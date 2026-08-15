"""Layer-wise residual-collapse profile for decoder-only transformers.

Post-norm (original-Transformer) decoders are hard to train because deep stacks
drift into a *rank-collapsed* regime where every token's residual representation
points in nearly the same direction. `A Mechanistic Diagnostic of Rank Collapse
in Post-Norm Decoder Transformers <https://arxiv.org/abs/2608.09417>`_ splits
that failure into a forward and a backward stage, and both are readable off a
single forward pass:

* **Forward — similarity amplification.** Causal attention acts approximately
  as a prefix-averaging operator, so mean pairwise token similarity grows with
  depth, while the MLP/SwiGLU branch contributes only a smaller damping term.
  Attributing each block's similarity increment to its attention and MLP
  sublayers reproduces that split directly from the cache.
* **Backward — repair incapacity.** Once the network enters the
  high-similarity regime the pre-normalization residual RMS grows, and the
  RMSNorm backward Jacobian :math:`(I - \\hat{x}\\hat{x}^\\top)/\\mathrm{RMS}(x)`
  scales as :math:`1/\\mathrm{RMS}(x)`. RMS growth therefore makes each norm
  contractive in the backward pass, and gradients reaching earlier layers decay
  geometrically — the collapsed state cannot repair itself.

Both stages are computed here from an
:class:`~transformer_lens.ActivationCache.ActivationCache` using its
stacked-residual primitives
(:meth:`~transformer_lens.ActivationCache.ActivationCache.stack_activation` and
shorthand ``__getitem__`` indexing), so the tool works for any model whose cache
exposes ``hook_resid_post`` — ``TransformerBridge`` and legacy ``HookedTransformer``
alike. The pre-norm RMS is taken from the residual tensors
themselves rather than from a particular norm's ``hook_scale`` so the profile
also runs on caches that were filtered with ``names_filter``.

The norm gain is intentionally omitted from the backward factor: it is a
per-model constant that does not change *across layers*, so the layer-wise decay
— the quantity the paper predicts — is unaffected.

Example::

    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.tools.analysis import residual_collapse_profile

    model = TransformerBridge.boot_transformers("gpt2", device="cpu")
    profile = residual_collapse_profile(model, "The quick brown fox jumps over")
    print(profile.summary())
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from jaxtyping import Float

from transformer_lens.ActivationCache import ActivationCache

# Inputs accepted for the prompt, mirroring the other tools in this package.
ProfileInput = Union[str, List[str], torch.Tensor]


@dataclass
class ResidualCollapseProfile:
    """Per-layer readout of the residual stream's collapse state.

    All leading axes are aligned with ``labels``: the embedding followed by each
    block's output, i.e. ``len(labels) == n_layers + 1``.

    Attributes:
        token_similarity:
            Mean pairwise cosine similarity between token vectors at each layer.
            This is the paper's scalar state variable; growth across depth is
            the forward signature of rank collapse.
        effective_rank:
            Entropy-based effective rank of the ``[position, d_model]`` hidden
            state matrix. Collapse drives this toward 1.
        attn_similarity_gain:
            Per-block similarity increment produced by the attention sublayer
            (``sim(resid_mid) - sim(resid_pre)``). ``None`` when the cache has
            no ``hook_resid_mid`` (parallel-residual blocks).
        mlp_similarity_gain:
            Per-block similarity increment produced by the MLP sublayer
            (``sim(resid_post) - sim(resid_mid)``), or ``None`` as above.
        residual_rms:
            RMS of the residual stream entering each block — the
            pre-normalization residual norm whose growth drives the backward
            contraction.
        norm_backward_factor:
            Upper bound on the norm of each block's input-norm backward Jacobian,
            ``1 / residual_rms``. Values below 1 are contractive.
        cumulative_norm_backward_factor:
            Cumulative product of ``norm_backward_factor`` — the geometric
            attenuation the paper predicts for gradients reaching layer 0.
        labels:
            Names aligned with the leading axis: ``"embed"`` then ``"0_post"``,
            ``"1_post"``, ...
    """

    token_similarity: Float[torch.Tensor, "layer"]
    effective_rank: Float[torch.Tensor, "layer"]
    residual_rms: Float[torch.Tensor, "block"]
    norm_backward_factor: Float[torch.Tensor, "block"]
    cumulative_norm_backward_factor: Float[torch.Tensor, "block"]
    labels: List[str]
    attn_similarity_gain: Optional[Float[torch.Tensor, "block"]] = None
    mlp_similarity_gain: Optional[Float[torch.Tensor, "block"]] = None

    def summary(self, similarity_threshold: float = 0.9) -> Dict[str, Any]:
        """Summarize the collapse state as plain Python scalars.

        Args:
            similarity_threshold:
                Token-similarity level at or above which the layer counts as
                being in the paper's high-similarity regime.
        """
        collapsed = self.token_similarity >= similarity_threshold
        onset = int(torch.argmax(collapsed.int()).item()) if bool(collapsed.any()) else None
        return {
            "high_similarity_onset_layer": onset,
            "final_token_similarity": float(self.token_similarity[-1].item()),
            "final_effective_rank": float(self.effective_rank[-1].item()),
            "final_residual_rms": float(self.residual_rms[-1].item()),
            "log10_cumulative_norm_backward_factor": float(
                torch.log10(self.cumulative_norm_backward_factor[-1].clamp_min(1e-30)).item()
            ),
        }


def _with_batch_dim(
    tensor: torch.Tensor, has_batch_dim: bool
) -> Float[torch.Tensor, "batch pos d_model"]:
    """Return a residual tensor with an explicit leading batch axis."""
    if has_batch_dim:
        return tensor
    return tensor.unsqueeze(0)


def _mean_offdiag_similarity(x: Float[torch.Tensor, "batch pos d_model"]) -> float:
    """Mean cosine similarity over all unordered pairs of distinct positions."""
    if x.shape[-2] < 2:
        return float("nan")
    normalized = torch.nn.functional.normalize(x.float(), dim=-1)
    sim = normalized @ normalized.transpose(-1, -2)
    n = sim.shape[-1]
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=sim.device), diagonal=1)
    return float(sim.masked_select(mask).view(sim.shape[0], -1).mean(dim=-1).mean().item())


def _effective_rank(x: Float[torch.Tensor, "batch pos d_model"]) -> float:
    """Entropy-based effective rank, ``exp(H(singular values))``, batch-meaned."""
    singular = torch.linalg.svdvals(x.float())
    probs = singular / singular.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy = -torch.where(probs > 0, probs * probs.log(), torch.zeros_like(probs)).sum(dim=-1)
    return float(entropy.exp().mean().item())


def residual_collapse_profile(
    model: Optional[Any] = None,
    input: Optional[ProfileInput] = None,
    *,
    cache: Optional[ActivationCache] = None,
    eps: float = 1e-5,
) -> ResidualCollapseProfile:
    """Compute a layer-wise residual-collapse profile.

    Runs the model once with caching unless a precomputed ``cache`` is passed,
    then reports the forward signature (token-similarity growth, effective-rank
    loss, and the attention-vs-MLP attribution of that growth) and the backward
    signature (pre-norm residual RMS growth and the resulting geometric
    contraction of the RMSNorm backward factor).

    Args:
        model:
            A ``TransformerBridge`` or ``HookedTransformer``. Optional only when
            a precomputed ``cache`` is supplied.
        input:
            Prompt to run — a string, list of strings, or token tensor.
        cache:
            Optional precomputed ``ActivationCache`` to reuse instead of running
            the model again.
        eps:
            Epsilon added to the mean square before the square root when
            computing residual RMS, matching the normalization default.

    Returns:
        A :class:`ResidualCollapseProfile` whose leading axes are aligned with
        ``labels``.

    Raises:
        ValueError: If neither ``input`` nor ``cache`` is provided, or if the
            cache lacks the per-block ``hook_resid_post`` activations.
    """
    if cache is None:
        if model is None or input is None:
            raise ValueError(
                "provide `model` and `input` to run the model, or a precomputed `cache`"
            )
        _, cache = model.run_with_cache(input)

    n_layers = cache.model.cfg.n_layers
    try:
        post_stack = cache.stack_activation("resid_post")
    except KeyError as err:
        raise ValueError(
            "residual_collapse_profile needs per-block `hook_resid_post` activations in the "
            "cache; re-run run_with_cache without a names_filter, or on an architecture whose "
            "blocks expose the standard residual hooks."
        ) from err

    has_batch = cache.has_batch_dim
    # Every per-layer tensor is given an explicit leading batch axis so the two
    # cache shapes (with and without the batch dim) produce identical profiles.
    # post_stack from stack_activation is [n_layers, *batch_and_pos, d_model].
    embed = cache["embed"] if "embed" in cache.cache_dict else cache[("resid_pre", 0)]
    trajectory = torch.stack(
        [_with_batch_dim(embed, has_batch)] + [_with_batch_dim(t, has_batch) for t in post_stack],
        dim=0,
    ).float()

    token_similarity = torch.tensor(
        [_mean_offdiag_similarity(t) for t in trajectory], device=trajectory.device
    )
    effective_rank = torch.tensor(
        [_effective_rank(t) for t in trajectory], device=trajectory.device
    )

    # Pre-norm residual RMS per block: the quantity whose growth makes the
    # RMSNorm backward factor contractive.
    resid_pre = torch.stack(
        [_with_batch_dim(cache[("resid_pre", l)], has_batch) for l in range(n_layers)], dim=0
    ).float()
    residual_rms = (
        resid_pre.pow(2).mean(dim=-1) + eps
    ).sqrt().flatten(start_dim=1).mean(dim=-1)
    norm_backward_factor = 1.0 / residual_rms

    labels = ["embed"] + [f"{layer}_post" for layer in range(n_layers)]

    attn_gain: Optional[torch.Tensor] = None
    mlp_gain: Optional[torch.Tensor] = None
    try:
        resid_mid = torch.stack(
            [_with_batch_dim(cache[("resid_mid", l)], has_batch) for l in range(n_layers)], dim=0
        ).float()
    except KeyError:
        # Parallel-residual blocks expose no post-attention residual, so the
        # branch attribution is unavailable rather than wrong.
        resid_mid = None

    if resid_mid is not None:
        pre_sim = torch.tensor(
            [_mean_offdiag_similarity(t) for t in resid_pre], device=resid_pre.device
        )
        mid_sim = torch.tensor(
            [_mean_offdiag_similarity(t) for t in resid_mid], device=resid_mid.device
        )
        post_sim = token_similarity[1:]
        attn_gain = mid_sim - pre_sim
        mlp_gain = post_sim - mid_sim

    return ResidualCollapseProfile(
        token_similarity=token_similarity,
        effective_rank=effective_rank,
        residual_rms=residual_rms,
        norm_backward_factor=norm_backward_factor,
        cumulative_norm_backward_factor=torch.cumprod(norm_backward_factor, dim=0),
        labels=labels,
        attn_similarity_gain=attn_gain,
        mlp_similarity_gain=mlp_gain,
    )
