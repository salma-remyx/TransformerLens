"""Subspace activation patching and the bidirectional illusion diagnostic.

Standard activation patching (see :mod:`transformer_lens.patching`) replaces a
*whole* activation — every direction of a residual-stream position, or an entire
attention head output. Subspace patching is the finer-grained variant used to
test the hypothesis "this feature lives in this low-dimensional subspace": keep
the corrupted activation, but overwrite only its projection onto a candidate
subspace with the clean run's value in that same subspace.

Background
----------
Given an activation ``a`` and a subspace spanned by the columns of an
orthonormal matrix ``B`` (shape ``[d_model, k]``), the subspace patch is

    a_patched = a_corrupted + B (B^T (a_clean - a_corrupted))

The orthogonal complement is untouched, so any output change is attributable to
the moved subspace *alone* — which is exactly what feature attribution claims
rest on.

The interpretability illusion
-----------------------------
Adapted from *"Is This the Subspace You Are Looking For? An Interpretability
Illusion for Subspace Activation Patching"* (arXiv:2311.17030). The paper shows
that the two things subspace patching is used for — *manipulating* behaviour and
*attributing* a feature to a subspace — can come apart. A patch can move the
output exactly as if the feature had changed, while the moved subspace is in
fact causally isolated and the change is produced by a dormant parallel pathway
recruited by the intervention.

The cheapest evidence against that illusion is **bidirectional patching**. Run
the same subspace patch in both directions over the same (clean, corrupted)
pair:

- corrupted → clean (the usual "recovery" direction): patch the clean subspace
  value *into* the corrupted run, metric should move toward the clean answer.
- clean → corrupted (the "corruption" direction): patch the corrupted subspace
  value *into* the clean run, metric should move toward the corrupted answer.

For a subspace that genuinely carries the feature, both directions move the
metric by a comparable amount. A causally disconnected subspace that merely
trips a parallel pathway shows a marked **asymmetry** — one direction moves the
output while the reverse patch does not, because there is nothing in the
subspace for the reverse patch to destroy. :func:`subspace_patch_asymmetry`
returns both sweeps and the per-site asymmetry score so this can be read
directly off the same ``patched_metric_output`` tensor the standard sweeps
produce, with no extra infrastructure.

Usage
-----
    from transformer_lens.tools.analysis import subspace_patch_asymmetry

    _, clean_cache = model.run_with_cache(clean_tokens)
    _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    # A candidate 1-D subspace (e.g. from an SAE latent, a probing classifier,
    # or a rank-1 model edit direction).
    direction = torch.nn.functional.normalize(some_vector, dim=-1)  # [d_model]

    sweeps = subspace_patch_asymmetry(
        model,
        clean_cache=clean_cache,
        corrupted_cache=corrupted_cache,
        clean_tokens=clean_tokens,
        corrupted_tokens=corrupted_tokens,
        patching_metric=metric,
        subspace=direction,
        activation_name="resid_pre",
        layer=7,
        pos=-1,
    )
    sweeps.summary()   # per-direction effect + asymmetry score

For a full (layer, position) sweep of one subspace, see
:func:`get_act_patch_resid_subspace_all_pos`, or plug
:func:`layer_pos_subspace_patch_setter` into
:func:`transformer_lens.patching.generic_activation_patch` directly (it is also
re-exported as ``transformer_lens.patching.layer_pos_subspace_patch_setter``).

References
----------
- Syed et al., "Is This the Subspace You Are Looking For? An Interpretability
  Illusion for Subspace Activation Patching" (2023), arXiv:2311.17030.
- Meng et al., "Locating and Editing Factual Associations in GPT" (2022) — the
  rank-1 editing case the paper gives a mechanistic account of.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Sequence, Union

import pandas as pd
import torch
from jaxtyping import Float

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.patching import generic_activation_patch

PatchingMetricT = Callable[[torch.Tensor], torch.Tensor]

ModelT = Union[HookedTransformer, TransformerBridge]


# ---------------------------------------------------------------------------
# Subspace basis handling
# ---------------------------------------------------------------------------


def orthonormalize(
    subspace: Float[torch.Tensor, "... d_model k"],
) -> Float[torch.Tensor, "... d_model k"]:
    """Return an orthonormal basis with the same column span as ``subspace``.

    Accepts ``[d_model, k]`` or a bare ``[d_model]`` direction (returned as
    ``[d_model, 1]``). Columns need not be orthogonal or unit-norm — a probing
    classifier's weights or a stack of SAE latents usually are not.

    Uses the reduced QR decomposition; the ``eps`` guard maps a numerically
    rank-deficient input to a zeroed column rather than an arbitrary one, so a
    rank-deficient basis silently patches nothing instead of patching noise.
    """
    subspace = subspace.to(dtype=torch.float32)
    if subspace.dim() == 1:
        subspace = subspace.unsqueeze(-1)
    if subspace.shape[-1] == 1:
        return torch.nn.functional.normalize(subspace, dim=-2)
    q, r = torch.linalg.qr(subspace, mode="reduced")
    # A rank-deficient input leaves near-zero diagonal entries in r; the
    # matching columns of q are numerically arbitrary, so drop them. The
    # threshold is scaled by the input's column norms (the same scale the
    # diagonal is measured on) — an absolute eps is far too tight for the
    # ~1e-7 diagonals LAPACK actually produces here.
    column_norms = torch.linalg.norm(subspace, dim=-2)
    threshold = torch.finfo(r.dtype).eps * max(1.0, float(column_norms.max())) * subspace.shape[-2]
    keep = torch.abs(torch.diagonal(r, dim1=-2, dim2=-1)) > threshold
    return q * keep.to(q.dtype)


def project(
    activation: Float[torch.Tensor, "... d_model"],
    basis: Float[torch.Tensor, "d_model k"],
) -> Float[torch.Tensor, "... d_model"]:
    """Project ``activation`` onto the column span of orthonormal ``basis``.

    Returns the projection as a full ``d_model``-shaped vector, so callers can
    subtract it (complement) or add it (patch) without reshaping.
    """
    coefficients = activation @ basis  # [..., k]
    return coefficients @ basis.transpose(-1, -2)  # [..., d_model]


# ---------------------------------------------------------------------------
# Subspace patch setters
# ---------------------------------------------------------------------------


def layer_pos_subspace_patch_setter(
    corrupted_activation: torch.Tensor,
    index: Sequence[int],
    clean_activation: torch.Tensor,
    subspace: Optional[Float[torch.Tensor, "d_model k"]] = None,
) -> torch.Tensor:
    """Patch only a subspace of one position, leaving the complement untouched.

    Compatible with :func:`transformer_lens.patching.generic_activation_patch`
    — same ``(corrupted_activation, index, clean_activation)`` contract, with
    the subspace supplied via :func:`functools.partial`. Index is
    ``[layer, pos]`` and the activation axis order is ``[batch, pos, ...]``,
    exactly like ``layer_pos_patch_setter``.

    Mathematically, at the indexed position ``p``::

        act[:, p] += B @ (B^T @ (clean[:, p] - corrupted[:, p]))

    so the moved content is the *difference* between the two runs confined to
    the subspace — the intervention tested by the paper.
    """
    if subspace is None:
        raise ValueError("subspace is required — pass it via functools.partial")
    layer, pos = index
    basis = orthonormalize(subspace).to(corrupted_activation.dtype)
    delta = clean_activation[:, pos, ...] - corrupted_activation[:, pos, ...]
    corrupted_activation[:, pos, ...] = corrupted_activation[:, pos, ...] + project(delta, basis)
    return corrupted_activation


def layer_subspace_patch_setter(
    corrupted_activation: torch.Tensor,
    index: Sequence[int],
    clean_activation: torch.Tensor,
    subspace: Optional[Float[torch.Tensor, "d_model k"]] = None,
) -> torch.Tensor:
    """Patch a subspace across every position of a layer's activation.

    Index is ``[layer]``; the activation axis order is ``[batch, pos, ...]``.
    Useful when the candidate subspace is a whole-layer feature direction
    rather than a position-localised one.
    """
    if subspace is None:
        raise ValueError("subspace is required — pass it via functools.partial")
    (layer,) = index
    basis = orthonormalize(subspace).to(corrupted_activation.dtype)
    delta = clean_activation - corrupted_activation
    corrupted_activation[:, ...] = corrupted_activation[:, ...] + project(delta, basis)
    return corrupted_activation


# ---------------------------------------------------------------------------
# Sweeps built on the existing generic patching loop
# ---------------------------------------------------------------------------


def get_act_patch_resid_subspace_all_pos(
    model: ModelT,
    corrupted_tokens: torch.Tensor,
    clean_cache: ActivationCache,
    patching_metric: PatchingMetricT,
    subspace: Float[torch.Tensor, "d_model k"],
    activation_name: str = "resid_pre",
) -> Float[torch.Tensor, "n_layers pos"]:
    """Sweep a subspace patch over every (layer, position).

    A drop-in sibling of ``get_act_patch_resid_pre`` that moves a subspace
    instead of the whole residual vector. Returns ``[n_layers, pos]``, entry
    ``(l, p)`` being the metric when only the subspace at layer ``l``,
    position ``p`` is patched from clean into corrupted.

    ``activation_name`` defaults to ``resid_pre`` but accepts any
    ``[batch, pos, d_model]`` activation (``resid_mid``, ``attn_out``,
    ``mlp_out``, ...).
    """
    return generic_activation_patch(
        model,
        corrupted_tokens,
        clean_cache,
        patching_metric,
        partial(layer_pos_subspace_patch_setter, subspace=subspace),
        activation_name,
        index_axis_names=("layer", "pos"),
    )


# ---------------------------------------------------------------------------
# Bidirectional illusion diagnostic
# ---------------------------------------------------------------------------


@dataclass
class SubspacePatchSweeps:
    """Paired subspace sweeps in both patch directions.

    Attributes:
        recovery: corrupted→clean sweep — the usual activation-patching
            direction. Each entry is the metric on the corrupted run with the
            clean subspace value patched in.
        corruption: clean→corrupted sweep — the reverse direction. Each entry
            is the metric on the clean run with the corrupted subspace value
            patched in.
        clean_metric: metric on the unpatched clean run.
        corrupted_metric: metric on the unpatched corrupted run.
    """

    recovery: torch.Tensor
    corruption: torch.Tensor
    clean_metric: float
    corrupted_metric: float

    def effects(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fraction of the clean-corrupted metric gap closed in each direction.

        ``recovery_effect`` is the standard normalised patching effect; the
        novel quantity here is ``corruption_effect``, measured by swapping the
        roles of the two runs. Both are relative to the same gap, so they are
        directly comparable — that comparability is what the asymmetry score
        below is computed from.
        """
        gap = self.clean_metric - self.corrupted_metric
        if gap == 0:
            raise ValueError(
                "clean and corrupted metrics are equal — the patch direction "
                "has no gap to close, so the asymmetry score is undefined"
            )
        recovery_effect = (self.recovery - self.corrupted_metric) / gap
        corruption_effect = (self.clean_metric - self.corruption) / gap
        return recovery_effect, corruption_effect

    def asymmetry(self) -> torch.Tensor:
        """Difference between the two directions' normalised effects.

        Near zero for a subspace that genuinely carries the feature; large in
        magnitude when one direction moves the output and the reverse patch
        does not — the signature of a causally disconnected subspace whose
        apparent effect is produced by a parallel pathway.
        """
        recovery_effect, corruption_effect = self.effects()
        return recovery_effect - corruption_effect

    def summary(self) -> dict[str, float]:
        """Scalar summary: both effects and the asymmetry, averaged over sites."""
        recovery_effect, corruption_effect = self.effects()
        return {
            "recovery_effect": float(recovery_effect.mean()),
            "corruption_effect": float(corruption_effect.mean()),
            "asymmetry": float(self.asymmetry().mean()),
        }


def _subspace_sweep(
    model: ModelT,
    run_tokens: torch.Tensor,
    source_cache: ActivationCache,
    patching_metric: PatchingMetricT,
    subspace: Float[torch.Tensor, "d_model k"],
    activation_name: str,
    layer: int,
    pos: Optional[int],
) -> torch.Tensor:
    """One-direction subspace sweep at a single layer (and optional position).

    Built on :func:`generic_activation_patch` with an explicit ``index_df``, so
    the sweep reuses the existing hook/partial machinery rather than a bespoke
    loop. The single-row dataframe means the result is a 1-element tensor.
    """
    if pos is None:
        setter = partial(layer_subspace_patch_setter, subspace=subspace)
        index_df = pd.DataFrame({"layer": [layer]})
    else:
        setter = partial(layer_pos_subspace_patch_setter, subspace=subspace)
        index_df = pd.DataFrame({"layer": [layer], "pos": [pos]})

    return generic_activation_patch(
        model,
        run_tokens,
        source_cache,
        patching_metric,
        setter,
        activation_name,
        index_df=index_df,
    )


def subspace_patch_asymmetry(
    model: ModelT,
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    patching_metric: PatchingMetricT,
    subspace: Float[torch.Tensor, "d_model k"],
    activation_name: str = "resid_pre",
    layer: int = 0,
    pos: Optional[int] = None,
) -> SubspacePatchSweeps:
    """Run the same subspace patch in both directions and score the asymmetry.

    This is the diagnostic the paper prescribes for distinguishing a subspace
    that carries a feature from one that merely looks like it does: patch the
    subspace clean→corrupted (recovery) and corrupted→clean (corruption) over
    the same site and compare how much of the clean-corrupted metric gap each
    direction closes. See :class:`SubspacePatchSweeps`.

    Args:
        model: The model to intervene on (HookedTransformer or TransformerBridge).
        clean_cache: Cached activations from the clean run.
        corrupted_cache: Cached activations from the corrupted run.
        clean_tokens: Tokens for the clean run.
        corrupted_tokens: Tokens for the corrupted run.
        patching_metric: Metric from logits to a scalar (same contract as
            ``generic_activation_patch``).
        subspace: Candidate subspace, shape ``[d_model, k]`` (or ``[d_model]``
            for a single direction). Need not be orthonormal.
        activation_name: Any ``[batch, pos, d_model]`` activation.
        layer: Layer to patch at.
        pos: Position to patch at. ``None`` patches every position.

    Returns:
        :class:`SubspacePatchSweeps` with both sweeps and the derived scores.
    """
    clean_metric = float(patching_metric(model(clean_tokens)).item())
    corrupted_metric = float(patching_metric(model(corrupted_tokens)).item())

    recovery = _subspace_sweep(
        model,
        run_tokens=corrupted_tokens,
        source_cache=clean_cache,
        patching_metric=patching_metric,
        subspace=subspace,
        activation_name=activation_name,
        layer=layer,
        pos=pos,
    )
    corruption = _subspace_sweep(
        model,
        run_tokens=clean_tokens,
        source_cache=corrupted_cache,
        patching_metric=patching_metric,
        subspace=subspace,
        activation_name=activation_name,
        layer=layer,
        pos=pos,
    )
    return SubspacePatchSweeps(
        recovery=recovery,
        corruption=corruption,
        clean_metric=clean_metric,
        corrupted_metric=corrupted_metric,
    )
