"""Cross-model activation transfer.

Implements the three-level protocol of `Architecture-Dependent Causal Transfer
of Activation States Across Large Language Models` (arXiv:2608.16347) on
TransformerBridge primitives, so that one model's hidden states can be
compared with, projected into, and causally injected into another model:

1. **Representational similarity** — :func:`mutual_knn_alignment` scores how
   often the same items are mutual k-nearest neighbours in both hidden spaces.
   The score is rank-based over cosine neighbours, so the activation-magnitude
   outliers that dominate CKA and Procrustes cannot move it;
   :func:`linear_cka` is provided for that comparison.
2. **Cross-model retrieval** — :class:`ActivationProjection` fits a linear map
   from the source to the target hidden space on paired activations and scores
   whether a projected held-out source state retrieves its matching target
   state (top-1 accuracy against a ``1/n`` chance rate, with a row-permuted
   negative control).
3. **Causal injection** — :func:`residual_injection_hooks` overwrites a target
   block's residual stream with projected source activations, using the same
   ``[batch, position, d_model]`` intervention contract as
   :meth:`JacobianLens.steering_hooks`, so the transfer can be tested during
   generation.

The paper's projection *network* is replaced by a closed-form
ridge-regularized linear map (deterministic, needs no training loop, and
linear readout is the standard alignment baseline), and its four-model study
with pre-registered statistics is left to downstream experiment code; this
module ships the reusable primitives. :func:`cross_model_transfer_report` runs
all three levels over one shared prompt set.

Example::

    >>> import torch
    >>> from transformer_lens.tools.analysis import (
    ...     ActivationProjection,
    ...     mutual_knn_alignment,
    ... )
    >>> _ = torch.manual_seed(0)
    >>> source = torch.randn(32, 16)
    >>> isometry = torch.linalg.qr(torch.randn(24, 16)).Q
    >>> target = source @ isometry.T
    >>> round(mutual_knn_alignment(source, target, k=5), 6)
    1.0
    >>> projection = ActivationProjection.fit(
    ...     source[:24], target[:24], source_layer=6, target_layer=6
    ... )
    >>> projection.retrieval_accuracy(source[24:], target[24:])
    1.0
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from jaxtyping import Bool, Float

DEFAULT_K_NEAREST = 10
DEFAULT_RIDGE = 1e-4
DEFAULT_N_PERMUTATIONS = 200


# --------------------------------------------------------------------------- #
# level 1 — representational similarity                                       #
# --------------------------------------------------------------------------- #


def mutual_knn_alignment(
    source: Float[torch.Tensor, "n d_source"],
    target: Float[torch.Tensor, "m d_target"],
    *,
    k: int = DEFAULT_K_NEAREST,
) -> float:
    """Fraction of neighbours shared by the two hidden spaces' geometries.

    For each paired item ``i`` (same row index in both matrices), take its
    ``k`` cosine-nearest neighbours among the other items in the source space
    and in the target space, and score the size of the intersection divided by
    ``k``, averaged over items. The score lies in ``[0, 1]``; unrelated
    geometries sit near the permutation null (see
    :func:`cross_model_transfer_report`). Because neighbour sets depend only
    on cosine *ranks*, a few positions with outsized activation norms cannot
    dominate the score — the robustness property the paper reports for this
    metric over CKA and Procrustes.

    Args:
        source: ``[n, d_source]`` activations, one row per item.
        target: ``[n, d_target]`` activations, row ``i`` paired with row ``i``
            of ``source``.
        k: Neighbourhood size, clamped to ``n - 1``.

    Returns:
        The mean mutual-neighbour overlap in ``[0, 1]``.
    """
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError(
            f"activations must be [n, d] matrices, got {tuple(source.shape)} "
            f"and {tuple(target.shape)}"
        )
    if source.shape[0] != target.shape[0]:
        raise ValueError(
            f"paired activations must share the item axis, got "
            f"{source.shape[0]} source rows and {target.shape[0]} target rows"
        )
    n_items = source.shape[0]
    if n_items < 2:
        raise ValueError("mutual k-NN alignment needs at least 2 paired items")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    effective_k = min(k, n_items - 1)
    source_mask = _cosine_neighbour_mask(source, effective_k)
    target_mask = _cosine_neighbour_mask(target, effective_k)
    overlap = (source_mask & target_mask).float().sum(dim=-1) / effective_k
    return float(overlap.mean().item())


def _cosine_neighbour_mask(
    activations: Float[torch.Tensor, "n d"], k: int
) -> Bool[torch.Tensor, "n n"]:
    """Boolean mask of each row's ``k`` cosine-nearest other rows."""
    normalized = torch.nn.functional.normalize(activations.float(), dim=-1)
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(-math.inf)
    neighbours = similarity.topk(k, dim=-1).indices
    mask = torch.zeros(similarity.shape, dtype=torch.bool, device=similarity.device)
    return mask.scatter(-1, neighbours, True)


def linear_cka(
    source: Float[torch.Tensor, "n d_source"],
    target: Float[torch.Tensor, "n d_target"],
) -> float:
    """Linear centered kernel alignment between two activation matrices.

    Kept alongside :func:`mutual_knn_alignment` because the paper's comparison
    found CKA less robust: its Frobenius-norm terms are weighted by activation
    magnitude, so outlier positions dominate.

    Args:
        source: ``[n, d_source]`` activations, one row per item.
        target: ``[n, d_target]`` activations, row-aligned with ``source``.

    Returns:
        CKA in ``[0, 1]`` for real activations.

    Raises:
        ValueError: If either matrix is column-constant (zero denominator).
    """
    if source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]:
        raise ValueError(
            "activations must be row-aligned [n, d] matrices, got "
            f"{tuple(source.shape)} and {tuple(target.shape)}"
        )
    centered_source = source.float() - source.float().mean(dim=0, keepdim=True)
    centered_target = target.float() - target.float().mean(dim=0, keepdim=True)
    denominator = (centered_source.T @ centered_source).norm() * (
        centered_target.T @ centered_target
    ).norm()
    if not denominator > 0:
        raise ValueError("linear CKA is undefined for column-constant activations")
    cross_norm = (centered_source.T @ centered_target).norm()
    return float((cross_norm / denominator).item())


# --------------------------------------------------------------------------- #
# paired-activation collection                                                #
# --------------------------------------------------------------------------- #


@dataclass
class PairedActivations:
    """Last-position residual activations for the same prompts in two models.

    Attributes:
        source: ``[n, d_source]`` rows, one per prompt.
        target: ``[n, d_target]`` rows, row ``i`` from the same prompt as
            ``source[i]``.
        prompts: The prompts the rows were collected from, in row order.
    """

    source: Float[torch.Tensor, "n d_source"]
    target: Float[torch.Tensor, "n d_target"]
    prompts: List[str]


def collect_paired_activations(
    source_model: Any,
    target_model: Any,
    prompts: Sequence[str],
    *,
    source_layer: int,
    target_layer: int,
) -> PairedActivations:
    """Cache paired mid-layer activations from two models on shared prompts.

    Runs each prompt through both models with
    ``run_with_cache(names_filter=[...])`` and keeps the final-position
    residual-stream activation of the requested block — the per-prompt summary
    state, which is well-defined even when the two tokenizers segment the
    prompt differently.

    Args:
        source_model: Raw ``TransformerBridge`` for the source model.
        target_model: Raw ``TransformerBridge`` for the target model.
        prompts: Shared prompt strings.
        source_layer: Source block whose ``hook_out`` residual to read.
        target_layer: Target block whose ``hook_out`` residual to read.

    Returns:
        The paired activation matrices and the prompts they came from.
    """
    _require_raw_bridge(source_model, "source_model")
    _require_raw_bridge(target_model, "target_model")
    if not prompts:
        raise ValueError("prompts must contain at least one prompt")
    source_hook = _resid_post_hook_name(source_layer)
    target_hook = _resid_post_hook_name(target_layer)
    source_rows: List[torch.Tensor] = []
    target_rows: List[torch.Tensor] = []
    for prompt in prompts:
        source_rows.append(_last_position_activation(source_model, prompt, source_hook))
        target_rows.append(_last_position_activation(target_model, prompt, target_hook))
    return PairedActivations(
        source=torch.stack(source_rows),
        target=torch.stack(target_rows),
        prompts=list(prompts),
    )


def _resid_post_hook_name(layer: int) -> str:
    """Bridge-native hook for the output of block ``layer``."""
    return f"blocks.{layer}.hook_out"


def _last_position_activation(model: Any, prompt: str, hook_name: str) -> torch.Tensor:
    """Return the ``[d_model]`` final-position residual cached at *hook_name*."""
    _, cache = model.run_with_cache(prompt, names_filter=[hook_name], return_cache_object=False)
    if hook_name not in cache:
        raise ValueError(f"{hook_name} was not cached for prompt {prompt!r}")
    activation = cache[hook_name]
    if activation.ndim != 3 or activation.shape[1] < 1:
        raise ValueError(
            f"{hook_name} must cache [batch, position, d_model] activations with at "
            f"least one position, got shape {tuple(activation.shape)}"
        )
    return activation[:, -1, :].to(device="cpu", dtype=torch.float32).squeeze(0)


def _require_raw_bridge(model: Any, role: str) -> None:
    """Require a raw (non-compatibility-mode) ``TransformerBridge``."""
    from transformer_lens.model_bridge import TransformerBridge

    if not isinstance(model, TransformerBridge):
        raise TypeError(
            f"{role} must be a TransformerBridge; cross-model transfer compares raw "
            "HuggingFace activation bases. Load it with "
            "TransformerBridge.boot_transformers(...)."
        )
    if getattr(model, "compatibility_mode", False):
        raise ValueError(
            f"compatibility mode is enabled on {role} and changes the residual basis "
            "being compared. Use a freshly booted TransformerBridge model."
        )


# --------------------------------------------------------------------------- #
# level 2 — learned projection and cross-model retrieval                       #
# --------------------------------------------------------------------------- #


@dataclass
class ActivationProjection:
    """Linear map from a source model's hidden space to a target model's.

    Fitted in closed form (ridge-regularized least squares with a bias term)
    on paired activations; the paper's contribution at this level is the
    retrieval evaluation, which the linear map serves directly.

    Attributes:
        weight: ``[d_target, d_source]`` projection matrix.
        bias: ``[d_target]`` bias.
        source_layer: Source block the map was fitted from.
        target_layer: Target block the map was fitted to.
        metadata: Free-form provenance recorded at fit time.
    """

    weight: Float[torch.Tensor, "d_target d_source"]
    bias: Float[torch.Tensor, "d_target"]
    source_layer: int
    target_layer: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        source: Float[torch.Tensor, "n d_source"],
        target: Float[torch.Tensor, "n d_target"],
        *,
        source_layer: int,
        target_layer: int,
        ridge: float = DEFAULT_RIDGE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ActivationProjection":
        """Solve the ridge-regularized least-squares projection ``Y ~ [X, 1] W``.

        Args:
            source: ``[n, d_source]`` paired source activations.
            target: ``[n, d_target]`` paired target activations.
            source_layer: Source block the activations came from.
            target_layer: Target block the activations came from.
            ridge: L2 penalty; ``0`` gives the unregularized least-squares
                (min-norm when ``n < d_source``) solution.
            metadata: Extra provenance stored on the projection.

        Returns:
            The fitted projection.
        """
        if source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]:
            raise ValueError(
                "paired activations must be row-aligned [n, d] matrices, got "
                f"{tuple(source.shape)} and {tuple(target.shape)}"
            )
        if source.shape[0] < 1:
            raise ValueError("fitting needs at least one paired activation")
        if ridge < 0:
            raise ValueError(f"ridge must be >= 0, got {ridge}")
        n_items, d_source = source.shape
        d_target = target.shape[1]
        design = torch.cat(
            [source.float(), torch.ones(n_items, 1, dtype=torch.float32)], dim=-1
        )
        responses = target.float()
        if ridge > 0:
            penalty = math.sqrt(ridge) * torch.eye(d_source + 1, dtype=torch.float32)
            design = torch.cat([design, penalty], dim=0)
            responses = torch.cat(
                [responses, torch.zeros(d_source + 1, d_target, dtype=torch.float32)],
                dim=0,
            )
        solution = torch.linalg.lstsq(design, responses).solution
        return cls(
            weight=solution[:-1].T.contiguous(),
            bias=solution[-1].contiguous(),
            source_layer=source_layer,
            target_layer=target_layer,
            metadata=dict(metadata) if metadata else {},
        )

    def project(self, source: Float[torch.Tensor, "batch d_source"]) -> torch.Tensor:
        """Map source-space activations into the target hidden space."""
        if source.ndim != 2 or source.shape[-1] != self.weight.shape[1]:
            raise ValueError(
                f"expected [batch, {self.weight.shape[1]}] source activations, "
                f"got {tuple(source.shape)}"
            )
        weight = self.weight.to(device=source.device)
        bias = self.bias.to(device=source.device)
        return source.float() @ weight.T + bias

    def retrieval_accuracy(
        self,
        source: Float[torch.Tensor, "n d_source"],
        target: Float[torch.Tensor, "n d_target"],
    ) -> float:
        """Top-1 accuracy of retrieving each target row via its projected pair.

        Each projected source row is matched against every target row by
        cosine similarity; the score is the fraction of items whose own target
        row is the argmax. Chance is ``1 / n``.

        Args:
            source: ``[n, d_source]`` held-out source activations.
            target: ``[n, d_target]`` held-out target activations, row-aligned.

        Returns:
            The fraction of rows retrieved correctly.
        """
        if source.shape[0] != target.shape[0] or source.shape[0] < 1:
            raise ValueError(
                "retrieval needs equally many non-empty source and target rows, got "
                f"{source.shape[0]} and {target.shape[0]}"
            )
        projected = torch.nn.functional.normalize(self.project(source), dim=-1)
        reference = torch.nn.functional.normalize(target.float().to(projected.device), dim=-1)
        similarity = projected @ reference.T
        correct = similarity.argmax(dim=-1) == torch.arange(
            target.shape[0], device=similarity.device
        )
        return float(correct.float().mean().item())


# --------------------------------------------------------------------------- #
# level 3 — causal injection during generation                                #
# --------------------------------------------------------------------------- #


def residual_injection_hooks(
    model: Any,
    residual: torch.Tensor,
    layer: int,
    *,
    positions: Optional[Sequence[int]] = None,
) -> List[Tuple[str, Callable[..., torch.Tensor]]]:
    """Hooks that overwrite a block's residual stream with *residual* rows.

    The target-model half of causal transfer: projected source activations
    replace the target model's own residual at the block output, using the
    same ``[batch, position, d_model]`` intervention contract as the
    ``JacobianLens`` steering/swap hooks.

    Args:
        model: The target ``TransformerBridge`` the hooks will run on.
        residual: ``[d_model]`` row broadcast over batch and positions, or
            ``[batch, d_model]`` rows broadcast over positions (e.g. the
            :meth:`ActivationProjection.project` output for one prompt).
        layer: Block whose ``hook_out`` residual to replace. Negative indices
            count from ``n_layers``.
        positions: Chunk-local positions to overwrite (negative indices
            allowed and normalized on every hook invocation). Defaults to all.

    Returns:
        ``[(hook_name, fn), ...]`` for ``model.hooks(fwd_hooks=...)`` or
        ``model.run_with_hooks(fwd_hooks=...)``.
    """
    d_model = model.cfg.d_model
    if residual.ndim not in (1, 2) or residual.shape[-1] != d_model:
        raise ValueError(
            f"residual must be [d_model] or [batch, d_model] with d_model={d_model}, "
            f"got {tuple(residual.shape)}"
        )
    resolved_layer = _normalize_layer(layer, model.cfg.n_layers)
    requested = None if positions is None else tuple(positions)
    if requested == ():
        raise ValueError("positions must contain at least one index")
    injected = residual.detach().to(dtype=torch.float32)

    def hook_fn(
        activation: Float[torch.Tensor, "batch pos d_model"], hook: Any
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        hook_name = getattr(hook, "name", "injection hook")
        if activation.ndim != 3 or activation.shape[-1] != d_model:
            raise ValueError(
                f"{hook_name} must have shape [batch, position, {d_model}], "
                f"got {tuple(activation.shape)}"
            )
        rows = injected.to(device=activation.device)
        if rows.ndim == 1:
            rows = rows.unsqueeze(0).expand(activation.shape[0], -1)
        if requested is None:
            selected = list(range(activation.shape[1]))
        else:
            selected = _normalize_positions(requested, activation.shape[1])
        output = activation.clone()
        output[:, selected, :] = rows.unsqueeze(1).to(dtype=activation.dtype)
        return output

    return [(_resid_post_hook_name(resolved_layer), hook_fn)]


def _normalize_positions(positions: Tuple[int, ...], seq_len: int) -> List[int]:
    """Normalize negative chunk-local positions and raise before indexing."""
    normalized = [position + seq_len if position < 0 else position for position in positions]
    out_of_range = [position for position in normalized if not 0 <= position < seq_len]
    if out_of_range:
        raise ValueError(
            f"positions {out_of_range} out of range for an activation chunk of length {seq_len}"
        )
    return normalized


def _normalize_layer(layer: int, n_layers: int) -> int:
    """Resolve negative layer indices and bounds-check."""
    resolved = layer + n_layers if layer < 0 else layer
    if not 0 <= resolved < n_layers:
        raise ValueError(f"layer {layer} out of range for a {n_layers}-layer model")
    return resolved


# --------------------------------------------------------------------------- #
# three-level report                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class ActivationTransferReport:
    """Levels 1 and 2 of the transfer protocol for one model pair.

    Level 3 (injection during generation) is inherently generation-time: use
    :attr:`projection` with :func:`residual_injection_hooks` on the target
    model.

    Attributes:
        source_layer: Source block the comparison used.
        target_layer: Target block the comparison used.
        n_prompts: Number of paired prompts collected.
        n_train: Prompts the projection was fitted on.
        n_holdout: Held-out prompts retrieval was scored on.
        mutual_knn: Mutual k-NN alignment over all paired activations.
        mutual_knn_null_mean: Mean alignment under paired-row permutation.
        mutual_knn_p_value: Permutation p-value for ``mutual_knn`` exceeding
            the null (``None`` when ``n_permutations=0``).
        cka: Linear CKA over all paired activations.
        retrieval_top1: Held-out top-1 retrieval accuracy via the projection.
        retrieval_chance: ``1 / n_holdout``.
        retrieval_permuted_top1: Retrieval accuracy against row-permuted
            held-out targets — the negative control, expected at chance.
        projection: The fitted projection, ready for injection hooks.
    """

    source_layer: int
    target_layer: int
    n_prompts: int
    n_train: int
    n_holdout: int
    mutual_knn: float
    mutual_knn_null_mean: float
    mutual_knn_p_value: Optional[float]
    cka: float
    retrieval_top1: float
    retrieval_chance: float
    retrieval_permuted_top1: float
    projection: ActivationProjection


def cross_model_transfer_report(
    source_model: Any,
    target_model: Any,
    prompts: Sequence[str],
    *,
    source_layer: int,
    target_layer: int,
    k: int = DEFAULT_K_NEAREST,
    holdout_count: Optional[int] = None,
    ridge: float = DEFAULT_RIDGE,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int = 0,
) -> ActivationTransferReport:
    """Run levels 1 and 2 of the transfer protocol for one model pair.

    Collects paired activations, scores representational similarity (with a
    permutation null), fits the projection on a training split, and scores
    held-out retrieval against a permuted negative control. The split is
    deterministic — the last ``holdout_count`` prompts are held out — and the
    permutation nulls are driven by a seeded generator, so a report is
    reproducible.

    Args:
        source_model: Raw ``TransformerBridge`` for the source model.
        target_model: Raw ``TransformerBridge`` for the target model.
        prompts: Shared prompt strings (at least 3, so both splits are
            non-empty).
        source_layer: Source block to compare and project from.
        target_layer: Target block to compare and project to.
        k: Neighbourhood size for mutual k-NN alignment.
        holdout_count: Prompts held out of the projection fit. Defaults to
            ``max(1, n // 5)``.
        ridge: Ridge penalty for the projection fit.
        n_permutations: Paired-row permutations for the alignment null;
            ``0`` skips the null and leaves the p-value ``None``.
        seed: Seed for the permutation generator.

    Returns:
        The report, carrying the fitted :attr:`ActivationTransferReport.projection`.
    """
    paired = collect_paired_activations(
        source_model,
        target_model,
        prompts,
        source_layer=source_layer,
        target_layer=target_layer,
    )
    n_prompts = paired.source.shape[0]
    resolved_holdout = max(1, n_prompts // 5) if holdout_count is None else holdout_count
    if not 1 <= resolved_holdout <= n_prompts - 1:
        raise ValueError(
            f"holdout_count must leave at least one prompt to fit on; got "
            f"{resolved_holdout} for {n_prompts} prompts"
        )
    if n_prompts < 3:
        raise ValueError("the report needs at least 3 prompts to split and score")
    train_source = paired.source[: n_prompts - resolved_holdout]
    train_target = paired.target[: n_prompts - resolved_holdout]
    holdout_source = paired.source[n_prompts - resolved_holdout :]
    holdout_target = paired.target[n_prompts - resolved_holdout :]

    projection = ActivationProjection.fit(
        train_source,
        train_target,
        source_layer=source_layer,
        target_layer=target_layer,
        ridge=ridge,
    )
    retrieval_top1 = projection.retrieval_accuracy(holdout_source, holdout_target)
    retrieval_chance = 1.0 / resolved_holdout
    generator = torch.Generator().manual_seed(seed)
    control_permutation = torch.randperm(resolved_holdout, generator=generator)
    retrieval_permuted_top1 = projection.retrieval_accuracy(
        holdout_source, holdout_target[control_permutation]
    )

    mutual_knn = mutual_knn_alignment(paired.source, paired.target, k=k)
    null_scores = [
        mutual_knn_alignment(
            paired.source, paired.target[torch.randperm(n_prompts, generator=generator)], k=k
        )
        for _ in range(n_permutations)
    ]
    return ActivationTransferReport(
        source_layer=source_layer,
        target_layer=target_layer,
        n_prompts=n_prompts,
        n_train=n_prompts - resolved_holdout,
        n_holdout=resolved_holdout,
        mutual_knn=mutual_knn,
        mutual_knn_null_mean=float(sum(null_scores) / len(null_scores)) if null_scores else 0.0,
        mutual_knn_p_value=(
            (1 + sum(score >= mutual_knn for score in null_scores)) / (1 + len(null_scores))
            if null_scores
            else None
        ),
        cka=linear_cka(paired.source, paired.target),
        retrieval_top1=retrieval_top1,
        retrieval_chance=retrieval_chance,
        retrieval_permuted_top1=retrieval_permuted_top1,
        projection=projection,
    )
