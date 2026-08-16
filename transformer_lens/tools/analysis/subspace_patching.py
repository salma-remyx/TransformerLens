"""Subspace Activation Patching and the Interpretability Illusion.

Implements subspace activation patching (Eq. 1 / Appendix A.1 of Makelov, Lange
& Nanda, 2023) on top of :func:`transformer_lens.patching.generic_activation_patch`,
together with the diagnostics the paper uses to tell a *faithful* subspace apart
from an *illusory* one.

Background
----------
Full-component activation patching replaces an activation outright. Subspace
patching is finer-grained: given a component activation ``act`` in ``R^d`` and a
linear subspace ``U`` with orthonormal basis ``V`` (columns), patching from a
clean run into a corrupted run *along U* replaces only the orthogonal projection
onto ``U`` and leaves the complement alone:

    act_patched = act_corrupted + V V^T (act_clean - act_corrupted)
                = (I - V V^T) act_corrupted + V V^T act_clean

For a 1-D subspace spanned by a unit vector ``v`` this reduces to
``act_patched = act_corrupted + (v . act_clean - v . act_corrupted) v``.

The illusion
------------
A patch can recover the clean metric while localising nothing. ``v`` may split as
``v = v_null + v_row`` where ``v_null`` is the orthogonal projection of ``v`` onto
``ker(W_out)`` and ``v_row`` the remainder, which lies in the rowspace
``(ker W_out)^⊥``. ``v_null`` is *causally disconnected* -- ``W_out v_null = 0``,
so writing along it never reaches the residual stream -- and ``v_row`` may be
*dormant*: barely activated differently by the two prompts, yet able to steer the
output when set to out-of-distribution values. Patching the sum couples them: the
(sizable) clean-vs-corrupted difference carried by ``v_null`` is *converted* into a
write along ``W_out v_row`` (Eq. 2-3 of the paper), so the patch "works" through a
pathway that is not the feature you were trying to localise.

This module exposes:

- :func:`subspace_patch_setter` -- an ``Eq. 1`` projection-swap ``patch_setter``,
  usable directly with ``generic_activation_patch`` / ``get_act_patch_*``.
- :func:`project_onto_subspace` -- the ``P_U`` / ``I - P_U`` pair.
- :func:`nullspace_rowspace_decomposition` -- split a direction against
  ``ker(W_out)`` vs ``rowspace(W_out)`` for an MLP layer (or any linear map).
- :func:`fractional_logit_diff_decrease` -- the paper's FLDD success metric
  (Eq. 4).
- :func:`projection_spread` -- dormancy check: how differently a direction is
  activated by two prompt classes.
- :func:`subspace_patch_faithfulness` -- single-call diagnostic combining the
  above, mirroring the paper's Section 5 experiments.

Usage
-----
    from transformer_lens.patching import generic_activation_patch
    from transformer_lens.tools.analysis.subspace_patching import (
        nullspace_rowspace_decomposition,
        subspace_patch_faithfulness,
    )

    _, clean_cache = model.run_with_cache(clean_tokens)
    v = ...  # some candidate direction in post-GELU MLP activation space

    report = subspace_patch_faithfulness(
        model, clean_tokens, corrupted_tokens, clean_cache,
        patching_metric=logit_diff, direction=v, layer=8,
    )
    if report.illusion_suspected:
        ...  # patch succeeds, but not through the causally-connected part

References
----------
- Makelov, Lange & Nanda, "Is This the Subspace You Are Looking for? An
  Interpretability Illusion for Subspace Activation Patching" (2023),
  arXiv:2311.17030.
- Geiger et al., "Finding Alignments for Interpretable Concept-based Models"
  (2023) -- the DAS subspace-intervention formulation this patching operator
  implements.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple, Optional, Sequence, Union, cast

import pandas as pd
import torch
from jaxtyping import Float

import transformer_lens.utilities as utils
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.patching import (
    PatchedActivation,
    generic_activation_patch,
)

# An orthonormal basis: rows are the ambient dimensions, columns span the subspace.
SubspaceBasis = Float[torch.Tensor, "d_model subspace_dim"]


# ---------------------------------------------------------------------------
# Projection helpers (Eq. 1 / Appendix A.1)
# ---------------------------------------------------------------------------


def _orthonormalize(
    vectors: Float[torch.Tensor, "num_vectors d_model"],
) -> SubspaceBasis:
    """Return an orthonormal basis (columns) for ``span(vectors)`` via QR.

    Rank-deficient input yields extra zero columns, which are harmless here:
    ``V @ V.T`` is still the correct orthogonal projection onto the span.
    """
    vectors = vectors.to(dtype=torch.float32).T  # [d_model, num_vectors]
    q, _ = torch.linalg.qr(vectors, mode="reduced")  # [d_model, rank]
    return q


def project_onto_subspace(
    activation: Float[torch.Tensor, "... d_model"],
    basis: SubspaceBasis,
    complement: bool = False,
) -> Float[torch.Tensor, "... d_model"]:
    """Orthogonally project ``activation`` onto the subspace spanned by ``basis``.

    Args:
        activation: Tensor whose last dimension is the activation space.
        basis: Orthonormal columns spanning the subspace ``U``.
        complement: If True, project onto ``U^⊥`` instead of ``U``.

    Returns:
        The projection, same shape as ``activation``.
    """
    # (... d_model) @ (d_model dim) -> (... dim); then back through (dim d_model).
    projection = (activation @ basis) @ basis.T
    if complement:
        return activation - projection
    return projection


def subspace_patch_setter(
    corrupted_activation,
    index: Sequence[int],
    clean_activation,
    basis: Optional[SubspaceBasis] = None,
    direction: Optional[Float[torch.Tensor, "d_model"]] = None,
) -> PatchedActivation:
    """Patch setter implementing Eq. 1: swap the projection onto a subspace.

    Usable as the ``patch_setter`` of
    :func:`transformer_lens.patching.generic_activation_patch` (bind ``basis`` /
    ``direction`` with ``functools.partial``), making a subspace patch one more
    variant alongside ``layer_pos_patch_setter`` and friends.

    For a 1-D subspace spanned by unit ``v`` this is
    ``act_corrupted + (v.act_clean - v.act_corrupted) v``; for a ``k``-D subspace
    with orthonormal basis ``V`` it is
    ``act_corrupted + V V^T (act_clean - act_corrupted)``. Positions / heads
    outside the index being patched are left untouched.

    Args:
        corrupted_activation: The activation from the corrupted run.
        index: The index being patched, ``[layer, pos]`` for ``[batch, pos, d]``
            activations.
        clean_activation: The corresponding activation from the clean run.
        basis: Orthonormal columns spanning the subspace. Mutually exclusive
            with ``direction``.
        direction: A single direction; normalised internally, so it need not be
            unit norm.

    Returns:
        The patched activation.
    """
    if (basis is None) == (direction is None):
        raise ValueError("Provide exactly one of `basis` or `direction`.")
    if basis is None and direction is not None:
        basis = _orthonormalize(direction.detach().cpu().reshape(1, -1))
    basis = basis.to(
        device=corrupted_activation.device, dtype=corrupted_activation.dtype
    )

    _, pos = index
    corrupted = corrupted_activation[:, pos, ...]
    clean = clean_activation[:, pos, ...]
    delta = project_onto_subspace(clean - corrupted, basis)
    corrupted_activation[:, pos, ...] = corrupted + delta
    return corrupted_activation


# ---------------------------------------------------------------------------
# ker(W_out) / rowspace(W_out) decomposition
# ---------------------------------------------------------------------------


def nullspace_rowspace_decomposition(
    direction: Float[torch.Tensor, "d_model"],
    linear_map: Float[torch.Tensor, "d_out d_model"],
) -> tuple[
    Float[torch.Tensor, "d_model"],
    Float[torch.Tensor, "d_model"],
    Float[torch.Tensor, ""],
]:
    """Split ``direction`` into its ``ker(W)`` and ``rowspace(W)`` components.

    ``v_null`` is the orthogonal projection onto the nullspace of ``linear_map``
    and is *causally disconnected*: ``W @ v_null = 0``, so writing along it never
    reaches the output of the component. ``v_row = v - v_null`` lies in
    ``(ker W)^⊥`` = rowspace of ``W``, the causally-connected part.

    Args:
        direction: Candidate subspace direction in the component's activation
            space.
        linear_map: The map the component applies to that activation space. For
            an MLP layer this is ``W_out`` (``[d_mlp, d_model]``); pass
            ``model.blocks[layer].mlp.W_out``.

    Returns:
        Tuple of ``(v_null, v_row, null_fraction)`` where ``null_fraction`` is
        ``||v_null|| / ||v||``, the share of the direction that is causally
        disconnected.
    """
    v = direction.detach().to(dtype=torch.float32).reshape(-1)
    weight = linear_map.detach().to(dtype=torch.float32)
    if weight.dim() != 2:
        raise ValueError(f"linear_map must be 2-D, got shape {tuple(weight.shape)}")

    # We decompose the DOMAIN of W -- the activation space R^d_mlp for W_out --
    # into the part W can see and the part it annihilates. With
    # W = U diag(S) Vh (full_matrices), U's columns are an orthonormal basis of
    # the domain: U[:, :rank] spans range(W), the rowspace component that
    # survives the multiplication, and U[:, rank:] spans ker(W), the causally
    # disconnected directions. For a tall W_out the nullspace is the large part
    # (d_mlp - d_model dimensions).
    u, singular_values, _ = torch.linalg.svd(weight, full_matrices=True)
    tolerance = max(weight.shape) * torch.finfo(singular_values.dtype).eps * singular_values.max()
    rank = int((singular_values > tolerance).sum().item())
    row_basis = u[:, :rank]
    null_basis = u[:, rank:]

    v_null = project_onto_subspace(v, null_basis)
    v_row = project_onto_subspace(v, row_basis)
    norm = v.norm()
    null_fraction = (v_null.norm() / norm) if norm > 0 else torch.zeros((), dtype=v.dtype)
    return v_null, v_row, null_fraction


def _get_mlp_w_out(
    model: Union[HookedTransformer, TransformerBridge], layer: int
) -> Float[torch.Tensor, "d_mlp d_model"]:
    """Fetch ``W_out`` of an MLP block, working for both model systems."""
    try:
        mlp = model.blocks[layer].mlp
    except (AttributeError, IndexError, TypeError) as error:
        raise AttributeError(
            f"Could not reach blocks[{layer}].mlp on this model ({error!r}); the "
            "illusion diagnostic applies to MLP post-nonlinearity activations."
        ) from error
    w_out = getattr(mlp, "W_out", None)
    if w_out is None:
        raise AttributeError(
            f"Layer {layer} has no MLP W_out; the illusion diagnostic applies to "
            "MLP post-nonlinearity activations (hook 'post')."
        )
    return cast(Float[torch.Tensor, "d_mlp d_model"], w_out)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def fractional_logit_diff_decrease(
    clean_metric: float,
    corrupted_metric: float,
    patched_metric: float,
) -> float:
    """The paper's FLDD metric (Eq. 4), generalised off the IOI template.

    The paper defines ``FLDD = (logitdiff(x) - logitdiff_patch(x)) /
    logitdiff(x)`` on the IOI logit difference, which also happens to be the
    corrupted-run value they patch away from. Written against the general
    clean/corrupted pair, the natural form is the fraction of the
    clean-minus-corrupted gap that the intervention closes:

        FLDD = (patched - corrupted) / (clean - corrupted)

    ``0`` means the patch changed nothing relative to the corrupted run; ``1``
    means it fully restored the clean behaviour; values above ``1`` overshoot
    and negative values push further from the clean answer than the corrupted
    run. Any scalar patching metric (logit difference, loss, accuracy) works,
    so this applies to arbitrary clean/corrupted prompt pairs.
    """
    denominator = clean_metric - corrupted_metric
    if abs(denominator) < 1e-12:
        raise ValueError(
            "clean and corrupted metrics coincide, so FLDD is undefined "
            "(the two prompts do not differ on this metric)."
        )
    return (patched_metric - corrupted_metric) / denominator


def projection_spread(
    direction: Float[torch.Tensor, "d_model"],
    activations_a: Float[torch.Tensor, "... d_model"],
    activations_b: Float[torch.Tensor, "... d_model"],
) -> Float[torch.Tensor, ""]:
    """How differently ``direction`` is activated by two prompt classes.

    Returns ``|mean_A(v.act) - mean_B(v.act)| / (std_A + std_B)``, the
    between-class separation of the projections on ``v`` in units of the
    within-class spread. A *dormant* direction has a small value: both classes
    activate it the same way, so a patch along it moves nothing.
    """
    v = direction.detach().to(dtype=torch.float32).reshape(-1)
    v = v / v.norm()

    def _stats(acts):
        projections = acts.detach().to(dtype=torch.float32).reshape(-1, v.shape[0]) @ v
        return projections.mean(), projections.std(unbiased=False)

    mean_a, std_a = _stats(activations_a)
    mean_b, std_b = _stats(activations_b)
    spread = std_a + std_b
    if float(spread) < 1e-12:
        return torch.zeros((), dtype=v.dtype)
    return (mean_a - mean_b).abs() / spread


# ---------------------------------------------------------------------------
# The single-call diagnostic
# ---------------------------------------------------------------------------


class SubspacePatchReport(NamedTuple):
    """Results of :func:`subspace_patch_faithfulness`.

    All ``fldd_*`` fields are :func:`fractional_logit_diff_decrease` values: the
    fraction of the clean-vs-corrupted metric gap each intervention closes.

    Attributes:
        fldd_full: FLDD when the full direction is patched.
        fldd_rowspace_only: FLDD when only the causally-connected (rowspace)
            part is patched.
        fldd_nullspace_only: FLDD when only the causally disconnected
            (``ker W_out``) part is patched. This is 0 in exact arithmetic --
            ``W_out`` annihilates the patch -- and is reported as a sanity check
            on numerical conditioning.
        null_fraction: ``||v_null|| / ||v||`` -- share of the direction inside
            ``ker(W_out)``.
        rowspace_spread: :func:`projection_spread` of ``v_row`` against the clean
            vs corrupted activations. Small means the causally-connected part is
            *dormant*.
        nullspace_spread: Same for ``v_null``. Large means the disconnected part
            is what actually distinguishes the two prompts.
        illusion_suspected: True when the full patch recovers a large share of
            the gap while the rowspace-only patch recovers little, and a
            non-trivial part of the direction is disconnected.
    """

    fldd_full: float
    fldd_rowspace_only: float
    fldd_nullspace_only: float
    null_fraction: float
    rowspace_spread: float
    nullspace_spread: float
    illusion_suspected: bool


def subspace_patch_faithfulness(
    model: Union[HookedTransformer, TransformerBridge],
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    clean_cache: ActivationCache,
    patching_metric: Callable[[torch.Tensor], torch.Tensor],
    direction: Float[torch.Tensor, "d_mlp"],
    layer: int,
    pos: int = -1,
    activation_name: str = "post",
) -> SubspacePatchReport:
    """Diagnose whether a direction's patching effect is faithful or illusory.

    Runs the paper's Section 5 experiments for one candidate direction ``v`` in
    the post-nonlinearity activation space of an MLP layer:

    1. decompose ``v = v_null + v_row`` against ``ker(W_out)`` vs
       ``rowspace(W_out)``;
    2. patch along ``v``, along ``v_row`` alone, and along ``v_null`` alone,
       converting each to an FLDD against the clean and corrupted baselines
       (routed through :func:`generic_activation_patch`, so the library's hook
       machinery is reused);
    3. measure how differently each part is activated by the clean vs corrupted
       prompts (dormancy check).

    Interpretation follows the paper: if patching ``v`` recovers the clean
    behaviour but patching ``v_row`` alone does not, the effect is carried by
    the *disconnected* component ``v_null`` coupling into the *dormant*
    ``v_row`` -- the patch changed the output without localising the feature.

    Args:
        model: The relevant model.
        clean_tokens: The input tokens for the clean run (used for the clean
            baseline of the FLDD conversion; must be the tokens ``clean_cache``
            was produced from).
        corrupted_tokens: The input tokens for the corrupted run.
        clean_cache: Cached activations from the clean run.
        patching_metric: Function from the model's logits to a scalar, as in
            ``generic_activation_patch``.
        direction: Candidate direction, length ``d_mlp``.
        layer: Layer index of the MLP whose post-activations are patched.
        pos: Sequence position to patch.
        activation_name: Cache shorthand for the activation; defaults to ``post``
            (post-nonlinearity MLP activations, the component the paper
            analyses).

    Returns:
        A :class:`SubspacePatchReport`.
    """
    w_out = _get_mlp_w_out(model, layer)
    v_null, v_row, null_fraction = nullspace_rowspace_decomposition(direction, w_out)
    post_name = utils.get_act_name(activation_name, layer=layer)

    def _patch(along: Float[torch.Tensor, "d_model"]) -> float:
        index_df = pd.DataFrame([{"layer": layer, "pos": pos}])
        metric = generic_activation_patch(
            model,
            corrupted_tokens,
            clean_cache,
            patching_metric,
            partial(subspace_patch_setter, direction=along),
            activation_name,
            index_df=index_df,
        )
        return float(metric.item())

    with torch.no_grad():
        # Baselines for the FLDD conversion: the clean run's metric alongside the
        # corrupted one, and the corrupted post-activations for the dormancy
        # check. The clean activations themselves already come from clean_cache.
        clean_logits = model.run_with_hooks(clean_tokens, fwd_hooks=[])
        corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

    metric_clean = float(patching_metric(clean_logits).item())
    metric_corrupted = float(patching_metric(corrupted_logits).item())

    with torch.no_grad():
        metric_full = _patch(direction.detach().cpu().reshape(-1))
        metric_row = _patch(v_row)
        metric_null = _patch(v_null)

    clean_post = clean_cache[post_name]
    corrupted_post = corrupted_cache[post_name]

    fldd_full = fractional_logit_diff_decrease(metric_clean, metric_corrupted, metric_full)
    fldd_row = fractional_logit_diff_decrease(metric_clean, metric_corrupted, metric_row)
    fldd_null = fractional_logit_diff_decrease(metric_clean, metric_corrupted, metric_null)

    rowspace_spread = float(projection_spread(v_row, clean_post, corrupted_post).item())
    nullspace_spread = float(
        projection_spread(v_null, clean_post, corrupted_post).item()
    )

    # The paper's signature of the illusion: the full patch recovers a
    # substantial share of the clean-vs-corrupted gap, the causally-connected
    # part alone recovers much less, and a non-trivial slice of the direction
    # sits inside ker(W_out) to do the (disconnected) carrying.
    illusion_suspected = bool(
        fldd_full > 0.5
        and abs(fldd_row) < 0.5 * abs(fldd_full)
        and null_fraction > 0.1
    )
    return SubspacePatchReport(
        fldd_full=fldd_full,
        fldd_rowspace_only=fldd_row,
        fldd_nullspace_only=fldd_null,
        null_fraction=float(null_fraction.item()),
        rowspace_spread=rowspace_spread,
        nullspace_spread=nullspace_spread,
        illusion_suspected=illusion_suspected,
    )
