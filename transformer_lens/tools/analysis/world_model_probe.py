"""World-model probe fidelity, decay, and restoration.

Large reasoning models build a linearly decodable representation of a task's
state space — an *emergent world model* — and then fail to *maintain* it while
they plan, which is where their errors come from.
`Transformers Struggle to Use Their Emergent World Models
<https://arxiv.org/abs/2608.07077>`_ demonstrates this on the Tower of Hanoi:
the Sierpinski-triangle state geometry is encoded near-perfectly at the end of
the prompt, decays across the generated plan, and performance is partially
recovered by re-injecting the prompt-time representation at inference.

This module makes that three-stage diagnose → localise → intervene pipeline a
library call instead of bespoke notebook code:

1. :func:`fit_world_model_probe` — least-squares readout of the task state
   from residual activations, plus a *fidelity* score. The paper's probes are
   trained on activations from a model trained on the task; here any cached
   run with known states serves as the training set, so the probe is a
   property of ``(model, layer, task)`` rather than of a training run.
2. :func:`probe_fidelity_by_position` — the probe applied along the sequence,
   which is the paper's "representation decay" measurement. A monotone drop
   from prompt end into the plan is the paper's signature failure.
3. :func:`restoration_hooks` — the causal stage. A forward hook that projects
   the residual stream back onto the prompt-time world-model subspace, so the
   plan-time representation is restored toward what the model encoded when it
   had the puzzle fully in view.

The probe and the intervention are linear, so they compose with the rest of the
toolkit: read out a fitted direction with ``lens_vectors``, compare a
``probe_fidelity_by_position`` curve before and after ``restoration_hooks``, or
sweep the injection layer band the way ``JacobianLens.steering_hooks`` sweeps
its own.

Example::

    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.tools.analysis import (
        fit_world_model_probe,
        probe_fidelity_by_position,
        restoration_hooks,
    )

    model = TransformerBridge.boot_transformers("Qwen/Qwen2.5-1.5B-Instruct")

    # One ground-truth state per cached position, e.g. Tower of Hanoi peg
    # occupancy as the prompt lays out each move.
    _, prompt_cache = model.run_with_cache(prompt)
    states = hanoi_states_per_position(prompt)  # [batch, pos, state_dim]

    probe = fit_world_model_probe(prompt_cache, layer=8, states=states)
    fidelity = probe_fidelity_by_position(prompt_cache, probe, states)
    print(fidelity.explained_variance)  # high at prompt end?

    # Then re-run generation with the prompt-time representation restored.
    with model.hooks(fwd_hooks=restoration_hooks(probe, alpha=1.0)):
        answer = model.generate(prompt)
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch
from jaxtyping import Float

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.utilities.components_utils import get_act_name

# Blocks whose output is not a plain residual stream (Mamba/SSM and friends).
# Mirrors the hybrid-architecture guard the sibling analysis tools use.
_HYBRID_VARIANT_NAMES = ("mamba", "ssm", "mixer", "linear_attn")


@dataclass
class WorldModelProbe:
    """A fitted linear readout of task state from the residual stream.

    Attributes:
        layer:
            The block whose output activations the probe reads.
        directions:
            The fitted readout rows, ``[n_components, d_model]``. Row ``i`` is
            the direction state component ``i`` is read out along, at the scale
            the least-squares fit chose — not unit-normalised, so that
            ``read()`` reproduces the fitted affine map exactly. Their span is
            the world-model subspace the interventions act on.
        intercepts:
            Per-component offset, ``[n_components]``. The residual stream is
            not centred, so states are recovered as
            ``h @ directions.T + intercepts``.
        state_dim:
            Number of state components the probe reads out.
        explained_variance:
            Training fidelity on the fit positions — the fraction of state
            variance the linear readout recovers. The paper reports
            near-perfect fidelity at the end of the prompt and a sharp drop
            during planning; this field is the "is there a world model at
            all?" number.
        fit_positions:
            Positions used for the fit, kept so decay curves can exclude them.
    """

    layer: int
    directions: Float[torch.Tensor, "n_components d_model"]
    intercepts: Float[torch.Tensor, "n_components"]
    state_dim: int
    explained_variance: float
    fit_positions: Tuple[int, ...]

    def read(
        self, activations: Float[torch.Tensor, "*batch_and_pos d_model"]
    ) -> Float[torch.Tensor, "*batch_and_pos n_components"]:
        """Read state estimates out of residual activations.

        Args:
            activations: Residual-stream vectors whose last axis is ``d_model``.

        Returns:
            Predicted state components, same leading dims as ``activations``.
        """
        return activations.float() @ self.directions.T.float() + self.intercepts.float()


def _validate_model_layer(model: Any, layer: int) -> int:
    """Normalise ``layer`` and refuse hybrid blocks the probe cannot read."""
    n_layers = model.cfg.n_layers
    resolved = layer + n_layers if layer < 0 else layer
    if not 0 <= resolved < n_layers:
        raise ValueError(f"layer {layer} out of range for a {n_layers}-layer model")

    layer_types = getattr(model, "layer_types", None)
    if callable(layer_types):
        variant = layer_types()[resolved]
        parts = variant.split("+") if isinstance(variant, str) else [str(variant)]
        hybrid = [part for part in parts if part in _HYBRID_VARIANT_NAMES]
        if hybrid:
            raise NotImplementedError(
                f"layer {resolved} is a {variant} block; world-model probes read the "
                f"single-stream residual output that attention + MLP blocks provide."
            )
    return resolved


def _layer_activations(
    cache: ActivationCache, layer: int
) -> Float[torch.Tensor, "batch pos d_model"]:
    """Pull one layer's residual output out of a cache, as ``[batch, pos, d]``.

    ``run_with_cache`` on a ``HookedTransformer`` stores ``blocks.{n}.hook_resid_post``;
    on a ``TransformerBridge`` it stores the canonical ``blocks.{n}.hook_out`` (the
    ``resid_post`` alias resolves to it, but is not itself a cache key). Both names
    are accepted so the probe works against either system's cache.
    """
    for key in (("resid_post", layer), (f"blocks.{layer}.hook_out",)):
        try:
            activations = cache[key]
            break
        except KeyError:
            continue
    else:
        raise KeyError(
            f"cache has no residual output for layer {layer}; run the model with "
            f"run_with_cache (or names_filter=[{get_act_name('resid_post', layer)!r}]) "
            f"so the probe can read it"
        )
    if activations.ndim == 2:
        activations = activations.unsqueeze(0)
    return activations


def _validate_states(
    states: Float[torch.Tensor, "batch pos state_dim"],
    activations: Float[torch.Tensor, "batch pos d_model"],
) -> Float[torch.Tensor, "batch pos state_dim"]:
    """Check state labels line up with the cached positions."""
    if states.shape[:2] != activations.shape[:2]:
        raise ValueError(
            f"states must supply one label per cached position: got batch x pos "
            f"{tuple(states.shape[:2])} for activations {tuple(activations.shape[:2])}"
        )
    if states.shape[-1] < 1:
        raise ValueError("states must have at least one component on the last axis")
    return states.float()


def _flatten(
    activations: Float[torch.Tensor, "batch pos d_model"],
    states: Float[torch.Tensor, "batch pos state_dim"],
) -> Tuple[Float[torch.Tensor, "n d_model"], Float[torch.Tensor, "n state_dim"]]:
    """Flatten batch and position into a single example axis."""
    n = activations.shape[0] * activations.shape[1]
    return (
        activations.reshape(n, -1).float(),
        states.reshape(n, -1).float(),
    )


def _explained_variance(
    predicted: Float[torch.Tensor, "n state_dim"], targets: Float[torch.Tensor, "n state_dim"]
) -> float:
    """Fraction of per-component state variance the readout recovers."""
    residual_sum = float((targets - predicted).pow(2).sum())
    total_sum = float((targets - targets.mean(dim=0)).pow(2).sum())
    if total_sum <= 0.0:
        # Degenerate labels (e.g. a constant state) carry no variance to explain;
        # the readout is trivially faithful iff it reproduces the constant.
        return 1.0 if residual_sum <= 1e-6 else 0.0
    return 1.0 - residual_sum / total_sum


def fit_world_model_probe(
    cache: ActivationCache,
    layer: int,
    states: Float[torch.Tensor, "batch pos state_dim"],
    *,
    fit_positions: Optional[Sequence[int]] = None,
    ridge: float = 1e-4,
) -> WorldModelProbe:
    """Fit a linear world-model probe on cached activations.

    Solves the ridge least-squares problem ``min ||h W + b - s||²`` over the
    cached positions, then reports how much state variance that linear readout
    recovers. The paper fits its probes on a task-trained model and reads
    state out of held-out positions; fitting on any run whose states are known
    gives the same object — a linear readout of task state from the residual
    stream — and the returned probe can then be scored on *other* positions or
    runs with :func:`probe_fidelity_by_position`.

    Args:
        cache: Activations from ``run_with_cache`` on the model being probed.
        layer: Block whose output the probe reads. Negative indices count from
            ``n_layers``; hybrid (Mamba/SSM) blocks are refused.
        states: Ground-truth task state per cached position, aligned with the
            cache's ``[batch, pos]`` grid. For the Tower of Hanoi this is e.g.
            the one-hot peg occupancy of every ring at each position.
        fit_positions: Positions to fit on. Defaults to every cached position;
            the paper fits on the prompt and scores on the plan, which is
            ``fit_positions=range(0, prompt_len)`` followed by a fidelity curve
            over the rest.
        ridge: L2 regulariser. Keeps the solve well-conditioned when ``d_model``
            exceeds the number of fitted positions, which is the common case
            when fitting on a short prompt.

    Returns:
        A :class:`WorldModelProbe` with the fitted readout directions,
        intercepts, and training fidelity.

    Raises:
        KeyError: If the cache has no residual output for ``layer``.
        ValueError: If ``states`` does not align with the cached positions.
        NotImplementedError: If ``layer`` is a hybrid block.
    """
    if ridge < 0:
        raise ValueError(f"ridge must be >= 0, got {ridge}")
    resolved_layer = _validate_model_layer(cache.model, layer)
    activations = _layer_activations(cache, resolved_layer)
    states = _validate_states(states, activations)

    if fit_positions is None:
        selected = activations
        selected_states = states
        positions: Tuple[int, ...] = tuple(range(activations.shape[1]))
    else:
        positions = tuple(fit_positions)
        if not positions:
            raise ValueError("fit_positions must contain at least one index")
        normalised = [
            position + activations.shape[1] if position < 0 else position for position in positions
        ]
        out_of_range = [
            position for position in normalised if not 0 <= position < activations.shape[1]
        ]
        if out_of_range:
            raise ValueError(
                f"fit_positions {out_of_range} out of range for a cache of "
                f"{activations.shape[1]} positions"
            )
        selected = activations[:, normalised, :]
        selected_states = states[:, normalised, :]

    flat_activations, flat_states = _flatten(selected, selected_states)
    design = torch.cat([flat_activations, torch.ones(flat_activations.shape[0], 1)], dim=1)
    gram = design.T @ design
    regulariser = ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
    regulariser[-1, -1] = 0.0  # leave the intercept unpenalised
    weights = torch.linalg.solve(gram + regulariser, design.T @ flat_states)

    directions = weights[:-1].T  # [n_components, d_model]
    intercepts = weights[-1]
    predicted = flat_activations @ directions.T + intercepts
    fidelity = _explained_variance(predicted, flat_states)

    norms = directions.norm(dim=-1, keepdim=True)
    if not bool(torch.all(torch.isfinite(norms)) and bool((norms > 0).all())):
        raise ValueError(
            f"probe readout at layer {resolved_layer} collapsed to a zero or "
            "non-finite direction; check that the fitted states vary across positions"
        )

    return WorldModelProbe(
        layer=resolved_layer,
        directions=directions,
        intercepts=intercepts,
        state_dim=int(directions.shape[0]),
        explained_variance=fidelity,
        fit_positions=positions,
    )


@dataclass
class ProbeFidelity:
    """Probe fidelity scored along a sequence.

    Attributes:
        layer: The layer the probe reads.
        positions: Position indices the scores are aligned with.
        explained_variance: Fidelity at each position — high where the
            world-model representation is intact, low where it has decayed.
        window: Number of positions averaged per score.
        baseline: Fidelity on the probe's own fit positions, for reference.
    """

    layer: int
    positions: List[int]
    explained_variance: List[float]
    window: int
    baseline: float


def probe_fidelity_by_position(
    cache: ActivationCache,
    probe: WorldModelProbe,
    states: Float[torch.Tensor, "batch pos state_dim"],
    *,
    window: int = 1,
) -> ProbeFidelity:
    """Score probe fidelity at each position of a cached run.

    This is the paper's representation-decay measurement: probe the world
    model at the end of the prompt, then keep probing as the model plans. A
    curve that starts high and falls is the paper's finding — the model *has*
    the world model and loses it.

    Args:
        cache: Activations of the run to score.
        probe: A probe fitted on this model at the same layer.
        states: Ground-truth state per position, aligned with the cache.
        window: Positions averaged per score. Fidelity at a single position of
            a multi-component state can be noisy; a small window (5–20) gives a
            smoother decay curve. Must be >= 1. Each window's score is an R²
            against the run's overall state variance, so scores are comparable
            across windows and to ``probe.explained_variance``.

    Returns:
        A :class:`ProbeFidelity` whose ``explained_variance`` is aligned with
        ``positions``.

    Raises:
        KeyError: If the cache has no residual output for the probe's layer.
        ValueError: On misaligned ``states`` or a non-positive ``window``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    activations = _layer_activations(cache, probe.layer)
    states = _validate_states(states, activations)

    flat_activations, flat_states = _flatten(activations, states)
    predicted = probe.read(flat_activations)
    squared_error = (predicted - flat_states).pow(2).sum(dim=-1)
    n_positions = activations.shape[1]

    # R² against the run's own state statistics. Centring within a window would
    # make the denominator collapse to near-zero for short windows and blow the
    # score up; against run-level variance the score is a stable per-window
    # read on how much state the probe still recovers.
    run_variance = float((flat_states - flat_states.mean(dim=0)).pow(2).sum())

    scores: List[float] = []
    position_indices: List[int] = []
    start = 0
    while start < n_positions:
        end = min(start + window, n_positions)
        window_slice = slice(start * activations.shape[0], end * activations.shape[0])
        residual = float(squared_error[window_slice].sum())
        window_total = run_variance * (end - start) / n_positions
        scores.append(1.0 - residual / window_total if window_total > 0 else 1.0)
        position_indices.append(end - 1)
        start = end

    return ProbeFidelity(
        layer=probe.layer,
        positions=position_indices,
        explained_variance=scores,
        window=window,
        baseline=probe.explained_variance,
    )


def restoration_hooks(
    probe: WorldModelProbe,
    *,
    alpha: float = 1.0,
    positions: Optional[Sequence[int]] = None,
) -> List[Tuple[str, Any]]:
    """Hooks that restore the prompt-time world-model representation.

    The paper's causal intervention: the representation is intact at the end of
    the prompt and decays during planning, and re-injecting the prompt-time
    representation at inference partially recovers performance. The hook
    projects the residual stream onto the probe's world-model subspace and
    moves it toward the position it had there at prompt time:

    ``h <- h + alpha * P (h_prompt - P h)``

    with ``P`` the orthogonal projector onto the probe directions and
    ``h_prompt`` the mean prompt-time activation captured by the hook on its
    first invocation. ``alpha=0`` is a no-op and ``alpha=1`` restores the
    subspace coordinates exactly; the orthogonal complement — everything the
    probe does not read — is left untouched, so the intervention cannot simply
    overwrite the plan.

    ``h_prompt`` is captured lazily rather than passed in, so the same hooks
    work under ``model.hooks(...)`` for a single forward pass or generation.

    Args:
        probe: The fitted probe supplying the subspace and layer.
        alpha: Restoration strength; ``0`` disables, ``1`` fully restores the
            subspace coordinates.
        positions: Chunk-local positions to restore (negative indices allowed).
            Defaults to all positions of every chunk — under generation each
            new chunk restores against the same prompt-time anchor.

    Returns:
        ``[(hook_name, fn), ...]`` for ``model.hooks(fwd_hooks=...)`` or
        ``model.run_with_hooks(fwd_hooks=...)``.

    Raises:
        ValueError: If ``alpha`` is outside ``[0, 1]`` or ``positions`` is empty.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    requested = None if positions is None else tuple(positions)
    if requested == ():
        raise ValueError("positions must contain at least one index")

    directions = probe.directions.float()  # [n_components, d_model]
    # The probe directions are stored unit-normalised but not orthogonal; the
    # pseudoinverse makes this the orthogonal projector onto their span, so
    # overlapping state readouts collapse to one shared projection instead of
    # being applied (and double-counted) in sequence.
    projector = directions.T @ torch.linalg.pinv(directions.T)  # [d_model, d_model]
    d_model = int(directions.shape[1])

    prompt_anchor: List[Optional[torch.Tensor]] = [None]
    device_projectors = {}

    def hook_fn(
        activation: Float[torch.Tensor, "batch pos d_model"], hook: Any
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        del hook
        if activation.ndim != 3 or activation.shape[-1] != d_model:
            raise ValueError(
                f"restoration expects [batch, position, {d_model}], "
                f"got {tuple(activation.shape)}"
            )
        local_projector = device_projectors.get(activation.device)
        if local_projector is None:
            local_projector = projector.to(device=activation.device, dtype=torch.float32)
            device_projectors[activation.device] = local_projector

        current = activation.float()
        if prompt_anchor[0] is None:
            # First chunk this hook sees defines the prompt-time anchor: the
            # subspace coordinates the model had when the world model was intact.
            prompt_anchor[0] = (current.reshape(-1, d_model) @ local_projector.T).reshape(
                current.shape
            )
            return activation

        anchor = prompt_anchor[0].to(device=current.device)
        if anchor.shape != current.shape:
            anchor = anchor.expand_as(current) if anchor.shape[0] == 1 else anchor
            if anchor.shape != current.shape:
                # Generation re-runs with a longer chunk; project the anchor
                # rows onto the current chunk length.
                anchor = anchor[:, : current.shape[1], :]
                if anchor.shape != current.shape:
                    raise ValueError(
                        f"prompt anchor shape {tuple(anchor.shape)} does not match "
                        f"activation {tuple(current.shape)}"
                    )

        if requested is None:
            normalized = list(range(activation.shape[1]))
        else:
            normalized = [
                position + activation.shape[1] if position < 0 else position
                for position in requested
            ]
        out_of_range = [
            position for position in normalized if not 0 <= position < activation.shape[1]
        ]
        if out_of_range:
            raise ValueError(
                f"positions {out_of_range} out of range for an activation chunk of "
                f"length {activation.shape[1]}"
            )

        restored = current.clone()
        selected = current[:, normalized, :]
        projected = selected @ local_projector.T
        anchor_selected = anchor[:, normalized, :]
        restored[:, normalized, :] = selected + alpha * (anchor_selected - projected)
        return restored.to(device=activation.device, dtype=activation.dtype)

    from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

    return [(_resid_post_hook_name(probe.layer), hook_fn)]
