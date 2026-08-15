"""Release diagnostics for linear steering interventions.

"Located" does not imply "releasable". A latent direction can be localized
correctly and still fail to convert into behavior when injected linearly.
This module surfaces the two failure modes of the detect-localize-release
pipeline as small audits anchored on the existing JacobianLens
interventions (:meth:`~transformer_lens.tools.analysis.jacobian_lens.JacobianLens.steering_hooks`
and ``swap_hooks``):

- **Bounded linear release** — the dose-response of a linearly injected
  direction can be monotone yet plateau far below the behavior change the
  experimenter intended. :func:`dose_response_sweep` sweeps the
  intervention strength ``alpha`` produced by ``steering_hooks``, measures
  the induced target-logit change, and reports monotonicity, the
  late-versus-early slope ratio, and whether the curve stays bounded away
  from a preregistered release margin.
- **Silent gate inversion** — a detector deciding *when* to intervene can
  fire on the wrong distribution and silently reduce the gated pipeline to
  the base model. :func:`gate_fire_audit` scores the gate's decisions
  against optional ground-truth "needs intervention" labels and against
  whether each fired intervention actually changed the model output.

Both audits are parameter-free diagnostics: they reuse the model's own
forward pass and the lens's own hooks, and add no learned components.

Adapted from the stress-test framing of `Located but Not Releasable:
Silent Gate Inversion and Bounded Linear Release
<https://arxiv.org/abs/2608.11822>`_ (arXiv:2608.11822). The paper's
preregistered 25.7M-parameter pipeline is not reproduced; the two
failure-mode detectors are re-expressed against TransformerLens
steering interventions.

Example::

    from transformer_lens.tools.analysis.release_diagnostics import (
        dose_response_sweep,
    )

    report = dose_response_sweep(
        lens, model, "The Eiffel Tower is in the city of", " Paris",
        layers=[8], alphas=[0.5, 1.0, 2.0, 4.0, 8.0],
        release_margin=2.0,
    )
    print(report.summary())
"""

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch

from transformer_lens.tools.analysis.jacobian_lens import JacobianLens, _to_token_ids

TokenInput = Union[str, int]
HookList = List[Tuple[str, Any]]

_DEFAULT_PLATEAU_FRACTION = 0.25


def _resolve_target_ids(
    model: Any, target_tokens: Optional[Union[TokenInput, Sequence[TokenInput]]], token: TokenInput
) -> List[int]:
    """Resolve the effect-measurement tokens, defaulting to the steered token."""
    return _to_token_ids(model, target_tokens if target_tokens is not None else token)


def _target_logit(
    logits: torch.Tensor, target_ids: Sequence[int], positions: Sequence[int]
) -> float:
    """Mean logit of the target tokens at the measured positions."""
    selected = logits[0, list(positions), :].float()
    return float(selected[..., list(target_ids)].mean().item())


def _slope(alphas: Sequence[float], gains: Sequence[float], start: int, end: int) -> float:
    """Secant slope of the gain curve between two sweep indices."""
    span = alphas[end] - alphas[start]
    if span == 0.0:
        return 0.0
    return (gains[end] - gains[start]) / span


@dataclass
class DoseResponseReport:
    """Result of a :func:`dose_response_sweep`.

    Attributes:
        alphas: The swept intervention strengths, in the given order.
        effects: Target-token logit at the measured positions for each
            alpha (and for the alpha=0 baseline run, last).
        gains: ``effects - baseline_effect``: the released behavior change.
        is_monotone: Whether the gains never decrease beyond ``atol``.
        early_slope: Secant slope of the gains over the first half of the sweep.
        late_slope: Secant slope of the gains over the second half.
        plateau_fraction: ``late_slope / early_slope``; values near zero mean
            the curve has plateaued (further alpha buys almost nothing).
        release_margin: Optional preregistered gain the release must reach.
        is_bounded: Whether the release is bounded: the final gain stays
            below ``release_margin`` when given, else whether the curve has
            plateaued (``plateau_fraction`` below the audit's threshold).
        token: The steered concept token.
        layers: The layers intervened at.
    """

    alphas: List[float]
    effects: List[float]
    baseline_effect: float
    gains: List[float]
    is_monotone: bool
    early_slope: float
    late_slope: float
    plateau_fraction: float
    release_margin: Optional[float]
    is_bounded: bool
    token: Union[str, int]
    layers: List[int]

    def summary(self) -> str:
        """One-line verdict in the paper's vocabulary."""
        margin = (
            f"margin {self.release_margin:g} not reached"
            if self.release_margin is not None
            else f"plateau_fraction {self.plateau_fraction:.3f}"
        )
        return (
            f"{'bounded' if self.is_bounded else 'unbounded'} linear release: "
            f"gain {self.gains[-1]:.3f} at alpha={self.alphas[-1]:g} "
            f"({'monotone' if self.is_monotone else 'non-monotone'}, {margin})"
        )


@dataclass
class GateFireReport:
    """Result of a :func:`gate_fire_audit`.

    Attributes:
        n_prompts: Number of audited prompts.
        fire_rate: Fraction of prompts on which the gate fired.
        effective_rate: Fraction of prompts where the gate fired *and* the
            intervention changed the model output beyond ``atol``.
        silent_noops: Indices of prompts where the gate fired but the
            output was indistinguishable from the base model.
        fire_on_needed: If labels were given, the gate's fire rate on
            prompts that actually need the intervention.
        fire_on_not_needed: If labels were given, the gate's fire rate on
            prompts that do not.
        inverted: If labels were given, whether the gate fires only where
            it should not and never where it should — the paper's complete
            silent inversion.
    """

    n_prompts: int
    fire_rate: float
    effective_rate: float
    silent_noops: List[int] = field(default_factory=list)
    fire_on_needed: Optional[float] = None
    fire_on_not_needed: Optional[float] = None
    inverted: Optional[bool] = None

    @property
    def reduced_to_base_model(self) -> bool:
        """True when every fired intervention was a silent no-op."""
        return self.fire_rate > 0.0 and self.effective_rate == 0.0

    def summary(self) -> str:
        """One-line verdict in the paper's vocabulary."""
        parts = [f"fire_rate {self.fire_rate:.3f}", f"effective_rate {self.effective_rate:.3f}"]
        if self.inverted is not None:
            parts.append("gate inverted" if self.inverted else "gate not inverted")
        elif self.reduced_to_base_model:
            parts.append("silently reduced to base model")
        return "gate audit: " + ", ".join(parts)


def dose_response_sweep(
    lens: JacobianLens,
    model: Any,
    prompt: str,
    token: TokenInput,
    layers: Sequence[int],
    alphas: Sequence[float],
    *,
    target_tokens: Optional[Union[TokenInput, Sequence[TokenInput]]] = None,
    positions: Optional[Sequence[int]] = None,
    release_margin: Optional[float] = None,
    atol: float = 1e-4,
) -> DoseResponseReport:
    """Sweep steering strength and audit the shape of the release.

    Runs the model on ``prompt`` under :meth:`JacobianLens.steering_hooks`
    at each alpha in ``alphas`` and measures the mean logit of the target
    token(s) at the measured positions. A monotone curve that flattens —
    or that never reaches ``release_margin`` — is the paper's bounded
    linear release: the direction is injected, the response saturates,
    and the intended behavior change is not delivered.

    Args:
        lens: The fitted lens supplying the steering direction.
        model: A raw ``TransformerBridge`` accepted by the lens.
        prompt: The prompt to steer on.
        token: The concept token to steer toward.
        layers: Layers to intervene at.
        alphas: Steering strengths to sweep, in increasing order is
            conventional but not required.
        target_tokens: Token(s) whose logits measure the released effect.
            Defaults to ``token`` itself.
        positions: Positions to measure (negative indices allowed).
            Defaults to the final position.
        release_margin: Optional preregistered gain the release must reach
            to count as sufficient.
        atol: Tolerance for the monotonicity check.

    Returns:
        A :class:`DoseResponseReport`.
    """
    if not alphas:
        raise ValueError("alphas must contain at least one strength")
    swept = [float(alpha) for alpha in alphas]
    target_ids = _resolve_target_ids(model, target_tokens, token)
    measured = list(positions) if positions is not None else [-1]
    tokens = model.to_tokens(prompt)

    base_logits = model(tokens)
    if base_logits is None:
        raise ValueError("the model forward pass returned no logits")
    baseline_effect = _target_logit(base_logits, target_ids, measured)

    effects: List[float] = []
    for alpha in swept:
        hooks = lens.steering_hooks(model, token, layers, alpha=alpha, positions=measured)
        with model.hooks(fwd_hooks=hooks):
            logits = model(tokens)
        if logits is None:
            raise ValueError("the model forward pass returned no logits")
        effects.append(_target_logit(logits, target_ids, measured))

    gains = [effect - baseline_effect for effect in effects]
    is_monotone = all(gains[i + 1] >= gains[i] - atol for i in range(len(gains) - 1))
    midpoint = len(swept) // 2
    early_slope = _slope(swept, gains, 0, midpoint)
    late_slope = _slope(swept, gains, midpoint, len(swept) - 1)
    plateau_fraction = late_slope / early_slope if early_slope > 0 else float("inf")
    is_bounded = (
        gains[-1] < release_margin
        if release_margin is not None
        else plateau_fraction < _DEFAULT_PLATEAU_FRACTION
    )
    return DoseResponseReport(
        alphas=swept,
        effects=effects,
        baseline_effect=baseline_effect,
        gains=gains,
        is_monotone=is_monotone,
        early_slope=early_slope,
        late_slope=late_slope,
        plateau_fraction=plateau_fraction,
        release_margin=release_margin,
        is_bounded=is_bounded,
        token=token,
        layers=list(layers),
    )


def gate_fire_audit(
    model: Any,
    prompts: Sequence[str],
    make_hooks: Callable[[str], HookList],
    gate: Callable[[str], bool],
    *,
    needs_intervention: Optional[Sequence[bool]] = None,
    atol: float = 1e-4,
) -> GateFireReport:
    """Audit a detector that decides when to apply an intervention.

    For each prompt, records whether the gate fired, whether the gated
    pipeline's output actually differs from the base model's, and — when
    ground-truth ``needs_intervention`` labels are supplied — whether the
    gate fires on the prompts that need it. A gate that fires only where
    it should not, or whose fired interventions never change the output,
    silently reduces the pipeline to the base model: the paper's silent
    gate inversion.

    Args:
        model: A model exposing ``to_tokens`` and ``hooks``.
        prompts: Prompts to audit.
        make_hooks: Builds the intervention hooks for one prompt (e.g. a
            closure over ``lens.steering_hooks``).
        gate: The detector under audit; returns True to intervene.
        needs_intervention: Optional per-prompt ground truth labels.
        atol: Outputs closer than this to the base model count as
            unchanged.

    Returns:
        A :class:`GateFireReport`.
    """
    if not prompts:
        raise ValueError("prompts must contain at least one prompt")
    if needs_intervention is not None and len(needs_intervention) != len(prompts):
        raise ValueError("needs_intervention must align with prompts")

    fired: List[bool] = []
    effective: List[bool] = []
    silent_noops: List[int] = []
    for index, prompt in enumerate(prompts):
        tokens = model.to_tokens(prompt)
        base_logits = model(tokens)
        if base_logits is None:
            raise ValueError("the model forward pass returned no logits")
        did_fire = bool(gate(prompt))
        fired.append(did_fire)
        hooks = make_hooks(prompt) if did_fire else []
        with model.hooks(fwd_hooks=hooks):
            gated_logits = model(tokens)
        if gated_logits is None:
            raise ValueError("the model forward pass returned no logits")
        changed = not torch.allclose(gated_logits.float(), base_logits.float(), atol=atol)
        effective.append(did_fire and changed)
        if did_fire and not changed:
            silent_noops.append(index)

    fire_rate = sum(fired) / len(prompts)
    effective_rate = sum(effective) / len(prompts)
    fire_on_needed: Optional[float] = None
    fire_on_not_needed: Optional[float] = None
    inverted: Optional[bool] = None
    if needs_intervention is not None:
        needed = [(fired[index], bool(label)) for index, label in enumerate(needs_intervention)]
        needed_fires = [fire for fire, label in needed if label]
        unneeded_fires = [fire for fire, label in needed if not label]
        fire_on_needed = sum(needed_fires) / len(needed_fires) if needed_fires else None
        fire_on_not_needed = (
            sum(unneeded_fires) / len(unneeded_fires) if unneeded_fires else None
        )
        inverted = bool(fire_on_not_needed) and not bool(fire_on_needed)
    return GateFireReport(
        n_prompts=len(prompts),
        fire_rate=fire_rate,
        effective_rate=effective_rate,
        silent_noops=silent_noops,
        fire_on_needed=fire_on_needed,
        fire_on_not_needed=fire_on_not_needed,
        inverted=inverted,
    )
