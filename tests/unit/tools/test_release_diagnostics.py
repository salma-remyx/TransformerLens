"""Unit tests for the release-diagnostics audits over JacobianLens steering."""

from typing import Any

import pytest
import torch

from transformer_lens.tools.analysis.jacobian_lens import JacobianLens
from transformer_lens.tools.analysis.release_diagnostics import (
    dose_response_sweep,
    gate_fire_audit,
)

from tests.unit.tools.test_jacobian_lens import _ToyBridge

D_MODEL = 6
N_LAYERS = 4
D_VOCAB = 11
SEQ_LEN = 9
ALPHAS = [0.0, 1.0, 2.0, 4.0, 8.0]


@pytest.fixture(scope="module")
def toy_model() -> _ToyBridge:
    return _ToyBridge()


@pytest.fixture(scope="module")
def identity_lens() -> JacobianLens:
    return JacobianLens({0: torch.eye(D_MODEL)}, n_prompts=1, d_model=D_MODEL)


def test_dose_response_sweep_reports_monotone_unbounded_growth(
    toy_model: _ToyBridge, identity_lens: JacobianLens
) -> None:
    """Steering along a real lens direction grows the target logit with alpha."""
    report = dose_response_sweep(
        identity_lens,
        toy_model,
        "a toy prompt",
        3,
        layers=[0],
        alphas=ALPHAS,
        positions=[-1],
    )
    assert report.alphas == ALPHAS
    assert len(report.effects) == len(ALPHAS)
    assert report.gains[0] == pytest.approx(0.0, abs=1e-4)
    assert report.gains == sorted(report.gains)
    assert report.is_monotone is True
    assert report.early_slope > 0.0
    # A truly linear toy keeps growing: not flagged as a plateaued release.
    assert report.plateau_fraction > 0.25
    assert report.is_bounded is False
    assert "unbounded linear release" in report.summary()


def test_dose_response_sweep_flags_bounded_release_against_margin(
    toy_model: _ToyBridge, identity_lens: JacobianLens
) -> None:
    """A preregistered margin above the reachable gain marks the release bounded."""
    report = dose_response_sweep(
        identity_lens,
        toy_model,
        "a toy prompt",
        3,
        layers=[0],
        alphas=ALPHAS,
        positions=[-1],
        release_margin=report_gain(toy_model, identity_lens) * 10.0,
    )
    assert report.is_bounded is True
    assert "bounded linear release" in report.summary()
    assert report.summary().startswith("bounded")


def report_gain(model: _ToyBridge, lens: JacobianLens) -> float:
    probe = dose_response_sweep(
        lens, model, "a toy prompt", 3, layers=[0], alphas=[8.0], positions=[-1]
    )
    return probe.gains[0]


def test_dose_response_sweep_zero_alpha_matches_baseline(
    toy_model: _ToyBridge, identity_lens: JacobianLens
) -> None:
    """alpha=0 disables the hook, so the measured effect equals the baseline."""
    report = dose_response_sweep(
        identity_lens, toy_model, "a toy prompt", 3, layers=[0], alphas=[0.0], positions=[-1]
    )
    assert report.effects[0] == pytest.approx(report.baseline_effect, abs=1e-5)
    assert report.gains[0] == pytest.approx(0.0, abs=1e-5)


def test_dose_response_sweep_rejects_empty_alphas(
    toy_model: _ToyBridge, identity_lens: JacobianLens
) -> None:
    with pytest.raises(ValueError, match="alphas"):
        dose_response_sweep(identity_lens, toy_model, "a toy prompt", 3, layers=[0], alphas=[])


def test_gate_fire_audit_detects_inverted_gate(
    toy_model: _ToyBridge, identity_lens: JacobianLens
) -> None:
    """A gate that fires only where the intervention is not needed is inverted."""
    prompts = ["a toy prompt", "another toy prompt"]

    def make_hooks(_prompt: str) -> list[tuple[str, Any]]:
        return identity_lens.steering_hooks(toy_model, 3, layers=[0], alpha=2.0)

    report = gate_fire_audit(
        toy_model,
        prompts,
        make_hooks,
        gate=lambda prompt: "another" in prompt,
        needs_intervention=[True, False],
    )
    assert report.n_prompts == 2
    assert report.fire_rate == pytest.approx(0.5)
    assert report.effective_rate == pytest.approx(0.5)
    assert report.fire_on_needed == pytest.approx(0.0)
    assert report.fire_on_not_needed == pytest.approx(1.0)
    assert report.inverted is True
    assert "gate inverted" in report.summary()


def test_gate_fire_audit_detects_silent_noops(
    toy_model: _ToyBridge, identity_lens: JacobianLens
) -> None:
    """A gate whose fired intervention cannot change the output is a silent no-op."""

    def noop_hooks(_prompt: str) -> list[tuple[str, Any]]:
        # alpha=0 leaves the residual stream untouched.
        return identity_lens.steering_hooks(toy_model, 3, layers=[0], alpha=0.0)

    report = gate_fire_audit(
        toy_model,
        ["a toy prompt"],
        noop_hooks,
        gate=lambda _prompt: True,
    )
    assert report.fire_rate == pytest.approx(1.0)
    assert report.effective_rate == pytest.approx(0.0)
    assert report.silent_noops == [0]
    assert report.reduced_to_base_model is True
    assert "silently reduced to base model" in report.summary()


def test_gate_fire_audit_aligned_gate_is_not_inverted(
    toy_model: _ToyBridge, identity_lens: JacobianLens
) -> None:
    def make_hooks(_prompt: str) -> list[tuple[str, Any]]:
        return identity_lens.steering_hooks(toy_model, 3, layers=[0], alpha=2.0)

    report = gate_fire_audit(
        toy_model,
        ["a toy prompt", "another toy prompt"],
        make_hooks,
        gate=lambda prompt: "another" not in prompt,
        needs_intervention=[True, False],
    )
    assert report.inverted is False
    assert report.fire_on_needed == pytest.approx(1.0)
    assert report.fire_on_not_needed == pytest.approx(0.0)


def test_gate_fire_audit_rejects_misaligned_labels(
    toy_model: _ToyBridge, identity_lens: JacobianLens
) -> None:
    with pytest.raises(ValueError, match="align"):
        gate_fire_audit(
            toy_model,
            ["a toy prompt"],
            lambda _prompt: [],
            gate=lambda _prompt: True,
            needs_intervention=[True, False],
        )
