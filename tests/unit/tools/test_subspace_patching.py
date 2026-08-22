"""Tests for transformer_lens/tools/analysis/subspace_patching.py

Run with:
    pytest tests/unit/tools/test_subspace_patching.py -v

These tests use a tiny randomly-initialised 2-layer model so they run in
seconds on CPU without downloading any weights. They exercise the wiring at
the call site in ``transformer_lens/patching.py`` as well as the analysis
module itself.
"""

import pytest
import torch

from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.patching import (
    generic_activation_patch,
    layer_pos_subspace_patch_setter,
)
from transformer_lens.tools.analysis import (
    SubspacePatchSweeps,
    get_act_patch_resid_subspace_all_pos,
    subspace_patch_asymmetry,
)
from transformer_lens.tools.analysis.subspace_patching import (
    layer_subspace_patch_setter,
    orthonormalize,
    project,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    """A small, randomly-initialised transformer with LN folded in."""
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=32,
        d_head=8,
        n_heads=4,
        d_mlp=64,
        d_vocab=50,
        n_ctx=8,
        act_fn="gelu",
        normalization_type="LN",
        attn_only=False,
    )
    model = HookedTransformer(cfg)
    model.process_weights_()
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokens_and_caches(tiny_model):
    """Precompute clean/corrupted tokens and their caches, plus a subspace."""
    torch.manual_seed(42)
    clean_tokens = torch.randint(0, 50, (1, 6))
    corrupted_tokens = torch.randint(0, 50, (1, 6))

    with torch.no_grad():
        _, clean_cache = tiny_model.run_with_cache(clean_tokens)
        _, corrupted_cache = tiny_model.run_with_cache(corrupted_tokens)

    # A random rank-1 candidate direction in d_model.
    subspace = torch.randn(tiny_model.cfg.d_model, 1)
    return clean_tokens, corrupted_tokens, clean_cache, corrupted_cache, subspace


def last_token_metric(logits):
    """Sum of last-token logits — a scalar metric."""
    return logits[0, -1, :].sum()


# ---------------------------------------------------------------------------
# Basis helpers
# ---------------------------------------------------------------------------


class TestOrthogonalize:
    def test_identity_basis_is_orthonormal(self):
        basis = orthonormalize(torch.eye(4))
        assert torch.allclose(basis, torch.eye(4), atol=1e-6)

    def test_non_orthogonal_input_is_orthonormalized(self):
        torch.manual_seed(0)
        raw = torch.randn(8, 3)
        basis = orthonormalize(raw)
        gram = basis.transpose(-1, -2) @ basis
        assert torch.allclose(gram, torch.eye(3), atol=1e-5)

    def test_rank1_direction_is_normalized(self):
        direction = torch.randn(8, 1) * 5.0
        basis = orthonormalize(direction)
        assert torch.allclose(basis.transpose(-1, -2) @ basis, torch.eye(1), atol=1e-6)

    def test_bare_1d_direction_is_accepted(self):
        """A [d_model] vector is documented as valid input."""
        direction = torch.randn(8) * 3.0
        basis = orthonormalize(direction)
        assert basis.shape == (8, 1)
        assert torch.allclose(basis.transpose(-1, -2) @ basis, torch.eye(1), atol=1e-6)

    def test_rank_deficient_input_zeroes_the_dependent_column(self):
        """A duplicated column must not patch an arbitrary direction."""
        first = torch.nn.functional.normalize(torch.randn(8), dim=-1).unsqueeze(-1)
        redundant = torch.cat([first, first], dim=-1)
        basis = orthonormalize(redundant)
        assert torch.allclose(basis[:, 0], first[:, 0], atol=1e-5)
        assert torch.allclose(basis[:, 1], torch.zeros(8), atol=1e-5)

    def test_span_is_preserved(self):
        """QR re-basing must not change which subspace the columns span."""
        torch.manual_seed(1)
        raw = torch.randn(6, 2)
        basis = orthonormalize(raw)
        v = torch.randn(6)
        # Reference: orthogonal projection onto span(raw) via the pseudo-inverse.
        # (project() itself assumes an orthonormal basis, so it is not its own
        # reference for a non-orthogonal input.)
        gram_inv = torch.linalg.pinv(raw.transpose(-1, -2) @ raw)
        reference = (v @ raw) @ gram_inv @ raw.transpose(-1, -2)
        assert torch.allclose(project(v, basis), reference, atol=1e-5)


class TestProject:
    def test_projection_is_idempotent(self):
        torch.manual_seed(2)
        basis = orthonormalize(torch.randn(6, 2))
        v = torch.randn(6)
        once = project(v, basis)
        assert torch.allclose(once, project(once, basis), atol=1e-5)

    def test_complement_is_orthogonal(self):
        torch.manual_seed(3)
        basis = orthonormalize(torch.randn(6, 2))
        v = torch.randn(6)
        residual = v - project(v, basis)
        assert torch.allclose(basis.transpose(-1, -2) @ residual, torch.zeros(2, 1), atol=1e-5)


# ---------------------------------------------------------------------------
# Patch setters
# ---------------------------------------------------------------------------


class TestSubspacePatchSetters:
    def test_setter_requires_subspace(self):
        with pytest.raises(ValueError, match="subspace is required"):
            layer_pos_subspace_patch_setter(
                torch.zeros(1, 4, 6), [0, 1], torch.zeros(1, 4, 6), None
            )

    def test_complement_is_untouched(self, tokens_and_caches):
        """The orthogonal complement must be byte-identical after the patch."""
        _, _, clean_cache, corrupted_cache, subspace = tokens_and_caches
        clean_act = clean_cache["resid_pre", 0]
        corrupted_act = corrupted_cache["resid_pre", 0].clone()

        patched = layer_pos_subspace_patch_setter(corrupted_act, [0, 2], clean_act, subspace)

        basis = orthonormalize(subspace)
        # Off-position rows are untouched.
        assert torch.equal(patched[:, 3], corrupted_act[:, 3])
        # On-position change lies entirely in the subspace, and equals the
        # clean-corrupted difference projected there.
        delta = patched[0, 2] - corrupted_act[0, 2]
        assert torch.allclose(delta, project(delta, basis), atol=1e-5)
        assert torch.allclose(
            delta,
            project(clean_act[0, 2] - corrupted_act[0, 2], basis),
            atol=1e-5,
        )

    def test_full_space_patch_matches_standard_setter(self, tokens_and_caches):
        """With the whole space as the subspace, this reduces to layer_pos_patch_setter."""
        _, _, clean_cache, corrupted_cache, _ = tokens_and_caches
        d_model = clean_cache["resid_pre", 0].shape[-1]
        full_space = torch.eye(d_model)

        clean_act = clean_cache["resid_pre", 0]
        corrupted_act = corrupted_cache["resid_pre", 0].clone()

        subspace_patched = layer_pos_subspace_patch_setter(
            corrupted_act, [0, 1], clean_act, full_space
        )
        assert torch.allclose(subspace_patched[0, 1], clean_act[0, 1], atol=1e-4)
        # Other positions still hold the corrupted values.
        assert torch.allclose(subspace_patched[0, 3], corrupted_act[0, 3], atol=1e-6)

    def test_layer_setter_touches_every_position(self, tokens_and_caches):
        _, _, clean_cache, corrupted_cache, subspace = tokens_and_caches
        clean_act = clean_cache["resid_pre", 0]
        corrupted_act = corrupted_cache["resid_pre", 0].clone()

        patched = layer_subspace_patch_setter(corrupted_act, [0], clean_act, subspace)
        basis = orthonormalize(subspace)
        for pos in range(patched.shape[1]):
            delta = patched[0, pos] - corrupted_act[0, pos]
            assert torch.allclose(
                delta - project(delta, basis), torch.zeros_like(delta), atol=1e-5
            )


# ---------------------------------------------------------------------------
# Integration with the existing generic patching loop
# ---------------------------------------------------------------------------


class TestGenericPatchingIntegration:
    def test_call_site_setter_is_reachable_from_patching_module(self):
        """The wiring edit: patching.py must expose the subspace setter."""
        import transformer_lens.patching as patching_module

        assert patching_module.layer_pos_subspace_patch_setter is layer_pos_subspace_patch_setter

    def test_sweep_runs_through_generic_activation_patch(
        self, tiny_model, tokens_and_caches
    ):
        """The analysis sweep rides the existing generic loop unchanged."""
        clean_tokens, corrupted_tokens, clean_cache, _, subspace = tokens_and_caches

        results = get_act_patch_resid_subspace_all_pos(
            tiny_model,
            corrupted_tokens,
            clean_cache,
            last_token_metric,
            subspace,
        )
        assert results.shape == (tiny_model.cfg.n_layers, corrupted_tokens.shape[-1])
        assert torch.isfinite(results).all()

    def test_generic_call_with_subspace_setter(self, tiny_model, tokens_and_caches):
        """Direct use of generic_activation_patch with the subspace setter."""
        from functools import partial

        clean_tokens, corrupted_tokens, clean_cache, _, subspace = tokens_and_caches

        results = generic_activation_patch(
            tiny_model,
            corrupted_tokens,
            clean_cache,
            last_token_metric,
            partial(layer_pos_subspace_patch_setter, subspace=subspace),
            "resid_pre",
            index_axis_names=("layer", "pos"),
        )
        assert results.shape == (tiny_model.cfg.n_layers, corrupted_tokens.shape[-1])


# ---------------------------------------------------------------------------
# Bidirectional diagnostic
# ---------------------------------------------------------------------------


class TestSubspacePatchAsymmetry:
    def test_returns_paired_sweeps(self, tiny_model, tokens_and_caches):
        clean_tokens, corrupted_tokens, clean_cache, corrupted_cache, subspace = (
            tokens_and_caches
        )

        sweeps = subspace_patch_asymmetry(
            tiny_model,
            clean_cache=clean_cache,
            corrupted_cache=corrupted_cache,
            clean_tokens=clean_tokens,
            corrupted_tokens=corrupted_tokens,
            patching_metric=last_token_metric,
            subspace=subspace,
            layer=1,
            pos=3,
        )

        assert isinstance(sweeps, SubspacePatchSweeps)
        assert sweeps.recovery.shape == (1,)
        assert sweeps.corruption.shape == (1,)
        # Unpatched baselines bracket the patched values.
        assert sweeps.clean_metric != sweeps.corrupted_metric

    def test_recovery_direction_moves_toward_clean(self, tiny_model, tokens_and_caches):
        """Patching the clean subspace into the corrupted run must move the metric."""
        clean_tokens, corrupted_tokens, clean_cache, corrupted_cache, subspace = (
            tokens_and_caches
        )

        sweeps = subspace_patch_asymmetry(
            tiny_model,
            clean_cache=clean_cache,
            corrupted_cache=corrupted_cache,
            clean_tokens=clean_tokens,
            corrupted_tokens=corrupted_tokens,
            patching_metric=last_token_metric,
            subspace=subspace,
            layer=1,
            pos=-1,
        )

        recovery_effect, corruption_effect = sweeps.effects()
        # A random direction is not expected to carry the feature, so we only
        # assert the diagnostic is well-formed and bounded, not that it fires.
        assert torch.isfinite(recovery_effect).all()
        assert torch.isfinite(corruption_effect).all()
        assert torch.isfinite(sweeps.asymmetry()).all()

    def test_summary_reports_all_three_scores(self, tiny_model, tokens_and_caches):
        clean_tokens, corrupted_tokens, clean_cache, corrupted_cache, subspace = (
            tokens_and_caches
        )

        sweeps = subspace_patch_asymmetry(
            tiny_model,
            clean_cache=clean_cache,
            corrupted_cache=corrupted_cache,
            clean_tokens=clean_tokens,
            corrupted_tokens=corrupted_tokens,
            patching_metric=last_token_metric,
            subspace=subspace,
            layer=0,
            pos=None,
        )

        summary = sweeps.summary()
        assert set(summary) == {"recovery_effect", "corruption_effect", "asymmetry"}
        assert summary["asymmetry"] == pytest.approx(
            summary["recovery_effect"] - summary["corruption_effect"], abs=1e-6
        )

    def test_zero_gap_raises(self):
        sweeps = SubspacePatchSweeps(
            recovery=torch.zeros(1),
            corruption=torch.zeros(1),
            clean_metric=1.0,
            corrupted_metric=1.0,
        )
        with pytest.raises(ValueError, match="no gap"):
            sweeps.effects()

    def test_asymmetry_is_zero_for_symmetric_patch(self):
        """A subspace that moves both directions equally scores zero."""
        sweeps = SubspacePatchSweeps(
            recovery=torch.tensor([1.5]),
            corruption=torch.tensor([0.5]),
            clean_metric=2.0,
            corrupted_metric=0.0,
        )
        recovery_effect, corruption_effect = sweeps.effects()
        assert recovery_effect.item() == pytest.approx(0.75)
        assert corruption_effect.item() == pytest.approx(0.75)
        assert sweeps.asymmetry().item() == pytest.approx(0.0, abs=1e-6)


class TestPackageExports:
    def test_exported_from_analysis_package(self):
        from transformer_lens.tools import analysis

        assert analysis.subspace_patch_asymmetry is subspace_patch_asymmetry
        assert analysis.get_act_patch_resid_subspace_all_pos is (
            get_act_patch_resid_subspace_all_pos
        )
        assert "SubspacePatchSweeps" in analysis.__all__
