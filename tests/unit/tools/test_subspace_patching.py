"""Tests for transformer_lens/tools/analysis/subspace_patching.py

Run with:
    pytest tests/unit/tools/test_subspace_patching.py -v

These tests use a tiny randomly-initialised model so they run in seconds on CPU
without downloading any weights.
"""

from functools import partial

import pandas as pd
import pytest
import torch

import transformer_lens.utilities as utils
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.patching import generic_activation_patch
from transformer_lens.tools.analysis.subspace_patching import (
    SubspacePatchReport,
    fractional_logit_diff_decrease,
    nullspace_rowspace_decomposition,
    project_onto_subspace,
    projection_spread,
    subspace_patch_faithfulness,
    subspace_patch_setter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    """A small, randomly-initialised transformer."""
    cfg = HookedTransformerConfig(
        n_layers=3,
        d_model=32,
        d_head=8,
        n_heads=4,
        d_mlp=96,
        d_vocab=50,
        n_ctx=8,
        act_fn="gelu",
        normalization_type="LN",
        attn_only=False,
    )
    model = HookedTransformer(cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokens_and_caches(tiny_model):
    """Precompute clean/corrupted tokens and the clean cache."""
    torch.manual_seed(42)
    clean_tokens = torch.randint(0, 50, (1, 6))
    corrupted_tokens = torch.randint(0, 50, (1, 6))
    with torch.no_grad():
        _, clean_cache = tiny_model.run_with_cache(clean_tokens)
    return clean_tokens, corrupted_tokens, clean_cache


def logit_diff(logits):
    """A logit-difference patching metric, as used in the paper."""
    return logits[0, -1, 7] - logits[0, -1, 11]


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------


class TestProjectOntoSubspace:
    def test_projection_and_complement_sum_to_identity(self):
        """P_U(x) + P_{U^⊥}(x) = x, and P_U is idempotent."""
        torch.manual_seed(0)
        basis = torch.linalg.qr(torch.randn(16, 3), mode="reduced")[0]
        x = torch.randn(5, 16)

        projection = project_onto_subspace(x, basis)
        complement = project_onto_subspace(x, basis, complement=True)

        assert torch.allclose(projection + complement, x, atol=1e-5)
        assert torch.allclose(project_onto_subspace(projection, basis), projection, atol=1e-5)

    def test_projection_is_in_the_subspace(self):
        """P_U(x) has no component outside the span of the basis."""
        torch.manual_seed(0)
        basis = torch.linalg.qr(torch.randn(16, 3), mode="reduced")[0]
        x = torch.randn(16)
        projection = project_onto_subspace(x, basis)
        # Residual against the basis must vanish.
        assert torch.allclose(projection @ basis, x @ basis, atol=1e-5)

    def test_higher_rank_subspace(self):
        """A 4-D basis projects onto all 4 dimensions, and a vector already in
        the span is a fixed point."""
        torch.manual_seed(1)
        basis = torch.linalg.qr(torch.randn(12, 4), mode="reduced")[0]
        in_span = basis @ torch.randn(4)  # [12], lies in the span
        assert torch.allclose(project_onto_subspace(in_span, basis), in_span, atol=1e-5)


# ---------------------------------------------------------------------------
# subspace_patch_setter (Eq. 1)
# ---------------------------------------------------------------------------


class TestSubspacePatchSetter:
    def test_matches_hand_computed_eq1(self, tiny_model, tokens_and_caches):
        """The setter reproduces act_patched = act_corr + (v·a_c - v·a_corr) v
        exactly, when routed through generic_activation_patch."""
        clean_tokens, corrupted_tokens, clean_cache = tokens_and_caches
        layer, pos = 1, 4
        act_name = utils.get_act_name("post", layer=layer)

        torch.manual_seed(5)
        v = torch.randn(tiny_model.cfg.d_mlp)
        v = v / v.norm()

        clean_post = clean_cache[act_name]

        def manual_eq1_hook(value, hook):
            c = clean_post[:, pos, :]
            v_d = v.to(value.dtype)
            patched = value[:, pos, :] + ((c - value[:, pos, :]) @ v_d).unsqueeze(-1) * v_d
            value = value.clone()
            value[:, pos, :] = patched
            return value

        with torch.no_grad():
            ref_logits = tiny_model.run_with_hooks(
                corrupted_tokens, fwd_hooks=[(act_name, manual_eq1_hook)]
            )
            reference = logit_diff(ref_logits).item()

            got = generic_activation_patch(
                tiny_model,
                corrupted_tokens,
                clean_cache,
                logit_diff,
                partial(subspace_patch_setter, direction=v),
                "post",
                index_df=pd.DataFrame([{"layer": layer, "pos": pos}]),
            ).item()

        assert got == pytest.approx(reference, abs=1e-6)

    def test_pure_nullspace_direction_is_a_no_op(self, tiny_model, tokens_and_caches):
        """The paper's core claim: patching a causally disconnected direction
        (inside ker(W_out)) leaves the model output bit-identical to the
        unpatched corrupted run."""
        clean_tokens, corrupted_tokens, clean_cache = tokens_and_caches
        layer, pos = 1, 4

        w_out = tiny_model.blocks[layer].mlp.W_out.detach()
        u, _, _ = torch.linalg.svd(w_out, full_matrices=True)
        v_null = u[:, tiny_model.cfg.d_model :][:, 0]

        with torch.no_grad():
            baseline_logits = tiny_model(corrupted_tokens)
            baseline = logit_diff(baseline_logits).item()

            got = generic_activation_patch(
                tiny_model,
                corrupted_tokens,
                clean_cache,
                logit_diff,
                partial(subspace_patch_setter, direction=v_null),
                "post",
                index_df=pd.DataFrame([{"layer": layer, "pos": pos}]),
            ).item()

        assert got == pytest.approx(baseline, abs=1e-5)

    def test_basis_and_direction_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="exactly one"):
            subspace_patch_setter(
                torch.zeros(1, 4, 8), [0, 1], torch.zeros(1, 4, 8)
            )
        with pytest.raises(ValueError, match="exactly one"):
            subspace_patch_setter(
                torch.zeros(1, 4, 8),
                [0, 1],
                torch.zeros(1, 4, 8),
                basis=torch.eye(8)[:, :2],
                direction=torch.ones(8),
            )

    def test_multi_dimensional_basis(self, tiny_model, tokens_and_caches):
        """A k-D basis patches the projection onto the whole span (App. A.1)."""
        clean_tokens, corrupted_tokens, clean_cache = tokens_and_caches
        layer, pos = 1, 4
        act_name = utils.get_act_name("post", layer=layer)
        clean_post = clean_cache[act_name]

        torch.manual_seed(6)
        basis = torch.linalg.qr(
            torch.randn(tiny_model.cfg.d_mlp, 5), mode="reduced"
        )[0]

        def manual_hook(value, hook):
            c = clean_post[:, pos, :]
            b = basis.to(value.dtype)
            delta = ((c - value[:, pos, :]) @ b) @ b.T
            value = value.clone()
            value[:, pos, :] = value[:, pos, :] + delta
            return value

        with torch.no_grad():
            ref_logits = tiny_model.run_with_hooks(
                corrupted_tokens, fwd_hooks=[(act_name, manual_hook)]
            )
            reference = logit_diff(ref_logits).item()
            got = generic_activation_patch(
                tiny_model,
                corrupted_tokens,
                clean_cache,
                logit_diff,
                partial(subspace_patch_setter, basis=basis),
                "post",
                index_df=pd.DataFrame([{"layer": layer, "pos": pos}]),
            ).item()

        assert got == pytest.approx(reference, abs=1e-6)


# ---------------------------------------------------------------------------
# nullspace / rowspace decomposition
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tall_map():
    torch.manual_seed(0)
    return torch.randn(96, 32)  # W_out-shaped: [d_mlp, d_model]


class TestNullspaceRowspaceDecomposition:
    def test_decomposition_is_a_partition(self, tall_map):
        """v_null + v_row = v and the two parts are orthogonal."""
        v = torch.randn(96)
        v_null, v_row, _ = nullspace_rowspace_decomposition(v, tall_map)
        assert torch.allclose(v_null + v_row, v, atol=1e-5)
        assert float((v_null @ v_row).abs()) < 1e-5

    def test_nullspace_part_is_annihilated(self, tall_map):
        """W_out v_null = 0: the nullspace part cannot reach the output."""
        v = torch.randn(96)
        v_null, _, _ = nullspace_rowspace_decomposition(v, tall_map)
        # Relative to the scale of W_out v itself, the nullspace part vanishes.
        assert float((tall_map.T @ v_null).norm()) < 1e-4 * float(
            (tall_map.T @ v).norm()
        )

    def test_rowspace_part_survives(self, tall_map):
        """W_out v_row != 0 for a generic v: the rowspace part is causal."""
        v = torch.randn(96)
        _, v_row, _ = nullspace_rowspace_decomposition(v, tall_map)
        assert float((tall_map.T @ v_row).norm()) > 1e-3

    def test_pure_nullspace_direction(self, tall_map):
        """A direction chosen inside ker(W) decomposes to null_fraction 1."""
        u, _, _ = torch.linalg.svd(tall_map, full_matrices=True)
        pure_null = u[:, 32:][:, 0]
        _, _, null_fraction = nullspace_rowspace_decomposition(pure_null, tall_map)
        assert float(null_fraction) == pytest.approx(1.0, abs=1e-5)

    def test_pure_rowspace_direction(self, tall_map):
        """A direction chosen in range(W) decomposes to null_fraction 0."""
        u, _, _ = torch.linalg.svd(tall_map, full_matrices=True)
        pure_row = u[:, 0]
        _, _, null_fraction = nullspace_rowspace_decomposition(pure_row, tall_map)
        assert float(null_fraction) == pytest.approx(0.0, abs=1e-5)

    def test_null_fraction_matches_dimension_ratio(self, tall_map):
        """For a random v, E[null_fraction^2] ≈ 1 - d_model/d_mlp."""
        torch.manual_seed(3)
        fractions = [
            float(nullspace_rowspace_decomposition(torch.randn(96), tall_map)[2])
            for _ in range(200)
        ]
        expected = (1 - 32 / 96) ** 0.5
        assert float(torch.tensor(fractions).mean()) == pytest.approx(expected, abs=0.05)

    def test_rejects_non_2d_map(self):
        with pytest.raises(ValueError, match="2-D"):
            nullspace_rowspace_decomposition(torch.randn(8), torch.randn(2, 3, 4))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_fldd_full_recovery_is_one(self):
        assert fractional_logit_diff_decrease(3.5, 0.2, 3.5) == pytest.approx(1.0)

    def test_fldd_no_op_is_zero(self):
        assert fractional_logit_diff_decrease(3.5, 0.2, 0.2) == pytest.approx(0.0)

    def test_fldd_overshoot_exceeds_one(self):
        # (5.0 - 0.2) / (3.5 - 0.2)
        assert fractional_logit_diff_decrease(3.5, 0.2, 5.0) == pytest.approx(4.8 / 3.3)

    def test_fldd_wrong_direction_is_negative(self):
        # (-1.0 - 0.2) / (3.5 - 0.2)
        assert fractional_logit_diff_decrease(3.5, 0.2, -1.0) == pytest.approx(-1.2 / 3.3)

    def test_fldd_undefined_when_clean_equals_corrupted(self):
        with pytest.raises(ValueError, match="coincide"):
            fractional_logit_diff_decrease(2.0, 2.0, 1.0)

    def test_projection_spread_zero_for_identical_classes(self):
        """A direction activated identically by both classes has zero spread
        (dormant)."""
        torch.manual_seed(0)
        v = torch.randn(16)
        v = v / v.norm()
        acts = torch.randn(20, 16)
        assert float(projection_spread(v, acts, acts)) == pytest.approx(0.0, abs=1e-9)

    def test_projection_spread_positive_for_shifted_classes(self):
        torch.manual_seed(0)
        v = torch.randn(16)
        v = v / v.norm()
        acts = torch.randn(200, 16)
        shifted = acts + 5.0 * v
        assert float(projection_spread(v, acts, shifted)) > 1.0


# ---------------------------------------------------------------------------
# End-to-end diagnostic
# ---------------------------------------------------------------------------


class TestSubspacePatchFaithfulness:
    def test_report_fields(self, tiny_model, tokens_and_caches):
        clean_tokens, corrupted_tokens, clean_cache = tokens_and_caches
        torch.manual_seed(1)
        v = torch.randn(tiny_model.cfg.d_mlp)

        with torch.no_grad():
            report = subspace_patch_faithfulness(
                tiny_model,
                clean_tokens,
                corrupted_tokens,
                clean_cache,
                logit_diff,
                direction=v,
                layer=1,
                pos=-1,
            )

        assert isinstance(report, SubspacePatchReport)
        for field in (
            "fldd_full",
            "fldd_rowspace_only",
            "fldd_nullspace_only",
            "null_fraction",
            "rowspace_spread",
            "nullspace_spread",
        ):
            assert isinstance(getattr(report, field), float)
        assert isinstance(report.illusion_suspected, bool)

    def test_nullspace_only_patch_has_zero_effect(self, tiny_model, tokens_and_caches):
        """FLDD of the ker(W_out)-only patch must be 0: the intervention is
        provably invisible to the model."""
        clean_tokens, corrupted_tokens, clean_cache = tokens_and_caches
        w_out = tiny_model.blocks[1].mlp.W_out.detach()
        u, _, _ = torch.linalg.svd(w_out, full_matrices=True)
        v_null = u[:, tiny_model.cfg.d_model :][:, 0]

        with torch.no_grad():
            report = subspace_patch_faithfulness(
                tiny_model,
                clean_tokens,
                corrupted_tokens,
                clean_cache,
                logit_diff,
                direction=v_null,
                layer=1,
                pos=-1,
            )

        assert report.null_fraction == pytest.approx(1.0, abs=1e-5)
        assert report.fldd_nullspace_only == pytest.approx(0.0, abs=1e-6)
        assert not report.illusion_suspected

    def test_pure_rowspace_direction_is_not_flagged(self, tiny_model, tokens_and_caches):
        """A fully causally-connected direction has null_fraction 0, so there is
        no disconnected component to carry an illusion."""
        clean_tokens, corrupted_tokens, clean_cache = tokens_and_caches
        w_out = tiny_model.blocks[1].mlp.W_out.detach()
        u, _, _ = torch.linalg.svd(w_out, full_matrices=True)
        v_row = u[:, 0]

        with torch.no_grad():
            report = subspace_patch_faithfulness(
                tiny_model,
                clean_tokens,
                corrupted_tokens,
                clean_cache,
                logit_diff,
                direction=v_row,
                layer=1,
                pos=-1,
            )

        assert report.null_fraction == pytest.approx(0.0, abs=1e-5)
        assert not report.illusion_suspected

    def test_full_equals_rowspace_when_null_fraction_is_zero(
        self, tiny_model, tokens_and_caches
    ):
        """With no disconnected component, v and v_row are the same patch."""
        clean_tokens, corrupted_tokens, clean_cache = tokens_and_caches
        w_out = tiny_model.blocks[1].mlp.W_out.detach()
        u, _, _ = torch.linalg.svd(w_out, full_matrices=True)
        v_row = u[:, 3]

        with torch.no_grad():
            report = subspace_patch_faithfulness(
                tiny_model,
                clean_tokens,
                corrupted_tokens,
                clean_cache,
                logit_diff,
                direction=v_row,
                layer=1,
                pos=-1,
            )

        assert report.fldd_full == pytest.approx(report.fldd_rowspace_only, abs=1e-6)
