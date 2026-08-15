"""Integration tests for the residual-collapse profile on real bridge models.

Loads one cached model from the CI allowlist and checks the profile end to end
against the cache the tool is built on, rather than against its own internals.
"""

import math

import pytest
import torch

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.tools.analysis import (
    ResidualCollapseProfile,
    residual_collapse_profile,
)
from transformer_lens.tools.analysis.residual_collapse_profile import (
    _effective_rank,
    _mean_offdiag_similarity,
)
from transformer_lens.utilities import get_act_name

PROMPT = "The quick brown fox jumps over the lazy dog"


def test_profile_from_real_bridge_cache(gpt2_bridge):
    """The tool runs on a TransformerBridge cache and returns aligned per-layer state."""
    profile = residual_collapse_profile(gpt2_bridge, PROMPT)

    assert isinstance(profile, ResidualCollapseProfile)
    n_layers = gpt2_bridge.cfg.n_layers
    assert len(profile.labels) == n_layers + 1
    assert profile.labels[0] == "embed"
    assert profile.labels[-1] == f"{n_layers - 1}_post"

    # Similarity and effective rank are aligned to labels; rms and the backward
    # factor cover one entry per block.
    assert profile.token_similarity.shape == (n_layers + 1,)
    assert profile.effective_rank.shape == (n_layers + 1,)
    assert profile.residual_rms.shape == (n_layers,)
    assert profile.norm_backward_factor.shape == (n_layers,)

    assert bool(torch.isfinite(profile.token_similarity).all())
    assert bool((profile.effective_rank >= 1.0).all())
    # The backward factor is exactly the reciprocal of the pre-norm residual RMS.
    assert torch.allclose(
        profile.norm_backward_factor, 1.0 / profile.residual_rms, atol=1e-6
    )
    assert torch.allclose(
        profile.cumulative_norm_backward_factor,
        torch.cumprod(profile.norm_backward_factor, dim=0),
    )


def test_profile_branch_gains_are_incremental(gpt2_bridge):
    """Attention and MLP gains decompose the per-block similarity increment."""
    profile = residual_collapse_profile(gpt2_bridge, PROMPT)

    assert profile.attn_similarity_gain is not None
    assert profile.mlp_similarity_gain is not None
    per_block_increment = profile.token_similarity[1:] - profile.token_similarity[:-1]
    # Attention + MLP sublayer gains reconstruct the block's total increment.
    assert torch.allclose(
        profile.attn_similarity_gain + profile.mlp_similarity_gain,
        per_block_increment,
        atol=1e-5,
    )


def test_profile_pre_norm_rms_matches_hook_scale(gpt2_bridge):
    """The pre-norm residual RMS matches the raw (uncentered) residual stream.

    The paper's backward factor is the RMSNorm denominator, which does not
    center the input. On a LayerNorm model ``ln1.hook_scale`` is centered, so it
    is a close but not exact reference — the residual's per-token mean is small
    relative to its norm. This pins the definition rather than trusting it.
    """
    _, cache = gpt2_bridge.run_with_cache(PROMPT)
    profile = residual_collapse_profile(cache=cache)

    for layer in range(gpt2_bridge.cfg.n_layers):
        resid_pre = cache.cache_dict[get_act_name("resid_pre", layer)].float()
        expected = (resid_pre.pow(2).mean(dim=-1) + 1e-5).sqrt().mean()
        assert torch.allclose(profile.residual_rms[layer], expected, atol=1e-6)

        hook_scale = cache.cache_dict[get_act_name("scale", layer, "ln1")].float().squeeze(-1)
        assert torch.allclose(
            profile.residual_rms[layer], hook_scale.mean(), rtol=5e-3
        ), "raw RMS drifted from the model's own normalization scale"


def test_profile_reuses_cache_without_second_forward(gpt2_bridge, monkeypatch):
    """Passing a cache skips the forward pass and yields the same trajectory."""
    _, cache = gpt2_bridge.run_with_cache(PROMPT)

    def _fail(*args, **kwargs):  # pragma: no cover - assert-not-called guard
        raise AssertionError("model should not be re-run when a cache is supplied")

    monkeypatch.setattr(gpt2_bridge, "run_with_cache", _fail)
    profile = residual_collapse_profile(cache=cache)

    assert profile.labels[-1] == f"{gpt2_bridge.cfg.n_layers - 1}_post"
    assert bool(torch.isfinite(profile.token_similarity).all())


def test_profile_without_resid_mid_degrades_to_none(gpt2_bridge):
    """Parallel-residual blocks report no branch attribution rather than wrong numbers.

    gpt2 has a mid residual; dropping it from the cache emulates a parallel block
    layout (GPT-J / GPT-NeoX / Falcon), where no distinct post-attention
    residual exists.
    """
    _, cache = gpt2_bridge.run_with_cache(PROMPT)
    filtered = {k: v for k, v in cache.cache_dict.items() if "resid_mid" not in k}
    partial = ActivationCache(filtered, cache.model, has_batch_dim=cache.has_batch_dim)
    profile = residual_collapse_profile(cache=partial)

    assert profile.attn_similarity_gain is None
    assert profile.mlp_similarity_gain is None
    # The forward and backward signatures are unaffected.
    assert profile.token_similarity.shape == (gpt2_bridge.cfg.n_layers + 1,)
    assert bool(torch.isfinite(profile.residual_rms).all())


def test_profile_requires_model_and_input_or_cache():
    """Missing both a forward-pass source and a cache is an argument error."""
    with pytest.raises(ValueError, match="precomputed `cache`"):
        residual_collapse_profile()


def test_profile_is_batch_dim_invariant(gpt2_bridge):
    """A cache with the batch dim removed yields the identical profile.

    ``ActivationCache`` methods are required to be robust to
    ``remove_batch_dim``; this profile reduces over batch and position, so both
    cache shapes must agree exactly.
    """
    _, batched = gpt2_bridge.run_with_cache(PROMPT)
    _, unbatched = gpt2_bridge.run_with_cache(PROMPT, remove_batch_dim=True)
    assert not unbatched.has_batch_dim

    from_batched = residual_collapse_profile(cache=batched)
    from_unbatched = residual_collapse_profile(cache=unbatched)

    assert torch.allclose(
        from_batched.token_similarity, from_unbatched.token_similarity, atol=1e-6
    )
    assert torch.allclose(from_batched.effective_rank, from_unbatched.effective_rank, atol=1e-6)
    assert torch.allclose(from_batched.residual_rms, from_unbatched.residual_rms, atol=1e-6)
    assert from_unbatched.attn_similarity_gain is not None
    assert from_batched.attn_similarity_gain is not None
    assert torch.allclose(
        from_batched.attn_similarity_gain,
        from_unbatched.attn_similarity_gain,
        atol=1e-6,
    )


def test_profile_raises_on_cache_without_resid_post(gpt2_bridge):
    """A names_filter cache missing the residual hooks fails loudly, not silently."""
    _, cache = gpt2_bridge.run_with_cache(PROMPT)
    filtered = {k: v for k, v in cache.cache_dict.items() if "hook_resid_post" not in k}
    partial = ActivationCache(filtered, cache.model, has_batch_dim=cache.has_batch_dim)

    with pytest.raises(ValueError, match="hook_resid_post"):
        residual_collapse_profile(cache=partial)


def test_metrics_on_known_tensors():
    """The similarity and effective-rank statistics behave on constructed input."""
    # Two orthogonal directions -> zero similarity, effective rank ~ 2.
    orthogonal = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    assert abs(_mean_offdiag_similarity(orthogonal)) < 1e-6
    assert 1.9 < _effective_rank(orthogonal) < 2.1

    # Identical rows -> similarity 1, effective rank 1.
    collapsed = torch.tensor([[[2.0, 1.0], [2.0, 1.0]]])
    assert abs(_mean_offdiag_similarity(collapsed) - 1.0) < 1e-6
    assert 1.0 <= _effective_rank(collapsed) < 1.1

    # A single position has no pairs to compare.
    assert math.isnan(_mean_offdiag_similarity(torch.ones(1, 1, 4)))


def test_profile_is_exported_from_analysis_package():
    from transformer_lens.tools import analysis

    assert analysis.residual_collapse_profile is residual_collapse_profile
    assert hasattr(analysis, "ResidualCollapseProfile")
