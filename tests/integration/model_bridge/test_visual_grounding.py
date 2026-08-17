"""Integration tests for visual lookback scoring over a real bridge cache.

Exercises transformer_lens.tools.analysis.visual_grounding against a genuine
``run_with_cache`` forward pass, so the hook names it reads and the shapes it
assumes are checked against what the bridge actually emits rather than against
a hand-built cache.

distilgpt2 has no image tokens, so the image span is a contiguous run of a
repeated token standing in for the expanded image placeholder. What is under
test — reading ``blocks.{i}.attn.hook_pattern``, masking the key positions,
gathering the response positions, combining with token log-probs — is exactly
the LVLM path.

Uses distilgpt2 (CI-cached).
"""

import pytest
import torch

from transformer_lens.tools.analysis import (
    find_image_span,
    lookback_score,
    visual_lookback,
)


@pytest.fixture()
def bridge(distilgpt2_bridge):
    """Alias the session fixture for concise test signatures."""
    return distilgpt2_bridge


@pytest.fixture()
def scored(bridge):
    """A cached forward pass plus the span/response bookkeeping, run once."""
    tokens = bridge.to_tokens("The quick brown fox jumps over the lazy dog")
    # The token at position 3 recurs contiguously later in the sequence; use that
    # run as the stand-in image span, as find_image_span would on an image id.
    image_token_id = int(tokens[0, 3])
    span = find_image_span(tokens[0], image_token_id)
    logits, cache = bridge.run_with_cache(tokens)
    return tokens, span, logits, cache


class TestVisualLookbackOnRealCache:
    """visual_lookback reads the patterns a real bridge caches."""

    def test_reads_every_block_pattern(self, scored, bridge):
        _tokens, span, _logits, cache = scored
        result = visual_lookback(cache, span)
        n_blocks = len(bridge.blocks)
        assert result.layers == list(range(n_blocks))
        assert result.per_layer_head.shape[0] == n_blocks

    def test_shapes_match_heads_and_response(self, scored):
        _tokens, span, _logits, cache = scored
        result = visual_lookback(cache, span)
        assert result.head_count == result.per_layer_head.shape[1]
        assert result.per_layer_head.shape[2] == len(result.response_positions)

    def test_scores_are_valid_attention_mass(self, scored):
        """Post-softmax attention summed over a key subset stays within [0, 1]."""
        _tokens, span, _logits, cache = scored
        result = visual_lookback(cache, span)
        assert (result.per_layer_head >= 0).all()
        assert (result.per_layer_head <= 1 + 1e-5).all()
        assert result.per_token.shape == (len(result.response_positions),)

    def test_layer_subset_selects_real_blocks(self, scored):
        _tokens, span, _logits, cache = scored
        result = visual_lookback(cache, span, layers=[0, 1])
        assert result.layers == [0, 1]
        assert result.per_layer_head.shape[0] == 2


class TestLookBackScoreOnRealCache:
    """lookback_score combines real logits with the grounding measurement."""

    def test_score_is_finite_and_ordered(self, scored):
        tokens, span, logits, cache = scored
        lookback = visual_lookback(cache, span)
        score = lookback_score(logits, tokens, lookback)
        assert torch.isfinite(score.log_likelihood)
        assert torch.isfinite(score.lookback)
        assert torch.isfinite(score.score)
        assert score.log_likelihood < 0  # a real model is not certainty-1
        assert 0 <= score.lookback <= 1

    def test_token_logprobs_align_with_response(self, scored):
        tokens, span, logits, cache = scored
        lookback = visual_lookback(cache, span)
        score = lookback_score(logits, tokens, lookback)
        assert len(score.token_logprobs) == len(lookback.response_positions)
        assert all(lp < 0 for lp in score.token_logprobs)

    def test_weight_zero_recovers_pure_likelihood(self, scored):
        tokens, span, logits, cache = scored
        lookback = visual_lookback(cache, span)
        plain = lookback_score(logits, tokens, lookback, weight=0.0)
        assert torch.allclose(plain.score, plain.log_likelihood, atol=1e-6)
