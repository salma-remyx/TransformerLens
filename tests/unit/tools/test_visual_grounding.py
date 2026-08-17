"""Unit tests for visual lookback scoring over an ActivationCache.

The scoring functions are cache-agnostic, so these tests drive them with a hand
-built ``ActivationCache`` (no model download): attention patterns whose mass is
known by construction make the expected lookback scores exact rather than
approximate.

The integration that matters — that the exported ``transformer_lens.tools.
analysis`` surface actually resolves and that the hook names the module reads
are the ones a real bridge emits — is asserted against the repo's own hook
inventory, not against a mock of it.
"""

import math

import pytest
import torch

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.tools import analysis
from transformer_lens.tools.analysis import (
    LookBackScore,
    VisualLookback,
    find_image_span,
    lookback_score,
    select_best_of_n,
    visual_lookback,
)

N_LAYERS, N_HEADS, SEQ_LEN = 2, 3, 10
IMAGE_SPAN = (2, 5)  # key positions 2, 3, 4 are image tokens
RESPONSE = list(range(6, 10))  # query positions 6-9 are the response


def _softmax_rows(scores: torch.Tensor) -> torch.Tensor:
    """Softmax over the last (key) axis of ``[head, query, key]`` scores."""
    return torch.softmax(scores, dim=-1)


def _build_pattern(image_mass_by_layer_head) -> torch.Tensor:
    """Build one layer's pattern with prescribed attention mass on the image span.

    ``image_mass_by_layer_head`` is ``[head, query]`` — the softmax mass each
    head's query position must place on key positions 2-4, the rest spread
    evenly over the remaining keys. Positions before the image span attend to
    themselves (causal, unused by the score).
    """
    head, query = image_mass_by_layer_head.shape
    pattern = torch.zeros(N_HEADS, SEQ_LEN, SEQ_LEN)
    for h in range(head):
        for q in range(SEQ_LEN):
            if q in RESPONSE:
                image = image_mass_by_layer_head[h, q]
                pattern[h, q, 2:5] = image / 3
                others = [k for k in range(SEQ_LEN) if k < 2 or k >= 5]
                pattern[h, q, others] = (1 - image) / len(others)
            else:
                pattern[h, q, q] = 1.0
    return _softmax_rows(torch.log(pattern.clamp_min(1e-9)))


def _cache_with(patterns, batch: int = 1) -> ActivationCache:
    """Wrap per-layer patterns in a real ActivationCache with batch dim intact."""
    cache_dict = {
        f"blocks.{layer}.attn.hook_pattern": pattern.unsqueeze(0).expand(batch, -1, -1, -1)
        for layer, pattern in enumerate(patterns)
    }
    return ActivationCache(cache_dict, model=None, has_batch_dim=True)


@pytest.fixture()
def lookback_cache() -> ActivationCache:
    """Two layers: head 0 looks back hard, heads 1-2 ignore the image.

    Mean per response position = (0.9 + 0.1 + 0.1) / 3 = 0.3667. A second batch
    element carries the same patterns so batch_index has something to select.
    """
    head_mass = torch.full((N_HEADS, SEQ_LEN), 0.1)
    head_mass[0, :] = 0.9
    return _cache_with([_build_pattern(head_mass) for _ in range(N_LAYERS)], batch=2)


class TestFindImageSpan:
    """find_image_span locates the expanded image placeholder run."""

    def test_returns_half_open_contiguous_span(self):
        tokens = torch.tensor([5, 5, 9, 9, 9, 9, 7])
        assert find_image_span(tokens, 9) == (2, 6)

    def test_accepts_batched_tokens(self):
        assert find_image_span(torch.tensor([[1, 4, 4, 2]]), 4) == (1, 3)

    def test_missing_token_id_raises(self):
        with pytest.raises(ValueError, match="not found"):
            find_image_span(torch.tensor([1, 2, 3]), 99)

    def test_non_contiguous_run_raises(self):
        tokens = torch.tensor([9, 1, 9])
        with pytest.raises(ValueError, match="non-contiguous"):
            find_image_span(tokens, 9)


class TestVisualLookback:
    """visual_lookback reads attention mass off the cache's hook_pattern keys."""

    def test_per_token_matches_constructed_attention_mass(self, lookback_cache):
        result = visual_lookback(lookback_cache, IMAGE_SPAN, response_positions=RESPONSE)
        expected = torch.full((len(RESPONSE),), (0.9 + 0.1 + 0.1) / 3)
        assert isinstance(result, VisualLookback)
        assert torch.allclose(result.per_token, expected, atol=1e-5)

    def test_reads_only_attn_pattern_hooks(self, lookback_cache):
        """Resid/mLP entries in the same cache are ignored, not confused for patterns."""
        lookback_cache.cache_dict["blocks.0.hook_resid_post"] = torch.zeros(1, SEQ_LEN, 8)
        result = visual_lookback(lookback_cache, IMAGE_SPAN, response_positions=RESPONSE)
        assert result.layers == [0, 1]
        assert result.per_layer_head.shape == (N_LAYERS, N_HEADS, len(RESPONSE))

    def test_head_and_layer_subset(self, lookback_cache):
        result = visual_lookback(
            lookback_cache, IMAGE_SPAN, response_positions=RESPONSE, layers=[1], heads=[0]
        )
        assert result.layers == [1]
        assert result.head_count == 1
        assert torch.allclose(result.per_token, torch.full((len(RESPONSE),), 0.9), atol=1e-5)

    def test_default_response_is_everything_after_the_image(self, lookback_cache):
        result = visual_lookback(lookback_cache, IMAGE_SPAN)
        assert result.response_positions == list(range(5, SEQ_LEN))

    def test_scores_are_attention_mass_in_unit_interval(self, lookback_cache):
        result = visual_lookback(lookback_cache, IMAGE_SPAN, response_positions=RESPONSE)
        assert (result.per_layer_head >= 0).all() and (result.per_layer_head <= 1).all()

    def test_position_tensor_span(self, lookback_cache):
        """An explicit position list works as a span, e.g. pooled/masked image tokens."""
        result = visual_lookback(
            lookback_cache, torch.tensor([2, 4]), response_positions=RESPONSE
        )
        # 2 of the 3 image positions, each carrying a third of head 0's mass -> 0.6
        assert torch.allclose(
            result.per_layer_head[1, 0], torch.full((len(RESPONSE),), 0.6), atol=1e-5
        )
        assert result.image_span == (2, 5)  # tightest covering range

    def test_position_list_span(self, lookback_cache):
        """A plain Python list of positions is accepted too, same as a tensor."""
        from_list = visual_lookback(lookback_cache, [2, 3, 4], response_positions=RESPONSE)
        from_range = visual_lookback(lookback_cache, IMAGE_SPAN, response_positions=RESPONSE)
        assert torch.allclose(from_list.per_token, from_range.per_token, atol=1e-6)

    def test_top_layers_ranks_by_mean_mass(self, lookback_cache):
        result = visual_lookback(lookback_cache, IMAGE_SPAN, response_positions=RESPONSE)
        assert [layer for layer, _ in result.top_layers()] == [0, 1]


class TestVisualLookbackValidation:
    """visual_lookback rejects malformed inputs loudly rather than scoring them."""

    def test_empty_cache_raises(self):
        with pytest.raises(ValueError, match="hook_pattern"):
            visual_lookback(ActivationCache({}, model=None), (0, 2))

    def test_unknown_layer_raises(self, lookback_cache):
        with pytest.raises(ValueError, match="no cached attention pattern"):
            visual_lookback(lookback_cache, IMAGE_SPAN, layers=[7])

    def test_out_of_range_span_raises(self, lookback_cache):
        with pytest.raises(ValueError, match="out of range"):
            visual_lookback(lookback_cache, (4, 99))

    def test_out_of_range_response_raises(self, lookback_cache):
        with pytest.raises(ValueError, match="outside"):
            visual_lookback(lookback_cache, IMAGE_SPAN, response_positions=[0, 99])

    def test_head_beyond_count_raises(self, lookback_cache):
        with pytest.raises(ValueError, match="heads"):
            visual_lookback(lookback_cache, IMAGE_SPAN, heads=[N_HEADS])


def _scores_for(lookback: VisualLookback, logit: float) -> LookBackScore:
    """Build a LookBackScore whose every response token has the same log-prob.

    Position ``p``'s token is predicted by the logits at ``p - 1``, so only those
    rows matter. Token 1 is the response token; against a single zero competitor
    its log-prob is ``logit - log1p(exp(logit))``, which stays exact in float32.
    """
    logits = torch.zeros(SEQ_LEN, D_VOCAB)
    tokens = torch.full((SEQ_LEN,), RESPONSE_TOKEN, dtype=torch.long)
    for p in lookback.response_positions:
        logits[p - 1, RESPONSE_TOKEN] = logit
    return lookback_score(logits, tokens, lookback)


D_VOCAB = 2
RESPONSE_TOKEN = 1  # the non-zero vocab entry, so a zero row predicts token 0


def _expected_logprob(logit: float) -> float:
    """Log-prob of the response token when its logit faces one zero competitor."""
    return logit - math.log1p(math.exp(logit))


class TestLookBackScore:
    """lookback_score adds the grounding signal the likelihood cannot see."""

    def test_uses_prediction_at_previous_position(self, lookback_cache):
        """Position 6's token is read from the logits at 5, not at 6."""
        lookback = visual_lookback(lookback_cache, IMAGE_SPAN, response_positions=RESPONSE)
        logits = torch.zeros(SEQ_LEN, D_VOCAB)
        logits[5, RESPONSE_TOKEN] = -12.0  # predicts position 6, the first response token
        tokens = torch.full((SEQ_LEN,), RESPONSE_TOKEN, dtype=torch.long)
        score = lookback_score(logits, tokens, lookback)
        assert math.isclose(score.token_logprobs[0], _expected_logprob(-12.0), abs_tol=1e-5)

    def test_score_combines_both_terms(self, lookback_cache):
        lookback = visual_lookback(lookback_cache, IMAGE_SPAN, response_positions=RESPONSE)
        score = _scores_for(lookback, -1.0)
        assert isinstance(score, LookBackScore)
        assert math.isclose(score.log_likelihood.item(), _expected_logprob(-1.0), abs_tol=1e-5)
        assert torch.allclose(score.lookback, lookback.per_token.mean(), atol=1e-6)
        assert math.isclose(
            score.score.item(), score.log_likelihood.item() + score.lookback.item(), abs_tol=1e-6
        )

    def test_batched_inputs_use_batch_index(self, lookback_cache):
        """Only batch element 1's logits are read, and its tokens are what's scored."""
        lookback = visual_lookback(
            lookback_cache, IMAGE_SPAN, response_positions=RESPONSE, batch_index=1
        )
        logits = torch.zeros(2, SEQ_LEN, D_VOCAB)
        logits[0, 5, RESPONSE_TOKEN] = 4.0  # would win if batch 0 were read instead
        logits[1, 5, RESPONSE_TOKEN] = -8.0  # predicts position 6, the first response token
        tokens = torch.full((2, SEQ_LEN), RESPONSE_TOKEN, dtype=torch.long)
        score = lookback_score(logits, tokens, lookback)
        assert math.isclose(score.token_logprobs[0], _expected_logprob(-8.0), abs_tol=1e-4)

    def test_position_zero_response_raises(self, lookback_cache):
        lookback = visual_lookback(lookback_cache, IMAGE_SPAN, response_positions=[0, 1])
        with pytest.raises(ValueError, match="no predecessor"):
            lookback_score(
                torch.zeros(SEQ_LEN, 5), torch.zeros(SEQ_LEN, dtype=torch.long), lookback
            )


class TestSelectBestOfN:
    """select_best_of_n is the Best-of-N decision the combined score drives."""

    def test_prefers_grounded_over_fluent(self, lookback_cache):
        """Likelihood alone picks the wrong response when grounding differs.

        Same model, two candidate responses: the fluent one is more likely but
        its heads place little mass on the image; the grounded one is less
        likely but looks back hard. LookBack picks the grounded one.
        """
        # Two responses to the same prompt, scored on the same model.
        ungrounded_cache = _cache_with(  # fluent — heads barely look at the image
            [_build_pattern(torch.full((N_HEADS, SEQ_LEN), 0.05)) for _ in range(N_LAYERS)]
        )
        grounded_cache = _cache_with(  # grounded — heads look back hard
            [_build_pattern(torch.full((N_HEADS, SEQ_LEN), 0.8)) for _ in range(N_LAYERS)]
        )

        fluent = _scores_for(
            visual_lookback(ungrounded_cache, IMAGE_SPAN, response_positions=RESPONSE), -0.5
        )
        grounded = _scores_for(
            visual_lookback(grounded_cache, IMAGE_SPAN, response_positions=RESPONSE), -0.8
        )
        assert fluent.log_likelihood > grounded.log_likelihood  # likelihood says #0
        assert select_best_of_n([fluent, grounded]) == 1  # the lookback term overrules it

    def test_weight_zero_degenerates_to_likelihood(self, lookback_cache):
        """At weight 0 the grounding signal is ignored and likelihood decides."""
        lookback = visual_lookback(lookback_cache, IMAGE_SPAN, response_positions=RESPONSE)
        likelier = _scores_for(lookback, -0.5)
        less_likely = _scores_for(lookback, -0.8)
        likelier.weight = less_likely.weight = 0.0
        assert select_best_of_n([less_likely, likelier]) == 1

    def test_mixed_weights_raise(self, lookback_cache):
        lookback = visual_lookback(lookback_cache, IMAGE_SPAN, response_positions=RESPONSE)
        a = _scores_for(lookback, -0.5)
        b = _scores_for(lookback, -0.6)
        b.weight = 2.0
        with pytest.raises(ValueError, match="mixed weights"):
            select_best_of_n([a, b])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            select_best_of_n([])


class TestAnalysisPackageIntegration:
    """The exported analysis surface resolves and the hook names are the repo's own."""

    def test_exports_from_analysis_package(self):
        assert analysis.visual_lookback is visual_lookback
        assert analysis.lookback_score is lookback_score
        assert analysis.find_image_span is find_image_span
        assert analysis.select_best_of_n is select_best_of_n
        for name in ("LookBackScore", "VisualLookback"):
            assert name in analysis.__all__

    def test_hook_names_match_repo_convention(self):
        """The keys visual_lookback reads are the ones the repo's own tests assert on.

        tests/integration/model_bridge/test_bridge_generate_return_cache.py documents
        ``blocks.0.attn.hook_pattern`` as the [batch, heads, query, key] pattern key;
        visual_grounding must derive its keys from the same layout rather than
        inventing a parallel one.
        """
        import transformer_lens.tools.analysis.visual_grounding as vg

        assert vg._PATTERN_KEY.format(layer=0) == "blocks.0.attn.hook_pattern"
