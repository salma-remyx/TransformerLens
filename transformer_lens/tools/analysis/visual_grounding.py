"""Visual grounding of generated tokens, via attention lookback to image tokens.

Vision-language models splice image tokens into the text sequence, so "does this
generated token actually refer to the image?" is a property of the *attention
patterns* rather than of the output logits: a response can be entirely fluent
while its positions place almost no attention mass on the image span. This
module measures that property straight off a cached forward pass.

Given the attention patterns ``run_with_cache`` already exposes
(``blocks.{i}.attn.hook_pattern``) and the sequence span occupied by image
tokens, :func:`visual_lookback` computes, for each response position, the
attention mass that position places on the image span — the *visual lookback
score*. :func:`lookback_score` combines it with the response's
length-normalised log-likelihood, and :func:`select_best_of_n` uses the
combined score to choose between candidate responses.

Adapted from "LookBack: Where and How to Score LVLM Responses via Visual
Reference Usage" (Cho et al., 2026, arXiv:2608.11847). The core mechanism —
per-token attention mass onto the image span, added to token likelihood — is
kept at full fidelity. The paper's auxiliary machinery is substituted: the
Best-of-N benchmark suite is cut (evaluation belongs downstream), and the
per-model combination weight is exposed as a parameter rather than tuned.
Where the paper fixes an aggregation over layers and heads, that aggregation is
also exposed (``layers`` / ``heads``) so a researcher can interrogate it.

Example::

    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.tools.analysis import (
        find_image_span,
        lookback_score,
        visual_lookback,
    )

    bridge = TransformerBridge.boot_transformers("Qwen/Qwen2.5-VL-7B-Instruct")
    inputs = bridge.processor(text=prompt, images=image, return_tensors="pt")
    logits, cache = bridge.run_with_cache(
        inputs["input_ids"], pixel_values=inputs["pixel_values"]
    )
    span = find_image_span(inputs["input_ids"][0], image_token_id)
    lookback = visual_lookback(cache, span)
    score = lookback_score(logits, inputs["input_ids"], lookback)
    print(score.log_likelihood.item(), score.lookback.item(), score.score.item())
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from jaxtyping import Float

from transformer_lens.ActivationCache import ActivationCache

# Attention-pattern cache key, "blocks.{i}.attn.hook_pattern" — the layout both
# HookedTransformer and TransformerBridge emit for decoder-only models, and the
# one the repo's own tests assert on. Encoder-decoder models prefix it
# ("encoder_blocks...."), so they are out of scope here rather than silently
# mis-scored.
_PATTERN_KEY = "blocks.{layer}.attn.hook_pattern"

# Anything a caller may use to describe which key positions are image tokens.
SpanLike = Union[Tuple[int, int], List[int], torch.Tensor]

# Anything a caller may use to describe which query positions are the response.
PositionLike = Union[Sequence[int], torch.Tensor, slice, None]


@dataclass
class VisualLookback:
    """Per-token attention mass onto the image span.

    Attributes:
        per_layer_head:
            Lookback per layer, head and response position, shape
            ``[layer, head, response]``. The leading axis is aligned with
            ``layers``; the head axis with ``range(head_count)``.
        per_token:
            ``per_layer_head`` averaged over the layer and head axes, shape
            ``[response]``. Each entry is in ``[0, 1]`` — it is a sum of
            softmax attention weights over the image span.
        layers:
            Original block indices the leading axis of ``per_layer_head``
            corresponds to (not contiguous when ``layers`` was passed a
            subset).
        head_count:
            Number of heads in the head axis.
        response_positions:
            Sequence positions the response axis corresponds to.
        image_span:
            The image span the scores were computed against.
    """

    per_layer_head: Float[torch.Tensor, "layer head response"]
    per_token: Float[torch.Tensor, "response"]
    layers: List[int]
    head_count: int
    response_positions: List[int]
    image_span: Tuple[int, int]
    batch_index: int = 0

    @property
    def mean(self) -> torch.Tensor:
        """Mean lookback over the response — a single scalar tensor."""
        return self.per_token.mean()

    def top_layers(self, k: int = 3) -> List[Tuple[int, float]]:
        """Return the ``k`` blocks whose mean lookback is highest.

        Useful for finding where visual reference actually happens before
        drilling into individual heads.
        """
        per_layer = self.per_layer_head.mean(dim=(1, 2))
        values, indices = torch.topk(per_layer, min(k, per_layer.shape[0]))
        return [(self.layers[i], v.item()) for i, v in zip(indices.tolist(), values)]


@dataclass
class LookBackScore:
    """Combined likelihood + visual-grounding score for one response.

    Attributes:
        log_likelihood:
            Length-normalised mean log-probability of the response tokens.
            Captures textual plausibility only.
        lookback:
            Mean visual lookback over the response tokens. Captures agreement
            with the image.
        score:
            ``log_likelihood + weight * lookback`` — the selection score.
        weight:
            The weight the combined score used.
        token_logprobs:
            Per-response-token log-probability, aligned with
            ``VisualLookback.per_token``.
    """

    log_likelihood: torch.Tensor
    lookback: torch.Tensor
    score: torch.Tensor
    weight: float
    token_logprobs: List[float] = field(default_factory=list)


def find_image_span(
    tokens: torch.Tensor,
    image_token_id: int,
) -> Tuple[int, int]:
    """Locate the contiguous run of image placeholder tokens.

    Most LVLM processors expand a single placeholder into a contiguous run of
    image tokens (Qwen-VL's ``<|image_pad|>``, LLaVA's ``<image>``), which is
    exactly the span the lookback score needs. Non-contiguous occurrences mean
    several images or interleaved text, which a single span cannot describe.

    Args:
        tokens:
            Token ids for one sequence — shape ``[pos]`` or ``[1, pos]``.
        image_token_id:
            The id the processor expands into image tokens.

    Returns:
        ``(start, stop)`` — half-open, suitable for ``range(start, stop)``.

    Raises:
        ValueError: If the token id does not occur, or occurs non-contiguously.
    """
    flat = tokens.detach().reshape(-1)
    where = (flat == image_token_id).nonzero(as_tuple=True)[0]
    if where.numel() == 0:
        raise ValueError(f"image token id {image_token_id} not found in the sequence")
    start, stop = int(where[0]), int(where[-1]) + 1
    if where.numel() != stop - start:
        raise ValueError(
            "image token id occurs non-contiguously "
            f"({where.numel()} occurrences spread over [{start}, {stop})); "
            "pass an explicit image span instead"
        )
    return start, stop


def _key_mask(image_span: SpanLike, seq_len: int, device: torch.device) -> torch.Tensor:
    """Convert an image span into a boolean mask over key positions.

    A ``(start, stop)`` pair (or 2-element list) is read as a half-open range;
    anything else must be an explicit list of positions. Both are validated
    against ``seq_len`` before use, so a bad span fails here rather than as a
    confusing index error deeper in.
    """
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    if isinstance(image_span, torch.Tensor):
        if image_span.dim() != 1:
            raise ValueError(f"image span tensor must be 1-D, got shape {tuple(image_span.shape)}")
        positions = image_span.reshape(-1).long().to(device)
    elif isinstance(image_span, tuple):
        if len(image_span) != 2:
            raise ValueError(f"image span tuple must be (start, stop), got {image_span!r}")
        start, stop = int(image_span[0]), int(image_span[1])
        if not 0 <= start < stop <= seq_len:
            raise ValueError(f"image span ({start}, {stop}) out of range for seq_len {seq_len}")
        positions = torch.arange(start, stop, device=device)
    else:
        positions = torch.tensor([int(p) for p in image_span], device=device)
        if len(positions) == 0:
            raise ValueError("image span is empty")
    if (positions < 0).any() or (positions >= seq_len).any():
        raise ValueError(f"image span positions fall outside [0, {seq_len})")
    mask[positions] = True
    return mask


def _response_indices(
    response_positions: PositionLike, image_span: Tuple[int, int], seq_len: int
) -> List[int]:
    """Normalise the response positions into an ascending list of ints.

    ``None`` means "everything after the image span". That includes any prompt
    text following the image, so callers who care about the distinction should
    pass explicit positions.
    """
    if response_positions is None:
        start, stop = image_span
        indices = list(range(stop, seq_len))
    elif isinstance(response_positions, slice):
        indices = list(range(*response_positions.indices(seq_len)))
    elif isinstance(response_positions, torch.Tensor):
        indices = response_positions.detach().reshape(-1).long().tolist()
    else:
        indices = [int(p) for p in response_positions]

    if not indices:
        raise ValueError("response_positions is empty — nothing to score")
    bad = [i for i in indices if not 0 <= i < seq_len]
    if bad:
        raise ValueError(f"response positions {bad} outside [0, {seq_len})")
    if len(set(indices)) != len(indices):
        raise ValueError("response_positions contains duplicates")
    return sorted(indices)


def _pattern_layers(cache: Union[ActivationCache, Dict[str, torch.Tensor]]) -> List[int]:
    """Block indices with an attention pattern in the cache, ascending."""
    keys = cache.keys()
    layers = []
    for name in keys:
        if not name.startswith("blocks.") or not name.endswith(".attn.hook_pattern"):
            continue
        layer = name.split(".")[1]
        if layer.isdigit():
            layers.append(int(layer))
    return sorted(set(layers))


def _select_layers(
    cache: Union[ActivationCache, Dict[str, torch.Tensor]], layers: Optional[Sequence[int]]
) -> List[int]:
    """Resolve the requested layer subset against what the cache actually has."""
    available = _pattern_layers(cache)
    if not available:
        raise ValueError(
            "no blocks.{i}.attn.hook_pattern entries in the cache — re-run the "
            "model with run_with_cache() and no names_filter, or pass "
            "pattern_keys explicitly for a non-standard hook layout"
        )
    if layers is None:
        return available
    wanted = list(layers)
    missing = [layer for layer in wanted if layer not in available]
    if missing:
        raise ValueError(
            f"requested layers {missing} have no cached attention pattern; "
            f"available layers are {available}"
        )
    return wanted


def visual_lookback(
    cache: Union[ActivationCache, Dict[str, torch.Tensor]],
    image_span: SpanLike,
    response_positions: PositionLike = None,
    *,
    layers: Optional[Sequence[int]] = None,
    heads: Optional[Sequence[int]] = None,
    batch_index: int = 0,
) -> VisualLookback:
    """Compute the visual lookback score for each response position.

    For every cached attention pattern, the score of a response position is the
    attention mass that position places on the image span::

        lookback[l, h, q] = sum_over_image_positions pattern[l][batch, h, q, k]

    Attention patterns are post-softmax, so each entry lies in ``[0, 1]`` and
    the score reads directly as "fraction of this position's attention that
    went to the image". Works with both ``HookedTransformer`` and
    ``TransformerBridge`` caches, which share the ``hook_pattern`` layout.

    Args:
        cache:
            The cache from ``run_with_cache`` over the full prompt + response
            sequence.
        image_span:
            Which key positions are image tokens — either a ``(start, stop)``
            half-open range (as :func:`find_image_span` returns) or a 1-D
            tensor of positions.
        response_positions:
            Which query positions belong to the response. ``None`` (the
            default) takes everything after the image span, which includes any
            prompt text that trails the image — pass explicit positions when
            that distinction matters.
        layers:
            Block indices to aggregate over. ``None`` uses every block with a
            cached pattern.
        heads:
            Head indices to aggregate over. ``None`` uses all of them.
        batch_index:
            Which batch element to read.

    Returns:
        A :class:`VisualLookback` with per-layer-head detail and the per-token
        mean.

    Raises:
        ValueError: If the cache has no attention patterns, the requested
            layers or the spans are invalid, or the selected blocks disagree on
            head count.
    """
    selected = _select_layers(cache, layers)
    first = cache[_PATTERN_KEY.format(layer=selected[0])]
    seq_len = first.shape[-1]
    device = first.device

    key_mask = _key_mask(image_span, seq_len, device)
    # A (start, stop) span bounds the default response ("everything after the
    # image"); an arbitrary position list carries no such bound, so the default
    # response falls back to the whole sequence.
    response_bound = image_span if isinstance(image_span, tuple) else (0, seq_len)
    queries = _response_indices(response_positions, response_bound, seq_len)

    query_index = torch.tensor(queries, device=device)
    head_sel = (
        None
        if heads is None
        else torch.tensor(sorted({int(h) for h in heads}), device=device)
    )

    per_layer_head = []
    for layer in selected:
        pattern = cache[_PATTERN_KEY.format(layer=layer)].detach()
        if pattern.dim() == 3:  # batch dim already removed
            pattern = pattern.unsqueeze(0)
        if pattern.dim() != 4:
            raise ValueError(
                f"layer {layer} pattern has shape {tuple(pattern.shape)}; expected "
                "[batch, head, query_pos, key_pos]"
            )
        if pattern.shape[-2] != seq_len:
            raise ValueError(
                f"layer {layer} pattern has {pattern.shape[-2]} query positions, "
                f"expected {seq_len}"
            )
        # [head, n_response] — attention mass each response position sends to the image.
        mass = pattern[batch_index][..., key_mask].sum(dim=-1)[:, query_index]
        if head_sel is not None:
            if head_sel.numel() and int(head_sel.max()) >= mass.shape[0]:
                raise ValueError(
                    f"requested head {int(head_sel.max())} but layer {layer} has "
                    f"{mass.shape[0]} heads"
                )
            mass = mass[head_sel]
        per_layer_head.append(mass)

    head_count = per_layer_head[0].shape[0]
    mismatched = [
        (layer, m.shape[0])
        for layer, m in zip(selected, per_layer_head)
        if m.shape[0] != head_count
    ]
    if mismatched:
        raise ValueError(
            f"selected blocks disagree on head count {mismatched}; aggregate them "
            "separately with one call per layer instead"
        )

    stacked = torch.stack(per_layer_head, dim=0).float()
    # Report the span back as a range when the caller gave one; an arbitrary
    # position list has no single range, so report the positions it covered.
    reported_span = (
        (int(image_span[0]), int(image_span[1]))
        if isinstance(image_span, tuple)
        else (int(key_mask.nonzero()[0]), int(key_mask.nonzero()[-1]) + 1)
    )
    return VisualLookback(
        per_layer_head=stacked,
        per_token=stacked.mean(dim=(0, 1)),
        layers=selected,
        head_count=head_count,
        response_positions=queries,
        image_span=reported_span,
        batch_index=batch_index,
    )


def lookback_score(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    lookback: VisualLookback,
    *,
    weight: float = 1.0,
) -> LookBackScore:
    """Combine token likelihood with visual grounding into a selection score.

    ``score = mean_token_logprob + weight * mean_visual_lookback``

    The two terms are complementary, which is the point: the likelihood is
    unchanged by swapping the image for a different one, so on its own it
    measures textual plausibility; the lookback term supplies the
    agreement-with-the-image signal the likelihood cannot see.

    Args:
        logits:
            Model logits over the full sequence, ``[batch, pos, d_vocab]`` or
            ``[pos, d_vocab]``.
        tokens:
            The token ids those logits came from, ``[batch, pos]`` or ``[pos]``.
        lookback:
            The :class:`VisualLookback` for the same forward pass — its
            ``response_positions`` drive the alignment.
        weight:
            How strongly visual grounding counts relative to likelihood. The
            paper tunes this per model; ``1.0`` is a reasonable default because
            the lookback term is bounded in ``[0, 1]`` while the likelihood
            term is not, so the weight mostly sets the scale of the tie-break.

    Returns:
        A :class:`LookBackScore`.

    Raises:
        ValueError: If the logits/tokens do not cover the response positions,
            or a response position has no predecessor to predict it from.
    """
    if logits.dim() == 3:
        logits = logits[lookback.batch_index]
    if tokens.dim() == 2:
        tokens = tokens[lookback.batch_index]
    if logits.shape[0] != tokens.shape[0]:
        raise ValueError(
            f"logits cover {logits.shape[0]} positions but tokens cover {tokens.shape[0]}"
        )

    positions = torch.tensor(lookback.response_positions, device=logits.device)
    if int(positions.min()) < 1:
        raise ValueError(
            f"response position {int(positions.min())} has no predecessor to "
            "predict it from — response_positions must start at 1 or later"
        )

    # Position p's token is predicted by the logits at p - 1.
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    predicted = logprobs[positions - 1]  # [response, d_vocab]
    targets = tokens[positions].long()  # [response]
    token_logprobs = predicted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    if token_logprobs.shape[0] != lookback.per_token.shape[0]:
        raise ValueError(
            f"{token_logprobs.shape[0]} scored tokens but lookback has "
            f"{lookback.per_token.shape[0]} — was the lookback computed on the "
            "same forward pass?"
        )

    log_likelihood = token_logprobs.mean()
    mean_lookback = lookback.per_token.mean().to(log_likelihood.device)
    return LookBackScore(
        log_likelihood=log_likelihood,
        lookback=mean_lookback,
        score=log_likelihood + weight * mean_lookback,
        weight=weight,
        token_logprobs=token_logprobs.tolist(),
    )


def select_best_of_n(candidates: Sequence[LookBackScore]) -> int:
    """Pick the index of the best-scoring candidate response.

    This is the paper's headline use: Best-of-N selection where the combined
    likelihood + lookback score, not likelihood alone, decides. Candidates
    must share a ``weight``, otherwise the comparison is not meaningful.

    Raises:
        ValueError: If ``candidates`` is empty, or the weights disagree.
    """
    if not candidates:
        raise ValueError("select_best_of_n needs at least one candidate")
    weights = {candidate.weight for candidate in candidates}
    if len(weights) > 1:
        raise ValueError(f"candidates use mixed weights {sorted(weights)}; compare like with like")
    scores = torch.stack([candidate.score for candidate in candidates])
    return int(torch.argmax(scores))
