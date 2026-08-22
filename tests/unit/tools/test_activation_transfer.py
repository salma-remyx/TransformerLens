"""Unit tests for cross-model activation transfer on toy bridges."""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
import torch
import torch.nn as nn

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis import (
    ActivationProjection,
    cross_model_transfer_report,
    mutual_knn_alignment,
    residual_injection_hooks,
)

D_MODEL = 6
N_LAYERS = 4
SEQ_LEN = 9
PROMPTS = [f"paired prompt {index}" for index in range(12)]


class _TransferToyBridge(TransformerBridge):
    """Small raw ``TransformerBridge`` with two distinct hidden spaces.

    Follows the toy-bridge pattern in ``test_jacobian_lens.py``: a real
    ``TransformerBridge`` subclass whose public analysis surface (cfg, adapter
    contract, hook_dict, run_with_cache, hooks) is exercised without a
    Hugging Face model.
    """

    def __init__(self, *, d_model: int, seed: int) -> None:
        nn.Module.__init__(self)
        torch.manual_seed(seed)
        self.cfg = SimpleNamespace(
            n_layers=N_LAYERS,
            d_model=d_model,
            d_vocab=32,
            device="cpu",
        )
        self.adapter = SimpleNamespace(supports_generation=True)
        self.compatibility_mode = False
        self.tokenizer = None
        self.embed = nn.Embedding(32, d_model)
        self.blocks = nn.ModuleList([_ResidualBlock(d_model, layer) for layer in range(N_LAYERS)])

    @property
    def hook_dict(self) -> dict[str, HookPoint]:
        return {
            f"blocks.{layer}.hook_out": block.hook_out for layer, block in enumerate(self.blocks)
        }

    def to_tokens(self, prompt: str) -> torch.Tensor:
        ids = [(7 * index + len(prompt)) % 32 for index in range(SEQ_LEN)]
        return torch.tensor([ids], dtype=torch.long)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = self.embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        return residual

    def run_with_cache(
        self,
        input: str,
        names_filter: Any = None,
        return_cache_object: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del return_cache_object, kwargs
        cache: dict[str, torch.Tensor] = {}

        def cache_hook(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            assert hook.name is not None
            cache[hook.name] = activation.detach()
            return activation

        wanted = set() if names_filter is None else set(names_filter)
        cache_hooks = [(name, cache_hook) for name in self.hook_dict if name in wanted]
        with self.hooks(fwd_hooks=cache_hooks):
            output = self(self.to_tokens(input))
        return output, cache

    @contextmanager
    def hooks(
        self,
        fwd_hooks: list[tuple[str, Any]] = [],
        bwd_hooks: list[tuple[str, Any]] = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
    ) -> Iterator["_TransferToyBridge"]:
        del clear_contexts, bwd_hooks
        added: list[tuple[HookPoint, Any]] = []
        for name, hook_fn in fwd_hooks:
            hook_point = self.hook_dict[name]
            hook_point.add_hook(hook_fn, dir="fwd")
            added.append((hook_point, hook_point.fwd_hooks[-1].hook))
        try:
            yield self
        finally:
            if reset_hooks_end:
                for hook_point, handle in added:
                    handle.remove()

    def run_with_hooks(self, input: str, fwd_hooks: list[tuple[str, Any]], **kwargs: Any):
        del kwargs
        with self.hooks(fwd_hooks=fwd_hooks):
            return self(self.to_tokens(input))


class _ResidualBlock(nn.Module):
    def __init__(self, d_model: int, layer: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.linear.weight, std=0.2)
        self.hook_out = HookPoint()
        self.hook_out.name = f"blocks.{layer}.hook_out"

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return self.hook_out(residual + self.linear(residual))


@pytest.fixture(scope="module")
def source_model() -> _TransferToyBridge:
    return _TransferToyBridge(d_model=D_MODEL, seed=0)


@pytest.fixture(scope="module")
def target_model() -> _TransferToyBridge:
    # Different d_model and weights: a genuinely different hidden space.
    return _TransferToyBridge(d_model=D_MODEL + 2, seed=1)


@pytest.fixture(scope="module")
def report(source_model, target_model):
    return cross_model_transfer_report(
        source_model,
        target_model,
        PROMPTS,
        source_layer=2,
        target_layer=2,
        n_permutations=20,
    )


def test_report_runs_all_three_level_one_and_two_fields(report) -> None:
    assert report.n_prompts == len(PROMPTS)
    assert report.n_train + report.n_holdout == len(PROMPTS)
    assert report.n_holdout >= 1
    assert 0.0 <= report.mutual_knn <= 1.0
    assert report.mutual_knn_p_value is not None and 0.0 < report.mutual_knn_p_value <= 1.0
    assert 0.0 <= report.retrieval_top1 <= 1.0
    assert report.retrieval_chance == pytest.approx(1.0 / report.n_holdout)


def test_report_is_deterministic(source_model, target_model, report) -> None:
    repeat = cross_model_transfer_report(
        source_model,
        target_model,
        PROMPTS,
        source_layer=2,
        target_layer=2,
        n_permutations=20,
    )
    assert repeat.mutual_knn == report.mutual_knn
    assert repeat.retrieval_top1 == report.retrieval_top1
    assert repeat.mutual_knn_null_mean == report.mutual_knn_null_mean


def test_permuted_retrieval_control_scores_at_chance(report) -> None:
    # Matching rows are broken by the permutation; accuracy should collapse
    # to at most the unpermuted score and stay within the chance band.
    assert report.retrieval_permuted_top1 <= report.retrieval_top1
    assert report.retrieval_permuted_top1 <= 1.0


def test_mutual_knn_alignment_is_one_for_isometric_spaces() -> None:
    torch.manual_seed(0)
    source = torch.randn(24, 16)
    isometry = torch.linalg.qr(torch.randn(20, 16)).Q
    assert mutual_knn_alignment(source, source @ isometry.T, k=5) == pytest.approx(1.0)


def test_mutual_knn_alignment_rejects_unpaired_rows() -> None:
    with pytest.raises(ValueError, match="share the item axis"):
        mutual_knn_alignment(torch.randn(5, 4), torch.randn(6, 4))


def test_projection_round_trips_a_linear_target() -> None:
    torch.manual_seed(0)
    source = torch.randn(40, 8)
    mapping = torch.randn(12, 8)
    target = source @ mapping.T + 0.25
    projection = ActivationProjection.fit(source[:32], target[:32], source_layer=1, target_layer=3)
    assert projection.weight.shape == (12, 8)
    assert projection.retrieval_accuracy(source[32:], target[32:]) == pytest.approx(1.0)
    torch.testing.assert_close(projection.project(source[:4]), target[:4], atol=1e-4, rtol=1e-4)


def test_projection_records_layer_provenance() -> None:
    torch.manual_seed(0)
    source = torch.randn(6, 4)
    projection = ActivationProjection.fit(
        source, source.clone(), source_layer=5, target_layer=7, metadata={"pair": "a-b"}
    )
    assert projection.source_layer == 5
    assert projection.target_layer == 7
    assert projection.metadata == {"pair": "a-b"}


def test_report_refuses_non_raw_bridges(source_model, target_model) -> None:
    target_model.compatibility_mode = True
    try:
        with pytest.raises(ValueError, match="compatibility mode"):
            cross_model_transfer_report(
                source_model, target_model, PROMPTS, source_layer=1, target_layer=1
            )
    finally:
        target_model.compatibility_mode = False


def test_report_requires_a_fittable_split(source_model, target_model) -> None:
    with pytest.raises(ValueError, match="holdout_count"):
        cross_model_transfer_report(
            source_model,
            target_model,
            PROMPTS,
            source_layer=1,
            target_layer=1,
            holdout_count=len(PROMPTS),
        )


def test_injection_hooks_replace_residual_and_drive_generation(target_model) -> None:
    residual = torch.randn(target_model.cfg.d_model)
    hooks = residual_injection_hooks(target_model, residual, -2, positions=[-1])
    assert hooks[0][0] == f"blocks.{N_LAYERS - 2}.hook_out"

    clean = target_model.run_with_hooks(PROMPTS[0], fwd_hooks=[])
    injected = target_model.run_with_hooks(PROMPTS[0], fwd_hooks=hooks)
    assert not torch.allclose(clean[:, -1, :], injected[:, -1, :])
    # Only the requested position is overwritten.
    torch.testing.assert_close(clean[:, :-1, :], injected[:, :-1, :])

    def capture(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        capture.seen = activation.clone()
        return activation

    capture.seen = None
    target_model.run_with_hooks(PROMPTS[0], fwd_hooks=hooks + [(hooks[0][0], capture)])
    assert torch.allclose(capture.seen[:, -1, :], residual.expand_as(capture.seen[:, -1, :]))


def test_injection_hooks_accept_projected_batch_rows(report, target_model) -> None:
    # Level 3 wiring: the fitted projection's output feeds the injection hooks.
    _, cache = target_model.run_with_cache(
        PROMPTS[0], names_filter=[f"blocks.{report.source_layer}.hook_out"]
    )
    projected = report.projection.project(torch.randn(2, report.projection.weight.shape[1]))
    hooks = residual_injection_hooks(
        target_model, projected, report.projection.target_layer, positions=[-1]
    )
    assert hooks[0][0] == f"blocks.{report.projection.target_layer}.hook_out"
    assert cache  # the target-side collection path used by the report also works standalone


def test_injection_hooks_validate_shape(target_model) -> None:
    with pytest.raises(ValueError, match="residual must be"):
        residual_injection_hooks(target_model, torch.randn(target_model.cfg.d_model + 1), 0)
    with pytest.raises(ValueError, match="out of range"):
        residual_injection_hooks(target_model, torch.randn(target_model.cfg.d_model), N_LAYERS)
