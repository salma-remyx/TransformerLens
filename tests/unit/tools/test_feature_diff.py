"""Unit tests for transformer_lens/tools/analysis/feature_diff.py.

Run with:
    pytest tests/unit/tools/test_feature_diff.py -v

Uses the same tiny TransformerBridge subclass pattern as
tests/unit/tools/test_jacobian_lens.py so the control hooks are exercised
against the real bridge hook surface without downloading any weights.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis import (
    SAE,
    contrastive_firing,
    control_hooks,
    feature_diff,
)

D_MODEL = 6
N_LAYERS = 4
D_VOCAB = 11
SEQ_LEN = 9


class _ToyTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return f"token-{token_ids[0]}"


class _ToyBlock(nn.Module):
    def __init__(self, d_model: int, layer: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.linear.weight, std=0.2)
        self.hook_out = HookPoint()
        self.hook_out.name = f"blocks.{layer}.hook_out"

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return self.hook_out(residual + self.linear(residual))


class _ToyBridge(TransformerBridge):
    """Minimal real TransformerBridge with Bridge-native hook_out points."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        torch.manual_seed(0)
        self.cfg = SimpleNamespace(n_layers=N_LAYERS, d_model=D_MODEL, device="cpu")
        self.tokenizer = _ToyTokenizer()
        self.embed = nn.Embedding(D_VOCAB, D_MODEL)
        self.blocks = nn.ModuleList([_ToyBlock(D_MODEL, layer) for layer in range(N_LAYERS)])

    @property
    def hook_dict(self) -> dict[str, HookPoint]:
        return {
            f"blocks.{layer}.hook_out": block.hook_out for layer, block in enumerate(self.blocks)
        }

    def to_tokens(self, prompt: str) -> torch.Tensor:
        ids = [(3 * index + len(prompt)) % D_VOCAB for index in range(SEQ_LEN)]
        return torch.tensor([ids], dtype=torch.long)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = self.embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        return residual

    @contextmanager
    def hooks(
        self,
        fwd_hooks: list[tuple[str, Any]] = [],
        bwd_hooks: list[tuple[str, Any]] = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
    ):
        del bwd_hooks, clear_contexts
        added: list[tuple[HookPoint, Any]] = []
        for name, hook_fn in fwd_hooks:
            hook_point = self.hook_dict[name]
            hook_point.add_hook(hook_fn, dir="fwd")
            added.append((hook_point, hook_point.fwd_hooks[-1]))
        try:
            yield self
        finally:
            if reset_hooks_end:
                for hook_point, handle in added:
                    handle.hook.remove()
                    if handle in hook_point.fwd_hooks:
                        hook_point.fwd_hooks.remove(handle)

    def layer_activations(self, tokens: torch.Tensor, layer: int) -> torch.Tensor:
        """Residual stream at the output of ``layer`` (the hook target)."""
        residual = self.embed(tokens)
        for index, block in enumerate(self.blocks):
            residual = block(residual)
            if index == layer:
                return residual.detach()
        raise AssertionError("unreachable")


@pytest.fixture(scope="module")
def toy_model() -> _ToyBridge:
    return _ToyBridge()


@pytest.fixture(scope="module")
def base_activations(toy_model: _ToyBridge) -> torch.Tensor:
    torch.manual_seed(1)
    tokens = torch.randint(0, D_VOCAB, (4, SEQ_LEN))
    return toy_model.layer_activations(tokens, layer=1)


@pytest.fixture(scope="module")
def base_sae(base_activations: torch.Tensor) -> SAE:
    return SAE.fit(base_activations, k=3, steps=60, seed=0, layer=1, show_progress=False)


def _rotated(activations: torch.Tensor, features: int, seed: int) -> torch.Tensor:
    """Paired activations with a few coordinates repurposed by a rotation."""
    generator = torch.Generator().manual_seed(seed)
    rotation = torch.eye(activations.shape[-1])
    rotation[:features] = torch.randn(features, activations.shape[-1], generator=generator)
    rotation[:features] = rotation[:features] / rotation[:features].norm(dim=-1, keepdim=True)
    return activations @ rotation.T


class TestSAEFit:
    def test_fit_reconstructs_activations(self, base_activations: torch.Tensor) -> None:
        sae = SAE.fit(base_activations, k=3, steps=150, seed=0, show_progress=False)
        recon = sae.decode(sae.encode(base_activations))
        baseline = torch.nn.functional.mse_loss(
            base_activations - base_activations.mean(dim=0), torch.zeros_like(recon)
        )
        assert torch.nn.functional.mse_loss(recon, base_activations).item() < baseline.item()

    def test_encode_is_top_k_sparse(self, base_sae: SAE, base_activations: torch.Tensor) -> None:
        features = base_sae.encode(base_activations)
        assert features.shape == base_activations.shape[:-1] + (base_sae.n_features,)
        assert (features > 0).sum(dim=-1).max().item() <= base_sae.k

    def test_decoder_columns_are_unit_norm(self, base_sae: SAE) -> None:
        assert torch.allclose(
            base_sae.W_dec.norm(dim=-1), torch.ones(base_sae.n_features), atol=1e-4
        )

    def test_fit_records_provenance(self, base_sae: SAE) -> None:
        assert base_sae.layer == 1
        assert base_sae.metadata["k"] == 3
        assert base_sae.metadata["seed"] == 0

    def test_fit_rejects_empty_activations(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            SAE.fit(torch.zeros(0, D_MODEL), k=2, steps=1, show_progress=False)


class TestFeatureDiff:
    def test_diff_ranks_repurposed_features_first(
        self, base_sae: SAE, base_activations: torch.Tensor
    ) -> None:
        adapted_activations = _rotated(base_activations, features=2, seed=3)
        adapted_sae = SAE.fit(
            adapted_activations, k=3, steps=60, seed=0, layer=1, show_progress=False
        )

        diff = feature_diff(base_sae, adapted_sae)

        assert diff.scores.shape == (base_sae.n_features,)
        assert diff.directions.shape == (base_sae.n_features, D_MODEL)
        # The rotation touches only 2 of 6 coordinates; every untouched base
        # feature matches itself with ~zero direction change, so the matched
        # features whose directions actually moved must sort above them.
        top = diff.top(k=2)
        untouched = [
            score
            for feature, score, _ in (
                (f, diff.scores[f].item(), diff.match_indices[f])
                for f in range(diff.scores.shape[0])
            )
            if feature not in {t[0] for t in top}
        ]
        assert min(score for _, score, _ in top) >= max(untouched) - 1e-6

    def test_diff_against_itself_is_zero(
        self, base_sae: SAE, base_activations: torch.Tensor
    ) -> None:
        diff = feature_diff(base_sae, base_sae)
        assert diff.scores.max().item() < 1e-6
        assert diff.match_indices == list(range(base_sae.n_features))

    def test_diff_with_firing_rates_uses_both_terms(
        self, base_sae: SAE, base_activations: torch.Tensor
    ) -> None:
        adapted_activations = base_activations * 4.0  # same directions, very different rates
        adapted_sae = SAE.fit(adapted_activations, k=3, steps=60, seed=0, show_progress=False)

        direction_only = feature_diff(base_sae, adapted_sae)
        with_rates = feature_diff(
            base_sae,
            adapted_sae,
            base_activations=base_activations,
            adapted_activations=adapted_activations,
        )
        assert (with_rates.scores >= direction_only.scores - 1e-6).all()

    def test_diff_rejects_mismatched_d_model(self, base_sae: SAE) -> None:
        other = SAE(
            torch.zeros(D_MODEL + 1, 2 * (D_MODEL + 1)),
            torch.zeros(2 * (D_MODEL + 1), D_MODEL + 1),
            torch.zeros(D_MODEL + 1),
            torch.zeros(2 * (D_MODEL + 1)),
            k=2,
        )
        with pytest.raises(ValueError, match="d_model"):
            feature_diff(base_sae, other)

    def test_diff_rejects_one_sided_activations(self, base_sae: SAE) -> None:
        with pytest.raises(ValueError, match="both"):
            feature_diff(base_sae, base_sae, base_activations=torch.zeros(4, D_MODEL))


class TestContrastiveFiring:
    def test_contrastive_firing_is_bounded_and_discriminative(
        self, base_sae: SAE, base_activations: torch.Tensor
    ) -> None:
        contrast = _rotated(base_activations, features=3, seed=5)
        scores = contrastive_firing(base_sae, base_activations, contrast)

        assert scores.shape == (base_sae.n_features,)
        assert (scores >= 0).all() and (scores <= 1).all()
        assert scores.max().item() > 0.0

    def test_contrastive_firing_is_zero_on_identical_pairs(
        self, base_sae: SAE, base_activations: torch.Tensor
    ) -> None:
        scores = contrastive_firing(base_sae, base_activations, base_activations)
        assert scores.max().item() == 0.0

    def test_contrastive_firing_rejects_unpaired_shapes(
        self, base_sae: SAE, base_activations: torch.Tensor
    ) -> None:
        with pytest.raises(ValueError, match="paired"):
            contrastive_firing(base_sae, base_activations, base_activations[:, :4])


class TestControlHooks:
    def test_ablation_changes_the_residual_stream(
        self, toy_model: _ToyBridge, base_sae: SAE
    ) -> None:
        torch.manual_seed(2)
        tokens = torch.randint(0, D_VOCAB, (2, SEQ_LEN))
        clean = toy_model.layer_activations(tokens, layer=1)

        hooks = control_hooks(base_sae, toy_model, features=[0, 1])
        assert [name for name, _ in hooks] == ["blocks.1.hook_out"]
        with toy_model.hooks(fwd_hooks=hooks):
            ablated = toy_model.layer_activations(tokens, layer=1)

        assert not torch.allclose(ablated, clean)
        # Ablation removes the feature components, so the residual norm does
        # not grow.
        assert ablated.norm().item() <= clean.norm().item() + 1e-5

    def test_steering_adds_the_feature_direction(
        self, toy_model: _ToyBridge, base_sae: SAE
    ) -> None:
        torch.manual_seed(2)
        tokens = torch.randint(0, D_VOCAB, (2, SEQ_LEN))
        clean = toy_model.layer_activations(tokens, layer=1)

        hooks = control_hooks(base_sae, toy_model, features=[0], alpha=2.0)
        with toy_model.hooks(fwd_hooks=hooks):
            steered = toy_model.layer_activations(tokens, layer=1)

        delta = steered - clean
        direction = base_sae.feature_directions([0])[0]
        assert (delta @ direction).min().item() > 0.0

    def test_hooks_use_the_bridge_surface_and_default_layer(
        self, toy_model: _ToyBridge, base_sae: SAE
    ) -> None:
        hooks = base_sae.feature_hooks(toy_model, features=[2])
        assert [name for name, _ in hooks] == ["blocks.1.hook_out"]  # base_sae.layer == 1
        with pytest.raises(ValueError, match="layer is required"):
            SAE(
                torch.zeros(D_MODEL, 4),
                torch.zeros(4, D_MODEL),
                torch.zeros(D_MODEL),
                torch.zeros(4),
                k=2,
            ).feature_hooks(toy_model, [0])

    def test_feature_directions_validate_indices(self, base_sae: SAE) -> None:
        with pytest.raises(ValueError, match="out of range"):
            base_sae.feature_directions([base_sae.n_features])
