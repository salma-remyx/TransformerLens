"""Unit tests for massive-activation detection (PAS/ISP morphology)."""

import pytest
import torch

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.massive_activations import (
    characterize_massive_activations,
    normalize_layer_types,
)

# Attention at layers 2 and 5; spikes in the residual stream feeding each.
HYBRID_LAYER_TYPES = ["mamba", "mamba", "attention", "mamba", "mamba", "attention"]
CANONICAL_LAYER_TYPES = [
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
]


class TestNormalizeLayerTypes:
    def test_aliases(self):
        assert normalize_layer_types(HYBRID_LAYER_TYPES) == CANONICAL_LAYER_TYPES

    def test_passthrough_of_other_types(self):
        assert normalize_layer_types(["mlp", "moe"]) == ["mlp", "moe"]


class TestCharacterizeMassiveActivations:
    def test_pas_and_isp(self):
        layer_max = torch.tensor([2.0, 500.0, 3.0, 25.0, 600.0, 2.5])
        layer_median = torch.tensor([0.5, 0.6, 0.5, 0.5, 0.55, 0.5])
        report = characterize_massive_activations(layer_max, layer_median, HYBRID_LAYER_TYPES)
        assert report.layer_types == CANONICAL_LAYER_TYPES
        assert report.massive_layers == [1, 4]
        assert report.pre_attention_spikes == [1, 4]
        # Layer 3 is an elevated linear-attention layer between the spikes.
        assert report.inter_spike_plateaus == [3]

    def test_spike_not_before_attention_is_not_pas(self):
        layer_max = torch.tensor([2.0, 500.0, 3.0, 2.5])
        layer_median = torch.tensor([0.5, 0.6, 0.5, 0.5])
        layer_types = ["mamba", "mamba", "mamba", "attention"]
        report = characterize_massive_activations(layer_max, layer_median, layer_types)
        assert report.massive_layers == [1]
        assert report.pre_attention_spikes == []

    def test_without_layer_types_detects_outliers_only(self):
        report = characterize_massive_activations(
            torch.tensor([1.0, 200.0]), torch.tensor([0.5, 0.5])
        )
        assert report.massive_layers == [1]
        assert report.pre_attention_spikes == []
        assert report.inter_spike_plateaus == []

    def test_validates_thresholds(self):
        with pytest.raises(ValueError, match="plateau_ratio"):
            characterize_massive_activations(torch.ones(2), torch.ones(2), plateau_ratio=200.0)
        with pytest.raises(ValueError, match="outlier_threshold"):
            characterize_massive_activations(torch.ones(2), torch.ones(2), outlier_threshold=0.0)


class _FakeCfg:
    def __init__(self, n_layers, layers_block_type=None):
        self.n_layers = n_layers
        if layers_block_type is not None:
            self.layers_block_type = layers_block_type


class _FakeModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.blocks = []


def _make_cache(layer_types, outliers):
    """Cache of ``resid_post`` activations with |x| = 0.5 everywhere plus a
    single injected outlier coordinate per layer in ``outliers``."""
    n_layers = len(layer_types)
    cache_dict = {}
    for layer in range(n_layers):
        resid = torch.full((1, 4, 8), 0.5)
        resid[..., 1::2] *= -1  # mix signs; |x| stays 0.5
        if layer in outliers:
            resid[0, 0, 0] = outliers[layer]
        cache_dict[f"blocks.{layer}.hook_resid_post"] = resid
    cfg = _FakeCfg(n_layers, layer_types)
    return ActivationCache(cache_dict, _FakeModel(cfg))


class TestDetectMassiveActivations:
    def test_hybrid_cache_reports_pas_and_isp(self):
        cache = _make_cache(HYBRID_LAYER_TYPES, {1: 500.0, 3: 20.0, 4: 600.0})
        report = cache.detect_massive_activations()
        assert report.massive_layers == [1, 4]
        assert report.pre_attention_spikes == [1, 4]
        assert report.inter_spike_plateaus == [3]
        assert report.layer_types == CANONICAL_LAYER_TYPES
        assert report.massive_ratio[1] > report.massive_ratio[3] > report.massive_ratio[0]

    def test_full_attention_fallback(self):
        # No layers_block_type and no SSM mixers: every layer counts as full
        # attention, so a spike preceding a layer is a pre-attention spike.
        cache_dict = {}
        for layer in range(4):
            resid = torch.full((1, 4, 8), 0.5)
            if layer == 1:
                resid[0, 0, 0] = 500.0
            cache_dict[f"blocks.{layer}.hook_resid_post"] = resid
        cache = ActivationCache(cache_dict, _FakeModel(_FakeCfg(4)))
        report = cache.detect_massive_activations()
        assert report.massive_layers == [1]
        assert report.pre_attention_spikes == [1]
        assert report.inter_spike_plateaus == []
