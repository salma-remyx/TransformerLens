"""Layerwise massive-activation morphology (PAS / ISP) on hybrid bridge caches.

Builds tiny synthetic hybrids (no Hub access) and verifies both halves of
``cache.massive_activation_profile``:

- the structural half on real caches: full-attention layer detection vs
  ``find_ssm_mixer``, per-layer max-|activation| reporting over ``hook_resid_pre``,
- the morphology half on injected activations: a spike immediately before a
  full attention layer is flagged as a pre-attention spike (PAS), and its
  channels persisting through the layers up to the next spike form a connected
  inter-spike plateau (ISP), per arXiv:2608.12149.
"""

import pytest
import torch

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    SSM2MixerBridge,
    find_ssm_mixer,
)
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)
from transformer_lens.utilities.massive_activation_profile import (
    full_attention_layer_indices,
)


class _Tok:
    pass


def _boot(hf_model, arch):
    cfg = build_bridge_config_from_hf(hf_model.config, arch, "tiny", torch.float32)
    from transformer_lens.factories.architecture_adapter_factory import (
        ArchitectureAdapterFactory,
    )

    adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
    return TransformerBridge(model=hf_model, adapter=adapter, tokenizer=_Tok())


@pytest.fixture(scope="module")
def granite_bridge():
    from transformers import AutoModelForCausalLM
    from transformers.models.granitemoehybrid import GraniteMoeHybridConfig

    torch.manual_seed(0)
    c = GraniteMoeHybridConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=32,
        shared_intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        layer_types=["mamba", "attention", "mamba"],
        mamba_n_heads=8,
        mamba_n_groups=2,
        mamba_d_state=16,
        mamba_d_head=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_chunk_size=16,
        position_embedding_type="rope",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    c.architectures = ["GraniteMoeHybridForCausalLM"]
    return _boot(AutoModelForCausalLM.from_config(c).eval(), "GraniteMoeHybridForCausalLM")


@pytest.fixture(scope="module")
def nemotronh_bridge():
    from transformers import AutoModelForCausalLM
    from transformers.models.nemotron_h import NemotronHConfig

    torch.manual_seed(0)
    c = NemotronHConfig(
        vocab_size=256,
        hidden_size=64,
        layers_block_type=["mamba", "attention", "mamba", "mlp"],
        num_attention_heads=4,
        num_key_value_heads=2,
        ssm_state_size=16,
        mamba_num_heads=4,
        mamba_head_dim=16,
        n_groups=2,
        conv_kernel=4,
        expand=2,
        intermediate_size=128,
        chunk_size=8,
    )
    c.architectures = ["NemotronHForCausalLM"]
    return _boot(AutoModelForCausalLM.from_config(c).eval(), "NemotronHForCausalLM")


TOKENS = torch.tensor([[1, 2, 3, 4, 5]])


def _run(bridge, use_cache=True):
    with torch.no_grad():
        _, cache = bridge.run_with_cache(TOKENS, use_cache=use_cache)
    return cache


def _with_injected(cache, injections, channel=7):
    """Clone a cache, adding an outlier on ``channel`` of the given layers.

    ``injections`` maps layer index to a multiplier of that layer's current
    max |activation|; the injected value lands at one position so per-channel
    maxima — but not their median — pick it up.
    """
    cloned = {k: v.clone() for k, v in cache.cache_dict.items()}
    for layer, scale in injections.items():
        key = f"blocks.{layer}.hook_resid_pre"
        cloned[key][0, 2, channel] = cloned[key].abs().max() * scale
    return ActivationCache(cloned, model=cache.model)


# ---------------------------------------------------------------------------
# Structural half — real caches
# ---------------------------------------------------------------------------


class TestFullAttentionDetection:
    def test_granite_matches_ssm_complement(self, granite_bridge):
        assert full_attention_layer_indices(granite_bridge) == [1]
        assert granite_bridge.blocks[1].attn is not None

    def test_nemotronh_via_mixer_slot(self, nemotronh_bridge):
        """NemotronH routes attention through the ``.mixer`` slot with no ``.attn``
        child; detection must still find it and drop the MLP-only layer."""
        assert full_attention_layer_indices(nemotronh_bridge) == [1]

    def test_no_blocks_returns_empty(self):
        assert full_attention_layer_indices(object()) == []


class TestProfileOnRealCaches:
    def test_layer_kinds_and_maxima(self, nemotronh_bridge):
        cache = _run(nemotronh_bridge)
        profile = cache.massive_activation_profile()
        assert profile.full_attention_layers == [1]
        assert profile.ssm_layers == [0, 2]
        kinds = [s.layer_kind for s in profile.layers]
        assert kinds == ["ssm", "full_attention", "ssm", "other"]
        # max|a| per layer matches a direct read of the pre-block residual.
        direct = cache["blocks.2.hook_resid_pre"].abs().max().item()
        assert profile.layers[2].max_abs == pytest.approx(direct)
        assert len(profile.summary()["max_abs_per_layer"]) == 4

    def test_random_init_tiny_model_has_no_massive_channels(self, granite_bridge):
        cache = _run(granite_bridge)
        profile = cache.massive_activation_profile()
        assert profile.spikes == []
        assert profile.plateaus == []
        assert all(not s.massive_channels for s in profile.layers)


# ---------------------------------------------------------------------------
# Morphology half — injected massive activations
# ---------------------------------------------------------------------------


class TestSpikeAndPlateauMorphology:
    def test_pre_attention_spike_detected(self, granite_bridge):
        # Spike at layer 0, full attention at layer 1 -> PAS; nothing after the
        # only attention layer, so no plateau is reported.
        cache = _with_injected(_run(granite_bridge), {0: 1e4})
        profile = cache.massive_activation_profile()
        assert profile.spikes == [0]
        assert profile.pre_attention_spikes == [0]
        assert profile.layers[0].massive_channels == (7,)
        assert profile.plateaus == []

    def test_spike_not_before_attention_is_not_pas(self, granite_bridge):
        # Same spike moved after the attention layer (layer 2): still a spike,
        # never a pre-attention spike.
        cache = _with_injected(_run(granite_bridge), {2: 1e4})
        profile = cache.massive_activation_profile()
        assert profile.spikes == [2]
        assert profile.pre_attention_spikes == []

    def test_connected_plateau_between_spikes(self, nemotronh_bridge):
        # Two spikes at 0 and 3 (treat 4 as full attention via the override so
        # both are PAS); channel 7 persists at 50x through layers 1-2 -> ISP.
        base = _run(nemotronh_bridge)
        cache = _with_injected(base, {0: 1e4, 3: 1e4, 1: 50, 2: 50})
        profile = cache.massive_activation_profile(
            full_attention_layers=[1, 4],
        )
        assert profile.pre_attention_spikes == [0, 3]
        assert len(profile.plateaus) == 1
        plateau = profile.plateaus[0]
        assert (plateau.start, plateau.end) == (1, 2)
        assert plateau.connected
        assert profile.fully_connected_plateaus == [plateau]

    def test_partial_plateau_when_persistence_breaks(self, nemotronh_bridge):
        # Same two spikes, but the channel vanishes at layer 2 -> partial ISP.
        base = _run(nemotronh_bridge)
        cache = _with_injected(base, {0: 1e4, 3: 1e4, 1: 50})
        profile = cache.massive_activation_profile(
            full_attention_layers=[1, 4],
        )
        assert len(profile.plateaus) == 1
        assert not profile.plateaus[0].connected
        assert profile.fully_connected_plateaus == []

    def test_ssmlayers_override_respected(self, granite_bridge):
        cache = _run(granite_bridge)
        profile = cache.massive_activation_profile(ssm_layers=[0])
        assert profile.ssm_layers == [0]
        assert profile.layers[2].layer_kind == "other"

    def test_str_rendering_mentions_pas(self, granite_bridge):
        cache = _with_injected(_run(granite_bridge), {0: 1e4})
        assert "pre-attention spike" in str(cache.massive_activation_profile())


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


class TestMissingResidual:
    def test_missing_residual_raises(self, granite_bridge):
        cache = _run(granite_bridge)
        stripped = {
            k: v
            for k, v in cache.cache_dict.items()
            if "hook_resid_pre" not in k and not k.startswith("blocks.0.hook_in")
        }
        broken = ActivationCache(stripped, model=cache.model)
        with pytest.raises(RuntimeError, match="pre-block residual"):
            broken.massive_activation_profile()
