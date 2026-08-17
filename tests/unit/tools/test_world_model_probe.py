"""Unit tests for the world-model probe, decay curve, and restoration hook.

Uses the same small synthetic-bridge pattern as ``test_jacobian_lens.py``: a
real ``ActivationCache`` from a tiny hooked model, no network access. The
residual stream carries a known linear "state" signal, so a fitted probe must
recover it exactly, a decayed copy must score lower, and the restoration hook
must move the representation back.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis import (
    fit_world_model_probe,
    probe_fidelity_by_position,
    restoration_hooks,
)

D_MODEL = 16
N_LAYERS = 3
D_VOCAB = 11
SEQ_LEN = 64
LAYER = 1


class _ToyBridge(TransformerBridge):
    """A tiny TransformerBridge whose layer-1 output encodes a known state."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        torch.manual_seed(0)
        self.cfg = SimpleNamespace(n_layers=N_LAYERS, d_model=D_MODEL, d_vocab=D_VOCAB)
        self.compatibility_mode = False
        self.tokenizer = None
        self.blocks = nn.ModuleList([_StateBlock(layer) for layer in range(N_LAYERS)])

    @property
    def hook_dict(self) -> dict[str, HookPoint]:
        return {
            f"blocks.{layer}.hook_out": block.hook_out
            for layer, block in enumerate(self.blocks)
        }

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seed = int(tokens.sum()) + tokens.shape[1]
        generator = torch.Generator().manual_seed(seed)
        residual = (
            torch.randn(tokens.shape[0], tokens.shape[1], D_MODEL, generator=generator) * 0.02
        )
        for block in self.blocks:
            residual = block(residual)
        return residual

    def layer_types(self) -> list[str]:
        return ["attn+mlp"] * N_LAYERS


# A fixed [2, D_MODEL] readout: state -> residual-stream direction. Layer 1
# writes the "world model" into the stream along these rows; the probe has to
# find them from activations alone.
torch.manual_seed(7)
_READOUT = torch.randn(2, D_MODEL)


class _StateBlock(nn.Module):
    """Identity block; layer 1 writes a linear function of a state signal."""

    def __init__(self, layer: int) -> None:
        super().__init__()
        self.layer = layer
        self.hook_out = HookPoint()
        self.hook_out.name = f"blocks.{layer}.hook_out"

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        out = residual
        if self.layer == LAYER:
            out = residual + _state_signal(residual) @ _READOUT
        return self.hook_out(out)


def _state_signal(residual: torch.Tensor) -> torch.Tensor:
    """Deterministic per-position 'task state' the probe must recover."""
    n = residual.shape[1]
    positions = torch.arange(n, dtype=torch.float32)
    return torch.stack([torch.sin(positions / 3.0), torch.cos(positions / 5.0)], dim=-1).unsqueeze(
        0
    )


def _run(model: _ToyBridge, tokens: torch.Tensor) -> torch.Tensor:
    return model(tokens)


def _cache_from_model(model: _ToyBridge) -> ActivationCache:
    """Run the toy model under the real cache machinery and return the cache."""
    tokens = torch.zeros(1, SEQ_LEN, dtype=torch.long)
    cache_dict: dict[str, torch.Tensor] = {}

    def cache_hook(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        assert hook.name is not None
        cache_dict[hook.name] = activation.detach()
        return activation

    with model.hooks(fwd_hooks=[(name, cache_hook) for name in model.hook_dict]):
        model(tokens)
    return ActivationCache(cache_dict, model, has_batch_dim=True)


@pytest.fixture(scope="module")
def toy_cache() -> ActivationCache:
    return _cache_from_model(_ToyBridge())


@pytest.fixture(scope="module")
def states() -> torch.Tensor:
    return _state_signal(torch.zeros(1, SEQ_LEN, D_MODEL))


def _layer_out(cache: ActivationCache, layer: int = LAYER) -> torch.Tensor:
    """Layer output under the Bridge-native cache name this toy produces."""
    return cache[f"blocks.{layer}.hook_out"]


def test_probe_recovers_the_linear_world_model(toy_cache, states):
    probe = fit_world_model_probe(toy_cache, layer=LAYER, states=states)
    assert probe.explained_variance > 0.99
    assert probe.state_dim == 2
    assert probe.read(_layer_out(toy_cache)).shape[-1] == 2


def test_probe_fitted_on_untouched_layer_is_weak(toy_cache, states):
    """Layer 0 carries no state signal, so a probe fitted there must not
    claim a world model — the fidelity score has to discriminate."""
    probe = fit_world_model_probe(toy_cache, layer=0, states=states)
    assert probe.explained_variance < 0.99


def test_probe_layer_round_trips_through_the_cache(toy_cache, states):
    probe = fit_world_model_probe(toy_cache, layer=-2, states=states)
    assert probe.layer == LAYER


def test_fidelity_curve_reads_high_across_an_intact_run(toy_cache, states):
    probe = fit_world_model_probe(toy_cache, layer=LAYER, states=states)
    fidelity = probe_fidelity_by_position(toy_cache, probe, states, window=8)
    assert fidelity.positions == [7, 15, 23, 31, 39, 47, 55, 63]
    assert fidelity.window == 8
    assert fidelity.baseline == probe.explained_variance
    assert all(score > 0.95 for score in fidelity.explained_variance)


def test_fidelity_curve_drops_when_the_representation_decays(toy_cache, states):
    """The paper's core measurement: score the *same* probe on a run where the
    world model has been destroyed late in the sequence. The early windows stay
    faithful and the late ones collapse."""
    probe = fit_world_model_probe(toy_cache, layer=LAYER, states=states)

    decayed = dict(toy_cache.cache_dict)
    intact = decayed[f"blocks.{LAYER}.hook_out"].clone()
    intact[:, SEQ_LEN // 2 :, :] = torch.randn_like(intact[:, SEQ_LEN // 2 :, :])
    decayed[f"blocks.{LAYER}.hook_out"] = intact
    decayed_cache = ActivationCache(decayed, toy_cache.model, has_batch_dim=True)

    fidelity = probe_fidelity_by_position(decayed_cache, probe, states, window=8)
    assert fidelity.explained_variance[0] > 0.9
    assert fidelity.explained_variance[-1] < 0.5


def test_restoration_hook_moves_the_stream_back_to_the_prompt_anchor(toy_cache, states):
    """The paper's causal stage: anchor on the intact run, then restore a run
    whose late positions have lost the state representation."""
    probe = fit_world_model_probe(toy_cache, layer=LAYER, states=states)
    hook_name, hook_fn = restoration_hooks(probe, alpha=1.0)[0]
    assert hook_name == f"blocks.{LAYER}.hook_out"

    intact = _layer_out(toy_cache)
    decayed = intact.clone()
    decayed[:, SEQ_LEN // 2 :, :] = torch.randn_like(decayed[:, SEQ_LEN // 2 :, :])

    hook = SimpleNamespace(name=hook_name)

    # First invocation only captures the prompt-time anchor...
    hook_fn(intact, hook)
    # ...the second restores the decayed half toward it.
    restored = hook_fn(decayed, hook)

    # Undecayed head is left alone.
    assert torch.allclose(restored[:, : SEQ_LEN // 2], decayed[:, : SEQ_LEN // 2])
    # The decayed tail is pulled back into the world-model subspace: its
    # subspace coordinates now match the intact run's, position for position.
    assert torch.allclose(
        _project(restored, probe), _project(intact, probe), atol=1e-5
    )
    # ...and everything outside the subspace (the injected noise) is untouched.
    assert torch.allclose(
        restored - _project(restored, probe),
        decayed - _project(decayed, probe),
        atol=1e-5,
    )
    assert not torch.allclose(restored, decayed)


def _project(activation: torch.Tensor, probe) -> torch.Tensor:
    """Project onto the world-model subspace the fitted probe spans."""
    basis = probe.directions
    projector = basis.T @ torch.linalg.pinv(basis.T)
    return activation @ projector.T


def test_restoration_alpha_zero_is_a_noop(toy_cache, states):
    probe = fit_world_model_probe(toy_cache, layer=LAYER, states=states)
    hook_fn = restoration_hooks(probe, alpha=0.0)[0][1]
    activation = _layer_out(toy_cache)
    hook_fn(activation, SimpleNamespace(name="hook"))
    assert torch.allclose(hook_fn(activation, SimpleNamespace(name="hook")), activation)


def test_restoration_rejects_out_of_range_alpha(toy_cache, states):
    probe = fit_world_model_probe(toy_cache, layer=LAYER, states=states)
    with pytest.raises(ValueError, match="alpha"):
        restoration_hooks(probe, alpha=1.5)


def test_hooks_run_under_the_real_hook_machinery(toy_cache, states):
    """The integration surface: the returned hook list drives ``model.hooks``.

    Two real forward passes. The first (a different token pattern, so a
    different stream) becomes the prompt-time anchor. The second pass runs
    with the hooks attached, and the layer-1 output block 2 consumes is
    restored toward that anchor's subspace coordinates."""
    probe = fit_world_model_probe(toy_cache, layer=LAYER, states=states)
    hook_name, hook_fn = restoration_hooks(probe, alpha=1.0)[0]

    model = _ToyBridge()
    anchor_pass = torch.zeros(1, SEQ_LEN, dtype=torch.long)
    intervened_pass = torch.ones(1, SEQ_LEN, dtype=torch.long)

    # Unhooked reference for the intervened tokens.
    baseline = _run(model, intervened_pass)

    captured: list[torch.Tensor] = []

    def capture(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        del hook
        captured.append(activation.detach().clone())
        return activation

    with model.hooks(fwd_hooks=[(hook_name, hook_fn), (hook_name, capture)]):
        _run(model, anchor_pass)  # first chunk: capture the anchor
        intervened = _run(model, intervened_pass)  # second chunk: restore

    assert len(captured) == 2
    # The anchor pass is returned unchanged; the intervened pass is not.
    assert not torch.allclose(intervened, baseline)
    # After restoration, both passes carry the same subspace coordinates.
    assert torch.allclose(
        _project(captured[1], probe), _project(captured[0], probe), atol=1e-5
    )


def test_hybrid_layer_is_refused(states):
    model = _ToyBridge()

    def hybrid_types() -> list[str]:
        return ["attn+mlp", "mamba+mlp", "attn+mlp"]

    model.layer_types = hybrid_types  # type: ignore[method-assign]
    cache = ActivationCache(
        {f"blocks.{layer}.hook_out": torch.randn(1, SEQ_LEN, D_MODEL) for layer in range(N_LAYERS)},
        model,
        has_batch_dim=True,
    )
    with pytest.raises(NotImplementedError, match="mamba"):
        fit_world_model_probe(cache, layer=LAYER, states=states)


def test_misaligned_states_are_rejected(toy_cache):
    with pytest.raises(ValueError, match="one label per cached position"):
        fit_world_model_probe(toy_cache, layer=LAYER, states=torch.zeros(1, 3, 2))


def test_empty_fit_positions_are_rejected(toy_cache, states):
    with pytest.raises(ValueError, match="at least one index"):
        fit_world_model_probe(toy_cache, layer=LAYER, states=states, fit_positions=[])


def test_capabilities_are_exported_from_the_analysis_package():
    from transformer_lens.tools import analysis

    assert analysis.fit_world_model_probe is fit_world_model_probe
    assert analysis.probe_fidelity_by_position is probe_fidelity_by_position
    assert analysis.restoration_hooks is restoration_hooks
    assert hasattr(analysis, "WorldModelProbe")
