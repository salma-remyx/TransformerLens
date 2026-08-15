"""Layerwise massive-activation morphology for hybrid linear-attention models.

Adapted from *Massive Activations in Hybrid Linear Attention Large Language
Models: Pre-Attention Spikes and Inter-Spike Plateaus* (arXiv:2608.12149),
which shows that massive activations (MAs) in layer-interleaved hybrid LLMs
spike immediately before full attention layers (pre-attention spikes, PAS) and
persist through the intervening linear-attention layers (inter-spike plateaus,
ISP), recovering the stable MA morphology of full attention LLMs as full
attention becomes denser.

This module ports that *analysis* — not the paper's pretraining or gating
interventions — onto a cached ``TransformerBridge`` forward pass: it reads each
block's pre-block residual, flags channels whose max |activation| dwarfs the
layer median (massive activations), and reports where they spike relative to
the full attention layers and whether they persist across the SSM layers
between successive spikes. Reached through
:meth:`transformer_lens.ActivationCache.ActivationCache.massive_activation_profile`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from transformer_lens.ActivationCache import ActivationCache

DEFAULT_OUTLIER_FACTOR = 100.0
"""A channel is *massive* when its max |activation| exceeds this factor times
the layer median of per-channel maxima."""

DEFAULT_PERSIST_FACTOR = 10.0
"""Weaker threshold used to trace a spike's channels through the layers after it."""

_RESID_KEYS = ("blocks.{layer}.hook_resid_pre", "blocks.{layer}.hook_in")


def full_attention_layer_indices(model: object) -> List[int]:
    """Structurally detect which blocks hold full (softmax) attention.

    Family-agnostic, mirroring ``find_ssm_mixer``: a block is full attention
    when its module tree contains a Q/K projection — either a realized bridge
    attention child (``attn.q``) or the wrapped HF module's ``q_proj`` /
    ``k_proj`` (hybrids like NemotronH route attention through the ``.mixer``
    slot). Pure-MLP hybrid blocks expose neither and are excluded.

    Args:
        model: A ``TransformerBridge``.

    Returns:
        Ascending list of full-attention block indices.
    """
    indices: List[int] = []
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return indices
    for i, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        if attn is not None and getattr(attn, "q", None) is not None:
            indices.append(i)
            continue
        names = {name.split(".")[-1] for name, _ in block.named_modules()}
        if "q_proj" in names or "k_proj" in names:
            indices.append(i)
    return indices


@dataclass(frozen=True)
class LayerStats:
    """Per-layer massive-activation statistics over the pre-block residual.

    Attributes:
        layer: Block index.
        layer_kind: ``"full_attention"``, ``"ssm"``, or ``"other"``.
        max_abs: Largest |activation| in the layer.
        median_abs: Median over channels of each channel's max |activation|.
        massive_channels: Channels above ``outlier_factor`` × ``median_abs``.
        persistent_channels: Channels above ``persist_factor`` × ``median_abs``
            (the weaker trace used to follow a spike across later layers).
    """

    layer: int
    layer_kind: str
    max_abs: float
    median_abs: float
    massive_channels: Tuple[int, ...] = ()
    persistent_channels: Tuple[int, ...] = ()


@dataclass(frozen=True)
class Plateau:
    """A run of layers carrying a spike's channels between two PAS layers.

    Attributes:
        start: First layer strictly between the two spikes.
        end: Last layer strictly between the two spikes.
        connected: True when *every* layer in the run still holds at least one
            persistent channel — the paper's fully-connected ISP morphology.
    """

    start: int
    end: int
    connected: bool

    def __len__(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class MassiveActivationProfile:
    """Layerwise MA morphology report: spikes, PAS alignment, and plateaus.

    Attributes:
        layers: One :class:`LayerStats` per block, in block order.
        full_attention_layers: Block indices holding full attention.
        ssm_layers: Block indices holding an SSM / linear-attention mixer.
        spikes: Layers with at least one massive channel.
        pre_attention_spikes: Spikes immediately followed by full attention.
        plateaus: One entry per adjacent pair of pre-attention spikes with at
            least one layer between them.
    """

    layers: List[LayerStats] = field(default_factory=list)
    full_attention_layers: List[int] = field(default_factory=list)
    ssm_layers: List[int] = field(default_factory=list)
    spikes: List[int] = field(default_factory=list)
    pre_attention_spikes: List[int] = field(default_factory=list)
    plateaus: List[Plateau] = field(default_factory=list)

    def layer(self, index: int) -> LayerStats:
        """Return the stats for a block index."""
        return self.layers[index]

    @property
    def fully_connected_plateaus(self) -> List[Plateau]:
        """Plateaus whose layers all hold a persistent massive channel."""
        return [p for p in self.plateaus if p.connected]

    def summary(self) -> Dict[str, object]:
        """JSON-friendly summary of the morphology."""
        return {
            "full_attention_layers": self.full_attention_layers,
            "ssm_layers": self.ssm_layers,
            "spikes": self.spikes,
            "pre_attention_spikes": self.pre_attention_spikes,
            "plateaus": [
                {"start": p.start, "end": p.end, "connected": p.connected}
                for p in self.plateaus
            ],
            "max_abs_per_layer": [stats.max_abs for stats in self.layers],
        }

    def __str__(self) -> str:
        header = (
            f"{'layer':>5}  {'kind':<13} {'max|a|':>9}  {'massive-ch':>10}  note"
        )
        rows = [header]
        spike_set = set(self.spikes)
        pas_set = set(self.pre_attention_spikes)
        for stats in self.layers:
            note = ""
            if stats.layer in pas_set:
                note = "pre-attention spike"
            elif stats.layer in spike_set:
                note = "spike"
            rows.append(
                f"{stats.layer:>5}  {stats.layer_kind:<13} {stats.max_abs:>9.2e}  "
                f"{len(stats.massive_channels):>10}  {note}"
            )
        for p in self.plateaus:
            state = "connected" if p.connected else "partial"
            rows.append(f"plateau layers {p.start}..{p.end} ({state})")
        return "\n".join(rows)


def _residual_for_layer(
    cache: "ActivationCache", layer: int
) -> Optional[torch.Tensor]:
    """Read a block's pre-block residual, preferring ``hook_resid_pre``."""
    for template in _RESID_KEYS:
        tensor = cache.cache_dict.get(template.format(layer=layer))
        if tensor is not None:
            return tensor
    return None


def profile_massive_activations(
    cache: "ActivationCache",
    outlier_factor: float = DEFAULT_OUTLIER_FACTOR,
    persist_factor: float = DEFAULT_PERSIST_FACTOR,
    full_attention_layers: Optional[List[int]] = None,
    ssm_layers: Optional[List[int]] = None,
) -> MassiveActivationProfile:
    """Profile massive-activation morphology layer by layer from a cache.

    For every block this reads the pre-block residual (``hook_resid_pre``,
    falling back to ``hook_in``), takes each channel's max |activation| over
    batch and position, and flags channels that exceed ``outlier_factor`` times
    the layer median as massive. It then reports the paper's two morphologies
    against the bridge's own layer layout: spikes landing immediately before a
    full attention layer (PAS), and the runs of layers between successive PAS
    that still carry those channels (ISP).

    Args:
        cache: An :class:`ActivationCache` from ``run_with_cache`` on a
            ``TransformerBridge`` (hybrid or pure transformer).
        outlier_factor: Multiple of the layer median defining a massive channel.
        persist_factor: Weaker multiple used to trace channels across layers.
        full_attention_layers: Optional explicit block indices, overriding the
            structural detection (useful for hybrids the detector does not
            cover, and for synthetic caches).
        ssm_layers: Optional explicit SSM block indices, overriding
            ``cache.ssm_layers()``.

    Returns:
        The completed :class:`MassiveActivationProfile`.

    Raises:
        RuntimeError: If no block exposes a readable pre-block residual.
    """
    from transformer_lens.model_bridge.generalized_components.ssm_protocol import (
        find_ssm_mixer,
    )

    model = cache.model
    n_layers = model.cfg.n_layers
    if full_attention_layers is None:
        full_attention_layers = full_attention_layer_indices(model)
    if ssm_layers is None:
        ssm_layers = list(cache.ssm_layers())
    attn_set = set(full_attention_layers)
    ssm_set = set(ssm_layers)

    stats: List[LayerStats] = []
    for i in range(n_layers):
        residual = _residual_for_layer(cache, i)
        if residual is None:
            raise RuntimeError(
                f"Block {i} has no cached pre-block residual "
                f"(looked for blocks.{i}.hook_resid_pre / hook_in); "
                "run_with_cache the bridge first."
            )
        flat = residual.detach().abs().reshape(-1, residual.shape[-1])
        channel_max = flat.amax(dim=0)
        median = channel_max.median().item()
        if median > 0:
            massive = torch.nonzero(channel_max > outlier_factor * median)
            persistent = torch.nonzero(channel_max > persist_factor * median)
            massive_channels = tuple(int(c) for c in massive.flatten().tolist())
            persistent_channels = tuple(int(c) for c in persistent.flatten().tolist())
        else:
            massive_channels = ()
            persistent_channels = ()
        kind = "full_attention" if i in attn_set else "ssm" if i in ssm_set else "other"
        stats.append(
            LayerStats(
                layer=i,
                layer_kind=kind,
                max_abs=float(channel_max.max().item()),
                median_abs=median,
                massive_channels=massive_channels,
                persistent_channels=persistent_channels,
            )
        )

    spikes = [s.layer for s in stats if s.massive_channels]
    pas = [i for i in spikes if i + 1 in attn_set]

    plateaus: List[Plateau] = []
    for first, second in zip(pas, pas[1:]):
        middle = stats[first + 1 : second]
        if not middle:
            continue
        connected = all(s.persistent_channels for s in middle)
        plateaus.append(
            Plateau(start=middle[0].layer, end=middle[-1].layer, connected=connected)
        )

    return MassiveActivationProfile(
        layers=stats,
        full_attention_layers=full_attention_layers,
        ssm_layers=ssm_layers,
        spikes=spikes,
        pre_attention_spikes=pas,
        plateaus=plateaus,
    )
