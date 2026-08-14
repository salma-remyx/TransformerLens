"""Massive-activation detection and PAS/ISP morphology characterization.

Detects *massive activations* (MAs) — residual-stream coordinates whose
magnitude dwarfs the layer median — and classifies the two architecture-
aligned morphologies they form in hybrid linear-attention (HLA) LLMs:

* **Pre-attention spikes (PAS)**: MAs spiking immediately before a
  full-attention layer.
* **Inter-spike plateaus (ISP)**: elevated (sub-outlier) magnitudes
  persisting through the linear-attention layers between successive spikes,
  connecting them. At the full-attention limit the plateau criterion is
  vacuous and every spike is a pre-attention spike, recovering the stable
  MA morphology of full-attention LLMs.

Adapted from "Massive Activations in Hybrid Linear Attention Large Language
Models: Pre-Attention Spikes and Inter-Spike Plateaus"
(https://arxiv.org/abs/2608.12149). The detection/characterization core is
ported directly; the paper's controlled-pretraining and output-gating
ablations are out of scope for this post-hoc analysis utility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch

_FULL_ATTENTION = "full_attention"
_LINEAR_ATTENTION = "linear_attention"

_LAYER_TYPE_ALIASES = {
    "attention": _FULL_ATTENTION,
    "mamba": _LINEAR_ATTENTION,
}


def normalize_layer_types(raw_types: Sequence[str]) -> List[str]:
    """Normalize per-layer mixer-type names to canonical TL names.

    Mirrors ``ArchitectureAdapter._canonical_layer_types``: ``mamba`` ->
    ``linear_attention``, ``attention`` -> ``full_attention``; anything else
    (``mlp``, ``moe``, ...) passes through unchanged.
    """
    return [_LAYER_TYPE_ALIASES.get(t, t) for t in raw_types]


@dataclass
class MassiveActivationReport:
    """Per-layer massive-activation statistics and PAS/ISP morphology.

    Attributes:
        layer_max: (n_layers,) max |residual-stream activation| per layer.
        layer_median: (n_layers,) median |residual-stream activation| per layer.
        massive_ratio: (n_layers,) ``layer_max / layer_median``.
        layer_types: Canonical per-layer mixer types used for morphology
            (empty when no mixer-type map was available).
        massive_layers: Layers whose ratio reaches ``outlier_threshold``.
        pre_attention_spikes: Massive layers immediately preceding a
            full-attention layer (PAS).
        inter_spike_plateaus: Linear-attention layers between successive
            pre-attention spikes whose ratio reaches ``plateau_ratio`` but
            stays below ``outlier_threshold`` (ISP).
    """

    layer_max: torch.Tensor
    layer_median: torch.Tensor
    massive_ratio: torch.Tensor
    layer_types: List[str]
    massive_layers: List[int] = field(default_factory=list)
    pre_attention_spikes: List[int] = field(default_factory=list)
    inter_spike_plateaus: List[int] = field(default_factory=list)


def characterize_massive_activations(
    layer_max: torch.Tensor,
    layer_median: torch.Tensor,
    layer_types: Optional[Sequence[str]] = None,
    outlier_threshold: float = 100.0,
    plateau_ratio: float = 10.0,
) -> MassiveActivationReport:
    """Classify massive-activation morphology from per-layer statistics.

    Args:
        layer_max: (n_layers,) max |residual-stream activation| per layer.
        layer_median: (n_layers,) median |residual-stream activation| per layer.
        layer_types: Per-layer mixer types (aliases are normalized
            internally). Required for PAS/ISP classification; when None or
            the wrong length, only outlier detection is performed.
        outlier_threshold: max/median ratio at which a layer counts as
            massive (the classic MA criterion; default 100).
        plateau_ratio: ratio at which a non-massive linear-attention layer
            between successive spikes counts as inter-spike plateau.

    Returns:
        A :class:`MassiveActivationReport`.

    Raises:
        ValueError: If the thresholds are not ordered
            ``0 < plateau_ratio <= outlier_threshold``, or the per-layer
            statistic shapes disagree.
    """
    if outlier_threshold <= 0:
        raise ValueError("outlier_threshold must be positive.")
    if plateau_ratio <= 0 or plateau_ratio > outlier_threshold:
        raise ValueError("plateau_ratio must be in (0, outlier_threshold].")

    layer_max = layer_max.detach().float().flatten()
    layer_median = layer_median.detach().float().flatten()
    if layer_max.shape != layer_median.shape:
        raise ValueError("layer_max and layer_median must have the same shape.")
    n_layers = layer_max.numel()

    ratio = layer_max / layer_median.clamp_min(torch.finfo(layer_median.dtype).tiny)
    ratios: List[float] = ratio.tolist()
    massive_layers = [l for l in range(n_layers) if ratios[l] >= outlier_threshold]

    types = normalize_layer_types(layer_types) if layer_types is not None else []
    if len(types) != n_layers:
        types = []

    pre_attention_spikes: List[int] = []
    inter_spike_plateaus: List[int] = []
    if types:
        pre_attention_spikes = [
            l for l in massive_layers if l + 1 < n_layers and types[l + 1] == _FULL_ATTENTION
        ]
        for spike, next_spike in zip(pre_attention_spikes, pre_attention_spikes[1:]):
            for l in range(spike + 1, next_spike):
                is_plateau = plateau_ratio <= ratios[l] < outlier_threshold
                if types[l] == _LINEAR_ATTENTION and is_plateau:
                    inter_spike_plateaus.append(l)

    return MassiveActivationReport(
        layer_max=layer_max,
        layer_median=layer_median,
        massive_ratio=ratio,
        layer_types=types,
        massive_layers=massive_layers,
        pre_attention_spikes=pre_attention_spikes,
        inter_spike_plateaus=inter_spike_plateaus,
    )
