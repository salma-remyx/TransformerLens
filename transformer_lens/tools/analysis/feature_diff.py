"""Dictionary diffing for feature discovery and control.

Adapted from `Multimodal Model Diffing for Feature Discovery and Control
<https://arxiv.org/abs/2608.09928v1>`_ (MMDiff). The paper trains sparse
autoencoders on a base language model and on its multimodally-adapted
counterpart over paired corpora, then *diffs* the two dictionaries to find
the features that multimodal training changed. This module keeps that
core — diff two SAE dictionaries over the same residual stream to rank
features by how far their decoder direction and firing statistics moved —
and substitutes the paper's auxiliary infrastructure with what the repo
already ships: instead of requiring externally fitted SAELens artifacts,
an SAE can be fitted natively from cached ``TransformerBridge``
activations, following the native-fitting precedent of
:class:`~transformer_lens.tools.analysis.jacobian_lens.JacobianLens`.
Diffing itself works on plain matrices too, so SAELens / published
dictionaries can be compared without any fitting here.

Modes kept from the paper:

- **Feature isolation** (:class:`FeatureDiff`, :func:`feature_diff`):
  match features between a base and an adapted dictionary and rank them
  by direction change and firing-rate change. Large scores are features
  the adaptation repurposed or suppressed; scores near zero are features
  it left alone.
- **Per-token contrastive firing** (:func:`contrastive_firing`): score
  each feature by how differently it fires on paired activations — the
  paper's task-specific feature detector, which only needs the decoder
  of one dictionary.
- **Feature-level control** (:class:`SAE.feature_hooks`,
  :class:`FeatureDiff.control_hooks`): hooks that ablate (project out)
  or steer along discovered feature directions, using the same
  ``blocks.{layer}.hook_out`` surface and hardening the Jacobian lens
  interventions use.

Deliberately out of scope: the paper's multimodal corpora, safety/OCR
benchmark suites, and any evaluation of downstream behaviour — those
belong to downstream work. Because the diff needs only cached residual
activations, the analysis applies unchanged to a multimodal tower's
cached hidden states, which is the repo-supported surface (full
multimodal generation is not).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from jaxtyping import Float
from tqdm.auto import tqdm

from transformer_lens.tools.analysis.jacobian_lens import (
    _cached_on_device,
    _make_intervention_hook,
    _normalize_layer,
    _resid_post_hook_name,
)

# ---------------------------------------------------------------------------
# dictionary (SAE) container + native fitting
# ---------------------------------------------------------------------------

DEFAULT_K = 8
DEFAULT_STEPS = 1000
DEFAULT_LR = 1e-3


class SAE:
    """A fitted sparse autoencoder dictionary over one residual stream.

    Top-K autoencoder: the encoder is a learned basis ``W_enc`` (with bias
    ``b_enc``) and only the ``k`` largest pre-activations survive, giving the
    sparse feature activations ``z = topk(relu(W_enc^T (x - b_dec)) + b_enc)``.
    The decoder ``W_dec`` holds one unit direction per feature and defines
    the feature directions used for diffing and control. Only unit-norm
    decoder columns make dictionary comparison meaningful, so columns are
    renormalized after every optimizer step (the TopK-SAE recipe; unit-norm
    decoding is also what MMDiff relies on when it diffs dictionaries).

    Attributes:
        W_dec: Decoder directions ``[n_features, d_model]``, fp32, unit rows.
        W_enc: Encoder basis ``[d_model, n_features]``, fp32.
        b_dec: Pre-encoder bias subtracted from the residual stream.
        b_enc: Per-feature encoder bias ``[n_features]``.
        k: Sparsity level (features kept per token).
        layer: Source layer the dictionary was fitted on, when known.
        metadata: Optional provenance (model name, corpus, hook name).
    """

    def __init__(
        self,
        W_enc: Float[torch.Tensor, "d_model n_features"],
        W_dec: Float[torch.Tensor, "n_features d_model"],
        b_dec: Float[torch.Tensor, "d_model"],
        b_enc: Float[torch.Tensor, "n_features"],
        k: int,
        *,
        layer: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if W_enc.shape[0] != W_dec.shape[1] or W_enc.shape[1] != W_dec.shape[0]:
            raise ValueError(
                f"W_enc {tuple(W_enc.shape)} and W_dec {tuple(W_dec.shape)} disagree on "
                "d_model / n_features"
            )
        if b_dec.shape != (W_dec.shape[1],) or b_enc.shape != (W_enc.shape[1],):
            raise ValueError("bias shapes must be [d_model] and [n_features]")
        self.W_enc = W_enc.detach().float().cpu()
        self.W_dec = W_dec.detach().float().cpu()
        self.b_dec = b_dec.detach().float().cpu()
        self.b_enc = b_enc.detach().float().cpu()
        self.k = int(k)
        self.layer = None if layer is None else int(layer)
        self.metadata: Dict[str, Any] = dict(metadata or {})

    @property
    def n_features(self) -> int:
        """Number of features in the dictionary."""
        return self.W_dec.shape[0]

    @property
    def d_model(self) -> int:
        """Residual-stream width the dictionary was fitted for."""
        return self.W_dec.shape[1]

    def __repr__(self) -> str:
        return (
            f"SAE(n_features={self.n_features}, d_model={self.d_model}, k={self.k}, "
            f"layer={self.layer})"
        )

    def encode(self, activations: torch.Tensor) -> Float[torch.Tensor, "*batch n_features"]:
        """Encode residual activations into sparse feature activations.

        Args:
            activations: Tensor whose last axis is ``d_model`` (any leading
                batch/position axes).

        Returns:
            Top-K sparse activations with the same leading axes.
        """
        if activations.shape[-1] != self.d_model:
            raise ValueError(
                f"activations have d_model {activations.shape[-1]}, dictionary was "
                f"fitted for {self.d_model}"
            )
        pre = torch.relu((activations.float() - self.b_dec) @ self.W_enc + self.b_enc)
        values, indices = torch.topk(pre, min(self.k, self.n_features), dim=-1)
        return torch.zeros_like(pre).scatter(-1, indices, values)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Reconstruct residual activations from feature activations."""
        return features @ self.W_dec + self.b_dec

    # ------------------------------------------------------------------ #
    # fitting                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def fit(
        cls,
        activations: torch.Tensor,
        *,
        k: int = DEFAULT_K,
        steps: int = DEFAULT_STEPS,
        lr: float = DEFAULT_LR,
        seed: int = 0,
        layer: Optional[int] = None,
        show_progress: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SAE":
        """Fit a top-K SAE on cached residual activations.

        Optimizes the standard reconstruction objective with a top-K sparsity
        penalty (no sparsity coefficient needed — top-K *is* the constraint),
        Adam on the repo's existing ``torch`` optimizer path, unit-norm decoder
        columns renormalized after every step. Activations are flattened over
        every axis but the last, so a cached ``[batch, pos, d_model]`` tensor
        can be passed as-is.

        Args:
            activations: ``[..., d_model]`` residual activations at one layer,
                typically ``cache[f"blocks.{layer}.hook_out"]``.
            k: Features kept per token.
            steps: Adam steps over minibatches of the flattened activations.
            lr: Learning rate.
            seed: Initialization / shuffle seed; the fit is deterministic
                given the activations and seed.
            layer: Source layer, recorded on the fitted dictionary.
            show_progress: Show a tqdm progress bar over steps.
            metadata: Extra provenance merged into :attr:`metadata`.

        Returns:
            The fitted :class:`SAE`.

        Raises:
            ValueError: If the activations are empty or ``k``/``steps`` are
                non-positive.
        """
        if activations.ndim < 1 or activations.shape[-1] < 1:
            raise ValueError("activations must have a trailing d_model axis")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if activations.numel() == 0:
            raise ValueError("cannot fit on an empty activation tensor")

        generator = torch.Generator().manual_seed(seed)
        flat = activations.detach().reshape(-1, activations.shape[-1]).float().cpu()
        d_model = flat.shape[1]
        n_features = max(4 * d_model, 8 * k)
        # Mean-center via the decoder bias, as in the TopK-SAE recipe.
        b_dec = flat.mean(dim=0).clone()

        # Near-identity init: identity rows make early decoding well-posed,
        # the small noise breaks symmetry between features.
        eye = torch.eye(n_features, d_model)
        W_dec = (eye + 0.02 * torch.randn(n_features, d_model, generator=generator)).clone()
        W_enc = W_dec.T.clone()
        b_enc = torch.zeros(n_features)
        for parameter in (W_dec, W_enc, b_enc):
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam([W_dec, W_enc, b_enc], lr=lr)

        batch = min(1024, flat.shape[0])
        iterator = tqdm(range(steps), desc="fitting SAE", disable=not show_progress)
        for _ in iterator:
            rows = torch.randint(0, flat.shape[0], (batch,), generator=generator)
            centred = flat[rows] - b_dec
            pre = torch.relu(centred @ W_enc + b_enc)
            values, indices = torch.topk(pre, min(k, n_features), dim=-1)
            sparse = torch.zeros_like(pre).scatter(-1, indices, values)
            recon = sparse @ W_dec
            loss = torch.nn.functional.mse_loss(recon, centred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                # Unit-norm decoder columns: the constraint dictionary diffing
                # and projection-based control both rely on.
                norms = W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
                W_enc.mul_(norms.T)  # keep W_dec @ W_enc^T scale-invariant
                W_dec.div_(norms)

        fit_metadata = {
            "k": k,
            "steps": steps,
            "lr": lr,
            "seed": seed,
            "n_tokens": int(flat.shape[0]),
        }
        fit_metadata.update(metadata or {})
        return cls(W_enc, W_dec, b_dec, b_enc, k, layer=layer, metadata=fit_metadata)

    # ------------------------------------------------------------------ #
    # control                                                            #
    # ------------------------------------------------------------------ #

    def feature_directions(self, features: Sequence[int]) -> Float[torch.Tensor, "n d_model"]:
        """Unit-normalized decoder directions for the given feature indices."""
        if not features:
            raise ValueError("features must contain at least one index")
        invalid = [f for f in features if not 0 <= f < self.n_features]
        if invalid:
            raise ValueError(f"feature ids {invalid} out of range for {self.n_features} features")
        vectors = self.W_dec[list(features)]
        norms = vectors.norm(dim=-1, keepdim=True)
        if (~torch.isfinite(norms) | (norms <= torch.finfo(torch.float32).eps)).any():
            raise ValueError("feature directions contain a zero or non-finite direction")
        return vectors / norms

    def feature_hooks(
        self,
        model: Any,
        features: Sequence[int],
        layer: Optional[int] = None,
        *,
        alpha: float = 0.0,
    ) -> List[Tuple[str, Any]]:
        """Hooks that ablate or steer along feature directions.

        With ``alpha == 0`` (ablation) each unit feature direction ``v̂`` is
        projected out of the residual stream: ``h <- h - (h·v̂) v̂``, applied
        sequentially for several features. With ``alpha != 0`` (steering) the
        direction is added instead, scaled by ``alpha`` times the activation's
        median per-position residual norm — the same norm-matched
        parameterization the Jacobian lens steering hooks use.

        Args:
            model: The model the hooks will run on; supplies the layer count
                when ``layer`` is negative.
            features: Feature indices from this dictionary.
            layer: Layer whose block output to intervene on. Defaults to the
                layer the dictionary was fitted on; negative indices count
                from ``model.cfg.n_layers``.
            alpha: Steering strength; ``0`` (the default) ablates.

        Returns:
            ``[(hook_name, fn), ...]`` for ``model.hooks(fwd_hooks=...)`` or
            ``model.run_with_hooks(fwd_hooks=...)``.
        """
        resolved = self._resolve_layer(model, layer)
        units = self.feature_directions(features)
        device_units: Dict[torch.device, torch.Tensor] = {}

        def transform(
            selected: Float[torch.Tensor, "batch pos d_model"],
            units: torch.Tensor = units,
            device_units: Dict[torch.device, torch.Tensor] = device_units,
        ) -> Float[torch.Tensor, "batch pos d_model"]:
            local_units = _cached_on_device(units, device_units, selected.device)
            result = selected.float()
            if alpha == 0.0:
                for unit in local_units:
                    coeff = result @ unit
                    result = result - coeff.unsqueeze(-1) * unit
            else:
                scale = alpha * selected.float().norm(dim=-1).median()
                result = result + scale * local_units.sum(dim=0)
            return result

        hook = _make_intervention_hook(transform, None, self.d_model)
        return [(_resid_post_hook_name(resolved), hook)]

    def _resolve_layer(self, model: Any, layer: Optional[int]) -> int:
        if layer is None:
            layer = self.layer
        if layer is None:
            raise ValueError(
                "layer is required (pass layer=, or fit/load the dictionary with a recorded layer)"
            )
        return _normalize_layer(layer, model.cfg.n_layers)


# ---------------------------------------------------------------------------
# dictionary diffing
# ---------------------------------------------------------------------------


@dataclass
class FeatureDiff:
    """Result of a :func:`feature_diff` call.

    Attributes:
        scores:
            Per-feature diff score, aligned with the base dictionary's
            feature axis. ``[n_features_base]``.
        directions:
            Unit decoder directions of the matched features — the adapted
            dictionary's direction where a match was found, otherwise the
            base dictionary's — aligned with ``scores``.
            ``[n_features_base, d_model]``.
        match_indices:
            Matched adapted-dictionary feature index per base feature, or
            ``-1`` when unmatched.
        layer: Source layer the diff was computed at.
        method: Matching method used (``"greedy"``).
    """

    scores: Float[torch.Tensor, "n_features"]
    directions: Float[torch.Tensor, "n_features d_model"]
    match_indices: List[int]
    layer: Optional[int]
    method: str = "greedy"

    def top(self, k: int = 10) -> List[Tuple[int, float, int]]:
        """Return the ``k`` highest-scoring ``(feature, score, match)`` triples.

        Unmatched features report ``match == -1``.
        """
        count = min(k, self.scores.shape[0])
        values, indices = torch.topk(self.scores, count)
        return [
            (int(feature), float(value), self.match_indices[feature])
            for feature, value in zip(indices.tolist(), values.tolist())
        ]

    def bottom(self, k: int = 10) -> List[Tuple[int, float, int]]:
        """Return the ``k`` lowest-scoring triples (most-suppressed features)."""
        count = min(k, self.scores.shape[0])
        values, indices = torch.topk(self.scores, count, largest=False)
        return [
            (int(feature), float(value), self.match_indices[feature])
            for feature, value in zip(indices.tolist(), values.tolist())
        ]


def feature_diff(
    base: SAE,
    adapted: SAE,
    *,
    base_activations: Optional[torch.Tensor] = None,
    adapted_activations: Optional[torch.Tensor] = None,
    activation_weight: float = 0.5,
) -> FeatureDiff:
    """Diff a base dictionary against its adapted counterpart.

    Features are matched by decoder-direction cosine similarity: each base
    feature takes its best adapted feature, and each adapted feature is
    matched to at most one base feature (greedy over all pairs, highest
    cosine first). A matched feature scores

    .. math::

        (1 - \\cos(v_{base}, v_{adapted})) + w \\cdot |\\log f_{adapted} - \\log f_{base}|

    where :math:`f` is the feature's mean firing rate (top-K membership)
    over the supplied activations. Direction change and firing-rate change
    are both magnitudes, so a large score marks a feature the adaptation
    repurposed or suppressed and a score near zero one it barely touched —
    use the sign of the underlying firing-rate change if you need the
    direction of the change. Unmatched base features score ``2.0`` (the
    maximum direction change; they were replaced by features with no
    counterpart) and keep their base direction — they are the strongest
    isolation signal, so they sort to the top.

    Args:
        base: Dictionary fitted on the base model's activations.
        adapted: Dictionary fitted on the adapted model's activations over
            the paired corpus. Must share ``d_model`` with ``base``.
        base_activations:
            Residual activations the base dictionary fires over; enables the
            firing-rate term. Optional — without it the diff is
            direction-only.
        adapted_activations:
            Paired activations for the adapted dictionary. Required whenever
            ``base_activations`` is given.
        activation_weight:
            Weight ``w`` on the firing-rate term relative to the direction
            term.

    Returns:
        A :class:`FeatureDiff` aligned with ``base``'s feature axis.

    Raises:
        ValueError: On mismatched ``d_model``, mismatched activation shapes,
            or ``adapted_activations`` missing when needed.
    """
    if base.d_model != adapted.d_model:
        raise ValueError(f"dictionaries disagree on d_model: {base.d_model} vs {adapted.d_model}")
    if (base_activations is None) != (adapted_activations is None):
        raise ValueError("pass both base_activations and adapted_activations, or neither")
    base_units = base.feature_directions(range(base.n_features))
    adapted_units = adapted.feature_directions(range(adapted.n_features))
    cosine = base_units @ adapted_units.T  # [n_base, n_adapted]

    pairs = _greedy_matches(cosine)
    match_indices = [-1] * base.n_features
    direction_change = torch.full((base.n_features,), 2.0)
    for base_idx, adapted_idx in pairs:
        match_indices[base_idx] = adapted_idx
        direction_change[base_idx] = 1.0 - cosine[base_idx, adapted_idx]

    directions = base_units.clone()
    if pairs:
        matched_base = torch.tensor([b for b, _ in pairs])
        matched_adapted = torch.tensor([a for _, a in pairs])
        directions[matched_base] = adapted_units[matched_adapted]

    scores = direction_change.clone()
    if base_activations is not None and adapted_activations is not None:
        base_rate = _mean_firing_rate(base, base_activations)
        adapted_rate = _mean_firing_rate(adapted, adapted_activations)
        rate_change = (adapted_rate + 1e-6).log() - (base_rate + 1e-6).log()
        scores = scores + activation_weight * rate_change.abs()

    return FeatureDiff(
        scores=scores,
        directions=directions,
        match_indices=match_indices,
        layer=base.layer,
    )


def _greedy_matches(cosine: torch.Tensor) -> List[Tuple[int, int]]:
    """Greedy one-to-one matching over a cosine-similarity matrix.

    Pairs are taken highest-cosine first; each row and column is used at
    most once. The threshold is 0 — a negative-cosine pair is not the same
    feature, so it is left unmatched rather than forced together.
    """
    if cosine.numel() == 0:
        return []
    values, flat_indices = cosine.flatten().sort(descending=True)
    rows = torch.div(flat_indices, cosine.shape[1], rounding_mode="floor").tolist()
    cols = (flat_indices % cosine.shape[1]).tolist()
    used_rows: set = set()
    used_cols: set = set()
    pairs: List[Tuple[int, int]] = []
    for value, row, col in zip(values.tolist(), rows, cols):
        if value <= 0.0:
            break
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(row)
        used_cols.add(col)
        pairs.append((row, col))
    return pairs


def _mean_firing_rate(dictionary: SAE, activations: torch.Tensor) -> torch.Tensor:
    """Fraction of tokens on which each feature is in the top-K, ``[n]``."""
    if activations.shape[-1] != dictionary.d_model:
        raise ValueError(
            f"activations have d_model {activations.shape[-1]}, dictionary was "
            f"fitted for {dictionary.d_model}"
        )
    features = dictionary.encode(activations)
    return (features > 0).float().mean(dim=tuple(range(features.ndim - 1)))


def contrastive_firing(
    dictionary: SAE,
    clean: torch.Tensor,
    contrast: torch.Tensor,
) -> Float[torch.Tensor, "n_features"]:
    """Per-token contrastive firing scores for one dictionary.

    Each feature is scored by the total variation distance between its
    top-K firing indicator on the clean activations and on the contrast
    activations, averaged over positions:

    .. math::

        s_f = \\frac{1}{T}\\sum_t |\\mathbb{1}[f \\text{ fires at } t \\mid clean]
        - \\mathbb{1}[f \\text{ fires at } t \\mid contrast]|

    A feature that fires on exactly one side of the pair scores 1; one that
    fires on both or neither scores 0. This is the paper's task-specific
    feature detector reduced to its parameter-free core — it isolates
    features whose firing is caused by the contrast (e.g. the visual
    tokens of a multimodal input) without any causal-ablation loop.

    Args:
        dictionary: The dictionary whose features are being scored.
        clean: ``[..., d_model]`` baseline activations.
        contrast: Same-shaped paired activations (e.g. the same prompts with
            the visual tokens present).

    Returns:
        Per-feature scores in ``[0, 1]``, aligned with the dictionary's
        feature axis.

    Raises:
        ValueError: If the two activation tensors disagree on shape or
            ``d_model``.
    """
    if clean.shape != contrast.shape:
        raise ValueError(
            f"clean {tuple(clean.shape)} and contrast {tuple(contrast.shape)} must be paired "
            "(same shape)"
        )
    clean_rate = _mean_firing_rate(dictionary, clean)
    contrast_rate = _mean_firing_rate(dictionary, contrast)
    return (contrast_rate - clean_rate).abs()


def control_hooks(
    dictionary: SAE,
    model: Any,
    features: Sequence[int],
    layer: Optional[int] = None,
    *,
    alpha: float = 0.0,
) -> List[Tuple[str, Any]]:
    """Hooks that ablate or steer along discovered feature directions.

    Thin wrapper over :meth:`SAE.feature_hooks` named for the diffing
    workflow: pass the feature indices a :class:`FeatureDiff` or
    :func:`contrastive_firing` call surfaced, plus the dictionary they came
    from.

    Args:
        dictionary: The dictionary the features belong to.
        model: The model the hooks will run on.
        features: Feature indices to intervene on.
        layer: Layer to intervene on. Defaults to the dictionary's recorded
            layer.
        alpha: Steering strength; ``0`` (the default) ablates the directions
            instead of adding them.

    Returns:
        ``[(hook_name, fn), ...]`` for ``model.hooks(fwd_hooks=...)``.
    """
    return dictionary.feature_hooks(model, features, layer, alpha=alpha)


# ---------------------------------------------------------------------------
# shared intervention helpers live in jacobian_lens (imported at module top)
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_K",
    "FeatureDiff",
    "SAE",
    "contrastive_firing",
    "control_hooks",
    "feature_diff",
]
