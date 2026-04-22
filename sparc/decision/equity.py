"""Spatial equity scoring.

Combines candidate intervention locations with one or more "equity layers"
(population density, vulnerability index, heat-burden, …) to produce an
equity weight per candidate.

Layers are passed in as either:
  * A list of float values (already aligned to candidates), OR
  * A dict ``{name: list[float]}`` (multiple layers, combined via mean
    of normalised values), OR
  * A GeoDataFrame-style mapping (handled by the caller before invoking
    this module).

This module is intentionally pure-Python so it can run without geopandas
available on the FastAPI server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass
class EquityScore:
    """Per-candidate equity contribution with provenance."""

    candidate: str
    weight: float
    layer_breakdown: dict[str, float]


def _minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    finite = [float(v) for v in values if v == v]  # drop NaN
    if not finite:
        return [0.0] * len(values)
    lo, hi = min(finite), max(finite)
    span = hi - lo
    if span == 0:
        return [1.0] * len(values)
    return [
        ((float(v) - lo) / span) if (v == v) else 0.0
        for v in values
    ]


def combine_equity_layers(
    layers: Mapping[str, Sequence[float]] | Sequence[float],
    candidate_names: Sequence[str],
    *,
    weights: Mapping[str, float] | None = None,
    invert: Mapping[str, bool] | None = None,
) -> list[EquityScore]:
    """Combine one or more equity layers into per-candidate weights.

    Each layer is min-max normalised to [0, 1], optionally inverted (so a
    layer encoding "advantage" becomes "disadvantage"), and then weighted
    averaged.  The output ``weight`` is always in [0, 1].

    Parameters
    ----------
    layers
        Either a flat sequence (single layer) or a mapping of name→values.
    candidate_names
        Ordered list of candidate names; must match the layer length.
    weights
        Per-layer weights for the convex combination. Defaults to equal.
    invert
        If True for a given layer, normalised values are flipped (1 − x).
        Use this to convert e.g. an "income" layer into a vulnerability
        proxy.
    """
    if isinstance(layers, Mapping):
        layer_dict: dict[str, Sequence[float]] = dict(layers)
    else:
        layer_dict = {"equity": list(layers)}

    invert = invert or {}
    weights = weights or {}

    n = len(candidate_names)
    normalised: dict[str, list[float]] = {}
    for name, vals in layer_dict.items():
        v = list(vals)
        if len(v) != n:
            raise ValueError(
                f"Layer '{name}' has {len(v)} values but {n} candidates supplied"
            )
        normed = _minmax(v)
        if invert.get(name, False):
            normed = [1.0 - x for x in normed]
        normalised[name] = normed

    # Convex combination.
    layer_names = list(normalised.keys())
    raw_w = [float(weights.get(k, 1.0)) for k in layer_names]
    total = sum(raw_w) or 1.0
    norm_w = [w / total for w in raw_w]

    out: list[EquityScore] = []
    for i, cand in enumerate(candidate_names):
        breakdown = {name: normalised[name][i] for name in layer_names}
        weight = sum(breakdown[name] * norm_w[j] for j, name in enumerate(layer_names))
        out.append(EquityScore(candidate=cand, weight=weight, layer_breakdown=breakdown))
    return out


def disparity_index(weights: Sequence[float]) -> float:
    """Return a 0-1 disparity index (0 = uniform, 1 = maximally uneven).

    Defined as the coefficient of variation of weights, clipped to [0, 1].
    Useful as a single-number summary for the UI.
    """
    if not weights:
        return 0.0
    n = len(weights)
    mean = sum(weights) / n
    if mean == 0:
        return 0.0
    var = sum((w - mean) ** 2 for w in weights) / n
    cv = (var ** 0.5) / mean
    return max(0.0, min(1.0, cv))
