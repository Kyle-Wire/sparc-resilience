"""
CorrelogramReport — typed output of Stage 0 (correlogram analysis).

This dataclass is the seam between Stage 0 and all downstream consumers.
Every Stage 0 result that matters to later stages (GWEN bandwidth selection,
spatial CV block sizing, stationarity diagnostics) is exposed here.

Downstream modules (e.g. :class:`~sparc.run.bandwidth_advisor.BandwidthAdvisor`)
read only this type — never the raw artifact dict — so that a field rename in
Stage 0 is caught at import time rather than silently producing wrong defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CorrelogramReport:
    """Typed output of the Stage 0 Correlogram Analysis.

    Parameters
    ----------
    bandwidths:
        Per-variable optimal bandwidth in metres, keyed by variable name.
        Derived from the zero-crossing of the empirical Moran's I curve.
    block_size:
        Spatial cross-validation block size in metres.  Used by Stage 1
        (GWEN) and Stage 2 (Spatial CV).
    block_size_source:
        How the block size was determined: ``"user"`` (config override),
        ``"correlogram"`` (target variable's autocorrelation range), or
        ``"correlogram_with_crossrange_uplift"`` (uplifted because a
        predictor couples to the target at a longer range).
    stationarity_warnings:
        Variable names whose Matérn κ diverges significantly from the
        PDE-derived κ, suggesting spatial non-stationarity.
    variable_effective_ranges:
        Per-variable effective autocorrelation range in metres.  For
        isotropic Matérn: effective_range ≈ √8ν / κ.
    cross_range_uplift:
        Optional dict describing a cross-range uplift event: the original
        block size, the uplifted size, and the responsible predictor.
    dataset_profile:
        Optional dataset-level summary (n_samples, spatial_extent, etc.)
        from :class:`~sparc.run.dataset_profiler.DatasetProfiler`.
    """

    bandwidths: Dict[str, float]
    block_size: float
    block_size_source: str
    stationarity_warnings: List[str]
    variable_effective_ranges: Dict[str, float]
    cross_range_uplift: Optional[Dict] = None
    dataset_profile: Optional[Dict] = None
