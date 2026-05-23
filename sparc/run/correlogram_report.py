"""
CorrelogramReport — typed output of Stage 0 (correlogram analysis).

This dataclass is the seam between Stage 0 and all downstream consumers.
Every Stage 0 result that matters to later stages (GWEN bandwidth selection,
spatial CV block sizing, stationarity diagnostics) is exposed here.

Downstream modules read only this type — never a raw artifact dict — so that
a field rename in Stage 0 is caught at import time rather than silently
producing wrong defaults.

The ``degradations`` field records every phase that fell back to a default
during Stage 0 execution.  A clean run has ``degradations == []``; any
swallowed exception is appended here instead of being silently discarded.
``is_clean`` provides a one-liner check.

``to_artifact()`` / ``from_artifact()`` form the single serialisation path
for all downstream consumers (Stage 2, desktop, report).  ``from_artifact``
raises ``ValueError`` on missing required keys so callers never get silent
None fallbacks.

``gwen_config(coords)`` is the Stage 0 → Stage 1 mapping.  It replaces
``BandwidthAdvisor.suggest()`` as the primary call site; BandwidthAdvisor
delegates here for backward compatibility.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from sparc.models.gwen_config import GWENConfig

# Clipping bounds for the bandwidth → k_neighbors formula (mirrors BandwidthAdvisor)
_MIN_K = 10
_MAX_K = 500

# Required keys that from_artifact() enforces
_REQUIRED_ARTIFACT_KEYS = ("bandwidths", "block_size", "block_size_source",
                            "stationarity_warnings", "variable_effective_ranges")


# ---------------------------------------------------------------------------
# Degradation — records a single phase fallback during Stage 0 execution
# ---------------------------------------------------------------------------


@dataclass
class Degradation:
    """Records one phase of Stage 0 that fell back to a default.

    Parameters
    ----------
    phase:
        Which sub-analysis degraded: ``"matern_fit"``, ``"anisotropy_fit"``,
        ``"cross_correlogram"``, ``"phase4_uplift"``, or ``"artifact_write"``.
    fallback_used:
        The default that was applied: ``"lsq"`` (Matérn fallback),
        ``"isotropic"``, ``"none"``, etc.
    reason:
        The exception message or a human-readable explanation.
    """

    phase: str
    fallback_used: str
    reason: str


# ---------------------------------------------------------------------------
# CorrelogramReport — typed Stage 0 output and the sole inter-stage seam
# ---------------------------------------------------------------------------


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
    degradations:
        List of :class:`Degradation` objects — one per Stage 0 phase that
        fell back to a default.  Empty on a clean run.
    sampling_config:
        The sampling parameters actually used during Stage 0 (n_samples,
        n_warmup, n_chains, n_perm, sample_cap).  Stored for reproducibility.
    matern_fits:
        Raw per-variable Matérn fit payload from Stage 0 Phase 1.
        Consumed by :meth:`~sparc.models.kernel_field.KernelField.from_report`.
    anisotropy_fits:
        Raw per-variable anisotropy fit payload from Stage 0 Phase 2.
        Consumed by :meth:`~sparc.models.kernel_field.KernelField.from_report`.
    effective_range_matrix:
        Raw V×V effective-range matrix payload from Stage 0 Phase 4.
        Consumed by :meth:`~sparc.models.kernel_field.KernelField.from_report`.
    """

    bandwidths: Dict[str, float]
    block_size: float
    block_size_source: str
    stationarity_warnings: List[str]
    variable_effective_ranges: Dict[str, float]
    cross_range_uplift: Optional[Dict] = None
    dataset_profile: Optional[Dict] = None
    degradations: List[Degradation] = field(default_factory=list)
    sampling_config: Optional[Dict] = None
    matern_fits: Optional[Dict] = None
    anisotropy_fits: Optional[Dict] = None
    effective_range_matrix: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Execution-health property
    # ------------------------------------------------------------------

    @property
    def is_clean(self) -> bool:
        """True when Stage 0 completed without any phase degradations."""
        return len(self.degradations) == 0

    # ------------------------------------------------------------------
    # Stage 0 → Stage 1 mapping  (bandwidth → k_neighbors)
    # ------------------------------------------------------------------

    def gwen_config(self, coords: np.ndarray) -> "GWENConfig":
        """Derive a :class:`~sparc.models.gwen_config.GWENConfig` from this report.

        Parameters
        ----------
        coords:
            ``(n, 2)`` array of projected spatial coordinates in metres.
            Used to estimate spatial density for the bandwidth→k mapping.

        Returns
        -------
        GWENConfig
            A validated configuration ready to pass to
            :meth:`~sparc.models.gwen.GWENModel.from_config`.
        """
        from sparc.models.gwen_config import GWENConfig

        k = self._k_neighbors(coords)
        return GWENConfig(k_neighbors=k)

    def _median_range(self) -> float:
        """Median effective range across all variables (metres)."""
        ranges = list(self.variable_effective_ranges.values())
        if not ranges:
            ranges = list(self.bandwidths.values())
        if not ranges:
            return 1_000.0
        return float(np.median(ranges))

    def _k_neighbors(self, coords: np.ndarray) -> int:
        """Compute k_neighbors from the median autocorrelation range."""
        n = len(coords)
        median_range = self._median_range()
        x_range = float(coords[:, 0].max() - coords[:, 0].min())
        y_range = float(coords[:, 1].max() - coords[:, 1].min())
        area = max(x_range * y_range, 1.0)
        circle_area = math.pi * median_range ** 2
        fraction = min(circle_area / area, 0.9)
        return int(max(_MIN_K, min(_MAX_K, round(n * fraction))))

    # ------------------------------------------------------------------
    # Artifact serialisation — the only path for inter-stage handoff
    # ------------------------------------------------------------------

    def to_artifact(self) -> Dict[str, Any]:
        """Serialise this report to a JSON-safe dict for the artifact store.

        Use :meth:`from_artifact` to restore.  The produced dict is the
        *only* authoritative representation that crosses the Stage 0 seam.
        """
        return {
            "bandwidths": dict(self.bandwidths),
            "block_size": float(self.block_size),
            "block_size_source": str(self.block_size_source),
            "stationarity_warnings": list(self.stationarity_warnings),
            "variable_effective_ranges": dict(self.variable_effective_ranges),
            "cross_range_uplift": self.cross_range_uplift,
            "dataset_profile": self.dataset_profile,
            "degradations": [asdict(d) for d in self.degradations],
            "sampling_config": self.sampling_config,
            "matern_fits": self.matern_fits,
            "anisotropy_fits": self.anisotropy_fits,
            "effective_range_matrix": self.effective_range_matrix,
        }

    @classmethod
    def from_artifact(cls, payload: Dict[str, Any]) -> "CorrelogramReport":
        """Restore a :class:`CorrelogramReport` from an artifact dict.

        Parameters
        ----------
        payload:
            Dict as produced by :meth:`to_artifact`.

        Raises
        ------
        ValueError
            If any required key is absent from *payload*.  This is deliberate
            — silent ``None`` fallbacks hide field renames and data loss.
        """
        for key in _REQUIRED_ARTIFACT_KEYS:
            if key not in payload:
                raise ValueError(
                    f"CorrelogramReport.from_artifact: required key '{key}' is missing "
                    f"from payload.  Available keys: {sorted(payload.keys())}"
                )

        raw_degradations = payload.get("degradations") or []
        degradations = [
            Degradation(
                phase=d["phase"],
                fallback_used=d["fallback_used"],
                reason=d["reason"],
            )
            for d in raw_degradations
        ]

        return cls(
            bandwidths=dict(payload["bandwidths"]),
            block_size=float(payload["block_size"]),
            block_size_source=str(payload["block_size_source"]),
            stationarity_warnings=list(payload["stationarity_warnings"]),
            variable_effective_ranges=dict(payload["variable_effective_ranges"]),
            cross_range_uplift=payload.get("cross_range_uplift"),
            dataset_profile=payload.get("dataset_profile"),
            degradations=degradations,
            sampling_config=payload.get("sampling_config"),
            matern_fits=payload.get("matern_fits"),
            anisotropy_fits=payload.get("anisotropy_fits"),
            effective_range_matrix=payload.get("effective_range_matrix"),
        )
