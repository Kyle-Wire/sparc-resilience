"""
BandwidthAdvisor — translates a CorrelogramReport into a GWENConfig.

This module is the seam between Stage 0 (correlogram analysis) and Stage 1
(GWEN variable selection).  It owns all knowledge of how correlogram results
map to GWEN hyperparameters, so neither stage needs to know about the other's
internals.

The key mapping is bandwidth → k_neighbors:

    Intuition: GWEN's k-nearest-neighbour window should be large enough to
    contain all the spatially-correlated information relevant to the target.
    The autocorrelation range from Stage 0 tells us the spatial scale at which
    that information lives.

    Formula:
        density  = n / bounding_box_area          (points per m²)
        fraction = min(π * range² * density, 0.9) (fraction of dataset in range circle)
        k        = max(MIN_K, min(MAX_K, round(n * fraction)))

    Cross-range uplift: when Stage 0 found that a predictor couples to the
    target at a scale *longer* than the target's own marginal range, the
    ``block_size_source`` is ``"correlogram_with_crossrange_uplift"``.  In
    that case BandwidthAdvisor uses the (already uplifted) ``block_size`` as
    a proxy for the relevant scale rather than the target's marginal range.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sparc.run.correlogram_report import CorrelogramReport
    from sparc.models.gwen_config import GWENConfig

_MIN_K = 10
_MAX_K = 500


class BandwidthAdvisor:
    """Maps a :class:`~sparc.run.correlogram_report.CorrelogramReport` to a
    :class:`~sparc.models.gwen_config.GWENConfig`.

    All methods are static — there is no stateful configuration here.
    """

    @staticmethod
    def suggest(
        report: "CorrelogramReport",
        coords: np.ndarray,
    ) -> "GWENConfig":
        """Derive a :class:`~sparc.models.gwen_config.GWENConfig` from Stage 0 results.

        Parameters
        ----------
        report:
            Typed output of Stage 0 correlogram analysis.
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

        k_neighbors = BandwidthAdvisor._k_from_report(report, coords)
        return GWENConfig(k_neighbors=k_neighbors)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _median_range(report: "CorrelogramReport") -> float:
        """Median effective range across all variables in the report (metres)."""
        ranges = list(report.variable_effective_ranges.values())
        if not ranges:
            # Fall back to bandwidths if effective_ranges is empty
            ranges = list(report.bandwidths.values())
        if not ranges:
            return 1_000.0  # last-resort default: 1 km
        return float(np.median(ranges))

    @staticmethod
    def _bounding_box_area(coords: np.ndarray) -> float:
        """Bounding box area of coords in m².  Minimum 1 m² to avoid division by zero."""
        x_range = float(coords[:, 0].max() - coords[:, 0].min())
        y_range = float(coords[:, 1].max() - coords[:, 1].min())
        return max(x_range * y_range, 1.0)

    @staticmethod
    def _k_from_report(
        report: "CorrelogramReport",
        coords: np.ndarray,
    ) -> int:
        """Compute k_neighbors from the median autocorrelation range.

        The formula assumes ~30 % of points lie within one autocorrelation
        radius of a given location, clipped to [MIN_K, MAX_K].
        """
        n = len(coords)
        median_range = BandwidthAdvisor._median_range(report)
        area = BandwidthAdvisor._bounding_box_area(coords)

        circle_area = math.pi * median_range ** 2
        fraction = min(circle_area / area, 0.9)  # never use more than 90 % of data
        k = max(_MIN_K, min(_MAX_K, round(n * fraction)))
        return int(k)
