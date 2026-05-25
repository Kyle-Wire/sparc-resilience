"""
_temporal.py — CAPA-first temporal alignment contract.

TemporalWindow records the CAPA measurement dates as canonical anchors.
All other sensors (Landsat, ERA5) search for data matching those dates
within a configurable tolerance window.

Design rationale:
  CAPA provides ground-truth air temperature at specific measurement dates.
  These dates are the prediction targets; covariate data (Landsat spectral
  state, ERA5 background) must be aligned to them — not the other way around.
  If CAPA has no dates, the entire pipeline degrades gracefully but the
  training labels are empty and model fit should not proceed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional


@dataclass
class TemporalWindow:
    """Temporal search contract anchored to CAPA measurement dates.

    Attributes
    ----------
    date_start, date_end : date
        Outer bounds of the study period.
    anchor_dates : list[date]
        CAPA measurement dates — the canonical alignment targets.
        Populated by ``from_capa_dates()``; empty means CAPA failed.
    tolerance_days : int
        Maximum acceptable offset (in days) between a requested anchor date
        and a found sensor scene.  Default 3.
    """

    date_start: date
    date_end: date
    anchor_dates: list[date] = field(default_factory=list)
    tolerance_days: int = 3

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_capa_dates(
        cls,
        capa_dates: list[date],
        *,
        date_start: date,
        date_end: date,
        tolerance_days: int = 3,
    ) -> "TemporalWindow":
        """Build a TemporalWindow anchored to CAPA measurement dates.

        Duplicates are removed; the list is sorted ascending.
        An empty ``capa_dates`` list is accepted — callers must check
        ``window.anchor_dates`` before requesting Landsat scenes.
        """
        deduped = sorted(set(capa_dates))
        return cls(
            date_start=date_start,
            date_end=date_end,
            anchor_dates=deduped,
            tolerance_days=tolerance_days,
        )

    @classmethod
    def from_range(
        cls,
        date_start: date,
        date_end: date,
        *,
        tolerance_days: int = 3,
    ) -> "TemporalWindow":
        """Build a TemporalWindow from a bare date range with no anchor dates.

        Use this when you have a study period but CAPA has not yet run (or is
        not applicable).  All adapters accept a :class:`TemporalWindow`; this
        factory lets callers avoid the legacy ``(date_start, date_end)`` pair.

        Parameters
        ----------
        date_start, date_end : date
            Outer bounds of the study period.
        tolerance_days : int
            Maximum offset (days) between anchor and scene.  Default 3.
        """
        return cls(
            date_start=date_start,
            date_end=date_end,
            anchor_dates=[],
            tolerance_days=tolerance_days,
        )

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------

    def search_range(self, anchor: date) -> tuple[date, date]:
        """Return the STAC search window for a single CAPA anchor date.

        The range is ``anchor ± tolerance_days``, clipped to
        ``[date_start, date_end]``.

        Parameters
        ----------
        anchor : date
            One CAPA measurement date.

        Returns
        -------
        (start, end)
            Inclusive date range to use when querying a STAC endpoint.
        """
        delta = timedelta(days=self.tolerance_days)
        raw_start = anchor - delta
        raw_end = anchor + delta
        return (
            max(raw_start, self.date_start),
            min(raw_end, self.date_end),
        )

    def closest_match(
        self,
        available_scenes: list[date],
        anchor: date,
    ) -> tuple[Optional[date], int]:
        """Find the closest available scene date to *anchor*.

        Issues a ``UserWarning`` when:
        - The closest scene is not an exact match (offset > 0).
        - The closest scene is beyond the configured tolerance
          (offset > tolerance_days) — a stronger warning with "beyond
          tolerance" language.

        Parameters
        ----------
        available_scenes : list[date]
            Scene dates returned by a STAC search.
        anchor : date
            The CAPA measurement date being matched.

        Returns
        -------
        (matched_date | None, offset_days)
            ``None`` when ``available_scenes`` is empty.
            ``offset_days`` is the absolute day difference (0 = exact).
        """
        if not available_scenes:
            return None, 0

        closest = min(available_scenes, key=lambda d: abs((d - anchor).days))
        offset = abs((closest - anchor).days)

        if offset > self.tolerance_days:
            warnings.warn(
                f"Landsat scene for CAPA anchor {anchor} is {offset} days away "
                f"(beyond tolerance of {self.tolerance_days} days). "
                "Using expanded window — spectral features may not match label date.",
                UserWarning,
                stacklevel=2,
            )
        elif offset > 0:
            warnings.warn(
                f"Landsat scene for CAPA anchor {anchor} has offset of {offset} day(s). "
                "No exact date match found within tolerance window.",
                UserWarning,
                stacklevel=2,
            )

        return closest, offset
