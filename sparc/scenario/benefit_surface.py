"""Per-cell benefit and cost surface for budget allocation.

This module owns the translation from scenario simulation outputs to the
inputs that :func:`sparc.scenario.budget.optimize` expects.  Two call sites
previously duplicated this logic:

  * :func:`sparc.run.scenarios.run_budget_optimization`  (pipeline / CLI path)
  * ``sparc.server.app:optimize_scenario``               (HTTP path)

Both paths share the same conceptual operation:
  1. Extract per-cell delta from scenario results.
  2. Flip sign so that a *beneficial* intervention (e.g. cooling) produces a
     *positive* benefit (the optimizer maximises benefit).
  3. Assemble a per-cell cost vector from unit costs.
  4. Optionally modulate benefits by an equity score.

:class:`BenefitSurface` names that operation and tests it in isolation,
independent of the simulator or the HTTP layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BenefitSurface:
    """Per-cell benefit and cost surface for a single treatment.

    Construct via :meth:`from_long_df` (pipeline path) or
    :meth:`from_results_gdf` (server path).  Both produce the same shape:
    parallel ``benefits`` and ``costs`` arrays aligned to the same set of
    ``cell_ids``.

    Parameters
    ----------
    cell_ids :
        Integer cell identifiers, shape ``(n,)``.
    benefits :
        Non-negative per-cell benefit values, shape ``(n,)``.
        Semantics: higher = better intervention outcome at that cell.
    costs :
        Per-cell cost (e.g. dollar cost for full treatment), shape ``(n,)``.
    """

    cell_ids: np.ndarray
    benefits: np.ndarray
    costs: np.ndarray

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_long_df(
        cls,
        df: pd.DataFrame,
        *,
        cell_ids: "list[int] | None" = None,
        unit_cost: float = 1.0,
    ) -> "BenefitSurface":
        """Build from a long-format results slice for *one* treatment variable.

        The DataFrame must contain the columns ``cell_id``, ``increment``,
        and ``delta_mean``.  For each unique ``cell_id``, the row with the
        largest absolute ``increment`` is selected (the "fully deployed"
        surface).

        Sign convention: ``benefit = |delta_mean|``.  The sign of ``delta_mean``
        is irrelevant here — the optimizer works with magnitudes only; the
        *direction* of the desired intervention is already encoded by which
        scenarios the caller chose to optimise.

        Parameters
        ----------
        df :
            Slice of ``results_long_df`` for a single treatment variable.
        cell_ids :
            Explicit cell id list.  When ``None``, inferred from ``df["cell_id"]``
            after increment filtering.
        unit_cost :
            Dollar cost per unit treatment applied to every cell uniformly.
        """
        work = df.copy()
        work["_abs_inc"] = work["increment"].astype(float).abs()

        if work["_abs_inc"].max() == 0.0:
            # No meaningful increment — return a zero surface.
            ids = np.asarray(df["cell_id"].astype(int).to_numpy()) if cell_ids is None else np.asarray(cell_ids, dtype=int)
            z = np.zeros(len(ids), dtype=float)
            return cls(cell_ids=ids, benefits=z, costs=np.full(len(ids), float(unit_cost)))

        max_inc = work["_abs_inc"].max()
        biggest = work[work["_abs_inc"] == max_inc].copy()

        ids = biggest["cell_id"].astype(int).to_numpy()
        benefits = np.abs(biggest["delta_mean"].astype(float).to_numpy())
        costs = np.full(len(ids), float(unit_cost), dtype=float)

        if cell_ids is not None:
            # Caller supplied explicit ordering — honour it.
            ids = np.asarray(cell_ids, dtype=int)
            # Build a lookup from cell_id → benefit
            _lookup = dict(zip(biggest["cell_id"].astype(int).tolist(), benefits.tolist()))
            benefits = np.array([_lookup.get(c, 0.0) for c in ids], dtype=float)
            costs = np.full(len(ids), float(unit_cost), dtype=float)

        return cls(cell_ids=ids, benefits=benefits, costs=costs)

    @classmethod
    def from_results_gdf(
        cls,
        gdf: pd.DataFrame,
        *,
        delta_col: str,
        unit_cost: float = 1.0,
    ) -> "BenefitSurface":
        """Build from a wide-format results GeoDataFrame for one delta column.

        Parameters
        ----------
        gdf :
            Wide-format results DataFrame (one row per cell).
        delta_col :
            Name of the column holding the per-cell delta (e.g. a
            ``total_Pct_Canopy_plus_25p`` column).
        unit_cost :
            Dollar cost per unit treatment applied to every cell uniformly.
        """
        n = len(gdf)
        raw = np.asarray(gdf[delta_col].to_numpy(), dtype=float)
        raw = np.nan_to_num(raw, nan=0.0)
        benefits = np.abs(raw)
        costs = np.full(n, float(unit_cost), dtype=float)
        ids = np.arange(n, dtype=int)
        return cls(cell_ids=ids, benefits=benefits, costs=costs)

    # ------------------------------------------------------------------
    # Equity modulation
    # ------------------------------------------------------------------

    def with_equity(
        self,
        equity_scores: "np.ndarray",
        equity_focus: float,
    ) -> "BenefitSurface":
        """Return a copy with benefits modulated by equity scores.

        Modulation formula::

            benefit_eff_i = benefit_i × [(1 − α) + α × equity_score_i]

        Parameters
        ----------
        equity_scores :
            Per-cell equity score in ``[0, 1]``, shape ``(n,)``.
        equity_focus :
            Blending weight α ∈ ``[0, 1]``.
            ``0`` ⇒ pure efficiency (no change); ``1`` ⇒ equity-first.
        """
        alpha = float(np.clip(equity_focus, 0.0, 1.0))
        if alpha == 0.0:
            return BenefitSurface(
                cell_ids=self.cell_ids.copy(),
                benefits=self.benefits.copy(),
                costs=self.costs.copy(),
            )
        eq = np.asarray(equity_scores, dtype=float)
        eq = np.clip(eq, 0.0, 1.0)
        modulated = self.benefits * ((1.0 - alpha) + alpha * eq)
        return BenefitSurface(
            cell_ids=self.cell_ids.copy(),
            benefits=modulated,
            costs=self.costs.copy(),
        )

    # ------------------------------------------------------------------
    # Budget optimizer interface
    # ------------------------------------------------------------------

    def to_optimize_args(self) -> dict:
        """Return keyword arguments suitable for :func:`sparc.scenario.budget.optimize`.

        Returns
        -------
        dict
            ``{"benefits": list[float], "costs": list[float]}``
        """
        return {
            "benefits": self.benefits.tolist(),
            "costs": self.costs.tolist(),
        }


__all__ = ["BenefitSurface"]
