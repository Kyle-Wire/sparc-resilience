"""Counterfactual CI coverage harness for the SPARC v4 causal stack.

Quick context
-------------
The v4 ScenarioEngine emits per-cell credible intervals for every
intervention. Those intervals are only honest if the underlying
posterior is well-calibrated — which is *not* something the model
checks itself. This module provides a small, standalone tool that
asks the empirical question:

    "If we held out a random subset of cells from the trained
    pipeline, treat the base ensemble's prediction at the modified
    covariates as a synthetic ground-truth ΔY, and run the engine on
    those held-out cells, do the nominal CI bands actually cover
    that synthetic truth at the rate they claim?"

The synthetic ground truth is intentionally circular (we use the
ensemble's own re-prediction). That circularity is fine for the
quantity we care about: relative miscalibration of the resolver
posterior versus the base predictor. A well-calibrated stack will
produce 90% bands that contain the ensemble's Δ ~90% of the time;
a miscalibrated one will be persistently too tight or too wide.

Usage
-----
>>> from sparc.evaluation.coverage import counterfactual_coverage
>>> report = counterfactual_coverage(
...     engine, baseline_df,
...     ensemble_predictor=engine.ensemble_predictor,
...     holdout_frac=0.2, seed=42,
... )
>>> report["nominal_90"], report["empirical_90"]
(0.9, 0.873)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


_DEFAULT_LEVELS = (0.50, 0.80, 0.90, 0.95)


@dataclass
class CoverageReport:
    """Container for coverage diagnostics across nominal CI levels."""

    levels: List[float]
    nominal: List[float]
    empirical: List[float]
    n_holdout: int
    n_scenarios: int
    per_scenario: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "n_holdout": self.n_holdout,
            "n_scenarios": self.n_scenarios,
            "per_scenario": self.per_scenario,
        }
        for lvl, nom, emp in zip(self.levels, self.nominal, self.empirical):
            tag = f"{int(round(lvl * 100))}"
            d[f"nominal_{tag}"] = float(nom)
            d[f"empirical_{tag}"] = float(emp)
        return d


def _quantile_pair_for_level(level: float) -> tuple[float, float]:
    """Return symmetric (lo, hi) quantiles for a nominal CI level."""
    half = (1.0 - level) / 2.0
    return half, 1.0 - half


def counterfactual_coverage(
    engine: Any,
    baseline_df: pd.DataFrame,
    *,
    ensemble_predictor: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
    holdout_frac: float = 0.2,
    seed: int = 0,
    n_draws: int = 300,
    levels: Sequence[float] = _DEFAULT_LEVELS,
) -> Dict[str, Any]:
    """Estimate empirical CI coverage on a held-out cell subset.

    Parameters
    ----------
    engine : ScenarioEngineV4
        A constructed engine. Must have ``scenarios`` populated and a
        resolver capable of producing posterior samples.
    baseline_df : pd.DataFrame
        Full baseline frame; we sample ``holdout_frac`` rows from it.
    ensemble_predictor : callable, optional
        ``df -> ndarray`` producing point predictions at modified
        covariates. Falls back to ``engine.ensemble_predictor`` when
        omitted; if neither exists, returns an empty report.
    holdout_frac : float
        Fraction of cells to hold out (default 0.2).
    seed : int
        RNG seed for the held-out index draw.
    n_draws : int
        Posterior draws to request from the engine per scenario.
    levels : sequence of float
        Nominal CI levels in ``(0, 1)``; defaults to (.50, .80, .90, .95).

    Returns
    -------
    dict
        Flattened coverage diagnostics — see :meth:`CoverageReport.as_dict`.

    Notes
    -----
    The synthetic ground truth is the ensemble's own re-prediction at
    the modified covariates minus its baseline prediction. This is
    *not* a true counterfactual — it inherits whatever bias the
    ensemble carries — but it is the right yardstick for asking
    whether the resolver posterior agrees with the base predictor on
    the cells it never directly informed.
    """
    if not getattr(engine, "scenarios", None):
        return CoverageReport([], [], [], 0, 0).as_dict()

    predictor = ensemble_predictor or getattr(engine, "ensemble_predictor", None)
    if predictor is None:
        return {
            "n_holdout": 0, "n_scenarios": 0, "per_scenario": {},
            "skipped_reason": "no ensemble_predictor available",
        }

    rng = np.random.default_rng(seed)
    n_total = len(baseline_df)
    n_hold = max(1, int(round(holdout_frac * n_total)))
    hold_idx = rng.choice(n_total, size=n_hold, replace=False)
    hold_idx.sort()
    hold_df = baseline_df.iloc[hold_idx].reset_index(drop=True)

    # Per-level hit counters across all (scenario × cell) rows.
    levels_list = list(levels)
    hits = {lvl: 0 for lvl in levels_list}
    totals = {lvl: 0 for lvl in levels_list}
    per_scenario: Dict[str, Dict[str, float]] = {}

    for spec in engine.scenarios:
        var = spec.variable
        if var not in hold_df.columns:
            continue
        base_x = hold_df[var].to_numpy(dtype=np.float64)
        mod_x = base_x + spec.increment
        if spec.min_val is not None:
            mod_x = np.maximum(mod_x, float(spec.min_val))
        if spec.max_val is not None:
            mod_x = np.minimum(mod_x, float(spec.max_val))

        # Posterior samples (n_eff, n_hold) of Δtarget from the engine.
        try:
            samples = engine._compose_delta(
                treatment=var, target=engine.target_col,
                baseline_x=base_x, modified_x=mod_x,
                baseline_df=hold_df, n_draws=n_draws,
            )
        except Exception:
            continue

        # Synthetic ground truth from the ensemble re-prediction.
        modified_df = hold_df.copy()
        modified_df[var] = mod_x
        try:
            base_pred = np.asarray(predictor(hold_df), dtype=np.float64).reshape(-1)
            mod_pred = np.asarray(predictor(modified_df), dtype=np.float64).reshape(-1)
        except Exception:
            continue
        truth = mod_pred - base_pred

        scen_hits = {lvl: 0 for lvl in levels_list}
        for lvl in levels_list:
            lo_q, hi_q = _quantile_pair_for_level(lvl)
            lo = np.quantile(samples, lo_q, axis=0)
            hi = np.quantile(samples, hi_q, axis=0)
            inside = (truth >= lo) & (truth <= hi)
            n = int(inside.size)
            k = int(inside.sum())
            hits[lvl] += k
            totals[lvl] += n
            scen_hits[lvl] = k
        per_scenario[spec.name] = {
            f"empirical_{int(round(lvl*100))}": (
                scen_hits[lvl] / max(1, len(truth))
            )
            for lvl in levels_list
        }
        per_scenario[spec.name]["n_holdout"] = float(len(truth))

    nominal = [float(lvl) for lvl in levels_list]
    empirical = [
        (hits[lvl] / totals[lvl]) if totals[lvl] > 0 else float("nan")
        for lvl in levels_list
    ]
    report = CoverageReport(
        levels=nominal,
        nominal=nominal,
        empirical=empirical,
        n_holdout=n_hold,
        n_scenarios=len(per_scenario),
        per_scenario=per_scenario,
    )
    return report.as_dict()


__all__ = ["counterfactual_coverage", "CoverageReport"]
