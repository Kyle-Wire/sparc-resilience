"""
maup_sensitivity — Modifiable Areal Unit Problem (MAUP) diagnostic.

The MAUP is the phenomenon whereby causal estimates change — sometimes flip
sign — purely because of the spatial unit chosen for aggregation.  SPARC's
continuous spatial modeling is resistant to MAUP by design, but this module
**formalises** that resistance as a diagnostic that can be attached to any
causal result.

Core function
-------------
``maup_sensitivity_analysis(data, treatment, outcome, confounders,
    coord_cols, cell_sizes, ...)``

    Reruns a lightweight DML ATE estimate at each supplied ``cell_size``
    (after aggregating the point data into a regular grid at that resolution)
    and reports scale-specific ATEs alongside a scalar **MAUP robustness
    score** in [0, 1].  A score of 1.0 is perfectly scale-invariant; < 0.8
    indicates that the causal claim should be interpreted cautiously.

MAUP robustness score definition
---------------------------------
.. math::

    \\text{MAUP-RS} = \\max\\!\\left(0,\\; 1 - \\frac{\\sigma_{\\text{ATE}}}{
    \\overline{|\\text{ATE}|} + \\varepsilon}\\right)

where :math:`\\sigma_{\\text{ATE}}` is the standard deviation of the ATEs
across scales and :math:`\\overline{|\\text{ATE}|}` is the mean absolute ATE.
Dividing by the signal magnitude makes the score relative (a CV-type metric)
and prevents degenerate inflation when the true ATE is near zero.

Typical usage::

    from sparc.causal.maup_sensitivity import maup_sensitivity_analysis

    results = maup_sensitivity_analysis(
        data=df,
        treatment="Pct_Impervious",
        outcome="AAT_z",
        confounders=["Elevation_m", "NDVI"],
        coord_cols=["POINT_X", "POINT_Y"],
        cell_sizes=[0.0, 100.0, 500.0, 2000.0],
    )
    print(results.maup_robustness_score)  # e.g. 0.94
    for s in results.scale_results:
        print(s.cell_size, s.ate, s.n_units)
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScaleResult:
    """ATE estimate at a single spatial scale."""
    cell_size: float
    """Grid cell size in the same units as ``coord_cols``.
    0.0 means the raw point-level data (no aggregation)."""
    n_units: int
    """Number of spatial units (points or grid cells) at this scale."""
    ate: float
    """Average Treatment Effect estimated by cross-fit DML at this scale."""
    ate_se: float
    """Approximate standard error of the ATE."""
    ate_ci_lower: float
    """95 % confidence interval lower bound."""
    ate_ci_upper: float
    """95 % confidence interval upper bound."""
    description: str = ""


@dataclass
class MAUPSensitivityResult:
    """Full multi-scale MAUP sensitivity report."""

    treatment: str
    outcome: str
    scale_results: List[ScaleResult] = field(default_factory=list)
    maup_robustness_score: float = float("nan")
    """Scalar in [0, 1].  1.0 = scale-invariant; < 0.8 = interpret cautiously."""
    sign_stable: bool = True
    """True iff the ATE estimate has the same sign across all valid scales."""
    interpretation: str = ""

    def summary_table(self) -> pd.DataFrame:
        """Return a tidy DataFrame of scale-by-scale results."""
        rows = []
        for s in self.scale_results:
            rows.append({
                "cell_size": s.cell_size,
                "n_units": s.n_units,
                "ate": round(s.ate, 6),
                "ate_se": round(s.ate_se, 6),
                "ci_lower": round(s.ate_ci_lower, 6),
                "ci_upper": round(s.ate_ci_upper, 6),
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Grid aggregation helpers
# ---------------------------------------------------------------------------

def _grid_aggregate(
    data: pd.DataFrame,
    coord_cols: list[str],
    cell_size: float,
    columns: list[str],
    agg: str = "mean",
) -> pd.DataFrame:
    """Aggregate ``data[columns]`` into a regular grid with spacing ``cell_size``.

    Each observation is assigned to a grid cell based on its coordinates.
    All numeric columns are aggregated by ``agg`` (default: mean) within
    each cell.  Cells with a single observation are retained as-is.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with ``coord_cols``.
    coord_cols : list[str]
        Two column names for (x, y) coordinates.
    cell_size : float
        Grid spacing in coordinate units.
    columns : list[str]
        Data columns to aggregate (must be numeric).
    agg : str
        Aggregation method passed to ``groupby.agg`` (default ``"mean"``).

    Returns
    -------
    pd.DataFrame
        Aggregated data with one row per grid cell (only cells with ≥ 1
        observation).
    """
    cx, cy = coord_cols
    # Bin coordinates into grid cell indices
    x_min, y_min = data[cx].min(), data[cy].min()
    gx = np.floor((data[cx].values - x_min) / cell_size).astype(int)
    gy = np.floor((data[cy].values - y_min) / cell_size).astype(int)
    cell_id = gx * 1_000_000 + gy  # unique cell key (sufficient for any reasonable grid)

    agg_data = data[columns].copy()
    agg_data["_cell"] = cell_id
    result = agg_data.groupby("_cell")[columns].agg(agg).reset_index(drop=True)
    return result.dropna(how="all").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Lightweight DML ATE estimator (no EconML dependency)
# ---------------------------------------------------------------------------

def _dml_ate(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[float, float]:
    """Cross-fit partial-linear DML ATE estimator.

    Uses HGB (HistGradientBoosting) nuisance models for outcome and
    treatment propensity.  Returns ``(ate, se)`` where ``se`` is estimated
    via the cross-fit influence function variance.

    Parameters
    ----------
    data : pd.DataFrame
    treatment, outcome : str
    confounders : list[str]
    n_splits : int
        Number of cross-fitting folds.
    random_state : int

    Returns
    -------
    (ate, se) : (float, float)
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold

    X = data[confounders].values.astype(float)
    T = data[treatment].values.astype(float)
    Y = data[outcome].values.astype(float)
    n = len(Y)

    # Fill any NaN in X with column means (robustness for aggregated cells)
    col_means = np.nanmean(X, axis=0)
    for k in range(X.shape[1]):
        mask = np.isnan(X[:, k])
        if mask.any():
            X[mask, k] = col_means[k]

    T_res = np.zeros(n)
    Y_res = np.zeros(n)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        T_tr, T_te = T[train_idx], T[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]

        # Outcome model: E[Y | X]
        m_Y = HistGradientBoostingRegressor(random_state=random_state)
        m_Y.fit(X_tr, Y_tr)
        Y_hat = m_Y.predict(X_te)

        # Treatment model: E[T | X]
        m_T = HistGradientBoostingRegressor(random_state=random_state)
        m_T.fit(X_tr, T_tr)
        T_hat = m_T.predict(X_te)

        T_res[test_idx] = T_te - T_hat
        Y_res[test_idx] = Y_te - Y_hat

    # Partially-linear moment: ATE = E[Y_res * T_res] / E[T_res²]
    denom = np.mean(T_res ** 2)
    if abs(denom) < 1e-12:
        return float("nan"), float("nan")

    psi = (Y_res - (np.mean(Y_res * T_res) / denom) * T_res) * T_res / denom
    ate = float(np.mean(Y_res * T_res) / denom)
    se = float(np.std(psi) / math.sqrt(n))
    return ate, se


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def maup_sensitivity_analysis(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
    coord_cols: list[str],
    cell_sizes: list[float] | None = None,
    n_splits: int = 5,
    min_units: int = 20,
    random_state: int = 42,
    z_ci: float = 1.96,
) -> MAUPSensitivityResult:
    """Run MAUP sensitivity analysis across multiple spatial aggregation scales.

    For each ``cell_size`` in ``cell_sizes`` the point data is aggregated
    onto a regular grid (``cell_size == 0`` uses raw point-level data), a
    cross-fit DML ATE is estimated, and the result is logged.  A scalar
    MAUP robustness score is computed from the coefficient of variation of
    ATE estimates across scales.

    Parameters
    ----------
    data : pd.DataFrame
        Point-level observations.  Must contain ``treatment``, ``outcome``,
        all ``confounders``, and both ``coord_cols``.
    treatment : str
        Treatment column name.
    outcome : str
        Outcome column name.
    confounders : list[str]
        Covariate column names used as nuisance controls.
    coord_cols : list[str]
        ``[x_col, y_col]`` — spatial coordinate column names.
    cell_sizes : list[float] or None
        Grid spacings to test.  ``0.0`` = raw point data.  When ``None``,
        defaults to ``[0.0, 100.0, 500.0, 2000.0]``.
    n_splits : int
        Cross-fitting folds for the DML nuisance models.
    min_units : int
        Skip a scale if aggregation yields fewer than ``min_units`` cells
        (too few observations for reliable estimation).
    random_state : int
    z_ci : float
        Normal quantile for confidence intervals (default 1.96 = 95 %).

    Returns
    -------
    MAUPSensitivityResult
    """
    if cell_sizes is None:
        # Derive sensible defaults from coordinate range
        cx, cy = coord_cols
        x_range = float(data[cx].max() - data[cx].min())
        default_scales = [0.0]
        for frac in [0.01, 0.05, 0.15]:
            cs = round(x_range * frac, 1)
            if cs > 0:
                default_scales.append(cs)
        cell_sizes = sorted(set(default_scales))

    all_cols = [treatment, outcome] + confounders
    missing = [c for c in all_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    scale_results: list[ScaleResult] = []
    valid_ates: list[float] = []

    for cs in cell_sizes:
        if cs == 0.0:
            agg_df = data[all_cols].dropna(subset=[treatment, outcome]).reset_index(drop=True)
            desc = "point-level (no aggregation)"
        else:
            agg_df = _grid_aggregate(
                data, coord_cols=coord_cols, cell_size=cs,
                columns=all_cols, agg="mean",
            )
            agg_df = agg_df.dropna(subset=[treatment, outcome]).reset_index(drop=True)
            desc = f"grid cell {cs}"

        n_units = len(agg_df)
        if n_units < min_units:
            logger.info(
                "MAUP: skipping cell_size=%.1f — only %d units (< min_units=%d)",
                cs, n_units, min_units,
            )
            scale_results.append(ScaleResult(
                cell_size=cs, n_units=n_units,
                ate=float("nan"), ate_se=float("nan"),
                ate_ci_lower=float("nan"), ate_ci_upper=float("nan"),
                description=f"{desc} — skipped (too few units)",
            ))
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ate, se = _dml_ate(
                agg_df, treatment=treatment, outcome=outcome,
                confounders=confounders, n_splits=min(n_splits, max(2, n_units // 5)),
                random_state=random_state,
            )

        ci_lo = ate - z_ci * se if math.isfinite(se) else float("nan")
        ci_hi = ate + z_ci * se if math.isfinite(se) else float("nan")

        scale_results.append(ScaleResult(
            cell_size=cs, n_units=n_units,
            ate=ate, ate_se=se,
            ate_ci_lower=ci_lo, ate_ci_upper=ci_hi,
            description=desc,
        ))
        if math.isfinite(ate):
            valid_ates.append(ate)
        logger.info(
            "MAUP: cell_size=%.1f  n=%d  ATE=%.4f  SE=%.4f",
            cs, n_units, ate, se,
        )

    # ---- Robustness score ----
    if len(valid_ates) < 2:
        maup_rs = float("nan")
        sign_stable = True
        interp = "Insufficient valid scales to compute MAUP robustness score."
    else:
        ate_arr = np.array(valid_ates)
        sigma = float(np.std(ate_arr, ddof=1))
        mean_abs = float(np.mean(np.abs(ate_arr)))
        maup_rs = float(max(0.0, 1.0 - sigma / (mean_abs + 1e-12)))
        sign_stable = bool(np.all(ate_arr > 0) or np.all(ate_arr < 0))
        if maup_rs >= 0.9:
            interp = "Excellent scale-invariance — causal estimate is MAUP-robust."
        elif maup_rs >= 0.8:
            interp = "Good scale-invariance — minor MAUP sensitivity detected."
        elif maup_rs >= 0.6:
            interp = (
                "Moderate MAUP sensitivity — interpret causal estimate cautiously "
                "across spatial scales."
            )
        else:
            interp = (
                "High MAUP sensitivity — causal estimate varies substantially with "
                "spatial aggregation. Results may be an artifact of scale choice."
            )
        if not sign_stable:
            interp += " WARNING: ATE sign flips across scales."

    logger.info(
        "MAUP robustness score for %s→%s: %.3f  sign_stable=%s",
        treatment, outcome, maup_rs, sign_stable,
    )

    return MAUPSensitivityResult(
        treatment=treatment,
        outcome=outcome,
        scale_results=scale_results,
        maup_robustness_score=maup_rs,
        sign_stable=sign_stable,
        interpretation=interp,
    )
