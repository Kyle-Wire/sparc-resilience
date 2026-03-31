"""
Causal Evaluation Framework
============================

Provides diagnostic tools to assess the quality of heterogeneous
treatment-effect estimates produced by Stage 3:

1. **Cumulative effect curves** – sort units by predicted CATE and
   plot cumulative Y vs. fraction treated to see if the ordering
   is informative (analogous to a QINI / uplift curve).

2. **RATE scorer** – uses ``econml.score.RScorer`` (when available) to
   score CATE models on held-out folds.

3. **Calibration check** – bins predicted CATE and compares with
   within-bin ATE estimated via OLS to verify calibration.

Called by Stage 3 (``run.causal_validation``) after all ATE/CATE
estimation is complete.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Cumulative Effect Curves (QINI-like)
# ---------------------------------------------------------------------------

def cumulative_effect_curve(
    cate: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    n_bins: int = 20,
) -> Dict[str, Any]:
    """
    Build a cumulative effect curve by sorting units from highest to
    lowest predicted CATE, then computing the running-mean outcome
    difference between treated and control within each progressive slice.

    Parameters
    ----------
    cate : array-like, shape (n,)
        Predicted individual treatment effects.
    Y : array-like, shape (n,)
        Observed outcome.
    T : array-like, shape (n,)
        Continuous treatment (will be binarised at median).
    n_bins : int
        Number of quantile bins.

    Returns
    -------
    dict with keys ``fractions``, ``cum_effects``, ``random_baseline``,
    ``area_ratio`` (> 1 means the CATE ordering is informative).
    """
    cate = np.asarray(cate, dtype=np.float64).ravel()
    Y = np.asarray(Y, dtype=np.float64).ravel()
    T = np.asarray(T, dtype=np.float64).ravel()

    # Binarise at median for curve construction
    T_bin = (T > np.median(T)).astype(int)

    order = np.argsort(-cate)  # highest CATE first
    Y_sorted = Y[order]
    T_sorted = T_bin[order]

    fractions = np.linspace(1.0 / n_bins, 1.0, n_bins)
    cum_effects = []

    n = len(Y)
    for frac in fractions:
        k = max(int(frac * n), 1)
        Y_k = Y_sorted[:k]
        T_k = T_sorted[:k]
        treated = T_k == 1
        control = T_k == 0
        if treated.sum() > 0 and control.sum() > 0:
            eff = float(Y_k[treated].mean() - Y_k[control].mean())
        else:
            eff = 0.0
        cum_effects.append(eff)

    cum_effects = np.array(cum_effects)

    # Random baseline: overall ATE
    treated_all = T_bin == 1
    control_all = T_bin == 0
    if treated_all.sum() > 0 and control_all.sum() > 0:
        baseline = float(Y[treated_all].mean() - Y[control_all].mean())
    else:
        baseline = 0.0

    random_line = np.full_like(cum_effects, baseline)

    # Area ratio (like AUUC / random AUUC)
    area_curve = float(np.trapz(cum_effects, fractions))
    area_random = float(np.trapz(random_line, fractions))
    area_ratio = area_curve / area_random if abs(area_random) > 1e-12 else 1.0

    return {
        'fractions': fractions.tolist(),
        'cum_effects': cum_effects.tolist(),
        'random_baseline': baseline,
        'area_ratio': float(area_ratio),
    }


# ---------------------------------------------------------------------------
# 2. RATE Scorer (econml.score.RScorer wrapper)
# ---------------------------------------------------------------------------

def rate_score(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    cate_model: Any,
    cv: int = 3,
) -> Optional[float]:
    """
    Score a CATE model using EconML's RScorer (Nie & Wager 2021).

    Returns the R-loss score (higher = better CATE fit).  Returns None
    if ``econml.score`` is unavailable.
    """
    try:
        from econml.score import RScorer
        from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    except ImportError:
        return None

    scorer = RScorer(
        model_y=HGB(max_iter=100, max_depth=3, random_state=42),
        model_t=HGB(max_iter=100, max_depth=3, random_state=42),
        discrete_treatment=False,
        cv=cv,
        random_state=42,
    )
    scorer.fit(Y, T.reshape(-1, 1), X=X)
    score = scorer.score(cate_model)
    return float(score)


# ---------------------------------------------------------------------------
# 3. CATE Calibration Check
# ---------------------------------------------------------------------------

def cate_calibration(
    cate: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    X_confounders: np.ndarray,
    n_bins: int = 5,
) -> Dict[str, Any]:
    """
    Check whether predicted CATE is well-calibrated by binning units
    into quantiles of predicted CATE and estimating within-bin ATE
    via OLS on the confounders.

    A well-calibrated model should show monotonically increasing
    within-bin ATE from the lowest to highest CATE bin.

    Returns
    -------
    dict with ``bin_cate_mean``, ``bin_ate``, ``is_monotone``,
    ``rank_correlation``.
    """
    cate = np.asarray(cate, dtype=np.float64).ravel()
    Y = np.asarray(Y, dtype=np.float64).ravel()
    T = np.asarray(T, dtype=np.float64).ravel()
    X_confounders = np.atleast_2d(np.asarray(X_confounders, dtype=np.float64))

    # Assign quantile bins
    bin_edges = np.percentile(cate, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-6  # ensure max value included
    bin_idx = np.digitize(cate, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    bin_cate_mean = []
    bin_ate = []

    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() < 10:
            bin_cate_mean.append(float(np.nan))
            bin_ate.append(float(np.nan))
            continue

        bin_cate_mean.append(float(cate[mask].mean()))

        # Within-bin ATE via partial regression (T → Y | X)
        Y_b = Y[mask]
        T_b = T[mask]
        X_b = X_confounders[mask]

        # Residualise Y and T on X, then regress residual Y on residual T
        design = np.column_stack([np.ones(X_b.shape[0]), X_b])
        try:
            beta_y, _, _, _ = np.linalg.lstsq(design, Y_b, rcond=None)
            beta_t, _, _, _ = np.linalg.lstsq(design, T_b, rcond=None)
            Y_resid = Y_b - design @ beta_y
            T_resid = T_b - design @ beta_t
            denom = float((T_resid ** 2).sum())
            if abs(denom) > 1e-12:
                ate_b = float((T_resid * Y_resid).sum() / denom)
            else:
                ate_b = float(np.nan)
        except Exception:
            ate_b = float(np.nan)

        bin_ate.append(ate_b)

    bin_cate_mean = np.array(bin_cate_mean)
    bin_ate = np.array(bin_ate)

    # Check monotonicity (within tolerance)
    valid = ~np.isnan(bin_ate)
    if valid.sum() >= 3:
        from scipy.stats import spearmanr
        result = spearmanr(bin_cate_mean[valid], bin_ate[valid])
        rho = float(result.statistic)  # type: ignore[union-attr]
        p = float(result.pvalue)       # type: ignore[union-attr]
        is_monotone = bool(rho > 0.6)
    else:
        rho: float = 0.0
        p: float = 1.0
        is_monotone = False

    return {
        'bin_cate_mean': bin_cate_mean.tolist(),
        'bin_ate': bin_ate.tolist(),
        'is_monotone': is_monotone,
        'rank_correlation': rho,
        'rank_pvalue': p,
    }


# ---------------------------------------------------------------------------
# 4. Run All Causal Diagnostics
# ---------------------------------------------------------------------------

def run_causal_diagnostics(
    validator,
    data: pd.DataFrame,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Run the full causal evaluation suite on a CausalValidator that
    has already been fitted.

    Parameters
    ----------
    validator : CausalValidator
        Must have ``ate_results``, ``structural_coeffs``, ``roles``,
        and optionally ``spatial_cate_results`` populated.
    data : pd.DataFrame
        The analysis dataset.
    output_dir : str
        Directory to write diagnostics JSON.

    Returns
    -------
    dict
        Full diagnostics payload.
    """
    target = validator.roles['outcomes'][0] if validator.roles.get('outcomes') else None
    if target is None:
        return {}

    diagnostics: Dict[str, Any] = {}

    for treatment in validator.roles.get('treatments', []):
        if treatment not in data.columns:
            continue

        diag: Dict[str, Any] = {}
        Y = data[target].values
        T = data[treatment].values

        # ---- Cumulative effect curve ----
        cate_info = validator.spatial_cate_results.get(treatment, {})
        cate_arr = cate_info.get('cate')
        if cate_arr is not None and len(cate_arr) == len(Y):
            diag['cumulative_effect_curve'] = cumulative_effect_curve(
                cate_arr, Y, T
            )
            print(f"    CumEffect({treatment}): area_ratio="
                  f"{diag['cumulative_effect_curve']['area_ratio']:.3f}")

        # ---- Calibration check ----
        if cate_arr is not None and len(cate_arr) == len(Y):
            # Confounders for calibration
            if validator.graph:
                from sparc.causal.dag_definition import compute_backdoor_set
                conf_cols = [
                    c for c in compute_backdoor_set(
                        validator.graph, treatment, target
                    )
                    if c in data.columns
                ]
            else:
                conf_cols = [
                    c for c in validator.roles.get('confounders', [])
                    if c in data.columns
                ]

            if conf_cols:
                diag['calibration'] = cate_calibration(
                    cate_arr, Y, T, data[conf_cols].values
                )
                cal = diag['calibration']
                monotone_str = "YES" if cal['is_monotone'] else "NO"
                print(f"    Calibration({treatment}): monotone={monotone_str}, "
                      f"rho={cal['rank_correlation']:.3f}")

        # ---- RATE score (expensive, only if CATE model is available) ----
        cate_model = cate_info.get('model')
        if cate_model is not None:
            if validator.graph:
                from sparc.causal.dag_definition import compute_backdoor_set
                conf_cols = [
                    c for c in compute_backdoor_set(
                        validator.graph, treatment, target
                    )
                    if c in data.columns
                ]
            else:
                conf_cols = [
                    c for c in validator.roles.get('confounders', [])
                    if c in data.columns
                ]
            if conf_cols:
                # Subsample for speed
                n_sub = min(5000, len(data))
                idx = np.random.RandomState(42).choice(
                    len(data), n_sub, replace=False
                )
                rs = rate_score(
                    Y[idx], T[idx],
                    data[conf_cols].values[idx],
                    cate_model, cv=3,
                )
                if rs is not None:
                    diag['rate_score'] = rs
                    print(f"    RScore({treatment}): {rs:.4f}")

        if diag:
            diagnostics[treatment] = diag

    # Save
    if diagnostics:
        os.makedirs(output_dir, exist_ok=True)
        diag_path = os.path.join(output_dir, 'causal_diagnostics.json')
        with open(diag_path, 'w', encoding='utf-8') as f:
            json.dump(diagnostics, f, indent=2, default=str)
        print(f"  Causal diagnostics saved: {diag_path}")

    return diagnostics
