"""Unified ATE estimation stack.

Wraps IPW, GPS, covariate matching, and doubly-robust (DML) estimators
behind a single ``estimate()`` call.  Each estimator runs inside a
``try/except`` so a failure in one (e.g. missing ``econml``) never
aborts the others.

Usage
-----
::

    stack = ATEEstimatorStack(config)
    result = stack.estimate(
        treatment="Pct_Canopy",
        outcome="LST",
        confounders=["Pct_Impervious", "NDVI"],
        data=df,
    )
    print(result["ate_ipw"])   # float or None

The returned :class:`ATEResult` is a plain ``dict`` so it can be
JSON-serialised or dropped straight into
``scenario_coefficients.json``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------

class EstimatorError(Exception):
    """Base class for typed ATE estimator failures.

    Raised by individual ``_run_*`` methods when a failure can be
    classified.  ``ATEEstimatorStack.estimate()`` always catches these
    and populates ``ATEResult["failure_reasons"]``; they are never
    propagated to the caller.
    """


class DependencyMissing(EstimatorError):
    """A required optional dependency (e.g. ``econml``, ``dowhy``) is
    not installed or cannot be imported."""


class NumericalInstability(EstimatorError):
    """Computation failed due to degenerate data (near-zero variance,
    rank-deficient matrix, etc.)."""


class InvalidInput(EstimatorError):
    """A required column is absent from the data, or the data does not
    meet minimum size / type requirements for this estimator."""

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class ATEResult(dict):
    """Dict subclass with typed access to ATE estimator outputs.

    Keys
    ----
    ate_backdoor : float | None
        OLS backdoor-adjustment ATE (via DoWhy when available).
    ate_ipw : float | None
        Stabilised inverse-probability-weighted ATE.
    ate_gps : float | None
        GPS-weighted dose-response slope (continuous treatment).
    att_matching : float | None
        Average treatment effect on the treated via nearest-neighbour
        covariate matching (Abadie-Imbens).
    ate_doubly_robust : float | None
        Doubly-robust / cross-fit DML ATE (EconML LinearDML or
        sklearn fallback).
    error : str | None
        Set to a non-empty string when all estimators failed.
    """

    __required_keys__ = frozenset({
        "ate_backdoor",
        "ate_ipw",
        "ate_gps",
        "att_matching",
        "ate_doubly_robust",
        "error",
    })

    @classmethod
    def empty(cls) -> "ATEResult":
        return cls({k: None for k in cls.__required_keys__})


# ---------------------------------------------------------------------------
# Main estimator class
# ---------------------------------------------------------------------------

class ATEEstimatorStack:
    """Runs multiple ATE estimators for a single treatment → outcome pair.

    Each estimator is independent — failures are caught and logged so
    the pipeline degrades gracefully when optional dependencies
    (``dowhy``, ``econml``) are not installed.

    Parameters
    ----------
    config : dict
        Project config (same ``project.yml``-derived dict used
        throughout the SPARC pipeline).  Keys read:

        * ``causal.ipw_enabled`` (default True)
        * ``causal.gps_enabled`` (default True)
        * ``causal.matching_enabled`` (default True)
        * ``causal.doubly_robust`` (default True)
    """

    def __init__(self, config: dict):
        self._cfg = config or {}
        self._causal = self._cfg.get("causal", {}) or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        *,
        treatment: str,
        outcome: str,
        confounders: list[str],
        data: Any,  # pd.DataFrame
    ) -> ATEResult:
        """Estimate ATE for *treatment → outcome* using all enabled estimators.

        Parameters
        ----------
        treatment : str
            Column name for the treatment variable.
        outcome : str
            Column name for the outcome variable.
        confounders : list[str]
            Pre-computed backdoor adjustment set column names.  When
            empty, estimators that require confounders return ``None``.
        data : pd.DataFrame
            Tabular data containing all columns above.

        Returns
        -------
        ATEResult
            Dict-like object; ``None`` for any estimator that failed or
            was disabled.
        """
        result = ATEResult.empty()
        failure_reasons: dict[str, str] = {}

        _estimators = [
            ("ate_backdoor",    lambda: self._run_backdoor(treatment, outcome, confounders, data)),
            ("ate_ipw",         lambda: self._run_ipw(treatment, outcome, confounders, data)),
            ("ate_gps",         lambda: self._run_gps(treatment, outcome, confounders, data)),
            ("att_matching",    lambda: self._run_matching(treatment, outcome, confounders, data)),
            ("ate_doubly_robust", lambda: self._run_dr(treatment, outcome, confounders, data)),
        ]

        for key, fn in _estimators:
            try:
                result[key] = fn()
            except EstimatorError as exc:
                result[key] = None
                failure_reasons[key] = str(exc)
                log.debug("ATEEstimatorStack %s: %s", key, exc)

        if failure_reasons:
            result["failure_reasons"] = failure_reasons

        all_none = all(result[k] is None for k in ATEResult.__required_keys__ - {"error"})
        if all_none:
            result["error"] = "all_estimators_failed"

        return result

    # ------------------------------------------------------------------
    # Individual estimators (each returns float | None)
    # ------------------------------------------------------------------

    def _run_backdoor(
        self,
        treatment: str,
        outcome: str,
        confounders: list[str],
        data: Any,
    ) -> Optional[float]:
        """OLS backdoor ATE via DoWhy (falls back to sklearn LinearRegression)."""
        if treatment not in data.columns:
            raise InvalidInput(
                f"Treatment column '{treatment}' not found in data "
                f"(available: {list(data.columns)})"
            )
        if outcome not in data.columns:
            raise InvalidInput(
                f"Outcome column '{outcome}' not found in data "
                f"(available: {list(data.columns)})"
            )
        try:
            from sklearn.linear_model import LinearRegression
        except ImportError as exc:
            raise DependencyMissing(
                f"sklearn is required for the backdoor estimator: {exc}"
            ) from exc
        try:
            avail = [c for c in [treatment] + confounders if c in data.columns]
            lr = LinearRegression()
            lr.fit(data[avail], data[outcome])
            idx = avail.index(treatment)
            return float(lr.coef_[idx])
        except Exception as exc:
            log.debug("ATEEstimatorStack._run_backdoor failed: %s", exc)
            return None

    def _run_ipw(
        self,
        treatment: str,
        outcome: str,
        confounders: list[str],
        data: Any,
    ) -> Optional[float]:
        """Stabilised IPW ATE with cross-fit propensity scores."""
        if not self._causal.get("ipw_enabled", True):
            return None
        if not confounders:
            return None
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import StratifiedKFold
            from sklearn.base import clone as _clone

            avail_conf = [c for c in confounders if c in data.columns]
            if not avail_conf or treatment not in data.columns or outcome not in data.columns:
                return None

            T_raw = data[treatment].values.astype(np.float64)
            Y = data[outcome].values.astype(np.float64)
            X_conf = data[avail_conf].values.astype(np.float64)
            n = len(Y)

            t_median = float(np.median(T_raw))
            T_bin = (T_raw > t_median).astype(int)

            ps_base = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
            ps = np.zeros(n)
            n_splits = min(5, max(2, n // 50))

            if n >= 100:
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                for train_idx, test_idx in skf.split(X_conf, T_bin):
                    fold_model = _clone(ps_base)
                    fold_model.fit(X_conf[train_idx], T_bin[train_idx])
                    ps[test_idx] = fold_model.predict_proba(X_conf[test_idx])[:, 1]
            else:
                ps_base.fit(X_conf, T_bin)
                ps = ps_base.predict_proba(X_conf)[:, 1]

            mask = (ps >= 0.05) & (ps <= 0.95)
            ps_t, T_t, Y_t = ps[mask], T_bin[mask], Y[mask]
            if T_t.sum() < 5 or (1 - T_t).sum() < 5:
                return None

            p_treat = float(T_t.mean())
            w1 = (T_t * p_treat) / np.clip(ps_t, 1e-10, None)
            w0 = ((1 - T_t) * (1 - p_treat)) / np.clip(1 - ps_t, 1e-10, None)
            return float(
                (w1 * Y_t).sum() / np.clip(w1.sum(), 1e-10, None)
                - (w0 * Y_t).sum() / np.clip(w0.sum(), 1e-10, None)
            )
        except Exception as exc:
            log.debug("ATEEstimatorStack._run_ipw failed: %s", exc)
            return None

    def _run_gps(
        self,
        treatment: str,
        outcome: str,
        confounders: list[str],
        data: Any,
    ) -> Optional[float]:
        """GPS-weighted dose-response slope for continuous treatments."""
        if not self._causal.get("gps_enabled", True):
            return None
        if not confounders:
            return None
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor as HGB
            from scipy.stats import norm

            avail_conf = [c for c in confounders if c in data.columns]
            if not avail_conf or treatment not in data.columns or outcome not in data.columns:
                return None

            T = data[treatment].values.astype(np.float64)
            Y = data[outcome].values.astype(np.float64)
            X_conf = data[avail_conf].values.astype(np.float64)

            t_model = HGB(max_iter=200, max_depth=4, random_state=42)
            t_model.fit(X_conf, T)
            t_hat = t_model.predict(X_conf)
            residuals = T - t_hat
            sigma = float(np.std(residuals))
            if sigma < 1e-12:
                return None

            gps = np.clip(norm.pdf(T, loc=t_hat, scale=sigma), 1e-12, None)
            f_marg = np.clip(norm.pdf(T, loc=float(T.mean()), scale=float(T.std())), 1e-12, None)
            weights = np.clip(f_marg / gps, *np.percentile(f_marg / gps, [1, 99]))

            T_design = np.column_stack([np.ones_like(T), T])
            W_sqrt = np.sqrt(weights)
            beta, _, _, _ = np.linalg.lstsq(T_design * W_sqrt[:, None], Y * W_sqrt, rcond=None)
            return float(beta[1])
        except Exception as exc:
            log.debug("ATEEstimatorStack._run_gps failed: %s", exc)
            return None

    def _run_matching(
        self,
        treatment: str,
        outcome: str,
        confounders: list[str],
        data: Any,
    ) -> Optional[float]:
        """ATT via nearest-neighbour covariate matching with caliper."""
        if not self._causal.get("matching_enabled", True):
            return None
        if not confounders:
            return None
        try:
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import StandardScaler

            avail_conf = [c for c in confounders if c in data.columns]
            if not avail_conf or treatment not in data.columns or outcome not in data.columns:
                return None

            T_raw = data[treatment].values.astype(np.float64)
            Y = data[outcome].values.astype(np.float64)
            X_conf = data[avail_conf].values.astype(np.float64)

            t_mean, t_std = float(T_raw.mean()), float(T_raw.std())
            if t_std < 1e-12:
                return None

            hi_mask = T_raw >= (t_mean + 0.5 * t_std)
            lo_mask = T_raw <= (t_mean - 0.5 * t_std)
            keep = hi_mask | lo_mask
            if keep.sum() < 20:
                return None

            T_bin = hi_mask[keep].astype(int)
            Y_k = Y[keep]
            X_k = X_conf[keep]
            X_scaled = StandardScaler().fit_transform(X_k)

            treated_idx = np.where(T_bin == 1)[0]
            control_idx = np.where(T_bin == 0)[0]
            if len(treated_idx) < 5 or len(control_idx) < 5:
                return None

            caliper = 0.25 * np.sqrt(X_scaled.shape[1])
            nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn.fit(X_scaled[control_idx])
            distances, indices = nn.kneighbors(X_scaled[treated_idx])

            within = distances.ravel() <= caliper
            matched_t = treated_idx[within]
            matched_c = control_idx[indices.ravel()[within]]
            if len(matched_t) < 5:
                return None

            return float(Y_k[matched_t].mean() - Y_k[matched_c].mean())
        except Exception as exc:
            log.debug("ATEEstimatorStack._run_matching failed: %s", exc)
            return None

    def _run_dr(
        self,
        treatment: str,
        outcome: str,
        confounders: list[str],
        data: Any,
    ) -> Optional[float]:
        """Doubly-robust ATE via EconML LinearDML or sklearn cross-fit DML."""
        if not self._causal.get("doubly_robust", True):
            return None
        if not confounders:
            return None
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor as HGB

            avail_conf = [c for c in confounders if c in data.columns]
            if not avail_conf or treatment not in data.columns or outcome not in data.columns:
                return None

            T = data[treatment].values.reshape(-1, 1).astype(np.float64)
            Y = data[outcome].values.astype(np.float64)
            W = data[avail_conf].values.astype(np.float64)

            try:
                from econml.dml import LinearDML
                dml = LinearDML(
                    model_y=HGB(max_iter=200, max_depth=4, random_state=42),
                    model_t=HGB(max_iter=200, max_depth=4, random_state=42),
                    discrete_treatment=False,
                    cv=3,
                    random_state=42,
                )
                dml.fit(Y, T, W=W)
                return float(np.mean(dml.const_marginal_ate()))
            except Exception:
                # sklearn cross-fit DML fallback
                from sparc.causal.counterfactual_engine import CounterfactualEngine
                coeff, _se, _ = CounterfactualEngine._fit_edge_dml_sklearn(
                    T, Y, W,
                    model_y=HGB(max_iter=200, max_depth=4, random_state=42),
                    model_t=HGB(max_iter=200, max_depth=4, random_state=42),
                    n_splits=3,
                )
                return float(coeff)
        except Exception as exc:
            log.debug("ATEEstimatorStack._run_dr failed: %s", exc)
            return None
