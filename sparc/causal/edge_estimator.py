"""EdgeEstimator — unified interface for per-edge structural coefficient estimation.

Replaces the three-path ``if estimator_name == 'dml' / 'hgb' / else`` dispatch
inside ``CausalValidator.fit()`` and ``CounterfactualEngine.fit()`` with three
adapter classes that all satisfy the same ``Coefficient``-returning interface.

Design
------
``Coefficient`` is a flat dataclass — one result, one place to look.  The three
separate dicts (``_edge_models``, ``_dml_models``, ``_coeff_ses``) in the current
god-object become a single ``dict[tuple[str,str], Coefficient]`` owned by
``EdgeFitter`` (C1).

``EdgeEstimator`` is a structural protocol (Python 3.12 ``typing.Protocol``).
No base class is needed — any object with a conforming ``fit`` signature works.

Usage::

    from sparc.causal.edge_estimator import OLSEdgeEstimator, Coefficient

    est = OLSEdgeEstimator()
    coeff: Coefficient = est.fit("Pct_Canopy", "AAT_z", df, confounders=["Elevation_m"])
    print(coeff.value, coeff.se)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Coefficient:
    """Result of fitting one DAG edge.

    Attributes
    ----------
    edge : tuple[str, str]
        ``(parent, child)`` pair.
    value : float
        Estimated marginal effect (pre-shrinkage).
    se : float
        Standard error, 0.0 when not available (OLS, HGB).
    model : Any
        Fitted model object for later non-linear prediction (HGB) or
        inference (DML).  None for OLS.
    metadata : dict
        Free-form diagnostic info: nuisance source, number of folds, etc.
    """

    edge: tuple
    value: float
    se: float = 0.0
    model: Any = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class EdgeEstimator(Protocol):
    """Structural protocol satisfied by OLS / HGB / DML adapters."""

    def fit(
        self,
        parent: str,
        child: str,
        data: pd.DataFrame,
        confounders: List[str],
    ) -> Coefficient:
        """Estimate the marginal effect of *parent* on *child* in *data*."""
        ...


# ---------------------------------------------------------------------------
# OLS adapter
# ---------------------------------------------------------------------------

class OLSEdgeEstimator:
    """Ordinary Least Squares with backdoor adjustment.

    Simple and fast; appropriate when the DAG is small or as a fallback.
    """

    def fit(
        self,
        parent: str,
        child: str,
        data: pd.DataFrame,
        confounders: List[str],
    ) -> Coefficient:
        from sklearn.linear_model import LinearRegression

        X_cols = [parent] + confounders
        lr = LinearRegression()
        lr.fit(data[X_cols], data[child])
        value = float(lr.coef_[0])
        return Coefficient(edge=(parent, child), value=value, se=0.0, model=None)


# ---------------------------------------------------------------------------
# HGB adapter
# ---------------------------------------------------------------------------

class HGBEdgeEstimator:
    """Monotone-constrained HistGradientBoosting with finite-difference marginal effect.

    Parameters
    ----------
    monotone : dict[str, int] or None
        Maps variable names to sign constraints (+1 / -1 / 0).  Applied only
        when ``child == target``.
    target : str or None
        Name of the outcome variable.  Required to apply monotone constraints.
    """

    def __init__(
        self,
        monotone: Optional[dict] = None,
        target: Optional[str] = None,
    ) -> None:
        self.monotone = monotone or {}
        self.target = target

    def fit(
        self,
        parent: str,
        child: str,
        data: pd.DataFrame,
        confounders: List[str],
    ) -> Coefficient:
        from sklearn.ensemble import HistGradientBoostingRegressor

        X_cols = [parent] + confounders

        mc = []
        for col in X_cols:
            if child == self.target:
                mc.append(self.monotone.get(col, 0))
            else:
                mc.append(0)

        hgb = HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=20,
            monotonic_cst=mc,
            random_state=42,
        )
        X = data[X_cols].values
        y = data[child].values
        hgb.fit(X, y)

        # Finite-difference marginal effect: mean ∂child/∂parent
        parent_idx = X_cols.index(parent)
        std = np.std(X[:, parent_idx])
        eps = std * 0.01
        if eps < 1e-8:
            eps = 1e-3  # fallback for near-zero-variance columns

        X_plus = X.copy()
        X_plus[:, parent_idx] += eps
        marginal = (hgb.predict(X_plus) - hgb.predict(X)) / eps
        value = float(np.mean(marginal))

        return Coefficient(edge=(parent, child), value=value, se=0.0, model=hgb)


# ---------------------------------------------------------------------------
# DML adapter
# ---------------------------------------------------------------------------

class DMLEdgeEstimator:
    """Debiased / Orthogonal ML (Chernozhukov et al. 2018).

    Wraps ``CounterfactualEngine._fit_edge_dml_sklearn`` so it can be
    used standalone without a ``CounterfactualEngine`` instance.

    Parameters
    ----------
    n_splits : int
        Number of cross-fitting folds.
    coords : np.ndarray or None
        (N, 2) spatial coordinates for spatial-block CV.
    nuisance : OOFNuisance or None
        Pre-fitted Stage 2 OOF outcome predictions.  When provided,
        model_y cross-fitting is bypassed (JD-1 path).
    """

    def __init__(
        self,
        n_splits: int = 5,
        coords: Optional[np.ndarray] = None,
        nuisance=None,  # OOFNuisance | None — avoid circular import
    ) -> None:
        self.n_splits = n_splits
        self.coords = coords
        self.nuisance = nuisance

    def fit(
        self,
        parent: str,
        child: str,
        data: pd.DataFrame,
        confounders: List[str],
    ) -> Coefficient:
        from sklearn.ensemble import HistGradientBoostingRegressor as HGB
        from sparc.causal.counterfactual_engine import CounterfactualEngine

        T = data[[parent]].values
        Y = data[child].values
        W = data[confounders].values if confounders else None

        precomputed_Y_res: Optional[np.ndarray] = None
        meta: dict = {}

        if self.nuisance is not None:
            preds = np.asarray(self.nuisance.predictions).ravel()
            if len(preds) == len(Y):
                precomputed_Y_res = Y - preds
                meta["nuisance_source"] = self.nuisance.source
            # If shape doesn't match, fall through to standard cross-fitting
            # (NuisanceCache.load already validates length, but defensive here)

        model_y = HGB(
            max_iter=100, max_depth=3, learning_rate=0.05,
            min_samples_leaf=20, random_state=42,
        )
        model_t = HGB(
            max_iter=100, max_depth=3, learning_rate=0.05,
            min_samples_leaf=20, random_state=42,
        )

        value, se, dml_model = CounterfactualEngine._fit_edge_dml_sklearn(
            T, Y, W,
            model_y=model_y if precomputed_Y_res is None else None,
            model_t=model_t,
            n_splits=self.n_splits,
            precomputed_Y_res=precomputed_Y_res,
        )

        return Coefficient(
            edge=(parent, child),
            value=float(value),
            se=float(se),
            model=dml_model,
            metadata=meta,
        )
