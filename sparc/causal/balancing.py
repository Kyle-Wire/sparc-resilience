"""Covariate-Balancing Propensity Score (CBPS) for causal balancing.

Wager (2025) audit Gap 9. Imai & Ratkovic (2014) / Wager Eq. 7.9-7.10.

Solves the convex problem:
    min_β  Σᵢ -W_i log e_β(X_i) - (1-W_i) log(1 - e_β(X_i))
    s.t.   Σᵢ X_i (W_i / e_β - (1-W_i)/(1 - e_β)) = 0

i.e., logistic likelihood + covariate-balance constraint via Lagrangian
penalty. We solve the unconstrained penalised version with L-BFGS, which
converges to the just-identified CBPS solution as the penalty grows.

For continuous treatments, ``binarize_continuous=True`` splits W on its
median (default) before fitting.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize

from ._audit import mark_addressed

GAP_9_IMPLEMENTED = True


def _logistic(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class CBPSEstimator:
    """Covariate-Balancing Propensity Score estimator."""

    def __init__(self,
                 binarize_continuous: bool = True,
                 binarize_threshold: Optional[float] = None,
                 lambda_balance: float = 5.0,
                 trim: float = 0.02):
        self.binarize_continuous = binarize_continuous
        self.binarize_threshold = binarize_threshold
        self.lambda_balance = lambda_balance
        self.trim = trim
        self.beta_: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None
        self.X_: Optional[np.ndarray] = None
        self.W_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ fit

    def fit(self, X: np.ndarray, W: np.ndarray) -> "CBPSEstimator":
        X = np.atleast_2d(X).astype(float)
        if X.shape[0] == 1 and X.shape[1] != len(W):
            X = X.T
        W = np.asarray(W, dtype=float).ravel()

        if self.binarize_continuous and not np.array_equal(np.unique(W), [0, 1]):
            thr = (self.binarize_threshold
                   if self.binarize_threshold is not None
                   else float(np.median(W)))
            W = (W > thr).astype(float)

        n, p = X.shape
        Xb = np.hstack([np.ones((n, 1)), X])

        def neg_loglik_balance(beta):
            z = Xb @ beta
            e = _logistic(z)
            e = np.clip(e, self.trim, 1 - self.trim)
            ll = W * np.log(e) + (1 - W) * np.log(1 - e)
            balance = (Xb.T @ (W / e - (1 - W) / (1 - e))) / n
            penalty = float(self.lambda_balance) * float(balance @ balance)
            return -ll.mean() + penalty

        beta0 = np.zeros(p + 1)
        res = minimize(neg_loglik_balance, beta0, method="L-BFGS-B",
                       options={"maxiter": 200, "ftol": 1e-8})
        self.beta_ = res.x
        e = np.clip(_logistic(Xb @ self.beta_), self.trim, 1 - self.trim)
        self.weights_ = W / e + (1 - W) / (1 - e)
        self.X_ = X
        self.W_ = W

        mark_addressed(9)
        return self

    # ------------------------------------------------------------------ ate

    def ate(self, Y: np.ndarray) -> float:
        if self.weights_ is None:
            raise RuntimeError("fit() must be called before ate()")
        Y = np.asarray(Y, dtype=float).ravel()
        w_treat = self.weights_ * (self.W_ == 1)
        w_ctrl = self.weights_ * (self.W_ == 0)
        mu1 = (w_treat * Y).sum() / w_treat.sum()
        mu0 = (w_ctrl * Y).sum() / w_ctrl.sum()
        return float(mu1 - mu0)

    def balance_diagnostics(self) -> dict:
        if self.weights_ is None:
            raise RuntimeError("fit() must be called first")
        X, W, w = self.X_, self.W_, self.weights_
        # Standardised mean differences before/after weighting.
        mean_t = X[W == 1].mean(axis=0)
        mean_c = X[W == 0].mean(axis=0)
        sd_pool = np.sqrt(0.5 * (X[W == 1].var(axis=0, ddof=1)
                                 + X[W == 0].var(axis=0, ddof=1)))
        smd_raw = (mean_t - mean_c) / np.where(sd_pool > 0, sd_pool, 1.0)

        wt_t = w[W == 1]
        wt_c = w[W == 0]
        mean_t_w = (wt_t[:, None] * X[W == 1]).sum(0) / wt_t.sum()
        mean_c_w = (wt_c[:, None] * X[W == 0]).sum(0) / wt_c.sum()
        smd_weighted = (mean_t_w - mean_c_w) / np.where(sd_pool > 0, sd_pool, 1.0)
        return {"smd_raw": smd_raw, "smd_weighted": smd_weighted}
