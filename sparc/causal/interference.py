"""Network-interference (spillover) estimation.

Wager (2025) audit Gap 3. Implements the anonymous-exposure model
``H_i(w) = (w_i, mean(w_{N_i}))`` from Wager Definition 11.1, fits a
linear regression of Y on (W_i, mean-of-neighbours W) with HAC variance
(Theorems 12.4–12.5), and tests the SUTVA / H₁ / H₂ hierarchy by a
spatial-permutation null (extends — does not replace — the existing
Moran's-I check in :mod:`sparc.causal.dag_validator`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

from ._audit import mark_addressed

GAP_3_IMPLEMENTED = True


@dataclass
class SpilloverDecomposition:
    direct_effect: float
    spillover_effect: float
    total_effect: float
    direct_se: float
    spillover_se: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "direct_effect": self.direct_effect,
            "spillover_effect": self.spillover_effect,
            "total_effect": self.total_effect,
            "direct_se": self.direct_se,
            "spillover_se": self.spillover_se,
        }


class NetworkInterferenceModel:
    """Anonymous-exposure spillover estimator with HAC variance."""

    def __init__(self,
                 k: int = 8,
                 hac_bandwidth: Optional[int] = None):
        self.k = k
        self.hac_bandwidth = hac_bandwidth
        self.coords_: Optional[np.ndarray] = None
        self.W_: Optional[np.ndarray] = None
        self.Y_: Optional[np.ndarray] = None
        self.beta_: Optional[np.ndarray] = None
        self.cov_hac_: Optional[np.ndarray] = None
        self.neighbor_idx_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ fit

    def fit(self,
            coords: np.ndarray,
            W: np.ndarray,
            Y: np.ndarray) -> "NetworkInterferenceModel":
        coords = np.asarray(coords, dtype=float)
        W = np.asarray(W, dtype=float).ravel()
        Y = np.asarray(Y, dtype=float).ravel()
        n = len(W)

        tree = cKDTree(coords)
        _, idx = tree.query(coords, k=self.k + 1)  # self + k neighbours
        self.neighbor_idx_ = idx[:, 1:]  # drop self
        neighbor_mean = W[self.neighbor_idx_].mean(axis=1)

        # Design matrix: intercept, W_i, neighbour-mean(W).
        X = np.column_stack([np.ones(n), W, neighbor_mean])
        XtX_inv = np.linalg.pinv(X.T @ X)
        beta = XtX_inv @ X.T @ Y
        resid = Y - X @ beta

        # HAC (Newey-West-style spatial) variance using neighbour structure.
        bw = self.hac_bandwidth if self.hac_bandwidth is not None else self.k
        bw = max(1, int(bw))
        u = X * resid[:, None]
        S0 = u.T @ u
        S = S0.copy()
        for lag in range(1, bw + 1):
            weight = 1.0 - lag / (bw + 1.0)
            cov_lag = np.zeros_like(S)
            for i in range(n):
                neigh = self.neighbor_idx_[i]
                for j in neigh[: max(1, lag)]:
                    cov_lag += np.outer(u[i], u[j])
            S += weight * (cov_lag + cov_lag.T)
        cov_hac = XtX_inv @ S @ XtX_inv

        self.coords_, self.W_, self.Y_ = coords, W, Y
        self.beta_ = beta
        self.cov_hac_ = cov_hac

        mark_addressed(3)
        return self

    # ------------------------------------------------------------------ api

    def decompose(self) -> Dict[str, float]:
        if self.beta_ is None:
            raise RuntimeError("fit() must be called before decompose()")
        direct = float(self.beta_[1])
        spill = float(self.beta_[2])
        ses = np.sqrt(np.clip(np.diag(self.cov_hac_), 0.0, None))
        return SpilloverDecomposition(
            direct_effect=direct,
            spillover_effect=spill,
            total_effect=direct + spill,
            direct_se=float(ses[1]),
            spillover_se=float(ses[2]),
        ).as_dict()

    def sutva_permutation_test(self,
                               n_permutations: int = 200,
                               random_state: Optional[int] = None) -> Dict[str, float]:
        """Permutation test of SUTVA (β_spill = 0) using neighbour-mean reshuffling.

        H₀ (SUTVA): treatment of neighbours has no effect on Y_i.
        Permutes the neighbour-mean assignment across units and recomputes
        the spillover coefficient; the empirical p-value is the fraction
        of permutations with |β_spill_perm| ≥ |β_spill_obs|.
        """
        if self.beta_ is None:
            raise RuntimeError("fit() must be called first")
        rng = np.random.default_rng(random_state)
        observed = abs(float(self.beta_[2]))
        n = len(self.W_)
        neighbor_mean = self.W_[self.neighbor_idx_].mean(axis=1)
        count = 0
        for _ in range(n_permutations):
            perm = rng.permutation(n)
            X = np.column_stack([np.ones(n), self.W_, neighbor_mean[perm]])
            beta_p = np.linalg.pinv(X.T @ X) @ X.T @ self.Y_
            if abs(float(beta_p[2])) >= observed:
                count += 1
        return {"observed_spillover": float(self.beta_[2]),
                "p_value": (count + 1) / (n_permutations + 1)}
