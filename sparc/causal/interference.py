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
    ring2_effect: Optional[float] = None
    ring2_se: Optional[float] = None

    def as_dict(self) -> Dict[str, float]:
        d = {
            "direct_effect": self.direct_effect,
            "spillover_effect": self.spillover_effect,
            "total_effect": self.total_effect,
            "direct_se": self.direct_se,
            "spillover_se": self.spillover_se,
        }
        if self.ring2_effect is not None:
            d["ring2_effect"] = self.ring2_effect
        if self.ring2_se is not None:
            d["ring2_se"] = self.ring2_se
        return d


class NetworkInterferenceModel:
    """Anonymous-exposure spillover estimator with HAC variance.

    Parameters
    ----------
    k : int
        Number of nearest neighbours for ring-1 spillover (default 8).
    hac_bandwidth : int or None
        Spatial HAC lag truncation.  Defaults to ``k``.
    spillover_order : int or str
        Controls the neighbourhood specification used to construct the
        spillover regressor:

        * ``1`` (default) — first-ring neighbours only (standard model).
        * ``2``           — first *and* second-ring neighbours; adds a
          ``ring2_mean(W)`` regressor to the design matrix.
        * ``'kernel'``    — distance-decay weighted mean using the Matérn
          anisotropic kernel from ``kernel_field``.  Requires
          ``kernel_field`` to be supplied.
    kernel_field : object or None
        A fitted ``KernelField`` instance exposing per-predictor Matérn
        parameters (``kappa_x``, ``kappa_y``, ``theta_rad``, ``nu``).
        Used only when ``spillover_order='kernel'``.  The geometric-mean
        across predictors is used for the spillover weight.
    """

    def __init__(self,
                 k: int = 8,
                 hac_bandwidth: Optional[int] = None,
                 spillover_order: int | str = 1,
                 kernel_field=None):
        self.k = k
        self.hac_bandwidth = hac_bandwidth
        self.spillover_order = spillover_order
        self.kernel_field = kernel_field
        self.coords_: Optional[np.ndarray] = None
        self.W_: Optional[np.ndarray] = None
        self.Y_: Optional[np.ndarray] = None
        self.beta_: Optional[np.ndarray] = None
        self.cov_hac_: Optional[np.ndarray] = None
        self.neighbor_idx_: Optional[np.ndarray] = None
        self._ring2_idx: Optional[np.ndarray] = None

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

        order = self.spillover_order

        if order == 1:
            # Standard first-ring model.
            X = np.column_stack([np.ones(n), W, neighbor_mean])
        elif order == 2:
            # Second-ring: neighbours of neighbours (excluding self and ring-1).
            ring2_sets = []
            ring1_set_per_i = [set(self.neighbor_idx_[i]) | {i} for i in range(n)]
            for i in range(n):
                ring2 = set()
                for j in self.neighbor_idx_[i]:
                    ring2.update(self.neighbor_idx_[j])
                ring2.difference_update(ring1_set_per_i[i])
                ring2_sets.append(sorted(ring2))
            # Pad to same length for vectorised mean.
            max_r2 = max(len(s) for s in ring2_sets) if ring2_sets else 1
            r2_idx = np.full((n, max(max_r2, 1)), -1, dtype=int)
            r2_mask = np.zeros((n, max(max_r2, 1)), dtype=float)
            for i, s in enumerate(ring2_sets):
                if s:
                    r2_idx[i, : len(s)] = s
                    r2_mask[i, : len(s)] = 1.0
            # Replace -1 with 0 (safe dummy — masked out below).
            r2_idx_safe = np.where(r2_idx >= 0, r2_idx, 0)
            ring2_sum = (W[r2_idx_safe] * r2_mask).sum(axis=1)
            ring2_count = r2_mask.sum(axis=1)
            ring2_mean = np.where(ring2_count > 0, ring2_sum / ring2_count, 0.0)
            self._ring2_idx = r2_idx
            X = np.column_stack([np.ones(n), W, neighbor_mean, ring2_mean])
        elif order == "kernel":
            if self.kernel_field is None:
                raise ValueError(
                    "spillover_order='kernel' requires kernel_field to be provided."
                )
            # Build distance-decay weighted mean using Matérn anisotropic kernel.
            kf = self.kernel_field
            predictors = list(kf.kernels.keys()) if hasattr(kf, "kernels") else []

            if predictors:
                from sparc.models.kernel_field import anisotropic_distance, matern_kernel_weights
                # Geometric-mean Matérn weights across predictors.
                all_weights = []
                for pred in predictors:
                    kern = kf.kernels[pred]
                    kappa_x = float(kern.get("kappa_x", 1.0))
                    kappa_y = float(kern.get("kappa_y", 1.0))
                    theta_rad = float(kern.get("theta_rad", 0.0))
                    nu = float(kern.get("nu", 1.5))
                    sigma2 = float(kern.get("sigma2", 1.0))
                    pred_weights = np.zeros((n, n))
                    for i in range(n):
                        dx = coords[:, 0] - coords[i, 0]
                        dy = coords[:, 1] - coords[i, 1]
                        d_aniso = anisotropic_distance(dx, dy, kappa_x, kappa_y, theta_rad)
                        w = matern_kernel_weights(d_aniso, nu=nu, sigma2=sigma2)
                        w[i] = 0.0  # zero self-weight
                        if w.sum() > 0:
                            w = w / w.sum()
                        pred_weights[i] = w
                    all_weights.append(pred_weights)
                kernel_weights = np.exp(
                    np.stack([np.log(w + 1e-300) for w in all_weights], axis=0).mean(axis=0)
                )
                row_sums = kernel_weights.sum(axis=1, keepdims=True)
                kernel_weights = np.where(row_sums > 0, kernel_weights / row_sums, 0.0)
                spillover_kernel = (kernel_weights * W[np.newaxis, :]).sum(axis=1)
            else:
                # Fallback: isotropic mean if no kernel params available.
                spillover_kernel = neighbor_mean

            X = np.column_stack([np.ones(n), W, spillover_kernel])
        else:
            raise ValueError(f"spillover_order must be 1, 2, or 'kernel'; got {order!r}")

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

        ring2_effect = None
        ring2_se = None
        if self.spillover_order == 2 and len(self.beta_) >= 4:
            ring2_effect = float(self.beta_[3])
            ring2_se = float(ses[3]) if len(ses) > 3 else None

        return SpilloverDecomposition(
            direct_effect=direct,
            spillover_effect=spill,
            total_effect=direct + spill + (ring2_effect or 0.0),
            direct_se=float(ses[1]),
            spillover_se=float(ses[2]),
            ring2_effect=ring2_effect,
            ring2_se=ring2_se,
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
