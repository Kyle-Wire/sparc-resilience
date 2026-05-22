"""
Spatial Lag (SLX) feature augmentation for SPARC.

For a uniform fishnet grid, each predictor X_j is augmented with its
neighborhood-average WX_j:

    WX_j(i) = (1/k) * Σ_{n ∈ N(i)} X_j(n)

where N(i) is the k nearest spatial neighbors of point i (queen's
contiguity, k=8 by default on a regular grid).

The augmented feature matrix [X | WX] lets base models (GWR, GWRF,
GGPGAM) distinguish own-location predictors from neighbor-average
predictors, capturing spatial spillover effects that local regression
alone cannot represent.  The complementary spatial lag of the treatment
variable is already handled post-hoc by NetworkInterferenceModel; this
module provides the *input-feature* side of SLX — spatially lagged
covariates fed *into* the model before fitting.

References
----------
LeSage & Pace, "Introduction to Spatial Econometrics", CRC 2009.
Halleck Vega & Elhorst, "The SLX Model", J. Regional Science, 2015.

Usage
-----
>>> slx = SpatialLagTransformer(k=8)
>>> X_aug, names_aug = slx.fit_transform(coords, X, feature_names)
# X_aug shape: (N, 2*P)
# names_aug:   ['x1', ..., 'xP', 'slx_x1', ..., 'slx_xP']
"""

from __future__ import annotations

import numpy as np

from sparc.features.geometry import build_knn_index


class SpatialLagTransformer:
    """
    Append spatially-lagged predictor averages (WX) to a feature matrix.

    Computes the queen-contiguity (or k-nearest-neighbor) spatial lag
    of every predictor column using a single KNN index built from
    projected spatial coordinates.  The transformation is fit once on
    the full dataset and can then be applied to any subset (train/test
    slice) without re-fitting.

    Parameters
    ----------
    k : int
        Number of neighbors.  On a regular fishnet grid, k=8 gives
        queen contiguity (N/S/E/W + 4 diagonals).  k=4 gives rook
        contiguity.
    """

    def __init__(self, k: int = 8) -> None:
        self.k = k
        self._knn_idx: np.ndarray | None = None   # (N, k)  int64

    # ------------------------------------------------------------------
    def fit(self, coords: np.ndarray) -> "SpatialLagTransformer":
        """Build the KNN index from projected coordinates.

        Parameters
        ----------
        coords : (N, 2) projected coordinate array (metres)
        """
        self._knn_idx = build_knn_index(np.asarray(coords, dtype=float), self.k)
        return self

    # ------------------------------------------------------------------
    def transform(
        self,
        X: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Compute WX and return the augmented feature matrix.

        Parameters
        ----------
        X : (N, P) feature matrix — **must be unscaled** (scaling happens
            downstream).  Row order must match the coords used in fit().
        feature_names : P-length list of predictor names.  If None,
            auto-generates 'f0', 'f1', ...

        Returns
        -------
        X_aug : (N, 2*P) — columns ordered [X | WX]
        names_aug : 2*P names — original names + 'slx_{name}'
        """
        if self._knn_idx is None:
            raise RuntimeError("Call fit() or fit_transform() before transform().")

        X = np.asarray(X, dtype=float)
        N, P = X.shape

        if len(self._knn_idx) != N:
            raise ValueError(
                f"X has {N} rows but KNN index has {len(self._knn_idx)} entries. "
                "Re-fit on the same coordinate array."
            )

        if feature_names is None:
            feature_names = [f"f{j}" for j in range(P)]

        # WX[i, j] = mean(X[knn[i, :], j])
        # knn_idx: (N, k) — all values are valid global indices (no -1 sentinel)
        WX = X[self._knn_idx].mean(axis=1)   # (N, k, P) -> mean over k -> (N, P)

        X_aug = np.hstack([X, WX])
        names_aug = list(feature_names) + [f"slx_{n}" for n in feature_names]
        return X_aug, names_aug

    # ------------------------------------------------------------------
    def fit_transform(
        self,
        coords: np.ndarray,
        X: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Fit on coords then transform X in one call.

        Parameters
        ----------
        coords : (N, 2) projected coordinates
        X : (N, P) unscaled feature matrix
        feature_names : optional P-length names

        Returns
        -------
        X_aug : (N, 2*P)
        names_aug : 2*P names
        """
        return self.fit(coords).transform(X, feature_names)
