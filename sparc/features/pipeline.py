"""
sparc/features/pipeline.py — Fold-scoped feature transform for Stage 2.

A single ``FeaturePipeline`` instance is created *per CV fold* so that:

* The ``StandardScaler`` is fit only on training data (no leakage from the
  test fold into the normalisation statistics).
* When ``use_laplacian=True``, the ``FoldAwareLaplacianEigenmap`` is also
  fit on training coords only and projects test points via Nyström extension.

For Stage 4 (scenario predictions) a pipeline should be fit on the full
dataset and persisted with ``save()`` / loaded with ``load()``.

Interface
---------
::

    fp = FeaturePipeline(use_laplacian=True, n_laplacian_components=50)

    # Inside a fold worker — train split only:
    X_train_out, names_out = fp.fit_transform(X_train, coords_train, feature_names)
    X_test_out            = fp.transform(X_test, coords_test)

    # Persist for deployment:
    fp.save(Path("output/feature_pipeline.pkl"))
    fp2 = FeaturePipeline.load(Path("output/feature_pipeline.pkl"))
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler


class FeaturePipeline:
    """Fold-scoped feature transform: StandardScaler + optional FoldAwareLaplacian.

    Parameters
    ----------
    use_laplacian : bool
        When *True*, appends ``n_laplacian_components`` Laplacian eigenmap
        features after the scaled predictors.  The eigenmap graph is built
        on training coordinates only; test points are projected via Nyström
        extension so no test-fold information enters the transform.
    n_laplacian_components : int
        Number of Laplacian eigenmaps to append.  Ignored when
        ``use_laplacian=False``.
    """

    def __init__(
        self,
        use_laplacian: bool = False,
        n_laplacian_components: int = 50,
    ) -> None:
        self.use_laplacian = use_laplacian
        self.n_laplacian_components = n_laplacian_components

        self._scaler: Optional[StandardScaler] = None
        self._laplacian = None
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        feature_names: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """Fit on *training* data and return transformed array + column names.

        Parameters
        ----------
        X : ndarray, shape (n_train, n_features)
            Raw (unscaled) feature matrix.
        coords : ndarray, shape (n_train, 2)
            Spatial coordinates corresponding to each row of *X*.
        feature_names : list[str]
            Names for each column of *X*.

        Returns
        -------
        X_out : ndarray, shape (n_train, n_features [+ n_laplacian_components])
        names_out : list[str]
        """
        X_arr = np.asarray(X, dtype=np.float64)
        coords_arr = np.asarray(coords, dtype=np.float64)

        # --- standard scaling ---
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_arr)
        self._feature_names = list(feature_names)
        out_names = list(feature_names)

        # --- optional fold-aware laplacian ---
        if self.use_laplacian:
            lapeig = self._build_laplacian()
            eig_train = lapeig.fit_transform(coords_arr)
            X_scaled = np.hstack([X_scaled, eig_train])
            out_names = out_names + (lapeig.eigenmap_cols_ or [
                f"LAPEIG_{i+1}" for i in range(eig_train.shape[1])
            ])

        return X_scaled, out_names

    def transform(
        self,
        X: np.ndarray,
        coords: np.ndarray,
    ) -> np.ndarray:
        """Apply fit-time transform to a new (test) array.

        Parameters
        ----------
        X : ndarray, shape (n_test, n_features)
        coords : ndarray, shape (n_test, 2)

        Returns
        -------
        ndarray, shape (n_test, n_features [+ n_laplacian_components])
        """
        if self._scaler is None:
            raise RuntimeError("Call fit_transform() before transform().")

        X_arr = np.asarray(X, dtype=np.float64)
        coords_arr = np.asarray(coords, dtype=np.float64)

        X_scaled = self._scaler.transform(X_arr)

        if self.use_laplacian and self._laplacian is not None:
            eig_test = self._laplacian.transform(coords_arr)
            X_scaled = np.hstack([X_scaled, eig_test])

        return X_scaled

    def save(self, path: Path | str) -> None:
        """Persist the fitted pipeline to *path* (pickle)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path | str) -> "FeaturePipeline":
        """Load a previously saved pipeline from *path*."""
        with open(Path(path), "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj)}, expected FeaturePipeline")
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_laplacian(self):
        """Construct (and cache) the FoldAwareLaplacianEigenmap instance."""
        from sparc.features.fold_aware_laplacian import FoldAwareLaplacianEigenmap
        self._laplacian = FoldAwareLaplacianEigenmap(
            n_components=self.n_laplacian_components,
            k_neighbors=10,
        )
        return self._laplacian


# ---------------------------------------------------------------------------
# FoldFeatureContext — ordering-safe alternative to FeaturePipeline
# ---------------------------------------------------------------------------

class FoldFeatureContext:
    """Fold-scoped feature transform that encodes train/test split at construction.

    Unlike :class:`FeaturePipeline`, which exposes a stateful ``fit_transform``
    / ``transform`` sequence, ``FoldFeatureContext`` accepts both train and test
    data at construction time.  The ordering constraint (fit on train, apply to
    test) is enforced structurally: ``test_transform()`` simply cannot be called
    before ``train_transform()`` because the scaler is built during that call.

    Interface
    ---------
    ::

        ctx = FoldFeatureContext(
            X_train, coords_train,
            X_test,  coords_test,
            feature_names,
            use_laplacian=False,
        )
        X_train_scaled, names = ctx.train_transform()
        X_test_scaled          = ctx.test_transform()

        # Persist for scenario inference:
        ctx.save(Path("output/fold_ctx.pkl"))
        loaded = FoldFeatureContext.load(Path("output/fold_ctx.pkl"))
    """

    def __init__(
        self,
        X_train: np.ndarray,
        coords_train: np.ndarray,
        X_test: np.ndarray,
        coords_test: np.ndarray,
        feature_names: list[str],
        use_laplacian: bool = False,
        n_laplacian_components: int = 50,
    ) -> None:
        self._X_train = np.asarray(X_train, dtype=np.float64)
        self._coords_train = np.asarray(coords_train, dtype=np.float64)
        self._X_test = np.asarray(X_test, dtype=np.float64)
        self._coords_test = np.asarray(coords_test, dtype=np.float64)
        self._feature_names = list(feature_names)
        self.use_laplacian = use_laplacian
        self.n_laplacian_components = n_laplacian_components

        self._scaler: Optional[StandardScaler] = None
        self._laplacian = None
        self._out_names: list[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_transform(self) -> tuple[np.ndarray, list[str]]:
        """Fit on training data and return (X_scaled, feature_names).

        Fits the StandardScaler (and optional Laplacian) on the training split
        stored at construction.  Safe to call multiple times — re-fits each
        time, which is intentional for pipeline restarts.

        Returns
        -------
        X_train_scaled : ndarray, shape (n_train, n_out_features)
        out_names : list[str]
        """
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(self._X_train)
        out_names = list(self._feature_names)

        if self.use_laplacian:
            self._laplacian = self._build_laplacian()
            eig_train = self._laplacian.fit_transform(self._coords_train)
            X_scaled = np.hstack([X_scaled, eig_train])
            out_names = out_names + (
                self._laplacian.eigenmap_cols_
                or [f"LAPEIG_{i+1}" for i in range(eig_train.shape[1])]
            )

        self._out_names = out_names
        self._fitted = True
        return X_scaled, out_names

    def test_transform(self) -> np.ndarray:
        """Apply the train-fit scale to the test split.

        Must be called after :meth:`train_transform`.

        Returns
        -------
        X_test_scaled : ndarray, shape (n_test, n_out_features)
        """
        if not self._fitted or self._scaler is None:
            raise RuntimeError(
                "Call train_transform() before test_transform(). "
                "The scaler must be fit on training data first."
            )

        X_scaled = self._scaler.transform(self._X_test)

        if self.use_laplacian and self._laplacian is not None:
            eig_test = self._laplacian.transform(self._coords_test)
            X_scaled = np.hstack([X_scaled, eig_test])

        return X_scaled

    def save(self, path: "Path | str") -> None:
        """Persist to *path* (pickle)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: "Path | str") -> "FoldFeatureContext":
        """Load a previously saved context from *path*."""
        with open(Path(path), "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj)}, expected FoldFeatureContext")
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_laplacian(self):
        from sparc.features.fold_aware_laplacian import FoldAwareLaplacianEigenmap
        return FoldAwareLaplacianEigenmap(
            n_components=self.n_laplacian_components,
            k_neighbors=10,
        )
