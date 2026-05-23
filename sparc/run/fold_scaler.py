"""FoldScalerContext — unified StandardScaler lifecycle for fold training.

Five separate ``StandardScaler`` instances in ``enhanced_spatial_cv.py``
previously had independent fit/save/load lifecycles.  This module
consolidates all of them into a single context object so:

- All scalers are fitted through one interface.
- All scalers are persisted atomically to one directory.
- The inference path rehydrates everything with one ``FoldScalerContext.load(dir)`` call.

Usage
-----
Training::

    ctx = FoldScalerContext(artifact_dir)
    ctx.fit_features(X_meta_features)
    ctx.fit_coords(coord_block)
    ctx.fit_laplacian(X_pca)
    ctx.save()

    X_scaled = ctx.transform_features(X_new)
    C_scaled = ctx.transform_coords(C_new)

Inference::

    ctx = FoldScalerContext.load(artifact_dir)
    X_scaled = ctx.transform_features(X_inference)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler


# Filenames within the artifact directory
_FEATURES_FILE = "fold_scaler_features.pkl"
_COORDS_FILE = "fold_scaler_coords.pkl"
_LAPLACIAN_FILE = "fold_scaler_laplacian.pkl"


class FoldScalerContext:
    """Owns three ``StandardScaler`` instances with unified save/load.

    Parameters
    ----------
    artifact_dir:
        Directory where scaler pickle files will be saved/loaded.

    Methods
    -------
    fit_features / transform_features
        Scaler for the meta-learner feature matrix.
    fit_coords / transform_coords
        Scaler for projected (X, Y) coordinates.
    fit_laplacian / transform_laplacian
        Scaler for Laplacian eigenmap embeddings.
    save()
        Persist all fitted scalers to *artifact_dir*.
    FoldScalerContext.load(artifact_dir)
        Restore from saved files.
    """

    def __init__(self, artifact_dir: "Path | str") -> None:
        self._dir = Path(artifact_dir)
        self._features: Optional[StandardScaler] = None
        self._coords: Optional[StandardScaler] = None
        self._laplacian: Optional[StandardScaler] = None

    # ------------------------------------------------------------------
    # Features scaler
    # ------------------------------------------------------------------

    def fit_features(self, X: np.ndarray) -> "FoldScalerContext":
        """Fit the feature scaler on *X* and return self for chaining."""
        self._features = StandardScaler()
        self._features.fit(X)
        return self

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Transform *X* using the fitted feature scaler.

        Raises
        ------
        RuntimeError
            If the features scaler has not been fitted yet.
        """
        if self._features is None:
            raise RuntimeError(
                "FoldScalerContext: features scaler has not been fitted. "
                "Call fit_features() before transform_features()."
            )
        return self._features.transform(X)

    # ------------------------------------------------------------------
    # Coordinates scaler
    # ------------------------------------------------------------------

    def fit_coords(self, C: np.ndarray) -> "FoldScalerContext":
        """Fit the coordinate scaler on *C* (projected X, Y) and return self."""
        self._coords = StandardScaler()
        self._coords.fit(C)
        return self

    def transform_coords(self, C: np.ndarray) -> np.ndarray:
        """Transform coordinate array *C*.

        Raises
        ------
        RuntimeError
            If the coords scaler has not been fitted yet.
        """
        if self._coords is None:
            raise RuntimeError(
                "FoldScalerContext: coords scaler has not been fitted. "
                "Call fit_coords() before transform_coords()."
            )
        return self._coords.transform(C)

    # ------------------------------------------------------------------
    # Laplacian eigenmaps scaler
    # ------------------------------------------------------------------

    def fit_laplacian(self, L: np.ndarray) -> "FoldScalerContext":
        """Fit the Laplacian eigenmap scaler on *L* and return self."""
        self._laplacian = StandardScaler()
        self._laplacian.fit(L)
        return self

    def transform_laplacian(self, L: np.ndarray) -> np.ndarray:
        """Transform Laplacian eigenvector matrix *L*.

        Raises
        ------
        RuntimeError
            If the laplacian scaler has not been fitted yet.
        """
        if self._laplacian is None:
            raise RuntimeError(
                "FoldScalerContext: laplacian scaler has not been fitted. "
                "Call fit_laplacian() before transform_laplacian()."
            )
        return self._laplacian.transform(L)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Pickle all fitted scalers into *artifact_dir*.

        Only scalers that have been fitted are written; unfit slots are
        skipped so a partially-configured context can be saved and later
        reloaded.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        if self._features is not None:
            self._pickle(self._features, self._dir / _FEATURES_FILE)
        if self._coords is not None:
            self._pickle(self._coords, self._dir / _COORDS_FILE)
        if self._laplacian is not None:
            self._pickle(self._laplacian, self._dir / _LAPLACIAN_FILE)

    @classmethod
    def load(cls, artifact_dir: "Path | str") -> "FoldScalerContext":
        """Restore a ``FoldScalerContext`` from *artifact_dir*.

        The features scaler is mandatory (raises ``FileNotFoundError`` if
        absent).  Coords and laplacian scalers are optional; their
        ``transform_*`` methods will raise ``RuntimeError`` if they weren't
        saved.

        Parameters
        ----------
        artifact_dir:
            Directory previously written by :meth:`save`.

        Raises
        ------
        FileNotFoundError
            If *artifact_dir* does not exist, or the features scaler file
            is missing (it is the minimal required scaler).
        """
        d = Path(artifact_dir)
        if not d.exists():
            raise FileNotFoundError(
                f"FoldScalerContext.load: artifact directory does not exist: {d}"
            )
        features_path = d / _FEATURES_FILE
        if not features_path.exists():
            raise FileNotFoundError(
                f"FoldScalerContext.load: features scaler not found at {features_path}. "
                "Was fit_features() called before save()?"
            )

        ctx = cls(d)
        ctx._features = cls._unpickle(features_path)

        coords_path = d / _COORDS_FILE
        if coords_path.exists():
            ctx._coords = cls._unpickle(coords_path)

        laplacian_path = d / _LAPLACIAN_FILE
        if laplacian_path.exists():
            ctx._laplacian = cls._unpickle(laplacian_path)

        return ctx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pickle(obj, path: Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _unpickle(path: Path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
