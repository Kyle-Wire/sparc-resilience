"""
Welford Online Scaler for SPARC V3 continual learning.

Incrementally computes feature-wise mean and variance via Welford's
algorithm.  Supports partial_fit(), transform(), inverse_transform(),
save/load, and merging of two scaler states for parallel or sequential
city processing.

This avoids re-normalising the entire dataset when a new city is added
to the training pool.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class WelfordScaler:
    """
    Feature-wise online StandardScaler using Welford's algorithm.

    Usage::

        scaler = WelfordScaler()
        scaler.partial_fit(X_providence)
        X_scaled = scaler.transform(X_providence)
        ...
        scaler.partial_fit(X_boston)          # update stats
        X_scaled = scaler.transform(X_boston) # new stats include both
    """

    n_features: int = 0
    count: int = 0
    mean: np.ndarray = field(default_factory=lambda: np.array([]))
    M2: np.ndarray = field(default_factory=lambda: np.array([]))

    def partial_fit(self, X: np.ndarray) -> "WelfordScaler":
        """
        Update running statistics with new data.

        Parameters
        ----------
        X : (N, D) array of feature values

        Returns
        -------
        self
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        if self.count == 0:
            self.n_features = n_features
            self.mean = np.zeros(n_features, dtype=np.float64)
            self.M2 = np.zeros(n_features, dtype=np.float64)

        if n_features != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {n_features}"
            )

        for row in X:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self.M2 += delta * delta2

        return self

    @property
    def var(self) -> np.ndarray:
        """Population variance for each feature."""
        if self.count < 2:
            return np.ones(self.n_features, dtype=np.float64)
        return self.M2 / self.count

    @property
    def std(self) -> np.ndarray:
        """Standard deviation for each feature (clamped ≥ 1e-8)."""
        return np.maximum(np.sqrt(self.var), 1e-8)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardise X using current mean/std.

        Parameters
        ----------
        X : (N, D) or (D,)

        Returns
        -------
        (N, D) or (D,) standardised array
        """
        if self.count == 0:
            raise RuntimeError("WelfordScaler not yet fitted.")
        return (X - self.mean) / self.std

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Undo standardisation."""
        if self.count == 0:
            raise RuntimeError("WelfordScaler not yet fitted.")
        return X_scaled * self.std + self.mean

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save scaler state to a pickle file."""
        state = {
            "n_features": self.n_features,
            "count": self.count,
            "mean": self.mean,
            "M2": self.M2,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> "WelfordScaler":
        """Load scaler state from a pickle file."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        scaler = cls()
        scaler.n_features = state["n_features"]
        scaler.count = state["count"]
        scaler.mean = state["mean"]
        scaler.M2 = state["M2"]
        return scaler

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    @staticmethod
    def merge(a: "WelfordScaler", b: "WelfordScaler") -> "WelfordScaler":
        """
        Merge two Welford scaler states into one.

        Uses the parallel Welford algorithm to combine statistics from
        two independent data streams.
        """
        if a.count == 0:
            return b
        if b.count == 0:
            return a
        if a.n_features != b.n_features:
            raise ValueError(
                f"Cannot merge scalers with {a.n_features} and "
                f"{b.n_features} features"
            )

        merged = WelfordScaler()
        merged.n_features = a.n_features
        merged.count = a.count + b.count

        delta = b.mean - a.mean
        merged.mean = (a.count * a.mean + b.count * b.mean) / merged.count
        merged.M2 = (
            a.M2 + b.M2
            + delta ** 2 * (a.count * b.count) / merged.count
        )

        return merged

    def __repr__(self) -> str:
        return (
            f"WelfordScaler(n_features={self.n_features}, count={self.count})"
        )
