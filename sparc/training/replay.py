"""
Experience Replay Coreset for SPARC V3 continual learning.

After training on a city, select a small coreset of representative data
points (default 400).  During training on a new city, the replay loss
anchors predictions on the coreset so the model preserves performance on
previously seen distributions.

Selection method: K-medoids clustering on the feature matrix.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CoresetSelector:
    """
    Select representative coreset points via K-medoids.

    Parameters
    ----------
    coreset_size : maximum number of representative points
    random_state : RNG seed for reproducibility
    """

    def __init__(self, coreset_size: int = 400, random_state: int = 42) -> None:
        self.coreset_size = coreset_size
        self.random_state = random_state

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Select coreset from feature matrix X and targets y.

        Parameters
        ----------
        X : (N, D) feature matrix
        y : (N,) target vector
        coords : (N, 2) optional coordinate matrix

        Returns
        -------
        dict with keys 'X', 'y', 'coords' (if provided), 'indices'
        """
        n = X.shape[0]
        k = min(self.coreset_size, n)

        if k >= n:
            # Data is small enough to keep everything
            result = {"X": X.copy(), "y": y.copy(), "indices": np.arange(n)}
            if coords is not None:
                result["coords"] = coords.copy()
            return result

        # K-medoids via greedy facility-location approximation
        rng = np.random.RandomState(self.random_state)
        indices = self._greedy_kmedoids(X, k, rng)

        result = {"X": X[indices].copy(), "y": y[indices].copy(), "indices": indices}
        if coords is not None:
            result["coords"] = coords[indices].copy()

        logger.info("Selected %d coreset points from %d samples", k, n)
        return result

    def _greedy_kmedoids(
        self, X: np.ndarray, k: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """
        Greedy K-medoids via facility-location objective.

        Algorithm:
        1. Start with a random point.
        2. At each step, add the point that maximally reduces the total
           assignment cost (sum of distances to nearest medoid).
        3. Repeat until k medoids selected.

        This is O(N*k) in distance evaluations using cached nearest-medoid
        distances.
        """
        n = X.shape[0]
        # Normalize features for distance computation
        X_norm = X - X.mean(axis=0)
        std = X_norm.std(axis=0)
        std[std < 1e-12] = 1.0
        X_norm /= std

        # Start with random medoid
        medoid_idx = rng.randint(n)
        selected = [medoid_idx]

        # Min distance to any selected medoid for each point
        min_dist = np.full(n, np.inf)
        self._update_min_dist(X_norm, X_norm[medoid_idx], min_dist)

        for _ in range(k - 1):
            # Greedy: pick point with maximum min-distance (facility location)
            remaining_mask = np.ones(n, dtype=bool)
            remaining_mask[selected] = False
            candidates = np.where(remaining_mask)[0]

            # Choose the point farthest from current medoid set
            best_idx = candidates[np.argmax(min_dist[candidates])]
            selected.append(best_idx)
            self._update_min_dist(X_norm, X_norm[best_idx], min_dist)

        return np.array(selected)

    @staticmethod
    def _update_min_dist(
        X: np.ndarray, new_medoid: np.ndarray, min_dist: np.ndarray
    ) -> None:
        """Update minimum distances with a new medoid (in-place)."""
        diff = X - new_medoid[np.newaxis, :]
        dist = np.sqrt((diff ** 2).sum(axis=1))
        np.minimum(min_dist, dist, out=min_dist)


def compute_replay_loss(
    model: nn.Module,
    coreset_X: torch.Tensor,
    coreset_y: torch.Tensor,
    coreset_coords: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute replay loss on the coreset.

    The model should predict coreset targets reasonably well to avoid
    catastrophic forgetting.

    Parameters
    ----------
    model : the current model being trained
    coreset_X : (K, D) feature tensor from past cities
    coreset_y : (K,) target tensor from past cities
    coreset_coords : (K, 2) optional coordinate tensor

    Returns
    -------
    replay_loss : scalar MSE loss on the coreset
    """
    model_device = next(model.parameters()).device
    X = coreset_X.to(model_device)
    y = coreset_y.to(model_device)

    # Build minimal inputs dict
    inputs = {"features": X}
    if coreset_coords is not None:
        inputs["coords"] = coreset_coords.to(model_device)

    # Forward pass (use model's forward signature)
    with torch.set_grad_enabled(True):
        T_pred, _, _ = model(**inputs)
        loss = nn.functional.mse_loss(T_pred.squeeze(), y.squeeze())

    return loss
