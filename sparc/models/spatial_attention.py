"""
SIREN layers and sparse spatial attention for SPARC V2.

Provides:
  - ``SIRENLayer``  — sinusoidal activation layer optimal for PDE solutions,
    with SIREN-specific initialization and pre-activation layer normalization.
  - ``SparseSpatialAttention`` — multi-head attention over a KNN spatial
    neighborhood.  O(N × max_neighbors) rather than O(N²).

Attention weights are interpretable as a spatial influence surface — each
point's weight distribution shows which other points influenced its
prediction and by how much.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SIREN Layer
# ---------------------------------------------------------------------------

class SIRENLayer(nn.Module):
    """
    Sinusoidal activation layer with SIREN-specific initialization.

    Gradient management:
      - LayerNorm before the sine prevents gradient accumulation.
      - SIREN init keeps pre-activations in a stable range.

    Parameters
    ----------
    in_dim, out_dim : int
    omega : float — frequency multiplier for ``sin(ω · Wx)``.
    is_first : bool — use wider init for the first layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        omega: float = 30.0,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.omega = omega
        self.layer_norm = nn.LayerNorm(in_dim)

        # SIREN-specific initialization
        if is_first:
            nn.init.uniform_(self.linear.weight, -1 / in_dim, 1 / in_dim)
        else:
            bound = math.sqrt(6 / in_dim) / omega
            nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        return torch.sin(self.omega * self.linear(x))


# ---------------------------------------------------------------------------
# Sparse Spatial Attention
# ---------------------------------------------------------------------------

class SparseSpatialAttention(nn.Module):
    """
    Multi-head attention restricted to a KNN spatial neighborhood.

    Parameters
    ----------
    d_model : int — feature dimensionality
    n_heads : int — number of attention heads
    max_neighbors : int — KNN neighborhood size
    dropout : float — attention dropout
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        max_neighbors: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.max_neighbors = max_neighbors

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------
    @staticmethod
    def build_knn_index(
        coords: np.ndarray, k: int = 128
    ) -> np.ndarray:
        """
        Build a KNN index using ``scipy.spatial.cKDTree``.

        Parameters
        ----------
        coords : (N, 2) array — projected coordinates
        k : int — number of neighbors (excluding self)

        Returns
        -------
        indices : (N, k) int array — neighbor indices (self excluded)
        """
        from scipy.spatial import cKDTree

        tree = cKDTree(coords)
        # k+1 because query returns the point itself
        _, indices = tree.query(coords, k=k + 1, workers=-1)
        return indices[:, 1:]  # exclude self

    # ------------------------------------------------------------------
    def forward(
        self,
        X: torch.Tensor,
        coords: torch.Tensor,
        knn_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        X : (N, d_model) — input features
        coords : (N, 2) — coordinates (used for potential distance weighting)
        knn_index : (N, max_neighbors) long — KNN indices

        Returns
        -------
        attended : (N, d_model) — attention output (residual + norm applied)
        weights  : (N, max_neighbors) — attention weights (averaged over heads)
        """
        N = X.shape[0]
        k = knn_index.shape[1]

        # Queries from each point, keys/values from its neighbors
        Q = self.W_q(X)                         # (N, d_model)
        K_all = self.W_k(X)                     # (N, d_model)
        V_all = self.W_v(X)                     # (N, d_model)

        # FIX: Mask invalid (-1) neighbor indices from batch-local remapping.
        # PyTorch treats -1 as "last element", silently corrupting attention.
        valid_mask = (knn_index >= 0) & (knn_index < N)  # (N, k) bool
        safe_idx = knn_index.clamp(min=0)                # replace -1 with 0 (masked out later)

        # Gather neighbor keys and values — O(N*k) memory, not O(N²)
        flat_idx = safe_idx.reshape(-1)          # (N*k,)
        K = K_all[flat_idx].reshape(N, k, self.d_model)  # (N, k, d_model)
        V = V_all[flat_idx].reshape(N, k, self.d_model)  # (N, k, d_model)

        # Reshape for multi-head attention
        Q = Q.view(N, self.n_heads, self.d_k).unsqueeze(2)        # (N, heads, 1, d_k)
        K = K.view(N, k, self.n_heads, self.d_k).permute(0, 2, 1, 3)  # (N, heads, k, d_k)
        V = V.view(N, k, self.n_heads, self.d_k).permute(0, 2, 1, 3)  # (N, heads, k, d_k)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_k)
        scores = (Q @ K.transpose(-2, -1)) / scale  # (N, heads, 1, k)

        # Mask invalid neighbor positions to -inf before softmax
        inv_mask = (~valid_mask).unsqueeze(1).unsqueeze(1)   # (N, 1, 1, k)
        scores = scores.masked_fill(inv_mask, -1e9)

        weights = F.softmax(scores, dim=-1)          # (N, heads, 1, k)
        weights = self.attn_dropout(weights)

        # Weighted value aggregation
        out = (weights @ V).squeeze(2)               # (N, heads, d_k)
        out = out.reshape(N, self.d_model)            # (N, d_model)
        out = self.W_o(out)                           # (N, d_model)

        # Residual connection + layer norm
        attended = self.output_norm(X + out)

        # Average attention weights across heads for interpretability
        avg_weights = weights.mean(dim=1).squeeze(1)  # (N, k)

        return attended, avg_weights

    # ------------------------------------------------------------------
    @staticmethod
    @torch.no_grad()
    def attention_influence_map(
        weights: torch.Tensor,
        knn_index: torch.Tensor,
        focus_point_idx: int,
        N: int,
    ) -> np.ndarray:
        """
        Return the spatial influence surface for a single focus point.

        Parameters
        ----------
        weights : (N, k) — attention weights from forward()
        knn_index : (N, k) — KNN indices
        focus_point_idx : int
        N : int — total number of points

        Returns
        -------
        influence : (N,) ndarray — influence weight of every point on the focus
        """
        influence = np.zeros(N)
        point_weights = weights[focus_point_idx].cpu().numpy()
        point_neighbors = knn_index[focus_point_idx].cpu().numpy()

        for w, idx in zip(point_weights, point_neighbors):
            if 0 <= idx < N:
                influence[idx] = w

        return influence
