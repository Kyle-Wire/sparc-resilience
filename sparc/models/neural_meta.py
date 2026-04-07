"""
Neural meta-learner for SPARC V2.

Four-stream fusion network:
  Stream 1 (Base)    — encodes differentiable surrogate predictions
  Stream 2 (Physics) — SIREN encoder for physics features
  Stream 3 (Spatial) — sparse spatial attention over coordinates
  Stream 4 (Alpha)   — process rate embedding

Dual output head:
  - Regression: continuous outcome prediction
  - Exceedance: P(outcome > threshold) per threshold

Interfaces:
  - ``predict_with_uncertainty`` — MC Dropout inference
  - ``predict_for_nuts``         — no-gradient prediction for NUTS likelihood
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sparc.models.spatial_attention import SIRENLayer, SparseSpatialAttention


class SPARCMetaLearner(nn.Module):
    """
    V2 end-to-end differentiable meta-learner.

    Parameters
    ----------
    n_base_models : int — number of differentiable surrogate outputs
    n_physics_features : int — dimension of physics input
    d_spatial : int — dimension of sinusoidal spatial encoding
    hidden_dim : int — hidden dimension for all streams
    dropout : float
    thresholds : list[float] — exceedance probability thresholds
    n_heads : int — attention heads for sparse spatial attention
    max_neighbors : int — KNN neighborhood size
    siren_omega : float — SIREN frequency parameter
    """

    def __init__(
        self,
        n_base_models: int,
        n_physics_features: int,
        d_spatial: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        thresholds: list[float] | None = None,
        n_heads: int = 4,
        max_neighbors: int = 128,
        siren_omega: float = 30.0,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.thresholds = thresholds or [0.25, 0.50, 0.75]

        # -----------------------------------
        # Stream 1: Base model encoder
        # -----------------------------------
        self.base_enc = nn.Sequential(
            nn.Linear(n_base_models, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # -----------------------------------
        # Stream 2: SIREN physics encoder
        # -----------------------------------
        self.physics_enc = nn.Sequential(
            SIRENLayer(n_physics_features, hidden_dim, omega=siren_omega, is_first=True),
            SIRENLayer(hidden_dim, hidden_dim, omega=siren_omega),
            SIRENLayer(hidden_dim, hidden_dim, omega=siren_omega),
        )

        # -----------------------------------
        # Stream 3: Sparse spatial attention
        # -----------------------------------
        self.spatial_enc = SparseSpatialAttention(
            d_model=d_spatial,
            n_heads=n_heads,
            max_neighbors=max_neighbors,
            dropout=dropout,
        )
        # Project spatial stream to hidden_dim
        self.spatial_proj = nn.Linear(d_spatial, hidden_dim)

        # -----------------------------------
        # Stream 4: Process rate embedding
        # -----------------------------------
        self.alpha_emb = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # -----------------------------------
        # Fusion
        # -----------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # -----------------------------------
        # Regression head
        # -----------------------------------
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # -----------------------------------
        # Exceedance heads (one per threshold)
        # -----------------------------------
        self.exceedance_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid(),
            )
            for _ in self.thresholds
        ])

    # ------------------------------------------------------------------
    def forward(
        self,
        base_preds: torch.Tensor,
        physics_feats: torch.Tensor,
        X_spatial: torch.Tensor,
        coords: torch.Tensor,
        knn_index: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """
        Parameters
        ----------
        base_preds   : (N, n_base_models) — surrogate predictions
        physics_feats: (N, n_physics_features) — physics inputs
        X_spatial    : (N, d_spatial) — sinusoidal spatial encoding
        coords       : (N, 2) — projected coordinates
        knn_index    : (N, max_neighbors) — KNN indices
        alpha        : (N, 1) — process rate from ProcessRateNet

        Returns
        -------
        T_pred       : (N,) — continuous outcome prediction
        exceedance   : list[(N,)] — P(outcome > thresh) per threshold
        attn_weights : (N, max_neighbors) — spatial attention weights
        """
        # Stream encoding
        h_base = self.base_enc(base_preds)           # (N, H)
        h_phys = self.physics_enc(physics_feats)      # (N, H)

        h_spatial, attn_weights = self.spatial_enc(X_spatial, coords, knn_index)
        h_spatial = self.spatial_proj(h_spatial)       # (N, H)

        h_alpha = self.alpha_emb(alpha)               # (N, H)

        # Fusion
        fused = torch.cat([h_base, h_phys, h_spatial, h_alpha], dim=-1)  # (N, 4H)
        fused = self.fusion(fused)                    # (N, H)

        # Dual output
        T_pred = self.regression_head(fused).squeeze(-1)  # (N,)
        exceedance = [head(fused).squeeze(-1) for head in self.exceedance_heads]

        return T_pred, exceedance, attn_weights

    # ------------------------------------------------------------------
    def predict_with_uncertainty(
        self,
        *args,
        n_samples: int = 100,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        MC Dropout inference — keep dropout active for uncertainty.

        Returns
        -------
        mean : (N,) — posterior mean prediction
        std  : (N,) — epistemic uncertainty (MC std)
        """
        self.train()  # keep dropout active
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                T_pred, _, _ = self.forward(*args, **kwargs)
                predictions.append(T_pred)

        predictions = torch.stack(predictions, dim=0)  # (S, N)
        return predictions.mean(dim=0), predictions.std(dim=0)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_for_nuts(
        self,
        X_dict: dict[str, torch.Tensor],
    ) -> np.ndarray:
        """
        No-gradient prediction interface for NUTS likelihood evaluation.

        Parameters
        ----------
        X_dict : dict with keys matching forward() signature

        Returns
        -------
        predictions : (N,) ndarray
        """
        self.eval()
        T_pred, _, _ = self.forward(**X_dict)
        return T_pred.cpu().numpy()
