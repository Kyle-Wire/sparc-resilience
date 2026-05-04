"""
Neural meta-learner for SPARC V2.

SharedTrunk + CityHead architecture for transfer learning:

  SharedTrunk (transfers across cities):
    - physics_enc (SIREN) — PDE-informed physics encoder
    - alpha_emb            — process rate embedding
    - trunk_fusion         — fuses physics + alpha into shared representation

  CityHead (city-specific, retrained per deployment):
    - base_enc     — encodes differentiable surrogate predictions
    - spatial_enc  — sparse spatial attention over coordinates
    - fusion       — combines trunk output with city-specific streams
    - regression_head   — continuous outcome prediction
    - exceedance_heads  — P(outcome > threshold) per threshold

Interfaces:
  - ``predict_with_uncertainty`` — MC Dropout inference
  - ``predict_for_nuts``         — no-gradient prediction for NUTS likelihood
  - ``save_trunk`` / ``load_trunk`` — persist / restore shared trunk weights
  - ``freeze_trunk`` / ``unfreeze_trunk`` — toggle gradient flow for transfer
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sparc.models.spatial_attention import SIRENLayer, SparseSpatialAttention
from sparc.models.pde_encoder import PDEInformedPhysicsEncoder


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
        time_embed_dim: int = 0,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.thresholds = thresholds or [0.25, 0.50, 0.75]
        self.time_embed_dim = time_embed_dim

        # =============================================================
        # SharedTrunk — physics encoder + process rate (transfers)
        # =============================================================

        # Stream 2: PDE-Informed physics encoder
        self.physics_enc = PDEInformedPhysicsEncoder(
            n_physics_features=n_physics_features,
            out_dim=hidden_dim,
            omega=siren_omega,
        )
        # Last w_source from forward pass (for diagnostics)
        self._last_w_source: torch.Tensor | None = None

        # Stream 4: Process rate embedding
        self.alpha_emb = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Optional: Time embedding for multi-snapshot temporal data
        # morning=0, midday=1, night=2
        self.time_embed: nn.Embedding | None = None
        trunk_fusion_input = hidden_dim * 2
        if time_embed_dim > 0:
            self.time_embed = nn.Embedding(3, time_embed_dim)
            trunk_fusion_input += time_embed_dim

        # Trunk fusion: physics + alpha [+ time] → shared representation
        self.trunk_fusion = nn.Sequential(
            nn.Linear(trunk_fusion_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # =============================================================
        # CityHead — city-specific streams + output heads
        # =============================================================

        # Stream 1: Base model encoder
        self.base_enc = nn.Sequential(
            nn.Linear(n_base_models, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Stream 3: Sparse spatial attention
        self.spatial_enc = SparseSpatialAttention(
            d_model=d_spatial,
            n_heads=n_heads,
            max_neighbors=max_neighbors,
            dropout=dropout,
        )
        # Project spatial stream to hidden_dim
        self.spatial_proj = nn.Linear(d_spatial, hidden_dim)

        # City fusion: trunk + base + spatial
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Exceedance heads (one per threshold)
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
        time_idx: torch.Tensor | None = None,
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
        time_idx     : (N,) int or None — snapshot index (0=morning, 1=midday, 2=night)

        Returns
        -------
        T_pred       : (N,) — continuous outcome prediction
        exceedance   : list[(N,)] — P(outcome > thresh) per threshold
        attn_weights : (N, max_neighbors) — spatial attention weights
        """
        # Stream encoding
        h_base = self.base_enc(base_preds)           # (N, H)
        h_phys, w_source = self.physics_enc(physics_feats)  # (N, H), (N, 1)
        self._last_w_source = w_source  # keep gradients for variance penalty

        h_spatial, attn_weights = self.spatial_enc(X_spatial, coords, knn_index)
        h_spatial = self.spatial_proj(h_spatial)       # (N, H)

        h_alpha = self.alpha_emb(alpha)               # (N, H)

        # Trunk fusion: physics + alpha [+ time] → shared representation
        trunk_parts = [h_phys, h_alpha]
        if self.time_embed is not None and time_idx is not None:
            h_time = self.time_embed(time_idx)         # (N, time_embed_dim)
            trunk_parts.append(h_time)
        h_trunk = self.trunk_fusion(torch.cat(trunk_parts, dim=-1))  # (N, H)

        # City fusion: trunk + base + spatial
        fused = torch.cat([h_trunk, h_base, h_spatial], dim=-1)  # (N, 3H)
        fused = self.fusion(fused)                    # (N, H)

        # Dual output
        T_pred = self.regression_head(fused).squeeze(-1)  # (N,)
        exceedance = [head(fused).squeeze(-1) for head in self.exceedance_heads]

        return T_pred, exceedance, attn_weights

    # ------------------------------------------------------------------
    def encode(
        self,
        physics_feats: torch.Tensor,
        alpha: torch.Tensor,
        time_idx: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the SharedTrunk embedding ``h_trunk`` only.

        Used by JEPA-style training (and downstream latent-space
        scenario rollouts) to obtain the transferable embedding without
        running the city-specific heads.  Mirrors the trunk fusion in
        ``forward`` exactly so ``encode`` and ``forward`` share weights.

        Parameters
        ----------
        physics_feats : (N, n_physics_features)
        alpha : (N, 1) — process rate
        time_idx : (N,) long or None — snapshot index for time embedding
        channel_mask : (n_physics_features,) or None
            Optional JEPA context mask (see ``PDEInformedPhysicsEncoder``).

        Returns
        -------
        h_trunk : (N, hidden_dim)
        """
        h_phys, _ = self.physics_enc(physics_feats, channel_mask=channel_mask)
        h_alpha = self.alpha_emb(alpha)

        trunk_parts = [h_phys, h_alpha]
        if self.time_embed is not None and time_idx is not None:
            trunk_parts.append(self.time_embed(time_idx))
        return self.trunk_fusion(torch.cat(trunk_parts, dim=-1))

    # ------------------------------------------------------------------
    def decode(
        self,
        h_trunk: torch.Tensor,
        base_preds: torch.Tensor,
        X_spatial: torch.Tensor,
        coords: torch.Tensor,
        knn_index: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Decode a (potentially predictor-rolled-out) trunk embedding back
        to the continuous outcome via the existing CityHead fusion +
        regression / exceedance heads.

        This is the JEPA Phase 2 ``latent_rollout`` decode step:

            h_state  = model.encode(physics, alpha)
            h_pred   = predictor(h_state, action_embed)
            T_pred   = model.decode(h_pred, base_preds, X_spatial, coords, knn)

        ``base_preds`` and the spatial inputs are taken from the
        unperturbed observation — only the trunk embedding has been
        rolled forward in latent space.

        Returns
        -------
        T_pred       : (N,)
        exceedance   : list[(N,)] — one tensor per threshold
        """
        h_base = self.base_enc(base_preds)
        h_spatial, _ = self.spatial_enc(X_spatial, coords, knn_index)
        h_spatial = self.spatial_proj(h_spatial)

        fused = torch.cat([h_trunk, h_base, h_spatial], dim=-1)
        fused = self.fusion(fused)

        T_pred = self.regression_head(fused).squeeze(-1)
        exceedance = [head(fused).squeeze(-1) for head in self.exceedance_heads]
        return T_pred, exceedance


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

    # ==================================================================
    # Trunk management (transfer learning)
    # ==================================================================

    _TRUNK_KEYS = {"physics_enc", "alpha_emb", "trunk_fusion", "time_embed"}

    def save_trunk(self, path: str | Path) -> None:
        """Save SharedTrunk weights (physics_enc + alpha_emb + trunk_fusion)."""
        trunk_state = {
            k: v for k, v in self.state_dict().items()
            if any(k.startswith(prefix) for prefix in self._TRUNK_KEYS)
        }
        torch.save(trunk_state, path)

    def load_trunk(self, path: str | Path, strict: bool = True) -> None:
        """Load SharedTrunk weights; CityHead keeps current (or random) weights."""
        trunk_state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(trunk_state, strict=False)

    def freeze_trunk(self) -> None:
        """Freeze SharedTrunk parameters — only CityHead trains."""
        for name, param in self.named_parameters():
            if any(name.startswith(prefix) for prefix in self._TRUNK_KEYS):
                param.requires_grad = False

    def unfreeze_trunk(self) -> None:
        """Unfreeze SharedTrunk parameters for joint fine-tuning."""
        for name, param in self.named_parameters():
            if any(name.startswith(prefix) for prefix in self._TRUNK_KEYS):
                param.requires_grad = True
