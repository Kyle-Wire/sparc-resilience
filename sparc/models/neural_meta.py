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
from sparc.features.geo_positional_encoding import GeoPositionalEncoding


class SpatialGatingHead(nn.Module):
    """Learned gate over the three differentiable surrogate outputs.

    Given the spatial embedding, trunk embedding, and the cross-surrogate
    standard deviation (an epistemic uncertainty proxy), produce a
    soft-attention weight vector of shape ``(N, n_surrogates)`` that
    dynamically down-weights surrogates with high prediction variance
    relative to the local spatial context.

    Parameters
    ----------
    hidden_dim : int
        Trunk / spatial hidden dimension ``H``.
    n_surrogates : int
        Number of surrogate streams (default 3: GWR, GWRF, GGPGAM).
    dropout : float
        Dropout rate applied before the output softmax.
    """

    def __init__(self, hidden_dim: int, n_surrogates: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),   # [h_spatial ‖ h_trunk ‖ std]
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_surrogates),
        )
        # Zero-initialise final layer so gates start at uniform 1/n_surrogates.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        h_spatial: torch.Tensor,   # (N, H)
        h_trunk: torch.Tensor,     # (N, H)
        surrogate_std: torch.Tensor,  # (N, 1)
    ) -> torch.Tensor:
        """Return soft gate weights ``(N, n_surrogates)``."""
        x = torch.cat([h_spatial, h_trunk, surrogate_std], dim=-1)  # (N, 2H+1)
        logits = self.mlp(x)                                          # (N, n_surr)
        return F.softmax(logits, dim=-1)                              # (N, n_surr)


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
    geo_pe_dim : int — geographic positional encoding dim (0 = disabled)
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
        init_bandwidth: float = 1000.0,
        geo_pe_dim: int = 0,
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

        # Optional: Geographic positional encoding (Space2Vec-style)
        # Encodes projected (x, y) coords as multi-scale Fourier features.
        # Ref: Mai et al. (2020) Space2Vec, ICLR 2020.
        self.geo_pe_dim = geo_pe_dim
        self.geo_pe_enc: GeoPositionalEncoding | None = None
        if geo_pe_dim > 0:
            self.geo_pe_enc = GeoPositionalEncoding(geo_pe_dim)
            trunk_fusion_input += geo_pe_dim

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

        # Spatial gating head — soft-weights the n_base_models surrogates
        # using spatial + trunk context + cross-surrogate std.  Produces a
        # weighted blend that is projected back to hidden_dim via blend_proj.
        self.spatial_gating = SpatialGatingHead(
            hidden_dim=hidden_dim,
            n_surrogates=n_base_models,
            dropout=dropout,
        )
        # Projects the scalar gated-blend (weighted surrogate mean, shape N,1)
        # up to hidden_dim so it can be residually added to the base stream.
        self.blend_proj = nn.Linear(1, hidden_dim)
        nn.init.zeros_(self.blend_proj.weight)
        nn.init.zeros_(self.blend_proj.bias)
        # Controls how much the gate blend replaces base_enc output.
        # Zero-initialised → starts as pure base_enc (backward compatible).
        self.gate_residual_weight = nn.Parameter(torch.zeros(1))

        # Stream 3: Sparse spatial attention
        self.spatial_enc = SparseSpatialAttention(
            d_model=d_spatial,
            n_heads=n_heads,
            max_neighbors=max_neighbors,
            dropout=dropout,
            init_bandwidth=init_bandwidth,
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
        surrogate_std: torch.Tensor | None = None,
        knn_dists: torch.Tensor | None = None,
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
        surrogate_std : (N, 1) or None — cross-surrogate std from _forward_surrogates.
            When None, zeros are used and the SpatialGatingHead defaults to
            uniform surrogate weights (fully backward compatible).
        knn_dists    : (N, max_neighbors) float or None — metric distances to KNN.
            Passed to SparseSpatialAttention to seed distance-biased attention
            from Stage 0 correlogram bandwidths.

        Returns
        -------
        T_pred       : (N,) — continuous outcome prediction
        exceedance   : list[(N,)] — P(outcome > thresh) per threshold
        attn_weights : (N, max_neighbors) — spatial attention weights
        """
        # Stream encoding
        h_base_raw = self.base_enc(base_preds)           # (N, H)
        h_phys, w_source = self.physics_enc(physics_feats)  # (N, H), (N, 1)
        self._last_w_source = w_source  # keep gradients for variance penalty

        h_spatial, attn_weights = self.spatial_enc(X_spatial, coords, knn_index, knn_dists=knn_dists)
        h_spatial = self.spatial_proj(h_spatial)       # (N, H)

        h_alpha = self.alpha_emb(alpha)               # (N, H)

        # Trunk fusion: physics + alpha [+ time] [+ geo-PE] → shared representation
        trunk_parts = [h_phys, h_alpha]
        if self.time_embed is not None and time_idx is not None:
            h_time = self.time_embed(time_idx)         # (N, time_embed_dim)
            trunk_parts.append(h_time)
        if self.geo_pe_enc is not None:
            h_geo = self.geo_pe_enc(coords)            # (N, geo_pe_dim)
            trunk_parts.append(h_geo)
        h_trunk = self.trunk_fusion(torch.cat(trunk_parts, dim=-1))  # (N, H)

        # SpatialGatingHead — soft-weight surrogates by epistemic uncertainty.
        # Defaults to zeros → uniform gate → no effect when surrogate_std=None.
        if surrogate_std is None:
            _std = torch.zeros(base_preds.shape[0], 1, device=base_preds.device)
        else:
            _std = surrogate_std  # (N, 1)

        gate_weights = self.spatial_gating(h_spatial, h_trunk, _std)  # (N, n_surr)
        # Weighted blend of raw surrogate predictions: (N,)
        blend = (base_preds * gate_weights).sum(dim=-1, keepdim=True)  # (N, 1)
        # Project blend to hidden_dim and residually mix with base_enc output.
        h_blend = self.blend_proj(blend)                               # (N, H)
        h_base = h_base_raw + self.gate_residual_weight * h_blend      # (N, H)
        # Store gate weights for diagnostics / divergence audit.
        self._last_gate_weights = gate_weights.detach()

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
        coords: torch.Tensor | None = None,
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
        coords : (N, 2) float or None — projected coordinates for geo-PE.
            When None and geo_pe_dim > 0, zeros are used (backward compatible).

        Returns
        -------
        h_trunk : (N, hidden_dim)
        """
        h_phys, _ = self.physics_enc(physics_feats, channel_mask=channel_mask)
        h_alpha = self.alpha_emb(alpha)

        trunk_parts = [h_phys, h_alpha]
        if self.time_embed is not None and time_idx is not None:
            trunk_parts.append(self.time_embed(time_idx))
        if self.geo_pe_enc is not None:
            if coords is not None:
                h_geo = self.geo_pe_enc(coords)
            else:
                h_geo = torch.zeros(
                    physics_feats.shape[0], self.geo_pe_dim,
                    device=physics_feats.device, dtype=physics_feats.dtype,
                )
            trunk_parts.append(h_geo)
        return self.trunk_fusion(torch.cat(trunk_parts, dim=-1))

    # ------------------------------------------------------------------
    def decode(
        self,
        h_trunk: torch.Tensor,
        base_preds: torch.Tensor,
        X_spatial: torch.Tensor,
        coords: torch.Tensor,
        knn_index: torch.Tensor,
        knn_dists: torch.Tensor | None = None,
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
        h_spatial, _ = self.spatial_enc(X_spatial, coords, knn_index, knn_dists=knn_dists)
        h_spatial = self.spatial_proj(h_spatial)

        fused = torch.cat([h_trunk, h_base, h_spatial], dim=-1)
        fused = self.fusion(fused)

        T_pred = self.regression_head(fused).squeeze(-1)
        exceedance = [head(fused).squeeze(-1) for head in self.exceedance_heads]
        return T_pred, exceedance


    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: "NeuralMetaConfig") -> "SPARCMetaLearner":
        """Construct a SPARCMetaLearner from a validated :class:`NeuralMetaConfig`.

        This is the preferred construction path.  It closes the wiring gap
        where ``time_embed_dim`` from the project config dict was silently
        discarded by the intermediate _ArchConfig object.
        """
        return cls(
            n_base_models=cfg.n_base_models,
            n_physics_features=cfg.n_physics_features,
            d_spatial=cfg.d_spatial,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            thresholds=cfg.thresholds,
            n_heads=cfg.n_heads,
            max_neighbors=cfg.max_neighbors,
            siren_omega=cfg.siren_omega,
            time_embed_dim=cfg.time_embed_dim,
            init_bandwidth=cfg.init_bandwidth,
            geo_pe_dim=cfg.geo_pe_dim,
        )

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
        was_training = self.training
        self.train()  # keep dropout active
        predictions = []

        with torch.inference_mode():
            for _ in range(n_samples):
                T_pred, _, _ = self.forward(*args, **kwargs)
                predictions.append(T_pred)

        if not was_training:
            self.eval()  # restore prior eval state
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

    # Canonical set of submodule name-prefixes that belong to the SharedTrunk.
    # ema_trunk.py and v2_neural_training.py import/reference this; keep in sync.
    _TRUNK_KEYS: frozenset[str] = frozenset(
        {"physics_enc", "alpha_emb", "trunk_fusion", "time_embed", "geo_pe_enc"}
    )

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


from sparc.models.neural_meta_config import NeuralMetaConfig  # noqa: F401
