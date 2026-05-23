"""SpatialANP — Attentive Neural Process for spatial few-shot / zero-shot inference.

Architecture
------------
- Deterministic path only (no latent variable z) for simplicity and stability.
- Encoder: MLP over context (x, y) pairs → fixed-size representation.
- Cross-attention: target_x attends over encoded context keys.
- Decoder: MLP over (target_x || aggregated_context) → (mean, log_std).
- Zero-shot: empty context → context summary is a learned prior vector.

Interface
---------
forward(context_x, context_y, target_x) → (mean, std)
    context_x : (K, x_dim)  — can be empty (K=0) for zero-shot
    context_y : (K, 1)
    target_x  : (N, x_dim)
    mean      : (N, 1)
    std       : (N, 1)  — always positive via softplus
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SpatialANP(nn.Module):
    """Attentive Neural Process for spatial prediction.

    Parameters
    ----------
    x_dim : int
        Dimensionality of the input features (e.g. 2 for lat/lon or more).
    hidden_dim : int
        Width of all hidden layers. Default 64.
    encoder_dim : int
        Size of the encoded context representation per point. Default 64.
    n_heads : int
        Number of attention heads for cross-attention. Default 4.
    """

    def __init__(
        self,
        x_dim: int,
        hidden_dim: int = 64,
        encoder_dim: int = 64,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.encoder_dim = encoder_dim

        # Encode each (x, y) context pair → R^encoder_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(x_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoder_dim),
        )

        # Encode target_x → query vectors for cross-attention
        self.target_key_encoder = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoder_dim),
        )

        # Cross-attention: target queries attend over context keys
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=encoder_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        # Learned prior used when context is empty (zero-shot)
        self.empty_context_prior = nn.Parameter(torch.zeros(1, encoder_dim))

        # Decoder: (target_x_enc || aggregated_context) → (mean, log_std)
        self.decoder = nn.Sequential(
            nn.Linear(encoder_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # → [mean, log_std]
        )

    def forward(
        self,
        context_x: Tensor,  # (K, x_dim)
        context_y: Tensor,  # (K, 1)
        target_x: Tensor,   # (N, x_dim)
    ) -> tuple[Tensor, Tensor]:
        """Predict mean and std at target locations.

        Returns
        -------
        mean : (N, 1)
        std  : (N, 1) — strictly positive
        """
        N = target_x.size(0)
        K = context_x.size(0)
        device = target_x.device

        # Encode target locations → query vectors   (N, encoder_dim)
        target_q = self.target_key_encoder(target_x)   # (N, encoder_dim)

        if K == 0:
            # Zero-shot: broadcast learned prior to all target points
            context_agg = self.empty_context_prior.expand(N, -1)  # (N, encoder_dim)
        else:
            # Encode context pairs → key/value vectors   (K, encoder_dim)
            ctx_in = torch.cat([context_x, context_y], dim=-1)  # (K, x_dim+1)
            ctx_enc = self.context_encoder(ctx_in)               # (K, encoder_dim)

            # MultiheadAttention expects (batch, seq, dim) — treat N/K as seq
            q = target_q.unsqueeze(0)    # (1, N, encoder_dim)
            k = ctx_enc.unsqueeze(0)     # (1, K, encoder_dim)
            v = ctx_enc.unsqueeze(0)     # (1, K, encoder_dim)

            attended, _ = self.cross_attn(q, k, v)  # (1, N, encoder_dim)
            context_agg = attended.squeeze(0)        # (N, encoder_dim)

        # Decode
        dec_in = torch.cat([target_q, context_agg], dim=-1)  # (N, 2*encoder_dim)
        out = self.decoder(dec_in)                            # (N, 2)
        mean = out[:, :1]                                     # (N, 1)
        log_std = out[:, 1:]                                  # (N, 1)
        std = F.softplus(log_std) + 1e-6                      # (N, 1) — strictly positive

        return mean, std

    # ------------------------------------------------------------------
    # SpatialTrunk protocol implementation
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save model state_dict to *path*."""
        import torch as _torch
        _torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        """Load model state_dict from *path*.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        import os
        import torch as _torch
        if not os.path.exists(path):
            raise FileNotFoundError(f"SpatialANP.load_checkpoint: no file at {path!r}")
        state = _torch.load(path, weights_only=True)
        self.load_state_dict(state)
