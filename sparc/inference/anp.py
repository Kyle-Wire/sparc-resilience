"""SpatialANP — Attentive Neural Process for spatial few-shot / zero-shot inference.

Architecture
------------
- Deterministic path only (no latent variable z) for simplicity and stability.
- Encoder: MLP over context (x, y) pairs → fixed-size representation.
- Cross-attention: target_x attends over encoded context keys.
- Decoder: MLP over (target_x || aggregated_context) → (mean, log_std).
- Zero-shot: empty context → context summary is a learned prior vector.

Two operating modes
-------------------
**Raw-feature mode** (default, backward-compatible):
    context_x : (K, x_dim)     raw features (coords + satellite bands)
    target_x  : (N, x_dim)

**Trunk-embedding mode** (station-conditioned UHI correction):
    context_x : (K, embed_dim) JEPA trunk embeddings at station locations
    target_x  : (N, embed_dim) JEPA trunk embeddings at all target pixels
    context_y : (K, 1)         UHI anomaly = station_obs_temp - ERA5_background
    Predicts residual correction on top of zero-shot ensemble field.
    Enabled by passing embed_dim > 0 to constructor or via from_trunk_config().

Matérn attention bias (optional)
---------------------------------
When matern_range_m > 0 is passed to forward(), pairwise distances between
context and target coordinates are used to compute a Matérn-3/2 spatial kernel
bias added to attention logits before softmax. This injects physics-derived
spatial correlation decay (from Stage 0 correlograms) into the attention
mechanism without requiring the model to learn it from scratch.

Interface
---------
forward(context_x, context_y, target_x,
        context_coords=None, target_coords=None, matern_range_m=None)
    → (mean, std)
    context_x      : (K, x_dim)  — can be empty (K=0) for zero-shot
    context_y      : (K, 1)
    target_x       : (N, x_dim)
    context_coords : (K, 2) UTM coords — required for Matérn bias
    target_coords  : (N, 2) UTM coords — required for Matérn bias
    matern_range_m : float effective range in metres (from Stage 0)
    mean           : (N, 1)
    std            : (N, 1)  — always positive via softplus
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _matern32_kernel(dist: Tensor, range_m: float) -> Tensor:
    """Matérn-3/2 spatial correlation kernel.

    k(d) = (1 + sqrt(3)*d/r) * exp(-sqrt(3)*d/r)

    Parameters
    ----------
    dist : (K, N) pairwise distances in metres
    range_m : effective range in metres (distance at which correlation ≈ 0.13)

    Returns
    -------
    weights : (K, N) in (0, 1]
    """
    sqrt3 = math.sqrt(3.0)
    scaled = sqrt3 * dist / range_m
    return (1.0 + scaled) * torch.exp(-scaled)


class SpatialANP(nn.Module):
    """Attentive Neural Process for spatial prediction.

    Parameters
    ----------
    x_dim : int
        Dimensionality of the input features (e.g. 2 for lat/lon, or 256 for
        JEPA trunk embeddings in trunk-embedding mode).
    hidden_dim : int
        Width of all hidden layers. Default 64.
    encoder_dim : int
        Size of the encoded context representation per point. Default 64.
    n_heads : int
        Number of attention heads for cross-attention. Default 4.
    trunk_embed_mode : bool
        When True, marks this model as operating in trunk-embedding space.
        Saved in checkpoint so from_checkpoint() can restore the flag.
        Does not change the architecture — purely informational for callers.
    """

    def __init__(
        self,
        x_dim: int,
        hidden_dim: int = 64,
        encoder_dim: int = 64,
        n_heads: int = 4,
        trunk_embed_mode: bool = False,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.encoder_dim = encoder_dim
        self.trunk_embed_mode = trunk_embed_mode

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
        context_coords: Optional[Tensor] = None,  # (K, 2) UTM metres
        target_coords: Optional[Tensor] = None,   # (N, 2) UTM metres
        matern_range_m: Optional[float] = None,
    ) -> tuple[Tensor, Tensor]:
        """Predict mean and std at target locations.

        Parameters
        ----------
        context_x : (K, x_dim) — features at station/context locations
        context_y : (K, 1) — observed UHI anomaly at stations
        target_x  : (N, x_dim) — features at target pixel locations
        context_coords : (K, 2) UTM coordinates in metres (required for Matérn bias)
        target_coords  : (N, 2) UTM coordinates in metres (required for Matérn bias)
        matern_range_m : effective range in metres from Stage 0 correlograms.
            When provided along with *_coords, adds Matérn-3/2 spatial kernel
            as an additive log-weight bias to cross-attention logits.

        Returns
        -------
        mean : (N, 1)
        std  : (N, 1) — strictly positive
        """
        N = target_x.size(0)
        K = context_x.size(0)

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

            # Matérn spatial bias: additive log-weight on attention logits
            # attn_mask shape for MHA: (N, K) — added to raw logits before softmax
            attn_mask: Optional[Tensor] = None
            if (
                matern_range_m is not None
                and matern_range_m > 0
                and context_coords is not None
                and target_coords is not None
            ):
                # Pairwise distances (N, K)
                # target_coords (N,2), context_coords (K,2)
                diff = target_coords.unsqueeze(1) - context_coords.unsqueeze(0)  # (N, K, 2)
                dist = torch.norm(diff, dim=-1)  # (N, K)
                matern_w = _matern32_kernel(dist, matern_range_m)  # (N, K) in (0,1]
                # Add small epsilon before log to avoid -inf at zero distance
                attn_mask = torch.log(matern_w + 1e-8)  # (N, K) — log-weight bias

            attended, _ = self.cross_attn(q, k, v, attn_mask=attn_mask)  # (1, N, encoder_dim)
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

    def save_checkpoint(self, path: str, climate_encoder=None) -> None:  # type: ignore[override]
        """Save model state_dict to *path*.

        Parameters
        ----------
        path :
            Destination file path.
        climate_encoder :
            Optional :class:`~sparc.models.climate_encoder.ClimateZoneEncoder`.
            When supplied the checkpoint is written as a *bundle* dict::

                {"anp": <state_dict>, "climate_encoder": <state_dict>}

            Bundle checkpoints are fully backward-compatible: all existing
            load methods detect the format automatically.
        """
        import torch as _torch
        if climate_encoder is not None:
            payload = {
                "anp": self.state_dict(),
                "climate_encoder": climate_encoder.state_dict(),
                "trunk_embed_mode": self.trunk_embed_mode,
            }
        else:
            payload = self.state_dict()
        _torch.save(payload, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model state_dict from *path*.

        Handles both plain checkpoints (raw ``state_dict``) and bundle
        checkpoints produced by :meth:`save_checkpoint` when a
        ``climate_encoder`` is supplied.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        import os
        import torch as _torch
        if not os.path.exists(path):
            raise FileNotFoundError(f"SpatialANP.load_checkpoint: no file at {path!r}")
        payload = _torch.load(path, weights_only=True)
        state = payload["anp"] if isinstance(payload, dict) and "anp" in payload else payload
        self.load_state_dict(state)

    @classmethod
    def from_checkpoint(cls, path: str) -> "SpatialANP":
        """Restore a SpatialANP from a saved checkpoint.

        Infers architecture dims (x_dim, hidden_dim, encoder_dim) from the
        saved weights so the caller does not need to track them separately.
        The returned model is in eval mode.

        Parameters
        ----------
        path :
            Path to a checkpoint produced by :meth:`save_checkpoint`.

        Returns
        -------
        SpatialANP

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        import os
        import torch as _torch
        if not os.path.exists(path):
            raise FileNotFoundError(f"SpatialANP.from_checkpoint: no file at {path!r}")
        payload = _torch.load(path, map_location="cpu", weights_only=True)
        # Support both plain state_dict and bundle format.
        state = payload["anp"] if isinstance(payload, dict) and "anp" in payload else payload
        # Infer architecture dims from weight shapes:
        #   target_key_encoder.0.weight : (hidden_dim, x_dim)
        #   empty_context_prior          : (1, encoder_dim)
        x_dim = int(state["target_key_encoder.0.weight"].shape[1])
        hidden_dim = int(state["target_key_encoder.0.weight"].shape[0])
        encoder_dim = int(state["empty_context_prior"].shape[1])
        trunk_embed_mode = bool(payload.get("trunk_embed_mode", False)) if isinstance(payload, dict) else False
        model = cls(x_dim=x_dim, hidden_dim=hidden_dim, encoder_dim=encoder_dim,
                    trunk_embed_mode=trunk_embed_mode)
        model.load_state_dict(state)
        model.eval()
        return model

    @classmethod
    def from_trunk_config(
        cls,
        trunk_checkpoint_path: str,
        hidden_dim: int = 128,
        encoder_dim: int = 128,
        n_heads: int = 4,
    ) -> "SpatialANP":
        """Create a SpatialANP configured for trunk-embedding space.

        Reads ``in_dim`` from the JEPA trunk checkpoint and sets ``x_dim``
        accordingly (default 256 hidden → 256 embedding dim).  The returned
        model is in trunk_embed_mode=True and ready for meta-training.

        Parameters
        ----------
        trunk_checkpoint_path :
            Path to ``jepa_pretrained_trunk.pt`` produced by
            ``train_multicity_jepa.py``.
        hidden_dim : int
            Hidden dimension for this ANP (independent of trunk hidden_dim).
            Default 128 — larger than the raw-feature ANP (64) because the
            256-dim embedding space is richer.
        encoder_dim : int
            Encoder output dimension. Default 128.
        n_heads : int
            Number of attention heads. Default 4.

        Returns
        -------
        SpatialANP in trunk-embedding mode, ready for meta-training.
        """
        import torch as _torch
        ckpt = _torch.load(trunk_checkpoint_path, map_location="cpu", weights_only=True)
        # JEPA trunk checkpoint keys: trunk_state, in_dim, hidden_dim, ...
        embed_dim: int = int(ckpt.get("hidden_dim", 256))  # trunk output = hidden_dim
        return cls(
            x_dim=embed_dim,
            hidden_dim=hidden_dim,
            encoder_dim=encoder_dim,
            n_heads=n_heads,
            trunk_embed_mode=True,
        )
