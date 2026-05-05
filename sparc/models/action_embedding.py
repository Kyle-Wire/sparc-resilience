"""Action embedding for the action-conditioned JEPA latent predictor.

V-JEPA 2-AC ("LeWorldModel") conditions the predictor on a discrete
action plus continuous magnitude/direction/time delta.  In SPARC we map
that to:

    action = concat(
        treatment_onehot,   # which physics feature was perturbed (+ "none")
        log1p(|Δx|),        # signed magnitude (compressed)
        sign(Δx),           # direction
        Δt_scaled,          # 0 for scenario, normalised time for temporal
    )                       # → ActionEmbedding → R^embed_dim

The "none"/unknown bucket lets the predictor be invoked with no
intervention (Δt > 0 only) and supports treatment vocabulary growth
across cities.

This module is pure; it has no global state and is config-flag
isolated from the rest of SPARC.  Used by ``LatentPredictor`` when
``action_dim > 0``; instantiated by the training loop and inference
helpers (``sparc/inference/latent_rollout.py``).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ActionEmbedding(nn.Module):
    """Embed (treatment, Δx, Δt) tuples into a fixed-dim action vector.

    Parameters
    ----------
    treatment_vocab : sequence of str
        Names of treatable physics features.  An implicit "__none__"
        slot is appended for the no-intervention case (used when only
        Δt > 0 conditioning is desired).
    embed_dim : int
        Output dimensionality of the action vector.
    treatment_embed_dim : int
        Hidden width of the treatment lookup table.  Smaller than
        ``embed_dim`` to keep the parameter count modest.
    """

    NONE_TOKEN = "__none__"

    def __init__(
        self,
        treatment_vocab: Sequence[str],
        embed_dim: int = 64,
        treatment_embed_dim: int = 16,
    ) -> None:
        super().__init__()

        # Append the "no treatment" sentinel token if not already present.
        vocab = list(treatment_vocab)
        if self.NONE_TOKEN not in vocab:
            vocab.append(self.NONE_TOKEN)
        self.treatment_vocab: list[str] = vocab
        self._token_to_idx: dict[str, int] = {tok: i for i, tok in enumerate(vocab)}
        self.none_idx: int = self._token_to_idx[self.NONE_TOKEN]

        self.embed_dim = int(embed_dim)
        self.treatment_embed_dim = int(treatment_embed_dim)

        # Treatment lookup table.
        self.treatment_lut = nn.Embedding(
            num_embeddings=len(vocab), embedding_dim=treatment_embed_dim,
        )

        # Continuous channel: [log1p|Δx|, sign(Δx), Δt_scaled] = 3 dims.
        # Project to embed_dim alongside the treatment embedding via a
        # small MLP with LayerNorm for stability.
        self.proj = nn.Sequential(
            nn.LayerNorm(treatment_embed_dim + 3),
            nn.Linear(treatment_embed_dim + 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    # --------------------------------------------------------------
    # Public helpers
    # --------------------------------------------------------------
    def token_index(self, treatment: str | None) -> int:
        """Map a treatment name (or None) to its vocabulary index.

        Unknown treatments fall back to the "__none__" slot so calls
        from new cities never raise — designed for vocabulary growth.
        """
        if treatment is None:
            return self.none_idx
        return self._token_to_idx.get(treatment, self.none_idx)

    def encode_actions(
        self,
        treatment_idx: torch.Tensor,
        delta_x: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of action tuples.

        Parameters
        ----------
        treatment_idx : (B,) long
            Vocabulary indices into ``treatment_vocab``.
        delta_x : (B,) float
            Signed magnitude of the intervention (in feature units).
        delta_t : (B,) float
            Time delta (in normalised units; 0.0 for pure scenario).

        Returns
        -------
        action_embed : (B, embed_dim) float
        """
        if treatment_idx.dim() != 1:
            raise ValueError(
                f"treatment_idx must be 1-D, got shape {tuple(treatment_idx.shape)}"
            )
        if delta_x.shape != treatment_idx.shape or delta_t.shape != treatment_idx.shape:
            raise ValueError(
                "delta_x and delta_t must share shape with treatment_idx"
            )

        treat_emb = self.treatment_lut(treatment_idx)  # (B, treatment_embed_dim)
        cont = torch.stack(
            [
                torch.log1p(delta_x.abs()),
                torch.sign(delta_x),
                delta_t,
            ],
            dim=-1,
        )  # (B, 3)
        x = torch.cat([treat_emb, cont], dim=-1)
        return self.proj(x)

    def forward(
        self,
        treatment_idx: torch.Tensor,
        delta_x: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode_actions(treatment_idx, delta_x, delta_t)
