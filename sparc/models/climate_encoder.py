"""
Köppen Climate Zone Encoder for SPARC V4.

Embeds the city's Köppen-Geiger climate classification into a learned
vector that conditions the shared trunk.  This allows the model to
adapt its physics priors based on the general climate regime.

V4 will use this alongside satellite features for zero-shot inference.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

# Standard 30 top-level Köppen zones (covers > 99% of populated areas)
KOPPEN_ZONES: list[str] = [
    "Af", "Am", "Aw", "As",                    # Tropical
    "BWh", "BWk", "BSh", "BSk",                # Arid
    "Csa", "Csb", "Csc",                       # Temperate, dry summer
    "Cwa", "Cwb", "Cwc",                       # Temperate, dry winter
    "Cfa", "Cfb", "Cfc",                       # Temperate, no dry season
    "Dsa", "Dsb", "Dsc", "Dsd",                # Continental, dry summer
    "Dwa", "Dwb", "Dwc", "Dwd",                # Continental, dry winter
    "Dfa", "Dfb", "Dfc", "Dfd",                # Continental, no dry season
    "ET", "EF",                                  # Polar
]

KOPPEN_TO_IDX: dict[str, int] = {z: i for i, z in enumerate(KOPPEN_ZONES)}


class ClimateZoneEncoder(nn.Module):
    """
    Learned embedding for Köppen climate zones.

    Parameters
    ----------
    embed_dim : size of the climate embedding vector (default: 8)
    n_zones : number of zone categories (default: len(KOPPEN_ZONES))
    """

    def __init__(self, embed_dim: int = 8, n_zones: int | None = None) -> None:
        super().__init__()
        if n_zones is None:
            n_zones = len(KOPPEN_ZONES)
        self.embed_dim = embed_dim
        self.n_zones = n_zones
        self.embedding = nn.Embedding(n_zones + 1, embed_dim, padding_idx=n_zones)

    def forward(self, zone_idx: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        zone_idx : (B,) integer tensor of zone indices
            Use KOPPEN_TO_IDX to map zone strings to indices.
            Unknown zones should use n_zones (padding idx).

        Returns
        -------
        (B, embed_dim) climate embedding vectors
        """
        return self.embedding(zone_idx)

    @staticmethod
    def zone_to_index(zone: str) -> int:
        """Convert a Köppen zone string to an embedding index."""
        return KOPPEN_TO_IDX.get(zone, len(KOPPEN_ZONES))

    @classmethod
    def from_checkpoint(cls, path: str) -> "ClimateZoneEncoder | None":
        """Load a :class:`ClimateZoneEncoder` from a bundle checkpoint.

        Parameters
        ----------
        path :
            Path to a bundle checkpoint produced by
            :meth:`~sparc.inference.anp.SpatialANP.save_checkpoint` with a
            ``climate_encoder`` argument.

        Returns
        -------
        ClimateZoneEncoder or None
            The restored encoder, or ``None`` when the checkpoint does not
            contain encoder weights (plain ANP checkpoint).
        """
        import torch as _torch
        payload = _torch.load(path, map_location="cpu", weights_only=True)
        if not (isinstance(payload, dict) and "climate_encoder" in payload):
            return None
        # Infer embed_dim and n_zones from saved weight shape (n_zones+1, embed_dim)
        weight = payload["climate_encoder"]["embedding.weight"]
        n_zones_plus_one, embed_dim = weight.shape
        encoder = cls(embed_dim=embed_dim, n_zones=n_zones_plus_one - 1)
        encoder.load_state_dict(payload["climate_encoder"])
        encoder.eval()
        return encoder
