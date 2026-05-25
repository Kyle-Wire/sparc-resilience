"""
Geographic Positional Encoding for SPARC SharedTrunk.

Encodes projected (x, y) coordinates as multi-scale sinusoidal Fourier features,
making the SharedTrunk geography-aware at scales from fine-grained local patterns
to city-wide gradients.

Inspired by Space2Vec (Mai et al. 2020) and NeRF positional encoding.  Unlike
the SinusoidalSpatialEncoding (which normalises coordinates to [0, 1] per-batch),
this module uses fixed physical-scale frequency bands (in metres) so that the
encoding is consistent across cities and independent of study-area extent.

Reference
---------
Mai G. et al. (2020) "Multi-Scale Representation Learning for Spatial Feature
Distributions using Grid Cells", ICLR 2020.  https://arxiv.org/abs/2003.00824
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class GeoPositionalEncoding(nn.Module):
    """Multi-scale geographic positional encoding for projected coordinates.

    Encodes ``(x, y)`` projected coordinates (metres) as sinusoidal Fourier
    features at ``output_dim // 4`` log-spaced spatial-scale bands, capturing
    patterns from fine-grained street-level to city-wide gradients.

    Each frequency band contributes four features:
    ``[sin(ω·x), cos(ω·x), sin(ω·y), cos(ω·y)]``
    where ``ω = 2π / scale`` (angular frequency for the given spatial scale).

    The result is a fixed-size ``(N, output_dim)`` tensor that can be
    concatenated to the SharedTrunk inputs before trunk fusion so that
    trunk weights become city-transferable yet geographically aware.

    Parameters
    ----------
    output_dim : int
        Output feature dimension.  Must be divisible by 4.
        ``output_dim // 4`` log-spaced scale bands are used.
    min_scale : float
        Finest spatial scale in metres (default 100 m ≈ street-block level).
    max_scale : float
        Coarsest spatial scale in metres (default 50 000 m ≈ city-wide).
    """

    def __init__(
        self,
        output_dim: int = 32,
        min_scale: float = 100.0,
        max_scale: float = 50_000.0,
    ) -> None:
        super().__init__()
        if output_dim % 4 != 0:
            raise ValueError(
                f"output_dim must be divisible by 4 (sin/cos × x/y per band), "
                f"got {output_dim}."
            )
        self.output_dim = output_dim
        n_bands = output_dim // 4

        # Angular frequencies: ω = 2π / scale, log-spaced from fine to coarse
        scales = torch.exp(
            torch.linspace(math.log(min_scale), math.log(max_scale), n_bands)
        )
        omegas = 2.0 * math.pi / scales  # (n_bands,) — rad per metre
        self.register_buffer("omegas", omegas)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode projected coordinates.

        Parameters
        ----------
        coords : (N, 2) float tensor — projected (x, y) in metres.

        Returns
        -------
        encoding : (N, output_dim) float tensor
        """
        # coords: (N, 2)  omegas: (n_bands,)
        x = coords[:, 0:1]   # (N, 1)
        y = coords[:, 1:2]   # (N, 1)

        # Broadcast: (N, n_bands)
        phases_x = x * self.omegas
        phases_y = y * self.omegas

        encoding = torch.cat(
            [
                torch.sin(phases_x),
                torch.cos(phases_x),
                torch.sin(phases_y),
                torch.cos(phases_y),
            ],
            dim=-1,
        )  # (N, 4 * n_bands = output_dim)

        return encoding
