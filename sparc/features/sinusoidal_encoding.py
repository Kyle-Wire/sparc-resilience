"""
Sinusoidal spatial encoding for SPARC V2.

Replaces Laplacian eigenmaps in the V2 neural path.  Encodes (x, y) projected
coordinates as sinusoidal features at multiple frequencies — the spatial
equivalent of transformer positional encoding.

Low frequencies  → broad regional patterns (climate gradients)
High frequencies → fine-scale local patterns (urban morphology)

Faster to compute than eigendecomposition, Fourier-interpretable, and
naturally captures multi-scale spatial periodicity.

Coordinates are normalized to [0, 1] before encoding so that frequency
values are meaningful regardless of the coordinate reference system.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalSpatialEncoding(nn.Module):
    """
    Multi-frequency sinusoidal encoding of 2-D spatial coordinates.

    Coordinates are internally normalized to [0, 1] using either
    explicitly provided bounds or bounds learned from the first
    ``fit()`` / ``forward()`` call.  This ensures that frequency
    values have consistent meaning regardless of the CRS or
    projection used.

    For each of ``n_frequencies // 2`` frequency bands the module produces
    four features per input point: ``sin(omega·x)``, ``cos(omega·x)``,
    ``sin(omega·y)``, ``cos(omega·y)``.

    Parameters
    ----------
    n_frequencies : int
        Total number of frequency bands (split equally across x and y).
        Output dimensionality = ``4 * (n_frequencies // 2)``.
    learnable_freqs : bool
        If *True* the frequency vector is a learnable parameter that is
        updated during training.  If *False* (default) frequencies are
        fixed log-spaced values in [1, max_freq].
    max_freq : float
        Highest spatial frequency (cycles per normalized coordinate
        unit).  With coordinates in [0, 1] a max_freq of 64 gives
        wavelengths down to ~1/64 of the study area extent.
    """

    def __init__(
        self,
        n_frequencies: int = 64,
        learnable_freqs: bool = False,
        max_freq: float = 64.0,
    ) -> None:
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_bands = n_frequencies // 2

        # Log-spaced frequencies from 1 to max_freq
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(max_freq),
                self.n_bands,
            )
        )

        if learnable_freqs:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)

        # Coordinate normalization bounds — set via fit() or first forward()
        self.register_buffer("coord_min", torch.zeros(2))
        self.register_buffer("coord_max", torch.ones(2))
        self._fitted = False

    # ------------------------------------------------------------------
    @property
    def output_dim(self) -> int:
        """Dimensionality of the encoding vector per point."""
        return 4 * self.n_bands

    # ------------------------------------------------------------------
    def fit(self, coords: torch.Tensor) -> "SinusoidalSpatialEncoding":
        """
        Compute and store coordinate normalization bounds.

        Parameters
        ----------
        coords : Tensor, shape ``(N, 2)``
            Projected (x, y) coordinates from the training set.

        Returns
        -------
        self
        """
        self.coord_min = coords.min(dim=0).values.detach().clone()
        self.coord_max = coords.max(dim=0).values.detach().clone()
        # Guard against zero extent (constant coordinate)
        extent = self.coord_max - self.coord_min
        extent = extent.clamp(min=1e-6)
        self.coord_max = self.coord_min + extent
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def _normalize(self, coords: torch.Tensor) -> torch.Tensor:
        """Normalize coordinates to [0, 1] using stored bounds."""
        if not self._fitted:
            self.fit(coords)
        return (coords - self.coord_min) / (self.coord_max - self.coord_min)

    # ------------------------------------------------------------------
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords : Tensor, shape ``(N, 2)``
            Projected (x, y) coordinates.

        Returns
        -------
        Tensor, shape ``(N, 4 * n_bands)``
            Sinusoidal encoding:
            ``[sin(omega_1·x), cos(omega_1·x), …, sin(omega_k·x),
              cos(omega_k·x), sin(omega_1·y), …, cos(omega_k·y)]``
        """
        normed = self._normalize(coords)
        x = normed[:, 0:1]  # (N, 1)
        y = normed[:, 1:2]  # (N, 1)

        freqs = self.freqs.unsqueeze(0)  # (1, n_bands)

        # Scale by 2*pi so that freq=1 → one full cycle across [0,1]
        omega = 2.0 * math.pi * freqs

        # (N, n_bands) each
        sin_x = torch.sin(x * omega)
        cos_x = torch.cos(x * omega)
        sin_y = torch.sin(y * omega)
        cos_y = torch.cos(y * omega)

        # (N, 4 * n_bands)
        return torch.cat([sin_x, cos_x, sin_y, cos_y], dim=-1)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_bands={self.n_bands}, "
            f"output_dim={self.output_dim}, "
            f"learnable={isinstance(self.freqs, nn.Parameter)}, "
            f"fitted={self._fitted})"
        )
