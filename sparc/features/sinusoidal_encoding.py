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
        kappa: Optional[float] = None,
        n_dims: int = 2,
    ) -> None:
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_bands = n_frequencies // 2
        self.max_freq = max_freq
        self._kappa = kappa
        self._learnable_freqs = learnable_freqs
        self.n_dims = n_dims

        # Log-spaced frequencies from 1 to max_freq (may be updated in fit())
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
        self.register_buffer("coord_min", torch.zeros(n_dims))
        self.register_buffer("coord_max", torch.ones(n_dims))
        self._fitted = False

    # ------------------------------------------------------------------
    @property
    def output_dim(self) -> int:
        """Dimensionality of the encoding vector per point."""
        return 2 * self.n_dims * self.n_bands

    # ------------------------------------------------------------------
    def fit(
        self,
        coords: torch.Tensor,
        kappa: Optional[float] = None,
    ) -> "SinusoidalSpatialEncoding":
        """
        Compute and store coordinate normalization bounds.

        If ``kappa`` is provided (or was set at construction time) the
        frequency bands are re-anchored so that the lowest band captures
        the Matérn correlation scale of the domain.  This ensures that
        sinusoidal features are aligned with the domain's actual spatial
        structure rather than a fixed frequency grid.

        Parameters
        ----------
        coords : Tensor, shape ``(N, 2)``
            Projected (x, y) coordinates from the training set.  Units
            must match those of ``kappa`` (typically metres for projected
            CRS).
        kappa : float, optional
            Matérn κ parameter from Stage 0 in units of 1 / coord_unit.
            Overrides the value supplied at construction if provided.

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

        # κ-informed frequency rescaling.
        # After normalization to [0, 1], the number of cycles per unit
        # corresponding to the Matérn correlation length (1/κ) is:
        #
        #   f_corr = κ × mean_extent_in_coord_units
        #
        # We set min_freq = f_corr / 4 (one octave below the correlation
        # scale) so the first band captures spatial patterns at and above
        # the correlation length.  max_freq is extended if f_corr × 4
        # exceeds the current max_freq.
        k = kappa if kappa is not None else self._kappa
        if k is not None and k > 0:
            mean_extent = float(extent.mean().item())
            f_corr = float(k) * mean_extent  # cycles per normalized unit
            if f_corr > 0:
                min_freq = max(0.5, f_corr / 4.0)
                new_max_freq = max(self.max_freq, f_corr * 4.0)
                new_freqs = torch.exp(
                    torch.linspace(
                        math.log(min_freq),
                        math.log(new_max_freq),
                        self.n_bands,
                        dtype=torch.float32,
                    )
                )
                if isinstance(self.freqs, nn.Parameter):
                    self.freqs.data.copy_(new_freqs)
                else:
                    self.freqs.copy_(new_freqs)

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

        freqs = self.freqs.unsqueeze(0)  # (1, n_bands)

        # Scale by 2*pi so that freq=1 → one full cycle across [0,1]
        omega = 2.0 * math.pi * freqs

        # Encode each spatial dimension as sin/cos pairs
        parts = []
        for d in range(self.n_dims):
            x_d = normed[:, d:d + 1]  # (N, 1)
            parts.append(torch.sin(x_d * omega))
            parts.append(torch.cos(x_d * omega))

        # (N, 2 * n_dims * n_bands)
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        kappa_str = f", kappa={self._kappa}" if self._kappa is not None else ""
        return (
            f"{self.__class__.__name__}("
            f"n_bands={self.n_bands}, "
            f"output_dim={self.output_dim}, "
            f"learnable={isinstance(self.freqs, nn.Parameter)}, "
            f"fitted={self._fitted}{kappa_str})"
        )
