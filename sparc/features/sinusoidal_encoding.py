"""
Sinusoidal spatial encoding for SPARC V2.

Replaces Laplacian eigenmaps in the V2 neural path.  Encodes (x, y) projected
coordinates as sinusoidal features at multiple frequencies — the spatial
equivalent of transformer positional encoding.

Low frequencies  → broad regional patterns (climate gradients)
High frequencies → fine-scale local patterns (urban morphology)

Faster to compute than eigendecomposition, Fourier-interpretable, and
naturally captures multi-scale spatial periodicity.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalSpatialEncoding(nn.Module):
    """
    Multi-frequency sinusoidal encoding of 2-D spatial coordinates.

    For each of ``n_frequencies // 2`` frequency bands the module produces
    four features per input point: ``sin(ω·x)``, ``cos(ω·x)``,
    ``sin(ω·y)``, ``cos(ω·y)``.

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
        Highest spatial frequency (cycles per coordinate unit).
    """

    def __init__(
        self,
        n_frequencies: int = 64,
        learnable_freqs: bool = False,
        max_freq: float = 1000.0,
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

    # ------------------------------------------------------------------
    @property
    def output_dim(self) -> int:
        """Dimensionality of the encoding vector per point."""
        return 4 * self.n_bands

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
            ``[sin(ω₁·x), cos(ω₁·x), …, sin(ω_k·x),
              cos(ω_k·x), sin(ω₁·y), …, cos(ω_k·y)]``
        """
        x = coords[:, 0:1]  # (N, 1)
        y = coords[:, 1:2]  # (N, 1)

        freqs = self.freqs.unsqueeze(0)  # (1, n_bands)

        # (N, n_bands) each
        sin_x = torch.sin(x * freqs)
        cos_x = torch.cos(x * freqs)
        sin_y = torch.sin(y * freqs)
        cos_y = torch.cos(y * freqs)

        # (N, 4 * n_bands)
        return torch.cat([sin_x, cos_x, sin_y, cos_y], dim=-1)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_bands={self.n_bands}, "
            f"output_dim={self.output_dim}, "
            f"learnable={isinstance(self.freqs, nn.Parameter)})"
        )
