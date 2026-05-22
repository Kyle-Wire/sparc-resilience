"""
Fractional Laplacian operator for SPARC PDE loss (uniform grid).

On a regular fishnet grid, the fractional Laplacian (−Δ)ˢ is computed
spectrally via 2-D FFT:

    (−Δ)ˢ u  =  IFFT2(|ξ|^{2s} · FFT2(u.reshape(rows, cols)))

where ξ = (2π·k/(rows·h), 2π·l/(cols·h)) are the spatial frequencies
and h is the uniform grid spacing in metres.

For s = 1 this recovers the standard Laplacian exactly.
For s ∈ (0, 1) it produces a fractional (nonlocal) operator that models
anomalous diffusion with Lévy-type power-law spatial decay — relevant for
air quality dispersion, flood inundation, and coastal transport scenarios
where long-range spatial coupling with α < 2 has been observed.

The DC component (k = l = 0, |ξ| = 0) is zeroed out so the operator
acts on the mean-subtracted field, consistent with the way standard
Laplacians are used in the SPARC PDE loss.

References
----------
Caffarelli & Silvestre (2007)
    "An extension problem for the fractional Laplacian."
    Comm. Partial Differential Equations, 32(7–9), 1245–1260.

Bucur & Valdinoci (2016)
    "Nonlocal Diffusion and Applications."
    Springer Lecture Notes, Vol. 20.

Usage
-----
>>> from sparc.physics.fractional_operators import fractional_laplacian_fft
>>> # field: (N,) torch.Tensor, h: float (metres), s: float in (0,1]
>>> frac_lap = fractional_laplacian_fft(field, h=30.0, s=0.75, grid_shape=(rows, cols))
>>> # frac_lap: (N,) torch.Tensor — same device/dtype as field
"""

from __future__ import annotations

import math

import torch


def fractional_laplacian_fft(
    field: torch.Tensor,
    h: float,
    s: float,
    grid_shape: tuple[int, int],
) -> torch.Tensor:
    """
    Compute the fractional Laplacian (−Δ)ˢ of a 2-D field on a uniform grid.

    Uses the spectral identity

        (−Δ)ˢ u = IFFT2(|ξ|^{2s} · FFT2(u))

    where ξ are the discrete spatial frequencies for a grid of shape
    ``(rows, cols)`` with uniform spacing ``h`` (in metres).

    Parameters
    ----------
    field : (N,) torch.Tensor
        Flattened field values in row-major order.  N must equal
        rows * cols exactly.
    h : float
        Uniform grid spacing in metres (same units as coordinates).
    s : float
        Fractional exponent in (0, 1].  s = 1 recovers the standard
        Laplacian.  s = 0.5 is the half-Laplacian (relevant for Lévy
        diffusion).
    grid_shape : (rows, cols)
        2-D shape of the fishnet grid.  rows * cols must equal N.

    Returns
    -------
    (N,) torch.Tensor — (−Δ)ˢ field, same device and dtype as input.

    Raises
    ------
    ValueError
        If rows * cols ≠ N, or if s ≤ 0 or s > 1.
    """
    if not (0.0 < s <= 1.0):
        raise ValueError(f"s must be in (0, 1], got {s}.")

    rows, cols = grid_shape
    N = field.shape[0]
    if rows * cols != N:
        raise ValueError(
            f"grid_shape {grid_shape} implies {rows * cols} points, "
            f"but field has {N} elements."
        )

    device = field.device
    orig_dtype = field.dtype

    # Work in float32 for FFT stability
    u = field.to(torch.float32).reshape(rows, cols)

    # ── Spatial frequency arrays ──────────────────────────────────────
    # torch.fft.fftfreq returns cycles-per-sample in [-0.5, 0.5).
    # Multiply by 2π/h to convert to radians-per-metre.
    freq_r = torch.fft.fftfreq(rows, d=h, device=device).mul_(2.0 * math.pi)    # (rows,)
    freq_c = torch.fft.rfftfreq(cols, d=h, device=device).mul_(2.0 * math.pi)   # (cols//2+1,)

    # |ξ|² = ξ_r² + ξ_c²  — shape (rows, cols//2+1)
    xi2 = freq_r[:, None].pow(2) + freq_c[None, :].pow(2)

    # ── Fractional filter |ξ|^{2s} ───────────────────────────────────
    # At DC (|ξ|=0): 0^{2s} = 0 for s>0, so DC is correctly zeroed out.
    frac_filter = xi2.pow(s)   # (rows, cols//2+1); float32

    # ── Spectral multiplication ───────────────────────────────────────
    U = torch.fft.rfft2(u)                          # (rows, cols//2+1) complex64
    frac_U = frac_filter * U                        # element-wise
    result = torch.fft.irfft2(frac_U, s=(rows, cols))  # (rows, cols) float32

    return result.reshape(N).to(orig_dtype)


def fractional_laplacian_loss(
    field: torch.Tensor,
    source_term: torch.Tensor,
    h: float,
    s: float,
    grid_shape: tuple[int, int],
) -> torch.Tensor:
    """
    Fractional diffusion residual loss: ||(−Δ)ˢ T − S||² (mean).

    Convenience wrapper for use in the PDE loss curriculum.

    Parameters
    ----------
    field : (N,) predicted temperature (or other scalar field)
    source_term : (N,) learned source term S(x)
    h, s, grid_shape : as in fractional_laplacian_fft

    Returns
    -------
    Scalar tensor — mean squared fractional diffusion residual.
    """
    frac_lap = fractional_laplacian_fft(field, h=h, s=s, grid_shape=grid_shape)
    residual = frac_lap - source_term
    std = residual.std().clamp(min=1e-6)
    return ((residual / std) ** 2).mean()
