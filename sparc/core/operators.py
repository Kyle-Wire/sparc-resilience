"""Discrete spatial operators shared by the physics model, the synthetic
city generator and the influence/focal-feature code.

The canopy-layer heat budget used throughout the core is the steady
advection–diffusion–relaxation equation (roadmap §2.1)

    (1 − L²∇² + v·∇) φ = q

with L = √(Kτ) (m) and v = u·τ (m).  It is discretised with the 5-point
Laplacian and *central* first differences, whose Fourier symbols are

    λ(k)   = (4/dx²)·sin²(kx·dx/2) + (4/dy²)·sin²(ky·dy/2)
    adv(k) = vx·sin(kx·dx)/dx + vy·sin(ky·dy)/dy

so the spectral solve Ĝ = 1/(1 + L²λ + i·adv) is the exact inverse of
:func:`apply_operator` on the (zero-padded, periodic) domain.  The advection
symbol vanishes at the Nyquist frequencies by construction (sin π = 0), so
real inverse FFTs are exact.

Every function has a numpy form; the ``*_torch`` variants are differentiable
in the field and in (L, vx, vy).
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------


def pad_cells(L: float, v_norm: float, dx: float, factor: float = 3.0, minimum: int = 4) -> int:
    """Cells of zero padding needed so the periodic solve does not wrap.

    The Green's function decays over ~L upwind/cross-wind and over ~|v|
    downwind when |v| ≫ L, so we pad ``factor·(L + |v|)`` on every side.
    """
    return max(minimum, int(math.ceil(factor * (abs(L) + abs(v_norm)) / dx)))


def _padded_shape(ny: int, nx: int, pad: int) -> tuple[int, int]:
    # Round up to an even size — cheap FFTs and a well-defined Nyquist bin.
    py = ny + 2 * pad
    px = nx + 2 * pad
    return py + (py % 2), px + (px % 2)


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------


def symbols(ny: int, nx: int, dy: float, dx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (λ, sx, sy) on the rfft2 frequency grid (ny, nx//2 + 1).

    ``λ`` is the (positive) discrete −Laplacian symbol; ``sx = sin(kx dx)/dx``
    and ``sy = sin(ky dy)/dy`` are the central-difference derivative symbols
    (the ``i`` is applied by the caller).
    """
    kx = 2.0 * np.pi * np.fft.rfftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    lam = (4.0 / dx**2) * np.sin(KX * dx / 2.0) ** 2 + (4.0 / dy**2) * np.sin(KY * dy / 2.0) ** 2
    sx = np.sin(KX * dx) / dx
    sy = np.sin(KY * dy) / dy
    return lam, sx, sy


# ---------------------------------------------------------------------------
# numpy solve / apply
# ---------------------------------------------------------------------------


def solve(
    q: np.ndarray,
    L: float,
    v: tuple[float, float] = (0.0, 0.0),
    dx: float = 30.0,
    dy: float | None = None,
    pad: int | None = None,
) -> np.ndarray:
    """Solve (1 − L²∇² + v·∇)φ = q on a zero-padded grid; returns φ cropped
    to q's shape.  ``q`` must be finite (fill no-data cells before calling).

    The DC gain is exactly 1 (Ĝ(0) = 1): a uniform source maps to itself.
    """
    dy = dx if dy is None else dy
    q = np.asarray(q, dtype=np.float64)
    if not np.all(np.isfinite(q)):
        raise ValueError("solve(): source q contains non-finite values; fill them first.")
    ny, nx = q.shape
    vx, vy = float(v[0]), float(v[1])
    if pad is None:
        pad = pad_cells(L, math.hypot(vx, vy), min(dx, dy))
    py, px = _padded_shape(ny, nx, pad)
    buf = np.zeros((py, px), dtype=np.float64)
    buf[pad:pad + ny, pad:pad + nx] = q
    lam, sx, sy = symbols(py, px, dy, dx)
    G = 1.0 / (1.0 + (L**2) * lam + 1j * (vx * sx + vy * sy))
    phi = np.fft.irfft2(np.fft.rfft2(buf) * G, s=(py, px))
    return phi[pad:pad + ny, pad:pad + nx]


def laplacian(phi: np.ndarray, dx: float, dy: float | None = None) -> np.ndarray:
    """5-point Laplacian; NaN on the one-cell border."""
    dy = dx if dy is None else dy
    out = np.full_like(phi, np.nan, dtype=np.float64)
    c = phi[1:-1, 1:-1]
    out[1:-1, 1:-1] = (
        (phi[1:-1, 2:] - 2.0 * c + phi[1:-1, :-2]) / dx**2
        + (phi[2:, 1:-1] - 2.0 * c + phi[:-2, 1:-1]) / dy**2
    )
    return out


def central_gradient(phi: np.ndarray, dx: float, dy: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Central differences (∂x, ∂y); NaN on the border.  Axis 1 is x, axis 0 is y."""
    dy = dx if dy is None else dy
    gx = np.full_like(phi, np.nan, dtype=np.float64)
    gy = np.full_like(phi, np.nan, dtype=np.float64)
    gx[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * dx)
    gy[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * dy)
    return gx, gy


def apply_operator(
    phi: np.ndarray, L: float, v: tuple[float, float] = (0.0, 0.0), dx: float = 30.0, dy: float | None = None
) -> np.ndarray:
    """(1 − L²∇² + v·∇)φ with the same stencils as :func:`solve`.  NaN on the border."""
    dy = dx if dy is None else dy
    gx, gy = central_gradient(phi, dx, dy)
    return phi - (L**2) * laplacian(phi, dx, dy) + v[0] * gx + v[1] * gy


def stencil_valid(mask: np.ndarray) -> np.ndarray:
    """Cells whose own value and all four cardinal neighbours are valid."""
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask)
    out[1:-1, 1:-1] = (
        mask[1:-1, 1:-1] & mask[1:-1, 2:] & mask[1:-1, :-2] & mask[2:, 1:-1] & mask[:-2, 1:-1]
    )
    return out


# ---------------------------------------------------------------------------
# Masked Gaussian smoothing (focal features / exposure mapping)
# ---------------------------------------------------------------------------


def _gaussian_kernel_hat(py: int, px: int, sigma_cells: float) -> np.ndarray:
    """rfft2 of a unit-sum periodic Gaussian kernel centred at (0, 0)."""
    ky = np.fft.fftfreq(py)
    kx = np.fft.rfftfreq(px)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    return np.exp(-2.0 * (np.pi**2) * (sigma_cells**2) * (KX**2 + KY**2))


def masked_gaussian(
    field: np.ndarray,
    mask: np.ndarray,
    sigma_cells: float,
    exclude_self: bool = False,
    min_weight: float = 1e-3,
) -> np.ndarray:
    """Mask-normalised Gaussian mean of ``field`` over valid cells.

    Returns ``(K∗(m·f)) / (K∗m)`` on every cell (NaN where the kernel mass on
    valid cells is below ``min_weight``).  With ``exclude_self`` the centre
    cell's own contribution is removed from numerator and denominator — the
    self-excluded neighbourhood mean used for exposure mapping.
    """
    field = np.asarray(field, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    f = np.where(m > 0, np.nan_to_num(field), 0.0)
    ny, nx = f.shape
    if sigma_cells <= 0:
        return np.where(m > 0, field, np.nan)
    pad = int(math.ceil(4.0 * sigma_cells)) + 1
    py, px = _padded_shape(ny, nx, pad)
    num = np.zeros((py, px))
    den = np.zeros((py, px))
    num[:ny, :nx] = f * m
    den[:ny, :nx] = m
    Kh = _gaussian_kernel_hat(py, px, sigma_cells)
    num_s = np.fft.irfft2(np.fft.rfft2(num) * Kh, s=(py, px))[:ny, :nx]
    den_s = np.fft.irfft2(np.fft.rfft2(den) * Kh, s=(py, px))[:ny, :nx]
    if exclude_self:
        w0 = self_weight(sigma_cells)
        num_s = num_s - w0 * f * m
        den_s = den_s - w0 * m
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num_s / den_s
    out[den_s < min_weight] = np.nan
    return out


def gaussian_conv(raster: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Plain (not mask-normalised) convolution with the unit-sum Gaussian.

    Used for adjoint sums: Σ_j K(j−i)·g_j = (K ∗ g)_i for the symmetric kernel.
    """
    r = np.nan_to_num(np.asarray(raster, dtype=np.float64))
    if sigma_cells <= 0:
        return r
    ny, nx = r.shape
    pad = int(math.ceil(4.0 * sigma_cells)) + 1
    py, px = _padded_shape(ny, nx, pad)
    buf = np.zeros((py, px))
    buf[:ny, :nx] = r
    return np.fft.irfft2(np.fft.rfft2(buf) * _gaussian_kernel_hat(py, px, sigma_cells), s=(py, px))[:ny, :nx]


def green_centre(L: float, v: tuple[float, float], dx: float, dy: float | None = None) -> float:
    """G(0): response of a cell to a unit source in itself (own-only effect)."""
    dy = dx if dy is None else dy
    n = 2 * pad_cells(L, math.hypot(*v), min(dx, dy)) + 1
    q = np.zeros((n, n))
    q[n // 2, n // 2] = 1.0
    return float(solve(q, L, v, dx=dx, dy=dy, pad=0)[n // 2, n // 2])


def green_mass(mask: np.ndarray, L: float, v: tuple[float, float], dx: float, dy: float | None = None) -> np.ndarray:
    """(Gᵀ ∗ 1_mask)_i = Σ_{j∈mask} G(j−i): total response inside the study
    area to a unit source at i (≈ 1 in the interior since Ĝ(0) = 1, smaller
    near edges where heat is carried off-domain).  The adjoint of the
    advection–diffusion operator flips the advection vector."""
    return solve(np.asarray(mask, dtype=np.float64), L, (-v[0], -v[1]), dx=dx, dy=dy)


def self_weight(sigma_cells: float, size: int = 4096) -> float:
    """Weight of the centre cell in the unit-sum discrete Gaussian kernel."""
    if sigma_cells <= 0:
        return 1.0
    # Discrete kernel normalised by its (periodic) Fourier construction:
    # w0 = mean over frequencies of the kernel hat = 1/N Σ_k Ĝ(k).
    n = int(max(64, min(size, 16 * math.ceil(sigma_cells) + 64)))
    k = np.fft.fftfreq(n)
    one_d = np.exp(-2.0 * (np.pi**2) * (sigma_cells**2) * k**2).mean()
    return float(one_d**2)


# ---------------------------------------------------------------------------
# torch variants (differentiable)
# ---------------------------------------------------------------------------


def solve_torch(q, L, vx, vy, dx: float, dy: float | None = None, pad: int = 16):
    """Differentiable spectral solve.  ``q`` is a (ny, nx) tensor; ``L``, ``vx``,
    ``vy`` are scalar tensors (or floats).  Returns φ cropped to q's shape."""
    import torch

    dy = dx if dy is None else dy
    ny, nx = q.shape
    py, px = _padded_shape(ny, nx, pad)
    buf = q.new_zeros((py, px))
    buf[pad:pad + ny, pad:pad + nx] = q
    lam, sx, sy = symbols(py, px, dy, dx)
    lam_t = torch.as_tensor(lam, dtype=q.dtype, device=q.device)
    sx_t = torch.as_tensor(sx, dtype=q.dtype, device=q.device)
    sy_t = torch.as_tensor(sy, dtype=q.dtype, device=q.device)
    denom = torch.complex(1.0 + (L**2) * lam_t, vx * sx_t + vy * sy_t)
    phi = torch.fft.irfft2(torch.fft.rfft2(buf) / denom, s=(py, px))
    return phi[pad:pad + ny, pad:pad + nx]


def apply_operator_torch(phi, L, vx, vy, dx: float, dy: float | None = None):
    """(1 − L²∇² + v·∇)φ on the interior (ny−2, nx−2) — same stencils as numpy."""
    dy = dx if dy is None else dy
    c = phi[1:-1, 1:-1]
    lap = (phi[1:-1, 2:] - 2.0 * c + phi[1:-1, :-2]) / dx**2 + (phi[2:, 1:-1] - 2.0 * c + phi[:-2, 1:-1]) / dy**2
    gx = (phi[1:-1, 2:] - phi[1:-1, :-2]) / (2.0 * dx)
    gy = (phi[2:, 1:-1] - phi[:-2, 1:-1]) / (2.0 * dy)
    return c - (L**2) * lap + vx * gx + vy * gy
