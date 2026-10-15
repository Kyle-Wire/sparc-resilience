"""Discrete spatial operators shared by the physics model, the synthetic
city generator and the influence/focal-feature code.

The canopy-layer heat budget used throughout the core is the steady
advection–diffusion–relaxation equation (roadmap §2.1)

    (1 − L²∇² + v·∇) φ = q

with L = √(Kτ) (m) and v = u·τ (m).  It is discretised with 3-point second
differences δ²_x, δ²_y and *central* first differences δ_x, δ_y,

    φ − D_x·δ²_x φ − D_y·δ²_y φ + vx·δ_x φ + vy·δ_y φ = q,

whose Fourier symbols are

    λ_x(k) = (4/dx²)·sin²(kx·dx/2),   λ_y(k) = (4/dy²)·sin²(ky·dy/2)
    adv(k) = vx·sin(kx·dx)/dx + vy·sin(ky·dy)/dy

so the spectral solve Ĝ = 1/(1 + D_x·λ_x + D_y·λ_y + i·adv) is the exact
inverse of :func:`apply_operator` on the (zero-padded, periodic) domain:
``apply_operator(solve(q)) = q`` on the interior to rounding, for either
scheme.  The advection symbol vanishes at the Nyquist frequencies by
construction (sin π = 0), so real inverse FFTs are exact.

Advection schemes (``scheme=``, :data:`SCHEMES`)
-----------------------------------------------
* ``"exponential"`` (default): exponentially fitted diffusion on each axis
  (Il'in 1969; the exponential scheme of Patankar 1980),

      D_x = L²·Pe_x·coth(Pe_x),   Pe_x = vx·dx / (2L²)      (likewise D_y),

  with D = L² exactly when the axis' velocity is 0.  Advection stays central.
* ``"central"``: D_x = D_y = L², the 5-point Laplacian of every run fitted
  before this scheme existed (old checkpoints keep it).

Why fit the diffusion.  With D = L² a neighbour's stencil weight is
−D/d² ± v/(2d), which turns positive once the cell Péclet number
|v_axis|·d/(2L²) exceeds 1: the Green's function then has negative lobes
(heating cools some of its neighbours; min G/max G = −0.81 at L = 30 m,
|v| = 1 km, dx = 30 m), the textbook failure of central differencing for
convection–diffusion (Patankar 1980).  Because x·coth(x) ≥ |x|, the fitted
D ≥ |v_axis|·d/2, every off-diagonal weight is ≤ 0, and with the +1
relaxation term the matrix is a strictly diagonally dominant M-matrix, so
G ≥ 0.  For small Pe, D = L²·(1 + Pe²/3 − …): the schemes agree to O(Pe²)
and coincide at v = 0.  The first moment of G is v for both schemes
(λ′(0) = 0, adv′(0) = v), so the downwind shift is kept, and Ĝ(0) = 1.
Both schemes are second order; the price of positivity is the O(Pe²)
extra diffusion, which on a smooth manufactured solution at Pe = 1.25 makes
the fitted scheme's error about 8× the central one's (still < 1 %), while
near a point source it is slightly more accurate (the verification tests).

Adjoint.  The second differences are symmetric and the central first
differences antisymmetric, and D depends on v only through the even
function Pe·coth(Pe), so the transpose of the operator is the same operator
with −v (used by :func:`green_mass` and the response-surface adjoint solve).

Accuracy oracle.  The continuous Green's function (a density, m⁻²) is

    G(r) = (1/2πL²)·exp(v·r / 2L²)·K₀(|r|·√(1/L² + |v|²/4L⁴)),

the steady state of the classical continuous point source in uniform flow
with first-order decay (Wilson & Miller 1978); ``solve`` of a unit cell
source divided by dx·dy converges to it at second order away from the
source (``tests/core/test_operator_verification.py``).

Every function has a numpy form; the ``*_torch`` variants are differentiable
in the field and in (L, vx, vy), with finite gradients at v = 0.
"""

from __future__ import annotations

import math

import numpy as np

#: Advection schemes: exponentially fitted diffusion (default) or plain central.
SCHEMES = ("exponential", "central")
DEFAULT_SCHEME = "exponential"
# Below this |Pe| the series 1 + Pe²/3 − Pe⁴/45 replaces Pe·coth(Pe): exact to
# rounding there (next term 2Pe⁶/945), and it keeps autograd finite at Pe = 0.
_SERIES_PE = 1e-2


def check_scheme(scheme: str) -> str:
    """Validate an advection scheme name; returns its canonical (lower-case) form."""
    s = str(scheme).lower()
    if s not in SCHEMES:
        raise ValueError(f"unknown advection scheme {scheme!r}; expected one of {SCHEMES}")
    return s


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


def symbols_axes(ny: int, nx: int, dy: float, dx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (λ_x, λ_y, sx, sy) on the rfft2 frequency grid (ny, nx//2 + 1).

    ``λ_x = (4/dx²)·sin²(kx dx/2)`` and ``λ_y`` are the (positive) symbols of
    the per-axis −second differences; ``sx = sin(kx dx)/dx`` and
    ``sy = sin(ky dy)/dy`` are the central-difference derivative symbols
    (the ``i`` is applied by the caller).
    """
    kx = 2.0 * np.pi * np.fft.rfftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    lam_x = (4.0 / dx**2) * np.sin(KX * dx / 2.0) ** 2
    lam_y = (4.0 / dy**2) * np.sin(KY * dy / 2.0) ** 2
    sx = np.sin(KX * dx) / dx
    sy = np.sin(KY * dy) / dy
    return lam_x, lam_y, sx, sy


def symbols(ny: int, nx: int, dy: float, dx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (λ, sx, sy) on the rfft2 frequency grid (ny, nx//2 + 1), with
    λ = λ_x + λ_y the 5-point −Laplacian symbol (see :func:`symbols_axes`)."""
    lam_x, lam_y, sx, sy = symbols_axes(ny, nx, dy, dx)
    return lam_x + lam_y, sx, sy


def _xcothx(x):
    """x·coth(x): even, ≥ max(1, |x|), with the removable singularity at 0
    replaced by its series."""
    x = np.abs(np.asarray(x, dtype=np.float64))
    small = x < _SERIES_PE
    safe = np.where(small, 1.0, x)
    x2 = x * x
    return np.where(small, 1.0 + x2 / 3.0 - x2 * x2 / 45.0, safe / np.tanh(safe))


def fitted_diffusion(L: float, v_axis: float, d: float) -> float:
    """Exponentially fitted diffusion coefficient on one axis (m²):
    D = L²·Pe·coth(Pe) with the cell Péclet number Pe = v_axis·d/(2L²).

    D = L² at v_axis = 0, D ≥ |v_axis|·d/2 always (the bound that makes the
    stencil weights non-positive), and D → |v_axis|·d/2 (first-order upwind)
    as Pe → ∞; L = 0 gives that upwind limit directly.
    """
    L2 = float(L) ** 2
    half = abs(float(v_axis)) * float(d) / 2.0
    if L2 == 0.0:
        return half
    return L2 * float(_xcothx(half / L2))


def _xcothx_torch(x):
    """Differentiable x·coth(x).  The double ``where`` keeps the unused
    branch finite, so gradients are finite (and exact) at x = 0."""
    import torch

    small = x.abs() < _SERIES_PE
    safe = torch.where(small, torch.ones_like(x), x)
    x2 = x * x
    return torch.where(small, 1.0 + x2 / 3.0 - x2 * x2 / 45.0, safe / torch.tanh(safe))


def fitted_diffusion_torch(L, v_axis, d: float):
    """:func:`fitted_diffusion` for tensors, differentiable in ``L`` and
    ``v_axis`` (including at v_axis = 0).  Floats are promoted to float64
    tensors; requires L > 0 (the fits bound L ≥ dx)."""
    import torch

    if not isinstance(L, torch.Tensor):
        L = torch.as_tensor(L, dtype=torch.float64)
    if not isinstance(v_axis, torch.Tensor):
        v_axis = torch.as_tensor(v_axis, dtype=L.dtype, device=L.device)
    L2 = L * L
    return L2 * _xcothx_torch(v_axis * (0.5 * float(d)) / L2)


def axis_diffusion(L: float, v: tuple[float, float], dx: float, dy: float | None = None,
                   scheme: str = DEFAULT_SCHEME) -> tuple[float, float]:
    """(D_x, D_y) in m² for ``scheme`` (L² on both axes for "central")."""
    dy = dx if dy is None else dy
    if check_scheme(scheme) == "central":
        return float(L) ** 2, float(L) ** 2
    return fitted_diffusion(L, v[0], dx), fitted_diffusion(L, v[1], dy)


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
    scheme: str = DEFAULT_SCHEME,
) -> np.ndarray:
    """Solve (1 − L²∇² + v·∇)φ = q on a zero-padded grid; returns φ cropped
    to q's shape.  ``q`` must be finite (fill no-data cells before calling).

    The DC gain is exactly 1 (Ĝ(0) = 1): a uniform source maps to itself.
    ``scheme`` selects the advection discretisation (module docstring).
    """
    scheme = check_scheme(scheme)
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
    lam_x, lam_y, sx, sy = symbols_axes(py, px, dy, dx)
    if scheme == "central":
        diff = (L**2) * (lam_x + lam_y)
    else:
        Dx, Dy = axis_diffusion(L, (vx, vy), dx, dy, scheme)
        diff = Dx * lam_x + Dy * lam_y
    G = 1.0 / (1.0 + diff + 1j * (vx * sx + vy * sy))
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


def second_differences(phi: np.ndarray, dx: float, dy: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis 3-point second differences (δ²_x, δ²_y); NaN on the one-cell
    border (so δ²_x + δ²_y is :func:`laplacian`)."""
    dy = dx if dy is None else dy
    d2x = np.full_like(phi, np.nan, dtype=np.float64)
    d2y = np.full_like(phi, np.nan, dtype=np.float64)
    c = phi[1:-1, 1:-1]
    d2x[1:-1, 1:-1] = (phi[1:-1, 2:] - 2.0 * c + phi[1:-1, :-2]) / dx**2
    d2y[1:-1, 1:-1] = (phi[2:, 1:-1] - 2.0 * c + phi[:-2, 1:-1]) / dy**2
    return d2x, d2y


def central_gradient(phi: np.ndarray, dx: float, dy: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Central differences (∂x, ∂y); NaN on the border.  Axis 1 is x, axis 0 is y."""
    dy = dx if dy is None else dy
    gx = np.full_like(phi, np.nan, dtype=np.float64)
    gy = np.full_like(phi, np.nan, dtype=np.float64)
    gx[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * dx)
    gy[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * dy)
    return gx, gy


def apply_operator(
    phi: np.ndarray, L: float, v: tuple[float, float] = (0.0, 0.0), dx: float = 30.0, dy: float | None = None,
    scheme: str = DEFAULT_SCHEME,
) -> np.ndarray:
    """(1 − L²∇² + v·∇)φ with the same stencils (and ``scheme``) as
    :func:`solve`.  NaN on the border."""
    scheme = check_scheme(scheme)
    dy = dx if dy is None else dy
    gx, gy = central_gradient(phi, dx, dy)
    if scheme == "central":
        return phi - (L**2) * laplacian(phi, dx, dy) + v[0] * gx + v[1] * gy
    Dx, Dy = axis_diffusion(L, v, dx, dy, scheme)
    d2x, d2y = second_differences(phi, dx, dy)
    return phi - Dx * d2x - Dy * d2y + v[0] * gx + v[1] * gy


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


def green_centre(L: float, v: tuple[float, float], dx: float, dy: float | None = None,
                 scheme: str = DEFAULT_SCHEME) -> float:
    """G(0): response of a cell to a unit source in itself (own-only effect)."""
    dy = dx if dy is None else dy
    n = 2 * pad_cells(L, math.hypot(*v), min(dx, dy)) + 1
    q = np.zeros((n, n))
    q[n // 2, n // 2] = 1.0
    return float(solve(q, L, v, dx=dx, dy=dy, pad=0, scheme=scheme)[n // 2, n // 2])


def green_mass(mask: np.ndarray, L: float, v: tuple[float, float], dx: float, dy: float | None = None,
               scheme: str = DEFAULT_SCHEME) -> np.ndarray:
    """(Gᵀ ∗ 1_mask)_i = Σ_{j∈mask} G(j−i): total response inside the study
    area to a unit source at i (≈ 1 in the interior since Ĝ(0) = 1, smaller
    near edges where heat is carried off-domain).  The adjoint of the
    advection–diffusion operator flips the advection vector (for both
    schemes: the fitted diffusion is even in v)."""
    return solve(np.asarray(mask, dtype=np.float64), L, (-v[0], -v[1]), dx=dx, dy=dy, scheme=scheme)


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


def _scalar_tensors(like, *xs):
    """Floats → scalar tensors with ``like``'s dtype/device (tensors pass through)."""
    import torch

    return tuple(x if isinstance(x, torch.Tensor) else torch.tensor(float(x), dtype=like.dtype, device=like.device)
                 for x in xs)


def solve_torch(q, L, vx, vy, dx: float, dy: float | None = None, pad: int = 16, scheme: str = DEFAULT_SCHEME):
    """Differentiable spectral solve.  ``q`` is a (ny, nx) tensor; ``L``, ``vx``,
    ``vy`` are scalar tensors (or floats).  Returns φ cropped to q's shape."""
    import torch

    scheme = check_scheme(scheme)
    dy = dx if dy is None else dy
    ny, nx = q.shape
    py, px = _padded_shape(ny, nx, pad)
    buf = q.new_zeros((py, px))
    buf[pad:pad + ny, pad:pad + nx] = q
    lam_x, lam_y, sx, sy = symbols_axes(py, px, dy, dx)
    sx_t = torch.as_tensor(sx, dtype=q.dtype, device=q.device)
    sy_t = torch.as_tensor(sy, dtype=q.dtype, device=q.device)
    if scheme == "central":
        re = 1.0 + (L**2) * torch.as_tensor(lam_x + lam_y, dtype=q.dtype, device=q.device)
    else:
        L, vx, vy = _scalar_tensors(q, L, vx, vy)
        re = (1.0 + fitted_diffusion_torch(L, vx, dx) * torch.as_tensor(lam_x, dtype=q.dtype, device=q.device)
              + fitted_diffusion_torch(L, vy, dy) * torch.as_tensor(lam_y, dtype=q.dtype, device=q.device))
    denom = torch.complex(re, vx * sx_t + vy * sy_t)
    phi = torch.fft.irfft2(torch.fft.rfft2(buf) / denom, s=(py, px))
    return phi[pad:pad + ny, pad:pad + nx]


def apply_operator_torch(phi, L, vx, vy, dx: float, dy: float | None = None, scheme: str = DEFAULT_SCHEME):
    """(1 − L²∇² + v·∇)φ on the interior (ny−2, nx−2) — same stencils (and
    ``scheme``) as numpy."""
    scheme = check_scheme(scheme)
    dy = dx if dy is None else dy
    c = phi[1:-1, 1:-1]
    d2x = (phi[1:-1, 2:] - 2.0 * c + phi[1:-1, :-2]) / dx**2
    d2y = (phi[2:, 1:-1] - 2.0 * c + phi[:-2, 1:-1]) / dy**2
    gx = (phi[1:-1, 2:] - phi[1:-1, :-2]) / (2.0 * dx)
    gy = (phi[2:, 1:-1] - phi[:-2, 1:-1]) / (2.0 * dy)
    if scheme == "central":
        return c - (L**2) * (d2x + d2y) + vx * gx + vy * gy
    Lt, vxt, vyt = _scalar_tensors(phi, L, vx, vy)
    return c - fitted_diffusion_torch(Lt, vxt, dx) * d2x - fitted_diffusion_torch(Lt, vyt, dy) * d2y + vx * gx + vy * gy
