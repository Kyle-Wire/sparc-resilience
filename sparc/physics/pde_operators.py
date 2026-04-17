"""
PDE finite-difference operators for SPARC V3.

Provides Laplacian, gradient, curvature, and Hessian operators using
cardinal (N/S/E/W) neighbor structure from ``_build_cardinal_neighbors``.

Grid spacing
------------
For irregular point clouds (e.g. Providence), the grid spacing ``h`` is
not fixed.  Two modes are available:

  ``spacing='local'``  (default)
      Per-point h_i estimated from cardinal neighbor distances.
      Laplacian becomes  (f_N + f_S + f_E + f_W - 4f) / h_i².

  ``spacing='global'``
      Single scalar h = median of all cardinal distances.
      Appropriate for regular grids or when local h is too noisy.

All operators expect projected coordinates in **meters** so that
Laplacian magnitudes are physically meaningful (units: field / m²).
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grid spacing estimation
# ---------------------------------------------------------------------------

def estimate_local_spacing(
    coords: torch.Tensor,
    neighbor_idx: torch.Tensor,
    spacing: str = "local",
) -> torch.Tensor:
    """
    Estimate the effective grid spacing h per point.

    Parameters
    ----------
    coords : (N, 2) — projected coordinates (x, y) in meters
    neighbor_idx : (N, 4) long — [North, South, East, West], -1 = missing
    spacing : ``'local'`` or ``'global'``

    Returns
    -------
    h : (N,) tensor — per-point grid spacing in meters.
        For ``spacing='global'``, all values are the same scalar.
    """
    N = coords.shape[0]
    device = coords.device

    # Compute distance to each valid cardinal neighbor
    dists = torch.zeros(N, 4, device=device)
    valid_count = torch.zeros(N, device=device)

    for k in range(4):
        valid_k = neighbor_idx[:, k] >= 0
        if valid_k.any():
            nbr_coords = coords[neighbor_idx[valid_k, k].clamp(min=0)]
            d = torch.norm(coords[valid_k] - nbr_coords, dim=1)
            dists[valid_k, k] = d
            valid_count[valid_k] += 1.0

    # Mean distance across valid neighbors per point
    dist_sum = dists.sum(dim=1)
    h_local = dist_sum / valid_count.clamp(min=1)

    # Points with no valid neighbors: use global median
    no_neighbors = valid_count == 0
    valid_h = h_local[~no_neighbors]

    if spacing == "global" or len(valid_h) == 0:
        h_global = valid_h.median() if len(valid_h) > 0 else torch.tensor(30.0, device=device)
        h = torch.full((N,), h_global.item(), device=device)
    else:
        h = h_local.clone()
        if no_neighbors.any():
            h[no_neighbors] = valid_h.median()

    # Clamp to prevent division by zero or extreme values
    h = h.clamp(min=1.0)

    # Log diagnostics
    logger.info(
        "Grid spacing: h_median=%.2f  h_std=%.2f  h_min=%.2f  h_max=%.2f  mode=%s",
        h.median().item(), h.std().item(), h.min().item(), h.max().item(), spacing,
    )

    return h


# ---------------------------------------------------------------------------
# Laplacian
# ---------------------------------------------------------------------------

def laplacian(
    field: torch.Tensor,
    neighbor_idx: torch.Tensor,
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Finite-difference Laplacian: ∇²f ≈ (f_N + f_S + f_E + f_W - 4f) / h².

    Parameters
    ----------
    field : (N,) — scalar field values
    neighbor_idx : (N, 4) long — [N, S, E, W], -1 = missing
    h : (N,) or scalar — grid spacing per point

    Returns
    -------
    lap : (M,) — Laplacian values for valid points
    valid : (N,) bool — mask of points with all 4 neighbors present
    """
    valid = (neighbor_idx >= 0).all(dim=1)

    f_n = field[neighbor_idx[valid, 0]]
    f_s = field[neighbor_idx[valid, 1]]
    f_e = field[neighbor_idx[valid, 2]]
    f_w = field[neighbor_idx[valid, 3]]
    f_c = field[valid]

    h_valid = h[valid] if isinstance(h, torch.Tensor) and h.dim() > 0 else h
    lap = (f_n + f_s + f_e + f_w - 4.0 * f_c) / (h_valid ** 2)

    return lap, valid


# ---------------------------------------------------------------------------
# Directional curvatures
# ---------------------------------------------------------------------------

def directional_curvatures(
    field: torch.Tensor,
    neighbor_idx: torch.Tensor,
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-axis second derivatives: ∂²f/∂x² and ∂²f/∂y².

    Uses E/W for x-axis and N/S for y-axis:
      ∂²f/∂x² ≈ (f_E + f_W - 2f) / h²
      ∂²f/∂y² ≈ (f_N + f_S - 2f) / h²

    Parameters
    ----------
    field : (N,)
    neighbor_idx : (N, 4) long — [N, S, E, W]
    h : (N,) or scalar

    Returns
    -------
    d2_dx2 : (M,) — second derivative in x
    d2_dy2 : (M,) — second derivative in y
    valid : (N,) bool
    """
    valid = (neighbor_idx >= 0).all(dim=1)

    f_n = field[neighbor_idx[valid, 0]]
    f_s = field[neighbor_idx[valid, 1]]
    f_e = field[neighbor_idx[valid, 2]]
    f_w = field[neighbor_idx[valid, 3]]
    f_c = field[valid]

    h_valid = h[valid] if isinstance(h, torch.Tensor) and h.dim() > 0 else h
    h2 = h_valid ** 2

    d2_dy2 = (f_n + f_s - 2.0 * f_c) / h2
    d2_dx2 = (f_e + f_w - 2.0 * f_c) / h2

    return d2_dx2, d2_dy2, valid


# ---------------------------------------------------------------------------
# Gradient magnitude
# ---------------------------------------------------------------------------

def gradient_magnitude(
    field: torch.Tensor,
    neighbor_idx: torch.Tensor,
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gradient magnitude from central differences.

      ∂f/∂x ≈ (f_E - f_W) / (2h)
      ∂f/∂y ≈ (f_N - f_S) / (2h)
      |∇f| = √((∂f/∂x)² + (∂f/∂y)²)

    Parameters
    ----------
    field : (N,)
    neighbor_idx : (N, 4) long — [N, S, E, W]
    h : (N,) or scalar

    Returns
    -------
    grad_mag : (M,) — gradient magnitude
    df_dx : (M,) — x-derivative
    df_dy : (M,) — y-derivative
    valid : (N,) bool
    """
    valid = (neighbor_idx >= 0).all(dim=1)

    f_n = field[neighbor_idx[valid, 0]]
    f_s = field[neighbor_idx[valid, 1]]
    f_e = field[neighbor_idx[valid, 2]]
    f_w = field[neighbor_idx[valid, 3]]

    h_valid = h[valid] if isinstance(h, torch.Tensor) and h.dim() > 0 else h
    two_h = 2.0 * h_valid

    df_dx = (f_e - f_w) / two_h
    df_dy = (f_n - f_s) / two_h
    grad_mag = torch.sqrt(df_dx ** 2 + df_dy ** 2 + 1e-12)

    return grad_mag, df_dx, df_dy, valid


# ---------------------------------------------------------------------------
# Hessian invariants
# ---------------------------------------------------------------------------

def hessian_invariants(
    field: torch.Tensor,
    neighbor_idx: torch.Tensor,
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hessian matrix invariants from the diagonal second derivatives.

    Since we only have cardinal neighbors, the off-diagonal ∂²f/∂x∂y
    is not directly available.  We compute:

      det(H) ≈ (∂²f/∂x²)(∂²f/∂y²)        [product of diagonal entries]
      tr(H)  = ∂²f/∂x² + ∂²f/∂y² = ∇²f   [Laplacian]
      anisotropy = |∂²f/∂x² - ∂²f/∂y²| / (|∂²f/∂x²| + |∂²f/∂y²| + ε)

    Interpretation:
      det(H) > 0 → elliptic point (local extremum)
      det(H) < 0 → saddle point (transition zone)
      anisotropy ≈ 0 → isotropic curvature
      anisotropy ≈ 1 → strongly directional

    Parameters
    ----------
    field : (N,)
    neighbor_idx : (N, 4) long
    h : (N,) or scalar

    Returns
    -------
    det_H : (M,)
    anisotropy : (M,)
    valid : (N,) bool
    """
    d2_dx2, d2_dy2, valid = directional_curvatures(field, neighbor_idx, h)

    det_H = d2_dx2 * d2_dy2

    denom = d2_dx2.abs() + d2_dy2.abs() + 1e-8
    anisotropy = (d2_dx2 - d2_dy2).abs() / denom

    return det_H, anisotropy, valid
