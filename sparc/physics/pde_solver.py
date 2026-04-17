"""
Iterative PDE forward solver for steady-state Poisson equation.

Solves  α∇²T = S  on irregular point clouds using Jacobi iteration
with the same cardinal-neighbor stencil as the training loss.

Usage (Stage 4 scenario simulation)::

    from sparc.physics.pde_solver import poisson_solve

    T_post, info = poisson_solve(
        T_init=T_baseline,           # (N,) z-scored baseline field
        source=S_new,                # (N,) modified source term (z-space)
        alpha=alpha_field,           # (N,) process rate (physical units)
        neighbor_idx=neighbor_idx,   # (N, 4) cardinal neighbors, -1 = missing
        h=h,                         # (N,) per-point grid spacing in meters
    )
    delta_z = T_post - T_baseline    # change in z-space
    delta_raw = delta_z * y_std      # convert to original units
"""

from __future__ import annotations

import torch


def poisson_solve(
    T_init: torch.Tensor,
    source: torch.Tensor,
    alpha: torch.Tensor,
    neighbor_idx: torch.Tensor,
    h: torch.Tensor,
    max_iter: int = 500,
    tol: float = 1e-4,
    omega: float = 1.0,
    boundary_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Solve α∇²T = S via Jacobi / SOR iteration.

    Interior points (all 4 cardinal neighbors valid) get the 5-point
    stencil update.  Boundary points hold their ``T_init`` values
    (quasi-Dirichlet from the learned baseline).

    Parameters
    ----------
    T_init : (N,) — initial / baseline temperature field (z-scored).
    source : (N,) — source term S from SourceTermNet (z-space, O(1)).
    alpha : (N,) — process rate from ProcessRateNet (physical units).
    neighbor_idx : (N, 4) int64 — cardinal neighbor indices
        [North, South, East, West].  -1 = missing.
    h : (N,) or scalar — per-point grid spacing in meters.
    max_iter : int — iteration cap (default 500).
    tol : float — convergence threshold on max absolute change.
    omega : float — SOR relaxation factor.  1.0 = standard Jacobi.
        Values in (1, 2) accelerate convergence.
    boundary_mask : (N,) bool, optional — if provided, points where
        True are held fixed (Dirichlet).  Default: auto-detect from
        ``neighbor_idx`` (any row with a -1 is a boundary point).

    Returns
    -------
    T : (N,) — converged temperature field (z-scored).
    info : dict with keys:
        ``converged`` (bool), ``iterations`` (int),
        ``max_residual`` (float), ``mean_residual`` (float).
    """
    N = T_init.shape[0]
    device = T_init.device
    dtype = T_init.dtype

    # --- Identify interior vs boundary points ---
    if boundary_mask is None:
        interior = (neighbor_idx >= 0).all(dim=1)  # (N,)
    else:
        interior = ~boundary_mask

    # --- Normalize α by detached mean (consistent with training) ---
    alpha_mean = alpha.mean().clamp(min=1e-8)
    alpha_norm = alpha / alpha_mean  # ~O(1)

    # --- Precompute RHS = h² × S / α_norm (interior only) ---
    h_sq = (h ** 2) if h.dim() > 0 else h * h
    rhs = h_sq * source / alpha_norm.clamp(min=1e-8)  # (N,)

    # --- Clamp neighbor_idx: replace -1 with 0 for gather safety ---
    # (boundary points won't be updated, so their gathered values don't matter)
    safe_idx = neighbor_idx.clamp(min=0)  # (N, 4)

    # --- Iterate ---
    T = T_init.clone()
    converged = False
    final_max_res = float("inf")
    final_mean_res = float("inf")
    it = 0

    for it in range(1, max_iter + 1):
        # Gather neighbor values: T[safe_idx[:, d]] for d in 0..3
        T_neighbors = T[safe_idx]  # (N, 4)
        neighbor_sum = T_neighbors.sum(dim=1)  # (N,)

        # Jacobi update for interior points
        T_jacobi = (neighbor_sum - rhs) / 4.0  # (N,)

        # SOR blend
        T_new = T.clone()
        T_new[interior] = (1.0 - omega) * T[interior] + omega * T_jacobi[interior]

        # Convergence check
        delta = (T_new[interior] - T[interior]).abs()
        final_max_res = float(delta.max()) if delta.numel() > 0 else 0.0
        final_mean_res = float(delta.mean()) if delta.numel() > 0 else 0.0

        T = T_new

        if final_max_res < tol:
            converged = True
            break

    info = {
        "converged": converged,
        "iterations": it,
        "max_residual": final_max_res,
        "mean_residual": final_mean_res,
    }
    return T, info
