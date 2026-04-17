"""
Boundary condition detection and loss functions for SPARC V3.

Supports three BC types:
  - Neumann (zero-flux): ∂T/∂n = 0 at domain edges
  - Dirichlet (fixed): T = T_boundary at water bodies
  - Robin (convective): ∂T/∂n + β(T - T_∞) = 0 at transitional boundaries

Boundary points are detected automatically from the cardinal neighbor
structure: any point missing one or more N/S/E/W neighbors is a boundary.
"""

from __future__ import annotations

import torch


def detect_boundary_points(
    neighbor_idx: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Identify boundary points from cardinal neighbor structure.

    Parameters
    ----------
    neighbor_idx : (N, 4) int64 — cardinal [N, S, E, W], -1 = missing

    Returns
    -------
    boundary_mask : (N,) bool — True for points on domain boundary
    edge_masks : dict with keys 'north', 'south', 'east', 'west'
        Each is (N,) bool indicating which edge(s) is missing.
    """
    missing = neighbor_idx < 0  # (N, 4)

    edge_masks = {
        "north": missing[:, 0],
        "south": missing[:, 1],
        "east": missing[:, 2],
        "west": missing[:, 3],
    }

    boundary_mask = missing.any(dim=1)  # (N,)
    return boundary_mask, edge_masks


def neumann_loss(
    T: torch.Tensor,
    boundary_mask: torch.Tensor,
    neighbor_idx: torch.Tensor,
    h: torch.Tensor | float,
) -> torch.Tensor:
    """
    Neumann (zero normal flux) boundary loss: ∂T/∂n ≈ 0.

    Approximated using one-sided finite differences at boundary points.
    For each missing cardinal neighbor, the gradient in that direction
    should be zero — i.e. the existing neighbor in the opposite direction
    should have the same temperature.

    Parameters
    ----------
    T : (N,) temperature field
    boundary_mask : (N,) bool
    neighbor_idx : (N, 4) int64 cardinal neighbors
    h : grid spacing (scalar or (N,))

    Returns
    -------
    loss : scalar Neumann BC loss
    """
    if not boundary_mask.any():
        return torch.tensor(0.0, device=T.device)

    N = T.shape[0]
    device = T.device
    total_residual = torch.tensor(0.0, device=device)
    n_terms = 0

    # Pairs: (missing_dir_index, opposite_dir_index)
    # N=0, S=1, E=2, W=3
    pairs = [(0, 1), (1, 0), (2, 3), (3, 2)]

    for missing_idx, opposite_idx in pairs:
        # Points missing this direction
        missing_mask = (neighbor_idx[:, missing_idx] < 0) & boundary_mask
        if not missing_mask.any():
            continue

        # Get opposite neighbor index
        opp_neighbors = neighbor_idx[missing_mask, opposite_idx]

        # Filter for points where opposite neighbor exists
        valid = opp_neighbors >= 0
        if not valid.any():
            continue

        T_boundary = T[missing_mask][valid]
        T_opposite = T[opp_neighbors[valid]]

        # ∂T/∂n ≈ (T_boundary - T_opposite) / h ≈ 0
        if isinstance(h, torch.Tensor) and h.dim() > 0:
            h_local = h[missing_mask][valid]
        else:
            h_local = h

        grad = (T_boundary - T_opposite) / (h_local + 1e-8)
        total_residual = total_residual + (grad ** 2).sum()
        n_terms += valid.sum().item()

    if n_terms > 0:
        return total_residual / n_terms
    return torch.tensor(0.0, device=device)


def dirichlet_loss(
    T: torch.Tensor,
    boundary_mask: torch.Tensor,
    T_boundary: float | torch.Tensor,
) -> torch.Tensor:
    """
    Dirichlet (fixed value) boundary loss: T = T_boundary.

    Parameters
    ----------
    T : (N,) temperature field
    boundary_mask : (N,) bool — True for Dirichlet boundary points
    T_boundary : scalar or (N_boundary,) target temperature

    Returns
    -------
    loss : scalar Dirichlet BC loss (MSE)
    """
    if not boundary_mask.any():
        return torch.tensor(0.0, device=T.device)

    T_at_boundary = T[boundary_mask]
    if isinstance(T_boundary, (int, float)):
        target = torch.full_like(T_at_boundary, T_boundary)
    else:
        target = T_boundary
    return ((T_at_boundary - target) ** 2).mean()


def robin_loss(
    T: torch.Tensor,
    boundary_mask: torch.Tensor,
    neighbor_idx: torch.Tensor,
    h: torch.Tensor | float,
    beta: float = 1.0,
    T_inf: float = 0.0,
) -> torch.Tensor:
    """
    Robin (convective) boundary loss: ∂T/∂n + β(T - T∞) = 0.

    Parameters
    ----------
    T : (N,) temperature field
    boundary_mask : (N,) bool
    neighbor_idx : (N, 4) cardinal neighbors
    h : grid spacing
    beta : convective transfer coefficient
    T_inf : far-field temperature

    Returns
    -------
    loss : scalar Robin BC loss
    """
    if not boundary_mask.any():
        return torch.tensor(0.0, device=T.device)

    device = T.device
    total_residual = torch.tensor(0.0, device=device)
    n_terms = 0

    pairs = [(0, 1), (1, 0), (2, 3), (3, 2)]
    for missing_idx, opposite_idx in pairs:
        missing_mask = (neighbor_idx[:, missing_idx] < 0) & boundary_mask
        if not missing_mask.any():
            continue
        opp = neighbor_idx[missing_mask, opposite_idx]
        valid = opp >= 0
        if not valid.any():
            continue

        T_b = T[missing_mask][valid]
        T_opp = T[opp[valid]]

        if isinstance(h, torch.Tensor) and h.dim() > 0:
            h_local = h[missing_mask][valid]
        else:
            h_local = h

        grad_n = (T_b - T_opp) / (h_local + 1e-8)
        robin_residual = grad_n + beta * (T_b - T_inf)
        total_residual = total_residual + (robin_residual ** 2).sum()
        n_terms += valid.sum().item()

    if n_terms > 0:
        return total_residual / n_terms
    return torch.tensor(0.0, device=device)


def compute_bc_loss(
    T: torch.Tensor,
    neighbor_idx: torch.Tensor,
    h: torch.Tensor | float,
    bc_specs: dict | None = None,
    water_mask: torch.Tensor | None = None,
    T_water: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute boundary condition loss — Dirichlet only at water-adjacent points.

    Domain-edge boundary points (detected from missing cardinal neighbors)
    are excluded from PDE loss by the Laplacian's valid mask, but receive
    no explicit BC constraint (no Neumann / Robin).

    Parameters
    ----------
    T : (N,) temperature field
    neighbor_idx : (N, 4) cardinal neighbors
    h : grid spacing
    bc_specs : optional dict with key ``dirichlet_weight``
    water_mask : (N,) bool — True for water-adjacent points
                 (Distance_from_water_m < 50).  Built from the distance
                 feature, NOT from cardinal-neighbor edge detection.
    T_water : Dirichlet temperature at water bodies

    Returns
    -------
    total_bc_loss : scalar
    loss_dict : per-term breakdown
    """
    if bc_specs is None:
        bc_specs = {}

    loss_dict: dict[str, float] = {}
    total = torch.tensor(0.0, device=T.device)

    # Dirichlet at water-adjacent points only
    w_dirichlet = bc_specs.get("dirichlet_weight", 0.05)
    if w_dirichlet > 0 and water_mask is not None and T_water is not None:
        diri = dirichlet_loss(T, water_mask, T_water)
        term = w_dirichlet * diri
        total = total + term
        loss_dict["bc_dirichlet"] = term.item()

    return total, loss_dict
