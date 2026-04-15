"""
Per-predictor spatial derivatives for physics feature enrichment.

Computes spatial derivatives (Laplacian, gradient magnitude, directional
curvatures, Hessian invariants) for each input predictor, plus composite
features that combine predictors in physically meaningful ways.

All derivatives are z-normalised before concatenation so they enter the
network at O(1) scale.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sparc.physics.pde_operators import (
    directional_curvatures,
    estimate_local_spacing,
    gradient_magnitude,
    hessian_invariants,
    laplacian,
)

logger = logging.getLogger(__name__)


def compute_predictor_derivatives(
    X: torch.Tensor,
    neighbor_idx: torch.Tensor,
    h: torch.Tensor,
    var_names: list[str],
) -> tuple[torch.Tensor, list[str]]:
    """
    Compute spatial derivatives for each predictor column.

    Per-predictor outputs (6 features each):
      - ∇² (Laplacian)
      - |∇| (gradient magnitude)
      - ∂²/∂x², ∂²/∂y² (directional curvatures)
      - det(H), anisotropy (Hessian invariants)

    Composite features (3 total):
      - thermal_stress_laplacian = ∇²(Impervious × (1 − Canopy))
      - cooling_potential_laplacian = ∇²(Canopy × NDVI)
      - albedo_canopy_gradient = |∇(Albedo × (1 − Canopy))|

    Parameters
    ----------
    X : (N, P) — predictor matrix (z-normalised)
    neighbor_idx : (N, 4) long — cardinal neighbor indices
    h : (N,) — per-point grid spacing
    var_names : list of P predictor names

    Returns
    -------
    derivatives : (N, D) — all derivative features (zero-filled where invalid)
    names : list of D feature names
    """
    N, P = X.shape
    device = X.device

    features: list[torch.Tensor] = []
    names: list[str] = []

    for j in range(P):
        field = X[:, j]
        vname = var_names[j] if j < len(var_names) else f"var_{j}"

        # Laplacian
        lap_vals, valid = laplacian(field, neighbor_idx, h)
        lap_full = torch.zeros(N, device=device)
        lap_full[valid] = lap_vals
        features.append(lap_full)
        names.append(f"∇²_{vname}")

        # Gradient magnitude
        gmag_vals, _, _, valid_g = gradient_magnitude(field, neighbor_idx, h)
        gmag_full = torch.zeros(N, device=device)
        gmag_full[valid_g] = gmag_vals
        features.append(gmag_full)
        names.append(f"|∇|_{vname}")

        # Directional curvatures
        dx2_vals, dy2_vals, valid_c = directional_curvatures(field, neighbor_idx, h)
        dx2_full = torch.zeros(N, device=device)
        dy2_full = torch.zeros(N, device=device)
        dx2_full[valid_c] = dx2_vals
        dy2_full[valid_c] = dy2_vals
        features.append(dx2_full)
        features.append(dy2_full)
        names.append(f"∂²/∂x²_{vname}")
        names.append(f"∂²/∂y²_{vname}")

        # Hessian invariants
        det_vals, aniso_vals, valid_h = hessian_invariants(field, neighbor_idx, h)
        det_full = torch.zeros(N, device=device)
        aniso_full = torch.zeros(N, device=device)
        det_full[valid_h] = det_vals
        aniso_full[valid_h] = aniso_vals
        features.append(det_full)
        features.append(aniso_full)
        names.append(f"det(H)_{vname}")
        names.append(f"aniso_{vname}")

    # ------------------------------------------------------------------
    # Composite features — physically meaningful combinations
    # ------------------------------------------------------------------
    name_lower = {n.lower(): i for i, n in enumerate(var_names)}

    # thermal_stress_laplacian = ∇²(Impervious × (1 − Canopy))
    imp_idx = name_lower.get("pct_impervious")
    can_idx = name_lower.get("pct_canopy")
    if imp_idx is not None and can_idx is not None:
        composite = X[:, imp_idx] * (1.0 - X[:, can_idx])
        comp_lap, comp_valid = laplacian(composite, neighbor_idx, h)
        comp_full = torch.zeros(N, device=device)
        comp_full[comp_valid] = comp_lap
        features.append(comp_full)
        names.append("∇²_thermal_stress")

    # cooling_potential_laplacian = ∇²(Canopy × NDVI)
    ndvi_idx = name_lower.get("ndvi")
    if can_idx is not None and ndvi_idx is not None:
        composite = X[:, can_idx] * X[:, ndvi_idx]
        comp_lap, comp_valid = laplacian(composite, neighbor_idx, h)
        comp_full = torch.zeros(N, device=device)
        comp_full[comp_valid] = comp_lap
        features.append(comp_full)
        names.append("∇²_cooling_potential")

    # albedo_canopy_gradient = |∇(Albedo × (1 − Canopy))|
    alb_idx = name_lower.get("albedo")
    if alb_idx is not None and can_idx is not None:
        composite = X[:, alb_idx] * (1.0 - X[:, can_idx])
        comp_gmag, _, _, comp_valid = gradient_magnitude(composite, neighbor_idx, h)
        comp_full = torch.zeros(N, device=device)
        comp_full[comp_valid] = comp_gmag
        features.append(comp_full)
        names.append("|∇|_albedo_canopy")

    # Stack all features
    derivatives = torch.stack(features, dim=1)  # (N, D)

    logger.info(
        "Input derivatives: %d predictors × 6 + %d composites = %d features",
        P, len(features) - 6 * P, derivatives.shape[1],
    )

    return derivatives, names


def normalize_derivatives(
    derivatives: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Z-normalise derivative features so they enter the network at O(1) scale.

    Parameters
    ----------
    derivatives : (N, D) — raw derivative features

    Returns
    -------
    normalized : (N, D) — zero-mean, unit-variance
    means : (D,) — per-feature means
    stds : (D,) — per-feature standard deviations
    """
    means = derivatives.mean(dim=0)
    stds = derivatives.std(dim=0).clamp(min=1e-8)
    normalized = (derivatives - means) / stds
    return normalized, means, stds
