"""
Multi-term PDE loss for SPARC V3.

Implements an 8-term physics-informed loss with staged sub-curriculum:
  1. heat_diffusion   — α∇²T - S ≈ 0
  2. energy_balance   — Q* - QH - QE ≈ 0
  3. directional      — consistent curvature ∂²T/∂x² + ∂²T/∂y²
  4. anisotropy       — penalize spurious isotropy where data is anisotropic
  5. gradient_flux    — Fourier's law flux consistency
  6. gaussian_curv    — penalize extreme curvature (det(H) regularizer)
  7. alpha_smooth     — spatial smoothness of learned α(s)
  8. alpha_prior      — deviation of α(s) from mixture prior

Terms activate progressively (staged sub-curriculum) to prevent
destabilizing a partially-converged network:
  - Epochs 1–10:  heat_diffusion only
  - Epochs 11–20: + energy_balance
  - Epochs 21–30: + directional + anisotropy
  - Epochs 31+:   all 8 terms

Each newly activated term ramps linearly over 5 epochs to avoid
step discontinuities in the loss landscape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from sparc.physics.pde_operators import (
    laplacian,
    directional_curvatures,
    gradient_magnitude,
    hessian_invariants,
)


@dataclass
class PDELossWeights:
    """Base weights for PDE loss terms (before curriculum scaling).

    These weights are multiplied by lambda_pde (from the curriculum) and
    the per-term stage activation weight.  After per-residual normalization
    each term is O(1), so these weights control relative importance only.
    """
    heat_diffusion: float = 1.0
    energy_balance: float = 0.50
    directional: float = 0.20
    anisotropy: float = 0.10
    gradient_flux: float = 0.10
    gaussian_curv: float = 0.05
    alpha_smooth: float = 0.10
    alpha_prior: float = 0.10
    # Temporal terms (activated only when multi-snapshot data is available)
    transient: float = 0.05
    nocturnal: float = 0.08


# Staged activation schedule: (term_name, activation_offset)
# Offsets are relative to pde_start_epoch (when outer curriculum
# first enables PDE lambda).  Compressed so all terms are fully
# active ~20 epochs after PDE turns on.
_ACTIVATION_SCHEDULE = [
    ("heat_diffusion", 0),
    ("energy_balance", 5),
    ("directional", 10),
    ("anisotropy", 10),
    ("gradient_flux", 15),
    ("gaussian_curv", 15),
    ("alpha_smooth", 15),
    ("alpha_prior", 15),
]

_RAMP_EPOCHS = 5  # epochs to ramp from 0→1 after activation


def _stage_weight(epoch: int, activation_epoch: int) -> float:
    """Compute staged curriculum weight: 0 before activation, ramp to 1."""
    if epoch < activation_epoch:
        return 0.0
    elapsed = epoch - activation_epoch
    if elapsed >= _RAMP_EPOCHS:
        return 1.0
    return elapsed / _RAMP_EPOCHS


def _normalize_residual(
    residual: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize residual by running std to prevent scale dominance."""
    if valid_mask is not None:
        r = residual[valid_mask]
    else:
        r = residual
    if r.numel() < 2:
        return residual
    std = r.std().clamp(min=1e-6)
    return residual / std


def _expand(values: torch.Tensor, valid: torch.Tensor, N: int) -> torch.Tensor:
    """Expand pre-filtered (M,) values back to full (N,) using valid mask."""
    full = torch.zeros(N, device=values.device, dtype=values.dtype)
    full[valid] = values
    return full


def compute_pde_loss(
    T_pred: torch.Tensor,
    alpha: torch.Tensor,
    source_term: torch.Tensor,
    neighbor_idx: torch.Tensor,
    h: torch.Tensor | float,
    weights: PDELossWeights | None = None,
    epoch: int = 0,
    alpha_prior_field: torch.Tensor | None = None,
    energy_residual: torch.Tensor | None = None,
    pde_start_epoch: int = 30,
    # Temporal / transient PDE terms (V3)
    T_prev: torch.Tensor | None = None,
    dt_hours: float | None = None,
    dT_dt_observed: torch.Tensor | None = None,
    nocturnal_dT_dt: torch.Tensor | None = None,
    T_night: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Multi-term PDE physics loss with staged activation.

    Parameters
    ----------
    T_pred : (N,) predicted temperature field
    alpha : (N, 1) learned process rate (spatial responsiveness)
    source_term : (N,) learned source term S(x)
    neighbor_idx : (N, 4) cardinal neighbor indices [N, S, E, W]
    h : (N,) or scalar — grid spacing
    weights : PDELossWeights or None (uses defaults)
    epoch : current training epoch (controls staged activation)
    alpha_prior_field : (N, 1) optional mixture prior for α
    energy_residual : (N,) optional pre-computed energy balance residual
    pde_start_epoch : epoch at which outer curriculum enables PDE lambda
                      (default 30 = ramp_end).  Internal sub-term schedule
                      offsets are relative to this value.
    T_prev : (N,) optional previous-snapshot temperature for transient loss
    dt_hours : float, hours between T_prev and T_pred snapshots
    dT_dt_observed : (N,) optional observed temporal derivative (heating/cooling rate)
    nocturnal_dT_dt : (N,) optional observed nocturnal cooling rate (S≈0)
    T_night : (N,) optional nighttime temperature field for nocturnal loss

    Returns
    -------
    total_loss : scalar PDE loss
    loss_dict : per-term breakdown {term_name: float}
    """
    if weights is None:
        weights = PDELossWeights()

    N = T_pred.shape[0]
    alpha_flat = alpha.squeeze(-1)  # (N,)
    loss_dict: dict[str, float] = {}
    total = torch.tensor(0.0, device=T_pred.device)

    # Relative epoch: schedule offsets are relative to PDE activation
    pde_epoch = max(0, epoch - pde_start_epoch)

    # Cache full-size operator results (expanded from valid-only)
    lap_T_full: torch.Tensor | None = None
    valid_lap: torch.Tensor | None = None
    d2_dx2_full: torch.Tensor | None = None
    d2_dy2_full: torch.Tensor | None = None
    valid_dir: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Term 1: Heat diffusion  α∇²T - S ≈ 0
    # Alpha normalized by detached mean so residual stays O(1) and
    # is consistent with the V2 physics residual in loss.py.
    # ------------------------------------------------------------------
    sw_heat = _stage_weight(pde_epoch, 0)
    alpha_mean = alpha_flat.detach().mean().clamp(min=1e-8)
    alpha_norm = alpha_flat / alpha_mean
    if sw_heat > 0:
        lap_T_raw, valid_lap = laplacian(T_pred, neighbor_idx, h)
        lap_T_full = _expand(lap_T_raw, valid_lap, N)
        heat_residual = alpha_norm * lap_T_full - source_term
        heat_residual = _normalize_residual(heat_residual, valid_lap)
        heat_loss = (heat_residual[valid_lap] ** 2).mean() if valid_lap.any() else torch.tensor(0.0, device=T_pred.device)
        term = weights.heat_diffusion * sw_heat * heat_loss
        total = total + term
        loss_dict["pde_heat_diffusion"] = term.item()
    else:
        loss_dict["pde_heat_diffusion"] = 0.0

    # ------------------------------------------------------------------
    # Term 2: Energy balance residual
    # ------------------------------------------------------------------
    sw_energy = _stage_weight(pde_epoch, 5)
    if sw_energy > 0 and energy_residual is not None:
        eb_norm = _normalize_residual(energy_residual)
        eb_loss = (eb_norm ** 2).mean()
        term = weights.energy_balance * sw_energy * eb_loss
        total = total + term
        loss_dict["pde_energy_balance"] = term.item()
    else:
        loss_dict["pde_energy_balance"] = 0.0

    # ------------------------------------------------------------------
    # Term 3: Directional curvature consistency
    # ------------------------------------------------------------------
    sw_dir = _stage_weight(pde_epoch, 10)
    if sw_dir > 0:
        d2_dx2_raw, d2_dy2_raw, valid_dir = directional_curvatures(T_pred, neighbor_idx, h)
        d2_dx2_full = _expand(d2_dx2_raw, valid_dir, N)
        d2_dy2_full = _expand(d2_dy2_raw, valid_dir, N)
        if lap_T_full is None:
            lap_T_raw, valid_lap = laplacian(T_pred, neighbor_idx, h)
            lap_T_full = _expand(lap_T_raw, valid_lap, N)
        combined_valid = valid_dir & valid_lap
        if combined_valid.any():
            dir_residual = d2_dx2_full + d2_dy2_full - lap_T_full
            dir_residual = _normalize_residual(dir_residual, combined_valid)
            dir_loss = (dir_residual[combined_valid] ** 2).mean()
        else:
            dir_loss = torch.tensor(0.0, device=T_pred.device)
        term = weights.directional * sw_dir * dir_loss
        total = total + term
        loss_dict["pde_directional"] = term.item()
    else:
        loss_dict["pde_directional"] = 0.0

    # ------------------------------------------------------------------
    # Term 4: Anisotropy penalty
    # ------------------------------------------------------------------
    sw_aniso = _stage_weight(pde_epoch, 10)
    if sw_aniso > 0:
        if d2_dx2_full is None:
            d2_dx2_raw, d2_dy2_raw, valid_dir = directional_curvatures(T_pred, neighbor_idx, h)
            d2_dx2_full = _expand(d2_dx2_raw, valid_dir, N)
            d2_dy2_full = _expand(d2_dy2_raw, valid_dir, N)
        if valid_dir.any():
            aniso = (d2_dx2_full - d2_dy2_full).abs()
            aniso = _normalize_residual(aniso, valid_dir)
            aniso_loss = (aniso[valid_dir] ** 2).mean()
        else:
            aniso_loss = torch.tensor(0.0, device=T_pred.device)
        term = weights.anisotropy * sw_aniso * aniso_loss
        total = total + term
        loss_dict["pde_anisotropy"] = term.item()
    else:
        loss_dict["pde_anisotropy"] = 0.0

    # ------------------------------------------------------------------
    # Term 5: Gradient–flux consistency (Fourier's law)
    # ------------------------------------------------------------------
    sw_grad = _stage_weight(pde_epoch, 15)
    if sw_grad > 0:
        grad_mag_raw, _, _, valid_g = gradient_magnitude(T_pred, neighbor_idx, h)
        grad_mag_full = _expand(grad_mag_raw, valid_g, N)
        if valid_g.any():
            flux = alpha_norm * grad_mag_full
            flux_norm = _normalize_residual(flux, valid_g)
            flux_loss = (flux_norm[valid_g] ** 2).mean()
        else:
            flux_loss = torch.tensor(0.0, device=T_pred.device)
        term = weights.gradient_flux * sw_grad * flux_loss
        total = total + term
        loss_dict["pde_gradient_flux"] = term.item()
    else:
        loss_dict["pde_gradient_flux"] = 0.0

    # ------------------------------------------------------------------
    # Term 6: Gaussian curvature regularizer
    # ------------------------------------------------------------------
    sw_gc = _stage_weight(pde_epoch, 15)
    if sw_gc > 0:
        det_H_raw, _, valid_h = hessian_invariants(T_pred, neighbor_idx, h)
        det_H_full = _expand(det_H_raw, valid_h, N)
        if valid_h.any():
            det_norm = _normalize_residual(det_H_full, valid_h)
            gc_loss = (det_norm[valid_h] ** 2).mean()
        else:
            gc_loss = torch.tensor(0.0, device=T_pred.device)
        term = weights.gaussian_curv * sw_gc * gc_loss
        total = total + term
        loss_dict["pde_gaussian_curv"] = term.item()
    else:
        loss_dict["pde_gaussian_curv"] = 0.0

    # ------------------------------------------------------------------
    # Term 7: Alpha spatial smoothness
    # ------------------------------------------------------------------
    sw_as = _stage_weight(pde_epoch, 15)
    if sw_as > 0:
        lap_alpha_raw, valid_a = laplacian(alpha_flat, neighbor_idx, h)
        lap_alpha_full = _expand(lap_alpha_raw, valid_a, N)
        if valid_a.any():
            as_norm = _normalize_residual(lap_alpha_full, valid_a)
            as_loss = (as_norm[valid_a] ** 2).mean()
        else:
            as_loss = torch.tensor(0.0, device=T_pred.device)
        term = weights.alpha_smooth * sw_as * as_loss
        total = total + term
        loss_dict["pde_alpha_smooth"] = term.item()
    else:
        loss_dict["pde_alpha_smooth"] = 0.0

    # ------------------------------------------------------------------
    # Term 8: Alpha prior deviation
    # ------------------------------------------------------------------
    sw_ap = _stage_weight(pde_epoch, 15)
    if sw_ap > 0 and alpha_prior_field is not None:
        ap_loss = F.mse_loss(alpha, alpha_prior_field)
        term = weights.alpha_prior * sw_ap * ap_loss
        total = total + term
        loss_dict["pde_alpha_prior"] = term.item()
    else:
        loss_dict["pde_alpha_prior"] = 0.0

    # ------------------------------------------------------------------
    # Term 9: Transient consistency  ∂T/∂t ≈ α·∇²T + S/ρc
    # Activated at epoch 0 when temporal data is available — clean signal.
    # ------------------------------------------------------------------
    sw_transient = _stage_weight(pde_epoch, 0) if (T_prev is not None and dt_hours) else 0.0
    if sw_transient > 0 and T_prev is not None and dt_hours:
        if lap_T_full is None:
            lap_T_raw, valid_lap = laplacian(T_pred, neighbor_idx, h)
            lap_T_full = _expand(lap_T_raw, valid_lap, N)
        dt_seconds = dt_hours * 3600.0
        # Observed temporal derivative from snapshots
        if dT_dt_observed is not None:
            obs_dTdt = dT_dt_observed
        else:
            obs_dTdt = (T_pred - T_prev) / dt_seconds
        # PDE-predicted temporal derivative
        pde_dTdt = alpha_norm * lap_T_full + source_term
        transient_residual = pde_dTdt - obs_dTdt
        transient_residual = _normalize_residual(transient_residual, valid_lap)
        if valid_lap is not None and valid_lap.any():
            transient_loss = (transient_residual[valid_lap] ** 2).mean()
        else:
            transient_loss = (transient_residual ** 2).mean()
        term = weights.transient * sw_transient * transient_loss
        total = total + term
        loss_dict["pde_transient"] = term.item()
    else:
        loss_dict["pde_transient"] = 0.0

    # ------------------------------------------------------------------
    # Term 10: Nocturnal calibration  α·∇²T_night ≈ dT/dt_cool
    # At night S≈0 → pure diffusion → clean α supervision.
    # Activated at epoch 0 (highest-quality physics signal).
    # ------------------------------------------------------------------
    sw_nocturnal = _stage_weight(pde_epoch, 0) if (nocturnal_dT_dt is not None and T_night is not None) else 0.0
    if sw_nocturnal > 0 and nocturnal_dT_dt is not None and T_night is not None:
        lap_T_night_raw, valid_night = laplacian(T_night, neighbor_idx, h)
        lap_T_night_full = _expand(lap_T_night_raw, valid_night, N)
        nocturnal_residual = alpha_norm * lap_T_night_full - nocturnal_dT_dt
        nocturnal_residual = _normalize_residual(nocturnal_residual, valid_night)
        if valid_night.any():
            nocturnal_loss = (nocturnal_residual[valid_night] ** 2).mean()
        else:
            nocturnal_loss = torch.tensor(0.0, device=T_pred.device)
        term = weights.nocturnal * sw_nocturnal * nocturnal_loss
        total = total + term
        loss_dict["pde_nocturnal"] = term.item()
    else:
        loss_dict["pde_nocturnal"] = 0.0

    return total, loss_dict
