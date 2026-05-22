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
    normalize_residual as _normalize_residual_pub,
)


@dataclass
class PDELossWeights:
    """Base weights for PDE loss terms (before curriculum scaling).

    These weights are multiplied by lambda_pde (from the curriculum) and
    the per-term stage activation weight.  After per-residual normalization
    each term is O(1), so these weights control relative importance only.
    """
    heat_diffusion: float = 1.0
    directional: float = 0.20
    anisotropy: float = 0.10
    gradient_flux: float = 0.10
    gaussian_curv: float = 0.05
    alpha_smooth: float = 0.10
    alpha_prior: float = 0.10
    # Temporal terms (activated only when multi-snapshot data is available)
    transient: float = 0.05
    nocturnal: float = 0.08
    # Sheaf Laplacian term 11 — multi-scale MAUP-resistance (activated when sheaf_laplacian provided)
    sheaf: float = 0.03
    # Fractional Laplacian term 12 — anomalous / non-local diffusion (activated when grid_shape provided)
    fractional_diffusion: float = 0.20


# Staged activation schedule: (term_name, activation_offset)
# Offsets are relative to pde_start_epoch (when outer curriculum
# first enables PDE lambda).  Compressed so all terms are fully
# active ~20 epochs after PDE turns on.
_ACTIVATION_SCHEDULE = [
    ("heat_diffusion", 0),
    ("directional", 10),
    ("anisotropy", 10),
    ("gradient_flux", 15),
    ("gaussian_curv", 15),
    ("alpha_smooth", 15),
    ("alpha_prior", 15),
    # Term 12 activates late — after local terms stabilise
    ("fractional_diffusion", 20),
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
    """Normalize residual by running std (delegates to public normalize_residual).

    Uses detach_scale=False so gradients flow through the denominator
    (the V3 PDE path convention — compare compute_physics_residual which
    uses detach_scale=True for the V2 path).
    """
    return _normalize_residual_pub(residual, valid_mask, detach_scale=False)


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
    pde_start_epoch: int = 30,
    # Temporal / transient PDE terms (V3)
    T_prev: torch.Tensor | None = None,
    dt_hours: float | None = None,
    dT_dt_observed: torch.Tensor | None = None,
    nocturnal_dT_dt: torch.Tensor | None = None,
    T_night: torch.Tensor | None = None,
    # Precomputed sparse Laplacian (optional — used when T_pred is full-N)
    sparse_laplacian: torch.Tensor | None = None,
    valid_laplacian_mask: torch.Tensor | None = None,
    # Sheaf Laplacian coboundary δ⁰ (optional — Term 11)
    sheaf_delta: torch.Tensor | None = None,
    # Fractional Laplacian grid info (optional — Term 12)
    grid_shape: tuple[int, int] | None = None,
    fractional_s: float = 0.75,
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

    # Choose Laplacian backend: sparse matmul (full-N, precomputed) or
    # 4-gather fallback (batched / no precomputed matrix).
    _use_sparse = (
        sparse_laplacian is not None
        and valid_laplacian_mask is not None
        and sparse_laplacian.shape[0] == N
    )

    def _laplacian_of(field: torch.Tensor):
        if _use_sparse:
            lap_full = torch.sparse.mm(
                sparse_laplacian, field.unsqueeze(-1)
            ).squeeze(-1)                               # (N,) — already expanded
            return lap_full, valid_laplacian_mask
        lap_raw, valid = laplacian(field, neighbor_idx, h)
        return _expand(lap_raw, valid, N), valid

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
        lap_T_full, valid_lap = _laplacian_of(T_pred)
        heat_residual = alpha_norm * lap_T_full - source_term
        heat_residual = _normalize_residual(heat_residual, valid_lap)
        heat_loss = (heat_residual[valid_lap] ** 2).mean() if valid_lap.any() else torch.tensor(0.0, device=T_pred.device)
        term = weights.heat_diffusion * sw_heat * heat_loss
        total = total + term
        loss_dict["pde_heat_diffusion"] = term.item()
    else:
        loss_dict["pde_heat_diffusion"] = 0.0

    # ------------------------------------------------------------------
    # Term 3: Directional curvature consistency
    # ------------------------------------------------------------------
    sw_dir = _stage_weight(pde_epoch, 10)
    if sw_dir > 0:
        d2_dx2_raw, d2_dy2_raw, valid_dir = directional_curvatures(T_pred, neighbor_idx, h)
        d2_dx2_full = _expand(d2_dx2_raw, valid_dir, N)
        d2_dy2_full = _expand(d2_dy2_raw, valid_dir, N)
        if lap_T_full is None:
            lap_T_full, valid_lap = _laplacian_of(T_pred)
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
        lap_alpha_full, valid_a = _laplacian_of(alpha_flat)
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
            lap_T_full, valid_lap = _laplacian_of(T_pred)
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
        lap_T_night_full, valid_night = _laplacian_of(T_night)
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

    # ------------------------------------------------------------------
    # Term 11: Sheaf restriction loss — multi-scale MAUP consistency
    # Penalizes predictions that are inconsistent across scales via the
    # sheaf coboundary operator δ⁰.  Fine scale = T_pred; coarse scale
    # = KNN-smoothed T_pred (acts as local average-pooling of T_pred
    # without requiring a separate multi-resolution forward pass).
    # Activated after nocturnal term (latest activation, optional data).
    # Only active when sheaf_delta is provided.
    # ------------------------------------------------------------------
    sw_sheaf = _stage_weight(pde_epoch, 20) if sheaf_delta is not None else 0.0
    if sw_sheaf > 0 and sheaf_delta is not None:
        # Build 2-scale section: stalk = [T_fine, T_coarse]
        # T_coarse: mean of KNN neighbors' T_pred (using neighbor_idx; 4 neighbors)
        nb_vals = T_pred[neighbor_idx.clamp(min=0)]   # (N, 4)
        valid_nb = (neighbor_idx >= 0).float()        # (N, 4)
        T_coarse = (nb_vals * valid_nb).sum(dim=1) / valid_nb.sum(dim=1).clamp(min=1.0)

        # Section f ∈ C⁰(ℱ): shape (N, 2) = [T_fine, T_coarse]
        f = torch.stack([T_pred, T_coarse], dim=-1)   # (N, 2)
        f_flat = f.reshape(-1, 1)                     # (N*2, 1)

        # ||δ⁰ f||² / (E*d)
        delta_f = torch.sparse.mm(sheaf_delta, f_flat)   # (E*2, 1)
        sheaf_loss = delta_f.pow(2).mean()
        sheaf_loss = _normalize_residual(sheaf_loss.unsqueeze(0)).squeeze(0)
        term = weights.sheaf * sw_sheaf * sheaf_loss
        total = total + term
        loss_dict["pde_sheaf"] = term.item()
    else:
        loss_dict["pde_sheaf"] = 0.0

    # ------------------------------------------------------------------
    # Term 12: Fractional Laplacian diffusion  (−Δ)ˢT − S ≈ 0
    # Non-local operator via FFT, captures anomalous/long-range coupling.
    # Requires a uniform fishnet grid; skipped when grid_shape is None.
    # Activated after local terms stabilise (offset 20).
    # ------------------------------------------------------------------
    sw_frac = _stage_weight(pde_epoch, 20) if grid_shape is not None else 0.0
    if sw_frac > 0 and grid_shape is not None:
        try:
            from sparc.physics.fractional_operators import fractional_laplacian_fft
            _h_scalar = float(h.mean()) if hasattr(h, 'mean') else float(h)
            frac_lap = fractional_laplacian_fft(
                T_pred.detach(),  # detach: operator is non-differentiable via autograd
                h=_h_scalar,
                s=fractional_s,
                grid_shape=grid_shape,
            )
            frac_residual = frac_lap - source_term
            frac_norm = _normalize_residual(frac_residual)
            frac_loss = (frac_norm ** 2).mean()
            term = weights.fractional_diffusion * sw_frac * frac_loss
            total = total + term
            loss_dict["pde_fractional_diffusion"] = term.item()
        except Exception:
            loss_dict["pde_fractional_diffusion"] = 0.0
    else:
        loss_dict["pde_fractional_diffusion"] = 0.0

    return total, loss_dict
