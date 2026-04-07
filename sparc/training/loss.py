"""
Joint loss function for SPARC V2 neural training.

Seven-term unified objective that combines data fidelity, physics
constraints, and model consistency into a single differentiable loss.

All terms back-propagate end-to-end through the meta-learner and
differentiable surrogates to the raw input features.

Terms
-----
1. MSE — data fidelity (regression)
2. Cross-entropy — exceedance probability calibration
3. Physics residual — PDE satisfaction (normalized)
4. Prediction smoothness — penalize implausible spatial spikes
5. Alpha field smoothness — process rate spatial continuity
6. Alpha prior regularization — curriculum-decayed
7. Spatial neighborhood consistency

(Surrogate fidelity is accepted but inactive — surrogates learn
end-to-end via the main loss terms, not by matching V1 outputs.)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Physics residual computation
# ---------------------------------------------------------------------------

def compute_physics_residual(
    T_pred: torch.Tensor,
    alpha: torch.Tensor,
    neighbor_idx: torch.Tensor,
    source_term: torch.Tensor,
    resolution: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Finite-difference Laplacian with normalized residual.

    Parameters
    ----------
    T_pred : (N,) — predicted outcome field
    alpha : (N,) — process rate per point
    neighbor_idx : (N, 4) long — indices [N, S, E, W]; -1 = missing
    source_term : (N,) — forcing / source term
    resolution : float — grid spacing in meters

    Returns
    -------
    normalized_residual : (M,) — physics residual / running std
    laplacian : (M,) — raw finite-difference Laplacian
    valid : (N,) bool mask — points with all 4 neighbors present
    """
    valid = (neighbor_idx != -1).all(dim=1)
    T = T_pred

    T_n = T[neighbor_idx[valid, 0]]
    T_s = T[neighbor_idx[valid, 1]]
    T_e = T[neighbor_idx[valid, 2]]
    T_w = T[neighbor_idx[valid, 3]]

    laplacian = (T_n + T_s + T_e + T_w - 4 * T[valid]) / (resolution ** 2)
    physics_residual = laplacian - (source_term[valid] / alpha[valid])

    # Normalize by residual magnitude — keeps physics loss on same scale as MSE
    residual_scale = physics_residual.detach().std().clamp(min=1e-6)
    normalized_residual = physics_residual / residual_scale

    return normalized_residual, laplacian, valid


# ---------------------------------------------------------------------------
# Prior weight decay
# ---------------------------------------------------------------------------

def get_prior_weight(epoch: int, lambda_prior_base: float) -> float:
    """
    Decay prior regularization over training.

    Early:  prior guides alpha near mixture model.
    Late:   data trusted to pull alpha away from prior.

    Schedule:
      epoch < 10:  full weight
      epoch < 30:  full weight
      epoch >= 30: exponential decay toward 10% of base
    """
    if epoch < 30:
        return lambda_prior_base
    decay = 0.1 + 0.9 * math.exp(-0.05 * (epoch - 30))
    return lambda_prior_base * decay


# ---------------------------------------------------------------------------
# Joint loss
# ---------------------------------------------------------------------------

def sparc_joint_loss(
    # Predictions
    T_pred: torch.Tensor,
    exceedance_preds: list[torch.Tensor],
    y_true: torch.Tensor,
    thresholds: list[float],
    # Process rate
    alpha: torch.Tensor,
    alpha_prior: torch.Tensor,
    # Spatial structure
    neighbor_idx: torch.Tensor,
    source_term: torch.Tensor,
    resolution: float,
    # Surrogate alignment
    surrogate_preds: list[torch.Tensor],
    surrogate_targets: list[torch.Tensor],
    # Lambda weights (from CMA-ES or config)
    lambda_physics: float,
    lambda_smooth: float,
    lambda_alpha_smooth: float,
    lambda_prior: float,
    lambda_base: float,
    lambda_neighbor: float,
    # Training phase (for curriculum)
    epoch: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the 8-term joint loss.

    Returns
    -------
    total : scalar Tensor — total loss for backward()
    loss_components : dict — each term's scalar value for logging
    """
    loss_components: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 1. Data fidelity — regression MSE
    # ------------------------------------------------------------------
    mse = F.mse_loss(T_pred.squeeze(), y_true)
    loss_components["mse"] = mse.item()

    # ------------------------------------------------------------------
    # 2. Exceedance probabilities — cross-entropy per threshold
    # ------------------------------------------------------------------
    ce = torch.tensor(0.0, device=T_pred.device)
    if exceedance_preds and thresholds:
        for exc_pred, thresh in zip(exceedance_preds, thresholds):
            target = (y_true > thresh).float()
            ce = ce + F.binary_cross_entropy(
                exc_pred.squeeze().clamp(1e-6, 1 - 1e-6), target
            )
        ce = ce * 0.5
    loss_components["cross_entropy"] = ce.item()

    # ------------------------------------------------------------------
    # 3. Physics residual — heat diffusion PDE (or domain equivalent)
    # ------------------------------------------------------------------
    physics = torch.tensor(0.0, device=T_pred.device)
    smooth = torch.tensor(0.0, device=T_pred.device)
    alpha_smooth = torch.tensor(0.0, device=T_pred.device)

    has_neighbors = neighbor_idx is not None and neighbor_idx.numel() > 0
    if has_neighbors and lambda_physics > 0:
        normalized_residual, laplacian, valid = compute_physics_residual(
            T_pred.squeeze(), alpha.squeeze(), neighbor_idx, source_term, resolution,
        )
        if valid.sum() > 0:
            physics = lambda_physics * normalized_residual.pow(2).mean()

            # 4. Prediction smoothness
            smooth = lambda_smooth * laplacian.pow(2).mean()

            # 5. Alpha field smoothness
            a = alpha.squeeze()
            a_n = a[neighbor_idx[valid, 0]]
            a_s = a[neighbor_idx[valid, 1]]
            a_e = a[neighbor_idx[valid, 2]]
            a_w = a[neighbor_idx[valid, 3]]
            alpha_lap = (a_n + a_s + a_e + a_w - 4 * a[valid]) / (resolution ** 2)
            alpha_smooth = lambda_alpha_smooth * alpha_lap.pow(2).mean()

    loss_components["physics"] = physics.item()
    loss_components["smooth"] = smooth.item()
    loss_components["alpha_smooth"] = alpha_smooth.item()

    # ------------------------------------------------------------------
    # 6. Alpha prior regularization — decayed by curriculum
    # ------------------------------------------------------------------
    prior_weight = get_prior_weight(epoch, lambda_prior)
    prior_reg = prior_weight * (alpha - alpha_prior).pow(2).mean()
    loss_components["alpha_prior"] = prior_reg.item()

    # ------------------------------------------------------------------
    # 7. Surrogate fidelity
    # ------------------------------------------------------------------
    surrogate_loss = torch.tensor(0.0, device=T_pred.device)
    if surrogate_preds and surrogate_targets:
        surrogate_loss = lambda_base * sum(
            F.mse_loss(pred, target)
            for pred, target in zip(surrogate_preds, surrogate_targets)
        )
    loss_components["surrogate"] = surrogate_loss.item()

    # ------------------------------------------------------------------
    # 8. Spatial neighborhood consistency
    # ------------------------------------------------------------------
    neighborhood = torch.tensor(0.0, device=T_pred.device)
    if has_neighbors and lambda_neighbor > 0:
        point_loss = (T_pred.squeeze() - y_true).pow(2)
        neighbor_loss = torch.zeros_like(point_loss)
        for k in range(4):
            valid_k = neighbor_idx[:, k] != -1
            neighbor_loss[valid_k] += 0.25 * point_loss[neighbor_idx[valid_k, k]]
        neighborhood = lambda_neighbor * (point_loss + 0.3 * neighbor_loss).mean()
    loss_components["neighborhood"] = neighborhood.item()

    # ------------------------------------------------------------------
    # Total
    # ------------------------------------------------------------------
    total = (
        mse + ce + physics + smooth + alpha_smooth
        + prior_reg + surrogate_loss + neighborhood
    )
    loss_components["total"] = total.item()

    return total, loss_components
