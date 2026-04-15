"""
Initial conditions for SPARC V3 physics-informed training.

Provides a warm-start initial condition T₀ from OLS baseline and
source-term scaling, plus a warmup-scheduled IC consistency loss
that gently nudges early predictions toward physical plausibility.

T₀ = 0.5 · X @ β_ols  +  0.5 · S / (α · k)

where:
  β_ols — OLS regression coefficients (simple linear baseline)
  S     — source term estimate
  α     — process rate from ProcessRateNet
  k     — thermal conductivity
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_initial_condition(
    X: torch.Tensor,
    y: torch.Tensor,
    alpha: torch.Tensor | None = None,
    source_term: torch.Tensor | None = None,
    k_thermal: float = 1.5,
) -> torch.Tensor:
    """
    Compute physics-informed initial condition T₀.

    Parameters
    ----------
    X : (N, D) feature matrix (standardized)
    y : (N,) target values (for OLS fit)
    alpha : (N, 1) optional process rate
    source_term : (N,) optional source term
    k_thermal : thermal conductivity

    Returns
    -------
    T0 : (N,) initial condition estimate
    """
    # OLS component: T_ols = X @ (X^T X)^{-1} X^T y
    # Use least-squares solver for numerical stability
    X_aug = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
    beta_ols, _, _, _ = torch.linalg.lstsq(X_aug, y.unsqueeze(1))
    T_ols = (X_aug @ beta_ols).squeeze(-1)

    if alpha is not None and source_term is not None:
        # Source-term component: T_source = S / (α · k)
        alpha_flat = alpha.squeeze(-1).clamp(min=1e-8)
        T_source = source_term / (alpha_flat * k_thermal)
        # Blend: 50% OLS, 50% source-term
        T0 = 0.5 * T_ols + 0.5 * T_source
    else:
        T0 = T_ols

    return T0


def ic_consistency_loss(
    T_pred: torch.Tensor,
    T0: torch.Tensor,
) -> torch.Tensor:
    """
    MSE between predicted field and initial condition.

    Parameters
    ----------
    T_pred : (N,) predicted temperature field
    T0 : (N,) initial condition

    Returns
    -------
    loss : scalar MSE
    """
    return F.mse_loss(T_pred, T0)


def warmup_ic_schedule(
    epoch: int,
    ic_loss: torch.Tensor,
    warmup_epochs: int = 20,
    max_weight: float = 0.05,
) -> torch.Tensor:
    """
    Apply warmup schedule to IC loss.

    IC weight = 0 for first ``warmup_epochs``, then ramps linearly
    to ``max_weight`` over the next ``warmup_epochs`` epochs, then
    stays constant.

    Parameters
    ----------
    epoch : current training epoch
    ic_loss : unweighted IC loss
    warmup_epochs : epochs before IC starts ramping
    max_weight : maximum IC loss weight

    Returns
    -------
    weighted_loss : IC loss multiplied by schedule weight
    """
    if epoch < warmup_epochs:
        return torch.tensor(0.0, device=ic_loss.device)

    elapsed = epoch - warmup_epochs
    ramp = min(1.0, elapsed / max(warmup_epochs, 1))
    weight = max_weight * ramp
    return weight * ic_loss
