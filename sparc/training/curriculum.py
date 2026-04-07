"""
Training curriculum and Stochastic Weight Averaging for SPARC V2.

Four-stage training schedule:

  Stage A — Representation Warmup  (epoch 0 – warmup_end)
      Physics off, surrogates track base models, alpha near prior.
  Stage B — Physics Activation     (warmup_end – ramp_end)
      Physics loss ramps in, alpha begins differentiating.
  Stage C — Joint Optimization     (ramp_end – swa_start)
      All lambdas at CMA-ES optimized values, joint gradient flow.
  Stage D — Stochastic Weight Averaging (final swa_epochs)
      Cyclic LR, weight averaging → flat optimum, better generalization.

Also provides ``capacity_sweep`` to identify the double-descent
interpolation threshold before selecting ``hidden_dim``.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lambda schedule
# ---------------------------------------------------------------------------

def get_lambda_schedule(
    epoch: int,
    target_lambdas: dict[str, float],
    warmup_end: int = 10,
    ramp_end: int = 30,
) -> dict[str, float]:
    """
    Return the lambda weight for each loss term at the given epoch.

    Stage A (< warmup_end):
        physics, smooth, alpha_smooth = 0
        base = 0.5, neighbor = 0.01, prior = target

    Stage B (warmup_end ≤ epoch < ramp_end):
        physics, smooth, alpha_smooth ramp linearly 0 → target
        base = 0.2, neighbor ramps, prior = target

    Stage C (≥ ramp_end):
        All at target values, prior decays later via get_prior_weight
    """
    schedule: dict[str, float] = {}
    ramp_duration = max(ramp_end - warmup_end, 1)

    for term, target in target_lambdas.items():
        if term in ("physics", "smooth", "alpha_smooth"):
            if epoch < warmup_end:
                schedule[term] = 0.0
            elif epoch < ramp_end:
                frac = (epoch - warmup_end) / ramp_duration
                schedule[term] = target * frac
            else:
                schedule[term] = target

        elif term == "base":
            if epoch < warmup_end:
                schedule[term] = 0.5
            elif epoch < ramp_end:
                schedule[term] = 0.2
            else:
                schedule[term] = target

        elif term == "neighbor":
            if epoch < warmup_end:
                schedule[term] = 0.01
            elif epoch < ramp_end:
                frac = (epoch - warmup_end) / ramp_duration
                schedule[term] = 0.01 + (target - 0.01) * frac
            else:
                schedule[term] = target

        elif term == "prior":
            # Prior handled separately via get_prior_weight in loss.py
            schedule[term] = target

        else:
            schedule[term] = target

    return schedule


# ---------------------------------------------------------------------------
# Stochastic Weight Averaging (SWA)
# ---------------------------------------------------------------------------

def apply_swa(
    model: nn.Module,
    process_rate_net: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    training_step_fn: Callable,
    swa_epochs: int = 20,
    swa_lr_max: float = 1e-2,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, nn.Module]:
    """
    Apply Stochastic Weight Averaging in the final training phase.

    Cyclic learning rate causes the model to traverse the flat optimum
    region.  Averaging produces a point near its center — best
    spatial generalization.  Typical improvement: +0.5 to +1.5 R².

    Parameters
    ----------
    model : SPARCMetaLearner
    process_rate_net : ProcessRateNet
    train_loader : iterable of batch dicts
    optimizer : pre-built optimizer
    training_step_fn : callable(model, prn, batch) → loss
    swa_epochs : number of SWA epochs
    swa_lr_max : peak learning rate for SWA cyclic schedule
    device : torch device

    Returns
    -------
    swa_model : averaged model
    swa_process_rate : averaged process rate network
    """
    from torch.optim.swa_utils import AveragedModel, SWALR

    swa_model = AveragedModel(model, device=device)
    swa_process_rate = AveragedModel(process_rate_net, device=device)

    swa_scheduler = SWALR(
        optimizer,
        swa_lr=swa_lr_max,
        anneal_epochs=max(swa_epochs // 2, 1),
        anneal_strategy="cos",
    )

    for epoch_i in range(swa_epochs):
        model.train()
        process_rate_net.train()

        for batch in train_loader:
            training_step_fn(model, process_rate_net, batch)

        swa_model.update_parameters(model)
        swa_process_rate.update_parameters(process_rate_net)
        swa_scheduler.step()

        logger.info("SWA epoch %d/%d — parameters averaged", epoch_i + 1, swa_epochs)

    # Update batch statistics (LayerNorm / BatchNorm) with training data
    with torch.no_grad():
        swa_model.train()
        for batch in train_loader:
            if isinstance(batch, dict):
                swa_model.module(**batch)
            else:
                swa_model.module(batch)

    return swa_model.module, swa_process_rate.module


# ---------------------------------------------------------------------------
# Capacity sweep (double descent)
# ---------------------------------------------------------------------------

def capacity_sweep(
    model_factory: Callable[..., nn.Module],
    train_fn: Callable,
    eval_fn: Callable,
    hidden_dims: list[int] | None = None,
    **factory_kwargs: Any,
) -> tuple[int, dict[int, float]]:
    """
    Run a hidden_dim sweep to find the double-descent interpolation threshold.

    Optimal ``hidden_dim`` lies beyond the threshold (second descent regime)
    where the model is overparameterized and the physics loss acts as
    implicit regularization.

    Parameters
    ----------
    model_factory : callable(hidden_dim=..., **kw) → nn.Module
    train_fn : callable(model) → trains for a short run
    eval_fn : callable(model) → returns spatial CV R² (float)
    hidden_dims : list of dims to sweep
    **factory_kwargs : extra kwargs forwarded to model_factory

    Returns
    -------
    optimal_dim : int — selected hidden_dim in the second descent regime
    results : dict mapping hidden_dim → CV R²
    """
    if hidden_dims is None:
        hidden_dims = [64, 128, 256, 512, 1024]

    results: dict[int, float] = {}

    for dim in hidden_dims:
        model = model_factory(hidden_dim=dim, **factory_kwargs)
        train_fn(model)
        cv_r2 = eval_fn(model)
        results[dim] = cv_r2
        logger.info("hidden_dim=%d: CV R²=%.4f", dim, cv_r2)

    # Find interpolation threshold (minimum R²)
    threshold_dim = min(results, key=results.get)
    # Select best dim beyond the threshold
    candidates_beyond = {d: r for d, r in results.items() if d > threshold_dim}

    if candidates_beyond:
        optimal_dim = max(candidates_beyond, key=candidates_beyond.get)
    else:
        # All dims are at or before threshold — pick the best overall
        optimal_dim = max(results, key=results.get)

    logger.info(
        "Interpolation threshold at hidden_dim=%d — selected hidden_dim=%d",
        threshold_dim, optimal_dim,
    )
    return optimal_dim, results
