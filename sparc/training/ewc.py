"""
Elastic Weight Consolidation (EWC) for SPARC V3 continual learning.

After training on a city, compute the diagonal Fisher information matrix
for the shared trunk parameters.  During subsequent city training, the
EWC penalty discourages large changes to parameters that were important
for previously learned cities.

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in
           neural networks", PNAS 2017.
"""

from __future__ import annotations

import logging
from typing import Iterator

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_fisher_matrix(
    model: nn.Module,
    data_loader: Iterator,
    trunk_keys: set[str],
    device: torch.device | str = "cpu",
    n_batches: int | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute the diagonal Fisher information matrix for trunk parameters.

    The Fisher is approximated as the mean of squared gradients of the
    log-likelihood (MSE loss) with respect to each trunk parameter,
    averaged over data samples.

    Parameters
    ----------
    model : the trained SPARCMetaLearner (or any nn.Module)
    data_loader : iterable of (inputs_dict, y_true) batches
        Each batch yields a dict of tensors matching model.forward() kwargs
        and a (B,) target tensor.
    trunk_keys : set of parameter name prefixes to compute Fisher for
        (e.g. {"physics_enc", "alpha_emb", "trunk_fusion", "time_embed"})
    device : computation device
    n_batches : max number of batches to use (None = all)

    Returns
    -------
    fisher : dict mapping param_name → (shape) diagonal Fisher values
    """
    model.eval()
    model.to(device)

    # Identify trunk parameters
    trunk_params = {
        name: param for name, param in model.named_parameters()
        if any(name.startswith(prefix) for prefix in trunk_keys)
        and param.requires_grad
    }

    # Initialize Fisher accumulators
    fisher = {name: torch.zeros_like(param) for name, param in trunk_params.items()}
    n_samples = 0

    for batch_idx, (inputs, y_true) in enumerate(data_loader):
        if n_batches is not None and batch_idx >= n_batches:
            break

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.to(device)

        model.zero_grad()

        # Forward pass
        if isinstance(inputs, dict):
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            T_pred, _, _ = model(**inputs)
        else:
            T_pred, _, _ = model(*[x.to(device) for x in inputs])

        # Log-likelihood ∝ -MSE (for Gaussian likelihood)
        loss = nn.functional.mse_loss(T_pred.squeeze(), y_true.squeeze())
        loss.backward()

        # Accumulate squared gradients
        batch_size = y_true.shape[0]
        for name, param in trunk_params.items():
            if param.grad is not None:
                fisher[name] += (param.grad.detach() ** 2) * batch_size

        n_samples += batch_size

    # Average over samples
    if n_samples > 0:
        for name in fisher:
            fisher[name] /= n_samples

    model.zero_grad()
    logger.info("Computed Fisher matrix for %d trunk parameters over %d samples",
                len(fisher), n_samples)

    return fisher


def ewc_penalty(
    model: nn.Module,
    fisher_matrices: list[dict[str, torch.Tensor]],
    optimal_params_list: list[dict[str, torch.Tensor]],
    trunk_keys: set[str],
) -> torch.Tensor:
    """
    Compute the EWC penalty across all previously learned cities.

    penalty = Σ_city Σ_i F_i * (θ_i - θ*_i)²

    Parameters
    ----------
    model : current model being trained
    fisher_matrices : list of Fisher dicts from previous cities
    optimal_params_list : list of optimal param dicts (θ*) from previous cities
    trunk_keys : set of trunk parameter prefixes

    Returns
    -------
    penalty : scalar tensor (differentiable)
    """
    if not fisher_matrices:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    penalty = torch.tensor(0.0, device=next(model.parameters()).device)

    current_params = {
        name: param for name, param in model.named_parameters()
        if any(name.startswith(prefix) for prefix in trunk_keys)
    }

    for fisher, optimal_params in zip(fisher_matrices, optimal_params_list):
        for name, param in current_params.items():
            if name in fisher and name in optimal_params:
                f = fisher[name].to(param.device)
                theta_star = optimal_params[name].to(param.device)
                penalty = penalty + (f * (param - theta_star) ** 2).sum()

    return penalty


def extract_trunk_params(
    model: nn.Module,
    trunk_keys: set[str],
) -> dict[str, torch.Tensor]:
    """
    Extract a detached copy of trunk parameter values (θ*).

    Used to store the optimal parameters after training on a city.
    """
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if any(name.startswith(prefix) for prefix in trunk_keys)
    }
