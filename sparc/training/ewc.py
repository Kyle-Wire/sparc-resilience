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
        # CU-10b: synchronise Fisher across DDP ranks before averaging
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                world_size = dist.get_world_size()
                n_samples_t = torch.tensor(
                    float(n_samples), device=next(iter(fisher.values())).device
                )
                dist.all_reduce(n_samples_t, op=dist.ReduceOp.SUM)
                for name in fisher:
                    dist.all_reduce(fisher[name], op=dist.ReduceOp.SUM)
                n_samples = int(n_samples_t.item())
        except Exception:
            pass
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


def wasserstein_trunk_alignment(
    trunk_activations_new: torch.Tensor,
    coreset_activations: torch.Tensor,
    blur: float = 0.01,
) -> torch.Tensor:
    """Sinkhorn (2-Wasserstein) distance between trunk activation distributions.

    Penalises distributional divergence of trunk activations across cities,
    complementing the EWC parameter-space penalty with a latent-geometry
    penalty.  This is especially useful when the trunk has converged to
    identical parameter values but still produces different activation
    distributions on new-city inputs.

    Uses ``geomloss.SamplesLoss("sinkhorn", p=2)`` when the package is
    installed; falls back to a pure-PyTorch sliced-Wasserstein approximation
    (mean of random 1-D projections) otherwise.

    Parameters
    ----------
    trunk_activations_new : (B, D) tensor
        Trunk activations on the current city's mini-batch.
    coreset_activations : (B, D) tensor
        Trunk activations stored from the previous city's coreset.
    blur : float
        Sinkhorn regularisation ε (only used by ``geomloss``).

    Returns
    -------
    loss : scalar tensor (differentiable)
    """
    if trunk_activations_new.shape[0] == 0 or coreset_activations.shape[0] == 0:
        return torch.tensor(0.0, device=trunk_activations_new.device)

    try:
        from geomloss import SamplesLoss  # type: ignore
        loss_fn = SamplesLoss("sinkhorn", p=2, blur=blur)
        ot_loss = loss_fn(trunk_activations_new, coreset_activations)
    except ImportError:
        # --- Pure-PyTorch sliced-Wasserstein fallback ---
        # Project onto 64 random unit directions and average 1-D Wasserstein.
        d = trunk_activations_new.shape[1]
        n_proj = min(64, d)
        device = trunk_activations_new.device
        dtype = trunk_activations_new.dtype

        g = torch.Generator(device=device)
        g.manual_seed(42)
        directions = torch.randn(d, n_proj, device=device, dtype=dtype,
                                 generator=g)
        directions = directions / (directions.norm(dim=0, keepdim=True) + 1e-8)

        proj_new = trunk_activations_new @ directions      # (B, n_proj)
        proj_cs  = coreset_activations   @ directions      # (B, n_proj)

        proj_new_sorted, _ = torch.sort(proj_new, dim=0)
        proj_cs_sorted,  _ = torch.sort(proj_cs,  dim=0)

        # Interpolate if batch sizes differ
        if proj_new_sorted.shape[0] != proj_cs_sorted.shape[0]:
            n_common = min(proj_new_sorted.shape[0], proj_cs_sorted.shape[0])
            proj_new_sorted = proj_new_sorted[:n_common]
            proj_cs_sorted  = proj_cs_sorted[:n_common]

        ot_loss = ((proj_new_sorted - proj_cs_sorted) ** 2).mean()

    # CU-10c: synchronise OT loss across DDP ranks
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.all_reduce(ot_loss, op=dist.ReduceOp.SUM)
            ot_loss = ot_loss / dist.get_world_size()
    except Exception:
        pass

    return ot_loss
