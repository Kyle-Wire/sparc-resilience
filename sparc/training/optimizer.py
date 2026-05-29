"""
Optimizer, scheduler, and training step for SPARC V2.

Provides:
  - ``build_optimizer``    — AdamW with 11 per-component learning rate groups
  - ``build_scheduler``    — CosineAnnealingWarmRestarts (T₀=10, T_mult=2)
  - ``warmup_scheduler``   — Linear warmup then hand-off to main scheduler
  - ``training_step``      — Forward / backward / clip / step
  - ``spatial_minibatch_sampler`` — Spatially contiguous batches preserving
    neighbor validity for the physics loss
"""

from __future__ import annotations

import logging
from typing import Generator, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-component optimizer
# ---------------------------------------------------------------------------

def build_optimizer(
    model: nn.Module,
    process_rate_net: nn.Module,
    surrogates: dict[str, nn.Module],
    base_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    source_term_net: nn.Module | None = None,
) -> torch.optim.AdamW:
    """
    AdamW with per-component learning rate groups.

    Different components have fundamentally different loss-landscape
    geometry.  The SIREN physics encoder needs a smaller LR; the process
    rate network operates in physical units and needs a tighter rate.

    Parameters
    ----------
    model : SPARCMetaLearner
    process_rate_net : ProcessRateNet
    surrogates : dict of DifferentiableGWR / GWRF / GGPGAM
    base_lr : default learning rate for generic components
    weight_decay : AdamW weight decay
    source_term_net : optional SourceTermNet
    """
    param_groups = []

    # Model sub-components (if present)
    # NOTE: physics_enc is split so that blend_net gets full LR while
    # SIREN (harmonic_enc) stays at 0.5×.  This prevents the blend
    # weight from collapsing to a near-constant scalar.
    component_lrs = {
        "base_enc":         base_lr,
        "spatial_enc":      base_lr,
        "alpha_emb":        base_lr,
        "fusion":           base_lr,
        "regression_head":  base_lr,
        "exceedance_heads": base_lr * 0.5,
    }

    for name, lr in component_lrs.items():
        sub = getattr(model, name, None)
        if sub is not None:
            params = list(sub.parameters())
            if params:
                param_groups.append({
                    "params": params,
                    "lr": lr,
                    "name": name,
                })

    # Split physics_enc into SIREN components (0.5× LR) and blend_net (1× LR)
    physics_enc = getattr(model, "physics_enc", None)
    if physics_enc is not None:
        blend_net = getattr(physics_enc, "blend_net", None)
        blend_ids = set(id(p) for p in blend_net.parameters()) if blend_net else set()

        siren_params = [p for p in physics_enc.parameters() if id(p) not in blend_ids]
        if siren_params:
            param_groups.append({
                "params": siren_params,
                "lr": base_lr * 0.5,
                "name": "physics_enc_siren",
            })
        if blend_net is not None:
            blend_params = list(blend_net.parameters())
            if blend_params:
                param_groups.append({
                    "params": blend_params,
                    "lr": base_lr,
                    "name": "physics_enc_blend_net",
                })

    # Catch any remaining model parameters not covered above
    named_params_so_far = set()
    for pg in param_groups:
        for p in pg["params"]:
            named_params_so_far.add(id(p))

    remaining = [p for p in model.parameters() if id(p) not in named_params_so_far]
    if remaining:
        param_groups.append({
            "params": remaining,
            "lr": base_lr,
            "name": "model_remainder",
        })

    # Process rate network — physical units, tight LR
    proc_params = list(process_rate_net.parameters())
    if proc_params:
        param_groups.append({
            "params": proc_params,
            "lr": base_lr * 0.01,  # 1e-5 at base_lr=1e-3
            "name": "process_rate",
        })

    # Differentiable surrogates
    for sname, smodule in surrogates.items():
        sparams = list(smodule.parameters())
        if sparams:
            param_groups.append({
                "params": sparams,
                "lr": base_lr * 0.5,
                "name": f"diff_{sname}",
            })

    # Source term network — learns forcing term, tight LR like process rate
    if source_term_net is not None:
        src_params = list(source_term_net.parameters())
        if src_params:
            param_groups.append({
                "params": src_params,
                "lr": base_lr * 0.01,
                "name": "source_term",
            })

    return torch.optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


# ---------------------------------------------------------------------------
# Learning rate scheduling
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    warmup_epochs: int = 10,
) -> torch.optim.lr_scheduler.SequentialLR:
    """
    Monotonic cosine decay preceded by linear warmup.

    Schedule:
      epoch 0–warmup:  linear ramp from 0 → base_lr
      epoch warmup+:   cosine decay to eta_min (no restarts)
    """
    # Linear warmup
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return max(epoch / warmup_epochs, 1e-4)
        return 1.0

    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Monotonic cosine decay (no restarts)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs - warmup_epochs, eta_min=1e-6,
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_epochs],
    )


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def training_step(
    model: nn.Module,
    process_rate_net: nn.Module,
    surrogates: dict[str, nn.Module],
    batch: dict[str, torch.Tensor],
    loss_fn,
    optimizer: torch.optim.Optimizer,
    clip_norm: float = 1.0,
    epoch: int = 0,
    loss_kwargs: dict | None = None,
) -> tuple[float, dict[str, float], float]:
    """
    Single training step: forward → loss → backward → clip → optimizer.

    Parameters
    ----------
    model : SPARCMetaLearner
    process_rate_net : ProcessRateNet
    surrogates : dict of differentiable surrogates
    batch : dict with keys consumed by the model and loss
    loss_fn : callable producing (total, components_dict)
    optimizer : AdamW from build_optimizer
    clip_norm : global gradient norm clip threshold
    epoch : current epoch (for curriculum)
    loss_kwargs : extra kwargs forwarded to loss_fn

    Returns
    -------
    loss_value : float
    loss_components : dict
    total_norm : float — gradient norm before clipping
    """
    optimizer.zero_grad()

    # Forward pass — model-specific; expects batch dict
    outputs = model(**batch)
    loss, loss_components = loss_fn(outputs, batch, epoch=epoch, **(loss_kwargs or {}))

    # Backward
    loss.backward()

    # Global gradient norm clipping across ALL parameters
    all_params = (
        list(model.parameters())
        + list(process_rate_net.parameters())
    )
    for s in surrogates.values():
        all_params.extend(s.parameters())

    total_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=clip_norm)

    if total_norm > clip_norm * 0.9:
        logger.debug(
            "Gradient norm %.3f near clip threshold %.3f", total_norm, clip_norm
        )

    optimizer.step()

    return loss.item(), loss_components, total_norm.item()


# ---------------------------------------------------------------------------
# Spatial mini-batch sampler
# ---------------------------------------------------------------------------

def spatial_minibatch_sampler(
    coords: np.ndarray,
    neighbor_idx: np.ndarray | None,
    batch_size: int = 2048,
    n_batches: int | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> Generator[np.ndarray, None, None]:
    """
    Generate spatially contiguous mini-batches such that each point's
    N/S/E/W neighbors are in the same batch.

    Strategy: sample a centroid, take all points within a radius that
    yields approximately ``batch_size`` points, then keep only those
    whose neighbors are all present.

    Parameters
    ----------
    coords : (N, 2) — projected coordinates
    neighbor_idx : (N, 4) — N/S/E/W neighbor indices (-1 = missing).
        Pass ``None`` to skip the neighbor-coherence filter (e.g. during
        pretraining phases before cardinal neighbors are available).
    batch_size : target batch size
    n_batches : if given, stop after this many batches (per rank)
    rank : DDP rank index (0-based); rank ``r`` receives every
        ``world_size``-th batch starting at offset ``r``, preserving
        geographic locality across ranks.
    world_size : total number of DDP ranks.  Default 1 disables
        partitioning and preserves the original single-process behaviour.
    """
    # Low-memory safety: clamp batch size on RAM-constrained machines so a
    # single batch fits comfortably in memory alongside the model and grads.
    try:
        from sparc.config.hardware_profile import detect_profile
        _profile = detect_profile()
        if _profile.tier == "low" and batch_size > _profile.batch_size:
            batch_size = _profile.batch_size
    except Exception:  # pragma: no cover - best-effort safety override
        pass

    N = len(coords)
    if N <= batch_size:
        # Full batch — no need to sub-sample
        yield np.arange(N)
        return

    global_count = 0
    rank_count = 0
    # Safety cap accounts for the extra batches generated for other ranks.
    max_iter = (n_batches or (N // batch_size + 1)) * world_size * 3

    for _ in range(max_iter):
        # Sample centroid
        centroid_idx = np.random.randint(0, N)
        centroid = coords[centroid_idx]

        # Find all points within radius
        distances = np.linalg.norm(coords - centroid, axis=1)
        radii = np.sort(distances)
        target_idx = min(batch_size, N) - 1
        radius = radii[target_idx]

        batch_idx = np.where(distances <= radius)[0]

        # Verify all neighbors present in batch.
        # When neighbor_idx is None (e.g. during EPA pretraining before cardinal
        # neighbors are built) skip the coherence filter entirely.
        if neighbor_idx is None:
            clean_batch = batch_idx
        else:
            neighbor_set = set(batch_idx.tolist())
            valid_in_batch = np.array([
                all(
                    neighbor_idx[i, k] == -1 or int(neighbor_idx[i, k]) in neighbor_set
                    for k in range(neighbor_idx.shape[1])
                )
                for i in batch_idx
            ])
            clean_batch = batch_idx[valid_in_batch]

        if len(clean_batch) >= batch_size // 2:
            # Partitioned DDP: rank r takes every world_size-th accepted batch
            # starting at offset r, interleaving geographic regions across ranks.
            if global_count % world_size == rank:
                yield clean_batch
                rank_count += 1
                if n_batches is not None and rank_count >= n_batches:
                    return
            global_count += 1
