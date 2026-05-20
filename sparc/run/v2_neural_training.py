"""
V2 neural training orchestrator for SPARC pipeline.

Plugs into Stage 3 of the enhanced pipeline: after base model OOF
predictions are collected (Stage 2), this module trains the neural
meta-learner + process-rate network + differentiable surrogates,
optionally runs CMA-ES, applies Stochastic Weight Averaging (SWA),
and packages outputs for Stage 4 (causal + scenarios).

Architecture (from SPARC V2 Development Document):

  Differentiable Surrogates  →  Stream 1 (Base)      ─┐
  SIREN Physics Encoder      →  Stream 2 (Physics)    ─┤─ Fusion → T_pred
  Sparse Spatial Attention   →  Stream 3 (Spatial)    ─┤           → exceedance
  Process Rate Embedding     →  Stream 4 (Alpha)      ─┘

Training uses the 8-term joint loss, 4-stage curriculum,
per-component optimizer, warmup scheduler, and optional SWA.

Usage (called from enhanced_spatial_cv.py when ``meta_learner == "neural"``)::

    from sparc.run.v2_neural_training import train_neural_meta

    result = train_neural_meta(
        base_predictions=base_predictions,
        y=y,
        coords=coords,
        feature_matrix=X,
        feature_names=feature_names,
        folds=folds,
        config=cfg,
        output_dir=stage2_dir,
    )
"""

from __future__ import annotations

import json
import logging
import os
import time as _time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Land cover classification for process-rate pretraining
# ---------------------------------------------------------------------------

def _classify_land_cover(
    feature_matrix: torch.Tensor | np.ndarray,
    feature_names: list[str],
    material_priors: dict[str, float],
) -> torch.Tensor:
    """
    Classify each point into a land cover class and return target α values.

    Classification uses Pct_Impervious, Pct_Canopy, and Distance_from_water_m
    with simple thresholds.  Returns (N, 1) tensor of target α values drawn
    from the material_priors table.

    Classes (in priority order):
      water_adjacent        — Distance_from_water_m < 50 AND Impervious < 30%
      vegetated             — Canopy > 40% AND Impervious < 20%
      open_green            — Canopy 15-40% AND Impervious < 30%
      low_density           — Impervious 30-50%
      moderate_density      — Impervious 50-70%
      high_density_urban    — Impervious > 70%
    """
    if isinstance(feature_matrix, torch.Tensor):
        fm = feature_matrix.detach().cpu().numpy()
    else:
        fm = np.asarray(feature_matrix)

    N = fm.shape[0]
    prior_values = list(material_priors.values())
    prior_names = list(material_priors.keys())
    default_val = np.median(prior_values)

    # Find feature indices — use what's available
    idx_imp = feature_names.index("Pct_Impervious") if "Pct_Impervious" in feature_names else None
    idx_can = feature_names.index("Pct_Canopy") if "Pct_Canopy" in feature_names else None
    idx_water = feature_names.index("Distance_from_water_m") if "Distance_from_water_m" in feature_names else None

    imp = fm[:, idx_imp] if idx_imp is not None else np.full(N, 50.0)
    can = fm[:, idx_can] if idx_can is not None else np.full(N, 20.0)
    dist_w = fm[:, idx_water] if idx_water is not None else np.full(N, 500.0)

    targets = np.full(N, default_val, dtype=np.float32)

    def _get(name: str) -> float:
        return material_priors.get(name, default_val)

    # Apply in reverse priority (later rules overwrite earlier)
    targets[:] = _get("moderate_density")
    targets[imp > 70] = _get("high_density_urban")
    targets[(imp >= 30) & (imp <= 50)] = _get("low_density")
    targets[(can > 15) & (can <= 40) & (imp < 30)] = _get("open_green")
    targets[(can > 40) & (imp < 20)] = _get("vegetated")
    targets[(dist_w < 50) & (imp < 30)] = _get("water_adjacent")

    return torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)


def _pretrain_process_rate_net(
    process_net: torch.nn.Module,
    physics_feats: torch.Tensor,
    feature_names: list[str],
    pr_col_idxs: list[int],
    material_priors: dict[str, float],
    n_epochs: int = 60,
    lr: float = 1e-3,
    lambda_mono: float = 0.1,
    device: torch.device | None = None,
    feat_mean: np.ndarray | None = None,
    feat_std: np.ndarray | None = None,
) -> float:
    """
    Pre-train ProcessRateNet toward land-cover-classified mixture prior
    with monotonicity regularization.

    Monotonicity penalty: for random pairs of points, if point A has
    higher Pct_Impervious than point B, penalize if predicted α(A) < α(B)
    (since higher impervious should yield higher thermal response rate).

    Returns final loss value.
    """
    if device is None:
        device = next(process_net.parameters()).device

    # Build classification targets from RAW (un-standardized) features.
    # physics_feats is z-scored; _classify_land_cover uses raw thresholds
    # (e.g. Pct_Impervious > 70, Distance_from_water_m < 50).
    if feat_mean is not None and feat_std is not None:
        fm_raw = physics_feats.cpu() * torch.tensor(feat_std) + torch.tensor(feat_mean)
    else:
        fm_raw = physics_feats.cpu()
    targets = _classify_land_cover(
        fm_raw, feature_names, material_priors,
    ).to(device)  # (N, 1)

    pr_input = physics_feats[:, pr_col_idxs]
    N = pr_input.shape[0]

    # Find impervious column index within the PR input subset
    imp_idx_in_pr = None
    for i, col_idx in enumerate(pr_col_idxs):
        if feature_names[col_idx] == "Pct_Impervious":
            imp_idx_in_pr = i
            break

    optimizer = torch.optim.Adam(process_net.parameters(), lr=lr)
    process_net.train()

    final_loss = 0.0
    for ep in range(n_epochs):
        pred = process_net(pr_input)  # (N, 1)

        # Gaussian NLL (MSE as proxy, equivalent when variance is constant)
        mse_loss = torch.nn.functional.mse_loss(pred, targets)

        # Monotonicity regularization via random pairs
        mono_loss = torch.tensor(0.0, device=device)
        if imp_idx_in_pr is not None and lambda_mono > 0:
            n_pairs = min(1024, N)
            idx_a = torch.randint(0, N, (n_pairs,), device=device)
            idx_b = torch.randint(0, N, (n_pairs,), device=device)
            imp_a = pr_input[idx_a, imp_idx_in_pr]
            imp_b = pr_input[idx_b, imp_idx_in_pr]
            alpha_a = pred[idx_a, 0]
            alpha_b = pred[idx_b, 0]

            # If imp_a > imp_b, we want alpha_a >= alpha_b
            # Penalize: max(0, alpha_b - alpha_a) when imp_a > imp_b
            mask = imp_a > imp_b + 1e-3  # meaningful difference
            if mask.any():
                violations = torch.relu(alpha_b[mask] - alpha_a[mask])
                mono_loss = violations.pow(2).mean()

        total_loss = mse_loss + lambda_mono * mono_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        final_loss = total_loss.item()

    return final_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _remap_indices_to_local(
    global_idx: np.ndarray,
    batch_idx: np.ndarray,
    neighbor_tensor: torch.Tensor,
) -> torch.Tensor:
    """Remap global neighbor indices to batch-local indices.

    Any global index not present in ``batch_idx`` is set to -1.
    """
    map_size = int(global_idx.max()) + 1 if len(global_idx) else 0
    if map_size == 0:
        return torch.full_like(neighbor_tensor, -1)

    device = neighbor_tensor.device

    # Build global→local map in torch — no .cpu().numpy() round-trip on
    # neighbor_tensor, which may live on GPU/MPS.
    local_map = torch.full((map_size,), -1, dtype=torch.long)
    batch_t = torch.as_tensor(batch_idx, dtype=torch.long)
    local_map[batch_t] = torch.arange(len(batch_idx), dtype=torch.long)
    local_map = local_map.to(device)

    nb = neighbor_tensor                        # (B, K) — stays on device
    valid = (nb >= 0) & (nb < map_size)
    nb_safe = nb.clamp(0, map_size - 1)        # safe index even for invalid entries
    return torch.where(valid, local_map[nb_safe], torch.full_like(nb, -1))


def _build_knn_index(
    coords: np.ndarray, max_neighbors: int, return_dists: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Build KNN index (N, max_neighbors) from projected coords.

    When *return_dists* is True, also returns (N, max_neighbors) distances.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    dists, indices = tree.query(coords, k=max_neighbors + 1)
    if return_dists:
        return indices[:, 1:], dists[:, 1:]
    return indices[:, 1:]  # exclude self


def _build_cardinal_neighbors(
    coords: np.ndarray,
    resolution: float | None = None,
    tol_factor: float = 1.5,
) -> tuple[np.ndarray, float]:
    """
    Build N/S/E/W cardinal neighbor indices for the physics Laplacian.

    Parameters
    ----------
    coords : (N, 2) — projected coordinates (x, y)
    resolution : grid spacing; auto-detected from nearest-neighbour
                 distances if None
    tol_factor : max distance tolerance as multiple of resolution

    Returns
    -------
    neighbor_idx : (N, 4) int64 — [North, South, East, West], -1 = missing
    resolution   : detected/used resolution
    """
    from scipy.spatial import cKDTree

    N = len(coords)
    tree = cKDTree(coords)

    if resolution is None:
        dists, _ = tree.query(coords, k=2)
        resolution = float(np.median(dists[:, 1]))

    tol = resolution * tol_factor

    # Offsets: North (+y), South (-y), East (+x), West (-x)
    offsets = np.array([
        [0, resolution],
        [0, -resolution],
        [resolution, 0],
        [-resolution, 0],
    ])

    neighbor_idx = np.full((N, 4), -1, dtype=np.int64)
    for k, offset in enumerate(offsets):
        targets = coords + offset
        dists, idxs = tree.query(targets, k=1)
        valid = dists < tol
        neighbor_idx[valid, k] = idxs[valid]

    # ------------------------------------------------------------------
    # Reject self-loops: edge points with no true neighbor in a direction
    # get assigned themselves (distance = resolution < tol).
    # ------------------------------------------------------------------
    arange_N = np.arange(N)
    for k in range(4):
        self_loop = neighbor_idx[:, k] == arange_N
        neighbor_idx[self_loop, k] = -1

    # ------------------------------------------------------------------
    # Reject wrong-direction assignments: ensure the found neighbor
    # actually lies in the expected cardinal direction.
    #   North: dy > 0,  South: dy < 0,  East: dx > 0,  West: dx < 0
    # ------------------------------------------------------------------
    _dir_axis = [1, 1, 0, 0]          # y, y, x, x
    _dir_sign = [1, -1, 1, -1]        # +, -, +, -
    for k in range(4):
        valid_mask = neighbor_idx[:, k] >= 0
        if not valid_mask.any():
            continue
        valid_idx = np.where(valid_mask)[0]
        delta = (coords[neighbor_idx[valid_idx, k], _dir_axis[k]]
                 - coords[valid_idx, _dir_axis[k]])
        wrong = (delta * _dir_sign[k]) <= 0
        neighbor_idx[valid_idx[wrong], k] = -1

    n_complete = int((neighbor_idx != -1).all(axis=1).sum())
    n_boundary = int((neighbor_idx == -1).any(axis=1).sum())
    logger.info(
        "Cardinal neighbors: %d/%d complete, %d boundary (res=%.2f)",
        n_complete, N, n_boundary, resolution,
    )
    return neighbor_idx, resolution


def _maybe_compile(
    mod: torch.nn.Module, *, mode: str = "reduce-overhead"
) -> torch.nn.Module:
    """Apply torch.compile when available (torch ≥ 2.0); no-op otherwise."""
    if not hasattr(torch, "compile"):
        return mod
    try:
        return torch.compile(mod, mode=mode)
    except Exception:
        return mod


def _pad_batch_to_size(
    b_idx: np.ndarray,
    target_size: int,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor | None]:
    """Pad *b_idx* to *target_size* by repeating the last element.

    Returns ``(padded_idx, valid_mask)``.  When ``len(b_idx) == target_size``
    the function is a no-op and returns ``(b_idx, None)`` to avoid the mask
    allocation on full batches.
    """
    n = len(b_idx)
    if n == target_size:
        return b_idx, None
    pad_n = target_size - n
    padded = np.concatenate([b_idx, np.full(pad_n, b_idx[-1], dtype=b_idx.dtype)])
    valid_mask = torch.zeros(target_size, dtype=torch.bool, device=device)
    valid_mask[:n] = True
    return padded, valid_mask


def _capture_cuda_graph(
    mod: torch.nn.Module,
    sample_inputs: tuple,
) -> torch.nn.Module:
    """Wrap *mod* forward with a CUDA Graph via ``make_graphed_callables``.

    Falls back to the original module when CUDA is unavailable, torch < 2.1
    does not have ``make_graphed_callables``, or capture raises any error.
    """
    if not torch.cuda.is_available():
        return mod
    if not hasattr(torch.cuda, "make_graphed_callables"):
        return mod
    try:
        return torch.cuda.make_graphed_callables(mod, sample_inputs)
    except Exception as exc:
        logger.warning("CUDA Graph capture skipped (%s)", exc)
        return mod



# CU-10a: Trunk parameter prefix set — used by EWC and DDP broadcast.
_TRUNK_KEYS: frozenset[str] = frozenset(
    {"physics_enc", "alpha_emb", "trunk_fusion", "time_embed"}
)


def _init_ddp_process_group(
    rank: int,
    world_size: int,
    backend: str = "nccl",
) -> None:
    """Initialise the default torch.distributed process group for DDP.

    Uses the ``env://`` init method — the launcher (``torch.multiprocessing.spawn``)
    sets ``MASTER_ADDR`` / ``MASTER_PORT`` before forking worker processes.

    Falls back to the ``gloo`` backend when NCCL is unavailable (CPU-only or
    non-CUDA builds) so that unit tests can still exercise the DDP paths.
    """
    import torch.distributed as dist

    if dist.is_initialized():
        return

    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    try:
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
    except Exception as exc:
        logger.warning(
            "DDP: NCCL init failed (%s); falling back to gloo backend", exc
        )
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
        )


def _destroy_ddp_process_group() -> None:
    """Tear down the distributed process group if one was initialised."""
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def _ddp_fold_worker(
    rank: int,
    world_size: int,
    fold_assignments: "list[tuple[int, np.ndarray, np.ndarray]]",
    shared_state: dict,
    oof_preds_shared: "torch.Tensor",
    oof_std_shared: "torch.Tensor",
) -> None:
    """Worker function spawned by ``torch.multiprocessing.spawn`` for DDP folds.

    Each rank processes the fold at ``fold_assignments[rank]``.  Results are
    written back into the pre-allocated shared-memory tensors
    ``oof_preds_shared`` and ``oof_std_shared``.

    Parameters
    ----------
    rank             : worker rank (0 … world_size-1)
    world_size       : total number of worker processes
    fold_assignments : list of ``(fold_idx, train_idx, test_idx)`` tuples;
                       rank ``r`` processes ``fold_assignments[r]``
    shared_state     : serialisable dict produced by ``train_neural_meta``
                       before the fold loop (see ``_build_fold_shared_state``)
    oof_preds_shared : shared-memory 1-D float32 tensor of length N
    oof_std_shared   : shared-memory 1-D float32 tensor of length N
    """
    _init_ddp_process_group(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    try:
        # Move tensors in the shared state to this worker's device.
        # Large tensors start on CPU (shared memory); we move them here so
        # each worker operates on its own GPU copy.
        s = dict(shared_state)
        raw_tensors = s.get("tensors", {})
        s["tensors"] = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in raw_tensors.items()
        }
        if s.get("alpha_targets") is not None:
            s["alpha_targets"] = s["alpha_targets"].to(device)

        if rank < len(fold_assignments):
            fold_idx, train_idx, test_idx = fold_assignments[rank]
            test_out, oof_p, oof_s = _exec_cv_fold(
                fold_idx, train_idx, test_idx, s, device
            )
            oof_preds_shared[test_out] = torch.from_numpy(oof_p)
            oof_std_shared[test_out] = torch.from_numpy(oof_s)
    finally:
        _destroy_ddp_process_group()


def _exec_cv_fold(
    fold_idx: int,
    train_idx: "np.ndarray",
    test_idx: "np.ndarray",
    ss: dict,
    device: "torch.device",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Train one CV fold and return ``(test_idx, oof_preds_slice, oof_std_slice)``.

    All outer-scope state that the fold body needs is packaged in *ss* (the
    shared-state dict produced by ``_build_fold_shared_state`` inside
    ``train_neural_meta``).  The function is module-level so it can be called
    from ``_ddp_fold_worker`` (via ``torch.multiprocessing.spawn``) as well
    as from the sequential fold loop.

    Parameters
    ----------
    fold_idx  : 0-based fold index (for logging)
    train_idx : row indices into the full dataset for this fold's training set
    test_idx  : row indices for the held-out set
    ss        : shared-state dict (see ``_build_fold_shared_state``)
    device    : torch device this fold runs on

    Returns
    -------
    test_idx       : same array as input (passed through for DDP convenience)
    oof_preds_fold : (len(test_idx),) float32 predictions in *normalised* space
    oof_std_fold   : (len(test_idx),) float32 MC-Dropout uncertainty
    """
    from sparc.models.neural_meta import SPARCMetaLearner
    from sparc.models.process_rate_net import ProcessRateNet, SourceTermNet
    from sparc.models.surrogates import (
        DifferentiableGGPGAM,
        DifferentiableGWR,
        DifferentiableGWRF,
    )
    from sparc.training.curriculum import get_lambda_schedule
    from sparc.training.loss import sparc_joint_loss
    from sparc.training.optimizer import (
        build_optimizer,
        build_scheduler,
        spatial_minibatch_sampler,
    )
    from sparc.models.ema_trunk import EMATrunk
    from sparc.models.latent_predictor import LatentPredictor
    from sparc.training.jepa_loss import (
        JEPALossWeights,  # noqa: F401 — imported for type compat
        jepa_curriculum_weight,
        jepa_loss,
        spatial_patch_mask,
    )
    from sparc.models.action_embedding import ActionEmbedding

    # ---- Unpack shared state ----
    tensors                   = ss["tensors"]
    coords                    = ss["coords_np"]
    alpha_targets             = ss["alpha_targets"]
    jepa_pretrained_trunk_state = ss["jepa_pretrained_trunk_state"]
    base_oof_predictions      = ss["base_oof_predictions"]
    hidden_dim                = ss["hidden_dim"]
    dropout                   = ss["dropout"]
    n_heads                   = ss["n_heads"]
    max_neighbors             = ss["max_neighbors"]
    siren_omega               = ss["siren_omega"]
    thresholds                = ss["thresholds"]
    thresholds_norm           = ss["thresholds_norm"]
    time_embed_dim            = ss["time_embed_dim"]
    n_base                    = ss["n_base"]
    n_physics                 = ss["n_physics"]
    n_physics_extended        = ss["n_physics_extended"]
    d_spatial                 = ss["d_spatial"]
    resolution                = ss["resolution"]
    y_mean                    = ss["y_mean"]
    y_std                     = ss["y_std"]
    pr_input_dim              = ss["pr_input_dim"]
    pr_col_idxs               = ss["pr_col_idxs"]
    prior_mean                = ss["prior_mean"]
    pr_cfg                    = ss["pr_cfg"]
    n_treatments              = ss["n_treatments"]
    material_priors           = ss["material_priors"]
    pr_pretrain_epochs        = ss["pr_pretrain_epochs"]
    n_epochs                  = ss["n_epochs"]
    pretrain_epochs           = ss["pretrain_epochs"]
    lr                        = ss["lr"]
    clip_norm                 = ss["clip_norm"]
    warmup_epochs             = ss["warmup_epochs"]
    ramp_epochs               = ss["ramp_epochs"]
    batch_size                = ss["batch_size"]
    target_lambdas            = ss["target_lambdas"]
    feature_names             = ss["feature_names"]
    gwrf_k                    = ss["gwrf_k"]
    predictor_bandwidths      = ss["predictor_bandwidths"]
    predictor_kernel_field    = ss["predictor_kernel_field"]
    jepa_enable               = ss["jepa_enable"]
    jepa_weights              = ss["jepa_weights"]
    jepa_mask_ratio           = ss["jepa_mask_ratio"]
    jepa_spatial_patch        = ss["jepa_spatial_patch"]
    jepa_n_patches            = ss["jepa_n_patches"]
    jepa_lambda               = ss["jepa_lambda"]
    jepa_ema_tau_start        = ss["jepa_ema_tau_start"]
    jepa_ema_tau_end          = ss["jepa_ema_tau_end"]
    jepa_warmup_steps         = ss["jepa_warmup_steps"]
    jepa_curric_start         = ss["jepa_curric_start"]
    jepa_curric_end           = ss["jepa_curric_end"]
    jepa_lambda_scenario      = ss["jepa_lambda_scenario"]
    jepa_lambda_latent_pde    = ss["jepa_lambda_latent_pde"]
    jepa_scenario_perturb_std = ss["jepa_scenario_perturb_std"]
    jepa_predictor_blocks     = ss["jepa_predictor_blocks"]
    jepa_predictor_film       = ss["jepa_predictor_film"]
    jepa_action_dim           = ss["jepa_action_dim"]
    _ewc_active               = ss["ewc_active"]
    _ewc_lambda               = ss["ewc_lambda"]
    _cont_fisher              = ss["cont_fisher"]
    _cont_optimal             = ss["cont_optimal"]
    _ot_active                = ss["ot_active"]
    _ot_lambda                = ss["ot_lambda"]
    _coreset_acts             = ss["coreset_acts"]
    _use_amp                  = ss["use_amp"]
    _use_cuda_graphs          = ss["use_cuda_graphs"]
    n_folds                   = ss["n_folds"]

    # ---- Per-fold AMP scaler (fresh per-process instance) ----
    _scaler = torch.cuda.amp.GradScaler(enabled=_use_amp)

    # ---- OT trunk hook (mutable; created fresh per fold) ----
    _trunk_acts_cv_bucket: list = []

    def _make_trunk_hook(bucket: list):
        def _hook(module, inp, out):
            bucket.append(out.detach())
        return _hook

    # ---- Conditionally import EWC / OT functions ----
    _ewc_penalty = None
    if _ewc_active:
        from sparc.training.ewc import ewc_penalty as _ewc_penalty  # type: ignore[assignment]

    _wta = None
    if _ot_active:
        from sparc.training.ewc import wasserstein_trunk_alignment as _wta  # type: ignore[assignment]

    # =====================================================================
    # Fold body — extracted from train_neural_meta (originally lines 1667–2342)
    # =====================================================================
    _fold_t0 = _time.perf_counter()
    logger.info(
        "Fold %d / %d  (%d train, %d test)",
        fold_idx + 1, n_folds, len(train_idx), len(test_idx),
    )

    # ---- Slice training data ----
    train_physics     = tensors["physics_feats"][train_idx]
    train_physics_ext = tensors["physics_feats_extended"][train_idx]
    train_spatial     = tensors["X_spatial"][train_idx]
    train_coords      = tensors["coords"][train_idx]
    train_y           = tensors["y"][train_idx]
    train_h           = tensors["h_field"][train_idx] if tensors["h_field"] is not None else None
    train_water_mask  = tensors["water_mask"][train_idx] if tensors.get("water_mask") is not None else None
    train_alpha_targets = alpha_targets[train_idx] if alpha_targets is not None else None

    # ---- Build fold-local KNN for spatial attention + GWRF kernel ----
    logger.info("  Building KNN index (k=%d) ...", max_neighbors)
    fold_knn_np, fold_knn_dists_np = _build_knn_index(
        coords[train_idx], max_neighbors, return_dists=True,
    )
    fold_knn       = torch.tensor(fold_knn_np,       dtype=torch.long,    device=device)
    fold_knn_dists = torch.tensor(fold_knn_dists_np, dtype=torch.float32, device=device)
    logger.info("  KNN ready.")

    # ---- Build fold-local cardinal neighbors for physics loss ----
    fold_cardinal_np, _ = _build_cardinal_neighbors(
        coords[train_idx], resolution=resolution,
    )
    fold_cardinal = torch.tensor(fold_cardinal_np, dtype=torch.long, device=device)

    # ---- Instantiate models ----
    process_net = ProcessRateNet(
        n_inputs=pr_input_dim,
        domain_config={
            "name": pr_cfg.get("name", "rate"),
            "units": pr_cfg.get("units", ""),
            "bounds": pr_cfg.get("bounds", [0.0, 1.0]),
            "prior_mean": pr_cfg.get("prior_mean", 0.5),
        },
        n_treatments=n_treatments,
    ).to(device)

    source_net = SourceTermNet(n_inputs=n_physics).to(device)

    model = SPARCMetaLearner(
        n_base_models=n_base,
        n_physics_features=n_physics_extended,
        d_spatial=d_spatial,
        hidden_dim=hidden_dim,
        dropout=dropout,
        thresholds=thresholds,
        n_heads=n_heads,
        max_neighbors=max_neighbors,
        siren_omega=siren_omega,
        init_bandwidth=float(predictor_bandwidths.mean()) if predictor_bandwidths is not None else 1000.0,
        time_embed_dim=time_embed_dim,
    ).to(device)

    # OT trunk alignment: register forward hook
    if _ot_active:
        _trunk_acts_cv_bucket.clear()
        model.trunk_fusion.register_forward_hook(_make_trunk_hook(_trunk_acts_cv_bucket))

    # Transfer JEPA-pretrained trunk weights if available
    if jepa_pretrained_trunk_state is not None:
        model.load_state_dict(jepa_pretrained_trunk_state, strict=False)
        logger.info("  Fold %d: transferred JEPA-pretrained trunk", fold_idx + 1)

    # ---- Differentiable surrogates ----
    surrogates = {
        "gwr": DifferentiableGWR(
            n_vars=n_physics, n_spatial_features=d_spatial,
            hidden_dim=hidden_dim,
            bandwidths=predictor_bandwidths,
            kernel_field=predictor_kernel_field,
            feature_names=feature_names,
        ).to(device),
        "gwrf": DifferentiableGWRF(
            n_vars=n_physics, n_spatial_features=d_spatial,
            hidden_dim=hidden_dim,
            kernel_field=predictor_kernel_field,
            feature_names=feature_names,
        ).to(device),
        "ggpgam": DifferentiableGGPGAM(
            n_vars=n_physics, n_spatial_features=d_spatial,
            hidden_dim=min(hidden_dim, 64),
        ).to(device),
    }

    # E-Perf-A: compile forward passes when torch >= 2.0
    model      = _maybe_compile(model)
    process_net = _maybe_compile(process_net)
    source_net  = _maybe_compile(source_net)
    for _k in list(surrogates):
        surrogates[_k] = _maybe_compile(surrogates[_k])

    # ---- Per-component optimizer + warmup scheduler ----
    optimizer = build_optimizer(
        model, process_net, surrogates, base_lr=lr,
        source_term_net=source_net,
    )
    scheduler = build_scheduler(optimizer, n_epochs, warmup_epochs=warmup_epochs)

    # ---- JEPA target trunk + latent predictor (Phase 1) ----
    ema_trunk:        "EMATrunk | None"    = None
    latent_predictor: "LatentPredictor | None" = None
    action_embed:     "ActionEmbedding | None" = None
    jepa_optimizer:   "torch.optim.Optimizer | None" = None
    if jepa_enable:
        ema_trunk = EMATrunk(
            model,
            tau_start=jepa_ema_tau_start,
            tau_end=jepa_ema_tau_end,
            warmup_steps=jepa_warmup_steps,
        ).to(device)
        use_action = (jepa_lambda_scenario > 0.0 or jepa_predictor_blocks > 1)
        action_dim = jepa_action_dim if use_action else 0
        latent_predictor = LatentPredictor(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            dropout=dropout,
            n_blocks=jepa_predictor_blocks,
            film=jepa_predictor_film,
        ).to(device)
        jepa_params = list(latent_predictor.parameters())
        if use_action:
            action_embed = ActionEmbedding(
                treatment_vocab=feature_names,
                embed_dim=action_dim,
            ).to(device)
            jepa_params += list(action_embed.parameters())
        jepa_optimizer = torch.optim.AdamW(jepa_params, lr=lr)

    # ---- Build per-fold V1 base-model targets (normalised) ----
    _fold_base_targets: "dict[str, torch.Tensor] | None" = None
    if base_oof_predictions is not None:
        _fold_base_targets = {}
        for sname in ("gwr", "gwrf", "ggpgam"):
            if sname in base_oof_predictions:
                _raw  = base_oof_predictions[sname][train_idx]
                _norm = (_raw - y_mean) / y_std
                _fold_base_targets[sname] = torch.tensor(
                    _norm, dtype=torch.float32, device=device,
                )
        if not _fold_base_targets:
            _fold_base_targets = None

    # ---- Surrogate pre-training (MSE only, no physics/meta) ----
    if pretrain_epochs > 0:
        logger.info(
            "  Pre-training surrogates (%d epochs, MSE only)...", pretrain_epochs,
        )
        _pretrain_surrogates(
            surrogates, train_physics, train_spatial, train_y,
            n_epochs=pretrain_epochs, lr=lr,
            base_targets=_fold_base_targets,
        )

    # ---- Surrogate fidelity targets ----
    with torch.no_grad():
        _surr_targets: dict = {}
        for sname, surr in surrogates.items():
            if _fold_base_targets and sname in _fold_base_targets:
                _surr_targets[sname] = _fold_base_targets[sname].detach()
                r2_vs = "vs V1 base model (anchor)"
            else:
                surr.eval()
                if sname == "gwr":
                    _out, _ = surr(train_physics, train_spatial)
                else:
                    _out = surr(train_physics, train_spatial)
                _surr_targets[sname] = _out.squeeze().detach()
                r2_vs = "vs y (no V1 target)"
            ss_res = ((_surr_targets[sname] - train_y) ** 2).sum()
            ss_tot = ((train_y - train_y.mean()) ** 2).sum()
            r2 = (1.0 - ss_res / (ss_tot + 1e-12)).item()
            logger.info("  Surrogate %s fidelity R²=%.4f %s", sname, r2, r2_vs)

    # ---- Validate surrogates before joint training ----
    if _fold_base_targets:
        from sparc.models.surrogates import validate_surrogates
        validation_results = validate_surrogates(
            surrogates=surrogates,
            true_predictions=_fold_base_targets,
            X=train_physics,
            spatial_features=train_spatial,
            threshold=0.85,
        )
        failed = [name for name, res in validation_results.items() if not res["passed"]]
        if failed:
            raise RuntimeError(
                f"Surrogate validation failed for: {failed}. "
                "Joint training aborted — fix surrogate architecture before proceeding."
            )
        logger.info("  All surrogates passed validation (R² >= 0.85)")

    # ---- ProcessRateNet pre-training toward mixture prior ----
    pr_pretrain_epochs_eff = pr_pretrain_epochs
    if pr_pretrain_epochs_eff > 0 and material_priors:
        logger.info(
            "  Pre-training ProcessRateNet with land-cover classification "
            "(%d epochs, %d materials, monotonicity reg)...",
            pr_pretrain_epochs_eff, len(material_priors),
        )
        pr_final_loss = _pretrain_process_rate_net(
            process_net=process_net,
            physics_feats=train_physics,
            feature_names=feature_names,
            pr_col_idxs=pr_col_idxs,
            material_priors=material_priors,
            n_epochs=pr_pretrain_epochs_eff,
            device=device,
            feat_mean=tensors["feat_mean"],
            feat_std=tensors["feat_std"],
        )
        logger.info("  ProcessRateNet pre-training done — final loss=%.6f", pr_final_loss)

    # ---- Training loop ----
    model.train()
    process_net.train()
    for _sv in surrogates.values():
        _sv.train()

    N_train = len(train_idx)
    use_minibatch = N_train > batch_size * 2

    if use_minibatch:
        logger.info(
            "  Using spatial minibatching: N=%d, batch_size=%d",
            N_train, batch_size,
        )

    for epoch in range(n_epochs):
        lambdas = get_lambda_schedule(
            epoch, target_lambdas,
            warmup_end=warmup_epochs,
            ramp_end=ramp_epochs,
        )

        # ---- Curriculum transition markers ----
        if epoch == 0:
            logger.info("[CURRICULUM] Stage A: Representation Warmup")
        elif epoch == warmup_epochs:
            logger.info("[CURRICULUM] Stage B: Physics Activation")
        elif epoch == ramp_epochs:
            logger.info("[CURRICULUM] Stage C: Joint Optimization")

        if use_minibatch:
            batches = list(spatial_minibatch_sampler(
                coords[train_idx],
                fold_cardinal_np,
                batch_size=batch_size,
                n_batches=max(1, N_train // batch_size),
            ))
        else:
            batches = [np.arange(N_train)]

        epoch_loss      = 0.0
        epoch_components: dict = {}
        n_batch_points  = 0

        for b_idx in batches:
            # CU-9b: pad last (short) batch to static size for CUDA Graph
            _valid_mask: "torch.Tensor | None" = None
            if _use_cuda_graphs and len(b_idx) < batch_size:
                b_idx, _valid_mask = _pad_batch_to_size(b_idx, batch_size, device)

            b_physics     = train_physics[b_idx]
            b_physics_ext = train_physics_ext[b_idx]
            b_spatial     = train_spatial[b_idx]
            b_coords      = train_coords[b_idx]
            b_y           = train_y[b_idx]
            b_h           = train_h[b_idx] if train_h is not None else resolution
            b_water_mask  = train_water_mask[b_idx] if train_water_mask is not None else None
            b_surr_targets = {k: v[b_idx] for k, v in _surr_targets.items()}

            # Remap KNN and cardinal indices to batch-local space
            b_knn = _remap_indices_to_local(
                np.arange(N_train), b_idx, fold_knn[b_idx],
            )
            b_cardinal = _remap_indices_to_local(
                np.arange(N_train), b_idx, fold_cardinal[b_idx],
            )

            # GWRF kernel
            b_gwrf_knn = _remap_indices_to_local(
                np.arange(N_train), b_idx,
                fold_knn[b_idx][:, :gwrf_k],
            )
            b_gwrf_dists = fold_knn_dists[b_idx][:, :gwrf_k]

            # Forward: surrogates → meta-learner (inside AMP context)
            _fwd_amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=_use_amp)
            _fwd_amp.__enter__()
            surrogate_preds, surrogate_std = _forward_surrogates(
                surrogates, b_physics, b_spatial,
                knn_index=b_gwrf_knn, knn_dists=b_gwrf_dists,
            )
            base_input = torch.stack(surrogate_preds, dim=1)

            pr_input   = b_physics[:, pr_col_idxs]
            alpha_full = process_net(pr_input)
            alpha = (
                alpha_full
                if alpha_full.shape[-1] == 1
                else alpha_full.mean(dim=-1, keepdim=True)
            )
            alpha_prior = torch.full_like(alpha, prior_mean)

            b_source = source_net(b_physics).squeeze(-1)

            T_pred, exceedance, _ = model(
                base_preds=base_input,
                physics_feats=b_physics_ext,
                X_spatial=b_spatial,
                coords=b_coords,
                knn_index=b_knn,
                alpha=alpha,
                surrogate_std=surrogate_std,
                knn_dists=fold_knn_dists[b_idx],
                time_idx=tensors["time_idx"][b_idx] if tensors["time_idx"] is not None else None,
            )
            _fwd_amp.__exit__(None, None, None)
            if _use_amp:
                T_pred = T_pred.float()
                if isinstance(exceedance, torch.Tensor):
                    exceedance = exceedance.float()

            # ---- JEPA Phase 1 ----
            b_jepa_h_pred            = None
            b_jepa_h_target          = None
            b_jepa_scenario_h_pred   = None
            b_jepa_scenario_h_target = None
            b_jepa_latent_h          = None
            b_jepa_latent_neighbor   = None
            b_jepa_latent_alpha      = None
            jepa_lambda_eff             = 0.0
            jepa_lambda_scenario_eff    = 0.0
            jepa_lambda_latent_pde_eff  = 0.0
            if jepa_enable:
                curric = jepa_curriculum_weight(
                    epoch,
                    warmup_start=jepa_curric_start,
                    warmup_end=jepa_curric_end,
                )
                jepa_lambda_eff            = jepa_lambda * curric
                jepa_lambda_scenario_eff   = jepa_lambda_scenario * curric
                jepa_lambda_latent_pde_eff = jepa_lambda_latent_pde * curric

                h_state_online: "torch.Tensor | None" = None

                if jepa_lambda_eff > 0.0:
                    if jepa_spatial_patch:
                        _patch_mask = spatial_patch_mask(
                            b_coords, mask_ratio=jepa_mask_ratio,
                            n_patches=jepa_n_patches,
                        )
                        b_phys_ctx = b_physics_ext.clone()
                        b_phys_ctx[_patch_mask] = 0.0
                        channel_mask = None
                    else:
                        b_phys_ctx = b_physics_ext
                        n_phys_feat = b_physics_ext.shape[-1]
                        keep_p = max(0.0, 1.0 - jepa_mask_ratio)
                        channel_mask = (
                            torch.rand(n_phys_feat, device=device) < keep_p
                        ).to(b_physics_ext.dtype)
                    h_context = model.encode(
                        physics_feats=b_phys_ctx,
                        alpha=alpha,
                        channel_mask=channel_mask,
                    )
                    b_jepa_h_pred   = latent_predictor(h_context)
                    b_jepa_h_target = ema_trunk.encode_target(
                        physics_feats=b_physics_ext, alpha=alpha,
                    )

                # ---- JEPA Phase 2: scenario distillation ----
                if (
                    jepa_lambda_scenario_eff > 0.0
                    and action_embed is not None
                    and len(feature_names) > 0
                ):
                    treat_pos = int(torch.randint(
                        0, len(feature_names), (1,), device=device,
                    ).item())
                    delta_x_val = float(
                        torch.randn((), device=device).item()
                        * jepa_scenario_perturb_std
                    )

                    b_phys_perturbed = b_physics_ext.clone()
                    b_phys_perturbed[:, treat_pos] = (
                        b_phys_perturbed[:, treat_pos] + delta_x_val
                    )

                    if h_state_online is None:
                        h_state_online = model.encode(
                            physics_feats=b_physics_ext, alpha=alpha,
                        )

                    n_pts = b_physics_ext.shape[0]
                    treat_idx = torch.full(
                        (n_pts,), treat_pos, dtype=torch.long, device=device,
                    )
                    dx = torch.full(
                        (n_pts,), delta_x_val,
                        dtype=b_physics_ext.dtype, device=device,
                    )
                    dt = torch.zeros_like(dx)
                    a_emb = action_embed(treat_idx, dx, dt)

                    b_jepa_scenario_h_pred   = latent_predictor(h_state_online, a_emb)
                    b_jepa_scenario_h_target = ema_trunk.encode_target(
                        physics_feats=b_phys_perturbed, alpha=alpha,
                    )

                # ---- JEPA Phase 2: latent PDE consistency ----
                if (
                    jepa_lambda_latent_pde_eff > 0.0
                    and b_cardinal is not None
                    and b_cardinal.numel() > 0
                ):
                    if h_state_online is None:
                        h_state_online = model.encode(
                            physics_feats=b_physics_ext, alpha=alpha,
                        )
                    b_jepa_latent_h        = h_state_online
                    b_jepa_latent_neighbor = b_cardinal
                    b_jepa_latent_alpha    = alpha

            total_loss, components = sparc_joint_loss(
                T_pred=T_pred,
                exceedance_preds=exceedance,
                y_true=b_y,
                thresholds=thresholds_norm,
                alpha=alpha,
                alpha_prior=alpha_prior,
                neighbor_idx=b_cardinal,
                source_term=b_source,
                resolution=resolution,
                surrogate_preds=surrogate_preds,
                surrogate_targets=[
                    b_surr_targets["gwr"],
                    b_surr_targets["gwrf"],
                    b_surr_targets["ggpgam"],
                ],
                lambda_physics=lambdas.get("physics", 0.0),
                lambda_smooth=lambdas.get("smooth", 0.0),
                lambda_alpha_smooth=lambdas.get("alpha_smooth", 0.0),
                lambda_prior=lambdas.get("prior", 1.0),
                lambda_base=lambdas.get("base", 0.0),
                lambda_neighbor=lambdas.get("neighbor", 0.0),
                epoch=epoch,
                h_field=b_h,
                lambda_pde=lambdas.get("pde", 0.0),
                lambda_bc=lambdas.get("bc", 0.0),
                water_mask=b_water_mask,
                T_water=tensors.get("T_water"),
                jepa_h_pred=b_jepa_h_pred,
                jepa_h_target=b_jepa_h_target,
                jepa_weights=jepa_weights,
                lambda_jepa=jepa_lambda_eff,
                jepa_scenario_h_pred=b_jepa_scenario_h_pred,
                jepa_scenario_h_target=b_jepa_scenario_h_target,
                lambda_jepa_scenario=jepa_lambda_scenario_eff,
                jepa_latent_h=b_jepa_latent_h,
                jepa_latent_neighbor_idx=b_jepa_latent_neighbor,
                jepa_latent_alpha=b_jepa_latent_alpha,
                lambda_jepa_latent_pde=jepa_lambda_latent_pde_eff,
                sparse_laplacian=tensors.get("sparse_laplacian"),
                valid_laplacian_mask=tensors.get("valid_laplacian_mask"),
                sheaf_delta=tensors.get("sheaf_delta"),
                valid_mask=_valid_mask,
            )

            # Variance penalty on w(s)
            if hasattr(model, "_last_w_source") and model._last_w_source is not None:
                total_loss = total_loss - 0.01 * model._last_w_source.var()

            # Alpha supervision loss
            if train_alpha_targets is not None:
                total_loss = total_loss + 0.1 * torch.nn.functional.mse_loss(
                    alpha, train_alpha_targets[b_idx]
                )

            # EWC penalty
            if _ewc_active and _ewc_penalty is not None:
                total_loss = total_loss + _ewc_lambda * _ewc_penalty(
                    model, _cont_fisher, _cont_optimal, _TRUNK_KEYS,
                )

            # OT trunk alignment
            if _ot_active and _wta is not None:
                _trunk_acts_cv = _trunk_acts_cv_bucket.pop() if _trunk_acts_cv_bucket else None
                if _trunk_acts_cv is not None:
                    ot_loss    = _wta(_trunk_acts_cv, _coreset_acts.to(_trunk_acts_cv.device))
                    total_loss = total_loss + _ot_lambda * ot_loss

            optimizer.zero_grad()
            if jepa_optimizer is not None:
                jepa_optimizer.zero_grad()
            _scaler.scale(total_loss).backward()

            all_params = (
                list(model.parameters()) + list(process_net.parameters())
            )
            for _sv2 in surrogates.values():
                all_params.extend(_sv2.parameters())
            if latent_predictor is not None:
                all_params.extend(latent_predictor.parameters())
            if action_embed is not None:
                all_params.extend(action_embed.parameters())
            _scaler.unscale_(optimizer)
            if jepa_optimizer is not None:
                _scaler.unscale_(jepa_optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, clip_norm)

            _scaler.step(optimizer)
            if jepa_optimizer is not None:
                _scaler.step(jepa_optimizer)
            _scaler.update()

            _any_jepa = (
                jepa_lambda_eff > 0.0
                or jepa_lambda_scenario_eff > 0.0
                or jepa_lambda_latent_pde_eff > 0.0
            )
            if ema_trunk is not None and _any_jepa:
                ema_trunk.update(model)

            bsize = int(_valid_mask.sum().item()) if _valid_mask is not None else len(b_idx)
            epoch_loss     += total_loss.item() * bsize
            n_batch_points += bsize
            for k, v in components.items():
                epoch_components[k] = epoch_components.get(k, 0) + v * bsize

        scheduler.step()

        epoch_loss /= max(n_batch_points, 1)
        for k in epoch_components:
            epoch_components[k] /= max(n_batch_points, 1)

        logger.info(
            "  Epoch %d/%d  loss=%.4f  "
            "[mse=%.3f phys=%.3f nbr=%.3f ce=%.3f "
            "pde=%.3f bc=%.3f prior=%.3f base=%.3f]  (%.1fs)",
            epoch + 1, n_epochs, epoch_loss,
            epoch_components.get("mse", 0),
            epoch_components.get("physics", 0),
            epoch_components.get("neighborhood", 0),
            epoch_components.get("cross_entropy", 0),
            epoch_components.get("pde_total", 0),
            epoch_components.get("bc_total", 0),
            epoch_components.get("alpha_prior", 0),
            epoch_components.get("surrogate", 0),
            _time.perf_counter() - _fold_t0,
        )

        # Convergence tracking
        if not hasattr(model, "_loss_history"):
            model._loss_history = []
        model._loss_history.append(epoch_loss)
        if len(model._loss_history) > 10:
            old_loss = model._loss_history[-11]
            rel_improvement = (old_loss - epoch_loss) / max(abs(old_loss), 1e-8)
            if rel_improvement < 0.001:
                logger.info("[CONVERGENCE] converged")
            elif rel_improvement < 0.01:
                logger.info("[CONVERGENCE] converging")

    logger.info(
        "  Fold %d training done in %.1fs",
        fold_idx + 1, _time.perf_counter() - _fold_t0,
    )

    # ---- OOF prediction (surrogates → meta-learner) ----
    model.eval()
    process_net.eval()
    for _sv3 in surrogates.values():
        _sv3.eval()

    full_knn_np, full_knn_dists_np = _build_knn_index(
        coords, max_neighbors, return_dists=True,
    )
    full_knn       = torch.tensor(full_knn_np,       dtype=torch.long,    device=device)
    full_knn_dists = torch.tensor(full_knn_dists_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        full_surr_preds, full_surr_std = _forward_surrogates(
            surrogates, tensors["physics_feats"], tensors["X_spatial"],
            knn_index=full_knn[:, :gwrf_k],
            knn_dists=full_knn_dists[:, :gwrf_k],
        )
        full_base_input = torch.stack(full_surr_preds, dim=1)
        full_pr         = tensors["physics_feats"][:, pr_col_idxs]
        full_alpha_full = process_net(full_pr)
        full_alpha = (
            full_alpha_full
            if full_alpha_full.shape[-1] == 1
            else full_alpha_full.mean(dim=-1, keepdim=True)
        )

    mean_pred, std_pred = model.predict_with_uncertainty(
        base_preds=full_base_input,
        physics_feats=tensors["physics_feats_extended"],
        X_spatial=tensors["X_spatial"],
        coords=tensors["coords"],
        knn_index=full_knn,
        alpha=full_alpha,
        knn_dists=full_knn_dists,
        n_samples=50,
    )

    oof_p_fold = mean_pred[test_idx].cpu().numpy()
    oof_s_fold = std_pred[test_idx].cpu().numpy()

    # CU-5: Release fold models immediately to free CUDA allocator memory.
    del model, surrogates, process_net, source_net
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return test_idx, oof_p_fold, oof_s_fold


def _forward_surrogates(
    surrogates: dict[str, torch.nn.Module],
    physics_feats: torch.Tensor,
    spatial_feats: torch.Tensor,
    knn_index: torch.Tensor | None = None,
    knn_dists: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    Run all three differentiable surrogates.

    Parameters
    ----------
    knn_index : (N, K) int tensor, optional
        Passed to GWRF for distance-weighted kernel predictions.
    knn_dists : (N, K) float tensor, optional
        Distances to K nearest neighbors for GWRF kernel.

    Returns
    -------
    preds_list : list of (N,) tensors [gwr_pred, gwrf_pred, ggpgam_pred]
    surrogate_std : (N, 1) tensor
        Std of the three surrogate predictions — epistemic disagreement
        proxy used by the SpatialGatingHead in SPARCMetaLearner.
    """
    gwr_pred, _beta = surrogates["gwr"](physics_feats, spatial_feats)
    gwrf_pred = surrogates["gwrf"](
        physics_feats, spatial_feats,
        neighbor_idx=knn_index, neighbor_dists=knn_dists,
    )
    ggpgam_pred = surrogates["ggpgam"](physics_feats, spatial_feats)
    preds_list = [gwr_pred, gwrf_pred, ggpgam_pred]
    surrogate_std = torch.stack(preds_list, dim=-1).std(dim=-1, keepdim=True)  # (N, 1)
    return preds_list, surrogate_std


def _pretrain_surrogates(
    surrogates: dict[str, torch.nn.Module],
    physics_feats: torch.Tensor,
    spatial_feats: torch.Tensor,
    y: torch.Tensor,
    n_epochs: int = 50,
    lr: float = 1e-3,
    base_targets: dict[str, torch.Tensor] | None = None,
) -> None:
    """
    Pre-train each surrogate independently with MSE to approximate the
    corresponding V1 base model.  When *base_targets* is provided, each
    surrogate trains against the V1 OOF predictions for that model
    (GWR, GWRF, GGPGAM). Falls back to *y* when V1 outputs are
    unavailable.
    """
    import torch.nn.functional as F

    for name, surrogate in surrogates.items():
        # Use V1 base-model OOF predictions as target when available
        target = base_targets[name] if base_targets and name in base_targets else y
        surrogate.train()
        optimizer = torch.optim.AdamW(surrogate.parameters(), lr=lr)
        for epoch in range(n_epochs):
            if name == "gwr":
                pred, _ = surrogate(physics_feats, spatial_feats)
            elif name == "gwrf":
                pred = surrogate(physics_feats, spatial_feats)
            else:  # ggpgam
                pred = surrogate(physics_feats, spatial_feats)
            loss = F.mse_loss(pred.squeeze(), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log final pre-train quality
        with torch.no_grad():
            if name == "gwr":
                pred, _ = surrogate(physics_feats, spatial_feats)
            elif name == "gwrf":
                pred = surrogate(physics_feats, spatial_feats)
            else:
                pred = surrogate(physics_feats, spatial_feats)
            pred = pred.squeeze()
            ss_res = ((pred - target) ** 2).sum()
            ss_tot = ((target - target.mean()) ** 2).sum()
            r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        lbl = f"vs {name}" if (base_targets and name in base_targets) else "vs y"
        logger.info(
            "  Surrogate pretrain %s: R²=%.4f %s (%d epochs)",
            name, r2.item(), lbl, n_epochs,
        )


def _prepare_tensors(
    y: np.ndarray,
    coords: np.ndarray,
    feature_matrix: np.ndarray,
    config: dict,
    device: torch.device,
) -> dict[str, Any]:
    """Convert numpy arrays to tensors, build spatial encoding, KNN, and cardinal neighbors."""
    from sparc.features.sinusoidal_encoding import SinusoidalSpatialEncoding

    neural_cfg = config.get("models", {}).get("neural", {})
    n_freq = neural_cfg.get("sinusoidal_frequencies", 64)
    max_neighbors = neural_cfg.get("max_neighbors", 128)

    # Sinusoidal spatial encoding
    encoder = SinusoidalSpatialEncoding(n_frequencies=n_freq)
    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)
    X_spatial = encoder(coords_t)

    # KNN index (for spatial attention) + distances (for GWRF kernel)
    knn_idx_np, knn_dist_np = _build_knn_index(
        coords, max_neighbors, return_dists=True,
    )
    knn_index = torch.tensor(knn_idx_np, dtype=torch.long, device=device)
    knn_dists = torch.tensor(knn_dist_np, dtype=torch.float32, device=device)

    # Cardinal neighbor index (for physics Laplacian)
    resolution = config.get("training", {}).get("resolution", None)
    cardinal_np, detected_res = _build_cardinal_neighbors(coords, resolution=resolution)
    cardinal_idx = torch.tensor(cardinal_np, dtype=torch.long, device=device)

    # Physics features — standardise to zero-mean / unit-variance so that
    # surrogate linear layers (especially DifferentiableGWR: y = X·β + β₀)
    # see O(1) inputs matching the z-normalised targets.
    feat_mean = feature_matrix.mean(axis=0).astype(np.float32)
    feat_std  = feature_matrix.std(axis=0).astype(np.float32)
    feat_std[feat_std == 0] = 1.0  # guard constant columns
    feature_matrix_scaled = (feature_matrix - feat_mean) / feat_std
    physics_t = torch.tensor(feature_matrix_scaled, dtype=torch.float32, device=device)

    # ---- V3: Compute per-predictor spatial derivatives ----
    feature_names = config.get("_feature_names", [])
    physics_cfg = config.get("physics", {})
    use_derivatives = physics_cfg.get("use_input_derivatives", True)

    deriv_feats = None
    deriv_names: list[str] = []
    h_field = None

    if use_derivatives and cardinal_idx is not None:
        from sparc.physics.pde_operators import estimate_local_spacing
        from sparc.physics.input_derivatives import (
            compute_predictor_derivatives,
            normalize_derivatives,
        )

        spacing_mode = physics_cfg.get("spacing_mode", "local")
        h_field = estimate_local_spacing(coords_t, cardinal_idx, spacing=spacing_mode)

        raw_derivs, deriv_names = compute_predictor_derivatives(
            physics_t, cardinal_idx, h_field, feature_names,
        )
        deriv_feats, deriv_means, deriv_stds = normalize_derivatives(raw_derivs)

        # Concatenate: physics_feats_extended = [original | derivatives]
        physics_extended = torch.cat([physics_t, deriv_feats], dim=1)
        logger.info(
            "Physics features extended: %d original + %d derivatives = %d total",
            physics_t.shape[1], deriv_feats.shape[1], physics_extended.shape[1],
        )
    else:
        physics_extended = physics_t
        logger.info("Input derivatives disabled — using %d original physics features", physics_t.shape[1])

    # Target — z-score normalise so surrogates' near-zero init is correct
    y_mean = float(np.mean(y))
    y_std = float(np.std(y)) or 1.0  # guard against constant target
    y_norm = (y - y_mean) / y_std
    y_t = torch.tensor(y_norm, dtype=torch.float32, device=device)

    # ---- Water mask: Distance_from_water_m < 50 m (Dirichlet BC) ----
    water_mask = None
    T_water = None
    if "Distance_from_water_m" in feature_names:
        idx_water = feature_names.index("Distance_from_water_m")
        water_mask = torch.tensor(
            feature_matrix[:, idx_water] < 50.0,
            dtype=torch.bool, device=device,
        )
        n_water = int(water_mask.sum().item())
        if n_water > 0:
            T_water = float(y_norm[water_mask.cpu().numpy()].mean())
            logger.info(
                "Water-adjacent BC mask: %d points (%.1f%%), T_water=%.4f (normalised)",
                n_water, 100 * n_water / len(y), T_water,
            )
        else:
            water_mask = None
            logger.info("No water-adjacent points found (Distance_from_water_m < 50)")

    # ---- Temporal snapshot indices (for time embedding) ----
    time_idx = None
    tcfg = config.get("temporal", {})
    if tcfg.get("snapshots"):
        from sparc.data.temporal import get_snapshot_time_indices
        import pandas as pd
        # Build a minimal DataFrame with enough length for the index builder.
        # The snapshot_column path requires the real DataFrame; single_snapshot
        # only needs len(df). Both paths are guarded by tcfg.get("enabled").
        _t_idx_np = get_snapshot_time_indices(
            pd.DataFrame({"_n": range(len(y))}), config,
        )
        if _t_idx_np is not None:
            time_idx = torch.tensor(_t_idx_np, dtype=torch.long, device=device)
            logger.info(
                "Temporal snapshot indices built: %d points, values %s",
                len(_t_idx_np), set(_t_idx_np.tolist()),
            )

    # ---- Precomputed sparse Laplacian (full-N, constant across epochs) ----
    _sparse_lap = None
    _valid_lap_mask = None
    if h_field is not None and cardinal_idx is not None:
        from sparc.physics.pde_operators import build_sparse_laplacian
        _sparse_lap, _valid_lap_mask = build_sparse_laplacian(cardinal_idx, h_field)
        logger.info(
            "Precomputed sparse Laplacian: N=%d, M=%d valid rows",
            cardinal_idx.shape[0], int(_valid_lap_mask.sum().item()),
        )

    # ---- Sheaf Laplacian δ⁰ (full-N, constant — Term 11) ----
    _sheaf_delta = None
    if knn_index is not None and knn_index.shape[0] > 0:
        try:
            from sparc.physics.pde_operators import build_sheaf_laplacian
            _sheaf_delta = build_sheaf_laplacian(knn_index, stalk_dim=2)
            logger.info(
                "Precomputed Sheaf coboundary δ⁰: N=%d, K=%d, shape=%s",
                knn_index.shape[0], knn_index.shape[1], tuple(_sheaf_delta.shape),
            )
        except Exception as _sheaf_build_exc:
            logger.debug("Sheaf Laplacian build failed (non-fatal): %s", _sheaf_build_exc)

    return {
        "physics_feats": physics_t,
        "physics_feats_extended": physics_extended,
        "X_spatial": X_spatial,
        "coords": coords_t,
        "knn_index": knn_index,
        "knn_dists": knn_dists,
        "cardinal_idx": cardinal_idx,
        "y": y_t,
        "encoder": encoder,
        "d_spatial": X_spatial.shape[1],
        "resolution": detected_res,
        "y_mean": y_mean,
        "y_std": y_std,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "h_field": h_field,
        "n_physics_original": physics_t.shape[1],
        "n_physics_extended": physics_extended.shape[1],
        "deriv_names": deriv_names,
        "water_mask": water_mask,
        "T_water": T_water,
        "time_idx": time_idx,
        "sparse_laplacian": _sparse_lap,
        "valid_laplacian_mask": _valid_lap_mask,
        "sheaf_delta": _sheaf_delta,
    }


def _load_correlogram_bandwidths(
    output_dir: Path,
    feature_names: list[str],
) -> np.ndarray | None:
    """
    Load per-predictor bandwidths from Stage 0 correlogram JSON.

    Searches for ``correlogram_analysis_results.json`` in
    ``{output_dir.parent}/Stage_0_Correlogram/``.

    Returns (n_features,) float32 array of per-predictor bandwidths,
    or None if unavailable.
    """
    import json

    # Stage_0 is a sibling of Stage_2 under the shared output root.
    # Walk up from output_dir until we find Stage_0_Correlogram as a sibling.
    # Prefer the artifact store (DB-only mode), fall back to disk walk.
    data = None
    try:
        from sparc.run.artifact_io import load_struct_path
        # Use a synthetic path; load_struct_path tries the store first via
        # (stage="0", artifact_id="correlogram_results") regardless of path.
        synthetic = output_dir.parent / "Stage_0_Correlogram" / "correlogram_analysis_results.json"
        try:
            data = load_struct_path(
                synthetic, stage="0", artifact_id="correlogram_results",
            )
        except FileNotFoundError:
            data = None
    except Exception:  # noqa: BLE001
        data = None

    if data is None:
        results_path = None
        search_dir = output_dir
        for _ in range(4):  # max 4 levels up
            candidate = search_dir.parent / "Stage_0_Correlogram" / "correlogram_analysis_results.json"
            if candidate.exists():
                results_path = candidate
                break
            search_dir = search_dir.parent

        if results_path is None:
            logger.info(
                "Correlogram results not found (searched up from %s) — using uniform bandwidths",
                output_dir,
            )
            return None

        try:
            with open(results_path, "r") as f:
                data = json.load(f)
        except Exception:  # noqa: BLE001
            return None

    try:
        # Extract per-variable bandwidths from individual_results
        individual = data.get("individual_results", {})
        bandwidths = []
        for fname in feature_names:
            if fname in individual:
                bw = individual[fname].get("optimal_bandwidth")
                if bw is not None and bw > 0:
                    bandwidths.append(float(bw))
                else:
                    bandwidths.append(None)
            else:
                bandwidths.append(None)

        # If any are missing, fall back to aggregate GWR bandwidth
        gwr_bw = data.get("model_bandwidths", {}).get("GWR")
        if gwr_bw is None:
            gwr_bw = data.get("summary_statistics", {}).get("bandwidth_range", {}).get("median", 1.0)

        for i in range(len(bandwidths)):
            if bandwidths[i] is None:
                bandwidths[i] = float(gwr_bw) if gwr_bw else 1.0

        bw_array = np.array(bandwidths, dtype=np.float32)
        logger.info(
            "Loaded per-predictor bandwidths from Stage 0 correlogram: %s",
            {fn: f"{bw:.1f}" for fn, bw in zip(feature_names, bw_array)},
        )
        return bw_array

    except Exception as e:
        logger.warning("Failed to load correlogram bandwidths: %s", e)
        return None


def _load_kernel_field(config: dict, feature_names: list[str]):
    """Phase 5b: load the matrix-aware ``KernelField`` from Stage-0
    artifacts so the surrogate's bandwidth conditioning becomes
    kernel-aware (κ_x, κ_y, θ, ν, σ², per-pair bandwidth-to-outcome).

    Returns ``None`` when the artifacts are not yet present so legacy
    runs proceed unchanged.
    """
    target_var = (config or {}).get("variables", {}).get("target", "")
    if not target_var or not feature_names:
        return None
    try:
        from sparc.registry.store import get_active_store
        store = get_active_store()
        if store is None:
            return None
    except Exception:
        return None

    def _safe(stage, aid):
        try:
            return store.read_struct(stage, aid)
        except Exception:
            return None

    matern = _safe("0", "correlogram_matern_fit")
    aniso = _safe("0", "correlogram_anisotropy")
    erm = _safe("0", "effective_range_matrix")
    if matern is None and aniso is None and erm is None:
        return None
    try:
        from sparc.models.kernel_field import KernelField
        kf = KernelField.from_artifacts(
            outcome_name=target_var,
            predictor_names=list(feature_names),
            matern_artifact=matern,
            anisotropy_artifact=aniso,
            effective_range_matrix=erm,
        )
        logger.info(
            "Surrogate KernelField assembled for outcome '%s' (%d predictors; "
            "matern=%s aniso=%s erm=%s)",
            target_var, len(kf.predictors),
            "yes" if matern else "no",
            "yes" if aniso else "no",
            "yes" if erm else "no",
        )
        return kf
    except Exception as e:
        logger.warning("KernelField assembly failed (legacy fallback): %s", e)
        return None


def _run_base_vs_surrogate_alignment(
    surrogate_gwr,
    X_spatial,
    feature_names: list[str],
) -> None:
    """Phase 5c audit: compute per-predictor cosine similarity between
    the classical ``GWRModel.coefficients_`` and the
    ``DifferentiableGWR`` surrogate's β surface.

    Persists ``stage=2, gwr_base_vs_surrogate_alignment`` to the active
    artifact store.  Skips silently when either model is unavailable.
    """
    if surrogate_gwr is None or X_spatial is None or not feature_names:
        return
    try:
        from sparc.registry.store import get_active_store
        store = get_active_store()
        if store is None:
            return
    except Exception:
        return

    # Load the classical GWR base model (persisted by Stage-2 spatial CV).
    gwr_model = None
    try:
        from sparc.run.artifact_io import load_blob_path
        from pathlib import Path as _P
        # Synthetic path; loader prefers store via (stage, artifact_id).
        gwr_model = load_blob_path(
            _P("gwr_model_full.pkl"),
            stage="2", artifact_id="gwr_model_full",
        )
    except Exception:
        gwr_model = None

    if gwr_model is None or not hasattr(gwr_model, "coefficients_"):
        logger.info(
            "Base/surrogate alignment skipped: classical GWR coefficients "
            "not available in artifact store"
        )
        return

    base_beta = np.asarray(gwr_model.coefficients_, dtype=np.float64)
    if base_beta.ndim != 2 or base_beta.shape[1] != len(feature_names):
        logger.warning(
            "Base/surrogate alignment skipped: shape mismatch "
            "(base_beta=%s, n_features=%d)",
            base_beta.shape, len(feature_names),
        )
        return

    from sparc.run.consistency_check import (
        collect_surrogate_beta, compute_base_vs_surrogate_alignment,
    )
    surrogate_beta = collect_surrogate_beta(
        surrogate_gwr, X_spatial, n_vars=len(feature_names),
    )
    # Surrogate may have been fit on a different N — only audit when N matches.
    if surrogate_beta.shape[0] != base_beta.shape[0]:
        logger.info(
            "Base/surrogate alignment skipped: sample-size mismatch "
            "(base N=%d, surrogate N=%d)",
            base_beta.shape[0], surrogate_beta.shape[0],
        )
        return

    payload = compute_base_vs_surrogate_alignment(
        base_beta, surrogate_beta, list(feature_names),
        warn_threshold=0.7, persist=True,
    )
    summary = payload.get("summary", {})
    logger.info(
        "Base/surrogate alignment: mean cos=%.3f, min cos=%.3f, "
        "%d/%d predictors below threshold",
        summary.get("mean_cosine_similarity", float("nan")),
        summary.get("min_cosine_similarity", float("nan")),
        summary.get("n_warnings", 0),
        summary.get("n_predictors", 0),
    )


def _resolve_treatment_list(config: dict) -> list[str]:
    """Resolve ordered treatment names for per-head alpha output.

    Resolution order (Phase 3 of v4):

    1. ``config['causal']['treatments']`` — explicit override list.
    2. ``causal.dag_file`` — load DAG and pull nodes with ``type='treatment'``
       in topological order.
    3. Empty list — caller defaults to ``n_treatments=1`` (legacy mode).
    """
    causal_cfg = (config or {}).get("causal", {}) or {}
    explicit = causal_cfg.get("treatments")
    if explicit:
        return [str(t) for t in explicit]

    dag_path = causal_cfg.get("dag_file")
    if not dag_path:
        return []
    try:
        from sparc.causal.dag_definition import (
            dag_to_networkx, get_node_roles, load_dag,
        )
        import networkx as _nx
        dag_def = load_dag(dag_path)
        G = dag_to_networkx(dag_def)
        roles = get_node_roles(G)
        treatments = roles.get("treatments", []) or []
        if not treatments:
            return []
        # Stable ordering: topological where possible, lexicographic fallback.
        try:
            topo = list(_nx.topological_sort(G))
            order = {n: i for i, n in enumerate(topo)}
            treatments = sorted(treatments, key=lambda n: order.get(n, len(order)))
        except Exception:
            treatments = sorted(treatments)
        return [str(t) for t in treatments]
    except Exception as exc:
        logger.warning(
            "Could not resolve treatment list from DAG (%s); "
            "falling back to single-head alpha.", exc,
        )
        return []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_neural_meta(
    y: np.ndarray,
    coords: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: dict,
    output_dir: str | Path,
    base_oof_predictions: dict[str, np.ndarray] | None = None,
    quick_eval: bool = False,
) -> dict[str, Any]:
    """
    Train the V2 neural meta-learner with differentiable surrogates,
    process-rate network, physics-informed loss, per-component
    optimizer, 4-stage curriculum, and Stochastic Weight Averaging.

    When *base_oof_predictions* is provided (dict mapping model name
    to (N,) OOF prediction arrays from V1 base models), the surrogates
    are pretrained against V1 outputs and the fidelity loss anchors
    them to those outputs.  Gradients from physics / MSE / smoothness
    still flow end-to-end through the surrogates during joint training.

    Returns dict with:
      - ``model``           — trained SPARCMetaLearner
      - ``process_rate``    — trained ProcessRateNet
      - ``surrogates``      — dict of trained DifferentiableGWR/GWRF/GGPGAM
      - ``oof_predictions`` — (N,) OOF predictions
      - ``oof_uncertainty``  — (N,) MC-Dropout std
      - ``metrics`` — dict with R², RMSE
    """
    from sparc.models.neural_meta import SPARCMetaLearner
    from sparc.models.process_rate_net import ProcessRateNet, SourceTermNet
    from sparc.models.surrogates import (
        DifferentiableGGPGAM,
        DifferentiableGWR,
        DifferentiableGWRF,
    )
    from sparc.training.curriculum import get_lambda_schedule
    from sparc.training.loss import sparc_joint_loss
    from sparc.training.optimizer import build_optimizer, build_scheduler

    # JEPA Phase 1 (optional, gated by config.jepa.enable).  Imports are
    # local so the dependency is only paid when the feature is on.
    from sparc.models.ema_trunk import EMATrunk
    from sparc.models.latent_predictor import LatentPredictor
    from sparc.training.jepa_loss import JEPALossWeights, jepa_curriculum_weight, jepa_loss, spatial_patch_mask
    # JEPA Phase 2: action-conditioned predictor + scenario distillation.
    from sparc.models.action_embedding import ActionEmbedding

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Config ----
    neural_cfg = config.get("models", {}).get("neural", {})
    training_cfg = config.get("training", {})
    optim_cfg = config.get("optimization", {})
    pr_cfg = config.get("process_rate", {})
    jepa_cfg = config.get("jepa", {}) or {}
    pr_pretrain_epochs = int(pr_cfg.get("pretrain_epochs", 0))

    hidden_dim = neural_cfg.get("hidden_dim", 256)
    dropout = neural_cfg.get("dropout", 0.1)
    n_heads = neural_cfg.get("n_heads", 4)
    max_neighbors = neural_cfg.get("max_neighbors", 128)
    siren_omega = neural_cfg.get("siren_omega", 30.0)
    thresholds = neural_cfg.get("exceedance_thresholds", [0.25, 0.50, 0.75])
    time_embed_dim = neural_cfg.get("time_embed_dim", 0)

    n_epochs = training_cfg.get("n_epochs", 100)
    swa_epochs = training_cfg.get("swa_epochs", max(int(n_epochs * 0.2), 5))
    pretrain_epochs = training_cfg.get("pretrain_epochs", 200)
    main_epochs = n_epochs  # SWA runs *on top* of main epochs for full retrain
    lr = training_cfg.get("learning_rate", 1e-3)
    clip_norm = optim_cfg.get("clip_norm", 1.0)
    warmup_epochs = training_cfg.get("warmup_epochs", 10)
    ramp_epochs = training_cfg.get("ramp_epochs", 30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        from sparc.config.hardware_profile import detect_profile
        if detect_profile().force_cpu:
            device = torch.device("cpu")
            logger.info("Low-memory tier detected — forcing CPU device for neural training")
    except Exception:  # pragma: no cover - best-effort safety override
        pass
    logger.info("V2 neural training on device: %s", device)

    # CU-2: Automatic Mixed Precision — active only on CUDA (Ampere/Hopper).
    # GradScaler handles FP16 loss scaling; bfloat16 doesn't need scaling but
    # we keep the Scaler for uniformity (it is a no-op when enabled=False).
    _use_amp: bool = device.type == "cuda"
    _scaler = torch.cuda.amp.GradScaler(enabled=_use_amp)
    if _use_amp:
        logger.info("AMP active (bfloat16)")

    # CU-9b: CUDA Graph batch padding — static shapes required for graph capture.
    # Reads the cuda_graphs flag from HardwareProfile; off by default.
    _use_cuda_graphs: bool = False
    try:
        from sparc.config.hardware_profile import detect_profile as _dp
        _use_cuda_graphs = device.type == "cuda" and bool(getattr(_dp(), "cuda_graphs", False))
    except Exception:
        pass
    if _use_cuda_graphs:
        logger.info("CUDA Graphs enabled — batches padded to static size=%d", batch_size)

    # ---- Prepare data ----
    # Stash feature_names into config for _prepare_tensors
    config["_feature_names"] = feature_names

    tensors = _prepare_tensors(
        y, coords, feature_matrix, config, device,
    )

    n_base = 3  # always 3 surrogates: GWR, GWRF, GGPGAM
    n_physics = feature_matrix.shape[1]
    n_physics_extended = tensors["n_physics_extended"]
    d_spatial = tensors["d_spatial"]
    resolution = tensors["resolution"]
    y_mean = tensors["y_mean"]
    y_std = tensors["y_std"]

    # ---- Load per-predictor bandwidths from Stage 0 correlogram ----
    predictor_bandwidths = _load_correlogram_bandwidths(output_dir, feature_names)
    # Phase 5b: matrix-aware kernel-field bundle (Matérn + anisotropy +
    # per-pair bandwidth-to-outcome).  None when Stage-0 artifacts
    # haven't been written; surrogates fall back to legacy bandwidth-only
    # conditioning in that case.
    predictor_kernel_field = _load_kernel_field(config, feature_names)

    # GWRF neighbor count (smaller than max_neighbors for spatial attention)
    gwrf_k = min(neural_cfg.get("gwrf_neighbors", 16), max_neighbors)

    # Convert exceedance thresholds to normalised space
    thresholds_norm = [(t - y_mean) / y_std for t in thresholds]

    # Process rate input columns — use all physics features when not configured
    pr_inputs = pr_cfg.get("inputs", feature_names)
    pr_input_dim = len(pr_inputs)
    pr_col_idxs = [feature_names.index(f) for f in pr_inputs if f in feature_names]
    if not pr_col_idxs:
        pr_col_idxs = list(range(min(pr_input_dim, n_physics)))

    prior_mean = pr_cfg.get("prior_mean", 0.5)

    # ---- JEPA settings (Phase 1, off by default) ----
    jepa_enable = bool(jepa_cfg.get("enable", False))
    jepa_mask_ratio = float(jepa_cfg.get("mask_ratio", 0.4))
    jepa_spatial_patch = bool(jepa_cfg.get("spatial_patch_masking", False))
    jepa_n_patches = int(jepa_cfg.get("n_patches", 4))
    jepa_lambda = float(jepa_cfg.get("lambda", 1.0))
    jepa_ema_tau_start = float(jepa_cfg.get("ema_tau_start", 0.99))
    jepa_ema_tau_end = float(jepa_cfg.get("ema_tau_end", 0.9999))
    jepa_warmup_steps = int(jepa_cfg.get("ema_warmup_steps", 1000))
    jepa_curric_start = int(jepa_cfg.get("curriculum_start", 5))
    jepa_curric_end = int(jepa_cfg.get("curriculum_end", 15))
    jepa_weights = JEPALossWeights(
        alignment=float(jepa_cfg.get("lambda_align", 1.0)),
        variance=float(jepa_cfg.get("lambda_variance", 1.0)),
        covariance=float(jepa_cfg.get("lambda_covariance", 0.04)),
        variance_gamma=float(jepa_cfg.get("variance_gamma", 1.0)),
    )
    # Phase 2 settings (only used when jepa.enable=true)
    jepa_action_dim = int(jepa_cfg.get("action_dim", 64))
    jepa_predictor_blocks = int(jepa_cfg.get("predictor_blocks", 1))
    jepa_predictor_film = bool(jepa_cfg.get("predictor_film", False))
    jepa_lambda_scenario = float(jepa_cfg.get("lambda_scenario", 0.0))
    jepa_lambda_latent_pde = float(jepa_cfg.get("lambda_latent_pde", 0.0))
    jepa_scenario_perturb_std = float(jepa_cfg.get("scenario_perturb_std", 0.5))
    if jepa_enable:
        logger.info(
            "  JEPA enabled: mask_ratio=%.2f lambda=%.3f ema_tau=%.4f→%.4f "
            "(warmup=%d) curriculum=[%d,%d]",
            jepa_mask_ratio, jepa_lambda,
            jepa_ema_tau_start, jepa_ema_tau_end, jepa_warmup_steps,
            jepa_curric_start, jepa_curric_end,
        )
        if jepa_lambda_scenario > 0 or jepa_lambda_latent_pde > 0 or jepa_predictor_blocks > 1:
            logger.info(
                "  JEPA Phase 2: action_dim=%d blocks=%d film=%s "
                "λ_scenario=%.3f λ_latent_pde=%.4f σ_perturb=%.3f",
                jepa_action_dim, jepa_predictor_blocks, jepa_predictor_film,
                jepa_lambda_scenario, jepa_lambda_latent_pde, jepa_scenario_perturb_std,
            )

    # ---- Phase 3 (v4): per-treatment alpha heads ----
    # Resolve the ordered list of treatments from the causal DAG (or
    # explicit config override).  When unavailable, fall back to a
    # single-treatment ProcessRateNet — this preserves legacy behavior.
    treatments = _resolve_treatment_list(config)
    n_treatments = max(1, len(treatments))
    if n_treatments > 1:
        logger.info(
            "  Phase 3 multi-head alpha enabled: %d treatments → %s",
            n_treatments, treatments,
        )

    # ---- Alpha supervision targets (land-cover classification) ----
    # Compute once from raw features; used as persistent prior during
    # joint training to prevent α from collapsing when w(s) is free.
    material_priors = pr_cfg.get("material_priors", {})
    if material_priors:
        alpha_targets = _classify_land_cover(
            feature_matrix, feature_names, material_priors,
        ).to(device)  # (N, 1)
    else:
        alpha_targets = None

    # Target lambda dict for curriculum schedule.
    # "base" = surrogate fidelity: soft alignment with pretrained targets.
    target_lambdas = {
        "physics": training_cfg.get("lambda_physics", 0.01),
        "smooth": training_cfg.get("lambda_smooth_pred", 0.01),
        "alpha_smooth": training_cfg.get("lambda_smooth_alpha", 0.001),
        "neighbor": training_cfg.get("lambda_neighbor", 0.1),
        "prior": training_cfg.get("lambda_alpha_prior", 0.01),
        "base": training_cfg.get("lambda_base", 0.2),
        "pde": training_cfg.get("lambda_pde", 0.05),
        "bc": training_cfg.get("lambda_bc", 0.02),
    }

    # OOF containers
    oof_preds = np.zeros(len(y), dtype=np.float32)
    oof_std = np.zeros(len(y), dtype=np.float32)

    # ==================================================================
    # Optional: capacity sweep to find optimal hidden_dim
    # ==================================================================
    if optim_cfg.get("capacity_sweep", False) and len(folds) > 0:
        from sparc.training.curriculum import capacity_sweep

        logger.info("Running capacity sweep to find optimal hidden_dim ...")
        sweep_train_idx, sweep_test_idx = folds[0]
        sweep_phys = tensors["physics_feats"]
        sweep_phys_ext = tensors["physics_feats_extended"]
        sweep_spat = tensors["X_spatial"]
        sweep_coords = tensors["coords"]
        sweep_y = tensors["y"]
        sweep_epochs = max(pretrain_epochs + 20, 30)

        # Precompute KNN/cardinal once — identical for every capacity trial
        _sweep_knn = torch.tensor(
            _build_knn_index(coords[sweep_train_idx], max_neighbors),
            dtype=torch.long, device=device,
        )
        _sweep_card_np, _ = _build_cardinal_neighbors(
            coords[sweep_train_idx], resolution=resolution,
        )
        _sweep_card = torch.tensor(_sweep_card_np, dtype=torch.long, device=device)
        _sweep_knn_full = torch.tensor(
            _build_knn_index(
                coords[np.concatenate([sweep_train_idx, sweep_test_idx])],
                max_neighbors,
            ),
            dtype=torch.long, device=device,
        )

        def _sweep_factory(hidden_dim: int, **_kw):
            """Create model bundle for capacity sweep."""
            _m = SPARCMetaLearner(
                n_base_models=n_base, n_physics_features=n_physics_extended,
                d_spatial=d_spatial, hidden_dim=hidden_dim, dropout=dropout,
                thresholds=thresholds, n_heads=n_heads,
                max_neighbors=max_neighbors, siren_omega=siren_omega,
                init_bandwidth=float(predictor_bandwidths.mean()) if predictor_bandwidths is not None else 1000.0,
            ).to(device)
            _s = {
                "gwr": DifferentiableGWR(
                    n_physics, d_spatial, hidden_dim,
                    bandwidths=predictor_bandwidths,
                    kernel_field=predictor_kernel_field,
                    feature_names=feature_names,
                ).to(device),
                "gwrf": DifferentiableGWRF(
                    n_physics, d_spatial, hidden_dim,
                    kernel_field=predictor_kernel_field,
                    feature_names=feature_names,
                ).to(device),
                "ggpgam": DifferentiableGGPGAM(
                    n_physics, d_spatial, min(hidden_dim, 64),
                ).to(device),
            }
            _p = ProcessRateNet(
                n_inputs=pr_input_dim,
                domain_config={
                    "name": pr_cfg.get("name", "rate"),
                    "units": pr_cfg.get("units", ""),
                    "bounds": pr_cfg.get("bounds", [0.0, 1.0]),
                    "prior_mean": prior_mean,
                },
                n_treatments=n_treatments,
            ).to(device)
            return {"model": _m, "surrogates": _s, "process": _p}

        def _sweep_train(bundle):
            """Short training: pretrain surrogates + brief joint training."""
            _s, _m, _p = bundle["surrogates"], bundle["model"], bundle["process"]
            _pretrain_surrogates(
                _s, sweep_phys[sweep_train_idx], sweep_spat[sweep_train_idx],
                sweep_y[sweep_train_idx], n_epochs=pretrain_epochs, lr=lr,
            )
            opt = build_optimizer(_m, _p, _s, base_lr=lr)
            _knn = _sweep_knn
            _card = _sweep_card
            _src = torch.zeros(len(sweep_train_idx), device=device)
            for _ in range(20):
                sp, _surr_std = _forward_surrogates(
                    _s, sweep_phys[sweep_train_idx], sweep_spat[sweep_train_idx],
                )
                bi = torch.stack(sp, dim=1)
                pr_in = sweep_phys[sweep_train_idx][:, pr_col_idxs]
                alpha = _p(pr_in)
                T_pred, exc, _ = _m(
                    base_preds=bi,
                    physics_feats=sweep_phys_ext[sweep_train_idx],
                    X_spatial=sweep_spat[sweep_train_idx],
                    coords=sweep_coords[sweep_train_idx],
                    knn_index=_knn, alpha=alpha,
                    surrogate_std=_surr_std,
                )
                # Cache surrogate outputs as their own targets (no V1 data in sweep)
                _surr_tgts = [s.detach() for s in sp]
                loss, _ = sparc_joint_loss(
                    T_pred=T_pred, exceedance_preds=exc,
                    y_true=sweep_y[sweep_train_idx],
                    thresholds=thresholds_norm, alpha=alpha,
                    alpha_prior=torch.full_like(alpha, prior_mean),
                    neighbor_idx=_card, source_term=_src,
                    resolution=resolution,
                    surrogate_preds=sp,
                    surrogate_targets=_surr_tgts,
                    lambda_physics=0.0, lambda_smooth=0.0,
                    lambda_alpha_smooth=0.0, lambda_prior=1.0,
                    lambda_base=0.0, lambda_neighbor=0.0, epoch=0,
                )
                opt.zero_grad()
                loss.backward()
                opt.step()

        def _sweep_eval(bundle):
            """Evaluate on held-out fold data, return R²."""
            _s, _m, _p = bundle["surrogates"], bundle["model"], bundle["process"]
            _m.eval(); _p.eval()
            for v in _s.values():
                v.eval()
            with torch.no_grad():
                _knn_full = _sweep_knn_full
                _n_train = len(sweep_train_idx)
                _n_test = len(sweep_test_idx)
                _all_phys = torch.cat([sweep_phys[sweep_train_idx], sweep_phys[sweep_test_idx]])
                _all_phys_ext = torch.cat([sweep_phys_ext[sweep_train_idx], sweep_phys_ext[sweep_test_idx]])
                _all_spat = torch.cat([sweep_spat[sweep_train_idx], sweep_spat[sweep_test_idx]])
                _all_coords = torch.cat([sweep_coords[sweep_train_idx], sweep_coords[sweep_test_idx]])
                sp, _surr_std_eval = _forward_surrogates(_s, _all_phys, _all_spat)
                bi = torch.stack(sp, dim=1)
                pr_in = _all_phys[:, pr_col_idxs]
                alpha = _p(pr_in)
                T_pred, _, _ = _m(
                    base_preds=bi,
                    physics_feats=_all_phys_ext,
                    X_spatial=_all_spat,
                    coords=_all_coords,
                    knn_index=_knn_full, alpha=alpha,
                    surrogate_std=_surr_std_eval,
                )
                # Extract predictions for test portion only
                _preds = T_pred[_n_train:].cpu().numpy().ravel()
                _true = sweep_y[sweep_test_idx].cpu().numpy().ravel()
            return float(r2_score(_true, _preds))

        hidden_dim, sweep_results = capacity_sweep(
            _sweep_factory, _sweep_train, _sweep_eval,
        )
        logger.info(
            "Capacity sweep selected hidden_dim=%d (results: %s)",
            hidden_dim, sweep_results,
        )

    # ==================================================================
    # CV Loop
    # ==================================================================
    batch_size = training_cfg.get("batch_size", 2048)
    # CU-3: Override with GPU-optimised batch size when CUDA is active.
    if device.type == "cuda":
        try:
            from sparc.config.hardware_profile import detect_profile as _dp
            _hp = _dp()
            if _hp.gpu_batch_size > batch_size:
                logger.info(
                    "GPU batch size override: %d → %d (%.1f GB VRAM)",
                    batch_size, _hp.gpu_batch_size, _hp.gpu_vram_gb,
                )
                batch_size = _hp.gpu_batch_size
        except Exception:  # pragma: no cover
            pass

    # --- Continual learning (EWC) config ---
    _cont = config.get("_continual", {})
    _ewc_lambda = float(_cont.get("ewc_lambda", 0.0))
    _cont_fisher = _cont.get("fisher_matrices", [])
    _cont_optimal = _cont.get("optimal_params_list", [])
    _ewc_active = _ewc_lambda > 0.0 and bool(_cont_fisher) and bool(_cont_optimal)
    if _ewc_active:
        from sparc.training.ewc import ewc_penalty as _ewc_penalty
        logger.info(
            "Continual learning: EWC active (lambda=%.3f, %d previous cities)",
            _ewc_lambda, len(_cont_fisher),
        )

    # --- Wasserstein trunk alignment (OT) config ---
    _ot_lambda = float(_cont.get("ot_lambda", 0.0))
    _coreset_acts = _cont.get("coreset_activations", None)  # (B, D) tensor or None
    _ot_active = _ot_lambda > 0.0 and _coreset_acts is not None
    if _ot_active:
        from sparc.training.ewc import wasserstein_trunk_alignment as _wta
        logger.info("Continual learning: OT alignment active (ot_lambda=%.3f)", _ot_lambda)

    # Buckets populated by forward hooks when OT is active
    _trunk_acts_cv_bucket: list = []
    _trunk_acts_ret_bucket: list = []
    _trunk_acts_swa_bucket: list = []

    def _make_trunk_hook(bucket: list):
        """Return a forward hook that appends trunk_fusion output (detached) to bucket."""
        def _hook(module, inp, out):
            bucket.append(out.detach())
        return _hook

    # ==================================================================
    # Optional: Spatial Contrastive pretext (2.SS-5)
    # Gated by config["continual"]["contrastive_pretext"]: true.
    # Trains SharedTrunk via InfoNCE over cross-city coresets so that
    # blocks from cities with similar Matérn bandwidths are aligned in
    # latent space.  Single-city runs with no registry coresets are a
    # no-op (zero coreset pool → skipped gracefully).
    # ==================================================================
    _cont_cfg = config.get("continual", {}) or {}
    _contrastive_enabled = _cont_cfg.get("contrastive_pretext", False)
    if _contrastive_enabled:
        from sparc.training.spatial_contrastive import (
            log_bandwidth_from_payload,
            spatial_contrastive_pretext,
        )
        from sparc.registry.city_registry import CityRegistry

        _cont_coresets: list[dict] = []

        # 1. Load cross-city coresets from registry (if available)
        _registry_path = _cont_cfg.get("registry_path", None)
        if _registry_path:
            try:
                _creg = CityRegistry(_registry_path)
                _cont_coresets = _creg.query_coresets_by_bandwidth_cluster(
                    threshold=float(_cont_cfg.get("bw_threshold", 0.3))
                )
                logger.info(
                    "Contrastive pretext: loaded %d cross-city coresets from registry",
                    len(_cont_coresets),
                )
            except Exception as _creg_exc:
                logger.warning("Could not load city registry coresets: %s", _creg_exc)

        # 2. Always include current-city coreset (feature matrix subsample)
        _current_log_bw = 0.0
        try:
            _vsba_store2 = get_active_store()
            if _vsba_store2 is not None and _vsba_store2.has("0", "correlogram_results"):
                _cur_corr = _vsba_store2.read_struct("0", "correlogram_results")
                _current_log_bw = log_bandwidth_from_payload(_cur_corr)
        except Exception:
            pass

        _n_current = min(512, len(feature_matrix))
        _rng_c = np.random.default_rng(0)
        _cur_idx = _rng_c.choice(len(feature_matrix), size=_n_current, replace=False)
        _cont_coresets.append({
            "X": feature_matrix[_cur_idx].astype(np.float32),
            "log_bw": _current_log_bw,
            "city": "_current",
        })

        logger.info(
            "Contrastive pretext: %d total coreset pools, current log_bw=%.3f",
            len(_cont_coresets), _current_log_bw,
        )

        try:
            # Build a fresh SPARCMetaLearner for pretext (same trunk dims)
            # then transfer weights back — avoids disrupting compiled model
            from sparc.models.neural_meta import SPARCMetaLearner
            _pt_model = SPARCMetaLearner(
                n_base_models=n_base,
                n_physics_features=n_physics,
                d_spatial=d_spatial,
                hidden_dim=hidden_dim,
                dropout=dropout,
                thresholds=thresholds,
                n_heads=n_heads,
                max_neighbors=max_neighbors,
                siren_omega=siren_omega,
                time_embed_dim=time_embed_dim,
            ).to(device)
            spatial_contrastive_pretext(
                model=_pt_model,
                coresets_with_bw=_cont_coresets,
                n_epochs=int(_cont_cfg.get("contrastive_epochs", 10)),
                lr=float(_cont_cfg.get("contrastive_lr", 1e-4)),
                temperature=float(_cont_cfg.get("contrastive_temperature", 0.07)),
                bw_threshold=float(_cont_cfg.get("bw_threshold", 0.3)),
                device=device,
            )
            # Transfer pretrained trunk weights to main model
            model.load_state_dict(_pt_model.state_dict(), strict=False)
            del _pt_model
            logger.info("Contrastive pretext trunk weights transferred to main model")
        except Exception as _ct_exc:
            logger.warning("Contrastive pretext failed (non-fatal): %s", _ct_exc)

    # ==================================================================
    # Optional: VSBA pretext — label-free fold quality scoring
    # Gated by config["models"]["spatial_cv"]["vsba_fold_scoring"]: true.
    # Trains a VariationalSpatialBlockAutoencoder on the full feature
    # matrix and scores each CV fold by its ELBO (higher = in-dist).
    # Results are logged; fold assignments are unchanged.
    # ==================================================================
    _vsba_enabled = (
        config.get("models", {})
        .get("spatial_cv", {})
        .get("vsba_fold_scoring", False)
    )
    if _vsba_enabled and len(folds) > 0:
        from sparc.run.enhanced_spatial_cv import _vsba_fold_quality_score

        # Load Stage 0 correlogram payload for Matérn seeding
        _vsba_corr_payload: dict | None = None
        try:
            from sparc.registry.store import get_active_store
            _vsba_store = get_active_store()
            if _vsba_store is not None and _vsba_store.has("0", "correlogram_results"):
                _vsba_corr_payload = _vsba_store.read_struct(
                    "0", "correlogram_results"
                )
        except Exception:
            pass

        logger.info(
            "VSBA fold quality scoring: %d folds, features=%d, corr_payload=%s",
            len(folds), feature_matrix.shape[1],
            "loaded" if _vsba_corr_payload else "absent (uniform prior)",
        )
        try:
            _vsba_fold_quality_score(
                feature_matrix.astype(np.float32),
                coords.astype(np.float32),
                folds,
                corr_payload=_vsba_corr_payload,
                n_epochs=int(
                    config.get("models", {})
                    .get("spatial_cv", {})
                    .get("vsba_n_epochs", 20)
                ),
                device=str(device),
            )
        except Exception as _vsba_exc:
            logger.warning("VSBA fold scoring failed (non-fatal): %s", _vsba_exc)

    # ==================================================================
    # Optional: JEPA self-supervised pretraining on full dataset
    # Gated by config["jepa"]["pretrain_epochs"] > 0 AND jepa_enable.
    # Trains trunk encoder + latent predictor via masked-context JEPA
    # before CV folds start.  Trunk weights are then transferred to each
    # fold's SPARCMetaLearner via load_state_dict (strict=False).
    # ==================================================================
    jepa_pretrain_epochs = int(jepa_cfg.get("pretrain_epochs", 0))
    jepa_pretrained_trunk_state: dict | None = None

    # ---- Load OOF GWR beta-map for JEPA target conditioning (Stage 2 artifact) ----
    _pt_beta_t: "torch.Tensor | None" = None
    _pt_beta_n_feat: int = n_physics
    try:
        from sparc.registry.store import get_active_store
        _oof_store = get_active_store()
        if _oof_store is not None and _oof_store.has("2", "oof_gwr_beta_map"):
            _oof_beta = np.asarray(
                _oof_store.read_blob("2", "oof_gwr_beta_map", trusted=True),
                dtype=np.float32,
            )
            _col_std = _oof_beta.std(axis=0, keepdims=True)
            _col_std[_col_std < 1e-8] = 1.0
            _oof_beta = (_oof_beta - _oof_beta.mean(axis=0, keepdims=True)) / _col_std
            _pt_beta_t = torch.tensor(_oof_beta, dtype=torch.float32, device=device)
            _pt_beta_n_feat = _oof_beta.shape[1]
            logger.info(
                "oof_intelligence loaded: gwr_local_coefficients (n=%d)", len(_oof_beta)
            )
    except Exception as _oof_e:
        logger.debug("OOF beta map not available (normal on first Stage 2 run): %s", _oof_e)

    if jepa_enable and jepa_pretrain_epochs > 0:
        logger.info(
            "JEPA pretraining: %d epochs on %d points (mask_ratio=%.2f)",
            jepa_pretrain_epochs, len(y), jepa_mask_ratio,
        )
        _pt_t0 = _time.perf_counter()

        # Build a temporary online model (trunk only used)
        _pt_model = SPARCMetaLearner(
            n_base_models=n_base,
            n_physics_features=n_physics_extended,
            d_spatial=d_spatial,
            hidden_dim=hidden_dim,
            dropout=dropout,
            thresholds=thresholds,
            n_heads=n_heads,
            max_neighbors=max_neighbors,
            siren_omega=siren_omega,
            init_bandwidth=float(predictor_bandwidths.mean()) if predictor_bandwidths is not None else 1000.0,
            time_embed_dim=time_embed_dim,
        ).to(device)

        _pt_ema_trunk = EMATrunk(
            _pt_model,
            tau_start=jepa_ema_tau_start,
            tau_end=jepa_ema_tau_end,
            warmup_steps=jepa_warmup_steps,
        ).to(device)

        _pt_latent_predictor = LatentPredictor(
            hidden_dim=hidden_dim,
            action_dim=0,
            dropout=dropout,
            n_blocks=jepa_predictor_blocks,
            film=jepa_predictor_film,
        ).to(device)

        _pt_params = (
            list(_pt_model.parameters()) + list(_pt_latent_predictor.parameters())
        )
        _pt_optimizer = torch.optim.AdamW(_pt_params, lr=lr)

        # Frozen beta-map projector: projects OOF GWR β-map → hidden_dim and
        # adds to EMA target encoding so the trunk learns spatial process structure.
        _pt_beta_proj: "torch.nn.Linear | None" = None
        if _pt_beta_t is not None:
            _pt_beta_proj = torch.nn.Linear(_pt_beta_n_feat, hidden_dim, bias=False).to(device)
            for _p in _pt_beta_proj.parameters():
                _p.requires_grad_(False)

        # Constant prior alpha (ProcessRateNet not trained yet)
        _pt_alpha = torch.full(
            (len(y), 1), prior_mean, dtype=torch.float32, device=device,
        )
        _pt_time_idx = tensors.get("time_idx")

        _pt_model.train()
        N_pt = len(y)
        _pt_use_minibatch = N_pt > batch_size * 2

        for _pt_epoch in range(jepa_pretrain_epochs):
            if _pt_use_minibatch:
                from sparc.training.optimizer import spatial_minibatch_sampler
                _pt_batches = spatial_minibatch_sampler(
                    coords, None, batch_size=batch_size,
                )
            else:
                _pt_batches = [np.arange(N_pt)]

            _pt_epoch_loss = 0.0
            _pt_n = 0

            for _pt_b_idx in _pt_batches:
                b_phys_ext = tensors["physics_feats_extended"][_pt_b_idx]
                b_alpha = _pt_alpha[_pt_b_idx]
                b_time = _pt_time_idx[_pt_b_idx] if _pt_time_idx is not None else None

                # --- Masking strategy ---
                if jepa_spatial_patch:
                    # Spatial patch masking: hide geographically contiguous regions
                    b_coords_t = coords_t[_pt_b_idx]
                    _pt_patch_mask = spatial_patch_mask(
                        b_coords_t, mask_ratio=jepa_mask_ratio, n_patches=jepa_n_patches,
                    )  # (B,) bool — True = masked point
                    b_phys_ctx = b_phys_ext.clone()
                    b_phys_ctx[_pt_patch_mask] = 0.0
                    channel_mask = None
                else:
                    # Legacy: random feature-channel masking
                    b_phys_ctx = b_phys_ext
                    n_phys_feat = b_phys_ext.shape[-1]
                    keep_p = max(0.0, 1.0 - jepa_mask_ratio)
                    channel_mask = (
                        torch.rand(n_phys_feat, device=device) < keep_p
                    ).to(b_phys_ext.dtype)

                h_context = _pt_model.encode(
                    physics_feats=b_phys_ctx,
                    alpha=b_alpha,
                    time_idx=b_time,
                    channel_mask=channel_mask,
                )
                h_pred = _pt_latent_predictor(h_context)

                with torch.no_grad():
                    # Target always sees the full (unmasked) features
                    h_target = _pt_ema_trunk.encode_target(
                        physics_feats=b_phys_ext,
                        alpha=b_alpha,
                        time_idx=b_time,
                    )
                    # Enrich target with OOF β-map spatial structure when available
                    if _pt_beta_proj is not None and _pt_beta_t is not None:
                        h_target = h_target + _pt_beta_proj(_pt_beta_t[_pt_b_idx])

                _pt_loss, _pt_components = jepa_loss(
                    h_pred, h_target, weights=jepa_weights,
                )

                _pt_optimizer.zero_grad()
                _pt_loss.backward()
                torch.nn.utils.clip_grad_norm_(_pt_params, clip_norm)
                _pt_optimizer.step()
                _pt_ema_trunk.update(_pt_model)

                _bsz = len(_pt_b_idx)
                _pt_epoch_loss += _pt_loss.item() * _bsz
                _pt_n += _bsz

            _pt_epoch_loss /= max(_pt_n, 1)

            if (_pt_epoch + 1) % max(1, jepa_pretrain_epochs // 5) == 0 or _pt_epoch == 0:
                logger.info(
                    "  JEPA pretrain epoch %d/%d  loss=%.4f  "
                    "[align=%.3f var=%.3f cov=%.4f]  (%.1fs)",
                    _pt_epoch + 1, jepa_pretrain_epochs, _pt_epoch_loss,
                    _pt_components.get("jepa_align", 0),
                    _pt_components.get("jepa_variance", 0),
                    _pt_components.get("jepa_covariance", 0),
                    _time.perf_counter() - _pt_t0,
                )

        # Extract trunk state for transfer into each fold model
        jepa_pretrained_trunk_state = {
            k: v.detach().clone()
            for k, v in _pt_model.state_dict().items()
            if any(k.startswith(pfx) for pfx in _TRUNK_KEYS)
        }
        logger.info(
            "JEPA pretraining done in %.1fs — trunk state (%d tensors) ready for transfer",
            _time.perf_counter() - _pt_t0, len(jepa_pretrained_trunk_state),
        )
        del _pt_model, _pt_ema_trunk, _pt_latent_predictor, _pt_optimizer, _pt_alpha, _pt_beta_proj

    # ---- DDP detection (CU-10a) ----
    _ddp_enabled: bool = False
    try:
        from sparc.config.hardware_profile import detect_profile as _dp_ddp
        _ddp_enabled = bool(getattr(_dp_ddp(), "ddp_enabled", False))
    except Exception:
        pass
    _gpu_count_ddp = getattr(__import__("torch").cuda, "device_count", lambda: 1)() if _ddp_enabled else 1
    _use_ddp = _ddp_enabled and _gpu_count_ddp > 1 and len(folds) > 1

    # ---- Build shared state dict for fold workers (CU-10a) ----
    _fold_shared: dict = {
        "tensors":                    tensors,
        "coords_np":                  coords,
        "alpha_targets":              alpha_targets,
        "jepa_pretrained_trunk_state": jepa_pretrained_trunk_state,
        "base_oof_predictions":       base_oof_predictions,
        "hidden_dim":                 hidden_dim,
        "dropout":                    dropout,
        "n_heads":                    n_heads,
        "max_neighbors":              max_neighbors,
        "siren_omega":                siren_omega,
        "thresholds":                 thresholds,
        "thresholds_norm":            thresholds_norm,
        "time_embed_dim":             time_embed_dim,
        "n_base":                     n_base,
        "n_physics":                  n_physics,
        "n_physics_extended":         n_physics_extended,
        "d_spatial":                  d_spatial,
        "resolution":                 resolution,
        "y_mean":                     y_mean,
        "y_std":                      y_std,
        "pr_input_dim":               pr_input_dim,
        "pr_col_idxs":                pr_col_idxs,
        "prior_mean":                 prior_mean,
        "pr_cfg":                     pr_cfg,
        "n_treatments":               n_treatments,
        "material_priors":            material_priors,
        "pr_pretrain_epochs":         pr_pretrain_epochs,
        "n_epochs":                   n_epochs,
        "pretrain_epochs":            pretrain_epochs,
        "lr":                         lr,
        "clip_norm":                  clip_norm,
        "warmup_epochs":              warmup_epochs,
        "ramp_epochs":                ramp_epochs,
        "batch_size":                 batch_size,
        "target_lambdas":             target_lambdas,
        "feature_names":              feature_names,
        "gwrf_k":                     gwrf_k,
        "predictor_bandwidths":       predictor_bandwidths,
        "predictor_kernel_field":     predictor_kernel_field,
        "jepa_enable":                jepa_enable,
        "jepa_weights":               jepa_weights,
        "jepa_mask_ratio":            jepa_mask_ratio,
        "jepa_spatial_patch":         jepa_spatial_patch,
        "jepa_n_patches":             jepa_n_patches,
        "jepa_lambda":                jepa_lambda,
        "jepa_ema_tau_start":         jepa_ema_tau_start,
        "jepa_ema_tau_end":           jepa_ema_tau_end,
        "jepa_warmup_steps":          jepa_warmup_steps,
        "jepa_curric_start":          jepa_curric_start,
        "jepa_curric_end":            jepa_curric_end,
        "jepa_lambda_scenario":       jepa_lambda_scenario,
        "jepa_lambda_latent_pde":     jepa_lambda_latent_pde,
        "jepa_scenario_perturb_std":  jepa_scenario_perturb_std,
        "jepa_predictor_blocks":      jepa_predictor_blocks,
        "jepa_predictor_film":        jepa_predictor_film,
        "jepa_action_dim":            jepa_action_dim,
        "ewc_active":                 _ewc_active,
        "ewc_lambda":                 _ewc_lambda,
        "cont_fisher":                _cont_fisher,
        "cont_optimal":               _cont_optimal,
        "ot_active":                  _ot_active,
        "ot_lambda":                  _ot_lambda,
        "coreset_acts":               _coreset_acts,
        "use_amp":                    _use_amp,
        "use_cuda_graphs":            _use_cuda_graphs,
        "n_folds":                    len(folds),
    }

    # ---- CV fold loop (CU-10a: DDP-parallel or sequential) ----
    if _use_ddp:
        import torch.multiprocessing as _tmp
        _world_size_ddp = min(_gpu_count_ddp, len(folds))
        _fold_list_ddp = [(fi, tr, te) for fi, (tr, te) in enumerate(folds)]
        _oof_preds_shared = torch.zeros(len(y), dtype=torch.float32).share_memory_()
        _oof_std_shared   = torch.zeros(len(y), dtype=torch.float32).share_memory_()
        # Move tensors to CPU shared memory before spawn
        _fold_shared["tensors"] = {
            k: (v.cpu().share_memory_() if isinstance(v, torch.Tensor) else v)
            for k, v in tensors.items()
        }
        if alpha_targets is not None:
            _fold_shared["alpha_targets"] = alpha_targets.cpu().share_memory_()
        _tmp.spawn(
            _ddp_fold_worker,
            nprocs=_world_size_ddp,
            args=(_world_size_ddp, _fold_list_ddp, _fold_shared,
                  _oof_preds_shared, _oof_std_shared),
            join=True,
        )
        oof_preds = _oof_preds_shared.numpy().copy()
        oof_std   = _oof_std_shared.numpy().copy()
    else:
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            test_out, oof_p, oof_s = _exec_cv_fold(
                fold_idx, train_idx, test_idx, _fold_shared, device,
            )
            oof_preds[test_out] = oof_p
            oof_std[test_out]   = oof_s

    # ---- Denormalise OOF predictions back to original scale ----
    oof_preds = oof_preds * y_std + y_mean
    oof_std = oof_std * y_std  # uncertainty scales linearly

    # ---- Handle NaN in OOF predictions ----
    nan_mask = np.isnan(oof_preds)
    n_nan = int(nan_mask.sum())
    if n_nan > 0:
        logger.warning("V2 Neural OOF has %d NaN predictions (%.1f%%) — replacing with mean", n_nan, 100.0 * n_nan / len(oof_preds))
        oof_preds[nan_mask] = np.nanmean(oof_preds)
        oof_std[nan_mask] = np.nanmean(oof_std)

    # ---- OOF Metrics ----
    r2 = float(r2_score(y, oof_preds))
    rmse = float(np.sqrt(mean_squared_error(y, oof_preds)))
    logger.info("V2 Neural Meta OOF  R²=%.4f  RMSE=%.4f", r2, rmse)

    # Quick-eval mode: return metrics only (used by CMA-ES trials)
    if quick_eval:
        return {
            "model": None,
            "process_rate": None,
            "source_term": None,
            "surrogates": None,
            "oof_predictions": oof_preds,
            "oof_uncertainty": oof_std,
            "metrics": {"r2": r2, "rmse": rmse},
            "encoder": tensors["encoder"],
            "meta_info": None,
        }

    # ==================================================================
    # Full retrain for deployment (main training + SWA)
    # ==================================================================
    logger.info(
        "Retraining on full dataset (%d main + %d SWA epochs)...",
        main_epochs, swa_epochs,
    )

    final_process = ProcessRateNet(
        n_inputs=pr_input_dim,
        domain_config={
            "name": pr_cfg.get("name", "rate"),
            "units": pr_cfg.get("units", ""),
            "bounds": pr_cfg.get("bounds", [0.0, 1.0]),
            "prior_mean": pr_cfg.get("prior_mean", 0.5),
        },
        n_treatments=n_treatments,
    ).to(device)

    final_model = SPARCMetaLearner(
        n_base_models=n_base,
        n_physics_features=n_physics_extended,
        d_spatial=d_spatial,
        hidden_dim=hidden_dim,
        dropout=dropout,
        thresholds=thresholds,
        n_heads=n_heads,
        max_neighbors=max_neighbors,
        siren_omega=siren_omega,
        init_bandwidth=float(predictor_bandwidths.mean()) if predictor_bandwidths is not None else 1000.0,
        time_embed_dim=time_embed_dim,
    ).to(device)

    # OT trunk alignment: register forward hook to capture trunk_fusion output
    if _ot_active:
        _trunk_acts_ret_bucket.clear()
        _trunk_acts_swa_bucket.clear()
        final_model.trunk_fusion.register_forward_hook(_make_trunk_hook(_trunk_acts_ret_bucket))
        final_model.trunk_fusion.register_forward_hook(_make_trunk_hook(_trunk_acts_swa_bucket))

    # Transfer JEPA-pretrained trunk weights if available
    if jepa_pretrained_trunk_state is not None:
        final_model.load_state_dict(jepa_pretrained_trunk_state, strict=False)
        logger.info("Full retrain: transferred JEPA-pretrained trunk")

    final_surrogates = {
        "gwr": DifferentiableGWR(
            n_vars=n_physics, n_spatial_features=d_spatial,
            hidden_dim=hidden_dim,
            bandwidths=predictor_bandwidths,
            kernel_field=predictor_kernel_field,
            feature_names=feature_names,
        ).to(device),
        "gwrf": DifferentiableGWRF(
            n_vars=n_physics, n_spatial_features=d_spatial,
            hidden_dim=hidden_dim,
            kernel_field=predictor_kernel_field,
            feature_names=feature_names,
        ).to(device),
        "ggpgam": DifferentiableGGPGAM(
            n_vars=n_physics, n_spatial_features=d_spatial,
            hidden_dim=min(hidden_dim, 64),
        ).to(device),
    }

    # FIX A1: Learned source term for full retrain
    final_source_net = SourceTermNet(n_inputs=n_physics).to(device)

    # ── E-Perf-A: compile forward passes when torch ≥ 2.0 ──────────────
    final_model = _maybe_compile(final_model)
    final_process = _maybe_compile(final_process)
    final_source_net = _maybe_compile(final_source_net)
    for _k in list(final_surrogates):
        final_surrogates[_k] = _maybe_compile(final_surrogates[_k])

    final_optimizer = build_optimizer(
        final_model, final_process, final_surrogates, base_lr=lr,
        source_term_net=final_source_net,
    )
    final_scheduler = build_scheduler(
        final_optimizer, main_epochs, warmup_epochs=warmup_epochs,
    )

    # ---- JEPA target trunk + latent predictor (final retrain) ----
    final_ema_trunk: EMATrunk | None = None
    final_latent_predictor: LatentPredictor | None = None
    final_action_embed: ActionEmbedding | None = None
    final_jepa_optimizer: torch.optim.Optimizer | None = None
    if jepa_enable:
        final_ema_trunk = EMATrunk(
            final_model,
            tau_start=jepa_ema_tau_start,
            tau_end=jepa_ema_tau_end,
            warmup_steps=jepa_warmup_steps,
        ).to(device)
        use_action_final = (
            jepa_lambda_scenario > 0.0 or jepa_predictor_blocks > 1
        )
        action_dim_final = jepa_action_dim if use_action_final else 0
        final_latent_predictor = LatentPredictor(
            hidden_dim=hidden_dim,
            action_dim=action_dim_final,
            dropout=dropout,
            n_blocks=jepa_predictor_blocks,
            film=jepa_predictor_film,
        ).to(device)
        final_jepa_params = list(final_latent_predictor.parameters())
        if use_action_final:
            final_action_embed = ActionEmbedding(
                treatment_vocab=feature_names,
                embed_dim=action_dim_final,
            ).to(device)
            final_jepa_params += list(final_action_embed.parameters())
        final_jepa_optimizer = torch.optim.AdamW(final_jepa_params, lr=lr)

    full_knn = tensors["knn_index"]
    full_knn_dists_rt = tensors["knn_dists"]
    full_cardinal = tensors["cardinal_idx"]

    final_model.train()
    final_process.train()
    final_source_net.train()
    for s in final_surrogates.values():
        s.train()

    _retrain_t0 = _time.perf_counter()

    # ---- Surrogate pre-training on full data ----
    # Build full-data base-model targets (normalised) if available
    _full_base_targets: dict[str, torch.Tensor] | None = None
    if base_oof_predictions is not None:
        _full_base_targets = {}
        for sname in ("gwr", "gwrf", "ggpgam"):
            if sname in base_oof_predictions:
                _raw = base_oof_predictions[sname]
                _norm = (_raw - y_mean) / y_std
                _full_base_targets[sname] = torch.tensor(
                    _norm, dtype=torch.float32, device=device,
                )
        if not _full_base_targets:
            _full_base_targets = None

    if pretrain_epochs > 0:
        logger.info(
            "  Pre-training surrogates on full data (%d epochs)...",
            pretrain_epochs,
        )
        _pretrain_surrogates(
            final_surrogates,
            tensors["physics_feats"],
            tensors["X_spatial"],
            tensors["y"],
            n_epochs=pretrain_epochs,
            lr=lr,
            base_targets=_full_base_targets,
        )

    # ---- Validate surrogates before full-retrain joint training ----
    if _full_base_targets:
        from sparc.models.surrogates import validate_surrogates
        validation_results = validate_surrogates(
            surrogates=final_surrogates,
            true_predictions=_full_base_targets,
            X=tensors["physics_feats"],
            spatial_features=tensors["X_spatial"],
            threshold=0.85,
        )
        failed = [name for name, res in validation_results.items()
                   if not res['passed']]
        if failed:
            raise RuntimeError(
                f"Full-retrain surrogate validation failed for: {failed}. "
                f"Joint training aborted — fix surrogate architecture "
                f"before proceeding."
            )
        logger.info("  All full-retrain surrogates passed validation (R² >= 0.85)")

    # ---- ProcessRateNet pre-training (full retrain) ----
    if pr_pretrain_epochs > 0 and material_priors:
        logger.info(
            "  Pre-training final ProcessRateNet with land-cover classification "
            "(%d epochs, monotonicity reg)...", pr_pretrain_epochs,
        )
        pr_final_loss_rt = _pretrain_process_rate_net(
            process_net=final_process,
            physics_feats=tensors["physics_feats"],
            feature_names=feature_names,
            pr_col_idxs=pr_col_idxs,
            material_priors=material_priors,
            n_epochs=pr_pretrain_epochs,
            device=device,
            feat_mean=tensors["feat_mean"],
            feat_std=tensors["feat_std"],
        )
        logger.info(
            "  Final ProcessRateNet pre-training done — loss=%.6f",
            pr_final_loss_rt,
        )

    # Cache surrogate fidelity targets — use base-model targets when
    # available (same logic as CV fold path), else cache pretrained output.
    with torch.no_grad():
        _final_surr_targets = {}
        for sname, surr in final_surrogates.items():
            if _full_base_targets and sname in _full_base_targets:
                _final_surr_targets[sname] = _full_base_targets[sname].detach()
            else:
                surr.eval()
                if sname == "gwr":
                    _out, _ = surr(tensors["physics_feats"], tensors["X_spatial"])
                else:
                    _out = surr(tensors["physics_feats"], tensors["X_spatial"])
                _final_surr_targets[sname] = _out.squeeze().detach()
    # Re-enable training mode
    for s in final_surrogates.values():
        s.train()

    # ---- Stages A/B/C: Main training ----
    N_full = len(y)
    retrain_loss_history: list[dict[str, float]] = []
    use_minibatch_retrain = N_full > batch_size * 2
    if use_minibatch_retrain:
        from sparc.training.optimizer import spatial_minibatch_sampler
        full_cardinal_np = full_cardinal.cpu().numpy()

    for epoch in range(main_epochs):
        lambdas = get_lambda_schedule(
            epoch, target_lambdas,
            warmup_end=warmup_epochs,
            ramp_end=ramp_epochs,
        )

        # ---- Curriculum transition markers ----
        if epoch == 0:
            logger.info("[CURRICULUM] Stage A: Representation Warmup")
        elif epoch == warmup_epochs:
            logger.info("[CURRICULUM] Stage B: Physics Activation")
        elif epoch == ramp_epochs:
            logger.info("[CURRICULUM] Stage C: Joint Optimization")

        if use_minibatch_retrain:
            rt_batches = list(spatial_minibatch_sampler(
                coords, full_cardinal_np,
                batch_size=batch_size,
                n_batches=max(1, N_full // batch_size),
            ))
        else:
            rt_batches = [np.arange(N_full)]

        rt_epoch_loss = 0.0
        rt_epoch_n = 0
        rt_epoch_components: dict[str, float] = {}

        for b_idx in rt_batches:
            # CU-9b: pad last (short) batch to static size for CUDA Graph
            _valid_mask_rt: torch.Tensor | None = None
            if _use_cuda_graphs and len(b_idx) < batch_size:
                b_idx, _valid_mask_rt = _pad_batch_to_size(b_idx, batch_size, device)

            b_phys = tensors["physics_feats"][b_idx]
            b_phys_ext = tensors["physics_feats_extended"][b_idx]
            b_spat = tensors["X_spatial"][b_idx]
            b_coord = tensors["coords"][b_idx]
            b_y = tensors["y"][b_idx]
            b_h_rt = tensors["h_field"][b_idx] if tensors["h_field"] is not None else resolution
            b_water_mask_rt = tensors["water_mask"][b_idx] if tensors.get("water_mask") is not None else None
            b_surr_tgt = {k: v[b_idx] for k, v in _final_surr_targets.items()}
            b_knn = _remap_indices_to_local(
                np.arange(N_full), b_idx, full_knn[b_idx],
            )
            b_card = _remap_indices_to_local(
                np.arange(N_full), b_idx, full_cardinal[b_idx],
            )

            # GWRF kernel neighbors
            b_gwrf_knn_rt = _remap_indices_to_local(
                np.arange(N_full), b_idx,
                full_knn[b_idx][:, :gwrf_k],
            )
            b_gwrf_dists_rt = full_knn_dists_rt[b_idx][:, :gwrf_k]

            # Forward: surrogates → meta-learner  (inside AMP context)
            _fwd_amp_rt = torch.autocast("cuda", dtype=torch.bfloat16, enabled=_use_amp)
            _fwd_amp_rt.__enter__()
            surrogate_preds, surrogate_std = _forward_surrogates(
                final_surrogates, b_phys, b_spat,
                knn_index=b_gwrf_knn_rt, knn_dists=b_gwrf_dists_rt,
            )
            base_input = torch.stack(surrogate_preds, dim=1)

            pr_input = b_phys[:, pr_col_idxs]
            alpha_full = final_process(pr_input)
            alpha = (
                alpha_full
                if alpha_full.shape[-1] == 1
                else alpha_full.mean(dim=-1, keepdim=True)
            )
            alpha_prior = torch.full_like(alpha, prior_mean)

            # FIX A1: learned source term
            b_src = final_source_net(b_phys).squeeze(-1)

            T_pred, exceedance, _ = final_model(
                base_preds=base_input,
                physics_feats=b_phys_ext,
                X_spatial=b_spat,
                coords=b_coord,
                knn_index=b_knn,
                alpha=alpha,
                surrogate_std=surrogate_std,
                knn_dists=full_knn_dists_rt[b_idx],
                time_idx=tensors["time_idx"][b_idx] if tensors["time_idx"] is not None else None,
            )
            # Close AMP context; cast mixed-precision outputs to FP32 for PDE loss.
            _fwd_amp_rt.__exit__(None, None, None)
            if _use_amp:
                T_pred = T_pred.float()
                if isinstance(exceedance, torch.Tensor):
                    exceedance = exceedance.float()

            # ---- JEPA forward (final retrain) ----
            b_jepa_h_pred = None
            b_jepa_h_target = None
            b_jepa_scenario_h_pred = None
            b_jepa_scenario_h_target = None
            b_jepa_latent_h = None
            b_jepa_latent_neighbor = None
            b_jepa_latent_alpha = None
            jepa_lambda_eff = 0.0
            jepa_lambda_scenario_eff = 0.0
            jepa_lambda_latent_pde_eff = 0.0
            if jepa_enable:
                curric = jepa_curriculum_weight(
                    epoch,
                    warmup_start=jepa_curric_start,
                    warmup_end=jepa_curric_end,
                )
                jepa_lambda_eff = jepa_lambda * curric
                jepa_lambda_scenario_eff = jepa_lambda_scenario * curric
                jepa_lambda_latent_pde_eff = jepa_lambda_latent_pde * curric

                h_state_online: torch.Tensor | None = None

                if jepa_lambda_eff > 0.0:
                    if jepa_spatial_patch:
                        _rt_coords = coords_t[b_idx]
                        _rt_patch_mask = spatial_patch_mask(
                            _rt_coords, mask_ratio=jepa_mask_ratio,
                            n_patches=jepa_n_patches,
                        )
                        b_phys_ctx_rt = b_phys_ext.clone()
                        b_phys_ctx_rt[_rt_patch_mask] = 0.0
                        channel_mask = None
                    else:
                        b_phys_ctx_rt = b_phys_ext
                        n_phys_feat = b_phys_ext.shape[-1]
                        keep_p = max(0.0, 1.0 - jepa_mask_ratio)
                        channel_mask = (
                            torch.rand(n_phys_feat, device=device) < keep_p
                        ).to(b_phys_ext.dtype)
                    b_jepa_h_pred = final_latent_predictor(
                        final_model.encode(
                            physics_feats=b_phys_ctx_rt,
                            alpha=alpha,
                            channel_mask=channel_mask,
                        )
                    )
                    b_jepa_h_target = final_ema_trunk.encode_target(
                        physics_feats=b_phys_ext, alpha=alpha,
                    )

                if (
                    jepa_lambda_scenario_eff > 0.0
                    and final_action_embed is not None
                    and len(feature_names) > 0
                ):
                    treat_pos = int(torch.randint(
                        0, len(feature_names), (1,), device=device,
                    ).item())
                    delta_x_val = float(
                        torch.randn((), device=device).item()
                        * jepa_scenario_perturb_std
                    )

                    b_phys_perturbed = b_phys_ext.clone()
                    b_phys_perturbed[:, treat_pos] = (
                        b_phys_perturbed[:, treat_pos] + delta_x_val
                    )

                    if h_state_online is None:
                        h_state_online = final_model.encode(
                            physics_feats=b_phys_ext, alpha=alpha,
                        )

                    n_pts = b_phys_ext.shape[0]
                    treat_idx = torch.full(
                        (n_pts,), treat_pos, dtype=torch.long, device=device,
                    )
                    dx = torch.full(
                        (n_pts,), delta_x_val,
                        dtype=b_phys_ext.dtype, device=device,
                    )
                    dt = torch.zeros_like(dx)
                    a_emb = final_action_embed(treat_idx, dx, dt)

                    b_jepa_scenario_h_pred = final_latent_predictor(
                        h_state_online, a_emb,
                    )
                    b_jepa_scenario_h_target = final_ema_trunk.encode_target(
                        physics_feats=b_phys_perturbed, alpha=alpha,
                    )

                if (
                    jepa_lambda_latent_pde_eff > 0.0
                    and b_card is not None
                    and b_card.numel() > 0
                ):
                    if h_state_online is None:
                        h_state_online = final_model.encode(
                            physics_feats=b_phys_ext, alpha=alpha,
                        )
                    b_jepa_latent_h = h_state_online
                    b_jepa_latent_neighbor = b_card
                    b_jepa_latent_alpha = alpha

            total_loss, components = sparc_joint_loss(
                T_pred=T_pred,
                exceedance_preds=exceedance,
                y_true=b_y,
                thresholds=thresholds_norm,
                alpha=alpha,
                alpha_prior=alpha_prior,
                neighbor_idx=b_card,
                source_term=b_src,
                resolution=resolution,
                surrogate_preds=surrogate_preds,
                surrogate_targets=[
                    b_surr_tgt["gwr"],
                    b_surr_tgt["gwrf"],
                    b_surr_tgt["ggpgam"],
                ],
                lambda_physics=lambdas.get("physics", 0.0),
                lambda_smooth=lambdas.get("smooth", 0.0),
                lambda_alpha_smooth=lambdas.get("alpha_smooth", 0.0),
                lambda_prior=lambdas.get("prior", 1.0),
                lambda_base=lambdas.get("base", 0.0),
                lambda_neighbor=lambdas.get("neighbor", 0.0),
                epoch=epoch,
                h_field=b_h_rt,
                lambda_pde=lambdas.get("pde", 0.0),
                lambda_bc=lambdas.get("bc", 0.0),
                water_mask=b_water_mask_rt,
                T_water=tensors.get("T_water"),
                jepa_h_pred=b_jepa_h_pred,
                jepa_h_target=b_jepa_h_target,
                jepa_weights=jepa_weights,
                lambda_jepa=jepa_lambda_eff,
                jepa_scenario_h_pred=b_jepa_scenario_h_pred,
                jepa_scenario_h_target=b_jepa_scenario_h_target,
                lambda_jepa_scenario=jepa_lambda_scenario_eff,
                jepa_latent_h=b_jepa_latent_h,
                jepa_latent_neighbor_idx=b_jepa_latent_neighbor,
                jepa_latent_alpha=b_jepa_latent_alpha,
                lambda_jepa_latent_pde=jepa_lambda_latent_pde_eff,
                sparse_laplacian=tensors.get("sparse_laplacian"),
                valid_laplacian_mask=tensors.get("valid_laplacian_mask"),
                sheaf_delta=tensors.get("sheaf_delta"),
                valid_mask=_valid_mask_rt,
            )

            # Variance penalty on w(s): encourage spatial diversity
            if hasattr(final_model, "_last_w_source") and final_model._last_w_source is not None:
                total_loss = total_loss - 0.01 * final_model._last_w_source.var()

            # Alpha supervision loss: maintain land-cover signal
            if alpha_targets is not None:
                total_loss = total_loss + 0.1 * torch.nn.functional.mse_loss(alpha, alpha_targets[b_idx])

            # EWC penalty (continual learning across cities)
            if _ewc_active:
                total_loss = total_loss + _ewc_lambda * _ewc_penalty(
                    final_model, _cont_fisher, _cont_optimal, _TRUNK_KEYS,
                )

            # Wasserstein trunk alignment (OT continual learning)
            if _ot_active:
                _trunk_acts_ret = _trunk_acts_ret_bucket.pop() if _trunk_acts_ret_bucket else None
                if _trunk_acts_ret is not None:
                    ot_loss = _wta(_trunk_acts_ret, _coreset_acts.to(_trunk_acts_ret.device))
                    total_loss = total_loss + _ot_lambda * ot_loss

            final_optimizer.zero_grad()
            if final_jepa_optimizer is not None:
                final_jepa_optimizer.zero_grad()
            _scaler.scale(total_loss).backward()
            all_p = list(final_model.parameters()) + list(final_process.parameters())
            for s in final_surrogates.values():
                all_p.extend(s.parameters())
            if final_latent_predictor is not None:
                all_p.extend(final_latent_predictor.parameters())
            if final_action_embed is not None:
                all_p.extend(final_action_embed.parameters())
            _scaler.unscale_(final_optimizer)
            if final_jepa_optimizer is not None:
                _scaler.unscale_(final_jepa_optimizer)
            torch.nn.utils.clip_grad_norm_(all_p, clip_norm)
            _scaler.step(final_optimizer)
            if final_jepa_optimizer is not None:
                _scaler.step(final_jepa_optimizer)
            _scaler.update()
            _any_jepa_rt = (
                jepa_lambda_eff > 0.0
                or jepa_lambda_scenario_eff > 0.0
                or jepa_lambda_latent_pde_eff > 0.0
            )
            if final_ema_trunk is not None and _any_jepa_rt:
                final_ema_trunk.update(final_model)

            bsz = int(_valid_mask_rt.sum().item()) if _valid_mask_rt is not None else len(b_idx)
            rt_epoch_loss += total_loss.item() * bsz
            rt_epoch_n += bsz
            for k, v in components.items():
                rt_epoch_components[k] = rt_epoch_components.get(k, 0) + v * bsz

        final_scheduler.step()
        rt_epoch_loss /= max(rt_epoch_n, 1)
        for k in rt_epoch_components:
            rt_epoch_components[k] /= max(rt_epoch_n, 1)
        retrain_loss_history.append(dict(rt_epoch_components))

        # Log every epoch for live training telemetry
        logger.info(
            "  Retrain %d/%d  loss=%.4f  "
            "[mse=%.3f phys=%.3f nbr=%.3f ce=%.3f "
            "pde=%.3f bc=%.3f prior=%.3f base=%.3f]  (%.1fs)",
            epoch + 1, main_epochs, rt_epoch_loss,
            rt_epoch_components.get("mse", 0),
            rt_epoch_components.get("physics", 0),
            rt_epoch_components.get("neighborhood", 0),
            rt_epoch_components.get("cross_entropy", 0),
            rt_epoch_components.get("pde_total", 0),
            rt_epoch_components.get("bc_total", 0),
            rt_epoch_components.get("alpha_prior", 0),
            rt_epoch_components.get("surrogate", 0),
            _time.perf_counter() - _retrain_t0,
        )

    # ---- Stage D: Stochastic Weight Averaging ----
    if swa_epochs > 0:
        from torch.optim.swa_utils import AveragedModel, SWALR

        logger.info("Starting SWA phase (%d epochs)...", swa_epochs)
        logger.info("[CURRICULUM] Stage D: Stochastic Weight Averaging")

        # Detach non-leaf tensors before deepcopy (AveragedModel uses deepcopy)
        if hasattr(final_model, "_last_w_source") and final_model._last_w_source is not None:
            final_model._last_w_source = final_model._last_w_source.detach()

        swa_model = AveragedModel(final_model, device=device)
        swa_process = AveragedModel(final_process, device=device)
        swa_surrogates_avg = {
            k: AveragedModel(v, device=device)
            for k, v in final_surrogates.items()
        }

        swa_lr_max = training_cfg.get("swa_lr_max", lr * 0.5)
        swa_scheduler = SWALR(
            final_optimizer,
            swa_lr=swa_lr_max,
            anneal_epochs=max(swa_epochs // 2, 1),
            anneal_strategy="cos",
        )

        for swa_ep in range(swa_epochs):
            epoch_global = main_epochs + swa_ep
            lambdas = get_lambda_schedule(
                epoch_global, target_lambdas,
                warmup_end=warmup_epochs,
                ramp_end=ramp_epochs,
            )

            if use_minibatch_retrain:
                swa_batches = list(spatial_minibatch_sampler(
                    coords, full_cardinal_np,
                    batch_size=batch_size,
                    n_batches=max(1, N_full // batch_size),
                ))
            else:
                swa_batches = [np.arange(N_full)]

            swa_epoch_loss = 0.0
            swa_epoch_n = 0

            for b_idx in swa_batches:
                # CU-9b: pad last (short) batch to static size for CUDA Graph
                _valid_mask_swa: torch.Tensor | None = None
                if _use_cuda_graphs and len(b_idx) < batch_size:
                    b_idx, _valid_mask_swa = _pad_batch_to_size(b_idx, batch_size, device)

                b_phys = tensors["physics_feats"][b_idx]
                b_phys_ext = tensors["physics_feats_extended"][b_idx]
                b_spat = tensors["X_spatial"][b_idx]
                b_coord = tensors["coords"][b_idx]
                b_y = tensors["y"][b_idx]
                b_h_swa = tensors["h_field"][b_idx] if tensors["h_field"] is not None else resolution
                b_water_mask_swa = tensors["water_mask"][b_idx] if tensors.get("water_mask") is not None else None
                b_surr_tgt = {k: v[b_idx] for k, v in _final_surr_targets.items()}
                b_knn = _remap_indices_to_local(
                    np.arange(N_full), b_idx, full_knn[b_idx],
                )
                b_card = _remap_indices_to_local(
                    np.arange(N_full), b_idx, full_cardinal[b_idx],
                )

                # GWRF kernel neighbors for SWA
                b_gwrf_knn_swa = _remap_indices_to_local(
                    np.arange(N_full), b_idx,
                    full_knn[b_idx][:, :gwrf_k],
                )
                b_gwrf_dists_swa = full_knn_dists_rt[b_idx][:, :gwrf_k]

                # Forward: surrogates → meta-learner  (inside AMP context)
                _fwd_amp_swa = torch.autocast("cuda", dtype=torch.bfloat16, enabled=_use_amp)
                _fwd_amp_swa.__enter__()
                surrogate_preds, surrogate_std_swa = _forward_surrogates(
                    final_surrogates, b_phys, b_spat,
                    knn_index=b_gwrf_knn_swa, knn_dists=b_gwrf_dists_swa,
                )
                base_input = torch.stack(surrogate_preds, dim=1)

                pr_input = b_phys[:, pr_col_idxs]
                alpha_full = final_process(pr_input)
                alpha = (
                    alpha_full
                    if alpha_full.shape[-1] == 1
                    else alpha_full.mean(dim=-1, keepdim=True)
                )
                alpha_prior = torch.full_like(alpha, prior_mean)

                # FIX A1: learned source term
                b_src = final_source_net(b_phys).squeeze(-1)

                T_pred, exceedance, _ = final_model(
                    base_preds=base_input,
                    physics_feats=b_phys_ext,
                    X_spatial=b_spat,
                    coords=b_coord,
                    knn_index=b_knn,
                    alpha=alpha,
                    surrogate_std=surrogate_std_swa,
                    knn_dists=full_knn_dists_rt[b_idx],
                    time_idx=tensors["time_idx"][b_idx] if tensors["time_idx"] is not None else None,
                )
                # Close AMP context; cast outputs to FP32 for PDE loss.
                _fwd_amp_swa.__exit__(None, None, None)
                if _use_amp:
                    T_pred = T_pred.float()
                    if isinstance(exceedance, torch.Tensor):
                        exceedance = exceedance.float()

                total_loss, _ = sparc_joint_loss(
                    T_pred=T_pred,
                    exceedance_preds=exceedance,
                    y_true=b_y,
                    thresholds=thresholds_norm,
                    alpha=alpha,
                    alpha_prior=alpha_prior,
                    neighbor_idx=b_card,
                    source_term=b_src,
                    resolution=resolution,
                    surrogate_preds=surrogate_preds,
                    surrogate_targets=[
                        b_surr_tgt["gwr"],
                        b_surr_tgt["gwrf"],
                        b_surr_tgt["ggpgam"],
                    ],
                    lambda_physics=lambdas.get("physics", 0.0),
                    lambda_smooth=lambdas.get("smooth", 0.0),
                    lambda_alpha_smooth=lambdas.get("alpha_smooth", 0.0),
                    lambda_prior=lambdas.get("prior", 1.0),
                    lambda_base=lambdas.get("base", 0.0),
                    lambda_neighbor=lambdas.get("neighbor", 0.0),
                    epoch=epoch_global,
                    h_field=b_h_swa,
                    lambda_pde=lambdas.get("pde", 0.0),
                    lambda_bc=lambdas.get("bc", 0.0),
                    water_mask=b_water_mask_swa,
                    T_water=tensors.get("T_water"),
                    sparse_laplacian=tensors.get("sparse_laplacian"),
                    valid_laplacian_mask=tensors.get("valid_laplacian_mask"),
                    sheaf_delta=tensors.get("sheaf_delta"),
                    valid_mask=_valid_mask_swa,
                )

                # Variance penalty on w(s): encourage spatial diversity
                if hasattr(final_model, "_last_w_source") and final_model._last_w_source is not None:
                    total_loss = total_loss - 0.01 * final_model._last_w_source.var()

                # Alpha supervision loss: maintain land-cover signal
                if alpha_targets is not None:
                    total_loss = total_loss + 0.1 * torch.nn.functional.mse_loss(alpha, alpha_targets[b_idx])

                # EWC penalty (continual learning across cities)
                if _ewc_active:
                    total_loss = total_loss + _ewc_lambda * _ewc_penalty(
                        final_model, _cont_fisher, _cont_optimal, _TRUNK_KEYS,
                    )

                # Wasserstein trunk alignment (OT continual learning)
                if _ot_active:
                    _trunk_acts_swa = _trunk_acts_swa_bucket.pop() if _trunk_acts_swa_bucket else None
                    if _trunk_acts_swa is not None:
                        ot_loss = _wta(_trunk_acts_swa, _coreset_acts.to(_trunk_acts_swa.device))
                        total_loss = total_loss + _ot_lambda * ot_loss

                final_optimizer.zero_grad()
                _scaler.scale(total_loss).backward()
                all_p = list(final_model.parameters()) + list(final_process.parameters())
                for s in final_surrogates.values():
                    all_p.extend(s.parameters())
                _scaler.unscale_(final_optimizer)
                torch.nn.utils.clip_grad_norm_(all_p, clip_norm)
                _scaler.step(final_optimizer)
                _scaler.update()

                bsz = int(_valid_mask_swa.sum().item()) if _valid_mask_swa is not None else len(b_idx)
                swa_epoch_loss += total_loss.item() * bsz
                swa_epoch_n += bsz

            swa_epoch_loss /= max(swa_epoch_n, 1)

            # Update SWA running averages
            swa_model.update_parameters(final_model)
            swa_process.update_parameters(final_process)
            for k in swa_surrogates_avg:
                swa_surrogates_avg[k].update_parameters(final_surrogates[k])
            swa_scheduler.step()

            if (swa_ep + 1) % 5 == 0 or swa_ep == 0:
                logger.info(
                    "  SWA epoch %d/%d  loss=%.4f  (%.1fs)",
                    swa_ep + 1, swa_epochs, swa_epoch_loss,
                    _time.perf_counter() - _retrain_t0,
                )
                # (SWA components are not tracked per-batch for speed)

        # Extract averaged weights
        final_model = swa_model.module
        final_process = swa_process.module
        final_surrogates = {k: v.module for k, v in swa_surrogates_avg.items()}
        logger.info("SWA complete — averaged weights extracted.")

    logger.info("Full retrain done in %.1fs", _time.perf_counter() - _retrain_t0)

    # ==================================================================
    # Save artifacts
    # ==================================================================
    artifact_dir = output_dir / "v2_neural"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Active store (single source of truth for db-resident artifacts).
    try:
        from sparc.registry.store import get_active_store
        _store = get_active_store()
    except Exception:
        _store = None

    # --- Model checkpoints + scalers (dual-write: db blobs + disk).
    # Consumer (scenario_simulator) still loads from disk paths; that
    # consumer migration is deferred to a future blob-cleanup pass.
    torch.save(final_model.state_dict(), artifact_dir / "neural_meta.pt")
    final_model.save_trunk(artifact_dir / "shared_trunk.pt")
    torch.save(final_process.state_dict(), artifact_dir / "process_rate_net.pt")
    torch.save(final_source_net.state_dict(), artifact_dir / "source_term_net.pt")
    for name, surr in final_surrogates.items():
        torch.save(surr.state_dict(), artifact_dir / f"surrogate_{name}.pt")
    joblib.dump(tensors["encoder"], artifact_dir / "sinusoidal_encoder.pkl")

    if _store is not None:
        try:
            _store.write_blob("2", "v2_neural_meta_state",
                              final_model.state_dict(),
                              serializer="torch", producer="v2_neural_training")
            _store.write_blob("2", "v2_process_rate_state",
                              final_process.state_dict(),
                              serializer="torch", producer="v2_neural_training")
            _store.write_blob("2", "v2_source_term_state",
                              final_source_net.state_dict(),
                              serializer="torch", producer="v2_neural_training")
            for name, surr in final_surrogates.items():
                _store.write_blob("2", f"v2_surrogate_{name}_state",
                                  surr.state_dict(),
                                  serializer="torch", producer="v2_neural_training")
            _store.write_blob("2", "v2_sinusoidal_encoder",
                              tensors["encoder"],
                              serializer="joblib", producer="v2_neural_training")
        except Exception as _e:
            logger.warning("Could not mirror neural checkpoints to artifacts.db: %s", _e)

    # Save alpha field for scenario simulation
    final_process.eval()
    with torch.no_grad():
        alpha_all = final_process(tensors["physics_feats"][:, pr_col_idxs])
    # Phase 3 (v4): alpha_all has shape (N, n_treatments).  The legacy
    # ``alpha_field.npy`` artifact remains 1-D (collapsed across heads)
    # so existing scenario-simulator readers are unaffected; the full
    # multi-head tensor is mirrored into artifacts.db with schema_version=2.
    alpha_full_np = alpha_all.cpu().numpy()  # (N, n_treatments)
    if alpha_full_np.shape[-1] == 1:
        alpha_field_np = alpha_full_np.squeeze(-1)
    else:
        alpha_field_np = alpha_full_np.mean(axis=-1)
    np.save(artifact_dir / "alpha_field.npy", alpha_field_np)
    np.save(artifact_dir / "alpha_field_coords.npy", coords)
    logger.info(
        "  Saved alpha_field.npy (%d points, %d treatment heads)",
        len(coords), alpha_full_np.shape[-1],
    )

    # Save cardinal neighbors and grid spacing for PDE forward solver
    cardinal_np = tensors["cardinal_idx"].cpu().numpy()
    np.save(artifact_dir / "cardinal_neighbors.npy", cardinal_np)
    grid_spacing_np = None
    if tensors["h_field"] is not None:
        grid_spacing_np = tensors["h_field"].cpu().numpy()
        np.save(artifact_dir / "grid_spacing.npy", grid_spacing_np)
    logger.info("  Saved cardinal_neighbors.npy + grid_spacing.npy")

    # Feature scaling stats (needed to standardize inputs at inference time)
    feat_scaling = {"feat_mean": tensors["feat_mean"], "feat_std": tensors["feat_std"]}
    np.savez(
        artifact_dir / "feature_scaling.npz",
        feat_mean=tensors["feat_mean"],
        feat_std=tensors["feat_std"],
    )

    oof_payload = {"predictions": oof_preds, "uncertainty": oof_std}
    np.savez(
        artifact_dir / "oof_results.npz",
        predictions=oof_preds,
        uncertainty=oof_std,
    )

    # Mirror numpy artifacts to artifacts.db (pickled dicts/arrays).
    if _store is not None:
        try:
            # Phase 3 (v4): persist full multi-head alpha (N, n_treatments)
            # alongside the collapsed (N,) field.  ``schema_version=2`` and
            # ``treatments`` metadata let the resolver index by treatment.
            _store.write_blob(
                "2", "v2_alpha_field", alpha_full_np,
                serializer="pickle", producer="v2_neural_training",
                consumers=[
                    "scenario_simulator",
                    "causal_effect_resolver",
                ],
                metadata={
                    "schema_version": 2,
                    "shape": list(alpha_full_np.shape),
                    "treatments": treatments if n_treatments > 1 else [],
                    "n_treatments": n_treatments,
                    "collapse": "mean" if n_treatments > 1 else "identity",
                },
            )
            _store.write_blob("2", "v2_alpha_field_coords", coords,
                              serializer="pickle", producer="v2_neural_training")
            _store.write_blob("2", "v2_cardinal_neighbors", cardinal_np,
                              serializer="pickle", producer="v2_neural_training")
            if grid_spacing_np is not None:
                _store.write_blob("2", "v2_grid_spacing", grid_spacing_np,
                                  serializer="pickle", producer="v2_neural_training")
            _store.write_blob("2", "v2_feature_scaling", feat_scaling,
                              serializer="pickle", producer="v2_neural_training")
            _store.write_blob("2", "v2_oof_results", oof_payload,
                              serializer="pickle", producer="v2_neural_training")
        except Exception as _e:
            logger.warning("Could not mirror neural numpy artifacts to artifacts.db: %s", _e)

    # Save per-epoch loss history for diagnostics
    if retrain_loss_history:
        _loss_keys = sorted(retrain_loss_history[0].keys())
        _loss_arrays = {
            k: np.array([ep.get(k, 0.0) for ep in retrain_loss_history])
            for k in _loss_keys
        }
        np.savez(artifact_dir / "loss_history.npz", **_loss_arrays)
        if _store is not None:
            try:
                _store.write_blob("2", "v2_loss_history", _loss_arrays,
                                  serializer="pickle", producer="v2_neural_training")
            except Exception as _e:
                logger.warning("Could not mirror loss_history to artifacts.db: %s", _e)
        logger.info(
            "  Saved loss_history.npz (%d epochs, %d components)",
            len(retrain_loss_history), len(_loss_keys),
        )

    meta_info = {
        "surrogate_names": ["gwr", "gwrf", "ggpgam"],
        "n_surrogates": n_base,
        "n_base_models": n_base,
        "n_physics_features": n_physics,
        "n_physics_original": tensors["n_physics_original"],
        "n_physics_extended": n_physics_extended,
        "has_alpha_field": True,
        "d_spatial": d_spatial,
        "hidden_dim": hidden_dim,
        "thresholds": thresholds,
        "thresholds_norm": thresholds_norm,
        "feature_names": feature_names,
        "process_rate_inputs": list(pr_inputs),
        "y_mean": y_mean,
        "y_std": y_std,
        "oof_r2": r2,
        "oof_rmse": rmse,
        "device": str(device),
        "n_epochs": n_epochs,
        "pretrain_epochs": pretrain_epochs,
        "main_epochs": main_epochs,
        "swa_epochs": swa_epochs,
        "resolution": resolution,
    }
    if _store is not None:
        try:
            _store.write_struct("2", "v2_neural_meta_info", meta_info,
                                producer="v2_neural_training")
        except Exception as _e:
            logger.warning("Could not write v2_neural_meta_info struct: %s", _e)
    with open(artifact_dir / "meta_info.json", "w") as f:
        json.dump(meta_info, f, indent=2)

    # Persist SpatialGatingHead weights — the final batch's softmax gate
    # vector (N, 3) showing how much each surrogate (GWR/GWRF/GGPGAM) was
    # weighted across the full training set.  Useful for diagnostics.
    try:
        gate_w = getattr(final_model, "_last_gate_weights", None)
        if gate_w is not None:
            gate_np = gate_w.cpu().numpy()  # (N_last_batch, 3)
            surrogate_names = ["gwr", "gwrf", "ggpgam"]
            gate_path = artifact_dir / "spatial_gate_weights.npz"
            np.savez(gate_path, gate_weights=gate_np, surrogate_names=np.array(surrogate_names))
            if _store is not None:
                try:
                    _store.write_blob(
                        "2", "v2_spatial_gate_weights",
                        {"gate_weights": gate_np, "surrogate_names": surrogate_names},
                        serializer="pickle", producer="v2_neural_training",
                        consumers=["desktop:model_performance", "report:/stage2/gating"],
                    )
                except Exception as _ge:
                    logger.warning("Could not persist gate weights to artifacts.db: %s", _ge)
            logger.info("  Saved spatial_gate_weights.npz (shape=%s)", gate_np.shape)
    except Exception as _ge:  # noqa: BLE001
        logger.warning("Gate weight persistence failed: %s", _ge)

    logger.info("V2 neural artifacts saved to %s", artifact_dir)

    # ==================================================================
    # Phase 5c: base / surrogate consistency check
    # Compares the surrogate's local-β surface against the classical
    # GWRModel's coefficients_ (when ``gwr_model_full`` is in the store).
    # Persisted as ``stage=2, gwr_base_vs_surrogate_alignment``.
    # ==================================================================
    try:
        _run_base_vs_surrogate_alignment(
            final_surrogates.get("gwr"),
            tensors["X_spatial"],
            feature_names,
        )
    except Exception as _e:  # noqa: BLE001
        logger.warning("Base/surrogate alignment audit failed: %s", _e)

    # ==================================================================
    # Export human-readable outputs (predictions, coefficients, maps)
    # ==================================================================
    _export_v2_outputs(
        final_model=final_model,
        final_process=final_process,
        final_surrogates=final_surrogates,
        tensors=tensors,
        coords=coords,
        y=y,
        oof_preds=oof_preds,
        oof_std=oof_std,
        feature_names=feature_names,
        pr_col_idxs=pr_col_idxs,
        gwrf_k=gwrf_k,
        thresholds=thresholds,
        thresholds_norm=thresholds_norm,
        y_mean=y_mean,
        y_std=y_std,
        artifact_dir=artifact_dir,
        device=device,
    )

    # ==================================================================
    # PDE diagnostic visualizations removed (Phase E):
    # the pipeline no longer produces PNG plots. Visualisations are
    # rendered live in the desktop app from artifacts.db data.
    # ==================================================================

    # --- Continual learning: extract trunk params + compute Fisher matrix ---
    from sparc.training.ewc import compute_fisher_matrix as _compute_fisher, extract_trunk_params as _extract_trunk
    _result_trunk = _extract_trunk(final_model, _TRUNK_KEYS)
    logger.info("Extracted trunk params for EWC (%d param tensors)", len(_result_trunk))

    def _fisher_data_loader():
        N_all = len(y)
        for _start in range(0, N_all, batch_size):
            _idx = np.arange(_start, min(_start + batch_size, N_all))
            with torch.no_grad():
                _b_phys = tensors["physics_feats"][_idx]
                _b_spat = tensors["X_spatial"][_idx]
                _b_coord = tensors["coords"][_idx]
                _b_knn = _remap_indices_to_local(
                    np.arange(N_all), _idx, full_knn[_idx]
                )
                _b_gwrf_knn = _b_knn[:, :gwrf_k]
                _b_gwrf_dists = full_knn_dists_rt[_idx][:, :gwrf_k]
                _b_preds, _ = _forward_surrogates(
                    final_surrogates, _b_phys, _b_spat,
                    knn_index=_b_gwrf_knn, knn_dists=_b_gwrf_dists,
                )
                _base_input = torch.stack(_b_preds, dim=1).detach()
                _pr_in = _b_phys[:, pr_col_idxs]
                _alpha_b = final_process(_pr_in).detach()
                if _alpha_b.shape[-1] > 1:
                    _alpha_b = _alpha_b.mean(dim=-1, keepdim=True)
            yield {
                "base_preds": _base_input,
                "physics_feats": tensors["physics_feats_extended"][_idx],
                "X_spatial": _b_spat,
                "coords": _b_coord,
                "knn_index": _b_knn,
                "alpha": _alpha_b,
            }, tensors["y"][_idx]

    _result_fisher = _compute_fisher(
        final_model, _fisher_data_loader(), _TRUNK_KEYS, device=device,
    )
    logger.info("Fisher matrix computed for EWC (%d trunk param tensors)", len(_result_fisher))

    return {
        "model": final_model,
        "process_rate": final_process,
        "source_term": final_source_net,
        "surrogates": final_surrogates,
        "oof_predictions": oof_preds,
        "oof_uncertainty": oof_std,
        "metrics": {"r2": r2, "rmse": rmse},
        "encoder": tensors["encoder"],
        "meta_info": meta_info,
        "fisher_matrix": _result_fisher,
        "trunk_state_dict": _result_trunk,
    }


# ---------------------------------------------------------------------------
# V2 output export (predictions, coefficients, maps, PDP curves)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _export_v2_outputs(
    *,
    final_model,
    final_process,
    final_surrogates,
    tensors: dict,
    coords: np.ndarray,
    y: np.ndarray,
    oof_preds: np.ndarray,
    oof_std: np.ndarray,
    feature_names: list[str],
    pr_col_idxs: list[int],
    gwrf_k: int,
    thresholds: list[float],
    thresholds_norm: list[float],
    y_mean: float,
    y_std: float,
    artifact_dir: Path,
    device: torch.device,
) -> None:
    """Export human-readable V2 neural outputs after training."""
    import pandas as pd

    logger.info("Exporting V2 neural readable outputs...")

    try:
        from sparc.registry.store import get_active_store
        _store = get_active_store()
    except Exception:
        _store = None
    _t0 = _time.perf_counter()

    final_model.eval()
    final_process.eval()
    for s in final_surrogates.values():
        s.eval()

    N = len(y)
    phys = tensors["physics_feats"]
    phys_ext = tensors["physics_feats_extended"]
    spat = tensors["X_spatial"]
    coords_t = tensors["coords"]
    knn_idx = tensors["knn_index"]
    knn_dists = tensors["knn_dists"]

    # --- 1. Full-data surrogate predictions + local coefficients ---
    gwrf_knn = knn_idx[:, :gwrf_k]
    gwrf_dists = knn_dists[:, :gwrf_k]

    gwr_pred, gwr_beta = final_surrogates["gwr"](phys, spat)
    gwrf_pred = final_surrogates["gwrf"](
        phys, spat, neighbor_idx=gwrf_knn, neighbor_dists=gwrf_dists,
    )
    ggpgam_pred = final_surrogates["ggpgam"](phys, spat)

    surrogate_preds = [gwr_pred, gwrf_pred, ggpgam_pred]
    base_input = torch.stack(surrogate_preds, dim=1)

    # --- 2. Process rate (alpha) ---
    pr_input = phys[:, pr_col_idxs]
    alpha_full = final_process(pr_input)  # (N, n_treatments)
    alpha = (
        alpha_full
        if alpha_full.shape[-1] == 1
        else alpha_full.mean(dim=-1, keepdim=True)
    )

    # --- 3. Full meta-learner inference ---
    T_pred_norm, exceedance_list, attn_weights = final_model(
        base_preds=base_input,
        physics_feats=phys_ext,
        X_spatial=spat,
        coords=coords_t,
        knn_index=knn_idx,
        alpha=alpha,
        knn_dists=knn_dists,
    )

    # De-normalise predictions
    T_pred_orig = T_pred_norm.cpu().numpy() * y_std + y_mean

    # --- Export 1: Predictions CSV ---
    pred_df = pd.DataFrame({
        "lon": coords[:, 0],
        "lat": coords[:, 1],
        "actual": y,
        "predicted_oof": oof_preds,
        "predicted_full": T_pred_orig,
        "residual_oof": y - oof_preds,
        "uncertainty": oof_std,
    })
    # Surrogate predictions (de-normalised)
    pred_df["surrogate_gwr"] = gwr_pred.cpu().numpy() * y_std + y_mean
    pred_df["surrogate_gwrf"] = gwrf_pred.cpu().numpy() * y_std + y_mean
    pred_df["surrogate_ggpgam"] = ggpgam_pred.cpu().numpy() * y_std + y_mean
    # Process rate
    pred_df["process_rate_alpha"] = alpha.squeeze(-1).cpu().numpy()

    # Exceedance probabilities
    for i, thresh in enumerate(thresholds):
        exc_prob = torch.sigmoid(exceedance_list[i]).cpu().numpy()
        pred_df[f"exceedance_p_{thresh:.2f}"] = exc_prob

    if _store is not None:
        try:
            _store.write_table("2", "v2_neural_predictions", pred_df,
                               producer="v2_neural_training")
        except Exception as _e:
            logger.warning("v2_neural_predictions store write failed: %s", _e)
            pred_df.to_csv(artifact_dir / "predictions.csv", index=False)
    else:
        pred_df.to_csv(artifact_dir / "predictions.csv", index=False)
    logger.info("  Saved predictions (%d rows, %d cols)", len(pred_df), len(pred_df.columns))

    # --- Export 2: GWR local coefficients ---
    coeff_df = pd.DataFrame({
        "lon": coords[:, 0],
        "lat": coords[:, 1],
    })
    beta_np = gwr_beta.cpu().numpy()  # (N, n_vars)
    for j, fname in enumerate(feature_names):
        coeff_df[f"beta_{fname}"] = beta_np[:, j]
    # Intercept from the coefficient head
    h_base = final_surrogates["gwr"].spatial_embed(spat)
    h_base = h_base + final_surrogates["gwr"].res_block(h_base)
    bw_ctx = final_surrogates["gwr"].bw_condition(final_surrogates["gwr"].bw_scale)
    h = h_base + bw_ctx
    all_coeff = final_surrogates["gwr"].coeff_head(h)
    coeff_df["beta_intercept"] = all_coeff[:, -1].cpu().numpy()

    if _store is not None:
        try:
            _store.write_table("2", "v2_neural_local_coefficients_gwr",
                               coeff_df, producer="v2_neural_training")
        except Exception as _e:
            logger.warning("local_coefficients_gwr store write failed: %s", _e)
            coeff_df.to_csv(artifact_dir / "local_coefficients_gwr.csv", index=False)
    else:
        coeff_df.to_csv(artifact_dir / "local_coefficients_gwr.csv", index=False)
    logger.info("  Saved local_coefficients_gwr (%d rows)", len(coeff_df))

    # --- Export 3: Process rate map ---
    alpha_df = pd.DataFrame({
        "lon": coords[:, 0],
        "lat": coords[:, 1],
        "alpha": alpha.squeeze(-1).cpu().numpy(),
    })
    if _store is not None:
        try:
            _store.write_table("2", "v2_neural_process_rate_map", alpha_df,
                               producer="v2_neural_training")
        except Exception as _e:
            logger.warning("process_rate_map store write failed: %s", _e)
            alpha_df.to_csv(artifact_dir / "process_rate_map.csv", index=False)
    else:
        alpha_df.to_csv(artifact_dir / "process_rate_map.csv", index=False)
    logger.info("  Saved process_rate_map")

    # --- Export 4: Meta-learner attention weights (summary stats) ---
    attn_np = attn_weights.cpu().numpy()  # (N, max_neighbors)
    attn_summary = pd.DataFrame({
        "lon": coords[:, 0],
        "lat": coords[:, 1],
        "attn_entropy": -np.sum(
            attn_np * np.log(attn_np + 1e-10), axis=1,
        ),
        "attn_max": attn_np.max(axis=1),
        "attn_effective_neighbors": 1.0 / np.sum(attn_np ** 2, axis=1),
    })
    if _store is not None:
        try:
            _store.write_table("2", "v2_neural_spatial_attention_summary",
                               attn_summary, producer="v2_neural_training")
        except Exception as _e:
            logger.warning("spatial_attention_summary store write failed: %s", _e)
            attn_summary.to_csv(artifact_dir / "spatial_attention_summary.csv", index=False)
    else:
        attn_summary.to_csv(artifact_dir / "spatial_attention_summary.csv", index=False)
    logger.info("  Saved spatial_attention_summary")

    # --- Export 5: PDP curves (sweep each feature) ---
    pdp_dir = artifact_dir / "pdp"
    pdp_dir.mkdir(exist_ok=True)

    # Sample a subset for PDP computation (full dataset too expensive)
    n_pdp = min(5000, N)
    pdp_rng = np.random.RandomState(42)
    pdp_idx = pdp_rng.choice(N, n_pdp, replace=False)
    pdp_idx.sort()

    n_grid = 50  # grid points per feature
    feat_mean = tensors["feat_mean"]
    feat_std_arr = tensors["feat_std"]

    for j, fname in enumerate(feature_names):
        # Get original-scale range for this feature
        raw_col = coords  # placeholder — we need raw feature values
        # Reconstruct raw feature values from scaled
        raw_vals = phys[:, j].cpu().numpy() * feat_std_arr[j] + feat_mean[j]
        grid = np.linspace(
            np.percentile(raw_vals, 1),
            np.percentile(raw_vals, 99),
            n_grid,
        )

        pdp_means = []
        pdp_q10 = []
        pdp_q90 = []

        for g_val in grid:
            # Scale the grid value
            g_scaled = float((g_val - feat_mean[j]) / feat_std_arr[j])

            # Copy physics features for the PDP subset and replace column j
            phys_mod = phys[pdp_idx].clone()
            phys_mod[:, j] = g_scaled

            # Extended features: replace col j in original portion, keep derivatives
            phys_ext_mod = phys_ext[pdp_idx].clone()
            phys_ext_mod[:, j] = g_scaled

            spat_sub = spat[pdp_idx]
            coords_sub = coords_t[pdp_idx]
            knn_sub = _remap_indices_to_local(
                np.arange(N), pdp_idx, knn_idx[pdp_idx],
            )
            gwrf_knn_sub = knn_sub[:, :gwrf_k]
            gwrf_dists_sub = knn_dists[pdp_idx][:, :gwrf_k]

            gwr_p, _ = final_surrogates["gwr"](phys_mod, spat_sub)
            gwrf_p = final_surrogates["gwrf"](
                phys_mod, spat_sub,
                neighbor_idx=gwrf_knn_sub, neighbor_dists=gwrf_dists_sub,
            )
            ggp_p = final_surrogates["ggpgam"](phys_mod, spat_sub)

            base_in = torch.stack([gwr_p, gwrf_p, ggp_p], dim=1)
            pr_in = phys_mod[:, pr_col_idxs]
            a_full = final_process(pr_in)
            a = (
                a_full
                if a_full.shape[-1] == 1
                else a_full.mean(dim=-1, keepdim=True)
            )

            t_pred, _, _ = final_model(
                base_preds=base_in,
                physics_feats=phys_ext_mod,
                X_spatial=spat_sub,
                coords=coords_sub,
                knn_index=knn_sub,
                alpha=a,
            )
            vals = t_pred.cpu().numpy() * y_std + y_mean
            pdp_means.append(float(np.mean(vals)))
            pdp_q10.append(float(np.percentile(vals, 10)))
            pdp_q90.append(float(np.percentile(vals, 90)))

        pdp_df = pd.DataFrame({
            fname: grid,
            "mean_prediction": pdp_means,
            "q10": pdp_q10,
            "q90": pdp_q90,
        })
        artifact_id = f"v2_neural_pdp::{fname}"
        wrote_to_store = False
        if _store is not None:
            try:
                _store.write_table("2", artifact_id, pdp_df,
                                   producer="v2_neural_training",
                                   consumers=["server:/results/pdp_curves",
                                              "scenario_simulator"],
                                   metadata={"variable": fname,
                                             "source": "neural_pde"})
                wrote_to_store = True
            except Exception as _e:
                logger.warning("pdp store write failed for %s: %s", fname, _e)
        if not wrote_to_store:
            pdp_df.to_csv(pdp_dir / f"pdp_{fname}.csv", index=False)
            try:
                from sparc.registry.run_registry import register_path
                register_path(pdp_dir / f"pdp_{fname}.csv", stage="2",
                              artifact_id=f"pdp::{fname}", format="csv",
                              producer="v2_neural_training",
                              consumers=["server:/results/pdp_curves"],
                              metadata={"variable": fname, "source": "neural_pde"})
            except Exception:
                pass

    logger.info("  Saved PDP curves for %d features → %s", len(feature_names), pdp_dir)

    # --- Export 6: Feature importance (gradient-based) ---
    with torch.enable_grad():
        phys_grad = phys.clone().detach().requires_grad_(True)
        gwr_p, _ = final_surrogates["gwr"](phys_grad, spat)
        gwrf_p = final_surrogates["gwrf"](
            phys_grad, spat,
            neighbor_idx=gwrf_knn, neighbor_dists=gwrf_dists,
        )
        ggp_p = final_surrogates["ggpgam"](phys_grad, spat)

        base_in = torch.stack([gwr_p, gwrf_p, ggp_p], dim=1)
        pr_in = phys_grad[:, pr_col_idxs]
        a_full = final_process(pr_in)
        a = (
            a_full
            if a_full.shape[-1] == 1
            else a_full.mean(dim=-1, keepdim=True)
        )

        # Build extended features with grad-tracked original portion
        n_orig = phys_grad.shape[1]
        phys_ext_grad = phys_ext.clone().detach()
        phys_ext_grad[:, :n_orig] = phys_grad  # grad flows through original cols

        # Need to enable grad for this pass
        final_model.train()  # enable dropout for grad flow
        t_pred, _, _ = final_model(
            base_preds=base_in,
            physics_feats=phys_ext_grad,
            X_spatial=spat,
            coords=coords_t,
            knn_index=knn_idx,
            alpha=a,
        )
        t_pred.sum().backward()
        final_model.eval()

    grad_importance = phys_grad.grad.abs().mean(dim=0).cpu().numpy()
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "gradient_importance": grad_importance,
        "normalised": grad_importance / (grad_importance.sum() + 1e-10),
    }).sort_values("gradient_importance", ascending=False)
    if _store is not None:
        try:
            _store.write_table("2", "v2_neural_feature_importance",
                               importance_df, producer="v2_neural_training")
        except Exception as _e:
            logger.warning("feature_importance store write failed: %s", _e)
            importance_df.to_csv(artifact_dir / "feature_importance.csv", index=False)
    else:
        importance_df.to_csv(artifact_dir / "feature_importance.csv", index=False)
    logger.info("  Saved feature_importance")

    elapsed = _time.perf_counter() - _t0
    logger.info("V2 output export complete (%.1fs)", elapsed)


# ---------------------------------------------------------------------------
# CMA-ES wrapper (optional)
# ---------------------------------------------------------------------------

def run_cma_es_search(
    y: np.ndarray,
    coords: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: dict,
    output_dir: str | Path,
) -> dict[str, float]:
    """
    Run CMA-ES hyperparameter search then retrain with best params.

    Returns the best hyperparameter dict (also saved to output_dir).
    """
    from sparc.training.cma_es import run_cma_es

    output_dir = Path(output_dir)

    def objective(params: dict[str, float]) -> float:
        """Evaluate one hyperparameter configuration via 1-fold quick check."""
        import copy
        trial_cfg = copy.deepcopy(config)
        trial_cfg.setdefault("training", {})
        for k, v in params.items():
            if k.startswith("lambda_"):
                trial_cfg["training"][k] = v
            elif k == "learning_rate":
                trial_cfg["training"]["learning_rate"] = v
            elif k == "dropout":
                trial_cfg.setdefault("models", {}).setdefault("neural", {})["dropout"] = v
            elif k == "clip_norm":
                trial_cfg.setdefault("optimization", {})["clip_norm"] = v
            elif k == "siren_omega":
                trial_cfg.setdefault("models", {}).setdefault("neural", {})["siren_omega"] = v

        # Disable capacity sweep inside CMA-ES trials — it's redundant
        # and adds ~30s per trial.  CMA-ES tunes lambdas, not architecture.
        trial_cfg.setdefault("optimization", {})["capacity_sweep"] = False

        # Quick eval: first fold only, skip retrain/export
        quick_folds = folds[:1]
        result = train_neural_meta(
            y=y,
            coords=coords,
            feature_matrix=feature_matrix,
            feature_names=feature_names,
            folds=quick_folds,
            config=trial_cfg,
            output_dir=output_dir / "cma_trial",
            quick_eval=True,
        )
        return -result["metrics"]["r2"]  # minimise negative R²

    cma_cfg = config.get("optimization", {})
    result = run_cma_es(
        objective,
        max_generations=cma_cfg.get("cma_max_generations", 30),
        population_size=cma_cfg.get("cma_population_size"),
        seed=42,
    )

    # Save
    try:
        from sparc.registry.store import get_active_store
        _store = get_active_store()
    except Exception:
        _store = None
    if _store is not None:
        try:
            _store.write_struct("2", "cma_es_best_params", result.best_params,
                                producer="v2_neural_training")
        except Exception as _e:
            logger.warning("cma_es_best_params struct write failed: %s", _e)
            with open(output_dir / "cma_es_best_params.json", "w") as f:
                json.dump(result.best_params, f, indent=2)
    else:
        with open(output_dir / "cma_es_best_params.json", "w") as f:
            json.dump(result.best_params, f, indent=2)

    logger.info("CMA-ES best R² = %.4f", -result.best_loss)
    return result.best_params
