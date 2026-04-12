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
    # Build global→local map
    local_map = -np.ones(global_idx.max() + 1 if len(global_idx) else 0,
                         dtype=np.int64)
    for local_i, global_i in enumerate(batch_idx):
        if global_i < len(local_map):
            local_map[global_i] = local_i

    nb = neighbor_tensor.cpu().numpy()
    remapped = np.full_like(nb, -1)
    valid = (nb >= 0) & (nb < len(local_map))
    remapped[valid] = local_map[nb[valid]]
    return torch.tensor(remapped, dtype=torch.long, device=neighbor_tensor.device)


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

    n_complete = int((neighbor_idx != -1).all(axis=1).sum())
    logger.info(
        "Cardinal neighbors: %d/%d complete (res=%.2f)", n_complete, N, resolution,
    )
    return neighbor_idx, resolution


def _forward_surrogates(
    surrogates: dict[str, torch.nn.Module],
    physics_feats: torch.Tensor,
    spatial_feats: torch.Tensor,
    knn_index: torch.Tensor | None = None,
    knn_dists: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """
    Run all three differentiable surrogates.

    Parameters
    ----------
    knn_index : (N, K) int tensor, optional
        Passed to GWRF for distance-weighted kernel predictions.
    knn_dists : (N, K) float tensor, optional
        Distances to K nearest neighbors for GWRF kernel.

    Returns list of (N,) tensors [gwr_pred, gwrf_pred, ggpgam_pred].
    """
    gwr_pred, _beta = surrogates["gwr"](physics_feats, spatial_feats)
    gwrf_pred = surrogates["gwrf"](
        physics_feats, spatial_feats,
        neighbor_idx=knn_index, neighbor_dists=knn_dists,
    )
    ggpgam_pred = surrogates["ggpgam"](physics_feats, spatial_feats)
    return [gwr_pred, gwrf_pred, ggpgam_pred]


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

    # Target — z-score normalise so surrogates' near-zero init is correct
    y_mean = float(np.mean(y))
    y_std = float(np.std(y)) or 1.0  # guard against constant target
    y_norm = (y - y_mean) / y_std
    y_t = torch.tensor(y_norm, dtype=torch.float32, device=device)

    return {
        "physics_feats": physics_t,
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

    # Stage_0 is a sibling of Stage_2 under the shared output root
    stage0_dir = output_dir.parent / "Stage_0_Correlogram"
    results_path = stage0_dir / "correlogram_analysis_results.json"

    if not results_path.exists():
        logger.info("Correlogram results not found at %s — using uniform bandwidths", results_path)
        return None

    try:
        with open(results_path, "r") as f:
            data = json.load(f)

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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Config ----
    neural_cfg = config.get("models", {}).get("neural", {})
    training_cfg = config.get("training", {})
    optim_cfg = config.get("optimization", {})
    pr_cfg = config.get("process_rate", {})

    hidden_dim = neural_cfg.get("hidden_dim", 256)
    dropout = neural_cfg.get("dropout", 0.1)
    n_heads = neural_cfg.get("n_heads", 4)
    max_neighbors = neural_cfg.get("max_neighbors", 128)
    siren_omega = neural_cfg.get("siren_omega", 30.0)
    thresholds = neural_cfg.get("exceedance_thresholds", [0.25, 0.50, 0.75])

    n_epochs = training_cfg.get("n_epochs", 100)
    swa_epochs = training_cfg.get("swa_epochs", max(int(n_epochs * 0.2), 5))
    pretrain_epochs = training_cfg.get("pretrain_epochs", 200)
    main_epochs = n_epochs  # SWA runs *on top* of main epochs for full retrain
    lr = training_cfg.get("learning_rate", 1e-3)
    clip_norm = optim_cfg.get("clip_norm", 1.0)
    warmup_epochs = training_cfg.get("warmup_epochs", 10)
    ramp_epochs = training_cfg.get("ramp_epochs", 30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("V2 neural training on device: %s", device)

    # ---- Prepare data ----
    tensors = _prepare_tensors(
        y, coords, feature_matrix, config, device,
    )

    n_base = 3  # always 3 surrogates: GWR, GWRF, GGPGAM
    n_physics = feature_matrix.shape[1]
    d_spatial = tensors["d_spatial"]
    resolution = tensors["resolution"]
    y_mean = tensors["y_mean"]
    y_std = tensors["y_std"]

    # ---- Load per-predictor bandwidths from Stage 0 correlogram ----
    predictor_bandwidths = _load_correlogram_bandwidths(output_dir, feature_names)

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

    # Target lambda dict for curriculum schedule.
    # "base" = surrogate fidelity: soft alignment with pretrained targets.
    target_lambdas = {
        "physics": training_cfg.get("lambda_physics", 0.01),
        "smooth": training_cfg.get("lambda_smooth_pred", 0.01),
        "alpha_smooth": training_cfg.get("lambda_smooth_alpha", 0.001),
        "neighbor": training_cfg.get("lambda_neighbor", 0.1),
        "prior": training_cfg.get("lambda_alpha_prior", 1.0),
        "base": training_cfg.get("lambda_base", 0.2),
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
        sweep_spat = tensors["X_spatial"]
        sweep_coords = tensors["coords"]
        sweep_y = tensors["y"]
        sweep_epochs = max(pretrain_epochs + 20, 30)

        def _sweep_factory(hidden_dim: int, **_kw):
            """Create model bundle for capacity sweep."""
            _m = SPARCMetaLearner(
                n_base_models=n_base, n_physics_features=n_physics,
                d_spatial=d_spatial, hidden_dim=hidden_dim, dropout=dropout,
                thresholds=thresholds, n_heads=n_heads,
                max_neighbors=max_neighbors, siren_omega=siren_omega,
            ).to(device)
            _s = {
                "gwr": DifferentiableGWR(
                    n_physics, d_spatial, hidden_dim,
                    bandwidths=predictor_bandwidths,
                ).to(device),
                "gwrf": DifferentiableGWRF(n_physics, d_spatial, hidden_dim).to(device),
                "ggpgam": DifferentiableGGPGAM(
                    n_physics, d_spatial, min(hidden_dim, 32),
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
            _knn = torch.tensor(
                _build_knn_index(coords[sweep_train_idx], max_neighbors),
                dtype=torch.long, device=device,
            )
            _card_np, _ = _build_cardinal_neighbors(
                coords[sweep_train_idx], resolution=resolution,
            )
            _card = torch.tensor(_card_np, dtype=torch.long, device=device)
            _src = torch.zeros(len(sweep_train_idx), device=device)
            for _ in range(20):
                sp = _forward_surrogates(
                    _s, sweep_phys[sweep_train_idx], sweep_spat[sweep_train_idx],
                )
                bi = torch.stack(sp, dim=1)
                pr_in = sweep_phys[sweep_train_idx][:, pr_col_idxs]
                alpha = _p(pr_in)
                T_pred, exc, _ = _m(
                    base_preds=bi,
                    physics_feats=sweep_phys[sweep_train_idx],
                    X_spatial=sweep_spat[sweep_train_idx],
                    coords=sweep_coords[sweep_train_idx],
                    knn_index=_knn, alpha=alpha,
                )
                loss, _ = sparc_joint_loss(
                    T_pred=T_pred, exceedance_preds=exc,
                    y_true=sweep_y[sweep_train_idx],
                    thresholds=thresholds_norm, alpha=alpha,
                    alpha_prior=torch.full_like(alpha, prior_mean),
                    neighbor_idx=_card, source_term=_src,
                    resolution=resolution,
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
                # FIX A5: build KNN from all sweep coords, not test-only
                _knn_full = torch.tensor(
                    _build_knn_index(
                        coords[np.concatenate([sweep_train_idx, sweep_test_idx])],
                        max_neighbors,
                    ),
                    dtype=torch.long, device=device,
                )
                _n_train = len(sweep_train_idx)
                _n_test = len(sweep_test_idx)
                _all_phys = torch.cat([sweep_phys[sweep_train_idx], sweep_phys[sweep_test_idx]])
                _all_spat = torch.cat([sweep_spat[sweep_train_idx], sweep_spat[sweep_test_idx]])
                _all_coords = torch.cat([sweep_coords[sweep_train_idx], sweep_coords[sweep_test_idx]])
                sp = _forward_surrogates(_s, _all_phys, _all_spat)
                bi = torch.stack(sp, dim=1)
                pr_in = _all_phys[:, pr_col_idxs]
                alpha = _p(pr_in)
                T_pred, _, _ = _m(
                    base_preds=bi,
                    physics_feats=_all_phys,
                    X_spatial=_all_spat,
                    coords=_all_coords,
                    knn_index=_knn_full, alpha=alpha,
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

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        _fold_t0 = _time.perf_counter()
        logger.info(
            "Fold %d / %d  (%d train, %d test)",
            fold_idx + 1, len(folds), len(train_idx), len(test_idx),
        )

        # ---- Slice training data ----
        train_physics = tensors["physics_feats"][train_idx]
        train_spatial = tensors["X_spatial"][train_idx]
        train_coords = tensors["coords"][train_idx]
        train_y = tensors["y"][train_idx]

        # ---- Build fold-local KNN for spatial attention + GWRF kernel ----
        logger.info("  Building KNN index (k=%d) ...", max_neighbors)
        fold_knn_np, fold_knn_dists_np = _build_knn_index(
            coords[train_idx], max_neighbors, return_dists=True,
        )
        fold_knn = torch.tensor(fold_knn_np, dtype=torch.long, device=device)
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
        ).to(device)

        # FIX A1: Learned source term instead of zeros
        source_net = SourceTermNet(n_inputs=n_physics).to(device)

        model = SPARCMetaLearner(
            n_base_models=n_base,
            n_physics_features=n_physics,
            d_spatial=d_spatial,
            hidden_dim=hidden_dim,
            dropout=dropout,
            thresholds=thresholds,
            n_heads=n_heads,
            max_neighbors=max_neighbors,
            siren_omega=siren_omega,
        ).to(device)

        # ---- Differentiable surrogates ----
        surrogates = {
            "gwr": DifferentiableGWR(
                n_vars=n_physics, n_spatial_features=d_spatial,
                hidden_dim=hidden_dim,
                bandwidths=predictor_bandwidths,
            ).to(device),
            "gwrf": DifferentiableGWRF(
                n_vars=n_physics, n_spatial_features=d_spatial,
                hidden_dim=hidden_dim,
            ).to(device),
            "ggpgam": DifferentiableGGPGAM(
                n_vars=n_physics, n_spatial_features=d_spatial,
                hidden_dim=min(hidden_dim, 32),
            ).to(device),
        }

        # ---- Per-component optimizer + warmup scheduler ----
        optimizer = build_optimizer(
            model, process_net, surrogates, base_lr=lr,
            source_term_net=source_net,
        )
        scheduler = build_scheduler(optimizer, n_epochs, warmup_epochs=warmup_epochs)

        # ---- Build per-fold V1 base-model targets (normalised) ----
        _fold_base_targets: dict[str, torch.Tensor] | None = None
        if base_oof_predictions is not None:
            _fold_base_targets = {}
            for sname in ("gwr", "gwrf", "ggpgam"):
                if sname in base_oof_predictions:
                    _raw = base_oof_predictions[sname][train_idx]
                    _norm = (_raw - y_mean) / y_std      # same normalisation as y
                    _fold_base_targets[sname] = torch.tensor(
                        _norm, dtype=torch.float32, device=device,
                    )
            if not _fold_base_targets:
                _fold_base_targets = None

        # ---- Surrogate pre-training (MSE only, no physics/meta) ----
        if pretrain_epochs > 0:
            logger.info(
                "  Pre-training surrogates (%d epochs, MSE only)...",
                pretrain_epochs,
            )
            _pretrain_surrogates(
                surrogates, train_physics, train_spatial, train_y,
                n_epochs=pretrain_epochs, lr=lr,
                base_targets=_fold_base_targets,
            )

        # ---- Surrogate fidelity targets ----
        # When V1 base-model OOF predictions are available, use those
        # directly as the fidelity anchor for Term 7 so surrogates
        # stay close to the V1 models.  Otherwise fall back to
        # caching the surrogate's own pretrained output.
        with torch.no_grad():
            _surr_targets = {}
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
                # Log fidelity target quality vs y
                ss_res = ((_surr_targets[sname] - train_y) ** 2).sum()
                ss_tot = ((train_y - train_y.mean()) ** 2).sum()
                r2 = (1.0 - ss_res / (ss_tot + 1e-12)).item()
                logger.info(
                    "  Surrogate %s fidelity R²=%.4f %s", sname, r2, r2_vs,
                )

        # ---- Validate surrogates before joint training ----
        # When base-model targets are available, gate on R² > threshold
        # to prevent garbage surrogates from corrupting the meta-learner.
        if _fold_base_targets:
            from sparc.models.surrogates import validate_surrogates
            validation_results = validate_surrogates(
                surrogates=surrogates,
                true_predictions=_fold_base_targets,
                X=train_physics,
                spatial_features=train_spatial,
                threshold=0.85,
            )
            failed = [name for name, res in validation_results.items()
                       if not res['passed']]
            if failed:
                raise RuntimeError(
                    f"Surrogate validation failed for: {failed}. "
                    f"Joint training aborted — fix surrogate architecture "
                    f"before proceeding."
                )
            logger.info("  All surrogates passed validation (R² >= 0.85)")

        # ---- Training loop ----
        model.train()
        process_net.train()
        for s in surrogates.values():
            s.train()

        N_train = len(train_idx)
        use_minibatch = N_train > batch_size * 2

        if use_minibatch:
            from sparc.training.optimizer import spatial_minibatch_sampler
            fold_cardinal_np, _ = _build_cardinal_neighbors(
                coords[train_idx], resolution=resolution,
            )
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

            if use_minibatch:
                # Spatial minibatch: iterate over contiguous batches
                batches = list(spatial_minibatch_sampler(
                    coords[train_idx],
                    fold_cardinal.cpu().numpy(),
                    batch_size=batch_size,
                    n_batches=max(1, N_train // batch_size),
                ))
            else:
                batches = [np.arange(N_train)]

            epoch_loss = 0.0
            epoch_components: dict = {}
            n_batch_points = 0

            for b_idx in batches:
                b_physics = train_physics[b_idx]
                b_spatial = train_spatial[b_idx]
                b_coords = train_coords[b_idx]
                b_y = train_y[b_idx]
                b_surr_targets = {
                    k: v[b_idx] for k, v in _surr_targets.items()
                }

                # Remap KNN and cardinal indices to batch-local space
                b_knn = _remap_indices_to_local(
                    np.arange(N_train), b_idx, fold_knn[b_idx],
                )
                b_cardinal = _remap_indices_to_local(
                    np.arange(N_train), b_idx, fold_cardinal[b_idx],
                )

                # GWRF kernel: use first gwrf_k neighbors + remapped distances
                b_gwrf_knn = _remap_indices_to_local(
                    np.arange(N_train), b_idx,
                    fold_knn[b_idx][:, :gwrf_k],
                )
                b_gwrf_dists = fold_knn_dists[b_idx][:, :gwrf_k]

                # Forward: surrogates → meta-learner
                surrogate_preds = _forward_surrogates(
                    surrogates, b_physics, b_spatial,
                    knn_index=b_gwrf_knn, knn_dists=b_gwrf_dists,
                )
                base_input = torch.stack(surrogate_preds, dim=1)

                pr_input = b_physics[:, pr_col_idxs]
                alpha = process_net(pr_input)
                alpha_prior = torch.full_like(alpha, prior_mean)

                # FIX A1: learned source term from physics features
                b_source = source_net(b_physics).squeeze(-1)

                T_pred, exceedance, _ = model(
                    base_preds=base_input,
                    physics_feats=b_physics,
                    X_spatial=b_spatial,
                    coords=b_coords,
                    knn_index=b_knn,
                    alpha=alpha,
                )

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
                )

                optimizer.zero_grad()
                total_loss.backward()

                # Global gradient norm clipping across all components
                all_params = (
                    list(model.parameters()) + list(process_net.parameters())
                )
                for s in surrogates.values():
                    all_params.extend(s.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm)

                optimizer.step()

                bsize = len(b_idx)
                epoch_loss += total_loss.item() * bsize
                n_batch_points += bsize
                for k, v in components.items():
                    epoch_components[k] = epoch_components.get(k, 0) + v * bsize

            scheduler.step()

            # Normalise accumulated loss
            epoch_loss /= max(n_batch_points, 1)
            for k in epoch_components:
                epoch_components[k] /= max(n_batch_points, 1)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "  Epoch %d/%d  loss=%.4f  "
                    "[mse=%.3f phys=%.3f nbr=%.3f ce=%.3f]  (%.1fs)",
                    epoch + 1, n_epochs, epoch_loss,
                    epoch_components.get("mse", 0),
                    epoch_components.get("physics", 0),
                    epoch_components.get("neighborhood", 0),
                    epoch_components.get("cross_entropy", 0),
                    _time.perf_counter() - _fold_t0,
                )

        logger.info(
            "  Fold %d training done in %.1fs",
            fold_idx + 1, _time.perf_counter() - _fold_t0,
        )

        # ---- OOF prediction (surrogates → meta-learner) ----
        # FIX A5: Build KNN from ALL coords so test points find true spatial
        # neighbours (including training points), not just other test points.
        model.eval()
        process_net.eval()
        for s in surrogates.values():
            s.eval()

        full_knn_np, full_knn_dists_np = _build_knn_index(
            coords, max_neighbors, return_dists=True,
        )
        full_knn = torch.tensor(full_knn_np, dtype=torch.long, device=device)
        full_knn_dists = torch.tensor(full_knn_dists_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            full_surr_preds = _forward_surrogates(
                surrogates, tensors["physics_feats"], tensors["X_spatial"],
                knn_index=full_knn[:, :gwrf_k],
                knn_dists=full_knn_dists[:, :gwrf_k],
            )
            full_base_input = torch.stack(full_surr_preds, dim=1)
            full_pr = tensors["physics_feats"][:, pr_col_idxs]
            full_alpha = process_net(full_pr)

        mean_pred, std_pred = model.predict_with_uncertainty(
            base_preds=full_base_input,
            physics_feats=tensors["physics_feats"],
            X_spatial=tensors["X_spatial"],
            coords=tensors["coords"],
            knn_index=full_knn,
            alpha=full_alpha,
            n_samples=50,
        )

        oof_preds[test_idx] = mean_pred[test_idx].cpu().numpy()
        oof_std[test_idx] = std_pred[test_idx].cpu().numpy()

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
    ).to(device)

    final_model = SPARCMetaLearner(
        n_base_models=n_base,
        n_physics_features=n_physics,
        d_spatial=d_spatial,
        hidden_dim=hidden_dim,
        dropout=dropout,
        thresholds=thresholds,
        n_heads=n_heads,
        max_neighbors=max_neighbors,
        siren_omega=siren_omega,
    ).to(device)

    final_surrogates = {
        "gwr": DifferentiableGWR(
            n_vars=n_physics, n_spatial_features=d_spatial,
            hidden_dim=hidden_dim,
            bandwidths=predictor_bandwidths,
        ).to(device),
        "gwrf": DifferentiableGWRF(
            n_vars=n_physics, n_spatial_features=d_spatial,
            hidden_dim=hidden_dim,
        ).to(device),
        "ggpgam": DifferentiableGGPGAM(
            n_vars=n_physics, n_spatial_features=d_spatial,
            hidden_dim=min(hidden_dim, 32),
        ).to(device),
    }

    # FIX A1: Learned source term for full retrain
    final_source_net = SourceTermNet(n_inputs=n_physics).to(device)

    final_optimizer = build_optimizer(
        final_model, final_process, final_surrogates, base_lr=lr,
        source_term_net=final_source_net,
    )
    final_scheduler = build_scheduler(
        final_optimizer, main_epochs, warmup_epochs=warmup_epochs,
    )

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

        for b_idx in rt_batches:
            b_phys = tensors["physics_feats"][b_idx]
            b_spat = tensors["X_spatial"][b_idx]
            b_coord = tensors["coords"][b_idx]
            b_y = tensors["y"][b_idx]
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

            surrogate_preds = _forward_surrogates(
                final_surrogates, b_phys, b_spat,
                knn_index=b_gwrf_knn_rt, knn_dists=b_gwrf_dists_rt,
            )
            base_input = torch.stack(surrogate_preds, dim=1)

            pr_input = b_phys[:, pr_col_idxs]
            alpha = final_process(pr_input)
            alpha_prior = torch.full_like(alpha, prior_mean)

            # FIX A1: learned source term
            b_src = final_source_net(b_phys).squeeze(-1)

            T_pred, exceedance, _ = final_model(
                base_preds=base_input,
                physics_feats=b_phys,
                X_spatial=b_spat,
                coords=b_coord,
                knn_index=b_knn,
                alpha=alpha,
            )

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
            )

            final_optimizer.zero_grad()
            total_loss.backward()
            all_p = list(final_model.parameters()) + list(final_process.parameters())
            for s in final_surrogates.values():
                all_p.extend(s.parameters())
            torch.nn.utils.clip_grad_norm_(all_p, clip_norm)
            final_optimizer.step()

            bsz = len(b_idx)
            rt_epoch_loss += total_loss.item() * bsz
            rt_epoch_n += bsz

        final_scheduler.step()
        rt_epoch_loss /= max(rt_epoch_n, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                "  Retrain %d/%d  loss=%.4f  (%.1fs)",
                epoch + 1, main_epochs, rt_epoch_loss,
                _time.perf_counter() - _retrain_t0,
            )

    # ---- Stage D: Stochastic Weight Averaging ----
    if swa_epochs > 0:
        from torch.optim.swa_utils import AveragedModel, SWALR

        logger.info("Starting SWA phase (%d epochs)...", swa_epochs)

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
                b_phys = tensors["physics_feats"][b_idx]
                b_spat = tensors["X_spatial"][b_idx]
                b_coord = tensors["coords"][b_idx]
                b_y = tensors["y"][b_idx]
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

                surrogate_preds = _forward_surrogates(
                    final_surrogates, b_phys, b_spat,
                    knn_index=b_gwrf_knn_swa, knn_dists=b_gwrf_dists_swa,
                )
                base_input = torch.stack(surrogate_preds, dim=1)

                pr_input = b_phys[:, pr_col_idxs]
                alpha = final_process(pr_input)
                alpha_prior = torch.full_like(alpha, prior_mean)

                # FIX A1: learned source term
                b_src = final_source_net(b_phys).squeeze(-1)

                T_pred, exceedance, _ = final_model(
                    base_preds=base_input,
                    physics_feats=b_phys,
                    X_spatial=b_spat,
                    coords=b_coord,
                    knn_index=b_knn,
                    alpha=alpha,
                )

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
                )

                final_optimizer.zero_grad()
                total_loss.backward()
                all_p = list(final_model.parameters()) + list(final_process.parameters())
                for s in final_surrogates.values():
                    all_p.extend(s.parameters())
                torch.nn.utils.clip_grad_norm_(all_p, clip_norm)
                final_optimizer.step()

                bsz = len(b_idx)
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

    torch.save(final_model.state_dict(), artifact_dir / "neural_meta.pt")
    torch.save(final_process.state_dict(), artifact_dir / "process_rate_net.pt")
    for name, surr in final_surrogates.items():
        torch.save(surr.state_dict(), artifact_dir / f"surrogate_{name}.pt")
    joblib.dump(tensors["encoder"], artifact_dir / "sinusoidal_encoder.pkl")

    # Feature scaling stats (needed to standardize inputs at inference time)
    np.savez(
        artifact_dir / "feature_scaling.npz",
        feat_mean=tensors["feat_mean"],
        feat_std=tensors["feat_std"],
    )

    np.savez(
        artifact_dir / "oof_results.npz",
        predictions=oof_preds,
        uncertainty=oof_std,
    )

    meta_info = {
        "surrogate_names": ["gwr", "gwrf", "ggpgam"],
        "n_surrogates": n_base,
        "n_base_models": n_base,
        "n_physics_features": n_physics,
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
    with open(artifact_dir / "meta_info.json", "w") as f:
        json.dump(meta_info, f, indent=2)

    logger.info("V2 neural artifacts saved to %s", artifact_dir)

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
    _t0 = _time.perf_counter()

    final_model.eval()
    final_process.eval()
    for s in final_surrogates.values():
        s.eval()

    N = len(y)
    phys = tensors["physics_feats"]
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
    alpha = final_process(pr_input)  # (N, 1)

    # --- 3. Full meta-learner inference ---
    T_pred_norm, exceedance_list, attn_weights = final_model(
        base_preds=base_input,
        physics_feats=phys,
        X_spatial=spat,
        coords=coords_t,
        knn_index=knn_idx,
        alpha=alpha,
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

    pred_df.to_csv(artifact_dir / "predictions.csv", index=False)
    logger.info("  Saved predictions.csv (%d rows, %d cols)", len(pred_df), len(pred_df.columns))

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

    coeff_df.to_csv(artifact_dir / "local_coefficients_gwr.csv", index=False)
    logger.info("  Saved local_coefficients_gwr.csv (%d rows)", len(coeff_df))

    # --- Export 3: Process rate map ---
    alpha_df = pd.DataFrame({
        "lon": coords[:, 0],
        "lat": coords[:, 1],
        "alpha": alpha.squeeze(-1).cpu().numpy(),
    })
    alpha_df.to_csv(artifact_dir / "process_rate_map.csv", index=False)
    logger.info("  Saved process_rate_map.csv")

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
    attn_summary.to_csv(artifact_dir / "spatial_attention_summary.csv", index=False)
    logger.info("  Saved spatial_attention_summary.csv")

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
            a = final_process(pr_in)

            t_pred, _, _ = final_model(
                base_preds=base_in,
                physics_feats=phys_mod,
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
        pdp_df.to_csv(pdp_dir / f"pdp_{fname}.csv", index=False)

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
        a = final_process(pr_in)

        # Need to enable grad for this pass
        final_model.train()  # enable dropout for grad flow
        t_pred, _, _ = final_model(
            base_preds=base_in,
            physics_feats=phys_grad,
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
    importance_df.to_csv(artifact_dir / "feature_importance.csv", index=False)
    logger.info("  Saved feature_importance.csv")

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
        trial_cfg = dict(config)
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

        # Quick eval: first fold only
        quick_folds = folds[:1]
        result = train_neural_meta(
            y=y,
            coords=coords,
            feature_matrix=feature_matrix,
            feature_names=feature_names,
            folds=quick_folds,
            config=trial_cfg,
            output_dir=output_dir / "cma_trial",
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
    with open(output_dir / "cma_es_best_params.json", "w") as f:
        json.dump(result.best_params, f, indent=2)

    logger.info("CMA-ES best R² = %.4f", -result.best_loss)
    return result.best_params
