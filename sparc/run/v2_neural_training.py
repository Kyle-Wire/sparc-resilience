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

def _build_knn_index(coords: np.ndarray, max_neighbors: int) -> np.ndarray:
    """Build KNN index (N, max_neighbors) from projected coords."""
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=max_neighbors + 1)
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
) -> list[torch.Tensor]:
    """
    Run all three differentiable surrogates.

    Returns list of (N,) tensors [gwr_pred, gwrf_pred, ggpgam_pred].
    """
    gwr_pred, _beta = surrogates["gwr"](physics_feats, spatial_feats)
    gwrf_pred = surrogates["gwrf"](physics_feats, spatial_feats)
    ggpgam_pred = surrogates["ggpgam"](physics_feats, spatial_feats)
    return [gwr_pred, gwrf_pred, ggpgam_pred]


def _pretrain_surrogates(
    surrogates: dict[str, torch.nn.Module],
    physics_feats: torch.Tensor,
    spatial_feats: torch.Tensor,
    y: torch.Tensor,
    n_epochs: int = 50,
    lr: float = 1e-3,
) -> None:
    """
    Pre-train each surrogate independently to fit the target with MSE only.

    Gives joint training a warm start rather than starting all
    surrogates from random initialization — analogous to V1 fitting
    GWR/GWRF/GGPGAM separately before the meta-learner.
    """
    import torch.nn.functional as F

    for name, surrogate in surrogates.items():
        surrogate.train()
        optimizer = torch.optim.AdamW(surrogate.parameters(), lr=lr)
        for epoch in range(n_epochs):
            if name == "gwr":
                pred, _ = surrogate(physics_feats, spatial_feats)
            elif name == "gwrf":
                pred = surrogate(physics_feats, spatial_feats)
            else:  # ggpgam
                pred = surrogate(physics_feats, spatial_feats)
            loss = F.mse_loss(pred.squeeze(), y)
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
            ss_res = ((pred - y) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        logger.info(
            "  Surrogate pretrain %s: R²=%.4f  (%d epochs)",
            name, r2.item(), n_epochs,
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

    # KNN index (for spatial attention)
    knn_index = torch.tensor(
        _build_knn_index(coords, max_neighbors),
        dtype=torch.long,
        device=device,
    )

    # Cardinal neighbor index (for physics Laplacian)
    resolution = config.get("training", {}).get("resolution", None)
    cardinal_np, detected_res = _build_cardinal_neighbors(coords, resolution=resolution)
    cardinal_idx = torch.tensor(cardinal_np, dtype=torch.long, device=device)

    # Physics features (the raw feature matrix)
    physics_t = torch.tensor(feature_matrix, dtype=torch.float32, device=device)

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
        "cardinal_idx": cardinal_idx,
        "y": y_t,
        "encoder": encoder,
        "d_spatial": X_spatial.shape[1],
        "resolution": detected_res,
        "y_mean": y_mean,
        "y_std": y_std,
    }


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
) -> dict[str, Any]:
    """
    Train the V2 neural meta-learner with differentiable surrogates,
    process-rate network, physics-informed loss, per-component
    optimizer, 4-stage curriculum, and Stochastic Weight Averaging.

    Surrogates *are* the base models — they learn end-to-end from raw
    features.  There is no V1 base-model output to match; gradients
    from physics / MSE / smoothness flow all the way through the
    surrogates to the input features.

    Returns dict with:
      - ``model``           — trained SPARCMetaLearner
      - ``process_rate``    — trained ProcessRateNet
      - ``surrogates``      — dict of trained DifferentiableGWR/GWRF/GGPGAM
      - ``oof_predictions`` — (N,) OOF predictions
      - ``oof_uncertainty``  — (N,) MC-Dropout std
      - ``metrics`` — dict with R², RMSE
    """
    from sparc.models.neural_meta import SPARCMetaLearner
    from sparc.models.process_rate_net import ProcessRateNet
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
    pretrain_epochs = training_cfg.get("pretrain_epochs", 50)
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

    # Convert exceedance thresholds to normalised space
    thresholds_norm = [(t - y_mean) / y_std for t in thresholds]

    # Process rate input columns
    pr_inputs = pr_cfg.get("inputs", feature_names[:3])
    pr_input_dim = len(pr_inputs)
    pr_col_idxs = [feature_names.index(f) for f in pr_inputs if f in feature_names]
    if not pr_col_idxs:
        pr_col_idxs = list(range(min(pr_input_dim, n_physics)))

    prior_mean = pr_cfg.get("prior_mean", 0.5)

    # Target lambda dict for curriculum schedule
    # NOTE: no "base" / surrogate-fidelity term — surrogates ARE the base
    # models, trained end-to-end via the main MSE + physics gradients.
    target_lambdas = {
        "physics": training_cfg.get("lambda_physics", 0.01),
        "smooth": training_cfg.get("lambda_smooth_pred", 0.01),
        "alpha_smooth": training_cfg.get("lambda_smooth_alpha", 0.001),
        "neighbor": training_cfg.get("lambda_neighbor", 0.1),
        "prior": training_cfg.get("lambda_alpha_prior", 1.0),
    }

    # OOF containers
    oof_preds = np.zeros(len(y), dtype=np.float32)
    oof_std = np.zeros(len(y), dtype=np.float32)

    # ==================================================================
    # CV Loop
    # ==================================================================
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

        # ---- Build fold-local KNN for spatial attention ----
        logger.info("  Building KNN index (k=%d) ...", max_neighbors)
        fold_knn = torch.tensor(
            _build_knn_index(coords[train_idx], max_neighbors),
            dtype=torch.long,
            device=device,
        )
        logger.info("  KNN ready.")

        # ---- Build fold-local cardinal neighbors for physics loss ----
        fold_cardinal_np, _ = _build_cardinal_neighbors(
            coords[train_idx], resolution=resolution,
        )
        fold_cardinal = torch.tensor(fold_cardinal_np, dtype=torch.long, device=device)
        source_term = torch.zeros(len(train_idx), device=device)

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
        optimizer = build_optimizer(model, process_net, surrogates, base_lr=lr)
        scheduler = build_scheduler(optimizer, n_epochs, warmup_epochs=warmup_epochs)

        # ---- Surrogate pre-training (MSE only, no physics/meta) ----
        if pretrain_epochs > 0:
            logger.info(
                "  Pre-training surrogates (%d epochs, MSE only)...",
                pretrain_epochs,
            )
            _pretrain_surrogates(
                surrogates, train_physics, train_spatial, train_y,
                n_epochs=pretrain_epochs, lr=lr,
            )

        # ---- Training loop ----
        model.train()
        process_net.train()
        for s in surrogates.values():
            s.train()

        for epoch in range(n_epochs):
            lambdas = get_lambda_schedule(
                epoch, target_lambdas,
                warmup_end=warmup_epochs,
                ramp_end=ramp_epochs,
            )

            # Forward: surrogates → meta-learner (end-to-end differentiable)
            # Gradients flow: loss → meta-learner → surrogates → features
            surrogate_preds = _forward_surrogates(
                surrogates, train_physics, train_spatial,
            )
            base_input = torch.stack(surrogate_preds, dim=1)  # (N, 3)

            pr_input = train_physics[:, pr_col_idxs]
            alpha = process_net(pr_input)
            alpha_prior = torch.full_like(alpha, prior_mean)

            T_pred, exceedance, _ = model(
                base_preds=base_input,
                physics_feats=train_physics,
                X_spatial=train_spatial,
                coords=train_coords,
                knn_index=fold_knn,
                alpha=alpha,
            )

            # No surrogate fidelity loss — surrogates learn end-to-end
            total_loss, components = sparc_joint_loss(
                T_pred=T_pred,
                exceedance_preds=exceedance,
                y_true=train_y,
                thresholds=thresholds_norm,
                alpha=alpha,
                alpha_prior=alpha_prior,
                neighbor_idx=fold_cardinal,
                source_term=source_term,
                resolution=resolution,
                surrogate_preds=[],
                surrogate_targets=[],
                lambda_physics=lambdas.get("physics", 0.0),
                lambda_smooth=lambdas.get("smooth", 0.0),
                lambda_alpha_smooth=lambdas.get("alpha_smooth", 0.0),
                lambda_prior=lambdas.get("prior", 1.0),
                lambda_base=0.0,
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
            scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "  Epoch %d/%d  loss=%.4f  "
                    "[mse=%.3f phys=%.3f nbr=%.3f ce=%.3f]  (%.1fs)",
                    epoch + 1, n_epochs, total_loss.item(),
                    components.get("mse", 0),
                    components.get("physics", 0),
                    components.get("neighborhood", 0),
                    components.get("cross_entropy", 0),
                    _time.perf_counter() - _fold_t0,
                )

        logger.info(
            "  Fold %d training done in %.1fs",
            fold_idx + 1, _time.perf_counter() - _fold_t0,
        )

        # ---- OOF prediction (surrogates → meta-learner) ----
        model.eval()
        process_net.eval()
        for s in surrogates.values():
            s.eval()

        test_physics = tensors["physics_feats"][test_idx]
        test_spatial = tensors["X_spatial"][test_idx]
        test_coords = tensors["coords"][test_idx]
        test_knn = torch.tensor(
            _build_knn_index(coords[test_idx], max_neighbors),
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            test_surr_preds = _forward_surrogates(
                surrogates, test_physics, test_spatial,
            )
            test_base_input = torch.stack(test_surr_preds, dim=1)
            pr_test = test_physics[:, pr_col_idxs]
            alpha_test = process_net(pr_test)

        mean_pred, std_pred = model.predict_with_uncertainty(
            base_preds=test_base_input,
            physics_feats=test_physics,
            X_spatial=test_spatial,
            coords=test_coords,
            knn_index=test_knn,
            alpha=alpha_test,
            n_samples=50,
        )

        oof_preds[test_idx] = mean_pred.cpu().numpy()
        oof_std[test_idx] = std_pred.cpu().numpy()

    # ---- Denormalise OOF predictions back to original scale ----
    oof_preds = oof_preds * y_std + y_mean
    oof_std = oof_std * y_std  # uncertainty scales linearly

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

    final_optimizer = build_optimizer(
        final_model, final_process, final_surrogates, base_lr=lr,
    )
    final_scheduler = build_scheduler(
        final_optimizer, main_epochs, warmup_epochs=warmup_epochs,
    )

    full_knn = tensors["knn_index"]
    full_cardinal = tensors["cardinal_idx"]
    full_source = torch.zeros(len(y), device=device)

    final_model.train()
    final_process.train()
    for s in final_surrogates.values():
        s.train()

    _retrain_t0 = _time.perf_counter()

    # ---- Surrogate pre-training on full data ----
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
        )

    # ---- Stages A/B/C: Main training ----
    for epoch in range(main_epochs):
        lambdas = get_lambda_schedule(
            epoch, target_lambdas,
            warmup_end=warmup_epochs,
            ramp_end=ramp_epochs,
        )

        surrogate_preds = _forward_surrogates(
            final_surrogates, tensors["physics_feats"], tensors["X_spatial"],
        )
        base_input = torch.stack(surrogate_preds, dim=1)

        pr_input = tensors["physics_feats"][:, pr_col_idxs]
        alpha = final_process(pr_input)
        alpha_prior = torch.full_like(alpha, prior_mean)

        T_pred, exceedance, _ = final_model(
            base_preds=base_input,
            physics_feats=tensors["physics_feats"],
            X_spatial=tensors["X_spatial"],
            coords=tensors["coords"],
            knn_index=full_knn,
            alpha=alpha,
        )

        total_loss, components = sparc_joint_loss(
            T_pred=T_pred,
            exceedance_preds=exceedance,
            y_true=tensors["y"],
            thresholds=thresholds_norm,
            alpha=alpha,
            alpha_prior=alpha_prior,
            neighbor_idx=full_cardinal,
            source_term=full_source,
            resolution=resolution,
            surrogate_preds=[],
            surrogate_targets=[],
            lambda_physics=lambdas.get("physics", 0.0),
            lambda_smooth=lambdas.get("smooth", 0.0),
            lambda_alpha_smooth=lambdas.get("alpha_smooth", 0.0),
            lambda_prior=lambdas.get("prior", 1.0),
            lambda_base=0.0,
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
        final_scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                "  Retrain %d/%d  loss=%.4f  "
                "[mse=%.3f phys=%.3f nbr=%.3f ce=%.3f]  (%.1fs)",
                epoch + 1, main_epochs, total_loss.item(),
                components.get("mse", 0),
                components.get("physics", 0),
                components.get("neighborhood", 0),
                components.get("cross_entropy", 0),
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

            surrogate_preds = _forward_surrogates(
                final_surrogates, tensors["physics_feats"], tensors["X_spatial"],
            )
            base_input = torch.stack(surrogate_preds, dim=1)

            pr_input = tensors["physics_feats"][:, pr_col_idxs]
            alpha = final_process(pr_input)
            alpha_prior = torch.full_like(alpha, prior_mean)

            T_pred, exceedance, _ = final_model(
                base_preds=base_input,
                physics_feats=tensors["physics_feats"],
                X_spatial=tensors["X_spatial"],
                coords=tensors["coords"],
                knn_index=full_knn,
                alpha=alpha,
            )

            total_loss, _ = sparc_joint_loss(
                T_pred=T_pred,
                exceedance_preds=exceedance,
                y_true=tensors["y"],
                thresholds=thresholds_norm,
                alpha=alpha,
                alpha_prior=alpha_prior,
                neighbor_idx=full_cardinal,
                source_term=full_source,
                resolution=resolution,
                surrogate_preds=[],
                surrogate_targets=[],
                lambda_physics=lambdas.get("physics", 0.0),
                lambda_smooth=lambdas.get("smooth", 0.0),
                lambda_alpha_smooth=lambdas.get("alpha_smooth", 0.0),
                lambda_prior=lambdas.get("prior", 1.0),
                lambda_base=0.0,
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

            # Update SWA running averages
            swa_model.update_parameters(final_model)
            swa_process.update_parameters(final_process)
            for k in swa_surrogates_avg:
                swa_surrogates_avg[k].update_parameters(final_surrogates[k])
            swa_scheduler.step()

            if (swa_ep + 1) % 5 == 0 or swa_ep == 0:
                logger.info(
                    "  SWA epoch %d/%d  loss=%.4f  (%.1fs)",
                    swa_ep + 1, swa_epochs, total_loss.item(),
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

    np.savez(
        artifact_dir / "oof_results.npz",
        predictions=oof_preds,
        uncertainty=oof_std,
    )

    meta_info = {
        "surrogate_names": ["gwr", "gwrf", "ggpgam"],
        "n_surrogates": n_base,
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

    return {
        "model": final_model,
        "process_rate": final_process,
        "surrogates": final_surrogates,
        "oof_predictions": oof_preds,
        "oof_uncertainty": oof_std,
        "metrics": {"r2": r2, "rmse": rmse},
        "encoder": tensors["encoder"],
        "meta_info": meta_info,
    }


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
