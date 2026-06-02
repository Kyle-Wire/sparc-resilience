"""
train_multicity_jepa.py -- Multi-city JEPA pretraining + per-city pipeline + LOO eval.

Phases
------
1. JEPA Pretraining  -- batch across all non-holdout city GeoParquets; saves shared trunk
2. Per-City Fine-Tuning -- 3 supervised heads (morning/midday/evening); EWC between cities
3. Leave-One-Out Eval -- zero-shot prediction on Philadelphia (holdout city)
4. Pipeline Stages    -- run stages 0/2/3 per city via RunContext.from_project()

Usage
-----
    python scripts/train_multicity_jepa.py
    python scripts/train_multicity_jepa.py --config configs/multicity_pilot.yml
    python scripts/train_multicity_jepa.py --skip-collect --stages 0,2,3
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_multicity_jepa")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/multicity_pilot.yml")
    p.add_argument("--skip-collect", action="store_true",
                   help="Skip data collection (assume GeoParquets already exist)")
    p.add_argument("--skip-pretrain", action="store_true",
                   help="Skip Phase 1 JEPA pretraining and load existing trunk checkpoint")
    p.add_argument("--stages", default=None,
                   help="Comma-separated pipeline stages to run, e.g. '0,2,3'")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _build_trunk(in_dim: int, hidden_dim: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim,    hidden_dim), nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


def _build_predictor(hidden_dim: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


def _build_head(hidden_dim: int, era5_dim: int = 4):
    """Supervised regression head: trunk_embedding + era5_features → scalar."""
    import torch.nn as nn
    in_dim = hidden_dim + era5_dim
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim // 2), nn.GELU(),
        nn.Linear(hidden_dim // 2, 1),
    )


# ---------------------------------------------------------------------------
# Phase 1 -- JEPA pretraining
# ---------------------------------------------------------------------------

def load_city_features(
    parquet_path: Path,
    feature_cols: list[str],
    coord_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load feature matrix and coords from a city GeoParquet."""
    import geopandas as gpd
    gdf = gpd.read_parquet(str(parquet_path))
    available_features = [c for c in feature_cols if c in gdf.columns]
    X = gdf[available_features].values.astype(np.float32)
    available_coords = [c for c in coord_cols if c in gdf.columns]
    if available_coords:
        C = gdf[available_coords].values.astype(np.float32)
    else:
        C = np.zeros((len(X), 2), dtype=np.float32)
    return X, C


def run_jepa_pretraining(
    all_X: np.ndarray,
    all_C: np.ndarray,
    jepa_cfg: dict,
    output_dir: Path,
) -> object:
    """Run JEPA spatial-patch pretraining on combined city features.

    Returns the trained online trunk (nn.Module).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sparc.training.jepa_loss import JEPALossWeights, jepa_loss, spatial_patch_mask

    hidden_dim = int(jepa_cfg.get("hidden_dim", 256))
    n_epochs   = int(jepa_cfg.get("n_epochs", 50))
    batch_size = int(jepa_cfg.get("batch_size", 2048))
    mask_ratio = float(jepa_cfg.get("mask_ratio", 0.40))
    n_patches  = int(jepa_cfg.get("n_patches", 4))
    ema_tau    = float(jepa_cfg.get("ema_tau", 0.99))
    lr         = float(jepa_cfg.get("lr", 1e-3))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("JEPA pretraining on device=%s  N=%d  D=%d", device, len(all_X), all_X.shape[1])

    # Normalize globally (replace NaN mean/std with safe defaults to avoid warnings)
    X_mean = np.where(np.isnan(np.nanmean(all_X, axis=0)), 0.0, np.nanmean(all_X, axis=0))
    X_std  = np.where(np.isnan(np.nanstd(all_X,  axis=0)), 1.0, np.nanstd(all_X, axis=0)) + 1e-6
    all_X  = np.nan_to_num((all_X - X_mean) / X_std, nan=0.0).astype(np.float32)

    C_mean = np.where(np.isnan(np.nanmean(all_C, axis=0)), 0.0, np.nanmean(all_C, axis=0))
    C_std  = np.where(np.isnan(np.nanstd(all_C,  axis=0)), 1.0, np.nanstd(all_C, axis=0)) + 1e-6
    all_C  = np.nan_to_num((all_C - C_mean) / C_std, nan=0.0).astype(np.float32)

    X_t = torch.tensor(all_X, device=device)
    C_t = torch.tensor(all_C, device=device)

    dataset    = TensorDataset(X_t, C_t)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    in_dim  = all_X.shape[1]
    online  = _build_trunk(in_dim, hidden_dim).to(device)
    target  = _build_trunk(in_dim, hidden_dim).to(device)
    predict = _build_predictor(hidden_dim).to(device)

    # Initialize EMA target = online weights
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.data.copy_(p_o.data)
        p_t.requires_grad_(False)

    opt     = torch.optim.AdamW(list(online.parameters()) + list(predict.parameters()), lr=lr)
    weights = JEPALossWeights()

    print()
    print("  Phase 1 -- JEPA Pretraining")
    print(f"  {'Epoch':>6}  {'Loss':>10}  {'Align':>10}  {'Var':>10}  {'Cov':>10}")
    print("  " + "─" * 50)

    for epoch in range(1, n_epochs + 1):
        epoch_losses = []
        for x_batch, c_batch in loader:
            mask = spatial_patch_mask(c_batch, mask_ratio=mask_ratio, n_patches=n_patches)

            # Context (visible) = ~masked; target (hidden) = masked
            x_context = x_batch.clone()
            x_context[mask] = 0.0  # zero-out masked positions

            h_online = online(x_context)          # (B, D)
            h_pred   = predict(h_online)           # (B, D)

            with torch.no_grad():
                h_target = target(x_batch)         # (B, D) -- full, EMA trunk

            loss, comps = jepa_loss(h_pred, h_target.detach(), weights)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # EMA update of target trunk
            with torch.no_grad():
                for p_t, p_o in zip(target.parameters(), online.parameters()):
                    p_t.data.mul_(ema_tau).add_(p_o.data, alpha=1.0 - ema_tau)

            epoch_losses.append(comps)

        if epoch % 5 == 0 or epoch == 1:
            avg = {k: float(np.mean([c[k] for c in epoch_losses])) for k in epoch_losses[0]}
            print(
                f"  {epoch:>6}  {avg['jepa_total']:>10.4f}"
                f"  {avg['jepa_align']:>10.4f}"
                f"  {avg['jepa_variance']:>10.4f}"
                f"  {avg['jepa_covariance']:>10.4f}"
            )

    print("  " + "─" * 50)

    # Save shared trunk
    trunk_path = output_dir / "jepa_pretrained_trunk.pt"
    torch.save({
        "trunk_state": online.state_dict(),
        "in_dim":      in_dim,
        "hidden_dim":  hidden_dim,
        "X_mean":      X_mean.tolist(),
        "X_std":       X_std.tolist(),
        "C_mean":      C_mean.tolist(),
        "C_std":       C_std.tolist(),
    }, str(trunk_path))
    log.info("Shared trunk saved → %s", trunk_path)

    return online, {"X_mean": X_mean, "X_std": X_std}


# ---------------------------------------------------------------------------
# Phase 2 -- Per-City Fine-Tuning with EWC
# ---------------------------------------------------------------------------

class _CityModel:
    """Container for trunk + 3 supervised heads."""

    def __init__(self, trunk, heads: dict, device, has_labels: bool = True,
                 has_labels_per_window: dict = None):
        self.trunk      = trunk
        self.heads      = heads   # {"morning": nn.Module, "midday": ..., "evening": ...}
        self.device     = device
        self.has_labels = has_labels  # False if city had no CAPA supervision at all
        # Per-window label availability: {"morning": True, "midday": True, "evening": False}
        self.has_labels_per_window = has_labels_per_window or {
            w: has_labels for w in ("morning", "midday", "evening")
        }

    def predict(self, x_land: object, era5_dict: dict) -> dict:
        import torch
        preds = {}
        with torch.no_grad():
            h = self.trunk(x_land)
            for window, head in self.heads.items():
                era5_feats = era5_dict.get(window)
                if era5_feats is not None:
                    inp = torch.cat([h, era5_feats], dim=-1)
                else:
                    inp = torch.cat([h, torch.zeros(h.shape[0], 4, device=self.device)], dim=-1)
                preds[window] = head(inp).squeeze(-1)
        return preds


def run_city_finetune(
    city_slug: str,
    parquet_path: Path,
    trunk,
    feature_cols_land: list[str],
    pilot_cfg: dict,
    fisher_matrices: list,
    optimal_params_list: list,
    output_dir: Path,
    device,
    normalizer: dict = None,
) -> tuple:
    """Fine-tune trunk + 3 heads on one city; update EWC state."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import geopandas as gpd
    from sparc.training.ewc import ewc_penalty, compute_fisher_matrix, extract_trunk_params

    mc           = pilot_cfg["multicity"]
    ewc_lambda   = float(mc["continual_learning"].get("ewc_lambda", 400.0))
    hidden_dim   = int(mc["jepa"].get("hidden_dim", 256))
    n_epochs     = max(50, int(mc["jepa"].get("n_epochs", 50)))
    batch_size   = min(512, int(mc["jepa"].get("batch_size", 2048)))
    lr           = float(mc["jepa"].get("lr", 1e-3))

    gdf = gpd.read_parquet(str(parquet_path))

    # Build feature tensors (apply same normalization used during JEPA pretraining)
    X_land_cols = [c for c in feature_cols_land if c in gdf.columns]
    X_raw = np.nan_to_num(gdf[X_land_cols].values.astype(np.float32), nan=0.0)
    if normalizer is not None:
        X_mean = normalizer["X_mean"][:X_raw.shape[1]]
        X_std  = normalizer["X_std"][:X_raw.shape[1]]
        X_raw  = (X_raw - X_mean) / X_std
    X_land = torch.tensor(X_raw, device=device)

    # ERA5 per-window tensors (apply simple fixed normalization: t2m /40, ws /15, wd /180, rh /100)
    _ERA5_SCALE = np.array([40.0, 15.0, 180.0, 100.0], dtype=np.float32)  # t2m(°C), ws(m/s), wd(°), rh(%)
    _era5_cols = {
        "morning": ["era5_morning_t2m","era5_morning_windspeed","era5_morning_winddir","era5_morning_rh"],
        "midday":  ["era5_midday_t2m", "era5_midday_windspeed", "era5_midday_winddir", "era5_midday_rh"],
        "evening": ["era5_evening_t2m","era5_evening_windspeed","era5_evening_winddir","era5_evening_rh"],
    }
    era5_tensors = {}
    for w, cols in _era5_cols.items():
        avail = [c for c in cols if c in gdf.columns]
        if avail:
            vals = np.nan_to_num(gdf[avail].values.astype(np.float32), nan=0.0)
            if vals.shape[1] < 4:
                pad = np.zeros((len(vals), 4 - vals.shape[1]), dtype=np.float32)
                vals = np.concatenate([vals, pad], axis=1)
            vals = vals / _ERA5_SCALE  # normalize to roughly [-1, 1] range
            era5_tensors[w] = torch.tensor(vals, device=device)
        else:
            era5_tensors[w] = torch.zeros(len(X_land), 4, device=device)

    # Label tensors: predict UHI anomaly above ERA5 t2m (°F) rather than absolute °F.
    # This bounds all targets to ~3-10°F range across all cities, removes inter-city
    # calibration shifts, and forces the head to learn spatial gradients only.
    # ERA5 t2m in era5_tensors has been divided by _ERA5_SCALE[0]=40, so un-scale:
    #   era5_t2m_°C = era5_tensor[:, 0] * 40.0
    #   era5_t2m_°F = era5_t2m_°C * 1.8 + 32.0
    _label_cols = {"morning": "aat_morning", "midday": "aat_midday", "evening": "aat_night"}
    labels = {}
    for w, col in _label_cols.items():
        era5_t2m_F = era5_tensors[w][:, 0] * 40.0 * 1.8 + 32.0  # ERA5 background in °F
        if col in gdf.columns:
            arr = gdf[col].values.astype(np.float32)
            valid = ~np.isnan(arr)
            anomaly = arr - era5_t2m_F.cpu().numpy()  # UHI anomaly: typically +3 to +10 °F
            labels[w] = torch.tensor(np.nan_to_num(anomaly, nan=0.0), device=device)
            labels[f"{w}_valid"] = torch.tensor(valid, device=device)
        else:
            labels[w] = torch.zeros(len(X_land), device=device)
            labels[f"{w}_valid"] = torch.zeros(len(X_land), dtype=torch.bool, device=device)

    # Rebuild heads from scratch; trunk is FROZEN — all heads share the same trunk embeddings
    # so the LOO ensemble can evaluate all heads on the same trunk state.
    trunk_local = trunk
    for p in trunk_local.parameters():
        p.requires_grad_(False)
    trunk_local.eval()

    heads = {
        w: _build_head(hidden_dim, era5_dim=4).to(device)
        for w in ("morning", "midday", "evening")
    }

    head_params = [p for head in heads.values() for p in head.parameters()]
    opt = torch.optim.AdamW(head_params, lr=lr)

    n = len(X_land)
    idx = torch.randperm(n, device=device)
    mse_loss = nn.MSELoss()
    any_labels = any(labels[f"{w}_valid"].sum() > 0 for w in ("morning", "midday", "evening"))
    has_labels_per_window = {w: bool(labels[f"{w}_valid"].sum() > 0) for w in ("morning", "midday", "evening")}

    log.info("[%s] fine-tuning  epochs=%d  N=%d", city_slug, n_epochs, n)

    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        n_batches  = 0
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            x_b = X_land[b]
            era5_b = {w: era5_tensors[w][b] for w in era5_tensors}
            h = trunk_local(x_b)

            sup_loss = torch.tensor(0.0, device=device)
            for w, head in heads.items():
                inp  = torch.cat([h, era5_b[w]], dim=-1)
                pred = head(inp).squeeze(-1)
                valid_mask = labels[f"{w}_valid"][b]
                if valid_mask.sum() > 0:
                    sup_loss = sup_loss + mse_loss(pred[valid_mask], labels[w][b][valid_mask])

            # Trunk is frozen — no EWC needed; supervised loss only
            loss = sup_loss

            if not loss.requires_grad:
                # No CAPA labels and no EWC anchors yet — nothing to train on this batch
                continue

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches  += 1

        if epoch % 5 == 0 or epoch == n_epochs:
            log.info("[%s]  epoch %3d/%d  loss=%.4f", city_slug, epoch, n_epochs, total_loss / max(n_batches, 1))

    # Save city model
    model_path = output_dir / city_slug / "model.pt"
    (output_dir / city_slug).mkdir(parents=True, exist_ok=True)
    torch.save({
        "trunk_state": trunk_local.state_dict(),
        "heads_state": {w: h.state_dict() for w, h in heads.items()},
        "city_slug":   city_slug,
    }, str(model_path))
    log.info("[%s] model saved → %s", city_slug, model_path)

    # Trunk is frozen — no Fisher/EWC needed
    fisher = {}
    theta_star = {}

    return trunk_local, fisher, theta_star, _CityModel(
        trunk_local, heads, device,
        has_labels=any_labels,
        has_labels_per_window=has_labels_per_window,
    )


def _ewc_penalty_sequential(trunk, fisher_matrices, optimal_params_list):
    """EWC penalty for a Sequential trunk (keys are string indices)."""
    import torch
    if not fisher_matrices:
        return torch.tensor(0.0, device=next(trunk.parameters()).device)
    penalty = torch.tensor(0.0, device=next(trunk.parameters()).device)
    current = {name: p for name, p in trunk.named_parameters()}
    for fisher, theta_star in zip(fisher_matrices, optimal_params_list):
        for name, param in current.items():
            if name in fisher and name in theta_star:
                f = fisher[name].to(param.device)
                ts = theta_star[name].to(param.device)
                penalty = penalty + (f * (param - ts) ** 2).sum()
    return penalty


def _compute_fisher_sequential(trunk, X_land, era5_tensors, labels, heads, device, batch_size):
    """Approximate diagonal Fisher for trunk using gradient accumulation."""
    import torch
    import torch.nn as nn
    trunk.eval()
    fisher = {name: torch.zeros_like(p) for name, p in trunk.named_parameters()}
    mse = nn.MSELoss()
    n = len(X_land)
    n_batches = 0
    for start in range(0, n, batch_size):
        b = slice(start, start + batch_size)
        x_b = X_land[b]
        trunk.zero_grad()
        h = trunk(x_b)
        loss = torch.tensor(0.0, device=device)
        for w, head in heads.items():
            era5_b = era5_tensors[w][b]
            inp = torch.cat([h, era5_b], dim=-1)
            pred = head(inp).squeeze(-1)
            valid = labels[f"{w}_valid"][b]
            if valid.sum() > 0:
                loss = loss + mse(pred[valid], labels[w][b][valid])
        if loss.item() > 0:
            loss.backward()
            for name, p in trunk.named_parameters():
                if p.grad is not None:
                    fisher[name] += (p.grad.detach() ** 2) * (b.stop - b.start if b.stop else batch_size)
            n_batches += 1
    if n_batches > 0:
        for name in fisher:
            fisher[name] /= n_batches * batch_size
    trunk.train()
    return fisher


# ---------------------------------------------------------------------------
# Phase 3 -- Leave-One-Out Eval
# ---------------------------------------------------------------------------

def run_loo_eval(
    holdout_path: Path,
    trunk,
    feature_cols_land: list[str],
    pilot_cfg: dict,
    output_dir: Path,
    device,
    city_models: dict = None,
    normalizer: dict = None,
) -> dict:
    """Zero-shot prediction on holdout city using ensemble of trained city heads.

    Uses CAPA-supervised city models to produce an ensemble prediction on the
    holdout city (Philadelphia). For each time window, only includes city-head
    combos where that city had valid CAPA labels for that specific window.
    Falls back to a random head only if no trained models are available.
    """
    import torch
    import geopandas as gpd

    mc         = pilot_cfg["multicity"]
    hidden_dim = int(mc["jepa"].get("hidden_dim", 256))

    log.info("")
    log.info("Phase 3 -- LOO Eval on %s", holdout_path)
    gdf = gpd.read_parquet(str(holdout_path))

    X_land_cols = [c for c in feature_cols_land if c in gdf.columns]
    X_raw = np.nan_to_num(gdf[X_land_cols].values.astype(np.float32), nan=0.0)
    if normalizer is not None:
        X_mean = normalizer["X_mean"][:X_raw.shape[1]]
        X_std  = normalizer["X_std"][:X_raw.shape[1]]
        X_raw  = (X_raw - X_mean) / X_std
    X_land = torch.tensor(X_raw, device=device)

    _ERA5_SCALE = np.array([40.0, 15.0, 180.0, 100.0], dtype=np.float32)
    _era5_cols = {
        "morning": ["era5_morning_t2m","era5_morning_windspeed","era5_morning_winddir","era5_morning_rh"],
        "midday":  ["era5_midday_t2m", "era5_midday_windspeed", "era5_midday_winddir", "era5_midday_rh"],
        "evening": ["era5_evening_t2m","era5_evening_windspeed","era5_evening_winddir","era5_evening_rh"],
    }
    era5_tensors = {}
    for w, cols in _era5_cols.items():
        avail = [c for c in cols if c in gdf.columns]
        if avail:
            vals = np.nan_to_num(gdf[avail].values.astype(np.float32), nan=0.0)
            if vals.shape[1] < 4:
                pad = np.zeros((len(vals), 4 - vals.shape[1]), dtype=np.float32)
                vals = np.concatenate([vals, pad], axis=1)
            vals = vals / _ERA5_SCALE
            era5_tensors[w] = torch.tensor(vals, device=device)
        else:
            era5_tensors[w] = torch.zeros(len(X_land), 4, device=device)

    _label_cols = {"morning": "aat_morning", "midday": "aat_midday", "evening": "aat_night"}

    # Collect CAPA-supervised city models for ensemble
    supervised_models = []
    if city_models:
        supervised_models = [m for m in city_models.values() if m.has_labels]
        log.info("LOO ensemble: %d supervised city models", len(supervised_models))

    trunk.eval()
    results = {}
    pred_cols = {}

    with torch.no_grad():
        h = trunk(X_land)

        if supervised_models:
            # Ensemble: average predictions from city heads that had labels for that specific window
            for w in ("morning", "midday", "evening"):
                window_models = [cm for cm in supervised_models
                                 if cm.has_labels_per_window.get(w, False)]
                if not window_models:
                    log.warning("No supervised models with %s labels — skipping %s window", w, w)
                    continue
                log.info("LOO %s ensemble: %d city heads", w, len(window_models))
                city_preds = []
                for cm in window_models:
                    cm.trunk.eval()
                    era5_feat = era5_tensors[w]
                    inp = torch.cat([h, era5_feat], dim=-1)
                    city_preds.append(cm.heads[w](inp).squeeze(-1))
                # Ensemble average of anomaly predictions
                pred_anomaly = torch.stack(city_preds, dim=0).mean(dim=0)
                # Reconstruct absolute °F: anomaly + ERA5 t2m background
                era5_t2m_F = era5_tensors[w][:, 0] * 40.0 * 1.8 + 32.0
                pred = (pred_anomaly + era5_t2m_F).cpu().numpy()
                pred_cols[f"pred_{w}"] = pred

                label_col = _label_cols[w]
                if label_col in gdf.columns:
                    true_vals = gdf[label_col].values.astype(np.float32)
                    valid = ~np.isnan(true_vals)
                    if valid.sum() > 0:
                        rmse = float(np.sqrt(np.mean((pred[valid] - true_vals[valid]) ** 2)))
                        mae  = float(np.mean(np.abs(pred[valid] - true_vals[valid])))
                        results[w] = {"rmse": rmse, "mae": mae, "n": int(valid.sum())}
        else:
            # Fallback: random heads (baseline only)
            log.warning("No supervised city models available — LOO predictions are from random heads")
            heads = {w: _build_head(hidden_dim, era5_dim=4).to(device) for w in ("morning", "midday", "evening")}
            for w, head in heads.items():
                inp  = torch.cat([h, era5_tensors[w]], dim=-1)
                pred = head(inp).squeeze(-1).cpu().numpy()
                pred_cols[f"pred_{w}"] = pred

    for col, vals in pred_cols.items():
        gdf[col] = vals

    # Save predictions
    pred_path = holdout_path.parent / "predictions.geoparquet"
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf.to_parquet(str(pred_path), index=False)
    log.info("Predictions saved → %s", pred_path)

    print()
    print("  Phase 3 -- Leave-One-Out Eval (Philadelphia, zero-shot)")
    print(f"  {'Window':<12}  {'RMSE':>8}  {'MAE':>8}  {'N':>8}")
    print("  " + "─" * 40)
    for w, metrics in results.items():
        print(f"  {w:<12}  {metrics['rmse']:>8.4f}  {metrics['mae']:>8.4f}  {metrics['n']:>8}")
    print("  " + "─" * 40)

    return results


# ---------------------------------------------------------------------------
# Phase 4 -- Pipeline stages per city
# ---------------------------------------------------------------------------

def run_pipeline_stages(
    city_cfg: dict,
    output_root: Path,
    stages: list[str],
) -> None:
    """Run SPARC pipeline stages for one city using its generated project.yml.

    Called inside the per-city learning loop (Phase 2, step a) BEFORE the
    JEPA supervised fine-tune step so that each city's causal/correlation
    analysis is grounded in the full pipeline before the trunk is updated.
    """
    import os
    from sparc.run.orchestrator import RunContext, PipelineOrchestrator

    city_slug = city_cfg["city_slug"]
    project_yml = output_root / city_slug / "project.yml"
    if not project_yml.exists():
        log.warning("[%s] project.yml not found at %s -- skipping pipeline stages", city_slug, project_yml)
        return

    # Set SPARC_PROJECT so load_config() calls inside each stage module
    # (which construct their own config objects without receiving ctx) pick up
    # this city's project.yml rather than falling back to brown4.csv defaults.
    prev_sparc_project = os.environ.get("SPARC_PROJECT")
    os.environ["SPARC_PROJECT"] = str(project_yml.resolve())
    try:
        ctx = RunContext.from_project(str(project_yml))
        orch = PipelineOrchestrator(ctx)
        for stage in stages:
            log.info("[%s] running pipeline stage %s ...", city_slug, stage)
            orch.run(stage)
            log.info("[%s] stage %s complete", city_slug, stage)
    except Exception as exc:
        log.warning("[%s] pipeline stage error: %s", city_slug, exc)
    finally:
        # Restore previous value so subsequent cities get their own project.yml
        if prev_sparc_project is None:
            os.environ.pop("SPARC_PROJECT", None)
        else:
            os.environ["SPARC_PROJECT"] = prev_sparc_project


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as fh:
        pilot_cfg = yaml.safe_load(fh)

    mc          = pilot_cfg["multicity"]
    output_root = Path(mc["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    # Determine pipeline stages
    if args.stages:
        run_stages = [s.strip() for s in args.stages.split(",")]
    else:
        run_stages = mc.get("pipeline", {}).get("stages", ["0", "2", "3"])

    # Separate holdout and training cities
    all_cities   = mc["pilot_cities"]
    holdout_cfgs = [c for c in all_cities if c.get("holdout", False)]
    train_cfgs   = [c for c in all_cities if not c.get("holdout", False)]

    # Feature columns
    feat_cols_land = mc["feature_columns"].get("land_surface", [])
    feat_cols_all  = (
        feat_cols_land
        + mc["feature_columns"].get("era5_morning", [])
        + mc["feature_columns"].get("era5_midday", [])
        + mc["feature_columns"].get("era5_evening", [])
    )
    coord_cols = mc.get("pipeline", {}).get("coordinate_columns", ["centroid_x", "centroid_y"])
    jepa_cfg   = mc["jepa"]

    # ── Optionally collect data first ──────────────────────────────────────
    if not args.skip_collect:
        log.info("Running data collection via collect_cities.py ...")
        from scripts.collect_cities import collect_one_city
        for city_cfg in all_cities:
            collect_one_city(city_cfg, pilot_cfg, output_root, resume=True)

    # ── Phase 1 -- JEPA Pretraining ─────────────────────────────────────────
    print()
    print("═" * 60)
    print("  Phase 1 -- JEPA Pretraining (all non-holdout cities)")
    print("═" * 60)

    all_X_list, all_C_list = [], []
    for city_cfg in train_cfgs:
        slug = city_cfg["city_slug"]
        pq   = output_root / slug / "data.geoparquet"
        if not pq.exists():
            log.warning("[%s] GeoParquet not found -- skipping from pretraining", slug)
            continue
        try:
            X, C = load_city_features(pq, feat_cols_land, coord_cols)
            all_X_list.append(X)
            all_C_list.append(C)
            log.info("[%s] loaded  X=%s  C=%s", slug, X.shape, C.shape)
        except Exception as exc:
            log.warning("[%s] load error: %s", slug, exc)

    if not all_X_list:
        log.error("No city data available for pretraining. Run collect_cities.py first.")
        sys.exit(1)

    # Align column counts (pad with zeros if different cities have different feature availability)
    max_feats = max(X.shape[1] for X in all_X_list)
    aligned_X = []
    for X in all_X_list:
        if X.shape[1] < max_feats:
            pad = np.zeros((X.shape[0], max_feats - X.shape[1]), dtype=np.float32)
            X = np.concatenate([X, pad], axis=1)
        aligned_X.append(X)

    combined_X = np.concatenate(aligned_X,  axis=0)
    combined_C = np.concatenate(all_C_list, axis=0)

    if args.skip_pretrain:
        # Load existing trunk checkpoint and reconstruct normalizer
        trunk_path = output_root / "jepa_pretrained_trunk.pt"
        if not trunk_path.exists():
            log.error("--skip-pretrain requested but no trunk checkpoint found at %s", trunk_path)
            sys.exit(1)
        import torch as _t
        _dev = _t.device("cuda" if _t.cuda.is_available() else "cpu")
        ckpt = _t.load(str(trunk_path), map_location=_dev)
        hidden_dim = ckpt.get("hidden_dim", int(jepa_cfg.get("hidden_dim", 256)))
        in_dim     = ckpt.get("in_dim", combined_X.shape[1])
        trunk = _build_trunk(in_dim, hidden_dim).to(_dev)
        trunk.load_state_dict(ckpt["trunk_state"])
        normalizer = {
            "X_mean": np.array(ckpt["X_mean"], dtype=np.float32),
            "X_std":  np.array(ckpt["X_std"],  dtype=np.float32),
        }
        log.info("Loaded pretrained trunk from %s (skipping Phase 1)", trunk_path)
    else:
        trunk, normalizer = run_jepa_pretraining(combined_X, combined_C, jepa_cfg, output_root)

    # ── Phase 2 -- Per-City Head Training (frozen trunk) ────────────────────
    # Trunk is FROZEN from Phase 1 JEPA pretraining.
    # Each city trains 3 supervised heads (morning/midday/evening) on top of
    # the fixed trunk embeddings. All heads share the same trunk state so the
    # LOO ensemble evaluates correctly at inference time.
    print()
    print("═" * 60)
    print("  Phase 2 -- Per-City Head Training (frozen trunk)")
    print("═" * 60)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.skip_pretrain:
        # Move the loaded trunk to the correct device (already set in skip block above)
        trunk = trunk.to(device)

    fisher_matrices:     list[dict] = []
    optimal_params_list: list[dict] = []
    city_models: dict = {}

    for city_idx, city_cfg in enumerate(train_cfgs):
        slug = city_cfg["city_slug"]
        pq   = output_root / slug / "data.geoparquet"

        print()
        print(f"  ── City {city_idx + 1}/{len(train_cfgs)}: {slug} ──────────────────────────")

        log.info("[%s] running pipeline stages %s ...", slug, run_stages)
        try:
            run_pipeline_stages(city_cfg, output_root, run_stages)
        except Exception as exc:
            log.warning("[%s] pipeline stage error: %s -- continuing with head training", slug, exc)

        if not pq.exists():
            log.warning("[%s] GeoParquet missing -- skipping supervised head training", slug)
            continue
        try:
            trunk, fisher, theta_star, city_model = run_city_finetune(
                city_slug=slug,
                parquet_path=pq,
                trunk=trunk,
                feature_cols_land=feat_cols_land,
                pilot_cfg=pilot_cfg,
                fisher_matrices=fisher_matrices,
                optimal_params_list=optimal_params_list,
                output_dir=output_root,
                device=device,
                normalizer=normalizer,
            )
            city_models[slug] = city_model
            log.info("[%s] head trained (%d cities done)", slug, len(city_models))
        except Exception as exc:
            log.warning("[%s] supervised head training error: %s", slug, exc)

    # ── Phase 3 -- LOO Eval on Philadelphia (zero-shot) ─────────────────────
    # Only runs AFTER all training cities have been processed.
    # The model has never seen Philadelphia -- this is the true zero-shot test.
    print()
    print("═" * 60)
    print("  Phase 3 -- Zero-Shot Leave-One-Out Eval (Philadelphia)")
    print("  (runs after all training cities processed -- never seen during training)")
    print("═" * 60)

    for holdout_cfg in holdout_cfgs:
        slug = holdout_cfg["city_slug"]
        pq   = output_root / slug / "data.geoparquet"
        if pq.exists():
            try:
                loo_results = run_loo_eval(
                    holdout_path=pq,
                    trunk=trunk,
                    feature_cols_land=feat_cols_land,
                    pilot_cfg=pilot_cfg,
                    output_dir=output_root,
                    device=device,
                    city_models=city_models,
                    normalizer=normalizer,
                )
            except Exception as exc:
                log.warning("LOO eval error for %s: %s", slug, exc)
        else:
            log.warning("Holdout city GeoParquet not found: %s -- skipping LOO eval", pq)

    print()
    print("All phases complete.")
    print(f"  Trunk checkpoint  : output/jepa_pretrained_trunk.pt")
    print(f"  City models       : output/cities/<slug>/model.pt")
    print(f"  LOO predictions   : output/cities/philadelphia_pa/predictions.geoparquet")
    print(f"  Cities trained    : {[c['city_slug'] for c in train_cfgs]}")
    print(f"  Cities evaluated  : {[c['city_slug'] for c in holdout_cfgs]}")
    print()


if __name__ == "__main__":
    main()
