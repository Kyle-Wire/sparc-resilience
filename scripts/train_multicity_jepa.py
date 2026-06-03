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
import torch
import torch.nn as nn
import yaml

# Maximize CPU parallelism — PyTorch defaults to half the available cores
import os as _os
import torch as _torch
_torch.set_num_threads(_os.cpu_count())
_torch.set_num_interop_threads(max(1, _os.cpu_count() // 2))
del _os, _torch

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
    p.add_argument("--skip-stages", action="store_true",
                   help="Skip auxiliary pipeline stages (correlogram/spatial analysis) — speeds up training runs")
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


class _FiLMHead(nn.Module):
    """FiLM-conditioned regression head: ERA5 modulates the trunk embedding.

    Instead of concatenating ERA5 to the trunk embedding (which forces the first
    linear layer to learn mixing), FiLM uses ERA5 to compute per-dimension scale
    (gamma) and shift (beta) applied to the trunk embedding *before* the MLP.
    This cleanly separates spatial structure (trunk) from weather conditioning (ERA5),
    matching the V-JEPA 2 principle of keeping the encoder pure and conditioning
    through the predictor pathway.

    era5_dim per window:
      morning: 8 (morning + evening)  — evening adds "preceding day" thermal context
               since morning UHI is driven by overnight heat retention, not active sun
      midday:  4 (midday only)        — cross-window chaining (train24) degraded by -0.474
      evening: 4 (evening only)       — same finding as midday
    """

    def __init__(self, hidden_dim: int, era5_dim: int):
        super().__init__()
        self.gamma_net = nn.Linear(era5_dim, hidden_dim)   # scale trunk dims
        self.beta_net  = nn.Linear(era5_dim, hidden_dim)   # shift trunk dims
        self.dropout   = nn.Dropout(0.15)
        self.proj1     = nn.Linear(hidden_dim, hidden_dim // 4)
        self.proj2     = nn.Linear(hidden_dim // 4, 1)

    def forward(self, h_trunk, era5_feat):
        """h_trunk: (N, D), era5_feat: (N, era5_dim) → (N, 1)"""
        import torch.nn.functional as F
        gamma = self.gamma_net(era5_feat)           # (N, D) — scale
        beta  = self.beta_net(era5_feat)            # (N, D) — shift
        h = self.dropout(F.gelu(h_trunk * gamma + beta))
        h = F.gelu(self.proj1(h))
        return self.proj2(h)


def _build_head(hidden_dim: int, era5_dim: int = 4, deep: bool = False) -> nn.Module:
    """Build a concat-MLP regression head with per-window depth (train32).

    train31 ablation revealed head depth is window-dependent:
      - Morning (overnight heat retention, simpler spatial pattern):
        2-layer is better (+0.461 vs +0.408 with 3-layer)
      - Midday/Evening (radiation-driven, richer feature interactions):
        3-layer is better (+0.126/+0.327 vs +0.016/+0.255 with 2-layer)

    deep=False → [h ‖ era5] → D/4 → 1          (morning)
    deep=True  → [h ‖ era5] → D/4 → D/8 → 1    (midday, evening)
    """
    in_dim = hidden_dim + era5_dim
    if deep:
        return nn.Sequential(
            nn.Linear(in_dim,            hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 4,   hidden_dim // 8),
            nn.GELU(),
            nn.Linear(hidden_dim // 8,   1),
        )
    return nn.Sequential(
        nn.Linear(in_dim,        hidden_dim // 4),
        nn.GELU(),
        nn.Dropout(0.15),
        nn.Linear(hidden_dim // 4, 1),
    )


# ---------------------------------------------------------------------------
# Climate zone and solar zenith helpers (train27)
# ---------------------------------------------------------------------------

# Köppen climate zone classification for each city.
# Used to weight the LOO ensemble — cities climatically similar to the
# holdout city (Philadelphia, Cfa) contribute more to the ensemble.
_KOPPEN_ZONE: dict[str, str] = {
    "philadelphia_pa": "Cfa",
    "atlanta_ga":      "Cfa",
    "raleigh_nc":      "Cfa",
    "wilmington_de":  "Cfa",   # added train32
    "baltimore_md":    "Cfa",   # added train32
    "charlotte_nc":    "Cfa",   # added train32
    "new_orleans_la":  "Cfa",   # added train32 (humid subtropical)
    "chicago_il":      "Dfa",
    "burlington_vt":   "Dfb",
    "providence_ri":   "Dfb",
    "seattle_wa":      "Cfb",
    "albuquerque_nm":  "BSk",
}

# Pairwise climate distance between zone codes.
# 0 = same zone, higher = more dissimilar.
# Based on: C/D letter (2pt gap), a/b/c subtype (1pt gap), B arid (3pt from C/D).
def _koppen_distance(zone_a: str, zone_b: str) -> float:
    if zone_a == zone_b:
        return 0.0
    letter_a, letter_b = zone_a[0], zone_b[0]
    # Arid (B) is very different from temperate/continental
    if "B" in (letter_a, letter_b):
        return 3.0
    # C vs D: different thermal regime
    letter_dist = 0.0 if letter_a == letter_b else 2.0
    # a/b/c precipitation subtype
    sub_a = zone_a[2] if len(zone_a) > 2 else ""
    sub_b = zone_b[2] if len(zone_b) > 2 else ""
    sub_dist = 0.0 if sub_a == sub_b else 1.0
    return letter_dist + sub_dist


def _koppen_weights(holdout_slug: str, city_slugs: list[str], temperature: float = 0.8) -> np.ndarray:
    """Softmax weights for ensemble cities based on Köppen distance to holdout.

    Closer climate zone → higher weight. temperature controls sharpness:
    lower = sharper (dominant city), higher = more uniform.
    """
    holdout_zone = _KOPPEN_ZONE.get(holdout_slug, "Cfa")
    distances = np.array([
        _koppen_distance(holdout_zone, _KOPPEN_ZONE.get(s, "Cfa"))
        for s in city_slugs
    ], dtype=np.float32)
    # Convert distance to similarity score: sim = exp(-dist / temperature)
    sims = np.exp(-distances / temperature)
    return sims / sims.sum()


def _solar_zenith_cos(lat_deg: float, lon_deg: float, date_str: str, hour_local: float) -> float:
    """Cosine of solar zenith angle for a city-window.

    cos(zenith) = sin(lat)*sin(dec) + cos(lat)*cos(dec)*cos(hour_angle)
    Returns cos(zenith) clipped to [0, 1] (negative = below horizon → 0).

    Args:
        lat_deg: city centroid latitude in degrees
        lon_deg: city centroid longitude in degrees
        date_str: ISO date string "YYYY-MM-DD"
        hour_local: approximate local solar hour (e.g. 7.0 = 7am, 14.0 = 2pm)
    """
    from math import sin, cos, radians, floor
    year_str, month_str, day_str = date_str.split("-")
    # Day-of-year
    import datetime
    doy = datetime.date(int(year_str), int(month_str), int(day_str)).timetuple().tm_yday
    # Solar declination (Spencer, 1971)
    B = radians(360 / 365 * (doy - 81))
    dec = radians(23.45 * sin(B))
    # Equation of time correction (minutes) — small but improves accuracy
    eot = 9.87 * sin(2 * B) - 7.53 * cos(B) - 1.5 * sin(B)
    # Local standard meridian (nearest 15°)
    lstm = 15 * round((-lon_deg) / 15)  # west lon → positive offset
    # True solar time
    tst = hour_local + (4 * (lstm - (-lon_deg)) + eot) / 60
    hour_angle = radians(15 * (tst - 12))
    lat = radians(lat_deg)
    cos_z = sin(lat) * sin(dec) + cos(lat) * cos(dec) * cos(hour_angle)
    return float(max(0.0, cos_z))


# Approximate local hours for each measurement window
_WINDOW_HOUR = {"morning": 7.0, "midday": 14.0, "evening": 19.0}


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
                 has_labels_per_window: dict = None, city_slug: str = "",
                 mean_embedding: object = None):
        self.trunk      = trunk
        self.heads      = heads   # {"morning": nn.Module, "midday": ..., "evening": ...}
        self.device     = device
        self.has_labels = has_labels  # False if city had no CAPA supervision at all
        self.city_slug  = city_slug
        # Per-window label availability: {"morning": True, "midday": True, "evening": False}
        self.has_labels_per_window = has_labels_per_window or {
            w: has_labels for w in ("morning", "midday", "evening")
        }
        # Mean trunk embedding for prototype-weighted ensemble (D,) tensor or None
        self.mean_embedding = mean_embedding

    def predict(self, x_land: object, era5_dict: dict) -> dict:
        import torch
        preds = {}
        with torch.no_grad():
            h = self.trunk(x_land)
            for window, head in self.heads.items():
                era5_feats = era5_dict.get(window)
                if era5_feats is None:
                    era5_feats = torch.zeros(h.shape[0], 4, device=self.device)
                preds[window] = head(torch.cat([h, era5_feats], dim=-1)).squeeze(-1)
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
    batch_size   = int(mc["jepa"].get("batch_size", 8192))  # large batch: fewer steps/epoch, faster on CPU
    lr           = float(mc["jepa"].get("lr", 1e-3))
    head_weight_decay = float(mc["jepa"].get("head_weight_decay", 0.1))
    seed         = int(mc["jepa"].get("seed", 42))

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

    # Trunk is FROZEN — JEPA pretraining already built good cross-city spatial
    # representations.  Unfreezing causes the trunk to drift city-by-city and
    # destroys its generalization (train11: -38°F bias; train12: -261°F bias).
    # The deeper 3-layer head now has enough capacity to decode the frozen
    # embeddings into UHI anomaly predictions without needing trunk adaptation.
    trunk_local = trunk
    for p in trunk_local.parameters():
        p.requires_grad_(False)
    trunk_local.eval()

    # Per-window ERA5 context tensors + solar zenith (train27).
    # Solar zenith cos(θ) is a city-level scalar (same for all pixels in a city)
    # that captures the radiation forcing intensity for each window.
    # It is appended to ERA5 so the FiLM head learns how much solar energy
    # is available to drive UHI at each time of day — without touching the trunk.
    # morning: [morning(4) + evening(4) + solar_zenith(1)] = 9-dim
    # midday:  [midday(4) + solar_zenith(1)] = 5-dim
    # evening: [evening(4) + solar_zenith(1)] = 5-dim
    # CRS is UTM (meters) — convert geometry centroid to WGS84 degrees for solar formula.
    _wgs84_pt = gdf.geometry.centroid.to_crs("EPSG:4326").iloc[0]
    city_lat  = float(_wgs84_pt.y)
    city_lon  = float(_wgs84_pt.x)
    date_str  = str(gdf["campaign_date"].iloc[0])[:10] if "campaign_date" in gdf.columns else "2021-07-15"
    solar_z = {
        w: float(_solar_zenith_cos(city_lat, city_lon, date_str, _WINDOW_HOUR[w]))
        for w in ("morning", "midday", "evening")
    }
    log.info("[%s] solar zenith cos  morning=%.3f  midday=%.3f  evening=%.3f  lat=%.2f  date=%s",
             city_slug, solar_z["morning"], solar_z["midday"], solar_z["evening"], city_lat, date_str)
    N = len(X_land)
    solar_tensors = {
        w: torch.full((N, 1), solar_z[w], device=device)
        for w in ("morning", "midday", "evening")
    }
    era5_inputs = {
        # train29: morning drops solar zenith — morning UHI is overnight heat *retention*,
        # not current radiation. Adding a low (0.2-0.4) solar value at 7am misleads the
        # head into thinking radiation matters for morning prediction (it doesn't).
        # Morning stays [morning(4) + evening(4)] = 8-dim (restored to t23 config).
        "morning": torch.cat([era5_tensors["morning"], era5_tensors["evening"]], dim=-1),  # (N, 8)
        # midday and evening keep solar zenith — midday at 2pm solar zenith 0.65-0.90
        # directly drives UHI via radiation onto impervious surfaces (train28: +0.191 vs +0.144).
        "midday":  torch.cat([era5_tensors["midday"],  solar_tensors["midday"]],  dim=-1),  # (N, 5)
        "evening": torch.cat([era5_tensors["evening"], solar_tensors["evening"]], dim=-1),  # (N, 5)
    }
    era5_dims = {"morning": 8, "midday": 5, "evening": 5}

    # Deterministic head init — fixes stochastic convergence across runs (T-2).
    import torch as _torch_seed
    _torch_seed.manual_seed(seed)

    heads = {
        w: _build_head(hidden_dim, era5_dim=era5_dims[w], deep=(w != "morning")).to(device)
        for w in ("morning", "midday", "evening")
    }

    head_params = [p for head in heads.values() for p in head.parameters()]
    # L2 weight decay on heads prevents covariate-shift extrapolation on OOD cities (T-1).
    opt = torch.optim.AdamW(head_params, lr=lr, weight_decay=head_weight_decay)

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
            with torch.no_grad():
                h = trunk_local(x_b)  # frozen trunk — no grad needed

            sup_loss = torch.tensor(0.0, device=device)
            for w, head in heads.items():
                pred = head(torch.cat([h, era5_inputs[w][b]], dim=-1)).squeeze(-1)
                valid_mask = labels[f"{w}_valid"][b]
                if valid_mask.sum() > 0:
                    sup_loss = sup_loss + mse_loss(pred[valid_mask], labels[w][b][valid_mask])

            loss = sup_loss  # trunk frozen — no EWC needed

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

    # Trunk is frozen — no Fisher/EWC needed (no trunk parameters were updated).
    fisher = {}
    theta_star = {}

    # Collect per-window UHI calibration stats for Option-C offset correction.
    # For each window: mean UHI anomaly (°F above ERA5) and mean ERA5 t2m (°F).
    uhi_stats: dict[str, dict] = {}
    for w in ("morning", "midday", "evening"):
        valid_mask = labels[f"{w}_valid"].cpu().numpy().astype(bool)
        if valid_mask.sum() > 10:
            anomaly_vals = labels[w].cpu().numpy()[valid_mask]
            era5_t2m_vals = (era5_tensors[w][:, 0].cpu().numpy() * 40.0 * 1.8 + 32.0)[valid_mask]
            uhi_stats[w] = {
                "mean_uhi_anomaly": float(anomaly_vals.mean()),
                "mean_era5_t2m_F":  float(era5_t2m_vals.mean()),
                "n": int(valid_mask.sum()),
            }

    # Compute mean trunk embedding for this city (used for prototype-weighted ensemble at LOO).
    # This is the centroid of the trunk's representation space for this city's pixels.
    with torch.no_grad():
        trunk_local.eval()
        mean_emb = trunk_local(X_land).mean(dim=0).cpu()  # (D,)

    return trunk_local, fisher, theta_star, _CityModel(
        trunk_local, heads, device,
        has_labels=any_labels,
        has_labels_per_window=has_labels_per_window,
        city_slug=city_slug,
        mean_embedding=mean_emb,
    ), uhi_stats


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
            pred = head(torch.cat([h, era5_b], dim=-1)).squeeze(-1)
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
    city_uhi_stats: list[dict] = None,
) -> dict:
    """Zero-shot prediction on holdout city using ensemble of trained city heads.

    Uses CAPA-supervised city models to produce an ensemble prediction on the
    holdout city (Philadelphia). For each time window, only includes city-head
    combos where that city had valid CAPA labels for that specific window.
    Falls back to a random head only if no trained models are available.

    Option-C UHI offset calibration: fits a simple linear model on training-city
    mean UHI anomalies vs mean ERA5 t2m, then applies a per-window bias correction
    to the ensemble prediction at inference time.
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

    # All 12 ERA5 features for each head (same as training)
    era5_all = torch.cat([
        era5_tensors["morning"],
        era5_tensors["midday"],
        era5_tensors["evening"],
    ], dim=-1)  # (N, 12)

    # Solar zenith for holdout city (train27) — same computation as training
    holdout_slug  = holdout_path.parent.name
    # CRS is UTM (meters) — convert geometry centroid to WGS84 degrees for solar formula.
    _wgs84_pt_loo = gdf.geometry.centroid.to_crs("EPSG:4326").iloc[0]
    city_lat  = float(_wgs84_pt_loo.y)
    city_lon  = float(_wgs84_pt_loo.x)
    date_str  = str(gdf["campaign_date"].iloc[0])[:10] if "campaign_date" in gdf.columns else "2021-07-15"
    solar_z_loo = {
        w: float(_solar_zenith_cos(city_lat, city_lon, date_str, _WINDOW_HOUR[w]))
        for w in ("morning", "midday", "evening")
    }
    log.info("LOO solar zenith cos  morning=%.3f  midday=%.3f  evening=%.3f  (%s)",
             solar_z_loo["morning"], solar_z_loo["midday"], solar_z_loo["evening"], holdout_slug)
    N_loo = len(X_land)
    solar_tensors_loo = {
        w: torch.full((N_loo, 1), solar_z_loo[w], device=device)
        for w in ("morning", "midday", "evening")
    }

    # Per-window ERA5 inputs matching the training configuration (train27: + solar zenith)
    era5_inputs_loo = {
        "morning": torch.cat([era5_tensors["morning"], era5_tensors["evening"]], dim=-1),  # (N, 8)
        "midday":  torch.cat([era5_tensors["midday"],  solar_tensors_loo["midday"]],  dim=-1),  # (N, 5)
        "evening": torch.cat([era5_tensors["evening"], solar_tensors_loo["evening"]], dim=-1),  # (N, 5)
    }

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
        if supervised_models:
            # All city models share the same final trunk (updated sequentially via EWC).
            # Compute the shared trunk embedding once per window, then apply each city head.
            # This guarantees coherent embeddings across the ensemble.
            shared_trunk = supervised_models[0].trunk
            shared_trunk.eval()
            for w in ("morning", "midday", "evening"):
                window_models = [cm for cm in supervised_models
                                 if cm.has_labels_per_window.get(w, False)]
                if not window_models:
                    log.warning("No supervised models with %s labels — skipping %s window", w, w)
                    continue
                log.info("LOO %s ensemble: %d city heads", w, len(window_models))
                h_shared_emb = shared_trunk(X_land)  # shared trunk embedding (computed once)
                era5_feat = era5_inputs_loo[w]  # per-window ERA5 (matches training config)
                era5_t2m_F = era5_tensors[w][:, 0] * 40.0 * 1.8 + 32.0
                city_preds = []
                for cm in window_models:
                    city_preds.append(cm.heads[w](torch.cat([h_shared_emb, era5_feat], dim=-1)).squeeze(-1))

                # Log per-city stats and identify outlier heads by mean prediction.
                # An outlier head is one whose spatial-mean absolute prediction deviates
                # more than the per-window robust-σ threshold from the group median (MAD).
                # MAD is much more resistant than mean/std to a single outlier in a small
                # ensemble (5 cities), where one bad head inflates std and lets others slip
                # through (e.g. Raleigh midday at 1.39σ survives a 1.5σ mean/std filter
                # but is correctly flagged at ~6 robust-σ under MAD).
                city_means = np.array([
                    float((cp + era5_t2m_F).cpu().mean()) for cp in city_preds
                ])
                med = float(np.median(city_means))
                mad = float(np.median(np.abs(city_means - med)))
                # Scale MAD to equivalent normal σ (MAD / 0.6745 ≈ σ for Gaussian).
                # Per-window thresholds: evening uses 1.5σ (tighter) to keep Chicago's
                # overnight cold-island head excluded even as the ensemble grows.
                # Morning/midday use 2.0σ — less aggressive, preserves useful diversity.
                _sigma_thresh = {"morning": 2.0, "midday": 2.0, "evening": 1.5}
                robust_std = mad / 0.6745
                if robust_std > 0:
                    keep_mask = np.abs(city_means - med) <= _sigma_thresh[w] * robust_std
                else:
                    keep_mask = np.ones(len(city_means), dtype=bool)
                for cm, cp, keep, cmean in zip(window_models, city_preds, keep_mask, city_means):
                    log.info("  per-city head [%s] %s mean_pred=%.2f°F  keep=%s",
                             cm.city_slug, w, cmean, "✓" if keep else "✗ (outlier — excluded)")
                if keep_mask.sum() == 0:
                    keep_mask[:] = True  # safety: never drop all cities
                trimmed_preds = [cp for cp, k in zip(city_preds, keep_mask) if k]
                trimmed_models = [cm for cm, k in zip(window_models, keep_mask) if k]
                if keep_mask.sum() < len(window_models):
                    log.info("  trimmed ensemble: %d/%d city heads kept", keep_mask.sum(), len(window_models))

                # train30: per-window ensemble weighting.
                # Morning UHI = overnight heat RETENTION driven by urban fabric density.
                # Chicago (Dfa) shares dense grid-pattern Rust Belt morphology with
                # Philadelphia, contributing strongly to morning spatial pattern despite
                # different climate zone. Köppen downweighting of Chicago from 0.25 → 0.04
                # caused morning regression from +0.467 → ~0 (t28/t29).
                # Solution: equal weight for morning (let MAD trimming do the work),
                # Köppen weighting for midday/evening (radiation-driven UHI where climate
                # zone correctly predicts relative intensity).
                if w == "morning":
                    kw_t = torch.ones(len(trimmed_preds), device=device) / len(trimmed_preds)
                    log.info("  morning: equal-weight ensemble (%d cities)", len(trimmed_preds))
                else:
                    koppen_w = _koppen_weights(
                        holdout_slug,
                        [cm.city_slug for cm in trimmed_models],
                        temperature=0.8,
                    )
                    for cm, kw in zip(trimmed_models, koppen_w):
                        log.info("  koppen weight [%s] %s zone=%s  w=%.3f",
                                 cm.city_slug, w,
                                 _KOPPEN_ZONE.get(cm.city_slug, "?"), kw)
                    kw_t = torch.tensor(koppen_w, dtype=torch.float32, device=device)
                stacked = torch.stack(trimmed_preds, dim=0)  # (K, N)
                pred_anomaly = (stacked * kw_t.unsqueeze(-1)).sum(dim=0)  # (N,)
                # Reconstruct absolute °F: anomaly + ERA5 t2m background (era5_t2m_F already computed above)
                pred = (pred_anomaly + era5_t2m_F).cpu().numpy()
                pred_cols[f"pred_{w}"] = pred

                label_col = _label_cols[w]
                if label_col in gdf.columns:
                    true_vals = gdf[label_col].values.astype(np.float32)
                    valid = ~np.isnan(true_vals)
                    if valid.sum() > 0:
                        rmse = float(np.sqrt(np.mean((pred[valid] - true_vals[valid]) ** 2)))
                        mae  = float(np.mean(np.abs(pred[valid] - true_vals[valid])))
                        bias = float(np.mean(pred[valid] - true_vals[valid]))
                        corr = float(np.corrcoef(pred[valid], true_vals[valid])[0, 1]) if valid.sum() > 2 else 0.0
                        results[w] = {"rmse": rmse, "mae": mae, "bias": bias, "corr": corr, "n": int(valid.sum())}
        else:
            # Fallback: random heads with shared pretrained trunk (baseline only)
            log.warning("No supervised city models available — LOO predictions are from random heads")
            h_shared = trunk(X_land)
            heads = {
                "morning": _build_head(hidden_dim, era5_dim=8, deep=False).to(device),
                "midday":  _build_head(hidden_dim, era5_dim=5, deep=True).to(device),
                "evening": _build_head(hidden_dim, era5_dim=5, deep=True).to(device),
            }
            for w, head in heads.items():
                pred = head(torch.cat([h_shared, era5_inputs_loo[w]], dim=-1)).squeeze(-1).cpu().numpy()
                pred_cols[f"pred_{w}"] = pred

    for col, vals in pred_cols.items():
        gdf[col] = vals

    # Option-C: UHI offset calibration.
    # Fit per-window linear model (era5_t2m → mean_uhi_anomaly) on training cities,
    # then apply bias correction to Philadelphia predictions.
    # Stores calibrated predictions alongside raw ones for comparison.
    calibrated_results: dict = {}
    if city_uhi_stats and pred_cols:
        for w in ("morning", "midday", "evening"):
            raw_key = f"pred_{w}"
            if raw_key not in pred_cols:
                continue
            # Gather training-city stats for this window
            train_era5_t2m = []
            train_uhi_mean = []
            for cs in city_uhi_stats:
                if w in cs and isinstance(cs[w], dict):
                    train_era5_t2m.append(cs[w]["mean_era5_t2m_F"])
                    train_uhi_mean.append(cs[w]["mean_uhi_anomaly"])
            if len(train_era5_t2m) < 2:
                log.info("Option-C: not enough training cities with %s labels — skipping calibration", w)
                continue
            x = np.array(train_era5_t2m, dtype=np.float64)
            y = np.array(train_uhi_mean, dtype=np.float64)
            # Simple OLS: slope and intercept
            x_c = x - x.mean()
            slope     = float(np.dot(x_c, y - y.mean()) / (np.dot(x_c, x_c) + 1e-9))
            intercept = float(y.mean() - slope * x.mean())

            # Philadelphia mean ERA5 t2m for this window
            era5_t2m_F_np = (era5_tensors[w][:, 0].cpu().numpy() * 40.0 * 1.8 + 32.0)
            holdout_mean_era5 = float(era5_t2m_F_np.mean())
            predicted_mean_uhi = slope * holdout_mean_era5 + intercept

            raw_pred = pred_cols[raw_key]
            raw_pred_anomaly = raw_pred - era5_t2m_F_np
            raw_mean_anomaly = float(raw_pred_anomaly.mean())
            bias_correction = predicted_mean_uhi - raw_mean_anomaly
            calibrated_pred = raw_pred + bias_correction

            cal_key = f"pred_{w}_calibrated"
            gdf[cal_key] = calibrated_pred

            log.info("Option-C  %s  slope=%.3f  intercept=%.3f  raw_mean_anomaly=%.2f°F  "
                     "predicted_uhi=%.2f°F  bias_correction=%.2f°F",
                     w, slope, intercept, raw_mean_anomaly, predicted_mean_uhi, bias_correction)

            label_col = _label_cols[w]
            if label_col in gdf.columns:
                true_vals = gdf[label_col].values.astype(np.float32)
                valid = ~np.isnan(true_vals)
                if valid.sum() > 0:
                    rmse_cal = float(np.sqrt(np.mean((calibrated_pred[valid] - true_vals[valid]) ** 2)))
                    mae_cal  = float(np.mean(np.abs(calibrated_pred[valid] - true_vals[valid])))
                    calibrated_results[w] = {
                        "rmse": rmse_cal, "mae": mae_cal, "n": int(valid.sum()),
                        "bias_correction": bias_correction,
                    }

    # Save predictions (raw + calibrated if available)
    pred_path = holdout_path.parent / "predictions.geoparquet"
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf.to_parquet(str(pred_path), index=False)
    log.info("Predictions saved → %s", pred_path)

    print()
    print("  Phase 3 -- Leave-One-Out Eval (Philadelphia, zero-shot)")
    print(f"  {'Window':<12}  {'RMSE (raw)':>10}  {'RMSE (cal)':>10}  {'Bias (raw)':>10}  {'Corr':>6}  {'N':>8}")
    print("  " + "─" * 65)
    for w in ("morning", "midday", "evening"):
        raw_m = results.get(w, {})
        cal_m = calibrated_results.get(w, {})
        rmse_r = f"{raw_m['rmse']:>10.4f}" if raw_m else "         —"
        rmse_c = f"{cal_m['rmse']:>10.4f}" if cal_m else "         —"
        bias_r = f"{raw_m.get('bias', float('nan')):>10.2f}" if raw_m else "         —"
        corr_r = f"{raw_m.get('corr', float('nan')):>6.3f}" if raw_m else "     —"
        n      = raw_m.get("n", "—")
        print(f"  {w:<12}  {rmse_r}  {rmse_c}  {bias_r}  {corr_r}  {n:>8}")
    print("  " + "─" * 65)

    return {"raw": results, "calibrated": calibrated_results}


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
    if not stages:
        log.debug("[%s] no pipeline stages specified -- skipping", city_slug)
        return
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
    if args.skip_stages:
        run_stages = []
        log.info("--skip-stages: auxiliary pipeline stages disabled for this run")
    elif args.stages:
        _stages_raw = [s.strip() for s in args.stages.split(",")]
        # Allow --stages none/skip/- to disable pipeline stages entirely
        run_stages = [s for s in _stages_raw if s.lower() not in ("none", "skip", "-", "")]
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

    # ── Phase 2 -- Per-City Head Training (unfrozen trunk, low lr) ─────────────
    # Trunk is UNFROZEN with a low lr (trunk_lr = lr * 0.1) so that CAPA heat
    # gradient supervision can shape the spatial representations learned in Phase 1.
    # EWC regularisation prevents catastrophic forgetting across cities.
    # Each city gets a deep-copied trunk so LOO can run each city's own trunk.
    print()
    print("═" * 60)
    print("  Phase 2 -- Per-City Head Training (frozen trunk, deeper heads)")
    print("═" * 60)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.skip_pretrain:
        # Move the loaded trunk to the correct device (already set in skip block above)
        trunk = trunk.to(device)

    fisher_matrices:     list[dict] = []
    optimal_params_list: list[dict] = []
    city_models: dict = {}
    city_uhi_stats_all: list[dict] = []   # for Option-C offset calibration

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
            # Pass the SHARED trunk — each city updates it sequentially in-place.
            # EWC prevents catastrophic forgetting of earlier cities.
            # At LOO time, all city heads share the same final trunk state,
            # so ensemble embeddings are coherent.  Per-city deep-copies created
            # incoherent embedding spaces and caused the ensemble to diverge (~-38°F bias).
            trunk, fisher, theta_star, city_model, uhi_stats = run_city_finetune(
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
            if uhi_stats:
                city_uhi_stats_all.append({"city": slug, **uhi_stats})
                for w, s in uhi_stats.items():
                    log.info("[%s] UHI calibration  %s  mean_anomaly=%.2f°F  era5_t2m=%.1f°F  n=%d",
                             slug, w, s["mean_uhi_anomaly"], s["mean_era5_t2m_F"], s["n"])
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
                    city_uhi_stats=city_uhi_stats_all,
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
