"""
eval_anp_capa_pseudo.py

E2 evaluation: quantify few-shot gain vs zero-shot using CAPA-labeled pixels
as pseudo weather stations.

For N in [0, 1, 2, 3, 5, 10, 20]:
  - Sample N labeled pixels as pseudo-stations (context)
  - Use remaining labeled pixels as targets
  - ANP predicts UHI for targets given context observations
  - Compute RMSE vs ground-truth CAPA labels
  - Repeat for n_trials random seeds and report mean ± std

N=0 is the zero-shot baseline: ANP uses its empty-context prior.

Produces a comparison table alongside zero-shot calibrated predictions.

Usage:
    python scripts/eval_anp_capa_pseudo.py \\
        --anp-checkpoint output/anp_station_conditioned.pt \\
        --predictions output/cities/philadelphia_pa/predictions.geoparquet \\
        --trunk output/jepa_pretrained_trunk.pt \\
        --window midday \\
        --n-values 0 1 2 3 5 10 20 \\
        --n-trials 10 --seed 42
"""
from __future__ import annotations

import argparse
import logging
import sys

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from sparc.inference.anp import SpatialANP

log = logging.getLogger("eval_anp_capa_pseudo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")

_FEAT_COLS = [
    "pct_impervious", "pct_canopy", "land_cover", "ndvi", "ndbi", "lst",
    "ndvi_s2", "ndbi_s2", "mndwi_s2", "albedo",
    "bldg_height_mean", "bldg_coverage", "svf", "cdc_svi",
    "elevation_m", "slope_deg", "aspect_deg", "distance_water_m",
]
_LABEL_COLS = {"morning": "aat_morning", "midday": "aat_midday", "evening": "aat_night"}
_ERA5_T2M   = {"morning": "era5_morning_t2m", "midday": "era5_midday_t2m", "evening": "era5_evening_t2m"}


def build_trunk(trunk_path: str, device: torch.device):
    ckpt  = torch.load(trunk_path, map_location="cpu", weights_only=False)
    in_d  = ckpt["in_dim"]
    h_d   = ckpt.get("hidden_dim", 256)
    x_m   = np.array(ckpt["X_mean"], dtype=np.float32)
    x_s   = np.array(ckpt["X_std"],  dtype=np.float32)
    trunk = nn.Sequential(
        nn.Linear(in_d, h_d), nn.GELU(),
        nn.Linear(h_d,  h_d), nn.GELU(),
        nn.Linear(h_d,  h_d),
    )
    trunk.load_state_dict(ckpt["trunk_state"])
    trunk.eval()
    return trunk.to(device), x_m, x_s


def compute_trunk_embeddings(gdf: gpd.GeoDataFrame,
                             trunk, x_mean: np.ndarray, x_std: np.ndarray,
                             device: torch.device) -> torch.Tensor:
    feat_cols = [c for c in _FEAT_COLS if c in gdf.columns]
    X_raw = gdf[feat_cols].values.astype(np.float32)
    X_raw = np.nan_to_num(
        (X_raw - x_mean[:X_raw.shape[1]]) / x_std[:X_raw.shape[1]], nan=0.0
    )
    with torch.no_grad():
        return trunk(torch.tensor(X_raw, device=device))


def era5_t2m_fahrenheit(gdf: gpd.GeoDataFrame, window: str) -> np.ndarray:
    col = _ERA5_T2M[window]
    if col not in gdf.columns:
        raise KeyError(f"ERA5 column {col!r} not in predictions file")
    # Column stores raw ERA5 t2m in °C — convert to °F (matches training convention)
    return gdf[col].values.astype(np.float32) * 9.0 / 5.0 + 32.0


def run_eval(
    anp: SpatialANP,
    trunk_emb: torch.Tensor,   # (N_total, 256)
    uhi_true: np.ndarray,       # (N_valid,) UHI anomaly in °F for labeled pixels
    labels_abs: np.ndarray,     # (N_valid,) absolute temperature in °F
    era5_labeled: np.ndarray,   # (N_valid,) ERA5 t2m in °F for labeled pixels
    labeled_idx: np.ndarray,    # indices into full gdf for labeled pixels
    n_context: int,
    n_trials: int,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Returns (mean_rmse, std_rmse) over n_trials random sub-samplings.

    RMSE is computed in absolute temperature space (°F) for comparability
    with zero-shot ensemble RMSE.
    """
    rmses = []
    for _ in range(n_trials):
        if n_context == 0:
            # Zero-shot: empty context → ANP uses empty_context_prior
            ctx_x = torch.zeros(0, trunk_emb.shape[1], device=device)
            ctx_y = torch.zeros(0, 1, device=device)
        else:
            ctx_idx = rng.choice(len(labeled_idx), size=n_context, replace=False)
            global_ctx_idx = labeled_idx[ctx_idx]
            ctx_x = trunk_emb[global_ctx_idx]             # (K, 256)
            ctx_y = torch.tensor(
                uhi_true[ctx_idx], dtype=torch.float32, device=device
            ).unsqueeze(-1)                                # (K, 1)

        # All labeled pixels are targets
        tgt_x = trunk_emb[labeled_idx]  # (N_valid, 256)

        with torch.no_grad():
            mu, _ = anp(context_x=ctx_x, context_y=ctx_y, target_x=tgt_x)

        uhi_pred  = mu.squeeze(-1).cpu().numpy()
        abs_pred  = era5_labeled + uhi_pred   # absolute temperature in °F
        rmse = float(np.sqrt(np.mean((abs_pred - labels_abs) ** 2)))
        rmses.append(rmse)

    return float(np.mean(rmses)), float(np.std(rmses))


def main():
    p = argparse.ArgumentParser(description="E2: ANP few-shot N-curve evaluation with CAPA pseudo-stations")
    p.add_argument("--anp-checkpoint", required=True, help="Trained ANP checkpoint (.pt)")
    p.add_argument("--predictions",    required=True, help="predictions.geoparquet with CAPA labels")
    p.add_argument("--trunk",          required=True, help="jepa_pretrained_trunk.pt")
    p.add_argument("--window",         default="midday", choices=["morning", "midday", "evening"])
    p.add_argument("--n-values",       nargs="+", type=int, default=[0, 1, 2, 3, 5, 10, 20],
                   help="Context sizes N to evaluate")
    p.add_argument("--n-trials",       type=int, default=10, help="Trials per N")
    p.add_argument("--seed",           type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    log.info("Loading ANP checkpoint: %s", args.anp_checkpoint)
    anp = SpatialANP.from_checkpoint(args.anp_checkpoint).to(device)
    anp.eval()

    log.info("Loading predictions: %s", args.predictions)
    gdf = gpd.read_parquet(args.predictions)

    log.info("Building trunk embeddings…")
    trunk, x_mean, x_std = build_trunk(args.trunk, device)
    trunk_emb = compute_trunk_embeddings(gdf, trunk, x_mean, x_std, device)

    w = args.window
    label_col = _LABEL_COLS[w]
    if label_col not in gdf.columns:
        log.error("Label column %r not in predictions file", label_col)
        sys.exit(1)

    labels = gdf[label_col].values.astype(np.float32)
    era5   = era5_t2m_fahrenheit(gdf, w)

    labeled_mask = ~np.isnan(labels)
    labeled_idx  = np.where(labeled_mask)[0]
    n_labeled    = int(labeled_mask.sum())
    log.info("Labeled pixels: %d / %d for window=%s", n_labeled, len(gdf), w)

    if n_labeled < 20:
        log.error("Too few labeled pixels (%d) for meaningful evaluation", n_labeled)
        sys.exit(1)

    # UHI ground truth = label - ERA5 background  (for context_y)
    labels_abs   = labels[labeled_idx]          # (N_valid,) absolute °F
    era5_labeled = era5[labeled_idx]            # (N_valid,) ERA5 °F
    uhi_true     = labels_abs - era5_labeled    # (N_valid,) UHI anomaly °F

    # Zero-shot baseline from saved calibrated predictions
    cal_col = f"pred_{w}_calibrated"
    if cal_col in gdf.columns:
        zs_pred   = gdf[cal_col].values.astype(np.float32)[labeled_idx]
        rmse_zs_cal = float(np.sqrt(np.mean((zs_pred - labels_abs) ** 2)))
    else:
        rmse_zs_cal = float("nan")
    raw_col = f"pred_{w}"
    if raw_col in gdf.columns:
        zs_raw  = gdf[raw_col].values.astype(np.float32)[labeled_idx]
        rmse_zs_raw = float(np.sqrt(np.mean((zs_raw - labels_abs) ** 2)))
    else:
        rmse_zs_raw = float("nan")

    rng = np.random.default_rng(args.seed)

    max_n = max(args.n_values)
    if max_n >= n_labeled:
        log.warning("Max N=%d >= n_labeled=%d; capping at n_labeled//2=%d",
                    max_n, n_labeled, n_labeled // 2)
        args.n_values = [v for v in args.n_values if v < n_labeled // 2]

    log.info("Running N-curve: N=%s  trials=%d", args.n_values, args.n_trials)
    rows = []
    for n in args.n_values:
        mean_r, std_r = run_eval(
            anp=anp,
            trunk_emb=trunk_emb,
            uhi_true=uhi_true,
            labels_abs=labels_abs,
            era5_labeled=era5_labeled,
            labeled_idx=labeled_idx,
            n_context=n,
            n_trials=args.n_trials,
            device=device,
            rng=rng,
        )
        rows.append((n, mean_r, std_r))
        log.info("  N=%2d  RMSE=%.4f ± %.4f °F", n, mean_r, std_r)

    print()
    print(f"  E2: ANP few-shot N-curve  (window={w})")
    print(f"  Zero-shot ensemble (raw):        RMSE = {rmse_zs_raw:.4f}°F")
    print(f"  Zero-shot ensemble (calibrated): RMSE = {rmse_zs_cal:.4f}°F")
    print()
    print(f"  {'N':>4}  {'RMSE (°F)':>10}  {'± std':>8}  {'Δ vs ZS-cal':>12}")
    print("  " + "-" * 42)
    for n, mean_r, std_r in rows:
        delta = mean_r - rmse_zs_cal
        delta_str = f"{delta:>+.4f}" if not np.isnan(rmse_zs_cal) else "       —"
        print(f"  {n:>4}  {mean_r:>10.4f}  {std_r:>8.4f}  {delta_str:>12}")
    print()


if __name__ == "__main__":
    main()
