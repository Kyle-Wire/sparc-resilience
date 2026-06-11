"""
eval_anp_relative_uhi.py

Evaluate the relative-UHI ANP with real ASOS airport stations.

At inference:
  1. airport_bg = mean(ASOS_UHI) for the window
  2. station_relative = ASOS_UHI - airport_bg  ≈ 0 (they define the background)
  3. ANP predicts relative UHI for all pixels
  4. final_temp = ERA5 + airport_bg + relative_UHI_predicted
"""
import json
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
import geopandas as gpd
from sparc.inference.anp import SpatialANP
from scipy.spatial import cKDTree

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ── Load trunk ─────────────────────────────────────────────────────────────────
ckpt = torch.load("output/cities/jepa_pretrained_trunk.pt", map_location="cpu", weights_only=False)
in_d = ckpt["in_dim"]; h_d = ckpt.get("hidden_dim", 256)
x_m  = torch.tensor(ckpt["X_mean"], dtype=torch.float32, device=device)
x_s  = torch.tensor(ckpt["X_std"],  dtype=torch.float32, device=device)
trunk = nn.Sequential(
    nn.Linear(in_d, h_d), nn.GELU(),
    nn.Linear(h_d, h_d),  nn.GELU(),
    nn.Linear(h_d, h_d),
)
trunk.load_state_dict({k.replace("online.", ""): v for k, v in ckpt["trunk_state"].items()})
trunk.eval().to(device)

# ── Load Philadelphia predictions ──────────────────────────────────────────────
feat_cols = [
    "pct_impervious", "pct_canopy", "land_cover", "ndvi", "ndbi", "lst",
    "ndvi_s2", "ndbi_s2", "mndwi_s2", "albedo",
    "bldg_height_mean", "bldg_coverage", "svf", "cdc_svi",
    "elevation_m", "slope_deg", "aspect_deg", "distance_water_m",
]
gdf = gpd.read_parquet("output/cities/philadelphia_pa/predictions.geoparquet")
X_raw  = torch.tensor(gdf[feat_cols].fillna(0).values, dtype=torch.float32, device=device)
X_norm = ((X_raw - x_m) / (x_s + 1e-6)).nan_to_num(nan=0.0)
with torch.no_grad():
    H = trunk(X_norm)  # (N, 256)

cx = gdf.geometry.centroid.x.values
cy = gdf.geometry.centroid.y.values
coords = torch.tensor(np.stack([cx, cy], axis=1), dtype=torch.float32, device=device)
kd = cKDTree(np.stack([cx, cy], axis=1))

# ── Load ASOS stations ─────────────────────────────────────────────────────────
st = pd.read_parquet("output/cities/philadelphia_pa/stations.parquet")

# ── Load relative-UHI ANP ──────────────────────────────────────────────────────
anp = SpatialANP.from_checkpoint("output/anp_relative_uhi.pt").to(device).eval()
anp_v2 = SpatialANP.from_checkpoint("output/anp_station_conditioned_v2.pt").to(device).eval()

windows = [
    ("morning", "aat_morning",  "era5_morning_t2m",  "pred_morning",  "pred_morning_calibrated"),
    ("midday",  "aat_midday",   "era5_midday_t2m",   "pred_midday",   "pred_midday_calibrated"),
    ("evening", "aat_night",    "era5_evening_t2m",  "pred_evening",  "pred_evening_calibrated"),
]

print(f"{'Window':<10}  {'ZS_cal':>7}  {'ANP_v2':>7}  {'RelUHI':>8}  {'stBG':>7}  {'N':>3}")
print("-" * 56)

for win, label_col, era5_col, pred_raw_col, pred_cal_col in windows:
    if label_col not in gdf.columns:
        continue
    valid = gdf[label_col].notna()
    if valid.sum() == 0:
        continue
    labels   = torch.tensor(gdf.loc[valid, label_col].values, dtype=torch.float32)
    pred_cal = torch.tensor(gdf.loc[valid, pred_cal_col].values, dtype=torch.float32) if pred_cal_col in gdf.columns else None
    era5_abs = torch.tensor(gdf.loc[valid, era5_col].values * 9 / 5 + 32, dtype=torch.float32)

    # Station context for this window
    st_w = st[st["window"] == win].dropna(subset=["uhi_anomaly_f"])
    if len(st_w) == 0:
        continue

    # Airport background = mean of all station absolute UHIs
    airport_bg = float(st_w["uhi_anomaly_f"].mean())

    # Station relative UHI ≈ 0 (they ARE the background)
    st_rel = st_w["uhi_anomaly_f"].values - airport_bg  # should be near zero

    # Station embeddings
    _, nn_idx = kd.query(np.stack([st_w["easting"].values, st_w["northing"].values], axis=1))
    st_h = H[nn_idx]
    st_y_rel = torch.tensor(st_rel, dtype=torch.float32, device=device).unsqueeze(1)
    st_c = coords[nn_idx]

    # Valid pixel embeddings
    valid_idx = torch.where(torch.tensor(valid.values))[0]
    H_valid = H[valid_idx]
    C_valid = coords[valid_idx]

    with torch.no_grad():
        # Relative UHI ANP (trained on relative UHI)
        mu_rel, _ = anp(
            context_x=st_h, context_y=st_y_rel,
            target_x=H_valid,
            context_coords=st_c, target_coords=C_valid,
        )
        uhi_rel = mu_rel.squeeze(-1).cpu()
        pred_rel = era5_abs + airport_bg + uhi_rel  # ERA5 + background + relative

        # ANP v2 (absolute, for comparison)
        st_y_abs = torch.tensor(st_w["uhi_anomaly_f"].values, dtype=torch.float32, device=device).unsqueeze(1)
        mu_v2, _ = anp_v2(context_x=st_h, context_y=st_y_abs, target_x=H_valid,
                          context_coords=st_c, target_coords=C_valid)
        pred_v2 = era5_abs + mu_v2.squeeze(-1).cpu()

    rmse_cal = float((pred_cal - labels).pow(2).mean().sqrt()) if pred_cal is not None else float("nan")
    rmse_v2  = float((pred_v2  - labels).pow(2).mean().sqrt())
    rmse_rel = float((pred_rel - labels).pow(2).mean().sqrt())
    print(f"{win:<10}  {rmse_cal:>7.2f}  {rmse_v2:>7.2f}  {rmse_rel:>8.2f}  {airport_bg:>+7.2f}  {len(st_w):>3}")

print()
print("ZS_cal  = zero-shot calibrated baseline")
print("ANP_v2  = absolute UHI ANP (airport stations hurt)")
print("RelUHI  = relative UHI ANP (airports define background)")
print("stBG    = mean airport UHI (the background shift applied)")
