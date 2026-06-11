"""E1 evaluation: ANP v2 with real ASOS weather stations."""
import sys, torch, numpy as np, pandas as pd
sys.path.insert(0, ".")
import geopandas as gpd
from sparc.inference.anp import SpatialANP
import torch.nn as nn
from scipy.spatial import cKDTree

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load trunk
ckpt = torch.load("output/cities/jepa_pretrained_trunk.pt", map_location="cpu", weights_only=False)
in_d = ckpt["in_dim"]
h_d = ckpt.get("hidden_dim", 256)
x_m = np.array(ckpt["X_mean"], dtype=np.float32)
x_s = np.array(ckpt["X_std"], dtype=np.float32)
trunk = nn.Sequential(
    nn.Linear(in_d, h_d), nn.GELU(),
    nn.Linear(h_d, h_d), nn.GELU(),
    nn.Linear(h_d, h_d),
)
trunk.load_state_dict({k.replace("online.", ""): v for k, v in ckpt["trunk_state"].items()})
trunk.eval().to(device)
x_m_t = torch.tensor(x_m, device=device)
x_s_t = torch.tensor(x_s, device=device)

# Load Philadelphia predictions
gdf = gpd.read_parquet("output/cities/philadelphia_pa/predictions.geoparquet")
feat_cols = [
    "pct_impervious", "pct_canopy", "land_cover", "ndvi", "ndbi", "lst",
    "ndvi_s2", "ndbi_s2", "mndwi_s2", "albedo", "bldg_height_mean",
    "bldg_coverage", "svf", "cdc_svi", "elevation_m", "slope_deg",
    "aspect_deg", "distance_water_m",
]
X_raw = torch.tensor(gdf[feat_cols].fillna(0).values, dtype=torch.float32, device=device)
X_norm = ((X_raw - x_m_t) / (x_s_t + 1e-6)).nan_to_num(nan=0.0)
with torch.no_grad():
    H = trunk(X_norm)

# Extract pixel centroids from geometry
cx = gdf.geometry.centroid.x.values
cy = gdf.geometry.centroid.y.values
coords = torch.tensor(np.stack([cx, cy], axis=1), dtype=torch.float32, device=device)
kd = cKDTree(np.stack([cx, cy], axis=1))

# Load ASOS stations
st = pd.read_parquet("output/cities/philadelphia_pa/stations.parquet")

# Load ANP v2
anp = SpatialANP.from_checkpoint("output/anp_station_conditioned_v2.pt").to(device).eval()

windows = [
    ("morning", "aat_morning",  "era5_morning_t2m", "pred_morning",  "pred_morning_calibrated"),
    ("midday",  "aat_midday",   "era5_midday_t2m",  "pred_midday",   "pred_midday_calibrated"),
    ("evening", "aat_night",    "era5_evening_t2m", "pred_evening",  "pred_evening_calibrated"),
]

print()
print(f"{'Window':<10}  {'ZS_raw':>7}  {'ZS_cal':>7}  {'ANP_equal':>9}  {'ANP_20km':>10}  {'ANP_5km':>9}  {'stUHI':>7}  {'N':>3}")
print("-" * 80)

for win, label_col, era5_col, pred_raw_col, pred_cal_col in windows:
    if label_col not in gdf.columns:
        continue
    valid = gdf[label_col].notna()
    if valid.sum() == 0:
        continue
    labels = torch.tensor(gdf.loc[valid, label_col].values, dtype=torch.float32)
    pred_raw = (
        torch.tensor(gdf.loc[valid, pred_raw_col].values, dtype=torch.float32)
        if pred_raw_col in gdf.columns else None
    )
    pred_cal = (
        torch.tensor(gdf.loc[valid, pred_cal_col].values, dtype=torch.float32)
        if pred_cal_col in gdf.columns else None
    )

    # Station context for this window
    st_win = st[st["window"] == win].dropna(subset=["uhi_anomaly_f"])
    if len(st_win) == 0:
        continue
    mean_st_uhi = st_win["uhi_anomaly_f"].mean()

    # Build station embeddings via nearest pixel
    _, nn_idx = kd.query(np.stack([st_win["easting"].values, st_win["northing"].values], axis=1))
    st_h = H[nn_idx].to(device)
    st_y = torch.tensor(st_win["uhi_anomaly_f"].values, dtype=torch.float32, device=device).unsqueeze(1)
    st_c = coords[nn_idx]

    # Valid pixel embeddings & coords
    valid_idx = torch.where(torch.tensor(valid.values))[0]
    H_valid = H[valid_idx]
    C_valid = coords[valid_idx]

    with torch.no_grad():
        # Matérn disabled: equal weight for all stations
        mu_equal, _ = anp(
            context_x=st_h,
            context_y=st_y,
            target_x=H_valid,
        )
        # Matérn enabled: stations decay by distance (range=20km)
        mu_matern, _ = anp(
            context_x=st_h,
            context_y=st_y,
            target_x=H_valid,
            context_coords=st_c,
            target_coords=C_valid,
            matern_range_m=20_000.0,
        )
        # Tight Matérn: range=5km (airport influence drops fast)
        mu_tight, _ = anp(
            context_x=st_h,
            context_y=st_y,
            target_x=H_valid,
            context_coords=st_c,
            target_coords=C_valid,
            matern_range_m=5_000.0,
        )
    uhi_equal  = mu_equal.squeeze(-1).cpu()
    uhi_matern = mu_matern.squeeze(-1).cpu()
    uhi_tight  = mu_tight.squeeze(-1).cpu()

    era5_abs = torch.tensor(gdf.loc[valid, era5_col].values * 9 / 5 + 32, dtype=torch.float32)
    rmse_raw = float((pred_raw - labels).pow(2).mean().sqrt()) if pred_raw is not None else float("nan")
    rmse_cal = float((pred_cal - labels).pow(2).mean().sqrt()) if pred_cal is not None else float("nan")
    rmse_equal  = float(((era5_abs + uhi_equal)  - labels).pow(2).mean().sqrt())
    rmse_matern = float(((era5_abs + uhi_matern) - labels).pow(2).mean().sqrt())
    rmse_tight  = float(((era5_abs + uhi_tight)  - labels).pow(2).mean().sqrt())
    mean_st_uhi = st_win["uhi_anomaly_f"].mean()
    print(f"{win:<10}  {rmse_raw:>7.2f}  {rmse_cal:>7.2f}  {rmse_equal:>9.2f}  {rmse_matern:>10.2f}  {rmse_tight:>9.2f}  {mean_st_uhi:>7.2f}  {len(st_win):>3}")

print()
print("ANP_equal  = all stations equal weight (no spatial decay)")
print("ANP_20km   = Matern-3/2 range=20km")
print("ANP_5km    = Matern-3/2 range=5km (tight: airports far away get near-zero weight)")
