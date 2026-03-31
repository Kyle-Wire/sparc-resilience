"""Temporary analysis script — delete after use."""
import numpy as np
import pandas as pd
import json

# 1. Sensitivity package: GGPGAM derivatives + GWRF curves
with open('brown_uhi_run/output/spatial_intelligence/sensitivity_package.json') as f:
    sp = json.load(f)

print("=== GGPGAM Derivatives ===")
if 'ggpgam_derivatives' in sp:
    for var, info in sp['ggpgam_derivatives'].items():
        md = info.get('mean_derivative', '?')
        sc = info.get('sign_consistent', '?')
        print(f"  {var:25s}  mean_deriv={md}  sign_ok={sc}")

print("\n=== GWRF Condition Curves ===")
if 'gwrf_condition_curves' in sp:
    for var, info in sp['gwrf_condition_curves'].items():
        if isinstance(info, dict):
            cf = info.get('curve_fit', {})
            print(f"  {var:25s}  r2={cf.get('r2','?')}  trend={cf.get('trend','?')}")

# 2. MGWR local coefficients
features = ['Distance_from_water_m', 'Pct_Impervious', 'Pct_Canopy',
            'NDVI', 'Albedo', 'Elevation_m']
mgwr = pd.read_csv(
    'brown_uhi_run/output/Stage_2_Spatial_CV/base_models_full/mgwr_local_coefficients.csv'
)
print(f"\nMGWR columns: {mgwr.columns.tolist()}")
print(f"Assumed feature order: {features}")
print()

for i, var in enumerate(features):
    col = f'feature_{i}'
    if col not in mgwr.columns:
        continue
    c = mgwr[col].values
    print(f"MGWR {var} ({col}):")
    print(f"  mean={c.mean():+.6f}  std={c.std():.6f}")
    print(f"  % positive: {100*np.mean(c>0):.1f}%   % negative: {100*np.mean(c<0):.1f}%")
    pcts = [np.percentile(c, q) for q in [5, 25, 50, 75, 95]]
    print(f"  p5={pcts[0]:+.6f}  p25={pcts[1]:+.6f}  p50={pcts[2]:+.6f}  p75={pcts[3]:+.6f}  p95={pcts[4]:+.6f}")
    print()

# 3. Also check the consensus weights from the scenario run
print("=== CONSENSUS WEIGHT INFO ===")
print("(from run output): ols=0.092, gwr=0.203, gwrf=0.355, ggpgam=0.349")
print()

# 4. Per-model decomposition using the GPKG
import geopandas as gpd
gdf = gpd.read_file('brown_uhi_run/output/Stage_4_Scenarios/scenario_results_reprediction.gpkg')

# Check what columns exist — any per-model delta columns?
delta_cols = [c for c in gdf.columns if 'delta' in c.lower() or 'repred' in c.lower()]
print(f"Delta/repred columns: {delta_cols[:10]}...")
print()

# Cross-tabulate warming vs Pct_Canopy baseline deciles for canopy+25
vals = gdf['repred_delta_Pct_Canopy_plus_25'].values
canopy_bl = gdf['Pct_Canopy'].values
deciles = pd.qcut(canopy_bl, 10, labels=False, duplicates='drop')

print("=== Canopy +25pp delta by baseline canopy decile ===")
for d in range(10):
    mask = deciles == d
    v = vals[mask]
    bl = canopy_bl[mask]
    print(f"  Decile {d} (canopy {bl.min():.0f}-{bl.max():.0f}): "
          f"mean delta={v.mean():+.5f}  warm={100*np.mean(v>0):.0f}%  cool={100*np.mean(v<0):.0f}%")

# Also by location (lat deciles)
print()
lat = gdf['latitude'].values if 'latitude' in gdf.columns else gdf.geometry.y.values
lat_dec = pd.qcut(lat, 10, labels=False, duplicates='drop')
print("=== Canopy +25pp delta by latitude decile (S to N) ===")
for d in range(10):
    mask = lat_dec == d
    v = vals[mask]
    la = lat[mask]
    print(f"  Lat decile {d} ({la.min():.0f}-{la.max():.0f}): "
          f"mean delta={v.mean():+.5f}  warm={100*np.mean(v>0):.0f}%  cool={100*np.mean(v<0):.0f}%")
