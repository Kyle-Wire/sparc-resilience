"""Standalone script to export V2 neural readable outputs from trained models."""
import torch
import json
import numpy as np
import joblib
import pandas as pd
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

project_path = Path(r"examples/providence_ri_uhi/providence_ri_uhi_run/project.yml")
base_dir = Path(r"examples/providence_ri_uhi/providence_ri_uhi_run/output/Stage_2_Spatial_CV")
v2_dir = base_dir / "v2_neural"
device = torch.device("cpu")

# Load config
with open(project_path, encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Load meta info
with open(v2_dir / "meta_info.json") as f:
    meta = json.load(f)

feature_names = meta["feature_names"]
n_physics = meta["n_physics_features"]
d_spatial = meta["d_spatial"]
hidden_dim = meta["hidden_dim"]
thresholds = meta["thresholds"]
thresholds_norm = meta["thresholds_norm"]
y_mean = meta["y_mean"]
y_std = meta["y_std"]
n_base = meta["n_base_models"]

# Load data
from sparc.data.data_utils import load_and_preprocess_data
data_cfg = config.get("data", {})
project_dir = project_path.parent
raw_csv = str(project_dir / data_cfg["file_path"])
target_col = data_cfg.get("target_column", "AAT_z")
identifier_col = data_cfg.get("identifier_column", "OBJECTID")
coord_columns = data_cfg.get("coord_columns", ["POINT_X", "POINT_Y"])
crs_cfg = config.get("crs", {})
initial_crs = crs_cfg.get("input", "EPSG:3438")
target_crs = crs_cfg.get("projected", "EPSG:26919")

gdf = load_and_preprocess_data(
    raw_data_path=raw_csv,
    identifier_col=identifier_col,
    target_col=target_col,
    coords_cols=coord_columns,
    predictor_cols=feature_names,
    initial_crs=initial_crs,
    target_crs=target_crs,
    output_dir=str(base_dir),
)
y = gdf[target_col].values
coords = gdf[["projected_X", "projected_Y"]].values.astype(np.float64)
feature_matrix = gdf[feature_names].values.astype(np.float32)
print(f"Data loaded: {len(y)} rows, {len(feature_names)} features")

# Prepare tensors
from sparc.run.v2_neural_training import _prepare_tensors
tensors = _prepare_tensors(y, coords, feature_matrix, config, device)

# Reconstruct models
from sparc.models.neural_meta import SPARCMetaLearner
from sparc.models.process_rate_net import ProcessRateNet
from sparc.models.surrogates import (
    DifferentiableGWR,
    DifferentiableGWRF,
    DifferentiableGGPGAM,
)

neural_cfg = config.get("models", {}).get("neural", {})
max_neighbors = neural_cfg.get("max_neighbors", 128)
n_heads = neural_cfg.get("n_heads", 4)
dropout = neural_cfg.get("dropout", 0.1)

model = SPARCMetaLearner(
    n_base_models=n_base,
    n_physics_features=n_physics,
    d_spatial=d_spatial,
    hidden_dim=hidden_dim,
    n_heads=n_heads,
    dropout=dropout,
    thresholds=thresholds_norm,
    max_neighbors=max_neighbors,
).to(device)
model.load_state_dict(
    torch.load(v2_dir / "neural_meta.pt", map_location=device, weights_only=True)
)
model.eval()

pr_cfg = config.get("process_rate", {})
# Use the actual inputs from meta_info (what the model was trained with)
pr_inputs = meta.get("process_rate_inputs", pr_cfg.get("inputs", feature_names))
pr_input_dim = len(pr_inputs)
pr_col_idxs = [feature_names.index(f) for f in pr_inputs if f in feature_names]
if not pr_col_idxs:
    pr_col_idxs = list(range(min(pr_input_dim, n_physics)))

prior_mean = pr_cfg.get("prior_mean", 0.5)

process = ProcessRateNet(
    n_inputs=pr_input_dim,
    domain_config={
        "name": pr_cfg.get("name", "rate"),
        "units": pr_cfg.get("units", ""),
        "bounds": pr_cfg.get("bounds", [0.0, 1.0]),
        "prior_mean": prior_mean,
    },
).to(device)
process.load_state_dict(
    torch.load(v2_dir / "process_rate_net.pt", map_location=device, weights_only=True)
)
process.eval()

bw = np.ones(n_physics, dtype=np.float32)
from sparc.run.v2_neural_training import _load_correlogram_bandwidths
predictor_bandwidths = _load_correlogram_bandwidths(base_dir, feature_names)
if predictor_bandwidths is not None:
    bw = predictor_bandwidths

surrogates = {}
surrogates["gwr"] = DifferentiableGWR(
    n_vars=n_physics, n_spatial_features=d_spatial,
    hidden_dim=hidden_dim, bandwidths=bw,
).to(device)
surrogates["gwr"].load_state_dict(
    torch.load(v2_dir / "surrogate_gwr.pt", map_location=device, weights_only=True)
)

surrogates["gwrf"] = DifferentiableGWRF(
    n_vars=n_physics, n_spatial_features=d_spatial, hidden_dim=hidden_dim,
).to(device)
surrogates["gwrf"].load_state_dict(
    torch.load(v2_dir / "surrogate_gwrf.pt", map_location=device, weights_only=True)
)

surrogates["ggpgam"] = DifferentiableGGPGAM(
    n_vars=n_physics, n_spatial_features=d_spatial, hidden_dim=min(hidden_dim, 32),
).to(device)
surrogates["ggpgam"].load_state_dict(
    torch.load(v2_dir / "surrogate_ggpgam.pt", map_location=device, weights_only=True)
)

for s in surrogates.values():
    s.eval()

print("All models loaded successfully")

# Load OOF results
oof = np.load(v2_dir / "oof_results.npz")
oof_preds = oof["predictions"]
oof_std = oof["uncertainty"]

gwrf_k = min(neural_cfg.get("gwrf_neighbors", 16), max_neighbors)

# Run export
from sparc.run.v2_neural_training import _export_v2_outputs

_export_v2_outputs(
    final_model=model,
    final_process=process,
    final_surrogates=surrogates,
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
    artifact_dir=v2_dir,
    device=device,
)
print("Export complete!")
