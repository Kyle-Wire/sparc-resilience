"""
PDE Physics Diagnostics — Validate α(s) field, physics loss activity,
ProcessRateNet behavior, blend weights, and Laplacian-source balance.

Run:  python scripts/pde_diagnostics.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "examples" / "providence_ri_uhi" / "providence_ri_uhi_run"
V2_DIR = RUN_DIR / "output" / "Stage_2_Spatial_CV" / "v2_neural"
DATA_FILE = ROOT / "examples" / "providence_ri_uhi" / "data" / "brown4.csv"

# ── Load artifacts ────────────────────────────────────────────────────
print("=" * 72)
print("  SPARC PDE Physics Diagnostics")
print("=" * 72)

# 1. Raw data
data = pd.read_csv(DATA_FILE)
print(f"\nDataset: {len(data)} rows, columns: {list(data.columns)}")

# 2. Alpha field
alpha = np.load(V2_DIR / "alpha_field.npy")
coords = np.load(V2_DIR / "alpha_field_coords.npy")
print(f"Alpha field: shape={alpha.shape}, range=[{alpha.min():.4f}, {alpha.max():.4f}], "
      f"mean={alpha.mean():.4f}, std={alpha.std():.4f}")

# 3. Predictions (has process_rate_alpha column)
preds = pd.read_csv(V2_DIR / "predictions.csv")

# 4. Meta info
with open(V2_DIR / "meta_info.json") as f:
    meta = json.load(f)

y_mean = meta["y_mean"]
y_std = meta["y_std"]

# 5. Project config
import yaml
with open(RUN_DIR / "project.yml", encoding="utf-8") as f:
    project_cfg = yaml.safe_load(f)

pr_cfg = project_cfg.get("process_rate", {})
physics_cfg = project_cfg.get("physics", {})

print(f"\nProcess rate config from project.yml:")
print(f"  name:       {pr_cfg.get('name', 'N/A')}")
print(f"  bounds:     {pr_cfg.get('bounds', 'NOT SET')}")
print(f"  prior_mean: {pr_cfg.get('prior_mean', 'NOT SET')}")
print(f"  inputs:     {pr_cfg.get('inputs', 'all features')}")

# Check if bounds mismatch
configured_bounds = pr_cfg.get("bounds", [0.0, 1.0])
if alpha.min() < configured_bounds[0] or alpha.max() > configured_bounds[1]:
    print(f"\n  ⚠️  BOUNDS MISMATCH: α field [{alpha.min():.4f}, {alpha.max():.4f}] "
          f"is outside configured bounds {configured_bounds}")
    print(f"  → The ProcessRateNet may be using default bounds [0.0, 1.0] "
          f"instead of project.yml bounds")
else:
    print(f"  ✓ α field is within configured bounds {configured_bounds}")


# ══════════════════════════════════════════════════════════════════════
# TEST 1: α(s) vs Land Cover Classification
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  TEST 1: α(s) Against Land Cover Classes")
print("=" * 72)

# Derive land cover classes from Pct_Impervious + Pct_Canopy + Distance_from_water_m
pct_imp = data["Pct_Impervious"].values
pct_can = data["Pct_Canopy"].values
dist_water = data["Distance_from_water_m"].values
ndvi = data["NDVI"].values

# Classification scheme:
#   - Water-adjacent (dist < 50m, low impervious): proxy for near-water
#   - High density urban: impervious >= 80%
#   - Moderate urban: 40% <= impervious < 80%
#   - Low density / mixed: 20% <= impervious < 40%
#   - Vegetated / parks: canopy >= 50% and impervious < 20%
#   - Open / green: impervious < 20% and canopy < 50%
classes = np.full(len(data), "unclassified", dtype=object)
classes[(pct_imp < 20) & (pct_can >= 50)] = "Vegetated/Parks"
classes[(pct_imp < 20) & (pct_can < 50)] = "Open/Green"
classes[(pct_imp >= 20) & (pct_imp < 40)] = "Low Density Urban"
classes[(pct_imp >= 40) & (pct_imp < 80)] = "Moderate Urban"
classes[(pct_imp >= 80)] = "High Density Urban"
classes[(dist_water < 50) & (pct_imp < 30)] = "Water-Adjacent"

# Expected physical ordering (highest to lowest α):
# Water-adjacent > Vegetated/Parks > Open/Green > Low Density > Moderate > High Density
expected_order = [
    "Water-Adjacent",
    "Vegetated/Parks",
    "Open/Green",
    "Low Density Urban",
    "Moderate Urban",
    "High Density Urban",
]

print("\nLand Cover Class           |  Count  |  Mean α  |  Std α  | Median α")
print("-" * 75)
class_means = {}
for cls in expected_order:
    mask = classes == cls
    n = mask.sum()
    if n > 0:
        m = alpha[mask].mean()
        s = alpha[mask].std()
        med = np.median(alpha[mask])
        class_means[cls] = m
        print(f"  {cls:<24s} | {n:>5d}   | {m:.4f}   | {s:.4f}  | {med:.4f}")
    else:
        print(f"  {cls:<24s} | {0:>5d}   |   N/A    |   N/A   |   N/A")

# Check ordering
print("\n  Expected ordering check (α should decrease from top to bottom):")
prev_cls = None
prev_mean = None
violations = 0
for cls in expected_order:
    if cls in class_means:
        if prev_mean is not None and class_means[cls] >= prev_mean:
            print(f"    ✗ {cls} (α={class_means[cls]:.4f}) ≥ {prev_cls} (α={prev_mean:.4f}) — VIOLATION")
            violations += 1
        elif prev_mean is not None:
            print(f"    ✓ {cls} (α={class_means[cls]:.4f}) < {prev_cls} (α={prev_mean:.4f})")
        prev_cls = cls
        prev_mean = class_means[cls]

if violations == 0:
    print("  ✓ Physical ordering is CORRECT ✓")
else:
    print(f"  ✗ {violations} ordering violation(s) — α may not be physically meaningful")

# Dynamic range
if class_means:
    vals = list(class_means.values())
    dr = max(vals) / min(vals) if min(vals) > 0 else float('inf')
    spread = max(vals) - min(vals)
    print(f"\n  Dynamic range: max/min = {dr:.2f}x, spread = {spread:.4f}")
    if dr < 1.5:
        print("  ⚠️  Dynamic range < 1.5× — α field is too compressed to be physically meaningful")
    elif dr < 2.0:
        print("  ⚡ Dynamic range 1.5–2.0× — marginal differentiation")
    else:
        print("  ✓ Dynamic range ≥ 2.0× — good physical differentiation")


# ══════════════════════════════════════════════════════════════════════
# TEST 2: α(s) Gradient Against Impervious + Canopy
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  TEST 2: α(s) Correlation with Pct_Impervious & Pct_Canopy")
print("=" * 72)

from scipy.stats import pearsonr, spearmanr

# α vs Pct_Impervious (expect negative)
r_pears_imp, p_pears_imp = pearsonr(alpha, pct_imp)
r_spear_imp, p_spear_imp = spearmanr(alpha, pct_imp)
print(f"\n  α(s) vs Pct_Impervious:")
print(f"    Pearson  r = {r_pears_imp:+.4f}  (p = {p_pears_imp:.2e})")
print(f"    Spearman ρ = {r_spear_imp:+.4f}  (p = {p_spear_imp:.2e})")
if r_pears_imp > -0.2:
    print(f"    ⚠️  Weak/wrong sign — α not responding to impervious surface")
elif r_pears_imp > -0.3:
    print(f"    ⚡ Marginal negative correlation (want r < -0.3 for paper)")
else:
    print(f"    ✓ Good negative correlation — α decreases with impervious")

# α vs Pct_Canopy (expect positive)
r_pears_can, p_pears_can = pearsonr(alpha, pct_can)
r_spear_can, p_spear_can = spearmanr(alpha, pct_can)
print(f"\n  α(s) vs Pct_Canopy:")
print(f"    Pearson  r = {r_pears_can:+.4f}  (p = {p_pears_can:.2e})")
print(f"    Spearman ρ = {r_spear_can:+.4f}  (p = {p_spear_can:.2e})")
if r_pears_can < 0.2:
    print(f"    ⚠️  Weak/wrong sign — α not responding to canopy")
elif r_pears_can < 0.3:
    print(f"    ⚡ Marginal positive correlation (want r > 0.3 for paper)")
else:
    print(f"    ✓ Good positive correlation — α increases with canopy")

# α vs NDVI (expect positive)
r_pears_ndvi, p_pears_ndvi = pearsonr(alpha, ndvi)
r_spear_ndvi, p_spear_ndvi = spearmanr(alpha, ndvi)
print(f"\n  α(s) vs NDVI:")
print(f"    Pearson  r = {r_pears_ndvi:+.4f}  (p = {p_pears_ndvi:.2e})")
print(f"    Spearman ρ = {r_spear_ndvi:+.4f}  (p = {p_spear_ndvi:.2e})")

# α vs Distance_from_water_m (expect positive — further from water → less cooling → context)
r_pears_dw, p_pears_dw = pearsonr(alpha, dist_water)
r_spear_dw, p_spear_dw = spearmanr(alpha, dist_water)
print(f"\n  α(s) vs Distance_from_water_m:")
print(f"    Pearson  r = {r_pears_dw:+.4f}  (p = {p_pears_dw:.2e})")
print(f"    Spearman ρ = {r_spear_dw:+.4f}  (p = {p_spear_dw:.2e})")

# α vs Albedo
albedo = data["Albedo"].values
r_pears_alb, p_pears_alb = pearsonr(alpha, albedo)
r_spear_alb, p_spear_alb = spearmanr(alpha, albedo)
print(f"\n  α(s) vs Albedo:")
print(f"    Pearson  r = {r_pears_alb:+.4f}  (p = {p_pears_alb:.2e})")
print(f"    Spearman ρ = {r_spear_alb:+.4f}  (p = {p_spear_alb:.2e})")


# ══════════════════════════════════════════════════════════════════════
# TEST 3: Physics Loss Activity
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  TEST 3: Physics Loss Activity (Training Loss Breakdown)")
print("=" * 72)

# Load oof_results.npz — check if loss_history was saved
oof_results = np.load(V2_DIR / "oof_results.npz", allow_pickle=True)
print(f"  oof_results.npz keys: {list(oof_results.keys())}")

# Check for loss_history in the npz or as a separate file
loss_file = V2_DIR / "loss_history.json"
has_loss_history = False

if "loss_history" in oof_results and oof_results["loss_history"] is not None:
    loss_history = oof_results["loss_history"]
    if isinstance(loss_history, np.ndarray):
        loss_history = loss_history.tolist()
    has_loss_history = True
    print(f"  Found loss_history in oof_results.npz ({len(loss_history)} epochs)")
elif loss_file.exists():
    with open(loss_file) as f:
        loss_history = json.load(f)
    has_loss_history = True
    print(f"  Found loss_history.json ({len(loss_history)} epochs)")
else:
    print("  ℹ️  No loss_history found in artifacts")
    print("  → Need to check retrain_loss_history from the training code")
    print("  → This was likely stored in oof_results.npz but may require re-examination")

# Try loading from saved fields in pde_diagnostics
# The loss history may be stored as Python object in npz
for key in oof_results.keys():
    arr = oof_results[key]
    print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}")

if has_loss_history and isinstance(loss_history, list) and len(loss_history) > 0:
    # Print last 5 epochs
    print(f"\n  Final training loss breakdown (last 5 epochs):")
    for epoch_data in loss_history[-5:]:
        if isinstance(epoch_data, dict):
            print(f"    Epoch: {epoch_data}")
        else:
            print(f"    {epoch_data}")

    # Check relative magnitudes
    if isinstance(loss_history[-1], dict):
        final = loss_history[-1]
        mse = final.get("mse", 0)
        physics = final.get("physics", 0)
        pde_total = final.get("pde_total", 0)
        total_prediction = mse
        total_physics = physics + pde_total
        if total_prediction > 0:
            ratio = total_physics / total_prediction
            print(f"\n  Physics-to-prediction ratio: {ratio:.4f}")
            if ratio < 0.01:
                print("  ⚠️  Physics loss < 1% of prediction loss — EFFECTIVELY INACTIVE")
                print("  → Increase lambda_pde to 0.10–0.20 in project.yml")
            elif ratio < 0.10:
                print("  ⚡ Physics 1–10% of prediction — MARGINAL influence")
            else:
                print("  ✓ Physics ≥10% of prediction — ACTIVE")


# ══════════════════════════════════════════════════════════════════════
# TEST 4: ProcessRateNet Pretraining Check
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  TEST 4: ProcessRateNet Architecture & Bounds Check")
print("=" * 72)

# Reload the actual ProcessRateNet and check its bounds
sys.path.insert(0, str(ROOT))
from sparc.models.process_rate_net import ProcessRateNet

# Load the saved state dict and reconstruct
state_dict = torch.load(V2_DIR / "process_rate_net.pt", map_location="cpu", weights_only=True)
print(f"  Saved state_dict keys: {list(state_dict.keys())}")

# Check the network input dim from weights
input_dim = state_dict["network.0.weight"].shape[1]
hidden_dim_pr = state_dict["network.0.weight"].shape[0]
print(f"  Input dim: {input_dim}, Hidden dim: {hidden_dim_pr}")

# Reconstruct with project.yml config
process_net_configured = ProcessRateNet(
    n_inputs=input_dim,
    domain_config={
        "name": pr_cfg.get("name", "rate"),
        "units": pr_cfg.get("units", ""),
        "bounds": pr_cfg.get("bounds", [0.0, 1.0]),
        "prior_mean": pr_cfg.get("prior_mean", 0.5),
    },
)
process_net_configured.load_state_dict(state_dict)
process_net_configured.eval()

print(f"  Configured bounds: [{process_net_configured.bounds_lo}, {process_net_configured.bounds_hi}]")
print(f"  Configured prior_mean: {process_net_configured.prior_mean}")

# Also reconstruct with DEFAULT bounds [0, 1] to see which matches the data
process_net_default = ProcessRateNet(
    n_inputs=input_dim,
    domain_config={
        "name": "rate",
        "units": "",
        "bounds": [0.0, 1.0],
        "prior_mean": 0.5,
    },
)
process_net_default.load_state_dict(state_dict)
process_net_default.eval()

# Load feature scaler to prepare inputs
feat_scaling = np.load(V2_DIR / "feature_scaling.npz")
feat_mean = feat_scaling["feat_mean"]
feat_std = feat_scaling["feat_std"]

feature_names = meta["feature_names"]
# Use actual process_rate_inputs from meta_info (what was actually used during training)
actual_pr_inputs = meta.get("process_rate_inputs", feature_names)
pr_col_idxs = [feature_names.index(f) for f in actual_pr_inputs if f in feature_names]
if not pr_col_idxs or len(pr_col_idxs) != input_dim:
    pr_col_idxs = list(range(input_dim))

print(f"  Actual PR inputs during training: {actual_pr_inputs} ({len(actual_pr_inputs)} features)")
cfg_pr_inputs = pr_cfg.get("inputs", "all features (default)")
print(f"  Configured PR inputs in project.yml: {cfg_pr_inputs}")
if isinstance(cfg_pr_inputs, list) and len(cfg_pr_inputs) != input_dim:
    print(f"  ⚠️  CONFIG MISMATCH: project.yml specifies {len(cfg_pr_inputs)} inputs "
          f"but model was trained with {input_dim}")
    print(f"  → process_rate.inputs config was NOT applied during training")

# Build raw features, then scale
raw_features = data[feature_names].values.astype(np.float32)
scaled_features = (raw_features - feat_mean) / (feat_std + 1e-8)
pr_input = torch.tensor(scaled_features[:, pr_col_idxs], dtype=torch.float32)

with torch.no_grad():
    alpha_configured = process_net_configured(pr_input).squeeze(-1).numpy()
    alpha_default = process_net_default(pr_input).squeeze(-1).numpy()

print(f"\n  With configured bounds {pr_cfg.get('bounds', [0.0, 1.0])}:")
print(f"    α range: [{alpha_configured.min():.4f}, {alpha_configured.max():.4f}], "
      f"mean={alpha_configured.mean():.4f}")

print(f"  With default bounds [0.0, 1.0]:")
print(f"    α range: [{alpha_default.min():.4f}, {alpha_default.max():.4f}], "
      f"mean={alpha_default.mean():.4f}")

# Compare with the saved alpha field
print(f"\n  Saved α field:  [{alpha.min():.4f}, {alpha.max():.4f}], mean={alpha.mean():.4f}")

# Which one matches?
diff_cfg = np.abs(alpha_configured - alpha).mean()
diff_def = np.abs(alpha_default - alpha).mean()
print(f"  Mean abs diff from configured bounds: {diff_cfg:.6f}")
print(f"  Mean abs diff from default bounds:    {diff_def:.6f}")

if diff_cfg < diff_def:
    print(f"  → Saved field matches CONFIGURED bounds")
    actual_bounds = pr_cfg.get("bounds", [0.0, 1.0])
else:
    print(f"  → Saved field matches DEFAULT bounds [0.0, 1.0]")
    print(f"  ⚠️  project.yml bounds were NOT used during training!")
    actual_bounds = [0.0, 1.0]


# ══════════════════════════════════════════════════════════════════════
# TEST 5: Mixture Prior Targets
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  TEST 5: Mixture Prior Differentiation")
print("=" * 72)

material_priors = pr_cfg.get("material_priors", {})
if material_priors:
    print(f"  Material priors from config:")
    for mat, val in material_priors.items():
        print(f"    {mat}: α = {val}")
    vals = list(material_priors.values())
    ratio = max(vals) / min(vals) if min(vals) > 0 else float('inf')
    print(f"  Range ratio: {ratio:.2f}x")
    if ratio < 2.0:
        print("  ⚠️  Prior targets span < 2× — insufficient gradient signal for differentiation")
    else:
        print("  ✓ Prior targets span ≥ 2×")
else:
    print("  ⚠️  NO material_priors configured in project.yml!")
    print("  → ProcessRateNet pretraining is SKIPPED (no mixture target)")
    print("  → α(s) is learned entirely from prediction loss — no physics anchoring")
    print("  → This is a major reason for the compressed α(s) field")
    print("  → Recommended: Add material_priors to process_rate section:")
    print("      material_priors:")
    print("        impervious: 0.25   # low responsiveness (locally equilibrated)")
    print("        soil:       0.45   # moderate")
    print("        vegetation: 0.65   # moderate-high (transpiration cooling)")
    print("        water:      0.85   # high (fast equilibration)")


# ══════════════════════════════════════════════════════════════════════
# TEST 6: w(s) Blend Weight vs Distance from Water
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  TEST 6: Blend Weight w(s) vs Distance from Water")
print("=" * 72)

# Check if w_source was saved
w_source = None
# It should be in pde_visualizations data — try loading from the model
# by reconstruction
try:
    from sparc.models.neural_meta import SPARCMetaLearner
    from sparc.models.process_rate_net import SourceTermNet
    from sparc.models.surrogates import (
        DifferentiableGWR, DifferentiableGWRF, DifferentiableGGPGAM,
    )
    import joblib

    # Load sinusoidal encoder
    encoder = joblib.load(V2_DIR / "sinusoidal_encoder.pkl")

    # Reconstruct model
    model = SPARCMetaLearner(
        n_base_models=meta["n_surrogates"],
        n_physics_features=meta["n_physics_extended"],
        d_spatial=meta["d_spatial"],
        hidden_dim=meta["hidden_dim"],
        dropout=0.1,
        thresholds=meta["thresholds_norm"],
        n_heads=4,
        max_neighbors=128,
    )
    model.load_state_dict(
        torch.load(V2_DIR / "neural_meta.pt", map_location="cpu", weights_only=True)
    )
    model.eval()

    # Check if model has _last_w_source after a forward pass
    # We need surrogates + data for a forward pass which is complex
    # Instead, check if w_source was stored in the PDE plots data
    print("  Model loaded. Checking for w_source from forward pass...")

    # Try a quick forward pass with a small subset
    # We need base predictions, spatial features, etc.
    # Load from predictions.csv
    surr_gwr = preds["surrogate_gwr"].values
    surr_gwrf = preds["surrogate_gwrf"].values
    surr_ggpgam = preds["surrogate_ggpgam"].values

    # Normalize targets
    surr_base = np.stack([
        (surr_gwr - y_mean) / y_std,
        (surr_gwrf - y_mean) / y_std,
        (surr_ggpgam - y_mean) / y_std,
    ], axis=1)

    # Use a subsample for the forward pass
    N_sub = min(5000, len(data))
    sub_idx = np.random.RandomState(42).choice(len(data), N_sub, replace=False)

    base_t = torch.tensor(surr_base[sub_idx], dtype=torch.float32)
    phys_t = torch.tensor(scaled_features[sub_idx], dtype=torch.float32)

    # Compute spatial features
    sub_coords = coords[sub_idx]
    X_spatial = encoder.transform(sub_coords)
    spat_t = torch.tensor(X_spatial, dtype=torch.float32)
    coords_t = torch.tensor(sub_coords, dtype=torch.float32)

    # Build KNN for subset
    from sklearn.neighbors import NearestNeighbors
    nn_model = NearestNeighbors(n_neighbors=min(128, N_sub - 1), algorithm="ball_tree")
    nn_model.fit(sub_coords)
    knn_dists_np, knn_idx_np = nn_model.kneighbors(sub_coords)
    knn_idx_t = torch.tensor(knn_idx_np, dtype=torch.long)

    # Compute physics_feats_extended for subset
    from sparc.physics.pde_operators import estimate_local_spacing
    from sparc.physics.input_derivatives import compute_predictor_derivatives, normalize_derivatives
    from sparc.run.v2_neural_training import _build_cardinal_neighbors

    cardinal_np, _ = _build_cardinal_neighbors(sub_coords, resolution=meta["resolution"])
    cardinal_t = torch.tensor(cardinal_np, dtype=torch.long)
    h_field = estimate_local_spacing(coords_t, cardinal_t, spacing="local")

    raw_derivs, deriv_names = compute_predictor_derivatives(
        phys_t, cardinal_t, h_field, meta["feature_names"],
    )
    deriv_feats, _, _ = normalize_derivatives(raw_derivs)
    phys_ext = torch.cat([phys_t, deriv_feats], dim=1)

    # Alpha for subset
    pr_input_sub = phys_t[:, pr_col_idxs]
    with torch.no_grad():
        alpha_sub = process_net_default(pr_input_sub) if diff_def < diff_cfg else process_net_configured(pr_input_sub)

    # Forward pass
    with torch.no_grad():
        T_pred, exc_preds, attn = model(
            base_preds=base_t,
            physics_feats=phys_ext,
            X_spatial=spat_t,
            coords=coords_t,
            knn_index=knn_idx_t,
            alpha=alpha_sub,
        )

    if hasattr(model, "_last_w_source") and model._last_w_source is not None:
        w_source = model._last_w_source.cpu().numpy().squeeze()
        print(f"  w(s) field: shape={w_source.shape}, range=[{w_source.min():.4f}, {w_source.max():.4f}]")
        print(f"  w(s) mean={w_source.mean():.4f}, std={w_source.std():.4f}")

        # Bin by distance from water
        dist_sub = dist_water[sub_idx]
        bins = [0, 50, 200, 500, 1000, 2000, np.inf]
        print(f"\n  Distance bins (w should decrease near water):")
        print(f"  {'Distance Range':<20s} | Count |  Mean w  |  Std w")
        print(f"  {'-'*55}")
        for i in range(len(bins) - 1):
            mask = (dist_sub >= bins[i]) & (dist_sub < bins[i + 1])
            n = mask.sum()
            if n > 0:
                label = f"{bins[i]:.0f}–{bins[i+1]:.0f}m" if bins[i+1] < np.inf else f"{bins[i]:.0f}m+"
                print(f"  {label:<20s} | {n:>5d} | {w_source[mask].mean():.4f}   | {w_source[mask].std():.4f}")
    else:
        print("  ℹ️  Model does not expose _last_w_source — w(s) not available")
        print("  → Check if PDEInformedPhysicsEncoder stores blend weights")

except Exception as exc:
    print(f"  ⚠️  Could not compute w(s): {exc}")
    import traceback
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
# TEST 7: Laplacian vs Source Term Scale Check
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  TEST 7: Laplacian Scale vs Source Term")
print("=" * 72)

try:
    from sparc.physics.pde_operators import laplacian

    # Use predictions to compute Laplacian
    actual_temp = data["AAT_z"].values.astype(np.float32)
    T_norm = (actual_temp - y_mean) / y_std
    T_t = torch.tensor(T_norm, dtype=torch.float32)

    # We need full cardinal neighbors
    from sparc.run.v2_neural_training import _build_cardinal_neighbors
    full_coords = coords  # already in projected CRS
    full_cardinal_np, _ = _build_cardinal_neighbors(full_coords, resolution=meta["resolution"])
    full_cardinal_t = torch.tensor(full_cardinal_np, dtype=torch.long)

    h = meta["resolution"]
    lap_T, valid = laplacian(T_t, full_cardinal_t, h)

    print(f"\n  Laplacian ∇²T:")
    print(f"    Valid points: {valid.sum()} / {len(T_t)}")
    print(f"    Mean: {lap_T.mean():.6f}")
    print(f"    Std:  {lap_T.std():.6f}")
    print(f"    Range: [{lap_T.min():.6f}, {lap_T.max():.6f}]")

    # Diffusion term: α * ∇²T
    alpha_valid = alpha[valid.numpy()] if valid.sum() < len(alpha) else alpha
    lap_np = lap_T.numpy()
    if len(alpha_valid) == len(lap_np):
        diffusion = alpha_valid * lap_np
    else:
        # Expand laplacian back to full size
        lap_full = np.zeros(len(alpha))
        lap_full[valid.numpy()] = lap_np
        diffusion = alpha * lap_full

    print(f"\n  Diffusion term α∇²T:")
    print(f"    Std:  {np.std(diffusion[diffusion != 0]):.6f}")

    # Estimate source term magnitude from the learned SourceTermNet
    # or from energy balance
    print(f"\n  Estimated magnitude comparison:")
    print(f"    α * std(∇²T) ≈ {alpha.mean():.4f} × {lap_T.std():.6f} = {alpha.mean() * lap_T.std().item():.6f}")
    print(f"    If source term S ≫ α∇²T by 10×+, then α is unidentifiable from steady-state data")

    # Check actual temperature variance for context
    print(f"\n  Temperature field stats (normalized):")
    print(f"    Mean: {T_norm.mean():.4f}, Std: {T_norm.std():.4f}")
    print(f"    Range: [{T_norm.min():.4f}, {T_norm.max():.4f}]")
    print(f"    Physical range: [{actual_temp.min():.1f}°F, {actual_temp.max():.1f}°F]")

except Exception as exc:
    print(f"  ⚠️  Could not compute Laplacian check: {exc}")
    import traceback
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  SUMMARY & RECOMMENDATIONS")
print("=" * 72)

issues = []

# 1. Check Gini / coefficient of variation
cv = alpha.std() / alpha.mean() if alpha.mean() > 0 else 0
print(f"\n  α(s) Coefficient of Variation: {cv:.4f}")
if cv < 0.10:
    issues.append("α(s) coefficient of variation < 10% — field is too uniform")
    print(f"  ⚠️  CV < 10% — α is nearly constant (network learned to ignore physics)")

# 2. Check dynamic range
if class_means:
    vals = list(class_means.values())
    dr = max(vals) / min(vals)
    if dr < 1.5:
        issues.append("α(s) dynamic range < 1.5× across land cover classes")

# 3. Bounds check
if alpha.min() < configured_bounds[0] - 0.01:
    issues.append(f"α values below configured lower bound ({configured_bounds[0]})")

# 4. Correlations
if abs(r_pears_imp) < 0.2:
    issues.append(f"α vs Pct_Impervious correlation too weak ({r_pears_imp:+.4f})")
if abs(r_pears_can) < 0.2:
    issues.append(f"α vs Pct_Canopy correlation too weak ({r_pears_can:+.4f})")

# 5. Material priors
if not material_priors:
    issues.append("No material_priors configured — pretraining skipped entirely")

if issues:
    print(f"\n  ❌ {len(issues)} issues found:")
    for i, issue in enumerate(issues, 1):
        print(f"    {i}. {issue}")
else:
    print(f"\n  ✅ No critical issues — α(s) field appears physically meaningful")

print("\n" + "=" * 72)
print("  Diagnostics complete")
print("=" * 72)
