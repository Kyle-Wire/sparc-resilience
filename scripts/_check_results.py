"""Quick diagnostic check after alpha supervision loss."""
import numpy as np
import torch
import pandas as pd

out = "examples/providence_ri_uhi/providence_ri_uhi_run/output/Stage_2_Spatial_CV/v2_neural"

# Alpha field
alpha = np.load(f"{out}/alpha_field.npy")
print("=== ALPHA FIELD ===")
print(f"  mean={alpha.mean():.4f}, std={alpha.std():.4f}, min={alpha.min():.4f}, max={alpha.max():.4f}")

# Loss history
lh = np.load(f"{out}/loss_history.npz", allow_pickle=True)
print(f"\n=== LOSS HISTORY ===")
keys = list(lh.keys())
print(f"  Epochs: {len(lh['total'])}")
print(f"  Components: {keys}")
print(f"  Total: {lh['total'][0]:.4f} -> {lh['total'][-1]:.4f}")
for k in sorted(keys):
    if k != "total":
        vals = lh[k]
        print(f"    {k}: {vals[0]:.4f} -> {vals[-1]:.4f}")

# Blend weights
print("\n=== BLEND WEIGHTS ===")
ckpt = torch.load(f"{out}/neural_meta.pt", map_location="cpu", weights_only=False)
for key in sorted(ckpt.keys()):
    if "blend" in key.lower() or "weight_net" in key.lower():
        v = ckpt[key]
        if hasattr(v, "shape"):
            print(f"  {key}: shape={v.shape}")
            if "weight" in key and v.dim() == 2:
                pos = (v > 0).sum().item()
                neg = (v <= 0).sum().item()
                print(f"    Positive: {pos}, Negative/Zero: {neg}")
            if "bias" in key and v.numel() == 1:
                print(f"    Value: {v.item():.6f}")
            elif "bias" in key:
                print(f"    Values: mean={v.mean():.6f}, std={v.std():.6f}")

# w(s) / predictions
preds = pd.read_csv(f"{out}/predictions.csv")
print(f"\n=== PREDICTIONS COLUMNS ===")
print(f"  {list(preds.columns)}")
if "w_source" in preds.columns:
    w = preds["w_source"].values
    print(f"\n=== w(s) FIELD ===")
    print(f"  mean={w.mean():.4f}, std={w.std():.4f}, min={w.min():.4f}, max={w.max():.4f}")
if "blend_weight" in preds.columns:
    w = preds["blend_weight"].values
    print(f"\n=== BLEND WEIGHT FIELD ===")
    print(f"  mean={w.mean():.4f}, std={w.std():.4f}, min={w.min():.4f}, max={w.max():.4f}")

# Alpha by impervious quartile
data = pd.read_csv("examples/providence_ri_uhi/data/brown4.csv")
imp = data["Pct_Impervious"].values
q = pd.qcut(imp, 4, labels=["Q1", "Q2", "Q3", "Q4"])
print("\n=== ALPHA BY IMPERVIOUS QUARTILE ===")
for label in ["Q1", "Q2", "Q3", "Q4"]:
    mask = q == label
    print(f"  {label}: mean_alpha={alpha[mask].mean():.4f}, std={alpha[mask].std():.4f}")

# Cardinal neighbors
cn = np.load(f"{out}/cardinal_neighbors.npy")
self_loops = (cn == np.arange(cn.shape[0])[:, None]).any(axis=1).sum()
boundary = (cn == -1).any(axis=1).sum()
print(f"\n=== CARDINAL NEIGHBORS ===")
print(f"  Self-loops: {self_loops}")
print(f"  Boundary points: {boundary}")

# BC loss check
print(f"\n=== BC LOSS CHECK ===")
bc_vals = lh.get("bc", None)
if bc_vals is not None:
    first_nonzero = next((i for i, v in enumerate(bc_vals) if v > 0), None)
    print(f"  BC activates at epoch: {first_nonzero}")
    print(f"  BC final: {bc_vals[-1]:.6f}")
    print(f"  BC max: {bc_vals.max():.6f}")

# v4 scenario_results — long-format table from artifact store
print(f"\n=== v4 SCENARIO_RESULTS (DB) ===")
try:
    from pathlib import Path
    from sparc.registry.run_registry import RunRegistry, set_active_registry
    from sparc.registry.store import ArtifactStore

    db_path = Path("examples/providence_ri_uhi/providence_ri_uhi_run/output/artifacts.db")
    if not db_path.exists():
        print(f"  No artifacts.db at {db_path}")
    else:
        reg = RunRegistry(db_path.parent)
        set_active_registry(reg)
        try:
            store = ArtifactStore(reg)
            if not store.has("4", "scenario_results"):
                print("  No (\"4\",\"scenario_results\") in store")
            else:
                df = store.read_table("4", "scenario_results")
                print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")
                if "mode" in df.columns:
                    for mode, sub in df.groupby("mode"):
                        dm = sub["delta_mean"].astype(float)
                        print(
                            f"  mode={mode}: n={len(sub)}, "
                            f"delta_mean=[{dm.min():.4f}, {dm.max():.4f}], "
                            f"|mean|={dm.abs().mean():.4f}"
                        )
        finally:
            set_active_registry(None)
except Exception as exc:  # noqa: BLE001
    print(f"  Error: {exc}")
