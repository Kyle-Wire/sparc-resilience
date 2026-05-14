"""
Standalone JEPA spatial-patch pretraining diagnostic.

Loads brown4.csv directly, builds minimal trunk + latent predictor,
runs the spatial-patch JEPA pretraining loop, and prints structured
results: loss curves, collapse diagnostics, and patch-coverage stats.

Usage:
    python scripts/run_jepa_standalone.py
"""
from __future__ import annotations

import sys, time, math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from sparc.training.jepa_loss import (
    JEPALossWeights,
    jepa_loss,
    spatial_patch_mask,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH   = ROOT / "my_project" / "brown4.csv"
HIDDEN_DIM  = 128
N_EPOCHS    = 30
BATCH_SIZE  = 1024
LR          = 1e-3
MASK_RATIO  = 0.40
N_PATCHES   = 4
EMA_TAU     = 0.99
SEED        = 42

torch.manual_seed(SEED)
device = torch.device("cpu")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH)
FEATURE_COLS = ["Pct_Canopy", "Pct_Impervious", "Distance_from_water_m",
                "Elevation_m", "NDVI", "Albedo"]
COORD_COLS   = ["POINT_X", "POINT_Y"]

# Drop rows with any NaN in features or coords
df = df[FEATURE_COLS + COORD_COLS].dropna().reset_index(drop=True)
N, D = len(df), len(FEATURE_COLS)
print(f"  {N:,} points  |  {D} features  |  {len(COORD_COLS)} coord dims")

# Normalise features and coords
X_raw   = df[FEATURE_COLS].values.astype(np.float32)
X_mean, X_std = X_raw.mean(0), X_raw.std(0) + 1e-6
X       = torch.tensor((X_raw - X_mean) / X_std, device=device)

C_raw   = df[COORD_COLS].values.astype(np.float32)
C_mean, C_std = C_raw.mean(0), C_raw.std(0) + 1e-6
coords  = torch.tensor((C_raw - C_mean) / C_std, device=device)

# Coord range info for context
coord_span_m = (C_raw.max(0) - C_raw.min(0))
print(f"  Coord span: {coord_span_m[0]:.0f} × {coord_span_m[1]:.0f} units  "
      f"(normalised to ~[-3,3])")

# ---------------------------------------------------------------------------
# Minimal trunk: 2-layer MLP  (stand-in for physics_enc + trunk_fusion)
# ---------------------------------------------------------------------------
class TinyTrunk(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TinyPredictor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

online  = TinyTrunk(D, HIDDEN_DIM).to(device)
ema     = TinyTrunk(D, HIDDEN_DIM).to(device)
predictor = TinyPredictor(HIDDEN_DIM).to(device)

# Initialise EMA = online
ema.load_state_dict(online.state_dict())
for p in ema.parameters():
    p.requires_grad_(False)

opt = torch.optim.AdamW(
    list(online.parameters()) + list(predictor.parameters()), lr=LR
)
weights = JEPALossWeights(alignment=1.0, variance=1.0, covariance=0.04,
                          variance_gamma=1.0)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print(f"\nRunning spatial-patch JEPA pretraining  "
      f"(mask_ratio={MASK_RATIO}, n_patches={N_PATCHES}, "
      f"hidden={HIDDEN_DIM}, epochs={N_EPOCHS})\n")

idx = torch.arange(N, device=device)
header = f"{'Epoch':>5}  {'Loss':>8}  {'Align':>8}  {'Var':>8}  {'Cov':>8}  "
header += f"{'MaskedFrac':>10}  {'Secs':>6}"
print(header)
print("-" * len(header))

t0 = time.perf_counter()
history: list[dict] = []

for epoch in range(1, N_EPOCHS + 1):
    perm = torch.randperm(N, device=device)
    ep_loss = ep_align = ep_var = ep_cov = 0.0
    ep_masked_frac = 0.0
    n_batches = 0

    for start in range(0, N, BATCH_SIZE):
        b_idx = perm[start : start + BATCH_SIZE]
        b_x   = X[b_idx]
        b_c   = coords[b_idx]

        # --- Spatial patch mask ---
        patch_mask = spatial_patch_mask(b_c, mask_ratio=MASK_RATIO,
                                        n_patches=N_PATCHES)
        b_x_ctx = b_x.clone()
        b_x_ctx[patch_mask] = 0.0
        masked_frac = patch_mask.float().mean().item()

        # --- Forward ---
        h_ctx  = online(b_x_ctx)
        h_pred = predictor(h_ctx)

        with torch.no_grad():
            h_tgt = ema(b_x)   # target always sees full features

        loss, components = jepa_loss(h_pred, h_tgt, weights)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(online.parameters()) + list(predictor.parameters()), 1.0
        )
        opt.step()

        # EMA update
        with torch.no_grad():
            for p_o, p_e in zip(online.parameters(), ema.parameters()):
                p_e.data.mul_(EMA_TAU).add_(p_o.data, alpha=1.0 - EMA_TAU)

        ep_loss        += components["jepa_total"]
        ep_align       += components["jepa_align"]
        ep_var         += components["jepa_variance"]
        ep_cov         += components["jepa_covariance"]
        ep_masked_frac += masked_frac
        n_batches      += 1

    n_batches = max(n_batches, 1)
    row = {
        "epoch":       epoch,
        "loss":        ep_loss / n_batches,
        "align":       ep_align / n_batches,
        "var":         ep_var / n_batches,
        "cov":         ep_cov / n_batches,
        "masked_frac": ep_masked_frac / n_batches,
        "secs":        time.perf_counter() - t0,
    }
    history.append(row)

    if epoch == 1 or epoch % 5 == 0 or epoch == N_EPOCHS:
        print(
            f"{epoch:>5}  {row['loss']:>8.4f}  {row['align']:>8.4f}  "
            f"{row['var']:>8.4f}  {row['cov']:>8.4f}  "
            f"{row['masked_frac']:>10.3f}  {row['secs']:>6.1f}s"
        )

# ---------------------------------------------------------------------------
# Summary diagnostics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DIAGNOSTICS")
print("=" * 60)

first  = history[0]
last   = history[-1]
best   = min(history, key=lambda r: r["loss"])

loss_drop_pct = 100 * (first["loss"] - last["loss"]) / (abs(first["loss"]) + 1e-9)
print(f"  Loss:  {first['loss']:.4f}  →  {last['loss']:.4f}  "
      f"({loss_drop_pct:+.1f}%)")
print(f"  Best:  {best['loss']:.4f}  at epoch {best['epoch']}")
print(f"  Align: {first['align']:.4f}  →  {last['align']:.4f}  "
      f"({'improving' if last['align'] < first['align'] else 'not improving'})")
print(f"  Var:   {first['var']:.4f}  →  {last['var']:.4f}  "
      f"({'collapsing!' if last['var'] > 0.5 else 'stable'})")
print(f"  Masked fraction (avg): {last['masked_frac']:.1%}  "
      f"(target {MASK_RATIO:.0%})")

# Collapse check: measure std of final EMA embeddings on a held-out sample
ema.eval()
with torch.no_grad():
    sample_idx = torch.randperm(N, device=device)[:2048]
    h_sample = ema(X[sample_idx])
    per_feat_std = h_sample.std(dim=0)
    dim_collapse_count = int((per_feat_std < 0.01).sum().item())

print(f"\n  Embedding std (mean across {HIDDEN_DIM} dims): "
      f"{per_feat_std.mean().item():.4f}")
print(f"  Collapsed dims (<0.01 std): {dim_collapse_count}/{HIDDEN_DIM}  "
      f"({'⚠ collapse detected' if dim_collapse_count > HIDDEN_DIM // 4 else '✓ healthy'})")

# Patch coverage vs target
print(f"\n  Spatial patch coverage: targeting {MASK_RATIO:.0%}, "
      f"achieved {last['masked_frac']:.1%} per batch on average")
if abs(last["masked_frac"] - MASK_RATIO) > 0.15:
    print("  ⚠  Coverage diverged from target — consider tuning n_patches")
else:
    print("  ✓  Coverage close to target")

print(f"\n  Total time: {last['secs']:.1f}s  "
      f"({last['secs']/N_EPOCHS:.2f}s/epoch)\n")
