#!/usr/bin/env python3
"""A3 Causal Deconfounding Diagnostics.

Measures whether the PI-JEPA trunk captures spatially-autocorrelated
structure in the treatment variables — if so, residualising on the trunk
embeddings reduces confounding for Stage 3 DML.

Two metrics are reported for each treatment column (pct_impervious,
pct_canopy, albedo, ndvi) before and after trunk-based residualisation:

  1. Moran's I  — spatial autocorrelation of the treatment / residual.
                  Lower I after = less spatial confounding remaining.
  2. R² of Ridge regression on PCA-16 of trunk embeddings — how much of
     the treatment's variance is captured by the trunk latent space.

Usage::

    python scripts/a3_causal_diagnostics.py

Outputs::
    output/a3_diagnostics.csv   — per-city per-treatment metrics
    output/a3_diagnostics.txt   — human-readable summary table
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  a3_diagnostics -- %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("a3_diagnostics")

# Treatment columns to analyse (subset present in data.geoparquet)
TREATMENT_COLS = ["pct_impervious", "pct_canopy", "albedo", "ndvi", "ndbi"]

# Max pixels per city to use (subsample to keep diagnostics fast)
MAX_N = 10_000

# PCA dims for residualisation
PCA_DIMS = 16

# Number of nearest neighbours for Moran's I spatial weights
MORANS_K = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_morans_i(values, coords, k=MORANS_K) -> float:
    """Compute Moran's I using sparse k-NN spatial weights (row-normalised).

    Returns I in [-1, 1].  NOTE: requires spatially-contiguous samples
    (random subsamples destroy spatial adjacency → I ≈ 0 regardless of
    true autocorrelation).  Use ``spatial_subsample`` for valid results.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    n = len(values)
    if n < k + 2:
        return float("nan")

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
    _, idxs = nbrs.kneighbors(coords)
    # idxs[:,0] is self — use columns 1..k

    z = values - values.mean()
    z_std = z.std()
    if z_std < 1e-8:
        return 0.0
    z = z / z_std

    # W is row-normalised (each row sums to 1), so S0 = n.
    # Wz[i] = mean of z[neighbours of i]
    Wz = np.mean(z[idxs[:, 1:]], axis=1)  # (N,)

    numerator   = float(np.dot(z, Wz))
    denominator = float(np.dot(z, z))
    S0 = float(n)  # row-normalised → each row sum = 1, total = n
    if S0 < 1e-8 or denominator < 1e-8:
        return 0.0

    return (n / S0) * (numerator / denominator)


def spatial_tile_subsample(df, coords, tile_n: int = 4_000) -> tuple:
    """Select a spatially-contiguous tile of ~tile_n points for Moran's I.

    Finds the spatial centroid of the city and returns the tile_n nearest
    points.  Unlike random subsampling, this preserves local adjacency.
    """
    import numpy as np

    if len(df) <= tile_n:
        return df, coords

    cx, cy = coords[:, 0].mean(), coords[:, 1].mean()
    dists = np.hypot(coords[:, 0] - cx, coords[:, 1] - cy)
    idx = np.argsort(dists)[:tile_n]
    return df.iloc[idx].reset_index(drop=True), coords[idx]


def load_city_data(geoparquet_path: Path, max_n: int = MAX_N):
    """Load geoparquet, return (X_df, coords_np)."""
    import numpy as np

    try:
        import geopandas as gpd
        gdf = gpd.read_parquet(str(geoparquet_path))
    except Exception:
        import pandas as pd
        gdf = pd.read_parquet(str(geoparquet_path))

    if len(gdf) > max_n:
        gdf = gdf.sample(n=max_n, random_state=42).reset_index(drop=True)

    # Extract coords
    if hasattr(gdf, "geometry") and gdf.geometry is not None:
        try:
            coords = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
        except Exception:
            coords = np.column_stack([gdf["lon"].values, gdf["lat"].values]) if "lon" in gdf.columns else np.zeros((len(gdf), 2))
    elif "lon" in gdf.columns and "lat" in gdf.columns:
        coords = np.column_stack([gdf["lon"].values, gdf["lat"].values])
    else:
        coords = np.zeros((len(gdf), 2))

    return gdf, coords


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_diagnostics(output_root: Path, trunk_path: Path | None = None) -> list[dict]:
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    # Locate trunk
    if trunk_path is None:
        candidates = [
            output_root / "jepa_pretrained_trunk.pt",
            output_root / "cities" / "jepa_pretrained_trunk_t66_best.pt",
        ]
        for c in candidates:
            if c.exists():
                trunk_path = c
                break

    if trunk_path is None or not trunk_path.exists():
        log.error("No PI-JEPA trunk found.  Pass --trunk or run training first.")
        return []

    log.info("Loading trunk from %s", trunk_path)
    from sparc.causal.jepa_trunk_adapter import JEPATrunkAdapter
    adapter = JEPATrunkAdapter(str(trunk_path), device="cpu")
    feat_names = adapter.feature_names
    log.info("Trunk features (%d): %s", len(feat_names), feat_names[:6])

    cities_dir = output_root / "cities"
    if not cities_dir.exists():
        log.error("Cities directory not found: %s", cities_dir)
        return []

    city_dirs = sorted([d for d in cities_dir.iterdir() if d.is_dir()])
    log.info("Found %d city directories", len(city_dirs))

    rows = []
    for city_dir in city_dirs:
        parquet = city_dir / "data.geoparquet"
        if not parquet.exists():
            continue
        slug = city_dir.name
        log.info("Processing %s ...", slug)

        try:
            df, coords = load_city_data(parquet)
        except Exception as exc:
            log.warning("[%s] Failed to load: %s", slug, exc)
            continue

        # Columns available in this city
        available_treat = [c for c in TREATMENT_COLS if c in df.columns]
        if not available_treat:
            log.warning("[%s] No treatment columns found", slug)
            continue

        # Physics features for trunk encoding
        phys_cols = [c for c in feat_names if c in df.columns]
        if not phys_cols:
            log.warning("[%s] No physics features overlap with trunk features", slug)
            continue

        X_raw = df[phys_cols].values.astype(np.float32)
        X_raw = np.nan_to_num(X_raw, nan=0.0)

        # Encode → PCA-16
        h_np = adapter.encode_by_names(X_raw, phys_cols)  # (N, 256)
        n_components = min(PCA_DIMS, h_np.shape[1], len(df) - 1)
        pca = PCA(n_components=n_components, random_state=42)
        h_pca = pca.fit_transform(h_np)  # (N, PCA_DIMS)

        # Spatial tile for Moran's I (contiguous patch preserves adjacency)
        df_tile, coords_tile = spatial_tile_subsample(df, coords, tile_n=4_000)
        X_tile = df_tile[[c for c in phys_cols if c in df_tile.columns]].values.astype(np.float32)
        X_tile = np.nan_to_num(X_tile, nan=0.0)
        phys_tile = [c for c in phys_cols if c in df_tile.columns]
        h_tile    = adapter.encode_by_names(X_tile, phys_tile)
        pca_tile  = PCA(n_components=min(PCA_DIMS, h_tile.shape[1], len(df_tile) - 1), random_state=42)
        h_pca_tile = pca_tile.fit_transform(h_tile)

        ridge = Ridge(alpha=1.0, fit_intercept=True)

        for col in available_treat:
            # --- R² on random subsample (valid for measuring explanatory power) ---
            y = df[col].values.astype(np.float64)
            mask = np.isfinite(y)
            if mask.sum() < 20:
                continue

            y_clean = y[mask]
            h_clean = h_pca[mask]

            ridge.fit(h_clean, y_clean)
            y_pred = ridge.predict(h_clean)
            ss_res = float(np.sum((y_clean - y_pred) ** 2))
            ss_tot = float(np.sum((y_clean - y_clean.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0

            # --- Moran's I on spatial tile (valid: preserves adjacency) ---
            y_tile = df_tile[col].values.astype(np.float64) if col in df_tile.columns else np.full(len(df_tile), np.nan)
            mask_tile = np.isfinite(y_tile)
            if mask_tile.sum() < 20:
                mi_before = float("nan")
                mi_after  = float("nan")
                delta_mi  = float("nan")
            else:
                y_t = y_tile[mask_tile]
                h_t = h_pca_tile[mask_tile]
                c_t = coords_tile[mask_tile]

                mi_before = compute_morans_i(y_t, c_t)
                ridge_tile = Ridge(alpha=1.0, fit_intercept=True)
                ridge_tile.fit(h_t, y_t)
                resid_tile = y_t - ridge_tile.predict(h_t)
                mi_after  = compute_morans_i(resid_tile, c_t)
                delta_mi  = mi_before - mi_after

            rows.append({
                "city": slug,
                "treatment": col,
                "n_r2": int(mask.sum()),
                "n_mi": int(mask_tile.sum()) if mask_tile.sum() >= 20 else 0,
                "mi_before": round(float(mi_before), 4),
                "mi_after": round(float(mi_after), 4),
                "delta_mi": round(float(delta_mi), 4),
                "r2_trunk": round(float(r2), 4),
            })
            log.info(
                "  [%s] %s  R2=%.3f  Moran_I(tile): %.3f -> %.3f  (delta=%.3f)",
                slug, col, r2, mi_before, mi_after, delta_mi,
            )

    return rows


def print_summary(rows: list[dict]) -> None:
    import numpy as np

    if not rows:
        print("No results to summarise.")
        return

    treatments = sorted({r["treatment"] for r in rows})
    print("\n" + "=" * 80)
    print("  A3 Causal Deconfounding Diagnostics - Summary")
    print()
    print("  PRIMARY metric: R2 (trunk explains treatment variance)")
    print("  R2 > 0.10: trunk captures spatial structure -> residualisation reduces confounding")
    print()
    print("  SECONDARY metric: Moran I on spatial tile (contiguous 4K-pixel patch)")
    print("  Delta-MI > 0: residualisation reduced spatial autocorrelation of treatment")
    print("=" * 80)
    print(f"  {'Treatment':<22} {'R2 trunk':>10} {'MI before':>12} {'MI after':>10} {'Delta MI':>10}")
    print("  " + "-" * 67)

    for trt in treatments:
        sub = [r for r in rows if r["treatment"] == trt]
        r2   = np.nanmean([r["r2_trunk"] for r in sub])
        mi_b = np.nanmean([r["mi_before"] for r in sub if not np.isnan(r["mi_before"])] or [float("nan")])
        mi_a = np.nanmean([r["mi_after"]  for r in sub if not np.isnan(r["mi_after"])]  or [float("nan")])
        d    = np.nanmean([r["delta_mi"]  for r in sub if not np.isnan(r["delta_mi"])]  or [float("nan")])
        r2_flag = " <<<" if r2 > 0.3 else ""
        print(f"  {trt:<22} {r2:>10.4f} {mi_b:>12.4f} {mi_a:>10.4f} {d:>10.4f}{r2_flag}")

    print("=" * 80)
    overall_r2    = float(np.nanmean([r["r2_trunk"] for r in rows]))
    overall_delta = float(np.nanmean([r["delta_mi"] for r in rows if not np.isnan(r["delta_mi"])]))
    print(f"\n  Mean R2 (trunk explains treatment):          {overall_r2:.4f}")
    print(f"  Mean delta-MI across all cities+treatments:  {overall_delta:+.4f}")
    if overall_r2 > 0.25:
        print("\n  --> A3 RECOMMENDED: trunk explains >25% of treatment variance")
        print("      Residualising treatments on trunk PCA-16 will remove substantial")
        print("      spatially-structured confounding from Stage 3 DML.")
    elif overall_r2 > 0.10:
        print("\n  --> A3 MARGINAL: some treatment variance captured by trunk")
    else:
        print("\n  --> A3 limited benefit for this trunk checkpoint (R2 too low)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="A3 causal deconfounding diagnostics")
    parser.add_argument("--output-root", default=str(ROOT / "output"),
                        help="Root output directory (default: output/)")
    parser.add_argument("--trunk", default=None,
                        help="Path to jepa_pretrained_trunk.pt (auto-detected if omitted)")
    parser.add_argument("--max-n", type=int, default=MAX_N,
                        help=f"Max pixels per city to subsample (default: {MAX_N})")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    trunk_path  = Path(args.trunk) if args.trunk else None

    rows = run_diagnostics(output_root, trunk_path)

    if rows:
        out_csv = output_root / "a3_diagnostics.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        log.info("Results saved -> %s", out_csv)

        out_txt = output_root / "a3_diagnostics.txt"
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_summary(rows)
        summary = buf.getvalue()
        out_txt.write_text(summary, encoding="utf-8")
        sys.stdout.write(summary)
        log.info("Summary saved -> %s", out_txt)
    else:
        log.error("No results produced — check that training has run and trunk exists.")


if __name__ == "__main__":
    main()
