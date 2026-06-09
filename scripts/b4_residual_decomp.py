"""B4 -- Residual Feature Correlation Decomposition

Loads Philadelphia predictions.geoparquet (zero-shot or few-shot) and the
ground-truth labels, computes residuals per time window, then correlates those
residuals with every available land feature column.

This identifies which spatial predictors the model is systematically missing --
they become the highest-priority targets for PI-JEPA feature engineering.

Outputs:
  output/diagnostics/residual_feature_corr.csv
  output/diagnostics/residual_feature_corr.png  (ranked bar chart)

Usage:
    python scripts/b4_residual_decomp.py [--config CONFIG]
    python scripts/b4_residual_decomp.py --pred-col pred_morning_calibrated
"""
from __future__ import annotations
import argparse, sys, logging
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("b4_residual_decomp")

# Focus features per PI-JEPA spec §B4
FOCUS_FEATURES = [
    "svf", "bldg_height_mean", "pct_impervious", "ndvi", "albedo",
    "distance_water_m", "pct_canopy", "lst", "land_cover", "elevation_m",
    "cdc_svi", "slope_deg", "aspect_deg",
]

# Prediction/label column pairs per window
WINDOWS = [
    ("morning",  "pred_morning_calibrated",  "aat_morning"),
    ("midday",   "pred_midday_calibrated",   "aat_midday"),
    ("evening",  "pred_evening_calibrated",  "aat_evening"),
]


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/multicity_pilot.yml")
    p.add_argument("--output-dir", default="output/diagnostics")
    p.add_argument("--city", default="philadelphia_pa",
                   help="Holdout city slug (default: philadelphia_pa)")
    return p.parse_args()


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r robust to NaN (uses only rows where both are finite)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return float("nan")
    x = x[mask]; y = y[mask]
    x = x - x.mean(); y = y - y.mean()
    denom = np.std(x) * np.std(y)
    if denom < 1e-10:
        return float("nan")
    return float(np.mean(x * y) / denom)


def main():
    args = _parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as fh:
        pilot_cfg = yaml.safe_load(fh)

    mc = pilot_cfg["multicity"]
    output_root = Path(mc["output_dir"])
    diag_dir = Path(args.output_dir)
    diag_dir.mkdir(parents=True, exist_ok=True)

    city_dir = output_root / args.city
    pred_path = city_dir / "predictions.geoparquet"
    data_path = city_dir / "data.geoparquet"

    if not pred_path.exists():
        log.error("Predictions not found: %s", pred_path)
        log.error("Run Phase 3 LOO eval first.")
        sys.exit(1)

    if not data_path.exists():
        log.error("City data not found: %s", data_path)
        sys.exit(1)

    import geopandas as gpd
    log.info("Loading predictions from %s", pred_path)
    pred_gdf = gpd.read_parquet(str(pred_path))
    log.info("Loading city data from %s", data_path)
    data_gdf = gpd.read_parquet(str(data_path))

    log.info("predictions shape: %s  cols: %s", pred_gdf.shape,
             [c for c in pred_gdf.columns if "pred" in c or "aat" in c])
    log.info("data shape: %s", data_gdf.shape)

    # Merge on geometry index (both should be aligned by construction)
    if len(pred_gdf) != len(data_gdf):
        log.warning("Row count mismatch: predictions=%d data=%d -- using inner index merge",
                    len(pred_gdf), len(data_gdf))
        merged = pred_gdf.join(data_gdf.drop(columns=["geometry"], errors="ignore"),
                               how="inner", rsuffix="_data")
    else:
        merged = pred_gdf.copy()
        for col in data_gdf.columns:
            if col not in merged.columns:
                merged[col] = data_gdf[col].values

    log.info("Merged shape: %s", merged.shape)

    # Identify available feature columns (skip geometry/pred/label cols)
    skip_prefixes = ("pred_", "aat_", "geometry", "lat", "lon", "grid_")
    feature_cols = [c for c in merged.columns
                    if not any(c.startswith(p) for p in skip_prefixes)
                    and merged[c].dtype in (np.float32, np.float64, float)
                    or (merged[c].dtype.kind in ("f", "i") and c not in ("aat_morning", "aat_midday", "aat_evening"))]
    # Exclude aat label columns explicitly
    feature_cols = [c for c in feature_cols if not c.startswith("aat_")]
    log.info("Feature columns to correlate: %d", len(feature_cols))

    # Compute residuals and correlations
    results = []
    for window_name, pred_col, label_col in WINDOWS:
        if pred_col not in merged.columns:
            log.warning("Window %s: pred col '%s' not in predictions -- skipping", window_name, pred_col)
            continue
        if label_col not in merged.columns:
            log.warning("Window %s: label col '%s' not in data -- skipping", window_name, label_col)
            continue

        pred = merged[pred_col].values.astype(np.float64)
        label = merged[label_col].values.astype(np.float64)
        residual = pred - label  # positive = model over-predicts

        n_valid = int(np.isfinite(residual).sum())
        bias = float(np.nanmean(residual))
        rmse = float(np.sqrt(np.nanmean(residual ** 2)))
        log.info("Window %s: bias=%.3f RMSE=%.3f n=%d", window_name, bias, rmse, n_valid)

        for feat in feature_cols:
            if feat not in merged.columns:
                continue
            x = merged[feat].values.astype(np.float64)
            r = _pearson_r(x, residual)
            is_focus = feat in FOCUS_FEATURES
            results.append({
                "window": window_name,
                "feature": feat,
                "pearson_r": round(r, 5) if r == r else None,
                "abs_r": round(abs(r), 5) if r == r else 0.0,
                "bias_window": round(bias, 3),
                "rmse_window": round(rmse, 3),
                "is_focus": is_focus,
            })

    if not results:
        log.error("No residual correlations computed.")
        sys.exit(1)

    # Sort by abs_r within each window
    import csv
    csv_path = diag_dir / "residual_feature_corr.csv"
    fieldnames = ["window", "feature", "pearson_r", "abs_r", "bias_window",
                  "rmse_window", "is_focus"]
    results_sorted = sorted(results, key=lambda r: (r["window"], -(r["abs_r"] or 0)))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_sorted)
    log.info("Residual correlation CSV written --> %s", csv_path)

    # Print top features per window
    print()
    print("=" * 70)
    print("  B4 Residual Feature Correlation (Philadelphia zero-shot)")
    print("=" * 70)
    for window_name, _, _ in WINDOWS:
        wrows = [r for r in results_sorted if r["window"] == window_name]
        if not wrows:
            continue
        bias_w = wrows[0]["bias_window"]
        rmse_w = wrows[0]["rmse_window"]
        print(f"\n  Window: {window_name}  (bias={bias_w:+.2f}F  RMSE={rmse_w:.2f}F)")
        print(f"  {'Feature':<25s} {'Pearson_r':>10s}  {'Focus?':>6s}")
        print("  " + "-" * 48)
        for row in wrows[:20]:
            r_str = f"{row['pearson_r']:>+.4f}" if row["pearson_r"] is not None else "   N/A"
            focus_str = "*" if row["is_focus"] else " "
            print(f"  {row['feature']:<25s} {r_str:>10s}  {focus_str:>6s}")

    print()
    print("  (* = focus variable per PI-JEPA spec §B4)")
    print()

    # Identify highest-signal focus features (mean |r| across windows)
    focus_results = [r for r in results if r["is_focus"] and r["pearson_r"] is not None]
    if focus_results:
        from collections import defaultdict
        feat_r: dict[str, list[float]] = defaultdict(list)
        for r in focus_results:
            feat_r[r["feature"]].append(abs(r["pearson_r"]))
        feat_mean_r = {f: np.mean(vs) for f, vs in feat_r.items()}
        ranked = sorted(feat_mean_r.items(), key=lambda x: -x[1])

        print("  Top focus features by mean |Pearson r| across windows:")
        for feat, mr in ranked[:10]:
            print(f"    {feat:<25s}  mean|r|={mr:.4f}")

    print(f"\n  CSV: {csv_path}")
    print("=" * 70)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        windows_to_plot = [w for w, _, _ in WINDOWS
                           if any(r["window"] == w for r in results_sorted)]
        fig, axes = plt.subplots(1, len(windows_to_plot), figsize=(5 * len(windows_to_plot), 6),
                                 sharey=False)
        if len(windows_to_plot) == 1:
            axes = [axes]

        for ax, window_name in zip(axes, windows_to_plot):
            wrows = [r for r in results_sorted if r["window"] == window_name][:20]
            feats = [r["feature"] for r in wrows]
            rs = [r["pearson_r"] or 0.0 for r in wrows]
            colors = ["#e74c3c" if r["is_focus"] else "#3498db" for r in wrows]
            bars = ax.barh(range(len(feats)), rs, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_yticks(range(len(feats)))
            ax.set_yticklabels(feats, fontsize=7)
            ax.axvline(0, color="black", linewidth=0.7)
            ax.set_xlabel("Pearson r (residual vs feature)")
            ax.set_title(f"Window: {window_name}\n"
                         f"bias={wrows[0]['bias_window']:+.1f}F  "
                         f"RMSE={wrows[0]['rmse_window']:.1f}F")
            ax.invert_yaxis()

        fig.suptitle("B4 Residual Feature Correlations -- Philadelphia LOO", fontsize=10)
        plt.tight_layout()
        png_path = diag_dir / "residual_feature_corr.png"
        fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("B4 plot saved --> %s", png_path)
        print(f"  Plot: {png_path}")
    except Exception as exc:
        log.warning("Matplotlib plot failed: %s", exc)


if __name__ == "__main__":
    main()
