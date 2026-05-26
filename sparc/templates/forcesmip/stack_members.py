#!/usr/bin/env python3
"""
stack_members.py — Stack all ForceSMIP members into a single cross-member dataset.

For each target variable (tos, tas, pr, psl), loads all 10 evaluation member CSVs,
merges the ensemble-mean forced truth, and outputs a single pooled CSV where
forcing indices (GMST_trend, AMV_index, IPO_index) vary across rows.

This enables SPARC to learn spatially-varying sensitivity coefficients for each
forcing, since within-member forcings are constant but across-member they differ.

Usage
-----
    python stack_members.py --variables tos tas pr psl
    python stack_members.py --variables tos --members 1A 1B 1C

Outputs
-------
    data/forcesmip_{var}_pooled.csv   — Stacked data ready for SPARC
    pooled_{var}_project.yml          — SPARC project config for pooled run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

ALL_MEMBERS = ["1A", "1B", "1C", "1D", "1E", "1F", "1G", "1H", "1I", "1J"]
DEFAULT_VARS = ["tos", "tas", "pr", "psl"]

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

PRIORS_MAP = {
    "tos": "physics/forcesmip_priors_tos.yml",
    "tas": "physics/forcesmip_priors_tas.yml",
    "pr": "physics/forcesmip_priors_pr.yml",
    "psl": "physics/forcesmip_priors_psl.yml",
}


def stack_variable(var: str, members: list[str]) -> pd.DataFrame:
    """Stack all members for one variable into a single DataFrame."""

    frames = []
    for member in members:
        raw_path = DATA_DIR / f"forcesmip_{var}_{member}.csv"
        if not raw_path.exists():
            print(f"  [SKIP] {raw_path.name} not found")
            continue

        df = pd.read_csv(raw_path)
        df["member"] = member

        # Merge ensemble-mean truth if available
        em_path = DATA_DIR / f"forcesmip_ensmean_{var}_{member}.csv"
        if em_path.exists():
            em = pd.read_csv(em_path)[["grid_cell_id", f"{var}_trend_forced"]]
            df = df.merge(em, on="grid_cell_id", how="left")
        else:
            df[f"{var}_trend_forced"] = float("nan")

        frames.append(df)
        print(f"  {var}_{member}: {len(df)} cells, "
              f"GMST={df['GMST_trend'].iloc[0]:.4f}, "
              f"AMV={df['AMV_index'].iloc[0]:.4f}, "
              f"IPO={df['IPO_index'].iloc[0]:.4f}")

    if not frames:
        raise FileNotFoundError(f"No data files found for variable '{var}'")

    pooled = pd.concat(frames, ignore_index=True)

    # Create unique row ID (member_gridcell) since grid_cell_id repeats
    pooled["row_id"] = pooled["member"] + "_" + pooled["grid_cell_id"].astype(str)

    # Encode member as numeric for models that need it
    member_map = {m: i for i, m in enumerate(sorted(pooled["member"].unique()))}
    pooled["member_id"] = pooled["member"].map(member_map)

    return pooled


def generate_pooled_project_yml(var: str, pooled_csv_name: str) -> dict:
    """Generate a SPARC project.yml config for the pooled dataset."""

    # Load the base project.yml to inherit settings
    base_path = SCRIPT_DIR / "project.yml"
    with open(base_path) as f:
        base = yaml.safe_load(f)

    target_col = f"{var}_trend"

    cfg = {
        "project": {
            "name": f"ForceSMIP-SPARC-Pooled-{var}",
            "description": f"Cross-member pooled analysis for {var} — spatially-varying forcing sensitivity",
            "domain": "climate_forced_response",
            "version": "2.0",
            "response_units": base["project"].get("response_units", "trend units"),
        },
        "data": {
            "file_path": f"data/{pooled_csv_name}",
            "target_column": target_col,
            "identifier_column": "row_id",
            "coord_columns": ["lon", "lat"],
            "areas_of_interest_file": base["data"].get("areas_of_interest_file", ""),
        },
        "crs": base["crs"],
        "predictors": [
            # These now VARY across rows (the key change)
            "GMST_trend",
            "AMV_index",
            "IPO_index",
            # Additional climate indices
            "PDO_index",
            "NAO_index",
            # Low-frequency filtered versions (>10-yr Butterworth)
            "GMST_lowfreq",
            "AMV_lowfreq",
            "IPO_lowfreq",
            # Spatial covariates
            "lat_abs",
            "land_fraction",
            "ocean_basin",
            # Member identifier (captures model-specific baseline bias)
            "member_id",
        ],
        "physics": {
            "priors_file": PRIORS_MAP.get(var, base["physics"].get("priors_file", "")),
            "caps_file": str(base["physics"].get("caps_file", "")),
        },
        "causal": {
            "dag_file": "causal/forcesmip_dag_pooled.yml",
            "estimator": "dml",
            "dml_shrinkage": 0.3,
            "dml_cv_folds": 5,
            "dag_blend_weight": 0.7,
            "estimate_cate": True,
            "ipw_enabled": True,
            "doubly_robust": True,
            "dose_response": True,
            "bootstrap_n": 200,
            "actionable_variables": ["GMST_trend", "AMV_index", "IPO_index", "PDO_index", "NAO_index"],
            "fixed_variables": ["lat_abs", "land_fraction", "ocean_basin", "member_id",
                                "GMST_lowfreq", "AMV_lowfreq", "IPO_lowfreq"],
        },
        "scenarios": [
            {
                "name": "warming_1C",
                "variable": "GMST_trend",
                "direction": "increase",
                "increments": [0.5, 1.0, 1.5],
                "unit": "°C global mean",
            },
            {
                "name": "AMV_positive",
                "variable": "AMV_index",
                "direction": "increase",
                "increments": [0.02, 0.05],
                "unit": "index units",
            },
            {
                "name": "IPO_positive",
                "variable": "IPO_index",
                "direction": "increase",
                "increments": [0.03, 0.06],
                "unit": "index units",
            },
        ],
        "output": {
            "base_dir": f"output/forcesmip_{var}_pooled",
        },
        "models": base.get("models", {}),
        "pipeline": {
            "random_seed": 42,
            "n_spatial_folds": 5,
            "fast_mode": False,
            "overwrite_outputs": False,
            "run_mc_uncertainty": False,
            "skip_pdp": False,  # PDP is useful now — forcings vary!
        },
        "flags": base.get("flags", {}),
        "benchmark_metrics": {
            "enabled": True,
            "area_weighted": True,
            "lat_column": "lat",
        },
        "fingerprint": {
            "enabled": True,
            "pattern_source": "ensemble_mean",
            "eof_n_modes": 3,
            "blend_alpha": 0.3,
            "truth_column": f"{var}_trend_forced",
        },
    }

    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Stack ForceSMIP members into cross-member pooled datasets"
    )
    parser.add_argument("--variables", nargs="+", default=DEFAULT_VARS)
    parser.add_argument("--members", nargs="+", default=ALL_MEMBERS)
    args = parser.parse_args()

    for var in args.variables:
        print(f"\n{'='*60}")
        print(f"Stacking {var} — {len(args.members)} members")
        print(f"{'='*60}")

        pooled = stack_variable(var, args.members)

        # Drop constant-valued forcing columns (zero variance = useless)
        constant_cols = []
        forcing_cols = [
            "GHG_forcing_index", "aerosol_forcing_index",
            "volcanic_forcing_index", "solar_forcing_index", "ENSO_detrended",
        ]
        for col in forcing_cols:
            if col in pooled.columns and pooled[col].std() < 1e-10:
                constant_cols.append(col)

        if constant_cols:
            print(f"\n  Dropping constant columns: {constant_cols}")
            pooled = pooled.drop(columns=constant_cols)

        # Summary
        target_col = f"{var}_trend"
        truth_col = f"{var}_trend_forced"
        print(f"\n  Pooled shape: {pooled.shape}")
        print(f"  Members: {sorted(pooled['member'].unique())}")
        print(f"  Target ({target_col}): mean={pooled[target_col].mean():.4f}, "
              f"std={pooled[target_col].std():.4f}")
        if truth_col in pooled.columns:
            valid_truth = pooled[truth_col].dropna()
            print(f"  Truth ({truth_col}): mean={valid_truth.mean():.4f}, "
                  f"std={valid_truth.std():.4f}, n={len(valid_truth)}")

        # Show forcing variation (the whole point)
        print(f"\n  Cross-member forcing variation:")
        for col in ["GMST_trend", "AMV_index", "IPO_index"]:
            if col in pooled.columns:
                print(f"    {col:20s}  range: {pooled[col].min():.4f} to "
                      f"{pooled[col].max():.4f}  std: {pooled[col].std():.4f}")

        # Save pooled CSV
        csv_name = f"forcesmip_{var}_pooled.csv"
        csv_path = DATA_DIR / csv_name
        pooled.to_csv(csv_path, index=False)
        print(f"\n  Saved: {csv_path} "
              f"({csv_path.stat().st_size / 1024 / 1024:.1f} MB)")

        # Generate and save project.yml
        cfg = generate_pooled_project_yml(var, csv_name)
        yml_path = SCRIPT_DIR / f"pooled_{var}_project.yml"
        with open(yml_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"  Saved: {yml_path}")

    print(f"\n{'='*60}")
    print("Done. Run pooled analysis with:")
    for var in args.variables:
        print(f"  python -m sparc run --project templates/forcesmip/pooled_{var}_project.yml")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
