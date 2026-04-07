"""Full V2 pipeline run on brown4.csv — real data."""
import numpy as np
import pandas as pd
import logging
import tempfile
import os
import sys
import json
import time
from sklearn.model_selection import KFold

# Enable logging from sparc modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


def main():
    # ----------------------------------------------------------------
    # Parse CLI:  --full  = all 54K rows,  default = 2000-row sample
    # ----------------------------------------------------------------
    full_mode = "--full" in sys.argv
    n_sample = None if full_mode else 2000

    print("=" * 64)
    print("  V2 SPARC Pipeline — Brown University UHI Dataset")
    print("=" * 64)

    # Load data
    df = pd.read_csv("examples/brown4.csv")
    total = len(df)

    if n_sample and n_sample < total:
        rng = np.random.default_rng(42)
        idx = rng.choice(total, size=n_sample, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        print(f"\nData: {len(df):,} rows (sampled from {total:,})")
    else:
        print(f"\nData: {len(df):,} rows (full dataset)")

    FEATURES = [
        "Pct_Canopy", "Pct_Impervious", "NDVI",
        "Albedo", "Elevation_m", "Distance_from_water_m",
    ]
    TARGET = "AAT_z"
    COORD_COLS = ["POINT_X", "POINT_Y"]

    X = df[FEATURES].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)
    coords = df[COORD_COLS].values.astype(np.float32)

    print(f"Target: {TARGET}  mean={y.mean():.2f}  std={y.std():.2f}")
    print(f"Features: {FEATURES}")

    # CV folds
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = [(tr, te) for tr, te in kf.split(X)]
    print(f"\nCV: {n_folds}-fold  (test sizes: {[len(f[1]) for f in folds]})")

    # ------------------------------------------------------------------
    # Config — production-like settings
    # ------------------------------------------------------------------
    n_epochs = 150 if full_mode else 25
    swa_epochs = 30 if full_mode else 5
    pretrain_epochs = 50 if full_mode else 10
    config = {
        "models": {
            "meta_learner": "neural",
            "neural": {
                "hidden_dim": 64,
                "dropout": 0.15,
                "n_heads": 4,
                "max_neighbors": 16,
                "siren_omega": 30.0,
                "sinusoidal_frequencies": 16,
                "exceedance_thresholds": [88.0, 90.0, 92.0],
                "n_base_models": 3,
            },
        },
        "training": {
            "n_epochs": n_epochs,
            "swa_epochs": swa_epochs,
            "swa_lr_max": 5e-4,
            "batch_size": 4096,  # not used in full-batch, kept for compat
            "learning_rate": 1e-3,
            "warmup_epochs": 10 if full_mode else 5,
            "ramp_epochs": 30 if full_mode else 20,
            "pretrain_epochs": pretrain_epochs,
            "lambda_physics": 0.05,
            "lambda_smooth_pred": 0.02,
            "lambda_smooth_alpha": 0.005,
            "lambda_alpha_prior": 1.0,
            "lambda_neighbor": 0.1,
            "feature_names": FEATURES,
        },
        "optimization": {"clip_norm": 1.0, "run_cma_es": False},
        "process_rate": {
            "enabled": True,
            "name": "thermal_admittance",
            "units": "W/m2/K",
            "bounds": [0.5, 12.0],
            "prior_mean": 4.0,
            "inputs": ["Pct_Impervious", "Albedo", "Elevation_m"],
        },
        "variables": {
            "target": TARGET,
            "coords": COORD_COLS,
        },
        "causal": {
            "inference": "bayesian",
            "mc3": {
                "n_iterations": 5000 if full_mode else 300,
                "n_chains": 4 if full_mode else 3,
                "burnin_fraction": 0.3,
            },
            "nuts": {
                "n_samples": 1000 if full_mode else 100,
                "n_warmup": 500 if full_mode else 50,
                "target_accept_rate": 0.80,
                "max_rows": 10000 if full_mode else 2000,
            },
        },
    }

    out_dir = os.path.join("examples", "providence_ri_uhi", "v2_run")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutput: {os.path.abspath(out_dir)}")

    # ==================================================================
    # STEP 1 — V2 Neural Meta-Learner Training
    # ==================================================================
    print("\n" + "=" * 64)
    print(f"  STEP 1: V2 Neural Meta-Learner  ({pretrain_epochs}pre+{n_epochs}+{swa_epochs}SWA epochs, {len(df):,} rows)")
    print(f"    Architecture: Raw Features -> Surrogates -> Meta-Learner (end-to-end)")
    print(f"    Surrogates: DifferentiableGWR, GWRF, GGPGAM")
    print(f"    Optimizer:  Per-component AdamW + warmup scheduler")
    print(f"    Loss terms: MSE + Exceedance + Physics + Smooth + AlphaSmooth")
    print(f"                + AlphaPrior + Neighborhood  (7-term, fully differentiable)")
    print("=" * 64)

    from sparc.run.v2_neural_training import train_neural_meta

    t0 = time.perf_counter()
    result = train_neural_meta(
        y=y,
        coords=coords,
        feature_matrix=X,
        feature_names=FEATURES,
        folds=folds,
        config=config,
        output_dir=out_dir,
    )
    t1 = time.perf_counter()

    r2 = result["metrics"]["r2"]
    rmse = result["metrics"]["rmse"]
    unc = result["oof_uncertainty"]
    print(f"\n  OOF R^2:          {r2:.4f}")
    print(f"  OOF RMSE:         {rmse:.4f}")
    print(f"  Uncertainty:      mean={unc.mean():.4f}  std={unc.std():.4f}")
    print(f"  Training time:    {t1 - t0:.1f}s")

    v2_dir = os.path.join(out_dir, "v2_neural")
    if os.path.isdir(v2_dir):
        artifacts = os.listdir(v2_dir)
        print(f"  Artifacts ({len(artifacts)}): {artifacts}")

    # Show surrogate info
    if result.get("surrogates"):
        print(f"  Surrogates:       {list(result['surrogates'].keys())}")

    # ==================================================================
    # STEP 2 — V2 Bayesian Causal (MC³ + NUTS)
    # ==================================================================
    print("\n" + "=" * 64)
    print("  STEP 2: Bayesian Causal Discovery (MC³ + NUTS)")
    print("=" * 64)

    from sparc.run.v2_bayesian_causal import run_bayesian_causal

    dag_def = {
        "nodes": [
            {"name": "Pct_Canopy", "type": "treatment"},
            {"name": "Pct_Impervious", "type": "treatment"},
            {"name": "NDVI", "type": "mediator"},
            {"name": "Albedo", "type": "treatment"},
            {"name": "Elevation_m", "type": "confounder"},
            {"name": "Distance_from_water_m", "type": "confounder"},
            {"name": "AAT_z", "type": "outcome"},
        ],
        "edges": [
            {"parent": "Pct_Canopy", "child": "AAT_z"},
            {"parent": "Pct_Impervious", "child": "AAT_z"},
            {"parent": "NDVI", "child": "AAT_z"},
            {"parent": "Albedo", "child": "AAT_z"},
            {"parent": "Elevation_m", "child": "AAT_z"},
            {"parent": "Distance_from_water_m", "child": "AAT_z"},
            {"parent": "Pct_Canopy", "child": "NDVI"},
            {"parent": "Pct_Impervious", "child": "NDVI"},
            {"parent": "Pct_Impervious", "child": "Albedo"},
        ],
    }

    bayesian_dir = os.path.join(out_dir, "bayesian")
    # Pass normalisation stats so NUTS can denormalise neural baseline
    config["training"]["y_mean"] = result["meta_info"]["y_mean"]
    config["training"]["y_std"] = result["meta_info"]["y_std"]
    t2 = time.perf_counter()
    bayesian_result = run_bayesian_causal(
        data=df,
        dag_def=dag_def,
        neural_model=result["model"],
        config=config,
        output_dir=bayesian_dir,
    )
    t3 = time.perf_counter()

    mc3 = bayesian_result["mc3_summary"]
    print(f"\n  MC³ results:")
    print(f"    Accepted: {mc3['n_accepted']} / {mc3['n_total']} "
          f"({mc3['acceptance_rate']:.1%})")
    print(f"    Best BIC: {mc3['best_score']:.1f}")
    print(f"    Time:     {t3 - t2:.1f}s")

    if bayesian_result.get("nuts_results"):
        nuts = bayesian_result["nuts_results"]
        print(f"\n  NUTS results:")
        print(f"    Acceptance:  {nuts['acceptance_rate']:.1%}")
        print(f"    Divergences: {nuts['n_divergences']}")
        print(f"    Treatment effects:")
        for t, bm, bs in zip(nuts["treatments"], nuts["beta_mean"], nuts["beta_std"]):
            print(f"      {t:25s}  beta = {bm:+.5f} +/- {bs:.5f}")

    if os.path.isdir(bayesian_dir):
        print(f"\n  Artifacts: {os.listdir(bayesian_dir)}")

    # ==================================================================
    # Summary
    # ==================================================================
    total_time = t3 - t0
    print("\n" + "=" * 64)
    print("  V2 PIPELINE COMPLETE")
    print("=" * 64)
    print(f"  Dataset:       {len(df):,} rows x {len(FEATURES)} features")
    print(f"  Architecture:  Surrogates + Neural Meta + ProcessRate + SWA")
    print(f"  Neural R^2:    {r2:.4f}")
    print(f"  Neural RMSE:   {rmse:.4f}")
    print(f"  MC³ accept:    {mc3['acceptance_rate']:.1%}")
    if bayesian_result.get("nuts_results"):
        print(f"  NUTS accept:   {nuts['acceptance_rate']:.1%}")
    print(f"  Total time:    {total_time:.1f}s")
    print(f"  Output:        {os.path.abspath(out_dir)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
