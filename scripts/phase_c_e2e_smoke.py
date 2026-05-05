"""Phase C end-to-end smoke test on the Brown UHI dataset.

Wires together every new Phase-C piece in a single script that runs in
a few minutes on a laptop:

  1. Load the brown4.csv UHI dataset (54k cells, sub-sample to 800 for speed).
  2. Fit a Stage-0 Matérn correlogram per predictor (κ posterior).
  3. Assemble a ``KernelField`` from those fits with synthetic per-predictor
     anisotropy (long axis along x for the impervious surfaces).
  4. Fit a kernel-field-aware ``BayesianSpatialCATE`` for one treatment
     (``Pct_Impervious`` → AAT_z).
  5. Build a causal PDP (C-2) with NUTS credible bands.
  6. Compute a CATE-vs-GWR divergence audit (C-3).
  7. Build the scenario routing audit (C-4 + anisotropic-frame metadata).
  8. Print a one-line health summary per stage.

Intended as a hands-on validation of the Bucket-C work, not a replacement
for the full SPARC pipeline.  Run with::

    python scripts/phase_c_e2e_smoke.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sparc.causal.causal_pdp import causal_pdp                        # noqa: E402
from sparc.causal.divergence_audit import cate_vs_gwr_divergence       # noqa: E402
from sparc.causal.spatial_cate import BayesianSpatialCATE              # noqa: E402
from sparc.interventions.scenario_routing_audit import (              # noqa: E402
    build_scenario_routing_audit,
)
from sparc.models.gwr import GWRModel                                  # noqa: E402
from sparc.models.kernel_field import KernelField, PredictorKernel     # noqa: E402


def _banner(msg: str) -> None:
    print(f"\n=== {msg} ===")


def _fit_matern_quick(coords: np.ndarray, values: np.ndarray) -> dict:
    """Lightweight Matérn fit using sparc.run.correlogram_matern_fit.

    Returns ``{"kappa": float, "samples": [..]}``.  Falls back to a
    coarse heuristic when the NUTS path is unavailable.
    """
    from sparc.run.correlogram_matern_fit import fit_matern

    # Cheap empirical correlogram on a small grid
    n = len(values)
    n_pairs = min(2000, n * (n - 1) // 2)
    rng = np.random.default_rng(0)
    i_idx = rng.integers(0, n, size=n_pairs)
    j_idx = rng.integers(0, n, size=n_pairs)
    mask = i_idx != j_idx
    i_idx = i_idx[mask]; j_idx = j_idx[mask]
    d = np.linalg.norm(coords[i_idx] - coords[j_idx], axis=1)
    z = (values - values.mean()) / (values.std() + 1e-12)
    prod = z[i_idx] * z[j_idx]
    n_lags = 12
    lag_edges = np.linspace(0, np.percentile(d, 95), n_lags + 1)
    lag_centers = 0.5 * (lag_edges[1:] + lag_edges[:-1])
    morans = np.zeros(n_lags)
    for k in range(n_lags):
        sel = (d >= lag_edges[k]) & (d < lag_edges[k + 1])
        if sel.sum() > 5:
            morans[k] = float(np.mean(prod[sel]))
    try:
        res = fit_matern(
            lag_centers, morans, method="bayes",
            n_samples=200, n_warmup=200, n_chains=1, seed=0,
        )
        return {
            "kappa": float(res.kappa_mean),
            "samples": [float(x) for x in res.kappa_samples.tolist()],
            "nu": float(res.nu),
        }
    except Exception as exc:  # noqa: BLE001
        # Heuristic fallback: log-linear slope
        valid = (morans > 1e-3) & (lag_centers > 0)
        if valid.sum() < 3:
            return {"kappa": 1.0 / float(np.median(d)), "samples": [], "nu": 1.5}
        slope, _ = np.polyfit(lag_centers[valid], np.log(morans[valid]), 1)
        return {
            "kappa": float(max(-slope, 1e-6)),
            "samples": [],
            "nu": 1.5,
            "fallback_reason": str(exc),
        }


def main() -> int:
    csv_path = REPO / "my_project" / "brown4.csv"
    if not csv_path.exists():
        print(f"[skip] brown4.csv not found at {csv_path}")
        return 0

    _banner("Phase C E2E — load + sub-sample brown4 UHI")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")
    rng = np.random.default_rng(42)
    sample = df.sample(n=min(800, len(df)), random_state=42).reset_index(drop=True)
    print(f"  Sub-sampled to {len(sample):,} rows for the smoke test")

    treatment = "Pct_Impervious"
    outcome = "AAT_z"
    confounders = ["Pct_Canopy", "NDVI", "Albedo", "Elevation_m",
                    "Distance_from_water_m"]
    coord_cols = ["POINT_X", "POINT_Y"]
    coords = sample[coord_cols].values.astype(np.float64)

    _banner("Step 1: Stage-0 Matérn fits per predictor")
    t0 = time.time()
    per_var_fits: dict[str, dict] = {}
    for var in [treatment, *confounders]:
        if var not in sample.columns:
            continue
        fit = _fit_matern_quick(coords, sample[var].values.astype(np.float64))
        per_var_fits[var] = {"kappa": fit}
        print(f"  κ({var}) = {fit['kappa']:.4g}, ν = {fit.get('nu', 1.5)}")
    print(f"  ({time.time() - t0:.1f}s)")

    _banner("Step 2: Assemble KernelField with synthetic anisotropy")
    # Inject anisotropy on impervious-like predictors (long axis along x)
    aniso_predictors = {treatment, "Pct_Canopy", "NDVI"}
    predictors: list[PredictorKernel] = []
    for var, fit in per_var_fits.items():
        kappa = fit["kappa"]["kappa"]
        nu = fit["kappa"].get("nu", 1.5)
        if var in aniso_predictors:
            predictors.append(PredictorKernel(
                name=var, kappa=float(kappa), nu=float(nu),
                kappa_x=float(kappa) * 0.5, kappa_y=float(kappa) * 2.0,
                theta_rad=0.0,
                kappa_posterior_samples=fit["kappa"].get("samples"),
                bandwidth_to_outcome=float(1.0 / max(kappa, 1e-9)),
            ))
        else:
            predictors.append(PredictorKernel(
                name=var, kappa=float(kappa), nu=float(nu),
                kappa_posterior_samples=fit["kappa"].get("samples"),
                bandwidth_to_outcome=float(1.0 / max(kappa, 1e-9)),
            ))

    var_names = [outcome] + list(per_var_fits.keys())
    n_v = len(var_names)
    range_matrix: list[list[float | None]] = [
        [None] * n_v for _ in range(n_v)
    ]
    for p in predictors:
        i = var_names.index(outcome)
        j = var_names.index(p.name)
        range_matrix[i][j] = p.bandwidth_to_outcome
        range_matrix[j][i] = p.bandwidth_to_outcome

    kf = KernelField(
        outcome_name=outcome,
        predictors=predictors,
        variable_names=var_names,
        range_matrix=range_matrix,
    )
    aniso_count = sum(1 for p in kf.predictors if p.is_anisotropic)
    print(f"  KernelField outcome={kf.outcome_name}, "
          f"{len(kf.predictors)} predictors, {aniso_count} anisotropic")

    _banner("Step 3: Bayesian-spatial-CATE (kernel-field-aware) for impervious")
    t0 = time.time()
    bayes_est = BayesianSpatialCATE(
        config={"pipeline": {"random_seed": 42},
                 "causal": {"bayesian_cate": {
                     "n_samples": 200, "n_warmup": 200,
                     "n_chains": 1, "max_tree_depth": 6,
                 }}},
        n_features=16, n_samples=200, n_warmup=200,
        kernel_lengthscale=0.20, kernel_field=kf,
    )
    bayes_est.estimate(
        sample, treatment=treatment, outcome=outcome,
        confounders=confounders, coord_cols=coord_cols,
    )
    audit_b = bayes_est._kernel_field_applied.get(treatment, {})
    cate = bayes_est.cate_estimates[treatment]
    print(f"  CATE: mean={cate.mean():+.4f}, std={cate.std():.4f}, "
          f"range=[{cate.min():+.4f}, {cate.max():+.4f}]")
    print(f"  Kernel field applied: warp={audit_b.get('applied_warp')}, "
          f"effective_lengthscale={audit_b.get('effective_lengthscale'):.4f}")
    print(f"  ({time.time() - t0:.1f}s)")

    _banner("Step 4: Causal PDP with NUTS credible bands")
    pdp = causal_pdp(bayes_est, treatment, sample[treatment].values, n_dose=15)
    print(f"  PDP source={pdp.source}, dose grid {pdp.dose_grid.size} pts")
    print(f"  Peak |slope|={pdp.peak_marginal_slope:.4f}, "
          f"saturation_dose={pdp.saturation_dose}")
    width = pdp.response_hdi_hi - pdp.response_hdi_lo
    print(f"  Mean 89% HDI width: {width.mean():.4f}")

    _banner("Step 5: Stage-2 GWR for divergence audit")
    t0 = time.time()
    X = sample[[treatment]].values.astype(np.float64)
    y = sample[outcome].values.astype(np.float64)
    gwr = GWRModel(kernel_field=kf, alpha=0.05,
                    min_points=30, use_constrained_regression=False)
    gwr.feature_names_ = [treatment]
    gwr.fit(X, y, coords=coords)
    gwr_beta = gwr.coefficients_[:, 0]
    print(f"  GWR β (impervious): mean={gwr_beta.mean():+.4f}, "
          f"std={gwr_beta.std():.4f}")
    print(f"  ({time.time() - t0:.1f}s)")

    _banner("Step 6: CATE-vs-GWR divergence audit")
    rep = cate_vs_gwr_divergence(gwr_beta, cate, treatment=treatment,
                                    threshold_sigma=2.0)
    print(f"  Pearson(β, τ) = {rep.pearson_correlation:+.3f}, "
          f"sign-agree = {rep.sign_agreement_rate:.2%}")
    print(f"  Mean |Δ| = {rep.mean_abs_divergence:.4f}, "
          f"flagged {rep.flagged_count}/{rep.cell_count} "
          f"({rep.flagged_fraction:.1%}) at thr={rep.threshold_used:.4f}")

    _banner("Step 7: Scenario routing audit + anisotropic-frame metadata")
    cate_multipliers = {treatment: cate / max(abs(cate.mean()), 1e-6)}
    audit = build_scenario_routing_audit(
        scenario_variables=[treatment, "Pct_Canopy", "NDVI"],
        cate_multipliers=cate_multipliers,
        n_cells=len(sample),
        kernel_field=kf,
    )
    payload = audit.to_payload()
    print(f"  Routing: {json.dumps(payload['summary'])}")
    print(f"  Anisotropy entries: {len(payload['anisotropy'])}")
    for r in payload["anisotropy"][:3]:
        if r["is_anisotropic"]:
            print(f"    {r['name']}: aspect={r['aspect_ratio']:.2f}, "
                  f"ecc={r['eccentricity']:.3f}, "
                  f"range_x={r['range_x']:.1f}m, range_y={r['range_y']:.1f}m")
        else:
            print(f"    {r['name']}: isotropic, "
                  f"range={r.get('range_x') or r.get('bandwidth_to_outcome')}")

    _banner("Phase C E2E smoke complete")
    print("  All seven steps finished without error.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
