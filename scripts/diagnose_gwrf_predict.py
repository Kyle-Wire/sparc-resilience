"""
Feedback-loop script for diagnosing & verifying the GWRF.predict() crash.

Usage:
    python scripts/diagnose_gwrf_predict.py          # runs the timing harness
    python scripts/diagnose_gwrf_predict.py --check  # also checks real data for NaN/Inf

Phase 1 loop:  fit GWRF(subsample_n=30) on 1000 synthetic rows, then time
               predict() on 1000 rows.  Pass = returns in < 30 s, results finite.
"""

import time
import numpy as np
import sys
import os

# Ensure repo root is importable when run as a plain script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("SPARC_NO_DB", "1")

N_FIT = 1000
N_PRED = 1_000
N_FEATURES = 6
RNG = np.random.default_rng(42)


def make_data(n):
    X = RNG.normal(size=(n, N_FEATURES)).astype(np.float32)
    y = X[:, 0] * 2 - X[:, 1] + RNG.normal(scale=0.2, size=n)
    coords = RNG.uniform(0, 1000, size=(n, 2))
    return X, y, coords


def run_loop():
    from sparc.models.gwrf import GWRFModel

    X_fit, y_fit, coords_fit = make_data(N_FIT)
    X_pred, _, coords_pred = make_data(N_PRED)

    print(f"Fitting GWRF on {N_FIT} rows …")
    model = GWRFModel(n_estimators=10, k_neighbors=30, min_samples_leaf=5,
                      n_jobs=1, subsample_n=30)
    model.fit(X_fit, y_fit, coords_fit)
    print(f"  {len(model.local_models)} local models fitted.")

    print(f"Predicting {N_PRED} rows …")
    t0 = time.perf_counter()
    preds, unc = model.predict(X_pred, coords_pred, k_blend=5)
    elapsed = time.perf_counter() - t0

    finite_pct = np.isfinite(preds).mean() * 100
    print(f"  Done in {elapsed:.2f}s  |  {finite_pct:.1f}% finite predictions")

    ok = elapsed < 30 and finite_pct > 90
    status = "PASS" if ok else "FAIL"
    print(f"\n[{status}]  predict({N_PRED} rows) completed in {elapsed:.2f}s")
    return ok


def check_real_data():
    """Check brown4.csv for NaN / Inf that could trigger Fortran crash."""
    import os, pandas as pd
    csv = os.path.join(os.path.dirname(__file__), "..", "brown4.csv")
    if not os.path.exists(csv):
        print("brown4.csv not found — skipping real-data NaN check.")
        return
    df = pd.read_csv(csv)
    predictors = ["Distance_from_water_m", "Pct_Impervious", "Pct_Canopy",
                  "NDVI", "Albedo", "Elevation_m"]
    cols = [c for c in predictors if c in df.columns]
    X = df[cols].values
    nan_mask = np.isnan(X)
    inf_mask = np.isinf(X)
    print(f"\nReal data check ({len(df)} rows, {len(cols)} features):")
    print(f"  NaN count : {nan_mask.sum()}")
    print(f"  Inf count : {inf_mask.sum()}")
    if nan_mask.any():
        rows_with_nan = np.where(nan_mask.any(axis=1))[0]
        print(f"  First NaN rows: {rows_with_nan[:10].tolist()}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check_real_data()

    ok = run_loop()
    sys.exit(0 if ok else 1)
