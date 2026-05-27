"""sparc.data.preprocessing — pure pipeline logic, no FastAPI.

Extracted from sparc.server.routes.data._generate() (D1 refactor).
``run_pipeline(df, config)`` is a synchronous generator that yields
step-result dicts with at minimum the keys ``step``, ``rows``, ``sha``.
"""

from __future__ import annotations

from typing import Generator, Any


def _hash_df(df) -> str:
    try:
        import pandas as pd
        return "%08x" % (int(pd.util.hash_pandas_object(df).sum()) % (2 ** 32))
    except Exception:
        return "00000000"


def _result(step_name: str, df, **extra) -> dict:
    return {"step": step_name, "done": True, "rows": len(df), "sha": _hash_df(df), **extra}


def run_pipeline(df, config: dict | None = None, _result_ref: dict | None = None) -> Generator[dict, None, Any]:
    """Yield step-result dicts for each preprocessing stage.

    Parameters
    ----------
    df:
        Input pandas DataFrame (will be copied internally).
    config:
        Optional project config dict (same structure as ``state.project_config``).
        Pass ``{}`` or ``None`` for a no-op run with all defaults.
    _result_ref:
        Optional mutable dict; on generator exhaustion ``_result_ref['df']``
        is set to the final transformed DataFrame so callers can retrieve it.

    Yields
    ------
    dict
        Keys: ``step`` (str), ``rows`` (int), ``sha`` (str), ``done`` (bool).
        The final dict has ``step == "__done__"``.
    """
    import numpy as np
    import pandas as pd

    if config is None:
        config = {}

    df = df.copy()
    if hasattr(df, "geometry"):
        df = pd.DataFrame(df.drop(columns="geometry"))

    step_hashes: dict = {}

    # Derive filesystem paths from config when available (all guarded by try/except)
    _project_dir = None
    try:
        from pathlib import Path as _Path
        _root = (config.get("paths") or {}).get("project_root")
        if _root:
            _project_dir = _Path(_root)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Step 1 — Ingest CSV
    # ------------------------------------------------------------------
    sha = _hash_df(df)
    step_hashes["ingest_csv"] = sha
    yield _result("Ingest CSV", df, sha=sha)

    # ------------------------------------------------------------------
    # Step 2 — Reproject CRS (coerce coordinate columns to numeric)
    # ------------------------------------------------------------------
    try:
        coord_cols = config.get("variables", {}).get("coordinates", []) or []
        for col in coord_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    except Exception as exc:
        print(f"  Preprocess step 2 (CRS): {exc}")
    sha = _hash_df(df)
    step_hashes["reproject_crs"] = sha
    yield _result("Reproject CRS", df, sha=sha)

    # ------------------------------------------------------------------
    # Step 3 — Deduplicate coords
    # ------------------------------------------------------------------
    coord_cols = config.get("variables", {}).get("coordinates", []) or []
    coord_cols = [c for c in coord_cols if c in df.columns]
    if coord_cols:
        before = len(df)
        df = df.drop_duplicates(subset=coord_cols)
        if before != len(df):
            print(f"  Deduplication: removed {before - len(df)} duplicate coord rows")
    sha = _hash_df(df)
    step_hashes["deduplicate_coords"] = sha
    yield _result("Deduplicate coords", df, sha=sha)

    # ------------------------------------------------------------------
    # Step 4 — Impute missing
    # ------------------------------------------------------------------
    target_col = config.get("variables", {}).get("target")
    predictor_cols = config.get("predictors", {}).get("base_model", []) or []
    essential = [c for c in ([target_col] + list(coord_cols) + list(predictor_cols)) if c and c in df.columns]
    for col in df.select_dtypes(include="number").columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    if essential:
        before = len(df)
        df = df.dropna(subset=essential)
        if before != len(df):
            print(f"  Impute: dropped {before - len(df)} rows with missing essential values")
    sha = _hash_df(df)
    step_hashes["impute_missing"] = sha
    yield _result("Impute missing", df, sha=sha)

    # ------------------------------------------------------------------
    # Step 5 — Derive features (inf → nan cleanup)
    # ------------------------------------------------------------------
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    sha = _hash_df(df)
    step_hashes["derive_features"] = sha
    yield _result("Derive features", df, sha=sha)

    # ------------------------------------------------------------------
    # Step 6 — Standardise (Welford z-score); saves scaler when project_dir known
    # ------------------------------------------------------------------
    try:
        from sparc.data.welford import WelfordScaler
        numeric_cols = list(df.select_dtypes(include="number").columns)
        if numeric_cols:
            X_num = df[numeric_cols].to_numpy(dtype=np.float64)
            scaler = WelfordScaler()
            scaler.partial_fit(X_num[~np.isnan(X_num).any(axis=1)])
            X_scaled = scaler.transform(X_num)
            df[numeric_cols] = X_scaled
            if _project_dir is not None:
                artifacts_dir = _project_dir / "artifacts"
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                scaler.save(artifacts_dir / "welford_scaler.pkl")
    except Exception as exc:
        print(f"  Preprocess step 6 (standardise): {exc}")
    sha = _hash_df(df)
    step_hashes["standardise"] = sha
    yield _result("Standardise (z-score)", df, sha=sha)

    # ------------------------------------------------------------------
    # Step 7 — Spatial block split
    # ------------------------------------------------------------------
    sha = _hash_df(df)
    step_hashes["spatial_block_split"] = sha
    yield _result("Spatial block split", df, sha=sha)

    # ------------------------------------------------------------------
    # Step 8 — Write cached arrow (parquet); guarded by project_dir
    # ------------------------------------------------------------------
    try:
        if _project_dir is not None:
            cache_path = _project_dir / "data_cache.parquet"
            df.to_parquet(cache_path, engine="pyarrow", index=False)
    except Exception as exc:
        print(f"  Preprocess step 8 (arrow cache): {exc}")
    sha = _hash_df(df)
    step_hashes["write_arrow"] = sha
    yield _result("Write cached arrow", df, sha=sha)

    # ------------------------------------------------------------------
    # Post-pipeline — versioning
    # ------------------------------------------------------------------
    try:
        if _project_dir is not None:
            from sparc.data.versioning import save_versioned
            data_dir = _project_dir / "data"
            save_versioned(df, data_dir, description="preprocess endpoint", settings={"step_hashes": step_hashes})
    except Exception as exc:
        print(f"  Preprocess: versioning failed: {exc}")

    # ------------------------------------------------------------------
    # Final sentinel
    # ------------------------------------------------------------------
    final_sha = _hash_df(df)
    if _result_ref is not None:
        _result_ref["df"] = df
    yield {"step": "__done__", "done": True, "rows": len(df), "sha": final_sha, "step_hashes": step_hashes}

    return df

