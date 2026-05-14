"""
OOF Spatial Intelligence Extraction Hooks
==========================================
Extracts spatially-varying model internals (GWR local β-coefficients,
GWRF local feature importances) from fitted fold models during Stage 2
cross-validation.  Called per-fold from the subprocess worker; aggregated
once all folds are complete.

Artifacts written to Stage 2 store:
  - ``oof_gwr_beta_map``        : (N, n_features) float32 ndarray
  - ``oof_gwrf_importance_map`` : (N, n_features) float32 ndarray
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Per-fold extraction (runs inside subprocess worker)
# ---------------------------------------------------------------------------

def extract_from_fitted_model(
    model,
    model_name: str,
    X_test,
    coords_test,
    feature_names: list[str],
    test_idx,
) -> dict | None:
    """Return a dict of spatial intelligence for one fold, or None on failure.

    Parameters
    ----------
    model : fitted GWRModel | GWRFModel
    model_name : 'gwr' | 'gwrf' | 'ggpgam'
    X_test : ndarray (n_test, n_features)
    coords_test : ndarray (n_test, 2)
    feature_names : list[str]
    test_idx : array-like of int — positions in the full N-sample array

    Returns
    -------
    dict with keys ``type``, ``test_idx``, ``beta_map`` or ``importance_map``,
    ``coords``, ``feature_names``; or None if extraction fails / not supported.
    """
    coords_test = np.asarray(coords_test)
    test_idx = np.asarray(test_idx)

    if model_name == "gwr":
        return _extract_gwr(model, coords_test, feature_names, test_idx)
    if model_name == "gwrf":
        return _extract_gwrf(model, coords_test, feature_names, test_idx)
    return None


def _extract_gwr(model, coords_test, feature_names, test_idx) -> dict | None:
    """Extract GWR local β-coefficients at test locations."""
    try:
        coefficients = getattr(model, "coefficients_", None)
        nn = getattr(model, "nn_", None)
        if coefficients is None or nn is None:
            return None
        # Find nearest training point for each test point
        _, indices = nn.kneighbors(coords_test)
        nearest = indices[:, 0]
        beta_map = coefficients[nearest].astype(np.float32)  # (n_test, n_features)
        return {
            "type": "gwr_local_coefficients",
            "test_idx": test_idx,
            "beta_map": beta_map,
            "coords": coords_test,
            "feature_names": feature_names,
        }
    except Exception as e:
        print(f"WARNING: GWR beta extraction failed: {e}")
        return None


def _extract_gwrf(model, coords_test, feature_names, test_idx) -> dict | None:
    """Extract GWRF local feature importances at test locations."""
    try:
        subsample_coords = getattr(model, "subsample_coords", None)
        local_models = getattr(model, "local_models", None)
        if subsample_coords is None or local_models is None:
            return None

        from sklearn.neighbors import BallTree

        tree = BallTree(subsample_coords)
        _, nearest_raw = tree.query(coords_test, k=1)
        nearest = nearest_raw[:, 0]

        n_features = len(feature_names)
        importance_map = np.zeros((len(coords_test), n_features), dtype=np.float32)
        for row_i, sub_idx in enumerate(nearest):
            rf = local_models[sub_idx]
            if rf is not None and hasattr(rf, "feature_importances_"):
                imp = rf.feature_importances_
                # Clip/pad to n_features in case of shape mismatch
                k = min(len(imp), n_features)
                importance_map[row_i, :k] = imp[:k].astype(np.float32)

        return {
            "type": "gwrf_spatial_importances",
            "test_idx": test_idx,
            "importance_map": importance_map,
            "coords": coords_test,
            "feature_names": feature_names,
        }
    except Exception as e:
        print(f"WARNING: GWRF importance extraction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Aggregation (runs in main process after all folds complete)
# ---------------------------------------------------------------------------

def aggregate_oof_intelligence(
    fold_results: list[dict],
    feature_names: list[str],
    n_samples: int,
    output_dir: str,
) -> dict:
    """Stitch per-fold dicts into full-sample arrays and write to artifact store.

    Parameters
    ----------
    fold_results : list of dicts returned by ``extract_from_fitted_model``
    feature_names : list[str]
    n_samples : int — total number of rows in the dataset
    output_dir : str — path to stage output dir (used for fallback file writes)

    Returns
    -------
    dict with keys ``gwr_beta_map`` and/or ``gwrf_importance_map``
        (whichever types were present in fold_results)
    """
    n_features = len(feature_names)
    gwr_array = np.zeros((n_samples, n_features), dtype=np.float32)
    gwr_coverage = np.zeros(n_samples, dtype=bool)
    gwrf_array = np.zeros((n_samples, n_features), dtype=np.float32)
    gwrf_coverage = np.zeros(n_samples, dtype=bool)

    for fold_dict in fold_results:
        if fold_dict is None:
            continue
        t = fold_dict.get("type", "")
        idx = np.asarray(fold_dict.get("test_idx", []))
        if len(idx) == 0:
            continue

        if t == "gwr_local_coefficients":
            data = fold_dict.get("beta_map")
            if data is not None and len(data) == len(idx):
                gwr_array[idx] = data
                gwr_coverage[idx] = True

        elif t == "gwrf_spatial_importances":
            data = fold_dict.get("importance_map")
            if data is not None and len(data) == len(idx):
                gwrf_array[idx] = data
                gwrf_coverage[idx] = True

    result = {}

    try:
        from sparc.registry.store import get_active_store
        store = get_active_store()
    except Exception:
        store = None

    if gwr_coverage.any():
        result["gwr_beta_map"] = gwr_array
        _write_array(store, "2", "oof_gwr_beta_map", gwr_array, output_dir)
        print(
            f"OOF GWR beta map: shape={gwr_array.shape}  "
            f"coverage={gwr_coverage.mean() * 100:.1f}%"
        )

    if gwrf_coverage.any():
        result["gwrf_importance_map"] = gwrf_array
        _write_array(store, "2", "oof_gwrf_importance_map", gwrf_array, output_dir)
        print(
            f"OOF GWRF importance map: shape={gwrf_array.shape}  "
            f"coverage={gwrf_coverage.mean() * 100:.1f}%"
        )

    return result


def _write_array(store, stage: str, artifact_id: str, arr: np.ndarray, output_dir: str) -> None:
    """Write ndarray to the artifact store as a blob (numpy serializer)."""
    try:
        if store is not None:
            store.write_blob(
                stage,
                artifact_id,
                arr,
                serializer="pickle",
                producer="oof_extraction_hooks.aggregate_oof_intelligence",
            )
            return
    except Exception as e:
        print(f"WARNING: store write failed for {artifact_id}: {e}")

    # Fallback: write as .npy file alongside output
    import os
    path = os.path.join(str(output_dir), f"{artifact_id}.npy")
    np.save(path, arr)
    print(f"  Fallback: saved {artifact_id} to {path}")
