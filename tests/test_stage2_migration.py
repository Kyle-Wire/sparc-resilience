"""Stage 2 (Spatial CV) migration tests.

Round-trip the user-facing artifacts produced by
``sparc.run.enhanced_spatial_cv`` after the artifacts.db migration.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sparc.registry.run_registry import RunRegistry, set_active_registry
from sparc.registry.store import ArtifactStore


@pytest.fixture()
def store(tmp_path):
    reg = RunRegistry(tmp_path)
    set_active_registry(reg)
    yield ArtifactStore(reg)
    set_active_registry(None)


def test_oof_predictions_table_roundtrip(store):
    df = pd.DataFrame({
        "OLS": [1.0, 2.0, 3.0],
        "GWR": [1.1, 2.1, 3.1],
        "GWRF": [0.9, 2.2, 2.9],
        "GGPGAM": [1.05, 2.05, 3.05],
    })
    store.write_table("2", "oof_predictions", df, producer="test")
    out = store.read_any("2", "oof_predictions")
    assert list(out.columns) == ["OLS", "GWR", "GWRF", "GGPGAM"]
    assert len(out) == 3


def test_ensemble_results_struct_roundtrip(store):
    payload = {
        "base_models": {
            "ols": {"r2": 0.42, "rmse": 1.1},
            "gwr": {"r2": 0.61, "rmse": 0.9},
            "gwrf": {"r2": 0.71, "rmse": 0.8},
        },
        "final_ensemble": {"r2": 0.78, "rmse": 0.7},
        "spatial_autocorrelation": {"moran_i": 0.15, "p_value": 0.001},
        "improvements": {"best_meta_vs_base": 0.07},
    }
    store.write_struct("2", "ensemble_results", payload, producer="test")
    out = store.read_any("2", "ensemble_results")
    assert out["base_models"]["gwrf"]["r2"] == pytest.approx(0.71)
    assert out["final_ensemble"]["r2"] == pytest.approx(0.78)


def test_ensemble_predictions_table_roundtrip(store):
    df = pd.DataFrame({
        "OBJECTID": [1, 2, 3],
        "actual": [10.0, 11.0, 12.0],
        "best_meta_approach": ["v2_neural", "v2_neural", "v2_neural"],
        "final_ensemble": [10.1, 10.9, 12.2],
    })
    store.write_table("2", "ensemble_predictions", df, producer="test")
    out = store.read_any("2", "ensemble_predictions")
    assert set(out.columns) == {"OBJECTID", "actual", "best_meta_approach", "final_ensemble"}


def test_feature_info_struct_roundtrip(store):
    payload = {
        "feature_names": ["albedo", "ndvi", "imperv"],
        "n_features": 3,
        "scaling_applied": False,
    }
    store.write_struct("2", "feature_info", payload, producer="test")
    out = store.read_any("2", "feature_info")
    assert out["feature_names"] == ["albedo", "ndvi", "imperv"]
    assert out["scaling_applied"] is False


def test_dataset_profile_stage2_struct_roundtrip(store):
    payload = {
        "n_obs": 1234,
        "n_features": 12,
        "spatial_extent_km2": 56.7,
        "recommended_bandwidth": 75,
    }
    store.write_struct("2", "dataset_profile", payload, producer="test")
    out = store.read_any("2", "dataset_profile")
    assert out["n_obs"] == 1234


def test_spatial_cv_predictions_geometry_roundtrip(store):
    pytest.importorskip("geopandas")
    pytest.importorskip("shapely")
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {
            "observed": [10.0, 11.0, 12.0],
            "pred_ols": [9.9, 11.1, 11.8],
            "residual_ols": [-0.1, 0.1, -0.2],
        },
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs="EPSG:4326",
    )
    store.write_table("2", "spatial_cv_predictions", gdf, producer="test")
    out = store.read_any("2", "spatial_cv_predictions")
    assert hasattr(out, "geometry")
    assert str(out.crs).upper() == "EPSG:4326"
    assert len(out) == 3
    assert "pred_ols" in out.columns
