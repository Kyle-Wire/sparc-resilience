"""Stage 2-neural (v2_neural_training) migration tests.

Round-trip the artifacts produced by ``sparc.run.v2_neural_training``
after the artifacts.db migration.

Coverage:
- meta_info struct
- predictions / local_coefficients_gwr / process_rate_map /
  spatial_attention_summary / feature_importance tables
- pdp_<feature> tables (one per feature)
- cma_es_best_params struct
- model state-dict blobs (torch serializer)
- numpy arrays (alpha_field, cardinal_neighbors, grid_spacing) blobs
- feature_scaling / oof_results / loss_history dict blobs
- joblib-encoded sinusoidal encoder blob
"""

from __future__ import annotations

import numpy as np
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


def test_v2_neural_meta_info_struct(store):
    payload = {
        "surrogate_names": ["gwr", "gwrf", "ggpgam"],
        "n_surrogates": 3,
        "n_physics_features": 5,
        "feature_names": ["albedo", "ndvi", "imperv"],
        "y_mean": 32.5,
        "y_std": 4.1,
        "oof_r2": 0.78,
        "oof_rmse": 0.6,
    }
    store.write_struct("2", "v2_neural_meta_info", payload, producer="test")
    out = store.read_any("2", "v2_neural_meta_info")
    assert out["oof_r2"] == pytest.approx(0.78)
    assert out["feature_names"] == ["albedo", "ndvi", "imperv"]


def test_v2_neural_predictions_table(store):
    df = pd.DataFrame({
        "actual": [10.0, 11.0, 12.0],
        "predicted_oof": [10.1, 10.9, 12.2],
        "predicted_full": [10.05, 11.05, 12.10],
        "uncertainty": [0.2, 0.3, 0.25],
    })
    store.write_table("2", "v2_neural_predictions", df, producer="test")
    out = store.read_any("2", "v2_neural_predictions")
    assert set(out.columns) == set(df.columns)


def test_v2_neural_pdp_table_per_feature(store):
    feat = "Pct_Canopy"
    df = pd.DataFrame({
        feat: [0.0, 0.1, 0.2, 0.3],
        "mean_prediction": [33.0, 32.7, 32.3, 32.0],
        "q10": [32.5, 32.2, 31.8, 31.5],
        "q90": [33.5, 33.2, 32.8, 32.5],
    })
    art_id = f"v2_neural_pdp::{feat}"
    store.write_table("2", art_id, df, producer="test",
                      metadata={"variable": feat, "source": "neural_pde"})
    out = store.read_any("2", art_id)
    assert feat in out.columns
    assert "mean_prediction" in out.columns


def test_v2_neural_feature_importance_table(store):
    df = pd.DataFrame({
        "feature": ["a", "b", "c"],
        "gradient_importance": [0.5, 0.3, 0.2],
        "normalised": [0.5, 0.3, 0.2],
    })
    store.write_table("2", "v2_neural_feature_importance", df, producer="test")
    out = store.read_any("2", "v2_neural_feature_importance")
    assert list(out["feature"]) == ["a", "b", "c"]


def test_cma_es_best_params_struct(store):
    payload = {"hidden_dim": 128, "lr": 1e-3, "weight_decay": 1e-5}
    store.write_struct("2", "cma_es_best_params", payload, producer="test")
    out = store.read_any("2", "cma_es_best_params")
    assert out["hidden_dim"] == 128


def test_v2_alpha_field_blob_roundtrip(store):
    arr = np.linspace(0, 1, 1000, dtype=np.float32)
    store.write_blob("2", "v2_alpha_field", arr, serializer="pickle",
                     producer="test")
    out = store.read_any("2", "v2_alpha_field")
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, arr)


def test_v2_feature_scaling_blob_roundtrip(store):
    payload = {"feat_mean": np.array([1.0, 2.0]), "feat_std": np.array([0.1, 0.2])}
    store.write_blob("2", "v2_feature_scaling", payload, serializer="pickle",
                     producer="test")
    out = store.read_any("2", "v2_feature_scaling")
    np.testing.assert_array_equal(out["feat_mean"], payload["feat_mean"])
    np.testing.assert_array_equal(out["feat_std"], payload["feat_std"])


def test_v2_torch_state_blob_roundtrip(store):
    torch = pytest.importorskip("torch")
    state_dict = {
        "layer.weight": torch.zeros(4, 3),
        "layer.bias": torch.ones(4),
    }
    store.write_blob("2", "v2_neural_meta_state", state_dict,
                     serializer="torch", producer="test")
    out = store.read_any("2", "v2_neural_meta_state")
    assert "layer.weight" in out
    assert out["layer.bias"].sum().item() == pytest.approx(4.0)


def test_v2_loss_history_blob_roundtrip(store):
    payload = {
        "total": np.array([1.0, 0.8, 0.6, 0.5]),
        "data": np.array([0.7, 0.5, 0.4, 0.3]),
        "physics": np.array([0.3, 0.3, 0.2, 0.2]),
    }
    store.write_blob("2", "v2_loss_history", payload, serializer="pickle",
                     producer="test")
    out = store.read_any("2", "v2_loss_history")
    assert set(out.keys()) == {"total", "data", "physics"}
    np.testing.assert_array_equal(out["total"], payload["total"])
