"""Stage 1 (GWEN) migration tests.

Verifies the writer/reader round-trips for the artifacts produced by
``sparc.run.gwen_variable_selection`` after the artifacts.db migration.
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


def test_gwen_results_struct_roundtrip(store):
    payload = {
        "timestamp": "2026-04-27T00:00:00",
        "total_features": 8,
        "selected_features": ["albedo", "ndvi", "imperv"],
        "selection_threshold": 0.1,
        "model_params": {"k_neighbors": 50, "sample_size": 1000, "cv_folds": 5,
                         "l1_ratios": [0.5, 0.7, 0.9], "n_alphas": 100},
        "selection_summary": {"features_selected": 3, "features_total": 8,
                              "selection_rate": 0.375},
        "feature_importance": [
            {"feature": "albedo", "selection_frequency": 0.92, "mean_local_coef": -0.4},
            {"feature": "ndvi", "selection_frequency": 0.81, "mean_local_coef": -0.2},
        ],
    }
    store.write_struct("1", "gwen_results", payload, producer="test")
    out = store.read_any("1", "gwen_results")
    assert out["total_features"] == 8
    assert "albedo" in out["selected_features"]


def test_gwen_variable_importance_table_roundtrip(store):
    df = pd.DataFrame({
        "feature": ["a", "b", "c"],
        "selection_frequency": [0.9, 0.7, 0.1],
        "mean_local_coef": [-0.5, 0.2, 0.0],
    })
    store.write_table("1", "gwen_variable_importance", df, producer="test")
    out = store.read_any("1", "gwen_variable_importance")
    assert list(out["feature"]) == ["a", "b", "c"]


def test_gwen_diagnostics_struct_roundtrip(store):
    diag = {
        "tuning_parameters": {"bandwidth_k": 50, "selected_alpha": 0.01},
        "stability_metrics": {
            "albedo": {"selection_frequency": 0.9, "mean_abs_coefficient": 0.4,
                       "coefficient_sign_stability": 1.0, "spatial_variability": 0.05,
                       "coefficient_range": "-0.5 to -0.3"},
        },
        "selection_rule": {"threshold": 0.1, "criterion": "selection_frequency >= threshold"},
    }
    store.write_struct("1", "gwen_diagnostics", diag, producer="test")
    out = store.read_any("1", "gwen_diagnostics")
    assert out["tuning_parameters"]["bandwidth_k"] == 50


def test_gwen_local_coefficients_struct_roundtrip(store):
    payload = {
        "feature_names": ["a", "b"],
        "local_coefficients": [[0.1, 0.2], [0.0, 0.3], [0.15, 0.0]],
        "local_alphas": [0.01, 0.01, 0.02],
        "local_l1_ratios": [0.5, 0.7, 0.5],
        "importance": [
            {"feature": "a", "selection_frequency": 0.66},
            {"feature": "b", "selection_frequency": 0.66},
        ],
    }
    store.write_struct("1", "gwen_local_coefficients", payload, producer="test")
    out = store.read_any("1", "gwen_local_coefficients")
    assert out["feature_names"] == ["a", "b"]
    assert len(out["local_coefficients"]) == 3


def test_gwen_stability_table_roundtrip(store):
    df = pd.DataFrame({
        "feature": ["a", "b"],
        "selection_stability": [1.0, 0.8],
        "sign_consistency": [1.0, 0.9],
        "rank_stability": [0.85, 0.7],
    })
    store.write_table("1", "gwen_stability", df, producer="test")
    out = store.read_any("1", "gwen_stability")
    assert set(out.columns) == {"feature", "selection_stability", "sign_consistency", "rank_stability"}
