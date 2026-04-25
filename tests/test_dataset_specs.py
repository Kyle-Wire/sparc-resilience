"""Tests for the canonical dataset specs (PR2 of the database refactor).

These tests pin the *layout contract*:
- every Dataset spec has a stable qualified name and storage format,
- save_dataset writes to the expected ``stageN_topic/<name>.<ext>`` path,
- load_dataset round-trips the value.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def store(tmp_path: Path):
    from sparc.run.pipeline_paths import PipelinePaths
    from sparc.run.result_store import ResultStore

    cfg = {
        "paths": {"project_root": str(tmp_path)},
        "output": {"base_dir": str(tmp_path / "output")},
    }
    paths = PipelinePaths.from_config(cfg)
    return ResultStore(paths)


# ---------------------------------------------------------------------------
# Spec inventory
# ---------------------------------------------------------------------------


def test_stage0_specs_present():
    from sparc.registry.datasets import STAGE0_DATASETS

    qnames = {d.qualified_name for d in STAGE0_DATASETS}
    assert qnames == {
        "stage0.profile",
        "stage0.results",
        "stage0.summary",
        "stage0.spatial_cv_config",
    }


def test_stage1_specs_present():
    from sparc.registry.datasets import STAGE1_DATASETS

    qnames = {d.qualified_name for d in STAGE1_DATASETS}
    assert qnames == {
        "stage1.results",
        "stage1.diagnostics",
        "stage1.importance",
        "stage1.selected_features",
    }


def test_all_datasets_lookup():
    from sparc.registry.datasets import ALL_DATASETS, datasets_for_stage

    assert "stage0.profile" in ALL_DATASETS
    assert "stage1.results" in ALL_DATASETS
    assert {d.name for d in datasets_for_stage("0")} == {
        "profile", "results", "summary", "spatial_cv_config",
    }
    assert {d.name for d in datasets_for_stage(1)} == {
        "results", "diagnostics", "importance", "selected_features",
    }


# ---------------------------------------------------------------------------
# Layout contract — files must land in stageN_topic/ with short names
# ---------------------------------------------------------------------------


def test_stage0_layout_paths(store, tmp_path):
    from sparc.registry.datasets import (
        STAGE0_DATASET_PROFILE,
        STAGE0_CORRELOGRAM_RESULTS,
        STAGE0_CORRELOGRAM_SUMMARY,
        STAGE0_SPATIAL_CV_CONFIG,
    )

    profile = {"size_tier": "medium", "spatial_extent": 12345.0}
    results = {"individual_results": {"lst": {"optimal_bandwidth": 800.0}}}
    summary_df = pd.DataFrame([{
        "Variable": "lst", "Optimal_Bandwidth": 800.0,
        "Effective_Range": 1200.0, "Block_Size": 1600.0,
        "Best_Kernel": "exponential", "Max_Moran_I": 0.42,
        "Significant_Lags": 5,
    }])
    cv_cfg = {
        "optimal_block_size": 1600.0,
        "model_bandwidths": {"gwr": 800.0},
        "validation_results": {"1600": {"gwr": "ok"}},
    }

    profile_path = store.save_dataset(STAGE0_DATASET_PROFILE, profile)
    results_path = store.save_dataset(STAGE0_CORRELOGRAM_RESULTS, results)
    summary_path = store.save_dataset(STAGE0_CORRELOGRAM_SUMMARY, summary_df)
    cv_path = store.save_dataset(STAGE0_SPATIAL_CV_CONFIG, cv_cfg)

    base = tmp_path / "output" / "stage0_correlogram"
    assert profile_path == base / "profile.json"
    assert results_path == base / "results.json"
    assert summary_path == base / "summary.parquet"
    assert cv_path == base / "spatial_cv_config.json"


def test_stage1_layout_paths(store, tmp_path):
    from sparc.registry.datasets import (
        STAGE1_GWEN_RESULTS,
        STAGE1_GWEN_DIAGNOSTICS,
        STAGE1_VARIABLE_IMPORTANCE,
        STAGE1_SELECTED_FEATURES,
    )

    results_path = store.save_dataset(STAGE1_GWEN_RESULTS, {"selected_features": ["a"]})
    diag_path = store.save_dataset(STAGE1_GWEN_DIAGNOSTICS, {"r2": 0.9})
    imp_path = store.save_dataset(
        STAGE1_VARIABLE_IMPORTANCE,
        pd.DataFrame([{
            "feature": "a", "selection_frequency": 1.0,
            "mean_local_coef": 0.5, "spatial_variability": 0.1,
        }]),
    )
    feat_path = store.save_dataset(STAGE1_SELECTED_FEATURES, ["a", "b", "c"])

    base = tmp_path / "output" / "stage1_gwen"
    assert results_path == base / "results.json"
    assert diag_path == base / "diagnostics.json"
    assert imp_path == base / "importance.parquet"
    assert feat_path == base / "selected_features.json"


# ---------------------------------------------------------------------------
# Round-trip via the registry
# ---------------------------------------------------------------------------


def test_stage0_results_round_trip(store):
    from sparc.registry.datasets import STAGE0_CORRELOGRAM_RESULTS

    payload = {
        "individual_results": {"lst": {"optimal_bandwidth": 800.0}},
        "summary_statistics": {"bandwidth_range": {"min": 100, "max": 2000}},
    }
    store.save_dataset(STAGE0_CORRELOGRAM_RESULTS, payload)

    loaded = store.load_dataset("0", "results")
    assert loaded == payload


def test_stage0_summary_round_trip_via_parquet(store):
    from sparc.registry.datasets import STAGE0_CORRELOGRAM_SUMMARY

    df = pd.DataFrame([
        {"Variable": "a", "Optimal_Bandwidth": 1.0, "Effective_Range": 2.0,
         "Block_Size": 3.0, "Best_Kernel": "exp", "Max_Moran_I": 0.5,
         "Significant_Lags": 4},
        {"Variable": "b", "Optimal_Bandwidth": 1.5, "Effective_Range": 2.5,
         "Block_Size": 3.5, "Best_Kernel": "gauss", "Max_Moran_I": 0.6,
         "Significant_Lags": 3},
    ])
    store.save_dataset(STAGE0_CORRELOGRAM_SUMMARY, df)

    loaded = store.load_dataset("0", "summary")
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), df)


def test_stage1_selected_features_round_trip(store):
    from sparc.registry.datasets import STAGE1_SELECTED_FEATURES

    feats = ["lst", "ndvi", "elev"]
    store.save_dataset(STAGE1_SELECTED_FEATURES, feats)
    loaded = store.load_dataset("1", "selected_features")
    assert loaded == feats


def test_dataset_writes_register_in_run_registry(store):
    from sparc.registry.datasets import (
        STAGE0_CORRELOGRAM_SUMMARY,
        STAGE1_GWEN_RESULTS,
    )

    store.save_dataset(STAGE0_CORRELOGRAM_SUMMARY, pd.DataFrame([{
        "Variable": "x", "Optimal_Bandwidth": 1.0, "Effective_Range": 2.0,
        "Block_Size": 3.0, "Best_Kernel": "exp", "Max_Moran_I": 0.5,
        "Significant_Lags": 4,
    }]))
    store.save_dataset(STAGE1_GWEN_RESULTS, {"selected_features": ["a"]})

    s0 = store.registry.lookup("0", "summary")
    s1 = store.registry.lookup("1", "results")
    assert s0 is not None and s0.format == "parquet"
    assert s1 is not None and s1.format == "json"
    assert s0.metadata["qualified_name"] == "stage0.summary"
    assert s1.metadata["qualified_name"] == "stage1.results"


# ---------------------------------------------------------------------------
# DuckDB views over the new layout
# ---------------------------------------------------------------------------


def test_duckdb_view_for_stage0_summary(store):
    pytest.importorskip("duckdb")
    from sparc.registry.datasets import STAGE0_CORRELOGRAM_SUMMARY

    df = pd.DataFrame([
        {"Variable": "a", "Optimal_Bandwidth": 100.0, "Effective_Range": 200.0,
         "Block_Size": 300.0, "Best_Kernel": "exp", "Max_Moran_I": 0.5,
         "Significant_Lags": 4},
        {"Variable": "b", "Optimal_Bandwidth": 110.0, "Effective_Range": 210.0,
         "Block_Size": 310.0, "Best_Kernel": "exp", "Max_Moran_I": 0.55,
         "Significant_Lags": 5},
    ])
    store.save_dataset(STAGE0_CORRELOGRAM_SUMMARY, df)

    con = store.registry.duckdb()
    rows = con.execute(
        "SELECT count(*) AS n, sum(Optimal_Bandwidth) AS sb FROM stage0_summary"
    ).fetchone()
    assert rows == (2, 210.0)
