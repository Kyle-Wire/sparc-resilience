"""Stage 4 (scenario_simulator) artifacts.db migration tests.

Round-trip the tables produced after the Phase C6 migration:
- scenario_summary, scenario_summary_dag, scenario_summary_hybrid,
  scenario_summary_reprediction
- scenario_results (GeoDataFrame with WKB geometry), and the mode-
  specific variants (_dag / _hybrid / _reprediction)
- scenario_mc_uncertainty, scenario_mc_consensus,
  scenario_mc_consensus_summary
- bma_scenario_summary
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


def _summary_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Variable": "Pct_Canopy", "Scenario": "Pct_Canopy_plus_5",
            "Increment": 5, "Direction": "increase", "Area": "All",
            "N_Cells": 100, "Avg_Var_Change": 5.0,
            "Repred_Delta_Mean": -0.12, "Repred_Delta_Std": 0.04,
            "Method": "base_model_consensus",
        },
        {
            "Variable": "Albedo", "Scenario": "Albedo_plus_0p1",
            "Increment": 0.1, "Direction": "increase", "Area": "All",
            "N_Cells": 100, "Avg_Var_Change": 0.1,
            "Repred_Delta_Mean": -0.05, "Repred_Delta_Std": 0.02,
            "Method": "base_model_consensus",
        },
    ])


@pytest.mark.parametrize("artifact_id", [
    "scenario_summary",
    "scenario_summary_dag",
    "scenario_summary_hybrid",
    "scenario_summary_reprediction",
])
def test_scenario_summary_table_roundtrip(store, artifact_id):
    df = _summary_frame()
    store.write_table("4", artifact_id, df, producer="test")
    out = store.read_any("4", artifact_id)
    assert len(out) == 2
    assert "Repred_Delta_Mean" in out.columns


@pytest.mark.parametrize("artifact_id", [
    "scenario_results",
    "scenario_results_dag",
    "scenario_results_hybrid",
    "scenario_results_reprediction",
])
def test_scenario_results_geo_table_roundtrip(store, artifact_id):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    n = 8
    rng = np.random.default_rng(0)
    pts = [Point(float(x), float(y))
           for x, y in zip(rng.uniform(-10, 10, n), rng.uniform(-10, 10, n))]
    gdf = gpd.GeoDataFrame(
        {
            "OBJECTID": np.arange(n),
            "pred_baseline": rng.standard_normal(n),
            "pred_Pct_Canopy_plus_5": rng.standard_normal(n),
        },
        geometry=pts,
        crs="EPSG:4326",
    )
    store.write_table("4", artifact_id, gdf,
                      geometry_col="geometry", crs="EPSG:4326",
                      producer="test")
    out = store.read_any("4", artifact_id)
    assert isinstance(out, gpd.GeoDataFrame)
    assert len(out) == n
    assert out.crs is not None and "4326" in str(out.crs)
    # Geometry round-trips
    assert out.geometry.iloc[0].x == pytest.approx(pts[0].x)
    assert out.geometry.iloc[0].y == pytest.approx(pts[0].y)


def test_scenario_mc_uncertainty_table(store):
    df = pd.DataFrame({
        "OBJECTID": np.arange(5),
        "total_Pct_Canopy_plus_5_q05": np.linspace(-0.2, -0.1, 5),
        "total_Pct_Canopy_plus_5_q50": np.linspace(-0.15, -0.05, 5),
        "total_Pct_Canopy_plus_5_q95": np.linspace(-0.10, 0.0, 5),
    })
    store.write_table("4", "scenario_mc_uncertainty", df, producer="test")
    out = store.read_any("4", "scenario_mc_uncertainty")
    assert len(out) == 5
    assert all(c in out.columns for c in (
        "total_Pct_Canopy_plus_5_q05",
        "total_Pct_Canopy_plus_5_q50",
        "total_Pct_Canopy_plus_5_q95",
    ))


def test_scenario_mc_consensus_table(store):
    df = pd.DataFrame({
        "OBJECTID": np.arange(4),
        "repred_delta_Pct_Canopy_plus_5_q05": [-0.2, -0.18, -0.16, -0.14],
        "repred_delta_Pct_Canopy_plus_5_q50": [-0.15, -0.13, -0.11, -0.09],
        "repred_delta_Pct_Canopy_plus_5_q95": [-0.10, -0.08, -0.06, -0.04],
    })
    store.write_table("4", "scenario_mc_consensus", df, producer="test")
    out = store.read_any("4", "scenario_mc_consensus")
    assert len(out) == 4


def test_scenario_mc_consensus_summary_table(store):
    df = pd.DataFrame([
        {"Scenario": "Pct_Canopy_plus_5", "Variable": "Pct_Canopy",
         "MC_Mean": -0.12, "MC_Std": 0.03,
         "MC_q05": -0.18, "MC_q50": -0.12, "MC_q95": -0.06},
    ])
    store.write_table("4", "scenario_mc_consensus_summary", df, producer="test")
    out = store.read_any("4", "scenario_mc_consensus_summary")
    assert "MC_q50" in out.columns
    assert float(out["MC_q50"].iloc[0]) == pytest.approx(-0.12)


def test_bma_scenario_summary_table(store):
    df = pd.DataFrame([
        {"Variable": "Pct_Canopy", "Increment": 5, "Area": "All",
         "Mean_Delta": -0.12, "Std_Delta": 0.03,
         "CI_5": -0.18, "CI_95": -0.06,
         "Exceedance_Prob": 0.05, "Inference": "nuts_posterior"},
    ])
    store.write_table("4", "bma_scenario_summary", df, producer="test")
    out = store.read_any("4", "bma_scenario_summary")
    assert "Inference" in out.columns
    assert out["Inference"].iloc[0] == "nuts_posterior"
