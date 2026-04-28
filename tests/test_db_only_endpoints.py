"""In-depth DB-only verification for live visualization + report/download endpoints.

Phase A — fixtures + a ``forbid_disk_reads`` sentinel that raises whenever
``pd.read_csv`` / ``pd.read_parquet`` / ``gpd.read_parquet`` / ``gpd.read_file``
is invoked against any path under a stage directory. This proves at runtime
(not just by inspection) that endpoints route through ``ArtifactStore`` and
``artifacts.db``, never through loose ``Stage_*`` files.

Phases B–E populate one test per endpoint exposed in the desktop's
``vizRegistry.ts``, every artifact download / native-render endpoint, and every
report/bundle endpoint. Phase E uses ``pytest.xfail(strict=True)`` for the
endpoints that currently bypass the store; once Phases F+G land they auto-flip
to passing and CI will catch any regression that re-introduces a disk read.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Point

from sparc.registry.run_registry import RunRegistry
from sparc.registry.store import ArtifactStore
from sparc.scenario import (
    PRED_BASELINE_COL,
    PRED_SCENARIO_PREFIX,
    SCENARIO_ATTRIBUTION,
    SCENARIO_LIBRARY,
    SCENARIO_STAGE,
    SCENARIO_TRAJECTORY,
    UNCERTAINTY_COLS,
)
from sparc.server import app as app_module


# ---------------------------------------------------------------------------
# Phase A — fixtures + sentinels
# ---------------------------------------------------------------------------


def _stage_dir_names() -> tuple[str, ...]:
    return (
        "Stage_0_Correlogram",
        "Stage_1_GWEN",
        "Stage_2_Spatial_CV",
        "Stage_3_Causal_Validation",
        "Stage_4_Scenarios",
    )


def _path_str(p: Any) -> str:
    try:
        return str(Path(p))
    except Exception:
        return str(p)


def _is_stage_path(target: Any) -> bool:
    """True if the path refers to a per-stage directory leak."""
    s = _path_str(target)
    return any(name in s for name in _stage_dir_names())


@pytest.fixture()
def attached_registry(tmp_path, monkeypatch):
    """Real ``RunRegistry`` swapped onto the FastAPI ``state``."""
    reg = RunRegistry(tmp_path, autoload=False)
    reg.add_register_listener(app_module._on_artifact_registered)
    monkeypatch.setattr(app_module.state, "registry", reg, raising=False)
    yield reg
    try:
        reg.remove_register_listener(app_module._on_artifact_registered)
    except Exception:
        pass


@pytest.fixture()
def project_config(tmp_path) -> dict:
    """Minimal project config wired to ``tmp_path``."""
    return {
        "output": {"base_dir": str(tmp_path), "disk_writes": False},
        "paths": {
            "project_root": str(tmp_path),
            "output_dir": str(tmp_path),
        },
        "variables": {"coordinates": ["lon", "lat"], "target": "y"},
        "crs": {"target_projected": "EPSG:4326"},
        "predictors": ["a", "b"],
        "causal": {"estimator": "bayesian", "actionable_variables": ["Pct_Canopy"]},
        "physics": {"monotone_constraints": {"Pct_Canopy": -1}},
        "pipeline": {"random_seed": 42, "n_spatial_folds": 5, "fast_mode": True},
        "project": {"name": "pytest", "domain": "uhi"},
    }


@pytest.fixture()
def seeded_state(attached_registry, project_config, monkeypatch, tmp_path):
    """Attach config + a tiny GeoDataFrame to ``state``."""
    monkeypatch.setattr(app_module.state, "project_config", project_config, raising=False)
    monkeypatch.setattr(app_module.state, "project_path", str(tmp_path / "project.yml"), raising=False)
    monkeypatch.setattr(
        app_module.state,
        "raw_project_yaml",
        {"project": project_config["project"]},
        raising=False,
    )
    monkeypatch.setattr(
        app_module.state,
        "data_summary",
        {"row_count": 3, "column_count": 4, "crs": "EPSG:4326"},
        raising=False,
    )
    gdf = gpd.GeoDataFrame(
        {"_": list(range(3))},
        geometry=[Point(i, i) for i in range(3)],
        crs="EPSG:4326",
    )
    monkeypatch.setattr(app_module.state, "data", gdf, raising=False)
    return attached_registry


@pytest.fixture()
def forbid_disk_reads(monkeypatch):
    """Sentinel: raise on any read targeting a per-stage directory.

    Reads to the canonical input parquet, the registry's own ``artifacts.db``,
    or any tempfile are allowed — only ``Stage_*`` paths trigger.
    """
    leaked: list[str] = []

    real_pd_read_csv = pd.read_csv
    real_pd_read_parquet = pd.read_parquet
    real_gpd_read_parquet = gpd.read_parquet
    real_gpd_read_file = gpd.read_file

    def _guard(name, real):
        def wrapper(filepath_or_buffer, *args, **kwargs):
            if hasattr(filepath_or_buffer, "read"):
                # In-memory buffer; allow.
                return real(filepath_or_buffer, *args, **kwargs)
            if _is_stage_path(filepath_or_buffer):
                msg = f"disk read leaked via {name}: {filepath_or_buffer}"
                leaked.append(msg)
                raise AssertionError(msg)
            return real(filepath_or_buffer, *args, **kwargs)

        return wrapper

    monkeypatch.setattr(pd, "read_csv", _guard("pd.read_csv", real_pd_read_csv))
    monkeypatch.setattr(pd, "read_parquet", _guard("pd.read_parquet", real_pd_read_parquet))
    monkeypatch.setattr(gpd, "read_parquet", _guard("gpd.read_parquet", real_gpd_read_parquet))
    monkeypatch.setattr(gpd, "read_file", _guard("gpd.read_file", real_gpd_read_file))

    yield leaked


def _make_gdf(n: int = 3, *, extra_cols: dict[str, Any] | None = None) -> gpd.GeoDataFrame:
    base = {"id": list(range(n))}
    if extra_cols:
        base.update(extra_cols)
    return gpd.GeoDataFrame(
        base,
        geometry=[Point(i, i) for i in range(n)],
        crs="EPSG:4326",
    )


def seed_all_stage_artifacts(store: ArtifactStore) -> None:
    """Write minimal artifacts for every (stage, id) the desktop UI consumes."""
    # Stage 0 — correlogram
    store.write_struct(
        "0",
        "correlogram_results",
        {
            "individual_results": {
                "x1": {"optimal_bandwidth": 1500.0, "effective_range": 1200.0, "max_moran_i": 0.4},
            }
        },
        producer="test",
    )

    # Stage 1 — GWEN
    store.write_table(
        "1",
        "gwen_variable_importance",
        pd.DataFrame({"variable": ["x1", "x2"], "importance": [0.6, 0.4]}),
        producer="test",
    )

    # Stage 2 — predictions (geo) + ensemble + neural PDP family
    preds = _make_gdf(3, extra_cols={"y_pred": [1.0, 2.0, 3.0], "y_true": [1.1, 2.1, 2.9]})
    store.write_table(
        "2",
        "spatial_cv_predictions",
        preds,
        geometry_col="geometry",
        crs="EPSG:4326",
        producer="test",
    )
    store.write_struct(
        "2",
        "ensemble_results",
        {
            "base_models": {
                "rf": {"r2": 0.75, "rmse": 0.5},
                "xgb": {"r2": 0.72, "rmse": 0.55},
            },
            "final_ensemble": {"r2": 0.81, "rmse": 0.42},
        },
        producer="test",
    )
    store.write_table(
        "2",
        "oof_predictions",
        pd.DataFrame({"rf": [1.0, 2.0], "xgb": [1.1, 1.9]}),
        producer="test",
    )

    # Stage 3 — causal coefficients, dose-response, sensitivity, diagnostics, NUTS, CATE
    store.write_struct(
        "3",
        "scenario_coefficients",
        {
            "effects": {
                "Pct_Canopy": {"coefficient": -0.05, "ci_lower": -0.08, "ci_upper": -0.02},
            }
        },
        producer="test",
    )
    store.write_struct(
        "3",
        "dose_response_curves",
        {
            "Pct_Canopy": {
                "x": [0.0, 0.5, 1.0],
                "y_mean": [0.0, -0.025, -0.05],
                "y_lower": [-0.01, -0.04, -0.07],
                "y_upper": [0.01, -0.01, -0.03],
            }
        },
        producer="test",
    )
    store.write_struct(
        "3",
        "causal_diagnostics",
        {
            "calibration": {"r2": 0.7, "n": 100},
            "rate": {"score": 0.05},
        },
        producer="test",
    )
    store.write_struct(
        "3",
        "nuts_summary",
        {"acceptance_rate": 0.85, "n_divergences": 0},
        producer="test",
    )
    store.write_table(
        "3",
        "parameter_posteriors",
        pd.DataFrame({"param": ["a", "b"], "mean": [0.1, -0.05], "sd": [0.01, 0.02]}),
        producer="test",
    )
    store.write_table(
        "3",
        "convergence_diagnostics",
        pd.DataFrame({"param": ["a", "b"], "r_hat": [1.0, 1.0], "ess": [400, 400]}),
        producer="test",
    )
    # CATE summary — long format
    cate_df = pd.DataFrame(
        [
            {
                "cell_id": i,
                "treatment": "Pct_Canopy",
                "cate_mean": -0.05 * (i + 1),
                "cate_std": 0.01,
                "cate_ci5": -0.05 * (i + 1) - 0.02,
                "cate_ci50": -0.05 * (i + 1),
                "cate_ci95": -0.05 * (i + 1) + 0.02,
                "multiplier_mean": 1.0 + 0.1 * i,
                "multiplier_ci5": 0.9,
                "multiplier_ci95": 1.1,
                "source": "bayesian_nuts",
            }
            for i in range(3)
        ]
    )
    store.write_table("3", "cate_summary", cate_df, producer="test")

    # Stage 4 — scenario results (geo, w/ baseline + scenario columns), summary, attribution, trajectory
    scen_results = _make_gdf(
        3,
        extra_cols={
            PRED_BASELINE_COL: [10.0, 11.0, 12.0],
            f"{PRED_SCENARIO_PREFIX}1": [9.0, 10.5, 11.5],
            f"{PRED_SCENARIO_PREFIX}2": [9.5, 11.0, 12.0],
            "Variable": ["Pct_Canopy"] * 3,
            "Increment": [10.0, 10.0, 10.0],
            "Scenario": ["Scenario1"] * 3,
            "pred_mean": [9.25, 10.75, 11.75],
            "pred_std": [0.1, 0.1, 0.1],
            "pred_p05": [9.1, 10.6, 11.6],
            "pred_p95": [9.4, 10.9, 11.9],
        },
    )
    store.write_table(
        "4",
        "scenario_results",
        scen_results,
        geometry_col="geometry",
        crs="EPSG:4326",
        producer="test",
    )
    store.write_table(
        "4",
        "scenario_summary",
        pd.DataFrame(
            [
                {"Scenario": "Scenario1", "Variable": "Pct_Canopy", "Increment": 10.0, "delta_mean": -0.5},
                {"Scenario": "Scenario2", "Variable": "Pct_Canopy", "Increment": 5.0, "delta_mean": -0.25},
            ]
        ),
        producer="test",
    )
    store.write_table(
        SCENARIO_STAGE,
        SCENARIO_ATTRIBUTION,
        pd.DataFrame(
            [
                {"scenario_id": 0, "variable": "Pct_Canopy", "contribution": -0.5},
                {"scenario_id": 0, "variable": "albedo", "contribution": -0.2},
                {"scenario_id": 1, "variable": "Pct_Canopy", "contribution": -0.7},
            ]
        ),
        producer="test",
    )
    store.write_table(
        SCENARIO_STAGE,
        SCENARIO_TRAJECTORY,
        pd.DataFrame(
            [
                {"scenario_id": 0, "t": 0, "geometry_id": "g0", "value": 1.0},
                {"scenario_id": 0, "t": 1, "geometry_id": "g0", "value": 1.5},
                {"scenario_id": 1, "t": 0, "geometry_id": "g0", "value": 2.0},
            ]
        ),
        producer="test",
    )
    store.write_struct(
        SCENARIO_STAGE,
        SCENARIO_LIBRARY,
        {"entries": [{"id": "lib1", "label": "Baseline", "parent": None, "t": 0}]},
        producer="test",
    )


# ---------------------------------------------------------------------------
# Sentinel sanity check
# ---------------------------------------------------------------------------


def test_sentinel_catches_disk_reads(forbid_disk_reads, tmp_path):
    """Self-test: the sentinel must raise when something reads a Stage_* path."""
    fake = tmp_path / "Stage_2_Spatial_CV" / "leak.csv"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("a,b\n1,2\n")
    with pytest.raises(AssertionError, match="disk read leaked"):
        pd.read_csv(fake)


def test_sentinel_allows_non_stage_reads(forbid_disk_reads, tmp_path):
    """Reads that don't target Stage_* paths must pass through."""
    safe = tmp_path / "input.csv"
    safe.write_text("a,b\n1,2\n")
    df = pd.read_csv(safe)
    assert df.shape == (1, 2)


# ---------------------------------------------------------------------------
# Phase B — live visualization endpoints
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_with_seed(seeded_state):
    """Seed every viz artifact and return a TestClient."""
    store = ArtifactStore(seeded_state)
    seed_all_stage_artifacts(store)
    with TestClient(app_module.app) as client:
        yield client, seeded_state


def _spy_read_table():
    """Return ``patch.object`` context capturing ArtifactStore.read_table calls."""
    return patch.object(
        ArtifactStore, "read_table", autospec=True, side_effect=ArtifactStore.read_table
    )


@pytest.mark.parametrize(
    "endpoint,must_contain",
    [
        ("/results/correlogram", "individual_results"),
        ("/results/gwen", "rows"),
        ("/results/spatial_cv/predictions", "FeatureCollection"),
        ("/results/causal", "effects"),
        ("/results/causal/dose_response", "Pct_Canopy"),
        ("/results/causal/diagnostics", "calibration"),
        ("/results/scenarios/nuts_summary", "acceptance_rate"),
        ("/results/scenarios/attribution", "records"),
        ("/results/scenarios/trajectory", "records"),
        ("/results/model_performance", "models"),
        ("/results/scenario_library/timeline", "entries"),
    ],
)
def test_viz_endpoint_reads_from_db_only(
    client_with_seed, forbid_disk_reads, endpoint, must_contain
):
    client, _reg = client_with_seed
    resp = client.get(endpoint)
    assert resp.status_code == 200, f"{endpoint}: {resp.status_code} / {resp.text[:200]}"
    text = resp.text
    assert must_contain in text, f"{endpoint}: missing {must_contain!r} in body"
    assert not forbid_disk_reads, f"{endpoint}: disk leaks {forbid_disk_reads}"


def test_cate_map_reads_from_db_only(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.get("/results/causal/cate_map?variable=Pct_Canopy")
    assert resp.status_code == 200, resp.text
    fc = resp.json()
    assert fc["type"] == "FeatureCollection"
    assert fc["variable"] == "Pct_Canopy"
    assert len(fc["features"]) == 3
    assert not forbid_disk_reads


def test_cate_map_variables_reads_from_db_only(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.get("/results/causal/cate_map/variables")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "Pct_Canopy" in (body.get("variables") or body)
    assert not forbid_disk_reads


def test_scenarios_list_reads_from_db_only(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.get("/results/scenarios")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "scenarios" in body
    assert body["scenarios"], "expected scenario rows from seeded data"
    assert not forbid_disk_reads


def test_scenarios_uncertainty_reads_from_db_only(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.get("/results/scenarios/uncertainty")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "geojson" in body
    assert not forbid_disk_reads


def test_results_manifest_reads_from_db_only(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.get("/results/manifest")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "stages" in body
    # Every seeded stage should appear in the manifest.
    assert {"0", "1", "2", "3", "4"}.issubset(body["stages"].keys())
    assert not forbid_disk_reads


def test_results_manifest_per_stage_reads_from_db_only(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.get("/results/manifest/3")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["stage"] == "3"
    assert "cate_summary" in body["artifacts"]
    assert not forbid_disk_reads


def test_results_availability_reflects_db_state(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.get("/results/availability")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["/results/correlogram"]["available"] is True
    assert body["/results/gwen"]["available"] is True
    assert body["/results/causal"]["available"] is True
    assert not forbid_disk_reads


def test_viz_endpoint_uses_artifact_store(client_with_seed, forbid_disk_reads):
    """Spy on ArtifactStore.read_table to confirm it's actually invoked."""
    client, _ = client_with_seed
    with _spy_read_table() as spy:
        resp = client.get("/results/scenarios/attribution")
    assert resp.status_code == 200
    assert spy.call_count >= 1, "Expected ArtifactStore.read_table to be called"


# Negative test — empty store, no disk fallback allowed.
def test_viz_endpoints_404_when_db_empty_and_no_disk_reads(
    seeded_state, forbid_disk_reads
):
    """With nothing in the DB, every viz endpoint must 404 — never hit disk."""
    with TestClient(app_module.app) as client:
        for endpoint in [
            "/results/correlogram",
            "/results/gwen",
            "/results/spatial_cv/predictions",
            "/results/causal",
            "/results/causal/dose_response",
            "/results/causal/diagnostics",
            "/results/scenarios/nuts_summary",
            "/results/scenarios/attribution",
            "/results/scenarios/trajectory",
            "/results/model_performance",
        ]:
            resp = client.get(endpoint)
            assert resp.status_code in (404, 503), (
                f"{endpoint}: expected 404/503, got {resp.status_code}"
            )
    assert not forbid_disk_reads


# ---------------------------------------------------------------------------
# Phase C — artifact native / render / download endpoints
# ---------------------------------------------------------------------------


def test_artifact_native_table_returns_csv(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.get("/artifacts/3/cate_summary")
    assert resp.status_code == 200, resp.text
    assert "text/csv" in resp.headers["content-type"]
    body = resp.text
    assert "cate_mean" in body and "Pct_Canopy" in body
    assert not forbid_disk_reads


def test_artifact_native_struct_returns_json(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.get("/artifacts/3/scenario_coefficients")
    assert resp.status_code == 200, resp.text
    assert "json" in resp.headers["content-type"]
    body = json.loads(resp.text)
    assert "effects" in body
    assert not forbid_disk_reads


def test_artifact_csv_endpoint(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.get("/artifacts/1/gwen_variable_importance.csv")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/csv")
    df = pd.read_csv(io.StringIO(resp.text))
    assert {"variable", "importance"}.issubset(df.columns)
    assert not forbid_disk_reads


def test_artifact_json_endpoint(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.get("/artifacts/0/correlogram_results.json")
    assert resp.status_code == 200, resp.text
    body = json.loads(resp.text)
    assert "individual_results" in body
    assert not forbid_disk_reads


def test_artifact_geojson_endpoint(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.get("/artifacts/2/spatial_cv_predictions.geojson")
    assert resp.status_code == 200, resp.text
    assert "geo+json" in resp.headers["content-type"]
    body = json.loads(resp.text)
    assert body["type"] == "FeatureCollection"
    assert len(body["features"]) == 3
    assert not forbid_disk_reads


def test_artifact_download_csv_format(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.get("/artifacts/3/cate_summary/download?fmt=csv")
    assert resp.status_code == 200, resp.text
    assert "text/csv" in resp.headers["content-type"]
    assert "cate_mean" in resp.text
    assert not forbid_disk_reads


def test_artifact_download_json_for_struct(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.get("/artifacts/3/scenario_coefficients/download?fmt=json")
    assert resp.status_code == 200, resp.text
    body = json.loads(resp.text)
    assert "effects" in body
    assert not forbid_disk_reads


def test_artifact_download_404_for_missing(seeded_state, forbid_disk_reads):
    with TestClient(app_module.app) as client:
        resp = client.get("/artifacts/3/does_not_exist/download")
    assert resp.status_code == 404
    assert not forbid_disk_reads


def test_stage_export_zip_contains_all_artifacts(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.post("/artifacts/3/export")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] in ("application/zip", "application/x-zip-compressed")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = {n.split("/")[-1] for n in zf.namelist()}
    # Stage 3 had: scenario_coefficients, dose_response_curves, causal_diagnostics,
    # nuts_summary, parameter_posteriors, convergence_diagnostics, cate_summary
    assert any("cate_summary" in n for n in names)
    assert any("scenario_coefficients" in n for n in names)
    assert not forbid_disk_reads


def test_artifacts_list_for_stage(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.get("/artifacts/3")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    ids = {a["artifact_id"] for a in body["artifacts"]}
    assert {"cate_summary", "scenario_coefficients", "dose_response_curves"}.issubset(ids)
    assert not forbid_disk_reads


# ---------------------------------------------------------------------------
# Phase D — report & bundle endpoints
# ---------------------------------------------------------------------------


def test_results_bundle_zip_uses_db_only(client_with_seed, forbid_disk_reads, tmp_path):
    client, _ = client_with_seed
    resp = client.get("/results/bundle")
    assert resp.status_code == 200, resp.text
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        assert "manifest.json" in names
        # Tables → csv or geojson; structs → json. At least one of each.
        assert any(n.endswith(".csv") for n in names)
        assert any(n.endswith(".json") for n in names)
        assert any(n.startswith("stage_4/scenario_results") for n in names)
    # And no Stage_* dirs were ever created on disk.
    for stage_name in _stage_dir_names():
        assert not (tmp_path / stage_name).exists(), (
            f"{stage_name} was created on disk despite db-only policy"
        )
    assert not forbid_disk_reads


def test_results_export_block_persists_to_registry(
    seeded_state, project_config, monkeypatch, tmp_path
):
    """POST /results/export round-trips a PNG screenshot into the registry."""
    import base64

    # Tiny 1x1 PNG.
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xfc\xff"
        b"\xff?\x03\x00\x06\x05\x02\x00\xa1\xfa\xc8\xc4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    payload = {
        "artifact_id": "test_chart",
        "stage": "export",
        "label": "smoke",
        "png_b64": base64.b64encode(png_bytes).decode("ascii"),
    }
    with TestClient(app_module.app) as client:
        resp = client.post("/results/export", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["registered"] is True
    # Verify the entry shows up in the manifest.
    found = False
    for stage_obj in seeded_state.manifest.stages.values():
        for art in stage_obj.artifacts.values():
            if art.id.startswith("export::test_chart::"):
                found = True
                break
    assert found, "exported PNG not registered in manifest"


def test_report_generate_markdown(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.post("/report/generate?format=markdown")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "markdown" in body
    assert "SPARC Pipeline Report" in body["markdown"]
    assert not forbid_disk_reads


def test_report_generate_json(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.post("/report/generate?format=json")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "report" in body and "markdown" in body
    assert body["report"].get("project", {}).get("name") == "pytest"
    assert not forbid_disk_reads


def test_report_pdf_routes_through_registry(
    client_with_seed, forbid_disk_reads, monkeypatch
):
    """POST /report/pdf must pass the active registry to generate_report_pdf."""
    client, reg = client_with_seed
    captured: dict[str, Any] = {}

    def fake_pdf(*, config, data_summary, causal_results, scenario_summary,
                 output_dir, registry):
        captured["registry"] = registry
        captured["output_dir"] = output_dir
        return b"%PDF-1.4 fake\n"

    import sparc.report as report_mod

    monkeypatch.setattr(report_mod, "generate_report_pdf", fake_pdf)
    resp = client.post("/report/pdf")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content.startswith(b"%PDF-1.4")
    assert captured["registry"] is reg
    assert not forbid_disk_reads


def test_report_pdf_falls_back_to_html(client_with_seed, forbid_disk_reads, monkeypatch):
    """When WeasyPrint is missing, /report/pdf falls back to HTML."""
    client, _ = client_with_seed

    def boom(**_kw):
        raise RuntimeError("WeasyPrint not installed")

    def fake_html(**_kw):
        return "<html><body>fallback</body></html>"

    import sparc.report as report_mod

    monkeypatch.setattr(report_mod, "generate_report_pdf", boom)
    monkeypatch.setattr(report_mod, "generate_report_html", fake_html)
    resp = client.post("/report/pdf")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("status") == "html_fallback"
    assert not forbid_disk_reads


def test_report_data_endpoint_uses_db_only(client_with_seed, forbid_disk_reads):
    """/results/report aggregates state from artifacts.db without disk reads."""
    client, _ = client_with_seed
    resp = client.get("/results/report")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["project"].get("name") == "pytest"
    assert "scenario_summary" in body
    assert not forbid_disk_reads


# ---------------------------------------------------------------------------
# Phase E — pre-fix xfail(strict=True) tests for the gap endpoints
# ---------------------------------------------------------------------------
#
# These tests start as XFAIL because the endpoints currently bypass
# ArtifactStore. Once Phases F+G land they auto-flip to PASS — and any
# future regression that re-introduces a disk read fails CI (strict=True).
# ---------------------------------------------------------------------------


def test_scenarios_variables_uses_db_only(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.get("/results/scenarios/variables")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "Pct_Canopy" in body["variables"]
    assert not forbid_disk_reads, f"disk leaks: {forbid_disk_reads}"


def test_scenarios_increment_uses_db_only(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.get(
        "/results/scenarios/increment?variable=Pct_Canopy&increment=10"
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "geojson" in body and "summary" in body
    assert not forbid_disk_reads, f"disk leaks: {forbid_disk_reads}"


def test_results_artifacts_uses_db_only(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.get("/results/artifacts")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Once migrated, artifacts come from the manifest — should be > 0 even
    # though no Stage_* dir exists on disk.
    assert body["total"] > 0
    assert not forbid_disk_reads, f"disk leaks: {forbid_disk_reads}"


def test_results_download_uses_db_only(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.get("/results/download/3/cate_summary.csv")
    # Once migrated, this should proxy through ArtifactStore.export.
    assert resp.status_code == 200, resp.text
    assert "cate_mean" in resp.text
    assert not forbid_disk_reads


def test_results_geopackage_uses_db_only(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.get("/results/geopackage")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/")
    assert resp.content[:4] == b"SQLi"  # GeoPackage = SQLite header "SQLite f"
    assert not forbid_disk_reads, f"disk leaks: {forbid_disk_reads}"


def test_report_docx_endpoint_streams_doc(client_with_seed, forbid_disk_reads):
    client, _ = client_with_seed
    resp = client.post("/report/docx")
    assert resp.status_code == 200, resp.text
    ctype = resp.headers["content-type"]
    assert "wordprocessingml" in ctype or "officedocument" in ctype
    # DOCX is a ZIP with PK signature.
    assert resp.content[:2] == b"PK"
    assert not forbid_disk_reads


# This one stays xfail forever: /report/data_bundle is removed from the
# frontend in Phase G and never implemented server-side.
@pytest.mark.xfail(
    strict=True,
    reason="/report/data_bundle is intentionally not implemented; UI uses /results/bundle instead",
)
def test_report_data_bundle_does_not_exist(seeded_state):
    with TestClient(app_module.app) as client:
        resp = client.post("/report/data_bundle")
    # Strict xfail: we *expect* this assertion to fail (route is 404).
    # If someone ever implements the endpoint, the test PASSES, the strict
    # xfail flips to XPASS, and CI fails — forcing a deliberate decision.
    assert resp.status_code == 200
