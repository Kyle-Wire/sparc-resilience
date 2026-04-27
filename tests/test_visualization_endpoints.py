"""Phase 7 — server tests for visualization endpoints + bundle download."""

from __future__ import annotations

import io
import json
import zipfile

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from sparc.registry.run_registry import RunRegistry
from sparc.registry.store import ArtifactStore
from sparc.scenario import (
    SCENARIO_ATTRIBUTION,
    SCENARIO_STAGE,
    SCENARIO_TRAJECTORY,
)
from sparc.server import app as app_module


@pytest.fixture()
def attached_registry(tmp_path, monkeypatch):
    reg = RunRegistry(tmp_path)
    reg.add_register_listener(app_module._on_artifact_registered)
    monkeypatch.setattr(app_module.state, "registry", reg, raising=False)
    yield reg
    reg.remove_register_listener(app_module._on_artifact_registered)


# ---------------------------------------------------------------------------
# /results/scenarios/attribution
# ---------------------------------------------------------------------------


def test_attribution_endpoint_returns_records(attached_registry):
    df = pd.DataFrame(
        [
            {"scenario_id": 0, "variable": "temp", "contribution": 0.5},
            {"scenario_id": 0, "variable": "albedo", "contribution": -0.2},
            {"scenario_id": 1, "variable": "temp", "contribution": 0.7},
        ]
    )
    ArtifactStore(attached_registry).write_table(SCENARIO_STAGE, SCENARIO_ATTRIBUTION, df)

    with TestClient(app_module.app) as client:
        resp = client.get("/results/scenarios/attribution")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert "records" in payload and "columns" in payload
    assert len(payload["records"]) == 3
    assert {"scenario_id", "variable", "contribution"}.issubset(payload["columns"])


def test_attribution_endpoint_404_when_missing(attached_registry):
    with TestClient(app_module.app) as client:
        resp = client.get("/results/scenarios/attribution")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /results/scenarios/trajectory
# ---------------------------------------------------------------------------


def test_trajectory_endpoint_filters_by_scenario_id(attached_registry):
    df = pd.DataFrame(
        [
            {"scenario_id": 0, "t": 0, "geometry_id": "g0", "value": 1.0},
            {"scenario_id": 0, "t": 1, "geometry_id": "g0", "value": 1.5},
            {"scenario_id": 1, "t": 0, "geometry_id": "g0", "value": 2.0},
        ]
    )
    ArtifactStore(attached_registry).write_table(SCENARIO_STAGE, SCENARIO_TRAJECTORY, df)

    with TestClient(app_module.app) as client:
        resp_all = client.get("/results/scenarios/trajectory")
        resp_one = client.get("/results/scenarios/trajectory?scenario_id=1")

    assert resp_all.status_code == 200
    assert len(resp_all.json()["records"]) == 3
    assert resp_one.status_code == 200
    one_records = resp_one.json()["records"]
    assert len(one_records) == 1
    assert one_records[0]["scenario_id"] == 1


# ---------------------------------------------------------------------------
# /results/bundle
# ---------------------------------------------------------------------------


def test_results_bundle_includes_manifest_and_artifacts(attached_registry, monkeypatch):
    # Bundle endpoint calls _ensure_registry_attached(); set a truthy stub.
    monkeypatch.setattr(app_module.state, "project_config", object(), raising=False)
    store = ArtifactStore(attached_registry)
    store.write_table("0", "preview", pd.DataFrame({"a": [1, 2, 3]}))
    store.write_struct("3", "diag", {"k": 1, "v": 2})

    with TestClient(app_module.app) as client:
        resp = client.get("/results/bundle")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/")

    buf = io.BytesIO(resp.content)
    with zipfile.ZipFile(buf) as zf:
        names = zf.namelist()
        assert "manifest.json" in names
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert "stages" in manifest
        assert any(n.startswith("stage_0/preview") for n in names)
        assert any(n.startswith("stage_3/diag") for n in names)


def test_results_bundle_404_without_registry(monkeypatch):
    """With no project loaded, bundle endpoint should return 4xx (not 200)."""
    monkeypatch.setattr(app_module.state, "registry", None, raising=False)
    monkeypatch.setattr(app_module.state, "project_config", None, raising=False)
    with TestClient(app_module.app) as client:
        resp = client.get("/results/bundle")
    assert resp.status_code in (400, 404, 503)
