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


# ---------------------------------------------------------------------------
# /results/causal/cate_map  — Bayesian β(s) (v4)
# ---------------------------------------------------------------------------


def _seed_cate_summary(registry, treatment: str = "Pct_Canopy"):
    """Write a minimal long-format cate_summary table for tests."""
    df = pd.DataFrame(
        [
            {
                "cell_id": i,
                "treatment": treatment,
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
    ArtifactStore(registry).write_table("3", "cate_summary", df)
    return df


def _attach_geometry(monkeypatch, n: int = 3):
    """Attach a tiny GeoDataFrame to ``state.data`` so the CATE endpoint can join."""
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {"_": list(range(n))},
        geometry=[Point(i, i) for i in range(n)],
        crs="EPSG:4326",
    )
    monkeypatch.setattr(app_module.state, "data", gdf, raising=False)
    monkeypatch.setattr(app_module.state, "project_config", {"loaded": True}, raising=False)


def test_cate_map_returns_coef_property(attached_registry, monkeypatch):
    _seed_cate_summary(attached_registry)
    _attach_geometry(monkeypatch)
    with TestClient(app_module.app) as client:
        resp = client.get("/results/causal/cate_map?variable=Pct_Canopy")
    assert resp.status_code == 200, resp.text
    fc = resp.json()
    assert fc["type"] == "FeatureCollection"
    assert fc["variable"] == "Pct_Canopy"
    assert "units" in fc and "ΔY per unit T" in fc["units"]
    assert fc["source"] == "bayesian_nuts"
    assert fc["with_uncertainty"] is False
    feats = fc["features"]
    assert len(feats) == 3
    p0 = feats[0]["properties"]
    # Canonical and back-compat property names — both should equal cate_mean.
    assert p0["coef_Pct_Canopy"] == pytest.approx(-0.05)
    assert p0["cate_Pct_Canopy"] == pytest.approx(-0.05)


def test_cate_map_with_uncertainty_includes_ci_and_ns(attached_registry, monkeypatch):
    _seed_cate_summary(attached_registry)
    _attach_geometry(monkeypatch)
    with TestClient(app_module.app) as client:
        resp = client.get(
            "/results/causal/cate_map?variable=Pct_Canopy&with_uncertainty=1"
        )
    assert resp.status_code == 200, resp.text
    fc = resp.json()
    assert fc["with_uncertainty"] is True
    p = fc["features"][0]["properties"]
    assert "coef_ci5_Pct_Canopy" in p
    assert "coef_ci95_Pct_Canopy" in p
    assert "coef_ns_Pct_Canopy" in p
    # All seeded CIs are negative (mean - 0.02 .. mean + 0.02 with mean ≤ -0.05),
    # so the CI does not bracket zero — coef_ns must be False.
    assert p["coef_ns_Pct_Canopy"] is False


def test_local_coefficients_endpoint_removed():
    """The legacy MGWR-coefficient endpoints are gone in v4."""
    with TestClient(app_module.app) as client:
        r1 = client.get("/results/local_coefficients?variable=Pct_Canopy")
        r2 = client.get("/results/local_coefficients/variables")
    # 404 = no route; 422 = matched a typed route (e.g. ``/results/{stage:int}``)
    # whose validation rejects the path. Either confirms the legacy endpoint
    # is gone.
    assert r1.status_code in (404, 422)
    assert r2.status_code in (404, 422)


# ---------------------------------------------------------------------------
# /scenarios/budget/optimize — benefit_source="cate" reads β(s) from DB
# ---------------------------------------------------------------------------


def test_budget_resolve_benefits_cate_reads_cate_mean(attached_registry, monkeypatch):
    _seed_cate_summary(attached_registry)
    _attach_geometry(monkeypatch)
    benefits, desc = app_module._resolve_benefits("cate", "Pct_Canopy")
    assert benefits.shape == (3,)
    assert benefits.tolist() == pytest.approx([-0.05, -0.10, -0.15])
    assert "Bayesian local coefficient" in desc


def test_budget_resolve_benefits_uniform_returns_ones(attached_registry, monkeypatch):
    _attach_geometry(monkeypatch, n=4)
    benefits, desc = app_module._resolve_benefits("uniform", None)
    assert benefits.shape == (4,)
    assert benefits.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert "uniform" in desc.lower()


def test_budget_resolve_benefits_rejects_unknown_source(monkeypatch):
    monkeypatch.setattr(app_module.state, "project_config", {"loaded": True}, raising=False)
    with pytest.raises(Exception):
        app_module._resolve_benefits("local_coef", "Pct_Canopy")
