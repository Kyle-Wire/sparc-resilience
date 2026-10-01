"""Projects API (api.md §5–5.3): templates, readiness, CRUD, files, data check, config round trip, impact."""

from __future__ import annotations

import gzip
import importlib.util
import json
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.studio.conftest import wait_for
from tests.studio.projects.conftest import create

SPINE_OK = ("data", "columns", "levers", "roles", "config_valid")


def spine(client, pid) -> dict:
    r = client.get(f"/api/projects/{pid}")
    assert r.status_code == 200, r.text
    return {row["key"]: row for row in r.json()["readiness"]}


def put_raw(client, pid, raw, version=None, **kw):
    headers = {"If-Match": str(version)} if version is not None else {}
    return client.put(f"/api/projects/{pid}/config", json={"raw": raw, **kw}, headers=headers)


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------

def test_synthetic_demo_template(client, demo, tmp_path):
    from sparc.core.synthetic import write_demo_project

    assert demo["demo"] is True and demo["template"] == "synthetic_demo"
    pdir = Path(demo["dir"])
    ref = tmp_path / "ref"
    write_demo_project(ref, n=40, seed=0)
    for rel in ("data/city.csv", "data/city.truth.json", "inputs/layers/demo_layers.parquet",
                "inputs/climate/demo_cmip6.csv", "config.yml"):
        assert (pdir / rel).read_bytes() == (ref / rel).read_bytes(), rel
    raw = client.get(f"/api/projects/{demo['id']}/config").json()["raw"]
    assert raw["data"]["crs"] == "EPSG:32619"
    rep = client.post(f"/api/projects/{demo['id']}/config/validate", json={}).json()
    assert rep["ok"] is True and [i for i in rep["issues"] if i["level"] == "error"] == []
    rows = spine(client, demo["id"])
    for key in SPINE_OK:
        assert rows[key]["state"] == "ok", rows[key]
    assert rows["climate_table"]["state"] == "ok" and rows["people_layers"]["state"] == "ok"
    assert rows["runs"]["state"] == "missing" and rows["runs"]["action"]["path"] == f"/p/{demo['id']}/launch"
    assert demo["readiness_score"]["done"] >= 7


def test_blank_template(client):
    p = create(client, "Blank one", "blank")["project"]
    text = Path(p["config_path"]).read_text()
    assert text.startswith("core:")
    rows = spine(client, p["id"])
    assert rows["data"]["state"] == "missing" and rows["config_valid"]["state"] == "missing"
    assert rows["data"]["action"]["path"] == f"/p/{p['id']}/setup/data"
    issues = client.post(f"/api/projects/{p['id']}/config/validate").json()["issues"]
    assert {"data.path", "data.target", "predictors"} <= {i["path"] for i in issues if i["level"] == "error"}


def test_slug_conflict_and_bad_template(client):
    create(client, "Twin", "blank")
    r = client.post("/api/projects", json={"name": "twin", "template": "blank"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
    r = client.post("/api/projects", json={"name": "x", "template": "nope"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation"


def test_providence_example_packaged(client, monkeypatch):
    from sparc.studio.projects import templates

    monkeypatch.setattr(templates, "repo_root", lambda: None)
    out = create(client, "Providence", "providence_example")
    p, warnings = out["project"], out["warnings"]
    assert out["imported_runs"] == [] and warnings == []
    pdir = Path(p["dir"])
    with gzip.open(templates.EXAMPLE_DIR / "brown4.csv.gz", "rb") as f:
        assert (pdir / "data" / "brown4.csv").read_bytes() == f.read()
    for rel in ("inputs/forcing/providence_2020-07-29.json", "inputs/climate/providence_cmip6_tasmax_jja.csv",
                "inputs/layers/providence_layers.parquet"):
        assert (pdir / rel).is_file(), rel
    text = (pdir / "config.yml").read_text()
    assert "# SPARC core pipeline" in text            # comments kept by the in-place path rewrite
    raw = yaml.safe_load(text)["core"]
    assert raw["data"]["path"] == "data/brown4.csv"
    assert raw["physics"]["forcing"] == "inputs/forcing/providence_2020-07-29.json"
    assert raw["climate"]["table"] == "inputs/climate/providence_cmip6_tasmax_jja.csv"
    assert raw["planner"]["layers"] == "inputs/layers/providence_layers.parquet"
    assert raw["output"]["dir"] == "runs"
    rep = client.post(f"/api/projects/{p['id']}/config/validate").json()
    assert [i for i in rep["issues"] if i["level"] == "error"] == []
    rows = spine(client, p["id"])
    assert all(rows[k]["state"] == "ok" for k in SPINE_OK + ("forcing", "climate_table", "people_layers"))


def test_providence_example_unavailable(client, monkeypatch):
    from sparc.studio.projects import templates

    monkeypatch.setattr(templates, "providence_sources", lambda: None)
    r = client.post("/api/projects", json={"name": "P", "template": "providence_example"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "example_unavailable"
    assert client.get("/api/projects").json() == []


def test_providence_import_without_run_registry(client, monkeypatch, tmp_path):
    """A checkout with runs, but no runs registry installed: the project is created with a warning."""
    from sparc.studio.projects import templates

    src = templates.providence_sources()
    fake_root = tmp_path / "checkout"
    for name in ("providence_uhi_fast", "placebo", "simcheck_a"):
        (fake_root / "output" / "core" / "providence" / name).mkdir(parents=True)
    monkeypatch.setattr(templates, "providence_sources", lambda: {**src, "root": fake_root})
    seen = []

    async def missing(sctx, module, func, *args, **kwargs):
        seen.append((module, func, args[0].name if args else None))
        return templates.MISSING

    monkeypatch.setattr(templates, "call_contract", missing)
    out = create(client, "Providence", "providence_example")
    assert out["imported_runs"] == []
    assert any(templates.UNAVAILABLE in w for w in out["warnings"])
    assert ("sparc.studio.runs.registry", "import_run", "providence_uhi_fast") in seen
    assert ("sparc.studio.studies.service", "import_study_dir", "simcheck_a") in seen


@pytest.mark.skipif(importlib.util.find_spec("sparc.studio.runs") is None
                    or importlib.util.find_spec("sparc.studio.runs.registry") is None,
                    reason="sparc.studio.runs.registry is not installed (backend-runs)")
def test_providence_example_in_repo_imports_runs(client):
    from sparc.studio.projects import templates

    root = templates.repo_root()
    if root is None or not (root / "output" / "core" / "providence").is_dir():
        pytest.skip("no output/core/providence in this checkout")
    out = create(client, "Providence", "providence_example", import_existing_runs=True)
    assert out["imported_runs"], out["warnings"]
    runs = client.get(f"/api/projects/{out['project']['id']}").json()["runs"]
    assert {r["id"] for r in runs} >= set(out["imported_runs"])


def test_call_contract(ctx):
    import asyncio

    from sparc.studio.projects.templates import MISSING, call_contract

    assert asyncio.run(call_contract(ctx, "sparc.studio.no_such_module", "f")) is MISSING
    mod = types.ModuleType("sparc_test_contract")

    def sync_fn(d, pid, sctx=None, config_path=None):
        return {"id": f"{d}:{pid}", "sctx": sctx is ctx, "config_path": config_path}

    async def async_fn(d, pid, kind=None):
        return {"id": f"{d}:{pid}:{kind}"}

    mod.sync_fn, mod.async_fn = sync_fn, async_fn
    sys.modules["sparc_test_contract"] = mod
    try:
        assert asyncio.run(call_contract(ctx, "sparc_test_contract", "sync_fn", "d", "p", config_path="c")) == \
            {"id": "d:p", "sctx": True, "config_path": "c"}
        assert asyncio.run(call_contract(ctx, "sparc_test_contract", "async_fn", "d", "p", kind="placebo")) == \
            {"id": "d:p:placebo"}
        assert asyncio.run(call_contract(ctx, "sparc_test_contract", "absent")) is MISSING
    finally:
        sys.modules.pop("sparc_test_contract", None)


# ---------------------------------------------------------------------------
# import an existing config
# ---------------------------------------------------------------------------

def _external_config(tmp_path) -> Path:
    from sparc.core.synthetic import write_demo_project

    ext = tmp_path / "elsewhere"
    write_demo_project(ext, n=24, seed=1)
    return ext / "config.yml"


def test_import_config_absolutises_paths(client, tmp_path):
    cfg = _external_config(tmp_path)
    r = client.post("/api/projects/import", json={"config_path": str(cfg), "name": "Imported"})
    assert r.status_code == 201, r.text
    body = r.json()
    raw = client.get(f"/api/projects/{body['project']['id']}/config").json()["raw"]
    assert raw["data"]["path"] == str((cfg.parent / "data" / "city.csv").resolve())
    assert raw["planner"]["layers"] == str((cfg.parent / "inputs/layers/demo_layers.parquet").resolve())
    assert body["runs"] == [] and body["studies"] == []
    rows = spine(client, body["project"]["id"])
    assert rows["data"]["state"] == "ok" and rows["config_valid"]["state"] == "ok"
    # the data folder outside the workspace is now an allowed root for path parameters
    r = client.get(f"/api/projects/{body['project']['id']}/files/inspect", params={"path": raw["data"]["path"]})
    assert r.status_code == 200 and r.json()["n_rows"] > 0


def test_import_config_copy_data(client, tmp_path):
    cfg = _external_config(tmp_path)
    r = client.post("/api/projects/import", json={"config_path": str(cfg), "name": "Copied", "copy_data": True})
    assert r.status_code == 201, r.text
    pid = r.json()["project"]["id"]
    raw = client.get(f"/api/projects/{pid}/config").json()["raw"]
    assert raw["data"]["path"] == "data/city.csv" and raw["climate"]["table"] == "inputs/climate/demo_cmip6.csv"
    pdir = Path(r.json()["project"]["dir"])
    assert (pdir / "data/city.csv").read_bytes() == (cfg.parent / "data/city.csv").read_bytes()


def test_import_config_errors(client, tmp_path):
    r = client.post("/api/projects/import", json={"config_path": str(tmp_path / "none.yml")})
    assert r.status_code == 404
    bad = tmp_path / "bad.yml"
    bad.write_text("core:\n  data: {path: x.csv, target: T}\n  predictors: []\n")
    r = client.post("/api/projects/import", json={"config_path": str(bad)})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation"
    assert any(e["path"] == "predictors" for e in r.json()["error"]["detail"]["errors"])
    assert client.get("/api/projects").json() == []


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def test_project_crud(client, ctx, demo, wait_job):
    pid = demo["id"]
    assert [p["id"] for p in client.get("/api/projects").json()] == [pid]
    r = client.patch(f"/api/projects/{pid}", json={"name": "Renamed", "report": {"place": "Nowhere"},
                                                    "headline_scenario": "canopy-increase-10",
                                                    "cost_model": {"canopy": {"per_unit": 2.5}}})
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["name"] == "Renamed" and p["report"]["place"] == "Nowhere"
    assert p["report"]["title"] == "Synthetic city (DEMO)"           # from the config's report block
    assert p["headline_scenario"] == "canopy-increase-10" and p["cost_model"]["canopy"] == {"per_unit": 2.5}
    rec = json.loads((Path(p["dir"]) / "project.json").read_text())
    assert rec["name"] == "Renamed" and rec["report"]["place"] == "Nowhere"
    assert client.patch(f"/api/projects/{pid}", json={"active_run_id": "nope"}).status_code == 404
    assert client.patch(f"/api/projects/{pid}", json={"archived": True}).json()["archived"] is True
    assert client.get("/api/projects").json() == []
    assert [p["id"] for p in client.get("/api/projects", params={"archived": True}).json()] == [pid]
    # a running job blocks deletion
    job = client.post("/api/jobs", json={"kind": "test.sleep", "params": {"seconds": 30}, "project_id": pid}).json()
    detail = client.get(f"/api/projects/{pid}").json()
    assert detail["project"]["active_jobs"] == 1 and detail["active_jobs"][0]["id"] == job["id"]
    r = client.delete(f"/api/projects/{pid}")
    assert r.status_code == 409 and r.json()["error"]["code"] == "active"
    client.post(f"/api/jobs/{job['id']}/cancel")
    wait_job(client, job["id"])
    assert client.delete(f"/api/projects/{pid}", params={"files": True}).json() == {"ok": True}
    assert not Path(p["dir"]).exists()
    assert client.get(f"/api/projects/{pid}").status_code == 404


def test_reindex_registers_project_folders(client, ctx, demo):
    from sparc.studio.db import reindex

    blank = create(client, "Gone", "blank")["project"]
    assert client.delete(f"/api/projects/{blank['id']}").status_code == 200       # files kept, marked deleted
    ctx.db.execute("DELETE FROM projects")
    summary = reindex(ctx.db, ctx.workspace)
    assert summary["projects"] == {"projects": 1}
    assert [p["id"] for p in client.get("/api/projects").json()] == [demo["id"]]
    assert client.get(f"/api/projects/{demo['id']}").json()["config_version"] >= 1


def test_detail_lists_runs_from_the_runs_table(client, ctx, demo):
    ctx.db.insert("runs", {"id": "20261001-120000-fast-a1b2", "project_id": demo["id"], "run_dir": "/tmp/x/run1",
                           "studio_dir": "/tmp/x/run1/studio", "origin": "studio", "status": "complete",
                           "mode": "fast", "created_utc": "2026-10-01T12:00:00Z",
                           "finished_utc": "2026-10-01T12:05:00Z", "r2": 0.8, "has_emulator": 0})
    d = client.get(f"/api/projects/{demo['id']}").json()
    run = d["runs"][0]
    assert run["id"] == "20261001-120000-fast-a1b2" and run["duration_s"] == 300.0 and run["mode"] == "fast"
    assert d["project"]["n_runs"] == 1 and d["project"]["last_run"]["r2"] == 0.8
    rows = {r["key"]: r for r in d["readiness"]}
    assert rows["runs"]["state"] == "warn"                            # fast only
    assert rows["emulator"]["action"]["kind"] == "build_emulator"


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

def _feet_csv() -> bytes:
    ft = 1200.0 / 3937.0
    xs, ys = np.meshgrid(np.arange(12), np.arange(10))
    df = pd.DataFrame({"OBJECTID": np.arange(xs.size), "POINT_X": 350000 + xs.ravel() * 30 / ft,
                       "POINT_Y": 260000 + ys.ravel() * 30 / ft, "AAT_z": 85 + (xs.ravel() % 7) * 0.5,
                       "Pct_Canopy": (xs.ravel() * 3) % 100, "Pct_Impervious": (ys.ravel() * 7) % 100,
                       "NDVI": 0.2 + 0.01 * xs.ravel(), "Location": ys.ravel() % 3})
    return df.to_csv(index=False).encode()


def test_files_upload_inspect_suggest_delete(client, demo):
    pid = demo["id"]
    body = _feet_csv()
    r = client.put(f"/api/projects/{pid}/files/data/heat.csv", content=body)
    assert r.status_code == 201, r.text
    up = r.json()
    assert up["path"] == "data/heat.csv" and up["bytes"] == len(body) and up["kind"] == "data"
    assert up["inspect"]["n_rows"] == 120 and up["inspect"]["n_rows_exact"] is True
    cols = {c["name"]: c for c in up["inspect"]["columns"]}
    assert cols["AAT_z"]["min"] == 85.0 and cols["Location"]["n_unique"] == 3
    assert len(up["inspect"]["preview"]) == 20
    r = client.put(f"/api/projects/{pid}/files/data/heat.csv", content=body)
    assert r.status_code == 409 and r.json()["error"]["code"] == "exists"
    assert client.put(f"/api/projects/{pid}/files/data/heat.csv", content=body,
                      headers={"X-Overwrite": "1"}).status_code == 201
    assert client.put(f"/api/projects/{pid}/files/data/run.exe", content=b"x").status_code == 415
    assert client.put(f"/api/projects/{pid}/files/bogus/a.csv", content=b"x").status_code == 422
    r = client.get(f"/api/projects/{pid}/files/inspect", params={"path": "data/heat.csv", "rows": 50})
    assert r.status_code == 200 and r.json()["n_rows"] == 120
    assert client.get(f"/api/projects/{pid}/files/inspect", params={"path": "../x.csv"}).status_code == 422
    s = client.post(f"/api/projects/{pid}/columns/suggest", json={"path": "data/heat.csv"}).json()
    assert (s["target"], s["id"], s["x"], s["y"], s["zone"]) == ("AAT_z", "OBJECTID", "POINT_X", "POINT_Y", "Location")
    assert s["coord_unit"] == "us_survey_foot" and s["crs_guess"] is None
    assert s["roles"] == {"canopy": "Pct_Canopy", "impervious": "Pct_Impervious", "ndvi": "NDVI"}
    assert "AAT_z" not in s["predictors"] and "NDVI" in s["predictors"]
    files = {f["path"]: f for f in client.get(f"/api/projects/{pid}/files").json()}
    assert files["data/city.csv"]["used_by"] == ["data.path"] and files["data/heat.csv"]["used_by"] == []
    r = client.delete(f"/api/projects/{pid}/files", params={"path": "data/city.csv"})
    assert r.status_code == 409 and r.json()["error"]["detail"]["used_by"] == ["data.path"]
    assert client.delete(f"/api/projects/{pid}/files", params={"path": "data/heat.csv"}).json() == {"ok": True}
    assert "data/heat.csv" not in {f["path"] for f in client.get(f"/api/projects/{pid}/files").json()}


def test_suggest_lonlat(client, demo):
    df = pd.DataFrame({"lon": [-71.40, -71.39, -71.38], "lat": [41.82, 41.83, 41.84], "temp_f": [85.0, 86.0, 87.0]})
    client.put(f"/api/projects/{demo['id']}/files/data/ll.csv", content=df.to_csv(index=False).encode())
    s = client.post(f"/api/projects/{demo['id']}/columns/suggest", json={"path": "data/ll.csv"}).json()
    assert (s["x"], s["y"], s["target"], s["crs_guess"]) == ("lon", "lat", "temp_f", "EPSG:4326")


# ---------------------------------------------------------------------------
# data check and preview
# ---------------------------------------------------------------------------

def test_data_check_and_preview(client, demo):
    pid = demo["id"]
    t0 = time.perf_counter()
    r = client.post(f"/api/projects/{pid}/data/check", json={})
    elapsed = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    assert elapsed < 2.0
    c = r.json()
    n = c["n_points"]
    assert n > 0 and c["n_input"] == n and c["n_dropped"] == 0
    assert c["grid"]["nx"] > 0 and c["grid"]["ny"] > 0 and c["grid"]["cell_m"] == 30.0
    assert isinstance(c["flags"], list) and {"canopy", "impervious", "albedo"} <= set(c["dose_scale"])
    ds = c["dose_scale"]["canopy"]
    assert len(ds["doses"]) == len(ds["doses_in_sd"]) == len(ds["percentile_reached"]) == 6
    assert c["columns_missing"] == [] and c["background"]["source"] == "field_median"
    tok = c["preview_token"]
    g = client.get(f"/api/projects/{pid}/data/preview/{tok}/grid.bin")
    assert g.status_code == 200 and g.headers["content-type"] == "application/octet-stream"
    offsets = json.loads(g.headers["x-sparc-offsets"])
    assert [(o["name"], o["dtype"]) for o in offsets] == [("ix", "int32"), ("iy", "int32"), ("lon", "float32"),
                                                          ("lat", "float32")]
    assert all(o["length"] == n and o["offset"] % 8 == 0 for o in offsets)
    assert len(g.content) == offsets[-1]["offset"] + 4 * n
    meta = json.loads(g.headers["x-sparc-grid"])
    assert meta["n"] == n and meta["has_lonlat"] is True and meta["crs"] == "EPSG:32619"
    assert set(meta["corners"]) == {"sw", "se", "nw", "ne"} and 41 < meta["corners"]["sw"][0] < 42
    ix = np.frombuffer(g.content, "<i4", n, offsets[0]["offset"])
    assert ix.min() >= 0 and ix.max() < meta["nx"]
    lat = np.frombuffer(g.content, "<f4", n, offsets[3]["offset"])
    assert np.all((lat > meta["bounds_lonlat"][1] - 1e-3) & (lat < meta["bounds_lonlat"][3] + 1e-3))
    col = client.get(f"/api/projects/{pid}/data/preview/{tok}/canopy.bin")
    assert col.status_code == 200 and col.headers["x-sparc-dtype"] == "float32"
    assert int(col.headers["x-sparc-length"]) == n and len(col.content) == 4 * n
    t = np.frombuffer(client.get(f"/api/projects/{pid}/data/preview/{tok}/T.bin").content, "<f4")
    assert len(t) == n and np.isfinite(t).all()
    assert client.get(f"/api/projects/{pid}/data/preview/{tok}/nope.bin").status_code == 404
    assert client.get(f"/api/projects/{pid}/data/preview/badtoken/grid.bin").status_code == 404


def test_data_check_patch_and_errors(client, demo):
    pid = demo["id"]
    r = client.post(f"/api/projects/{pid}/data/check",
                    json={"config_patch": {"data": {"coarse_m": 60}, "predictors": ["canopy", "missing_col"]}})
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["coarse"]["cell_m"] == 60.0 and c["columns_missing"] == ["missing_col"]
    tok = c["preview_token"]
    col = client.get(f"/api/projects/{pid}/data/preview/{tok}/canopy.bin")
    assert len(col.content) == 4 * c["n_points"]
    r = client.post(f"/api/projects/{pid}/data/check", json={"config_patch": {"data": {"target": "nope"}}})
    assert r.status_code == 422 and r.json()["error"]["detail"]["errors"][0]["code"] == "missing_column"
    blank = create(client, "Empty", "blank")["project"]
    r = client.post(f"/api/projects/{blank['id']}/data/check")
    assert r.status_code == 422
    r = client.post(f"/api/projects/{pid}/data/check", json={"config_patch": {"data": {"path": "data/gone.csv"}}})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_yaml_round_trip_keeps_core_block(client, demo):
    pid = demo["id"]
    cfg = client.get(f"/api/projects/{pid}/config").json()
    assert cfg["yaml"].startswith("core:") and cfg["comments_preserved"] is False
    assert cfg["effective"]["cv"]["n_folds"] == 3 and cfg["effective"]["models"]["ols"] is True
    changed = {c["path"]: c for c in cfg["changed_from_defaults"]}
    assert changed["cv.n_folds"] == {"path": "cv.n_folds", "value": 3, "default": 5}
    raw = cfg["raw"]
    raw["stacker"]["epochs"] = 150
    r = put_raw(client, pid, raw, cfg["version"], note="fewer epochs")
    assert r.status_code == 200, r.text
    saved = r.json()
    assert saved["version"] == cfg["version"] + 1 and "epochs: 150" in saved["diff"]
    assert [i for i in saved["issues"] if i["level"] == "error"] == []
    text = Path(demo["config_path"]).read_text()
    assert text.startswith("core:") and yaml.safe_load(text)["core"]["stacker"]["epochs"] == 150
    # a block without core: is wrapped into one
    r = client.put(f"/api/projects/{pid}/config", json={"yaml": yaml.safe_dump(raw)})
    assert r.status_code == 200
    assert Path(demo["config_path"]).read_text().startswith("core:")
    # YAML text with a core: block is stored as written (comments survive)
    commented = "# my notes\n" + yaml.safe_dump({"core": raw}, sort_keys=False)
    v = client.put(f"/api/projects/{pid}/config", json={"yaml": commented}).json()["version"]
    assert Path(demo["config_path"]).read_text() == commented
    assert client.get(f"/api/projects/{pid}/config/history/{v}").json()["yaml"] == commented
    hist = client.get(f"/api/projects/{pid}/config/history").json()
    assert [h["version"] for h in hist] == sorted((h["version"] for h in hist), reverse=True)
    assert any(h["note"] == "fewer epochs" for h in hist)


def test_if_match_conflict(client, demo):
    pid = demo["id"]
    cfg = client.get(f"/api/projects/{pid}/config").json()
    v = cfg["version"]
    raw = cfg["raw"]
    raw["cv"]["seed"] = 7
    assert put_raw(client, pid, raw, v).status_code == 200
    raw["cv"]["seed"] = 8
    r = put_raw(client, pid, raw, v)                                # stale
    assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
    assert r.json()["error"]["detail"]["current_version"] == v + 1
    r = client.patch(f"/api/projects/{pid}/config/sections/stacker", json={"value": {"epochs": 99}},
                     headers={"If-Match": str(v)})
    assert r.status_code == 409
    r = client.patch(f"/api/projects/{pid}/config/sections/stacker", json={"value": {"epochs": 99}},
                     headers={"If-Match": f'"{v + 1}"'})
    assert r.status_code == 200 and r.json()["version"] == v + 2
    assert client.get(f"/api/projects/{pid}/config").json()["raw"]["stacker"] == {"epochs": 99}
    r = client.patch(f"/api/projects/{pid}/config/sections/stacker", json={"value": None})
    assert "stacker" not in client.get(f"/api/projects/{pid}/config").json()["raw"]
    assert client.patch(f"/api/projects/{pid}/config/sections/bogus", json={"value": 1}).status_code == 422
    assert client.put(f"/api/projects/{pid}/config", json={"raw": raw}, headers={"If-Match": "x"}).status_code == 422


def test_edits_outside_studio_become_versions(client, demo):
    pid = demo["id"]
    v = client.get(f"/api/projects/{pid}/config").json()["version"]
    path = Path(demo["config_path"])
    path.write_text(path.read_text().replace("epochs: 200", "epochs: 180"))
    cfg = client.get(f"/api/projects/{pid}/config").json()
    assert cfg["version"] == v + 1 and cfg["raw"]["stacker"]["epochs"] == 180
    assert client.get(f"/api/projects/{pid}/config/history").json()[0]["note"] == "edited outside Studio"
    r = put_raw(client, pid, cfg["raw"], v)
    assert r.status_code == 409


def test_yaml_errors(client, demo):
    r = client.put(f"/api/projects/{demo['id']}/config", json={"yaml": "core:\n  data: [unclosed\n"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "yaml_error"
    assert r.json()["error"]["detail"]["line"] >= 2
    r = client.post(f"/api/projects/{demo['id']}/config/validate", json={"yaml": "- a list"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "yaml_error"


def test_validate_candidate(client, demo):
    pid = demo["id"]
    raw = client.get(f"/api/projects/{pid}/config").json()["raw"]
    raw["actionable"]["canopy"]["doses"] = [0, 500]
    rep = client.post(f"/api/projects/{pid}/config/validate", json={"raw": raw}).json()
    assert rep["ok"] is False and any(i["code"] == "dose_out_of_bounds" for i in rep["issues"])
    assert rep["fast_overrides"]["data.subsample"]["to"] == 8000 and rep["coarse_preview"]["cell_m"] == 60.0


def test_config_schema_endpoint(client):
    s = client.get("/api/config/schema").json()
    assert s["type"] == "object" and "physics" in s["properties"]
    phys = s["$defs"]["PhysicsSection"]["properties"]
    assert phys["sw_down"]["x-ui"]["unit"] == "W/m²" and phys["shade_form"]["x-ui"]["advanced"] is True
    assert set(phys["sw_down"]["x-ui"]) == {"group", "advanced", "unit", "help", "enum_labels"}


def _copy_fixture_run(demo, fixture: Path, run_id: str = "20261001-212149-fast-f1x0") -> Path:
    d = Path(demo["dir"]) / "runs" / run_id
    shutil.copytree(fixture, d)
    return d


def test_impact_preview(client, demo, synth_run_dir):
    pid = demo["id"]
    _copy_fixture_run(demo, synth_run_dir)
    side = json.loads((synth_run_dir / "checkpoint.json").read_text())
    base = client.post(f"/api/projects/{pid}/config/impact").json()
    assert base["changed_sections"] == []
    run = base["runs"][0]
    assert run["run_id"] == "20261001-212149-fast-f1x0" and run["checkpoint_done"] == side["done"]
    assert "core" not in run["changed_sections"]          # same demo config (n=40, seed=0) as the fixture
    raw = client.get(f"/api/projects/{pid}/config").json()["raw"]
    raw["stacker"]["tune_lambda"] = [0.0, 0.5, 2.0]
    out = client.post(f"/api/projects/{pid}/config/impact", json={"raw": raw}).json()
    assert out["changed_sections"] == ["core"]
    raw["cv"]["seed"] = 7
    out = client.post(f"/api/projects/{pid}/config/impact", json={"raw": raw}).json()
    run = out["runs"][0]
    assert "core" in run["changed_sections"] and run["refit_from"] == "S1" and "refit from S1" in run["phrase"]
    raw2 = client.get(f"/api/projects/{pid}/config").json()["raw"]
    raw2["optimize"]["budget"] = 999.0
    out = client.post(f"/api/projects/{pid}/config/impact",
                      json={"yaml": yaml.safe_dump({"core": raw2})}).json()
    assert out["changed_sections"] == ["s7"] and "s7" in out["runs"][0]["changed_sections"]
    assert "core" not in out["runs"][0]["changed_sections"]


def test_impact_lists_indexed_runs(client, ctx, demo, synth_run_dir, tmp_path):
    elsewhere = tmp_path / "outside" / "run_a"
    shutil.copytree(synth_run_dir, elsewhere)
    ctx.db.insert("runs", {"id": "r_outside", "project_id": demo["id"], "run_dir": str(elsewhere),
                           "studio_dir": str(elsewhere / "studio"), "origin": "imported", "status": "imported",
                           "label": "Imported fast run"})
    out = client.post(f"/api/projects/{demo['id']}/config/impact").json()
    assert [(r["run_id"], r["label"]) for r in out["runs"]] == [("r_outside", "Imported fast run")]


def test_startup_scan_registers_folders(make_app, tmp_path):
    """A project folder written while the server was down (e.g. by a features job) is registered at start."""
    from fastapi.testclient import TestClient

    from tests.studio.conftest import AUTH

    app = make_app()
    with TestClient(app, headers=AUTH) as c:
        p = c.post("/api/projects", json={"name": "Kept", "template": "blank"}).json()["project"]
        ws = app.state.studio.workspace
    from sparc.studio.db import Database

    db = Database(ws.db_path)
    db.execute("DELETE FROM projects")
    db.close()
    app2 = make_app()
    with TestClient(app2, headers=AUTH) as c:
        wait_for(lambda: c.get("/api/projects").json(), 10, what="the startup scan")
        assert [x["id"] for x in c.get("/api/projects").json()] == [p["id"]]
