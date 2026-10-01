"""Input jobs (api.md §5.4, §8) with the network fetchers faked, link-into-config, input views, station lookup.

Jobs run in real worker processes (``fake_worker.py``: the worker with the
fetchers patched).  Live network tests are marked ``network``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.studio.projects.conftest import create
from tests.studio.projects.fakes import ISD_ROWS


def launch(client, pid, kind, body=None):
    r = client.post(f"/api/projects/{pid}/inputs/{kind}", json=body if body is not None else {})
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["kind"] == f"input.{kind}" and job["lane"] == "network" and job["executor"] == "process"
    assert job["project_id"] == pid
    return job


def finish(client, wait_job, job, timeout=90):
    done = wait_job(client, job["id"], timeout=timeout)
    if done["status"] != "succeeded":
        raise AssertionError(f"{job['kind']} {done['status']}: {done.get('error')}")
    return done


def raw_of(client, pid):
    return client.get(f"/api/projects/{pid}/config").json()["raw"]


def test_forcing_job_links_config(client, demo, fake_network, wait_job):
    pid = demo["id"]
    v0 = client.get(f"/api/projects/{pid}/config").json()["version"]
    job = launch(client, pid, "forcing", {"date": "2020-07-29", "hours": [15, 16], "tz": "America/New_York",
                                          "station": "72507014765"})
    done = finish(client, wait_job, job)
    res = done["result"]
    assert res["path"] == "inputs/forcing/forcing_2020-07-29_15-16.json" and res["linked"] is True
    assert res["sw_down"] == 636.0 and res["wind"] == [0.4, 7.6] and res["checks"]
    cfg = client.get(f"/api/projects/{pid}/config").json()
    assert cfg["raw"]["physics"]["forcing"] == res["path"] and cfg["version"] == v0 + 1
    assert cfg["effective"]["physics"]["sw_down"] == 636.0          # the forcing file's values apply
    note = client.get(f"/api/projects/{pid}/config/history").json()[0]["note"]
    assert "physics.forcing" in note and job["id"] in note
    st = client.get(f"/api/projects/{pid}/inputs").json()["forcing"]
    assert st == {"path": res["path"], "date": "2020-07-29", "physics": {"window": "day", "sw_down": 636.0,
                                                                         "lw_net": -71.0, "wind": [0.4, 7.6]},
                  "checks": res["checks"], "linked": True}
    view = client.get(f"/api/projects/{pid}/inputs/forcing/view").json()
    assert view["station"]["station"] == "72507014765"
    temps = next(r for r in view["compare"] if r["name"].startswith("air temperature"))
    assert temps["era5"] == 87.8 and temps["station"] == pytest.approx(30.8 * 9 / 5 + 32)
    rows = {r["key"]: r for r in client.get(f"/api/projects/{pid}").json()["readiness"]}
    assert rows["forcing"]["state"] == "ok"


def test_forcing_validation(client, demo):
    pid = demo["id"]
    for bad in ({"date": "2020-13-01", "hours": [15, 16], "tz": "UTC"},
                {"date": "2020-07-29", "hours": [16, 15], "tz": "UTC"},
                {"date": "2020-07-29", "hours": [15, 16], "tz": "Mars/Olympus"},
                {"date": "2020-07-29", "hours": [15, 16], "tz": "UTC", "lat": 41.0},
                {"date": "2020-07-29", "hours": [15, 16], "tz": "UTC", "extra": 1}):
        r = client.post(f"/api/projects/{pid}/inputs/forcing", json=bad)
        assert r.status_code == 422, bad


def test_layers_job_links_config(client, demo, fake_network, wait_job):
    pid = demo["id"]
    done = finish(client, wait_job, launch(client, pid, "layers"))
    res = done["result"]
    assert res == {"path": "inputs/layers/layers.parquet", "n_cells": res["n_cells"],
                   "people_total": res["people_total"], "linked": True}
    lay = pd.read_parquet(Path(demo["dir"]) / res["path"])
    assert len(lay) == res["n_cells"] > 0 and {"id", "x_m", "y_m", "people", "lc_built"} <= set(lay.columns)
    assert raw_of(client, pid)["planner"]["layers"] == res["path"]
    st = client.get(f"/api/projects/{pid}/inputs").json()["layers"]
    assert st["path"] == res["path"] and st["linked"] is True and st["n"] == res["n_cells"]
    assert st["people_total"] == pytest.approx(res["people_total"])
    view = client.get(f"/api/projects/{pid}/inputs/layers/view").json()
    assert view["totals"]["people"] == pytest.approx(res["people_total"])


def test_cmip6_job_links_table_and_periods(client, demo, fake_network, wait_job):
    pid = demo["id"]
    body = {"experiments": ["ssp245", "ssp585"], "periods": {"2041-2060": [2041, 2060], "2061-2080": [2061, 2080]}}
    done = finish(client, wait_job, launch(client, pid, "cmip6", body))
    res = done["result"]
    assert res["path"] == "inputs/climate/cmip6_tasmax_6-7-8.csv" and res["linked"] is True
    assert res["n_models"] == 3 and res["experiments"] == ["ssp245", "ssp585"]
    assert res["periods"] == ["2041-2060", "2061-2080"] and res["skipped_models"] == []
    raw = raw_of(client, pid)
    assert raw["climate"]["table"] == res["path"] and raw["climate"]["source"] == "table"
    assert raw["climate"]["periods"] == {"2041-2060": [2041, 2060], "2061-2080": [2061, 2080]}
    assert raw["climate"]["experiments"] == ["ssp245", "ssp585"]
    st = client.get(f"/api/projects/{pid}/inputs").json()["climate"]
    assert st["n_models"] == 3 and st["linked"] is True and st["periods"] == res["periods"]
    view = client.get(f"/api/projects/{pid}/inputs/climate/view").json()
    assert len(view["rows"]) == 12 and {s["experiment"] for s in view["summary"]} == {"ssp245", "ssp585"}
    assert all(s["p10"] <= s["median"] <= s["p90"] for s in view["summary"])
    rep = client.post(f"/api/projects/{pid}/config/validate").json()
    assert [i for i in rep["issues"] if i["level"] == "error"] == []


def test_ghcn_job(client, demo, fake_network, wait_job, ctx):
    pid = demo["id"]
    done = finish(client, wait_job, launch(client, pid, "ghcn", {"station": "USW00014765"}))
    res = done["result"]
    assert res["years"] == [1995, 2014] and Path(res["path"]) == ctx.workspace.cache_dir / "ghcn_USW00014765.csv"
    assert Path(res["path"]).is_file()
    assert client.get(f"/api/projects/{pid}/inputs").json()["ghcn"] == {"station": "USW00014765",
                                                                        "years": [1995, 2014]}
    assert raw_of(client, pid)["planner"]["ghcn_station"] == "USW00014765"
    blank = create(client, "No station", "blank")["project"]
    r = client.post(f"/api/projects/{blank['id']}/inputs/ghcn", json={})
    assert r.status_code == 422


def test_station_lookup_needs_the_cached_index(client, demo, fake_network, wait_job):
    pid = demo["id"]
    r = client.get(f"/api/projects/{pid}/forcing/stations", params={"lat": 41.82, "lon": -71.41})
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["detail"]["missing"] == "isd-history.csv"
    assert err["action"] == {"kind": "fetch_input", "label": "Fetch station list", "method": "POST",
                             "path": f"/api/projects/{pid}/inputs/stations", "body": {}}
    done = finish(client, wait_job, launch(client, pid, "stations"))
    assert done["result"]["n_stations"] == len(ISD_ROWS)
    rows = client.get(f"/api/projects/{pid}/forcing/stations", params={"lat": 41.82, "lon": -71.41,
                                                                        "limit": 3}).json()
    assert [r["usaf_wban"] for r in rows] == ["72507014765", "72507454752", "72508894746"]
    assert rows[0]["name"].startswith("PROVIDENCE") and rows[0]["begin"] == "1942-01-01"
    assert rows[0]["dist_km"] < rows[1]["dist_km"] < rows[2]["dist_km"]
    # default site: the demo's data centroid (EPSG:32619)
    rows = client.get(f"/api/projects/{pid}/forcing/stations").json()
    assert len(rows) == len(ISD_ROWS) - 1 and all(r["usaf_wban"] != "99999999999" for r in rows)


def test_features_new_project(client, demo, fake_network, wait_job):
    pid = demo["id"]
    done = finish(client, wait_job, launch(client, pid, "features"))
    res = done["result"]
    assert res["path"] == "inputs/features/open_features.parquet" and res["linked"] is True
    roles = {a["role"] for a in res["agreement"]}
    assert roles == {"canopy", "impervious", "ndvi", "albedo", "elevation", "water_distance"}
    assert all(a["pearson_r"] > 0.9 for a in res["agreement"])
    opid = res["open_project_id"]
    other = client.get(f"/api/projects/{opid}")
    assert other.status_code == 200, other.text                 # registered before the job was reported final
    op = other.json()["project"]
    assert op["name"] == "Demo city_open" and op["template"] == "features_open"
    raw = raw_of(client, opid)
    assert raw["predictors"] == [f"open_{r}" for r in ("albedo", "canopy", "impervious", "ndvi", "elevation",
                                                       "water_distance")]
    assert raw["physics"]["roles"]["canopy"] == "open_canopy" and set(raw["actionable"]) == \
        {"open_canopy", "open_impervious", "open_albedo"}
    assert raw["causal"]["treatments"] == ["open_canopy"] and raw["optimize"]["variable"] == "open_canopy"
    assert raw["data"]["join"][-1] == {"path": "inputs/features/open_features.parquet", "key": "id", "right_key": "id"}
    assert Path(raw["data"]["path"]).is_absolute()
    rep = client.post(f"/api/projects/{opid}/config/validate").json()
    assert [i for i in rep["issues"] if i["level"] == "error"] == [], rep["issues"]
    check = client.post(f"/api/projects/{opid}/data/check").json()
    assert check["columns_missing"] == [] and check["n_points"] > 0
    st = client.get(f"/api/projects/{pid}/inputs").json()["features"]
    assert st["path"] == res["path"] and len(st["agreement"]) == 6
    view = client.get(f"/api/projects/{pid}/inputs/features/view").json()
    assert len(view["scatter_bins"]) == 6 and np.sum(view["scatter_bins"][0]["counts"]) == check["n_points"]


def test_features_bootstrap_on_a_predictorless_project(client, fake_network, wait_job, tmp_path):
    """A project with only target, id, x/y and a CRS gets its predictors from open data (SPEC §9.3)."""
    from sparc.core.synthetic import write_demo_project

    src = tmp_path / "src"
    write_demo_project(src, n=24, seed=2)
    table = pd.read_csv(src / "data" / "city.csv")[["id", "x", "y", "T"]]
    p = create(client, "New city", "blank")["project"]
    pid = p["id"]
    r = client.put(f"/api/projects/{pid}/files/data/points.csv", content=table.to_csv(index=False).encode())
    assert r.status_code == 201
    cfg = client.get(f"/api/projects/{pid}/config").json()
    r = client.patch(f"/api/projects/{pid}/config/sections/data", headers={"If-Match": str(cfg["version"])},
                     json={"value": {"path": "data/points.csv", "target": "T", "id": "id", "x": "x", "y": "y",
                                     "crs": "EPSG:32619"}})
    assert r.status_code == 200, r.text
    assert client.post(f"/api/projects/{pid}/inputs/features", json={"target": "bogus"}).status_code == 422
    done = finish(client, wait_job, launch(client, pid, "features", {"target": "this_project"}))
    res = done["result"]
    assert res["agreement"] == [] and res["open_project_id"] is None and res["linked"] is True
    raw = raw_of(client, pid)
    assert raw["predictors"] == [f"open_{r}" for r in ("albedo", "canopy", "impervious", "ndvi", "elevation",
                                                       "water_distance")]
    assert raw["physics"]["roles"] == {r: f"open_{r}" for r in ("albedo", "canopy", "impervious", "ndvi",
                                                                "elevation", "water_distance")}
    assert raw["data"]["join"] == [{"path": "inputs/features/open_features.parquet", "key": "id",
                                    "right_key": "id"}]
    check = client.post(f"/api/projects/{pid}/data/check").json()
    assert check["columns_missing"] == [] and check["n_points"] == len(table)
    rows = {r["key"]: r for r in client.get(f"/api/projects/{pid}").json()["readiness"]}
    assert rows["roles"]["state"] == "ok" and rows["columns"]["state"] == "ok"


def test_input_preconditions(client):
    p = create(client, "Bare", "blank")["project"]
    pid = p["id"]
    r = client.post(f"/api/projects/{pid}/inputs/layers", json={})
    assert r.status_code == 422 and r.json()["error"]["code"] == "requirements"
    assert r.json()["error"]["detail"]["missing"] == ["data.path", "data.target"]
    cfg = client.get(f"/api/projects/{pid}/config").json()
    client.patch(f"/api/projects/{pid}/config/sections/data", headers={"If-Match": str(cfg["version"])},
                 json={"value": {"path": "data/x.csv", "target": "T"}})
    for kind in ("layers", "features", "cmip6"):
        body = {} if kind != "forcing" else None
        r = client.post(f"/api/projects/{pid}/inputs/{kind}", json=body)
        code = r.json()["error"]["code"]
        assert r.status_code == 422 and code in ("needs_crs", "requirements"), (kind, r.text)
    r = client.post(f"/api/projects/{pid}/inputs/cmip6", json={})
    assert r.json()["error"]["code"] == "needs_crs"
    assert r.json()["error"]["action"]["path"] == f"/p/{pid}/setup/data"
    r = client.post(f"/api/projects/{pid}/inputs/forcing", json={"date": "2020-07-29", "hours": [15, 16],
                                                                   "tz": "UTC"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "needs_crs"


def test_link_endpoint_preview_and_apply(client, demo):
    pid = demo["id"]
    forcing = {"date": "2020-07-18", "physics": {"window": "day", "sw_down": 700.0, "lw_net": -90.0,
                                                 "wind": [1.0, 2.0]}, "checks": ["ok"]}
    up = client.put(f"/api/projects/{pid}/files/forcing/day.json", content=json.dumps(forcing).encode()).json()
    v0 = client.get(f"/api/projects/{pid}/config").json()["version"]
    r = client.post(f"/api/projects/{pid}/link", json={"kind": "forcing", "path": up["path"]})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["applied"] is False and "+    forcing: inputs/forcing/day.json" in out["yaml_diff"]
    assert client.get(f"/api/projects/{pid}/config").json()["version"] == v0
    out = client.post(f"/api/projects/{pid}/link", json={"kind": "forcing", "path": up["path"], "apply": True}).json()
    assert out["applied"] is True and out["version"] == v0 + 1
    assert raw_of(client, pid)["physics"]["forcing"] == "inputs/forcing/day.json"
    assert client.get(f"/api/projects/{pid}/inputs").json()["forcing"]["linked"] is True
    r = client.post(f"/api/projects/{pid}/link", json={"kind": "layers", "path": "inputs/layers/none.parquet"})
    assert r.status_code == 404
    # a climate table with non-default periods links them too
    tab = pd.read_csv(Path(demo["dir"]) / "inputs/climate/demo_cmip6.csv")
    tab = tab[tab.period == "2041-2060"].assign(period="2050-2070")
    client.put(f"/api/projects/{pid}/files/climate/other.csv", content=tab.to_csv(index=False).encode())
    out = client.post(f"/api/projects/{pid}/link", json={"kind": "climate", "path": "inputs/climate/other.csv",
                                                         "apply": True}).json()
    raw = raw_of(client, pid)
    assert out["applied"] and raw["climate"]["table"] == "inputs/climate/other.csv"
    assert raw["climate"]["periods"] == {"2050-2070": [2050, 2070]}


def test_link_features_new_project(client, demo):
    pid = demo["id"]
    check = client.post(f"/api/projects/{pid}/data/check").json()
    ids = np.arange(check["n_points"])
    df = pd.DataFrame({"id": ids, **{f"open_{r}": np.linspace(0, 1, len(ids)) for r in
                                      ("canopy", "impervious", "ndvi", "albedo", "elevation", "water_distance")}})
    client.put(f"/api/projects/{pid}/files/features/open_features.parquet", content=df.to_parquet(index=False))
    path = "inputs/features/open_features.parquet"
    prev = client.post(f"/api/projects/{pid}/link", json={"kind": "features_new_project", "path": path}).json()
    assert prev["applied"] is False and "open_canopy" in prev["yaml_diff"] and prev["new_project_id"] is None
    out = client.post(f"/api/projects/{pid}/link", json={"kind": "features_new_project", "path": path,
                                                         "apply": True}).json()
    new = client.get(f"/api/projects/{out['new_project_id']}").json()
    assert new["project"]["name"] == "Demo city_open"
    assert (Path(new["project"]["dir"]) / path).read_bytes() == (Path(demo["dir"]) / path).read_bytes()
    out = client.post(f"/api/projects/{pid}/link", json={"kind": "features_join", "path": path, "apply": True})
    raw = raw_of(client, pid)
    assert out.status_code == 200 and raw["data"]["join"][-1]["path"] == path
    assert "open_canopy" in raw["predictors"] and raw["physics"]["roles"]["canopy"] == "canopy"


def test_open_project_raw_renames_by_role(tmp_path):
    from sparc.studio.projects.inputs import open_project_raw

    raw = yaml.safe_load(Path(__file__).resolve().parents[3].joinpath("configs/core_providence.yml").read_text())["core"]
    out = open_project_raw(raw, tmp_path, "inputs/features/open_features.parquet")
    ref = yaml.safe_load(Path(__file__).resolve().parents[3].joinpath("configs/core_providence_open.yml")
                         .read_text())["core"]
    for key in ("predictors", "actionable", "mediators", "scenarios", "joint_scenarios"):
        if key == "predictors":
            assert sorted(out[key]) == sorted(ref[key])
        elif key in ("scenarios", "joint_scenarios"):
            assert [s["name"] for s in out[key]] == [s["name"] for s in ref[key]]
        else:
            assert set(out[key]) == set(ref[key])
    assert out["causal"]["treatments"] == ref["causal"]["treatments"]
    assert out["causal"]["confounders"] == ref["causal"]["confounders"]
    assert out["physics"]["roles"] == ref["physics"]["roles"]
    assert Path(out["data"]["path"]).is_absolute() and out["name"] == "providence_uhi_open"


def test_input_views_unknown(client, demo):
    assert client.get(f"/api/projects/{demo['id']}/inputs/bogus/view").json()["error"]["code"] == "unknown_view"
    assert client.get(f"/api/projects/{demo['id']}/inputs/forcing/view").status_code == 404


def test_kinds_are_listed_with_hosts(client):
    kinds = {k["kind"]: k for k in client.get("/api/meta").json()["job_kinds"]}
    for k in ("input.forcing", "input.layers", "input.features", "input.cmip6", "input.ghcn", "input.stations"):
        assert kinds[k]["lane"] == "network" and kinds[k]["network_hosts"], k
    assert kinds["input.forcing"]["params_schema"]["additionalProperties"] is False


# ---------------------------------------------------------------------------
# live network (deselected with -m "not network")
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_live_station_index(client, demo, wait_job):
    done = finish(client, wait_job, launch(client, demo["id"], "stations"), timeout=300)
    assert done["result"]["n_stations"] > 10000
    rows = client.get(f"/api/projects/{demo['id']}/forcing/stations", params={"lat": 41.72, "lon": -71.43}).json()
    assert rows and rows[0]["dist_km"] < 30


@pytest.mark.network
def test_live_ghcn(client, demo, wait_job):
    done = finish(client, wait_job, launch(client, demo["id"], "ghcn", {"station": "USW00014765"}), timeout=300)
    assert done["result"]["years"][0] <= 1995 <= done["result"]["years"][1]
