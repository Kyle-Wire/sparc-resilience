"""The Heat tab (``GET /runs/{rid}/views/heat``), heat map layers and scenario verdicts."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from sparc.core import heat as heatmod
from tests.studio.runs.conftest import f32


def heat(client, rid, **params) -> dict:
    r = client.get(f"/api/runs/{rid}/views/heat", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def test_heat_without_campaign_humidity_asks_for_a_dewpoint(client, fixture_run):
    rid, _ = fixture_run
    vm = heat(client, rid)
    s = vm["sections"]
    assert list(s) == ["brief", "humidity", "categories", "kpis", "today", "futures", "hist", "verdicts"]
    assert s["humidity"]["needs_input"] is True and s["humidity"]["dewpoint_C"] is None
    assert [p["dewpoint_C"] for p in s["humidity"]["presets"]] == [10.0, 16.0, 21.0, 24.0]
    assert [c["id"] for c in s["categories"]] == ["below", "caution", "extreme_caution", "danger", "extreme_danger"]
    assert s["today"] is None and s["brief"] is None
    tabs = {t["id"]: t["availability"] for t in client.get(f"/api/runs/{rid}/outputs").json()["tabs"]}
    assert tabs["heat"] == "ready"
    # no campaign humidity: no heat layers
    keys = {m["key"] for g in client.get(f"/api/runs/{rid}/layers").json()["groups"] for m in g["layers"]}
    assert "heat_index" not in keys


def test_heat_with_a_chosen_dewpoint_counts_cells_by_category(client, fixture_run, synth):
    rid, _ = fixture_run
    s = heat(client, rid, dewpoint_C=21)["sections"]
    pred = pd.read_parquet(synth / "predictions.parquet")
    units = json.loads((synth / "manifest.json").read_text())["config"]["data"].get("target_units", "degF")
    hi = heatmod.heat_index_today(pred["target"].to_numpy(float), units, 21.0)
    codes = heatmod.category_codes(hi)
    assert s["humidity"]["user_set"] is True and s["humidity"]["source"] == "your setting"
    today = s["today"]
    if today["measure"] == "people":                       # the demo carries a population layer
        w = np.nan_to_num(f32(client.get(f"/api/runs/{rid}/layers/people.bin").content).astype(float))
    else:
        w = np.ones(len(pred))
    assert today["total"] == pytest.approx(w.sum(), rel=1e-5)
    expected = {c[0]: float(w[codes == i].sum()) for i, c in enumerate(heatmod.CATEGORIES)}
    assert today["unadapted"]["counts"] == pytest.approx(expected, rel=1e-5, abs=1e-3)
    assert today["unadapted"]["ec_or_worse"] == pytest.approx(float(w[codes >= 2].sum()), rel=1e-5, abs=1e-3)
    assert today["unadapted"]["max_hi"] == pytest.approx(float(hi.max()), rel=1e-6)
    assert sum(s["hist"]["counts"]) == pytest.approx(w.sum(), rel=1e-5)
    assert [b["id"] for b in s["brief"]][:2] == ["how_hot", "who"]
    assert s["kpis"][0]["id"] == "ec_today"
    # a more humid afternoon never lowers the heat index
    wetter = heat(client, rid, dewpoint_C=24)["sections"]["today"]["unadapted"]
    assert wetter["mean_hi"] >= today["unadapted"]["mean_hi"]
    assert client.get(f"/api/runs/{rid}/views/heat", params={"dewpoint_C": 80}).status_code == 422


def test_heat_futures_and_package(client, demo, place_run, synth):
    """With climate projections and a joint scenario: futures carry the humidity band and the model spread, and the
    package lowers the count."""
    def edit(d):
        m = json.loads((d / "manifest.json").read_text())
        m["climate"] = {"units": "degF", "projections": [
            {"experiment": "ssp245", "label": "SSP2-4.5", "period": "2041-2060", "n_models": 5,
             "warming": {"median": 3.0, "p10": 2.0, "p90": 4.5}}]}
        (d / "manifest.json").write_text(json.dumps(m))
        pred = pd.read_parquet(d / "predictions.parquet")
        deltas = pd.read_parquet(d / "scenario_deltas.parquet") if (d / "scenario_deltas.parquet").exists() else \
            pd.DataFrame({"id": pred["id"]})
        deltas["Cool package"] = -1.5
        deltas.to_parquet(d / "scenario_deltas.parquet")

    def edit_config(raw):
        raw["joint_scenarios"] = [{"name": "Cool package", "interventions": []}]

    rd = place_run(demo, "20260101-000000-heat-0001", edit=edit, edit_config=edit_config)
    rid = rd.name
    s = heat(client, rid, dewpoint_C=21)["sections"]
    assert s["today"]["package"] == "Cool package"
    assert s["today"]["adapted"]["mean_hi"] < s["today"]["unadapted"]["mean_hi"]
    f = s["futures"][0]
    assert f["label"] == "SSP2-4.5" and f["warming_F"] == pytest.approx(3.0)
    u = f["unadapted"]
    lo_h, hi_h = u["ec_range_humidity"]
    lo_f, hi_f = u["ec_range_full"]
    assert lo_h <= hi_h and lo_f <= lo_h and hi_f >= hi_h          # the model spread widens the humidity band
    assert u["constant_rh"]["mean_hi"] >= u["constant_dewpoint"]["mean_hi"]
    assert f["adapted"]["constant_rh"]["mean_hi"] < u["constant_rh"]["mean_hi"]
    assert s["hist"]["adapted_counts"] is not None
    assert any(b["id"] == "helps" for b in s["brief"]) and any(b["id"] == "future" for b in s["brief"])


def test_scenario_and_uncertainty_rows_carry_verdicts(client, demo, place_run):
    def edit(d):
        m = json.loads((d / "manifest.json").read_text())
        names = [s["name"] for s in m.get("scenarios") or []][:2]
        assert names
        rows = [{"scenario": names[0], "estimate": -0.4, "estimation_95": [-0.7, -0.1], "specification": [-0.5, -0.3],
                 "envelope": [-0.7, -0.1], "sign_stability": 1.0, "frac_extrapolated": 0.0}]
        if len(names) > 1:
            rows.append({"scenario": names[1], "estimate": -0.1, "estimation_95": [-0.3, 0.1],
                         "envelope": [-0.3, 0.1], "sign_stability": 0.6})
        m["uncertainty"] = {"scenarios": rows}
        (d / "manifest.json").write_text(json.dumps(m))

    rid = place_run(demo, "20260101-000000-verd-0001", edit=edit).name
    unc = client.get(f"/api/runs/{rid}/views/uncertainty").json()["sections"]["rows"]
    assert unc[0]["verdict"]["verdict"] == "robust" and unc[0]["verdict"]["label"] == "Robust"
    if len(unc) > 1:
        assert unc[1]["verdict"]["verdict"] == "not_established"
    sc = {r["name"]: r for r in client.get(f"/api/runs/{rid}/views/scenarios").json()["sections"]["rows"]}
    assert sc[unc[0]["label"]]["verdict"]["verdict"] == "robust"
    v = heat(client, rid)["sections"]["verdicts"]
    assert v[0]["scenario"] == unc[0]["label"] and v[0]["envelope"] == [-0.7, -0.1]


def test_providence_heat(client, ctx, providence_runs, tmp_path, monkeypatch):
    """The recorded full Providence run: campaign dewpoint from the station, residents from the planner pack,
    heat layers, and the package's verdict."""
    full = providence_runs / "providence_uhi"
    if not (full / "planner" / "planner_cells.parquet").is_file():
        pytest.skip("no full Providence run with the planner pack")
    from tests.studio.runs.conftest import REPO

    monkeypatch.chdir(REPO)                                         # configs/forcing/… relative to the checkout
    light = tmp_path / "providence_uhi"
    (light / "planner").mkdir(parents=True)
    for name in ("predictions.parquet", "scenario_deltas.parquet", "climate.json", "uncertainty.json",
                 "run_state.json", "influence.json"):
        if (full / name).is_file():
            shutil.copy2(full / name, light / name)
    shutil.copy2(full / "planner" / "planner_cells.parquet", light / "planner" / "planner_cells.parquet")
    m = json.loads((full / "manifest.json").read_text())
    m["provenance"]["config_dir"] = str(tmp_path / "absent")
    (light / "manifest.json").write_text(json.dumps(m))
    row = ctx.services["registry"].index_run_dir(light, origin="imported")
    rid = row["id"]
    s = heat(client, rid)["sections"]
    h = s["humidity"]
    assert h["needs_input"] is False and h["user_set"] is False and h["dewpoint_C"] == pytest.approx(16.7, abs=0.05)
    t = s["today"]
    assert t["measure"] == "people" and t["total"] > 150_000
    assert 20_000 < t["unadapted"]["ec_or_worse"] < 60_000
    assert t["adapted"]["ec_or_worse"] < t["unadapted"]["ec_or_worse"]
    assert s["futures"] and all(f["unadapted"]["ec_range_full"][1] > t["unadapted"]["ec_or_worse"]
                                for f in s["futures"])
    pkg = next(v for v in s["verdicts"] if v["scenario"] == t["package"])
    assert pkg["verdict"] == "robust"
    canopy = [v for v in s["verdicts"] if v["scenario"].lower().startswith("canopy")]
    assert canopy and all(v["verdict"] == "not_established" for v in canopy)
    keys = {m["key"] for g in client.get(f"/api/runs/{rid}/layers").json()["groups"] for m in g["layers"]}
    assert {"heat_index", "heat_cat", "heat_index_pkg", "heat_cat_pkg"} <= keys
    hi = f32(client.get(f"/api/runs/{rid}/layers/heat_index.bin").content)
    assert hi.size == len(pd.read_parquet(light / "predictions.parquet"))
    assert 80 < float(np.nanmedian(hi)) < 110
