"""Budget plans (SPEC §7.10) and climate × adaptation (SPEC §7.11)."""

from __future__ import annotations

import base64
import json

import numpy as np
import pandas as pd
import pytest


def _vr_from_files(run_dir, cfg_raw, var="canopy"):
    """A VariableResponse straight from the run files (independent of the Studio rebuild)."""
    from sparc.core.response import VariableResponse

    df = pd.read_parquet(run_dir / f"response_{var}.parquet")
    curves = json.loads((run_dir / "response_curves.json").read_text())[var]
    maps = df.drop(columns=["id", "x_m", "y_m"])
    return VariableResponse(variable=var, direction=cfg_raw["actionable"][var].get("direction", "increase"),
                            doses=[float(d) for d in curves["curve"]["dose"]], curve=pd.DataFrame(curves["curve"]),
                            maps=maps, summary=curves["summary"])


def _dose(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype="<f4").astype(np.float64)


def test_plan_preview_equals_planned_allocation(client, run_ctx, synth_run):
    from sparc.core.optimize import planned_allocation

    rid, rd = synth_run
    r = client.post(f"/api/runs/{rid}/plans/preview", json={"lever": "canopy", "budget": 1500, "min_dose": 2.0})
    assert r.status_code == 200, r.text
    got = r.json()
    want = planned_allocation(_vr_from_files(rd, run_ctx.cfg_raw), 1500.0, min_dose=2.0)
    assert got["planned_total"] == pytest.approx(want["planned_total_cooling"], rel=1e-12)
    assert got["total_cost"] == pytest.approx(want["total_cost"], rel=1e-12)
    assert got["n_cells_treated"] == want["n_cells_treated"]
    assert got["gini"] == pytest.approx(want["gini"])
    assert got["min_dose_dropped_cost"] == pytest.approx(want["min_dose_dropped_cost"])
    dose = _dose(got["dose"])
    np.testing.assert_allclose(dose, want["dose"], rtol=1e-6)
    # n_cells_treated counts cells, not segments
    assert got["n_cells_treated"] == int((dose > 0).sum())
    assert [p["budget"] for p in got["pareto"]] == [375.0, 750.0, 1500.0, 3000.0]
    assert all(p["n_segments"] >= p["n_cells"] for p in got["pareto"])
    assert got["caption"] and got["objective"] == "total cooling"


def test_region_cap_zeroes_cells_outside(client, synth_run):
    rid, _ = synth_run
    region = {"kind": "top", "column": "pred:target", "frac": 0.3, "direction": "highest"}
    r = client.post(f"/api/runs/{rid}/plans/preview", json={"lever": "canopy", "budget": 3000,
                                                             "cap": {"region": region}})
    assert r.status_code == 200, r.text
    dose = _dose(r.json()["dose"])
    mask = np.asarray(client.post(f"/api/runs/{rid}/selection/resolve", json={"selection": region}).json()["mask"])
    m = np.unpackbits(np.frombuffer(base64.b64decode(str(mask)), dtype=np.uint8), bitorder="little")[:dose.size]
    assert dose[m == 0].max() == 0.0 and dose[m == 1].max() > 0
    assert "selected region" in r.json()["constraint"]


def test_plan_objectives_and_equity(client, synth_run):
    rid, _ = synth_run
    base = client.post(f"/api/runs/{rid}/plans/preview", json={"lever": "canopy", "budget": 1000}).json()
    ppl = client.post(f"/api/runs/{rid}/plans/preview", json={
        "lever": "canopy", "budget": 1000, "objective": "people", "cap": {"plantable": True},
        "equity": {"source": "share_60_plus", "focus": 0.5}})
    assert ppl.status_code == 200, ppl.text
    out = ppl.json()
    assert out["objective"].startswith("resident-weighted") and "plantable space" in out["constraint"]
    assert not np.allclose(_dose(out["dose"]), _dose(base["dose"]))
    bad = client.post(f"/api/runs/{rid}/plans/preview", json={"lever": "albedo", "budget": 10,
                                                               "cap": {"plantable": True}})
    assert bad.status_code == 422


def test_plan_needs_responses(client, synth_run):
    rid, rd = synth_run
    (rd / "response_albedo.parquet").unlink()
    r = client.post(f"/api/runs/{rid}/plans/preview", json={"lever": "albedo", "budget": 10})
    assert r.status_code == 422 and r.json()["error"]["code"] == "needs_responses"


def test_plan_store_field_kit_and_scenario(client, ctx, run_ctx, synth_run):
    from sparc.studio.engine.compile import compile_scenario

    rid, _ = synth_run
    r = client.post(f"/api/runs/{rid}/plans", json={"params": {"lever": "canopy", "budget": 2000}, "name": "Trees",
                                                     "verify": False})
    assert r.status_code == 201, r.text
    plan = r.json()["plan"]
    assert r.json()["job"] is None and plan["realised"] is None
    assert [p["id"] for p in client.get(f"/api/runs/{rid}/plans").json()] == [plan["id"]]
    dose = np.frombuffer(client.get(f"/api/plans/{plan['id']}/layers/dose.bin").content, dtype="<f4")
    assert int((dose > 0).sum()) == plan["planned"]["n_cells_treated"]
    kit = client.post(f"/api/plans/{plan['id']}/field-kit", json={"n_sites": 8, "min_spacing_m": 60,
                                                                   "n_pairs": 5, "min_distance_m": 60}).json()
    ids = {str(i) for i in np.asarray(run_ctx.grid.ids)}
    assert len(kit["cells"]) == plan["planned"]["n_cells_treated"]
    c0 = kit["cells"][0]
    assert c0["rank"] == 1 and str(c0["id"]) in ids and np.isfinite(c0["lon"]) and np.isfinite(c0["lat"])
    assert all(a["planned_benefit"] >= b["planned_benefit"] for a, b in zip(kit["cells"], kit["cells"][1:]))
    assert kit["sites"] and all(str(s["id"]) in ids and s["lon"] is not None and s["lat"] is not None
                                for s in kit["sites"])
    assert kit["pairs"] and all(str(p["treated_id"]) in ids and str(p["control_id"]) in ids
                                and p["treated_lon"] is not None and p["control_lat"] is not None for p in kit["pairs"])
    sc = client.post(f"/api/plans/{plan['id']}/to-scenario")
    assert sc.status_code == 201
    edit = sc.json()["doc"]["edits"][0]
    assert edit["mode"] == "per_cell" and edit["per_cell_ref"] == f"plan:{plan['id']}"
    comp = compile_scenario(run_ctx, sc.json()["doc"], db=ctx.db)
    np.testing.assert_allclose(comp.edits[0].per_point, dose.astype(np.float64))
    assert client.delete(f"/api/plans/{plan['id']}").json() == {"ok": True}
    assert client.get(f"/api/plans/{plan['id']}").status_code == 404


def test_climate_explore_equals_summarize_projections(client, run_ctx, synth_run):
    from sparc.core.climate import summarize_projections

    rid, _ = synth_run
    info = client.get(f"/api/runs/{rid}/climate/factors").json()
    assert info["present"] and "ssp245" in info["experiments"] and info["models"]
    r = client.post(f"/api/runs/{rid}/climate/explore", json={
        "adaptations": [{"kind": "configured", "slug": "cooling-package"}], "thresholds": [88.0, 92.5],
        "statistic": "p90"})
    assert r.status_code == 200, r.text
    got = r.json()
    factors = pd.read_csv(run_ctx.cfg.resolve_path(run_ctx.cfg_raw["climate"]["table"]))
    delta = pd.read_parquet(run_ctx.run_dir / "scenario_deltas.parquet")["Cooling package"].to_numpy(float)
    want = summarize_projections(np.asarray(run_ctx.data.target_raw, float), factors, {"Cooling package": delta},
                                 [88.0, 92.5], to_units=1.8)
    assert got["thresholds"] == [88.0, 92.5] and got["adaptation"] == ["Cooling package"]
    assert len(got["projections"]) == len(want["projections"])
    for g, w in zip(got["projections"], want["projections"]):
        assert (g["experiment"], g["period"]) == (w["experiment"], w["period"])
        assert g["warming"]["median"] == pytest.approx(w["warming"]["median"])
        assert g["warming"]["selected"] == pytest.approx(w["warming"]["p90"])
        for gv, wv in zip(g["variants"], w["variants"]):
            assert gv["name"] == wv["name"] and gv["mean"] == pytest.approx(wv["mean"])
            for t, sh in wv["share_at_or_above"].items():
                assert gv["share_at_or_above"][t] == pytest.approx(sh)
    assert got["people_exposure"] and set(got["people_exposure"][0]["people_ge"]) == {"88", "92.5"}


def test_climate_without_factors_is_404(client, ctx, synth_run):
    from pathlib import Path

    rid, rd = synth_run
    proj = ctx.db.fetchone("SELECT dir FROM projects")
    (Path(proj["dir"]) / "inputs" / "climate" / "demo_cmip6.csv").unlink()
    (rd / "climate.json").unlink()
    m = json.loads((rd / "manifest.json").read_text())
    m.pop("climate", None)
    (rd / "manifest.json").write_text(json.dumps(m))
    info = client.get(f"/api/runs/{rid}/climate/factors").json()
    assert info["present"] is False and info["action"]["kind"] == "fetch_input"
    r = client.post(f"/api/runs/{rid}/climate/explore", json={"adaptations": []})
    assert r.status_code == 404 and r.json()["error"]["code"] == "no_climate_factors"
    assert r.json()["error"]["action"]["path"].endswith("/inputs/cmip6")
