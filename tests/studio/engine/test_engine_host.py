"""The engine host (SPEC §7.6, api.md §13) through the API: open, exact scenarios with fold ticks, the result
directory, the cache, cancel, the LRU, reconnect after a server restart, incompatible checkpoints, restart."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.studio.conftest import AUTH, wait_for
from tests.studio.engine.conftest import ENGINE_RUN_ID, kill_host, write_incompatible_checkpoint

pytestmark = pytest.mark.slow

SHADE = {"name": "Shade", "edits": [{"lever": "canopy", "mode": "add", "amount": 10,
                                     "where": {"kind": "top", "column": "pred:target", "frac": 0.2,
                                               "direction": "highest"}}]}


def _events(ws, jid) -> list[dict]:
    p = Path(ws.job_dir(jid)) / "events.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _open(client, wait_job, rid) -> dict:
    r = client.post(f"/api/runs/{rid}/engine/open")
    assert r.status_code in (200, 202), r.text
    if r.status_code == 202:
        job = wait_job(client, r.json()["id"], timeout=180)
        assert job["status"] == "succeeded", job
    return client.get(f"/api/runs/{rid}/engine").json()


def _scenario(client, pid, doc=SHADE) -> dict:
    r = client.post(f"/api/projects/{pid}/scenarios", json={"doc": doc})
    assert r.status_code == 201, r.text
    return r.json()


def _host_pid(client) -> int | None:
    return client.get("/api/engine").json()["pid"]


def test_open_scenario_ticks_result_dir_and_cache(client, ctx, demo, engine_run, wait_job):
    rid, rd = engine_run
    st = client.get(f"/api/runs/{rid}/engine").json()
    assert st["state"] == "cold" and st["action"]["kind"] == "open_engine" and st["est_rss_mb"] > 0
    st = _open(client, wait_job, rid)
    assert st["state"] == "ready" and st["code_match"] is True and st["loaded_utc"]
    assert (rd / "studio" / "engine" / "base_fold.npy").is_file()
    meta = json.loads((rd / "studio" / "engine" / "base_fold.json").read_text())
    assert meta["ckpt_key"] and meta["code_sha"]
    host = client.get("/api/engine").json()
    assert host["state"] == "ready" and [r["run_id"] for r in host["runs"]] == [rid]
    # an already loaded run answers 200 with its status
    assert client.post(f"/api/runs/{rid}/engine/open").status_code == 200
    sc = _scenario(client, demo["id"])
    r = client.post(f"/api/scenarios/{sc['id']}/run", json={"run_id": rid})
    assert r.status_code == 202, r.text
    job = wait_job(client, r.json()["job"]["id"], timeout=180)
    assert job["status"] == "succeeded", job
    res_id = job["result"]["result_id"]
    K = int(json.loads((rd / "manifest.json").read_text())["cv"]["n_folds"])
    ticks = [(e["k"], e["n"]) for e in _events(ctx.workspace, job["id"]) if e["type"] == "tick"
             and e.get("unit") == "engine_pass"]
    assert ticks == [(k, K) for k in range(1, K + 1)]
    # the result directory (api.md §12.2)
    rdir = rd / "studio" / "results" / res_id
    assert {"spec.json", "summary.json", "cells.parquet", "folds.npy", "warnings.json"} <= {p.name for p in rdir.iterdir()}
    spec = json.loads((rdir / "spec.json").read_text())
    assert spec["scenario_id"] == sc["id"] and spec["revision"] == 1 and spec["content_hash"] == sc["content_hash"]
    assert spec["run_id"] == rid and spec["ckpt_key"] and spec["code_sha"] and spec["compiled"]["levers"]["canopy"]
    cells = pd.read_parquet(rdir / "cells.parquet")
    assert list(cells.columns) == ["id", "delta", "delta_sd", "extrapolation", "realized_canopy"]
    assert all(cells[c].dtype == np.float32 for c in cells.columns[1:])
    folds = np.load(rdir / "folds.npy")
    assert folds.shape == (K, len(cells)) and folds.dtype == np.float32
    np.testing.assert_allclose(folds.mean(axis=0), cells["delta"].to_numpy(), atol=1e-5)
    full = client.get(f"/api/results/{res_id}").json()
    assert full["summary"]["has_folds"] and full["city"]["se"] is not None and full["plain"]["headline"]
    assert client.get(f"/api/scenarios/{sc['id']}").json()["status"] == "exact"
    # an identical second run is a cache hit
    r = client.post(f"/api/scenarios/{sc['id']}/run", json={"run_id": rid})
    assert r.status_code == 200 and r.json()["cached"]["id"] == res_id
    # forced: a new result, same numbers
    r = client.post(f"/api/scenarios/{sc['id']}/run", json={"run_id": rid, "force": True})
    assert r.status_code == 202
    job2 = wait_job(client, r.json()["job"]["id"], timeout=180)
    cells2 = pd.read_parquet(rd / "studio" / "results" / job2["result"]["result_id"] / "cells.parquet")
    np.testing.assert_allclose(cells2["delta"].to_numpy(), cells["delta"].to_numpy(), atol=1e-6)


def test_cancel_between_folds_keeps_the_host(client, ctx, demo, engine_run, wait_job):
    rid, _ = engine_run
    _open(client, wait_job, rid)
    pid = _host_pid(client)
    ids = [_scenario(client, demo["id"], {**SHADE, "name": f"s{i}", "edits": [{**SHADE["edits"][0], "amount": 1 + i}]})["id"]
           for i in range(40)]
    r = client.post("/api/scenarios/run-batch", json={"run_id": rid, "scenario_ids": ids})
    assert r.status_code == 202
    jid = r.json()["job"]["id"]
    wait_for(lambda: any(e["type"] == "tick" and e.get("unit") == "engine_pass"
                         for e in _events(ctx.workspace, jid)), 120, what="the first fold tick")
    assert client.post(f"/api/jobs/{jid}/cancel").status_code == 202
    job = wait_job(client, jid, timeout=120)
    assert job["status"] == "cancelled"
    assert any(e["type"] == "cancel.ack" for e in _events(ctx.workspace, jid))
    assert len(list((Path(ctx.db.fetchone("SELECT studio_dir FROM runs WHERE id = ?", (rid,))["studio_dir"])
                     / "results").glob("*/summary.json"))) < 40
    assert _host_pid(client) == pid and client.get("/api/engine").json()["state"] == "ready"
    r = client.post(f"/api/scenarios/{ids[0]}/run", json={"run_id": rid, "force": True})
    assert wait_job(client, r.json()["job"]["id"], timeout=120)["status"] == "succeeded"
    assert _host_pid(client) == pid


def _second_run(demo, place_run, engine_run_dir) -> str:
    rid = ENGINE_RUN_ID.replace("e0e0", "e1e1")
    place_run(demo, engine_run_dir, rid)
    return rid


def test_lru_evicts_by_count(client, demo, engine_run, place_run, engine_run_dir, wait_job):
    rid_a, _ = engine_run
    rid_b = _second_run(demo, place_run, engine_run_dir)
    assert client.put("/api/settings", json={"engine_max_runs": 1}).status_code == 200
    _open(client, wait_job, rid_a)
    _open(client, wait_job, rid_b)
    host = client.get("/api/engine").json()
    assert [r["run_id"] for r in host["runs"]] == [rid_b] and host["max_runs"] == 1
    assert client.get(f"/api/runs/{rid_a}/engine").json()["state"] == "cold"
    # explicit eviction
    st = client.delete(f"/api/runs/{rid_b}/engine").json()
    assert st["state"] == "cold" and client.get("/api/engine").json()["runs"] == []


def test_lru_evicts_by_memory_budget(client, demo, engine_run, place_run, engine_run_dir, wait_job, monkeypatch):
    monkeypatch.setenv("SPARC_STUDIO_ENGINE_SLACK_GB", "1000")    # no recycle on this tiny budget
    rid_a, _ = engine_run
    rid_b = _second_run(demo, place_run, engine_run_dir)
    assert client.put("/api/settings", json={"engine_max_runs": 4, "engine_mem_budget_gb": 0.01}).status_code == 200
    _open(client, wait_job, rid_a)
    _open(client, wait_job, rid_b)
    host = client.get("/api/engine").json()
    assert [r["run_id"] for r in host["runs"]] == [rid_b] and host["budget_gb"] == 0.01


def test_incompatible_checkpoint_state(client, ctx, demo, place_run, engine_run_dir, wait_job):
    rid = ENGINE_RUN_ID.replace("e0e0", "dead")
    rd = place_run(demo, engine_run_dir, rid)
    write_incompatible_checkpoint(rd)
    r = client.post(f"/api/runs/{rid}/engine/open")
    assert r.status_code == 202
    job = wait_job(client, r.json()["id"], timeout=120)
    assert job["status"] == "failed" and job["error"]["type"] == "Incompatible"
    st = client.get(f"/api/runs/{rid}/engine").json()
    assert st["state"] == "incompatible" and "GhostEnsemble" in st["error"]["message"]
    assert st["action"]["kind"] == "run_job" and "Refit S2/S3" in st["action"]["label"]


def test_untrusted_run_is_refused(client, ctx, demo, engine_run_dir, tmp_path):
    outside = tmp_path / "elsewhere" / "run"
    shutil.copytree(engine_run_dir, outside)
    r = client.post("/api/runs/import", json={"dir": str(outside), "project_id": demo["id"],
                                               "config_path": demo["config_path"]})
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    r = client.post(f"/api/runs/{rid}/engine/open")
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "untrusted_pickle" and "execute code" in err["message"]
    assert err["action"]["path"] == "/api/runs/import" and err["action"]["body"]["trust_pickles"] is True
    assert client.get(f"/api/runs/{rid}/engine").json()["action"]["body"]["trust_pickles"] is True
    assert client.post("/api/runs/import", json=err["action"]["body"]).status_code == 201
    assert client.post(f"/api/runs/{rid}/engine/open").status_code == 202


def test_host_survives_a_server_restart(make_app, demo, engine_run_dir, wait_job, client, ctx, place_run):
    from fastapi.testclient import TestClient

    rid = ENGINE_RUN_ID
    place_run(demo, engine_run_dir, rid)
    _open(client, wait_job, rid)
    pid = _host_pid(client)
    ids = [_scenario(client, demo["id"], {**SHADE, "name": f"r{i}", "edits": [{**SHADE["edits"][0], "amount": 2 + i}]})["id"]
           for i in range(12)]
    jid = client.post("/api/scenarios/run-batch", json={"run_id": rid, "scenario_ids": ids}).json()["job"]["id"]
    wait_for(lambda: any(e["type"] == "tick" for e in _events(ctx.workspace, jid)), 120, what="the batch to start")
    client.__exit__(None, None, None)                    # server shutdown with an engine job running
    app2 = make_app()
    with TestClient(app2, headers=AUTH) as c2:
        assert c2.get("/api/engine").json()["pid"] == pid       # reconnected to the same host
        job = wait_job(c2, jid, timeout=180)
        assert job["status"] == "succeeded", job
        assert len(job["result"]["result_ids"]) == 12
        sc = c2.get(f"/api/scenarios/{ids[-1]}").json()
        assert sc["status"] == "exact" and len(sc["results"]) == 1
        # Force stop = host restart: the next request starts a new host
        assert c2.post("/api/engine/restart").status_code == 202
        assert c2.get("/api/engine").json()["state"] == "absent"
        st = _open(c2, wait_job, rid)
        assert st["state"] == "ready" and _host_pid(c2) != pid
    kill_host(app2.state.studio.workspace)


def test_sweep_plans_rerun_and_across_runs(client, ctx, demo, engine_run, place_run, engine_run_dir, wait_job):
    rid, rd = engine_run
    _open(client, wait_job, rid)
    # sweep: one exact run per dose, a fitted curve, sweep_point results
    region = {"kind": "top", "column": "pred:target", "frac": 0.3, "direction": "highest"}
    r = client.post(f"/api/runs/{rid}/sweeps", json={"lever": "canopy", "doses": [5, 10, 20, 30], "selection": region})
    assert r.status_code == 202, r.text
    job = wait_job(client, r.json()["job"]["id"], timeout=180)
    assert job["status"] == "succeeded", job
    sw = client.get(f"/api/sweeps/{r.json()['sweep_id']}").json()
    assert [p["dose"] for p in sw["curve"]] == [5, 10, 20, 30] and len(sw["points"]) == 4
    assert all(p["region"] is not None and p["realized"] > 0 for p in sw["curve"])
    assert sw["fit"]["model"] in ("linear", "saturating", "sigmoid", "insufficient")
    assert sw["status"] == "succeeded" and sw["pipeline_curve"]
    # two fits, each on its own axis: the requested dose (the curve's x) and the neighbourhood dose
    from sparc.studio.scenarios.sweeps import fit_curve

    benefit = [0.0] + [-p["region"]["estimate"] for p in sw["curve"]]
    neigh = [p["neighbourhood_dose"] for p in sw["curve"]]
    assert all(0 < nd < p["dose"] for nd, p in zip(neigh, sw["curve"]))      # regional: smoothed below the dose
    assert sw["fit"] == pytest.approx(fit_curve([0.0, 5, 10, 20, 30], benefit, "dose"))
    assert sw["fit_neighbourhood"] == pytest.approx(fit_curve([0.0] + neigh, benefit, "neighbourhood_dose"))
    kinds = {x["kind"] for x in client.get(f"/api/runs/{rid}/scenarios").json()["results"]}
    assert kinds == {"sweep_point"}
    # plan: verify (closed loop) and the frontier
    r = client.post(f"/api/runs/{rid}/plans", json={"params": {"lever": "canopy", "budget": 1500}, "name": "Trees"})
    assert r.status_code == 201
    plan, vjob = r.json()["plan"], r.json()["job"]
    vjob = wait_job(client, vjob["id"], timeout=180)
    assert vjob["status"] == "succeeded", vjob
    plan = client.get(f"/api/plans/{plan['id']}").json()
    assert plan["realised"]["result_id"] == vjob["result"]["result_id"] and plan["realised"]["total"] != 0
    loop = client.get(f"/api/plans/{plan['id']}/layers/closed_loop_delta.bin")
    assert loop.status_code == 200
    fjob = client.post(f"/api/plans/{plan['id']}/verify", json={"frontier": True}).json()
    fjob = wait_job(client, fjob["id"], timeout=180)
    assert fjob["status"] == "succeeded", fjob
    fr = client.get(f"/api/plans/{plan['id']}").json()["frontier"]
    assert [p["budget"] for p in fr] == [375.0, 750.0, 1500.0, 3000.0] and all(p["realised"] > 0 for p in fr)
    kit = client.post(f"/api/plans/{plan['id']}/field-kit", json={}).json()
    assert kit["cells"][0]["closed_loop_delta"] is not None
    # re-run a configured scenario exactly: a "configured" result with folds
    r = client.post(f"/api/runs/{rid}/configured/cooling-package/rerun-exact")
    assert r.status_code == 202
    cj = wait_job(client, r.json()["id"], timeout=180)
    assert cj["status"] == "succeeded", cj
    res = client.get(f"/api/results/{cj['result']['result_id']}").json()
    assert res["summary"]["kind"] == "configured" and res["summary"]["has_folds"]
    import pandas as pd

    stored = pd.read_parquet(rd / "scenario_deltas.parquet")["Cooling package"].to_numpy()
    assert res["city"]["estimate"] == pytest.approx(float(stored.mean()), abs=1e-6)
    # check across runs: estimate, then a heavy job that loads each engine in its own process
    rid_b = ENGINE_RUN_ID.replace("e0e0", "e2e2")
    place_run(demo, engine_run_dir, rid_b)
    sc = _scenario(client, demo["id"])
    est = client.post(f"/api/scenarios/{sc['id']}/across-runs/estimate", json={"run_ids": [rid, rid_b, "nope"]}).json()
    assert [r["ok"] for r in est["runs"]] == [True, True, False]
    assert est["total_s"] > 0 and est["peak_rss_gb"] > 0
    aj = client.post(f"/api/scenarios/{sc['id']}/across-runs", json={"run_ids": [rid, rid_b]})
    assert aj.status_code == 202
    aj = wait_job(client, aj.json()["id"], timeout=300)
    assert aj["status"] == "succeeded", aj
    rows = aj["result"]["rows"]
    assert all(r["ok"] for r in rows) and aj["result"]["sign_stability"] == 1.0
    assert rows[0]["city"]["estimate"] == pytest.approx(rows[1]["city"]["estimate"], abs=1e-9)
    # the across-runs band feeds the specification of the next exact result
    rj = wait_job(client, client.post(f"/api/scenarios/{sc['id']}/run", json={"run_id": rid}).json()["job"]["id"],
                  timeout=180)
    unc = client.get(f"/api/results/{rj['result']['result_id']}").json()["uncertainty"]
    assert unc["specification"] is not None and "specification: check across runs" in unc["sources"]


def test_host_recycles_and_reopens_the_mru_run(client, ctx, demo, engine_run, wait_job, monkeypatch):
    monkeypatch.setenv("SPARC_STUDIO_ENGINE_SLACK_GB", "-1000")     # every open exceeds budget + slack
    rid, _ = engine_run
    r = client.post(f"/api/runs/{rid}/engine/open")
    assert wait_job(client, r.json()["id"], timeout=180)["status"] == "succeeded"
    # the host recycled after replying, and the server queued a re-open of the most recent run
    reopen = wait_for(lambda: ctx.db.fetchone("SELECT * FROM jobs WHERE kind = 'engine.open' AND label LIKE "
                                              "'Re-open%'"), 60, what="the re-open job")
    done = wait_job(client, reopen["id"], timeout=180)
    assert done["status"] == "succeeded"
    # it recycled again, but a run is re-opened at most once per window: no thrash
    time.sleep(2.0)
    assert ctx.db.fetchval("SELECT COUNT(*) FROM jobs WHERE kind = 'engine.open'") == 2
    wait_for(lambda: client.get("/api/engine").json()["state"] == "absent", 30, what="the recycled host to exit")
