"""Engine states, preflight and bookkeeping that need no engine host (api.md §7.2, §7.8)."""

from __future__ import annotations

import json

import pytest


def test_engine_states_without_a_host(client, ctx, synth_run):
    rid, rd = synth_run
    st = client.get(f"/api/runs/{rid}/engine").json()
    assert st["state"] == "no_checkpoint"
    r = client.post(f"/api/runs/{rid}/engine/open")
    assert r.status_code == 409 and r.json()["error"]["code"] == "no_checkpoint"
    (rd / "checkpoint.pkl").write_bytes(b"x" * 1000)
    st = client.get(f"/api/runs/{rid}/engine").json()
    assert st["state"] == "cold" and st["action"]["path"] == f"/api/runs/{rid}/engine/open"
    host = client.get("/api/engine").json()
    s = ctx.settings()
    assert host["state"] == "absent" and host["pid"] is None and host["runs"] == []
    assert host["budget_gb"] == s.engine_mem_budget_gb and host["max_runs"] == s.engine_max_runs
    assert client.get("/api/health").json()["engine"]["state"] == "absent"


def test_memory_preflight_refuses_with_holders_and_action(client, ctx, synth_run, monkeypatch):
    from sparc.studio.jobs import resources

    rid, rd = synth_run
    (rd / "checkpoint.pkl").write_bytes(b"x" * 10_000_000)
    monkeypatch.setattr(resources, "memory_available_gb", lambda exclude_rss_mb=0.0: 0.5)
    r = client.post(f"/api/runs/{rid}/engine/open")
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "engine_memory"
    assert err["detail"]["needed_gb"] == pytest.approx(3.5 * 0.01 + 0.3 + 1.0, abs=0.01)
    assert err["detail"]["available_gb"] == 0.5 and isinstance(err["detail"]["holders"], list)
    assert ctx.db.fetchval("SELECT COUNT(*) FROM jobs WHERE kind = 'engine.open'") == 0


def test_incompatible_state_follows_the_checkpoint(client, ctx, synth_run):
    from sparc.core.session import checkpoint_key

    rid, rd = synth_run
    (rd / "checkpoint.pkl").write_bytes(b"x" * 1000)
    (rd / "studio" / "engine").mkdir(parents=True, exist_ok=True)
    status = {"state": "incompatible", "ckpt_key": checkpoint_key(rd),
              "error": {"type": "Incompatible", "message": "AttributeError: Can't get attribute 'Old'"}}
    (rd / "studio" / "engine" / "status.json").write_text(json.dumps(status))
    st = client.get(f"/api/runs/{rid}/engine").json()
    assert st["state"] == "incompatible" and "Old" in st["error"]["message"]
    assert st["action"]["kind"] == "run_job" and st["action"]["path"] == f"/api/runs/{rid}/rerun"
    (rd / "checkpoint.pkl").write_bytes(b"y" * 2000)            # a refit wrote a new checkpoint
    assert client.get(f"/api/runs/{rid}/engine").json()["state"] == "cold"


def test_engine_kinds_are_registered(client):
    kinds = {k["kind"]: k for k in client.get("/api/meta").json()["job_kinds"]}
    for k in ("engine.open", "engine.scenario", "engine.batch", "engine.rerun_configured", "engine.sweep",
              "engine.plan_verify", "engine.plan_frontier"):
        assert kinds[k]["lane"] == "engine" and kinds[k]["executor"] == "engine"
    assert kinds["scenario.across_runs"]["lane"] == "heavy"
    for k in ("export.decision_pack", "export.plan_pack", "export.compare_pack"):
        assert kinds[k]["lane"] == "medium" and kinds[k]["executor"] == "process"
    from sparc.studio.db import _REINDEX_HOOKS
    from sparc.studio.jobs.executors import registered_executors

    assert "engine" in registered_executors()
    assert {"engine.results", "scenarios"} <= {h[1] for h in _REINDEX_HOOKS}


def test_sweep_listing_and_delete(client, ctx, run_ctx, synth_run):
    from sparc.studio.scenarios import sweeps
    from sparc.studio.workspace import utc_now

    rid, _ = synth_run
    params = sweeps.create(ctx.db, run_ctx, "canopy", [10, 5, 0, 20], None)
    assert params["doses"] == [5.0, 10.0, 20.0]
    rows = client.get(f"/api/runs/{rid}/sweeps").json()
    assert [r["id"] for r in rows] == [params["id"]] and rows[0]["lever"] == "canopy"
    out = client.get(f"/api/sweeps/{params['id']}").json()
    assert out["curve"] == [] and out["pipeline_curve"]["dose"][0] == 0.0
    ctx.db.insert("jobs", {"id": "j_running1", "kind": "engine.sweep", "lane": "engine", "executor": "engine",
                           "params_json": "{}", "status": "running", "job_dir": "/nonexistent",
                           "created_utc": utc_now()})
    ctx.db.update("sweeps", {"id": params["id"]}, {"job_id": "j_running1"})
    r = client.delete(f"/api/sweeps/{params['id']}")
    assert r.status_code == 409 and r.json()["error"]["code"] == "active"
    ctx.db.update("jobs", {"id": "j_running1"}, {"status": "failed"})
    assert client.delete(f"/api/sweeps/{params['id']}").json() == {"ok": True}
    assert client.get(f"/api/runs/{rid}/sweeps").json() == []
    bad = client.post(f"/api/runs/{rid}/sweeps", json={"lever": "elevation", "doses": [1]})
    assert bad.status_code in (409, 422)
