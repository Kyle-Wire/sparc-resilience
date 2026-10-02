"""Engine states, preflight and bookkeeping that need no engine host (api.md §7.2, §7.8)."""

from __future__ import annotations

import json
import sys
import threading
import time

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


def test_scenario_run_checks_memory_unless_cached(client, ctx, demo, run_ctx, synth_run, monkeypatch):
    """An exact run on a run that is not loaded makes the host load it: the memory preflight applies, except for
    a cache hit, which needs no engine."""
    from sparc.studio.jobs import resources
    from tests.studio.engine.conftest import make_result

    rid, rd = synth_run
    (rd / "checkpoint.pkl").write_bytes(b"x" * 10_000_000)
    sc = client.post(f"/api/projects/{demo['id']}/scenarios", json={"doc": {
        "name": "s", "edits": [{"lever": "canopy", "mode": "add", "amount": 10}]}}).json()
    monkeypatch.setattr(resources, "memory_available_gb", lambda exclude_rss_mb=0.0: 0.5)
    r = client.post(f"/api/scenarios/{sc['id']}/run", json={"run_id": rid})
    assert r.status_code == 409 and r.json()["error"]["code"] == "engine_memory"
    r = client.post("/api/scenarios/run-batch", json={"run_id": rid, "scenario_ids": [sc["id"]]})
    assert r.status_code == 409 and r.json()["error"]["code"] == "engine_memory"
    assert ctx.db.fetchval("SELECT COUNT(*) FROM jobs") == 0
    row = make_result(ctx, run_ctx, scenario=sc)
    r = client.post(f"/api/scenarios/{sc['id']}/run", json={"run_id": rid})
    assert r.status_code == 200 and r.json()["cached"]["id"] == row["id"]


def test_executor_indexes_results_before_the_job_ends(ctx, demo, run_ctx, synth_run, tmp_path):
    """The executor indexes a succeeded job's result directories (and moves the scenario to ``exact``) before
    it reports the exit, so a client that sees the job succeed also sees its results."""
    import anyio

    from sparc.studio.engine.executor import EngineExecutor
    from sparc.studio.scenarios import library
    from tests.studio.engine.conftest import make_result

    rid, _ = synth_run
    srow = library.create(ctx.db, ctx.workspace, demo["id"], {"name": "s", "edits": [
        {"lever": "canopy", "mode": "add", "amount": 10}]})
    sc = {"id": srow["id"], "revision": 1, "content_hash": srow["content_hash"]}
    row = make_result(ctx, run_ctx, scenario=sc)
    ctx.db.execute("DELETE FROM results WHERE id = ?", (row["id"],))      # as the host left it: files only
    library.sync_status(ctx.db, srow["id"])
    assert library.get_row(ctx.db, srow["id"])["status"] == "draft"
    job_dir = tmp_path / "j_fake"
    job_dir.mkdir()
    (job_dir / "result.json").write_text(json.dumps({"status": "succeeded", "exit_code": 0, "error": None,
                                                     "result": {"result_id": row["id"], "results": [row["id"]]}}))
    ex = EngineExecutor(poll_s=0.01)
    ex.bind(ctx)
    job = {"id": "j_fake", "kind": "engine.scenario", "run_id": rid, "job_dir": str(job_dir),
           "params_json": json.dumps({"run_id": rid, "scenario_id": srow["id"], "revision": 1})}
    exit_info = anyio.run(ex.wait, job)
    assert exit_info.code == 0
    assert ctx.db.fetchone("SELECT id FROM results WHERE id = ?", (row["id"],)) is not None
    assert library.get_row(ctx.db, srow["id"])["status"] == "exact"


class _FakeListener:
    def __init__(self):
        self.closed = threading.Event()

    def accept(self):
        self.closed.wait(5)
        raise OSError("listener closed")

    def close(self):
        self.closed.set()


def test_host_idle_watchdog():
    """The host stops after ``idle_min`` without a loaded run or a request in progress, and not before."""
    from types import SimpleNamespace

    from sparc.studio.engine.host import Host

    lst = _FakeListener()
    host = Host(lst, idle_min=0.001, watch_s=0.02)                 # 60 ms
    host.sessions["run"] = SimpleNamespace()                       # a loaded run keeps it up
    t = threading.Thread(target=host.serve, daemon=True)
    t.start()
    time.sleep(0.4)
    assert not host.stopping.is_set()
    with host.state_lock:
        host.sessions.clear()
        host.busy = {"op": "scenario"}                             # so does a request in progress
    time.sleep(0.4)
    assert not host.stopping.is_set()
    with host.state_lock:
        host.busy = None
        host.last_activity = time.monotonic()
    assert host.stopping.wait(3.0)
    t.join(3.0)
    assert not t.is_alive() and lst.closed.is_set()


def test_incompatible_at_the_first_evaluation(tmp_path):
    """A session opened from a cached baseline pass whose first engine pass fails with pickle drift becomes
    ``incompatible``: recorded for the checkpoint, evicted, and the cached pass dropped; once a session has
    evaluated, the same error is reported as itself."""
    from types import SimpleNamespace

    from sparc.core.session import IncompatibleCheckpoint, checkpoint_key
    from sparc.studio.engine import ops
    from sparc.studio.engine.host import Entry, Host

    rd = tmp_path / "run"
    sd = rd / "studio"
    (sd / "engine").mkdir(parents=True)
    (rd / "checkpoint.pkl").write_bytes(b"x")
    for name in ("base_fold.npy", "base_fold.json"):
        (sd / "engine" / name).write_bytes(b"{}")

    def broken_run(spec, keep_frame=False):
        raise AttributeError("'DecisionTreeRegressor' object has no attribute 'monotonic_cst'")

    session = SimpleNamespace(engine=SimpleNamespace(clip_support=True, mediators=None, run=broken_run),
                              set_threads=lambda n: None)

    def fake_op(op, sess, payload, job_id=None):
        return ops.evaluate(sess, [{"variable": "canopy", "mode": "add", "amount": 1.0}], None, "x")

    host = Host(None, engine_dir=sd / "engine")
    req = {"op": "scenario", "run_id": "r1", "run_dir": str(rd), "threads": 1, "payload": {}}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ops, "run_op", fake_op)
        host.sessions["r1"] = Entry(session=session, run_dir=str(rd), studio_dir=str(sd), loaded_utc="t",
                                    last_used_utc="t", est_rss_mb=1.0, code_match=True)
        with pytest.raises(IncompatibleCheckpoint, match="monotonic_cst"):
            host._dispatch(req)
        st = json.loads((sd / "engine" / "status.json").read_text())
        assert st["state"] == "incompatible" and st["ckpt_key"] == checkpoint_key(rd)
        assert "r1" not in host.sessions and host.status()["incompatible"]["r1"]["type"] == "Incompatible"
        assert not (sd / "engine" / "base_fold.npy").exists() and not (sd / "engine" / "base_fold.json").exists()
        # a session that already evaluated: the original error, no state change
        (sd / "engine" / "status.json").unlink()
        host.sessions["r1"] = Entry(session=session, run_dir=str(rd), studio_dir=str(sd), loaded_utc="t",
                                    last_used_utc="t", est_rss_mb=1.0, code_match=True, evaluated=True)
        with pytest.raises(AttributeError, match="monotonic_cst"):
            host._dispatch(req)
        assert not (sd / "engine" / "status.json").exists() and "r1" in host.sessions


def test_loading_state_reports_progress_and_step(client, ctx, synth_run):
    """While an ``engine.open`` job runs, the run's engine is ``loading`` with the job's progress and a readable
    step (the unpickle tick label once the tailer has one)."""
    from sparc.studio.workspace import utc_now

    rid, rd = synth_run
    (rd / "checkpoint.pkl").write_bytes(b"x" * 1000)
    ctx.db.insert("jobs", {"id": "j_open1", "kind": "engine.open", "lane": "engine", "executor": "engine",
                           "run_id": rid, "params_json": json.dumps({"run_id": rid}), "status": "running",
                           "job_dir": "/nonexistent", "created_utc": utc_now(), "progress": 0.42,
                           "current_path": json.dumps(["task:unpickle[checkpoint.pkl]"])})
    st = client.get(f"/api/runs/{rid}/engine").json()
    assert st["state"] == "loading" and st["job_id"] == "j_open1"
    assert st["progress"] == pytest.approx(0.42) and st["step"] == "Loading checkpoint"
    # a live tailer's tick label refines the step
    from types import SimpleNamespace

    ctx.jobs.tailers["j_open1"] = SimpleNamespace(state={"tick": {"label": "312/525 MB", "frac": 0.6},
                                                         "current_path": ["task:unpickle[checkpoint.pkl]"]})
    try:
        assert client.get(f"/api/runs/{rid}/engine").json()["step"] == "Loading checkpoint 312/525 MB"
    finally:
        ctx.jobs.tailers.pop("j_open1", None)
    ctx.db.update("jobs", {"id": "j_open1"}, {"status": "failed", "error_json": json.dumps(
        {"type": "UnpicklingError", "message": "truncated"})})
    st = client.get(f"/api/runs/{rid}/engine").json()
    assert st["state"] == "error" and st["error"]["message"] == "truncated"
    assert st["action"]["kind"] == "open_engine"


@pytest.mark.skipif(sys.platform == "win32", reason="AF_UNIX paths only (Windows uses a named pipe)")
def test_host_starts_in_a_deep_workspace(tmp_path):
    """A workspace whose ``engine/host.sock`` is longer than AF_UNIX allows binds a short temp-dir socket
    instead; ``host.json`` records it, clients reach the host through it, and a stop removes it."""
    from pathlib import Path
    from types import SimpleNamespace

    from sparc.studio.engine.client import EngineClient
    from sparc.studio.engine.host import unix_bind_path

    deep = tmp_path.joinpath(*(["a-rather-long-workspace-folder-name"] * 4))
    nominal = deep / "engine" / "host.sock"
    assert len(str(nominal)) > 108
    bound = unix_bind_path(nominal)
    assert len(bound) <= 100 and bound == unix_bind_path(nominal) != unix_bind_path(tmp_path / "engine" / "host.sock")
    assert unix_bind_path(tmp_path / "host.sock") == str(tmp_path / "host.sock")

    cl = EngineClient(SimpleNamespace(engine_dir=deep / "engine"), idle_min=5.0, threads=1)
    try:
        info = cl.start()
        assert info["sock"] == bound and Path(bound).exists()
        assert cl.status() is not None
    finally:
        cl.stop()
    assert not Path(bound).exists() and cl.info() is None
