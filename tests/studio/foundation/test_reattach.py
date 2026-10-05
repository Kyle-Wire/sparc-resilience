"""Reattach after a server restart (SPEC §5.9): live jobs are picked up again by pid + create_time + cmdline
token; jobs that vanished while the server was down become ``interrupted``."""

from __future__ import annotations

import json
import os

import psutil
from fastapi.testclient import TestClient

from tests.studio.conftest import AUTH, read_jsonl, wait_for


def _start_job(make_app, wait_job, kind="test.sleep", until=None, **params):
    app = make_app()
    with TestClient(app, headers=AUTH) as client:
        jid = client.post("/api/jobs", json={"kind": kind, "params": params}).json()["id"]
        wait_job(client, jid, ("running",))
        wait_for(lambda: client.get(f"/api/jobs/{jid}").json()["progress"], 10, what="a flushed projection")
        if until is not None:
            wait_for(lambda: until(app.state.studio, jid), 30, what="the first app to flush rows")
        pid = app.state.studio.db.fetchval("SELECT pid FROM jobs WHERE id = ?", (jid,))
        ws = app.state.studio.workspace
    # the first app is gone; its job keeps running in its own session
    assert psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    return jid, pid, ws


def _kill(pid):
    from tests.studio.conftest import kill_tree

    try:
        kill_tree(pid)
    except (ProcessLookupError, psutil.Error):
        pass
    try:
        os.waitpid(pid, 0)                     # it is a child of this test process
    except (ChildProcessError, OSError):
        pass


def test_job_survives_restart_and_is_reattached(make_app, wait_job):
    def flushed_metrics(ctx, jid):
        return ctx.db.fetchval("SELECT COUNT(*) FROM metrics WHERE job_id = ?", (jid,))

    jid, pid, ws = _start_job(make_app, wait_job, kind="test.events", until=flushed_metrics, seconds=8.0)
    try:
        app2 = make_app()
        with TestClient(app2, headers=AUTH) as client:
            job = client.get(f"/api/jobs/{jid}").json()
            assert job["status"] == "running"
            assert jid in app2.state.studio.jobs.tailers                # tailing again
            done = wait_job(client, jid, timeout=30)
            assert done["status"] == "succeeded" and done["progress"] == 1.0
            assert done["result"]["ok"] is True
            snap = client.get(f"/api/jobs/{jid}/tracker").json()
            assert snap["stages"]["S7"]["state"] == "done"
            assert snap["spans"] and all(s["status"] == "ok" for s in snap["spans"])
            # rows flushed by the first app are not inserted again by the second
            evs = [ev for _, ev in read_jsonl(ws.job_dir(jid) / "events.jsonl")]
            db = app2.state.studio.db
            assert db.fetchval("SELECT COUNT(*) FROM metrics WHERE job_id = ?", (jid,)) == \
                sum(1 for ev in evs if ev["type"] == "metric")
            assert db.fetchval("SELECT COUNT(*) FROM artifacts WHERE job_id = ?", (jid,)) == 8
            assert db.fetchval("SELECT COUNT(*) FROM checkpoints WHERE job_id = ?", (jid,)) == 2
            assert db.fetchval("SELECT count FROM warnings WHERE job_id = ? AND code = 'qa.coarse'", (jid,)) == 2
            assert db.fetchval("SELECT COUNT(*) FROM spans WHERE job_id = ?", (jid,)) == len(snap["spans"])
            statuses = [ev["status"] for ev in evs if ev["type"] == "job.status"]
            assert statuses == ["queued", "starting", "running", "succeeded"]
    finally:
        _kill(pid)


def test_job_killed_while_down_is_interrupted(make_app, wait_job):
    jid, pid, ws = _start_job(make_app, wait_job, seconds=60.0)
    _kill(pid)
    assert not psutil.pid_exists(pid)
    app2 = make_app()
    with TestClient(app2, headers=AUTH) as client:
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "interrupted"
        assert job["error"]["type"] == "Interrupted" and job["error"]["message"].startswith("process vanished")
        assert jid not in app2.state.studio.jobs.tailers
        evs = [ev for _, ev in read_jsonl(ws.job_dir(jid) / "events.jsonl")]
        assert evs[-1]["type"] == "job.status" and evs[-1]["status"] == "interrupted"
        snap = client.get(f"/api/jobs/{jid}/tracker").json()
        assert all(s["status"] != "running" for s in snap["spans"])


def test_create_time_mismatch_is_not_reattached(make_app, wait_job):
    jid, pid, ws = _start_job(make_app, wait_job, seconds=60.0)
    try:
        state_path = ws.job_dir(jid) / "state.json"
        state = json.loads(state_path.read_text())
        state["proc_create_time"] = state["proc_create_time"] - 1000.0      # "a different process with that pid"
        state_path.write_text(json.dumps(state))
        app2 = make_app()
        with TestClient(app2, headers=AUTH) as client:
            job = client.get(f"/api/jobs/{jid}").json()
            assert job["status"] == "interrupted"
            assert jid not in app2.state.studio.jobs.tailers
        assert psutil.pid_exists(pid)                                    # a foreign process is never signalled
    finally:
        _kill(pid)


def test_job_that_finished_while_down_keeps_its_result(make_app, wait_job):
    jid, pid, ws = _start_job(make_app, wait_job, seconds=1.0)
    wait_for(lambda: (ws.job_dir(jid) / "result.json").exists(), 30, what="the job to finish")
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    app2 = make_app()
    with TestClient(app2, headers=AUTH) as client:
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "succeeded" and job["result"]["ok"] is True


def test_reindex_keeps_a_live_job_and_reattaches_it(make_app, wait_job, tmp_path):
    """``--reindex`` while a job runs: the rebuilt row stays live, reattach tails it to the end and the job
    takes its run's write lock again (``run_locks`` is emptied by the reindex)."""
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    app = make_app()
    with TestClient(app, headers=AUTH) as client:
        app.state.studio.db.insert("runs", {"id": "r1", "run_dir": str(run_dir), "studio_dir": str(run_dir / "studio"),
                                            "origin": "studio", "status": "complete"})
        r = client.post("/api/jobs", json={"kind": "test.lock", "params": {"seconds": 4.0}, "run_id": "r1"})
        jid = r.json()["id"]
        wait_job(client, jid, ("running",))
        pid = app.state.studio.db.fetchval("SELECT pid FROM jobs WHERE id = ?", (jid,))
        ws = app.state.studio.workspace
    try:
        for suffix in ("", "-wal", "-shm"):
            p = ws.root / f"studio.sqlite{suffix}"
            if p.exists():
                p.unlink()
        app2 = make_app(reindex=True)
        with TestClient(app2, headers=AUTH) as client:
            db = app2.state.studio.db
            assert client.get(f"/api/jobs/{jid}").json()["status"] == "running"
            assert jid in app2.state.studio.jobs.tailers
            assert db.fetchval("SELECT job_id FROM run_locks WHERE run_id = 'r1'") == jid
            done = wait_job(client, jid, timeout=30)
            assert done["status"] == "succeeded" and done["result"]["run_id"] == "r1"
            assert db.fetchval("SELECT COUNT(*) FROM run_locks") == 0
    finally:
        _kill(pid)
