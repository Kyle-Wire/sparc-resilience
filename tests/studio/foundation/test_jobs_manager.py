"""JobManager with the test kinds: lanes, priority, chains, per-run locks, cancel, kill, start timeout, results."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import psutil
import pytest
from fastapi.testclient import TestClient

from tests.studio.conftest import AUTH, read_jsonl, wait_for


def submit(client, kind, **params):
    extra = {k: params.pop(k) for k in ("run_id", "priority", "after_job_id", "project_id") if k in params}
    r = client.post("/api/jobs", json={"kind": kind, "params": params, **extra})
    assert r.status_code == 202, r.text
    return r.json()


def status(client, jid):
    return client.get(f"/api/jobs/{jid}").json()["status"]


def events_of(ctx, jid):
    return [ev for _, ev in read_jsonl(ctx.workspace.job_dir(jid) / "events.jsonl")]


def pgid_members(pgid: int) -> list[int]:
    """Live processes of a job's process group (POSIX); on Windows, which has no process groups to list, the job's
    process tree rooted at ``pgid`` (the worker's pid)."""
    if not hasattr(os, "getpgid"):
        try:
            root = psutil.Process(pgid)
            return [root.pid] + [c.pid for c in root.children(recursive=True) if c.is_running()]
        except psutil.Error:
            return []
    out = []
    for p in psutil.process_iter(["pid"]):
        try:
            if os.getpgid(p.info["pid"]) == pgid and p.status() != psutil.STATUS_ZOMBIE:
                out.append(p.info["pid"])
        except (ProcessLookupError, psutil.NoSuchProcess, PermissionError, psutil.AccessDenied):
            continue
    return out


def add_run(ctx, run_id, tmp_path):
    d = tmp_path / "runs" / run_id
    d.mkdir(parents=True)
    ctx.db.insert("runs", {"id": run_id, "run_dir": str(d), "studio_dir": str(d / "studio"), "origin": "studio",
                           "status": "complete"})
    return d


# ---------------------------------------------------------------------------


def test_job_lifecycle_and_result_json(client, ctx, wait_job):
    job = submit(client, "test.events", seconds=0.5)
    assert job["status"] == "queued" and job["lane"] == "heavy" and job["executor"] == "process"
    done = wait_job(client, job["id"])
    assert done["status"] == "succeeded" and done["exit_code"] == 0 and done["error"] is None
    assert done["progress"] == 1.0 and done["result"]["ok"] is True
    job_dir = ctx.workspace.job_dir(job["id"])
    result = json.loads((job_dir / "result.json").read_text())
    assert result["status"] == "succeeded" and result["exit_code"] == 0 and result["result"]["ok"] is True
    state = json.loads((job_dir / "state.json").read_text())
    assert state["pid"] and state["proc_create_time"] and state["cmdline_token"] == str(job_dir)
    assert state["executor"] == "process" and state["status"] == "succeeded"
    jobj = json.loads((job_dir / "job.json").read_text())
    assert jobj["threads"] >= 1 and jobj["context"]["workspace"] == str(ctx.workspace.root)
    statuses = [ev["status"] for ev in events_of(ctx, job["id"]) if ev["type"] == "job.status"]
    assert statuses == ["queued", "starting", "running", "succeeded"]
    # flushed projection rows
    assert ctx.db.fetchval("SELECT COUNT(*) FROM spans WHERE job_id = ?", (job["id"],)) > 10
    assert ctx.db.fetchval("SELECT count FROM warnings WHERE job_id = ? AND code = 'qa.coarse'", (job["id"],)) == 2
    assert ctx.db.fetchval("SELECT COUNT(*) FROM artifacts WHERE job_id = ?", (job["id"],)) == 8
    row = ctx.db.fetchone("SELECT last_cursor, progress FROM jobs WHERE id = ?", (job["id"],))
    assert row["last_cursor"] > 0 and row["progress"] == 1.0


def test_failure_fills_error(client, ctx, wait_job):
    job = submit(client, "test.fail", message="boom here")
    done = wait_job(client, job["id"])
    assert done["status"] == "failed" and done["exit_code"] == 1
    assert done["error"]["type"] == "RuntimeError" and "boom here" in done["error"]["message"]
    assert "Traceback" in done["error"]["traceback_tail"]
    result = json.loads((ctx.workspace.job_dir(job["id"]) / "result.json").read_text())
    assert result["status"] == "failed" and result["error"]["message"] == "boom here"
    snap = client.get(f"/api/jobs/{job['id']}/tracker").json()
    assert snap["stages"]["S0"]["state"] == "failed"
    retry = client.post(f"/api/jobs/{job['id']}/retry").json()
    assert retry["parent_job_id"] == job["id"] and retry["kind"] == "test.fail"
    assert client.post(f"/api/jobs/{retry['id']}/retry").status_code == 409     # still active


def test_validation_errors(client):
    r = client.post("/api/jobs", json={"kind": "test.sleep", "params": {"seconds": "soon", "extra": 1}})
    assert r.status_code == 422
    paths = {e["path"] for e in r.json()["error"]["detail"]["errors"]}
    assert {"params.seconds", "params.extra"} <= paths
    r = client.post("/api/jobs", json={"kind": "no.such.kind", "params": {}})
    assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_kind"
    r = client.post("/api/jobs", json={"kind": "test.lock", "params": {}})
    assert r.status_code == 422                                                # needs a run
    r = client.post("/api/jobs", json={"kind": "test.lock", "params": {}, "run_id": "missing"})
    assert r.status_code == 404
    assert client.get("/api/jobs/j_nope").status_code == 404


def test_lane_limit_and_priority(client, ctx, wait_job):
    a = submit(client, "test.sleep", seconds=1.5)
    wait_job(client, a["id"], ("running",))
    b = submit(client, "test.sleep", seconds=0.3)
    c = submit(client, "test.sleep", seconds=0.3, priority=5)
    time.sleep(0.6)
    assert status(client, b["id"]) == "queued" and status(client, c["id"]) == "queued"   # heavy lane: 1 slot
    q = client.get("/api/queue").json()
    heavy = next(lane for lane in q["lanes"] if lane["lane"] == "heavy")
    assert heavy["slots"] == 1 and heavy["running"] == [a["id"]] and heavy["queued"] == [c["id"], b["id"]]
    wait_job(client, a["id"])
    wait_job(client, c["id"], ("running", "succeeded"))
    assert status(client, b["id"]) == "queued"                                  # priority 5 went first
    wait_job(client, b["id"])
    listing = client.get("/api/jobs", params={"kind": "test.sleep", "limit": 2}).json()
    assert [j["id"] for j in listing["items"]] == [c["id"], b["id"]] and listing["next_cursor"]
    rest = client.get("/api/jobs", params={"kind": "test.sleep", "cursor": listing["next_cursor"]}).json()
    assert [j["id"] for j in rest["items"]] == [a["id"]] and rest["next_cursor"] is None


def test_queue_pause_resume_and_priority_patch(client, wait_job):
    client.post("/api/queue/pause")
    j = submit(client, "test.sleep", seconds=0.1)
    time.sleep(0.6)
    assert status(client, j["id"]) == "queued"
    assert client.patch(f"/api/jobs/{j['id']}", json={"priority": 3}).json()["priority"] == 3
    assert client.post("/api/queue/resume").json()["paused"] is False
    wait_job(client, j["id"])
    assert client.patch(f"/api/jobs/{j['id']}", json={"priority": 1}).status_code == 409


def test_after_job_chain(client, ctx, wait_job):
    a = submit(client, "test.sleep", seconds=0.8)
    b = submit(client, "test.sleep", seconds=0.1, after_job_id=a["id"])
    wait_job(client, b["id"], ("blocked",))
    blocked = client.get(f"/api/jobs/{b['id']}").json()["blocked"]
    assert a["id"] in blocked["reason"] and blocked["reason"].startswith("waiting for")
    assert wait_job(client, b["id"])["status"] == "succeeded"
    f = submit(client, "test.fail")
    g = submit(client, "test.sleep", seconds=0.1, after_job_id=f["id"])
    done = wait_job(client, g["id"])
    assert done["status"] == "cancelled" and done["error"]["type"] == "DependencyFailed"


def test_deleting_a_succeeded_dependency_keeps_its_dependents(client, ctx, wait_job):
    """A chained job starts when its dependency **succeeds** (SPEC §10.3): deleting that finished job (Activity,
    or retention) while the dependent still waits for a slot must not cancel it as ``DependencyFailed``."""
    a = submit(client, "test.sleep", seconds=0.1)
    assert wait_job(client, a["id"])["status"] == "succeeded"
    assert client.post("/api/queue/pause").status_code == 200           # the dependent waits for its slot
    b = submit(client, "test.sleep", seconds=0.1, after_job_id=a["id"])
    assert client.delete(f"/api/jobs/{a['id']}").status_code == 200
    for _ in range(3):
        client.portal.call(ctx.jobs.schedule)
    assert status(client, b["id"]) == "queued", client.get(f"/api/jobs/{b['id']}").json()
    assert client.post("/api/queue/resume").status_code == 200
    assert wait_job(client, b["id"])["status"] == "succeeded"


def test_per_run_lock_blocks(client, ctx, wait_job, tmp_path):
    add_run(ctx, "r1", tmp_path)
    add_run(ctx, "r2", tmp_path)
    assert client.put("/api/settings", json={"medium_slots": 2}).status_code == 200
    # a holds r1's lock until it is cancelled, so the checks below do not race its end on a slow machine
    a = submit(client, "test.lock", seconds=120, run_id="r1")
    wait_job(client, a["id"], ("running",))
    assert ctx.db.fetchval("SELECT job_id FROM run_locks WHERE run_id = 'r1'") == a["id"]
    b = submit(client, "test.lock", seconds=0.1, run_id="r1")
    c = submit(client, "test.lock", seconds=0.1, run_id="r2")
    blocked = wait_job(client, b["id"], ("blocked",))
    assert blocked["blocked"]["reason"] == "waiting for test.lock on this run"
    assert wait_job(client, c["id"])["status"] == "succeeded"                  # other run, free lane slot
    assert status(client, b["id"]) == "blocked"
    assert client.post(f"/api/jobs/{a['id']}/cancel").status_code == 202      # releasing r1's lock unblocks b
    assert wait_job(client, a["id"])["status"] == "cancelled"
    assert wait_job(client, b["id"])["status"] == "succeeded"
    assert ctx.db.fetchval("SELECT COUNT(*) FROM run_locks") == 0


def _wait_first_tick(ctx, jid, timeout=60.0):
    """Until the worker has ticked: its signal handlers are installed by then.  A cancel that lands earlier
    (the job is "running" from spawn; Windows takes seconds to import) ends the process with the platform's
    termination code instead of 130, which the manager also reports as cancelled."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if any(ev.get("type") == "tick" for ev in events_of(ctx, jid)):
                return
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    raise AssertionError(f"job {jid} never ticked")


def test_cancel_as_soon_as_running_reads_cancelled(client, ctx, wait_job):
    """A cancel right after spawn may beat the worker's signal handlers: the process then ends with SIGTERM's or
    CTRL_BREAK's code rather than 130, and the job must still read cancelled, not failed."""
    from sparc.studio.jobs.manager import _TERMINATED_CODES

    j = submit(client, "test.sleep", seconds=60)
    wait_job(client, j["id"], ("running",))
    assert client.post(f"/api/jobs/{j['id']}/cancel").status_code == 202
    done = wait_job(client, j["id"], timeout=30)
    assert done["status"] == "cancelled" and done["exit_code"] in (130, *_TERMINATED_CODES)


def test_cancel_exits_130(client, ctx, wait_job):
    j = submit(client, "test.sleep", seconds=60)
    wait_job(client, j["id"], ("running",))
    _wait_first_tick(ctx, j["id"])
    r = client.post(f"/api/jobs/{j['id']}/cancel")
    assert r.status_code == 202 and r.json()["status"] in ("cancelling", "cancelled")
    done = wait_job(client, j["id"], timeout=20)
    assert done["status"] == "cancelled" and done["exit_code"] == 130
    evs = events_of(ctx, j["id"])
    assert any(e["type"] == "cancel.requested" and e["by"] == "user" for e in evs)
    assert any(e["type"] == "cancel.ack" for e in evs)
    assert json.loads((ctx.workspace.job_dir(j["id"]) / "result.json").read_text())["status"] == "cancelled"
    assert client.post(f"/api/jobs/{j['id']}/cancel").status_code == 409


def test_cancel_queued_job_is_immediate(client, wait_job):
    client.post("/api/queue/pause")
    j = submit(client, "test.sleep", seconds=5)
    r = client.post(f"/api/jobs/{j['id']}/cancel")
    assert r.json()["status"] == "cancelled"
    client.post("/api/queue/resume")


@pytest.fixture
def grace_client(make_app):
    with TestClient(make_app(kill_grace_s=1.0), headers=AUTH) as c:
        yield c


def test_kill_after_grace_removes_whole_group(grace_client, wait_job):
    client = grace_client
    ctx = client.app.state.studio
    j = submit(client, "test.ignore_sigterm", seconds=120, workers=2)
    wait_job(client, j["id"], ("running",))
    pid = ctx.db.fetchval("SELECT pid FROM jobs WHERE id = ?", (j["id"],))
    pgid = os.getpgid(pid) if hasattr(os, "getpgid") else pid
    assert pgid == pid                                         # its own session / process group
    wait_for(lambda: len(pgid_members(pgid)) >= 3, 30, what="the worker's two children")
    members = pgid_members(pgid)
    client.post(f"/api/jobs/{j['id']}/cancel")
    time.sleep(0.4)
    assert status(client, j["id"]) == "cancelling" and psutil.pid_exists(pid)   # SIGTERM is ignored
    r = client.post(f"/api/jobs/{j['id']}/kill")
    if r.status_code == 409:                                   # still inside the grace period
        assert r.json()["error"]["code"] == "not_ready"
        time.sleep(1.0)
        r = client.post(f"/api/jobs/{j['id']}/kill")
    assert r.status_code == 202
    done = wait_job(client, j["id"], timeout=20)
    assert done["status"] == "cancelled"
    wait_for(lambda: not pgid_members(pgid), 10, what="the process group to disappear")
    assert not any(psutil.pid_exists(p) and psutil.Process(p).status() != psutil.STATUS_ZOMBIE for p in members)
    evs = events_of(ctx, j["id"])
    assert [e["by"] for e in evs if e["type"] == "cancel.requested"] == ["user", "kill"]


def test_kill_before_grace_needs_force(client, wait_job):
    j = submit(client, "test.sleep", seconds=60)
    wait_job(client, j["id"], ("running",))
    r = client.post(f"/api/jobs/{j['id']}/kill")
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_ready"
    r = client.post(f"/api/jobs/{j['id']}/kill", json={"force_now": True})
    assert r.status_code == 202
    assert wait_job(client, j["id"], timeout=20)["status"] == "cancelled"
    assert client.post(f"/api/jobs/{j['id']}/kill", json={"force_now": True}).status_code == 409


def test_kill_pool_job_removes_pool_children(client, ctx, wait_job):
    j = submit(client, "test.pool", seconds=120, workers=2)
    wait_job(client, j["id"], ("running",))
    pid = ctx.db.fetchval("SELECT pid FROM jobs WHERE id = ?", (j["id"],))
    proc = psutil.Process(pid)
    wait_for(lambda: len(proc.children(recursive=True)) >= 2, 60, what="pool workers")
    children = [c.pid for c in proc.children(recursive=True)]
    if hasattr(os, "getpgid"):
        assert all(os.getpgid(c) == pid for c in children)
    assert client.post(f"/api/jobs/{j['id']}/kill", json={"force_now": True}).status_code == 202
    assert wait_job(client, j["id"], timeout=20)["status"] == "cancelled"
    wait_for(lambda: not pgid_members(pid), 10, what="pool processes to disappear")


def test_cancel_pool_job(client, ctx, wait_job):
    j = submit(client, "test.pool", seconds=60, workers=2)
    wait_job(client, j["id"], ("running",))
    wait_for(lambda: any(e["type"] == "task.start" and e["name"] == "pool_task" for e in events_of(ctx, j["id"])),
             60, what="pool tasks")
    client.post(f"/api/jobs/{j['id']}/cancel")
    done = wait_job(client, j["id"], timeout=30)
    assert done["status"] == "cancelled" and done["exit_code"] == 130
    pid = ctx.db.fetchval("SELECT pid FROM jobs WHERE id = ?", (j["id"],))
    wait_for(lambda: not pgid_members(pid), 15, what="pool processes to exit")


def test_starting_timeout(make_app, wait_job):
    app = make_app(start_timeout_s=1.0)
    with TestClient(app, headers=AUTH) as client:
        ctx = app.state.studio
        ctx.jobs.executors["process"].command = lambda job: [
            sys.executable, "-c", "import time, sys; time.sleep(60)", job["job_dir"]]
        j = submit(client, "test.sleep", seconds=1)
        done = wait_job(client, j["id"], timeout=20)
        assert done["status"] == "failed"
        assert done["error"]["message"] == "worker did not start"
        pid = ctx.db.fetchval("SELECT pid FROM jobs WHERE id = ?", (j["id"],))
        wait_for(lambda: not pgid_members(pid), 10, what="the stuck worker to be killed")


def test_delete_job(client, ctx, wait_job):
    j = submit(client, "test.sleep", seconds=0.1)
    wait_job(client, j["id"])
    d = ctx.workspace.job_dir(j["id"])
    assert client.delete(f"/api/jobs/{j['id']}", params={"files": "true"}).json() == {"ok": True}
    assert client.get(f"/api/jobs/{j['id']}").status_code == 404 and not d.exists()


def test_tracker_and_reading_endpoints(client, ctx, wait_job):
    j = submit(client, "test.events", seconds=0.5)
    wait_job(client, j["id"])
    jid = j["id"]
    snap = client.get(f"/api/jobs/{jid}/tracker").json()
    assert snap["job"]["id"] == jid and snap["cursor"] > 0
    assert snap["stages"]["S2_S3"]["state"] == "done" and snap["stages"]["cv_curve"]["state"] == "not_requested"
    assert [n["id"] for n in snap["plan"]][:3] == ["S0", "S1", "S2_S3"]
    assert any(w["code"] == "qa.coarse" and w["count"] == 2 for w in snap["warnings"])
    assert {a["relpath"] for a in snap["artifacts"]} >= {"qa.json", "predictions.csv", "report.md"}
    assert snap["checkpoints"][-1]["done"] == ["S3", "S4"]
    spans = client.get(f"/api/jobs/{jid}/spans").json()
    run = next(s for s in spans if s["kind"] == "run")
    under = client.get(f"/api/jobs/{jid}/spans", params={"under": run["span_id"], "max_depth": 1}).json()
    assert {s["kind"] for s in under} == {"run", "stage"}
    first = client.get(f"/api/jobs/{jid}/events", params={"limit": 3}).json()
    assert len(first["events"]) == 3 and first["eof"] is False
    assert first["events"][0]["cursor"] == 0 and first["events"][0]["type"] == "job.status"
    nxt = client.get(f"/api/jobs/{jid}/events", params={"after": first["next_cursor"]}).json()
    assert nxt["events"][0]["cursor"] > first["next_cursor"] and nxt["eof"] is True
    only = client.get(f"/api/jobs/{jid}/events", params={"types": "warning"}).json()["events"]
    assert {e["type"] for e in only} == {"warning"}
    logs = client.get(f"/api/jobs/{jid}/logs", params={"level": "warning"}).json()["lines"]
    assert logs and all(line["level"] in ("warning", "error") for line in logs)
    assert client.get(f"/api/jobs/{jid}/logs", params={"q": "uniform"}).json()["lines"]
    assert client.get(f"/api/jobs/{jid}/logs", params={"stage": "S1"}).json()["lines"][0]["logger"] == \
        "warning:influence.few_cells"
    m = client.get(f"/api/jobs/{jid}/metrics", params={"names": "influence.range_m,stacker_r2"}).json()
    assert len(m["influence.range_m"]) == 2 and m["stacker_r2"][0]["value"] == pytest.approx(0.66)
    assert client.get(f"/api/jobs/{jid}/warnings").json() == snap["warnings"]
    raw = client.get(f"/api/jobs/{jid}/logs/raw", params={"stream": "events"})
    assert raw.status_code == 200 and raw.text.count("\n") == len(read_jsonl(ctx.workspace.job_dir(jid) / "events.jsonl"))
    txt = client.get(f"/api/jobs/{jid}/logs/raw", params={"stream": "text"}).text
    assert "qa.coarse" in txt and "STAGE" in txt
    assert client.get(f"/api/jobs/{jid}/logs/raw", params={"stream": "stderr"}).status_code == 200
    assert isinstance(client.get(f"/api/jobs/{jid}/resources").json(), list)


def test_settings_validation(client):
    s = client.get("/api/settings").json()
    assert s["heavy_slots"] == 1 and s["network_slots"] == 2 and s["upload_max_gb"] == 2
    r = client.put("/api/settings", json={"thread_budget": 2, "threads_heavy": 2, "engine_threads": 2})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation"
    r = client.put("/api/settings", json={"watch_roots": ["/definitely/not/here"]})
    assert r.status_code == 422 and r.json()["error"]["detail"]["errors"][0]["path"] == "watch_roots.0"
    r = client.put("/api/settings", json={"thread_budget": 4, "threads_heavy": 3, "engine_threads": 2,
                                          "keep_job_logs_days": None})
    assert r.status_code == 200 and r.json()["threads_heavy"] == 3
    assert client.get("/api/settings").json()["threads_heavy"] == 3
    assert client.put("/api/settings", json={"bogus": 1}).status_code == 422


# ---------------------------------------------------------------------------
# server-side hooks, preflight, labelling, external runs, retention
# ---------------------------------------------------------------------------

def test_server_side_hooks_run_in_the_server(client, ctx, wait_job, monkeypatch):
    from sparc.studio.jobs import kinds

    seen, finished = [], []

    def on_event(sctx, job, ev):
        seen.append((os.getpid(), job["id"], ev["type"]))
        if ev["type"] == "run.dir":                       # hooks may write through the server's DB handle
            sctx.db.insert("settings", {"key": f"hook:{job['id']}", "value_json": json.dumps(ev["run_dir"])})

    def on_finish(sctx, job, result):
        finished.append((os.getpid(), job["status"], result))

    k = kinds.get_kind("test.events")
    monkeypatch.setattr(k, "on_event", on_event)
    monkeypatch.setattr(k, "on_finish", on_finish)
    jid = submit(client, "test.events", seconds=0.3)["id"]
    wait_job(client, jid)
    wait_for(lambda: finished, 10, what="on_finish")
    assert {pid for pid, _, _ in seen} == {os.getpid()} and {j for _, j, _ in seen} == {jid}
    assert [t for _, _, t in seen].count("artifact") == 8
    assert {t for _, _, t in seen} == {"run.start", "run.dir", "artifact"}       # the default hook types
    assert finished == [(os.getpid(), "succeeded", finished[0][2])] and finished[0][2]["ok"] is True
    assert Path(ctx.db.get_setting(f"hook:{jid}")).name == "run"


def test_memory_preflight_blocks_with_actions(client, ctx, wait_job, monkeypatch):
    from sparc.studio.jobs import kinds

    k = kinds.get_kind("test.sleep")
    monkeypatch.setattr(k, "estimate", lambda sctx, job, params: {"peak_ram_gb": 1e6, "est_s": 5.0})
    jid = submit(client, "test.sleep", seconds=0.1)["id"]
    assert client.get(f"/api/jobs/{jid}").json()["eta_s"] == 5.0              # estimate shown while queued
    blocked = wait_job(client, jid, ("blocked",))["blocked"]
    assert "GB of memory" in blocked["reason"]
    monkeypatch.setattr(k, "estimate", lambda sctx, job, params: {"peak_ram_gb": 0.01})
    assert wait_job(client, jid)["status"] == "succeeded"


def test_custom_preflight_failure_is_fatal(client, wait_job, monkeypatch):
    from sparc.studio.jobs import kinds

    k = kinds.get_kind("test.sleep")
    monkeypatch.setattr(k, "preflight", lambda sctx, job, params: [{"reason": "inputs are missing", "fatal": True}])
    done = wait_job(client, submit(client, "test.sleep", seconds=0.1)["id"])
    assert done["status"] == "failed" and done["error"]["message"] == "inputs are missing"


def test_out_of_memory_labelling(ctx):
    from sparc.studio.jobs.executors import ExitInfo

    mgr = ctx.jobs
    row = {"id": "j_oom", "peak_rss_mb": 15000.0}
    mgr.samples["j_oom"] = {"rss_mb": 9000.0, "avail_mb": 500.0}
    status, error, _ = mgr._final_status(row, ExitInfo(-9), None, None)
    assert status == "failed" and error["type"] == "OutOfMemory"
    assert error["message"].startswith("likely out of memory (peak 14.6 GB of ")
    mgr.samples["j_oom"] = {"rss_mb": 100.0, "avail_mb": 8000.0}
    status, error, _ = mgr._final_status(row, ExitInfo(-9), None, None)
    assert status == "failed" and error["type"] == "Killed"
    status, error, _ = mgr._final_status({"id": "j_gone"}, ExitInfo(None, reattached=True), None, None)
    assert status == "interrupted" and error["message"] == "process vanished"
    status, _, _ = mgr._final_status({"id": "j_x"}, ExitInfo(None), None, {"run_status": "succeeded", "result": {}})
    assert status == "succeeded"
    status, error, _ = mgr._final_status({"id": "j_y"}, ExitInfo(3), None, None)
    assert status == "failed" and error["message"] == "the worker exited with code 3"


def test_sigterm_after_a_cancel_is_cancelled(ctx, tmp_path):
    """A cancel whose SIGTERM arrives before the worker (or the replay runner) installed its handlers ends the
    process with -15/143: that is the requested cancel, not a failure; an unrequested SIGTERM still fails."""
    from sparc.studio.jobs.executors import ExitInfo

    mgr = ctx.jobs
    for code in (-15, 143):
        status, error, _ = mgr._final_status({"id": "j_c", "status": "cancelling", "job_dir": str(tmp_path)},
                                             ExitInfo(code), None, None)
        assert (status, error) == ("cancelled", None)
    row = {"id": "j_t", "status": "running", "job_dir": str(tmp_path)}
    status, error, _ = mgr._final_status(row, ExitInfo(-15), None, None)
    assert status == "failed" and error["message"] == "the worker exited with signal 15"
    (tmp_path / "cancel").touch()
    assert mgr._final_status(row, ExitInfo(-15), None, None)[0] == "cancelled"


def test_storage_low_is_broadcast(tmp_path, monkeypatch):
    import asyncio
    import shutil

    from sparc.studio.events import EventHub
    from sparc.studio.jobs import resources

    monkeypatch.setattr(shutil, "disk_usage", lambda p: types_ns(total=10, used=9, free=100 * 1024 ** 2))
    hub = EventHub()
    sampler = resources.ResourceSampler(db=None, hub=hub, workspace_root=tmp_path, live_jobs=lambda: {})
    asyncio.run(sampler._check_storage(1000.0))
    asyncio.run(sampler._check_storage(1001.0))                                   # not repeated within 5 min
    asyncio.run(sampler._check_storage(1000.0 + resources.STORAGE_REPEAT_S + 1))  # … but again later
    lows = [r for r in hub.ring if r["type"] == "storage.low"]
    assert len(lows) == 2
    assert lows[0]["free_bytes"] == 100 * 1024 ** 2 and lows[0]["threshold_bytes"] == 2 * 1024 ** 3


def types_ns(**kw):
    import types

    return types.SimpleNamespace(**kw)


def test_worker_environment(ctx):
    from sparc.studio.jobs.executors import worker_env

    job = {"id": "j_env", "job_dir": str(ctx.workspace.job_dir("j_env")), "threads": 3}
    env = worker_env(job, test_kinds=True, base={"SPARC_PROGRESS_LEVEL": "debug", "PATH": "/bin"})
    d = ctx.workspace.job_dir("j_env")
    assert env["SPARC_PROGRESS"] == str(d / "events.jsonl") and env["SPARC_JOB_ID"] == "j_env"
    assert env["SPARC_CANCEL_FILE"] == str(d / "cancel")
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "3"
    assert env["PYTHONUNBUFFERED"] == "1" and env["SPARC_STUDIO_TEST_KINDS"] == "1"
    assert "SPARC_PROGRESS_LEVEL" not in env
    import sparc

    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(Path(sparc.__file__).resolve().parent.parent)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX sessions; Windows workers get their own process group "
                                                 "(CREATE_NEW_PROCESS_GROUP), checked by the cancel tests")
def test_worker_runs_in_its_own_session(client, ctx, wait_job):
    jid = submit(client, "test.sleep", seconds=5)["id"]
    wait_job(client, jid, ("running",))
    pid = ctx.db.fetchval("SELECT pid FROM jobs WHERE id = ?", (jid,))
    assert os.getsid(pid) == pid and os.getpgid(pid) == pid and os.getsid(pid) != os.getsid(0)
    client.post(f"/api/jobs/{jid}/kill", json={"force_now": True})
    wait_job(client, jid)


def test_external_pseudo_job(client, ctx, tmp_path):
    from sparc.studio.events import append_event

    add_run(ctx, "r_ext", tmp_path)
    events = tmp_path / "cli" / "events.jsonl"
    append_event(events, "log", job_id="cli", logger="sparc", level="INFO", msg="external running")
    job = client.portal.call(lambda: ctx.jobs.create_external("run.external", run_id="r_ext", events_path=events,
                                                              label="CLI run", pid=12345))
    jid = job["id"]
    assert job["lane"] == "none" and job["executor"] == "external" and job["status"] == "running"
    wait_for(lambda: client.get(f"/api/jobs/{jid}/events").json()["events"], 10, what="external events")
    assert client.get(f"/api/jobs/{jid}/events").json()["events"][0]["msg"] == "external running"
    r = client.post(f"/api/jobs/{jid}/cancel")
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_cancellable"
    assert client.post(f"/api/jobs/{jid}/kill", json={"force_now": True}).status_code == 409
    client.portal.call(ctx.jobs.detach_external, jid, "succeeded")
    assert client.get(f"/api/jobs/{jid}").json()["status"] == "succeeded"
    assert client.get(f"/api/jobs/{jid}/tracker").json()["cursor"] == 0
    r = client.post("/api/jobs", json={"kind": "run.external", "params": {}})
    assert r.status_code in (404, 422)                                         # never created by clients


def test_retention_deletes_old_finished_jobs(client, ctx, wait_job):

    jid = submit(client, "test.sleep", seconds=0.1)["id"]
    wait_job(client, jid)
    ctx.db.execute("UPDATE jobs SET finished_utc = '2020-01-01T00:00:00Z' WHERE id = ?", (jid,))
    assert client.portal.call(ctx.jobs.apply_retention) == 0                     # default: keep
    client.put("/api/settings", json={"keep_job_logs_days": 7})
    assert client.portal.call(ctx.jobs.apply_retention) == 1
    assert client.get(f"/api/jobs/{jid}").status_code == 404
    assert not ctx.workspace.job_dir(jid).exists()


def test_output_written_events(client, ctx, tmp_path):
    """An artifact event of a job attached to a run is announced on the global stream as ``output.written``."""
    ctx.jobs.tailers["j_fake"] = types_ns(run_id="r_out", kind="test.lock", path=None)
    try:
        client.portal.call(ctx.jobs._on_event, "j_fake", 10,
                           {"type": "artifact", "path": "predictions.parquet", "role": "predictions", "bytes": 5,
                            "stage": "S2_S3", "span_path": ["run:x", "stage:S2_S3"]})
        client.portal.call(ctx.jobs._on_event, "j_fake", 20,             # a nested study child: not announced
                           {"type": "artifact", "path": "child.json", "role": "x", "bytes": 1,
                            "span_path": ["task:placebo_kind[shift]", "run:child"]})
    finally:
        ctx.jobs.tailers.pop("j_fake", None)
    recs = [r for r in ctx.hub.ring if r["type"] == "output.written"]
    assert len(recs) == 1
    rec = recs[0]
    assert rec["run_id"] == "r_out" and rec["relpath"] == "predictions.parquet"
    assert rec["bytes"] == 5 and rec["stage"] == "S2_S3" and "output_id" in rec


def test_job_context_resolves_dirs_and_run_config(tmp_path):
    """JobContext: dirs from job.json's context, then the read-only DB; run_config() in SPEC §4.3 order."""
    import yaml

    from sparc.studio.db import Database
    from sparc.studio.jobs.kinds import JobContext
    from sparc.studio.workspace import Workspace

    ws = Workspace(tmp_path / "ws").ensure()
    db = Database(ws.db_path)
    db.migrate()
    run_dir = tmp_path / "runs" / "r1"
    (run_dir / "studio").mkdir(parents=True)
    db.insert("runs", {"id": "r1", "run_dir": str(run_dir), "studio_dir": str(run_dir / "studio"),
                       "origin": "studio", "status": "complete"})
    db.insert("projects", {"id": "p_1", "slug": "demo", "name": "Demo", "dir": str(tmp_path / "proj"),
                           "config_path": str(tmp_path / "proj" / "config.yml")})
    job_dir = ws.job_dir("j_ctx")
    job_dir.mkdir()
    (job_dir / "job.json").write_text(json.dumps({"id": "j_ctx", "kind": "test.sleep", "params": {},
                                                  "run_id": "r1", "project_id": "p_1", "threads": 2}))
    ctx = JobContext(job_dir)
    assert ctx.run_dir == run_dir and ctx.studio_dir == run_dir / "studio"            # from the DB
    assert ctx.project_dir == tmp_path / "proj" and ctx.threads == 2
    assert ctx.workspace.root == ws.root and ctx.cache_dir == ws.cache_dir
    db.close()
    raw = {"data": {"path": "x.csv", "target": "T"}, "predictors": ["Pct_Canopy"]}
    # 3. an explicit config path
    cfg_path = tmp_path / "explicit.yml"
    cfg_path.write_text(yaml.safe_dump(raw))
    ctx.params["config_path"] = str(cfg_path)
    assert ctx.run_config().base_dir == tmp_path
    # 2. manifest.config + provenance.config_dir
    (run_dir / "manifest.json").write_text(json.dumps({"config": raw, "provenance": {"config_dir": "/m/dir"}}))
    assert ctx.run_config().base_dir == Path("/m/dir")
    # 1. the launch snapshot wins
    (run_dir / "studio" / "launch.json").write_text(json.dumps({"config_raw": raw, "config_dir": "/launch/dir",
                                                                "args": {"fast": True}}))
    assert ctx.run_config().base_dir == Path("/launch/dir") and ctx.launch["args"] == {"fast": True}
    ctx.emit_result({"a": 1})
    assert ctx.result == {"a": 1}


# ---------------------------------------------------------------------------
# status transitions are atomic and ordered
# ---------------------------------------------------------------------------

def test_cancel_wins_over_a_stale_start(client, ctx):
    """The scheduler read a queued job, then awaited; a cancel landed meanwhile: the start must not undo it."""
    from sparc.studio.jobs import kinds

    client.post("/api/queue/pause")
    try:
        jid = submit(client, "test.sleep", seconds=5)["id"]
        stale = ctx.jobs.get_row(jid)
        assert stale["status"] == "queued"
        assert client.post(f"/api/jobs/{jid}/cancel").json()["status"] == "cancelled"
        started = client.portal.call(ctx.jobs._start, stale, kinds.get_kind("test.sleep"), 1)
        assert started is False
        row = ctx.jobs.get_row(jid)
        assert row["status"] == "cancelled" and row["pid"] is None and jid not in ctx.jobs.tailers
        statuses = [ev["status"] for ev in events_of(ctx, jid) if ev["type"] == "job.status"]
        assert statuses == ["queued", "cancelled"]
        # … and a stale block cannot resurrect it either
        client.portal.call(ctx.jobs._block, stale, "waiting for something")
        assert ctx.jobs.get_row(jid)["status"] == "cancelled"
    finally:
        client.post("/api/queue/resume")


def test_status_line_is_on_disk_before_the_row_changes(client, ctx, wait_job, monkeypatch):
    """Whoever sees a new status in SQLite (e.g. a job stream deciding to send ``end``) finds its line."""
    seen = []
    orig = ctx.db.aupdate

    async def aupdate(table, key, values):
        if table == "jobs" and "status" in values:
            lines = [ev for ev in events_of(ctx, key["id"]) if ev["type"] == "job.status"]
            seen.append((values["status"], lines[-1]["status"] if lines else None))
        return await orig(table, key, values)

    monkeypatch.setattr(ctx.db, "aupdate", aupdate)
    jid = submit(client, "test.sleep", seconds=0.2)["id"]
    assert wait_job(client, jid)["status"] == "succeeded"
    assert [s for s, _ in seen] == ["starting", "running", "succeeded"]
    assert all(row == line for row, line in seen), seen


def test_default_settings_fit_the_thread_budget(monkeypatch):
    from sparc.studio import settings as settings_mod

    for cpu, engine in ((1, 1), (2, 2), (8, 2)):
        monkeypatch.setattr(settings_mod, "machine", lambda cpu=cpu: {"cpu_count": cpu, "mem_total_gb": 4.0})
        d = settings_mod.default_settings()
        s = settings_mod.Settings(**d)                        # the defaults satisfy their own budget rule
        assert s.thread_budget == cpu and s.threads_heavy == max(1, cpu - 1) and s.engine_threads == engine


def test_odd_stored_errors_and_actions_never_break_the_job_list(client, ctx, wait_job):
    from sparc.studio.jobs.manager import _valid_actions

    jid = submit(client, "test.sleep", seconds=0.1)["id"]
    wait_job(client, jid)
    for stored, expected in (('"plain text"', {"type": "Error", "message": "plain text"}),
                             ('{"message": "no type"}', {"type": "Error", "message": "no type"}),
                             ('{}', None)):
        ctx.db.execute("UPDATE jobs SET error_json = ? WHERE id = ?", (stored, jid))
        err = client.get(f"/api/jobs/{jid}").json()["error"]
        assert (err if err is None else {k: err[k] for k in ("type", "message")}) == expected
        assert client.get("/api/jobs").status_code == 200
    kept = _valid_actions([{"kind": "evict_engine", "label": "Evict"}, {"kind": "open", "label": "Show", "path": "/x"},
                           "not an action"], "test.sleep")
    assert kept == [{"kind": "open", "label": "Show", "path": "/x"}]


def test_tracker_reports_the_log_cap(client, wait_job, monkeypatch):
    from sparc.studio.routes import jobs as jobs_routes

    jid = submit(client, "test.sleep", seconds=0.1)["id"]
    wait_job(client, jid)
    assert client.get(f"/api/jobs/{jid}/tracker").json()["log_capped"] is False
    monkeypatch.setattr(jobs_routes, "LOG_CAP_BYTES", 100)              # stands in for 200 MB
    assert client.get(f"/api/jobs/{jid}/tracker").json()["log_capped"] is True
