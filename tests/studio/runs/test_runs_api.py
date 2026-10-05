"""Runs API (api.md §6, §6.1): list, detail, plan, launch and resume through the replay runner, import in place,
status board, timeline, metadata endpoints, outputs, views, docs, files and the dictionary."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.studio.conftest import wait_for
from tests.studio.runs.conftest import RUN_ID, create_demo

VIEWS = ("overview", "data", "accuracy", "distance", "influence", "response", "scenarios", "climate", "heat", "causal",
         "budget", "planner", "uncertainty", "provenance")


def run_status(client, rid: str) -> str:
    return client.get(f"/api/runs/{rid}").json()["run"]["status"]


# ---------------------------------------------------------------------------
# list and detail
# ---------------------------------------------------------------------------

def test_list_and_detail(client, demo, fixture_run, synth):
    rid, rd = fixture_run
    manifest = json.loads((synth / "manifest.json").read_text())
    page = client.get("/api/runs").json()
    item = next(r for r in page["items"] if r["id"] == rid)
    assert item["status"] == "complete" and item["origin"] == "studio" and item["mode"] == "fast"
    assert item["project_id"] == demo["id"] and item["n_points"] == 1120
    assert item["r2"] == pytest.approx(manifest["metrics"]["stacker"]["r2"])
    assert item["rmse"] == pytest.approx(manifest["metrics"]["stacker"]["rmse"])
    assert item["label"] == "synthetic_demo"
    proj = client.get(f"/api/projects/{demo['id']}/runs").json()
    assert [r["id"] for r in proj["items"]] == [rid]
    assert client.get("/api/runs", params={"status": "running"}).json()["items"] == []

    d = client.get(f"/api/runs/{rid}").json()
    assert d["run"]["id"] == rid and d["header"]["run_dir"] == str(rd)
    assert d["header"]["n_points"] == 1120 and d["header"]["fast"] is True and d["header"]["cell_m"] == 30.0
    assert d["launch"]["args"]["fast"] is True
    states = {s["id"]: s["state"] for s in d["stages"]}
    assert states["S0"] == "done" and states["S7"] == "done"
    ck = d["checkpoint"]                                   # the fixture keeps checkpoint.json, not the pickle
    assert ck["present"] is False and set(ck["done"]) >= {"S3", "S4"} and ck["fingerprint"] == "68c17c7b36fce387"
    assert ck["resumable"] is False and ck["reason"] == "the run is complete"
    assert d["outputs_summary"]["present"] >= 20
    assert client.get("/api/runs/nope").status_code == 404


def test_stage_timings_recorded_from_manifest(ctx, fixture_run, synth):
    rid, _ = fixture_run
    manifest = json.loads((synth / "manifest.json").read_text())
    rows = ctx.db.fetchall("SELECT stage, seconds, source FROM stage_timings WHERE run_id = ?", (rid,))
    assert rows and {r["source"] for r in rows} == {"manifest"}
    stages = (manifest.get("timings_detail") or {}).get("stages") or manifest["timings_s"]
    got = {r["stage"]: r["seconds"] for r in rows}
    for k, v in stages.items():
        assert got[k] == pytest.approx(float(v))


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def test_plan_matches_core_plan_stages(client, demo):
    from sparc.core.config import load_core_config
    from sparc.core.pipeline import ALL_STAGES, plan_stages

    r = client.post(f"/api/projects/{demo['id']}/runs/plan", json={"mode": "fast"})
    assert r.status_code == 200, r.text
    plan = r.json()
    core = plan_stages(load_core_config(demo["config_path"]), stages=ALL_STAGES, fast=True)
    assert [(n["id"], n["state"]) for n in plan["nodes"]] == [(n["id"], n["state"]) for n in core]
    for n in plan["nodes"]:
        if n["state"] == "will_run":
            assert n["est_s"] is not None and n["est_lo"] <= n["est_s"] <= n["est_hi"]
    assert plan["est_lo"] <= plan["total_est_s"] <= plan["est_hi"]
    checks = {p["check"]: p for p in plan["preflight"]}
    assert {"memory", "disk", "files", "threads"} <= set(checks)
    assert all(p["ok"] for p in checks.values())
    assert plan["est_peak_rss_gb"] > 0 and plan["network_hosts"] == []

    coarse = client.post(f"/api/projects/{demo['id']}/runs/plan",
                         json={"mode": "coarse", "coarse_m": 60, "stages": ["S0", "S1"]}).json()
    core = plan_stages(load_core_config(demo["config_path"]), stages=["S0", "S1"], coarse=60.0)
    assert [(n["id"], n["state"]) for n in coarse["nodes"]] == [(n["id"], n["state"]) for n in core]
    assert client.post(f"/api/projects/{demo['id']}/runs/plan", json={"mode": "warp"}).status_code == 422
    assert client.post("/api/projects/p_nope/runs/plan", json={"mode": "fast"}).status_code == 404


# ---------------------------------------------------------------------------
# launch, cancel, resume (replay runner)
# ---------------------------------------------------------------------------

def test_launch_replay_completes(client, ctx, demo, replay_runner, wait_job):
    r = client.post(f"/api/projects/{demo['id']}/runs", json={"mode": "fast", "label": "first"})
    assert r.status_code == 202, r.text
    body = r.json()
    rid, jid = body["run"]["id"], body["job"]["id"]
    assert body["job"]["kind"] == "run.core" and body["chain"] == []
    assert body["run"]["label"] == "first" and rid.endswith(rid.rsplit("-", 1)[-1]) and "-fast-" in rid
    job = wait_job(client, jid, timeout=120)
    assert job["status"] == "succeeded", job
    wait_for(lambda: run_status(client, rid) == "complete", 30, what="run complete")

    rd = Path(demo["dir"]) / "runs" / rid
    launch = json.loads((rd / "studio" / "launch.json").read_text())
    assert launch["run_id"] == rid and launch["job_id"] == jid and launch["args"]["fast"] is True
    assert Path(launch["config_raw"]["data"]["path"]).is_absolute()
    assert (rd / "manifest.json").is_file() and (rd / "predictions.parquet").is_file()

    evs = client.get(f"/api/jobs/{jid}/events").json()["events"]
    types = [e["type"] for e in evs if not e["type"].startswith("job.")]
    assert types[0] == "run.start" and types[-1] == "run.end" and "stage.end" in types
    tracker = client.get(f"/api/jobs/{jid}/tracker").json()
    assert tracker["stages"]["S2_S3"]["state"] == "done" and tracker["stages"]["finish"]["state"] == "done"
    row = ctx.db.fetchone("SELECT last_job_id, status FROM runs WHERE id = ?", (rid,))
    assert row["last_job_id"] == jid and row["status"] == "complete"
    n_events = ctx.db.fetchval("SELECT COUNT(*) FROM stage_timings WHERE run_id = ? AND source = 'events'", (rid,))
    assert n_events > 0


def test_launch_chains_then_jobs(client, ctx, demo, replay_runner, wait_job, monkeypatch):
    """``then`` queues the post-run jobs one after another (``after_job_id``); a cancelled run cancels them."""
    from pydantic import BaseModel, ConfigDict

    from sparc.studio.jobs.kinds import KINDS, JobKind

    class NoParams(BaseModel):
        model_config = ConfigDict(extra="forbid")

    for kind in ("post.planner", "post.writeup"):               # the studies item registers the real ones
        monkeypatch.setitem(KINDS, kind, JobKind(kind=kind, fn=lambda ctx, p: {}, lane="medium", executor="process",
                                                 label=kind, params=NoParams, needs_run=True, locks_run=True))
    monkeypatch.setenv("SPARC_STUDIO_REPLAY_SPEED", "4")
    r = client.post(f"/api/projects/{demo['id']}/runs", json={"mode": "fast", "then": ["post.planner", "post.writeup"]})
    assert r.status_code == 202, r.text
    body = r.json()
    jid, chain = body["job"]["id"], body["chain"]
    assert [j["kind"] for j in chain] == ["post.planner", "post.writeup"]
    assert chain[0]["after_job_id"] == jid and chain[1]["after_job_id"] == chain[0]["id"]
    assert all(j["run_id"] == body["run"]["id"] and j["status"] in ("queued", "blocked") for j in chain)
    wait_for(lambda: any(e["type"] == "stage.start" for e in client.get(f"/api/jobs/{jid}/events").json()["events"]),
             60, what="the run to start")
    assert client.post(f"/api/jobs/{jid}/cancel").status_code in (200, 202)
    assert wait_job(client, jid, timeout=60)["status"] == "cancelled"
    for j in chain:
        assert wait_job(client, j["id"], timeout=30)["status"] == "cancelled"
    bad = client.post(f"/api/projects/{demo['id']}/runs", json={"mode": "fast", "then": ["post.nope"]})
    assert bad.status_code == 422                                # not a registered kind


def test_cancel_then_resume_reports_cached_stages(client, ctx, demo, replay_runner, wait_job, monkeypatch):
    monkeypatch.setenv("SPARC_STUDIO_REPLAY_SPEED", "6")
    body = client.post(f"/api/projects/{demo['id']}/runs", json={"mode": "fast"}).json()
    rid, jid = body["run"]["id"], body["job"]["id"]
    rd = Path(demo["dir"]) / "runs" / rid
    wait_for(lambda: (rd / "checkpoint.json").is_file(), 60, what="the first checkpoint")
    assert client.post(f"/api/jobs/{jid}/cancel").status_code in (200, 202)
    job = wait_job(client, jid, timeout=60)
    assert job["status"] == "cancelled", job
    wait_for(lambda: run_status(client, rid) == "partial", 30, what="run partial")
    detail = client.get(f"/api/runs/{rid}").json()
    assert detail["checkpoint"]["resumable"] is True and "S3" in detail["checkpoint"]["done"]

    monkeypatch.setenv("SPARC_STUDIO_REPLAY_SPEED", "400")
    r = client.post(f"/api/runs/{rid}/resume", json={})
    assert r.status_code == 202, r.text
    jid2 = r.json()["id"]
    assert r.json()["params"]["resume"] is True
    assert client.post(f"/api/runs/{rid}/resume", json={}).status_code == 409   # one run job at a time
    job2 = wait_job(client, jid2, timeout=120)
    assert job2["status"] == "succeeded", job2
    tracker = client.get(f"/api/jobs/{jid2}/tracker").json()
    cached = {k for k, v in tracker["stages"].items() if v["state"] == "cached"}
    assert {"S1", "S2_S3"} <= cached
    wait_for(lambda: run_status(client, rid) == "complete", 30, what="run complete after resume")
    evs = client.get(f"/api/jobs/{jid2}/events").json()["events"]
    assert any(e["type"] == "stage.skip" and e["reason"] == "checkpoint" for e in evs)
    assert client.post(f"/api/runs/{rid}/resume", json={}).status_code == 409    # complete: nothing to resume


def test_resume_with_current_config_keeps_history(client, demo, place_run, monkeypatch, replay_runner, wait_job):
    rd = place_run(demo, edit=lambda d: _mark_cancelled(d))
    rid = RUN_ID
    assert run_status(client, rid) == "partial"
    r = client.post(f"/api/runs/{rid}/resume", json={"use_current_config": True})
    assert r.status_code == 202, r.text
    wait_job(client, r.json()["id"], timeout=120)
    hist = sorted(p.name for p in (rd / "studio").glob("launch.*.json"))
    assert hist == ["launch.1.json"]
    launch = json.loads((rd / "studio" / "launch.json").read_text())
    assert launch["previous"] == "launch.1.json" and launch["run_id"] == rid
    assert {"changed_sections", "refit_from", "phrase"} <= set(launch["impact"])


def _mark_cancelled(rd: Path) -> None:
    st = json.loads((rd / "run_state.json").read_text())
    st.update(status="cancelled", done=["S3", "S4"])
    (rd / "run_state.json").write_text(json.dumps(st))
    (rd / "manifest.json").unlink()


def _run_core_jobs(ctx, rid: str) -> list[dict]:
    return ctx.db.fetchall("SELECT id, status, params_json FROM jobs WHERE run_id = ? AND kind = 'run.core' "
                           "ORDER BY created_utc", (rid,))


@pytest.mark.parametrize("current", [False, True])
def test_concurrent_resumes_queue_one_job(client, ctx, demo, place_run, monkeypatch, current):
    """Two resume requests at once (a double click, two tabs) used to both pass the "active job" check before
    either queued its job, so a second ``run.core`` re-ran the run once the first had completed it (and, with
    ``use_current_config``, renamed ``launch.json`` twice).  One is queued; the other is ``409 active``."""
    import threading

    from sparc.studio.runs import launch as launchmod

    rd = place_run(demo, edit=_mark_cancelled)
    rid = RUN_ID
    assert client.post("/api/queue/pause").status_code == 200       # nothing starts: only the queue is checked
    slow = launchmod.checkpoint_info

    def slow_info(*a, **kw):
        time.sleep(0.3)                                               # both requests are past the check by now
        return slow(*a, **kw)

    monkeypatch.setattr(launchmod, "checkpoint_info", slow_info)
    gate = threading.Barrier(2)
    out: list = []

    def resume():
        gate.wait()
        out.append(client.post(f"/api/runs/{rid}/resume", json={"use_current_config": current}))

    threads = [threading.Thread(target=resume) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    codes = sorted(r.status_code for r in out)
    assert codes == [202, 409], [r.text for r in out]
    refused = next(r for r in out if r.status_code == 409).json()["error"]
    assert refused["code"] == "active"
    jobs = _run_core_jobs(ctx, rid)
    assert len(jobs) == 1 and jobs[0]["id"] == next(r for r in out if r.status_code == 202).json()["id"]
    hist = sorted(p.name for p in (rd / "studio").glob("launch.*.json"))
    assert hist == (["launch.1.json"] if current else [])


def test_retry_of_a_run_core_job_keeps_the_resume_guards(client, ctx, demo, place_run):
    """``POST /api/jobs/{jid}/retry`` of an old ``run.core`` job is a resume: it is refused like one (``409
    active`` while a job runs on the run, ``409 not_resumable`` once the run is complete), never a second
    pipeline pass over the run."""
    rd = place_run(demo, edit=_mark_cancelled)
    rid = RUN_ID
    assert client.post("/api/queue/pause").status_code == 200
    first = client.post(f"/api/runs/{rid}/resume", json={})
    assert first.status_code == 202, first.text
    old = first.json()["id"]
    assert client.post(f"/api/jobs/{old}/cancel").json()["status"] == "cancelled"
    again = client.post(f"/api/runs/{rid}/resume", json={})
    assert again.status_code == 202, again.text

    r = client.post(f"/api/jobs/{old}/retry")                         # another job is queued on the run
    assert r.status_code == 409 and r.json()["error"]["code"] == "active", r.text
    assert r.json()["error"]["detail"]["job_id"] == again.json()["id"]
    assert len(_run_core_jobs(ctx, rid)) == 2

    assert client.post(f"/api/jobs/{again.json()['id']}/cancel").json()["status"] == "cancelled"
    r = client.post(f"/api/jobs/{old}/retry")                         # a partial run: the retry resumes it
    assert r.status_code == 202, r.text
    assert r.json()["params"]["resume"] is True and r.json()["parent_job_id"] == old
    assert client.get(f"/api/runs/{rid}").json()["run"]["last_job_id"] == r.json()["id"]
    assert client.post(f"/api/jobs/{r.json()['id']}/cancel").json()["status"] == "cancelled"

    st = json.loads((rd / "run_state.json").read_text())             # the run completes (another way)
    st.update(status="succeeded")
    (rd / "run_state.json").write_text(json.dumps(st))
    ctx.services["registry"].refresh(rid)
    assert run_status(client, rid) == "complete"
    r = client.post(f"/api/jobs/{old}/retry")
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_resumable", r.text
    assert len(_run_core_jobs(ctx, rid)) == 3


def test_a_queued_resume_of_a_run_completed_meanwhile_does_not_start(client, ctx, demo, place_run, replay_runner,
                                                                    wait_job):
    """Start-time backstop: a ``run.core`` resume that reaches the scheduler once its run is complete (queued
    through ``POST /api/jobs``, which has no run-level checks) fails its preflight instead of re-running the
    finished run's S0, S7 and finish."""
    rd = place_run(demo, edit=_mark_cancelled)
    rid = RUN_ID
    assert client.post("/api/queue/pause").status_code == 200
    r = client.post("/api/jobs", json={"kind": "run.core", "run_id": rid, "params": {"run_id": rid, "resume": True}})
    assert r.status_code == 202, r.text
    st = json.loads((rd / "run_state.json").read_text())
    st.update(status="succeeded")
    (rd / "run_state.json").write_text(json.dumps(st))
    before = (rd / "run_state.json").stat().st_mtime_ns
    assert client.post("/api/queue/resume").status_code == 200
    job = wait_job(client, r.json()["id"], timeout=30)
    assert job["status"] == "failed" and job["error"]["detail"]["code"] == "not_resumable", job
    assert (rd / "run_state.json").stat().st_mtime_ns == before
    assert run_status(client, rid) == "complete"


def test_rerun_creates_a_new_run(client, demo, fixture_run, replay_runner, wait_job):
    rid, _ = fixture_run
    r = client.post(f"/api/runs/{rid}/rerun", json={"label": "again"})
    assert r.status_code == 202, r.text
    new = r.json()["run"]["id"]
    assert new != rid and r.json()["run"]["label"] == "again"
    assert wait_job(client, r.json()["job"]["id"], timeout=120)["status"] == "succeeded"
    wait_for(lambda: run_status(client, new) == "complete", 30, what="rerun complete")


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------

def _outside_copy(synth: Path, dest: Path, edit=None) -> Path:
    shutil.copytree(synth, dest, ignore=shutil.ignore_patterns("events.jsonl"))
    m = json.loads((dest / "manifest.json").read_text())
    m["provenance"]["config_dir"] = str(dest.parent / "no-such-config-dir")
    (dest / "manifest.json").write_text(json.dumps(m))
    if edit is not None:
        edit(dest)
    return dest


def test_import_in_place(client, ctx, demo, synth, tmp_path):
    src = _outside_copy(synth, tmp_path / "outside" / "synth_fast")
    before = sorted(p.name for p in src.iterdir())
    r = client.post("/api/runs/import", json={"dir": str(src), "project_id": demo["id"]})
    assert r.status_code == 422 and r.json()["error"]["code"] == "needs_config"
    assert not any(ctx.workspace.imports_dir.glob("*/studio/import.json"))

    r = client.post("/api/runs/import", json={"dir": str(src), "project_id": demo["id"],
                                              "config_path": demo["config_path"], "trust_pickles": True})
    assert r.status_code == 201, r.text
    run = r.json()
    assert run["origin"] == "imported" and run["status"] == "complete" and run["project_id"] == demo["id"]
    assert sorted(p.name for p in src.iterdir()) == before                     # nothing written in place
    rec = json.loads((ctx.workspace.imports_dir / run["id"] / "studio" / "import.json").read_text())
    assert rec["dir"] == str(src.resolve()) and rec["trust_pickles"] is True
    from sparc.studio.runs.registry import pickle_trusted

    assert pickle_trusted(run["id"]) is True
    assert client.get(f"/api/runs/{run['id']}/config").json()["source"] == "import"
    # a restarted server registers the folder (and the import config's folder) as path roots again
    from sparc.studio.security import PathGuard

    ctx.paths = PathGuard(ctx.workspace.root)
    ctx.services["registry"].scan()
    assert {src.resolve(), Path(demo["config_path"]).resolve().parent} <= set(ctx.paths.roots())
    # idempotent: the same folder keeps its id
    again = client.post("/api/runs/import", json={"dir": str(src), "config_path": demo["config_path"],
                                                  "project_id": demo["id"]})
    assert again.status_code == 201 and again.json()["id"] == run["id"]
    # imported in place: deleting outputs needs force_files
    assert client.delete(f"/api/runs/{run['id']}", params={"what": "outputs"}).status_code == 409
    gone = client.delete(f"/api/runs/{run['id']}").json()
    assert src.is_dir() and gone["freed_bytes"] > 0                              # the folder stays
    assert client.get(f"/api/runs/{run['id']}").status_code == 404


@pytest.mark.parametrize("how", ["absolute", "relative"])
def test_import_never_uses_an_unsafe_launch_run_id(client, ctx, demo, synth, tmp_path, how):
    """An imported folder's ``studio/launch.json`` is untrusted: a ``run_id`` that is not one plain path segment
    must neither name the side folder (``<ws>/imports/<run_id>``) nor be removed when the import is refused."""
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "thesis.txt").write_text("keep me")
    rid = str(victim) if how == "absolute" else os.path.relpath(victim, ctx.workspace.imports_dir)

    def plant(d: Path) -> None:
        (d / "studio").mkdir(exist_ok=True)
        (d / "studio" / "launch.json").write_text(json.dumps({"run_id": rid}))

    bare = tmp_path / "downloaded"
    bare.mkdir()
    (bare / "manifest.json").write_text("{}")
    plant(bare)
    r = client.post("/api/runs/import", json={"dir": str(bare)})
    assert r.status_code == 422 and r.json()["error"]["code"] == "needs_config", r.text
    assert sorted(p.name for p in victim.iterdir()) == ["thesis.txt"]      # neither removed nor written into

    src = _outside_copy(synth, tmp_path / "outside" / "synth_fast", plant)
    r = client.post("/api/runs/import", json={"dir": str(src), "project_id": demo["id"],
                                              "config_path": demo["config_path"]})
    assert r.status_code == 201, r.text
    run = r.json()
    assert "/" not in run["id"] and ".." not in run["id"]
    assert (ctx.workspace.imports_dir / run["id"] / "studio" / "import.json").is_file()
    assert sorted(p.name for p in victim.iterdir()) == ["thesis.txt"]


def test_import_of_a_copied_studio_run_keeps_the_original(client, ctx, demo, fixture_run, tmp_path):
    """A Studio run folder copied elsewhere carries the original's ``launch.json`` ``run_id``: importing the copy
    gets its own id instead of replacing the original's row."""
    rid, rd = fixture_run
    copy = tmp_path / "elsewhere" / rid
    shutil.copytree(rd, copy)
    r = client.post("/api/runs/import", json={"dir": str(copy), "project_id": demo["id"],
                                              "config_path": demo["config_path"]})
    assert r.status_code == 201, r.text
    assert r.json()["id"] != rid
    assert Path(ctx.db.fetchval("SELECT run_dir FROM runs WHERE id = ?", (rid,))).resolve() == rd.resolve()
    assert client.get(f"/api/runs/{rid}").json()["run"]["origin"] == "studio"


def test_import_verifies_ids_and_folds(client, demo, synth, tmp_path):
    def shuffle_ids(d):
        p = pd.read_parquet(d / "predictions.parquet")
        p["id"] = p["id"].to_numpy()[::-1]
        p.to_parquet(d / "predictions.parquet")

    def other_folds(d):
        p = pd.read_parquet(d / "predictions.parquet")
        p["fold"] = (p["fold"].to_numpy() + 1) % 3
        p.to_parquet(d / "predictions.parquet")

    for name, edit in (("ids", shuffle_ids), ("folds", other_folds)):
        src = _outside_copy(synth, tmp_path / name, edit)
        r = client.post("/api/runs/import", json={"dir": str(src), "config_path": demo["config_path"]})
        assert r.status_code == 422, (name, r.text)
        assert r.json()["error"]["code"] == "mismatch"
    assert client.post("/api/runs/import", json={"dir": str(tmp_path / "none")}).status_code == 404


def test_reindex_rebuilds_run_rows(ctx, fixture_run):
    from sparc.studio import db as dbmod

    rid, _ = fixture_run
    assert any(name == "runs" and order == 10 for order, name, _ in dbmod._REINDEX_HOOKS)
    ctx.db.update("runs", {"id": rid}, {"status": "imported", "label": None})
    summary = dbmod.reindex(ctx.db, ctx.workspace)
    assert summary["runs"]["workspace"] == 1
    row = ctx.db.fetchone("SELECT status, origin, label FROM runs WHERE id = ?", (rid,))
    assert row["status"] == "complete" and row["origin"] == "studio" and row["label"] == "synthetic_demo"


# ---------------------------------------------------------------------------
# board, timeline, metadata
# ---------------------------------------------------------------------------

def test_status_board_and_timeline(client, demo, fixture_run):
    rid, _ = fixture_run
    board = client.get(f"/api/projects/{demo['id']}/status-board").json()
    cols = [c["id"] for c in board["columns"]]
    assert cols[:2] == ["S0", "S1"] and "S7" in cols
    row = next(r for r in board["rows"] if r["run"]["id"] == rid)
    assert row["cells"]["S0"]["state"] == "done" and row["cells"]["S7"]["state"] == "done"
    tl = client.get(f"/api/runs/{rid}/timeline").json()
    assert tl["source"] == "manifest"
    secs = {s["id"]: s["seconds"] for s in tl["stages"]}
    assert secs["S2_S3"] > 0 and all(s["state"] in ("done", "cached", "skipped", "failed") for s in tl["stages"])


def test_waiting_post_jobs_are_running_cells_with_a_reason():
    """A post-run job queued behind its run (a launch's "then" chain) shows as a running cell whose reason is
    "queued" (BoardCell has no queued state); a started job has no reason."""
    from sparc.studio.runs.statusboard import _job_cell

    base = {"state": "not_run", "seconds": None, "progress": None, "reason": None, "job_id": None,
            "study_id": None, "action": None}
    for st, reason in (("queued", "queued"), ("blocked", "blocked"), ("running", None), ("cancelling", None)):
        c = _job_cell({"id": "j1", "status": st, "progress": 0.25 if st == "running" else None}, base)
        assert (c["state"], c["reason"], c["job_id"]) == ("running", reason, "j1")
    assert _job_cell({"id": "j2", "status": "succeeded"}, base)["state"] == "done"


def test_manifest_config_provenance_environment(client, demo, fixture_run, synth):
    rid, _ = fixture_run
    m = client.get(f"/api/runs/{rid}/manifest").json()
    assert m["name"] == "synthetic_demo" and m["created_utc"] == "2026-10-01T21:21:49+00:00"
    cfg = client.get(f"/api/runs/{rid}/config").json()
    assert cfg["source"] == "launch" and cfg["config_dir"] == demo["dir"]
    assert cfg["effective"]["cv"]["n_folds"] == 3                       # fast mode applied
    assert "data:" in cfg["yaml"] and cfg["vs_project_diff"] == []      # launch snapshot vs the same config
    import yaml

    cfgp = Path(demo["config_path"])
    raw = yaml.safe_load(cfgp.read_text())
    raw.get("core", raw)["cv"]["seed"] = 7
    cfgp.write_text(yaml.safe_dump(raw))
    diff = client.get(f"/api/runs/{rid}/config").json()["vs_project_diff"]
    assert [d["path"] for d in diff] == ["cv.seed"] and diff[0]["project"] == 7
    prov = client.get(f"/api/runs/{rid}/provenance").json()
    assert prov["launch"]["run_id"] == rid and prov["environment"]
    env = client.get(f"/api/runs/{rid}/environment").json()
    lines = [ln.strip() for ln in (synth / "environment.txt").read_text().splitlines() if "==" in ln]
    assert lines and set(lines) <= set(env["packages"])
    same = client.get(f"/api/runs/{rid}/environment", params={"diff_with": rid}).json()["diff"]
    assert same["added"] == [] and same["removed"] == [] and same["changed"] == []


def test_patch_and_delete(client, ctx, demo, fixture_run):
    rid, rd = fixture_run
    r = client.patch(f"/api/runs/{rid}", json={"label": "keep", "pinned": True, "notes": "n"})
    assert r.status_code == 200 and r.json()["label"] == "keep" and r.json()["pinned"] is True
    freed = client.delete(f"/api/runs/{rid}/checkpoint").json()["freed_bytes"]
    assert freed > 0 and not (rd / "checkpoint.json").exists()
    assert client.get(f"/api/runs/{rid}").json()["checkpoint"]["done"] == []
    assert client.delete(f"/api/runs/{rid}", params={"what": "bogus"}).status_code == 422
    out = client.delete(f"/api/runs/{rid}").json()
    assert out["freed_bytes"] > 500_000 and not rd.exists()
    assert ctx.db.fetchone("SELECT id FROM runs WHERE id = ?", (rid,)) is None


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL and pid checks are POSIX")
def test_temporaries_of_a_killed_worker_are_removed(client, ctx, demo, fixture_run):
    """A worker killed while it saved the checkpoint (Force stop, the OOM killer) left
    ``.checkpoint.pkl.<pid>.<tid>.tmp`` - up to a whole checkpoint - that nothing removed: the end of the
    ``run.core`` job sweeps it, and deleting the checkpoint frees one left by an older server too."""
    from sparc.studio.runs.kinds import run_core_on_finish
    from tests.core.test_runio import killed_writer_tmp

    rid, rd = fixture_run
    stale = killed_writer_tmp(rd / "checkpoint.pkl")
    run_core_on_finish(ctx, {"id": "j_x", "run_id": rid, "project_id": demo["id"]}, None)
    assert not stale.exists() and (rd / "checkpoint.json").exists()

    stale = killed_writer_tmp(rd / "checkpoint.pkl")
    ck = (rd / "checkpoint.json").stat().st_size
    assert client.delete(f"/api/runs/{rid}/checkpoint").json()["freed_bytes"] == ck + 65536
    assert not stale.exists() and not (rd / "checkpoint.json").exists()


def test_delete_waits_for_the_engine_off_the_event_loop(client, ctx, fixture_run):
    """Deleting a checkpoint evicts the run from the engine host, which waits for the host's work lock (up to
    120 s while it serves another run): the wait must not freeze every other request."""
    import asyncio
    import threading

    rid, _ = fixture_run
    entered, release = threading.Event(), threading.Event()
    seen: dict = {}

    class SlowEngine:
        def evict(self, run_id):
            try:
                asyncio.get_running_loop()
                seen["on_loop"] = True
            except RuntimeError:
                seen["on_loop"] = False
            entered.set()
            release.wait(5)
            return True

    old = ctx.services.get("engine")
    ctx.services["engine"] = SlowEngine()
    out: dict = {}
    t = threading.Thread(target=lambda: out.update(r=client.delete(f"/api/runs/{rid}/checkpoint")))
    t.start()
    try:
        assert entered.wait(10)
        t0 = time.monotonic()
        assert client.get("/api/health").status_code == 200
        elapsed = time.monotonic() - t0
    finally:
        release.set()
        t.join(30)
        if old is None:
            ctx.services.pop("engine", None)
        else:
            ctx.services["engine"] = old
    assert out["r"].status_code == 200 and seen["on_loop"] is False
    assert elapsed < 2.0


# ---------------------------------------------------------------------------
# outputs, views, docs, files, dictionary
# ---------------------------------------------------------------------------

def test_a_tab_is_being_computed_only_by_a_job_that_makes_it(client, ctx, fixture_run):
    """While only ``post.emulator`` ran on a finished run, the Planner and Uncertainty tabs read "being
    computed" (any active job counted); now a tab is ``running`` only while an active job makes its outputs."""
    rid, _ = fixture_run

    def fake_job(kind: str) -> str:
        jid = "j_" + kind.replace(".", "_")
        ctx.db.insert("jobs", {"id": jid, "kind": kind, "lane": "heavy", "executor": "process", "params_json": "{}",
                               "status": "running", "job_dir": str(ctx.workspace.job_dir(jid)), "run_id": rid,
                               "created_utc": "2026-10-02T00:00:00Z"})
        return jid

    def tabs() -> dict:
        return {t["id"]: t["availability"] for t in client.get(f"/api/runs/{rid}/outputs").json()["tabs"]}

    em = fake_job("post.emulator")
    t = tabs()
    assert t["planner"] == "missing" and t["uncertainty"] == "missing", t
    assert t["track"] == "running" and t["accuracy"] == "ready"
    assert client.get(f"/api/runs/{rid}/views/planner").json()["availability"] == "missing"
    ctx.db.update("jobs", {"id": em}, {"status": "succeeded"})
    pl = fake_job("post.planner")
    t = tabs()
    assert t["planner"] == "running" and t["uncertainty"] == "missing", t
    ctx.db.update("jobs", {"id": pl}, {"status": "succeeded"})
    assert tabs()["planner"] == "missing" and tabs()["track"] == "ready"


def test_outputs_and_tabs(client, fixture_run):
    rid, _ = fixture_run
    out = client.get(f"/api/runs/{rid}/outputs").json()
    by_id = {o["id"]: o for o in out["outputs"]}
    assert by_id["predictions"]["state"] == "present" and by_id["causal"]["state"] == "present"
    assert by_id["response_maps:canopy"]["state"] == "present"
    tabs = {t["id"]: t["availability"] for t in out["tabs"]}
    for t in ("overview", "data", "accuracy", "influence", "response", "scenarios", "causal", "budget", "map",
              "files", "provenance"):
        assert tabs[t] == "ready", (t, tabs[t])
    assert tabs["planner"] == "missing" and tabs["uncertainty"] == "missing"
    one = client.get(f"/api/runs/{rid}/outputs/influence").json()
    assert one["ranges_m"] and one["_meta"]["stale"] is False
    batch = client.get(f"/api/runs/{rid}/outputs/batch", params={"ids": "physics,planner"}).json()
    assert "physics" in batch["results"] and [m["id"] for m in batch["missing"]] == ["planner"]


def test_stale_outputs(client, demo, place_run):
    def drop_causal(rd):
        m = json.loads((rd / "manifest.json").read_text())
        m.pop("causal")
        m.pop("optimize")
        (rd / "manifest.json").write_text(json.dumps(m))
        old = time.mktime(time.strptime("2026-09-30", "%Y-%m-%d"))
        for f in ("causal.json", "allocation.parquet"):
            os.utime(rd / f, (old, old))

    rid = RUN_ID
    place_run(demo, edit=drop_causal)
    by_id = {o["id"]: o["state"] for o in client.get(f"/api/runs/{rid}/outputs").json()["outputs"]}
    assert by_id["causal"] == "stale" and by_id["allocation"] == "stale" and by_id["predictions"] == "present"
    detail = client.get(f"/api/runs/{rid}").json()
    assert detail["sections"]["causal"]["stale"] is True and detail["sections"]["causal"]["source"] == "file"


@pytest.mark.parametrize("view", VIEWS)
def test_views(client, fixture_run, view):
    rid, _ = fixture_run
    r = client.get(f"/api/runs/{rid}/views/{view}")
    assert r.status_code == 200, r.text
    vm = r.json()
    assert vm["view"] == view and vm["units"]["target"] == "°F" and vm["demo"] is True
    if view in ("planner", "uncertainty"):
        assert vm["availability"] == "missing" and vm["missing"]
    else:
        assert vm["availability"] in ("ready", "partial"), vm["availability"]


def test_view_contents(client, fixture_run, synth):
    rid, _ = fixture_run
    manifest = json.loads((synth / "manifest.json").read_text())
    acc = client.get(f"/api/runs/{rid}/views/accuracy").json()
    assert acc["sections"]["live"] is False
    stack = next(m for m in acc["sections"]["models"] if m["kind"] == "stack")
    assert stack["r2"] == pytest.approx(manifest["metrics"]["stacker"]["r2"])
    assert acc["sections"]["cv_design"]["n_folds"] == 3
    assert sum(acc["sections"]["cv_design"]["test_sizes"]) == 1120
    sc = client.get(f"/api/runs/{rid}/views/scenarios").json()["sections"]
    names = [s["name"] for s in sc["rows"]]
    assert names == [s["name"] for s in manifest["scenarios"]]
    assert "canopy-increase-plus-10" in [s["slug"] for s in sc["rows"]]
    assert client.get(f"/api/runs/{rid}/views/nope").status_code == 404


def test_docs_files_and_dictionary(client, fixture_run, synth):
    rid, rd = fixture_run
    docs = {d["id"]: d for d in client.get(f"/api/runs/{rid}/docs").json()}
    assert docs["report"]["present"] and docs["methods"]["present"] and docs["model_card"]["present"]
    rep = client.get(f"/api/runs/{rid}/docs/report").json()
    assert rep["markdown"] == (synth / "report.md").read_text(encoding="utf-8")

    outside = rd.parent.parent / "outside"
    outside.mkdir()
    (rd / "linked").symlink_to(outside)                       # a link out of the run folder is never listed
    files = {f["relpath"]: f for f in client.get(f"/api/runs/{rid}/files").json()}
    assert "linked" not in files
    assert files["predictions.parquet"]["output_id"] == "predictions" and files["studio"]["dir"] is True
    assert files["predictions.parquet"]["in_manifest"] is False and files["causal.json"]["in_manifest"] is True
    raw = client.get(f"/api/runs/{rid}/files/raw", params={"path": "influence.json"})
    assert raw.status_code == 200 and raw.content == (synth / "influence.json").read_bytes()
    assert client.get(f"/api/runs/{rid}/files/raw", params={"path": "../../../etc/passwd"}).status_code in (403, 404,
                                                                                                         422)
    part = client.get(f"/api/runs/{rid}/files/raw", params={"path": "influence.json"}, headers={"Range": "bytes=0-9"})
    assert part.status_code == 206 and part.content == (synth / "influence.json").read_bytes()[:10]
    assert client.get(f"/api/runs/{rid}/files/raw", params={"path": "report.md", "as": "xlsx"}).status_code == 415

    csv = client.get(f"/api/runs/{rid}/files/raw", params={"path": "predictions.parquet", "as": "csv"})
    assert csv.status_code == 200
    head = csv.text.splitlines()[0].split(",")
    assert head[:3] == ["id", "lon", "lat"] and {"x_m", "y_m", "pred"} <= set(head)
    assert len(csv.text.strip().splitlines()) == 1121
    html = client.get(f"/api/runs/{rid}/files/raw", params={"path": "report.md", "as": "html"})
    assert html.status_code == 200 and "<h1" in html.text and "<script" not in html.text
    gj = client.get(f"/api/runs/{rid}/files/raw", params={"path": "allocation.parquet", "as": "geojson"}).json()
    assert gj["type"] == "FeatureCollection" and len(gj["features"]) == 1120
    lon, lat = gj["features"][0]["geometry"]["coordinates"]
    assert -180 <= lon <= 180 and -90 <= lat <= 90

    tbl = client.get(f"/api/runs/{rid}/files/table", params={"path": "predictions.parquet", "limit": 5,
                                                              "columns": "id,pred"}).json()
    assert [c["name"] for c in tbl["columns"]] == ["id", "pred"] and len(tbl["rows"]) == 5 and tbl["n_rows"] == 1120
    pred = pd.read_parquet(synth / "predictions.parquet")
    assert tbl["rows"][0][1] == pytest.approx(float(pred["pred"].iloc[0]))
    assert np.isfinite(tbl["rows"][4][1])

    rows = client.get(f"/api/runs/{rid}/dictionary").json()
    outputs = {r["output"] for r in rows}
    assert {"predictions.parquet", "response_canopy.parquet", "scenario_deltas.parquet"} <= outputs
    assert "planner/planner_cells.parquet" not in outputs                     # only files the run has
    assert next(r for r in rows if r["output"] == "predictions.parquet" and r["column"] == "pred")["unit"] == "°F"


def test_second_project_runs_are_separate(client, demo, fixture_run):
    other = create_demo(client, "Other city")
    assert client.get(f"/api/projects/{other['id']}/runs").json()["items"] == []
    assert len(client.get(f"/api/projects/{demo['id']}/runs").json()["items"]) == 1
