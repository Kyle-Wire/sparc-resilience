"""Replay runner (api.md §14): copies outputs at their artifact events, rewrites ids and paths, cancels at an
event boundary, resumes from the last saved checkpoint; and the executor picks it for run.core."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from sparc.studio.jobs.executors import spawn_command
from tests.studio.conftest import read_jsonl

ROOT = Path(__file__).resolve().parents[3]
PLACEHOLDER = "/tmp/sparc-fixture"


def _ev(seq, type_, ts, path=("run:demo",), span="9:1", **fields):
    return {"v": 1, "type": type_, "seq": seq, "ts": ts, "t_rel": ts - 1000.0, "pid": 999, "job": "j_fixture",
            "lvl": "info", "span": span, "parent": None, "path": list(path), "ctx": {}, **fields}


def make_fixture(d: Path) -> Path:
    """A three-stage recorded run: S0 → S2_S3 (checkpoint S3) → S4 (checkpoint S3+S4)."""
    d.mkdir(parents=True)
    for name, text in {"qa.json": "{}", "predictions.parquet": "pq", "response_curves.json": "{}",
                       "manifest.json": json.dumps({"metrics": {"stacker_r2": 0.7, "stacker_rmse": 0.9}})}.items():
        (d / name).write_text(text)
    (d / "run_state.json").write_text(json.dumps({"schema": 1, "status": "succeeded", "host": "vm", "stage": "finish",
                                                   "events_path": f"{PLACEHOLDER}/events.jsonl", "meta": {"n_points": 9},
                                                   "done": ["S3", "S4"], "fingerprint": "f" * 16}))
    (d / "checkpoint.json").write_text(json.dumps({"schema": 1, "fingerprint": "f" * 16, "done": ["S3", "S4"],
                                                   "bytes": 100, "sections": {"core": "x"}}))
    nodes = [{"id": "S0", "label": "S0", "state": "will_run", "units": {"s0_load": 1}},
             {"id": "S2_S3", "label": "S2", "state": "will_run", "units": {"base_fit:ols": 1, "checkpoint_save": 1}},
             {"id": "S4", "label": "S4", "state": "will_run", "units": {"engine_pass": 1}}]
    s0, s2, s4 = ("run:demo", "stage:S0"), ("run:demo", "stage:S2_S3"), ("run:demo", "stage:S4")
    evs = [
        _ev(1, "run.start", 1000.0, name="demo", stages=["S0", "S2_S3", "S4"], fast=True, resume=False,
            run_meta={}),
        _ev(2, "run.plan", 1000.1, nodes=nodes, total_units={}, n_points=9),
        _ev(3, "run.dir", 1000.2, run_dir=f"{PLACEHOLDER}/run", fingerprint="f" * 16),
        _ev(4, "stage.start", 1000.3, s0, "9:2", stage="S0", label="S0"),
        {**_ev(5, "artifact", 1000.4, s0, "9:2", role="qa", bytes=2, stage="S0"), "path": "qa.json",
         "span_path": list(s0)},
        _ev(6, "stage.end", 1000.5, s0, "9:2", stage="S0", status="ok", elapsed_s=0.2, summary={}),
        _ev(7, "stage.start", 1001.0, s2, "9:3", stage="S2_S3", label="S2"),
        _ev(8, "task.end", 1001.5, s2 + ("task:base_model[ols]",), "9:4", name="base_model", key="ols",
            unit="base_fit:ols", status="ok", elapsed_s=0.5, metrics={}),
        {**_ev(9, "artifact", 1001.6, s2, "9:3", role="predictions", bytes=2, stage="S2_S3"),
         "path": "predictions.parquet", "span_path": list(s2)},
        _ev(10, "checkpoint", 1001.7, s2, "9:3", action="saved", done=["S3"], bytes=100, elapsed_s=0.1),
        _ev(11, "stage.end", 1001.8, s2, "9:3", stage="S2_S3", status="ok", elapsed_s=0.8, summary={}),
        _ev(12, "stage.start", 1002.0, s4, "9:5", stage="S4", label="S4"),
        _ev(13, "tick", 1002.5, s4, "9:5", k=1, n=1, unit="engine_pass", frac=1.0, label="", pass_s=0.5),
        {**_ev(14, "artifact", 1002.6, s4, "9:5", role="response", bytes=2, stage="S4"),
         "path": "response_curves.json", "span_path": list(s4)},
        _ev(15, "checkpoint", 1002.7, s4, "9:5", action="saved", done=["S3", "S4"], bytes=100, elapsed_s=0.1),
        _ev(16, "stage.end", 1002.8, s4, "9:5", stage="S4", status="ok", elapsed_s=0.8, summary={}),
        {**_ev(17, "artifact", 1002.9, role="manifest", bytes=2, stage="finish"), "path": "manifest.json",
         "span_path": ["run:demo"]},
        _ev(18, "run.end", 1003.0, status="succeeded", elapsed_s=3.0, timings_s={"S0": 0.2}, done=["S3", "S4"],
            error=None),
    ]
    with open(d / "events.jsonl", "w") as f:
        for e in evs:
            f.write(json.dumps(e) + "\n")
    (d / "FIXTURE.json").write_text(json.dumps({"n": 3, "seed": 0, "mode": "fast", "generator": "test"}))
    return d


def make_job(tmp_path: Path, name: str, run_dir: Path, *, resume=False) -> Path:
    job_dir = tmp_path / "ws" / "jobs" / name
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(json.dumps({
        "id": name, "kind": "run.core", "params": {"run_id": "r1", "resume": resume}, "run_id": "r1", "threads": 1,
        "context": {"workspace": str(tmp_path / "ws"), "run_dir": str(run_dir), "studio_dir": str(run_dir / "studio")}}))
    return job_dir


def replay(fixture, job_dir, speed="1000", **kw):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPARC_")}
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run([sys.executable, "-m", "sparc.studio.jobs.replay", str(fixture), str(job_dir),
                           "--speed", speed], env=env, cwd=str(job_dir), capture_output=True, text=True, timeout=120,
                          **kw)


def test_replay_copies_outputs_and_rewrites(tmp_path):
    fixture = make_fixture(tmp_path / "fixture")
    run_dir = tmp_path / "ws" / "projects" / "demo" / "runs" / "r1"
    job_dir = make_job(tmp_path, "j_one", run_dir)
    proc = replay(fixture, job_dir)
    assert proc.returncode == 0, proc.stderr
    for name in ("qa.json", "predictions.parquet", "response_curves.json", "manifest.json"):
        assert (run_dir / name).read_text() == (fixture / name).read_text()
    evs = [ev for _, ev in read_jsonl(job_dir / "events.jsonl")]
    assert all(e["job"] == "j_one" and e["pid"] != 999 for e in evs)
    assert next(e for e in evs if e["type"] == "run.dir")["run_dir"] == str(run_dir)
    assert [e["type"] for e in evs][-2:] == ["run.end", "job.result"]
    state = json.loads((run_dir / "run_state.json").read_text())
    assert state["status"] == "succeeded" and state["job"] == "j_one"
    assert state["events_path"] == str(job_dir / "events.jsonl") and state["pid"] != 999
    assert json.loads((run_dir / "checkpoint.json").read_text())["done"] == ["S3", "S4"]
    result = json.loads((job_dir / "result.json").read_text())
    assert result["status"] == "succeeded" and result["exit_code"] == 0
    assert result["result"]["metrics"] == {"r2": 0.7, "rmse": 0.9, "coverage": None}


def test_replay_cancel_then_resume(tmp_path):
    fixture = make_fixture(tmp_path / "fixture")
    # stretch the recorded timing so the cancel lands after the S2_S3 checkpoint and before S4 ends
    run_dir = tmp_path / "ws" / "runs" / "r1"
    job_dir = make_job(tmp_path, "j_first", run_dir)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPARC_")}
    env["PYTHONPATH"] = str(ROOT)
    p = subprocess.Popen([sys.executable, "-m", "sparc.studio.jobs.replay", str(fixture), str(job_dir), "--speed",
                          "1"], env=env, cwd=str(job_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    import time

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if (run_dir / "checkpoint.json").exists():
            break
        time.sleep(0.02)
    (job_dir / "cancel").touch()
    out, err = p.communicate(timeout=60)
    assert p.returncode == 130, err
    state = json.loads((run_dir / "run_state.json").read_text())
    assert state["status"] == "cancelled" and state["done"] == ["S3"]
    evs = [ev for _, ev in read_jsonl(job_dir / "events.jsonl")]
    assert [e["type"] for e in evs][-2:] == ["cancel.ack", "run.end"] and evs[-1]["status"] == "cancelled"
    assert json.loads((job_dir / "result.json").read_text())["status"] == "cancelled"

    job2 = make_job(tmp_path, "j_resume", run_dir, resume=True)
    proc = replay(fixture, job2)
    assert proc.returncode == 0, proc.stderr
    evs = [ev for _, ev in read_jsonl(job2 / "events.jsonl")]
    plan = next(e for e in evs if e["type"] == "run.plan")
    assert {n["id"]: n["state"] for n in plan["nodes"]} == {"S0": "will_run", "S2_S3": "cached", "S4": "will_run"}
    assert next(e for e in evs if e["type"] == "run.start")["resume"] is True
    skips = [e for e in evs if e["type"] == "stage.skip"]
    assert [(e["stage"], e["reason"]) for e in skips] == [("S2_S3", "checkpoint")]
    assert not any(e["type"] == "stage.start" and e["stage"] == "S2_S3" for e in evs)
    assert any(e["type"] == "checkpoint" and e["action"] == "loaded" and e["done"] == ["S3"] for e in evs)
    assert json.loads((run_dir / "run_state.json").read_text())["status"] == "succeeded"
    from sparc.studio.jobs import tracker

    st = tracker.replay((c, e) for c, e in read_jsonl(job2 / "events.jsonl"))
    assert st["stages"]["S2_S3"]["state"] == "cached" and st["stages"]["S4"]["state"] == "done"
    assert st["progress"] == 1.0


def test_executor_uses_replay_for_run_core(monkeypatch, tmp_path):
    job = {"id": "j_x", "kind": "run.core", "job_dir": str(tmp_path / "j_x")}
    monkeypatch.delenv("SPARC_STUDIO_RUNNER", raising=False)
    assert spawn_command(job)[1:3] == ["-m", "sparc.studio.jobs.worker"]
    monkeypatch.setenv("SPARC_STUDIO_RUNNER", "replay:tests/studio/fixtures/synth_run")
    cmd = spawn_command(job)
    assert cmd[1:3] == ["-m", "sparc.studio.jobs.replay"] and Path(cmd[3]).parts[-3:] == ("studio", "fixtures", "synth_run")
    assert cmd[-2:] == ["--speed", "20"]
    assert spawn_command({**job, "kind": "test.sleep"})[2] == "sparc.studio.jobs.worker"


def test_replay_of_the_synth_run_fixture(tmp_path, synth_run_dir):
    run_dir = tmp_path / "run"
    job_dir = make_job(tmp_path, "j_synth", run_dir)
    proc = replay(synth_run_dir, job_dir, speed="100000")
    assert proc.returncode == 0, proc.stderr
    assert (run_dir / "predictions.parquet").read_bytes() == (synth_run_dir / "predictions.parquet").read_bytes()
    state = json.loads((run_dir / "run_state.json").read_text())
    assert state["status"] == "succeeded" and state["events_path"] == str(job_dir / "events.jsonl")
    from sparc.studio.jobs import tracker

    st = tracker.replay((c, e) for c, e in read_jsonl(job_dir / "events.jsonl"))
    assert st["run_status"] == "succeeded" and st["progress"] == 1.0
    assert st["run"]["run_dir"] == str(run_dir)


def test_cached_stages_follow_core_checkpoint_keys():
    from sparc.studio.jobs.replay import cached_stages

    assert cached_stages(["S3"]) == {"S1", "S2_S3"}
    assert cached_stages(["S3", "baselines", "S4"]) == {"S1", "S2_S3", "baselines", "S4"}
    assert cached_stages([]) == set()
