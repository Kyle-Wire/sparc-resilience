"""The runs registry (SPEC §5.8, §5.11, §5.14): watch roots, the ``run.external`` pseudo-job of a live CLI run
(tailed events, or events synthesised from ``run_state.json``), status rules and lineage (``run_meta``, or the
legacy name patterns flagged *inferred*)."""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import pytest

from sparc.studio.events import encode_line
from tests.studio.conftest import wait_for


def _now(offset_s: float = 0.0) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + offset_s))


def _state(rd: Path, **fields) -> None:
    st = {"schema": 1, "status": "running", "pid": os.getpid(), "job": None, "host": socket.gethostname(),
          "started_utc": _now(-30), "updated_utc": _now(), "stage": "S1", "done": [], "fingerprint": "ab" * 8,
          "events_path": None, "error": None, "meta": {"n_points": 9}}
    st.update(fields)
    (rd / "run_state.json").write_text(json.dumps(st))


class Events:
    """A CLI ``--progress`` events file."""

    def __init__(self, path: Path, name: str):
        self.path, self.name, self.seq = path, name, 0

    def add(self, type_: str, stage: str | None = None, **fields) -> None:
        self.seq += 1
        path = [f"run:{self.name}"] + ([f"stage:{stage}"] if stage else [])
        ev = {"v": 1, "type": type_, "seq": self.seq, "ts": round(time.time(), 3), "t_rel": float(self.seq),
              "pid": os.getpid(), "job": "", "lvl": "info", "span": f"1:{self.seq}", "parent": None, "path": path,
              "ctx": {}, **fields}
        if stage:
            ev["stage"] = stage
        with open(self.path, "ab") as f:
            f.write(encode_line(ev))


@pytest.fixture
def watch_root(client, tmp_path) -> Path:
    root = tmp_path / "cli_runs"
    root.mkdir()
    r = client.put("/api/settings", json={"watch_roots": [str(root)]})
    assert r.status_code == 200, r.text
    return root


def tick(client, ctx) -> None:
    client.portal.call(ctx.services["registry"].watch_tick)


def run_row(ctx, run_dir: Path) -> dict:
    return ctx.db.fetchone("SELECT * FROM runs WHERE run_dir = ?", (str(run_dir.resolve()),))


def external_job(ctx, rid: str) -> dict | None:
    return ctx.db.fetchone("SELECT * FROM jobs WHERE run_id = ? AND kind = 'run.external' ORDER BY created_utc DESC",
                           (rid,))


def test_live_cli_run_gets_a_pseudo_job(client, ctx, watch_root, wait_job):
    rd = watch_root / "city_live"
    rd.mkdir()
    ev = Events(rd / "events.jsonl", "city_live")
    ev.add("run.start", name="city_live", stages=["S0", "S1", "S2"], fast=True, coarse=None, resume=False,
           cv_curve=None, config_sha256="", code_sha256="", run_meta={})
    ev.add("stage.start", "S0", label="Load data", est_s=None)
    _state(rd, stage="S0", events_path=str(rd / "events.jsonl"))
    tick(client, ctx)
    row = run_row(ctx, rd)
    assert row["status"] == "external_live" and row["origin"] == "external_live"
    job = external_job(ctx, row["id"])
    assert job["status"] == "running" and job["lane"] == "none" and job["executor"] == "external"
    jid = job["id"]
    evs = wait_for(lambda: [e for e in client.get(f"/api/jobs/{jid}/events").json()["events"]
                            if e["type"] in ("run.start", "stage.start")] or None, 20, what="tailed events")
    assert [e["type"] for e in evs] == ["run.start", "stage.start"]
    assert client.get(f"/api/runs/{row['id']}").json()["run"]["status"] == "external_live"
    assert client.post(f"/api/jobs/{jid}/cancel").status_code == 409            # read-only
    assert client.delete(f"/api/runs/{row['id']}").status_code == 409           # live
    tick(client, ctx)
    assert ctx.db.fetchval("SELECT COUNT(*) FROM jobs WHERE kind = 'run.external'") == 1   # one pseudo-job

    ev.add("stage.end", "S0", status="ok", elapsed_s=0.1, summary={})
    ev.add("run.end", status="succeeded", elapsed_s=1.0, timings_s={"S0": 0.1}, done=[], error=None)
    (rd / "manifest.json").write_text(json.dumps({"name": "city_live", "created_utc": _now(), "timings_s": {}}))
    _state(rd, status="succeeded", stage="finish", events_path=str(rd / "events.jsonl"))
    tick(client, ctx)
    assert wait_job(client, jid, timeout=20)["status"] == "succeeded"
    row = run_row(ctx, rd)
    assert row["status"] == "complete" and row["origin"] == "imported"
    tr = client.get(f"/api/jobs/{jid}/tracker").json()
    assert tr["stages"]["S0"]["state"] == "done"


def test_cli_run_without_events_file_synthesises_stages(client, ctx, watch_root, wait_job):
    rd = watch_root / "city_quiet"
    rd.mkdir()
    _state(rd, stage="S2_S3", done=[])
    tick(client, ctx)
    row = run_row(ctx, rd)
    job = external_job(ctx, row["id"])
    assert row["status"] == "external_live" and job is not None
    jid = job["id"]
    synth = Path(job["job_dir"]) / "events.jsonl"
    assert synth.is_file() and synth.parent.parent == ctx.workspace.jobs_dir

    def types():
        return [(e["type"], e.get("stage")) for e in client.get(f"/api/jobs/{jid}/events").json()["events"]
                if not e["type"].startswith("job.")]

    wait_for(lambda: ("stage.start", "S2_S3") in types(), 20, what="synthesised stage.start")
    _state(rd, stage="S4", done=["S3"])
    tick(client, ctx)
    wait_for(lambda: ("stage.start", "S4") in types(), 20, what="the next stage")
    assert ("stage.end", "S2_S3") in types()
    _state(rd, status="failed", stage="S4", done=["S3"], error={"type": "ValueError", "message": "boom"})
    tick(client, ctx)
    assert wait_job(client, jid, timeout=20)["status"] == "failed"
    assert ("run.end", None) in types()
    assert run_row(ctx, rd)["status"] == "partial"                    # S3 is in the checkpoint


def test_stale_heartbeat_is_not_live(client, ctx, watch_root):
    rd = watch_root / "city_stale"
    rd.mkdir()
    _state(rd, updated_utc=_now(-600))
    tick(client, ctx)
    row = run_row(ctx, rd)
    assert row["status"] == "interrupted" and external_job(ctx, row["id"]) is None
    _state(rd, updated_utc=_now(-600), done=["S3"])
    ctx.services["registry"].refresh(row["id"])
    assert run_row(ctx, rd)["status"] == "partial"


def test_lineage_explicit_and_inferred(client, ctx, watch_root):
    def run(name: str, **manifest) -> Path:
        rd = watch_root / name
        rd.mkdir()
        (rd / "manifest.json").write_text(json.dumps({"name": name, "created_utc": "2026-10-01T10:00:00+00:00",
                                                      **manifest}))
        return rd

    parent = run("city")
    child = run("city_placebo_grf")
    variant = run("city_mv_no_physics")
    ctx.services["registry"].scan()
    pid = run_row(ctx, parent)["id"]
    explicit = run("other", run_meta={"parent_run_id": pid, "role": "reproduction", "study_id": None})
    unrelated = run("elsewhere_placebo_grf")
    ctx.services["registry"].scan()
    page = {r["id"]: r for r in client.get("/api/runs", params={"limit": 50}).json()["items"]}
    c, v, e, u = (run_row(ctx, d)["id"] for d in (child, variant, explicit, unrelated))
    assert page[c]["parent_run_id"] == pid and page[v]["parent_run_id"] == pid and page[e]["parent_run_id"] == pid
    assert page[u]["parent_run_id"] is None
    info = {rid: json.loads(run_row(ctx, d)["stages_json"]) for rid, d in ((c, child), (v, variant), (e, explicit))}
    assert info[c]["link"] == "inferred" and info[c]["role"] == "placebo:grf"
    assert info[v]["link"] == "inferred" and info[v]["role"] == "variant:no_physics"
    assert info[e]["link"] == "explicit"
    detail = client.get(f"/api/runs/{c}").json()
    assert any(f["code"] == "lineage.inferred" for f in detail["flags"])
    kids = {k["id"] for k in client.get(f"/api/runs/{pid}").json()["children"]}
    assert kids == {c, v, e}
    # stable ids: a reindex of the same folders finds the same ids
    ctx.db.execute("DELETE FROM runs")
    ctx.services["registry"].scan()
    assert run_row(ctx, child)["id"] == c and run_row(ctx, parent)["id"] == pid


def test_imported_run_ids_follow_the_documented_shape(ctx, watch_root):
    rd = watch_root / "city"
    rd.mkdir()
    (rd / "manifest.json").write_text(json.dumps({"name": "city", "created_utc": "2026-09-30T17:34:37+00:00",
                                                  "fast_mode": True}))
    row = ctx.services["registry"].index_run_dir(rd, origin="imported")
    head, mode, tail = row["id"].rsplit("-", 2)
    assert head == "20260930-173437" and mode == "fast" and len(tail) == 4
    assert row["status"] == "complete" and row["mode"] == "fast"
    assert (ctx.workspace.imports_dir / row["id"] / "studio" / "import.json").is_file()
    assert not (rd / "studio").exists()


def test_run_core_kind_hooks(ctx, demo, place_run):
    from sparc.studio.jobs.kinds import KINDS
    from sparc.studio.runs import kinds as K

    spec = KINDS["run.core"]
    assert spec.lane == "heavy" and spec.locks_run and spec.needs_run
    assert KINDS["run.external"].executor == "external" and KINDS["run.external"].lane == "none"
    rd = place_run(demo)
    rid = rd.name
    plan = [{"id": "S0", "state": "will_run", "units": {"s0_load": 1}},
            {"id": "S2_S3", "state": "will_run", "checkpoint_key": "S3", "units": {"base_fit:ols": 3}},
            {"id": "S4", "state": "will_run", "checkpoint_key": "S4", "units": {"engine_pass": 2}},
            {"id": "S6", "state": "disabled", "units": {"causal": 1}}]
    info = json.loads(ctx.db.fetchone("SELECT stages_json FROM runs WHERE id = ?", (rid,))["stages_json"])
    info.update(plan=plan, n_folds=3)
    ctx.db.update("runs", {"id": rid}, {"stages_json": json.dumps(info)})
    job = {"run_id": rid, "params": {"run_id": rid, "resume": False, "use_current_config": True}}
    est = K.run_core_estimate(ctx, job, K.RunCoreParams(run_id=rid))
    assert est["units"] == {"s0_load": 1, "base_fit:ols": 3, "engine_pass": 2} and est["n_cells"] == 1120
    assert est["peak_ram_gb"] > 0 and est["disk_bytes"] > 0
    est = K.run_core_estimate(ctx, job, K.RunCoreParams(run_id=rid, resume=True))
    assert est["units"] == {"s0_load": 1}                         # S3 and S4 come from the checkpoint
    assert K.run_core_retry_params(ctx, job) == {"run_id": rid, "resume": True}
    with pytest.raises(Exception):
        K.RunCoreParams(run_id=rid, bogus=1)
