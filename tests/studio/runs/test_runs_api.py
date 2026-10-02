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

VIEWS = ("overview", "data", "accuracy", "distance", "influence", "response", "scenarios", "climate", "causal",
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
    bad = client.post(f"/api/projects/{demo['id']}/runs", json={"mode": "fast", "then": ["post.emulator"]})
    assert bad.status_code == 422                                # not registered in this build


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


# ---------------------------------------------------------------------------
# outputs, views, docs, files, dictionary
# ---------------------------------------------------------------------------

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
