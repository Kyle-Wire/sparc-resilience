"""Cross-item contracts of the runs item (SPEC §10.2, §4.3): ``import_run`` as backend-projects calls it, the
``RunSummary`` both items build, the launch snapshot's path rule (shared with the config service's impact
preview), the resume impact, threads on resume and the project's active run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from tests.studio.conftest import wait_for
from tests.studio.runs.conftest import RUN_ID


def _outside_copy(synth: Path, dest: Path) -> Path:
    shutil.copytree(synth, dest, ignore=shutil.ignore_patterns("events.jsonl", "FIXTURE.json"))
    m = json.loads((dest / "manifest.json").read_text())
    m["provenance"]["config_dir"] = str(dest.parent / "no-such-config-dir")
    (dest / "manifest.json").write_text(json.dumps(m))
    return dest


def test_project_import_registers_runs_through_the_contract(client, demo, synth, tmp_path):
    """``POST /api/projects/import`` calls ``runs.registry.import_run(dir, project_id, config_path=…,
    trust_pickles=…)`` lazily; the run lands in the new project with the same summary on both endpoints."""
    src = _outside_copy(synth, tmp_path / "cli" / "synth_fast")
    r = client.post("/api/projects/import", json={"name": "Imported city", "config_path": demo["config_path"],
                                                  "run_dirs": [str(src)]})
    assert r.status_code == 201, r.text
    body = r.json()
    pid = body["project"]["id"]
    assert [x["origin"] for x in body["runs"]] == ["imported"]
    assert not any("import" in w and ("unavailable" in w or "not imported" in w) for w in body["warnings"])
    rid = body["runs"][0]["id"]
    listed = next(x for x in client.get("/api/runs").json()["items"] if x["id"] == rid)
    assert listed["project_id"] == pid and listed["status"] == "complete"
    detail_runs = client.get(f"/api/projects/{pid}").json()["runs"]
    assert detail_runs == [listed] == client.get(f"/api/projects/{pid}/runs").json()["items"]

    # a folder whose ids do not match its config aborts the import and removes the new project
    bad = _outside_copy(synth, tmp_path / "cli" / "bad")
    import pandas as pd

    p = pd.read_parquet(bad / "predictions.parquet")
    p["id"] = p["id"].to_numpy()[::-1]
    p.to_parquet(bad / "predictions.parquet")
    r = client.post("/api/projects/import", json={"name": "Broken", "config_path": demo["config_path"],
                                                  "run_dirs": [str(bad)]})
    assert r.status_code == 422 and r.json()["error"]["code"] == "mismatch"
    assert all(x["name"] != "Broken" for x in client.get("/api/projects").json())


def test_import_run_takes_the_server_context(ctx, demo, synth, tmp_path):
    from sparc.studio.runs.registry import import_run, pickle_trusted

    src = _outside_copy(synth, tmp_path / "elsewhere" / "run")
    out = import_run(src, demo["id"], config_path=Path(demo["config_path"]), trust_pickles=False, sctx=ctx)
    assert out["project_id"] == demo["id"] and out["origin"] == "imported"
    assert pickle_trusted(out["id"], sctx=ctx) is False
    again = import_run(str(src), demo["id"], config_path=demo["config_path"], trust_pickles=True, sctx=ctx)
    assert again["id"] == out["id"] and pickle_trusted(out["id"], sctx=ctx) is True


def test_launch_snapshot_follows_the_config_service_rule(ctx, demo):
    """The launch snapshot's ``config_raw`` is what the config service's impact preview compares
    Studio-launched runs with (``config_service.launch_raw``)."""
    from sparc.studio.projects.config_service import launch_raw
    from sparc.studio.runs.launch import absolutise
    from sparc.studio.runs.reader import load_config_raw

    pdir = Path(demo["dir"])
    raw = load_config_raw(demo["config_path"])
    raw["physics"]["forcing"] = "inputs/forcing/day.json"
    raw["data"]["path"] = "~/not-expanded.csv"                  # core resolves "~" literally, so does the rule
    mine = absolutise(raw, pdir, cache_dir=ctx.workspace.cache_dir, runs_dir=pdir / "runs")
    assert mine == launch_raw(raw, pdir, ctx.workspace)
    assert mine["data"]["path"] == str((pdir / "~/not-expanded.csv").resolve())
    assert mine["climate"]["cache"] == str(ctx.workspace.cache_dir.resolve())
    assert mine["output"]["dir"] == str((pdir / "runs").resolve())
    # join tables keep their list form and their keys
    absolute = str(Path(pdir.anchor) / "abs" / "x.csv")                  # absolute on every platform
    raw["data"]["join"] = [{"path": "data/extra.csv", "key": "id"}, {"path": absolute, "key": "id"}]
    joined = absolutise(raw, pdir, cache_dir=ctx.workspace.cache_dir, runs_dir=pdir / "runs")["data"]["join"]
    assert joined == [{"path": str((pdir / "data/extra.csv").resolve()), "key": "id"},
                      {"path": absolute, "key": "id"}]


def test_resume_impact_compares_the_launch_form(ctx, demo, place_run):
    from sparc.core.config import core_config_from_dict
    from sparc.core.pipeline import fingerprint_sections
    from sparc.studio.runs.launch import impact, project_row

    rd = place_run(demo)
    launch = json.loads((rd / "studio" / "launch.json").read_text())
    cfg = core_config_from_dict(launch["config_raw"], base_dir=launch["config_dir"])
    side = json.loads((rd / "checkpoint.json").read_text())
    side["sections"] = fingerprint_sections(cfg, True)          # as core wrote it for this launch (fast mode)
    (rd / "checkpoint.json").write_text(json.dumps(side))
    project = project_row(ctx.db, demo["id"])
    run = ctx.services["reader"].get(RUN_ID)
    assert impact(run, project, ctx.workspace) == {
        "changed_sections": [], "refit_from": None,
        "phrase": "the current config matches the checkpoint: nothing would be refitted"}
    doc = yaml.safe_load(Path(demo["config_path"]).read_text())
    doc["core"]["optimize"]["budget"] = 3000.0
    Path(demo["config_path"]).write_text(yaml.safe_dump(doc, sort_keys=False))
    imp = impact(ctx.services["reader"].get(RUN_ID), project, ctx.workspace)
    assert imp["changed_sections"] == ["s7"] and imp["refit_from"] == "S7"


def test_resume_keeps_the_snapshot_threads(client, demo, place_run, replay_runner, wait_job):
    def cancelled(rd):
        st = json.loads((rd / "run_state.json").read_text())
        st.update(status="cancelled", done=["S3"])
        (rd / "run_state.json").write_text(json.dumps(st))
        (rd / "manifest.json").unlink()

    rd = place_run(demo, edit=cancelled)
    launch = json.loads((rd / "studio" / "launch.json").read_text())
    launch["args"]["threads"] = 1
    (rd / "studio" / "launch.json").write_text(json.dumps(launch))
    r = client.post(f"/api/runs/{RUN_ID}/resume", json={})
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["params"]["threads"] == 1
    done = wait_job(client, job["id"], timeout=120)
    assert done["status"] == "succeeded" and done["threads"] == 1
    # a refused resume of a complete run with the current config shows what it would refit
    wait_for(lambda: client.get(f"/api/runs/{RUN_ID}").json()["run"]["status"] == "complete", 30, what="complete")
    r = client.post(f"/api/runs/{RUN_ID}/resume", json={"use_current_config": True})
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_resumable"
    assert {"checkpoint", "impact"} <= set(r.json()["error"]["detail"])


def test_completed_run_becomes_the_active_run(client, ctx, demo, replay_runner, wait_job):
    body = client.post(f"/api/projects/{demo['id']}/runs", json={"mode": "fast"}).json()
    rid = body["run"]["id"]
    assert wait_job(client, body["job"]["id"], timeout=120)["status"] == "succeeded"
    wait_for(lambda: client.get(f"/api/projects/{demo['id']}").json()["project"]["active_run_id"] == rid, 30,
             what="the active run")
    record = json.loads((Path(demo["dir"]) / "project.json").read_text())
    assert record["active_run_id"] == rid                         # project.json kept in sync
    # a second run does not take over an existing active run
    body2 = client.post(f"/api/projects/{demo['id']}/runs", json={"mode": "fast"}).json()
    assert wait_job(client, body2["job"]["id"], timeout=120)["status"] == "succeeded"
    wait_for(lambda: client.get(f"/api/runs/{body2['run']['id']}").json()["run"]["status"] == "complete", 30,
             what="second run complete")
    assert client.get(f"/api/projects/{demo['id']}").json()["project"]["active_run_id"] == rid
    # deleting the active run clears it
    assert client.delete(f"/api/runs/{rid}").status_code == 200
    assert client.get(f"/api/projects/{demo['id']}").json()["project"]["active_run_id"] is None
    assert json.loads((Path(demo["dir"]) / "project.json").read_text())["active_run_id"] is None


@pytest.mark.parametrize("what", ["studies", "duration"])
def test_run_summaries_agree_with_the_project_service(ctx, demo, fixture_run, what):
    """``registry.run_summary`` and ``projects.service.run_summary`` give the same object for the same row."""
    from sparc.studio import db as dbmod
    from sparc.studio.projects.service import project_runs
    from sparc.studio.runs.registry import run_summary

    rid, _ = fixture_run
    if what == "studies":
        ctx.db.insert("study_links", {"run_id": rid, "study_id": "st_aaaa0001", "attached": 1})
        ctx.db.insert("study_links", {"run_id": rid, "study_id": "st_aaaa0002", "attached": 0})
    else:
        row = ctx.db.fetchone("SELECT stages_json FROM runs WHERE id = ?", (rid,))
        info = dbmod.loads(row["stages_json"], {})
        info.pop("duration_s", None)
        ctx.db.update("runs", {"id": rid}, {"stages_json": dbmod.dumps(info), "finished_utc": "2026-10-01T21:30:00Z",
                                            "created_utc": "2026-10-01T21:21:18Z"})
    row = ctx.db.fetchone("SELECT * FROM runs WHERE id = ?", (rid,))
    reg = ctx.services["registry"]
    mine = reg.summaries([row])[0]
    theirs = next(r for r in project_runs(ctx.db, demo["id"]) if r["id"] == rid)
    assert mine == theirs
    if what == "studies":
        assert mine["studies"] == ["st_aaaa0001"]
    else:
        assert mine["duration_s"] == pytest.approx(522.0)
    assert run_summary(row, ["x"])["studies"] == ["x"]
