"""Studies and post-run actions (SPEC §8, api.md §9).

Jobs run in real worker processes with the core functions faked (``fake_worker.py``): the dispatch passes the
run's launch-snapshot config, the workspace cache, the study's ``children`` folder, the explicit ``run_meta``
and the reproduction's ``config_dir`` / ``out_dir``; the kinds write the ``.md`` / ``.json`` files the CLI
used to; child runs register live with ``origin = study_child``; attaching a finished study enqueues
``post.uncertainty``; a refitted run makes its studies stale.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.studio.conftest import wait_for
from tests.studio.studies.conftest import RUN_ID, snapshot_title
from tests.studio.studies.fakes import simcheck_row, write_variant


def _post(client, path, body=None, status=202):
    r = client.post(path, json=body if body is not None else {})
    assert r.status_code == status, r.text
    return r.json()


def _done(client, wait_job, job, status="succeeded", timeout=60):
    j = wait_job(client, job["id"], timeout=timeout)
    assert j["status"] == status, j
    return j


def _refit(ctx, run_dir: Path, fp: str) -> None:
    """What a refit leaves behind: a checkpoint with another fingerprint (the registry re-derives the row)."""
    for name in ("checkpoint.json", "run_state.json"):
        doc = json.loads((run_dir / name).read_text())
        doc["fingerprint"] = fp
        (run_dir / name).write_text(json.dumps(doc))
    ctx.services["registry"].refresh(run_dir.name)


# ---------------------------------------------------------------------------
# post-run actions
# ---------------------------------------------------------------------------

def test_post_actions_run_on_the_launch_snapshot(client, ctx, fixture_run, fake_studies, calls, wait_job):
    rid, rd = fixture_run
    title = snapshot_title(rd)
    job = _post(client, f"/api/runs/{rid}/actions/baselines", {"models": ["idw", "hgb"]})
    assert job["kind"] == "post.baselines" and job["run_id"] == rid and job["lane"] == "medium"
    _done(client, wait_job, job)
    (call,) = calls(job)
    assert call["fn"] == "baselines_for_run" and call["models"] == ["idw", "hgb"]
    assert call["cfg"]["title"] == title                      # the launch snapshot, not the project config
    assert Path(call["run_dir"]) == rd
    assert json.loads((rd / "baselines.json").read_text())["best_baseline"] == "idw"

    job = _post(client, f"/api/runs/{rid}/actions/planner", {"package": "cooling-package", "thresholds": [88]})
    _done(client, wait_job, job)
    (call,) = calls(job)
    assert Path(call["cache_dir"]) == ctx.workspace.cache_dir
    assert call["package"] == "Cooling package" and call["thresholds"] == [88.0] and call["cfg"]["title"] == title
    res = client.get(f"/api/jobs/{job['id']}").json()["result"]
    assert res["people_total"] == 1234.0 and "planner/planner.json" in res["files"] and res["hot_days"] is True

    r = client.post(f"/api/runs/{rid}/actions/planner", json={"package": "no such scenario"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation"

    job = _post(client, f"/api/runs/{rid}/actions/writeup")
    _done(client, wait_job, job)
    assert client.get(f"/api/jobs/{job['id']}").json()["result"]["files"] == ["methods.md", "model_card.md"]
    rows = {r["kind"]: r for r in client.get(f"/api/runs/{rid}/studies").json()}
    assert rows["baselines"]["state"] == "done" and rows["baselines"]["headline"] == "the stack beats every baseline"
    assert rows["planner"]["state"] == "done" and rows["planner"]["job_id"]


def test_planner_without_layers_is_refused(client, ctx, demo, place_run):
    def drop(raw):
        raw.pop("planner", None)

    rd = place_run(demo, edit_config=drop)
    r = client.post(f"/api/runs/{RUN_ID}/actions/planner", json={})
    assert r.status_code == 422, r.text
    err = r.json()["error"]
    assert err["code"] == "requirements" and err["detail"]["missing"] == ["planner.layers"]
    rows = {x["kind"]: x for x in client.get(f"/api/runs/{RUN_ID}/studies").json()}
    assert rows["planner"]["requirements"] == {"ok": False, "missing": ["planner.layers"]}
    assert rows["emulator"]["requirements"]["missing"] == ["checkpoint"]
    assert not (rd / "planner").exists()


def test_emulator_needs_the_checkpoint(client, fixture_run, fake_studies, calls, wait_job):
    rid, rd = fixture_run
    r = client.post(f"/api/runs/{rid}/actions/emulator", json={})
    assert r.status_code == 409 and r.json()["error"]["code"] == "no_checkpoint"
    (rd / "checkpoint.pkl").write_bytes(b"not a real checkpoint")
    job = _post(client, f"/api/runs/{rid}/actions/emulator", {"patches": 3})
    assert job["lane"] == "heavy"
    _done(client, wait_job, job)
    assert calls(job)[0]["n_patches"] == 3
    res = client.get(f"/api/jobs/{job['id']}").json()["result"]
    assert set(res["levers"]) == {"canopy", "impervious", "albedo"}
    assert res["levers"]["canopy"] == {"patch_pass_rate": 0.9, "uniform_rel_err": 0.05}


def test_unknown_kinds_and_params(client, fixture_run):
    rid, _ = fixture_run
    assert client.post(f"/api/runs/{rid}/actions/nope", json={}).json()["error"]["code"] == "unknown_kind"
    assert client.post(f"/api/runs/{rid}/studies/nope", json={}).json()["error"]["code"] == "unknown_kind"
    r = client.post(f"/api/runs/{rid}/actions/baselines", json={"models": ["kriging-ish"]})
    assert r.status_code == 422 and r.json()["error"]["detail"]["errors"][0]["path"].startswith("params")
    r = client.post(f"/api/runs/{rid}/studies/placebo", json={"kinds": ["shift"], "bogus": 1})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# studies
# ---------------------------------------------------------------------------

def test_placebo_dispatch_children_and_files(client, ctx, fixture_run, fake_studies, calls, wait_job):
    rid, rd = fixture_run
    title = snapshot_title(rd)
    out = _post(client, f"/api/runs/{rid}/studies/placebo", {"kinds": ["shift", "rotate"], "coarse_m": 120})
    study, job = out["study"], out["job"]
    assert study["kind"] == "placebo" and study["target_run_id"] == rid and job["study_id"] == study["id"]
    assert study["attached_runs"] == [rid] and study["origin"] == "studio"
    sdir = Path(study["out_dir"])
    assert sdir == Path(demo_dir(ctx, study)) / "studies" / study["id"]
    _done(client, wait_job, job)
    (call,) = calls(job)
    assert call["fn"] == "run_placebo_suite" and call["cfg"]["title"] == title
    assert Path(call["children_dir"]) == sdir / "children" and call["resume"] is True
    assert call["kinds"] == ["shift", "rotate"] and call["coarse"] == 120.0
    assert call["run_meta"] == {"study_id": study["id"], "parent_run_id": rid, "project_id": study["project_id"],
                                "origin": "study_child"}
    # placebo.json / placebo.md written by the kind
    pz = json.loads((sdir / "placebo.json").read_text())
    assert pz["n_placebos"] == 2 and (sdir / "placebo.md").read_text().startswith("| run | layer |")
    # children registered (live, through the run.dir events) with the explicit link
    kids = ctx.db.fetchall("SELECT * FROM runs WHERE study_id = ?", (study["id"],))
    assert len(kids) == 2
    assert {k["origin"] for k in kids} == {"study_child"} and {k["parent_run_id"] for k in kids} == {rid}
    res = client.get(f"/api/jobs/{job['id']}").json()["result"]
    assert sorted(res["children"]) == sorted(k["id"] for k in kids)
    s = client.get(f"/api/studies/{study['id']}").json()
    assert s["status"] == "succeeded" and len(s["children"]) == 2
    assert s["summary"]["n_placebos"] == 2 and s["summary"]["n_pass_model"] == 1
    assert s["summary"]["headline"] == "model passes 1/2, causal passes 2/2"
    view = client.get(f"/api/studies/{study['id']}/view").json()
    assert view["n_placebos"] == 2 and len(view["rows"]) == 2
    verdicts = {c["kind"]: c["verdict"] for c in view["children"]}
    assert verdicts == {"shift": "passes", "rotate": "model fails"}
    assert all(c["run_id"] for c in view["children"])
    # the mirror file
    mirror = json.loads((sdir / "study.json").read_text())
    assert mirror["id"] == study["id"] and mirror["links"] == [{"run_id": rid, "attached": True}]


def demo_dir(ctx, study) -> str:
    return ctx.db.fetchone("SELECT dir FROM projects WHERE id = ?", (study["project_id"],))["dir"]


def test_simcheck_dispatch_continue_and_live_grid(client, ctx, fixture_run, fake_studies, calls, wait_job):
    rid, rd = fixture_run
    ctx.db.set_setting("threads_heavy", 2)
    ctx.reload_settings()
    r = client.post(f"/api/runs/{rid}/studies/simcheck", json={"design": {"physics": 2}, "workers": 2, "threads": 2})
    assert r.status_code == 422 and "threads_heavy" in r.text
    out = _post(client, f"/api/runs/{rid}/studies/simcheck",
                {"design": {"physics": 2, "null": 1}, "coarse_m": 120, "epochs": 20, "workers": 2, "threads": 1})
    study, job = out["study"], out["job"]
    assert job["threads"] in (None, 2)
    _done(client, wait_job, job)
    (call,) = calls(job)
    sdir = Path(study["out_dir"])
    assert Path(call["out_dir"]) == sdir and call["design"] == {"physics": 2, "null": 1}
    assert call["workers"] == 2 and call["threads"] == 1 and call["epochs"] == 20 and call["coarse"] == 120.0
    assert (sdir / "simcheck_summary.md").read_text().startswith("| generator |")
    view = client.get(f"/api/studies/{study['id']}/view").json()
    assert [(g["generator"], g["seed"], g["status"]) for g in view["grid"]] == [
        ("physics", 0, "done"), ("physics", 1, "done"), ("null", 0, "done")]
    assert view["generators"]["physics"]["n"] == 2 and view["eta_s"] == 0.0
    # continue the same study with more replicates: resumes into its folder
    out2 = _post(client, f"/api/runs/{rid}/studies/simcheck",
                 {"design": {"physics": 3, "null": 1}, "continue_study_id": study["id"], "workers": 1})
    assert out2["study"]["id"] == study["id"]
    _done(client, wait_job, out2["job"])
    rows = [json.loads(x) for x in (sdir / "simcheck.jsonl").read_text().splitlines()]
    assert sorted((r["generator"], r["seed"]) for r in rows) == [("null", 0), ("physics", 0), ("physics", 1),
                                                                 ("physics", 2)]
    s = client.get(f"/api/studies/{study['id']}").json()
    assert s["summary"]["n_rows"] == 4 and s["summary"]["bias_correction"]["share_range"]


def test_simcheck_view_is_live_from_the_jsonl(client, ctx, fixture_run):
    """While a study runs, the grid comes from ``simcheck.jsonl``; missing pairs are pending."""
    from sparc.studio.studies import service, views

    rid, _ = fixture_run
    pid = ctx.db.fetchone("SELECT project_id FROM runs WHERE id = ?", (rid,))["project_id"]
    row = service.create_study(ctx.db, project_id=pid, kind="simcheck", target_run_id=rid,
                               params={"design": {"physics": 2, "own_only": 1}, "workers": 1})
    sdir = Path(row["out_dir"])
    bad = simcheck_row("physics", 1)
    bad["gate"] = {"pass": False, "attempt": 2}
    lines = [json.dumps(simcheck_row("physics", 0)), json.dumps(bad), '{"generator": "own_only", "seed"']
    (sdir / "simcheck.jsonl").write_text("\n".join(lines))
    v = views.simcheck_view(ctx.db, row)
    cells = {(g["generator"], g["seed"]): g for g in v["grid"]}
    assert cells[("physics", 0)]["status"] == "done" and cells[("physics", 1)]["status"] == "gate_fail"
    assert cells[("physics", 1)]["gate_attempt"] == 2 and cells[("own_only", 0)]["status"] == "pending"


def test_multiverse_dispatch_custom_variants(client, ctx, fixture_run, fake_studies, calls, wait_job):
    rid, rd = fixture_run
    r = client.post(f"/api/runs/{rid}/studies/multiverse", json={"variants": ["no_such_variant"]})
    assert r.status_code == 422
    r = client.post(f"/api/runs/{rid}/studies/multiverse", json={"custom_variants": {"baseline": {"cv.block_m": 1}}})
    assert r.status_code == 422
    out = _post(client, f"/api/runs/{rid}/studies/multiverse",
                {"variants": ["baseline", "blocks_1km"], "custom_variants": {"wide_lag": {"influence.max_lag_m": 3000}},
                 "coarse_m": 120})
    study, job = out["study"], out["job"]
    _done(client, wait_job, job)
    (call,) = calls(job)
    sdir = Path(study["out_dir"])
    assert Path(call["out_dir"]) == sdir and Path(call["cfg"]["output_dir"]) == sdir / "children"
    assert call["extra_variants"] == {"wide_lag": {"influence.max_lag_m": 3000}}
    assert call["run_meta"]["study_id"] == study["id"] and call["run_meta"]["parent_run_id"] == rid
    assert (sdir / "multiverse_summary.md").read_text().startswith("| scenario |")
    kids = ctx.db.fetchall("SELECT id, origin, parent_run_id FROM runs WHERE study_id = ?", (study["id"],))
    assert len(kids) == 3 and all(k["origin"] == "study_child" and k["parent_run_id"] == rid for k in kids)
    view = client.get(f"/api/studies/{study['id']}/view").json()
    assert [v["name"] for v in view["variants"]] == ["baseline", "blocks_1km", "wide_lag"]
    assert all(v["status"] == "done" and v["run_id"] for v in view["variants"])
    assert view["stability"]["sign_stability_min"] == 1.0 and "Canopy Increase +10" in view["effects"]
    assert set(view["priority"]) == {"blocks_1km", "wide_lag"}
    s = client.get(f"/api/studies/{study['id']}").json()
    assert s["summary"]["sign_stability_min"] == 1.0 and s["summary"]["headline"].startswith("signs stable")


def test_reproduce_passes_config_dir_and_out_dir(client, ctx, demo, fixture_run, fake_studies, calls, wait_job):
    rid, rd = fixture_run
    out = _post(client, f"/api/runs/{rid}/studies/reproduce", {"stages": ["S0", "S1"], "tol_r2": 0.02})
    study, job = out["study"], out["job"]
    _done(client, wait_job, job)
    (call,) = calls(job)
    sdir = Path(study["out_dir"])
    assert Path(call["config_dir"]) == Path(demo["dir"])                # launch.json config_dir
    assert Path(call["out_dir"]) == sdir / "children" / f"{rd.name}_reproduce"
    assert call["stages"] == ["S0", "S1"] and call["tol_r2"] == 0.02
    assert call["run_meta"]["parent_run_id"] == rid
    res = client.get(f"/api/jobs/{job['id']}").json()["result"]
    child = ctx.db.fetchone("SELECT * FROM runs WHERE id = ?", (res["child_run_id"],))
    assert child is not None and child["parent_run_id"] == rid and child["origin"] == "reproduction"
    assert res["pass"] is True and res["n_hard_fail"] == 0
    view = client.get(f"/api/studies/{study['id']}/view").json()
    assert view["pass"] is True and [c["check"] for c in view["checks"]][:2] == ["cv design", "R² stacker"]
    assert client.get(f"/api/studies/{study['id']}").json()["summary"]["pass"] is True


def test_benchmark_writes_json_and_md(client, demo, fake_studies, calls, wait_job):
    out = _post(client, f"/api/projects/{demo['id']}/studies/benchmark", {"n": 48, "ab": False, "epochs": 30})
    study, job = out["study"], out["job"]
    assert study["target_run_id"] is None and job["run_id"] is None
    _done(client, wait_job, job)
    (call,) = calls(job)
    assert call == {"fn": "run_benchmark", "seed": 0, "spatial_plus_ab": False, "epochs": 30, "n": 48}
    sdir = Path(study["out_dir"])
    assert json.loads((sdir / "benchmark.json").read_text())["n"] == 48
    assert (sdir / "benchmark.md").read_text().startswith("| setting | model |")
    view = client.get(f"/api/studies/{study['id']}/view").json()
    assert set(view["runs"]) == {"spatial_plus"}
    assert client.get(f"/api/studies/{study['id']}").json()["summary"]["headline"] == "stack effect share 0.92"


def test_requirements_roles_for_placebo_and_simcheck(client, ctx, demo, place_run):
    def no_roles(raw):
        raw["physics"]["roles"].pop("canopy", None)
        raw["physics"]["roles"].pop("impervious", None)

    place_run(demo, edit_config=no_roles)
    r = client.post(f"/api/runs/{RUN_ID}/studies/placebo", json={"kinds": ["shift"]})
    assert r.status_code == 422 and r.json()["error"]["code"] == "requirements"
    assert r.json()["error"]["detail"]["missing"] == ["physics.roles.canopy", "physics.roles.impervious"]
    r = client.post(f"/api/runs/{RUN_ID}/studies/simcheck", json={"design": {"physics": 1}})
    assert r.status_code == 422 and r.json()["error"]["code"] == "requirements"


# ---------------------------------------------------------------------------
# attach, auto uncertainty, staleness
# ---------------------------------------------------------------------------

def test_attach_enqueues_uncertainty_and_stale_vs_fingerprint(client, ctx, demo, place_run, fake_studies, calls,
                                                              wait_job):
    from sparc.studio.studies import service

    rd = place_run(demo)
    other = place_run(demo, "20261001-222222-fast-c3d4")
    # a finished multiverse study on the first run
    out = _post(client, f"/api/runs/{RUN_ID}/studies/multiverse", {"variants": ["baseline"], "coarse_m": 120})
    _done(client, wait_job, out["job"])
    sid = out["study"]["id"]
    # its end auto-enqueued post.uncertainty on the run it is attached to (auto_uncertainty on by default)
    unc = wait_for(lambda: ctx.db.fetchone("SELECT * FROM jobs WHERE run_id = ? AND kind = 'post.uncertainty'",
                                           (RUN_ID,)), 30, what="auto uncertainty")
    _done(client, wait_job, {"id": unc["id"]})
    (call,) = calls({"id": unc["id"]})
    assert Path(call["multiverse_dir"]) == Path(out["study"]["out_dir"])
    # attaching it to another run enqueues post.uncertainty there
    res = _post(client, f"/api/studies/{sid}/attach", {"run_id": "20261001-222222-fast-c3d4"}, status=200)
    assert res["job"] is not None and res["job"]["kind"] == "post.uncertainty"
    assert res["job"]["run_id"] == "20261001-222222-fast-c3d4"
    assert sorted(res["study"]["attached_runs"]) == sorted([RUN_ID, "20261001-222222-fast-c3d4"])
    _done(client, wait_job, res["job"])
    assert (other / "uncertainty.json").exists()
    # with auto_uncertainty off, attaching does not enqueue
    r = client.put("/api/settings", json={"auto_uncertainty": False})
    assert r.status_code == 200, r.text
    res = _post(client, f"/api/studies/{sid}/detach", {"run_id": "20261001-222222-fast-c3d4"}, status=200)
    assert res["job"] is None and res["study"]["attached_runs"] == [RUN_ID]
    res = _post(client, f"/api/studies/{sid}/attach", {"run_id": "20261001-222222-fast-c3d4"}, status=200)
    assert res["job"] is None
    # stale: the run is refitted after the study ran (its checkpoint fingerprint changes); the other run, which
    # the study was attached to, carries the same fingerprint as the one recorded
    assert client.get(f"/api/studies/{sid}").json()["stale_vs"] == []
    _refit(ctx, rd, "b" * 16)
    s = client.get(f"/api/studies/{sid}").json()
    assert s["stale_vs"] == [RUN_ID]
    rows = {r["kind"]: r for r in client.get(f"/api/runs/{RUN_ID}/studies").json()}
    assert rows["multiverse"]["state"] == "stale" and rows["multiverse"]["attached"] is True
    assert service.stale_vs(ctx.db, service.get_row(ctx.db, sid)) == [RUN_ID]
    del rd


def test_attach_hook_enqueues_through_the_server_loop(client, ctx, fixture_run, fake_studies, wait_job):
    """``study_on_finish`` runs in a hook thread: the submission is scheduled on the server loop."""
    from sparc.studio.studies import kinds, service

    rid, _ = fixture_run
    pid = ctx.db.fetchone("SELECT project_id FROM runs WHERE id = ?", (rid,))["project_id"]
    row = service.create_study(ctx.db, project_id=pid, kind="placebo", target_run_id=rid, params={"kinds": ["shift"]})
    kinds.study_on_finish(ctx, {"id": "j_none", "study_id": row["id"], "run_id": rid, "project_id": pid,
                                "status": "succeeded"}, {})
    job = wait_for(lambda: ctx.db.fetchone("SELECT id FROM jobs WHERE run_id = ? AND kind = 'post.uncertainty'",
                                           (rid,)), 20, what="the auto-enqueued job")
    _done(client, wait_job, job)
    # a second finish while one is waiting does not queue a duplicate
    assert service.get_row(ctx.db, row["id"])["status"] == "succeeded"


# ---------------------------------------------------------------------------
# reading, estimates, merge, import, reindex, delete
# ---------------------------------------------------------------------------

def test_status_rows_cover_every_kind(client, fixture_run):
    rid, _ = fixture_run
    rows = client.get(f"/api/runs/{rid}/studies").json()
    assert [r["kind"] for r in rows] == ["baselines", "planner", "emulator", "uncertainty", "writeup", "placebo",
                                         "simcheck", "multiverse", "reproduce", "literature", "benchmark"]
    by = {r["kind"]: r for r in rows}
    assert by["baselines"]["state"] == "done"                    # the fixture's S2_S3 baselines
    assert by["placebo"]["state"] == "not_run" and by["placebo"]["attached"] is False
    assert by["placebo"]["action"]["path"] == f"/api/runs/{rid}/studies/placebo"
    assert by["emulator"]["action"]["kind"] == "build_emulator"
    assert by["literature"]["state"] == "done" and "within ×2" in by["literature"]["headline"]
    assert all(r["estimate"]["est_lo"] <= r["estimate"]["est_s"] <= r["estimate"]["est_hi"]
               for r in rows if r["estimate"])
    assert by["planner"]["requirements"]["ok"] is True


def test_estimate_endpoint(client, fixture_run):
    rid, _ = fixture_run
    a = client.post("/api/studies/estimate", json={"kind": "placebo", "run_id": rid,
                                                    "params": {"kinds": ["shift"], "coarse_m": 120}}).json()
    b = client.post("/api/studies/estimate", json={"kind": "placebo", "run_id": rid,
                                                    "params": {"kinds": ["grf", "shift", "rotate"], "coarse_m": 120}}).json()
    assert a["n_children"] == 1 and b["n_children"] == 3 and b["est_s"] == pytest.approx(3 * a["est_s"], rel=0.01)
    assert a["est_lo"] < a["est_s"] < a["est_hi"] and a["est_peak_rss_gb"] > 0 and a["est_disk_gb"] >= 0
    one = client.post("/api/studies/estimate", json={"kind": "simcheck", "run_id": rid,
                                                      "params": {"design": {"physics": 4}, "workers": 1}}).json()
    two = client.post("/api/studies/estimate", json={"kind": "simcheck", "run_id": rid,
                                                      "params": {"design": {"physics": 4}, "workers": 2}}).json()
    assert two["est_s"] == pytest.approx(one["est_s"] / 2, rel=0.01)
    r = client.post("/api/studies/estimate", json={"kind": "simcheck", "params": {}})
    assert r.status_code == 422
    assert client.post("/api/studies/estimate", json={"kind": "nope", "params": {}}).status_code == 404


def test_simcheck_merge_inline(client, ctx, fixture_run):
    from sparc.studio.studies import service

    rid, _ = fixture_run
    pid = ctx.db.fetchone("SELECT project_id FROM runs WHERE id = ?", (rid,))["project_id"]
    ids = []
    for gen in ("physics", "additive"):
        row = service.create_study(ctx.db, project_id=pid, kind="simcheck", target_run_id=rid,
                                   params={"design": {gen: 2}}, status="succeeded")
        Path(row["out_dir"], "simcheck.jsonl").write_text(
            "\n".join(json.dumps(simcheck_row(gen, s, 0.7 + 0.2 * (gen == "additive"))) for s in range(2)))
        ids.append(row["id"])
    out = client.post("/api/studies/simcheck/merge", json={"study_ids": ids}).json()
    assert set(out["summary"]["generators"]) == {"physics", "additive"} and out["summary"]["n_rows"] == 4
    assert out["markdown"].startswith("| generator |") and "Effect share" in out["markdown"]
    r = client.post("/api/studies/simcheck/merge", json={"study_ids": ["st_nope0000"]})
    assert r.status_code == 404


def test_import_study_dirs(client, ctx, demo, fixture_run, tmp_path):
    """Placebo, simcheck (one study per sub-folder, matched to the run's uncertainty sources) and multiverse
    folders import in place; a second import finds the same study."""
    from sparc.studio.studies.service import import_study_dir

    rid, rd = fixture_run
    root = tmp_path / "providence"
    (root / "placebo").mkdir(parents=True)
    (root / "placebo" / "placebo.json").write_text(json.dumps({"n_placebos": 3, "n_pass_model": 2, "n_pass_causal": 3,
                                                                 "rows": [], "kinds": ["grf", "shift", "rotate"]}))
    for sub in ("null", "physics"):
        d = root / "simcheck" / sub
        d.mkdir(parents=True)
        (d / "simcheck.jsonl").write_text("\n".join(json.dumps(simcheck_row(sub, s)) for s in range(2)))
    mv = root / "multiverse"
    mv.mkdir()
    for i, n in enumerate(("baseline", "blocks_1km")):
        write_variant(mv, n, rd, 1 + 0.1 * i)
    (rd / "uncertainty.json").write_text(json.dumps({"scenarios": [], "sources": {
        "run": str(rd), "multiverse": None, "simcheck": [str(root / "simcheck" / "physics")]}}))
    pz = import_study_dir(root / "placebo", demo["id"], kind="placebo", target_run_id=rid, sctx=ctx)
    assert pz["origin"] == "imported" and pz["summary"]["n_placebos"] == 3 and pz["status"] == "succeeded"
    sc = import_study_dir(root / "simcheck", demo["id"], kind="simcheck", target_run_id=rid, sctx=ctx)
    rows = ctx.db.fetchall("SELECT id, out_dir FROM studies WHERE kind = 'simcheck'")
    assert sorted(Path(r["out_dir"]).name for r in rows) == ["null", "physics"]
    assert Path(sc["out_dir"]).name == "null" and sc["summary"]["label"] == "simcheck/null"
    phys = next(r for r in rows if r["out_dir"].endswith("physics"))
    # only the folder the run's uncertainty report used is attached
    links = {r["study_id"]: r["attached"] for r in ctx.db.fetchall("SELECT * FROM study_links WHERE run_id = ?", (rid,))}
    assert links[phys["id"]] == 1 and links[sc["id"]] == 0
    m = import_study_dir(mv, demo["id"], target_run_id=rid, sctx=ctx)
    assert m["kind"] == "multiverse" and m["summary"]["sign_stability_min"] == 1.0
    again = import_study_dir(mv, demo["id"], target_run_id=rid, sctx=ctx)
    assert again["id"] == m["id"]
    view = client.get(f"/api/studies/{m['id']}/view").json()
    assert [v["name"] for v in view["variants"]] == ["baseline", "blocks_1km"]
    # the run hub's uncertainty sources resolve these rows by folder
    from sparc.studio.errors import ApiError

    with pytest.raises(ApiError):
        import_study_dir(tmp_path / "nothing", demo["id"], sctx=ctx)
    # imported studies cannot be resumed
    r = client.post(f"/api/studies/{m['id']}/resume", json={})
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_resumable"


def test_reindex_rebuilds_studies_from_their_mirrors(client, ctx, fixture_run, fake_studies, wait_job):
    from sparc.studio import db as dbmod

    rid, _ = fixture_run
    out = _post(client, f"/api/runs/{rid}/studies/placebo", {"kinds": ["shift"], "coarse_m": 120})
    _done(client, wait_job, out["job"])
    before = client.get(f"/api/studies/{out['study']['id']}").json()
    summary = dbmod.reindex(ctx.db, ctx.workspace)
    assert summary["studies"]["studies"] >= 1
    after = client.get(f"/api/studies/{out['study']['id']}").json()
    for k in ("kind", "target_run_id", "out_dir", "status", "params", "summary", "attached_runs", "origin"):
        assert after[k] == before[k], k
    assert len(after["children"]) == 1


def test_delete_study(client, ctx, fixture_run, fake_studies, wait_job):
    rid, _ = fixture_run
    out = _post(client, f"/api/runs/{rid}/studies/placebo", {"kinds": ["shift"], "coarse_m": 120})
    _done(client, wait_job, out["job"])
    sid, sdir = out["study"]["id"], Path(out["study"]["out_dir"])
    assert client.delete(f"/api/studies/{sid}", params={"files": "true"}).json() == {"ok": True}
    assert client.get(f"/api/studies/{sid}").status_code == 404 and not sdir.exists()
    assert ctx.db.fetchone("SELECT id FROM runs WHERE study_id = ?", (sid,)) is None


def test_resume_reuses_the_study(client, ctx, fixture_run, fake_studies, calls, wait_job):
    rid, _ = fixture_run
    out = _post(client, f"/api/runs/{rid}/studies/placebo", {"kinds": ["shift"], "coarse_m": 120})
    _done(client, wait_job, out["job"])
    job = _post(client, f"/api/studies/{out['study']['id']}/resume")
    assert job["study_id"] == out["study"]["id"] and job["kind"] == "study.placebo"
    _done(client, wait_job, job)
    assert calls(job)[0]["resume"] is True
    assert client.get(f"/api/studies/{out['study']['id']}").json()["job_id"] == job["id"]


# ---------------------------------------------------------------------------
# truth vs recovered
# ---------------------------------------------------------------------------

def test_truth_on_the_demo(client, ctx, fixture_run):
    rid, rd = fixture_run
    rows = client.get(f"/api/runs/{rid}/truth").json()["rows"]
    by = {(r["quantity"], r["scenario"]): r for r in rows}
    m = json.loads((rd / "manifest.json").read_text())
    s10 = next(s for s in m["scenarios"] if s["name"] == "Canopy Increase +10")
    row = by[("canopy_scenario", "Canopy Increase +10")]
    assert row["recovered"] == pytest.approx(s10["mean_delta"]) and row["se"] == pytest.approx(s10["mean_delta_se"])
    assert row["share"] == pytest.approx(s10["mean_delta"] / row["truth"]) and row["unit"] == "°F"
    assert {q for q, _ in by} == {"canopy_scenario", "footprint_mean", "L_m", "influence_radius_m", "noise_sd"}
    assert by[("L_m", None)]["truth"] == 150.0 and by[("L_m", None)]["recovered"] == pytest.approx(
        m["physics"]["L_m"]["mean"])
    assert by[("influence_radius_m", None)]["recovered"] == m["influence"]["ranges_m"]["canopy"]
    assert by[("noise_sd", None)]["recovered"] == pytest.approx(m["metrics"]["stacker"]["rmse"])
    # not a demo project → 404
    ctx.db.execute("UPDATE projects SET demo = 0")
    assert client.get(f"/api/runs/{rid}/truth").status_code == 404
    shutil.rmtree(rd / "studio", ignore_errors=True)
