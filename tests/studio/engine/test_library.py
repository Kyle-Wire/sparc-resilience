"""The scenario library (SPEC §7.4, api.md §7.4): CRUD, revisions, mirrors, templates, ladders, promote, designs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.studio.engine.conftest import make_result

DOC = {"name": "Corridor", "tags": ["district"],
       "edits": [{"lever": "canopy", "mode": "add", "amount": 10,
                  "where": {"kind": "top", "column": "pred:target", "frac": 0.2, "direction": "highest"}}],
       "regions": {"hot": {"kind": "top", "column": "pred:target", "frac": 0.1, "direction": "highest"}}}


def _create(client, pid, doc=DOC):
    r = client.post(f"/api/projects/{pid}/scenarios", json={"doc": doc})
    assert r.status_code == 201, r.text
    return r.json()


def test_crud_mirror_and_listing(client, ctx, demo):
    pid = demo["id"]
    sc = _create(client, pid)
    assert sc["revision"] == 1 and sc["parent_id"] is None and sc["status"] == "draft" and sc["results"] == []
    mirror = Path(demo["dir"]) / "scenarios" / f"{sc['id']}.json"
    m = json.loads(mirror.read_text())
    assert m["schema"] == 1 and [r["id"] for r in m["revisions"]] == [sc["id"]]
    assert m["revisions"][0]["edits"][0]["lever"] == "canopy" and m["revisions"][0]["content_hash"] == sc["content_hash"]
    # cosmetic patches keep the hash
    r = client.patch(f"/api/scenarios/{sc['id']}", json={"name": "Corridor 2", "tags": ["a", "b"], "notes": "n"})
    assert r.status_code == 200 and r.json()["doc"]["name"] == "Corridor 2"
    assert r.json()["content_hash"] == sc["content_hash"]
    rows = client.get(f"/api/projects/{pid}/scenarios", params={"tag": "a"}).json()
    assert [x["id"] for x in rows] == [sc["id"]] and rows[0]["latest"] is None
    assert client.get(f"/api/projects/{pid}/scenarios", params={"q": "nothing"}).json() == []
    r = client.patch(f"/api/scenarios/{sc['id']}", json={"archived": True})
    assert r.json()["status"] == "archived"
    assert client.get(f"/api/projects/{pid}/scenarios").json() == []
    assert len(client.get(f"/api/projects/{pid}/scenarios", params={"archived": True}).json()) == 1
    assert client.delete(f"/api/scenarios/{sc['id']}").json() == {"ok": True}
    assert not mirror.exists()


def test_patch_with_exact_result_conflicts_and_fork(client, ctx, demo, run_ctx):
    sc = _create(client, demo["id"])
    # editing before any exact result is fine
    doc2 = {**DOC, "edits": [{**DOC["edits"][0], "amount": 12}]}
    r = client.patch(f"/api/scenarios/{sc['id']}", json={"doc": doc2})
    assert r.status_code == 200 and r.json()["content_hash"] != sc["content_hash"]
    make_result(ctx, run_ctx, scenario={"id": sc["id"], "content_hash": r.json()["content_hash"]})
    from sparc.studio.scenarios import library

    library.sync_status(ctx.db, sc["id"])
    assert client.get(f"/api/scenarios/{sc['id']}").json()["status"] == "exact"
    doc3 = {**DOC, "edits": [{**DOC["edits"][0], "amount": 20}]}
    r = client.patch(f"/api/scenarios/{sc['id']}", json={"doc": doc3})
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "conflict_revision" and err["action"]["path"] == f"/api/scenarios/{sc['id']}/fork"
    assert client.patch(f"/api/scenarios/{sc['id']}", json={"name": "renamed"}).status_code == 200
    r = client.post(f"/api/scenarios/{sc['id']}/fork", json={"doc": doc3})
    assert r.status_code == 201
    fk = r.json()
    assert fk["revision"] == 2 and fk["parent_id"] == sc["id"] and fk["status"] == "draft"
    assert fk["doc"]["edits"][0]["amount"] == 20
    parent = client.get(f"/api/scenarios/{sc['id']}").json()
    assert parent["children"] == [fk["id"]] and len(parent["results"]) == 1
    # one mirror file holds the lineage
    m = json.loads((Path(demo["dir"]) / "scenarios" / f"{sc['id']}.json").read_text())
    assert [(r["id"], r["revision"], r["parent_id"]) for r in m["revisions"]] == [
        (sc["id"], 1, None), (fk["id"], 2, sc["id"])]
    r = client.delete(f"/api/scenarios/{sc['id']}")
    assert r.status_code == 409 and r.json()["error"]["code"] == "has_children"
    assert client.delete(f"/api/scenarios/{sc['id']}", params={"force": True, "results": True}).status_code == 200
    assert ctx.db.fetchval("SELECT COUNT(*) FROM results") == 0


def test_run_endpoint_checkpoint_cache_and_blocking(client, ctx, demo, run_ctx, synth_run):
    rid, rd = synth_run
    sc = _create(client, demo["id"])
    r = client.post(f"/api/scenarios/{sc['id']}/run", json={"run_id": rid})
    assert r.status_code == 409 and r.json()["error"]["code"] == "no_checkpoint"
    (rd / "checkpoint.pkl").write_bytes(b"not a real checkpoint")
    make_result(ctx, run_ctx, scenario={"id": sc["id"], "content_hash": sc["content_hash"]})
    r = client.post(f"/api/scenarios/{sc['id']}/run", json={"run_id": rid})
    assert r.status_code == 200, r.text
    assert r.json()["cached"]["scenario_id"] == sc["id"] and r.json()["job"] is None
    bad = _create(client, demo["id"], {"name": "bad", "edits": [{"lever": "elevation", "mode": "add", "amount": 1}]})
    r = client.post(f"/api/scenarios/{bad['id']}/run", json={"run_id": rid})
    assert r.status_code == 422 and r.json()["error"]["detail"]["warnings"][0]["code"] == "not_actionable"


def test_templates(client, ctx, demo, synth_run):
    from sparc.studio.errors import ApiError
    from sparc.studio.scenarios import templates

    rid, _ = synth_run
    listing = client.get("/api/scenario-templates").json()
    assert [t["id"] for t in listing] == ["cool_roofs", "shade_hottest", "fill_plantable", "depave",
                                          "footprint_priority", "around_sites", "configured_package"]
    params = {"around_sites": {"sites": [[-69.0, 41.0]]}}
    for t in listing:
        r = client.post(f"/api/projects/{demo['id']}/scenarios/from-template",
                        json={"template": t["id"], "params": params.get(t["id"], {}), "run_id": rid})
        assert r.status_code == 201, (t["id"], r.text)
        doc = r.json()["doc"]
        assert doc["anchor_run_id"] == rid and doc["edits"]
        comp = client.post(f"/api/runs/{rid}/compile", json={"scenario": doc})
        assert comp.status_code == 200, (t["id"], comp.text)
    pkg = client.get(f"/api/projects/{demo['id']}/scenarios", params={"tag": "template:configured_package"}).json()
    assert pkg[0]["name"] == "Cooling package"
    bare = SimpleNamespace(cfg_raw={"physics": {"roles": {}}}, grid=SimpleNamespace(crs=None), data=None, run_id="r")
    with pytest.raises(ApiError) as exc:
        templates.build(bare, "fill_plantable", {})
    assert exc.value.code == "template_unavailable" and exc.value.detail["missing"] == ["canopy_role", "layers"]


def test_ladder(client, demo, synth_run):
    rid, _ = synth_run
    sc = _create(client, demo["id"])
    r = client.post(f"/api/scenarios/{sc['id']}/ladder", json={"edit_index": 0, "amounts": [5, 10, 20], "run_id": rid})
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["job"] is None                      # the fixture run has no checkpoint
    assert [s["doc"]["edits"][0]["amount"] for s in out["scenarios"]] == [5, 10, 20]
    tags = {t for s in out["scenarios"] for t in s["doc"]["tags"] if t.startswith("ladder:")}
    assert len(tags) == 1 and all(s["parent_id"] == sc["id"] for s in out["scenarios"])
    bad = client.post(f"/api/scenarios/{sc['id']}/ladder", json={"edit_index": 3, "amounts": [1]})
    assert bad.status_code == 422


def test_promote(client, ctx, demo):
    pid = demo["id"]
    one = _create(client, pid, {"name": "More trees", "edits": [{"lever": "canopy", "mode": "add", "amount": 15}]})
    r = client.post(f"/api/scenarios/{one['id']}/promote", json={})
    out = r.json()
    assert out["eligible"] and out["names"] == ["More trees +15"] and out["version"] is None
    assert "More trees" in out["yaml_diff"]
    pkg = _create(client, pid, {"name": "Pkg", "edits": [{"lever": "canopy", "mode": "add", "amount": 5},
                                                         {"lever": "impervious", "mode": "add", "amount": -10}]})
    v0 = ctx.db.fetchval("SELECT MAX(version) FROM config_versions WHERE project_id = ?", (pid,)) or 0
    out = client.post(f"/api/scenarios/{pkg['id']}/promote", json={"apply": True}).json()
    assert out["eligible"] and out["names"] == ["Pkg"] and out["version"] > v0
    text = Path(demo["config_path"]).read_text()
    assert "Pkg" in text and "decrease" in text
    sel = _create(client, pid, DOC)
    out = client.post(f"/api/scenarios/{sel['id']}/promote", json={}).json()
    assert not out["eligible"] and "city-wide" in out["reason"]
    dec = _create(client, pid, {"name": "Less paving", "edits": [{"lever": "impervious", "mode": "add", "amount": -10}]})
    assert client.post(f"/api/scenarios/{dec['id']}/promote", json={}).json()["names"] == ["Less paving −10"]


def test_design_csv_round_trip(client, ctx, demo, run_ctx, synth_run):
    rid, _ = synth_run
    sc = _create(client, demo["id"])
    r = client.get(f"/api/scenarios/{sc['id']}/design.csv", params={"run_id": rid})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    lines = r.text.strip().splitlines()
    assert lines[0] == "id,lever,change" and len(lines) - 1 == int(round(0.2 * run_ctx.grid.n))
    r = client.post(f"/api/runs/{rid}/designs/import", content=r.text.encode(), headers={"Content-Type": "text/csv"})
    assert r.status_code == 201, r.text
    imp = r.json()
    assert imp["mode"] == "change" and imp["levers"] == ["canopy"] and imp["unknown_ids"] == []
    doc = {"name": "imported", "edits": [{"lever": "canopy", "mode": "per_cell",
                                          "per_cell_ref": f"blob:{imp['blobs']['canopy']}"}]}
    comp = client.post(f"/api/runs/{rid}/compile", json={"scenario": doc}).json()
    assert comp["levers"]["canopy"]["n_cells"] == len(lines) - 1
    assert comp["levers"]["canopy"]["mean_requested"] == pytest.approx(10.0, rel=1e-6)
    # absolute targets: change = value − current
    ids = np.asarray(run_ctx.grid.ids)
    x = run_ctx.data.frame["canopy"].to_numpy(float)
    body = f"id,lever,value\n{ids[0]},canopy,{x[0] + 4}\n{ids[1]},canopy,{x[1] + 6}\nghost,canopy,1\n"
    imp = client.post(f"/api/runs/{rid}/designs/import", content=body.encode()).json()
    assert imp["mode"] == "value" and imp["unknown_ids"] == ["ghost"] and imp["n_rows"] == 3
    raw = Path(ctx.db.fetchone("SELECT path FROM blobs WHERE id = ?", (imp["blobs"]["canopy"],))["path"]).read_bytes()
    np.testing.assert_allclose(np.frombuffer(raw[8:], dtype="<f4"), [4, 6], rtol=1e-5)


def test_design_csv_import_rejects_bad_cells(client, ctx, demo, run_ctx, synth_run):
    """A number column with text in it ("abc") is a 422 that names the row, not a 500; an id the run does not
    have ("inf", "ghost") is an unknown id; empty and NA cells are skipped."""
    rid, _ = synth_run
    ids = np.asarray(run_ctx.grid.ids)
    r = client.post(f"/api/runs/{rid}/designs/import", content=f"id,lever,change\n{ids[0]},canopy,2\nzz,canopy,abc\n"
                    .encode(), headers={"Content-Type": "text/csv"})
    assert r.status_code == 422, r.text
    err = r.json()["error"]
    assert err["code"] == "validation" and err["detail"]["errors"][0]["code"] == "number"
    assert "row 3" in err["detail"]["errors"][0]["message"] and "abc" in err["detail"]["errors"][0]["message"]
    for bad_id in ("inf", "-inf", "nan", "1e400"):
        body = f"id,lever,change\n{ids[0]},canopy,2\n{bad_id},canopy,1\n{ids[1]},canopy,N/A\n{ids[2]},canopy,\n"
        r = client.post(f"/api/runs/{rid}/designs/import", content=body.encode(), headers={"Content-Type": "text/csv"})
        assert r.status_code == 201, (bad_id, r.text)
        imp = r.json()
        assert imp["levers"] == ["canopy"] and len(imp["unknown_ids"]) == 1 and imp["n_rows"] == 4
        raw = Path(ctx.db.fetchone("SELECT path FROM blobs WHERE id = ?", (imp["blobs"]["canopy"],))["path"]).read_bytes()
        np.testing.assert_allclose(np.frombuffer(raw[4:], dtype="<f4"), [2.0])


def test_cache_hit_of_another_scenario_is_adopted(client, ctx, demo, run_ctx, synth_run):
    """Two scenarios with the same content share the cache key: the second one's "Run exact" is a cache hit that
    it gets as its own result (listed, exact, inspectable), and deleting the first keeps it."""
    rid, rd = synth_run
    (rd / "checkpoint.pkl").write_bytes(b"not a real checkpoint")
    a = _create(client, demo["id"])
    b = _create(client, demo["id"], {**DOC, "name": "Same content, other name"})
    assert a["content_hash"] == b["content_hash"]
    made = make_result(ctx, run_ctx, scenario={"id": a["id"], "content_hash": a["content_hash"]})
    r = client.post(f"/api/scenarios/{b['id']}/run", json={"run_id": rid})
    assert r.status_code == 200 and r.json()["job"] is None
    hit = r.json()["cached"]
    assert hit["scenario_id"] == b["id"] and hit["id"] != made["id"]
    got = client.get(f"/api/scenarios/{b['id']}").json()
    assert got["status"] == "exact" and [x["id"] for x in got["results"]] == [hit["id"]]
    res = client.get(f"/api/results/{hit['id']}").json()
    assert res["scenario"]["id"] == b["id"] and res["scenario"]["name"] == "Same content, other name"
    assert res["city"]["estimate"] == pytest.approx(client.get(f"/api/results/{made['id']}").json()["city"]["estimate"])
    # a second run of b hits its own copy; deleting a (and its results) leaves b's result in place
    assert client.post(f"/api/scenarios/{b['id']}/run", json={"run_id": rid}).json()["cached"]["id"] == hit["id"]
    assert client.delete(f"/api/scenarios/{a['id']}", params={"results": True}).status_code == 200
    assert client.get(f"/api/results/{hit['id']}").status_code == 200


def test_preview_marks_scenario_previewed(client, ctx, demo, run_ctx, synth_run, fake_emulator):
    rid, rd = synth_run
    fake_emulator(rd, run_ctx)
    sc = _create(client, demo["id"])
    r = client.post(f"/api/runs/{rid}/preview", json={"edits": DOC["edits"], "request_seq": 1, "scenario_id": sc["id"]})
    assert r.status_code == 200
    assert client.get(f"/api/scenarios/{sc['id']}").json()["status"] == "previewed"
    # The Lab previews an edit before autosaving it: saving the content just previewed keeps
    # "previewed" (a name change does too); saving content no preview showed is a draft again.
    edited = [{**DOC["edits"][0], "amount": 15}]
    r = client.post(f"/api/runs/{rid}/preview", json={"edits": edited, "request_seq": 2, "scenario_id": sc["id"]})
    assert r.status_code == 200
    s = client.patch(f"/api/scenarios/{sc['id']}", json={"doc": {**DOC, "name": "Corridor 15", "edits": edited}}).json()
    assert s["status"] == "previewed"
    s = client.patch(f"/api/scenarios/{sc['id']}", json={"doc": {**DOC, "edits": [{**edited[0], "amount": 25}]}}).json()
    assert s["status"] == "draft"


def test_deleting_a_run_drops_its_results_plans_and_comparisons(client, ctx, demo, run_ctx, synth_run):
    """``DELETE /api/runs/{rid}`` removes the run folder, and with it ``studio/results|plans|sweeps|comparisons|
    blobs``: their rows go too, so a scenario whose only exact result lived there is no longer ``exact`` (it can
    be edited again) and the plan and comparison of the deleted run answer 404 instead of serving its data."""
    rid, rd = synth_run
    sc = _create(client, demo["id"])
    other = _create(client, demo["id"], {**DOC, "name": "Untouched"})
    made = make_result(ctx, run_ctx, scenario={"id": sc["id"], "content_hash": sc["content_hash"]})
    from sparc.studio.scenarios import library

    library.sync_status(ctx.db, sc["id"])
    library.write_mirror(ctx.db, ctx.workspace, sc["id"])
    assert client.get(f"/api/scenarios/{sc['id']}").json()["status"] == "exact"
    plan = client.post(f"/api/runs/{rid}/plans", json={"params": {"lever": "canopy", "budget": 2000}, "name": "Trees",
                                                        "verify": False})
    assert plan.status_code == 201, plan.text
    plid = plan.json()["plan"]["id"]
    cmp_ = client.post(f"/api/runs/{rid}/compare", json={"items": [{"kind": "result", "id": made["id"]},
                                                                   {"kind": "baseline"}], "regions": []})
    assert cmp_.status_code == 201, cmp_.text
    region = client.post(f"/api/runs/{rid}/regions", json={"name": "Hot", "spec": {"kind": "filter", "column": "obs",
                                                                                  "op": ">", "value": 88.5}})
    assert region.status_code == 201, region.text

    assert client.delete(f"/api/runs/{rid}").status_code == 200 and not rd.exists()
    for table in ("results", "plans", "sweeps", "comparisons", "blobs", "regions"):
        assert ctx.db.fetchval(f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", (rid,)) == 0, table
    got = client.get(f"/api/scenarios/{sc['id']}").json()
    assert got["status"] == "draft" and got["results"] == []
    listed = {x["id"]: x for x in client.get(f"/api/projects/{demo['id']}/scenarios").json()}
    assert listed[sc["id"]]["latest"] is None and listed[sc["id"]]["status"] == "draft"
    assert listed[other["id"]]["status"] == "draft"
    mirror = json.loads((Path(demo["dir"]) / "scenarios" / f"{sc['id']}.json").read_text())
    assert mirror["revisions"][0]["status"] == "draft"
    doc2 = {**DOC, "edits": [{**DOC["edits"][0], "amount": 12}]}
    assert client.patch(f"/api/scenarios/{sc['id']}", json={"doc": doc2}).status_code == 200
    assert client.get(f"/api/plans/{plid}").status_code == 404
    assert client.get(f"/api/comparisons/{cmp_.json()['id']}").status_code == 404


def _finished_check(ctx, sid: str, result: dict, finished: str) -> dict:
    """A succeeded ``scenario.across_runs`` job row with ``result`` (what the heavy job leaves behind)."""
    from sparc.studio.workspace import new_id

    job = {"id": new_id("job"), "kind": "scenario.across_runs", "lane": "heavy", "executor": "process",
           "scenario_id": sid, "params_json": json.dumps({"scenario_id": sid, "run_ids": ["a", "b"]}),
           "status": "succeeded", "job_dir": "/nonexistent", "created_utc": finished, "finished_utc": finished,
           "result_json": json.dumps(result)}
    ctx.db.insert("jobs", job)
    return job


def test_across_runs_band_is_the_checked_contents_and_reaches_its_results(client, ctx, demo, run_ctx, synth_run):
    """The spread across runs is the specification band of the content it evaluated: an exact result of that
    content (served from the cache afterwards, never recomputed) takes it when the check finishes, and a draft
    edited after its check does not inherit it."""
    from sparc.studio.engine import kinds
    from sparc.studio.engine.compile import compile_scenario

    rid, _ = synth_run
    sc = _create(client, demo["id"])
    assert compile_scenario(run_ctx, sc["doc"], db=ctx.db).content_hash == sc["content_hash"]
    row = make_result(ctx, run_ctx, scenario=sc)              # exact first: no band yet
    before = client.get(f"/api/results/{row['id']}").json()["uncertainty"]
    assert before["specification"] is None and before["envelope_excludes_zero"] is True
    check = {"rows": [{"run_id": "a", "ok": True, "city": {"estimate": -0.9}, "error": None},
                      {"run_id": "b", "ok": True, "city": {"estimate": 0.2}, "error": None}],
             "sign_stability": 0.5, "spread": 1.1, "content_hash": sc["content_hash"]}
    job = _finished_check(ctx, sc["id"], check, "2026-10-05T01:00:00Z")
    kinds.across_on_finish(ctx, {**job, "params": json.loads(job["params_json"])}, check)
    after = client.get(f"/api/results/{row['id']}").json()["uncertainty"]
    assert after["specification"] == [-0.9, 0.2] and "specification: check across runs" in after["sources"]
    assert after["envelope"][0] <= -0.9 and after["envelope"][1] >= 0.2 and after["envelope_excludes_zero"] is False
    assert after["estimation_95"] == before["estimation_95"] and after["causal_band"] == before["causal_band"]
    assert kinds._specification(ctx, sc["id"], sc["content_hash"]) == [-0.9, 0.2]
    # a check of a draft that is edited afterwards is not the band of the new content
    draft = _create(client, demo["id"], {**DOC, "name": "Draft"})
    _finished_check(ctx, draft["id"], {**check, "content_hash": draft["content_hash"]}, "2026-10-05T02:00:00Z")
    edited = client.patch(f"/api/scenarios/{draft['id']}", json={"doc": {
        **DOC, "name": "Draft", "edits": [{**DOC["edits"][0], "amount": 30}]}}).json()
    assert edited["content_hash"] != draft["content_hash"]
    srow = ctx.db.fetchone("SELECT * FROM scenarios WHERE id = ?", (draft["id"],))
    item = kinds.compiled_item(ctx, run_ctx, srow, {"studio_dir": str(run_ctx.studio_dir)})
    assert item["specification"] is None
    assert kinds._specification(ctx, draft["id"], draft["content_hash"]) == [-0.9, 0.2]
    # a check recorded before checks named their content is not tied to any content
    legacy = _create(client, demo["id"], {**DOC, "name": "Legacy"})
    _finished_check(ctx, legacy["id"], {k: v for k, v in check.items() if k != "content_hash"}, "2026-10-05T03:00:00Z")
    assert kinds._specification(ctx, legacy["id"], legacy["content_hash"]) is None
