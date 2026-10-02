"""Real placebo studies on the synthetic city (slow): ``study.placebo`` with ``kinds=["shift"]`` at a coarse
120 m cell refits the pipeline once, in a real worker, against the fixture run's launch snapshot.  The child
run registers with ``origin = study_child`` and its parent and study ids, the kind writes ``placebo.json`` /
``placebo.md``, and the study view carries the shift layer's verdict.  A second test runs the size of the
end-to-end check: the n=96 demo at 60 m.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_real_placebo_shift_registers_its_child_and_a_verdict(client, ctx, fixture_run, wait_job):
    rid, _ = fixture_run
    r = client.post(f"/api/runs/{rid}/studies/placebo", json={"kinds": ["shift"], "coarse_m": 120})
    assert r.status_code == 202, r.text
    study, job = r.json()["study"], r.json()["job"]
    j = wait_job(client, job["id"], timeout=900)
    assert j["status"] == "succeeded", j
    kids = ctx.db.fetchall("SELECT * FROM runs WHERE study_id = ?", (study["id"],))
    assert len(kids) == 1
    child = kids[0]
    assert child["origin"] == "study_child" and child["parent_run_id"] == rid and child["status"] == "complete"
    # followed by the study's job, never as a CLI run with a pseudo-job of its own
    assert ctx.db.fetchone("SELECT id FROM jobs WHERE kind = 'run.external' AND run_id = ?", (child["id"],)) is None
    assert Path(child["run_dir"]).parent == Path(study["out_dir"]) / "children"
    m = json.loads((Path(child["run_dir"]) / "manifest.json").read_text())
    assert m["run_meta"]["role"] == "placebo:shift" and m["run_meta"]["study_id"] == study["id"]
    sdir = Path(study["out_dir"])
    pz = json.loads((sdir / "placebo.json").read_text())
    assert pz["kinds"] == ["shift"] and pz["n_placebos"] >= 1 and (sdir / "placebo.md").is_file()
    view = client.get(f"/api/studies/{study['id']}/view").json()
    rows = [x for x in view["rows"] if x["placebo"]]
    assert rows and all(isinstance(x["verdict"], dict) and "model_pass" in x["verdict"] for x in rows)
    (c,) = view["children"]
    assert c["kind"] == "shift" and c["run_id"] == child["id"] and c["verdict"] in (
        "passes", "model fails", "causal fails", "fails")
    s = client.get(f"/api/studies/{study['id']}").json()
    assert s["status"] == "succeeded" and s["summary"]["n_placebos"] == pz["n_placebos"]
    assert [x["id"] for x in s["children"]] == [child["id"]]
    assert j["result"]["children"] == [child["id"]]


def test_real_placebo_on_the_n96_demo_at_60m(client, ctx, wait_job):
    """The size the end-to-end check uses (SPEC §14.5): the n=96 demo city at a 60 m coarse cell (≈1,600
    cells, so MGWR tuning runs), launched from a Studio run's snapshot of the demo config."""
    from sparc.studio.runs.launch import mode_args
    from tests.studio.runs.conftest import launch_snapshot

    r = client.post("/api/projects", json={"name": "Demo 96", "template": "synthetic_demo",
                                           "options": {"n": 96, "seed": 0}})
    assert r.status_code == 201, r.text
    demo = r.json()["project"]
    rid = "20261002-090000-fast-9696"
    rd = Path(demo["dir"]) / "runs" / rid
    (rd / "studio").mkdir(parents=True)
    (rd / "studio" / "launch.json").write_text(json.dumps(launch_snapshot(ctx, demo, rid, mode_args("fast"))))
    ctx.services["registry"].index_run_dir(rd, origin="studio", studio_dir=rd / "studio")
    r = client.post(f"/api/runs/{rid}/studies/placebo", json={"kinds": ["shift"], "coarse_m": 60})
    assert r.status_code == 202, r.text
    study, job = r.json()["study"], r.json()["job"]
    j = wait_job(client, job["id"], timeout=1500)
    assert j["status"] == "succeeded", j
    (child,) = ctx.db.fetchall("SELECT * FROM runs WHERE study_id = ?", (study["id"],))
    assert child["origin"] == "study_child" and child["parent_run_id"] == rid
    assert ctx.db.fetchone("SELECT id FROM jobs WHERE kind = 'run.external' AND run_id = ?", (child["id"],)) is None
    m = json.loads((Path(child["run_dir"]) / "manifest.json").read_text())
    assert (m["qa"]["coarse"] or {}).get("cell_m") == 60.0 and m["n_points"] > 1000
    view = client.get(f"/api/studies/{study['id']}/view").json()
    assert [c["kind"] for c in view["children"]] == ["shift"] and view["children"][0]["verdict"]
