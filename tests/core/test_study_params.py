"""Study parameters Studio relies on (SPEC §11 item 13): placebo ``children_dir`` / ``resume`` / ``run_meta``,
multiverse ``extra_variants`` / ``run_meta``, reproduce ``out_dir`` / ``run_meta``."""

from __future__ import annotations

import json
import os

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    from sparc.core.synthetic import write_demo_project

    return write_demo_project(tmp_path_factory.mktemp("study_params"), n=32, seed=0)


def _light(demo, out_dir=None):
    from sparc.core.config import load_core_config

    cfg = load_core_config(demo["config_path"])
    cfg.raw["models"] = {"ols": True, "mgwr": False, "gwrf": False, "gam": True, "physics": False}
    cfg.raw["stacker"].update(epochs=20, tune_lambda=[0.0])
    cfg.raw["cv"]["baselines"] = False
    cfg.raw["causal"]["n_boot"] = 10
    cfg.raw["actionable"] = {"canopy": {**cfg.raw["actionable"]["canopy"], "doses": [0, 5, 10, 20, 30]}}
    cfg.raw["scenarios"] = cfg.raw["scenarios"][:1]
    cfg.raw["joint_scenarios"] = []
    if out_dir is not None:
        cfg.raw["output"]["dir"] = str(out_dir)
    return cfg


def _manifest(run_dir) -> dict:
    return json.loads((run_dir / "manifest.json").read_text())


def test_placebo_children_dir_resume_and_run_meta(demo, tmp_path):
    from sparc.core import progress
    from sparc.core.placebo import child_run_dir, run_placebo_suite

    children = tmp_path / "studies" / "s_1" / "children"
    meta = {"study_id": "s_1", "project_id": "p"}
    res = run_placebo_suite(_light(demo), kinds=("shift",), coarse=60.0, children_dir=children, run_meta=meta)
    child = child_run_dir(children, "synthetic_demo", "shift", 60.0)
    assert child == children / "synthetic_demo_placebo_shift_coarse60"
    assert res["runs"]["shift"]["run_dir"] == str(child)
    m = _manifest(child)
    assert m["run_meta"] == {**meta, "role": "placebo:shift"}
    assert (child / "input_frame.parquet").exists() and m["provenance"]["input_frame"] == "input_frame.parquet"

    events: list[dict] = []
    progress.configure(events.append, heartbeat_s=0)
    try:
        again = run_placebo_suite(_light(demo), kinds=("shift",), coarse=60.0, children_dir=children, resume=True,
                                  run_meta=meta)
    finally:
        progress.reset()
    assert [e["action"] for e in events if e["type"] == "checkpoint"][:1] == ["loaded"]
    cached = {e["stage"] for e in events if e["type"] == "stage.skip" and e["reason"] == "checkpoint"}
    assert {"S2_S3", "S4", "S5", "S6"} <= cached
    assert again["rows"] == res["rows"]
    task = next(e for e in events if e["type"] == "task.start" and e["name"] == "placebo_kind")
    assert task["key"] == "shift" and task["ctx"] == {"placebo": "shift"}
    nested = next(e for e in events if e["type"] == "run.start")
    assert nested["path"][0] == "task:placebo_kind[shift]" and nested["ctx"] == {"placebo": "shift"}
    assert nested["run_meta"]["role"] == "placebo:shift"


def test_multiverse_extra_variants_and_run_meta(demo, tmp_path):
    from sparc.core import progress
    from sparc.core.multiverse import run_multiverse

    sink = tmp_path / "events.jsonl"
    progress.configure(str(sink), heartbeat_s=0)
    try:
        cfg = _light(demo, out_dir=tmp_path / "children")
        summ = run_multiverse(cfg, tmp_path / "mv", variants=["baseline"], coarse=60.0, workers=1,
                              extra_variants={"no_gam": {"models.gam": False}}, run_meta={"study_id": "m_1"})
    finally:
        progress.reset()
    assert set(summ["runs"]) == {"baseline", "no_gam"}
    out = json.loads((tmp_path / "mv" / "no_gam.json").read_text())
    assert out["changes"] == {"models.gam": False}
    child = tmp_path / "children" / "synthetic_demo_mv_no_gam_coarse60"
    m = _manifest(child)
    assert m["run_meta"] == {"study_id": "m_1", "role": "variant:no_gam"}
    assert m["config"]["models"]["gam"] is False
    evs = [json.loads(line) for line in sink.read_text().splitlines()]
    variants = [e for e in evs if e["type"] == "task.end" and e["name"] == "variant"]
    assert {(e["key"], e["unit"], e["status"]) for e in variants} == {("baseline", "variant:baseline", "ok"),
                                                                      ("no_gam", "variant:no_gam", "ok")}
    assert all(e["ctx"].get("variant") == e["key"] for e in variants)   # reported from the pool worker
    assert os.getpid() not in {e["pid"] for e in variants}
    assert os.getpid() in {e["pid"] for e in evs if e["type"] == "tick"}       # the parent's progress ticks
    ticks = [e for e in evs if e["type"] == "tick" and e["unit"] == "variants"]
    assert ticks[-1]["k"] == ticks[-1]["n"] == 2
    with pytest.raises(ValueError, match="built-in"):
        run_multiverse(cfg, tmp_path / "mv2", variants=["baseline"], extra_variants={"no_gwrf": {}})


def test_reproduce_out_dir_and_run_meta(demo, tmp_path):
    from sparc.core.pipeline import run_core
    from sparc.core.reproduce import reproduce

    original = run_core(_light(demo), stages=("S0", "S1", "S2", "S3"), run_dir=tmp_path / "runs" / "original")
    out_dir = tmp_path / "studies" / "r_1" / "reproduction"
    res = reproduce(original.run_dir, stages=("S0", "S1", "S2", "S3"), out_dir=out_dir,
                    run_meta={"study_id": "r_1", "parent_run_id": "original"})
    assert res["pass"] and res["reproduction"] == str(out_dir)
    assert (out_dir / "reproduce.json").exists() and (out_dir / "manifest.json").exists()
    assert _manifest(out_dir)["run_meta"] == {"study_id": "r_1", "parent_run_id": "original", "role": "reproduction"}
    assert json.loads((out_dir / "reproduce.json").read_text())["pass"] is True


def test_simcheck_replicates_report_from_the_pool(demo, tmp_path):
    """``run_simcheck`` is unchanged in what it computes; its pool workers report replicate tasks into the
    parent's sink and the parent emits share / oof_r2 metrics and a replicate tick."""
    from sparc.core import progress
    from sparc.core.simcheck import run_simcheck

    sink = tmp_path / "events.jsonl"
    progress.configure(str(sink), heartbeat_s=0)
    try:
        summ = run_simcheck(_light(demo), {"null": 1}, tmp_path / "sc", coarse=90.0, epochs=20, workers=1,
                            product="direct")
    finally:
        progress.reset()
    assert summ["n_rows"] == 1 and summ["n_errors"] == 0
    assert (tmp_path / "sc" / "simcheck_summary.json").exists()
    evs = [json.loads(line) for line in sink.read_text().splitlines()]
    rep = [e for e in evs if e["type"] == "task.end" and e["name"] == "replicate"]
    assert [(e["key"], e["unit"], e["status"]) for e in rep] == [("null/0", "replicate:null", "ok")]
    assert rep[0]["ctx"] == {"generator": "null", "seed": 0} and rep[0]["pid"] != os.getpid()
    nested = [e for e in evs if e["type"] == "run.start"]
    assert nested and nested[0]["path"][0] == "task:replicate[null/0]"
    assert {e["name"] for e in evs if e["type"] == "metric" and e["pid"] == os.getpid()} >= {"share", "oof_r2"}
    tick = [e for e in evs if e["type"] == "tick" and e["unit"] == "replicates"]
    assert tick[-1]["k"] == tick[-1]["n"] == 1
    again = run_simcheck(_light(demo), {"null": 1}, tmp_path / "sc", coarse=90.0, epochs=20, product="direct")
    assert again == summ                                                # resumed: nothing re-run
