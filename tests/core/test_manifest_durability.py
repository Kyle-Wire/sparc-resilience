"""``_finish`` keeps post-run sections and adds the schema-2 fields (SPEC §11 item 9)."""

from __future__ import annotations

import copy
import json

import pytest

S0_S3 = ("S0", "S1", "S2", "S3")


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    from sparc.core.config import load_core_config
    from sparc.core.synthetic import write_demo_project

    demo = write_demo_project(tmp_path_factory.mktemp("durability"), n=24, seed=0)
    c = load_core_config(demo["config_path"])
    c.raw["models"] = {"ols": True, "mgwr": False, "gwrf": False, "gam": False, "physics": False}
    c.raw["stacker"].update(epochs=20, tune_lambda=[0.0])
    c.raw["cv"]["baselines"] = False
    return c


def _manifest(run_dir) -> dict:
    return json.loads((run_dir / "manifest.json").read_text())


def test_schema_2_fields(cfg, tmp_path):
    from sparc.core.pipeline import run_core

    res = run_core(copy.deepcopy(cfg), stages=S0_S3, fast=True, run_dir=tmp_path / "run",
                   run_meta={"studio_run_id": "r1", "project_id": "p1", "origin": "studio"})
    m = _manifest(res.run_dir)
    assert m["schema_version"] == 2
    assert m["stages_run"] == ["S0", "S1", "S2_S3"]
    assert m["run_meta"] == {"studio_run_id": "r1", "project_id": "p1", "origin": "studio"}
    td = m["timings_detail"]
    assert set(td["stages"]) == {"S0", "S1", "S2_S3"} and all(v >= 0 for v in td["stages"].values())
    assert set(td["fold_model_s"]) == {"ols"} and len(td["fold_model_s"]["ols"]) == 3
    assert {"base_models_s", "stackers_s", "stacker_candidates_s"} <= set(td["ensemble"])
    assert "physics_fit_s" not in td                                  # no physics model in this config
    assert m["timings_s"].keys() == {"S0", "S1", "S2_S3"}


def test_rerunning_a_subset_keeps_post_run_sections(cfg, tmp_path):
    from sparc.core import runio
    from sparc.core.pipeline import POST_RUN_KEYS, run_core

    run_dir = tmp_path / "run"
    run_core(copy.deepcopy(cfg), stages=S0_S3, fast=True, run_dir=run_dir)
    placebo = {"rows": [], "n_pass_model": 2, "n_pass_causal": 3, "n_placebos": 3, "coarse_m": 60.0,
               "layer_correlation_with_original": {}}
    post = {"planner": {"n_cells": 1}, "uncertainty": {"scenarios": []}, "placebo": placebo,
            "emulator": {"levers": {}}, "simcheck": {"n_rows": 2}, "multiverse": {"runs": {}}}
    runio.update_manifest(run_dir, post, source="test")
    runio.update_manifest(run_dir, {"baselines": {"verdict": "post-run", "rows": {}}}, source="baselines")
    before = _manifest(run_dir)

    res = run_core(copy.deepcopy(cfg), stages=S0_S3, fast=True, resume=True, run_dir=run_dir)
    m = _manifest(run_dir)
    for k, v in post.items():
        assert m[k] == v, k
    assert m["post_run"] == before["post_run"]
    assert set(POST_RUN_KEYS) <= set(m)
    assert m["baselines"] == {"verdict": "post-run", "rows": {}}       # not recomputed by this invocation
    assert m["created_utc"] >= before["created_utc"] and m["schema_version"] == 2
    assert res.manifest["planner"] == post["planner"]
    assert "2/3 pass for the model" in (run_dir / "methods.md").read_text()   # the writeup sees the merged manifest

    run_core(copy.deepcopy(cfg), stages=("S0", "S1"), resume=True, run_dir=run_dir)
    m = _manifest(run_dir)
    assert m["uncertainty"] == post["uncertainty"] and m["planner"] == post["planner"]
    assert m["stages_run"] == ["S0", "S1"]


def test_baselines_computed_in_run_replace_a_post_run_section(cfg, tmp_path):
    from sparc.core import runio
    from sparc.core.pipeline import run_core

    run_dir = tmp_path / "run"
    with_bl = copy.deepcopy(cfg)
    with_bl.raw["cv"]["baselines"] = ["idw"]
    run_core(copy.deepcopy(with_bl), stages=S0_S3, fast=True, run_dir=run_dir)
    runio.update_manifest(run_dir, {"baselines": {"verdict": "stale"}}, source="baselines")
    run_core(copy.deepcopy(with_bl), stages=S0_S3, fast=True, resume=True, run_dir=run_dir)
    m = _manifest(run_dir)
    assert m["baselines"]["verdict"] != "stale" and "idw" in m["baselines"]["rows"]

    # an in-run section (no post_run entry) is not carried into a run that does not compute baselines
    run_dir2 = tmp_path / "run2"
    run_core(copy.deepcopy(with_bl), stages=S0_S3, fast=True, run_dir=run_dir2)
    run_core(copy.deepcopy(cfg), stages=S0_S3, fast=True, run_dir=run_dir2)
    assert "baselines" not in _manifest(run_dir2)


def test_post_run_actions_record_their_sections_through_update_manifest(tmp_path):
    """baselines, emulator, planner and uncertainty edit the manifest only through ``runio.update_manifest``
    (``post_run[]`` history), write their files atomically and report them as artifacts."""
    from sparc.core import progress
    from sparc.core.baselines import baselines_for_run
    from sparc.core.config import load_core_config
    from sparc.core.emulator import emulator_for_run
    from sparc.core.pipeline import POST_RUN_KEYS, run_core
    from sparc.core.planner import planner_pack
    from sparc.core.synthetic import write_demo_project
    from sparc.core.uncertainty import uncertainty_report

    demo = write_demo_project(tmp_path / "project", n=32, seed=0)

    def light():
        c = load_core_config(demo["config_path"])
        c.raw["models"] = {"ols": True, "mgwr": False, "gwrf": False, "gam": True, "physics": False}
        c.raw["stacker"].update(epochs=20, tune_lambda=[0.0])
        c.raw["cv"]["baselines"] = False
        c.raw["causal"]["n_boot"] = 10
        c.raw["actionable"] = {"canopy": {**c.raw["actionable"]["canopy"], "doses": [0, 10, 20]}}
        c.raw["scenarios"] = c.raw["scenarios"][:1]
        c.raw["joint_scenarios"] = []
        return c

    run_dir = tmp_path / "run"
    run_core(light(), stages=("S0", "S1", "S2", "S3", "S4", "S5", "S7"), run_dir=run_dir)
    events: list[dict] = []
    progress.configure(events.append, heartbeat_s=0)
    try:
        bl = baselines_for_run(run_dir, light(), models=("idw",))
        em = emulator_for_run(run_dir, light(), n_patches=2)
        pl = planner_pack(run_dir, light(), export=False)
        un = uncertainty_report(run_dir)
    finally:
        progress.reset()
    m = _manifest(run_dir)
    history = [(e["section"], e["source"]) for e in m["post_run"]]
    assert history == [("baselines", "baselines"), ("emulator", "emulator"), ("planner", "planner"),
                       ("uncertainty", "uncertainty")]
    assert m["baselines"] == json.loads((run_dir / "baselines.json").read_text()) and bl["rows"]["idw"]
    assert set(m["emulator"]["levers"]) == {"canopy"} and m["emulator"]["files"] == ["emulator.npz", "emulator.json"]
    assert m["emulator"]["levers"]["canopy"]["patch_pass_rate"] == em["levers"]["canopy"]["validation"]["patch_pass_rate"]
    assert m["planner"]["n_cells"] == pl["n_cells"] and m["uncertainty"]["scenarios"] == un["scenarios"]
    arts = {e["path"] for e in events if e["type"] == "artifact"}
    assert {"baselines.json", "emulator.npz", "emulator.json", "planner/planner.json", "planner/hex_250m.csv",
            "planner/planner_cells.parquet", "uncertainty.json", "uncertainty.md", "manifest.json"} <= arts
    assert {e["name"] for e in events if e["type"] == "task.start"} >= {"load_run", "unpickle", "engine_init", "lever",
                                                                        "exposure", "hexagons", "baseline_model"}
    assert not list(run_dir.rglob("*.tmp"))

    # a resumed re-run of the fitting stages keeps every post-run section
    run_core(light(), stages=("S0", "S1", "S2", "S3", "S4", "S5", "S7"), run_dir=run_dir, resume=True)
    again = _manifest(run_dir)
    for k in ("baselines", "emulator", "planner", "uncertainty", "post_run"):
        assert again[k] == m[k], k
    assert set(POST_RUN_KEYS) - {"simcheck", "multiverse", "placebo"} <= set(again)
