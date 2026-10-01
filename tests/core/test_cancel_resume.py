"""Cancel at safe points, then resume (SPEC §5.9, §14.1).

A callable sink calls ``progress.request_cancel()`` when it sees the trigger
event; the run stops at the next ``check_cancel()`` with ``Cancelled``,
``run_state.json`` says ``cancelled`` and ``checkpoint.json`` agrees with the
pickle.  ``resume=True`` then completes with the same numbers as an
uninterrupted run (both single-threaded, fixed seeds).
"""

from __future__ import annotations

import copy
import json
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

TRIGGERS = {
    "s1_end": lambda e: e["type"] == "stage.end" and e["stage"] == "S1",
    "fold1_mgwr_tick": lambda e: (e["type"] == "tick" and e.get("unit") == "mgwr_score"
                                  and "task:fold[1/3]" in e["path"] and "task:base_model[mgwr]" in e["path"]),
    "stacker_candidate_2": lambda e: e["type"] == "task.start" and e["name"] == "stacker_candidate" and e["k"] == 2,
    "s4_dose_2": lambda e: e["type"] == "task.start" and e["name"] == "dose" and e["k"] == 2,
    "s6_treatment_1": lambda e: e["type"] == "task.start" and e["name"] == "treatment" and e["k"] == 1,
}
SECTIONS = ("metrics", "scenarios", "response", "causal", "optimize", "baselines", "climate", "lambda_scores", "cv")


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    from sparc.core.synthetic import write_demo_project

    return write_demo_project(tmp_path_factory.mktemp("cancel_resume"), n=40, seed=0)


def _cfg(demo):
    """MGWR stays in (its tuning ticks are a cancel point); GWRF and physics are left out to keep this quick."""
    from sparc.core.config import load_core_config

    cfg = load_core_config(demo["config_path"])
    cfg.raw["models"] = {"ols": True, "mgwr": True, "gwrf": False, "gam": True, "physics": False}
    cfg.raw["stacker"].update(epochs=30, tune_lambda=[0.0])
    cfg.raw["cv"]["baselines"] = ["idw"]
    cfg.raw["causal"]["n_boot"] = 20
    for v, d in (("canopy", [0, 10, 20]), ("impervious", [0, 10]), ("albedo", [0, 0.1])):
        cfg.raw["actionable"][v]["doses"] = d
    return cfg


def _run(cfg, run_dir, events, level="info", trigger=None, resume=False):
    from sparc.core import progress
    from sparc.core.pipeline import run_core

    fired = []

    def sink(ev):
        events.append(ev)
        if trigger is not None and not fired and trigger(ev):
            fired.append(ev)
            progress.request_cancel()

    progress.configure(sink, level=level, heartbeat_s=0)
    try:
        with progress.limit_threads(1):
            return run_core(cfg, fast=True, run_dir=run_dir, resume=resume)
    finally:
        progress.reset()


def _numbers(obj, prefix=""):
    """Flattened numeric leaves of a manifest section."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _numbers(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _numbers(v, f"{prefix}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, float(obj)


def _assert_same_numbers(a: dict, b: dict):
    for sec in SECTIONS:
        na, nb = dict(_numbers(a.get(sec), sec)), dict(_numbers(b.get(sec), sec))
        assert na.keys() == nb.keys(), sec
        for k, v in na.items():
            w = nb[k]
            assert (math.isnan(v) and math.isnan(w)) or abs(v - w) <= 1e-9, (k, v, w)


@pytest.fixture(scope="module")
def reference(demo, tmp_path_factory):
    events: list[dict] = []
    res = _run(_cfg(demo), tmp_path_factory.mktemp("reference") / "run", events)
    return res.manifest


@pytest.mark.parametrize("point", list(TRIGGERS))
def test_cancel_then_resume(demo, reference, tmp_path, point):
    from sparc.core import progress
    from sparc.core.pipeline import CHECKPOINT, CHECKPOINT_JSON, RUN_STATE

    run_dir = tmp_path / "run"
    events: list[dict] = []
    level = "debug" if point == "fold1_mgwr_tick" else "info"
    with pytest.raises(progress.Cancelled):
        _run(_cfg(demo), run_dir, events, level=level, trigger=TRIGGERS[point])
    assert any(TRIGGERS[point](e) for e in events), "the trigger event never came"
    end = [e for e in events if e["type"] == "run.end"]
    assert len(end) == 1 and end[0]["status"] == "cancelled"
    assert any(e["type"] == "cancel.ack" for e in events)
    rs = json.loads((run_dir / RUN_STATE).read_text())
    assert rs["status"] == "cancelled" and rs["error"] is None
    if (run_dir / CHECKPOINT).exists():
        side = json.loads((run_dir / CHECKPOINT_JSON).read_text())
        with open(run_dir / CHECKPOINT, "rb") as fh:
            state = pickle.load(fh)
        assert side["done"] == sorted(state["done"]) == rs["done"]
        assert side["fingerprint"] == state["fingerprint"] == rs["fingerprint"]
    else:
        assert not (run_dir / CHECKPOINT_JSON).exists() and rs["done"] == []
    expected_done = {"s1_end": [], "fold1_mgwr_tick": [], "stacker_candidate_2": [], "s4_dose_2": ["S3", "baselines"],
                     "s6_treatment_1": ["S3", "S4", "S5", "baselines", "climate"]}[point]
    assert rs["done"] == expected_done
    stopped = {"s1_end": "S2_S3", "fold1_mgwr_tick": "S2_S3", "stacker_candidate_2": "S2_S3", "s4_dose_2": "S4",
               "s6_treatment_1": "S6"}[point]
    if point != "s1_end":                                    # S1 end: cancelled before the next stage started
        assert [e for e in events if e["type"] == "stage.end"][-1] == next(
            e for e in events if e["type"] == "stage.end" and e["stage"] == stopped)
        assert next(e for e in events if e["type"] == "stage.end" and e["stage"] == stopped)["status"] == "cancelled"

    events2: list[dict] = []
    res = _run(_cfg(demo), run_dir, events2, resume=True)
    rs = json.loads((run_dir / RUN_STATE).read_text())
    assert rs["status"] == "succeeded"
    skipped = {e["stage"] for e in events2 if e["type"] == "stage.skip" and e["reason"] == "checkpoint"}
    cached = {"S3": {"S1", "S2_S3"}, "baselines": {"baselines"}, "S4": {"S4"}, "S5": {"S5"}, "climate": {"climate"}}
    assert skipped == set().union(*(cached[k] for k in expected_done)) if expected_done else not skipped
    if expected_done:
        assert [e["action"] for e in events2 if e["type"] == "checkpoint"][0] == "loaded"
    _assert_same_numbers(reference, res.manifest)


def test_cli_exits_130_and_records_the_cancel(demo, tmp_path):
    """The CLI maps ``Cancelled`` to exit 130; a pre-existing cancel file stops the run at its first safe point."""
    cancel = tmp_path / "cancel"
    cancel.touch()
    events = tmp_path / "events.jsonl"
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPARC_")}
    env.update(PYTHONPATH=str(REPO), OMP_NUM_THREADS="1", SPARC_CANCEL_FILE=str(cancel))
    out = subprocess.run([sys.executable, "-m", "sparc.core", "run", "--project", str(demo["config_path"]),
                          "--stages", "S0,S1", "--progress", str(events), "--job-id", "j_cancel", "--quiet"],
                         cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 130, out.stderr[-2000:]
    run_dir = Path(demo["config_path"]).parent / "runs" / "synthetic_demo"
    rs = json.loads((run_dir / "run_state.json").read_text())
    assert rs["status"] == "cancelled" and rs["job"] == "j_cancel" and rs["events_path"] == str(events.resolve())
    evs = [json.loads(line) for line in events.read_text().splitlines()]
    kinds = [e["type"] for e in evs]
    assert "cancel.ack" in kinds and evs[kinds.index("run.end")]["status"] == "cancelled"
    assert not any(e["type"] == "stage.start" for e in evs)          # cancelled before S0 began
    assert copy.deepcopy(rs["done"]) == []
