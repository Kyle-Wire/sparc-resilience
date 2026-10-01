"""The tracker projection (jobs/tracker.py): golden replay, unit accounting, stage states, children."""

from __future__ import annotations

import json
import math

import pytest

from sparc.studio.jobs import tracker
from sparc.studio.jobs.tailer import iter_lines, parse_line

EXPECTED_TYPES = {"run.start", "run.plan", "run.dir", "stage.start", "stage.end", "stage.skip", "task.start",
                  "task.end", "tick", "metric", "artifact", "checkpoint", "warning", "log", "run.end", "job.status",
                  "job.result"}


def _replay(path):
    st = tracker.new_state()
    progress = []
    for cursor, raw in iter_lines(path):
        tracker.reduce(st, parse_line(raw), cursor)
        progress.append(st["progress"])
    return st, progress


def _close(a, b, path="$"):
    if isinstance(a, float) or isinstance(b, float):
        assert isinstance(a, (int, float)) and isinstance(b, (int, float)), f"{path}: {a!r} vs {b!r}"
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), f"{path}: {a!r} vs {b!r}"
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f"{path}: keys {sorted(set(a) ^ set(b))}"
        for k in a:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def test_fixture_is_schema_complete(selftest_events):
    types = {ev["type"] for _, ev in selftest_events}
    assert EXPECTED_TYPES <= types
    plan = next(ev for _, ev in selftest_events if ev["type"] == "run.plan")
    assert {n["state"] for n in plan["nodes"]} == {"will_run", "skipped"}
    assert any(ev["type"] == "tick" and ev.get("pass_s") for _, ev in selftest_events)
    assert any(ev["type"] == "metric" and ev.get("tags") for _, ev in selftest_events)


def test_replay_matches_golden_projection(fixtures_dir):
    st, _ = _replay(fixtures_dir / "selftest_events.jsonl")
    golden = json.loads((fixtures_dir / "selftest_projection.golden.json").read_text())
    _close(json.loads(json.dumps(st)), golden["state"])
    _close(tracker.to_contract(st), golden["contract"])


def test_progress_is_monotonic_and_completes(fixtures_dir):
    st, progress = _replay(fixtures_dir / "selftest_events.jsonl")
    vals = [p or 0.0 for p in progress]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
    assert st["progress"] == 1.0
    assert st["status"] == "succeeded" and st["run_status"] == "succeeded"


def test_contract_shape(fixtures_dir):
    st, _ = _replay(fixtures_dir / "selftest_events.jsonl")
    c = tracker.to_contract(st)
    assert c["stages"]["S2_S3"] == {"state": "done", "reason": None}
    assert c["stages"]["baselines"]["state"] == "disabled"
    assert c["stages"]["cv_curve"]["state"] == "not_requested"
    assert c["stages"]["S6"] == {"state": "skipped", "reason": "no_treatments"}
    assert c["done_units"]["base_fit:ols"] == 3 and c["done_units"]["engine_pass"] == 6
    assert {"code": "qa.coarse", "count": 2} in c["warnings"]
    assert c["artifacts"] == sorted(c["artifacts"]) and "predictions.csv" in c["artifacts"]


def test_snapshot_parts(fixtures_dir):
    st, _ = _replay(fixtures_dir / "selftest_events.jsonl")
    snap = tracker.snapshot_parts(st)
    assert snap["cursor"] == st["cursor"] > 0
    assert all(sp["status"] == "ok" for sp in snap["spans"])
    assert "influence.range_m{predictor=Pct_Canopy}" in snap["metrics_latest"]
    assert len(snap["metric_series"]["candidate_rmse{candidate=mean}"]) == 1
    assert snap["checkpoints"][0]["done"] == ["S3"]


# ---------------------------------------------------------------------------
# rules on hand-written events
# ---------------------------------------------------------------------------

def ev(type_, path=("run:r",), span=None, **fields):
    return {"v": 1, "type": type_, "seq": 0, "ts": fields.pop("ts", 100.0), "pid": 1, "job": "j", "lvl": "info",
            "span": span, "parent": fields.pop("parent", None), "path": list(path), "ctx": fields.pop("ctx", {}),
            **fields}


def _run(events):
    st = tracker.new_state()
    for i, e in enumerate(events):
        tracker.reduce(st, e, i * 100)
    return st


PLAN = ev("run.plan", nodes=[
    {"id": "S0", "label": "S0", "state": "will_run", "units": {"s0_load": 1}},
    {"id": "S2_S3", "label": "S2", "state": "will_run", "units": {"base_fit:ols": 2, "checkpoint_save": 1}},
    {"id": "S4", "label": "S4", "state": "will_run", "units": {"engine_pass": 2}},
    {"id": "S6", "label": "S6", "state": "skipped", "reason": "no_treatments", "units": {}},
], total_units={}, n_points=10)


def test_unit_accounting_rules():
    p = ("run:r", "stage:S2_S3", "task:fold[1/2]")
    st = _run([
        ev("run.start", span="1:1", name="r"), PLAN,
        ev("stage.start", ("run:r", "stage:S0"), "1:2", stage="S0"),
        ev("stage.end", ("run:r", "stage:S0"), "1:2", stage="S0", status="ok", elapsed_s=0.3),
        ev("stage.start", ("run:r", "stage:S2_S3"), "1:3", stage="S2_S3"),
        ev("task.start", p, "1:4", name="base_model", key="ols", unit="base_fit:ols"),
        ev("task.end", p, "1:4", name="base_model", key="ols", unit="base_fit:ols", status="ok", elapsed_s=2.0),
        ev("task.start", p, "1:5", name="base_model", key="ols", unit="base_fit:ols"),
        ev("task.end", p, "1:5", name="base_model", key="ols", unit="base_fit:ols", status="error", elapsed_s=1.0),
        ev("checkpoint", ("run:r", "stage:S2_S3"), "1:3", action="saved", done=["S3"], elapsed_s=0.5),
    ])
    assert st["done_units"] == {"s0_load": 1, "base_fit:ols": 1, "checkpoint_save": 1}   # the error does not count
    assert st["unit_obs"]["base_fit:ols"] == [2.0]
    assert st["stage_done"]["S2_S3"] == {"base_fit:ols": 1, "checkpoint_save": 1}
    w = {u: tracker.unit_weight(u) for u in ("s0_load", "base_fit:ols", "checkpoint_save", "engine_pass")}
    total = w["s0_load"] + 2 * w["base_fit:ols"] + w["checkpoint_save"] + 2 * w["engine_pass"]
    done = w["s0_load"] + w["base_fit:ols"] + w["checkpoint_save"]
    assert st["progress"] == pytest.approx(done / total)
    assert st["stages"]["S6"]["state"] == "skipped"


def test_ticks_partial_and_k_equals_n():
    p = ("run:r", "stage:S4", "task:dose[1/1]")
    events = [ev("run.start", span="1:1", name="r"), PLAN,
              ev("stage.start", ("run:r", "stage:S4"), "1:2", stage="S4"),
              ev("task.start", p, "1:3", name="dose", k=1, n=1)]
    st = _run(events + [ev("tick", p, "1:3", k=1, n=4, unit="engine_pass", frac=0.25)])
    assert st["done_units"] == {}
    assert st["stage_partial"]["S4"] == {"engine_pass": 0.25}
    assert st["stages"]["S4"]["progress"] == pytest.approx(0.125)
    st = _run(events + [ev("tick", p, "1:3", k=4, n=4, unit="engine_pass", frac=1.0, pass_s=13.0),
                        ev("tick", p, "1:3", k=1, n=4, unit="engine_pass", frac=0.25)])
    assert st["done_units"] == {"engine_pass": 1}
    assert st["unit_obs"]["engine_pass"] == [13.0]
    assert st["stages"]["S4"]["progress"] == pytest.approx(0.625)
    # the span ending clears its partial
    st = _run(events + [ev("tick", p, "1:3", k=2, n=4, unit="engine_pass"),
                        ev("task.end", p, "1:3", name="dose", k=1, n=1, status="ok")])
    assert st["partial"] == {} and st["stage_partial"] == {}


def test_stage_states_and_finish():
    st = _run([ev("run.start", span="1:1", name="r"), PLAN,
               ev("stage.skip", stage="S0", reason="checkpoint"),
               ev("stage.start", ("run:r", "stage:S2_S3"), "1:2", stage="S2_S3"),
               ev("run.end", span="1:1", status="cancelled", elapsed_s=3.0, timings_s={}, done=[])])
    s = st["stages"]
    assert s["S0"] == {**s["S0"], "state": "cached", "reason": "checkpoint"}
    assert s["S2_S3"]["state"] == "cancelled"
    assert s["S4"]["state"] == "not_reached"
    assert s["S6"]["state"] == "skipped"


def test_replanned_stage_is_not_rewound():
    st = _run([ev("run.start", span="1:1", name="r"), PLAN,
               ev("stage.start", ("run:r", "stage:S0"), "1:2", stage="S0"),
               ev("stage.end", ("run:r", "stage:S0"), "1:2", stage="S0", status="ok", elapsed_s=0.3),
               PLAN])
    assert st["stages"]["S0"]["state"] == "done"
    assert st["stages"]["S2_S3"]["state"] == "planned"


def test_nested_runs_feed_children_not_the_rail():
    t = ("task:placebo_kind[shift]",)
    child = t + ("run:city_placebo_shift",)
    st = _run([
        ev("task.start", t, "1:1", name="placebo_kind", key="shift", ctx={"placebo": "shift"}),
        ev("run.start", child, "1:2", name="city_placebo_shift", ctx={"placebo": "shift"},
           run_meta={"studio_run_id": "r-child", "role": "placebo:shift"}),
        ev("run.plan", child, "1:2", nodes=[{"id": "S0", "label": "S0", "state": "will_run",
                                              "units": {"s0_load": 1}}], total_units={}),
        ev("stage.start", child + ("stage:S0",), "1:3", stage="S0"),
        ev("stage.end", child + ("stage:S0",), "1:3", stage="S0", status="ok", elapsed_s=1.0),
        ev("warning", child + ("stage:S0",), "1:3", code="qa.coarse", message="m", data={}),
        ev("run.end", child, "1:2", status="succeeded", elapsed_s=1.0, timings_s={}, done=[]),
        ev("task.end", t, "1:1", name="placebo_kind", key="shift", status="ok", elapsed_s=2.0),
    ])
    assert st["stages"] is None and st["plan"] is None and st["done_units"] == {}
    rows = tracker.children_rows(st)
    assert len(rows) == 1
    assert rows[0]["key"] == "placebo:shift" and rows[0]["run_id"] == "r-child"
    assert rows[0]["status"] == "succeeded" and rows[0]["progress"] == 1.0
    assert st["progress"] == 1.0
    assert st["warnings"]                                   # child warnings are listed with the job's
    assert "1:2" in st["spans"]                             # and child spans are in the shared tree


def test_unknown_and_invalid_events_become_logs():
    st = tracker.new_state()
    e1 = parse_line(b'{"v":1,"type":"mystery","ts":1.0,"path":[],"ctx":{}}')
    e2 = parse_line(b'{"v":1,"type":"tick","ts":1.0,"path":[],"ctx":{},"k":"x"}')
    e3 = parse_line(b"not json")
    for e in (e1, e2, e3):
        assert e["type"] == "log" and e["lvl"] == "warning"
        tracker.reduce(st, e, 0)
    assert e1["orig_type"] == "mystery" and "tick" in e2["msg"]


def test_metric_key_format():
    assert tracker.metric_key("stacker_r2", {}) == "stacker_r2"
    assert tracker.metric_key("candidate_rmse", {"candidate": 0.1}) == "candidate_rmse{candidate=0.1}"
    assert tracker.metric_key("x", {"b": 2.0, "a": "z"}) == "x{a=z,b=2}"


def test_spans_collapse_beyond_limit():
    events = [ev("run.start", span="1:0", name="r")]
    for k in range(30):
        events.append(ev("task.start", ("run:r", f"task:fold[{k}]"), f"1:f{k}", name="fold", k=k, parent="1:0"))
        for j in range(80):
            p = ("run:r", f"task:fold[{k}]", "task:step")
            events.append(ev("task.start", p, f"1:{k}:{j}", name="step", parent=f"1:f{k}"))
            events.append(ev("task.end", p, f"1:{k}:{j}", name="step", status="ok", elapsed_s=0.5, parent=f"1:f{k}"))
    st = _run(events)
    rows = tracker.spans_list(st)
    assert len(rows) < tracker.MAX_SPANS
    agg = [r for r in rows if r["metrics"].get("aggregated")]
    assert len(agg) == 30 and all(r["metrics"]["aggregated"] == 80 for r in agg)
    assert agg[0]["elapsed_s"] == pytest.approx(40.0)
    assert len(tracker.spans_list(st, under="1:f3", max_depth=1)) == 81


def test_synth_run_fixture_replays_to_done(synth_run_dir):
    st, progress = _replay(synth_run_dir / "events.jsonl")
    vals = [p or 0.0 for p in progress]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))
    assert st["run_status"] == "succeeded" and st["progress"] == 1.0
    states = {s["state"] for s in st["stages"].values()}
    assert states <= {"done", "skipped", "cached", "disabled", "not_requested"}


def test_snapshot_plan_is_always_wire_valid():
    """A ``run.plan`` node without a label, with an unknown state or a non-stage id never breaks the tracker
    endpoint: the snapshot labels it with its id, counts an unknown state as skipped, drops a foreign id."""
    from sparc.studio.schemas.common import PlanNode

    st = _run([ev("run.plan", nodes=[
        {"id": "S0", "units": {"s0_load": 1}},
        {"id": "replicates", "label": "Replicates", "state": "will_run", "units": {"replicate:null": 4}},
        {"id": "S4", "label": "Response", "state": "weird", "units": {}},
    ], total_units={})])
    plan = tracker.snapshot_parts(st)["plan"]
    assert [n["id"] for n in plan] == ["S0", "S4"]
    assert plan[0]["label"] == "S0" and plan[0]["state"] == "will_run" and plan[1]["state"] == "skipped"
    for node in plan:
        PlanNode.model_validate(node)


def test_snapshots_of_a_core_run_are_wire_valid(synth_run_dir):
    """Every 20th prefix of the recorded core run gives a ``TrackerSnapshot`` the response model accepts."""
    from sparc.studio.jobs.eta import projection_eta
    from sparc.studio.schemas.common import TrackerSnapshot

    job = {"id": "j_x", "kind": "run.core", "lane": "heavy", "executor": "process", "label": "Run",
           "status": "running", "created_utc": "2026-10-01T00:00:00Z"}
    st = tracker.new_state()
    checked = 0
    for i, (cursor, raw) in enumerate(iter_lines(synth_run_dir / "events.jsonl")):
        tracker.reduce(st, parse_line(raw), cursor)
        if i % 20 == 0:
            parts = tracker.snapshot_parts(st, eta=projection_eta(st))
            TrackerSnapshot.model_validate({"job": job, **parts, "resources": []})
            checked += 1
    snap = TrackerSnapshot.model_validate({"job": job, **tracker.snapshot_parts(st), "resources": []})
    assert checked > 5 and snap.plan and {n.id for n in snap.plan} <= set(tracker.STAGE_IDS)
