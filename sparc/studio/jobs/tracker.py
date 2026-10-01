"""The tracker projection: ``reduce(state, event)`` over a job's ``events.jsonl`` (SPEC §5.4–5.6, §5.10).

The state is a plain JSON-compatible dict built only from events, so the
same reducer serves the live tailer, ``GET /api/jobs/{jid}/tracker`` for
finished jobs, reindexing, and the cross-language contract with the web
client's ``applyEvent`` (:func:`to_contract`).

**Ownership.**  Events whose span ancestry contains a ``run:`` element below
the top level belong to a *nested* run (a study child, SPEC §5.5).  They feed
that child's own sub-projection in ``children`` and the shared span tree and
warning list, never the job's own stages, units or progress.  For
``artifact`` events the ancestry is ``span_path`` (``path`` is the file).

**Stages.**  ``run.plan`` seeds them (``will_run`` → planned, ``cached`` →
cached, ``skipped`` → not_requested / disabled (``disabled_by_config:*``) /
skipped); ``stage.start`` → running; ``stage.end`` ok/error/cancelled →
done/failed/cancelled; ``stage.skip`` maps its reason the same way
(``checkpoint`` → cached).  A re-emitted plan never rewinds a stage that has
started.  When the run (``run.end``) or the job (final ``job.status``) ends,
planned stages become ``not_reached`` and running ones take the end status.

**Unit accounting** (one rule for Python and TS).  A planned unit ``U`` is
completed by a ``task.end`` with ``status: ok`` and ``unit == U``; by a
``tick`` with ``unit == U`` and ``k == n``; by ``stage.end{ok}`` of S0 / S1
(``s0_load`` / ``s1_influence``); by ``checkpoint{saved}``
(``checkpoint_save``).  The latest ``tick`` with ``k < n`` of a span adds
``k/n`` of one unit of its ``unit`` until that span completes a unit or
ends.  Completions are attributed to the stage in the event's ancestry.

**Progress** = Σ_U w(U)·min(planned_U, done_U + partial_U) / Σ_U w(U)·planned_U
with ``w`` the seed rates (:func:`sparc.studio.jobs.eta.unit_weight`) and
``planned`` summed over the plan's ``will_run`` nodes.  Without a plan it is
the mean child progress (study jobs) or the ``frac`` of the latest
shallowest tick.  It is 1.0 once the own run or the job has succeeded.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from sparc.studio.jobs.eta import unit_weight

__all__ = ["STAGE_IDS", "SERIES_NAMES", "SERIES_PREFIXES", "new_state", "reduce", "replay", "metric_key",
           "stage_of", "is_nested", "unit_completion", "to_contract", "snapshot_parts", "spans_list",
           "children_rows", "MAX_SPANS", "MAX_DEPTH"]

STAGE_IDS = ("S0", "S1", "S2_S3", "baselines", "cv_curve", "S4", "S5", "climate", "S6", "S7", "finish")
SERIES_NAMES = frozenset({"candidate_rmse", "heldout_rmse", "mean_benefit"})
SERIES_PREFIXES = ("scenario.", "cv_row.", "influence.")
MAX_SPANS = 2000
MAX_DEPTH = 4
GAP_S = 45.0
OBS_KEEP = 50
SERIES_KEEP = 2000
FINAL_JOB = ("succeeded", "failed", "cancelled", "interrupted")
_STAGE_END = {"ok": "done", "error": "failed", "cancelled": "cancelled"}
_SPAN_STATUS = {"succeeded": "ok", "failed": "error", "cancelled": "cancelled", "interrupted": "error",
                "ok": "ok", "error": "error"}
_SETTLED = ("done", "failed", "cancelled", "running")


def new_state() -> dict:
    return {
        "cursor": -1, "n_events": 0, "first_ts": None, "last_ts": None,
        "run": None, "plan": None, "n_points": None, "stages": None,
        "spans": {}, "running": [],
        "metrics_latest": {}, "metric_series": {},
        "warnings": {}, "artifacts": {}, "checkpoints": [],
        "hb_last": {}, "heartbeat_gaps": [],
        "planned_units": {}, "done_units": {}, "stage_done": {}, "partial": {}, "stage_partial": {},
        "unit_obs": {}, "stage_elapsed": {},
        "progress": None, "current_path": None, "stage": None, "tick": None,
        "status": None, "exit_code": None, "error": None, "result": None,
        "run_status": None, "finished": False, "cancel": None,
        "children": {}, "child_of": {},
    }


def metric_key(name: str, tags: dict | None) -> str:
    """``name`` without tags, else ``name{k1=v1,k2=v2}`` with tags sorted by key (api.md §3)."""
    if not tags:
        return name
    return name + "{" + ",".join(f"{k}={_tag(v)}" for k, v in sorted(tags.items())) + "}"


def _tag(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        return str(int(v)) if abs(v) < 1e15 else repr(v)
    return str(v)


def _ancestry(ev: dict) -> list:
    p = ev.get("span_path") if ev.get("type") == "artifact" else ev.get("path")
    return p if isinstance(p, list) else []


def stage_of(path: Iterable) -> str | None:
    """The stage id named in a span path (``stage:<id>[…]``), or None."""
    for el in path or ():
        if isinstance(el, str) and el.startswith("stage:"):
            s = el[6:]
            return s.split("[", 1)[0]
    return None


def _child_index(path: list) -> int | None:
    for i, el in enumerate(path):
        if i >= 1 and isinstance(el, str) and el.startswith("run:"):
            return i
    return None


def is_nested(ev: dict) -> bool:
    """True when ``ev`` belongs to a nested run (a study child), not to the job's own run."""
    return _child_index(_ancestry(ev)) is not None


def unit_completion(ev: dict) -> tuple[str, float | None] | None:
    """``(unit, seconds)`` when ``ev`` completes one planned unit (SPEC §5.4 unit accounting), else None.

    ``seconds`` feeds the calibration table: ``elapsed_s`` of a ``task.end`` /
    ``stage.end`` / ``checkpoint``, ``pass_s`` of a completing tick (None when absent).
    """
    t = ev.get("type")
    if t == "task.end":
        if ev.get("status") == "ok" and ev.get("unit"):
            return str(ev["unit"]), _num(ev.get("elapsed_s"))
        return None
    if t == "tick":
        k, n, unit = ev.get("k"), ev.get("n"), ev.get("unit")
        if unit and isinstance(k, (int, float)) and isinstance(n, (int, float)) and n > 0 and k >= n:
            return str(unit), _num(ev.get("pass_s"))
        return None
    if t == "stage.end":
        if ev.get("status") == "ok" and ev.get("stage") in ("S0", "S1"):
            return ("s0_load" if ev["stage"] == "S0" else "s1_influence"), _num(ev.get("elapsed_s"))
        return None
    if t == "checkpoint" and ev.get("action") == "saved":
        return "checkpoint_save", _num(ev.get("elapsed_s"))
    return None


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _weight_sum(units: dict) -> float:
    return sum(unit_weight(u) * float(n) for u, n in units.items() if n)


def replay(events: Iterable[tuple[int, dict]], state: dict | None = None) -> dict:
    """Fold ``(cursor, event)`` pairs into a projection."""
    state = state or new_state()
    for cursor, ev in events:
        reduce(state, ev, cursor)
    return state


def reduce(state: dict, ev: dict, cursor: int | None = None) -> dict:
    """Apply one event (in file order) to ``state``; returns ``state`` (updated in place)."""
    if cursor is not None:
        state["cursor"] = cursor
    state["n_events"] += 1
    ts = ev.get("ts")
    if isinstance(ts, (int, float)):
        if state["first_ts"] is None:
            state["first_ts"] = ts
        state["last_ts"] = ts if state["last_ts"] is None else max(state["last_ts"], ts)
    t = ev.get("type")
    path = _ancestry(ev)
    idx = _child_index(path)
    if idx is not None:
        _child_event(state, ev, path, idx, cursor)
        _span_event(state, ev, path)
        if t == "warning":
            _warning(state, ev, cursor)
        _update_current(state)
        _update_progress(state)
        return state
    handler = _HANDLERS.get(t)
    if handler is not None:
        handler(state, ev, path, cursor)
    _update_current(state)
    _update_progress(state)
    return state


# ---------------------------------------------------------------------------
# handlers (own events)
# ---------------------------------------------------------------------------

def _run_start(state, ev, path, cursor):
    _span_event(state, ev, path)
    run = state["run"] or {}
    run.update({"name": ev.get("name"), "stages": ev.get("stages"), "fast": ev.get("fast"),
                "coarse": ev.get("coarse"), "resume": ev.get("resume"), "cv_curve": ev.get("cv_curve"),
                "run_meta": ev.get("run_meta") or {}, "started_ts": ev.get("ts")})
    state["run"] = run
    state["run_status"] = "running"


def _run_dir(state, ev, path, cursor):
    run = state["run"] or {}
    run.update({"run_dir": ev.get("run_dir"), "fingerprint": ev.get("fingerprint")})
    state["run"] = run


def _run_plan(state, ev, path, cursor):
    nodes = [dict(n) for n in ev.get("nodes") or [] if isinstance(n, dict) and n.get("id")]
    state["plan"] = nodes
    if ev.get("n_points") is not None:
        state["n_points"] = ev.get("n_points")
    stages = state["stages"] if state["stages"] is not None else {}
    planned: dict[str, float] = {}
    for node in nodes:
        sid = node["id"]
        st = stages.get(sid)
        if st is None or st["state"] not in _SETTLED:
            new_state_, reason = _plan_state(node)
            stages[sid] = _stage(stages.get(sid), new_state_, reason)
            if node.get("est_s") is not None:
                stages[sid]["est_s"] = node.get("est_s")
        if node.get("state", "will_run") == "will_run":
            for u, n in (node.get("units") or {}).items():
                planned[u] = planned.get(u, 0.0) + float(n or 0)
    state["stages"] = stages
    state["planned_units"] = planned


def _plan_state(node: dict) -> tuple[str, str | None]:
    s = node.get("state", "will_run")
    reason = node.get("reason")
    if s == "will_run":
        return "planned", reason
    if s == "cached":
        return "cached", reason or "checkpoint"
    return _skip_state(reason), reason


def _skip_state(reason: str | None) -> str:
    r = reason or ""
    if r == "checkpoint":
        return "cached"
    if r == "not_requested":
        return "not_requested"
    if r.startswith("disabled_by_config"):
        return "disabled"
    return "skipped"


def _stage(prev: dict | None, state_: str, reason: str | None = None) -> dict:
    st = dict(prev) if prev else {"state": "planned", "reason": None, "started_ts": None, "ended_ts": None,
                                  "elapsed_s": None, "est_s": None, "progress": None}
    st["state"] = state_
    st["reason"] = reason
    return st


def _stages(state) -> dict:
    if state["stages"] is None:
        state["stages"] = {}
    return state["stages"]


def _stage_start(state, ev, path, cursor):
    _span_event(state, ev, path)
    sid = ev.get("stage")
    if not sid:
        return
    stages = _stages(state)
    st = _stage(stages.get(sid), "running", None)
    st["started_ts"] = ev.get("ts")
    st["ended_ts"] = st["elapsed_s"] = None
    if ev.get("est_s") is not None:
        st["est_s"] = ev.get("est_s")
    stages[sid] = st


def _stage_end(state, ev, path, cursor):
    _span_event(state, ev, path)
    sid = ev.get("stage")
    if not sid:
        return
    stages = _stages(state)
    status = ev.get("status", "ok")
    st = _stage(stages.get(sid), _STAGE_END.get(status, "failed"), None)
    st["ended_ts"] = ev.get("ts")
    el = ev.get("elapsed_s")
    if el is None and st.get("started_ts") is not None and ev.get("ts") is not None:
        el = ev["ts"] - st["started_ts"]
    st["elapsed_s"] = el
    stages[sid] = st
    if el is not None:
        state["stage_elapsed"][sid] = el
    if status == "ok" and sid in ("S0", "S1"):
        _complete(state, sid, "s0_load" if sid == "S0" else "s1_influence", el)


def _stage_skip(state, ev, path, cursor):
    sid = ev.get("stage")
    if not sid:
        return
    stages = _stages(state)
    cur = stages.get(sid)
    if cur is not None and cur["state"] in ("done", "failed", "cancelled"):
        return
    reason = ev.get("reason")
    stages[sid] = _stage(cur, _skip_state(reason), reason)


def _task_start(state, ev, path, cursor):
    _span_event(state, ev, path)


def _task_end(state, ev, path, cursor):
    _span_event(state, ev, path)
    span = ev.get("span")
    if span and state["partial"].pop(span, None) is not None:
        _recompute_partial(state)
    unit = ev.get("unit")
    if ev.get("status") == "ok" and unit:
        _complete(state, stage_of(path), unit, ev.get("elapsed_s"))
    metrics = ev.get("metrics")
    if isinstance(metrics, dict) and metrics:
        tags = {"task": ev.get("name")}
        if ev.get("key") is not None:
            tags["key"] = ev.get("key")
        for name, value in metrics.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                _metric(state, str(name), value, None, tags, ev.get("ts"), cursor)


def _tick(state, ev, path, cursor):
    k, n, unit = ev.get("k"), ev.get("n"), ev.get("unit")
    span = ev.get("span")
    depth = len(path)
    frac = ev.get("frac")
    if frac is None and isinstance(k, (int, float)) and isinstance(n, (int, float)) and n:
        frac = k / n
    if frac is not None:
        cur = state["tick"]
        if cur is None or depth <= cur["depth"]:
            state["tick"] = {"depth": depth, "frac": max(0.0, min(1.0, float(frac)))}
    if not unit or not isinstance(k, (int, float)) or not isinstance(n, (int, float)) or n <= 0:
        return
    stage = stage_of(path)
    if k >= n:
        state["partial"].pop(span, None)
        _complete(state, stage, unit, ev.get("pass_s"))
    else:
        state["partial"][span] = [stage, unit, max(0.0, float(k) / float(n))]
    _recompute_partial(state)


def _metric_ev(state, ev, path, cursor):
    _metric(state, ev.get("name"), ev.get("value"), ev.get("unit"), ev.get("tags") or {}, ev.get("ts"), cursor)


def _artifact(state, ev, path, cursor):
    rel = ev.get("path")
    if not isinstance(rel, str):
        return
    stage = ev.get("stage") or stage_of(path)
    row = state["artifacts"].get(rel)
    if row is None:
        state["artifacts"][rel] = {"relpath": rel, "role": ev.get("role") or "", "bytes": int(ev.get("bytes") or 0),
                                   "stage": stage, "ts": ev.get("ts")}
    else:
        row.update(role=ev.get("role") or row["role"], bytes=int(ev.get("bytes") or 0), stage=stage or row["stage"],
                   ts=ev.get("ts"))


def _checkpoint(state, ev, path, cursor):
    action = ev.get("action")
    state["checkpoints"].append({"action": action, "done": list(ev.get("done") or []), "bytes": ev.get("bytes"),
                                 "ts": ev.get("ts")})
    if action == "saved":
        _complete(state, stage_of(path), "checkpoint_save", ev.get("elapsed_s"))


def _warning(state, ev, cursor):
    code = str(ev.get("code") or "warning")
    msg = str(ev.get("message") or "")
    h = hashlib.sha1(msg.encode("utf-8", "replace")).hexdigest()[:12]
    key = f"{code}|{h}"
    row = state["warnings"].get(key)
    if row is None:
        state["warnings"][key] = {"code": code, "lvl": ev.get("lvl") or "warning", "message": msg, "count": 1,
                                  "stage": stage_of(_ancestry(ev)), "first_cursor": cursor if cursor is not None else -1,
                                  "data": ev.get("data") if isinstance(ev.get("data"), dict) else {},
                                  "msg_hash": h}
    else:
        row["count"] += 1


def _warning_ev(state, ev, path, cursor):
    _warning(state, ev, cursor)


def _heartbeat(state, ev, path, cursor):
    ts = ev.get("ts")
    if not isinstance(ts, (int, float)):
        return
    pid = str(ev.get("pid"))
    last = state["hb_last"].get(pid)
    if last is not None and ts - last > GAP_S:
        state["heartbeat_gaps"].append({"from_ts": last, "to_ts": ts})
    state["hb_last"][pid] = ts


def _cancel_requested(state, ev, path, cursor):
    state["cancel"] = {"by": ev.get("by"), "ts": ev.get("ts")}


def _run_end(state, ev, path, cursor):
    _span_event(state, ev, path)
    status = ev.get("status")
    state["run_status"] = status
    run = state["run"] or {}
    run.update({"ended_ts": ev.get("ts"), "elapsed_s": ev.get("elapsed_s"), "done": ev.get("done") or [],
                "timings_s": ev.get("timings_s") or {}, "error": ev.get("error")})
    state["run"] = run
    _finish_stages(state, status)
    if status == "succeeded":
        state["finished"] = True


def _job_status(state, ev, path, cursor):
    status = ev.get("status")
    state["status"] = status
    if ev.get("exit_code") is not None:
        state["exit_code"] = ev.get("exit_code")
    if ev.get("error") is not None:
        state["error"] = ev.get("error")
    if status in FINAL_JOB:
        state["finished"] = True
        _finish_stages(state, status)
        span_status = _SPAN_STATUS.get(status, "error")
        for sid in list(state["running"]):
            sp = state["spans"].get(sid)
            if sp is not None and sp["status"] == "running":
                sp["status"] = span_status
                sp["ended_ts"] = ev.get("ts")
                if sp.get("started_ts") is not None and ev.get("ts") is not None:
                    sp["elapsed_s"] = round(ev["ts"] - sp["started_ts"], 4)
        state["running"] = []
        state["partial"] = {}
        _recompute_partial(state)


def _job_result(state, ev, path, cursor):
    if isinstance(ev.get("result"), dict):
        state["result"] = ev.get("result")


def _finish_stages(state, status: str | None) -> None:
    stages = state["stages"]
    if not stages:
        return
    end_state = {"succeeded": "done", "failed": "failed", "cancelled": "cancelled",
                 "interrupted": "failed"}.get(status or "", "failed")
    for sid, st in stages.items():
        if st["state"] == "planned":
            st["state"] = "not_reached"
        elif st["state"] == "running":
            st["state"] = end_state


_HANDLERS = {
    "run.start": _run_start, "run.dir": _run_dir, "run.plan": _run_plan,
    "stage.start": _stage_start, "stage.end": _stage_end, "stage.skip": _stage_skip,
    "task.start": _task_start, "task.end": _task_end, "tick": _tick, "metric": _metric_ev,
    "artifact": _artifact, "checkpoint": _checkpoint, "warning": _warning_ev, "heartbeat": _heartbeat,
    "cancel.requested": _cancel_requested, "run.end": _run_end, "job.status": _job_status,
    "job.result": _job_result,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _complete(state, stage: str | None, unit: str, seconds) -> None:
    state["done_units"][unit] = state["done_units"].get(unit, 0) + 1
    if stage:
        sd = state["stage_done"].setdefault(stage, {})
        sd[unit] = sd.get(unit, 0) + 1
    if isinstance(seconds, (int, float)) and seconds > 0:
        obs = state["unit_obs"].setdefault(unit, [])
        obs.append(round(float(seconds), 6))
        if len(obs) > OBS_KEEP:
            del obs[: len(obs) - OBS_KEEP]
    _recompute_partial(state)


def _recompute_partial(state) -> None:
    per_stage: dict[str, dict[str, float]] = {}
    for stage, unit, frac in state["partial"].values():
        d = per_stage.setdefault(stage or "", {})
        d[unit] = d.get(unit, 0.0) + frac
    state["stage_partial"] = per_stage


def _metric(state, name, value, unit, tags, ts, cursor) -> None:
    if not name:
        return
    key = metric_key(name, tags)
    state["metrics_latest"][key] = {"value": value, "unit": unit, "tags": dict(tags or {}), "ts": ts}
    if (name in SERIES_NAMES or name.startswith(SERIES_PREFIXES)) and isinstance(value, (int, float)) \
            and not isinstance(value, bool):
        series = state["metric_series"].setdefault(key, [])
        series.append({"ts": ts, "value": value})
        if len(series) > SERIES_KEEP:
            del series[: len(series) - SERIES_KEEP]


def _span_event(state, ev, path) -> None:
    """Open or close the span an ``*.start`` / ``*.end`` event describes."""
    t = ev.get("type") or ""
    if not (t.endswith(".start") or t.endswith(".end")):
        return
    sid = ev.get("span")
    if not sid:
        return
    kind = t.split(".", 1)[0]
    if kind not in ("run", "stage", "task"):
        return
    spans = state["spans"]
    sp = spans.get(sid)
    if t.endswith(".start"):
        name = ev.get("stage") if kind == "stage" else ev.get("name")
        sp = {"span_id": sid, "parent_id": ev.get("parent"), "kind": kind, "name": str(name or kind),
              "key": ev.get("key") if kind == "task" else None,
              "k": ev.get("k") if kind == "task" else None, "n": ev.get("n") if kind == "task" else None,
              "unit": ev.get("unit") if kind == "task" else None, "status": "running",
              "started_ts": ev.get("ts"), "ended_ts": None, "elapsed_s": None,
              "ctx": ev.get("ctx") or {}, "metrics": {}, "path": list(path), "depth": len(path)}
        spans[sid] = sp
        state["running"].append(sid)
        return
    if sp is None:                            # an end whose start we never saw
        name = ev.get("stage") if kind == "stage" else ev.get("name")
        sp = {"span_id": sid, "parent_id": ev.get("parent"), "kind": kind, "name": str(name or kind),
              "key": ev.get("key") if kind == "task" else None, "k": ev.get("k") if kind == "task" else None,
              "n": ev.get("n") if kind == "task" else None, "unit": ev.get("unit") if kind == "task" else None,
              "status": "running", "started_ts": ev.get("ts"), "ended_ts": None, "elapsed_s": None,
              "ctx": ev.get("ctx") or {}, "metrics": {}, "path": list(path), "depth": len(path)}
        spans[sid] = sp
    status = ev.get("status")
    sp["status"] = _SPAN_STATUS.get(status, "error") if kind == "run" else (status if status in (
        "ok", "error", "cancelled") else "error")
    sp["ended_ts"] = ev.get("ts")
    el = ev.get("elapsed_s")
    if el is None and sp.get("started_ts") is not None and ev.get("ts") is not None:
        el = round(ev["ts"] - sp["started_ts"], 4)
    sp["elapsed_s"] = el
    if kind == "task" and isinstance(ev.get("metrics"), dict):
        sp["metrics"] = ev["metrics"]
    elif kind == "stage" and isinstance(ev.get("summary"), dict):
        sp["metrics"] = ev["summary"]
    if sid in state["running"]:
        state["running"].remove(sid)


def _update_current(state) -> None:
    run_sp = None
    for sid in reversed(state["running"]):
        sp = state["spans"].get(sid)
        if sp is not None and sp["status"] == "running":
            run_sp = sp
            break
    state["current_path"] = list(run_sp["path"]) if run_sp else None
    stage = None
    for sid in reversed(state["running"]):
        sp = state["spans"].get(sid)
        if sp is not None and sp["kind"] == "stage" and _child_index(sp["path"]) is None:
            stage = sp["name"]
            break
    state["stage"] = stage


def _update_progress(state) -> None:
    if state["status"] == "succeeded" or state["run_status"] == "succeeded":
        state["progress"] = 1.0
    else:
        p = _own_progress(state)
        if p is None and state["children"]:
            ps = [c["state"]["progress"] or 0.0 for c in state["children"].values()]
            p = sum(ps) / len(ps)
        if p is None and state["tick"] is not None:
            p = state["tick"]["frac"]
        state["progress"] = None if p is None else round(min(1.0, max(0.0, p)), 12)
    stages = state["stages"]
    if stages:
        for node in state["plan"] or []:
            st = stages.get(node["id"])
            if st is None:
                continue
            if st["state"] == "done":
                st["progress"] = 1.0
            elif st["state"] in ("running", "planned"):
                st["progress"] = _node_progress(state, node)
            else:
                st["progress"] = None


def _own_progress(state) -> float | None:
    planned = state["planned_units"]
    total = _weight_sum(planned)
    if total <= 0:
        return None
    partial: dict[str, float] = {}
    for _stage, unit, frac in state["partial"].values():
        partial[unit] = partial.get(unit, 0.0) + frac
    done = state["done_units"]
    got = 0.0
    for u, n in planned.items():
        if n:
            got += unit_weight(u) * min(float(n), float(done.get(u, 0)) + partial.get(u, 0.0))
    return got / total


def _node_progress(state, node) -> float | None:
    units = node.get("units") or {}
    total = _weight_sum(units)
    if total <= 0:
        return None
    done = state["stage_done"].get(node["id"], {})
    part = state["stage_partial"].get(node["id"], {})
    got = sum(unit_weight(u) * min(float(n), float(done.get(u, 0)) + float(part.get(u, 0.0)))
              for u, n in units.items() if n)
    return round(got / total, 12)


# ---------------------------------------------------------------------------
# nested runs (study children)
# ---------------------------------------------------------------------------

def _child_event(state, ev, path, idx, cursor) -> None:
    prefix = json.dumps(path[: idx + 1])
    key = state["child_of"].get(prefix)
    if key is None:
        if ev.get("type") != "run.start":
            return
        key = _child_key(state, ev, path[idx])
        state["child_of"][prefix] = key
        meta = ev.get("run_meta") or {}
        state["children"][key] = {"key": key, "label": meta.get("role") or ev.get("name") or key,
                                  "run_id": meta.get("studio_run_id"), "job_id": None, "state": new_state()}
    child = state["children"][key]
    sub = dict(ev)
    if sub.get("type") == "artifact":
        sub["span_path"] = path[idx:]
    else:
        sub["path"] = path[idx:]
    reduce(child["state"], sub, cursor)


def _child_key(state, ev, run_el: str) -> str:
    ctx = ev.get("ctx") or {}
    meta = ev.get("run_meta") or {}
    if ctx.get("placebo") is not None:
        key = f"placebo:{ctx['placebo']}"
    elif ctx.get("variant") is not None:
        key = f"variant:{ctx['variant']}"
    elif meta.get("role"):
        key = str(meta["role"])
    else:
        key = run_el[4:] or "child"
    base, i = key, 2
    while key in state["children"]:
        key = f"{base}#{i}"
        i += 1
    return key


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def _child_status(cs: dict) -> str:
    if cs["run_status"] in ("succeeded", "failed", "cancelled"):
        return cs["run_status"]
    return "running" if cs["run_status"] == "running" else "planned"


def children_rows(state) -> list[dict]:
    rows = []
    for key, c in state["children"].items():
        cs = c["state"]
        metrics = {k: v["value"] for k, v in cs["metrics_latest"].items()
                   if "{" not in k and isinstance(v.get("value"), (int, float, str, bool))}
        rows.append({"job_id": c.get("job_id"), "run_id": c.get("run_id"), "key": key, "label": c["label"],
                     "status": _child_status(cs), "progress": cs["progress"], "metrics": metrics})
    return rows


def _span_out(sp: dict) -> dict:
    return {k: sp.get(k) for k in ("span_id", "parent_id", "kind", "name", "key", "k", "n", "unit", "status",
                                   "started_ts", "ended_ts", "elapsed_s", "ctx", "metrics")}


def spans_list(state, *, under: str | None = None, max_depth: int = MAX_DEPTH, limit: int = MAX_SPANS) -> list[dict]:
    """Span rows (depth ≤ ``max_depth`` below ``under``).  Beyond ``limit`` rows, spans deeper than two
    levels are aggregated per (parent, name) into one row each (``metrics.aggregated`` = count)."""
    spans = state["spans"]
    if under:
        root = spans.get(under)
        if root is None:
            return []
        base = root["depth"]
        rows = [sp for sp in spans.values()
                if sp["span_id"] == under or (_has_ancestor(spans, sp, under) and sp["depth"] - base <= max_depth)]
    else:
        base = min((sp["depth"] for sp in spans.values()), default=1) - 1
        rows = [sp for sp in spans.values() if sp["depth"] - base <= max_depth]
    if len(rows) <= limit:
        return [_span_out(sp) for sp in rows]
    keep = [sp for sp in rows if sp["depth"] - base <= 2]
    groups: dict[tuple, dict] = {}
    for sp in rows:
        if sp["depth"] - base <= 2:
            continue
        g = groups.get((sp["parent_id"], sp["name"]))
        if g is None:
            g = groups[(sp["parent_id"], sp["name"])] = {
                "span_id": f"{sp['parent_id']}/{sp['name']}*", "parent_id": sp["parent_id"], "kind": sp["kind"],
                "name": sp["name"], "key": None, "k": None, "n": 0, "unit": sp.get("unit"), "status": "ok",
                "started_ts": sp["started_ts"], "ended_ts": sp["ended_ts"], "elapsed_s": 0.0, "ctx": {},
                "metrics": {"aggregated": 0}}
        g["n"] += 1
        g["metrics"]["aggregated"] += 1
        g["elapsed_s"] = round((g["elapsed_s"] or 0.0) + (sp.get("elapsed_s") or 0.0), 4)
        if sp["started_ts"] is not None and (g["started_ts"] is None or sp["started_ts"] < g["started_ts"]):
            g["started_ts"] = sp["started_ts"]
        if sp["ended_ts"] is None or g["ended_ts"] is None:
            g["ended_ts"] = None if sp["status"] == "running" else (g["ended_ts"] or sp["ended_ts"])
        elif sp["ended_ts"] > g["ended_ts"]:
            g["ended_ts"] = sp["ended_ts"]
        rank = {"running": 3, "error": 2, "cancelled": 1, "ok": 0}
        if rank.get(sp["status"], 0) > rank.get(g["status"], 0):
            g["status"] = sp["status"]
    return [_span_out(sp) for sp in keep] + list(groups.values())


def _has_ancestor(spans: dict, sp: dict, anc: str) -> bool:
    seen = 0
    cur = sp.get("parent_id")
    while cur and seen < 64:
        if cur == anc:
            return True
        nxt = spans.get(cur)
        cur = nxt.get("parent_id") if nxt else None
        seen += 1
    return False


def snapshot_parts(state, *, eta: dict | None = None) -> dict:
    """The projection part of ``TrackerSnapshot`` (api.md §3; ``job`` and ``resources`` are added by the caller)."""
    stages = None
    if state["stages"] is not None:
        stages = {}
        est = (eta or {}).get("stages") or {}
        for sid, st in state["stages"].items():
            row = {k: st.get(k) for k in ("state", "reason", "started_ts", "ended_ts", "elapsed_s", "est_s",
                                          "progress")}
            if sid in est and st["state"] in ("planned", "running"):
                row["est_s"] = est[sid]
            stages[sid] = row
    warnings = [{k: w[k] for k in ("code", "lvl", "message", "count", "stage", "first_cursor", "data")}
                for w in state["warnings"].values()]
    return {
        "cursor": state["cursor"],
        "plan": state["plan"],
        "stages": stages,
        "spans": spans_list(state),
        "metrics_latest": state["metrics_latest"],
        "metric_series": state["metric_series"],
        "warnings": warnings,
        "artifacts": list(state["artifacts"].values()),
        "checkpoints": state["checkpoints"],
        "heartbeat_gaps": state["heartbeat_gaps"],
        "children": children_rows(state),
    }


def to_contract(state) -> dict:
    """Canonical projection shared with the web client's ``toContract`` (SPEC §14.3)::

        {stages: {id: {state, reason}}, progress, done_units: {unit: n},
         warnings: [{code, count}] (summed per code, sorted by code), artifacts: [relpath] (sorted)}
    """
    per_code: dict[str, int] = {}
    for w in state["warnings"].values():
        per_code[w["code"]] = per_code.get(w["code"], 0) + int(w["count"])
    return {
        "stages": {sid: {"state": st["state"], "reason": st.get("reason")}
                   for sid, st in sorted((state["stages"] or {}).items())},
        "progress": state["progress"],
        "done_units": {u: n for u, n in sorted(state["done_units"].items())},
        "warnings": [{"code": c, "count": n} for c, n in sorted(per_code.items())],
        "artifacts": sorted(state["artifacts"]),
    }
