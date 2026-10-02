"""Cross-language reducer contract (SPEC §14.3).

The server's projection (``sparc.studio.jobs.tracker.reduce`` → ``to_contract``) and the web
client's (``applyEvent`` → ``toContract`` in ``studio-web/src/stores/tracker.ts``) replay the
recorded synthetic run ``tests/studio/fixtures/synth_run/events.jsonl``. After **every** event
they must agree: identical stage states, done units, warnings and artifacts, and progress within
1e-9. The client side runs in vitest (``studio-web/src/contract/reducer.contract.test.ts``) and
writes its projections to ``SPARC_CONTRACT_OUT``; the test is skipped when Node or the web
dependencies are absent.

Both sides are also held to the committed golden projection
``tests/studio/fixtures/reducer_projection.golden.json`` (progress after each event, stage
states wherever they change, and the final contract). After an intended change to the reducer
rules, rewrite it from the Python reducer with::

    python tests/studio/test_reducer_contract.py --update
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WEB = ROOT / "studio-web"
FIXTURE = "tests/studio/fixtures/synth_run/events.jsonl"
GOLDEN = HERE / "fixtures" / "reducer_projection.golden.json"
TOL = 1e-9


def _events() -> list[tuple[int, dict, bytes]]:
    from sparc.studio.jobs.tailer import iter_lines, parse_line

    return [(cursor, parse_line(raw), raw) for cursor, raw in iter_lines(ROOT / FIXTURE)]


def python_trajectory() -> dict:
    """The canonical projection after every event, as the server folds the job log."""
    from sparc.studio.jobs import tracker

    state = tracker.new_state()
    steps = []
    for cursor, ev, _ in _events():
        tracker.reduce(state, ev, cursor)
        steps.append({"cursor": cursor, "type": ev["type"], **tracker.to_contract(state)})
    # a round trip through JSON gives both sides the same value types (tuples → lists, …)
    return json.loads(json.dumps({"fixture": FIXTURE, "n_events": len(steps),
                                  "final": tracker.to_contract(state), "steps": steps}))


def compact(traj: dict) -> dict:
    """The golden form: progress per event and the stage states wherever they change."""
    changes, prev = [], {}
    for s in traj["steps"]:
        diff = {sid: [st["state"], st["reason"]] for sid, st in s["stages"].items()
                if prev.get(sid) != st}
        if diff:
            changes.append([s["cursor"], diff])
        prev = s["stages"]
    return {"fixture": traj["fixture"], "n_events": traj["n_events"], "final": traj["final"],
            "progress": [[s["cursor"], s["progress"]] for s in traj["steps"]], "stage_changes": changes}


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=0.0, abs_tol=TOL)


def contract_diffs(got: dict, want: dict, where: str) -> list[str]:
    out = []
    for key in ("stages", "done_units", "warnings", "artifacts"):
        if got.get(key) != want.get(key):
            out.append(f"{where}: {key} differ:\n    got  {got.get(key)}\n    want {want.get(key)}")
    if not _close(got.get("progress"), want.get("progress")):
        out.append(f"{where}: progress {got.get('progress')!r} vs {want.get('progress')!r}")
    return out


def _fail_on(diffs: list[str], what: str) -> None:
    if diffs:
        shown = "\n".join(diffs[:12]) + (f"\n  … {len(diffs) - 12} more" if len(diffs) > 12 else "")
        raise AssertionError(f"{what} ({len(diffs)} differences):\n{shown}")


# ---------------------------------------------------------------------------
# Python side
# ---------------------------------------------------------------------------

def test_fixture_events_are_what_the_wire_carries():
    """Every fixture line validates unchanged, so the client's raw replay sees what the server streams."""
    events = _events()
    assert len(events) > 100 and events[0][0] == 0
    changed = [cursor for cursor, ev, raw in events if ev != json.loads(raw)]
    assert not changed, f"lines the server would rewrite (as log events): cursors {changed[:10]}"
    assert events[-1][1]["type"] == "run.end"


def test_python_reducer_matches_golden():
    assert GOLDEN.exists(), f"{GOLDEN.name} is missing: run `python {Path(__file__).relative_to(ROOT)} --update`"
    golden = json.loads(GOLDEN.read_text("utf-8"))
    mine = compact(python_trajectory())
    assert mine["fixture"] == golden["fixture"] and mine["n_events"] == golden["n_events"]
    diffs = contract_diffs(mine["final"], golden["final"], "final")
    assert [c for c, _ in mine["progress"]] == [c for c, _ in golden["progress"]]
    diffs += [f"progress at cursor {c}: {p!r} vs {g!r}"
              for (c, p), (_, g) in zip(mine["progress"], golden["progress"]) if not _close(p, g)]
    if mine["stage_changes"] != golden["stage_changes"]:
        diffs.append(f"stage changes differ:\n    got  {mine['stage_changes']}\n    want {golden['stage_changes']}")
    _fail_on(diffs, "The Python reducer no longer gives the golden projection; if the rule change is "
                    f"intended, run `python {Path(__file__).relative_to(ROOT)} --update`")


def test_python_projection_shape():
    traj = python_trajectory()
    final = traj["final"]
    assert final["progress"] == 1.0
    assert final["stages"]["cv_curve"]["state"] == "disabled"
    ran = {sid for sid, st in final["stages"].items() if st["state"] == "done"}
    assert {"S0", "S1", "S2_S3", "S4", "S5", "S6", "S7", "finish"} <= ran
    # every artifact the fixture recorded, and nothing else
    recorded = {ev["path"] for _, ev, _ in _events() if ev["type"] == "artifact"}
    assert set(final["artifacts"]) == recorded
    progress = [s["progress"] for s in traj["steps"] if s["progress"] is not None]
    assert all(b >= a - 1e-12 for a, b in zip(progress, progress[1:])), "progress went down"
    assert 0.0 < progress[len(progress) // 2] < 1.0


# ---------------------------------------------------------------------------
# both sides
# ---------------------------------------------------------------------------

def _node_ready() -> str | None:
    if shutil.which("npm") is None or shutil.which("node") is None:
        return "Node/npm not installed"
    if not (WEB / "node_modules" / "vitest").is_dir():
        return "studio-web dependencies not installed (npm --prefix studio-web ci)"
    return None


def test_ts_reducer_matches_python(tmp_path):
    reason = _node_ready()
    if reason:
        pytest.skip(reason)
    out = tmp_path / "ts_projection.json"
    env = dict(os.environ, SPARC_CONTRACT_OUT=str(out), CI="1", NO_COLOR="1")
    proc = subprocess.run(["npm", "--prefix", str(WEB), "exec", "--", "vitest", "run", "src/contract"],
                          cwd=WEB, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"vitest failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-2000:]}"
    assert out.exists(), f"vitest wrote no projection to SPARC_CONTRACT_OUT:\n{proc.stdout[-2000:]}"
    ts = json.loads(out.read_text("utf-8"))
    py = python_trajectory()
    assert ts["fixture"] == py["fixture"] and ts["n_events"] == py["n_events"]
    assert len(ts["steps"]) == len(py["steps"])
    diffs = contract_diffs(ts["final"], py["final"], "final")
    for t, p in zip(ts["steps"], py["steps"]):
        assert (t["cursor"], t["type"]) == (p["cursor"], p["type"])
        diffs += contract_diffs(t, p, f"after {p['type']} at cursor {p['cursor']}")
    _fail_on(diffs, "The TS and Python reducers disagree")


def _update() -> int:
    sys.path.insert(0, str(ROOT))
    golden = compact(python_trajectory())
    GOLDEN.write_text(json.dumps(golden, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN.relative_to(ROOT)} ({golden['n_events']} events, "
          f"{len(golden['stage_changes'])} stage changes)")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--update"]:
        sys.exit(_update())
    print(f"usage: python {Path(__file__).relative_to(ROOT)} --update", file=sys.stderr)
    sys.exit(2)
