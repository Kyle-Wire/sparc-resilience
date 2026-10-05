"""sparc.core.runio: atomic writes, NaN → null, run_lock and concurrent update_manifest."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sparc.core import runio

ROOT = Path(runio.__file__).resolve().parents[2]


def _strict_json(path):
    def reject(name):
        raise ValueError(f"non-standard JSON constant {name}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def _leftovers(d: Path):
    return sorted(p.name for p in d.iterdir() if p.name.endswith(".tmp"))


def test_write_json_atomic_maps_nan_to_null(tmp_path):
    obj = {"nan": float("nan"), "inf": [float("inf"), -float("inf"), 1.5], "np32": np.float32("nan"),
           "np64": np.float64(2.5), "int": np.int64(3), "flag": np.bool_(True), "arr": np.array([1.0, np.nan]),
           "path": tmp_path / "x", "set": {"b", "a"}, 7: "int key",
           "series": pd.Series([0.5, np.nan]), "when": pd.Timestamp("2026-10-01T12:00:00")}
    out = runio.write_json_atomic(tmp_path / "m.json", obj)
    text = out.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    got = _strict_json(out)
    assert got == {"nan": None, "inf": [None, None, 1.5], "np32": None, "np64": 2.5, "int": 3, "flag": True,
                   "arr": [1.0, None], "path": str(tmp_path / "x"), "set": ["a", "b"], "7": "int key",
                   "series": [0.5, None], "when": "2026-10-01T12:00:00"}
    assert _leftovers(tmp_path) == []


def test_failed_write_leaves_the_old_file(tmp_path):
    target = tmp_path / "report.md"
    runio.write_text_atomic(target, "old\n")
    with pytest.raises(RuntimeError):
        with runio.atomic_open(target, "w", encoding="utf-8") as f:
            f.write("half of the new")
            raise RuntimeError("killed mid-write")
    assert target.read_text(encoding="utf-8") == "old\n" and _leftovers(tmp_path) == []
    with pytest.raises(ValueError):
        with runio.atomic_open(target, "a"):
            pass


def test_parquet_npz_bytes_round_trip(tmp_path):
    df = pd.DataFrame({"id": np.arange(5), "v": np.linspace(0, 1, 5).astype("float32")})
    runio.write_parquet_atomic(df, tmp_path / "p.parquet")
    back = pd.read_parquet(tmp_path / "p.parquet")
    pd.testing.assert_frame_equal(back, df)
    assert list(back.columns) == ["id", "v"]                 # index=False by default
    runio.write_npz_atomic(tmp_path / "scenario_detail.npz", ids=np.arange(3), f0=np.ones((2, 3), "float32"))
    runio.write_npz_atomic(tmp_path / "emulator.bin", compressed=True, a=np.zeros(1000))
    with np.load(tmp_path / "scenario_detail.npz") as z:
        assert z["f0"].dtype == np.float32 and z["ids"].tolist() == [0, 1, 2]
    with np.load(tmp_path / "emulator.bin") as z:              # used as given: no ".npz" appended
        assert z["a"].shape == (1000,)
    assert (tmp_path / "emulator.bin").stat().st_size < 1000   # compressed
    runio.write_bytes_atomic(tmp_path / "sub" / "b.bin", b"\x00\x01")
    assert (tmp_path / "sub" / "b.bin").read_bytes() == b"\x00\x01"
    assert _leftovers(tmp_path) == []


def test_readers_never_see_a_partial_file(tmp_path):
    target = tmp_path / "big.json"
    payloads = [{"which": w, "data": [w] * 200_000} for w in ("a", "b")]
    runio.write_json_atomic(target, payloads[0])
    stop = threading.Event()
    errors = []

    def writer():
        i = 0
        while not stop.is_set():
            runio.write_json_atomic(target, payloads[i % 2], indent=None)
            i += 1

    t = threading.Thread(target=writer)
    t.start()
    try:
        deadline = time.monotonic() + 1.0
        reads = 0
        while time.monotonic() < deadline:
            try:
                got = json.loads(target.read_text(encoding="utf-8"))
                assert len(got["data"]) == 200_000 and set(got["data"]) == {got["which"]}
                reads += 1
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))
    finally:
        stop.set()
        t.join()
    assert errors == [] and reads > 0


def test_update_manifest_records_history_and_nulls(tmp_path):
    runio.write_json_atomic(tmp_path / "manifest.json", {"name": "r", "metrics": {"stacker": {"r2": 0.8}}})
    m1 = runio.update_manifest(tmp_path, {"baselines": {"rmse": float("nan")}}, source="post.baselines")
    m2 = runio.update_manifest(tmp_path, {"planner": {"people": 10}, "uncertainty": {"x": [1, float("inf")]}},
                               source="post.planner")
    on_disk = _strict_json(tmp_path / "manifest.json")
    assert on_disk == m2
    assert on_disk["name"] == "r" and on_disk["metrics"] == {"stacker": {"r2": 0.8}}
    assert on_disk["baselines"] == {"rmse": None} and on_disk["uncertainty"] == {"x": [1, None]}
    assert [(h["section"], h["source"]) for h in on_disk["post_run"]] == [
        ("baselines", "post.baselines"), ("planner", "post.planner"), ("uncertainty", "post.planner")]
    assert all(h["at_utc"].endswith("Z") for h in on_disk["post_run"])
    assert m1["post_run"][0]["section"] == "baselines"
    with pytest.raises(ValueError):
        runio.update_manifest(tmp_path, {"post_run": []}, source="x")
    assert (tmp_path / runio.LOCK_NAME).exists()


_PROC_UPDATES = """
import sys
from sparc.core import runio
run_dir, tag, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.stdout.write("ready\\n"); sys.stdout.flush()
sys.stdin.readline()                                   # start together with the threads
for i in range(n):
    runio.update_manifest(run_dir, {f"{tag}_{i}": {"i": i, "v": float("nan")}}, source=tag)
"""


def test_update_manifest_concurrent_threads_and_processes_lose_nothing(tmp_path):
    runio.write_json_atomic(tmp_path / "manifest.json", {"name": "run", "keep": True})
    n_each = 6
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    procs = [subprocess.Popen([sys.executable, "-c", _PROC_UPDATES, str(tmp_path), f"proc{p}", str(n_each)],
                              cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
             for p in range(2)]
    for p in procs:
        assert p.stdout.readline().strip() == "ready"
    barrier = threading.Barrier(8)
    errors = []

    def thread_updates(t):
        try:
            barrier.wait()
            for i in range(n_each):
                runio.update_manifest(tmp_path, {f"thread{t}_{i}": {"i": i, "v": float("inf")}}, source=f"thread{t}")
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=thread_updates, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for p in procs:
        p.stdin.write("go\n")
        p.stdin.flush()
    for t in threads:
        t.join(timeout=120)
    for p in procs:
        assert p.wait(timeout=120) == 0
    assert errors == []
    m = _strict_json(tmp_path / "manifest.json")
    expected = {f"thread{t}_{i}" for t in range(8) for i in range(n_each)} | \
               {f"proc{p}_{i}" for p in range(2) for i in range(n_each)}
    assert expected <= set(m) and m["keep"] is True and m["name"] == "run"
    assert all(m[k]["v"] is None for k in expected)
    assert sorted(h["section"] for h in m["post_run"]) == sorted(expected)
    assert _leftovers(tmp_path) == []


def test_run_lock_serialises_threads_and_is_reentrant(tmp_path):
    counter = {"n": 0}

    def bump():
        for _ in range(20):
            with runio.run_lock(tmp_path):
                v = counter["n"]
                time.sleep(0.0005)
                counter["n"] = v + 1

    threads = [threading.Thread(target=bump) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert counter["n"] == 120
    with runio.run_lock(tmp_path):
        with runio.run_lock(tmp_path):                       # re-entrant in one thread
            runio.update_manifest(tmp_path, {"a": 1}, source="t")
    assert _strict_json(tmp_path / "manifest.json")["a"] == 1


_HOLD = """
import sys, time
from sparc.core import runio
with runio.run_lock(sys.argv[1]):
    sys.stdout.write("locked\\n"); sys.stdout.flush()
    time.sleep(float(sys.argv[2]))
"""


def test_run_lock_excludes_other_processes(tmp_path):
    p = subprocess.Popen([sys.executable, "-c", _HOLD, str(tmp_path), "1.5"], cwd=ROOT,
                         env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "locked"
        with pytest.raises(TimeoutError):
            with runio.run_lock(tmp_path, timeout=0.2):
                pass
        t0 = time.monotonic()
        with runio.run_lock(tmp_path, timeout=30):            # granted once the holder exits
            waited = time.monotonic() - t0
        assert waited > 0.2
    finally:
        p.wait(timeout=30)


def killed_writer_tmp(target: Path) -> Path:
    """A process killed with SIGKILL inside ``atomic_open(target)`` (Studio's Force stop, the OOM killer while
    pickling a checkpoint): its hidden temporary sibling stays behind.  Returns that file."""
    code = ("import sys, time\nfrom sparc.core import runio\n"
            "with runio.atomic_open(sys.argv[1], 'wb') as f:\n"
            "    f.write(b'x' * 65536); f.flush(); print('ready', flush=True); time.sleep(60)\n")
    p = subprocess.Popen([sys.executable, "-c", code, str(target)], stdout=subprocess.PIPE, cwd=ROOT)
    try:
        assert p.stdout.readline().strip() == b"ready"
    finally:
        p.kill()
        p.wait(10)
        p.stdout.close()
    left = list(target.parent.glob(f".{target.name}.{p.pid}.*.tmp"))
    assert len(left) == 1, sorted(x.name for x in target.parent.iterdir())
    return left[0]


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL and pid checks are POSIX")
def test_temporaries_of_killed_writers_are_swept(tmp_path):
    """Nothing removed the temporary of a writer killed mid-write (its name holds the dead pid, so the next
    write never reuses it): the next atomic write of that file removes it, and ``remove_stale_tmp`` sweeps a
    folder; a live writer's temporary is never touched."""
    target = tmp_path / "checkpoint.pkl"
    stale = killed_writer_tmp(target)
    live = tmp_path / f".checkpoint.pkl.{os.getpid()}.1.tmp"            # this process: alive
    live.write_bytes(b"in progress")
    runio.write_bytes_atomic(target, b"new checkpoint")
    assert not stale.exists() and live.exists() and target.read_bytes() == b"new checkpoint"

    other = killed_writer_tmp(tmp_path / "checkpoint.json")
    unrelated = tmp_path / ".notes.tmp"
    unrelated.write_text("not ours")
    assert runio.remove_stale_tmp(tmp_path) == 65536
    assert not other.exists() and live.exists() and unrelated.exists()
    assert runio.remove_stale_tmp(tmp_path / "missing") == 0


def test_replace_rides_out_a_windows_sharing_violation(tmp_path, monkeypatch):
    """On Windows a reader holding the target open makes os.replace fail for a moment: retry, then succeed."""
    import os as _os

    from sparc.core import runio

    src, dst = tmp_path / "a", tmp_path / "b"
    src.write_text("new", "utf-8")
    dst.write_text("old", "utf-8")
    real, calls = _os.replace, []

    def flaky(a, b):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(13, "The process cannot access the file because it is being used")
        real(a, b)

    monkeypatch.setattr(runio.os, "replace", flaky)
    monkeypatch.setattr(runio.os, "name", "nt")
    monkeypatch.setattr(runio.time, "sleep", lambda s: None)
    runio.replace(src, dst)
    assert dst.read_text("utf-8") == "new" and len(calls) == 3
    monkeypatch.setattr(runio.os, "name", "posix")                 # elsewhere the first error is raised
    calls.clear()
    src.write_text("x", "utf-8")
    import pytest as _pytest

    with _pytest.raises(PermissionError):
        runio.replace(src, dst)
    assert len(calls) == 1
