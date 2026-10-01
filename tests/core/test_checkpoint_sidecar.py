"""``checkpoint.json``, ``fingerprint_sections`` and ``checkpoint_status`` (SPEC §5.9, §11 item 6)."""

from __future__ import annotations

import copy
import json
import os
import pickle
from pathlib import Path

import pytest

S0_S3 = ("S0", "S1", "S2", "S3")


def _light(cfg):
    cfg.raw["models"] = {"ols": True, "mgwr": False, "gwrf": False, "gam": False, "physics": False}
    cfg.raw["stacker"].update(epochs=20, tune_lambda=[0.0])
    cfg.raw["cv"]["baselines"] = ["idw"]
    return cfg


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    from sparc.core.synthetic import write_demo_project

    return write_demo_project(tmp_path_factory.mktemp("sidecar"), n=24, seed=0)


def _cfg(demo):
    from sparc.core.config import load_core_config

    return _light(load_core_config(demo["config_path"]))


def _collect(level: str = "info"):
    from sparc.core import progress

    events: list[dict] = []
    progress.configure(events.append, level=level, heartbeat_s=0)
    return events


@pytest.fixture
def events():
    from sparc.core import progress

    evs = _collect()
    yield evs
    progress.reset()


@pytest.fixture
def no_unpickle(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the checkpoint must not be unpickled")

    monkeypatch.setattr(pickle, "load", boom)
    monkeypatch.setattr(pickle, "loads", boom)
    monkeypatch.setattr(pickle, "Unpickler", boom)


def test_fingerprint_sections_pinpoint_the_changed_section(demo):
    from sparc.core.pipeline import FINGERPRINT_SECTIONS, _fingerprint, fingerprint_sections

    cfg = _cfg(demo)
    base = fingerprint_sections(cfg, True)
    assert list(base) == list(FINGERPRINT_SECTIONS) and all(len(v) == 16 for v in base.values())

    def changed(edit, fast=True, **kw):
        c = copy.deepcopy(cfg)
        edit(c.raw)
        now = fingerprint_sections(c, fast, **kw)
        return sorted(k for k in base if now[k] != base[k])

    assert changed(lambda r: r["stacker"].update(tune_lambda=[0.0, 0.5]), fast=False) == ["core"]
    assert changed(lambda r: r["stacker"].update(tune_lambda=[0.0, 0.5])) == []     # fast mode sets [0, 0.1]
    assert changed(lambda r: r["stacker"].update(epochs=30)) == ["core"]
    assert changed(lambda r: r["models"].update(gam=True)) == ["core"]
    assert changed(lambda r: r["cv"].update(n_folds=4)) == []        # fast mode caps the folds at 3 either way
    assert changed(lambda r: r["cv"].update(n_folds=4), fast=False) == ["core"]
    assert changed(lambda r: r["cv"]["distance_curve"].update(enabled=True)) == []     # reporting only
    assert changed(lambda r: r["cv"]["distance_curve"].update(block_m=[0, 300])) == ["core"]
    assert changed(lambda r: None, coarse=60.0) == ["core"]
    assert changed(lambda r: r["mediators"].clear()) == ["s4"]
    assert changed(lambda r: r["actionable"]["canopy"].update(doses=[0, 5])) == ["s4"]
    assert changed(lambda r: r["scenarios"].pop()) == ["s5"]
    assert changed(lambda r: r["climate"].update(thresholds=[91, 96])) == ["climate"]
    assert changed(lambda r: r["causal"].update(n_boot=50)) == ["s6"]
    assert changed(lambda r: r["optimize"].update(budget=10.0)) == ["s7"]
    assert changed(lambda r: r["optimize"].update(equity_column="people")) == ["s7"]
    assert changed(lambda r: r["planner"].update(paved_plantable_share=0.5)) == ["s7"]
    assert changed(lambda r: r["report"].update(title="x")) == []

    # the resume fingerprint covers the whole effective config, so every change above invalidates it
    a, b = copy.deepcopy(cfg), copy.deepcopy(cfg)
    b.raw["report"]["title"] = "other"
    assert _fingerprint(a, True, None) != _fingerprint(b, True, None)

    data = Path(demo["files"]["data"])
    st = data.stat()
    try:
        os.utime(data, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        now = fingerprint_sections(cfg, True)
        assert [k for k in base if now[k] != base[k]] == ["data"]
    finally:
        os.utime(data, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert fingerprint_sections(cfg, True) == base


def test_instrumentation_modules_are_not_fingerprinted(demo, monkeypatch):
    from sparc.core.pipeline import _code_digest, _fingerprint, fingerprint_sections

    cfg = _cfg(demo)
    fp0, code0, secs0 = _fingerprint(cfg, False, None), _code_digest(), fingerprint_sections(cfg, False)
    real = Path.read_bytes
    edited: set[str] = set()

    def read_bytes(self):
        data = real(self)
        return data + b"\n# edited\n" if self.name in edited else data

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    edited.update({"progress.py", "runio.py"})
    assert _fingerprint(cfg, False, None) == fp0 and _code_digest() == code0
    assert fingerprint_sections(cfg, False) == secs0
    edited.add("stacker.py")
    assert _fingerprint(cfg, False, None) != fp0
    now = fingerprint_sections(cfg, False)
    assert [k for k in secs0 if now[k] != secs0[k]] == ["code"]


def test_checkpoint_json_matches_the_pickle_and_the_events(demo, tmp_path, events):
    from sparc.core.pipeline import CHECKPOINT, CHECKPOINT_JSON, _code_digest, fingerprint_sections, run_core

    cfg = _cfg(demo)
    res = run_core(cfg, stages=S0_S3, fast=True, run_dir=tmp_path / "run")
    side = json.loads((res.run_dir / CHECKPOINT_JSON).read_text())
    with open(res.run_dir / CHECKPOINT, "rb") as fh:
        state = pickle.load(fh)
    assert side["schema"] == 1 and side["done"] == sorted(state["done"]) == ["S3", "baselines"]
    assert side["fingerprint"] == state["fingerprint"] and len(side["fingerprint"]) == 16
    assert side["bytes"] == (res.run_dir / CHECKPOINT).stat().st_size
    assert side["code_sha256"] == _code_digest()
    assert side["sections"] == fingerprint_sections(res.cfg, True)
    saved = [e for e in events if e["type"] == "checkpoint"]
    assert [e["action"] for e in saved] == ["saved", "saved"]
    assert saved[-1]["done"] == side["done"] and saved[-1]["bytes"] == side["bytes"]
    assert saved[-1]["fingerprint"] == side["fingerprint"]
    run_dir_ev = next(e for e in events if e["type"] == "run.dir")
    assert run_dir_ev["fingerprint"] == side["fingerprint"]


def test_checkpoint_status_never_unpickles(demo, tmp_path):
    from sparc.core.pipeline import CHECKPOINT, CHECKPOINT_JSON, checkpoint_status, run_core

    cfg = _cfg(demo)
    res = run_core(copy.deepcopy(cfg), stages=S0_S3, fast=True, run_dir=tmp_path / "run")
    run_dir = res.run_dir
    mp = pytest.MonkeyPatch()
    try:
        def boom(*a, **k):
            raise AssertionError("the checkpoint must not be unpickled")

        for name in ("load", "loads", "Unpickler"):
            mp.setattr(pickle, name, boom)
        st = checkpoint_status(run_dir)
        assert st["present"] and st["done"] == ["S3", "baselines"]
        assert st["bytes"] == (run_dir / CHECKPOINT).stat().st_size and st["saved_utc"]
        assert st["matches"] == {"data": None, "code": True, "config": None} and st["changed_sections"] is None
        st = checkpoint_status(run_dir, cfg)                            # fast comes from the manifest
        assert st["changed_sections"] == [] and st["fingerprint_match"] is True
        assert st["matches"] == {"data": True, "code": True, "config": True}
        other = copy.deepcopy(cfg)
        other.raw["stacker"]["epochs"] = 25
        st = checkpoint_status(run_dir, other, fast=True)
        assert st["changed_sections"] == ["core"] and st["fingerprint_match"] is False
        assert st["matches"] == {"data": True, "code": True, "config": False}
        assert checkpoint_status(run_dir, cfg, fast=False)["fingerprint_match"] is False
        (run_dir / CHECKPOINT_JSON).unlink()                            # a checkpoint from before the sidecar
        st = checkpoint_status(run_dir, cfg)
        assert st["present"] and st["done"] is None and st["matches"]["code"] is None
        assert checkpoint_status(tmp_path / "nowhere")["present"] is False
    finally:
        mp.undo()


def test_resume_mismatch_is_decided_by_the_sidecar(demo, tmp_path, events, no_unpickle):
    from sparc.core import pipeline

    cfg = _cfg(demo)
    pipeline.run_core(copy.deepcopy(cfg), stages=("S0", "S1"), run_dir=tmp_path / "run")
    # S0–S1 save no checkpoint; write one for a different config by hand through the real writer
    cfg_a = copy.deepcopy(cfg)
    pipeline.apply_mode_overrides(cfg_a, fast=True)
    state = {"fingerprint": pipeline._fingerprint(cfg_a, True, None), "done": {"S3"}}
    pipeline._save_checkpoint(tmp_path / "run", state, pipeline.fingerprint_sections(cfg_a, True))
    events.clear()
    cfg_b = copy.deepcopy(cfg)
    cfg_b.raw["causal"]["n_boot"] = 7
    res = pipeline.run_core(cfg_b, stages=("S0", "S1"), fast=True, resume=True, run_dir=tmp_path / "run")
    mism = [e for e in events if e["type"] == "checkpoint"]
    assert [e["action"] for e in mism] == ["mismatch"] and mism[0]["changed_sections"] == ["s6"]
    assert any(e["type"] == "warning" and e["code"] == "checkpoint.mismatch" for e in events)
    assert [e["stage"] for e in events if e["type"] == "stage.start"] == ["S0", "S1", "finish"]
    assert res.influence is not None


def test_resume_loads_the_checkpoint_and_marks_cached_stages(demo, tmp_path, events):
    from sparc.core.pipeline import run_core

    cfg = _cfg(demo)
    first = run_core(copy.deepcopy(cfg), stages=S0_S3, fast=True, run_dir=tmp_path / "run")
    events.clear()
    again = run_core(copy.deepcopy(cfg), stages=S0_S3, fast=True, resume=True, run_dir=tmp_path / "run")
    assert [e["action"] for e in events if e["type"] == "checkpoint"] == ["loaded"]
    skips = {e["stage"]: e["reason"] for e in events if e["type"] == "stage.skip"}
    assert skips == {"S1": "checkpoint", "S2_S3": "checkpoint", "baselines": "checkpoint",
                     "cv_curve": "disabled_by_config:cv.distance_curve.enabled", "S4": "not_requested",
                     "S5": "not_requested", "climate": "requires_S5", "S6": "not_requested", "S7": "not_requested"}
    plans = [e for e in events if e["type"] == "run.plan"]
    states = {n["id"]: n["state"] for n in plans[0]["nodes"]}     # the first plan already knows (sidecar)
    assert states["S1"] == states["S2_S3"] == states["baselines"] == "cached"
    assert plans[-1]["total_units"] == {"s0_load": 1}
    assert again.ensemble.oof_pred.tolist() == first.ensemble.oof_pred.tolist()


def test_a_failed_save_is_not_claimed_as_done(demo, tmp_path, events, monkeypatch):
    """When a checkpoint save fails (a full disk, a second signal), ``run_state.json`` and ``run.end`` keep the
    done set of the last checkpoint that was written: they never claim a stage the checkpoint does not hold."""
    from sparc.core import pipeline

    real = pipeline._save_checkpoint

    def failing(run_dir, state, sections=None):
        if "baselines" in state["done"]:
            raise OSError(28, "No space left on device")
        return real(run_dir, state, sections)

    monkeypatch.setattr(pipeline, "_save_checkpoint", failing)
    run_dir = tmp_path / "run"
    with pytest.raises(OSError):
        pipeline.run_core(_cfg(demo), stages=S0_S3, fast=True, run_dir=run_dir)
    rs = json.loads((run_dir / pipeline.RUN_STATE).read_text())
    side = json.loads((run_dir / pipeline.CHECKPOINT_JSON).read_text())
    assert rs["status"] == "failed" and rs["error"]["type"] == "OSError"
    assert rs["done"] == side["done"] == ["S3"]
    end = next(e for e in events if e["type"] == "run.end")
    assert end["status"] == "failed" and end["done"] == ["S3"]
