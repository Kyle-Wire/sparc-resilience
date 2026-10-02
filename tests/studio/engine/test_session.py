"""``sparc.core.session`` (SPEC §7.6, §11 item 17): config resolution, open_run parity, base_fold, threads."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest


def test_config_for_run_prefers_the_launch_snapshot(synth_run):
    from sparc.core.session import config_for_run

    rid, rd = synth_run
    cfg = config_for_run(rd)
    assert str(cfg.base_dir) == str(rd.parent.parent)          # the project dir of launch.json
    assert int(cfg.raw["cv"]["n_folds"]) <= 3                   # fast mode overrides applied
    (rd / "studio" / "launch.json").unlink()
    cfg2 = config_for_run(rd)                                   # manifest config next
    assert cfg2.raw["data"]["target"] == cfg.raw["data"]["target"]


def _emulator_path_engine(run_dir, cfg):
    """The loading path of ``emulator.emulator_for_run`` (load_run + pickle + mediators + ScenarioEngine)."""
    from sparc.core.baselines import load_run
    from sparc.core.mediators import MediatorChain
    from sparc.core.scenarios import ScenarioEngine

    data, _folds, _m, _pred = load_run(run_dir, cfg)
    with open(run_dir / "checkpoint.pkl", "rb") as fh:
        st = pickle.load(fh)
    ens, inf = st["ensemble"], st["influence"]
    med = MediatorChain(cfg.mediators).fit(data.frame) if cfg.mediators else None
    return ScenarioEngine(data, cfg, ens, dict(inf.ranges_m), med)


@pytest.mark.slow
def test_open_run_matches_emulator_path_and_s5(engine_run_dir):
    import copy

    from sparc.core import progress
    from sparc.core.scenarios import specs_from_config
    from sparc.core.session import config_for_run, open_run

    events = []
    progress.configure(events.append, heartbeat_s=0)
    try:
        session = open_run(engine_run_dir, threads=1)
    finally:
        progress.reset()
    tasks = [e["name"] for e in events if e["type"] == "task.start"]
    assert tasks[:4] == ["load_run", "unpickle", "mediators", "engine_init"]
    ticks = [e for e in events if e["type"] == "tick" and e["unit"] == "unpickle"]
    assert ticks and ticks[-1]["k"] == ticks[-1]["n"] and ticks[-1]["label"].endswith(" MB")
    assert any(e["type"] == "task.end" and e.get("unit") == "mediator_fit" and e["status"] == "ok" for e in events)
    other = _emulator_path_engine(engine_run_dir, copy.deepcopy(config_for_run(engine_run_dir)))
    stored = pd.read_parquet(engine_run_dir / "scenario_deltas.parquet")
    specs = specs_from_config(session.cfg)
    assert specs
    for spec in specs:
        a = session.engine.run(spec).delta
        b = other.run(spec).delta
        np.testing.assert_allclose(a, b, atol=1e-9, rtol=0)
        np.testing.assert_allclose(a, stored[spec.name].to_numpy(float), atol=1e-9, rtol=0)
    # the cached baseline pass gives identical results
    again = open_run(engine_run_dir, threads=1, base_fold=session.base_fold)
    np.testing.assert_array_equal(again.base_fold, session.base_fold)
    for spec in specs[:3]:
        np.testing.assert_array_equal(again.engine.run(spec).delta, session.engine.run(spec).delta)
    assert again.load_seconds["engine_init"] <= session.load_seconds["engine_init"] + 0.5
    assert session.code_match is True and session.checkpoint_key


@pytest.mark.slow
def test_gwrf_threads_override(engine_run_dir):
    from sparc.core.session import open_run

    s = open_run(engine_run_dir, threads=3)
    models = s.gwrf_models()
    assert models, "the demo config fits GWRF base models"
    assert all(m.n_jobs == 3 for m in models)
    s.set_threads(2)
    assert all(m.n_jobs == 2 for m in models)


@pytest.mark.slow
def test_incompatible_checkpoint(tmp_path, engine_run_dir):
    import shutil

    from sparc.core.session import IncompatibleCheckpoint, open_run
    from tests.studio.engine.conftest import write_incompatible_checkpoint

    rd = tmp_path / "run"
    shutil.copytree(engine_run_dir, rd)
    write_incompatible_checkpoint(rd)
    with pytest.raises(IncompatibleCheckpoint) as exc:
        open_run(rd, threads=1)
    assert "GhostEnsemble" in str(exc.value)
