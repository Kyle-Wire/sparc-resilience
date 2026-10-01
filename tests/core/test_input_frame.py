"""Frame-input runs are re-readable (SPEC §11 item 22): ``run_core(frame=…, write=True)`` keeps its table in
``input_frame.parquet`` and ``baselines.load_run`` rebuilds the data and folds from it."""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_placebo_style_frame_run_is_rebuilt_from_input_frame(tmp_path):
    from sparc.core.baselines import load_run
    from sparc.core.config import load_core_config
    from sparc.core.pipeline import checkpoint_status, run_core
    from sparc.core.placebo import shift_layer
    from sparc.core.synthetic import write_demo_project

    demo = write_demo_project(tmp_path / "project", n=32, seed=2)
    cfg = load_core_config(demo["config_path"])
    cfg.raw["models"] = {"ols": True, "mgwr": False, "gwrf": False, "gam": False, "physics": False}
    cfg.raw["stacker"].update(epochs=20, tune_lambda=[0.0])
    cfg.raw["cv"]["baselines"] = ["idw"]
    df = pd.read_csv(demo["files"]["data"])
    df = df.iloc[np.random.default_rng(0).permutation(len(df))[: len(df) - 40]].reset_index(drop=True)
    from sparc.core.data import prepare_frame

    full = prepare_frame(df, cfg)
    frame = df.copy()
    frame["canopy"] = shift_layer(full.frame["canopy"].to_numpy(float), full.grid)   # differs from data.path
    run_dir = tmp_path / "children" / "demo_placebo_shift"
    res = run_core(cfg, stages=("S0", "S1", "S2", "S3"), frame=frame, run_dir=run_dir,
                   run_meta={"role": "placebo:shift"})
    assert (run_dir / "input_frame.parquet").exists()
    assert res.manifest["provenance"]["input_frame"] == "input_frame.parquet"
    assert res.manifest["provenance"]["input_kind"] == "frame"
    assert res.manifest["run_meta"] == {"role": "placebo:shift"}
    pd.testing.assert_frame_equal(pd.read_parquet(run_dir / "input_frame.parquet"), frame)

    cfg2 = load_core_config(demo["config_path"])
    cfg2.raw.update({k: res.cfg.raw[k] for k in ("models", "stacker", "cv")})
    data, folds, m, pred = load_run(run_dir, cfg2)
    np.testing.assert_array_equal(data.ids, res.data.ids)
    np.testing.assert_array_equal(data.ids, pred["id"].to_numpy())
    np.testing.assert_array_equal(folds.fold_id, res.ensemble.folds.fold_id)
    np.testing.assert_array_equal(data.frame["canopy"].to_numpy(), res.data.frame["canopy"].to_numpy())
    assert not np.array_equal(data.frame["canopy"].to_numpy(), full.frame["canopy"].to_numpy()[: data.n])

    st = checkpoint_status(run_dir, res.cfg, fast=False)
    assert st["present"] and st["fingerprint_match"] is True and st["changed_sections"] == []


def test_file_runs_write_no_input_frame(tmp_path):
    from sparc.core.config import load_core_config
    from sparc.core.pipeline import run_core
    from sparc.core.synthetic import write_demo_project

    demo = write_demo_project(tmp_path / "project", n=16, seed=0)
    res = run_core(load_core_config(demo["config_path"]), stages=("S0", "S1"), run_dir=tmp_path / "run")
    assert not (res.run_dir / "input_frame.parquet").exists()
    assert "input_frame" not in res.manifest["provenance"]


def test_input_frame_keeps_the_index_so_the_data_identity_matches(tmp_path):
    """A frame with a non-default index (e.g. rows filtered out upstream) reads back hash-identical, so
    ``checkpoint_status`` recognises the data of the run."""
    from sparc.core.config import load_core_config
    from sparc.core.pipeline import _data_identity, checkpoint_status, run_core
    from sparc.core.synthetic import write_demo_project

    demo = write_demo_project(tmp_path / "project", n=24, seed=1)
    cfg = load_core_config(demo["config_path"])
    cfg.raw["models"] = {"ols": True, "mgwr": False, "gwrf": False, "gam": False, "physics": False}
    cfg.raw["stacker"].update(epochs=10, tune_lambda=[0.0])
    cfg.raw["cv"]["baselines"] = False
    df = pd.read_csv(demo["files"]["data"])
    frame = df[df["id"] % 7 != 3]                                   # a gappy index, not a RangeIndex
    assert not isinstance(frame.index, pd.RangeIndex)
    run_dir = tmp_path / "run"
    res = run_core(cfg, stages=("S0", "S1", "S2", "S3"), frame=frame, run_dir=run_dir)
    back = pd.read_parquet(run_dir / "input_frame.parquet")
    pd.testing.assert_frame_equal(back, frame)
    assert _data_identity(res.cfg, back) == _data_identity(res.cfg, frame)
    st = checkpoint_status(run_dir, res.cfg, fast=False)
    assert st["matches"]["data"] is True and st["fingerprint_match"] is True
