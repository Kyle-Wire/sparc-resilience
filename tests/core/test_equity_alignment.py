"""``optimize.equity_column`` is carried through S0 as an auxiliary column (SPEC §11 item 14): the scores stay
aligned with the cells by id after dropped non-finite rows, a fast window and coarse cells."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _score(ids) -> np.ndarray:
    return (np.asarray(ids, float) * 37 % 101) / 100.0              # a deterministic score per id


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    from sparc.core.config import load_core_config
    from sparc.core.synthetic import write_demo_project

    d = write_demo_project(tmp_path_factory.mktemp("equity"), n=32, seed=1)
    cfg = load_core_config(d["config_path"])
    df = pd.read_csv(d["files"]["data"])
    df["equity"] = _score(df["id"])
    rng = np.random.default_rng(0)
    bad = rng.choice(len(df), 60, replace=False)
    df.loc[bad[:30], "canopy"] = np.nan                            # dropped rows (non-finite predictor)
    df.loc[bad[30:], "T"] = np.inf                                 # dropped rows (non-finite target)
    df.loc[rng.choice(len(df), 25, replace=False), "equity"] = np.nan   # missing scores are kept as NaN
    cfg.raw["optimize"]["equity_column"] = "equity"
    return cfg, df


def test_aux_column_follows_dropped_rows_and_window(demo):
    import copy

    from sparc.core.data import prepare_frame

    cfg, df = demo
    c = copy.deepcopy(cfg)
    c.raw["data"]["subsample"] = 400                                # a centred window, like fast mode
    data = prepare_frame(df, c)
    assert data.n < len(df) - 60 and "equity" not in data.frame.columns and "equity" not in data.X.columns
    by_id = df.set_index("id")["equity"]
    np.testing.assert_array_equal(data.aux["equity"], by_id.loc[data.ids].to_numpy())
    ok = np.isfinite(data.aux["equity"])
    np.testing.assert_allclose(data.aux["equity"][ok], _score(data.ids)[ok])
    assert data.with_frame(data.frame, c.raw.get("encodings")).aux is data.aux


def test_aux_column_is_averaged_onto_coarse_cells(demo):
    import copy

    from sparc.core.data import _fine_to_coarse, prepare_frame

    cfg, df = demo
    c = copy.deepcopy(cfg)
    c.raw["data"]["coarse_m"] = 90.0
    data = prepare_frame(df, c)
    keep = np.isfinite(df[["T", "x", "y"] + c.predictors].to_numpy(float)).all(axis=1)
    fine = df.loc[keep].reset_index(drop=True)
    inv, members, _, _ = _fine_to_coarse(fine["x"].to_numpy(float), fine["y"].to_numpy(float), 30.0, 90.0)
    expect = fine.assign(cell=inv).groupby("cell")["equity"].mean().to_numpy()   # mean of the finite members
    np.testing.assert_allclose(data.aux["equity"], expect, equal_nan=True)
    assert data.n == members.size


def test_pipeline_passes_scores_aligned_by_id(demo, monkeypatch):
    import copy

    import sparc.core.optimize as optimize
    from sparc.core.pipeline import equity_scores, run_core

    cfg, df = demo
    c = copy.deepcopy(cfg)
    c.raw["models"] = {"ols": True, "mgwr": False, "gwrf": False, "gam": True, "physics": False}
    c.raw["stacker"].update(epochs=20, tune_lambda=[0.0])
    c.raw["cv"]["baselines"] = False
    c.raw["influence"]["n_perm"] = 3
    c.raw["actionable"] = {"canopy": {**c.raw["actionable"]["canopy"], "doses": [0, 10, 20]}}
    c.raw["data"]["subsample"] = 500
    c.raw["optimize"]["equity_focus"] = 0.5
    seen = {}
    real = optimize.optimise_allocation

    def spy(engine, vr, budget, **kw):
        seen["scores"] = kw["equity_scores"]
        seen["ids"] = engine.data.ids
        return real(engine, vr, budget, **kw)

    monkeypatch.setattr(optimize, "optimise_allocation", spy)
    res = run_core(c, stages=("S4", "S7"), frame=df, write=False)
    assert "scores" in seen and res.optimize.get("n_cells_treated", 0) > 0
    ids = seen["ids"]
    raw = df.set_index("id")["equity"].loc[ids].to_numpy()
    ok = np.isfinite(raw)
    assert not ok.all()                                             # some scores were missing in the input
    np.testing.assert_array_equal(seen["scores"][ok], raw[ok])
    np.testing.assert_allclose(seen["scores"][~ok], raw[ok].mean())   # missing scores → the mean score
    np.testing.assert_array_equal(seen["scores"], equity_scores(c, res.data))


def test_no_equity_column_means_no_scores(demo):
    import copy

    from sparc.core.data import prepare_frame
    from sparc.core.pipeline import equity_scores

    cfg, df = demo
    c = copy.deepcopy(cfg)
    c.raw["optimize"]["equity_column"] = None
    data = prepare_frame(df, c)
    assert data.aux is None and equity_scores(c, data) is None
    c.raw["optimize"]["equity_column"] = "not_a_column"
    assert equity_scores(c, prepare_frame(df, c)) is None
