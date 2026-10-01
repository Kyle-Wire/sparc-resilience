"""Negative-control layers and placebo-run configuration."""

from __future__ import annotations

import numpy as np
import pytest

from sparc.core import placebo as P


def test_placebo_layers_keep_distribution_and_break_location(synthetic_core_data):
    cfg, data = synthetic_core_data
    can = data.frame["canopy"].to_numpy(float)
    for layer in (P.shift_layer(can, data.grid), P.rotate_layer(can, data.grid)):
        assert layer.shape == can.shape and np.isfinite(layer).all()
        assert abs(np.median(layer) - np.median(can)) < 0.25 * can.std()     # same kind of values
        assert abs(P.layer_correlation(layer, can)) < 0.5                     # different place
    g = P.grf_layer(data.grid, can, 300.0, seed=0)
    assert np.allclose(np.sort(g), np.sort(can))                              # rank-matched marginal
    # smooth: neighbouring cells agree far more than random pairs
    r = data.grid.rasterize(g)
    dx_corr = np.corrcoef(r[:, :-1][np.isfinite(r[:, :-1]) & np.isfinite(r[:, 1:])],
                          r[:, 1:][np.isfinite(r[:, :-1]) & np.isfinite(r[:, 1:])])[0, 1]
    assert dx_corr > 0.8


def test_placebo_config_builds_sd_dose_ladders_and_treatments(synthetic_core_data):
    cfg, data = synthetic_core_data
    sds = {"canopy": 20.0, P.GRF: 20.0}
    pc = P.placebo_config(cfg, "grf", ["canopy", P.GRF], sds)
    assert P.GRF in pc.predictors and pc.name.endswith("_placebo_grf")
    assert pc.actionable[P.GRF]["doses"] == [0.0, 10.0, 20.0, 40.0]
    assert pc.raw["causal"]["treatments"] == ["canopy", P.GRF]
    assert pc.raw["causal"]["contrast"][P.GRF] == pytest.approx(20.0)
    assert P.GRF not in pc.raw["causal"]["confounders"][P.GRF] and "ndvi" not in pc.raw["causal"]["confounders"][P.GRF]
    assert not pc.raw["climate"]["enabled"] and not pc.raw["optimize"]["enabled"]
    assert cfg.name == "synthetic_city" and P.GRF not in cfg.predictors          # original untouched
    names = [s["name"] for s in pc.raw["scenarios"]]
    assert names[0].startswith("canopy (sd ")


def test_judge_uses_se_and_real_effect_scale():
    real = {"model": [{"dose_sd": 1.0, "mean_delta": -1.0, "se": 0.1}]}
    small = {"model": [{"dose_sd": 1.0, "mean_delta": 0.05, "se": 0.01}],
             "causal_theta_sum_per_sd": 0.01, "causal_se_per_sd": 0.02}
    v = P.judge(small, real)
    assert not v["model_within_2se"] and v["model_below_10pct_of_real"] and v["model_pass"] and v["causal_ci_covers_zero"]
    big = {"model": [{"dose_sd": 1.0, "mean_delta": -0.5, "se": 0.05}],
           "causal_theta_sum_per_sd": -0.4, "causal_se_per_sd": 0.05}
    v = P.judge(big, real)
    assert not v["model_pass"] and not v["causal_ci_covers_zero"]
