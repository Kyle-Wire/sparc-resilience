"""Uncertainty report: components kept apart and combined only as an envelope."""

from __future__ import annotations

import pytest

from sparc.core.uncertainty import scenario_uncertainty, uncertainty_markdown


def test_components_and_envelope():
    m = {"config": {"physics": {"roles": {"canopy": "can"}}},
         "scenarios": [
             {"name": "Canopy +10", "mean_delta": -0.30, "mean_delta_se": 0.05, "mean_realized": {"can": 10.0},
              "causal_linear": {"lo": -0.25, "hi": 0.0, "model_within": False}},
             {"name": "Albedo +0.1", "mean_delta": -0.80, "mean_delta_se": 0.10, "mean_realized": {"alb": 0.1}}]}
    mv = {"effects": {"Canopy +10": {"baseline": -0.28, "values": {"baseline": -0.28, "a": -0.20, "b": -0.40},
                                     "sign_stability": 1.0}}}
    sc = {"bias_correction": {"share_range": [0.8, 1.6], "stable": False}}
    u = scenario_uncertainty(m, mv, sc)
    can, alb = u["scenarios"]
    assert can["estimation_95"] == pytest.approx([-0.398, -0.202])
    assert can["specification"] == pytest.approx([-0.30 * 0.40 / 0.28, -0.30 * 0.20 / 0.28])   # ratios
    assert can["specification_mode"] == "ratio"
    assert can["attribution"] == pytest.approx([-0.375, -0.1875])
    assert can["envelope"] == pytest.approx([-0.30 * 0.40 / 0.28, -0.1875]) and can["envelope_excludes_zero"]
    assert "attribution" not in alb and "specification" not in alb          # only canopy is simulated
    assert "⚠" in uncertainty_markdown(u, "°F")


def test_offset_fallback_and_null_artifact():
    m = {"config": {"physics": {"roles": {"canopy": "can"}}},
         "scenarios": [{"name": "Canopy +5", "mean_delta": -0.10, "mean_delta_se": 0.04, "mean_realized": {"can": 5.0}},
                       {"name": "Canopy +20", "mean_delta": -0.80, "mean_delta_se": 0.05,
                        "mean_realized": {"can": 20.0}}]}
    mv = {"effects": {"Canopy +5": {"baseline": 0.0, "values": {"baseline": 0.0, "a": -0.05}, "sign_stability": 0.5}}}
    sc = {"generators": {"null": {"n": 12, "null_mean_delta": -0.2, "null_mean_delta_se": 0.05},
                         "null/direct": {"n": 12, "null_mean_delta": 0.0, "null_mean_delta_se": 0.02},
                         "physics": {"n": 20, "share_median": 1.1}}}
    small, big = scenario_uncertainty(m, mv, sc)["scenarios"]
    assert small["specification_mode"] == "offset" and small["specification"] == pytest.approx([-0.15, -0.10])
    rf, direct = small["null_artifact"]
    assert rf["product"] == "rf" and rf["delta"] == pytest.approx(-0.1) and not rf["distinguishable"]
    assert direct["product"] == "direct" and direct["distinguishable"]
    assert big["null_artifact"][0]["delta"] == pytest.approx(-0.4) and big["null_artifact"][0]["distinguishable"]
    md = uncertainty_markdown({"scenarios": [small, big]}, "°F")
    assert "spurious change" in md and "direct (no forest)" in md
