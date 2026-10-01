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
    assert can["specification"] == pytest.approx([-0.42, -0.22])
    assert can["attribution"] == pytest.approx([-0.375, -0.1875])
    assert can["envelope"] == pytest.approx([-0.42, -0.1875]) and can["envelope_excludes_zero"]
    assert "attribution" not in alb and "specification" not in alb          # only canopy is simulated
    assert "⚠" in uncertainty_markdown(u, "°F")
