"""Regression tests for the G6 saturation fixes (roadmap Appendix A25, A27).

A25 — the √ diminishing-return taper in ScenarioSimulator is opt-in
      (identity unless a threshold is configured) and is skipped when a
      usable fitted condition curve exists.
A27 — causal_pdp builds R(d) = ∫_{t0}^{d} Ê[τ | T≈s] ds instead of the
      linear-by-construction mean(τ)·(d − t0), so saturation is detectable.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from sparc.causal.causal_pdp import (
    _detect_saturation,
    causal_pdp,
    causal_pdp_bayesian,
    causal_pdp_frequentist,
    tau_by_level_response,
)
from sparc.interventions.scenario_simulator import ScenarioSimulator


# ---------------------------------------------------------------------------
# A25 — opt-in taper
# ---------------------------------------------------------------------------


def _bare_sim(caps=None, curves=None) -> ScenarioSimulator:
    sim = ScenarioSimulator.__new__(ScenarioSimulator)
    sim.config = {"caps": caps or {}}
    sim._condition_curves = curves or {}
    sim._condition_curve_min_r2 = 0.5
    return sim


_GOOD_CURVE = {"r2": 0.9, "grid_values": [0, 50, 100], "pdp_values": [0, -1, -1.2]}


class TestOptInTaper:
    def test_no_threshold_configured_is_identity(self):
        sim = _bare_sim()
        thr = sim._get_diminishing_threshold("Pct_Canopy")
        assert math.isinf(thr)
        delta = np.array([-40.0, -5.0, 0.0, 5.0, 40.0])
        np.testing.assert_array_equal(sim._diminishing_return(delta, thr), delta)

    def test_explicit_threshold_applies_taper(self):
        sim = _bare_sim(caps={"diminishing_return_thresholds": {"Pct_Canopy": 10.0}})
        thr = sim._get_diminishing_threshold("Pct_Canopy")
        assert thr == 10.0
        out = sim._diminishing_return(np.array([5.0, 20.0]), thr)
        assert out[0] == 5.0
        assert out[1] == pytest.approx(10.0 + math.sqrt(10.0) * math.sqrt(10.0))
        # other variables stay untapered
        assert math.isinf(sim._get_diminishing_threshold("Albedo"))

    def test_explicit_default_key_is_honoured(self):
        sim = _bare_sim(caps={"diminishing_return_thresholds": {"default": 7.0}})
        assert sim._get_diminishing_threshold("Albedo") == 7.0

    def test_usable_condition_curve_skips_taper(self):
        sim = _bare_sim(
            caps={"diminishing_return_thresholds": {"Pct_Canopy": 10.0}},
            curves={"Pct_Canopy": dict(_GOOD_CURVE)},
        )
        assert math.isinf(sim._get_diminishing_threshold("Pct_Canopy"))

    def test_low_r2_curve_does_not_skip_configured_taper(self):
        sim = _bare_sim(
            caps={"diminishing_return_thresholds": {"Pct_Canopy": 10.0}},
            curves={"Pct_Canopy": dict(_GOOD_CURVE, r2=0.1)},
        )
        assert sim._get_diminishing_threshold("Pct_Canopy") == 10.0

    @pytest.mark.parametrize("path", [
        "sparc/templates/uhi/physics/caps.yml",
        "templates/uhi/physics/caps.yml",
        "sparc/interventions/config/caps.yml",
    ])
    def test_shipped_caps_do_not_enable_taper(self, path):
        import yaml
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        caps = yaml.safe_load((root / path).read_text(encoding="utf-8"))
        assert "diminishing_return_thresholds" not in caps
        assert caps["combined_constraints"]["canopy_impervious_sum"]["enforcement"] == "warning"

    def test_template_copies_identical(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        a = (root / "sparc/templates/uhi/physics/caps.yml").read_bytes()
        b = (root / "templates/uhi/physics/caps.yml").read_bytes()
        assert a == b


# ---------------------------------------------------------------------------
# A27 — τ-by-level integral dose response
# ---------------------------------------------------------------------------


def _treatment(n=4000, seed=0):
    return np.random.default_rng(seed).uniform(0.0, 100.0, n)


class _FakeBayes:
    def __init__(self, tau_post):
        self._tau_post = tau_post
        self._posterior_samples = {"Pct_Canopy": tau_post}

    def posterior_samples(self, treatment):
        return self._tau_post


class TestTauByLevelIntegral:
    def test_declining_marginal_effect_gives_concave_curve_with_knee(self):
        T = _treatment()
        tau = 1.0 - 0.01 * T                       # a − b·T
        est = SimpleNamespace(cate_estimates={"Pct_Canopy": tau},
                              cate_intervals={"Pct_Canopy": (tau - 0.1, tau + 0.1)})
        curve = causal_pdp_frequentist(est, "Pct_Canopy", T)
        slopes = np.gradient(curve.response_mean, curve.dose_grid)
        assert np.all(np.diff(slopes) < 0)         # concave
        assert curve.saturation_index is not None
        assert curve.saturation_dose is not None
        # analytic knee: slope 1 − 0.01 d drops below ½·peak(≈0.95) near d ≈ 52
        assert 40.0 < curve.saturation_dose < 65.0
        assert curve.method == "tau_by_level_integral"
        assert curve.to_payload()["method"] == "tau_by_level_integral"
        assert np.all(curve.response_hdi_lo <= curve.response_mean + 1e-12)
        assert np.all(curve.response_hdi_hi >= curve.response_mean - 1e-12)

    def test_constant_tau_gives_straight_line_and_no_knee(self):
        T = _treatment()
        tau = np.full_like(T, 0.3)
        est = SimpleNamespace(cate_estimates={"Pct_Canopy": tau}, cate_intervals={})
        curve = causal_pdp_frequentist(est, "Pct_Canopy", T)
        t0 = curve.diagnostics["baseline_dose"]
        np.testing.assert_allclose(curve.response_mean, 0.3 * (curve.dose_grid - t0), atol=1e-9)
        assert curve.saturation_index is None
        idx, _, _ = _detect_saturation(curve.response_mean, curve.dose_grid)
        assert idx is None

    def test_sign_correct_below_baseline(self):
        T = _treatment()
        R, _ = tau_by_level_response(np.full_like(T, 2.0), T, np.array([10.0, 90.0]), t0=50.0)
        assert R[0] == pytest.approx(-80.0)
        assert R[1] == pytest.approx(80.0)

    def test_bayesian_path_per_draw(self):
        T = _treatment(1500)
        rng = np.random.default_rng(1)
        tau_post = (1.0 - 0.01 * T)[None, :] + rng.normal(0, 0.05, (40, T.size))
        curve = causal_pdp(_FakeBayes(tau_post), "Pct_Canopy", T)
        assert curve.source == "bayesian"
        assert curve.saturation_dose is not None
        assert curve.diagnostics["method"] == "tau_by_level_integral"
        assert np.all(curve.response_hdi_lo <= curve.response_hdi_hi)

    def test_bayesian_constant_tau_no_knee_and_per_cell_shape(self):
        T = _treatment(300)
        tau_post = np.full((20, T.size), -0.4)
        curve = causal_pdp_bayesian(_FakeBayes(tau_post), "Pct_Canopy", T,
                                    keep_per_cell=True)
        assert curve.saturation_index is None
        assert curve.per_cell_mean.shape == (T.size, curve.dose_grid.size)
        t0 = curve.diagnostics["baseline_dose"]
        np.testing.assert_allclose(curve.per_cell_mean[0], -0.4 * (curve.dose_grid - t0), atol=1e-9)
