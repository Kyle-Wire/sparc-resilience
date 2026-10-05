"""Heat index, NWS categories, heat-risk exposure and plain verdicts."""

import numpy as np
import pytest

from sparc.core import forcing, heat


@pytest.mark.parametrize("t,rh", [(70, 50), (80, 40), (85, 40), (95, 10), (84, 90), (100, 60), (110, 5)])
def test_heat_index_matches_the_scalar_nws_formula(t, rh):
    assert float(heat.heat_index_f(t, rh)) == pytest.approx(forcing.heat_index_F(t, rh), abs=1e-9)


def test_nws_reference_values_and_categories():
    assert float(heat.heat_index_f(90, 70)) == pytest.approx(105.9, abs=0.5)     # NWS chart: 106 °F
    assert float(heat.heat_index_f(96, 40)) == pytest.approx(101.1, abs=0.5)     # NWS chart: 101 °F
    assert heat.category_codes([79.9, 80, 89.9, 90, 102.9, 103, 124.9, 125]).tolist() == [0, 1, 1, 2, 2, 3, 3, 4]


def test_humidity_from_dewpoint_falls_with_temperature():
    rh = heat.rh_from_dewpoint([20.0, 30.0, 35.0], 16.7)
    assert rh[0] > rh[1] > rh[2] and 40 < rh[1] < 50


def test_heat_risk_cases_people_and_humidity_band():
    t = np.array([86.0, 90.0, 94.0, 98.0])
    people = np.array([100.0, 200.0, 300.0, 400.0])
    cool = np.array([-1.0, -1.0, -2.0, -2.0])
    r = heat.heat_risk(t, "degF", 18.0, people=people, futures={"mid": 3.0}, adaptation=cool)
    today = next(c for c in r["cases"] if c["case"] == "today" and not c["adapted"])
    assert sum(today["people"].values()) == pytest.approx(1000.0)
    mid = {(c["adapted"], c["humidity"]): c for c in r["cases"] if c["case"] == "mid"}
    assert set(mid) == {(a, h) for a in (False, True) for h in heat.HUMIDITY_ASSUMPTIONS}
    # constant relative humidity is the upper bound; the package lowers heat index
    assert mid[(False, "constant_rh")]["person_mean_hi"] > mid[(False, "constant_dewpoint")]["person_mean_hi"]
    assert mid[(True, "constant_dewpoint")]["person_mean_hi"] < mid[(False, "constant_dewpoint")]["person_mean_hi"]
    # Celsius targets give the same answer as their Fahrenheit equivalent
    rc = heat.heat_risk((t - 32) * 5 / 9, "degC", 18.0, people=people)
    assert rc["cases"][0]["person_mean_hi"] == pytest.approx(today["person_mean_hi"])


def test_campaign_dewpoint_prefers_the_station():
    td, src = heat.campaign_dewpoint({"station": {"td_C": 16.7, "name": "PVD"}, "era5": {"d2m_C": 16.1}})
    assert td == 16.7 and "PVD" in src
    assert heat.campaign_dewpoint({"era5": {"d2m_C": 16.1}})[0] == 16.1
    assert heat.campaign_dewpoint({}) == (None, None)


def _row(**kw):
    base = {"scenario": "s", "estimate": -0.4, "estimation_95": [-0.78, -0.03], "envelope": [-0.78, -0.03],
            "sign_stability": 1.0, "frac_extrapolated": 0.05}
    return {**base, **kw}


def test_verdicts():
    assert heat.effect_verdict(_row())["verdict"] == "robust"
    assert heat.effect_verdict(_row(envelope=[-0.9, 0.1]))["verdict"] == "direction"
    nul = _row(null_artifact=[{"product": "rf", "delta": -0.18, "distinguishable": False}])
    v = heat.effect_verdict(nul)
    assert v["verdict"] == "not_established" and "no planted effect" in v["reasons"][0]
    assert heat.effect_verdict(_row(frac_extrapolated=1.0))["verdict"] == "not_established"
    q = heat.effect_verdict(_row(frac_extrapolated=0.3, model_within_causal=False))
    assert q["verdict"] == "robust" and len(q["qualifiers"]) == 2
    assert heat.effect_verdict(_row(estimation_95=[-0.5, 0.2], envelope=[-0.6, 0.3], sign_stability=0.5))[
        "verdict"] == "not_established"
    assert heat.effect_verdict({"scenario": "x"})["verdict"] == "unknown"
