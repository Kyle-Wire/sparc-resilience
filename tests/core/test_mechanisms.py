"""Physics mechanism register and physical-units panel (``sparc.core.mechanisms``; PHYSICS_PLAN W0 and W3).

Unit tests drive :func:`evaluate` with hand-made per-fold parameters (Providence-like numbers: a = 45 °F per
unit q, SW↓ = 635.8 W/m², ERA5 2 m temperature 304.12 K); the integration test runs a small synthetic
pipeline and checks ``mechanisms.json``, the manifest section, the report texts, the results page and the CLI.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sparc.core import mechanisms as M

ROLES = {"albedo": "Albedo", "canopy": "Pct_Canopy", "impervious": "Pct_Impervious", "ndvi": "NDVI",
         "elevation": "Elevation_m", "water_distance": "Distance_from_water_m"}
F_TO_K = 5.0 / 9.0
RHO_CP = 101325.0 / (287.05 * 304.12) * 1005.0           # ≈ 1166 J m⁻³ K⁻¹


def _gamma_F_per_m(k_per_km: float) -> float:
    """γ in °F per metre for a lapse rate in K/km."""
    return k_per_km / 1000.0 / F_TO_K


def _fold(a=45.0, gamma_k_km=-8.0, L=256.0, window="day", warn=None, roles=tuple(ROLES), v=0.0, **kw):
    return {"a": a, "b": 0.0, "gamma": _gamma_F_per_m(gamma_k_km), "beta_x": 0.0, "beta_y": 0.0, "L_m": L,
            "vx_m": v, "vy_m": 0.0, "v_norm_m": abs(v), "s": 0.34, "shade_form": "saturating", "kappa_canopy": 2.0,
            "shade_c0": 0.4, "shade_width": 0.1, "a1": 0.3, "a1n": 0.5, "e0": -0.6, "e1": 2.0, "e2": 1.0, "e3": 1.0,
            "w": 50.0, "window": window, "roles_used": sorted(roles), "fit_warning": warn,
            "priors": {"s": [0.6, 0.3], "a1": [0.3, 0.3], "a1n": [0.5, 0.3]}, "L_bounds_m": [30.0, 2000.0], **kw}


def _folds(a_values=(40.0, 42.0, 45.0, 48.0, 50.0), gammas=None, **kw):
    gammas = gammas or [-8.0] * len(a_values)
    return [_fold(a=a, gamma_k_km=g, **kw) for a, g in zip(a_values, gammas)]


def _cfg(units="degF", kind="product", window="day", sw=635.8, lw=-71.5, t2m=304.12, wind=(-1.158, 6.567),
         roles=None, hours=(15, 16), forcing=True):
    """Providence-like config: the 15–16 EDT forcing window (``forcing=False``: no forcing file at all)."""
    ph = {"enabled": True, "window": window, "sw_down": sw, "lw_net": lw, "wind": list(wind) if wind else None,
          "roles": dict(ROLES if roles is None else roles)}
    if forcing:
        ph["forcing"] = "forcing/providence_2020-07-29.json"
        ph["forcing_info"] = {"file": ph["forcing"], "date": "2020-07-29", "t2m_K": t2m,
                              "hours_local": list(hours) if hours else None}
    return {"data": {"target_units": units, "target_kind": kind}, "physics": ph,
            "actionable": {"Pct_Canopy": {}, "Pct_Impervious": {}, "Albedo": {}}}


def _a_for_ra(ra_values, norm=635.8):
    """Gains a (°F per unit q) that give these r_a (s/m) with Providence's ρc_p."""
    return [r / (F_TO_K * RHO_CP / norm) for r in ra_values]


def _row(reg, rid):
    return next(r for r in reg["rows"] if r["id"] == rid)


def _panel(reg, key):
    return next((r for r in reg["panel"]["rows"] if r["key"] == key), None)


# ---------------------------------------------------------------------------
# r_a = a·ρc_p/norm
# ---------------------------------------------------------------------------

def test_ra_from_providence_numbers_is_about_46_s_per_m_and_supported():
    folds = _folds()
    reg = M.evaluate({}, folds, _cfg())
    ra = _panel(reg, "r_a")
    expected = 45.0 * F_TO_K * RHO_CP / 635.8
    assert reg["panel"]["rho_cp"] == pytest.approx(RHO_CP) and RHO_CP == pytest.approx(1166.5, abs=1.0)
    assert ra["value"] == pytest.approx(expected) and 45.0 < ra["value"] < 47.0
    per = np.array([a * F_TO_K * RHO_CP / 635.8 for a in (40.0, 42.0, 45.0, 48.0, 50.0)])
    K = len(per)
    assert ra["per_fold"] == pytest.approx(per.tolist())
    assert ra["se"] == pytest.approx(math.sqrt((K - 1) / K * np.sum((per - per.mean()) ** 2)))   # jackknife
    assert ra["status"] == "inside" and ra["published"]["low"] == 20.0 and ra["published"]["high"] == 100.0
    r = _row(reg, "M6a")
    assert r["status"] == "supported" and r["label"] == "Supported"
    assert r["statistic"]["unit"] == "s/m" and r["interval"][0] < expected < r["interval"][1]
    assert "masson2000" in r["refs"] and "crawford2018" in r["refs"]


def test_ra_degC_target_matches_the_same_gain_in_kelvin():
    f = M.evaluate({}, _folds(), _cfg(units="degF"))
    c = M.evaluate({}, _folds(a_values=[a * F_TO_K for a in (40.0, 42.0, 45.0, 48.0, 50.0)]), _cfg(units="degC"))
    assert _panel(c, "r_a")["value"] == pytest.approx(_panel(f, "r_a")["value"])
    assert _panel(c, "r_a")["se"] == pytest.approx(_panel(f, "r_a")["se"])
    assert c["panel"]["to_kelvin"] == 1.0 and f["panel"]["to_kelvin"] == pytest.approx(F_TO_K)


def test_night_window_normalises_by_abs_lw_net_and_gamma_is_not_testable():
    reg = M.evaluate({}, _folds(window="night"), _cfg(window="night"))
    ra = _panel(reg, "r_a")
    assert reg["panel"]["norm"] == pytest.approx(71.5) and reg["panel"]["norm_label"] == "|LW_net|"
    assert ra["value"] == pytest.approx(45.0 * F_TO_K * RHO_CP / 71.5)
    # the ≈ 20–100 s/m range is for the daytime, unstable afternoon: no night verdict, the value stays reported
    assert ra["status"] == "not_comparable" and ra["published"] is None and "no night-time" in ra["status_note"]
    m6 = _row(reg, "M6a")
    assert m6["status"] == "not_testable" and "night" in m6["reason"] and "no night-time" in m6["reason"]
    assert m6["statistic"]["value"] == pytest.approx(ra["value"]) and m6["basis"] is None
    assert "|LW_net|" in m6["test"] and "SW↓" not in m6["test"] and "|LW_net|" in ra["derivation"]
    assert "a·ρc_p/|LW_net|" in next(m for m in reg["mechanisms"] if m["id"] == "M6")["chain"]
    assert "a·ρc_p/SW↓" in _row(M.evaluate({}, _folds(), _cfg()), "M6a")["test"]
    g = _row(reg, "M7a")
    assert g["status"] == "not_testable" and "night" in g["reason"] and "inversions" in g["reason"]
    assert g["statistic"]["value"] == pytest.approx(-8.0)            # the value is still reported
    assert _panel(reg, "gamma")["status"] == "not_comparable"
    assert _panel(reg, "a1n") is not None and _panel(reg, "a1") is None and _panel(reg, "s") is None


def test_missing_t2m_falls_back_to_293_K_and_says_so():
    reg = M.evaluate({}, _folds(), _cfg(t2m=None))
    P = reg["panel"]
    assert P["t2m_K"] == pytest.approx(293.15) and not P["t2m_known"] and "assumed" in P["t2m_note"]
    assert _panel(reg, "r_a")["value"] == pytest.approx(45.0 * F_TO_K * M.rho_cp(293.15) / 635.8)
    assert "assumed" in _row(reg, "M6a")["note"]


def test_ra_outside_range_is_contradicted_and_gates_every_lever():
    reg = M.evaluate({}, _folds(a_values=(1000.0, 1010.0, 990.0)), _cfg())
    r = _row(reg, "M6a")
    assert r["status"] == "contradicted" and _panel(reg, "r_a")["status"] == "outside"
    assert reg["gates"]["contradicted"] == ["M6"]
    for lever in ("Pct_Canopy", "Pct_Impervious", "Albedo"):
        g = reg["gates"]["levers"][lever]
        assert "M6" in g["contradicted"] and "do not cite" in g["statement"]


# ---------------------------------------------------------------------------
# γ against the lapse-rate band
# ---------------------------------------------------------------------------

def test_gamma_inside_band_is_supported():
    reg = M.evaluate({}, _folds(gammas=[-8.5, -7.9, -8.1, -7.6, -8.2]), _cfg())
    g = _panel(reg, "gamma")
    assert g["unit"] == "K/km" and g["value"] == pytest.approx(-8.06) and g["status"] == "inside"
    assert _row(reg, "M7a")["status"] == "supported" and not reg["gates"]["statements"]


def test_gamma_minus_26_with_small_se_is_contradicted():
    reg = M.evaluate({}, _folds(gammas=[-26.2, -25.8, -26.0, -26.1, -25.9]), _cfg())
    r = _row(reg, "M7a")
    assert r["status"] == "contradicted" and r["interval"][1] < -9.8
    assert r["statistic"]["value"] == pytest.approx(-26.0)
    assert reg["summary"]["contradicted"] == ["M7a"] and reg["gates"]["contradicted"] == ["M7"]
    assert any(s.startswith("Elevation enters as a statistical term, not as a lapse rate") for s in
               reg["gates"]["statements"])
    assert all("M7" not in g["contradicted"] for g in reg["gates"]["levers"].values())   # not a lever's mechanism


def test_gamma_wide_interval_overlapping_the_band_is_not_testable():
    # γ = −26.0 ± 47.1 K/km: the ±2 SE interval (−120 to +68) holds the band, zero, inversions and −26 alike,
    # so it cannot tell inside from outside (a parameter the data cannot pin down cannot be Supported, plan W3)
    reg = M.evaluate({}, _folds(gammas=[-60.0, 5.0, -40.0, -5.0, -30.0]), _cfg())
    g, r = _panel(reg, "gamma"), _row(reg, "M7a")
    assert g["value"] == pytest.approx(-26.0) and g["status"] == "outside" and g["interval_overlaps"] is True
    assert g["interval_informative"] is False and "too wide" in g["status_note"]
    assert r["status"] == "not_testable" and "too wide" in r["reason"] and r["basis"] is None
    assert r["interval"][0] < -9.8 and r["interval"][1] > 0 and not reg["gates"]["statements"]


def test_gamma_interval_entirely_outside_is_contradicted_however_wide():
    reg = M.evaluate({}, _folds(gammas=[-40.0, -30.0, -35.0, -38.0, -32.0]), _cfg())   # −35 ± 7.4 K/km
    r = _row(reg, "M7a")
    assert r["status"] == "contradicted" and r["interval"][1] < -9.8
    assert r["interval"][1] - r["interval"][0] > 3.3                 # wider than the band, still excludes it


def test_ra_wide_interval_reaching_negative_values_is_not_testable():
    # r_a = 58.5 ± 130.7 s/m: the ±2 SE interval (−203 to 320) reaches impossible negative values
    reg = M.evaluate({}, _folds(a_values=(2.0, 150.0, 10.0, 120.0, 5.0)), _cfg())
    ra, r = _panel(reg, "r_a"), _row(reg, "M6a")
    assert ra["value"] == pytest.approx(58.5, abs=0.1) and ra["status"] == "inside"
    assert ra["interval_informative"] is False
    assert r["status"] == "not_testable" and "too wide" in r["reason"] and r["interval"][0] < 0
    lv = reg["gates"]["levers"]["Pct_Canopy"]
    assert "hypothesis" in lv["statement"] and not lv["supported"]


def test_ra_regression_folds_far_outside_the_range_are_not_supported():
    # reviewer case: per-fold r_a 1285, 721, 2368 s/m (r_a = 1458 ± 966, ±2 SE −475 to 3391)
    per = [1285.0, 721.0, 2368.0]
    day = M.evaluate({}, _folds(a_values=_a_for_ra(per)), _cfg())
    ra, r = _panel(day, "r_a"), _row(day, "M6a")
    assert ra["per_fold"] == pytest.approx(per) and ra["status"] == "outside"
    assert r["status"] == "not_testable" and "too wide" in r["reason"]
    night = M.evaluate({}, _folds(a_values=_a_for_ra(per, 71.5), window="night"), _cfg(window="night"))
    assert _panel(night, "r_a")["per_fold"] == pytest.approx(per)
    assert _row(night, "M6a")["status"] == "not_testable" and "night" in _row(night, "M6a")["reason"]
    from sparc.core.writeup import model_card_markdown

    card = model_card_markdown({"mechanisms": night, "config": _cfg(window="night")})
    assert "What holds: no mechanism is supported yet" in card and "1458" not in card.split("Contradicted")[0]


def test_ra_interval_reaching_zero_is_not_testable():
    chk = M._range_check({"value": 30.0, "se": 16.0}, M.RA_RANGE, "r_a", positive=True)
    assert chk["verdict"] == "not_testable" and "≤ 0" in chk["why"] and chk["status"] == "inside"
    assert M._range_check({"value": 30.0, "se": 16.0}, M.RA_RANGE, "r_a")["verdict"] == "supported"


def test_one_usable_fold_gives_no_verdict():
    # one fold with the physics on, two anti-physical: no jackknife interval, so neither Supported nor Contradicted
    reg = M.evaluate({}, [_fold(gamma_k_km=-26.0), _fold(a=0.0, warn="disabled"), _fold(a=0.0, warn="disabled")],
                     _cfg())
    for rid in ("M6a", "M7a"):
        r = _row(reg, rid)
        assert r["status"] == "not_testable" and "one fold" in r["reason"] and r["interval"] is None, rid
    assert _row(reg, "M7a")["statistic"]["value"] == pytest.approx(-26.0)
    assert not reg["gates"]["statements"] and not reg["gates"]["contradicted"]
    assert _panel(reg, "gamma")["se"] is None and _panel(reg, "gamma")["status"] == "outside"
    single = M.evaluate({}, [_fold(gamma_k_km=-9.9)], _cfg())
    assert _row(single, "M7a")["status"] == "not_testable" and _row(single, "M6a")["status"] == "not_testable"


def test_ranges_apply_to_the_afternoon_window_only():
    for hours in ((6, 7), (19, 20)):
        reg = M.evaluate({}, _folds(gammas=[-26.0] * 5), _cfg(hours=hours))
        for rid, key in (("M6a", "r_a"), ("M7a", "gamma")):
            r = _row(reg, rid)
            assert r["status"] == "not_testable" and "afternoon" in r["reason"], (hours, rid)
            assert _panel(reg, key)["status"] == "not_comparable" and _panel(reg, key)["value"] is not None
        assert not reg["gates"]["statements"]
    unknown = M.evaluate({}, _folds(), _cfg(hours=None))
    assert _row(unknown, "M7a")["status"] == "not_testable" and "time of day unknown" in _row(unknown, "M7a")["reason"]
    assert _row(M.evaluate({}, _folds(), _cfg(hours=(14, 17))), "M6a")["status"] == "supported"


def test_ra_with_the_generic_default_forcing_is_not_testable():
    reg = M.evaluate({}, _folds(), _cfg(sw=800.0, lw=-100.0, t2m=None, forcing=False))
    r, ra = _row(reg, "M6a"), _panel(reg, "r_a")
    assert r["status"] == "not_testable" and "generic default SW↓ = 800" in r["reason"]
    assert ra["status"] == "not_comparable" and ra["value"] is not None and not reg["panel"]["forcing_known"]
    assert "assumed" in ra["derivation"] and "no forcing file" in ra["derivation"]
    assert "assumed (no forcing file)" in reg["panel"]["timescales"]["assumption"]
    # values set in the config without a file are the campaign's, but the time of day is then unknown
    own = M.evaluate({}, _folds(), _cfg(forcing=False))
    assert own["panel"]["forcing_known"] and "time of day unknown" in _row(own, "M6a")["reason"]
    assert "generic default" not in _row(own, "M6a")["reason"]


def test_missing_elevation_role_makes_gamma_not_testable():
    roles = {k: v for k, v in ROLES.items() if k != "elevation"}
    reg = M.evaluate({}, _folds(roles=tuple(roles)), _cfg(roles=roles))
    r = _row(reg, "M7a")
    assert r["status"] == "not_testable" and "elevation" in r["reason"]
    assert _panel(reg, "gamma")["value"] is None and _panel(reg, "gamma")["status"] == "not_comparable"
    assert _row(reg, "M6a")["status"] == "supported"


# ---------------------------------------------------------------------------
# physics disabled / absent, basis
# ---------------------------------------------------------------------------

def test_physics_disabled_in_every_fold_makes_physics_rows_not_testable():
    warn = "fitted source amplitude a = -1 ≤ 0 (source anti-correlated with ΔT); physics term disabled (a = 0)"
    reg = M.evaluate({}, [_fold(a=0.0, warn=warn) for _ in range(3)], _cfg())
    for rid in ("M6a", "M7a"):
        r = _row(reg, rid)
        assert r["status"] == "not_testable" and "disabled" in r["reason"] and r["basis"] is None
    assert reg["physics_status"] == "disabled" and reg["panel"]["n_active"] == 0
    assert _panel(reg, "r_a")["value"] is None and reg["inputs"]["fit_warnings"] == [warn]
    for lever, g in reg["gates"]["levers"].items():
        assert g["in_physics"] is False and not g["mechanisms"] and not g["transport"], lever
        assert "disabled" in g["statement"] and "statistical only" in g["statement"]
        assert "acts in the physics" not in g["statement"]


def test_physics_disabled_in_some_folds_uses_the_others_and_says_so():
    folds = [_fold(a=44.0), _fold(a=0.0, warn="disabled"), _fold(a=46.0)]
    reg = M.evaluate({}, folds, _cfg())
    r = _row(reg, "M6a")
    assert r["status"] == "supported" and "1 of 3 folds" in r["note"]
    assert _panel(reg, "r_a")["n"] == 2 and reg["panel"]["n_active"] == 2


def test_no_physics_model_marks_the_evaluable_rows_not_testable():
    reg = M.evaluate({"config": _cfg()}, None)
    assert reg["physics_status"] == "absent"
    assert _row(reg, "M6a")["status"] == "not_testable" and "no physics model" in _row(reg, "M6a")["reason"]
    assert _row(reg, "M7a")["status"] == "not_testable"
    for lever, g in reg["gates"]["levers"].items():
        assert g["in_physics"] is False and g["physics_status"] == "absent", lever
        assert "absent" in g["statement"] and "acts in the physics" not in g["statement"]
    off = _cfg()
    off["physics"]["enabled"] = False
    reg = M.evaluate({}, _folds(), off)
    assert all(g["in_physics"] is False for g in reg["gates"]["levers"].values())


def test_non_temperature_units_cannot_be_put_in_physical_units():
    reg = M.evaluate({}, _folds(), _cfg(units="zscore"))
    assert _row(reg, "M6a")["status"] == "not_testable" and "temperature scale" in _row(reg, "M6a")["reason"]
    assert _panel(reg, "r_a") is None and _panel(reg, "L")["value"] == pytest.approx(256.0)


def test_basis_product_is_indicative_and_null_is_unspecified():
    prod = M.evaluate({}, _folds(), _cfg(kind="product"))
    for r in prod["rows"]:
        if r["status"] in ("supported", "contradicted"):
            assert r["basis"] == "indicative: fitted to an interpolated product, not measurements (W7)"
        else:
            assert r["basis"] is None
    assert any(r["status"] == "supported" for r in prod["rows"])
    none = M.evaluate({}, _folds(), _cfg(kind=None))
    assert none["target_kind"] is None and "unspecified" in none["basis"]
    assert "unspecified" in _row(none, "M6a")["basis"]
    sim = M.evaluate({}, _folds(), _cfg(kind="simulated"))
    assert "indicative" not in _row(sim, "M6a")["basis"] and "simulated" in _row(sim, "M6a")["basis"]


# ---------------------------------------------------------------------------
# the register itself
# ---------------------------------------------------------------------------

def test_register_is_complete_and_every_reference_resolves():
    assert [m["id"] for m in M.REGISTER] == [f"M{i}" for i in range(1, 11)]
    for m in M.REGISTER:
        assert m["name"] and m["chain"] and m["signatures"]
        assert [s["sig"] for s in m["signatures"]] == list("abc"[:len(m["signatures"])])
        for s in m["signatures"]:
            assert s["then"] and s["test"] and s["workstream"].startswith("W")
            assert s["kind"] in ("evaluate", "multi_window", "workstream", "missing_term")
        for k in m["refs"]:
            assert k in M.REFERENCES, k
    reg = M.evaluate({}, _folds(), _cfg())
    assert {r["status"] for r in reg["rows"]} <= set(M.VERDICTS)
    assert all(r["label"] == M.VERDICT_LABELS[r["status"]] for r in reg["rows"])
    assert all(k in M.REFERENCES for k in reg["references"])
    assert all(k in M.REFERENCES for m in reg["mechanisms"] for k in m["refs"])
    assert M.VERDICT_LABELS == {"supported": "Supported", "contradicted": "Contradicted",
                                "not_testable": "Not testable with this data", "not_yet_tested": "Not yet tested"}
    for r in reg["rows"]:
        if r["status"] in ("not_testable", "not_yet_tested"):
            assert r["reason"], r["id"]


def test_untestable_signatures_name_their_reason_or_workstream():
    reg = M.evaluate({}, _folds(), _cfg())
    for rid in ("M1a", "M2a", "M3a", "M3b", "M6b", "M7b"):
        r = _row(reg, rid)
        assert r["status"] == "not_testable" and ("window" in r["reason"]), rid
    assert "W6" in _row(reg, "M1a")["reason"]
    for rid, ws in (("M1b", "W4"), ("M4a", "W4"), ("M4b", "W4"), ("M5a", "W4"), ("M4c", "W7"), ("M5b", "W7"),
                    ("M8a", "W2"), ("M9a", "W8"), ("M10a", "W8")):
        r = _row(reg, rid)
        assert r["status"] == "not_yet_tested" and ws in r["reason"], rid
    assert "not in the model" in _row(reg, "M9a")["reason"]
    assert {m["id"]: m["status"] for m in reg["mechanisms"]}["M6"] == "supported"
    assert {m["id"]: m["status"] for m in reg["mechanisms"]}["M2"] == "not_testable"


def test_advection_selection_is_context_on_m5_not_its_verdict():
    adv = {"kept": False, "delta_rmse_mean": 0.003, "delta_rmse_se": 0.017}
    reg = M.evaluate({"physics_advection": adv}, _folds(), _cfg())
    r = _row(reg, "M5a")
    assert r["status"] == "not_yet_tested" and "dropped" in r["context"] and "not the signature test" in r["context"]
    assert _row(reg, "M5b")["status"] == "not_yet_tested" and _row(reg, "M5b")["context"] is None   # once per mechanism


def test_source_parameters_carry_priors_as_information():
    reg = M.evaluate({}, _folds(), _cfg())
    s = _panel(reg, "s")
    assert s["prior"]["mean"] == 0.6 and s["prior"]["sd"] == 0.3 and "power-scaling" in s["prior"]["note"]
    assert s["status"] == "not_comparable"
    for key in ("kappa_canopy", "a1", "e1", "e2", "e3", "w", "ef0"):
        assert _panel(reg, key) is not None, key
    assert "Grimmond & Oke (1999)" in _panel(reg, "a1")["status_note"]
    assert reg["panel"]["identifiability"].startswith("not yet tested: W3 profile likelihood")
    assert reg["panel"]["prior_sensitivity"].startswith("not yet tested: W3 power-scaling")


def test_implied_time_scales_are_assumptions_with_an_advection_bound():
    reg = M.evaluate({}, _folds(v=600.0), _cfg())
    ts = reg["panel"]["timescales"]
    ra = _panel(reg, "r_a")["value"]
    assert [t["h_m"] for t in ts["rows"]] == [20.0, 50.0] and "assumed" in ts["assumption"]
    assert ts["rows"][0]["tau_s"] == pytest.approx(ra * 20.0)
    assert ts["rows"][0]["omega_tau"] == pytest.approx(2 * math.pi / 86400 * ra * 20.0)
    assert 0.05 < ts["rows"][0]["omega_tau"] < 0.2
    speed = math.hypot(-1.158, 6.567)
    assert ts["advection_bound"]["tau_min_s"] == pytest.approx(600.0 / speed)
    assert "lower bound" in ts["advection_bound"]["text"]
    assert M.evaluate({}, _folds(), _cfg())["panel"]["timescales"]["advection_note"].startswith("Advection off")


def test_advection_row_says_whether_the_out_of_fold_test_ran():
    untested = _panel(M.evaluate({}, _folds(v=600.0), _cfg()), "v")
    assert "without the out-of-fold test" in untested["status_note"] and "kept after" not in untested["status_note"]
    adv = {"kept": True, "delta_rmse_mean": -0.021, "delta_rmse_se": 0.004}
    kept = _panel(M.evaluate({"physics_advection": adv}, _folds(v=600.0), _cfg()), "v")
    assert "kept by the out-of-fold test" in kept["status_note"] and "−0.021" in kept["status_note"]


def test_manifest_summary_fallback_uses_the_jackknife_over_n_folds():
    summary = {"a": {"mean": 45.0, "sd": 3.0}, "gamma": {"mean": _gamma_F_per_m(-8.0), "sd": 0.0001},
               "L_m": {"mean": 256.0, "sd": 20.0}, "v_norm_m": {"mean": 0.0, "sd": 0.0}}
    m = {"physics": summary, "cv": {"n_folds": 5}, "config": _cfg()}
    reg = M.evaluate(m, None)
    ra = _panel(reg, "r_a")
    scale = F_TO_K * RHO_CP / 635.8
    assert reg["inputs"]["source"] == "manifest summary"
    assert ra["value"] == pytest.approx(45.0 * scale) and ra["se"] == pytest.approx(3.0 * 2.0 * scale)
    assert _row(reg, "M6a")["status"] == "supported"


def test_lever_gates_list_the_mechanisms_that_explain_each_lever():
    cfg = _cfg()
    cfg["actionable"]["Unrelated"] = {}
    reg = M.evaluate({}, _folds(), cfg)
    lv = reg["gates"]["levers"]
    assert [m["id"] for m in lv["Pct_Canopy"]["mechanisms"]] == ["M1", "M2"]
    assert [m["id"] for m in lv["Albedo"]["mechanisms"]] == ["M1"]
    assert [m["id"] for m in lv["Pct_Impervious"]["mechanisms"]] == ["M2", "M3"]
    assert all([m["id"] for m in g["transport"]] == ["M4", "M5", "M6"] for c, g in lv.items() if c != "Unrelated")
    # M6 (the overall gain) is supported, but it tests no lever's own pathway: every lever stays a hypothesis
    assert {m["id"]: m["status"] for m in lv["Pct_Canopy"]["transport"]}["M6"] == "supported"
    for col in ("Pct_Canopy", "Pct_Impervious", "Albedo"):
        g = lv[col]
        assert g["in_physics"] and not g["supported"] and not g["contradicted"], col
        assert "hypothesis" in g["statement"] and "Transport and scale checks" in g["statement"], col
    assert lv["Unrelated"]["in_physics"] is False and "statistical only" in lv["Unrelated"]["statement"]


def test_markdown_renders_the_register_panel_and_errors():
    reg = M.evaluate({}, _folds(gammas=[-26.0] * 5), _cfg())
    md = M.mechanisms_markdown(reg)
    assert "| mechanism | IF it holds, THEN … | test | verdict | statistic |" in md
    assert "Fitted physics in physical units" in md and "Aerodynamic resistance" in md
    assert "**Gates:** Elevation enters as a statistical term" in md and "Implied time scales" in md
    assert "unavailable: boom" in M.mechanisms_markdown({"error": "boom"})
    json.dumps(reg, allow_nan=False)


def test_writeups_give_each_not_testable_signature_its_own_reason():
    from sparc.core.writeup import methods_markdown, model_card_markdown

    cfg = _cfg(window="night")
    reg = M.evaluate({}, _folds(window="night"), cfg)
    m = {"mechanisms": reg, "config": cfg}
    card, methods = model_card_markdown(m), methods_markdown(m)
    for text in (card, methods):
        assert "one time window or a missing input" not in text
        assert "M7(a) (night window: no single physical range at night" in text
        assert "M6(a) (night window: the verified reference list gives no night-time" in text
        assert "(one time window; needs the multi-window fit (W6))" in text
    line = next(x for x in card.splitlines() if x.startswith("- Not testable with this data"))
    assert line.count("M1(a)") == 1 and "M3(b) (one time window: the hysteresis coefficient" in line


# ---------------------------------------------------------------------------
# config, forcing and fingerprints
# ---------------------------------------------------------------------------

def test_forcing_records_t2m_and_old_configs_read_it_from_the_file(tmp_path):
    from sparc.core.forcing import apply_forcing_file

    src = Path(__file__).resolve().parents[2] / "configs" / "forcing" / "providence_2020-07-29.json"
    if not src.exists():
        pytest.skip("Providence forcing file not present")
    ph = apply_forcing_file({}, src, label="forcing/providence_2020-07-29.json")
    assert ph["forcing_info"]["t2m_K"] == pytest.approx(304.12, abs=0.01)
    (tmp_path / "forcing").mkdir()
    (tmp_path / "forcing" / src.name).write_text(src.read_text())
    old = {"forcing": "forcing/" + src.name, "forcing_info": {"date": "2020-07-29"}}
    assert M.forcing_t2m_K(old, tmp_path) == pytest.approx(304.12, abs=0.01)
    assert M.forcing_t2m_K({"forcing": None}, tmp_path) is None
    # an old run directory: forcing_info without t2m_K or hours_local → both read from the forcing file
    run = tmp_path / "run"
    run.mkdir()
    cfg = _cfg()
    cfg["physics"]["forcing_info"] = {"date": "2020-07-29"}
    (run / "manifest.json").write_text(json.dumps({"config": cfg, "provenance": {"config_dir": str(tmp_path)}}))
    (run / "physics.json").write_text(json.dumps(_folds()))
    reg = M.mechanisms_for_run(run, write=False)
    assert reg["panel"]["t2m_known"] and reg["panel"]["afternoon"] and _row(reg, "M6a")["status"] == "supported"


def test_target_kind_is_validated_and_does_not_change_fit_fingerprints():
    import pandas as pd

    from sparc.core.config import core_config_from_dict
    from sparc.core.pipeline import _fingerprint, fingerprint_sections
    from sparc.core.synthetic import synthetic_city_config

    raw = synthetic_city_config()
    assert raw["data"]["target_kind"] == "simulated"
    a = core_config_from_dict(raw)
    raw2 = synthetic_city_config()
    raw2["data"]["target_kind"] = None
    b = core_config_from_dict(raw2)
    frame = pd.DataFrame({"x": [0.0, 30.0]})
    assert fingerprint_sections(a, True, frame)["core"] == fingerprint_sections(b, True, frame)["core"]
    assert _fingerprint(a, True, None, "x") == _fingerprint(b, True, None, "x")
    raw2["data"]["target_kind"] = "interpolated"
    with pytest.raises(ValueError, match="target_kind"):
        core_config_from_dict(raw2)


def test_providence_configs_declare_a_product_target():
    import yaml

    repo = Path(__file__).resolve().parents[2]
    for name in ("core_providence.yml", "core_providence_open.yml"):
        p = repo / "configs" / name
        if p.exists():
            assert yaml.safe_load(p.read_text())["core"]["data"]["target_kind"] == "product"


def test_packaged_studio_providence_example_matches_the_repo_config():
    # Studio uses the packaged copy when there is no repo checkout: same product, same data block
    import yaml

    root = Path(__file__).resolve().parents[2]
    repo = root / "configs" / "core_providence.yml"
    packaged = root / "sparc" / "studio" / "examples" / "providence" / "core_providence.yml"   # templates.EXAMPLE_DIR
    if not (repo.exists() and packaged.exists()):
        pytest.skip("Providence configs not present")
    a, b = (yaml.safe_load(x.read_text())["core"]["data"] for x in (repo, packaged))
    a.pop("path", None), b.pop("path", None)
    assert b == a and b["target_kind"] == "product"


# ---------------------------------------------------------------------------
# integration: a small synthetic run, the results page and the CLI
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    from sparc.core.config import load_core_config
    from sparc.core.pipeline import run_core
    from sparc.core.synthetic import write_demo_project

    out = tmp_path_factory.mktemp("mech_demo")
    info = write_demo_project(out, n=40, seed=0)
    cfg = load_core_config(info["config_path"])
    cfg.raw["models"] = {"ols": True, "mgwr": False, "gwrf": False, "gam": False, "physics": True}
    cfg.raw["stacker"]["epochs"] = 30
    cfg.raw["stacker"]["tune_lambda"] = [0.0]
    cfg.raw["cv"]["baselines"] = False
    res = run_core(cfg, stages=("S0", "S1", "S2", "S3"), fast=True)
    return info["config_path"], cfg, res


def test_run_writes_mechanisms_json_and_the_manifest_section(small_run):
    _, _, res = small_run
    run = res.run_dir
    reg = json.loads((run / "mechanisms.json").read_text())
    man = json.loads((run / "manifest.json").read_text())
    assert reg["rows"] and man["mechanisms"]["rows"] == reg["rows"]
    assert reg["target_kind"] == "simulated" and reg["inputs"]["source"] == "physics.json"
    assert {r["id"] for r in reg["rows"]} >= {"M1a", "M6a", "M7a", "M10a"}
    assert _panel(reg, "r_a") is not None and _panel(reg, "L")["value"] > 0
    assert "Physics: what holds" in (run / "report.md").read_text()
    assert "## Physics mechanisms" in (run / "methods.md").read_text()
    assert "## Physics mechanisms" in (run / "model_card.md").read_text()


def test_register_failure_is_a_warning_not_a_crash(small_run, monkeypatch):
    from sparc.core.report import build_manifest

    def boom(*a, **k):
        raise RuntimeError("boom")

    from sparc.core import progress

    warned = []
    monkeypatch.setattr(M, "evaluate", boom)
    monkeypatch.setattr(progress, "warn", lambda code, msg, **kw: warned.append((code, msg)))
    m = build_manifest(small_run[2], {}, True)
    assert m["mechanisms"] == {"error": "RuntimeError: boom"}
    assert [c for c, _ in warned] == ["physics.mechanisms_failed"] and "boom" in warned[0][1]


def test_results_page_shows_the_register(small_run):
    from sparc.core.results_page import collect, render_page

    _, cfg, res = small_run
    blob = collect(res.run_dir, cfg)
    assert blob["mechanisms"]["rows"] == json.loads((res.run_dir / "mechanisms.json").read_text())["rows"]
    html = render_page(blob, cfg)
    assert 'id="mechCard"' in html and "Physics: what holds" in html and '"mechanisms":{' in html


def test_cli_recomputes_the_register_of_a_run(small_run, capsys):
    from sparc.core import cli

    cfg_path, _, res = small_run
    run = res.run_dir
    (run / "mechanisms.json").unlink()
    assert cli.main(["mechanisms", "-r", str(run)]) == 0
    out = capsys.readouterr().out
    assert "| mechanism | IF it holds, THEN …" in out and (run / "mechanisms.json").exists()
    man = json.loads((run / "manifest.json").read_text())
    assert any(e["section"] == "mechanisms" for e in man["post_run"])
    assert json.loads((run / "mechanisms.json").read_text())["target_kind"] == "simulated"
    # -p replaces the run's config: a copy that declares a product target must change the basis
    import yaml

    raw = yaml.safe_load(Path(cfg_path).read_text())
    (raw.get("core", raw))["data"]["target_kind"] = "product"
    prod = Path(cfg_path).with_name("product_copy.yml")
    prod.write_text(yaml.safe_dump(raw))
    assert cli.main(["mechanisms", str(run), "-p", str(prod)]) == 0
    reg = json.loads((run / "mechanisms.json").read_text())
    assert reg["target_kind"] == "product" and reg["basis"].startswith("indicative")
    assert all(r["basis"].startswith("indicative") for r in reg["panel"]["rows"] if r["value"] is not None)
    assert json.loads((run / "manifest.json").read_text())["mechanisms"]["target_kind"] == "product"
    assert cli.main(["mechanisms", "-r", str(run / "missing")]) == 2
