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
    folds = [_fold(a=44.0), _fold(a=0.0, warn="disabled"), _fold(a=46.0), _fold(a=45.0), _fold(a=43.0)]
    reg = M.evaluate({}, folds, _cfg())
    r = _row(reg, "M6a")
    assert r["status"] == "supported" and "1 of 5 folds" in r["note"]
    ra = _panel(reg, "r_a")
    assert ra["n"] == 4 and ra["n_groups"] == 5 and reg["panel"]["n_active"] == 4
    # the delete-a-group inflation stays the K = 5 one: s·(K−1)/√K with s the sample sd of the 4 active folds
    per = np.asarray(ra["per_fold"])
    assert ra["se"] == pytest.approx(per.std(ddof=1) * 4 / math.sqrt(5))
    assert ra["se"] > per.std() * math.sqrt(3)                      # the 4-replicate formula understates it


def test_a_subset_of_the_folds_gives_no_verdict():
    # reviewer case: γ −12 and −11 K/km in 2 of 5 folds.  The 2-replicate formula gave −11.5 ± 0.5 (Contradicted);
    # the K = 5 inflation gives ± 1.26, and a subset chosen by the fit is not a valid jackknife interval anyway
    folds = [_fold(gamma_k_km=-12.0), _fold(gamma_k_km=-11.0)] + [_fold(a=0.0, warn="disabled") for _ in range(3)]
    reg = M.evaluate({}, folds, _cfg())
    g, r = _panel(reg, "gamma"), _row(reg, "M7a")
    assert g["se"] == pytest.approx(math.sqrt(0.5) * 4 / math.sqrt(5)) and g["interval_informative"] is False
    assert r["status"] == "not_testable" and "2 of 5 folds" in r["reason"] and "not a valid jackknife" in r["reason"]
    assert r["basis"] is None and not reg["gates"]["statements"] and not reg["gates"]["contradicted"]
    assert _row(reg, "M6a")["status"] == "not_testable"
    # three of five is still a subset; two of three passes K−1 but leaves one degree of freedom
    three = [_fold(gamma_k_km=g) for g in (-26.0, -25.9, -26.1)] + [_fold(a=0.0, warn="x") for _ in range(2)]
    assert _row(M.evaluate({}, three, _cfg()), "M7a")["status"] == "not_testable"
    two = [_fold(gamma_k_km=-26.0), _fold(gamma_k_km=-26.2), _fold(a=0.0, warn="x")]
    r2 = _row(M.evaluate({}, two, _cfg()), "M7a")
    assert r2["status"] == "not_testable" and "only 2 folds" in r2["reason"] and "degree of freedom" in r2["reason"]
    # four of five is enough: −26 K/km with a small SE is Contradicted
    four = [_fold(gamma_k_km=g) for g in (-26.0, -25.9, -26.1, -26.2)] + [_fold(a=0.0, warn="x")]
    assert _row(M.evaluate({}, four, _cfg()), "M7a")["status"] == "contradicted"
    # two folds in a two-fold run: one degree of freedom too
    assert _row(M.evaluate({}, _folds(a_values=(44.0, 46.0)), _cfg()), "M6a")["status"] == "not_testable"


def test_point_estimate_outside_the_range_is_not_supported():
    # reviewer cases: the ±2 SE interval only reaches the range from an estimate outside it
    for st, band, sym, pos in (({"value": 130.0, "se": 17.0}, M.RA_RANGE, "r_a", True),
                               ({"value": -5.0, "se": 0.8}, M.LAPSE_BAND, "γ", False),
                               ({"value": 12.0, "se": 4.5}, M.RA_RANGE, "r_a", True)):
        chk = M._range_check(st, band, sym, positive=pos)
        assert chk["verdict"] == "not_testable" and chk["status"] == "outside", st
        assert chk["overlaps"] and chk["informative"] and "inconclusive" in chk["why"], st
    # through evaluate: folds with r_a 118–142 s/m are neither Supported nor cited for any lever
    reg = M.evaluate({}, _folds(a_values=_a_for_ra([118.0, 125.0, 130.0, 136.0, 142.0])), _cfg())
    r = _row(reg, "M6a")
    assert r["status"] == "not_testable" and "inconclusive" in r["reason"] and r["interval"][0] < 100.0
    assert _panel(reg, "r_a")["status"] == "outside"
    assert {m["id"]: m["status"] for m in reg["mechanisms"]}["M6"] == "not_testable"
    assert all("(supported)" not in g["statement"] for g in reg["gates"]["levers"].values())
    from sparc.core.writeup import model_card_markdown

    card = model_card_markdown({"mechanisms": reg, "config": _cfg()})
    holds = next(x for x in card.splitlines() if x.startswith("- What holds"))
    assert "M6(a)" not in holds and "M7(a)" in holds                    # γ = −8 K/km is inside the band
    # γ just outside the band with a tight interval reaching it: inconclusive, not Supported
    near = M.evaluate({}, _folds(gammas=[-5.5, -4.5, -5.0, -5.4, -4.6]), _cfg())      # −5.0 ± 0.81 K/km
    assert _row(near, "M7a")["interval"][0] < -6.5 < _row(near, "M7a")["interval"][1]
    assert _row(near, "M7a")["status"] == "not_testable" and "inconclusive" in _row(near, "M7a")["reason"]


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


def test_night_window_levers_have_no_sunlight_pathway():
    # the night source is Q* = LW_net: no albedo or shade term, so M1 is not in it (PhysicsModel._q_points_torch)
    reg = M.evaluate({}, _folds(window="night"), _cfg(window="night", hours=(0, 1)))
    lv = reg["gates"]["levers"]
    assert lv["Albedo"]["in_physics"] is False and not lv["Albedo"]["mechanisms"]
    assert "does not enter the physics source in this night window" in lv["Albedo"]["statement"]
    assert "statistical only" in lv["Albedo"]["statement"] and "M1" not in lv["Albedo"]["contradicted"]
    assert [m["id"] for m in lv["Pct_Canopy"]["mechanisms"]] == ["M2"] and lv["Pct_Canopy"]["in_physics"]
    assert [m["id"] for m in lv["Pct_Impervious"]["mechanisms"]] == ["M2", "M3"]
    assert all("through M1" not in g["statement"] and "M1 Absorbed" not in g["statement"] for g in lv.values())
    m1 = next(m for m in reg["mechanisms"] if m["id"] == "M1")
    assert m1["acts_in_run"] is False and "does not act" in _row(reg, "M1a")["note"]
    # by day the same config keeps M1 for albedo and canopy
    day = M.evaluate({}, _folds(), _cfg())["gates"]["levers"]
    assert [m["id"] for m in day["Albedo"]["mechanisms"]] == ["M1"] and day["Albedo"]["in_physics"]
    assert [m["id"] for m in day["Pct_Canopy"]["mechanisms"]] == ["M1", "M2"]


def test_grid_limited_L_from_the_operator_resolution_keys():
    """W1's grid_resolution keys reach the panel: past cell Péclet 1 on both axes the hybrid scheme's diffusion
    no longer depends on L, so L is not identified (no SE); past 1 on one axis only it is weakly identified."""
    from sparc.core import operators as ops

    def fold(L, v):
        res = ops.grid_resolution(L, v, 30.0, 30.0, "hybrid")
        return _fold(L=L, vx_m=v[0], vy_m=v[1], v_norm_m=math.hypot(*v), advection_scheme="hybrid", **res,
                     resolution_warning=None if res["L_resolved"] else "grid-limited")

    flat = [fold(L, (700.0, 700.0)) for L in (40.0, 42.0, 44.0)]
    assert all(f["L_elasticity_x"] == 0 and f["L_elasticity_y"] == 0 for f in flat)
    L = _panel(M.evaluate({}, flat, _cfg()), "L")
    assert L["grid_limited"]["every_fold"] and L["grid_limited"]["k"] == 3
    assert L["se"] is None and L["interval"] is None and "not identified" in L["status_note"]
    assert "grid-limited in 3 of 3 folds: not identified" in L["value_text"]

    one_axis = [fold(L, (900.0, 0.0)) for L in (100.0, 105.0, 110.0)]          # Pe_x ≈ 1.1–1.4, Pe_y = 0
    L1 = _panel(M.evaluate({}, one_axis, _cfg()), "L")
    assert not L1["grid_limited"]["every_fold"] and L1["se"] is not None and "weakly identified" in L1["value_text"]

    ok = [fold(L, (100.0, 0.0)) for L in (250.0, 256.0, 262.0)]
    assert "grid_limited" not in _panel(M.evaluate({}, ok, _cfg()), "L")
    assert "grid_limited" not in _panel(M.evaluate({}, _folds(), _cfg()), "L")      # old runs: no keys


def test_fit_bounds_match_the_physics_transforms():
    from sparc.core import physics

    assert M.FIT_BOUNDS["kappa_canopy"][:2] == physics.KAPPA_BOUNDS
    assert M.FIT_BOUNDS["shade_c0"][:2] == physics.C0_BOUNDS and M.FIT_BOUNDS["shade_width"][:2] == physics.CW_BOUNDS
    assert M.FLOOR_TOL["w"] == pytest.approx(1e-3 * physics._W_SCALE)


def test_parameters_at_a_fit_bound_are_flagged_not_reported_as_precise():
    # reviewer case (Providence fast run): L = 644.967 m, the upper fit bound, in every fold → "645 ± 0 m"
    hi = 644.9672339344687
    folds = _folds(a_values=(44.0, 45.0, 46.0), L=hi - 3e-6, L_bounds_m=[30.0, hi])
    for f, e2 in zip(folds, (1e-12, 1e-13, 0.8)):                   # e₂ at its softplus floor in 2 of 3 folds
        f["e2"] = e2
    reg = M.evaluate({}, folds, _cfg())
    L = _panel(reg, "L")
    assert L["at_bound"]["side"] == "upper" and L["at_bound"]["k"] == 3 and L["at_bound"]["every_fold"]
    assert L["se"] is None and L["interval"] is None and "not identified" in L["status_note"]
    assert "± 0" not in L["value_text"] and "upper fit bound" in L["value_text"]
    ctx = _row(reg, "M4a")["context"]
    assert "upper fit bound" in ctx and "± 0" not in ctx
    e2 = _panel(reg, "e2")
    assert e2["at_bound"]["side"] == "lower" and e2["at_bound"]["k"] == 2 and e2["se"] is not None
    assert "understates" in e2["status_note"]
    assert "L" in reg["panel"]["at_bound"] and "e2" in reg["panel"]["at_bound"]
    assert any("fit bound" in n for n in reg["panel"]["notes"])
    assert "at_bound" not in _panel(reg, "e1")                         # e₁ = 2.0: interior
    md = M.mechanisms_markdown(reg)
    assert "645 ± 0" not in md and "| at fit bound |" in md
    from sparc.core.writeup import methods_markdown

    methods = methods_markdown({"mechanisms": reg, "config": _cfg()})
    assert "mixing length L = 645 m (at the upper fit bound in 3 of 3 folds" in methods and "645 ± 0" not in methods
    # an interior L keeps its SE and no flag
    inner = _panel(M.evaluate({}, _folds(L=256.0), _cfg()), "L")
    assert "at_bound" not in inner and inner["se"] is not None
    # s at the bottom of its sigmoid in one fold
    s_folds = _folds(a_values=(44.0, 45.0, 46.0))
    s_folds[2]["s"] = 2.6e-53
    s = _panel(M.evaluate({}, s_folds, _cfg()), "s")
    assert s["at_bound"]["side"] == "lower" and s["at_bound"]["k"] == 1


def test_time_scales_from_an_r_a_that_is_not_pinned_down_say_so():
    # r_a = 58.5 ± 130.7 s/m (±2 SE −203 to 320): the point time scales would mean nothing on their own
    reg = M.evaluate({}, _folds(a_values=(2.0, 150.0, 10.0, 120.0, 5.0)), _cfg())
    ts = reg["panel"]["timescales"]
    assert ts["informative"] is False and "not pinned down" in ts["caveat"] and "reaches r_a ≤ 0" in ts["caveat"]
    assert ts["rows"][0]["tau_min_range"][0] == 0.0                     # clipped at zero
    assert ts["rows"][1]["omega_tau_range"][1] > ts["rows"][1]["omega_tau"]
    md = M.mechanisms_markdown(reg)
    assert "not informative" in md and "±2 SE: τ 0–" in md
    # a precise r_a: informative, with its range
    ok = M.evaluate({}, _folds(), _cfg())["panel"]["timescales"]
    assert ok["informative"] and ok["caveat"] is None
    lo, hi = ok["rows"][0]["tau_min_range"]
    assert 0 < lo < ok["rows"][0]["tau_min"] < hi
    # one usable fold: no interval, so not informative either
    one = M.evaluate({}, [_fold(), _fold(a=0.0, warn="x"), _fold(a=0.0, warn="x")], _cfg())["panel"]["timescales"]
    assert one["informative"] is False and "one fold" in one["caveat"] and one["rows"][0]["tau_min_range"] is None


def test_catalog_defines_r_a_with_the_run_normaliser():
    from sparc.core.catalog import DATA_DICTIONARY

    text = DATA_DICTIONARY["mechanisms.json"]["r_a"][2]
    assert "|LW_net|" in text and "SW↓" in text and "5/9" in text


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


def test_physics_json_is_used_only_when_it_belongs_to_the_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    summary = {"a": {"mean": 45.0, "sd": 3.0}, "gamma": {"mean": _gamma_F_per_m(-8.0), "sd": 0.0},
               "L_m": {"mean": 256.0, "sd": 0.0}, "v_norm_m": {"mean": 0.0, "sd": 0.0}}
    (run / "manifest.json").write_text(json.dumps({"config": _cfg(), "physics": summary, "cv": {"n_folds": 5}}))
    # a file from an earlier fit: its fold means do not match this run's summary → the summary is used
    (run / "physics.json").write_text(json.dumps(_folds(a_values=(1000.0, 1010.0, 990.0))))
    reg = M.mechanisms_for_run(run, write=False)
    assert reg["inputs"]["source"] == "manifest summary" and "does not match" in reg["inputs"]["note"]
    assert _panel(reg, "r_a")["value"] == pytest.approx(45.0 * F_TO_K * RHO_CP / 635.8)
    (run / "physics.json").write_text(json.dumps(_folds(a_values=(40.0, 42.0, 45.0, 48.0, 50.0))))
    reg = M.mechanisms_for_run(run, write=False)
    assert reg["inputs"]["source"] == "physics.json" and "note" not in reg["inputs"]
    # the manifest records no physics fit: the file is ignored and the register says physics is absent
    (run / "manifest.json").write_text(json.dumps({"config": _cfg(), "physics": None}))
    reg = M.mechanisms_for_run(run, write=False)
    assert reg["physics_status"] == "absent" and "records no physics fit" in reg["inputs"]["note"]
    # a config without a physics model has none, whatever files are passed in
    off = _cfg()
    off["models"] = {"physics": False}
    reg = M.evaluate({}, _folds(), off)
    assert reg["physics_status"] == "absent" and _panel(reg, "r_a")["value"] is None
    assert "models.physics is off" in _row(reg, "M6a")["reason"]
    assert all(not g["in_physics"] for g in reg["gates"]["levers"].values())


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
# W3: the identifiability study in the register (hand-made reports for the folds of _folds())
# ---------------------------------------------------------------------------

_LABELS = {"L": ("Mixing length L", "m"), "s": ("Canopy shading strength s", "fraction of absorbed sunlight"),
           "kappa_canopy": ("Shade saturation scale κ", "canopy fraction"), "a1": ("Impervious heat storage a₁",
                                                                                   "fraction of Q*"),
           "e2": ("Evaporative-fraction slope on canopy e₂", "logit per canopy fraction"),
           "w": ("Water-cell sink w", "W/m²")}


def _prow(key, cls="identified", ci=(0.2, 0.5), run=None, data=None, share=None, at_bound=None, psens=None,
          conflict=None, open_sides=(), sides_at_limit=()):
    """One parameter row of an identifiability report (the fields the register reads)."""
    label, unit = _LABELS[key]
    return {"key": key, "label": label, "unit": unit, "estimate_run": run, "estimate_data": data,
            "estimate_run_folds": None, "at_bound": at_bound, "held": "bound" if at_bound else None,
            "se_naive": 0.01, "se_robust": 0.1, "adjust_factor": 0.01, "ci95": list(ci) if ci else None,
            "ci95_naive": [0.3, 0.4], "identifiability": cls, "open_sides": list(open_sides),
            "sides_at_limit": list(sides_at_limit), "pull": 0.5, "prior": None, "prior_share_run": share,
            "posterior": {"method": "laplace", "mean": run, "sd": 0.1}, "psens": psens, "conflict": conflict,
            "profile": {"grid": [0.1, 0.3, 0.6], "lambda_adj": [5.0, 0.0, 6.0], "lambda_naive": [50.0, 0.0, 60.0]},
            "note": None}


def _ident(folds, rows, *, n=1000, cv=None, a_se=None, gamma_se=None, units="degF", window="day", **run):
    """An identifiability report made for ``folds`` (their fold means recorded as the study records them) with
    parameter ``rows``; ``a_se`` / ``gamma_se``: sandwich SEs of the gain a (°F per unit q) and γ (°F per m)."""
    act = [f for f in folds if f["a"] > 0 and not f.get("fit_warning")]
    pk = {"L": "L_m", "v": "v_norm_m"}

    def mean(k):
        return sum(f[pk.get(k, k)] for f in act) / len(act)

    a_m, g_m = mean("a"), mean("gamma")
    lin = [{"key": "a", "label": "Source gain a", "unit": "target units per unit q", "estimate_data": a_m,
            "se_naive_data": 0.5, "se_robust_data": a_se or 1.0, "estimate_run": a_m, "se_naive_run": 0.5,
            "se_robust_run": a_se or 1.0, "adjust_factor": 0.01},
           {"key": "gamma", "label": "Elevation coefficient γ", "unit": "target units per m", "estimate_data": g_m,
            "se_naive_data": 1e-5, "se_robust_data": gamma_se or 1e-4, "estimate_run": g_m, "se_naive_run": 1e-5,
            "se_robust_run": gamma_se or 1e-4, "adjust_factor": 0.01}]
    rep = {"schema": 1, "status": "ok", "n": n, "n_clusters": 23, "coarse_m": None, "units": units, "window": window,
           "cluster": {"kind": "cv_block", "block_m": 2000.0}, "settings": {"score_covariance": "blocks"},
           "posterior": {"laplace_adequate": True}, "warnings": [], "parameters": rows, "linear": lin,
           "run": {"n_points_run": n, "cv": cv, "fold_means": {r["key"]: mean(r["key"]) for r in rows}}}
    rep["run"].update(run)
    return rep


def _m(n=1000, **kw):
    return {"n_points": n, "cv": {"n_folds": 5, "block_m": 2000.0, "n_blocks": 23}, **kw}


def test_identifiability_fills_the_panel_and_replaces_the_not_yet_tested_notes():
    folds = _folds()
    rep = _ident(folds, [_prow("L", ci=(200.0, 420.0), run=256.0, data=270.0, share=0.05),
                         _prow("s", ci=(0.15, 0.55), run=0.34, data=0.3, share=0.4,
                               psens={"prior": 0.01, "likelihood": 0.2, "diagnosis": "none", "method": "laplace"},
                               conflict={"z": 1.0, "p": 0.32, "flag": False, "reason": None})])
    reg = M.evaluate(_m(), folds, _cfg(), identifiability=rep)
    ib = reg["identifiability"]
    assert ib["status"] == "used" and ib["source"] == "argument" and ib["not_identified"] == []
    s = _panel(reg, "s")
    idn = s["identifiability"]
    assert idn["class"] == "identified" and idn["ci95"] == [0.15, 0.55] and idn["prior_share_run"] == 0.4
    assert idn["posterior"]["mean"] == 0.34 and idn["psens"]["likelihood"] == 0.2 and idn["conflict"]["p"] == 0.32
    assert "identified by the data alone" in idn["text"] and "prior share 0.40" in idn["text"]
    assert "not yet tested" not in s["prior"]["note"] and "power-scaling" in s["prior"]["note"]
    assert "W3 profile likelihood (identifiability column)" in s["status_note"]
    P = reg["panel"]
    assert P["identifiability_study"] and not P["identifiability"].startswith("not yet tested")
    assert "23 CV blocks of 2000 m" in P["identifiability"] and "Kallioinen" in P["prior_sensitivity"]
    assert _panel(reg, "L")["identifiability"]["ci95"] == [200.0, 420.0]
    assert _panel(reg, "a1").get("identifiability") is None          # not profiled: no columns
    assert {"raue2009", "chandler2007", "ribatet2012", "kallioinen2024", "evans2006"} <= set(reg["references"])
    assert "Identifiability from the data alone" in _row(reg, "M4a")["context"]
    # the reference strings are the study's own
    from sparc.core import identifiability as ID

    assert all(M.REFERENCES[k] == ID.REFERENCES[k] for k in ID.REFERENCES)
    md = M.mechanisms_markdown(reg)
    assert "**Identifiability from the data alone (W3).**" in md and "| Canopy shading strength s | 0.34 | 0.34 |" in md
    assert "| parameter | run (fold mean) | full-data refit |" in md
    assert "come later" not in md


def test_non_identified_s_is_set_by_the_penalty_in_the_register_and_the_lever_statement():
    folds = _folds()
    rep = _ident(folds, [_prow("s", cls="structurally_non_identifiable", ci=(None, None), run=0.36, data=0.9,
                               share=0.83),
                         _prow("e2", ci=(0.0, 2.9), run=0.5, data=0.6, sides_at_limit=("lower",))])
    reg = M.evaluate(_m(identifiability=rep), folds, _cfg())         # read from the manifest section
    assert reg["identifiability"]["status"] == "used" and reg["identifiability"]["source"] == "manifest"
    assert reg["identifiability"]["not_identified"] == ["s"]
    lever = reg["gates"]["levers"]["Pct_Canopy"]
    st = lever["statement"]
    # the run's own value is its fold mean (0.34); the study's full-data refit (0.36) is labelled as such (review X15)
    assert ("Canopy shading strength s: not identified by the data alone (flat profile likelihood); its full-data "
            "refit value 0.36 is set mostly by the penalty (prior share 0.83; the run's own value, its fold mean, is "
            "0.34).") in st
    assert lever["identifiability"]["s"]["run_value"] == pytest.approx(0.34)
    assert lever["identifiability"]["s"]["estimate_run"] == 0.36 and "run value" not in st
    assert "the data alone do not exclude e₂ = 0" in st
    assert lever["not_identified"] == ["s"] and set(lever["identifiability"]) == {"s", "e2"}
    assert "identifiability" not in reg["gates"]["levers"]["Albedo"]                # no parameter of its own
    imp = reg["gates"]["levers"]["Pct_Impervious"]
    assert "identifiability" not in imp                                            # a₁, e₃ not profiled
    m1 = next(m for m in reg["mechanisms"] if m["id"] == "M1")
    assert m1["identifiability"] == {"keys": ["s"], "not_identified": ["s"]}
    assert "Canopy shading strength s: not identified" in _row(reg, "M1a")["context"]
    assert _row(reg, "M1b").get("context") is None                                 # once per mechanism
    assert [x["key"] for x in _row(reg, "M2a")["identifiability"]] == ["e2"]


def test_s_on_its_lower_bound_is_bounded_but_does_not_exclude_zero():
    folds = _folds()
    rep = _ident(folds, [_prow("s", ci=(0.0, 0.49), run=0.335, data=1e-9, share=0.93,
                               at_bound={"side": "lower", "bound": 0.0},
                               psens={"prior": 0.097, "likelihood": 0.115, "diagnosis": "prior_data_conflict",
                                      "method": "profile"},
                               conflict={"z": None, "p": None, "flag": None, "reason": "on a fit bound"})])
    reg = M.evaluate(_m(), folds, _cfg(), identifiability=rep)
    p = _panel(reg, "s")["identifiability"]
    assert p["reaches_zero"] and p["class"] == "identified"
    assert p["text"] == ("Canopy shading strength s: the data alone do not exclude s = 0 (the data-only fit sits at 0; "
                         "95% profile interval 0 to 0.49); its full-data refit value 0.335 is set mostly by the "
                         "penalty (prior share 0.93; the run's own value, its fold mean, is 0.34); power-scaling flags a potential prior–data conflict (prior and likelihood both "
                         "sensitive).")
    assert "| identified; does not exclude 0 |" in M.mechanisms_markdown(reg)
    assert "not testable" in M.mechanisms_markdown(reg)              # the conflict check at a bound


@pytest.mark.parametrize("change, why", [
    ({"fold_L": 300.0}, "different physics fit"),
    ({"n": 999}, "999 cells"),
    ({"cv": {"n_folds": 4, "block_m": 2000.0, "n_blocks": 23}}, "cross-validation design"),
    ({"input_sha256": "b" * 64}, "different input data"),
    ({"units": "degC"}, "target units"),
    ({"window": "night"}, "window"),
])
def test_a_study_made_for_another_run_is_ignored_with_a_note(change, why):
    folds = _folds()
    rep = _ident(folds, [_prow("s", cls="structurally_non_identifiable", ci=None, run=0.34, data=0.9, share=0.83)],
                 n=change.get("n", 1000), cv=change.get("cv"), units=change.get("units", "degF"),
                 window=change.get("window", "day"), input_sha256=change.get("input_sha256"))
    if "fold_L" in change:
        rep["run"]["fold_means"]["L"] = change["fold_L"]
    man = _m(provenance={"input_sha256": "a" * 64})
    reg = M.evaluate(man, folds, _cfg(), identifiability=rep)
    ib = reg["identifiability"]
    assert ib["status"] == "ignored" and why in ib["note"] and "re-run `sparc core identifiability`" in ib["note"]
    assert all("identifiability" not in r for r in reg["panel"]["rows"])
    assert "identifiability" not in reg["gates"]["levers"]["Pct_Canopy"]
    assert reg["panel"]["identifiability"].startswith("not yet tested")
    assert "*Identifiability (W3):* identifiability study ignored" in M.mechanisms_markdown(reg)


def test_a_study_is_ignored_without_an_active_physics_term_and_absent_when_disabled():
    folds = _folds()
    rep = _ident(folds, [_prow("s")])
    off = M.evaluate(_m(), _folds(a_values=(-1.0, -2.0, -3.0)), _cfg(), identifiability=rep)
    assert off["identifiability"]["status"] == "ignored" and "no active physics term" in off["identifiability"]["note"]
    assert M.evaluate(_m(identifiability=rep), folds, _cfg(), identifiability=False)["identifiability"] == \
        {"status": "absent", "source": None, "note": None}
    assert M.evaluate(_m(), folds, _cfg())["identifiability"]["status"] == "absent"
    bad = M.evaluate(_m(), folds, _cfg(), identifiability={"schema": 9, "parameters": []})
    assert bad["identifiability"]["status"] == "ignored" and "known schema" in bad["identifiability"]["note"]


def test_a_key_parameter_not_identified_blocks_supported_but_never_makes_contradicted(monkeypatch):
    folds = _folds()
    L_bad = _prow("L", cls="practically_non_identifiable", ci=(150.0, None), run=256.0, data=260.0,
                  open_sides=("upper",))
    reg = M.evaluate(_m(), folds, _cfg(), identifiability=_ident(folds, [L_bad]))
    assert _row(reg, "M6a")["status"] == "supported"              # L is M4's key parameter, not M6's
    assert "stays open on the upper side" in _panel(reg, "L")["identifiability"]["text"]
    monkeypatch.setitem(M.KEY_PARAMS, "M6", ("a", "L"))
    reg = M.evaluate(_m(), folds, _cfg(), identifiability=_ident(folds, [L_bad]))
    r = _row(reg, "M6a")
    assert r["status"] == "not_testable" and r["label"] == "Not testable with this data" and r["identifiability_gate"]
    assert "mixing length L is not identified by the data alone" in r["reason"] and r["basis"] is None
    assert r["statistic"]["value"] == pytest.approx(_panel(reg, "r_a")["value"])     # the statistic is kept
    assert {m["id"]: m["status"] for m in reg["mechanisms"]}["M6"] == "not_testable"
    assert "M6a" in reg["summary"]["not_testable"] and "M6a" not in reg["summary"]["supported"]
    ok = _prow("L", ci=(200.0, 420.0), run=256.0, data=260.0)
    assert _row(M.evaluate(_m(), folds, _cfg(), identifiability=_ident(folds, [ok])), "M6a")["status"] == "supported"
    far = _folds(a_values=(1000.0, 1010.0, 990.0))
    reg = M.evaluate(_m(), far, _cfg(), identifiability=_ident(far, [L_bad]))
    assert _row(reg, "M6a")["status"] == "contradicted" and not _row(reg, "M6a").get("identifiability_gate")


def test_sandwich_se_of_r_a_and_gamma_beside_the_jackknife_with_a_note_beyond_a_factor_of_two():
    folds = _folds(gammas=[-7.5, -8.0, -8.5, -8.0, -8.0])
    plain = M.evaluate({}, folds, _cfg())
    jk, jk_g = _panel(plain, "r_a")["se"], _panel(plain, "gamma")["se"]
    scale = F_TO_K * RHO_CP / 635.8
    rep = _ident(folds, [], a_se=3.0 * jk / scale, gamma_se=_gamma_F_per_m(1.5 * jk_g))
    reg = M.evaluate(_m(), folds, _cfg(), identifiability=rep)
    ra, g = _panel(reg, "r_a"), _panel(reg, "gamma")
    sw = ra["sandwich"]
    assert sw["value"] == pytest.approx(45.0 * scale) and sw["se"] == pytest.approx(3.0 * jk)
    assert sw["ratio"] == pytest.approx(3.0) and "differ by more than a factor of 2" in sw["note"]
    assert sw["note"] in ra["status_note"] and sw["note"] in _row(reg, "M6a")["note"]
    assert _row(reg, "M6a")["status"] == "supported"                 # the verdict stays on the jackknife
    assert "block sandwich" in _row(reg, "M6a")["context"] and "the verdict uses the jackknife" in _row(reg, "M6a")["context"]
    assert g["sandwich"]["value"] == pytest.approx(-8.0) and g["sandwich"]["se"] == pytest.approx(1.5 * jk_g)
    assert g["sandwich"]["ratio"] == pytest.approx(1.5) and g["sandwich"]["note"] is None   # within a factor of 2
    md = M.mechanisms_markdown(reg)
    assert "*Aerodynamic resistance r_a:* full-data reference fit" in md and "Note: the block-sandwich SE" in md


def test_lever_text_flags_a_conflict_and_a_poor_laplace_approximation():
    folds = _folds()
    rep = _ident(folds, [_prow("s", ci=(0.6, 0.9), run=0.5, data=0.75, share=0.2,
                               psens={"prior": 0.2, "likelihood": 0.01, "diagnosis": "weak_likelihood",
                                      "method": "laplace"},
                               conflict={"z": 2.5, "p": 0.012, "flag": True, "reason": None})])
    rep["posterior"]["laplace_adequate"] = False
    st = M.evaluate(_m(), folds, _cfg(), identifiability=rep)["gates"]["levers"]["Pct_Canopy"]["statement"]
    assert "identified by the data alone (data-only estimate 0.75; 95% profile interval 0.6 to 0.9); prior share 0.20" in st
    # power-scaling of a poor Laplace approximation is not presented as a finding (PHYSICS_PLAN W3 review X10)
    assert "weak likelihood" not in st
    assert "its posterior summaries and power-scaling are not reported (they rest on a Laplace approximation that is poor" in st
    assert "the prior–data conflict check finds a conflict (p = 0.012)" in st


@pytest.mark.parametrize("used", ["blocks", "neighbours"])
def test_reports_describe_the_block_sandwich_and_the_adjacent_block_sensitivity(used):
    """PHYSICS_PLAN W3 review X14: the register, its markdown and the writeups say that the intervals are Chandler &
    Bate's block sandwich by default, the adjacent-block (Conley-type) variant is a sensitivity, and the class rests
    on the wider of the two; both intervals are quoted; r_a/γ's sandwich is named after the V the study used."""
    from sparc.core.writeup import methods_markdown, model_card_markdown

    folds = _folds()
    L = _prow("L", ci=(200.0, 420.0), run=256.0, data=270.0, share=0.05)
    L["ci95_neighbours"] = [190.0, 480.0]
    L["ci95_conservative"] = [190.0, 480.0]
    s = _prow("s", ci=(0.15, 0.55), run=0.34, data=0.3, share=0.4)
    s["ci95_neighbours"] = [0.15, 0.55]                       # equal: quoted once
    jk = _panel(M.evaluate({}, folds, _cfg()), "r_a")["se"]
    rep = _ident(folds, [L, s], a_se=jk / (F_TO_K * RHO_CP / 635.8))
    rep["covariance"] = {"computed": ["blocks", "neighbours"], "used": used}
    m = {"name": "t", "config": _cfg(), **_m()}
    reg = M.evaluate(m, folds, _cfg(), identifiability=rep)
    m["mechanisms"] = reg
    P = reg["panel"]
    for text in (P["identifiability"], M.mechanisms_markdown(reg)):
        assert "Chandler & Bate 2007" in text and "block sandwich" in text and "Conley 1999" in text
        assert "adjacent blocks" in text and "wider of the two" in text
    Lp = _panel(reg, "L")["identifiability"]
    assert Lp["ci95_neighbours"] == [190.0, 480.0] and Lp["ci95_conservative"] == [190.0, 480.0]
    assert ("95% profile interval 200 to 420 m with the block sandwich, 190 to 480 m with adjacent blocks"
            in Lp["text"])
    assert "with adjacent blocks" not in _panel(reg, "s")["identifiability"]["text"]
    md = M.mechanisms_markdown(reg)
    assert "| data-only 95% CI (block sandwich) | with adjacent blocks |" in md
    assert "| 200 to 420 m | 190 to 480 m |" in md
    sw = _panel(reg, "r_a")["sandwich"]
    if used == "blocks":
        assert sw["kind"] == "block sandwich" and "(block sandwich, clusters = the CV blocks" in sw["text"]
    else:
        assert sw["kind"] == "adjacent-block sandwich" and "Conley 1999" in sw["text"]
    sec = methods_markdown(m)
    sec = sec[sec.index("## Identifiability"):]
    sec = sec[:sec.index("\n## ", 3)]
    assert "the block sandwich of Chandler & Bate 2007, the default" in sec and "Conley 1999" in sec
    assert "wider of the two 95% intervals" in sec
    assert "mixing length L (200 to 420 m; 190 to 480 m with adjacent blocks)" in sec
    card = model_card_markdown(m)
    csec = card[card.index("## Identifiability"):]
    assert "Chandler & Bate 2007's block sandwich" in csec and "Conley 1999" in csec and "wider of the two" in csec
    # nothing says the intervals are adjusted by the blocks alone
    assert "as clusters (Chandler & Bate 2007)" not in sec + csec + md


def test_writeups_report_identifiability_from_the_register():
    from sparc.core.writeup import methods_markdown, model_card_markdown

    folds = _folds()
    rep = _ident(folds, [_prow("L", ci=(200.0, 420.0), run=256.0, data=270.0, share=0.05),
                         _prow("s", ci=(0.0, 0.49), run=0.335, data=0.0, share=0.93,
                               at_bound={"side": "lower", "bound": 0.0},
                               psens={"prior": 0.1, "likelihood": 0.1, "diagnosis": "prior_data_conflict",
                                      "method": "profile"}),
                         _prow("kappa_canopy", cls="structurally_non_identifiable", ci=(None, 20.0), run=2.3,
                               data=19.5, share=0.99)])
    m = {"name": "t", "config": _cfg(), **_m()}
    m["mechanisms"] = M.evaluate(m, folds, _cfg(), identifiability=rep)
    md = methods_markdown(m)
    sec = md[md.index("## Identifiability"):]
    sec = sec[:sec.index("\n## ", 3)]
    assert "Raue et al. 2009" in sec and "23 cross-validation blocks of 2000 m" in sec
    assert "Identified by the data alone: mixing length L (200 to 420 m)." in sec
    assert "not excluded from zero: canopy shading strength s (0 to 0.49)" in sec
    assert "Not identified by the data alone: shade saturation scale κ (structurally non-identifiable)." in sec
    assert ("s = 0.335 in the full-data refit (run's fold mean 0.34), prior share 0.93; κ = 2.3 in the full-data "
            "refit (run's fold mean 2), prior share 0.99") in sec
    assert "power-scaling flags a potential prior–data conflict for s" in sec
    assert "key parameters not identified: M1 (κ)" in sec and "against the fold jackknife" in sec
    card = model_card_markdown(m)
    csec = card[card.index("## Identifiability"):]
    assert "- **What the data pin down.**" in csec and "- **What the penalty sets.**" in csec and "- **Conflict.**" in csec
    # a study made for another fit: one line, no numbers from it
    rep["run"]["fold_means"]["L"] = 300.0
    m["mechanisms"] = M.evaluate(m, folds, _cfg(), identifiability=rep)
    md = methods_markdown(m)
    assert "## Identifiability\n\nNot used: identifiability study ignored" in md and "200 to 420" not in md
    # no study: no section; a study the register predates: say so
    m["mechanisms"] = M.evaluate(m, folds, _cfg(), identifiability=False)
    assert "## Identifiability" not in methods_markdown(m) and "## Identifiability" not in model_card_markdown(m)
    m["mechanisms"].pop("identifiability")
    m["identifiability"] = rep
    assert "the mechanism register predates it" in methods_markdown(m)


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


def test_cli_ignores_a_physics_json_left_by_an_earlier_run(small_run, tmp_path):
    # reviewer case: re-run the same config without physics into a directory that holds an old physics.json
    import copy
    import shutil

    from sparc.core import cli
    from sparc.core.pipeline import run_core

    _, cfg, res = small_run
    run = tmp_path / "rerun"
    shutil.copytree(res.run_dir, run)
    cfg2 = copy.deepcopy(cfg)
    cfg2.raw["models"] = dict(cfg.raw["models"], physics=False)
    run_core(cfg2, stages=("S0", "S1", "S2", "S3"), fast=True, run_dir=run)
    man = json.loads((run / "manifest.json").read_text())
    assert man["physics"] is None and (run / "physics.json").exists()       # the pipeline leaves the old file
    assert man["mechanisms"]["physics_status"] == "absent"
    assert cli.main(["mechanisms", "-r", str(run)]) == 0
    reg = json.loads((run / "mechanisms.json").read_text())
    assert reg["physics_status"] == "absent" and "records no physics fit" in reg["inputs"]["note"]
    assert _row(reg, "M6a")["status"] == "not_testable" and "no physics model" in _row(reg, "M6a")["reason"]
    assert all(r["value"] is None for r in reg["panel"]["rows"])
    assert json.loads((run / "manifest.json").read_text())["mechanisms"]["physics_status"] == "absent"
