"""Pooling cities (sparc.core.identify.pool): recovers a planted pooled effect and tau, LOCO coverage near
nominal, reference values, and the per-window pooling of the batch driver's run table."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sparc.core.identify import pool as P


def _cities(rng, k, mu=-0.10, tau=0.05, se_range=(0.02, 0.08)):
    se = rng.uniform(*se_range, k)
    return mu + tau * rng.standard_normal(k) + se * rng.standard_normal(k), se


def test_no_heterogeneity_is_the_fixed_effect_mean():
    y, se = np.array([-0.1, -0.1, -0.1]), np.array([0.02, 0.03, 0.04])
    f = P.random_effects(y, se)
    w = 1 / se ** 2
    assert f["tau2"] == 0.0 and f["mu"] == pytest.approx(-0.1) and f["se"] == pytest.approx(np.sqrt(1 / w.sum()))


def test_dersimonian_laird_reference_value():
    # hand-computed: equal SEs 0.1, y = (0, 0.2, 0.4): w = 100, Q = 100*0.08 = 8, C = 300 - 30000/300 = 200,
    # tau² = (8 - 2) / 200 = 0.03
    assert P.dersimonian_laird([0.0, 0.2, 0.4], [0.1, 0.1, 0.1]) == pytest.approx(0.03)
    # with equal SEs REML has the closed form tau² = var(y, ddof=1) - s² = 0.04 - 0.01
    assert P.reml_tau2([0.0, 0.2, 0.4], [0.1, 0.1, 0.1]) == pytest.approx(0.03, abs=1e-8)


def test_recovers_the_planted_effect_and_tau():
    rng = np.random.default_rng(7)
    y, se = _cities(rng, 400)
    f = P.random_effects(y, se)
    assert f["method"] == "reml"
    assert f["mu"] == pytest.approx(-0.10, abs=0.01) and f["tau"] == pytest.approx(0.05, abs=0.01)
    assert f["pi_lo"] < -0.10 - 1.5 * 0.05 and f["pi_hi"] > -0.10 + 1.5 * 0.05


def test_loco_coverage_is_near_nominal():
    rng = np.random.default_rng(11)
    cov = [P.loco(*_cities(rng, 12))["coverage"] for _ in range(200)]
    assert 0.92 <= float(np.mean(cov)) <= 0.98


def test_stouffer_combines_cooling_evidence():
    s = P.stouffer([-1.0, -1.0, -1.0, -1.0])
    assert s["z"] == pytest.approx(-2.0) and s["p_cooling"] == pytest.approx(0.02275, abs=1e-4)


def test_pool_runs_takes_only_pooled_runs_per_window(tmp_path):
    rows = [{"city": c, "window": "evening", "pooled": ok, "street_100": e, "street_100_se": s}
            for c, ok, e, s in (("providence_ri", True, -0.077, 0.019), ("seattle_wa", False, -0.033, 0.053),
                                ("a", True, -0.05, 0.03), ("b", True, -0.02, 0.04), ("c", True, -0.06, 0.02))]
    rows.append({"city": "providence_ri", "window": "midday", "pooled": False, "street_100": 0.02, "street_100_se": 0.08})
    res = P.pool_runs(rows)
    ev = res["windows"]["evening"]
    assert ev["cities"] == ["providence_ri", "a", "b", "c"] and ev["fit"]["k"] == 4 and len(ev["loco"]["rows"]) == 4
    assert res["windows"]["midday"]["k"] == 0
    md = P.pool_markdown(res)
    assert "| evening | 4 (providence_ri, a, b, c) |" in md and "| midday | 0 |" in md
    json.dumps(res)


def test_cli_pool_reads_the_driver_table(tmp_path):
    import pandas as pd

    from sparc.core.identify.__main__ import main

    pd.DataFrame([{"city": "providence_ri", "run": "2020-07-29 evening", "window": "evening", "pooled": True,
                   "street_100": -0.077, "street_100_se": 0.019},
                  {"city": "seattle_wa", "run": "2020-07-27 evening", "window": "evening", "pooled": False,
                   "street_100": -0.033, "street_100_se": 0.053}]).to_csv(tmp_path / "run_status.csv", index=False)
    assert main(["pool", "--status", str(tmp_path / "run_status.csv")]) == 0
    res = json.loads((tmp_path / "pool.json").read_text())
    assert res["windows"]["evening"]["fit"]["mu"] == pytest.approx(-0.077) and res["windows"]["evening"]["k"] == 1


def test_street_300_has_its_own_pool_entry_flag():
    """Agent decision A4: street_300 is pooled under the same rule applied to its own verdict."""
    rows = [{"city": "a", "window": "evening", "pooled": True, "pooled_300": False, "street_100": -0.05,
             "street_100_se": 0.02, "street_300": -0.10, "street_300_se": 0.03},
            {"city": "b", "window": "evening", "pooled": False, "pooled_300": True, "street_100": -0.01,
             "street_100_se": 0.03, "street_300": -0.12, "street_300_se": 0.04}]
    assert P.pool_runs(rows)["windows"]["evening"]["cities"] == ["a"]
    r3 = P.pool_runs(rows, "street_300")
    assert r3["windows"]["evening"]["cities"] == ["b"] and r3["windows"]["evening"]["fit"]["mu"] == pytest.approx(-0.12)
    assert P.pool_markdown(r3).startswith("# Pooled canopy effect (+10 pp within 300 m")


def test_leave_one_out_sensitivity_and_exclusions():
    """Owner decision 12: every pooled result with each city left out in turn, and pooled without named cities."""
    rows = [{"city": c, "window": "evening", "pooled": True, "street_100": e, "street_100_se": 0.03}
            for c, e in (("a", -0.08), ("b", -0.05), ("c", -0.02))]
    res = P.pool_runs(rows)
    lo = res["windows"]["evening"]["leave_one_out"]
    assert [r["left_out"] for r in lo] == ["a", "b", "c"] and all(r["k"] == 2 for r in lo)
    assert lo[0]["mu"] == pytest.approx(-0.035) and lo[2]["mu"] == pytest.approx(-0.065)
    assert "## Each city left out in turn" in P.pool_markdown(res)
    ex = P.pool_runs(rows, exclude=("a", "c"))
    assert ex["windows"]["evening"]["cities"] == ["b"] and "Excluded from every window: a, c." in P.pool_markdown(ex)
