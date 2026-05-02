"""Wager (2025) causal-inference audit — gap-by-gap test suite.

One test per gap. Every test must pass before the audit is "closed".
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Phase 0 — registry exists and is wired correctly
# ---------------------------------------------------------------------------

def test_audit_registry_exists():
    from sparc.causal import _audit
    assert len(_audit.GAP_SPECS) == 10
    assert set(_audit.GAP_SPECS.keys()) == set(range(1, 11))
    assert all(gid in _audit.WAGER2025_GAPS_ADDRESSED for gid in range(1, 11))


def test_audit_report_renders():
    from sparc.causal import _audit
    text = _audit.report()
    for gid in range(1, 11):
        assert f"Gap {gid:2d}" in text


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------

def test_gap1_overlap_detects_extrapolation():
    from sparc.causal.overlap import GeneralizedPropensityOverlap

    rng = np.random.default_rng(42)
    n = 600
    X = rng.normal(size=(n, 2))
    W = X[:, 0] * 0.5 + rng.normal(scale=0.5, size=n)

    ov = GeneralizedPropensityOverlap().fit(X, W)
    in_dom = ov.assess_scenario(x_baseline=np.zeros(2), delta_w=0.0)
    far = ov.assess_scenario(x_baseline=np.array([6.0, 6.0]), delta_w=8.0)

    assert in_dom["flag"] == "HIGH"
    assert far["flag"] == "EXTRAPOLATED"


def test_gap2_rate_null_effect_covers_zero():
    from sparc.causal.cate_validation import compute_rate

    rng = np.random.default_rng(0)
    n = 500
    scores = rng.normal(size=n)
    dr_scores = rng.normal(size=n)  # null effect
    out = compute_rate(scores, dr_scores, n_bootstrap=200, random_state=0)
    lo, hi = out["rate_ci"]
    assert lo <= 0.0 <= hi


def test_gap9_cbps_recovers_treatment_effect():
    from sparc.causal.balancing import CBPSEstimator

    rng = np.random.default_rng(1)
    n = 800
    X = rng.normal(size=(n, 3))
    logit = 0.5 * X[:, 0] - 0.3 * X[:, 1]
    p = 1 / (1 + np.exp(-logit))
    W = (rng.uniform(size=n) < p).astype(float)
    Y = 1.5 * W + X @ np.array([1.0, -0.5, 0.2]) + rng.normal(scale=0.5, size=n)

    est = CBPSEstimator().fit(X, W)
    tau = est.ate(Y)
    assert abs(tau - 1.5) < 0.3


def test_gap10_edge_nuisance_cache_does_not_cross_pollinate():
    from sparc.causal.counterfactual_engine import EdgeNuisanceCache

    cache = EdgeNuisanceCache()
    fit_a = cache.get_or_fit(
        parent="X", child="Ya", fold_id=0, controls_hash="h",
        fit_fn=lambda: ("model_for_Ya", 1.0),
    )
    fit_b = cache.get_or_fit(
        parent="X", child="Yb", fold_id=0, controls_hash="h",
        fit_fn=lambda: ("model_for_Yb", 2.0),
    )
    assert fit_a != fit_b
    assert cache.size() == 2


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

def test_gap3_spillover_recovers_known_decay():
    from sparc.causal.interference import NetworkInterferenceModel

    rng = np.random.default_rng(7)
    n = 400
    coords = rng.uniform(0, 10, size=(n, 2))
    W = rng.normal(size=n)
    # Manually build spillover: Y = 1.0*W_i + 0.5*mean(neighbors)
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=9)
    neighbor_mean = np.array([W[idx[i, 1:]].mean() for i in range(n)])
    Y = 1.0 * W + 0.5 * neighbor_mean + rng.normal(scale=0.3, size=n)

    model = NetworkInterferenceModel(k=8).fit(coords, W, Y)
    decomp = model.decompose()
    assert abs(decomp["direct_effect"] - 1.0) < 0.3
    assert abs(decomp["spillover_effect"] - 0.5) < 0.3


def test_gap4_iv_consistency_under_unmeasured_confounding():
    from sparc.causal.iv import CrossFitIV

    rng = np.random.default_rng(2)
    n = 1500
    Z = rng.normal(size=n)
    U = rng.normal(size=n)  # unmeasured confounder
    W = 0.7 * Z + 0.5 * U + rng.normal(scale=0.3, size=n)
    Y = 1.2 * W + 1.0 * U + rng.normal(scale=0.3, size=n)
    # OLS would be biased upward; IV should recover ~1.2.

    iv = CrossFitIV(n_folds=3).fit(Z=Z, W=W, Y=Y)
    assert abs(iv.coef_ - 1.2) < 0.2


def test_gap5_mte_flat_for_linear_treatment_effect():
    from sparc.causal.iv import LocalIV

    rng = np.random.default_rng(3)
    n = 2000
    Z = rng.uniform(size=n)
    W = (Z + rng.normal(scale=0.1, size=n) > 0.5).astype(float)
    Y = 0.8 * W + rng.normal(scale=0.2, size=n)

    mte = LocalIV().fit(Z=Z, W=W, Y=Y)
    curve = mte.curve(grid=np.linspace(0.2, 0.8, 7))
    # Constant treatment effect → MTE roughly flat.
    assert curve.std() < 0.4


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------

def test_gap6_sequential_aipw_handles_treatment_confounder_feedback():
    from sparc.causal.dynamic import SequentialAIPW

    rng = np.random.default_rng(4)
    n = 800
    # Two periods with treatment-confounder feedback (Hernán-Robins style):
    #  L1 -> A1 -> L2 -> A2 -> Y, with A1 affecting L2 affecting Y.
    L1 = rng.normal(size=n)
    A1 = (L1 + rng.normal(scale=0.5, size=n) > 0).astype(float)
    L2 = 0.5 * L1 + 0.4 * A1 + rng.normal(scale=0.5, size=n)
    A2 = (0.5 * L2 + rng.normal(scale=0.5, size=n) > 0).astype(float)
    # True effect of always-treat vs. never-treat = 0 by construction.
    Y = 0.0 * (A1 + A2) + L1 + 0.3 * L2 + rng.normal(scale=0.3, size=n)

    est = SequentialAIPW().fit(
        history=[
            {"L": L1.reshape(-1, 1), "A": A1},
            {"L": L2.reshape(-1, 1), "A": A2},
        ],
        Y=Y,
    )
    psi_treat = est.evaluate_regime([1.0, 1.0])
    psi_ctrl = est.evaluate_regime([0.0, 0.0])
    assert abs(psi_treat - psi_ctrl) < 0.4  # null recovered


def test_gap7_did_recovers_known_att():
    from sparc.causal.panel import DifferenceInDifferences

    rng = np.random.default_rng(5)
    n_units = 100
    n_periods = 6
    treat_period = 3
    # Half treated, half control. ATT = 2.0 starting at treat_period.
    treated_ids = np.arange(n_units // 2)
    rows = []
    for u in range(n_units):
        unit_fe = rng.normal()
        for t in range(n_periods):
            time_fe = 0.1 * t
            treated = (u in treated_ids) and t >= treat_period
            y = unit_fe + time_fe + 2.0 * treated + rng.normal(scale=0.2)
            rows.append({"unit": u, "time": t, "y": y,
                         "treat": int(u in treated_ids),
                         "post": int(t >= treat_period)})
    import pandas as pd
    df = pd.DataFrame(rows)
    did = DifferenceInDifferences().fit(df, outcome="y",
                                        unit="unit", time="time",
                                        treat="treat", post="post")
    assert abs(did.att_ - 2.0) < 0.3


# ---------------------------------------------------------------------------
# Phase 4
# ---------------------------------------------------------------------------

def test_gap8_policy_learning_finds_optimal_under_budget():
    from sparc.decision.policy_learning import EmpiricalWelfareMaximizer

    rng = np.random.default_rng(6)
    n = 200
    # Each unit has a feature x; treatment effect = x - 0.3.
    x = rng.uniform(size=n)
    aipw_scores = (x - 0.3) + rng.normal(scale=0.2, size=n)
    cost = np.ones(n)
    budget = 60  # treat ~30% of units

    pol = EmpiricalWelfareMaximizer(policy_class="linear",
                                    budget=budget).fit(
        X=x.reshape(-1, 1), aipw_scores=aipw_scores, cost=cost,
    )
    treated = pol.recommend()
    assert treated.sum() <= budget
    # Optimal policy treats high-x units; mean x of treated should be > 0.5.
    assert x[treated.astype(bool)].mean() > 0.5


# ---------------------------------------------------------------------------
# Phase 5 — integration smoke test
# ---------------------------------------------------------------------------

def test_phase5_audit_registry_addresses_all_ten():
    from sparc.causal import _audit
    _audit._autodetect()
    n_done = sum(_audit.WAGER2025_GAPS_ADDRESSED.values())
    assert n_done == 10, _audit.report()
