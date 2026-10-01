"""Simulation check on a real-like layout: generators, product emulation, gate, summary."""

from __future__ import annotations

import numpy as np
import pytest

from sparc.core import simcheck as S


@pytest.fixture(scope="module")
def layout(synthetic_core_data, synthetic_city):
    cfg, data = synthetic_core_data
    return S.Layout(cfg=cfg, df=synthetic_city.frame, data=data,
                    roles=cfg.raw["physics"]["roles"])


@pytest.mark.parametrize("kind", S.GENERATORS)
def test_generators_plant_the_calibrated_canopy_effect(layout, kind):
    gen = S.Generator(kind, layout, np.random.default_rng(0))
    td = gen.truth_delta()
    target = 0.0 if kind == "null" else S.TRUE_EFFECT_F
    assert float(np.mean(td)) == pytest.approx(target, abs=1e-6)
    sig = gen.signal(gen.C0, gen.I0)
    assert np.isfinite(sig).all() and sig.std() < 10.0


def test_product_emulation_classes_values_and_gate(layout):
    rng = np.random.default_rng(1)
    feats = S._product_features(layout)
    X = S._covariates(layout)
    gen = S.Generator("additive", layout, rng)
    T = 88.0 + gen.signal(gen.C0, gen.I0) + 0.5 * rng.standard_normal(gen.C0.size)
    prod = S.emulate_product(T, layout, rng, feats, street_m=150.0, int_share=0.7)
    assert np.mean(np.isclose(prod, np.round(prod))) == pytest.approx(0.7, abs=0.03)
    assert np.corrcoef(prod, T)[0, 1] > 0.5                      # the product still tracks the field
    real = {"mean": 88.0, "sd": float(prod.std()), "integer_share": 0.7,
            "residual_range_m": S.residual_range(prod, X, layout.grid)}
    st = S.gate_stats(prod, layout, X, real)
    assert st["pass"] and st["sd_ratio"] == pytest.approx(1.0)


def test_residual_range_tracks_field_scale(layout):
    from sparc.core.synthetic import gaussian_random_field

    g = layout.grid
    X = np.zeros((g.n_points, 1))
    r = []
    for cells in (2.0, 8.0):
        z = gaussian_random_field(g.shape, cells, np.random.default_rng(3))[g.iy, g.ix]
        r.append(S.residual_range(z, X, g))
    assert r[1] > 2.0 * r[0]


def test_summary_rates_and_bias_correction():
    rows = []
    for s in range(10):
        rows.append({"generator": "additive", "seed": s, "gate": {"pass": True}, "share": 0.9 + 0.01 * s,
                     "ci_covers_truth": s < 9, "causal_covers_truth": True, "rank_corr": 0.6,
                     "interval_coverage": 0.9, "oof_r2": 0.6})
        rows.append({"generator": "physics", "seed": s, "gate": {"pass": True}, "share": 0.95,
                     "ci_covers_truth": True, "causal_covers_truth": True, "interval_coverage": 0.9, "oof_r2": 0.6})
        rows.append({"generator": "null", "seed": s, "gate": {"pass": True}, "share": None,
                     "significant": s == 0, "causal_significant": False, "interval_coverage": 0.9, "oof_r2": 0.5})
    rows.append({"generator": "null", "seed": 99, "error": "boom"})
    summ = S.summarize(rows)
    g = summ["generators"]
    assert g["null"]["false_positive_rate"] == pytest.approx(0.1) and g["additive"]["ci_coverage"] == pytest.approx(0.9)
    b = summ["bias_correction"]
    assert b["stable"] and b["correction_factor"] == pytest.approx(1.0 / np.median([0.945, 0.95]))
    assert summ["n_errors"] == 1
    assert "null" in S.simcheck_markdown(summ)
