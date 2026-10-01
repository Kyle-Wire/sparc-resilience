"""Planner pack pieces: plantable space, exposure, equity, hot days, hexagons, sites, S7 cap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparc.core import planner as P


def test_plantable_headroom_caps_by_open_land_and_existing_canopy():
    lay = pd.DataFrame({"lc_grass": [0.5, 0.0, 0.0], "lc_bare": [0.0, 0.0, 0.1], "lc_built": [0.5, 1.0, 0.0]})
    h = P.plantable_headroom(np.array([10.0, 0.0, 95.0]), lay, paved_share=0.2)
    assert h == pytest.approx([60.0, 20.0, 5.0])


def test_exposure_counts_people_above_thresholds():
    t = np.array([88.0, 90.0, 91.0, 95.5])
    p = np.array([10.0, 20.0, 30.0, 40.0])
    rows = P.exposure_table(t, p, [90, 95], {"future": 1.0}, adaptation=np.full(4, -1.5))
    today, today_ad, fut, fut_ad = rows
    assert today["people_ge_90"] == 90.0 and today["people_ge_95"] == 40.0
    assert today_ad["people_ge_90"] == 40.0 and fut["people_ge_90"] == 90.0
    assert fut_ad["people_ge_90"] == 70.0 and today["person_mean_temp"] == pytest.approx(92.3)


def test_concentration_index_sign():
    rank = np.arange(100.0)
    assert P.concentration_index(rank + 1.0, rank) > 0.2
    assert P.concentration_index(np.ones(100), rank) == pytest.approx(0.0, abs=1e-12)
    assert P.concentration_index(100.0 - rank, rank) < -0.2


def test_hot_days_per_summer_with_offsets_and_shift():
    idx = pd.date_range("1995-06-01", "1996-08-31", freq="D")
    idx = idx[idx.month.isin([6, 7, 8])]
    tmax = pd.Series(np.where(np.arange(idx.size) % 10 == 0, 92.0, 85.0), index=idx)   # 10% of days at 92
    d = P.hot_days(tmax, np.array([0.0, 6.0]), [90], years=(1995, 1996))
    assert d["90"][0] == pytest.approx(9.5) and d["90"][1] == pytest.approx(92.0)        # 19 of 184 days; all
    d2 = P.hot_days(tmax, np.array([0.0]), [90], years=(1995, 1996), shift=5.0)
    assert d2["90"][0] == pytest.approx(92.0)


def test_hexagons_partition_points_and_sum_people():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 2000, 3000), rng.uniform(0, 2000, 3000)
    h = P.summarize_hex(x, y, {"people": np.ones(3000), "t": x / 100.0}, 250.0)
    assert h["people"].sum() == pytest.approx(3000.0) and h["n_cells"].sum() == 3000
    key, cx, cy = P.hex_ids(x, y, 250.0)
    assert np.max(np.hypot(x - cx, y - cy)) <= 250.0 / np.sqrt(3) + 1e-6       # within the hexagon's radius


def test_logger_sites_and_matched_controls():
    rng = np.random.default_rng(1)
    n = 2000
    can, imp = rng.uniform(0, 80, n), rng.uniform(0, 100, n)
    x, y = rng.uniform(0, 8000, n), rng.uniform(0, 8000, n)
    sites = P.logger_sites(can, imp, rng.uniform(size=n), x, y, n=16, min_spacing_m=300.0)
    assert 8 <= len(sites) <= 16 and set(sites["role"]) == {"low canopy", "high canopy"}
    d = np.hypot(sites.x_m.to_numpy()[:, None] - sites.x_m.to_numpy()[None], sites.y_m.to_numpy()[:, None]
                 - sites.y_m.to_numpy()[None])
    assert d[np.triu_indices(len(sites), 1)].min() >= 300.0
    treated = (x < 2000) & (y < 2000)
    pairs = P.matched_controls(treated, np.column_stack([can, imp]), x, y, min_distance_m=1000.0, n_pairs=10)
    assert len(pairs) == 10 and pairs["control"].is_unique and not treated[pairs["control"]].any()


def test_s7_cap_limits_allocation():
    from types import SimpleNamespace

    from sparc.core.optimize import build_segments

    n = 50
    maps = pd.DataFrame({"footprint_effect_per_unit": np.full(n, -0.02), "saturation_scale_ds": np.full(n, 20.0),
                         "headroom": np.full(n, 60.0)})
    vr = SimpleNamespace(direction="increase", maps=maps, doses=[0, 10, 20, 40])
    full = build_segments(vr, 1.0)
    capped = build_segments(vr, 1.0, cap=np.full(n, 5.0))
    assert capped.groupby("cell")["x_max"].sum().max() == pytest.approx(5.0)
    assert full.groupby("cell")["x_max"].sum().max() == pytest.approx(40.0)
    w = build_segments(vr, 1.0, benefit_weight=np.linspace(0.5, 1.5, n))
    assert w.groupby("cell")["benefit_per_unit"].max().is_monotonic_increasing
