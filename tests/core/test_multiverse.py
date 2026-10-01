"""Multiverse: variant configs, rank agreement and the stability summary."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sparc.core import multiverse as M


def test_apply_variant_sets_dotted_keys_and_drops_forcing(synthetic_core_data):
    cfg, _ = synthetic_core_data
    cfg = __import__("copy").deepcopy(cfg)
    cfg.raw["physics"]["forcing_info"] = {"date": "x"}
    v = M.apply_variant(cfg, M.VARIANTS["generic_forcing"], "generic_forcing")
    assert v.raw["physics"]["sw_down"] == 800.0 and v.raw["physics"]["wind"] is None
    assert "forcing_info" not in v.raw["physics"] and v.name.endswith("_mv_generic_forcing")
    v = M.apply_variant(cfg, M.VARIANTS["blocks_3km"], "blocks_3km")
    assert v.raw["cv"]["block_m"] == 3000.0 and cfg.raw["cv"]["block_m"] == "auto"
    assert not v.raw["cv"]["baselines"] and not v.raw["climate"]["enabled"]


def test_rank_agreement_identical_and_shuffled():
    rng = np.random.default_rng(0)
    a = rng.normal(size=2000)
    same = M.rank_agreement(a, a * 2.0)
    assert same["kendall_tau"] == pytest.approx(1.0) and same["top_decile_jaccard"] == pytest.approx(1.0)
    rnd = M.rank_agreement(a, rng.permutation(a))
    assert abs(rnd["kendall_tau"]) < 0.1 and rnd["top_decile_jaccard"] < 0.2


def test_rank_agreement_all_nan_map_is_nan_not_an_error():
    a = np.random.default_rng(0).normal(size=50)
    out = M.rank_agreement(a, np.full(50, np.nan))
    assert np.isnan(out["kendall_tau"]) and np.isnan(out["top_decile_jaccard"])
    assert M._median_finite([out["kendall_tau"], 0.4, 0.8]) == pytest.approx(0.6)
    assert M._median_finite([out["kendall_tau"]]) is None


def test_summarize_effect_and_priority_stability(tmp_path):
    ids = np.arange(500)
    rng = np.random.default_rng(1)
    m = rng.normal(size=500)
    for name, delta, noise in (("baseline", -0.5, 0.0), ("blocks_1km", -0.4, 0.1), ("no_physics", 0.05, 2.0)):
        (tmp_path / f"{name}.json").write_text(json.dumps({
            "variant": name, "label": name, "r2": 0.5, "rmse": 1.0, "block_m": 2000, "stacker_choice": "x",
            "seconds": 1, "scenarios": {"Canopy +10": {"mean_delta": delta, "se": 0.1}}}))
        np.savez(tmp_path / f"{name}_maps.npz", ids=ids, canopy=m + noise * rng.normal(size=500))
    s = M.summarize(tmp_path, ["baseline", "blocks_1km", "no_physics"])
    assert s["effects"]["Canopy +10"]["sign_stability"] == pytest.approx(2 / 3)
    assert s["priority"]["blocks_1km"]["canopy"]["kendall_tau"] > 0.8 > s["priority"]["no_physics"]["canopy"]["kendall_tau"]
    assert "sign stable" in M.multiverse_markdown(s, "°F")
