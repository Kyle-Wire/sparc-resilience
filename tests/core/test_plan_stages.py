"""``plan_stages`` (SPEC §5.4, §11 item 4) is built on the gating predicates ``run_core`` uses: its node states
equal the ``stage.start`` / ``stage.skip`` events of real runs, and its unit counts reproduce the reference
counts of the recorded Providence run."""

from __future__ import annotations

import copy
import itertools

import pytest

ALL = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")
S0_S3 = ("S0", "S1", "S2", "S3")


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    from sparc.core.synthetic import write_demo_project

    return write_demo_project(tmp_path_factory.mktemp("plan_stages"), n=40, seed=0)


def _light(demo):
    """The demo config with cheap models: node states do not depend on the model set."""
    from sparc.core.config import load_core_config

    cfg = load_core_config(demo["config_path"])
    cfg.raw["models"] = {"ols": True, "mgwr": False, "gwrf": False, "gam": True, "physics": False}
    cfg.raw["stacker"].update(epochs=20, tune_lambda=[0.0])
    cfg.raw["cv"]["baselines"] = ["idw"]
    cfg.raw["cv"]["distance_curve"] = {"enabled": False, "block_m": [0, 300]}
    cfg.raw["causal"]["n_boot"] = 10
    for v, d in (("canopy", [0, 10]), ("impervious", [0, 10]), ("albedo", [0, 0.1])):
        cfg.raw["actionable"][v]["doses"] = d
    return cfg


def _toggle(cfg, climate=True, causal=True, optimize=True, baselines=True, partitions=None):
    cfg.raw["climate"]["enabled"] = climate
    cfg.raw["causal"]["enabled"] = causal
    cfg.raw["optimize"]["enabled"] = optimize
    if not baselines:
        cfg.raw["cv"]["baselines"] = False
    if partitions is not None:
        cfg.raw["cv"]["distance_curve"]["block_m"] = partitions
    return cfg


def _run_events(cfg, stages, **kw) -> list[dict]:
    from sparc.core import progress
    from sparc.core.pipeline import run_core

    events: list[dict] = []
    progress.configure(events.append, heartbeat_s=0)
    try:
        run_core(cfg, stages=stages, write=False, **kw)
    finally:
        progress.reset()
    return events


def _assert_plan_matches_run(cfg, stages, fast=False, coarse=None, cv_curve=None):
    from sparc.core.pipeline import STAGE_IDS, plan_stages

    nodes = plan_stages(copy.deepcopy(cfg), stages, fast=fast, coarse=coarse, cv_curve=cv_curve, write=False)
    events = _run_events(copy.deepcopy(cfg), stages, fast=fast, coarse=coarse, cv_curve=cv_curve)
    started = [e["stage"] for e in events if e["type"] == "stage.start"]
    skipped = {e["stage"]: e["reason"] for e in events if e["type"] == "stage.skip"}
    assert sorted(started + list(skipped), key=STAGE_IDS.index) == list(STAGE_IDS)    # each node exactly once
    for n in nodes:
        if n["state"] == "will_run":
            assert n["id"] in started, (n, skipped.get(n["id"]))
        else:
            assert skipped.get(n["id"]) == n["reason"], (n, n["id"] in started)
    emitted = [e for e in events if e["type"] == "run.plan"][-1]["nodes"]
    assert [(n["id"], n["state"], n["reason"]) for n in emitted] == [(n["id"], n["state"], n["reason"]) for n in nodes]
    return {n["id"]: (n["state"], n["reason"]) for n in nodes}


CASES = {
    "all_fast": (dict(), ALL, dict(fast=True)),
    "s0_s1": (dict(), ("S0", "S1"), dict()),
    "s0_s3_coarse_curve": (dict(), S0_S3, dict(fast=True, coarse=60.0, cv_curve=True)),
    "s6_only": (dict(), ("S6",), dict(coarse=60.0)),
    "no_climate_no_budget": (dict(climate=False), ("S0", "S1", "S2", "S3", "S4", "S5", "S7"), dict(coarse=60.0)),
    "all_off": (dict(causal=False, optimize=False, baselines=False, partitions=[50_000]), ALL,
                dict(coarse=60.0, cv_curve=True)),
}


@pytest.mark.parametrize("case", list(CASES))
def test_plan_states_equal_real_run_events(demo, case):
    toggles, stages, kw = CASES[case]
    cfg = _toggle(_light(demo), **toggles)
    if case == "no_climate_no_budget":
        cfg.raw["optimize"]["budget"] = None
    states = _assert_plan_matches_run(cfg, stages, **kw)
    expected = {
        "all_fast": {"cv_curve": ("skipped", "disabled_by_config:cv.distance_curve.enabled"), "S7": ("will_run", None)},
        "s0_s1": {"S1": ("will_run", None), "S2_S3": ("skipped", "not_requested"), "climate": ("skipped", "requires_S5")},
        "s0_s3_coarse_curve": {"cv_curve": ("will_run", None), "S4": ("skipped", "not_requested")},
        "s6_only": {"S0": ("will_run", "required_by:S6"), "S2_S3": ("will_run", "required_by:S6"),
                    "S4": ("will_run", "required_by:S6"), "S5": ("skipped", "not_requested"),
                    "climate": ("skipped", "requires_S5"), "S6": ("will_run", None),
                    "S7": ("skipped", "not_requested")},
        "no_climate_no_budget": {"climate": ("skipped", "disabled_by_config:climate.enabled"),
                                 "S6": ("skipped", "not_requested"), "S7": ("skipped", "no_budget")},
        "all_off": {"baselines": ("skipped", "disabled_by_config:cv.baselines"),
                    "cv_curve": ("skipped", "no_partitions"), "S6": ("skipped", "disabled_by_config:causal.enabled"),
                    "S7": ("skipped", "disabled_by_config:optimize.enabled")},
    }[case]
    for nid, st in expected.items():
        assert states[nid] == st, nid


def test_more_skip_reasons_from_the_shared_predicates(demo):
    from sparc.core.pipeline import plan_stages

    cfg = _light(demo)
    cfg.raw["causal"]["treatments"] = []
    cfg.raw["optimize"]["variable"] = "elevation"                     # not a lever: no S4 response
    by = {n["id"]: (n["state"], n["reason"]) for n in plan_stages(cfg, ("S6", "S7"), n_points=1, extent=1e4, dx=30)}
    assert by["S6"] == ("skipped", "no_treatments") and by["S7"] == ("skipped", "no_responses")
    assert by["S4"] == ("will_run", "required_by:S6")
    nodes = plan_stages(_light(demo), ALL, done={"S3", "baselines", "S4"}, n_points=1, extent=1e4, dx=30)
    by = {n["id"]: (n["state"], n["reason"], n["units"]) for n in nodes}
    assert by["S1"][:2] == by["S2_S3"][:2] == by["baselines"][:2] == by["S4"][:2] == ("cached", "checkpoint")
    assert by["S5"][2]["engine_pass"] == 8 + 1                         # the engine's baseline pass moves to S5
    assert by["S7"][2] == {"engine_pass": 1, "pareto": 1}
    assert {n["id"]: n["checkpoint_key"] for n in nodes}["S2_S3"] == "S3"


def test_plan_stages_counts_partitions_with_s0_on_demand(demo):
    from sparc.core.pipeline import plan_stages

    cfg = _light(demo)
    cfg.raw["cv"]["distance_curve"]["block_m"] = [0, 300, 50_000]     # 50 km is beyond a third of the extent
    nodes = plan_stages(cfg, S0_S3, fast=True, cv_curve=True)           # n_points / extent / dx from S0
    curve = next(n for n in nodes if n["id"] == "cv_curve")
    s2 = next(n for n in nodes if n["id"] == "S2_S3")
    assert curve["state"] == "will_run"
    assert curve["units"]["base_fit:ols"] == 2 * s2["units"]["base_fit:ols"] == 2 * 3
    assert curve["units"]["baseline_fit:idw"] == 2 * 3 and curve["units"]["checkpoint_save"] == 1


def test_reference_unit_counts_of_the_recorded_providence_run(providence_config_path):
    """SPEC §5.4: 5 folds, five base models, no wind (no advection refits), tune_lambda [0, 0.1, 1] (C = 5), no
    in-run baselines, a CV curve with 3 partitions, levers with 7/7/5 doses and 9/9/6 marginal passes, 12
    scenarios, 3 treatments, table climate and S7 — computed without running anything."""
    from sparc.core.config import load_core_config
    from sparc.core.pipeline import plan_stages, plan_totals

    cfg = load_core_config(providence_config_path)
    cfg.raw["physics"]["wind"] = None
    cfg.raw["stacker"]["tune_lambda"] = [0.0, 0.1, 1.0]
    cfg.raw["cv"]["baselines"] = False
    cfg.raw["cv"]["distance_curve"] = {"enabled": True, "block_m": [0, 500, 1000]}
    nodes = plan_stages(cfg, ALL, n_points=54701, extent=8600.0, dx=30.0, block_m=2000.0)
    by = {n["id"]: n for n in nodes}
    models = ("mgwr", "gwrf", "gam", "physics", "ols")
    s2 = {**{f"base_fit:{m}": 5 for m in models}, "stacker_fit:mean": 5, "stacker_fit:nnls": 5,
          "stacker_fit:residual": 15}
    assert by["S0"]["units"] == {"s0_load": 1} and by["S1"]["units"] == {"s1_influence": 1}
    assert by["S2_S3"]["units"] == {**s2, "checkpoint_save": 1}
    assert by["baselines"]["state"] == "skipped"
    assert by["cv_curve"]["units"] == {**{k: 3 * v for k, v in s2.items()}, "checkpoint_save": 1}
    assert by["S4"]["units"] == {"engine_pass": 44, "checkpoint_save": 1}                   # 1 + 19 doses + 24
    assert by["S5"]["units"] == {"engine_pass": 12, "checkpoint_save": 1}
    assert by["climate"]["units"] == {"checkpoint_save": 1}                                  # table: no downloads
    s6 = {f"causal_step:{s}": 3 for s in ("dml", "spillover", "cate", "dr", "sens", "audit")}
    assert by["S6"]["units"] == {"engine_pass": 24, **s6, "checkpoint_save": 1}
    assert by["S7"]["units"] == {"engine_pass": 1, "pareto": 1}
    assert by["finish"]["units"] == {}
    tot = plan_totals(nodes)
    assert tot["engine_pass"] == 44 + 12 + 24 + 1 and tot["base_fit:mgwr"] == 20
    # seed-rate seconds of the reference table (checkpoint_save at 2 s each)
    rate = {"base_fit:mgwr": 170, "base_fit:gwrf": 23, "base_fit:gam": 6, "base_fit:physics": 4, "base_fit:ols": 0.1,
            "stacker_fit:mean": 0.2, "stacker_fit:nnls": 0.2, "stacker_fit:residual": 14, "engine_pass": 13,
            "causal_step:dml": 10, "causal_step:spillover": 15, "causal_step:cate": 1, "causal_step:dr": 22,
            "causal_step:sens": 0.5, "causal_step:audit": 0.5, "s0_load": 0.3, "s1_influence": 3.3,
            "checkpoint_save": 2, "pareto": 0.5}
    est = sum(rate[k] * v for k, v in tot.items())
    assert est == pytest.approx(6116, rel=0.01)                        # SPEC §5.4: ≈6,116 s vs 6,097 recorded

    cmip = copy.deepcopy(cfg)
    cmip.raw["climate"]["source"] = "cmip6"
    by = {n["id"]: n for n in plan_stages(cmip, ALL, n_points=54701, extent=8600.0, dx=30.0, block_m=2000.0)}
    assert by["climate"]["units"]["climate_model"] == 24                # until the catalogue is read


@pytest.mark.slow
def test_full_gating_matrix(demo):
    subsets = [("S0", "S1"), S0_S3, ("S5",), ("S6",), ("S7",), ALL]
    toggles = [dict(), dict(climate=False, causal=False, optimize=False, baselines=False)]
    for stages, fast, coarse, cv_curve, tg in itertools.product(subsets, (False, True), (None, 60.0),
                                                               (None, True), toggles):
        cfg = _toggle(_light(demo), **tg)
        _assert_plan_matches_run(cfg, stages, fast=fast, coarse=coarse, cv_curve=cv_curve)
