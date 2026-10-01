"""Cost model and ETA (SPEC §5.4): seed-table calibration against the recorded Providence run, scaling,
history medians and ranges, online refinement, and the calibration feed."""

from __future__ import annotations

import pytest

from sparc.studio.jobs import eta, tracker
from sparc.studio.jobs.tailer import iter_lines, parse_line

# Recorded full Providence run (configs/core_providence.yml): manifest timings_s, seconds (SPEC §5.4).
PROVIDENCE_TIMINGS_S = {"S0": 0.26, "S1": 3.3, "S2_S3": 1289, "cv_curve": 3653, "S4": 535, "S5": 159, "S6": 444,
                        "S7": 13}
PROVIDENCE_TOTAL_S = 6096.6
N_CELLS, THREADS = 54_701, 4
MODELS = ("mgwr", "gwrf", "gam", "physics", "ols")
K = 5


def _s2_s3(with_checkpoint: bool = True) -> dict:
    units = {f"base_fit:{m}": K for m in MODELS}
    units.update({"stacker_fit:mean": K, "stacker_fit:nnls": K, "stacker_fit:residual": 3 * K})   # C = 2 + 3
    if with_checkpoint:
        units["checkpoint_save"] = 1
    return units


# Reference unit counts of the recorded run (SPEC §5.4): 5 folds, five base models, no advection, tune_lambda
# [0, 0.1, 1], CV curve with 3 partitions, 3 levers with 7/7/5 doses and 9/9/6 marginal passes, 12 scenarios,
# 3 treatments, table climate, S7 on.
REFERENCE_UNITS = {
    "S0": {"s0_load": 1},
    "S1": {"s1_influence": 1},
    "S2_S3": _s2_s3(),
    "cv_curve": {u: 3 * n for u, n in _s2_s3(with_checkpoint=False).items()},
    "S4": {"engine_pass": 1 + (7 + 7 + 5) + (9 + 9 + 6)},
    "S5": {"engine_pass": 12},
    "S6": {"engine_pass": 3 * 8, **{f"causal_step:{s}": 3 for s in ("dml", "spillover", "cate", "dr", "sens", "audit")}},
    "S7": {"engine_pass": 1, "pareto": 1},
}
SEED_SECONDS = {"S0": 0.3, "S1": 3.3, "S2_S3": 1229.6, "cv_curve": 3682.5, "S4": 572, "S5": 156, "S6": 459,
                "S7": 13.5}


def test_recorded_total_constant():
    assert sum(PROVIDENCE_TIMINGS_S.values()) == pytest.approx(PROVIDENCE_TOTAL_S, abs=0.1)


def test_reference_unit_counts():
    assert REFERENCE_UNITS["S4"]["engine_pass"] == 44
    assert sum(REFERENCE_UNITS["S2_S3"].values()) == 5 * 5 + 5 + 5 + 15 + 1


@pytest.mark.parametrize("stage", sorted(SEED_SECONDS))
def test_seed_table_matches_the_spec_per_stage(stage):
    est, lo, hi = eta.estimate_units(REFERENCE_UNITS[stage], N_CELLS, THREADS)
    assert est == pytest.approx(SEED_SECONDS[stage], abs=0.06)
    assert lo == pytest.approx(0.75 * est) and hi == pytest.approx(1.25 * est)


def test_seed_table_predicts_providence_within_20_percent():
    total = sum(eta.estimate_units(u, N_CELLS, THREADS)[0] for u in REFERENCE_UNITS.values())
    assert total == pytest.approx(6116.2, abs=0.5)                          # the spec's "≈6,116 s"
    assert abs(total - PROVIDENCE_TOTAL_S) / PROVIDENCE_TOTAL_S < 0.20
    nodes = [{"id": sid, "label": sid, "state": "will_run", "units": u} for sid, u in REFERENCE_UNITS.items()]
    est = eta.estimate_nodes(nodes, N_CELLS, THREADS)
    assert sum(n["est_s"] for n in est) == pytest.approx(total, abs=0.1)


def test_scaling():
    assert eta.scale("base_fit:mgwr", N_CELLS, 4) == pytest.approx(1.0)
    assert eta.scale("base_fit:mgwr", 2 * N_CELLS, 4) == pytest.approx(2 ** 1.3)
    assert eta.scale("base_fit:gwrf", 2 * N_CELLS, 4) == pytest.approx(2 ** 1.15)
    assert eta.scale("engine_pass", 2 * N_CELLS, 4) == pytest.approx(2.0)
    assert eta.scale("engine_pass", N_CELLS, 2) == pytest.approx(2 ** 0.7)
    assert eta.scale("climate_model", 10 * N_CELLS, 1) == 1.0
    assert eta.scale("replicate:physics", 10 * N_CELLS, 1) == 1.0


def test_weights_and_wildcards():
    assert eta.unit_weight("baseline_fit:idw") == 12.0
    assert eta.unit_weight("replicate:null") == 254.0
    assert eta.unit_weight("variant:no_physics") == 1200.0
    assert eta.unit_weight("something_new") == 1.0
    assert eta.unit_costs()["stacker_fit:residual"] == 14.0


def test_history_median_and_ranges():
    model = eta.CostModel({"engine_pass": [10, 11, 12, 13, 50], "adv_refit": [3.0]})
    assert model.rate("engine_pass") == 12
    lo, hi = model.rate_range("engine_pass")
    assert (lo, hi) == (11, 13)
    assert model.rate("adv_refit") == 3.0
    assert model.rate_range("adv_refit") == pytest.approx((2.25, 3.75))     # < 3 observations: ±25%
    assert model.rate("base_fit:mgwr") == 170.0                              # no history: seed
    model = eta.CostModel({"engine_pass": list(range(1, 100))})
    assert len(model.history["engine_pass"]) == 20                           # the last 20 observations


def test_history_from_db_is_normalised(tmp_path):
    from sparc.studio.db import Database

    db = Database(tmp_path / "s.sqlite")
    db.migrate()
    host = eta.host_id()
    # 2× the cells and 4 threads: a 26 s pass is a 13 s reference pass
    db.executemany("INSERT INTO unit_timings (host_id, unit, seconds, n_cells, threads, mode, job_id, ts) "
                   "VALUES (?,?,?,?,?,?,?,?)",
                   [(host, "engine_pass", 26.0, 2 * N_CELLS, 4, "full", "j_x", float(i)) for i in range(5)]
                   + [("otherhost", "engine_pass", 1.0, N_CELLS, 4, "full", "j_y", 0.0)])
    model = eta.CostModel.from_db(db, host)
    assert model.rate("engine_pass") == pytest.approx(13.0)
    db.close()


def _states(path):
    st = tracker.new_state()
    for cursor, raw in iter_lines(path):
        ev = parse_line(raw)
        tracker.reduce(st, ev, cursor)
        yield ev, st


def test_online_refinement_uses_in_job_means_after_fold_0(fixtures_dir):
    """On the selftest fixture: once fold 1 is done, the remaining base fits are priced at their in-job means."""
    path = fixtures_dir / "selftest_events.jsonl"
    for ev, st in _states(path):
        if ev["type"] == "task.end" and ev["name"] == "fold" and ev.get("k") == 1:
            assert st["unit_obs"]["base_fit:ols"] and st["unit_obs"]["base_fit:gam"]
            result = eta.projection_eta(st, eta.CostModel(), threads=1)
            mean_ols = sum(st["unit_obs"]["base_fit:ols"]) / len(st["unit_obs"]["base_fit:ols"])
            mean_gam = sum(st["unit_obs"]["base_fit:gam"]) / len(st["unit_obs"]["base_fit:gam"])
            rem = st["planned_units"]["base_fit:ols"] - st["done_units"]["base_fit:ols"]
            seed_part = sum(n * eta.CostModel().seconds(u, st["n_points"], 1)
                            for u, n in {"stacker_fit:mean": 3, "stacker_fit:nnls": 3, "checkpoint_save": 1}.items())
            assert result["stages"]["S2_S3"] == pytest.approx(rem * (mean_ols + mean_gam) + seed_part, abs=1e-3)  # ms rounding
            break
    else:
        pytest.fail("fold 1 never ended in the fixture")


def test_online_refinement_converges():
    """Seed rates far from this machine: the ETA error collapses after the first completed fold."""
    plan = [{"id": "S2_S3", "label": "", "state": "will_run", "units": {"base_fit:ols": 5}}]
    st = tracker.new_state()
    events = [{"type": "run.plan", "ts": 0.0, "path": ["run:r"], "nodes": plan, "total_units": {},
               "n_points": N_CELLS},
              {"type": "stage.start", "ts": 0.0, "path": ["run:r", "stage:S2_S3"], "span": "1:1", "stage": "S2_S3"}]
    for e in events:
        tracker.reduce(st, e, 0)
    actual_total = 5 * 2.0                                  # each fit really takes 2 s (seed: 0.1 s)
    before = eta.projection_eta(st, eta.CostModel(), threads=THREADS)["eta_s"]
    errors = [abs(before - actual_total) / actual_total]
    for k in range(1, 5):
        p = ["run:r", "stage:S2_S3", f"task:fold[{k}/5]"]
        tracker.reduce(st, {"type": "task.start", "ts": 2.0 * (k - 1), "path": p, "span": f"1:{k + 1}",
                            "name": "base_model", "unit": "base_fit:ols"}, k)
        tracker.reduce(st, {"type": "task.end", "ts": 2.0 * k, "path": p, "span": f"1:{k + 1}", "name": "base_model",
                            "unit": "base_fit:ols", "status": "ok", "elapsed_s": 2.0}, k)
        est = eta.projection_eta(st, eta.CostModel(), threads=THREADS)
        remaining = actual_total - 2.0 * k
        errors.append(abs(est["eta_s"] - remaining) / remaining)
    assert errors[0] > 0.9                                  # the prior was 20x too optimistic
    assert max(errors[1:]) < 1e-9                           # fold 0 done: exact from then on


def test_cv_curve_partitions_at_095_of_s2_s3():
    st = tracker.new_state()
    plan = [{"id": "S2_S3", "label": "", "state": "will_run", "units": {"base_fit:ols": 5, "checkpoint_save": 1}},
            {"id": "cv_curve", "label": "", "state": "will_run", "units": {"base_fit:ols": 15}}]
    for i, e in enumerate([
        {"type": "run.plan", "ts": 0.0, "path": ["run:r"], "nodes": plan, "total_units": {}},
        {"type": "stage.start", "ts": 0.0, "path": ["run:r", "stage:S2_S3"], "span": "1:2", "stage": "S2_S3"},
        {"type": "stage.end", "ts": 100.0, "path": ["run:r", "stage:S2_S3"], "span": "1:2", "stage": "S2_S3",
         "status": "ok", "elapsed_s": 100.0},
    ]):
        tracker.reduce(st, e, i)
    out = eta.projection_eta(st, eta.CostModel())
    assert out["stages"]["cv_curve"] == pytest.approx(3 * 0.95 * 100.0)
    assert out["eta_s"] == pytest.approx(285.0)
    assert out["eta_lo"] == pytest.approx(285.0 * 0.75)


def test_launch_estimates():
    assert eta.peak_ram_gb(54_701, 5) == pytest.approx(55e3 * 54_701 * 5 ** 0.5 / 1e9 + 0.8)
    assert eta.checkpoint_bytes(54_701) == pytest.approx(525e6, rel=0.01)


def test_calibration_feed_writes_unit_timings(client, ctx, wait_job):
    jid = client.post("/api/jobs", json={"kind": "test.events", "params": {"seconds": 0.5}}).json()["id"]
    assert wait_job(client, jid)["status"] == "succeeded"
    rows = ctx.db.fetchall("SELECT unit, seconds, n_cells, threads FROM unit_timings WHERE job_id = ?", (jid,))
    units = {r["unit"] for r in rows}
    assert {"base_fit:ols", "base_fit:gam", "engine_pass", "s0_load", "checkpoint_save"} <= units
    assert all(r["seconds"] > 0 and r["n_cells"] == 40 for r in rows)
    body = client.get("/api/timings", params={"unit": "base_fit"}).json()
    assert {r["unit"] for r in body["unit_rates"]} == {"base_fit:ols", "base_fit:gam"}
    assert all(r["n"] == 3 for r in body["unit_rates"])
