"""Result statistics (SPEC §7.7–7.8), result storage and the results / compare endpoints."""

from __future__ import annotations

import numpy as np
import pytest

from sparc.studio.engine import stats as S
from sparc.studio.runs.common import likely
from tests.studio.engine.conftest import make_result


def _core_result(run_ctx, name="Canopy Increase +10"):
    from sparc.core.scenarios import ScenarioResult

    det = run_ctx.scenario_detail()
    folds = np.asarray(det["folds"][name], dtype=np.float64)
    delta = folds.mean(axis=0)
    n = delta.size
    return ScenarioResult(name=name, baseline=np.zeros(n), scenario=delta, delta=delta,
                          delta_sd=np.asarray(det["sd"][name], dtype=np.float64),
                          extrapolation=np.asarray(det["ex"][name], dtype=np.float64),
                          realized={"canopy": np.full(n, 10.0)}, delta_folds=folds)


def test_region_all_equals_core_summary(run_ctx):
    res = _core_result(run_ctx)
    core = res.summary()
    out = S.build_result(result_id="res_x", kind="exact", run_id=run_ctx.run_id, created_utc="t", job_id=None,
                         delta=res.delta, delta_sd=res.delta_sd, extrapolation=res.extrapolation, folds=res.delta_folds,
                         realized=res.realized, grid=run_ctx.grid, unit="°F")
    reg = {r["name"]: r for r in out["regions"]}["all"]
    assert reg["mean"]["estimate"] == pytest.approx(core["mean_delta"], abs=1e-12)
    assert reg["mean"]["se"] == pytest.approx(core["mean_delta_se"], abs=1e-12)
    assert out["city"]["estimate"] == pytest.approx(core["mean_delta"], abs=1e-12)
    assert out["city"]["se"] == pytest.approx(core["mean_delta_se"], abs=1e-12)
    assert out["p10"] == pytest.approx(core["p10_delta"]) and out["p90"] == pytest.approx(core["p90_delta"])
    assert out["mean_delta_sd"] == pytest.approx(core["mean_delta_sd"])


def test_paired_se():
    rng = np.random.default_rng(1)
    K, n = 5, 400
    common = rng.normal(0, 1.0, (K, 1)) + rng.normal(0, 0.05, (K, n))
    a = -0.5 + common + rng.normal(0, 0.02, (K, n))
    b = -0.3 + common + rng.normal(0, 0.02, (K, n))
    assert S.paired_se(a, a) == pytest.approx(0.0, abs=1e-15)
    paired = S.paired_se(a, b)
    indep = np.hypot(S.masked_se(a), S.masked_se(b))
    assert 0 < paired <= indep
    assert paired < 0.2 * indep                      # correlated folds: much tighter
    lk = S.pair_likely(a.mean(0), a, S.masked_se(a), b.mean(0), b, S.masked_se(b), np.ones(n, bool), "°F")
    assert lk["paired"] is True and lk["se"] == pytest.approx(paired)
    lk = S.pair_likely(a.mean(0), None, S.masked_se(a), b.mean(0), b, S.masked_se(b), np.ones(n, bool), "°F")
    assert lk["paired"] is False and lk["se"] == pytest.approx(indep)


def test_pair_phrase_is_relative():
    """A − B of two cooling scenarios reads as one cooling less than the other, never "Warms A vs B …
    Confident it warms" (both items here cool the city); the numbers and confidence are those of A − B."""
    rng = np.random.default_rng(2)
    K, n = 5, 300
    common = rng.normal(0, 0.1, (K, n))
    common -= common.mean()
    jitter = rng.normal(0, 0.01, (K, n))
    jitter -= jitter.mean()

    def pair(ma, mb, where="the city"):
        a, b = ma + common, mb + common + jitter
        return S.pair_likely(a.mean(0), a, S.masked_se(a), b.mean(0), b, S.masked_se(b), np.ones(n, bool), "°F",
                             labels=("Shade", "Cooling package"), where=where)

    lk = pair(-0.34, -1.307)
    assert lk["estimate"] == pytest.approx(0.967, abs=0.01) and lk["confidence"] == "confident_warms"
    assert lk["phrase"].startswith("Shade cools the city 0.97 °F less than Cooling package (likely range ")
    assert lk["phrase"].endswith("Confident there is a difference.")
    assert "Warms" not in lk["phrase"] and "warms" not in lk["phrase"]
    assert pair(-1.307, -0.34)["phrase"].startswith("Shade cools the city 0.97 °F more than Cooling package")
    assert pair(0.5, 0.2)["phrase"].startswith("Shade warms the city 0.30 °F more than Cooling package")
    assert pair(-0.5, 0.3)["phrase"].startswith("Shade cools the city by 0.50 °F and Cooling package warms it by "
                                                "0.30 °F, a 0.80 °F difference")
    assert pair(-0.3, -0.3002)["phrase"].endswith("Could be no difference.")
    zero = np.zeros(n)
    b = -0.3 + common
    lk = S.pair_likely(zero, np.zeros_like(b), 0.0, b.mean(0), b, S.masked_se(b), np.ones(n, bool), "°F",
                       labels=("Baseline", "Shade"), where="the city")
    assert lk["phrase"].startswith("Baseline leaves the city unchanged and Shade cools it by 0.30 °F, a 0.30 °F "
                                   "difference")


def test_paired_se_on_configured_scenarios(run_ctx):
    det = run_ctx.scenario_detail()
    fa = np.asarray(det["folds"]["Canopy Increase +10"], dtype=np.float64)
    fb = np.asarray(det["folds"]["Canopy Increase +20"], dtype=np.float64)
    assert S.paired_se(fa, fb) <= np.hypot(S.masked_se(fa), S.masked_se(fb))


def test_spill_shares_and_rings(run_ctx):
    res = _core_result(run_ctx)
    n = res.delta.size
    edited = np.zeros(n, dtype=bool)
    edited[: n // 5] = True
    realized = {"canopy": np.where(edited, 10.0, 0.0)}
    out = S.build_result(result_id="r", kind="exact", run_id=run_ctx.run_id, created_utc="t", job_id=None,
                         delta=res.delta, delta_sd=res.delta_sd, extrapolation=res.extrapolation, folds=res.delta_folds,
                         realized=realized, grid=run_ctx.grid, unit="°F", lever_ranges={"canopy": 120.0})
    sp = out["spill"]
    total = float(np.sum(res.delta))
    assert sp["inside"] + sp["outside"] == pytest.approx(total)
    inside_share = sp["inside"] / (sp["inside"] + sp["outside"])
    assert inside_share + sp["outside_share"] == pytest.approx(1.0)
    assert sp["outside_share"] == pytest.approx(float(np.sum(res.delta[~edited])) / total)
    rings = sp["rings"]
    assert rings[0]["r_m"] == 0 and rings[0]["n"] == int(edited.sum())
    assert rings[0]["mean"] == pytest.approx(float(res.delta[edited].mean()))
    assert sum(r["n"] for r in rings) <= n and all(r["se"] is not None for r in rings)
    assert sp["lever_ranges"] == {"canopy": 120.0}
    names = [r["name"] for r in out["regions"]]
    assert names[:4] == ["all", "edited", "edited + ring", "outside"]
    reg = {r["name"]: r for r in out["regions"]}
    assert reg["edited"]["n_cells"] + reg["outside"]["n_cells"] == n
    assert reg["edited + ring"]["n_cells"] > reg["edited"]["n_cells"]
    assert out["extrapolated_edited"] == pytest.approx(float(np.mean(res.extrapolation[edited] > 1)))
    rz = out["realized"]["canopy"]
    assert rz["requested_mean"] == rz["realized_mean"] == 10.0 and rz["clipped_share"] == 0.0


def test_extrapolated_edited_is_null_when_not_computed(run_ctx):
    """A preview (no extrapolation scores) or a scenario that edits nothing has no extrapolated share: null,
    never 0 (a draft pack's brief printed "Edited cells outside observed conditions: 0%")."""
    res = _core_result(run_ctx)
    n = res.delta.size
    common = dict(result_id="r", kind="exact", run_id=run_ctx.run_id, created_utc="t", job_id=None, delta=res.delta,
                  delta_sd=None, folds=None, grid=run_ctx.grid, unit="°F")
    draft = S.build_result(extrapolation=None, realized=res.realized, draft=True, **common)
    assert draft["summary"]["frac_extrapolated_edited"] is None and draft["extrapolated_edited"] is None
    nothing = S.build_result(extrapolation=res.extrapolation, realized={"canopy": np.zeros(n)}, **common)
    assert nothing["extrapolated_edited"] is None
    inside = S.build_result(extrapolation=np.zeros(n), realized=res.realized, **common)
    assert inside["extrapolated_edited"] == 0.0           # computed and none outside: a real 0


@pytest.mark.parametrize("est,se", [(-0.5, 0.1), (-0.5, 0.3), (-0.05, 0.02), (0.3, 0.1), (0.3, 0.2), (-0.1, 0.0511),
                                    (-0.1, 0.0510)])
def test_confidence_iff_interval_below_zero(est, se):
    lk = likely(est, se, "°F")
    hi, lo = est + 1.96 * se, est - 1.96 * se
    assert (lk["confidence"] == "confident_cools") == (hi < 0)
    assert (lk["confidence"] == "confident_warms") == (lo > 0)
    assert (lk["confidence"] == "could_be_zero") == (lo <= 0 <= hi)
    card = S.plain_card(city=lk, edited=None, edited_all=True, unit="°F", frac_extrapolated_edited=0.23,
                        causal={"model_within": False})
    want = {"confident_cools": "Confident it cools", "confident_warms": "Confident it warms",
            "could_be_zero": "Could be zero"}[lk["confidence"]]
    assert card["confidence"] == want
    assert card["qualifiers"] == ["partly outside observed conditions (23% of edited cells)",
                                  "independent causal check disagrees"]


def test_plain_card_headline():
    city = likely(-0.021, 0.004, "°F")
    ed = likely(-0.62, 0.0918, "°F")
    card = S.plain_card(city=city, edited=ed, edited_all=False, unit="°F", frac_extrapolated_edited=0.1, causal=None,
                        draft=True)
    assert card["headline"].startswith("Cools the edited area by 0.62 °F (likely range 0.44–0.80 °F).")
    assert "City-wide: 0.021 °F cooler." in card["headline"]
    assert card["qualifiers"] == ["preview only — not verified"]


def test_result_storage_cache_and_stale(client, ctx, run_ctx, synth_run):
    from sparc.studio.engine import store

    rid, rd = synth_run
    row = make_result(ctx, run_ctx)
    got = client.get(f"/api/results/{row['id']}")
    assert got.status_code == 200, got.text
    res = got.json()
    assert res["summary"]["kind"] == "exact" and res["summary"]["has_folds"] and not res["stale"]
    assert res["plain"]["headline"] and res["regions"][0]["name"] == "all"
    assert res["impacts"] is not None and res["impacts"]["exposure"]          # the demo has people layers
    layer = client.get(f"/api/results/{row['id']}/layers/delta.bin")
    assert layer.status_code == 200 and layer.headers["X-SPARC-Length"] == str(run_ctx.grid.n)
    assert client.get(f"/api/results/{row['id']}/layers/delta.bin",
                      headers={"If-None-Match": layer.headers["ETag"]}).status_code == 304
    hit = store.find_cached(ctx.db, rid, "h", store.checkpoint_key(rd), store.code_sha())
    assert hit is not None and hit["id"] == row["id"]
    listing = client.get(f"/api/runs/{rid}/scenarios").json()
    assert [r["id"] for r in listing["results"]] == [row["id"]]
    conf = {c["slug"]: c for c in listing["configured"]}
    assert conf["cooling-package"]["has_folds"] is True and conf["cooling-package"]["layer_key"] == "sc:cooling-package"
    assert [e["mode"] for e in conf["cooling-package"]["doc"]["edits"]] == ["add", "add", "add"]
    # the core code (or checkpoint) changes: the result goes stale but stays readable
    ctx.db.execute("UPDATE results SET code_sha = 'older' WHERE id = ?", (row["id"],))
    res = client.get(f"/api/results/{row['id']}").json()
    assert res["stale"] is True and res["summary"]["stale"] is True
    assert store.find_cached(ctx.db, rid, "h", store.checkpoint_key(rd), store.code_sha()) is None
    assert client.delete(f"/api/results/{row['id']}").json() == {"ok": True}
    assert client.get(f"/api/results/{row['id']}").status_code == 404


def test_compare_endpoint_paired(client, ctx, run_ctx, synth_run):
    rid, _ = synth_run
    row = make_result(ctx, run_ctx)
    r = client.post(f"/api/runs/{rid}/compare", json={"items": [
        {"kind": "result", "id": row["id"]}, {"kind": "configured", "slug": "canopy-increase-plus-20"},
        {"kind": "baseline"}], "regions": []})
    assert r.status_code == 201, r.text
    cmp_ = r.json()
    pairs = {(p["a"], p["b"]): p for p in cmp_["pairs"]}
    det = run_ctx.scenario_detail()
    fa = np.asarray(det["folds"]["Canopy Increase +10"], dtype=np.float64)
    fb = np.asarray(det["folds"]["Canopy Increase +20"], dtype=np.float64)
    assert pairs[(0, 1)]["city"]["paired"] is True
    assert pairs[(0, 1)]["city"]["se"] == pytest.approx(S.paired_se(fa, fb), rel=1e-5)
    assert pairs[(0, 2)]["city"]["se"] == pytest.approx(S.masked_se(fa), rel=1e-5)     # vs baseline: A's own SE
    # both items cool the city: the pair is worded relative to each other, never as A warming it
    ph = pairs[(0, 1)]["city"]["phrase"]
    assert cmp_["items"][0]["city"]["estimate"] < 0 and cmp_["items"][1]["city"]["estimate"] < 0
    assert ph.startswith(f"{cmp_['items'][0]['label']} cools the city ") and " than " in ph
    assert "warms" not in ph.lower() and "Confident it" not in ph
    assert cmp_["needs_exact"] == []
    lay = client.get(f"/api/runs/{rid}/layers/{pairs[(0, 1)]['layer_key']}.bin")
    assert lay.status_code == 200
    assert client.get(f"/api/comparisons/{cmp_['id']}").json()["id"] == cmp_["id"]
    assert [c["id"] for c in client.get(f"/api/runs/{rid}/comparisons").json()] == [cmp_["id"]]
    assert client.delete(f"/api/comparisons/{cmp_['id']}").json() == {"ok": True}
    assert client.get(f"/api/runs/{rid}/comparisons").json() == []


def test_compare_unverified_plan_has_only_city_totals(client, ctx, run_ctx, synth_run):
    """An unverified plan's planned benefit is a footprint total on each treated cell (°F·cells), not a ΔT map:
    its city mean (Σ/n) and cost are compared, but no edited-area mean, regions, equity, exposure or difference
    map (it read "edited −2.62 °F" where core's own closed loop cools treated cells by 0.81 °F)."""
    from pathlib import Path

    rid, _ = synth_run
    plan = client.post(f"/api/runs/{rid}/plans", json={"params": {"lever": "canopy", "budget": 2000}, "name": "P",
                                                        "verify": False}).json()["plan"]
    row = make_result(ctx, run_ctx)
    region = client.post(f"/api/runs/{rid}/regions", json={"name": "Hot", "spec": {
        "kind": "top", "column": "pred:target", "frac": 0.3, "direction": "highest"}}).json()
    r = client.post(f"/api/runs/{rid}/compare", json={"items": [
        {"kind": "plan", "id": plan["id"]}, {"kind": "result", "id": row["id"]}, {"kind": "baseline"}],
        "regions": [region["id"]]})
    assert r.status_code == 201, r.text
    cmp_ = r.json()
    it = cmp_["items"][0]
    n = run_ctx.grid.n
    assert it["label"] == "Plan: P (planned)" and it["per_cell"] is False and it["edited"] is None
    assert it["city"]["estimate"] == pytest.approx(-plan["planned"]["planned_total"] / n, rel=1e-5)
    assert cmp_["items"][1]["per_cell"] is True and cmp_["items"][2]["per_cell"] is True
    pairs = {(p["a"], p["b"]): p for p in cmp_["pairs"]}
    for k in ((0, 1), (0, 2)):
        assert pairs[k]["layer_key"] is None and pairs[k]["regions"] == {}
        assert pairs[k]["city"]["estimate"] is not None
    assert pairs[(1, 2)]["layer_key"] == f"cmp:{cmp_['id']}:1__2" and pairs[(1, 2)]["regions"]
    cdir = Path(ctx.db.fetchone("SELECT dir FROM comparisons WHERE id = ?", (cmp_["id"],))["dir"])
    assert sorted(f.name for f in cdir.glob("diff_*.npy")) == ["diff_1__2.npy"]
    assert it["label"] not in cmp_["equity"]
    assert all(e["item"] != it["label"] for e in cmp_["exposure"])
    assert cmp_["cooling_per_cost"][it["label"]] == pytest.approx(
        plan["planned"]["planned_total"] / plan["planned"]["total_cost"], rel=1e-5)
    assert {"kind": "plan", "id": plan["id"]} in cmp_["needs_exact"]
    meta = {m["key"]: m for g in client.get(f"/api/runs/{rid}/layers").json()["groups"] for m in g["layers"]}
    pb = meta[f"plan:{plan['id']}:planned_benefit"]
    assert pb["unit"] == "°F·cells" and "footprint total" in pb["label"]


def test_reindex_rebuilds_lab_tables(client, ctx, run_ctx, synth_run):
    from sparc.studio.db import reindex
    from sparc.studio.scenarios import plans

    rid, _ = synth_run
    row = make_result(ctx, run_ctx)
    plans.create(ctx.db, run_ctx, {"lever": "canopy", "budget": 300.0}, "plan")
    client.post(f"/api/runs/{rid}/compare", json={"items": [{"kind": "result", "id": row["id"]}, {"kind": "baseline"}]})
    sc = client.post(f"/api/projects/{run_ctx.project_id}/scenarios",
                     json={"doc": {"name": "s", "edits": [{"lever": "canopy", "mode": "add", "amount": 3}]}}).json()
    before = {t: ctx.db.fetchval(f"SELECT COUNT(*) FROM {t}") for t in ("results", "plans", "comparisons", "scenarios")}
    summary = reindex(ctx.db, ctx.workspace)
    after = {t: ctx.db.fetchval(f"SELECT COUNT(*) FROM {t}") for t in ("results", "plans", "comparisons", "scenarios")}
    assert before == after == {"results": 1, "plans": 1, "comparisons": 1, "scenarios": 1}
    assert summary["engine.results"]["results"] == 1 and summary["scenarios"]["scenarios"] == 1
    assert client.get(f"/api/scenarios/{sc['id']}").json()["doc"]["name"] == "s"
