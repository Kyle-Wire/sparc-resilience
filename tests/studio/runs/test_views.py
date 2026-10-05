"""Run-hub ViewModels against the web client's contract (``studio-web/src/api/runs.ts`` shapes, filled in
``studio-web/src/pages/runhub/__fixtures__/views.json``), the overview's outputs grid and study chips, the
uncertainty sources, the caveats (parity with the results page's JavaScript) and the cell inspector's S-curve."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.studio.runs.conftest import REPO, RUN_ID

WEB_VIEWS = REPO / "studio-web" / "src" / "pages" / "runhub" / "__fixtures__" / "views.json"
TEMPLATE = REPO / "sparc" / "core" / "results_page" / "template.html"
VIEWS = ("overview", "data", "accuracy", "distance", "influence", "response", "scenarios", "climate", "causal",
         "budget", "planner", "uncertainty", "provenance")
#: maps keyed by data (levers, treatments, models, thresholds …): their values share one shape
DYNAMIC = {"levers", "treatments", "by_model", "hashes", "platform", "share", "values", "present", "launch", "weights",
           "nuisance_r2", "realized", "ranges_m", "directional"}


def _kind(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "num"
    return {str: "str", list: "list", dict: "dict"}[type(v)]


def shape_errors(ours, web, path: str, dynamic: bool = False) -> list[str]:
    """Where ``ours`` departs from the shape of ``web``: a missing key or a different JSON type (null matches
    anything: every section and most fields are nullable)."""
    ko, kw = _kind(ours), _kind(web)
    if "null" in (ko, kw):
        return []
    if ko != kw:
        return [f"{path}: {ko} where the client expects {kw}"]
    errs: list[str] = []
    if ko == "dict":
        if dynamic:
            a = next((v for v in ours.values() if v is not None), None)
            b = next((v for v in web.values() if v is not None), None)
            return shape_errors(a, b, f"{path}.*") if a is not None and b is not None else []
        for k, v in web.items():
            if k not in ours:
                errs.append(f"{path}.{k}: missing")
            else:
                errs += shape_errors(ours[k], v, f"{path}.{k}", k in DYNAMIC)
    elif ko == "list":
        a = next((v for v in ours if v is not None), None)
        b = next((v for v in web if v is not None), None)
        if a is not None and b is not None:
            errs += shape_errors(a, b, f"{path}[]")
    return errs


def get_view(client, rid: str, view: str) -> dict:
    r = client.get(f"/api/runs/{rid}/views/{view}")
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# the web client's shapes
# ---------------------------------------------------------------------------

def test_views_match_the_web_client_shapes(client, fixture_run):
    if not WEB_VIEWS.is_file():
        pytest.skip("studio-web view fixtures are not in this checkout")
    web = json.loads(WEB_VIEWS.read_text("utf-8"))
    rid, _ = fixture_run
    errors = []
    for view in VIEWS:
        vm = get_view(client, rid, view)
        assert set(vm) == set(web[view]), view
        assert list(vm["sections"]) == list(web[view]["sections"]), view          # api.md §6.1 keys, in order
        errors += shape_errors(vm["sections"], web[view]["sections"], view)
    assert errors == []


def test_section_contents_follow_the_client_rules(client, fixture_run, synth):
    """The rules the run hub relies on (impl report of frontend-run-hub): x-major obs/pred bins, run row indices,
    predictor layer keys, scenario slugs matching the ``sc:<slug>`` layers."""
    rid, _ = fixture_run
    pred = pd.read_parquet(synth / "predictions.parquet")
    acc = get_view(client, rid, "accuracy")["sections"]
    b = acc["obs_pred_bins"]
    counts = np.asarray(b["counts"])
    x_counts, _ = np.histogram(pred["pred"], bins=b["x_edges"])                 # x = predicted
    y_counts, _ = np.histogram(pred["target"], bins=b["y_edges"])               # y = observed
    np.testing.assert_array_equal(counts.sum(axis=1), x_counts)
    np.testing.assert_array_equal(counts.sum(axis=0), y_counts)
    assert set(acc["resid_hist"]) == {"edges", "counts"}

    layers = {m["key"] for g in client.get(f"/api/runs/{rid}/layers").json()["groups"] for m in g["layers"]}
    inf = get_view(client, rid, "influence")["sections"]
    assert inf["ranges"] and all(r["predictor"] in layers for r in inf["ranges"])
    rows = get_view(client, rid, "scenarios")["sections"]["rows"]
    assert all(f"sc:{r['slug']}" in layers for r in rows)
    for key in ("obs", "resid", "fold", "dist_train_m", "fp_canopy", "own_canopy", "cls_canopy", "d90_canopy",
                "A_canopy", "cate_canopy", "mslope_canopy", "mslope_own_canopy", "alloc_dose", "alloc_delta"):
        assert key in layers, key
    top = get_view(client, rid, "budget")["sections"]["top_cells"]
    alloc = pd.read_parquet(synth / "allocation.parquet")
    assert top and all(alloc["dose"].iloc[c["row"]] == pytest.approx(c["dose"]) for c in top)
    assert all(c["id"] == int(alloc["id"].iloc[c["row"]]) for c in top)
    causal = get_view(client, rid, "causal")["sections"]["treatments"]
    for t, tr in causal.items():
        assert {"label", "unit", "forest", "audit", "dr_curve", "model_pd_curve", "cate", "cate_layer", "sensitivity",
                "controls"} <= set(tr)
        assert tr["cate_layer"] in (None, f"cate_{t}")


def test_budget_top_cells_rank_by_cooling_not_by_row(client, demo, place_run):
    """"Top cells by cooling": among the treated cells, the 25 with the most closed-loop cooling, in that order;
    the many cells sharing one dose are not listed in row order, and without ``closed_loop_delta`` the larger
    doses come first."""
    rid, rd = RUN_ID, place_run(demo)
    alloc = pd.read_parquet(rd / "allocation.parquet")
    treated = alloc[alloc["dose"] > 0]
    assert treated["dose"].round(6).value_counts().max() > 25                    # many cells share one dose
    top = get_view(client, rid, "budget")["sections"]["top_cells"]
    assert [c["rank"] for c in top] == list(range(1, 26))
    best = (-treated["closed_loop_delta"]).sort_values(ascending=False, kind="stable")
    assert [c["row"] for c in top] == [int(i) for i in best.index[:25]]
    benefit = [c["benefit"] for c in top]
    assert benefit == sorted(benefit, reverse=True) and benefit[0] == pytest.approx(best.iloc[0])
    assert all(c["dose"] > 0 for c in top)
    # without the closed-loop column: by dose, ties by row
    rd2 = place_run(demo, "20260101-000000-nocl-0001",
                    edit=lambda d: pd.read_parquet(d / "allocation.parquet").drop(columns="closed_loop_delta")
                    .to_parquet(d / "allocation.parquet"))
    top2 = get_view(client, "20260101-000000-nocl-0001", "budget")["sections"]["top_cells"]
    assert rd2.is_dir() and all(c["benefit"] is None for c in top2)
    by_dose = treated["dose"].sort_values(ascending=False, kind="stable")
    assert [c["row"] for c in top2] == [int(i) for i in by_dose.index[:25]]


def test_response_literature_sends_the_extrapolated_share(client, fixture_run, synth):
    """The literature panel's SPARC rows carry the extrapolated share of the scenario they scale from
    (``manifest.literature.sparc.<quantity>.frac_extrapolated``)."""
    rid, _ = fixture_run
    lit = json.loads((synth / "manifest.json").read_text())["literature"]["sparc"]
    rows = get_view(client, rid, "response")["sections"]["literature"]["sparc"]
    assert {r["quantity"] for r in rows} == set(lit)
    for r in rows:
        assert r["scenario"] == lit[r["quantity"]]["scenario"]
        assert r["frac_extrapolated"] == pytest.approx(lit[r["quantity"]]["frac_extrapolated"])


def test_causal_view_reads_the_manifest_layout(client, demo, place_run):
    """Without causal.json the view is built from the manifest's per-treatment summary
    (``{<treatment>: {...}, _flags: [...]}``), not left empty."""
    rd = place_run(demo, drop=("causal.json",))
    m = json.loads((rd / "manifest.json").read_text())
    assert "treatments" not in m["causal"]
    sec = get_view(client, RUN_ID, "causal")["sections"]
    assert set(sec["treatments"]) == {t for t in m["causal"] if not t.startswith("_")}
    tr = next(iter(sec["treatments"].values()))
    assert any(f["id"] == "theta_sum" and f["est"] is not None for f in tr["forest"])


# ---------------------------------------------------------------------------
# overview: outputs grid, study chips, stage reasons
# ---------------------------------------------------------------------------

def test_overview_grid_chips_and_stage_reasons(client, ctx, demo, fixture_run):
    rid, _ = fixture_run
    vm = get_view(client, rid, "overview")
    s = vm["sections"]
    grid = {g["id"]: g for g in s["outputs_grid"]}
    assert grid["predictions"]["state"] == "present" and grid["predictions"]["action"] is None
    # cv_curve is disabled in the config: the remedy is a new run with it, from the prefilled Launch page
    cv = grid["cv_distance"]
    assert cv["state"] == "missing"
    assert cv["action"] == {"kind": "open", "label": "Re-run with the CV curve", "method": "GET",
                            "path": f"/p/{demo['id']}/launch?from={rid}"}
    # one row per post-run action that has not run, carrying its action; no rows for optional files
    planner = [g for g in s["outputs_grid"] if g["produced_by"] == "post:planner"]
    assert [g["id"] for g in planner] == ["planner"]
    assert planner[0]["action"]["path"] == f"/api/runs/{rid}/actions/planner"
    assert "events" not in grid and "checkpoint" not in grid and "results_page" not in grid
    timing = {t["stage"]: t for t in s["timings"]}
    assert timing["cv_curve"]["state"] == "skipped"
    assert timing["cv_curve"]["reason"] == "disabled_by_config:cv.distance_curve.enabled"
    chips = {c["kind"]: c for c in s["studies"]}
    assert list(chips) == ["baselines", "planner", "emulator", "uncertainty", "placebo", "multiverse", "simcheck",
                           "reproduce"]
    assert chips["baselines"]["state"] == "done" and "hgb" in chips["baselines"]["headline"]
    assert chips["placebo"] == {"kind": "placebo", "state": "not_run", "headline": None, "study_id": None}

    # a finished placebo study of the run, through the DB contract with the studies item
    ctx.db.insert("studies", {"id": "st_plac0001", "project_id": demo["id"], "kind": "placebo", "target_run_id": rid,
                              "out_dir": str(Path(demo["dir"]) / "studies" / "st_plac0001"), "status": "succeeded",
                              "summary_json": json.dumps({"n_placebos": 3, "n_pass_model": 2, "n_pass_causal": 3}),
                              "created_utc": "2026-10-01T22:00:00Z", "updated_utc": "2026-10-01T22:00:00Z"})
    chips = {c["kind"]: c for c in get_view(client, rid, "overview")["sections"]["studies"]}
    assert chips["placebo"]["state"] == "done" and chips["placebo"]["study_id"] == "st_plac0001"
    assert chips["placebo"]["headline"] == "model 2/3 and causal 3/3 placebos passed"
    board = client.get(f"/api/projects/{demo['id']}/status-board").json()
    cells = next(r for r in board["rows"] if r["run"]["id"] == rid)["cells"]
    assert cells["cv_curve"]["state"] == "disabled" and cells["placebo"]["study_id"] == "st_plac0001"


def test_partial_run_offers_resume(client, demo, place_run):
    def cancelled(rd):
        st = json.loads((rd / "run_state.json").read_text())
        st.update(status="cancelled", done=["S3"])
        (rd / "run_state.json").write_text(json.dumps(st))
        (rd / "manifest.json").unlink()
        for f in ("causal.json", "causal_cells.parquet"):
            (rd / f).unlink()

    place_run(demo, edit=cancelled)
    out = client.get(f"/api/runs/{RUN_ID}/outputs").json()
    causal = next(o for o in out["outputs"] if o["id"] == "causal")
    assert causal["state"] == "missing"
    assert causal["action"] == {"kind": "resume", "label": "Resume to compute S6", "method": "POST",
                                "path": f"/api/runs/{RUN_ID}/resume", "body": {}}
    tab = next(t for t in out["tabs"] if t["id"] == "causal")
    assert tab["availability"] in ("partial", "missing") and tab["missing"][0]["action"]["kind"] == "resume"


# ---------------------------------------------------------------------------
# uncertainty sources
# ---------------------------------------------------------------------------

def test_uncertainty_sources_come_from_the_studies_index(client, ctx, demo, place_run):
    studies = Path(demo["dir"]) / "studies"
    mv, sc, other = studies / "st_mv000001", studies / "st_sc000001", studies / "st_mv000002"

    def with_uncertainty(rd):
        (rd / "uncertainty.json").write_text(json.dumps({
            "scenarios": [{"scenario": "Canopy Increase +10", "estimate": -1.0, "estimation_95": [-1.5, -0.5],
                           "specification": [-1.2, -0.8], "envelope": [-1.6, -0.4]}],
            "climate": [], "sources": {"run": str(rd), "multiverse": str(mv), "simcheck": [str(sc)]}}))

    place_run(demo, edit=with_uncertainty)
    rid = RUN_ID
    for sid, kind, out_dir, status in (("st_mv000001", "multiverse", mv, "succeeded"),
                                       ("st_sc000001", "simcheck", sc, "succeeded"),
                                       ("st_mv000002", "multiverse", other, "running")):
        ctx.db.insert("studies", {"id": sid, "project_id": demo["id"], "kind": kind, "target_run_id": rid,
                                  "out_dir": str(out_dir), "status": status, "created_utc": "2026-10-01T22:00:00Z",
                                  "updated_utc": "2026-10-01T22:00:00Z"})
    ctx.db.insert("study_links", {"run_id": rid, "study_id": "st_mv000001", "attached": 1})
    ctx.db.insert("study_links", {"run_id": rid, "study_id": "st_sc000001", "attached": 0})
    vm = get_view(client, rid, "uncertainty")
    assert vm["availability"] == "ready"
    src = {s["study_id"]: s for s in vm["sections"]["sources"]}
    assert src["st_mv000001"] == {"kind": "multiverse", "label": "st_mv000001", "study_id": "st_mv000001",
                                  "attached": True, "state": "done"}
    assert src["st_sc000001"]["attached"] is False and src["st_sc000001"]["kind"] == "simcheck"
    assert src["st_mv000002"]["attached"] is False and src["st_mv000002"]["state"] == "running"
    row = vm["sections"]["rows"][0]
    assert row["id"] == "canopy-increase-plus-10" and {x["id"] for x in row["layers"]} == {
        "estimation_95", "specification", "envelope"}
    # RunSummary.studies lists the attached studies (same as the project endpoints)
    summ = next(r for r in client.get("/api/runs").json()["items"] if r["id"] == rid)
    assert summ["studies"] == ["st_mv000001"]
    assert next(r for r in client.get(f"/api/projects/{demo['id']}").json()["runs"] if r["id"] == rid) == summ


# ---------------------------------------------------------------------------
# caveats: the results page's logic
# ---------------------------------------------------------------------------

def _template_caveats(data: dict) -> list[str]:
    """Run the template's own ``caveats()`` (with its ``fmt`` / ``sfmt``) in node on ``data``."""
    src = TEMPLATE.read_text("utf-8")
    fmt = re.search(r"^const fmt = .*$", src, re.M).group(0)
    sfmt = re.search(r"^const sfmt = .*$", src, re.M).group(0)
    body = re.search(r"^function caveats\(\) \{.*?^\}$", src, re.M | re.S).group(0)
    js = "\n".join([
        f"const DATA = {json.dumps(data)};",
        'const U = DATA.units === "degF" ? "°F" : (DATA.units || "");',
        "const CL = DATA.climate;",
        "const out = [];",
        "const $ = () => ({ append: (x) => out.push(x), set textContent(v) {} });",
        "const el = (tag, attrs, text) => text;",
        fmt, sfmt, body, "caveats();", "process.stdout.write(JSON.stringify(out));"])
    res = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def _page_data(run_dir: Path, m: dict) -> dict:
    """The fields of ``scripts/results_page/build_page.collect`` that ``caveats()`` reads."""
    placebo = None
    if (run_dir / "placebo.json").exists():
        pz = json.loads((run_dir / "placebo.json").read_text("utf-8"))
        placebo = {k: pz.get(k) for k in ("rows", "n_pass_model", "n_pass_causal", "n_placebos", "coarse_m")}
    cfg = m.get("config") or {}
    return {"qa": m.get("qa"), "cv_distance": m.get("cv_distance"), "simcheck": m.get("simcheck"),
            "uncertainty": m.get("uncertainty"), "placebo": placebo, "scenarios": m.get("scenarios"),
            "physics": m.get("physics"), "forcing": (cfg.get("physics") or {}).get("forcing_info"),
            "climate": m.get("climate"), "caveats_extra": list((cfg.get("report") or {}).get("caveats") or []),
            "units": (cfg.get("data") or {}).get("target_units", "degF"), "timings": m.get("timings_s") or {},
            "name": m.get("name"), "created": m.get("created_utc")}


def _check_parity(run_dir: Path, m: dict) -> list[str]:
    from sparc.studio.runs.caveats import caveats_for

    ctx = SimpleNamespace(manifest=m, cfg_raw=m.get("config") or {}, run_dir=run_dir)
    ours = caveats_for(ctx)
    assert ours == _template_caveats(_page_data(run_dir, m))
    return ours


@pytest.fixture
def node():
    if shutil.which("node") is None or not TEMPLATE.is_file():
        pytest.skip("node or the results-page template is not available")


def test_caveats_match_the_results_page(node, synth, tmp_path):
    m = json.loads((synth / "manifest.json").read_text())
    ours = _check_parity(synth, m)
    assert any("Climate futures" in c for c in ours)
    # one simcheck generator, a null simulation the product reproduces, a placebo floor, a CV curve
    m2 = json.loads(json.dumps(m))
    m2["simcheck"] = {"bias_correction": {"share_range": [1.13, 1.13], "n_generators": 1}}
    m2["uncertainty"] = {"scenarios": [
        {"scenario": "Canopy Increase +5", "estimate": -0.09,
         "null_artifact": [{"product": "rf", "delta": -0.08, "distinguishable": False}]},
        {"scenario": "Canopy Increase +10", "estimate": -0.17,
         "null_artifact": [{"product": "rf", "delta": -0.18, "distinguishable": False},
                           {"product": "direct", "delta": -0.06, "distinguishable": False}]}]}
    m2["cv_distance"] = {"rows": [{"main": True, "fold_r2_min": 0.61, "fold_r2_max": 0.83}]}
    m2["config"].setdefault("report", {})["caveats"] = ["A project caveat."]
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "placebo.json").write_text(json.dumps({"n_placebos": 3, "n_pass_model": 1, "n_pass_causal": 2}))
    m2["placebo"] = {"n_placebos": 3, "n_pass_model": 1, "n_pass_causal": 2}
    ours = _check_parity(rd, m2)
    text = " ".join(ours)
    assert "about 1.13 times the truth (one effect generator so far)" in text
    assert '"Canopy Increase +10"' in text and "−0.18 °F" in text and "−0.06 °F" in text
    assert "in 2 of 3 tests" in text and ours[-1] == "A project caveat."


def test_caveat_without_simcheck_states_no_uncomputed_share(synth):
    """Without a simulation check the recovered share is not computed for the run, so the caveat names no share
    (no canned "about two-thirds") and no direction: on the demo the run recovers 80-91% of the truth, and other
    seeds overstate it."""
    from sparc.studio.runs.caveats import caveats_for

    m = json.loads((synth / "manifest.json").read_text())
    assert not m.get("simcheck")
    cav = caveats_for(SimpleNamespace(manifest=m, cfg_raw=m.get("config") or {}, run_dir=synth))
    eff = [c for c in cav if c.startswith("Effect sizes")]
    assert len(eff) == 1 and "not computed" in eff[0] and "too small or too large" in eff[0], cav
    assert not any("two-thirds" in c or "attenuated" in c for c in cav)


def test_caveats_match_the_results_page_on_providence(node, providence_runs):
    full = providence_runs / "providence_uhi"
    if not (full / "manifest.json").is_file():
        pytest.skip("no full Providence run")
    m = json.loads((full / "manifest.json").read_text())
    ours = _check_parity(full, m)
    assert len(ours) >= 5


# ---------------------------------------------------------------------------
# cell inspector curves
# ---------------------------------------------------------------------------

def test_cell_curves_rebuild_every_fitted_shape(client, fixture_run, synth):
    rid, _ = fixture_run
    resp = pd.read_parquet(synth / "response_impervious.parquet")

    def sig(z):
        return 1.0 / (1.0 + np.exp(-z))

    i = int(np.flatnonzero(resp["curve_model"].to_numpy() == "sigmoid")[0])
    row = resp.iloc[i]
    curve = client.get(f"/api/runs/{rid}/cells/{i}").json()["curves"]["impervious"]
    assert curve["model"] == "sigmoid" and curve["inflection"] == pytest.approx(row["inflection_dose"], rel=1e-6)
    d, b = np.asarray(curve["dose"]), np.asarray(curve["benefit"])
    assert b[0] == pytest.approx(0.0, abs=1e-12) and b[-1] <= row["max_cooling_A"] * (1 + 1e-6)
    # the rebuilt width reproduces the stored slope at today's dose (A·σ(−D0/w)/w) and the curve has that form
    d0, A = float(row["inflection_dose"]), float(row["max_cooling_A"])
    slopes = {w: A * sig(-d0 / w) / w for w in d0 * np.array([cw / cd for cd in np.linspace(0.15, 0.85, 8)
                                                                for cw in (0.04, 0.08, 0.15)])}
    w = min(slopes, key=lambda x: abs(slopes[x] - row["marginal_benefit_per_unit"]))
    assert slopes[w] == pytest.approx(row["marginal_benefit_per_unit"], rel=1e-3)
    s0 = sig(-d0 / w)
    np.testing.assert_allclose(b, A * (sig((d - d0) / w) - s0) / (1 - s0), rtol=1e-6)

    j = int(np.flatnonzero(resp["curve_model"].to_numpy() == "linear")[0])
    lin = client.get(f"/api/runs/{rid}/cells/{j}").json()["curves"]["impervious"]
    np.testing.assert_allclose(lin["benefit"], resp["marginal_benefit_per_unit"].iloc[j] * np.asarray(lin["dose"]),
                               rtol=1e-6)
    k = int(np.flatnonzero(resp["curve_model"].to_numpy() == "insufficient")[0])
    ins = client.get(f"/api/runs/{rid}/cells/{k}").json()["curves"]["impervious"]
    assert ins["model"] == "insufficient" and ins["benefit"] == [] and ins["dose"] == []
