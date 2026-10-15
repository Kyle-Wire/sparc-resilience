"""The packaged results page (``sparc.core.results_page``, SPEC §6.9).

Built on the committed synthetic fixture run (``tests/studio/fixtures/synth_run``, recorded on the demo city of
``write_demo_project(n=40, seed=0)``): the HTML shell, the config-driven catalogue (no Providence names), folds
rebuilt without the checkpoint, one buffer mask per fold, the stage-file fallbacks and the computed Pareto
caption.  The Playwright check (``scripts/results_page/check_page.py``) is marked slow.
"""

from __future__ import annotations

import base64
import copy
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SYNTH = REPO / "tests" / "studio" / "fixtures" / "synth_run"
CHROMIUM = os.environ.get("SPARC_CHROMIUM", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome")


@pytest.fixture(autouse=True)
def _single_thread(monkeypatch):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(var, "1")


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    """``(config path, CoreConfig)`` of the demo city the fixture run was recorded on."""
    if not (SYNTH / "manifest.json").is_file():
        pytest.skip("tests/studio/fixtures/synth_run is not present")
    from sparc.core.config import load_core_config
    from sparc.core.synthetic import write_demo_project

    out = tmp_path_factory.mktemp("demo")
    info = write_demo_project(out, n=40, seed=0)
    return Path(info["config_path"]), load_core_config(info["config_path"])


@pytest.fixture(scope="module")
def built(demo, tmp_path_factory):
    """``(page path, collected blob)`` of the fixture run."""
    from sparc.core.results_page import build_results_page, collect

    cfg_path, cfg = demo
    out = tmp_path_factory.mktemp("page") / "results.html"
    path = build_results_page(SYNTH, cfg_path, out)
    return path, collect(SYNTH, cfg)


def _data_blob(html: str) -> dict:
    m = re.search(r"const DATA = (\{.*?\});\n</script>", html, re.S)
    assert m, "no DATA payload"
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_page_has_an_html_shell_and_no_placeholders(built):
    path, _ = built
    html = path.read_text("utf-8")
    assert html.startswith("<!doctype html>\n<html lang=\"en\">")
    head = html[:600]
    assert '<meta charset="utf-8">' in head and 'name="viewport"' in head
    assert "<title>Synthetic city (DEMO)</title>" in head
    assert html.rstrip().endswith("</body>\n</html>")
    assert "{{" not in html and "/*__DATA__*/" not in html
    # the fixture's grid (30 m cells, 3 folds of 0.39 km blocks) reaches the prose
    assert "Each pixel is one 30 m cell" in html
    assert "jackknife over the three fold models" in html and "Models on the main 0.39 km blocks" in html


def test_catalog_is_generated_from_the_config(built):
    path, blob = built
    data = _data_blob(path.read_text("utf-8"))
    keys = [c["key"] for c in data["catalog"]]
    assert not [k for k in keys if "Pct_" in k or "Albedo" in k or k.startswith("hd_")]
    assert {"in_canopy", "in_impervious", "fp_canopy", "own_canopy", "fp_impervious", "fp_albedo",
            "cls_impervious", "alloc_dose", "__folds"} <= set(keys)
    assert all(k.startswith("__") or k in data["layers"] for k in keys)
    by_key = {c["key"]: c for c in data["catalog"]}
    assert by_key["fp_albedo"]["name"] == "Albedo: footprint per +0.01" and by_key["fp_albedo"]["mult"] == 0.01
    assert by_key["fp_canopy"]["name"] == "Canopy: footprint per +1 pp"
    assert by_key["cls_impervious"]["name"].startswith("Impervious")
    assert "three fold models" in by_key["sc_0"]["desc"]
    assert data["var_names"]["canopy"] == "Tree canopy" and data["unit_label"] == "°F"
    assert "Pct_Canopy" not in json.dumps({k: data[k] for k in ("var_names", "levers", "roles", "catalog")})


def test_folds_are_rebuilt_without_the_checkpoint(demo, monkeypatch):
    """No ``checkpoint.pkl`` in the fixture, and unpickling is refused: the folds come from ``load_run``."""
    import pickle

    from sparc.core.baselines import load_run
    from sparc.core.results_page import collect

    assert not (SYNTH / "checkpoint.pkl").exists()
    monkeypatch.setattr(pickle, "load", lambda *a, **k: pytest.fail("the results page unpickled something"))
    _, cfg = demo
    blob = collect(SYNTH, cfg)
    _data, folds, _m, pred = load_run(SYNTH, copy.deepcopy(cfg))
    fold = np.frombuffer(base64.b64decode(blob["fold"]), np.uint8)
    np.testing.assert_array_equal(fold, np.where(folds.fold_id >= 0, folds.fold_id, 255))
    np.testing.assert_array_equal(fold, pred["fold"].to_numpy())
    assert blob["n_folds"] == folds.n_folds == len(blob["excl"])
    for k, b64 in enumerate(blob["excl"]):
        mask = np.frombuffer(base64.b64decode(b64), np.uint8)
        np.testing.assert_array_equal(mask, (~folds.train_masks[k] & ~folds.test_masks[k]).astype(np.uint8))


def test_more_than_eight_folds(demo, monkeypatch):
    """One u8 mask per fold: ten folds no longer overflow an 8-bit field."""
    import sparc.core.baselines as bl
    from sparc.core.cv import make_spatial_folds
    from sparc.core.results_page import collect

    real = bl.load_run

    def ten_folds(run_dir, cfg):
        data, _folds, m, pred = real(run_dir, cfg)
        return data, make_spatial_folds(data.coords, n_folds=10, block_m=150.0, buffer_m=50.0, seed=1), m, pred

    monkeypatch.setattr(bl, "load_run", ten_folds)
    _, cfg = demo
    blob = collect(SYNTH, cfg)
    assert blob["n_folds"] == 10 and len(blob["excl"]) == 10
    data, folds, _m, _p = ten_folds(SYNTH, copy.deepcopy(cfg))
    for k in (8, 9):
        mask = np.frombuffer(base64.b64decode(blob["excl"][k]), np.uint8)
        np.testing.assert_array_equal(mask, (~folds.train_masks[k] & ~folds.test_masks[k]).astype(np.uint8))
    assert "ten fold models" in next(c["desc"] for c in blob["catalog"] if c["key"] == "sc_0")


def test_stage_files_fill_a_manifest_without_sections(demo, tmp_path):
    """causal, cv_distance and optimize fall back to their stage files."""
    from sparc.core.report import _causal_summary
    from sparc.core.results_page import collect

    run = tmp_path / "run"
    shutil.copytree(SYNTH, run, ignore=shutil.ignore_patterns("events.jsonl"))
    m = json.loads((run / "manifest.json").read_text())
    for k in ("causal", "cv_distance", "optimize"):
        m.pop(k, None)
    (run / "manifest.json").write_text(json.dumps(m))
    (run / "optimize.json").unlink()
    opt = {"variable": "canopy", "budget": 100.0, "pareto": {"points": [
        {"budget": 50.0, "total_benefit": 6.0}, {"budget": 100.0, "total_benefit": 10.0},
        {"budget": 200.0, "total_benefit": 19.0}]}}
    (run / "optimize.json").write_text(json.dumps(opt))
    cvd = {"rows": [{"label": "0.39 km blocks", "main": True, "block_m": 390.0, "buffer_m": 130.0,
                     "fold_r2_min": 0.4, "fold_r2_max": 0.7}]}
    (run / "cv_distance.json").write_text(json.dumps(cvd))
    _, cfg = demo
    blob = collect(run, cfg)
    assert blob["causal"] == json.loads(json.dumps(_causal_summary(json.loads((run / "causal.json").read_text()))))
    assert blob["cv_distance"] == cvd
    assert blob["optimize"]["pareto"] == opt["pareto"]
    assert "raises total cooling 1.90×" in blob["pareto_caption"]


def test_pareto_caption_is_written_from_the_numbers():
    from sparc.core.results_page import pareto_caption

    def opt(*pts):
        return {"budget": 100.0, "pareto": {"points": [{"budget": b, "total_benefit": v} for b, v in pts]}}

    assert pareto_caption(None) is None and pareto_caption({"budget": 1}) is None
    flat = pareto_caption(opt((100, 10.0), (200, 19.5)))
    assert "raises total cooling 1.95×" in flat and "barely diminish" in flat
    mid = pareto_caption(opt((100, 10.0), (200, 17.4)))
    assert "1.74×" in mid and "about 74%" in mid and "only slightly" not in mid
    steep = pareto_caption(opt((100, 10.0), (200, 12.0)))
    assert "diminish sharply" in steep and "about 20%" in steep
    none = pareto_caption(opt((100, 10.0), (200, 10.0)))
    assert "buys no further cooling" in none
    other = pareto_caption(opt((25, 4.0), (50, 7.0)), "°C")
    assert other.startswith("Total planned cooling (°C summed over cells)") and "from 0.25× to 0.5×" in other


def test_layer_catalog_lists_every_hot_days_case(demo):
    from sparc.core.results_page import layer_catalog

    _, cfg = demo
    layers = {k: {} for k in ("obs", "people", "hd_90_today", "hd_90_ssp245_mid", "hd_95_today", "hd_95_ssp245_mid",
                              "hd_95_ssp585_late")}
    cat = layer_catalog(cfg, layers, units="°F", background=80.0, n_folds=5)
    hd = [(c["key"], c["name"]) for c in cat if c["key"].startswith("hd_")]
    assert [k for k, _ in hd] == ["hd_90_today", "hd_90_ssp245_mid", "hd_95_today", "hd_95_ssp245_mid",
                                  "hd_95_ssp585_late"]
    assert dict(hd)["hd_95_ssp245_mid"] == "Hot afternoons ≥ 95 °F (SSP2-4.5, 2041–2060)"
    assert dict(hd)["hd_95_ssp585_late"] == "Hot afternoons ≥ 95 °F (SSP5-8.5, 2081–2100)"


def test_catalog_follows_renamed_roles(demo):
    """Another city's column names: the keys and labels follow the config's roles and levers."""
    from sparc.core.results_page import layer_catalog, lever_info, variable_names

    _, cfg = demo
    cfg = copy.deepcopy(cfg)
    cfg.raw["physics"]["roles"] = {"canopy": "tree_pct", "impervious": "paved"}
    cfg.raw["actionable"] = {"tree_pct": {"min": 0, "max": 100, "unit": "pp"},
                             "paved": {"min": 0, "max": 100, "unit": "pp", "direction": "decrease"},
                             "roof_refl": {"min": 0.05, "max": 0.9}}
    levers = lever_info(cfg)
    assert levers["roof_refl"]["albedo_like"] and not levers["tree_pct"]["albedo_like"]
    names = variable_names(cfg)
    assert names["tree_pct"] == "Tree canopy" and names["paved"] == "Impervious cover"
    layers = {k: {} for k in ("in_tree_pct", "in_paved", "fp_tree_pct", "own_tree_pct", "fp_paved", "fp_roof_refl",
                              "cls_paved", "alloc_dose")}
    cat = {c["key"]: c for c in layer_catalog(cfg, layers, units="°C", background=30.0, block_m=1500, n_folds=4,
                                             optimize_variable="tree_pct")}
    assert cat["in_tree_pct"]["name"] == "Tree canopy" and cat["fp_roof_refl"].get("mult") == 0.01
    assert cat["cls_paved"]["name"] == "Impervious removal: response shape"
    assert cat["alloc_dose"]["name"] == "Planned canopy increase"
    assert "budget optimiser ranks" in cat["fp_tree_pct"]["desc"]
    assert "1.5 km block" in next(c for c in layer_catalog(cfg, {"pred": {}}, units="°C", background=30.0,
                                                            block_m=1500) if c["key"] == "pred")["desc"]


def test_corners_and_page_lat_lon_know_the_hemisphere(tmp_path):
    """Corners from a southern-hemisphere CRS are negative latitudes, and the page writes °S / °E."""
    from types import SimpleNamespace

    from sparc.core.results_page import TEMPLATE, _corners

    g = SimpleNamespace(x0=334000.0, y0=6250000.0, nx=10, ny=10, dx=30.0, dy=30.0)
    cfg = SimpleNamespace(data={"crs": "EPSG:32756"}, coord_scale=1.0)     # UTM 56S (Sydney)
    c = _corners(cfg, g)
    assert c["sw"][0] < -30 and c["sw"][1] > 150
    # data reprojected at load (data.reproject_to): the run frame is that CRS in metres, not data.crs scaled
    from pyproj import Transformer

    rep = SimpleNamespace(data={"crs": "EPSG:4326", "reproject_to": "EPSG:32756"}, coord_scale=1.0)
    lon, lat = Transformer.from_crs("EPSG:32756", "EPSG:4326", always_xy=True).transform(334000.0, 6250000.0)
    assert _corners(rep, g)["sw"] == pytest.approx([lat, lon])
    if shutil.which("node") is None:
        pytest.skip("node is not available")
    src = TEMPLATE.read_text("utf-8")
    line = re.search(r"^const latlonText = .*$", src, re.M).group(0)
    js = line + "\nprocess.stdout.write(JSON.stringify([latlonText(-33.86, 151.21), latlonText(41.82, -71.41, 3, ' ')]));"
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30, check=True).stdout)
    assert out == ["33.8600°S, 151.2100°E", "41.820°N 71.410°W"]


def test_build_page_script_is_a_shim(demo, tmp_path):
    """``scripts/results_page/build_page.py`` builds the same page through the packaged builder."""
    cfg_path, _ = demo
    out = tmp_path / "p.html"
    res = subprocess.run([sys.executable, str(REPO / "scripts" / "results_page" / "build_page.py"), str(SYNTH),
                          str(cfg_path), "--out", str(out)], capture_output=True, text=True, timeout=300,
                         cwd=str(tmp_path))
    assert res.returncode == 0, res.stderr
    assert out.read_text("utf-8").startswith("<!doctype html>")
    assert not (REPO / "scripts" / "results_page" / "template.html").exists()
    assert len((REPO / "scripts" / "results_page" / "build_page.py").read_text().splitlines()) <= 15


@pytest.mark.slow
def test_page_passes_check_page(built, demo):
    """Every theme and layer opens in Chromium without a JavaScript error (``check_page.py``)."""
    pytest.importorskip("playwright")
    if not Path(CHROMIUM).exists():
        pytest.skip(f"no Chromium at {CHROMIUM}")
    path, _ = built
    cfg_path, _ = demo
    res = subprocess.run([sys.executable, str(REPO / "scripts" / "results_page" / "check_page.py"), str(path),
                          str(SYNTH), str(cfg_path), "--chromium", CHROMIUM],
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "page OK" in res.stdout
    n = int(re.search(r"(\d+) map layers opened", res.stdout).group(1))
    assert n >= 30


# ---------------------------------------------------------------------------
# W3: "Can the data pin down the physics?" (identifiability card)
# ---------------------------------------------------------------------------

def _study_for(run: Path, *, L_mean=None) -> dict:
    """A hand-made identifiability report that belongs to ``run`` (its cells, CV design and the physics fold
    means), profiling L (identified) and s (on its lower bound, prior-dominated)."""
    m = json.loads((run / "manifest.json").read_text())
    folds = [p for p in json.loads((run / "physics.json").read_text()) if p["a"] > 0 and not p.get("fit_warning")]

    def mean(k):
        return float(np.mean([p[k] for p in folds]))

    def row(key, label, unit, cls, ci, run_v, data_v, share, at_bound=None):
        return {"key": key, "label": label, "unit": unit, "estimate_run": run_v, "estimate_data": data_v,
                "at_bound": at_bound, "held": "bound" if at_bound else None, "se_robust": 0.1, "adjust_factor": 0.01,
                "ci95": ci, "ci95_naive": [ci[0], ci[0] + 0.1 * (ci[1] - ci[0])], "identifiability": cls,
                "open_sides": [], "sides_at_limit": [], "pull": 0.3, "prior": None, "prior_share_run": share,
                "posterior": {"method": "laplace", "mean": run_v, "sd": 0.1},
                "psens": {"prior": 0.1, "likelihood": 0.12, "diagnosis": "prior_data_conflict", "method": "profile"},
                "conflict": {"z": None, "p": None, "flag": None, "reason": "on a fit bound"},
                "profile": {"grid": [ci[0], 0.5 * (ci[0] + ci[1]), ci[1] * 1.2],
                            "lambda_adj": [0.0, 1.0, 6.0], "lambda_naive": [0.0, 40.0, 300.0]}, "note": None}

    L = mean("L_m")
    params = [row("L", "Mixing length L", "m", "identified", [0.7 * L, 1.5 * L], L, 1.05 * L, 0.05),
              row("s", "Canopy shading strength s", "fraction of absorbed sunlight", "identified", [0.0, 0.49],
                  mean("s"), 0.0, 0.93, {"side": "lower", "bound": 0.0})]
    a = mean("a")
    lin = [{"key": "a", "label": "Source gain a", "unit": "target units per unit q", "estimate_data": a,
            "se_naive_data": 0.5, "se_robust_data": 5.0, "estimate_run": a, "se_naive_run": 0.5, "se_robust_run": 5.0,
            "adjust_factor": 0.01}]
    return {"schema": 1, "status": "ok", "n": m["n_points"], "n_clusters": m["cv"]["n_blocks"], "coarse_m": None,
            "units": m["config"]["data"]["target_units"], "window": "day",
            "cluster": {"kind": "cv_block", "block_m": m["cv"]["block_m"]},
            "settings": {"chi2_95": 3.841458820694124, "psens_threshold": 0.05}, "posterior": {"laplace_adequate": False},
            "warnings": [], "parameters": params, "linear": lin,
            "run": {"n_points_run": m["n_points"], "cv": m["cv"],
                    "fold_means": {"L": L if L_mean is None else L_mean, "s": mean("s")}}}


@pytest.fixture(scope="module")
def studied(demo, tmp_path_factory):
    """``(run dir, page html, blob)`` of a copy of the fixture run with an identifiability study."""
    from sparc.core.results_page import collect, render_page

    _, cfg = demo
    run = tmp_path_factory.mktemp("ident") / "synth_run"
    shutil.copytree(SYNTH, run)
    (run / "identifiability.json").write_text(json.dumps(_study_for(run)))
    blob = collect(run, cfg)
    return run, render_page(blob, cfg), blob


def test_identifiability_card_is_hidden_without_a_study(built):
    path, blob = built
    html = path.read_text("utf-8")
    assert blob["identifiability"] is None and _data_blob(html)["identifiability"] is None
    assert '<div class="card" id="identCard" hidden>' in html
    assert blob["mechanisms"]["identifiability"]["status"] == "absent"


def test_identifiability_card_shows_the_study_and_its_profiles(studied):
    run, html, blob = studied
    card = blob["identifiability"]
    assert card["status"] == "used" and card["n_clusters"] == 16 and card["block_m"] == 390.0 and card["laplace_adequate"] is False
    assert [r["key"] for r in card["rows"]] == ["L", "s"]
    s = card["rows"][1]
    assert s["class"] == "identified" and s["reaches_zero"] and s["prior_share_run"] == 0.93
    assert s["profile"]["lambda_adj"] == [0.0, 1.0, 6.0] and s["ci95"] == [0.0, 0.49]
    assert "set mostly by the penalty (prior share 0.93)" in s["text"]
    assert card["sandwich"] and card["sandwich"][0]["label"] == "Aerodynamic resistance r_a"
    assert card["sandwich"][0]["jackknife_se"] != card["sandwich"][0]["sandwich_se"]
    # the register on the page was recomputed with the study (the fixture's manifest has none)
    assert blob["mechanisms"]["identifiability"]["status"] == "used"
    assert next(r for r in blob["mechanisms"]["panel"]["rows"] if r["key"] == "s")["identifiability"]["ci95"] == [0.0, 0.49]
    data = _data_blob(html)
    assert data["identifiability"]["rows"][0]["profile"]["grid"] == card["rows"][0]["profile"]["grid"]
    assert "Can the data pin down the physics?" in html and "function identifiabilityCard()" in html
    assert "identifiabilityCard();" in html and 'id="identProfiles"' in html


def test_identifiability_card_with_a_study_of_another_fit_says_so(demo, tmp_path):
    from sparc.core.results_page import collect

    _, cfg = demo
    run = tmp_path / "synth_run"
    shutil.copytree(SYNTH, run)
    (run / "identifiability.json").write_text(json.dumps(_study_for(run, L_mean=12345.0)))
    card = collect(run, cfg)["identifiability"]
    assert card["status"] == "ignored" and "different physics fit" in card["note"] and "rows" not in card


@pytest.mark.slow
def test_identifiability_card_renders_in_chromium(studied, tmp_path):
    """The card's script runs without a console error and fills the table and one profile chart per parameter
    (headless Chromium's ``--dump-dom``; no Playwright needed)."""
    if not Path(CHROMIUM).exists():
        pytest.skip(f"no Chromium at {CHROMIUM}")
    _, html, _ = studied
    page = tmp_path / "page.html"
    page.write_text(html, encoding="utf-8")
    res = subprocess.run([CHROMIUM, "--headless=new", "--no-sandbox", "--disable-gpu", "--enable-logging=stderr",
                          "--v=0", "--virtual-time-budget=5000", "--dump-dom", page.as_uri()],
                         capture_output=True, text=True, timeout=180)
    assert res.returncode == 0, res.stderr[-2000:]
    assert "CONSOLE" not in res.stderr and "Uncaught" not in res.stderr, res.stderr[-2000:]
    dom = res.stdout
    card = dom[dom.index('id="identCard"') - 30:]
    card = card[:card.index('id="identWarn"') + 400]
    assert 'id="identCard" hidden' not in dom
    assert card.count("<figure>") == 2 and card.count('role="img"') == 2
    assert "0 not excluded" in card and "identified" in card and "potential prior–data conflict" in card
    assert "Laplace (Gaussian) approximation" in card
