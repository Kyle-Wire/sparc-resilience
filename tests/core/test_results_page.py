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
    # the fixture's grid (30 m cells, 3 folds of 0.39 km blocks) reaches the prose, in plain words
    assert "Each square is one place, 30 m across" in html
    assert "from the three models trained on different test areas" in html and "Models on the main 0.39 km blocks" in html


def test_catalog_is_generated_from_the_config(built):
    path, blob = built
    data = _data_blob(path.read_text("utf-8"))
    keys = [c["key"] for c in data["catalog"]]
    assert not [k for k in keys if "Pct_" in k or "Albedo" in k or k.startswith("hd_")]
    assert {"in_canopy", "in_impervious", "fp_canopy", "own_canopy", "fp_impervious", "fp_albedo",
            "cls_impervious", "alloc_dose", "__folds"} <= set(keys)
    assert all(k.startswith("__") or k in data["layers"] for k in keys)
    by_key = {c["key"]: c for c in data["catalog"]}
    assert by_key["fp_albedo"]["name"] == "Reflectivity: total effect of +0.01, nearby included"
    assert by_key["fp_albedo"]["mult"] == 0.01 and by_key["marg_albedo"]["mult"] == 0.01
    assert by_key["fp_canopy"]["name"] == "Tree canopy: total effect of +1 pt, nearby included"
    assert by_key["own_canopy"]["name"] == "Tree canopy: effect of +1 pt on the spot"
    assert by_key["cls_impervious"]["name"] == "Paving removed: how the cooling builds up"
    assert by_key["d90_canopy"]["name"] == "Tree canopy added: amount for 90% of the cooling" and by_key["d90_canopy"]["unit"] == "pts"
    assert "the three models trained on different test areas" in by_key["sc_0"]["desc"]
    assert data["layers"]["cls_canopy"]["labels"] == ["Hit the limit, or too little room to tell", "Diminishing returns",
                                                      "Steady returns", "Slow start, then strong"]
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
    assert "the ten models trained on different test areas" in next(c["desc"] for c in blob["catalog"] if c["key"] == "sc_0")


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
    assert other.startswith("Total planned cooling (°C·places, added up over every place)") and "from 0.25× to 0.5×" in other
    assert "cells" not in flat + mid + steep + none + other


def test_layer_catalog_lists_every_hot_days_case(demo):
    from sparc.core.results_page import layer_catalog

    _, cfg = demo
    layers = {k: {} for k in ("obs", "people", "hd_90_today", "hd_90_ssp245_mid", "hd_95_today", "hd_95_ssp245_mid",
                              "hd_95_ssp585_late")}
    cat = layer_catalog(cfg, layers, units="°F", background=80.0, n_folds=5)
    hd = [(c["key"], c["name"]) for c in cat if c["key"].startswith("hd_")]
    assert [k for k, _ in hd] == ["hd_90_today", "hd_90_ssp245_mid", "hd_95_today", "hd_95_ssp245_mid",
                                  "hd_95_ssp585_late"]
    # emissions futures and periods in the glossary's words (sparc/core/glossary.json), as Studio says them
    assert dict(hd)["hd_90_today"] == "Hot afternoons ≥ 90 °F per summer, today"
    assert dict(hd)["hd_95_ssp245_mid"] == "Hot afternoons ≥ 95 °F per summer, moderate emissions, mid-century (2041–2060)"
    assert dict(hd)["hd_95_ssp585_late"] == "Hot afternoons ≥ 95 °F per summer, very high emissions, late century (2081–2100)"


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
    assert names["tree_pct"] == "Tree canopy" and names["paved"] == "Paving"
    layers = {k: {} for k in ("in_tree_pct", "in_paved", "fp_tree_pct", "own_tree_pct", "fp_paved", "fp_roof_refl",
                              "cls_paved", "alloc_dose")}
    cat = {c["key"]: c for c in layer_catalog(cfg, layers, units="°C", background=30.0, block_m=1500, n_folds=4,
                                             optimize_variable="tree_pct")}
    assert cat["in_tree_pct"]["name"] == "Tree canopy" and cat["fp_roof_refl"].get("mult") == 0.01
    assert cat["cls_paved"]["name"] == "Paving removed: how the cooling builds up"
    assert cat["alloc_dose"]["name"] == "Planned canopy increase"
    assert "budget plan ranks places by" in cat["fp_tree_pct"]["desc"]
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


def _node(js: str):
    if shutil.which("node") is None:
        pytest.skip("node is not available")
    return json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30, check=True).stdout)


def test_chart_ticks_and_dates_are_round():
    """Chart axes step by 1, 2, 2.5 or 5 × 10^k (the budget's 0/250/…/1,000, not max/4), and dates read "1 Oct 2026"."""
    from sparc.core.results_page import TEMPLATE

    src = TEMPLATE.read_text("utf-8")
    nice = re.search(r"^function niceTicks\(.*?^\}$", src, re.M | re.S).group(0)
    dates = "\n".join(re.search(rf"^const {n} = .*$", src, re.M).group(0) for n in ("MONTHS", "dateWords"))
    out = _node(nice + "\n" + dates + "\nprocess.stdout.write(JSON.stringify([niceTicks(0, 920, 4).ticks, niceTicks(-2.44, 0, 4).ticks, "
                "niceTicks(0, 1.85, 5).ticks, niceTicks(0, 8.7, 4).ticks, niceTicks(0, 315, 3).ticks, "
                "dateWords('2026-10-01T21:21:49+00:00'), dateWords('')]));")
    assert out[0] == [0, 250, 500, 750, 1000]
    assert out[1] == [-2.5, -2, -1.5, -1, -0.5, 0]
    assert out[2] == [0, 0.5, 1, 1.5, 2] and out[3] == [0, 2, 4, 6, 8, 10] and out[4] == [0, 100, 200, 300, 400]
    assert out[5:] == ["1 Oct 2026", ""]


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
# the brand system, the tab bar and the glossary (Studio redesign phase 5)
# ---------------------------------------------------------------------------

STUDIO = REPO / "studio-web" / "src"


def _js_object(src: str, name: str) -> str:
    """The ``const <name> = { … };`` literal of the template."""
    m = re.search(rf"^const {name} = (\{{.*?^\}});$", src, re.M | re.S)
    assert m, f"no const {name} in the template"
    return m.group(1)


def _ramps(src: str) -> dict:
    """The template's RAMPS literal ({name: [hex, …]})."""
    return {k: re.findall(r'"(#[0-9a-f]{6})"', v)
            for k, v in re.findall(r"^\s*(\w+): \[([^\]]*)\]", _js_object(src, "RAMPS"), re.M)}


def _css_block(css: str, selector: str) -> dict:
    """``{--token: value}`` of the first rule block for ``selector`` (whitespace normalised)."""
    i = css.index(selector + " {")
    body = css[i + len(selector) + 2:css.index("}", i)]
    return {k: " ".join(v.split()) for k, v in re.findall(r"(--[\w-]+):\s*([^;]+);", body)}


def test_ramps_match_studio_palette():
    """RAMPS carry Studio's exact ramp stops (palette.json), categorical and heat colours (palette.ts)."""
    from sparc.core.results_page import TEMPLATE

    fixture = STUDIO / "test" / "fixtures" / "palette.json"
    palette_ts = STUDIO / "theme" / "palette.ts"
    if not fixture.is_file() or not palette_ts.is_file():
        pytest.skip("studio-web is not present")
    ramps = _ramps(TEMPLATE.read_text("utf-8"))
    studio = json.loads(fixture.read_text("utf-8"))["ramps"]
    assert {k: ramps[k] for k in studio} == studio
    ts = palette_ts.read_text("utf-8")
    for theme in ("light", "dark"):
        row = re.search(rf"^\s*{theme}: \{{ (s1: .*?) \}},?$", ts, re.M).group(1)
        cols = dict(re.findall(r'(s[1-6]): "(#[0-9a-f]{6})"', row))
        assert ramps[f"cat{theme.title()}"] == [cols[f"s{i}"] for i in range(1, 7)]
        heat = re.search(rf"^\s*{theme}: (\[[^\]]*\])", ts[ts.index("HEAT_COLORS"):], re.M).group(1)
        assert ramps[f"heat{theme.title()}"] == json.loads(heat)


def test_tokens_match_studio():
    """Every colour and font token the page defines has Studio's value, in light and dark (tokens.css)."""
    from sparc.core.results_page import TEMPLATE

    tokens = STUDIO / "theme" / "tokens.css"
    if not tokens.is_file():
        pytest.skip("studio-web is not present")
    page, studio = TEMPLATE.read_text("utf-8"), tokens.read_text("utf-8")
    for selector in (":root", ':root[data-theme="dark"]', ':root:not([data-theme="light"])'):
        ours, theirs = _css_block(page, selector), _css_block(studio, selector)
        shared = set(ours) & set(theirs)
        assert {"--page", "--ink", "--accent", "--s1", "--hc0"} <= shared
        assert {k: ours[k] for k in shared} == {k: theirs[k] for k in shared}, selector
    assert "--accent-strong" in _css_block(page, ":root") and "--grad-ramp" in _css_block(page, ":root")
    assert "family=Inter" in page and "JetBrains+Mono" in page


def test_tab_bar_and_old_anchors(built):
    """Six tabs in the story's order; every anchor of the earlier page still resolves (an id or an alias)."""
    path, _ = built
    html = path.read_text("utf-8")
    tabs = re.findall(r'<a href="#([\w-]+)" data-sec="\1"(?: data-short="([^"]+)")?>([^<]+)</a>', html)
    assert [(t, label) for t, _, label in tabs] == [("summary", "Summary"), ("map", "Map"), ("cools", "What cools"),
                                                    ("heat-risk", "Heat risk"), ("trust", "How much to trust this"),
                                                    ("details", "Model details")]
    # a phone shows the logo and the short labels (data-short), so all six tabs fit
    assert {t: short for t, short, _ in tabs if short} == {"cools": "Cooling", "heat-risk": "Heat", "trust": "Trust",
                                                         "details": "Details"}
    tabs = [(t, label) for t, _, label in tabs]
    ids = set(re.findall(r'\bid="([\w-]+)"', html)) | set(re.findall(r'checkCard\("([\w-]+)"', html))  # + the trust checks
    assert {t for t, _ in tabs} <= ids and {"accuracy", "ranges", "distance", "cause"} <= ids
    alias = dict(re.findall(r'"?([\w-]+)"?: "([\w-]+)"', _js_object(html, "ALIAS")))
    for old in ("map", "accuracy", "influence", "saturation", "scenarios", "heat", "climate", "causal", "identify",
                "budget", "planner", "validation", "caveats"):
        assert alias.get(old, old) in ids, old
    assert alias["influence"] == "reach" and alias["saturation"] == "diminishing"
    assert '<details class="details-group" id="detailsGroup">' in html          # Model details starts collapsed
    # the things to keep in mind sit in How much to trust this, after the four checks (Studio: Summary and Trust), and
    # the summary's link and the trust tab's note point there; the old anchors still resolve
    trust, details = html.index('<section class="tab-sec" id="trust"'), html.index('<section class="tab-sec" id="details"')
    assert trust < html.index('id="checks"') < html.index('<section class="block" id="caveats"') < details
    assert alias["keep-in-mind"] == alias["limitations"] == "caveats"
    assert 'read the <a href="#caveats">things to keep in mind</a> below' in html
    # the summary's line is one link: "Read the 4 things to keep in mind before using these numbers →"
    assert 'el("a", { href: "#caveats" }, `Read the ${n === 1' in html and "before using these numbers →" in html
    assert "How much to trust this › ${k}" not in html and "Model details › ${k}" not in html
    # What cools: each change's curve, then its reach; the diminishing-returns table and every layer's reach and
    # direction (one table, with the method sentence) are in Model details; the old reach-by-layer anchor leads there
    assert html.index('id="diminishing"') < html.index('id="reach"') < html.index('id="scenarios"')
    assert details < html.index('id="returns-detail"') < html.index('id="spatial"')
    assert html.index("averaged over neighbourhoods of half, one and two times") > details
    assert alias["layer-reach"] == "spatial" and "chart-influence-all" not in html
    assert 'href="#returns-detail">Model details › Diminishing returns by change</a>' in html
    assert re.search(r"\stitle=", html.split("<script>")[0]) is None           # no title= tooltips in the markup


def test_glossary_reaches_the_page(built):
    """DATA.glossary holds every term the page names (from sparc/core/glossary.json), and nothing it does not."""
    from sparc.core.glossary import glossary
    from sparc.core.results_page import TEMPLATE

    path, blob = built
    data = _data_blob(path.read_text("utf-8"))
    terms = glossary()["terms"]
    src = TEMPLATE.read_text("utf-8")
    named = set(re.findall(r'data-(?:tip|gloss)="([^"]+)"', src)) | set(re.findall(r'\b(?:infotip|term)\("([^"]+)"', src))
    named |= set(re.findall(r'\[(?:"[^"\n]*[A-Z ][^"\n]*"|`[^`\n]*`), "([a-z][a-z0-9_]*)"\]', src))
    assert named and named <= set(terms), sorted(named - set(terms))
    assert named <= set(data["glossary"]["terms"])
    assert data["glossary"]["terms"]["r2"] == {k: terms["r2"][k] for k in ("label", "tip", "technical")}
    assert {"trust", "trustRanges", "shapes", "heat", "levers"} <= set(data["glossary"])
    assert data["glossary"] == blob["glossary"] and len(json.dumps(data["glossary"])) < 40_000


#: words a planner should not meet outside Model details (phase 5 plain language; the glossary keeps them in its tips)
JARGON = re.compile(r"emulat|ΔT|RMSE|R²|\bCATE\b|\bDAG\b|jackknife|\bSE\b|stacker|advection|forcing|anisotrop|kernel|"
                    r"\bpriors?\b|mediator|censored|saturat|sigmoid|Gini|Kendall|held[- ]out|\bfolds?\b|quintile|\bSSP|"
                    r"envelope|extrapolat|coverage|half-width|\bdose|own-cell|footprint|closed-loop|open-loop|CMIP|\bcells?\b|"
                    r"\bpp\b|NDVI|[Ii]mpervious|[Aa]lbedo|\bNWS\b|HRSL|pathway", re.I)


def test_map_layers_speak_plainly(built, demo):
    """Every map theme, layer name and description is in plain words (the technical names live in Model details)."""
    from sparc.core.results_page import layer_catalog

    _, blob = built
    _, cfg = demo
    keys = ("obs", "pred", "err", "hw", "people", "plantable", "heat_index", "heat_cat", "alloc_dose", "alloc_delta",
            "hd_90_today", "hd_90_ssp245_mid", "hd_95_ssp585_late")
    full = layer_catalog(cfg, {**blob["layers"], **{k: {} for k in keys}}, units="°F", background=80.0, block_m=390,
                         n_folds=3, scen_layers=blob["scen_layers"], climate=True, sites=True, design=True,
                         optimize_variable="canopy")
    assert {c["group"] for c in full} == {"Temperature", "Land cover", "Cooling per step", "Diminishing returns",
                                          "Scenarios", "Budget plan", "Future climate", "Heat risk", "People & hot days",
                                          "Paint a design", "Test areas"}
    scenario_names = {s["name"] for s in blob["scen_layers"]}           # the config's own scenario names
    for c in full:
        text = " ".join([c["group"], c["desc"]] + ([] if c["name"] in scenario_names else [c["name"]]))
        assert not JARGON.search(text), (c["key"], JARGON.search(text).group(0), text)
    for labels in (blob["layers"][k]["labels"] for k in blob["layers"] if k.startswith("cls_")):
        assert not any(JARGON.search(x) for x in labels), labels


def test_page_wording_follows_studio():
    """The page says what Studio says: the glossary's labels, Studio's diminishing-returns sentence and table, the Lab's
    quick-estimate words, cooler/warmer changes and the ← cooler · warmer → legend."""
    from sparc.core.results_page import TEMPLATE

    src = TEMPLATE.read_text("utf-8")
    for phrase in ("each extra step of ${word}", "About 90% of the possible cooling comes by", "Where most benefit is reached",
                   "Local cooling per step", "Cooling reach per step", '"Area changed"', "Rough for large changes — the quick "
                   "estimate assumes effects simply add up.", "← cooler", "warmer →", "Small (60 m)", "Very large (500 m)",
                   "What the evidence supports", "Things to keep in mind"):
        assert phrase in src, phrase
    # phase 5 review: shape shares in words (the numbers in the bar, which accounts for every place), plateaus beyond the
    # tested amounts not given as numbers, the causal slope's unit, the quick estimate's check behind a question, the
    # budget's cooling totals in plain words with the unit in an ⓘ, and the levers' names on the design buttons
    for phrase in ('"In most places"', 'label: "No clear pattern"', '"— (not reached in tested amounts)"',
                   '" added by a place and all its neighbours"', "<summary>How reliable is the quick estimate?</summary>",
                   '"Checked vs planned", infotip("closed_loop")', "`Total cooling, added up (${U})`", "}, leverName(v));",
                   "scenLabel(s)", "`Changed here: ${ch}`"):
        assert phrase in src, phrase
    # gone from the planner's view: data-QA facts and the run date in the facts row, the repository path in the footer,
    # charts that scroll sideways or shrink their text on a phone, "2.5k" counts, raw scenario names in the chart
    for gone in ('["Readings in whole degrees"', '["Run date"', '["Code version"', "sparc-resilience repository",
                 "min-width: 620px", "#chart-influence svg", "kfmt", "viewBox: `0 0 980", '"t-label" }, s.name))', " · changed here: "):
        assert gone not in src, gone
    # phase 5 review 2: no grid-size strip on the Summary, no repeated grade rule or reach chart, budget tiles in °F (the
    # added-up totals in the chart's caption), a phone can scroll past the map and tap a value, the group's summary
    # says Show / Hide, painted places are outlined and the legend names them
    for gone in ('id="facts"', "That is good:", "See heat exposure", "`${U}, added up`", "Most cooling possible (${U})",
                 "(from the {{N_FOLDS}} models", "hover over one for"):
        assert gone not in src, gone
    for phrase in ("touch-action: pan-y pinch-zoom", ".mapbox.painting canvas { touch-action: none; }", "hover over or tap one",
                   '"Hide the model details"', "getComputedTextLength()", '"Your changes (the places you painted)"', '"No change"',
                   'el("p", { "data-grade-line": "true" }, g.line)', '"Most cooling possible (typical place)"',
                   ".tablewrap table:not(.fit) { min-width: 600px; }"):
        assert phrase in src, phrase
    # repeated concepts come from the shared glossary, not copies: shapes, heat levels, verdicts, emissions, periods
    for read in ("PLAIN_GLOSS.shapes", "PLAIN_GLOSS.emissions", "PLAIN_GLOSS.periods", "PLAIN_GLOSS.quickTrust",
                 '(DATA.glossary || {}).heat', 'gterm("where_changed")', 'gterm("citywide")', '"verdict_robust"'):
        assert read in src, read
    # the static page outside Model details names no technical term
    static = src[src.index('<main class="wrap">'):src.index('<section class="tab-sec" id="details"')]
    visible = re.sub(r"\{\{\w+\}\}|<[^>]+>", " ", static)                       # text only, placeholders out
    assert not JARGON.search(visible), JARGON.search(visible).group(0)

