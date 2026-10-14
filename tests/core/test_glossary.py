"""sparc/core/glossary.json: the plain words shared by SPARC Studio and the results page.

Studio imports the file directly (studio-web/src/content/glossary.ts), so its shape is a contract:
every term has a plain label and a short tip, and the fixed sections cover every key the UI asks for.
"""

from __future__ import annotations

import json
import sys
from importlib import resources
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MAX_TIP = 240


@pytest.fixture(scope="module")
def glossary() -> dict:
    text = resources.files("sparc.core").joinpath("glossary.json").read_text("utf-8")
    return json.loads(text)


def _nonempty(v) -> bool:
    return isinstance(v, str) and v.strip() == v and bool(v)


def test_loads_with_version_and_sections(glossary):
    assert glossary["version"] == 1
    for section in ("terms", "levers", "modes", "confidence", "shapes", "heat", "trust", "engine"):
        assert isinstance(glossary[section], dict) and glossary[section], section


def test_every_term_has_label_and_short_tip(glossary):
    terms = glossary["terms"]
    assert len(terms) >= 35
    for tid, t in terms.items():
        assert tid == tid.lower() and " " not in tid, tid
        assert set(t) <= {"label", "tip", "technical", "more"}, (tid, set(t))
        assert _nonempty(t.get("label")), tid
        assert _nonempty(t.get("tip")), tid
        assert len(t["tip"]) <= MAX_TIP, (tid, len(t["tip"]))
        for opt in ("technical", "more"):
            if opt in t:
                assert _nonempty(t[opt]), (tid, opt)


def test_core_terms_present(glossary):
    terms = glossary["terms"]
    for tid, label in {
        "r2": "Accuracy score",
        "rmse": "Typical error",
        "coverage": "Range reliability",
        "emulator_preview": "Quick estimate",
        "run_exact": "Run full model",
        "compile": "Check the change",
        "cate": "Effect by place",
        "dag_audit": "Cause-and-effect check",
        "gini": "Unevenness of benefit",
        "interval_95": "Likely range",
        "mediators": "Knock-on effects",
    }.items():
        assert terms[tid]["label"] == label, tid


def test_fixed_sections_cover_their_keys(glossary):
    assert set(glossary["confidence"]) == {"confident_cools", "confident_warms", "could_be_zero", "unknown"}
    for c in glossary["confidence"].values():
        assert _nonempty(c["chip"]) and _nonempty(c["sentence"])
    assert set(glossary["shapes"]) == {"saturating", "linear", "sigmoid", "censored"}
    assert set(glossary["trust"]) == {"good", "fair", "limited"}
    for g in glossary["trust"].values():
        assert _nonempty(g["label"]) and _nonempty(g["sentence"])
    for section in ("shapes", "heat"):
        for k, v in glossary[section].items():
            assert _nonempty(v["label"]) and _nonempty(v["tip"]) and len(v["tip"]) <= MAX_TIP, (section, k)
    for k, lv in glossary["levers"].items():
        assert _nonempty(lv["name"]) and _nonempty(lv["oneLine"]) and lv["icon"] in {"trees", "paving", "roofs", "lever"}, k
    for k, m in glossary["modes"].items():
        assert _nonempty(m["label"]) and _nonempty(m["help"]) and _nonempty(m["verb"]) and isinstance(m["planner"], bool), k
    # every edit mode of the scenario engine has plain words
    assert {"add", "set", "scale", "floor", "ceiling", "fill_headroom", "to_percentile", "per_cell"} <= set(glossary["modes"])
    engine_states = {"no_checkpoint", "cold", "queued", "loading", "ready", "busy", "incompatible", "error"}
    assert set(glossary["engine"]["states"]) == engine_states


def test_heat_categories_match_the_nws_bands():
    from sparc.core.heat import CATEGORIES

    text = resources.files("sparc.core").joinpath("glossary.json").read_text("utf-8")
    heat = json.loads(text)["heat"]
    assert [k for k, *_ in CATEGORIES] == list(heat)
    assert [label for _, label, *_ in CATEGORIES] == [v["label"] for v in heat.values()]


def test_emissions_pathways_in_plain_words(glossary):
    em = glossary["emissions"]
    assert [em[c]["label"] for c in ("ssp126", "ssp245", "ssp370", "ssp585")] == ["low emissions", "moderate emissions", "high emissions", "very high emissions"]
    assert em["ssp245"]["technical"] == "SSP2-4.5"
    assert glossary["periods"]["2041-2060"]["label"] == "mid-century (2041–2060)"
    assert glossary["periods"]["2081-2100"]["label"] == "late century (2081–2100)"


def test_glossary_ships_in_the_wheel():
    """setuptools package-data covers sparc/core/glossary.json (Studio and the results page read it)."""
    if sys.version_info < (3, 11):
        pytest.skip("tomllib needs Python 3.11")
    import tomllib

    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    data = cfg["tool"]["setuptools"]["package-data"]
    assert "glossary.json" in data.get("sparc.core", []) or "*.json" in data.get("*", [])
