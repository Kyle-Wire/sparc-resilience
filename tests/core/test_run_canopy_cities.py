"""scripts/run_canopy_cities.py: status rows, the pool-entry rule and a batch that never aborts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def drv():
    spec = importlib.util.spec_from_file_location("run_canopy_cities", REPO / "scripts" / "run_canopy_cities.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _estimate():
    def run(name, window, est, se):
        return {"run": name, "window": window, "qa": {"n_samples": 2000},
                "street": {"street_100": {"estimate": est, "se": se, "lo": est - 1.96 * se, "hi": est + 1.96 * se},
                           "street_300": {"estimate": 2 * est, "se": se}}}
    return {"runs": [run("2020-07-29 morning", "morning", -0.03, 0.03), run("2020-07-29 midday", "midday", 0.02, 0.08),
                     run("2020-07-29 evening", "evening", -0.08, 0.02), {"run": "2020-07-30 night", "error": "off grid"}],
            "headline": {"street_100": {"estimate": -0.03, "lo": -0.07, "hi": 0.02}},
            "kilometre": {"result": {"estimate": 0.08}, "lab_status": "underpowered"}}


def _lab(status, reps=36, floor=True):
    return {"n_reps": {"null": reps, "additive": reps}, "designs": {"street_100": {"verdict": {"status": status},
                                                                                  "floor": {"valid": floor}}}}


def test_pool_entry_needs_a_trustworthy_verdict_on_36_replicates(drv):
    labs = {"morning": _lab("trustworthy"), "midday": _lab("direction only"), "evening": _lab("trustworthy", reps=12)}
    rows = drv.run_rows("providence_ri", _estimate(), labs)
    assert [r["run"] for r in rows] == ["2020-07-29 morning", "2020-07-29 midday", "2020-07-29 evening"]
    assert [r["pooled"] for r in rows] == [True, False, False]          # evening: only 12 replicates
    assert rows[0]["street_100_se"] == 0.03 and rows[1]["route_verdict"] == "direction only"
    c = drv.city_row("providence_ri", {"lab": {"status": "ok", "seconds": 60.0}}, _lab("trustworthy"), _estimate(), rows)
    assert c["status"] == "ok" and c["runs_pooled"] == 1 and c["runs"] == 3 and c["lab_street_100"] == "trustworthy"
    assert c["wind_shift_lab"] == "underpowered" and c["minutes"] == 1.0
    md = drv.status_markdown([c], rows)
    assert "| providence_ri | ok | trustworthy | -0.030 (-0.070 to +0.020) | 1/3 |" in md


def test_two_runs_in_one_window_are_named_by_run(drv):
    assert drv.route_tag("2021-07-09 morning", "morning", ["morning", "midday"]) == "morning"
    assert drv.route_tag("2021-07-10 morning", "morning", ["morning", "morning"]) == "2021-07-10 morning"


def test_a_failing_city_is_recorded_and_the_batch_goes_on(drv, monkeypatch, tmp_path):
    def run_city(slug, args):
        if slug == "bad_city":
            raise RuntimeError("boom")
        if slug == "no_setup":
            steps = {"setup": {"status": "failed", "seconds": 5.0, "error": "exit 1"}}
            return drv.city_row(slug, steps, None, None, []), []
        rows = drv.run_rows(slug, _estimate(), {"morning": _lab("trustworthy")})
        return drv.city_row(slug, {}, _lab("trustworthy"), _estimate(), rows), rows

    monkeypatch.setattr(drv, "run_city", run_city)
    monkeypatch.setattr(drv, "REPO", tmp_path)
    assert drv.main(["bad_city", "no_setup", "good_city"]) == 0
    import pandas as pd

    st = pd.read_csv(tmp_path / "output" / "core" / "pooled" / "city_status.csv")
    assert st["city"].tolist() == ["bad_city", "no_setup", "good_city"]
    assert st["status"].tolist() == ["failed at driver", "failed at setup", "ok"]
    assert "boom" in st["error"].iloc[0]
    runs = pd.read_csv(tmp_path / "output" / "core" / "pooled" / "run_status.csv")
    assert runs["pooled"].tolist() == [True, False, False]


def test_the_status_table_is_cumulative_across_batches(drv, monkeypatch, tmp_path):
    """Running one city must not erase the others' rows (Seattle's batch overwrote three cities' status)."""
    def run_city(slug, args):
        rows = drv.run_rows(slug, _estimate(), {"morning": _lab("trustworthy")})
        return drv.city_row(slug, {}, _lab("trustworthy"), _estimate(), rows), rows

    monkeypatch.setattr(drv, "run_city", run_city)
    monkeypatch.setattr(drv, "REPO", tmp_path)
    drv.main(["providence_ri", "burlington_vt"])
    drv.main(["seattle_wa", "burlington_vt"])
    import pandas as pd

    st = pd.read_csv(tmp_path / "output" / "core" / "pooled" / "city_status.csv")
    assert st["city"].tolist() == ["providence_ri", "seattle_wa", "burlington_vt"]
    runs = pd.read_csv(tmp_path / "output" / "core" / "pooled" / "run_status.csv")
    assert runs.groupby("city").size().to_dict() == {"burlington_vt": 3, "providence_ri": 3, "seattle_wa": 3}
    assert "| providence_ri |" in (tmp_path / "output" / "core" / "pooled" / "city_status.md").read_text()


def test_rows_carry_the_street_300_verdict_and_its_pool_flag(drv):
    lab = _lab("direction only")
    lab["designs"]["street_300"] = {"verdict": {"status": "trustworthy"}}
    rows = drv.run_rows("x", _estimate(), {"morning": lab})
    assert rows[0]["pooled"] is False and rows[0]["pooled_300"] is True and rows[0]["route_verdict_300"] == "trustworthy"
    assert rows[1]["pooled_300"] is False and rows[1]["route_verdict_300"] == "not run"


def test_pool_doc_rewrites_only_its_generated_blocks():
    spec = importlib.util.spec_from_file_location("pool_doc", REPO / "scripts" / "pool_doc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = "intro\n<!-- generated:pool:start -->\nold\n<!-- generated:pool:end -->\nafter\n"
    new = mod.replace_block(text, "pool", "| a |")
    assert new == "intro\n<!-- generated:pool:start -->\n| a |\n<!-- generated:pool:end -->\nafter\n"
    with pytest.raises(SystemExit):
        mod.replace_block(text, "status", "x")
