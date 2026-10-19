"""Synthetic validation of the LST pilot's if-then logic (L5): in a world where LST carries a hidden heat source the
harness must find B and C better than A; where LST is only land cover plus noise it must say "not shown"; where the
10:00 surface pattern differs from the 15:00 air pattern it must report the loss."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sparc.core.lstair import synthetic as S
from sparc.core.lstair.evaluate import Settings

FAST = Settings(physics_cell_m=30.0, physics_iter=30, within_city=False, gbm=False, boot=1000)


def test_worlds_plant_what_they_say():
    inf = S.make_city(0, "informative", n=48)
    red = S.make_city(0, "redundant", n=48)
    null = S.make_city(0, "null", n=48)
    for c in (inf, red):
        t = c.table
        assert len(t) > 150 and abs(t["dTa"].median()) < 1e-12 and c.rasters["canopy"].shape == (48, 48)
        assert set(S.SCENARIOS) == {"informative", "redundant", "mismatch", "null"}
    # in the redundant world LST is (almost) a linear function of the cell's land cover; in the informative one not
    def r2_on_land_cover(c):
        t = c.table
        X = np.column_stack([np.ones(len(t))] + [t[k] for k in ("canopy", "impervious", "albedo", "ndvi")])
        b, *_ = np.linalg.lstsq(X, t["lst"], rcond=None)
        return 1 - np.var(t["lst"] - X @ b) / np.var(t["lst"])
    assert r2_on_land_cover(red) > 0.9 > r2_on_land_cover(inf) and r2_on_land_cover(null) < 0.2
    # the null world's control field has the LST's spread but no relation to the air
    assert abs(null.table["lst"].std() / red.table["lst"].std() - 1) < 0.5
    assert abs(np.corrcoef(null.table["lst"], null.table["dTa"])[0, 1]) < 0.2
    with pytest.raises(ValueError):
        S.make_city(0, "nope")


def test_fast_validation_every_world_gives_its_answer(tmp_path):
    v = S.validate(n_cities=8, seeds=(0,), n=44, settings=FAST, out=tmp_path)
    rows = {r["scenario"]: r for r in v["rows"]}
    inf, red, mis, null = rows["informative"], rows["redundant"], rows["mismatch"], rows["null"]
    assert null["B-A"]["verdict"] == "not shown" and null["C-C0"]["verdict"] == "not shown" and null["as_expected"]
    assert v["mode"].startswith("single draws") and set(v["rates"]) == set(S.SCENARIOS)
    assert inf["B-A"]["verdict"] == "LST adds information beyond land cover" and inf["B-A"]["ci95"][1] < 0
    assert inf["C-A"]["verdict"] == "physics with LST beats land cover" and inf["C-A"]["ci95"][1] < 0
    assert red["B-A"]["verdict"] == "not shown"
    sh = mis["share_of_informative_gain"]
    assert sh["B-A"] < 0.5 and sh["C-C0"] < 0.5                       # most of LST's gain is lost at the wrong hour
    assert v["all_as_expected"]
    md = (tmp_path / "validation.md").read_text()
    assert "| informative |" in md and "All as expected: yes" in md
    assert json.loads((tmp_path / "validation.json").read_text())["all_as_expected"] is True


def test_written_world_runs_through_the_cli(tmp_path):
    from sparc.core.cli import main

    S.write_world(S.make_world("informative", 4, seed=3, n=40), tmp_path / "data")
    rc = main(["lstair", "evaluate", "--data", str(tmp_path / "data"), "--out", str(tmp_path / "res"), "--no-gbm",
               "--physics-cell", "30", "--physics-iter", "20", "--boot", "300", "--no-within-city"])
    assert rc == 0
    res = json.loads((tmp_path / "res" / "results.json").read_text())
    w = res["windows"]["af"]
    assert len(w["per_city"]) == 4 and w["paired"]["B-A"]["verdict"] == "not testable"      # 4 < 5 cities
    assert w["paired"]["B-A"]["mean_diff_mse"] < 0


def test_cli_validate_fast(tmp_path):
    from sparc.core.cli import main

    assert main(["lstair", "validate", "--out", str(tmp_path), "--cities", "8", "--size", "40", "--fast"]) == 0
    assert (tmp_path / "validation.md").exists()


@pytest.mark.slow
def test_full_validation_three_seeds_with_boosting(tmp_path):
    """The fuller run: 10 cities per world, three seeds, both model families, 4000 bootstraps."""
    v = S.validate(n_cities=10, seeds=(0, 1, 2), n=64, settings=Settings(physics_cell_m=30.0, within_city=False),
                   out=tmp_path)
    bad = [(r["scenario"], r["seed"], r["B-A"]["verdict"], r["C-A"]["verdict"]) for r in v["rows"]
           if not r.get("as_expected", True)]
    assert v["all_as_expected"], bad


def test_claim_rates_and_the_rate_mode():
    """R3: rates over seeds with Clopper–Pearson intervals; the null worlds' counts are checked against the level."""
    def row(sc, seed, b, c0="not shown", c="not shown"):
        return {"scenario": sc, "seed": seed, "status": "done", "B-A": {"verdict": b}, "C-A": {"verdict": c},
                "C-C0": {"verdict": c0}}
    claim = S.CLAIMS["B-A"]
    rows = [row("null", i, claim if i == 0 else "not shown") for i in range(40)]
    rows += [row("informative", i, claim if i % 4 else "not shown", c=S.CLAIMS["C-A"]) for i in range(40)]
    r = S.claim_rates(rows)
    assert r["null"]["B-A"]["claims"] == 1 and r["null"]["B-A"]["calibrated"]
    assert r["null"]["B-A"]["ci95"][0] < 0.025 < r["null"]["B-A"]["ci95"][1]
    assert r["informative"]["B-A"]["rate"] == 0.75 and "calibrated" not in r["informative"]["B-A"]
    bad = S.claim_rates([row("null", i, claim if i < 8 else "not shown") for i in range(40)])
    assert bad["null"]["B-A"]["calibrated"] is False
    assert "calibrated" not in r["null"]["C-A"] and "calibrated" in r["null"]["C-C0"]


@pytest.mark.slow
def test_claim_rates_over_many_seeds(tmp_path):
    """The rate run: 20 seeds × 4 worlds at 8 cities (spline ridge, short physics fits)."""
    v = S.validate(n_cities=8, seeds=tuple(range(20)), n=44, settings=FAST, out=tmp_path)
    assert v["mode"] == "rates over seeds" and v["all_as_expected"], v["checks"]
