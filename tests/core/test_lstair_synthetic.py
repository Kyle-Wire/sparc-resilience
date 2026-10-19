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
    for c in (inf, red):
        t = c.table
        assert len(t) > 150 and abs(t["dTa"].median()) < 1e-12 and c.rasters["canopy"].shape == (48, 48)
        assert set(S.SCENARIOS) == {"informative", "redundant", "mismatch"}
    # in the redundant world LST is (almost) a linear function of the cell's land cover; in the informative one not
    def r2_on_land_cover(c):
        t = c.table
        X = np.column_stack([np.ones(len(t))] + [t[k] for k in ("canopy", "impervious", "albedo", "ndvi")])
        b, *_ = np.linalg.lstsq(X, t["lst"], rcond=None)
        return 1 - np.var(t["lst"] - X @ b) / np.var(t["lst"])
    assert r2_on_land_cover(red) > 0.9 > r2_on_land_cover(inf)
    with pytest.raises(ValueError):
        S.make_city(0, "nope")


def test_fast_validation_every_world_gives_its_answer(tmp_path):
    v = S.validate(n_cities=8, seeds=(0,), n=44, settings=FAST, out=tmp_path)
    rows = {r["scenario"]: r for r in v["rows"]}
    inf, red, mis = rows["informative"], rows["redundant"], rows["mismatch"]
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
