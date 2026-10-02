"""The output catalog (``sparc/core/catalog.py``, SPEC §6.1): every file a run, a post-run job or a study writes
maps to one entry; scenario slugs; per-lever expansion; the data dictionary."""

from __future__ import annotations

import fnmatch
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from sparc.core import catalog as C

REPO = Path(__file__).resolve().parents[3]

POST_RUN = {
    "emulator.npz": "emulator", "emulator.json": "emulator", "uncertainty.json": "uncertainty",
    "uncertainty.md": "uncertainty_md", "placebo.json": "placebo_copy", "planner/planner.json": "planner",
    "planner/planner_cells.parquet": "planner_cells", "planner/hex_250m.csv": "planner_hex:250",
    "planner/hex_500m.csv": "planner_hex:500", "planner/hex_1000m.csv": "planner_hex_other",
    "planner/hexagons.gpkg": "planner_gpkg", "planner/geotiff/canopy.tif": "planner_geotiff",
    "planner/logger_sites.csv": "logger_sites", "planner/before_after_pairs.csv": "before_after_pairs",
    "results.html": "results_page", "reproduce.json": "reproduce", "cv_distance.json": "cv_distance",
    "checkpoint.pkl": "checkpoint", "input_frame.parquet": "input_frame",
}
STUDY = {
    "placebo.json": "placebo_study", "placebo.md": "placebo_md", "simcheck.jsonl": "simcheck_rows",
    "simcheck_summary.json": "simcheck_summary", "simcheck_summary.md": "simcheck_md",
    "multiverse_summary.json": "multiverse_summary", "multiverse_summary.md": "multiverse_md",
    "benchmark.json": "benchmark", "benchmark.md": "benchmark_md",
    "no_physics_maps.npz": "multiverse_maps:{variant}", "no_physics.json": "multiverse_variant:{variant}",
}
STUDIO = {
    "launch.json": "launch", "launch.2.json": "launch_history", "import.json": "import_record",
    "cache/grid.npz": "grid_cache", "engine/base.npy": "engine_cache", "results/res_1/cells.parquet": "studio_results",
    "plans/plan_1/plan.json": "studio_plans", "sweeps/sw_1.json": "studio_sweeps",
    "comparisons/cmp_1/delta.npy": "studio_comparisons", "blobs/b_1.bin": "studio_blobs",
}


def _files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_every_fixture_file_matches_one_output(synth):
    files = _files(synth)
    assert "predictions.parquet" in files and "manifest.json" in files
    unmatched = [f for f in files if C.match_output(f) is None and not C.is_ignored(f)]
    assert unmatched == []
    assert [f for f in files if C.is_ignored(f)] == ["FIXTURE.json"]
    assert C.match_output("response_canopy.parquet").id == "response_maps:{var}"
    assert C.match_output("response_canopy.parquet").var_of("response_canopy.parquet") == "canopy"
    ids = [s.id for s in C.OUTPUTS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("table,root", [(POST_RUN, "run"), (STUDY, "study"), (STUDIO, "studio")])
def test_post_run_study_and_studio_names(table, root):
    for rel, oid in table.items():
        spec = C.match_output(rel, root=root)
        assert spec is not None and spec.id == oid, (rel, spec and spec.id)
    assert C.is_ignored("studio/cache/grid.npz") and C.is_ignored("x.tmp") and C.is_ignored("planner/a.tmp")
    assert C.is_ignored("__pycache__/m.pyc") and not C.is_ignored("report.md")


def test_providence_run_files_match(providence_runs):
    for name in ("providence_uhi_fast", "providence_uhi"):
        d = providence_runs / name
        if not d.is_dir():
            continue
        files = [f for f in _files(d) if not C.is_ignored(f)]
        assert [f for f in files if C.match_output(f) is None] == [], name


def test_expand_outputs_per_lever():
    specs = C.expand_outputs({"actionable": {"canopy": {}, "albedo": {}}})
    ids = [s.id for s in specs]
    assert "response_maps:canopy" in ids and "response_maps:albedo" in ids and "response_maps:{var}" not in ids
    canopy = C.output_by_id("response_maps:canopy", specs)
    assert canopy.files == ("response_canopy.parquet",) and canopy.matches("response_canopy.parquet")
    assert not canopy.matches("response_albedo.parquet")
    assert all(s.root == "run" for s in specs)
    assert {s.root for s in C.expand_outputs(None, root=None)} == {"run", "study", "studio"}


@pytest.mark.parametrize("name,slug", [
    ("Canopy Increase +10", "canopy-increase-plus-10"),
    ("Impervious Decrease −10", "impervious-decrease-minus-10"),
    ("Impervious Decrease -20", "impervious-decrease-minus-20"),
    ("Albedo Increase +0.1", "albedo-increase-plus-0-1"),
    ("Cooling package", "cooling-package"),
    ("  ***  ", "scenario"),
    ("Well-known plan", "well-known-plan"),
])
def test_scenario_slug(name, slug):
    assert C.scenario_slug(name) == slug


def test_scenario_slug_collisions():
    assert C.scenario_slug("Canopy +5", {"canopy-plus-5"}) == "canopy-plus-5-2"
    assert C.scenario_slug("Canopy +5", {"canopy-plus-5", "canopy-plus-5-2"}) == "canopy-plus-5-3"


def _described(rows, output: str, column: str) -> dict | None:
    for r in rows:
        if r["output"] == output and fnmatch.fnmatchcase(column, r["column"]):
            return r
    return None


def test_dictionary_covers_every_fixture_column(synth):
    rows = C.dictionary_rows({"data": {"target_units": "degF"},
                              "actionable": {"canopy": {"unit": "pp"}, "impervious": {"unit": "pp"},
                                             "albedo": {"unit": "fraction"}}})
    for p in sorted(synth.glob("*.parquet")):
        for col in pd.read_parquet(p).columns:
            assert _described(rows, p.name, col) is not None, (p.name, col)
    assert _described(rows, "predictions.parquet", "pred")["unit"] == "°F"
    cate = _described(rows, "causal_cells.parquet", "cate:canopy")
    assert cate is not None and "°F" in cate["unit"]
    assert any(r["output"] == "response_albedo.parquet" for r in rows)


def test_units_for():
    u = C.units_for({"data": {"target_units": "degC"}, "actionable": {"canopy": {"unit": "pp"}, "x": {}}})
    assert u["target"] == "°C" and u["lever:canopy"] == "pp" and u["lever:x"] == "units"
    assert C.unit_label("degF") == "°F" and C.unit_label(None) == ""


def test_catalog_import_is_light():
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPARC_")}
    env["PYTHONPATH"] = str(REPO)
    code = ("import sys, sparc.core.catalog; "
            "print(sorted(m for m in ('numpy', 'pandas', 'torch', 'sklearn') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(REPO), capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
