"""validate_deep: each check of SPEC §9.4 on the synthetic demo config, and the mode previews."""

from __future__ import annotations

import copy
import json

import pytest
import yaml

from sparc.studio.projects.validate import coarse_preview, fast_overrides, validate_deep, validate_report


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory):
    from sparc.core.synthetic import write_demo_project

    d = tmp_path_factory.mktemp("demo")
    write_demo_project(d, n=40, seed=0)
    return d


@pytest.fixture
def raw(demo_dir):
    return yaml.safe_load((demo_dir / "config.yml").read_text())["core"]


def codes(issues, level="error"):
    return {(i["path"], i["code"]) for i in issues if i["level"] == level}


def test_demo_config_has_no_errors(raw, demo_dir):
    issues = validate_deep(raw, demo_dir)
    assert [i for i in issues if i["level"] == "error"] == []
    assert all(set(i) >= {"level", "path", "code", "message"} for i in issues)


def test_missing_column(raw, demo_dir):
    raw["predictors"] = raw["predictors"] + ["nope"]
    assert ("predictors.6", "missing_column") in codes(validate_deep(raw, demo_dir))
    raw["data"]["target"] = "TT"
    assert ("data.target", "missing_column") in codes(validate_deep(raw, demo_dir))


def test_lever_not_a_predictor(raw, demo_dir):
    raw["actionable"]["nope"] = {"min": 0, "max": 1, "doses": [0, 0.1]}
    assert ("actionable.nope", "lever_not_predictor") in codes(validate_deep(raw, demo_dir))


def test_dose_outside_bounds_and_zero(raw, demo_dir):
    raw["actionable"]["canopy"]["doses"] = [0, 5, 150]
    raw["actionable"]["albedo"]["doses"] = [0.05, -0.1]
    issues = validate_deep(raw, demo_dir)
    errs = codes(issues)
    assert ("actionable.canopy.doses.2", "dose_out_of_bounds") in errs
    assert ("actionable.albedo.doses.1", "dose_out_of_bounds") in errs
    zero = [i for i in issues if i["code"] == "dose_zero_missing"]
    assert zero and zero[0]["level"] == "warn" and zero[0]["fix"]["value"][0] == 0


def test_role_mapped_to_a_non_predictor(raw, demo_dir):
    raw["physics"]["roles"]["canopy"] = "tree_cover"
    assert ("physics.roles.canopy", "role_not_predictor") in codes(validate_deep(raw, demo_dir))


def test_missing_roles_warn(raw, demo_dir):
    del raw["physics"]["roles"]["impervious"]
    warns = [i for i in validate_deep(raw, demo_dir) if i["code"] == "roles_missing"]
    assert warns and "placebo, simcheck, planner and emulator" in warns[0]["message"]


def test_scenario_variable_not_actionable(raw, demo_dir):
    raw["scenarios"].append({"name": "Greener", "variable": "ndvi", "direction": "increase", "increments": [0.1]})
    raw["joint_scenarios"][0]["interventions"].append({"variable": "elevation", "increment": 1})
    errs = codes(validate_deep(raw, demo_dir))
    assert ("scenarios.3.variable", "scenario_not_actionable") in errs
    assert ("joint_scenarios.0.interventions.3.variable", "scenario_not_actionable") in errs


def test_causal_treatment_not_a_predictor(raw, demo_dir):
    raw["causal"]["treatments"] = ["canopy", "shade"]
    raw["causal"]["confounders"]["canopy"].append("income")
    errs = codes(validate_deep(raw, demo_dir))
    assert ("causal.treatments.1", "causal_not_predictor") in errs
    assert ("causal.confounders.canopy.4", "causal_not_predictor") in errs


def test_climate_table_required_with_source_table(raw, demo_dir):
    raw["climate"]["table"] = None
    assert ("climate.table", "climate_table_missing") in codes(validate_deep(raw, demo_dir))
    raw["climate"]["table"] = "inputs/climate/gone.csv"
    assert ("climate.table", "missing_file") in codes(validate_deep(raw, demo_dir))
    raw["climate"]["source"] = "cmip6"
    assert not any(p == "climate.table" for p, _ in codes(validate_deep(raw, demo_dir)))


def test_people_objective_needs_layers_and_crs(raw, demo_dir):
    raw["optimize"]["objective"] = "people"
    raw["planner"] = {}
    assert ("planner.layers", "needs_layers") in codes(validate_deep(raw, demo_dir))
    raw["planner"] = {"layers": "inputs/layers/demo_layers.parquet"}
    raw["data"]["crs"] = None
    errs = codes(validate_deep(raw, demo_dir))
    assert ("data.crs", "needs_crs") in errs
    assert ("planner.layers", "needs_layers") not in errs


def test_forcing_file_missing_or_invalid(raw, demo_dir, tmp_path):
    raw["physics"]["forcing"] = "inputs/forcing/none.json"
    assert ("physics.forcing", "missing_file") in codes(validate_deep(raw, demo_dir))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"date": "2020-07-29"}))
    raw["physics"]["forcing"] = str(bad)
    assert ("physics.forcing", "forcing_invalid") in codes(validate_deep(raw, demo_dir))


def test_required_keys_files_and_predictors(demo_dir):
    issues = validate_deep({"data": {}, "predictors": []}, demo_dir)
    errs = codes(issues)
    assert {("data.target", "required"), ("data.path", "required"), ("predictors", "no_predictors")} <= errs
    issues = validate_deep({"data": {"path": "data/gone.csv", "target": "T"}, "predictors": ["a"]}, demo_dir)
    assert ("data.path", "missing_file") in codes(issues)


def test_types_units_and_unknown_keys(raw, demo_dir):
    raw["cv"]["n_folds"] = "five"
    raw["data"]["coord_unit"] = "furlong"
    raw["stacker"]["mystery"] = 3
    issues = validate_deep(raw, demo_dir)
    errs = codes(issues)
    assert ("cv.n_folds", "type") in errs
    assert any(p == "data.coord_unit" for p, _ in errs)
    assert ("stacker.mystery", "unknown_key") in codes(issues, "info")


def test_lever_and_equity_columns(raw, demo_dir):
    raw["optimize"]["variable"] = "ndvi"
    raw["optimize"]["equity_column"] = "income"
    raw["mediators"]["ndvi"]["parents"].append("grass")
    errs = codes(validate_deep(raw, demo_dir))
    assert ("optimize.variable", "optimize_not_actionable") in errs
    assert ("optimize.equity_column", "missing_column") in errs
    assert ("mediators.ndvi.parents.2", "mediator_not_predictor") in errs


def test_climate_without_site_or_crs(raw, demo_dir):
    raw["data"]["crs"] = None
    raw["climate"].pop("site", None)
    assert ("data.crs", "needs_crs") in codes(validate_deep(raw, demo_dir))
    raw["climate"]["site"] = [41.8, -71.4]
    assert ("data.crs", "needs_crs") not in codes(validate_deep(raw, demo_dir))


def test_validate_report_previews(raw, demo_dir):
    rep = validate_report(copy.deepcopy(raw), demo_dir)
    assert rep["ok"] is True
    fo = fast_overrides(raw, demo_dir)
    assert fo["data.subsample"] == {"from": None, "to": 8000}
    assert fo["stacker.tune_lambda"]["to"] == [0.0, 0.1]
    cp = coarse_preview(raw, demo_dir)
    assert cp["cell_m"] == 60.0 and cp["fine_cell_m"] == 30.0 and 0 < cp["n_cells"] < cp["n_input"] and cp["ok"]
    assert coarse_preview(raw, demo_dir, cell_m=30.0)["ok"] is False
