"""CoreConfigModel against core's DEFAULTS, the keys DEFAULTS leaves out, and the x-ui hints."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from sparc.core.config import DEFAULTS
from sparc.studio.projects.config_schema import CoreConfigModel, config_json_schema, defaults_diff, model_paths

REPO = Path(__file__).resolve().parents[3]


def test_model_covers_every_default_key_with_the_same_default():
    diff = defaults_diff()
    assert diff["missing"] == [], f"CoreConfigModel lacks DEFAULTS keys: {diff['missing']}"
    assert diff["different"] == [], f"defaults differ: {diff['different']}"


def test_the_diff_catches_a_new_default():
    d = copy.deepcopy(DEFAULTS)
    d["stacker"]["new_knob"] = 1
    d["brand_new_section"] = {"x": 1}
    d["cv"]["n_folds"] = 7
    diff = defaults_diff(d)
    assert set(diff["missing"]) == {"stacker.new_knob", "brand_new_section"}
    assert [x["path"] for x in diff["different"]] == ["cv.n_folds"]


def test_keys_core_reads_beyond_defaults():
    paths = model_paths()
    extra = {"planner.layers", "planner.paved_plantable_share", "planner.ghcn_station", "report.title",
             "report.limitations", "report.caveats", "response.clip_to_support", "physics.albedo_map",
             "physics.shade_form", "physics.priors", "physics.num_threads", "influence.n_bins", "influence.families",
             "influence.jack_side", "causal.dag_audit.edges", "causal.dag_audit.n_iter", "output.scenario_detail",
             "physics.roles.water_distance", "data.join"}
    assert extra <= paths, extra - paths


def test_real_configs_validate_and_dag_audit_dict_form():
    for name in ("core_providence.yml", "core_providence_open.yml"):
        raw = yaml.safe_load((REPO / "configs" / name).read_text())["core"]
        m = CoreConfigModel.model_validate(raw)
        assert m.report.title == "Providence Heat Model"
    m = CoreConfigModel.model_validate({"causal": {"dag_audit": {"n_boot": 3}}})
    assert m.causal.dag_audit.n_boot == 3 and m.causal.dag_audit.n_iter == 2000
    assert CoreConfigModel.model_validate({"causal": {"dag_audit": True}}).causal.dag_audit is True
    m = CoreConfigModel.model_validate({"stacker": {"future_flag": 1}})
    assert m.stacker.model_extra == {"future_flag": 1}                  # unknown keys pass through


def test_json_schema_hints():
    s = config_json_schema()
    roles = s["$defs"]["RolesSection"]["properties"]
    assert set(roles) == {"albedo", "canopy", "impervious", "ndvi", "elevation", "water_distance"}
    for defn in s["$defs"].values():
        for name, prop in defn.get("properties", {}).items():
            assert "x-ui" in prop, name
            assert prop["x-ui"]["group"] in ("data", "levers", "physics", "scenarios", "analysis", "about", "run")
    clim = s["$defs"]["ClimateSection"]["properties"]
    assert clim["experiments"]["x-ui"]["enum_labels"]["ssp245"] == "SSP2-4.5"
