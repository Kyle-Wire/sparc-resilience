"""
Session-state manager for the SPARC Streamlit UI.

Centralises all configuration fields matching the ``project.yml`` schema so
that every page reads/writes the same shared state dictionary.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

# ── Default values (mirrors templates/blank/project.yml) ─────────────────────

_BLANK_DEFAULTS: dict[str, Any] = {
    # -- project metadata --
    "project_name": "My Spatial Analysis",
    "project_description": "",
    "project_domain": "custom",
    "project_version": "1.0.0",
    "project_author": "",
    "response_units": "",

    # -- data --
    "data_file_path": "",
    "target_column": "",
    "identifier_column": "",
    "coord_x": "",
    "coord_y": "",
    "areas_of_interest_file": "",

    # -- CRS --
    "crs_input": "",
    "crs_projected": "",

    # -- predictors & constraints --
    "predictors": [],
    "monotone_constraints": {},          # {var_name: -1 | 0 | 1}
    "actionable_variables": [],
    "fixed_variables": [],

    # -- physics --
    "physics_priors_file": "",
    "physics_caps_file": "",
    "priors": {},                        # {var: {value, units, uncertainty, source, confidence, description}}
    "caps_variable_bounds": {},
    "caps_delta_constraints": {},
    "caps_monotonicity": {},
    "caps_diminishing_return_thresholds": {},
    "caps_validation": {
        "max_single_cell_delta": 10.0,
        "typical_max_delta": 5.0,
        "max_zone_avg_delta": 4.0,
        "sign_violation_action": "error",
        "magnitude_violation_action": "warning",
    },
    "calibration_shrinkage_weight": 0.7,
    "calibration_max_deviation": 0.5,
    "calibration_adaptive_weighting": True,

    # -- causal DAG --
    "dag_nodes": [],                     # [{name, type, description}]
    "dag_edges": [],                     # [{parent, child, mechanism}]
    "causal_estimator": "hgb",
    "causal_estimate_cate": False,

    # -- pipeline --
    "random_seed": 42,
    "n_spatial_folds": 5,
    "fast_mode": False,
    "overwrite_outputs": False,
    "run_mc_uncertainty": False,
    "n_mc_draws": 50,

    # -- flags --
    "use_gwen_selection": True,
    "include_gwrf_in_ensemble": True,
    "use_laplacian_eigenmaps_in_ols": False,

    # -- model hyper-parameters --
    "gwr_bandwidth": None,
    "gwr_kernel": "gaussian",
    "gwr_alpha": 0.1,
    "gwr_min_points": 50,

    "gwrf_n_estimators": 100,
    "gwrf_k_neighbors": 100,
    "gwrf_min_samples_leaf": 5,
    "gwrf_n_jobs": 1,

    "ggpgam_n_splines": 5,
    "ggpgam_n_spatial_bases": 10,
    "ggpgam_lam": 0.6,
    "ggpgam_max_iter": 100,

    "meta_algorithm": "lightgbm",
    "meta_n_optuna_trials": 25,
    "meta_include_base_features": True,
    "meta_include_laplacian_pca": True,

    "dk_hidden_layers": [64, 32, 16],
    "dk_dropout_rate": 0.2,
    "dk_epochs": 100,
    "dk_learning_rate": 0.001,
    "dk_batch_size": 256,

    "cv_block_size": 300,
    "cv_buffer_size": 300,
    "cv_block_size_source": "user",
    "cv_method": "block",
    "cv_stratify_y": True,

    "gwen_sample_size": 5000,
    "gwen_k_neighbors": 500,
    "gwen_cv_folds": 5,
    "gwen_selection_threshold": 0.1,

    "laplacian_n_eigenmaps": 150,
    "laplacian_k_for_swm": 10,

    # -- scenarios --
    "scenarios": [],                     # [{name, variable, direction, increments, min_val, max_val, unit}]
    "joint_scenarios": [],               # [{name, auto_propagate_dag, interventions: [{variable, direction, increment}]}]

    # -- output / working dir --
    "output_base_dir": "output",
    "working_dir": "",                   # root project folder (where YAMLs get written)

    # -- stage dirs (rarely changed) --
    "stage_dir_1": "Stage_1_Correlogram_Analysis",
    "stage_dir_2": "Stage_2_Spatial_CV",
    "stage_dir_3": "Stage_3_Causal_Validation",
    "stage_dir_4": "Stage_4_Scenarios",
    "stage_dir_final": "Final_Interpretation_Results",

    # -- internal UI-state (not serialized to YAML) --
    "_csv_columns": [],
    "_csv_preview": None,
}


def init_state() -> None:
    """Populate ``st.session_state`` with default values (no-op if already set)."""
    for key, val in _BLANK_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(val)


_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"


def available_templates() -> list[str]:
    """Return sorted list of available template names."""
    if not _TEMPLATES_DIR.is_dir():
        return []
    return sorted(
        d.name for d in _TEMPLATES_DIR.iterdir()
        if d.is_dir() and (d / "project.yml").exists()
    )


def load_template(name: str = "uhi") -> None:
    """Overwrite state from a bundled template's ``project.yml``."""
    # templates/ lives at repo root: sparc/ui/state.py → parent.parent.parent
    tpl_dir = Path(__file__).resolve().parent.parent.parent / "templates" / name
    yml_path = tpl_dir / "project.yml"
    if not yml_path.exists():
        st.error(f"Template '{name}' not found at {yml_path}")
        return
    with open(yml_path) as f:
        cfg = yaml.safe_load(f)
    _apply_yaml_to_state(cfg, tpl_dir)

    # Load DAG
    dag_file = tpl_dir / "causal" / "dag.yml"
    if dag_file.exists():
        with open(dag_file) as f:
            dag = yaml.safe_load(f) or {}
        st.session_state["dag_nodes"] = dag.get("nodes", [])
        st.session_state["dag_edges"] = dag.get("edges", [])

    # Load physics priors
    priors_file = tpl_dir / "physics" / "priors.yml"
    if priors_file.exists():
        with open(priors_file) as f:
            priors = yaml.safe_load(f) or {}
        st.session_state["priors"] = priors.get("coefficients", {})
        cal = priors.get("calibration", {})
        if cal:
            st.session_state["calibration_shrinkage_weight"] = cal.get("shrinkage_weight", 0.7)
            st.session_state["calibration_max_deviation"] = cal.get("max_deviation", 0.5)
            st.session_state["calibration_adaptive_weighting"] = cal.get("adaptive_weighting", True)

    # Load physics caps
    caps_file = tpl_dir / "physics" / "caps.yml"
    if caps_file.exists():
        with open(caps_file) as f:
            caps = yaml.safe_load(f) or {}
        st.session_state["caps_variable_bounds"] = caps.get("variable_bounds", {})
        st.session_state["caps_delta_constraints"] = caps.get("delta_constraints", {})
        st.session_state["caps_monotonicity"] = caps.get("monotonicity", {})
        st.session_state["caps_diminishing_return_thresholds"] = caps.get("diminishing_return_thresholds", {})
        if "validation" in caps:
            st.session_state["caps_validation"] = caps["validation"]

    st.toast(f"Loaded **{name}** template")


def load_from_yaml(path: str) -> None:
    """Import an existing ``project.yml`` (and its referenced DAG / physics files)."""
    p = Path(path).resolve()
    if not p.exists():
        st.error(f"File not found: {p}")
        return
    with open(p) as f:
        cfg = yaml.safe_load(f)
    project_dir = p.parent
    _apply_yaml_to_state(cfg, project_dir)
    st.session_state["working_dir"] = str(project_dir)

    # Load DAG if referenced
    dag_path_str = cfg.get("causal", {}).get("dag_file", "")
    if dag_path_str:
        dag_path = (project_dir / dag_path_str).resolve()
        if dag_path.exists():
            with open(dag_path) as f:
                dag = yaml.safe_load(f) or {}
            st.session_state["dag_nodes"] = dag.get("nodes", [])
            st.session_state["dag_edges"] = dag.get("edges", [])

    # Load priors
    priors_path_str = cfg.get("physics", {}).get("priors_file", "")
    if priors_path_str:
        priors_path = (project_dir / priors_path_str).resolve()
        if priors_path.exists():
            with open(priors_path) as f:
                priors = yaml.safe_load(f) or {}
            st.session_state["priors"] = priors.get("coefficients", {})
            cal = priors.get("calibration", {})
            if cal:
                st.session_state["calibration_shrinkage_weight"] = cal.get("shrinkage_weight", 0.7)
                st.session_state["calibration_max_deviation"] = cal.get("max_deviation", 0.5)

    # Load caps
    caps_path_str = cfg.get("physics", {}).get("caps_file", "")
    if caps_path_str:
        caps_path = (project_dir / caps_path_str).resolve()
        if caps_path.exists():
            with open(caps_path) as f:
                caps = yaml.safe_load(f) or {}
            st.session_state["caps_variable_bounds"] = caps.get("variable_bounds", {})
            st.session_state["caps_delta_constraints"] = caps.get("delta_constraints", {})
            st.session_state["caps_monotonicity"] = caps.get("monotonicity", {})
            st.session_state["caps_diminishing_return_thresholds"] = caps.get("diminishing_return_thresholds", {})
            if "validation" in caps:
                st.session_state["caps_validation"] = caps["validation"]

    st.toast("Imported project configuration")


# ─── private helpers ──────────────────────────────────────────────────────────

def _apply_yaml_to_state(cfg: dict, base_dir: Path) -> None:
    """Map a parsed project.yml dict into session-state keys."""
    proj = cfg.get("project", {})
    st.session_state["project_name"]        = proj.get("name", "")
    st.session_state["project_description"] = proj.get("description", "")
    st.session_state["project_domain"]      = proj.get("domain", "custom")
    st.session_state["project_version"]     = proj.get("version", "1.0.0")
    st.session_state["project_author"]      = proj.get("author", "")
    st.session_state["response_units"]      = proj.get("response_units", "")

    data = cfg.get("data", {})
    st.session_state["data_file_path"]       = data.get("file_path", "")
    st.session_state["target_column"]        = data.get("target_column", "")
    st.session_state["identifier_column"]    = data.get("identifier_column", "")
    coords = data.get("coord_columns", ["", ""])
    st.session_state["coord_x"] = coords[0] if coords else ""
    st.session_state["coord_y"] = coords[1] if len(coords) > 1 else ""
    st.session_state["areas_of_interest_file"] = data.get("areas_of_interest_file", "")

    crs = cfg.get("crs", {})
    st.session_state["crs_input"]     = crs.get("input", "")
    st.session_state["crs_projected"] = crs.get("projected", "")

    st.session_state["predictors"] = cfg.get("predictors", [])

    phys = cfg.get("physics", {})
    st.session_state["monotone_constraints"]  = phys.get("monotone_constraints", {})
    st.session_state["physics_priors_file"]   = phys.get("priors_file", "")
    st.session_state["physics_caps_file"]     = phys.get("caps_file", "")

    causal = cfg.get("causal", {})
    st.session_state["causal_estimator"]      = causal.get("estimator", "hgb")
    st.session_state["causal_estimate_cate"]  = causal.get("estimate_cate", False)
    st.session_state["actionable_variables"]  = causal.get("actionable_variables", [])
    st.session_state["fixed_variables"]       = causal.get("fixed_variables", [])

    pipe = cfg.get("pipeline", {})
    st.session_state["random_seed"]         = pipe.get("random_seed", 42)
    st.session_state["n_spatial_folds"]     = pipe.get("n_spatial_folds", 5)
    st.session_state["fast_mode"]           = pipe.get("fast_mode", False)
    st.session_state["overwrite_outputs"]   = pipe.get("overwrite_outputs", False)
    st.session_state["run_mc_uncertainty"]  = pipe.get("run_mc_uncertainty", False)
    st.session_state["n_mc_draws"]          = pipe.get("n_mc_draws", 50)

    flags = cfg.get("flags", {})
    st.session_state["use_gwen_selection"]            = flags.get("use_gwen_selection", True)
    st.session_state["include_gwrf_in_ensemble"]      = flags.get("include_gwrf_in_ensemble", True)
    st.session_state["use_laplacian_eigenmaps_in_ols"] = flags.get("use_laplacian_eigenmaps_in_ols", False)

    # Model hyperparams
    models = cfg.get("models", {})
    gwr = models.get("gwr", {})
    st.session_state["gwr_bandwidth"]   = gwr.get("bandwidth")
    st.session_state["gwr_kernel"]      = gwr.get("kernel", "gaussian")
    st.session_state["gwr_alpha"]       = gwr.get("alpha", 0.1)
    st.session_state["gwr_min_points"]  = gwr.get("min_points", 50)

    gwrf = models.get("gwrf", {})
    st.session_state["gwrf_n_estimators"]    = gwrf.get("n_estimators", 100)
    st.session_state["gwrf_k_neighbors"]     = gwrf.get("k_neighbors", 100)
    st.session_state["gwrf_min_samples_leaf"] = gwrf.get("min_samples_leaf", 5)
    st.session_state["gwrf_n_jobs"]          = gwrf.get("n_jobs", 1)

    gam = models.get("ggpgam", {})
    st.session_state["ggpgam_n_splines"]       = gam.get("n_splines", 5)
    st.session_state["ggpgam_n_spatial_bases"]  = gam.get("n_spatial_bases", 10)
    st.session_state["ggpgam_lam"]              = gam.get("lam", 0.6)
    st.session_state["ggpgam_max_iter"]         = gam.get("max_iter", 100)

    meta = models.get("meta_ensemble", {})
    st.session_state["meta_algorithm"]            = meta.get("algorithm", "lightgbm")
    st.session_state["meta_n_optuna_trials"]      = meta.get("n_optuna_trials", 25)
    st.session_state["meta_include_base_features"] = meta.get("include_base_features", True)
    st.session_state["meta_include_laplacian_pca"] = meta.get("include_laplacian_pca", True)

    dk = models.get("deep_kriging", {})
    st.session_state["dk_hidden_layers"]  = dk.get("hidden_layers", [64, 32, 16])
    st.session_state["dk_dropout_rate"]   = dk.get("dropout_rate", 0.2)
    st.session_state["dk_epochs"]         = dk.get("epochs", 100)
    st.session_state["dk_learning_rate"]  = dk.get("learning_rate", 0.001)
    st.session_state["dk_batch_size"]     = dk.get("batch_size", 256)

    cv = models.get("spatial_cv", {})
    st.session_state["cv_block_size"]        = cv.get("block_size", 300)
    st.session_state["cv_buffer_size"]       = cv.get("buffer_size", 300)
    st.session_state["cv_block_size_source"] = cv.get("block_size_source", "user")
    st.session_state["cv_method"]            = cv.get("method", "block")
    st.session_state["cv_stratify_y"]        = cv.get("stratify_y", True)

    gwen = cfg.get("gwen", {})
    st.session_state["gwen_sample_size"]          = gwen.get("sample_size", 5000)
    st.session_state["gwen_k_neighbors"]          = gwen.get("k_neighbors", 500)
    st.session_state["gwen_cv_folds"]             = gwen.get("cv_folds", 5)
    st.session_state["gwen_selection_threshold"]  = gwen.get("selection_threshold", 0.1)

    lap = cfg.get("laplacian", {})
    st.session_state["laplacian_n_eigenmaps"] = lap.get("n_eigenmaps", 150)
    st.session_state["laplacian_k_for_swm"]   = lap.get("k_for_swm", 10)

    st.session_state["scenarios"]       = cfg.get("scenarios", [])
    st.session_state["joint_scenarios"] = cfg.get("joint_scenarios", [])

    out = cfg.get("output", {})
    st.session_state["output_base_dir"] = out.get("base_dir", "output")
    sdirs = out.get("stage_dirs", {})
    st.session_state["stage_dir_1"]     = sdirs.get("stage_1", "Stage_1_Correlogram_Analysis")
    st.session_state["stage_dir_2"]     = sdirs.get("stage_2", "Stage_2_Spatial_CV")
    st.session_state["stage_dir_3"]     = sdirs.get("stage_3", "Stage_3_Causal_Validation")
    st.session_state["stage_dir_4"]     = sdirs.get("stage_4", "Stage_4_Scenarios")
    st.session_state["stage_dir_final"] = sdirs.get("final", "Final_Interpretation_Results")
