"""
YAML Generator — converts Streamlit session state into the SPARC config files.

Writes: project.yml, causal/dag.yml, physics/priors.yml, physics/caps.yml
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml


def generate_all(output_dir: str) -> Path:
    """Write every config file and return the ``project.yml`` path."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    _write_project_yml(out)
    _write_dag_yml(out)
    _write_priors_yml(out)
    _write_caps_yml(out)

    return out / "project.yml"


# ─── project.yml ──────────────────────────────────────────────────────────────

def _write_project_yml(out: Path) -> None:
    s = st.session_state

    cfg: dict = {
        "project": {
            "name": s["project_name"],
            "description": s["project_description"],
            "domain": s["project_domain"],
            "version": s["project_version"],
            "author": s["project_author"],
            "response_units": s["response_units"],
        },
        "data": {
            "file_path": s["data_file_path"],
            "target_column": s["target_column"],
            "identifier_column": s["identifier_column"],
            "coord_columns": [s["coord_x"], s["coord_y"]],
        },
        "crs": {
            "input": s["crs_input"],
            "projected": s["crs_projected"],
        },
        "predictors": s["predictors"],
        "physics": {
            "monotone_constraints": s["monotone_constraints"],
        },
        "output": {
            "base_dir": s["output_base_dir"],
            "stage_dirs": {
                "stage_1": s["stage_dir_1"],
                "stage_2": s["stage_dir_2"],
                "stage_3": s["stage_dir_3"],
                "stage_4": s["stage_dir_4"],
                "final": s["stage_dir_final"],
            },
        },
        "pipeline": {
            "random_seed": s["random_seed"],
            "n_spatial_folds": s["n_spatial_folds"],
            "fast_mode": s["fast_mode"],
            "overwrite_outputs": s["overwrite_outputs"],
            "run_mc_uncertainty": s["run_mc_uncertainty"],
            "n_mc_draws": s["n_mc_draws"],
        },
        "flags": {
            "use_gwen_selection": s["use_gwen_selection"],
            "include_gwrf_in_ensemble": s["include_gwrf_in_ensemble"],
            "use_laplacian_eigenmaps_in_ols": s["use_laplacian_eigenmaps_in_ols"],
        },
        "models": {
            "gwr": {
                "bandwidth": s["gwr_bandwidth"],
                "kernel": s["gwr_kernel"],
                "alpha": s["gwr_alpha"],
                "min_points": s["gwr_min_points"],
            },
            "gwrf": {
                "n_estimators": s["gwrf_n_estimators"],
                "k_neighbors": s["gwrf_k_neighbors"],
                "min_samples_leaf": s["gwrf_min_samples_leaf"],
                "n_jobs": s["gwrf_n_jobs"],
            },
            "ggpgam": {
                "n_splines": s["ggpgam_n_splines"],
                "n_spatial_bases": s["ggpgam_n_spatial_bases"],
                "lam": s["ggpgam_lam"],
                "max_iter": s["ggpgam_max_iter"],
            },
            "meta_ensemble": {
                "algorithm": s["meta_algorithm"],
                "n_optuna_trials": s["meta_n_optuna_trials"],
                "include_base_features": s["meta_include_base_features"],
                "include_laplacian_pca": s["meta_include_laplacian_pca"],
            },
            "deep_kriging": {
                "hidden_layers": s["dk_hidden_layers"],
                "dropout_rate": s["dk_dropout_rate"],
                "epochs": s["dk_epochs"],
                "learning_rate": s["dk_learning_rate"],
                "batch_size": s["dk_batch_size"],
            },
            "spatial_cv": {
                "block_size": s["cv_block_size"],
                "buffer_size": s["cv_buffer_size"],
                "block_size_source": s["cv_block_size_source"],
                "method": s["cv_method"],
                "stratify_y": s["cv_stratify_y"],
            },
        },
        "gwen": {
            "sample_size": s["gwen_sample_size"],
            "k_neighbors": s["gwen_k_neighbors"],
            "cv_folds": s["gwen_cv_folds"],
            "selection_threshold": s["gwen_selection_threshold"],
        },
        "laplacian": {
            "n_eigenmaps": s["laplacian_n_eigenmaps"],
            "k_for_swm": s["laplacian_k_for_swm"],
        },
    }

    # Optional: areas of interest
    if s["areas_of_interest_file"]:
        cfg["data"]["areas_of_interest_file"] = s["areas_of_interest_file"]

    # Physics file references
    if s["priors"]:
        cfg["physics"]["priors_file"] = "physics/priors.yml"
    if s["caps_variable_bounds"] or s["caps_delta_constraints"]:
        cfg["physics"]["caps_file"] = "physics/caps.yml"

    # Causal section (only if DAG nodes defined)
    if s["dag_nodes"]:
        cfg["causal"] = {
            "dag_file": "causal/dag.yml",
            "estimator": s["causal_estimator"],
            "estimate_cate": s["causal_estimate_cate"],
        }
        if s["actionable_variables"]:
            cfg["causal"]["actionable_variables"] = s["actionable_variables"]
        if s["fixed_variables"]:
            cfg["causal"]["fixed_variables"] = s["fixed_variables"]

    # Scenarios
    if s["scenarios"]:
        cfg["scenarios"] = s["scenarios"]
    if s["joint_scenarios"]:
        cfg["joint_scenarios"] = s["joint_scenarios"]

    _dump(out / "project.yml", cfg)


# ─── causal/dag.yml ──────────────────────────────────────────────────────────

def _write_dag_yml(out: Path) -> None:
    nodes = st.session_state["dag_nodes"]
    edges = st.session_state["dag_edges"]
    if not nodes:
        return

    dag = {
        "nodes": nodes,
        "edges": edges,
        "structural_equations": {},
    }

    dag_dir = out / "causal"
    dag_dir.mkdir(parents=True, exist_ok=True)
    _dump(dag_dir / "dag.yml", dag)


# ─── physics/priors.yml ──────────────────────────────────────────────────────

def _write_priors_yml(out: Path) -> None:
    priors = st.session_state["priors"]
    if not priors:
        return

    s = st.session_state
    data = {
        "coefficients": priors,
        "scale_harmonization": {"target_resolution": 30, "factors": {}},
        "calibration": {
            "shrinkage_weight": s["calibration_shrinkage_weight"],
            "max_deviation": s["calibration_max_deviation"],
            "adaptive_weighting": s["calibration_adaptive_weighting"],
        },
        "sensitivity": {
            "tornado_uncertainty": 0.20,
            "n_samples": 1000,
        },
    }

    phys_dir = out / "physics"
    phys_dir.mkdir(parents=True, exist_ok=True)
    _dump(phys_dir / "priors.yml", data)


# ─── physics/caps.yml ────────────────────────────────────────────────────────

def _write_caps_yml(out: Path) -> None:
    s = st.session_state
    has_caps = s["caps_variable_bounds"] or s["caps_delta_constraints"]
    if not has_caps:
        return

    data = {
        "variable_bounds": s["caps_variable_bounds"],
        "delta_constraints": s["caps_delta_constraints"],
        "monotonicity": s["caps_monotonicity"],
        "capacity_rules": {},
        "combined_constraints": {},
        "validation": s["caps_validation"],
        "diminishing_return_thresholds": s["caps_diminishing_return_thresholds"],
        "sign_violation_strategy": "dampen",
        "sign_violation_dampen_factor": 0.50,
        "mgwr_reliability_threshold": 0.01,
    }

    phys_dir = out / "physics"
    phys_dir.mkdir(parents=True, exist_ok=True)
    _dump(phys_dir / "caps.yml", data)


# ─── helper ──────────────────────────────────────────────────────────────────

def _dump(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
