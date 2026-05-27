"""Scenario library and execution routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field

from sparc.server import deps

router = APIRouter(tags=["scenarios"])


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _scenario_library_dir() -> Path:
    state = deps.state
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")
    from sparc.run.pipeline_paths import PipelinePaths
    paths = PipelinePaths.from_config(state.project_config)
    return paths.output_dir


def _to_geojson(data: Any) -> dict:
    """Convert a GeoDataFrame to GeoJSON."""
    import geopandas as gpd

    if isinstance(data, gpd.GeoDataFrame):
        return data.__geo_interface__
    if isinstance(data, dict) and "spatial" in data:
        return data["spatial"].__geo_interface__
    raise HTTPException(400, "Data is not spatial; use format=json")


def _to_json(data: Any) -> Any:
    """Convert a DataFrame or dict to JSON-serializable form."""
    import pandas as pd
    import geopandas as gpd

    if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
        if isinstance(data, gpd.GeoDataFrame):
            data = pd.DataFrame(data.drop(columns="geometry"))
        return {"rows": data.to_dict(orient="records")}
    if isinstance(data, dict):
        return data
    return {"data": str(data)}


def _read_cate_column_from_store(variable: str, column: str = "cate_mean"):
    """Return per-cell values of *column* for *variable* from cate_summary, or None."""
    state = deps.state
    if state.registry is None:
        return None
    try:
        from sparc.registry.run_registry import set_active_registry
        from sparc.registry.store import ArtifactStore

        set_active_registry(state.registry)
        try:
            store = ArtifactStore(state.registry)
            if not store.has("3", "cate_summary"):
                return None
            df = store.read_table("3", "cate_summary")
            if df is None or "treatment" not in df.columns:
                return None
            sub = df[df["treatment"] == variable]
            if sub.empty or column not in sub.columns:
                return None
            sub = sub.sort_values("cell_id")
            return np.asarray(sub[column].to_numpy(), dtype=float)
        finally:
            set_active_registry(None)
    except Exception:
        return None


def _resolve_benefits(source: str, variable: Optional[str]) -> tuple[np.ndarray, str]:
    """Pull a per-cell benefit vector from the configured source (DB-only)."""
    state = deps.state

    if source == "uniform":
        if state.data is None:
            raise HTTPException(400, "Cannot determine cell count: project data not loaded.")
        n = len(state.data)
        return np.ones(n, dtype=float), "uniform unit benefits (1.0 per cell)"

    if not variable:
        raise HTTPException(400, f"benefit_source='{source}' requires `variable`.")

    if state.project_config is None:
        raise HTTPException(400, "No project loaded.")

    if source == "cate":
        vals = _read_cate_column_from_store(variable, "cate_mean")
        if vals is None:
            raise HTTPException(
                404,
                f"Bayesian CATE for '{variable}' not available — run Stage 3 "
                "with `causal.estimate_cate: true` and at least one treatment.",
            )
        return vals, f"Bayesian local coefficient β(s) for '{variable}'"

    raise HTTPException(400, f"Unknown benefit_source: {source}")


def _resolve_per_cell_column(column: Optional[str], n_cells: int, default: float) -> np.ndarray:
    """Read a per-cell vector from the active dataset's CSV, or fall back to default."""
    if column is None:
        return np.full(n_cells, default, dtype=float)
    state = deps.state
    if state.project_config is None:
        return np.full(n_cells, default, dtype=float)
    cfg = state.project_config
    data_file = cfg.get("data", {}).get("file_path", cfg.get("paths", {}).get("raw_csv_path"))
    if not data_file:
        return np.full(n_cells, default, dtype=float)
    import pandas as pd
    df = pd.read_csv(data_file)
    if column not in df.columns:
        raise HTTPException(404, f"Column '{column}' not in dataset.")
    vals = df[column].to_numpy(dtype=float)[:n_cells]
    if len(vals) < n_cells:
        vals = np.concatenate([vals, np.full(n_cells - len(vals), default)])
    return vals


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RuntimeScenarioSpec(BaseModel):
    """A single user-defined scenario submitted at runtime (no project.yml required)."""

    name: str = "Unnamed Scenario"
    interventions: dict[str, float] = Field(
        ...,
        description=(
            "Mapping of variable→signed magnitude. Positive = increase, "
            "negative = decrease. E.g. {'Pct_Canopy': 25, 'Pct_Impervious': -15}."
        ),
    )
    unit_costs: dict[str, float] = Field(
        default_factory=dict,
        description="$/unit-of-change per variable. E.g. {'Pct_Canopy': 500}.",
    )
    budget: Optional[float] = Field(None, gt=0, description="Dollar budget (optional).")
    equity_focus: float = Field(
        0.0, ge=0.0, le=1.0,
        description="0=pure ROI, 1=equity-weighted ROI.",
    )

    class Config:
        extra = "ignore"


class RunScenariosBody(BaseModel):
    """Optional body for POST /scenarios/run."""

    scenarios: Optional[list[RuntimeScenarioSpec]] = None

    class Config:
        extra = "ignore"


class OptimizeScenarioBody(BaseModel):
    """Request body for POST /scenarios/optimize (single scenario)."""

    scenario: RuntimeScenarioSpec
    solver: Literal["greedy", "greedy_2opt", "milp", "auto"] = "auto"
    pareto_sweep: bool = True

    class Config:
        extra = "ignore"


class BudgetOptimizeRequest(BaseModel):
    """User-facing request for the budget optimizer.

    Source semantics
    ----------------
    `benefit_source` decides where per-cell unit benefits come from:

    - "cate"    — Bayesian per-cell coefficient β(s) for `variable`
    - "uniform" — every cell has unit benefit 1.0
    """

    budget: float = Field(..., gt=0, description="Total budget cap in cost units.")
    benefit_source: Literal["cate", "uniform"] = "cate"
    variable: Optional[str] = Field(
        None,
        description="Treatment variable (required for cate source).",
    )
    cost_column: Optional[str] = Field(
        None,
        description="Column in the active dataset holding per-cell unit cost.",
    )
    x_max_column: Optional[str] = Field(
        None,
        description="Column with per-cell maximum allocation (defaults to 1.0).",
    )
    solver: Literal["greedy", "greedy_2opt", "milp", "auto"] = "auto"
    pareto_sweep: bool = Field(
        True,
        description="Whether to additionally sweep the budget for the Pareto frontier.",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/scenarios/library")
async def get_scenario_library():
    """Return the scenario library (DB-first, with disk fallback)."""
    from sparc.scenario.library import build_timeline

    state = deps.state
    if state.registry is not None:
        from sparc.registry.store import ArtifactStore
        from sparc.scenario import SCENARIO_LIBRARY, SCENARIO_STAGE

        store = ArtifactStore(state.registry)
        if store.has(SCENARIO_STAGE, SCENARIO_LIBRARY):
            return store.read_struct(SCENARIO_STAGE, SCENARIO_LIBRARY)

    return build_timeline(_scenario_library_dir())


@router.post("/scenarios/library")
async def post_scenario_library(payload: dict = Body(...)):
    from sparc.scenario.library import append_entry
    from sparc.registry.provenance import compute_config_hash

    state = deps.state
    scenario = payload.get("scenario")
    if not isinstance(scenario, dict):
        raise HTTPException(400, "scenario (dict) required")
    cfg_hash = compute_config_hash(state.project_config or {}) if state.project_config else None
    entry = append_entry(
        _scenario_library_dir(),
        scenario,
        author=str(payload.get("author") or "anonymous"),
        comment=str(payload.get("comment") or ""),
        parent_id=payload.get("parent_id"),
        config_hash=cfg_hash,
    )
    return entry


@router.post("/scenarios/chain")
async def post_scenarios_chain(payload: dict = Body(...)):
    """Run a multi-step latent rollout through a sequence of interventions.

    Body::

        {
          "actions": [
            {"treatment": "Pct_Canopy", "delta_x": 10, "delta_t": 1.0},
            ...
          ],
          "mode": "latent"   // "latent" | "reencode"
        }

    Returns per-step mean delta and cumulative delta over the study area.
    """
    state = deps.state
    actions_raw = payload.get("actions", [])
    if not isinstance(actions_raw, list) or len(actions_raw) == 0:
        raise HTTPException(400, "'actions' list with at least one item required")
    mode = str(payload.get("mode", "latent"))

    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    try:
        import numpy as _np
        from sparc.run.pipeline_paths import PipelinePaths
        _paths = PipelinePaths.from_config(state.project_config)
        _v2_dir = _paths.stage2_dir / "v2_neural"
        _meta_path = _v2_dir / "meta_info.json"
        if not _meta_path.exists():
            raise HTTPException(424, "Stage 1 neural model not available — run Stage 1 first")

        import json as _json
        import torch as _torch
        from sparc.models.neural_meta import SPARCMetaLearner
        from sparc.models.process_rate_net import ProcessRateNet
        from sparc.models.latent_predictor import LatentPredictor
        from sparc.models.action_embedding import ActionEmbedding
        from sparc.inference.latent_rollout import multi_step_latent_rollout

        with open(_meta_path) as _mf:
            _mi = _json.load(_mf)

        _model = SPARCMetaLearner(
            n_base_models=_mi["n_base_models"],
            n_physics_features=_mi.get("n_physics_extended", _mi["n_physics_features"]),
            d_spatial=_mi["d_spatial"],
            hidden_dim=_mi["hidden_dim"],
            thresholds=_mi.get("thresholds", [0.25, 0.50, 0.75]),
        )
        _model.load_state_dict(
            _torch.load(_v2_dir / "neural_meta.pt", map_location="cpu", weights_only=True)
        )
        _model.eval()

        _process = ProcessRateNet(
            n_inputs=_mi.get("pr_input_dim", _mi["n_physics_features"])
        )
        _process.load_state_dict(
            _torch.load(_v2_dir / "process_rate_net.pt", map_location="cpu", weights_only=True)
        )
        _process.eval()

        _hidden_dim = _mi["hidden_dim"]
        _predictor = LatentPredictor(hidden_dim=_hidden_dim)
        _predictor_path = _v2_dir / "latent_predictor.pt"
        if _predictor_path.exists():
            _predictor.load_state_dict(
                _torch.load(_predictor_path, map_location="cpu", weights_only=True)
            )
        _predictor.eval()

        _treat_cols = (
            state.project_config.get("treatments", [])
            or state.project_config.get("predictors", [])
        )
        _action_embed = ActionEmbedding(
            n_treatments=max(len(_treat_cols), 1), hidden_dim=_hidden_dim
        )
        _ae_path = _v2_dir / "action_embedding.pt"
        if _ae_path.exists():
            _action_embed.load_state_dict(
                _torch.load(_ae_path, map_location="cpu", weights_only=True)
            )
        _action_embed.eval()

        _tensors_path = _v2_dir / "tensors.npz"
        if not _tensors_path.exists():
            raise HTTPException(424, "Physics tensors not found in v2_neural directory")
        _tensors = _np.load(str(_tensors_path), allow_pickle=False)
        _phys = _torch.tensor(_tensors["physics_feats"], dtype=_torch.float32)
        _base = _torch.tensor(
            _tensors.get("base_preds", _np.zeros((_phys.shape[0], 1))), dtype=_torch.float32
        )
        _spatial = _torch.tensor(
            _tensors.get("X_spatial", _np.zeros((_phys.shape[0], 2))), dtype=_torch.float32
        )
        _coords = _torch.tensor(
            _tensors.get("coords", _np.zeros((_phys.shape[0], 2))), dtype=_torch.float32
        )
        _knn = _torch.tensor(
            _tensors.get("knn_index", _np.zeros((_phys.shape[0], 8), dtype=_np.int64)),
            dtype=_torch.long,
        )

        _pr_col_idxs = _mi.get(
            "pr_col_idxs",
            list(
                range(
                    min(
                        _process.n_inputs if hasattr(_process, "n_inputs") else _phys.shape[1],
                        _phys.shape[1],
                    )
                )
            ),
        )
        with _torch.no_grad():
            _alpha = _process(_phys[:, _pr_col_idxs])

        _y_mean = float(_mi.get("y_mean", 0.0))
        _y_std = float(_mi.get("y_std", 1.0))

        actions = []
        for a in actions_raw:
            actions.append(
                (str(a["treatment"]), float(a.get("delta_x", 0.0)), float(a.get("delta_t", 1.0)))
            )

        result = multi_step_latent_rollout(
            model=_model,
            predictor=_predictor,
            action_embed=_action_embed,
            physics_feats=_phys,
            base_preds=_base,
            X_spatial=_spatial,
            coords=_coords,
            knn_index=_knn,
            alpha=_alpha,
            actions=actions,
            mode=mode,
            max_steps=10,
            y_mean=_y_mean,
            y_std=_y_std,
        )

        steps_out = []
        for i, step in enumerate(result.steps):
            steps_out.append(
                {
                    "step": i + 1,
                    "treatment": step.treatment,
                    "delta_x": step.delta_x,
                    "mean_delta": float(_np.mean(step.delta)),
                    "p5_delta": float(_np.percentile(step.delta, 5)),
                    "p95_delta": float(_np.percentile(step.delta, 95)),
                }
            )

        cumulative = float(_np.mean(result.final.delta)) if result.n_steps > 0 else 0.0
        return {
            "n_steps": result.n_steps,
            "steps": steps_out,
            "cumulative_mean_delta": cumulative,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Chain rollout failed: {exc}") from exc


@router.post("/scenarios/run")
async def run_scenarios(body: Optional[RunScenariosBody] = Body(default=None)):
    """Execute scenarios.

    Accepts an optional JSON body with a ``scenarios`` list. When provided,
    those runtime specs are used and the ``project.yml`` scenarios block is
    ignored. When no body is provided, the project config scenarios are used
    as before (backward compatible). Returns 400 if neither source has
    scenarios.
    """
    state = deps.state
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    runtime_specs: list[RuntimeScenarioSpec] = (body.scenarios or []) if body else []

    if runtime_specs:
        from sparc.scenario.spec_translator import InterventionSpec, ScenarioSpecTranslator
        _sim_cfg = ScenarioSpecTranslator.to_simulator_config(
            [InterventionSpec(name=s.name, interventions=s.interventions) for s in runtime_specs]
        )
        scenarios = _sim_cfg.scenarios
        interaction_scenarios = _sim_cfg.joint_scenarios
    else:
        scenarios = state.project_config.get("scenarios", [])
        interaction_scenarios = state.project_config.get("interaction_scenarios", [])

    if not scenarios and not interaction_scenarios:
        raise HTTPException(
            400,
            "No scenarios to run. Define scenarios in the Scenario Builder "
            "or add a 'scenarios:' block to project.yml.",
        )

    from sparc.interventions.scenario_simulator import ScenarioSimulator
    import pandas as pd

    config = dict(state.project_config)
    config["scenarios"] = scenarios
    config["interaction_scenarios"] = interaction_scenarios

    sim = ScenarioSimulator(config)
    sim.load_models()

    csv_path = config["paths"]["raw_csv_path"]
    data = pd.read_csv(csv_path)

    dag_file = config.get("causal", {}).get("dag_file")
    has_dag = dag_file and Path(dag_file).exists()
    requested_mode = config.get("pipeline", {}).get("scenario_mode", "auto")

    from sparc.run.scenario_engine_selector import ScenarioEngineSelector
    selector = ScenarioEngineSelector(config, sim, data)
    scenario_mode = selector.resolve_mode(requested_mode, has_dag)
    summary_df, results_gdf = selector.run(scenario_mode, has_dag=has_dag)

    conservation_violations = []
    try:
        from sparc.interventions.physics_priors import ConservationChecker
        checker = ConservationChecker()
        for scenario in scenarios:
            var = scenario["variable"]
            for inc in scenario.get("increments", []):
                direction = scenario.get("direction", "increase")
                delta_signed = -inc if direction == "decrease" else inc
                col_label = (
                    f"total_{var}_{'minus' if delta_signed < 0 else 'plus'}_"
                    f"{str(inc).replace('.', 'p')}"
                )
                if hasattr(results_gdf, "columns") and col_label in results_gdf.columns:
                    deltas = {var: np.full(len(data), delta_signed)}
                    target_deltas = results_gdf[col_label].values
                    violations = checker.check(
                        data, deltas, target_deltas=target_deltas, verbose=True
                    )
                    conservation_violations.extend(violations)
    except Exception as e:
        print(f"  [CONSERVATION] Check skipped ({e})")

    state.store_result(4, {"summary": summary_df, "spatial": results_gdf})

    return {
        "status": "complete",
        "n_scenarios": len(scenarios) + len(interaction_scenarios),
        "summary_rows": len(summary_df),
        "scenario_mode": scenario_mode,
        "conservation_violations": len(conservation_violations),
    }


@router.post("/scenarios/optimize")
async def optimize_scenario(body: OptimizeScenarioBody):
    """Run a scenario through the causal final function and return a budget-optimal allocation."""
    state = deps.state
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    spec = body.scenario
    if not spec.interventions:
        raise HTTPException(400, "scenario.interventions must not be empty")
    if spec.budget is None:
        raise HTTPException(400, "scenario.budget is required for optimization")

    import pandas as pd
    from sparc.interventions.scenario_simulator import ScenarioSimulator
    from sparc.scenario.budget import optimize as _budget_opt, pareto_sweep as _budget_sweep

    from sparc.scenario.spec_translator import InterventionSpec, ScenarioSpecTranslator
    _sim_cfg = ScenarioSpecTranslator.to_simulator_config(
        [InterventionSpec(name=spec.name, interventions=spec.interventions)]
    )
    scenarios = _sim_cfg.scenarios
    interaction_scenarios = _sim_cfg.joint_scenarios
    config = dict(state.project_config)
    config["scenarios"] = scenarios
    config["interaction_scenarios"] = interaction_scenarios

    sim = ScenarioSimulator(config)
    sim.load_models()

    csv_path = config["paths"]["raw_csv_path"]
    data = pd.read_csv(csv_path)
    n_cells = len(data)

    dag_file = config.get("causal", {}).get("dag_file")
    has_dag = bool(dag_file and Path(dag_file).exists())

    if has_dag:
        summary_df, results_gdf, *_ = sim.run_with_causal_dag(data, verbose=False)
    else:
        summary_df, results_gdf, *_ = sim.run(verbose=False)

    from sparc.scenario.benefit_surface import BenefitSurface
    delta_col: Optional[str] = None
    if results_gdf is not None and hasattr(results_gdf, "columns"):
        total_cols = [c for c in results_gdf.columns if c.startswith("total_")]
        if total_cols:
            delta_col = total_cols[0]

    if delta_col is None or results_gdf is None:
        raise HTTPException(
            500,
            "Scenario simulation produced no per-cell delta columns. "
            "Ensure Stage 3 has been run and CATE estimates are available.",
        )

    raw_deltas = np.asarray(results_gdf[delta_col].to_numpy(), dtype=float)
    raw_deltas = np.nan_to_num(raw_deltas, nan=0.0)

    equity_scores = np.ones(n_cells, dtype=float)
    if spec.equity_focus > 0:
        try:
            eq_store = None
            if state.registry is not None:
                from sparc.registry.store import ArtifactStore
                eq_store = ArtifactStore(state.registry)

            if eq_store is not None and eq_store.has("0", "equity_layer"):
                eq_df = eq_store.read_table("0", "equity_layer")
                if eq_df is not None and "equity_score" in eq_df.columns:
                    equity_scores = np.asarray(
                        eq_df["equity_score"].to_numpy(), dtype=float
                    )[:n_cells]
                    if len(equity_scores) < n_cells:
                        equity_scores = np.pad(
                            equity_scores,
                            (0, n_cells - len(equity_scores)),
                            constant_values=np.nanmean(equity_scores),
                        )
            else:
                from sparc.data.census_equity import get_per_cell_equity
                census_key = None
                try:
                    from sparc.config.user_preferences import load_preferences
                    prefs = load_preferences()
                    census_key = (prefs.get("api_keys") or {}).get("census")
                except Exception:
                    pass
                equity_scores = get_per_cell_equity(data, census_api_key=census_key)
        except Exception as _eq_err:
            import warnings as _w
            _w.warn(f"Equity layer unavailable: {_eq_err}. Using uniform equity.")

    alpha = float(np.clip(spec.equity_focus, 0.0, 1.0))
    total_unit_cost = sum(
        spec.unit_costs.get(var, 0.0) * abs(mag)
        for var, mag in spec.interventions.items()
    )
    if total_unit_cost <= 0:
        total_unit_cost = 1.0
        import warnings as _w
        _w.warn(
            "No unit_costs provided for scenario interventions. "
            "Using abstract cost=1 per cell."
        )
    surface = BenefitSurface.from_results_gdf(
        results_gdf, delta_col=delta_col, unit_cost=total_unit_cost
    ).with_equity(equity_scores, spec.equity_focus)

    alloc_result = _budget_opt(
        **surface.to_optimize_args(),
        budget=spec.budget,
        solver=body.solver,
    )
    allocation = np.asarray(alloc_result.allocation, dtype=float)

    import geopandas as gpd

    if isinstance(results_gdf, gpd.GeoDataFrame):
        out_gdf = results_gdf[["geometry"]].copy()
    else:
        raise HTTPException(500, "Scenario results are not a GeoDataFrame.")

    out_gdf = out_gdf.iloc[:n_cells].copy()
    out_gdf["allocation"] = allocation
    out_gdf["projected_delta"] = raw_deltas
    out_gdf["equity_score"] = equity_scores
    out_gdf["estimated_cost"] = allocation * surface.costs

    if out_gdf.crs is not None and str(out_gdf.crs) != "EPSG:4326":
        out_gdf = out_gdf.to_crs(epsg=4326)

    allocation_geojson = out_gdf.__geo_interface__

    pareto_out: Optional[dict] = None
    if body.pareto_sweep:
        try:
            sweep = _budget_sweep(
                **surface.to_optimize_args(),
                budget=spec.budget,
                solver=body.solver,
            )
            pareto_out = sweep.to_dict()
        except Exception as _pe:
            import warnings as _w
            _w.warn(f"Pareto sweep failed: {_pe}")

    from sparc.scenario.budget import _gini

    return {
        "status": "complete",
        "scenario_name": spec.name,
        "budget": spec.budget,
        "equity_focus": alpha,
        "allocation_geojson": allocation_geojson,
        "pareto": pareto_out,
        "summary": {
            "total_projected_delta": float(np.sum(np.abs(raw_deltas) * allocation)),
            "total_estimated_cost": float(alloc_result.total_cost),
            "n_cells_treated": int(alloc_result.n_treated),
            "n_cells_fully_treated": int(alloc_result.n_fully_treated),
            "allocation_gini": float(_gini(allocation)),
            "equity_gini": float(_gini(equity_scores * (allocation > 0).astype(float))),
            "solver": alloc_result.solver,
            "delta_column": delta_col,
        },
    }


@router.get("/scenarios/results")
async def scenario_results(format: str = Query("geojson", regex="^(json|geojson)$")):
    """Return scenario simulation results."""
    state = deps.state
    result = state.get_result(4)
    if result is None:
        raise HTTPException(404, "No scenario results. Run /scenarios/run first.")

    if format == "geojson" and "spatial" in result:
        return _to_geojson(result["spatial"])

    if "summary" in result:
        return _to_json(result["summary"])

    return _to_json(result)


@router.post("/scenarios/budget/optimize")
async def budget_optimize(req: BudgetOptimizeRequest):
    """Allocate `req.budget` across cells to maximise expected benefit."""
    from sparc.scenario.budget import optimize as _opt, pareto_sweep as _sweep

    benefits, benefit_desc = _resolve_benefits(req.benefit_source, req.variable)
    n_cells = int(benefits.size)
    if n_cells == 0:
        raise HTTPException(404, "No cells in benefit source.")

    costs = _resolve_per_cell_column(req.cost_column, n_cells, default=1.0)
    x_max = _resolve_per_cell_column(req.x_max_column, n_cells, default=1.0)

    result = _opt(
        benefits=benefits,
        budget=req.budget,
        costs=costs,
        x_max=x_max,
        solver=req.solver,
    )

    out: dict = {
        "result": result.to_dict(),
        "n_cells": n_cells,
        "benefit_source": req.benefit_source,
        "benefit_description": benefit_desc,
    }

    if req.pareto_sweep:
        sweep = _sweep(
            benefits=benefits,
            budget=req.budget,
            costs=costs,
            x_max=x_max,
            solver=req.solver,
        )
        out["pareto"] = sweep.to_dict()

    return out
