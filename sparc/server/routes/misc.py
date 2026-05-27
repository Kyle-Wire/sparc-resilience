"""sparc.server.routes.misc — miscellaneous singleton routes.

Covers: shutdown, debug/paths, context/layers, gwen approval gate,
insights/headline aggregator, and panels/availability.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from sparc.server import deps

router = APIRouter()

state = deps.state


# ---------------------------------------------------------------------------
# Panel availability helpers (used only by GET /panels/availability)
# ---------------------------------------------------------------------------

_PANEL_SPECS: dict[str, dict] = {
    "overview":            {"stage": "0", "specs": [("0", "correlogram_results")]},
    "headline":            {"stage": "2", "specs": [("2", "ensemble_results")]},
    "model_performance":   {"stage": "2", "specs": [("2", "ensemble_results")]},
    "dataset_profile":     {"stage": "0", "specs": [("2", "dataset_profile")]},
    "correlogram":         {"stage": "0", "specs": [("0", "correlogram_results")]},
    "kernel_field":        {"stage": "0", "specs": [
        ("1", "kernel_field"),
        ("0", "kernel_field"),
        ("0", "cross_correlogram_kernel_field"),
    ]},
    "predictions_map":     {"stage": "2", "specs": [("2", "spatial_cv_predictions")]},
    "pdp":                 {"stage": "2", "specs": [
        ("3", "dose_response_curves"),
        ("2", "gwrf_condition_curves"),
        ("2", "v2_neural_pdp::*"),
    ]},
    "dose_response":       {"stage": "3", "specs": [("3", "dose_response_curves")]},
    "cate":                {"stage": "3", "specs": [("3", "cate_summary")]},
    "divergence":          {"stage": "3", "specs": [("3", "cate_vs_gwr_divergence")]},
    "sensitivity":         {"stage": "3", "specs": [
        ("3", "scenario_coefficients"),
        ("3", "causal_diagnostics"),
    ]},
    "negative_control":    {"stage": "3", "specs": [("3", "cate_summary")]},
    "scenario_strip":      {"stage": "4", "specs": [("4", "scenario_results")]},
    "scenario_map":        {"stage": "4", "specs": [("4", "scenario_results")]},
    "scenario_uncertainty":{"stage": "4", "specs": [("4", "scenario_results")]},
    "scenario_trajectory": {"stage": "4", "specs": [("4", "scenario_results")]},
    "equity_cost":         {"stage": "4", "specs": [("4", "scenario_results")]},
    "artifact_browser":    {"stage": "0", "specs": [("0", "correlogram_results")]},
}

_STAGE_HINTS: dict[str, str] = {
    "0": "Run Stage 0 (Correlogram) to populate this panel.",
    "1": "Run Stage 1 (GWEN) to populate this panel.",
    "2": "Run Stage 2 (Spatial CV) to populate this panel.",
    "3": "Run Stage 3 (Causal Validation) to populate this panel.",
    "4": "Run Stage 4 (Scenarios) to populate this panel.",
}

_STAGE_LABELS: dict[str, str] = {
    "0": "STAGE 0 · CORRELOGRAM",
    "1": "STAGE 1 · GWEN",
    "2": "STAGE 2 · SPATIAL CV",
    "3": "STAGE 3 · CAUSAL",
    "4": "STAGE 4 · SCENARIOS",
}


def _spec_matches(spec: tuple[str, str], db_artifacts: set[tuple[str, str]],
                  prefix_index: dict[tuple[str, str], bool]) -> bool:
    stage_id, art_id = spec
    if art_id.endswith("::*"):
        prefix = art_id[: -1]  # keep '::' but drop the trailing '*'
        return prefix_index.get((stage_id, prefix), False)
    return (stage_id, art_id) in db_artifacts


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/shutdown", tags=["server"])
async def shutdown(request: Request):
    client_host = request.client.host if request.client else None
    if client_host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(status_code=403, detail="shutdown is localhost-only")

    import signal

    async def _terminate_soon() -> None:
        await asyncio.sleep(0.1)
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except Exception as exc:  # pragma: no cover
            print(f"Failed to deliver SIGINT during /shutdown: {exc}")

    asyncio.create_task(_terminate_soon())
    return {"status": "shutting down"}


@router.get("/debug/paths", tags=["debug"])
async def debug_paths():
    """Diagnostic endpoint: show resolved output paths and whether expected files exist."""
    if state.project_config is None:
        return {"error": "No project loaded"}

    from pathlib import Path
    from sparc.run.pipeline_paths import PipelinePaths

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception as exc:
        return {"error": f"Cannot resolve paths: {exc}"}

    expected_files = {
        "stage1_gwen_csv": str(paths.stage1_dir / "gwen_variable_importance.csv"),
        "stage1_gwen_json": str(paths.stage1_dir / "gwen_results.json"),
        "stage2_predictions_gpkg": str(paths.stage2_dir / "spatial_cv_predictions.gpkg"),
        "stage3_coefficients": str(paths.stage3_dir / "scenario_coefficients.json"),
        "stage3_dose_response": str(paths.stage3_dir / "dose_response_curves.json"),
        "stage4_scenario_gpkg": str(paths.stage4_dir / "scenario_results.gpkg"),
    }

    return {
        "output_dir": str(paths.output_dir),
        "stage1_dir": str(paths.stage1_dir),
        "stage2_dir": str(paths.stage2_dir),
        "stage3_dir": str(paths.stage3_dir),
        "stage4_dir": str(paths.stage4_dir),
        "final_dir": str(paths.final_dir),
        "files": {k: {"path": v, "exists": Path(v).exists()} for k, v in expected_files.items()},
        "config_output_base_dir": state.project_config.get("output", {}).get("base_dir"),
        "config_paths_output_dir": state.project_config.get("paths", {}).get("output_dir"),
    }


@router.get("/context/layers", tags=["context"])
async def get_context_layers(domain: str | None = Query(default=None)):
    """Return the basemap+overlay tile catalog, optionally filtered by *domain*."""
    from sparc.server.context_layers import get_catalog
    if domain is None and state.project_config is not None:
        proj = state.project_config.get("project") or {}
        domain = proj.get("domain")
    return get_catalog(domain)


# ------------------------------------------------------------------
# GWEN approval gate
# ------------------------------------------------------------------

@router.get("/gwen/status", tags=["gwen"])
async def get_gwen_status():
    """Return GWEN approval state and variable importance rows."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths

    paths = PipelinePaths.from_config(state.project_config)
    approval_path = paths.gwen_approved
    approved = approval_path.exists()
    stage1_complete = False
    rows = None

    try:
        if state.registry is not None:
            from sparc.registry.store import ArtifactStore
            store = ArtifactStore(state.registry)
            stage1_complete = store.has("1", "gwen_variable_importance") or store.has("1", "gwen_results")
            if store.has("1", "gwen_variable_importance"):
                df = store.read_any("1", "gwen_variable_importance")
                rows = df.to_dict(orient="records") if df is not None else None
            elif store.has("1", "gwen_results"):
                r = store.read_any("1", "gwen_results")
                rows = r.get("rows") if isinstance(r, dict) else None
    except Exception:
        pass

    return {
        "approved": approved,
        "approval_path": str(approval_path),
        "stage1_complete": stage1_complete,
        "rows": rows,
    }


@router.post("/gwen/approve", tags=["gwen"])
async def post_gwen_approve():
    """Write the ``gwen_approved.txt`` sentinel so the pipeline can proceed to Stage 2."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    import time
    from sparc.run.pipeline_paths import PipelinePaths

    paths = PipelinePaths.from_config(state.project_config)
    approval_path = paths.gwen_approved
    approval_path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    approval_path.write_text(f"approved at {ts}\n", encoding="utf-8")
    return {"approved": True, "approval_path": str(approval_path), "approved_at": ts}


@router.delete("/gwen/approve", tags=["gwen"])
async def delete_gwen_approve():
    """Revoke approval by removing the ``gwen_approved.txt`` sentinel."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths

    paths = PipelinePaths.from_config(state.project_config)
    approval_path = paths.gwen_approved
    if approval_path.exists():
        approval_path.unlink()
    return {"approved": False, "approval_path": str(approval_path)}


# ------------------------------------------------------------------
# Insights headline aggregator
# ------------------------------------------------------------------

@router.get("/insights/headline", tags=["insights"])
async def get_insights_headline():
    """Return a single 'headline' payload for the Insights page.

    Shape::

        {
          "best": InterventionCandidate | null,
          "alternatives": int,
          "score": float | null,
          "sensitivity": {"treatment": str | null, "e_value": float | null, "robust": bool} | null,
          "n_treatments": int,
          "n_outcomes": int,
          "warnings": [str, ...],
        }
    """
    from pathlib import Path

    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    best: dict | None = None
    score: float | None = None
    n_alt = 0
    warnings: list[str] = []

    # 1) Decision candidates → "best"
    try:
        from sparc.server.routes.decision import get_decision_candidates as _get_candidates
        cand_payload = await _get_candidates()
        candidates = (cand_payload or {}).get("candidates", []) if isinstance(cand_payload, dict) else []
    except HTTPException as exc:
        candidates = []
        warnings.append(f"decision: {exc.detail}")

    if candidates:
        scored: list[tuple[float, dict]] = []
        for c in candidates:
            try:
                eff = float(c.get("mean_effect"))
            except (TypeError, ValueError):
                continue
            cost = abs(float(c.get("cost", 1.0) or 1.0)) or 1.0
            scored.append((eff / cost, c))
        if scored:
            scored.sort(key=lambda p: p[0], reverse=True)
            score, best = scored[0]
            n_alt = max(0, len(candidates) - 1)

    # 2) Strongest sensitivity (E-value) across reported effects
    sensitivity_top: dict | None = None
    try:
        from sparc.server.routes.results_extended import get_causal_sensitivity as _get_sensitivity
        sens = await _get_sensitivity()
        effects = []
        if isinstance(sens, dict):
            effects = sens.get("effects") or sens.get("rows") or []
        if not isinstance(effects, list):
            effects = []
        best_e = None
        for row in effects:
            if not isinstance(row, dict):
                continue
            try:
                ev = float(row.get("e_value"))
            except (TypeError, ValueError):
                continue
            if best_e is None or ev > best_e["e_value"]:
                best_e = {"treatment": row.get("treatment") or row.get("variable"), "e_value": ev}
        if best_e is not None:
            best_e["robust"] = best_e["e_value"] >= 2.0
            sensitivity_top = best_e
    except HTTPException as exc:
        warnings.append(f"sensitivity: {exc.detail}")

    # 3) Cheap DAG structure counts
    n_treatments = 0
    n_outcomes = 0
    try:
        dag_file = state.project_config.get("causal", {}).get("dag_file")
        if dag_file and Path(dag_file).exists():
            from sparc.causal.dag_definition import load_dag, dag_to_networkx, get_node_roles
            dag = load_dag(dag_file)
            G = dag_to_networkx(dag)
            roles = get_node_roles(G)
            n_treatments = len(roles.get("treatments", []))
            n_outcomes = len(roles.get("outcomes", []))
    except Exception as exc:
        warnings.append(f"dag: {exc}")

    return {
        "best": best,
        "alternatives": n_alt,
        "score": score,
        "sensitivity": sensitivity_top,
        "n_treatments": n_treatments,
        "n_outcomes": n_outcomes,
        "warnings": warnings,
    }


# ------------------------------------------------------------------
# Panel availability
# ------------------------------------------------------------------

@router.get("/panels/availability", tags=["panels"])
async def get_panels_availability():
    """Per-panel availability snapshot for the desktop insights workspace."""
    db_artifacts: set[tuple[str, str]] = set()
    prefix_index: dict[tuple[str, str], bool] = {}
    if state.registry is not None:
        try:
            man = state.registry.manifest
            for stage_obj in man.stages.values():
                stage_str = str(stage_obj.stage)
                for art in stage_obj.artifacts.values():
                    if art.partial:
                        continue
                    db_artifacts.add((stage_str, art.id))
                    sep = art.id.find("::")
                    if sep > 0:
                        prefix_index[(stage_str, art.id[: sep + 2])] = True
        except Exception:
            db_artifacts = set()
            prefix_index = {}

    is_running = bool(state.is_running)
    current_stage = (
        str(state.current_stage) if state.current_stage is not None else None
    )
    last_errors: dict[str, str] = getattr(state, "stage_errors", {}) or {}

    panels_out: dict[str, dict] = {}
    for panel_id, cfg in _PANEL_SPECS.items():
        stage = cfg["stage"]
        specs: list[tuple[str, str]] = list(cfg["specs"])
        matched = [s for s in specs if _spec_matches(s, db_artifacts, prefix_index)]
        missing = [f"{s[0]}:{s[1]}" for s in specs if s not in matched]

        if matched and not missing:
            status = "ready"
        elif matched:
            status = "partial"
        else:
            status = "awaiting"

        if stage in last_errors and not matched:
            status = "failed"
        if is_running and current_stage == stage and not matched:
            status = "running"

        panels_out[panel_id] = {
            "status": status,
            "stage": stage,
            "stage_label": _STAGE_LABELS.get(stage, f"STAGE {stage}"),
            "missing": missing,
            "matched": [f"{s[0]}:{s[1]}" for s in matched],
            "hint": (
                last_errors[stage] if status == "failed"
                else _STAGE_HINTS.get(stage, "Run the relevant stage.")
            ),
        }

    return {
        "panels": panels_out,
        "is_running": is_running,
        "current_stage": current_stage,
    }
