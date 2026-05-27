"""sparc.server.routes.results_extended — extracted /results/* route handlers.

All 37 inline /results/* handlers from app.py are served here.
Routes are declared in specificity order so FastAPI declaration-order
matching resolves correctly (specific before generic, and all named routes
before the parameterised /results/{stage} catch-all).
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

from sparc.server import deps

router = APIRouter(tags=["results"])


# ---------------------------------------------------------------------------
# Local helpers (moved from app.py — only used by routes in this module)
# ---------------------------------------------------------------------------

class BlockExportRequest(BaseModel):
    artifact_id: str
    stage: str = "export"
    label: Optional[str] = None
    png_b64: str  # data URL or raw base64

    class Config:
        extra = "ignore"


_RENDER_MIME = {
    "csv":     "text/csv",
    "json":    "application/json",
    "geojson": "application/geo+json",
    "html":    "text/html",
    "pkl":     "application/octet-stream",
    "joblib":  "application/octet-stream",
    "pt":      "application/octet-stream",
    "bin":     "application/octet-stream",
}


def _load_neural_pdp(paths) -> dict:
    """Read PINN neural-network PDP curves (DB-only)."""
    out: dict = {}
    if deps.state.registry is None:
        return out
    try:
        from sparc.registry.run_registry import set_active_registry
        from sparc.registry.store import ArtifactStore
        set_active_registry(deps.state.registry)
        try:
            _store = ArtifactStore(deps.state.registry)
            manifest_stages = getattr(deps.state.registry.manifest, "stages", {}) or {}
            stage2 = manifest_stages.get("2", {}) or {}
            for art_id in stage2.keys():
                if not art_id.startswith("v2_neural_pdp::"):
                    continue
                feat_col = art_id.split("::", 1)[1]
                try:
                    df = _store.read_any("2", art_id)
                except Exception:
                    continue
                if df is None or len(df) == 0 or feat_col not in df.columns:
                    continue
                grid = [float(v) for v in df[feat_col].tolist()]
                pdp_vals = [float(v) for v in df["mean_prediction"].tolist()]
                if {"q10", "q90"}.issubset(df.columns):
                    pdp_std = [
                        (float(q90) - float(q10)) / 2.56
                        for q10, q90 in zip(df["q10"].tolist(), df["q90"].tolist())
                    ]
                else:
                    pdp_std = None
                out[feat_col] = {
                    "grid_values": grid,
                    "pdp_values": pdp_vals,
                    "pdp_std": pdp_std,
                    "source": "neural_pde",
                }
        finally:
            set_active_registry(None)
    except Exception:
        pass
    return out


def _load_gwrf_pdp(paths) -> dict | None:
    """Load GWRF condition curves dict from artifacts.db (DB-only)."""
    if deps.state.registry is None:
        return None
    try:
        from sparc.registry.run_registry import set_active_registry
        from sparc.registry.store import ArtifactStore
        set_active_registry(deps.state.registry)
        try:
            _store = ArtifactStore(deps.state.registry)
            for stage_id, art_id in [
                ("2", "gwrf_condition_curves"),
                ("3", "gwrf_condition_curves"),
            ]:
                if _store.has(stage_id, art_id):
                    data = _store.read_any(stage_id, art_id)
                    if isinstance(data, dict):
                        for _var, curve in data.items():
                            if isinstance(curve, dict):
                                curve.setdefault("source", "gwrf")
                        return data
        finally:
            set_active_registry(None)
    except Exception:
        return None
    return None


def _read_cate_column_from_store(variable: str, column: str = "multiplier_mean"):
    """Return per-cell values of *column* for *variable* from cate_summary table."""
    if deps.state.registry is None:
        return None
    try:
        from sparc.registry.run_registry import set_active_registry
        from sparc.registry.store import ArtifactStore
        import numpy as np
        set_active_registry(deps.state.registry)
        try:
            store = ArtifactStore(deps.state.registry)
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


def _read_cate_multiplier_from_store(variable: str):
    """Back-compat wrapper: return multiplier_mean (used by scenario simulator)."""
    return _read_cate_column_from_store(variable, "multiplier_mean")


def _read_cate_coefficient_from_store(variable: str, *, with_ci: bool = False):
    """Return the Bayesian per-cell coefficient β(s) for *variable*."""
    mean = _read_cate_column_from_store(variable, "cate_mean")
    if mean is None:
        return None
    src = _read_cate_column_from_store(variable, "source")
    source_label = None
    if src is not None and len(src):
        try:
            source_label = str(src[0])
        except Exception:
            source_label = None
    out: dict[str, Any] = {"mean": mean, "source": source_label}
    if with_ci:
        out["ci5"] = _read_cate_column_from_store(variable, "cate_ci5")
        out["ci95"] = _read_cate_column_from_store(variable, "cate_ci95")
    return out


def _ensure_registry_attached() -> None:
    if deps.state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if deps.state.registry is None:
        from sparc.server.app import _attach_registry
        _attach_registry(deps.state.project_config)
    if deps.state.registry is None:
        raise HTTPException(503, "Run registry unavailable")


def _to_geojson(data: Any) -> dict:
    """Convert a GeoDataFrame or dict with spatial key to GeoJSON."""
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


# ---------------------------------------------------------------------------
# Routes — specificity order: named routes first, parameterised last
# ---------------------------------------------------------------------------

@router.post("/results/export")
async def export_block(req: BlockExportRequest):
    """Persist a screenshot of a UI block (map / chart / plot) to disk and the registry."""
    if deps.state.project_config is None:
        raise HTTPException(400, "No project loaded")

    import base64
    import time
    from sparc.run.pipeline_paths import PipelinePaths

    payload = req.png_b64
    if payload.startswith("data:"):
        try:
            payload = payload.split(",", 1)[1]
        except IndexError:
            raise HTTPException(400, "Malformed data URL in png_b64")
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise HTTPException(400, f"Invalid base64 PNG: {exc}")
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise HTTPException(400, "png_b64 does not contain a PNG signature")

    try:
        paths = PipelinePaths.from_config(deps.state.project_config)
    except Exception as exc:
        raise HTTPException(500, f"Cannot resolve paths: {exc}")

    exports_dir = paths.output_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    safe_label = "".join(c for c in (req.label or "") if c.isalnum() or c in "-_") or "snapshot"
    fname = f"{req.stage}__{req.artifact_id}__{safe_label}__{ts}.png"
    out_path = exports_dir / fname
    out_path.write_bytes(raw)

    entry_dict: dict[str, Any] = {
        "saved_to": str(out_path),
        "size_bytes": len(raw),
        "registered": False,
    }

    reg = deps.state.registry
    if reg is not None:
        try:
            entry = reg.register_artifact(
                stage=req.stage,
                artifact_id=f"export::{req.artifact_id}::{ts}",
                path=out_path,
                format="png",
                producer="desktop:export",
                consumers=["user:export", "report:figures"],
                metadata={
                    "exported_from": req.artifact_id,
                    "label": req.label or "",
                    "exported_at": ts,
                },
            )
            entry_dict["registered"] = True
            entry_dict["artifact"] = entry.model_dump()
        except Exception as exc:
            entry_dict["registry_error"] = str(exc)

    return entry_dict


@router.get("/results/kernel_field")
async def get_kernel_field_artifact():
    """Return the canonical KernelField artifact (Stage 0 → Stage 1)."""
    store = deps.get_open_store()
    try:
        for stage in ("1", "0"):
            if store.has(stage, "kernel_field"):
                payload = store.read_any(stage, "kernel_field")
                if isinstance(payload, dict):
                    payload.setdefault("source", "kernel_field")
                return payload
        if store.has("0", "cross_correlogram_kernel_field"):
            payload = store.read_any("0", "cross_correlogram_kernel_field")
            if isinstance(payload, dict):
                payload = dict(payload)
                payload["source"] = "cross_correlogram"
            return payload
    finally:
        from sparc.registry.run_registry import set_active_registry
        set_active_registry(None)
    raise deps.missing_artifact_response(
        artifact_id="kernel_field", stage="0/1",
        hint=(
            "No KernelField artifact written by Stages 0/1 and no "
            "cross_correlogram_kernel_field fallback found in Stage 0."
        ),
    )


@router.get("/results/gwen")
async def get_gwen_data():
    """Return GWEN variable importance as a row-oriented table."""
    store = deps.get_open_store()
    try:
        if store.has("1", "gwen_variable_importance"):
            df = store.read_any("1", "gwen_variable_importance")
            return {"rows": df.to_dict(orient="records")}
        if store.has("1", "gwen_results"):
            return store.read_any("1", "gwen_results")
    finally:
        from sparc.registry.run_registry import set_active_registry
        set_active_registry(None)
    raise deps.missing_artifact_response(
        artifact_id="gwen_variable_importance", stage="1",
        hint="Stage 1 (GWEN) has not produced gwen_variable_importance. Run Stage 1.",
    )


@router.get("/results/model_performance")
async def get_model_performance():
    """Return per-model R²/RMSE for the R² bar chart on the Results page."""
    cached = deps.state.result_cache.get("2", "model_performance_response")
    if cached is not None:
        return cached

    models: list[dict] = []

    r2_result = deps.state.get_result(2)
    if isinstance(r2_result, dict):
        indiv = (r2_result.get("performance") or {}).get("individual_models")
        if indiv is None:
            indiv = r2_result.get("base_models")
        if isinstance(indiv, dict):
            for name, metrics in indiv.items():
                if isinstance(metrics, dict) and metrics.get("r2") is not None:
                    models.append({
                        "name": name.upper(),
                        "r2": metrics["r2"],
                        "rmse": metrics.get("rmse"),
                    })
        ens = r2_result.get("final_ensemble") or r2_result.get("meta_ensemble_best")
        if isinstance(ens, dict) and ens.get("r2") is not None:
            models.append({
                "name": "Ensemble",
                "r2": ens["r2"],
                "rmse": ens.get("rmse"),
            })

    if not models:
        store = deps.get_open_store()
        try:
            if store.has("2", "ensemble_results"):
                data = store.read_any("2", "ensemble_results")
                base = data.get("base_models", {})
                for name, metrics in base.items():
                    if isinstance(metrics, dict) and metrics.get("r2") is not None:
                        models.append({
                            "name": name.upper(),
                            "r2": metrics["r2"],
                            "rmse": metrics.get("rmse"),
                        })
                ens = data.get("final_ensemble") or data.get("meta_ensemble_best")
                if isinstance(ens, dict) and ens.get("r2") is not None:
                    models.append({
                        "name": "Ensemble",
                        "r2": ens["r2"],
                        "rmse": ens.get("rmse"),
                    })
        finally:
            from sparc.registry.run_registry import set_active_registry
            set_active_registry(None)

    if not models:
        raise deps.missing_artifact_response(
            artifact_id="ensemble_results", stage="2",
            hint="Stage 2 has not produced ensemble_results. Run Stage 2.",
        )

    models.sort(key=lambda m: m.get("r2") or 0, reverse=True)
    result = {"models": models}
    deps.state.result_cache.set("2", "model_performance_response", result)
    return result


@router.get("/results/spatial_cv/predictions")
async def get_spatial_cv_predictions():
    """Return spatial CV predictions as GeoJSON (DB-only)."""
    cached = deps.state.result_cache.get("2", "spatial_cv_predictions_geojson")
    if cached is not None:
        return cached
    store = deps.get_open_store()
    try:
        if not store.has("2", "spatial_cv_predictions"):
            raise deps.missing_artifact_response(
                artifact_id="spatial_cv_predictions", stage="2",
                hint=(
                    "Stage 2 has not produced spatial_cv_predictions. "
                    "Run Stage 2 to populate the predictions map."
                ),
            )
        gdf = store.read_any("2", "spatial_cv_predictions")
    finally:
        from sparc.registry.run_registry import set_active_registry
        set_active_registry(None)

    import geopandas as gpd
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise HTTPException(
            500,
            "Artifact (2, spatial_cv_predictions) is not a GeoDataFrame; "
            "the producer must include a geometry column.",
        )
    if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
    unit_str = ""
    if "_unit" in gdf.columns and len(gdf) > 0:
        unit_str = str(gdf["_unit"].iloc[0])
    if not unit_str:
        raw = getattr(deps.state, "raw_project_yaml", None) or {}
        unit_str = (
            (raw.get("output") or {}).get("response_units")
            or (raw.get("project") or {}).get("response_units")
            or ""
        )
    geojson = dict(gdf.__geo_interface__)
    if unit_str:
        geojson["unit"] = unit_str
    deps.state.result_cache.set("2", "spatial_cv_predictions_geojson", geojson)
    return geojson


@router.get("/results/causal/cate_map/variables")
async def get_cate_variables():
    """Return list of variables that have spatial CATE multiplier maps."""
    if deps.state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths

    try:
        paths = PipelinePaths.from_config(deps.state.project_config)
    except Exception:
        raise HTTPException(404, "Cannot resolve output paths")

    variables: list[str] = []
    diagnostics: dict[str, Any] = {}

    if deps.state.registry is not None:
        try:
            from sparc.registry.run_registry import set_active_registry
            from sparc.registry.store import ArtifactStore
            set_active_registry(deps.state.registry)
            try:
                store = ArtifactStore(deps.state.registry)
                if store.has("3", "cate_summary"):
                    df = store.read_table("3", "cate_summary")
                    if df is not None and "treatment" in df.columns:
                        variables.extend(
                            sorted(str(t) for t in df["treatment"].unique())
                        )
                        diagnostics["cate_summary_hits"] = len(variables)
            finally:
                set_active_registry(None)
        except Exception as exc:
            diagnostics["cate_summary_error"] = str(exc)

    if not variables and deps.state.registry is not None:
        for art in deps.state.registry.list_for_stage("3"):
            if art.id.startswith("cate_multiplier::") and not art.partial:
                var = art.metadata.get("variable") or art.id.split("::", 1)[1]
                variables.append(var)
        diagnostics["registry_hits"] = len(variables)

    sorted_vars = sorted(set(variables))
    payload: dict[str, Any] = {"variables": sorted_vars}
    if not sorted_vars:
        causal_cfg = deps.state.project_config.get("causal", {}) or {}
        inference = (causal_cfg.get("inference") or "").lower()
        estimate_cate = causal_cfg.get("estimate_cate", True)
        reasons: list[str] = []
        if not estimate_cate:
            reasons.append("`causal.estimate_cate` is false in project.yml.")
        if inference == "bayesian":
            reasons.append(
                "Bayesian inference path historically skipped CATE; this is "
                "fixed in newer Stage 3 runs but the existing run was "
                "produced before the fix."
            )
        if not reasons:
            reasons.append(
                "Stage 3 finished but did not write spatial CATE multipliers. "
                "Check the DAG for at least one treatment node."
            )
        payload["empty_reason"] = " ".join(reasons)
        payload["next_action"] = (
            "Re-run Stage 3 with `causal.estimate_cate: true` and at least "
            "one treatment in your DAG."
        )
    if diagnostics:
        payload["diagnostics"] = diagnostics
    return payload


@router.get("/results/causal/cate_map")
async def get_cate_map(
    variable: str = Query(...),
    with_uncertainty: bool = Query(
        False,
        description=(
            "When true, also emit per-cell coef_ci5_<var> / coef_ci95_<var>"
            " properties so the frontend can stipple cells whose 90% CI crosses zero."
        ),
    ),
):
    """Return the Bayesian per-cell coefficient β(s) for *variable* as GeoJSON."""
    if deps.state.project_config is None:
        raise HTTPException(400, "No project loaded")

    payload = _read_cate_coefficient_from_store(variable, with_ci=with_uncertainty)
    if payload is None:
        raise deps.missing_artifact_response(
            artifact_id=f"cate_summary[{variable}]", stage="3",
            hint=(
                f"No CATE map for variable '{variable}'. Re-run Stage 3 with "
                "`causal.estimate_cate: true` and at least one treatment in your DAG."
            ),
        )

    if deps.state.data is None or not hasattr(deps.state.data, "geometry"):
        raise HTTPException(
            500,
            "Project data has no geometry; cannot project CATE coefficients "
            "onto the map. Re-load the project so the input dataset is "
            "parsed with coordinate columns.",
        )

    import geopandas as gpd

    src = deps.state.data
    coef = payload["mean"]
    n = min(len(src), len(coef))
    cols: dict[str, Any] = {
        f"coef_{variable}": coef[:n],
        f"cate_{variable}": coef[:n],  # back-compat alias
    }
    if with_uncertainty:
        ci5 = payload.get("ci5")
        ci95 = payload.get("ci95")
        if ci5 is not None and ci95 is not None:
            cols[f"coef_ci5_{variable}"] = ci5[:n]
            cols[f"coef_ci95_{variable}"] = ci95[:n]
            import numpy as _np
            cols[f"coef_ns_{variable}"] = (
                (_np.asarray(ci5[:n], dtype=float) <= 0.0)
                & (_np.asarray(ci95[:n], dtype=float) >= 0.0)
            )

    gdf = gpd.GeoDataFrame(cols, geometry=src.geometry.values[:n], crs=src.crs)
    if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
    fc = gdf.__geo_interface__
    fc["variable"] = variable
    fc["units"] = "ΔY per unit T (posterior mean of β(s))"
    fc["source"] = payload.get("source") or "bayesian_nuts"
    fc["with_uncertainty"] = bool(with_uncertainty)
    return fc


@router.get("/results/causal/dose_response")
async def get_dose_response():
    """Return dose-response curves (DB-only)."""
    return await deps.read_or_404(
        "3", "dose_response_curves",
        hint=(
            "Dose-response curves are produced by Stage 3 (Causal Validation). "
            "They are skipped when `causal.inference: bayesian`. Re-run Stage 3 "
            "in frequentist mode or check the Causal Diagnostics panel for the "
            "Bayesian posterior."
        ),
    )


@router.get("/results/causal/sensitivity")
async def get_causal_sensitivity():
    """Return E-values + tipping-point analysis for the current causal payload."""
    from sparc.causal.sensitivity import annotate_causal_payload

    payload = await get_causal_results()  # type: ignore[misc]
    if not isinstance(payload, dict):
        raise HTTPException(500, "Causal payload is not a dict")
    annotated = annotate_causal_payload(payload)
    sens = annotated.get("sensitivity") or {}
    return sens


@router.get("/results/causal/negative_control")
async def get_causal_negative_control(
    variable: str = Query(...),
    n_permutations: int = Query(1000, ge=50, le=10000),
):
    """Permutation negative-control test on the spatial CATE values (DB-only)."""
    from sparc.evaluation.negative_controls import permutation_test_cate

    arr_np = _read_cate_multiplier_from_store(variable)
    if arr_np is None:
        raise deps.missing_artifact_response(
            artifact_id=f"cate_summary[{variable}]", stage="3",
            hint=(
                f"No CATE map for variable '{variable}'. Re-run Stage 3 "
                "with `causal.estimate_cate: true`."
            ),
        )
    arr = arr_np.astype(float).tolist()
    res = permutation_test_cate(arr, n_permutations=n_permutations)
    return {
        "variable": variable,
        "n": res.n,
        "mean_observed": res.mean_observed,
        "mean_null": res.mean_null,
        "std_null": res.std_null,
        "p_value": res.p_value,
        "z_score": res.z_score,
        "n_permutations": res.n_permutations,
        "passed": res.passed,
        "interpretation": (
            "p > 0.05 — observed CATE indistinguishable from null (negative-control PASS)."
            if res.passed
            else f"p = {res.p_value:.4f} — CATE is significantly non-zero (real treatment effect)."
        ),
    }


@router.get("/results/causal/diagnostics")
async def get_causal_diagnostics():
    """Return CATE diagnostics (calibration, cumulative effects, RATE)."""
    return await deps.read_or_404(
        "3", "causal_diagnostics",
        hint="Stage 3 (Causal Validation) has not produced causal_diagnostics.",
    )


@router.get("/results/causal")
async def get_causal_results():
    """Return causal validation results (DB-only)."""
    store = deps.get_open_store()
    try:
        if store.has("3", "scenario_coefficients"):
            return store.read_any("3", "scenario_coefficients")
        if store.has("3", "causal_diagnostics"):
            return store.read_any("3", "causal_diagnostics")
    finally:
        from sparc.registry.run_registry import set_active_registry
        set_active_registry(None)

    mem_result = deps.state.get_result(3)
    if mem_result is not None:
        return mem_result

    raise deps.missing_artifact_response(
        artifact_id="scenario_coefficients", stage="3",
        hint=(
            "Stage 3 (Causal Validation) has not produced "
            "`scenario_coefficients`. Run Stage 3 to populate the causal panel."
        ),
    )


@router.get("/results/neural_pdp")
async def get_neural_pdp():
    """Return PINN-derived PDP curves (DB-only)."""
    curves = _load_neural_pdp(None)
    if not curves:
        raise deps.missing_artifact_response(
            artifact_id="v2_neural_pdp::*", stage="2",
            hint="Neural PDP curves not available — run Stage 2 (v2_neural training) first.",
        )
    return curves


@router.get("/results/pdp_curves")
async def get_pdp_curves():
    """Return partial dependence / condition curves from any available source (DB-only)."""
    if deps.state.project_config is None:
        raise HTTPException(400, "No project loaded")

    gwrf = _load_gwrf_pdp(None) or {}
    neural = _load_neural_pdp(None) or {}

    available_sources: list[str] = []
    if neural:
        available_sources.append("neural_pde")
    if gwrf:
        available_sources.append("gwrf")

    causal_curves: dict = {}
    if deps.state.registry is not None:
        try:
            from sparc.registry.run_registry import set_active_registry
            from sparc.registry.store import ArtifactStore
            set_active_registry(deps.state.registry)
            try:
                _store = ArtifactStore(deps.state.registry)
                if _store.has("3", "dose_response_curves"):
                    cd = _store.read_any("3", "dose_response_curves")
                    if isinstance(cd, dict):
                        for var, curve in cd.items():
                            if isinstance(curve, dict):
                                curve.setdefault("source", "causal_dose_response")
                                causal_curves[var] = curve
                        if causal_curves:
                            available_sources.append("causal_dose_response")
            finally:
                set_active_registry(None)
        except Exception:
            pass

    merged: dict = {}
    merged.update(gwrf)
    merged.update(neural)

    if not merged and not causal_curves:
        raise deps.missing_artifact_response(
            artifact_id="pdp_curves", stage="2",
            hint=(
                "No response-curve data found. Stage 2 produces neural PDP "
                "and GWRF condition curves; Stage 3 (frequentist mode) produces "
                "causal dose-response curves. Re-run the relevant stage."
            ),
        )

    merged["_meta"] = {
        "available_sources": available_sources,
        "by_source": {
            "neural_pde": neural,
            "gwrf": gwrf,
            "causal_dose_response": causal_curves,
        },
    }
    return merged


@router.get("/results/scenarios/routing_audit")
async def get_scenario_routing_audit():
    """Return the scenario routing + anisotropic-frame audit (Stage 4, Phase C-4)."""
    return await deps.read_or_404(
        "4", "scenario_routing_audit",
        hint="Stage 4 has not produced scenario_routing_audit.",
    )


@router.get("/results/scenarios/nuts_summary")
async def get_nuts_summary():
    """Return NUTS posterior summaries, convergence diagnostics, BMA coefficients (DB-only)."""
    store = deps.get_open_store()
    try:
        result: dict = {}
        if store.has("3", "nuts_summary"):
            ns = store.read_any("3", "nuts_summary") or {}
            result["acceptance_rate"] = ns.get("acceptance_rate")
            result["n_divergences"] = ns.get("n_divergences")
        if store.has("3", "parameter_posteriors"):
            df = store.read_any("3", "parameter_posteriors")
            if df is not None:
                result["posteriors"] = df.to_dict(orient="records")
        if store.has("3", "convergence_diagnostics"):
            df = store.read_any("3", "convergence_diagnostics")
            if df is not None:
                result["convergence"] = df.to_dict(orient="records")
        if store.has("3", "bma_coefficients"):
            df = store.read_any("3", "bma_coefficients")
            if df is not None:
                result["bma"] = df.to_dict(orient="records")
    finally:
        from sparc.registry.run_registry import set_active_registry
        set_active_registry(None)

    if not result:
        raise deps.missing_artifact_response(
            artifact_id="nuts_summary", stage="3",
            hint=(
                "No NUTS results found. Run Stage 3 with "
                "`causal.inference: bayesian` to populate posterior summaries."
            ),
        )
    return result


@router.get("/results/scenarios/detail")
async def get_scenario_detail():
    """Return scenario results as GeoJSON with delta columns + summary table."""
    cached = deps.state.result_cache.get("4", "scenario_detail_response")
    if cached is not None:
        return cached

    if deps.state.registry is None:
        raise deps.missing_artifact_response(
            artifact_id="scenario_results", stage="4",
            hint="No active run registry. Load a project first.",
        )

    from sparc.registry.store import ArtifactStore
    from sparc.scenario import (
        DELTA_SCENARIO_PREFIX,
        PRED_BASELINE_COL,
        PRED_SCENARIO_PREFIX,
        ScenarioBundle,
    )

    store = ArtifactStore(deps.state.registry)
    bundle = ScenarioBundle.from_store(store)

    if "results" not in bundle.available or bundle.results is None:
        scenarios_cfg = (deps.state.project_config or {}).get("scenarios") or []
        scenario_count = len(scenarios_cfg) if isinstance(scenarios_cfg, list) else 0
        raise deps.missing_artifact_response(
            artifact_id="scenario_results", stage="4",
            hint=(
                f"Stage 4 should auto-run the {scenario_count} scenario(s) "
                "defined in project.yml. If Stage 4 finished without "
                "writing scenario results to the database, re-run Stage 4 "
                "or use the Scenario Runner page."
                if scenario_count
                else "No `scenarios:` block found in project.yml."
            ),
        )

    import geopandas as gpd

    gdf = bundle.results
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise HTTPException(
            500,
            f"Artifact {bundle.results_artifact_id} is not a GeoDataFrame; "
            "scenario producer must include a geometry column.",
        )

    if PRED_BASELINE_COL in gdf.columns:
        for col in list(gdf.columns):
            if not col.startswith(PRED_SCENARIO_PREFIX):
                continue
            delta_col = col.replace(PRED_SCENARIO_PREFIX, DELTA_SCENARIO_PREFIX)
            if delta_col in gdf.columns:
                continue
            try:
                gdf[delta_col] = (
                    gdf[col].astype(float) - gdf[PRED_BASELINE_COL].astype(float)
                )
            except Exception:
                pass

    if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
    geojson_data = gdf.__geo_interface__

    summary_records: list[dict[str, Any]] = []
    if bundle.summary is not None:
        try:
            summary_records = bundle.summary.to_dict(orient="records")
        except Exception:
            summary_records = []

    response = {
        "geojson": geojson_data,
        "summary": summary_records,
        "results_artifact_id": bundle.results_artifact_id,
        "summary_artifact_id": bundle.summary_artifact_id,
    }
    deps.state.result_cache.set("4", "scenario_detail_response", response)
    return response


@router.get("/results/scenarios/attribution")
async def get_scenario_attribution():
    """Return the per-scenario x per-variable attribution table."""
    from sparc.registry.store import ArtifactStore
    from sparc.scenario import SCENARIO_ATTRIBUTION, SCENARIO_STAGE

    if deps.state.registry is None:
        raise deps.missing_artifact_response(
            artifact_id=SCENARIO_ATTRIBUTION, stage=SCENARIO_STAGE,
            hint="No active run registry.",
        )
    store = ArtifactStore(deps.state.registry)
    if not store.has(SCENARIO_STAGE, SCENARIO_ATTRIBUTION):
        raise deps.missing_artifact_response(
            artifact_id=SCENARIO_ATTRIBUTION, stage=SCENARIO_STAGE,
            hint="Scenario simulator did not produce an attribution table.",
        )
    df = store.read_table(SCENARIO_STAGE, SCENARIO_ATTRIBUTION)
    return {"records": df.to_dict(orient="records"), "columns": list(df.columns)}


@router.get("/results/scenarios/trajectory")
async def get_scenario_trajectory(scenario_id: int | None = None):
    """Return the long-format scenario trajectory table, optionally filtered."""
    from sparc.registry.store import ArtifactStore
    from sparc.scenario import SCENARIO_STAGE, SCENARIO_TRAJECTORY

    if deps.state.registry is None:
        raise deps.missing_artifact_response(
            artifact_id=SCENARIO_TRAJECTORY, stage=SCENARIO_STAGE,
            hint="No active run registry.",
        )
    store = ArtifactStore(deps.state.registry)
    if not store.has(SCENARIO_STAGE, SCENARIO_TRAJECTORY):
        raise deps.missing_artifact_response(
            artifact_id=SCENARIO_TRAJECTORY, stage=SCENARIO_STAGE,
            hint="Scenario simulator did not produce a trajectory table.",
        )
    df = store.read_table(SCENARIO_STAGE, SCENARIO_TRAJECTORY)
    if scenario_id is not None and "scenario_id" in df.columns:
        df = df[df["scenario_id"] == scenario_id]
    return {"records": df.to_dict(orient="records"), "columns": list(df.columns)}


@router.get("/results/scenarios/uncertainty")
async def get_scenario_uncertainty(scenario_id: int | None = None):
    """Return per-feature uncertainty (mean/std/p05/p95) as GeoJSON."""
    from sparc.registry.store import ArtifactStore
    from sparc.scenario import (
        UNCERTAINTY_COLS,
        ScenarioBundle,
        pred_column,
    )

    if deps.state.registry is None:
        raise deps.missing_artifact_response(
            artifact_id="scenario_results", stage="4",
            hint="No active run registry.",
        )
    store = ArtifactStore(deps.state.registry)
    bundle = ScenarioBundle.from_store(store)
    if bundle.results is None:
        raise deps.missing_artifact_response(
            artifact_id="scenario_results", stage="4",
            hint="Scenario results not in database.",
        )
    if not bundle.has_uncertainty():
        raise deps.missing_artifact_response(
            artifact_id="scenario_results", stage="4",
            hint=(
                "Active scenario_results table has no uncertainty columns "
                f"({', '.join(UNCERTAINTY_COLS)}). Re-run with the MC-Dropout "
                "ensemble enabled."
            ),
        )
    import geopandas as gpd

    gdf = bundle.results
    keep = [c for c in UNCERTAINTY_COLS if c in gdf.columns]
    if scenario_id is not None:
        sc = bundle.get_scenario(scenario_id)
        if sc is not None:
            keep.append(sc.pred_column)
            if sc.delta_column:
                keep.append(sc.delta_column)
    if isinstance(gdf, gpd.GeoDataFrame):
        out = gdf[[*keep, gdf.geometry.name]].copy()
        if out.crs is not None and str(out.crs) != "EPSG:4326":
            out = out.to_crs(epsg=4326)
        mc_uncertainty_records: list[dict[str, Any]] | None = None
        mc_consensus_records: list[dict[str, Any]] | None = None
        mc_consensus_summary_records: list[dict[str, Any]] | None = None
        if bundle.mc_uncertainty is not None and hasattr(bundle.mc_uncertainty, "to_dict"):
            try:
                mc_uncertainty_records = bundle.mc_uncertainty.to_dict(orient="records")
            except Exception:
                mc_uncertainty_records = None
        if bundle.mc_consensus is not None and hasattr(bundle.mc_consensus, "to_dict"):
            try:
                mc_consensus_records = bundle.mc_consensus.to_dict(orient="records")
            except Exception:
                mc_consensus_records = None
        if bundle.mc_consensus_summary is not None and hasattr(
            bundle.mc_consensus_summary, "to_dict"
        ):
            try:
                mc_consensus_summary_records = bundle.mc_consensus_summary.to_dict(
                    orient="records"
                )
            except Exception:
                mc_consensus_summary_records = None
        return {
            "geojson": out.__geo_interface__,
            "results_artifact_id": bundle.results_artifact_id,
            "mc_uncertainty": mc_uncertainty_records,
            "mc_consensus": mc_consensus_records,
            "mc_consensus_summary": mc_consensus_summary_records,
        }
    raise HTTPException(500, "scenario results table is not a GeoDataFrame")


@router.get("/results/scenarios/variables")
async def get_scenario_variables():
    """Return the list of scenario variables and their available increments."""
    if deps.state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if deps.state.registry is None:
        raise HTTPException(404, "No active run registry")

    from sparc.registry.store import ArtifactStore
    from sparc.scenario import SCENARIO_STAGE, SCENARIO_SUMMARY_VARIANTS

    store = ArtifactStore(deps.state.registry)
    summary_df = None
    for variant in SCENARIO_SUMMARY_VARIANTS:
        if store.has(SCENARIO_STAGE, variant):
            summary_df = store.read_table(SCENARIO_STAGE, variant)
            break
    if summary_df is None:
        raise HTTPException(404, "No scenario summary found in artifacts.db")

    variables: dict[str, dict] = {}
    for _, row in summary_df.iterrows():
        var = str(row.get("Variable", ""))
        inc = row.get("Increment")
        if not var or inc is None:
            continue
        try:
            inc = float(inc)
        except (TypeError, ValueError):
            continue
        if var not in variables:
            variables[var] = {"increments": [], "sign": "plus" if inc >= 0 else "minus"}
        if inc not in variables[var]["increments"]:
            variables[var]["increments"].append(inc)

    for info in variables.values():
        info["increments"].sort(key=lambda x: abs(x))

    return {"variables": variables}


@router.get("/results/scenarios/increment")
async def get_scenario_increment(variable: str = Query(...), increment: float = Query(...)):
    """Return GeoJSON filtered to a specific variable+increment scenario (DB-only)."""
    if deps.state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if deps.state.registry is None:
        raise HTTPException(404, "No active run registry")

    from sparc.registry.store import ArtifactStore
    from sparc.scenario import (
        SCENARIO_STAGE,
        SCENARIO_SUMMARY_VARIANTS,
        SCENARIO_RESULTS_VARIANTS,
    )

    store = ArtifactStore(deps.state.registry)

    summary_df = None
    for variant in SCENARIO_SUMMARY_VARIANTS:
        if store.has(SCENARIO_STAGE, variant):
            summary_df = store.read_table(SCENARIO_STAGE, variant)
            break
    if summary_df is None:
        raise HTTPException(404, "No scenario summary found in artifacts.db")

    mask = (
        (summary_df["Variable"].str.lower() == variable.lower()) &
        (summary_df["Increment"].astype(float).round(6) == round(float(increment), 6))
    )
    matched = summary_df[mask]
    if matched.empty:
        raise HTTPException(404, f"No scenario found for {variable} at increment {increment}")

    gdf = None
    for variant in SCENARIO_RESULTS_VARIANTS:
        if store.has(SCENARIO_STAGE, variant):
            gdf = store.read_table(SCENARIO_STAGE, variant)
            break
    if gdf is None:
        raise HTTPException(404, "No spatial scenario results found in artifacts.db")

    import geopandas as gpd
    if not isinstance(gdf, gpd.GeoDataFrame) and "geometry" in gdf.columns:
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")

    scenario_label = matched.iloc[0].get("Scenario", "")
    pred_col = f"pred_{scenario_label}"
    delta_col = f"delta_{scenario_label}"
    baseline_col = "pred_baseline" if "pred_baseline" in gdf.columns else None

    cols_to_keep = ["geometry"]
    if baseline_col:
        cols_to_keep.append(baseline_col)
    if pred_col in gdf.columns:
        cols_to_keep.append(pred_col)
    if delta_col in gdf.columns:
        cols_to_keep.append(delta_col)
    elif pred_col in gdf.columns and baseline_col:
        gdf[delta_col] = gdf[pred_col].astype(float) - gdf[baseline_col].astype(float)
        cols_to_keep.append(delta_col)

    result_gdf = gdf[cols_to_keep].copy()
    if isinstance(result_gdf, gpd.GeoDataFrame):
        if result_gdf.crs is not None and str(result_gdf.crs) != "EPSG:4326":
            result_gdf = result_gdf.to_crs(epsg=4326)

    return {
        "geojson": result_gdf.__geo_interface__,
        "summary": matched.to_dict(orient="records"),
    }


@router.get("/results/scenarios")
async def list_scenarios():
    """Enumerate available scenarios from the active scenario_results table."""
    from sparc.registry.store import ArtifactStore
    from sparc.scenario import ScenarioBundle

    if deps.state.registry is None:
        raise deps.missing_artifact_response(
            artifact_id="scenario_results", stage="4",
            hint="No active run registry. Load a project first.",
        )
    store = ArtifactStore(deps.state.registry)
    bundle = ScenarioBundle.from_store(store)
    return {
        "results_artifact_id": bundle.results_artifact_id,
        "summary_artifact_id": bundle.summary_artifact_id,
        "available": sorted(bundle.available),
        "has_uncertainty": bundle.has_uncertainty(),
        "scenarios": [
            {
                "index": sid,
                "pred_column": (rec := bundle.get_scenario(sid)).pred_column,
                "delta_column": rec.delta_column,
            }
            for sid in bundle.list_scenarios()
        ],
    }


@router.get("/results/report")
async def get_report_data():
    """Compile all stage results into a structured payload for the report view (DB-only)."""
    cfg = deps.state.project_config or {}
    report: dict[str, Any] = {}

    raw = deps.state.raw_project_yaml or {}
    report["project"] = raw.get("project", {})
    report["data_summary"] = deps.state.data_summary or {}

    predictors = cfg.get("predictors", {})
    if isinstance(predictors, list):
        report["predictors"] = predictors
    elif isinstance(predictors, dict):
        report["predictors"] = predictors.get("base_model", [])
    else:
        report["predictors"] = []

    report["causal"] = cfg.get("causal", {})
    report["physics"] = cfg.get("physics", {})
    report["pipeline"] = cfg.get("pipeline", {})

    if deps.state.registry is None:
        return report

    from sparc.registry.run_registry import set_active_registry
    from sparc.registry.store import ArtifactStore
    set_active_registry(deps.state.registry)
    try:
        store = ArtifactStore(deps.state.registry)

        if store.has("0", "correlogram_results"):
            corr_data = store.read_any("0", "correlogram_results")
            individual = (corr_data or {}).get("individual_results", {})
            report["correlogram"] = {
                var: {
                    "optimal_bandwidth": info.get("optimal_bandwidth"),
                    "effective_range": info.get("effective_range"),
                    "max_moran_i": info.get("max_moran_i"),
                }
                for var, info in individual.items()
            }

        if store.has("1", "gwen_variable_importance"):
            gwen_df = store.read_any("1", "gwen_variable_importance")
            if gwen_df is not None:
                report["gwen"] = gwen_df.to_dict(orient="records")

        if store.has("2", "oof_predictions"):
            oof_df = store.read_any("2", "oof_predictions")
            if oof_df is not None:
                report["spatial_cv_models"] = list(oof_df.columns)
        if store.has("2", "ensemble_results"):
            report["ensemble_results"] = store.read_any("2", "ensemble_results")

        for art_id, key in (
            ("scenario_coefficients", "causal_results"),
            ("dose_response_curves", "dose_response"),
            ("propensity_diagnostics", "propensity_diagnostics"),
            ("causal_diagnostics", "causal_diagnostics"),
        ):
            if store.has("3", art_id):
                report[key] = store.read_any("3", art_id)

        from sparc.scenario import ScenarioBundle

        bundle = ScenarioBundle.from_store(store)
        if bundle.summary is not None and hasattr(bundle.summary, "to_dict"):
            try:
                report["scenario_summary"] = bundle.summary.to_dict(orient="records")
            except Exception:
                pass
        elif bundle.mc_consensus_summary is not None and hasattr(
            bundle.mc_consensus_summary, "to_dict"
        ):
            try:
                report["scenario_summary"] = bundle.mc_consensus_summary.to_dict(
                    orient="records"
                )
            except Exception:
                pass
        report["scenario_results_artifact_id"] = bundle.results_artifact_id
        report["scenario_summary_artifact_id"] = bundle.summary_artifact_id
    finally:
        set_active_registry(None)

    report["plots"] = {}
    return report


@router.get("/results/dataset/profile")
async def get_dataset_profile():
    """Return per-column diagnostics for the loaded project dataset."""
    if deps.state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if deps.state.data is None:
        raise HTTPException(404, "No dataset loaded — upload a CSV first")

    import numpy as np
    import pandas as pd

    df = deps.state.data
    target = (deps.state.project_config.get("variables") or {}).get("target")
    target_series = None
    if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        target_series = df[target]

    cols_out: list[dict] = []
    for col in df.columns:
        s = df[col]
        n = len(s)
        miss = float(s.isna().sum()) / max(n, 1)
        entry: dict = {
            "name": col,
            "dtype": str(s.dtype),
            "missing_pct": round(miss * 100.0, 3),
        }
        if pd.api.types.is_numeric_dtype(s):
            arr = s.dropna().to_numpy()
            if arr.size > 0:
                entry["n_unique"] = int(pd.Series(arr).nunique())
                entry["min"] = float(np.min(arr))
                entry["max"] = float(np.max(arr))
                entry["mean"] = float(np.mean(arr))
                entry["std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
                if arr.size >= 3 and entry["std"] > 0:
                    try:
                        entry["skew"] = float(pd.Series(arr).skew())
                        entry["kurtosis"] = float(pd.Series(arr).kurtosis())
                    except Exception:
                        entry["skew"] = None
                        entry["kurtosis"] = None
                else:
                    entry["skew"] = None
                    entry["kurtosis"] = None
                if target_series is not None and col != target:
                    try:
                        corr = float(s.corr(target_series))
                        entry["corr_target"] = corr if np.isfinite(corr) else None
                    except Exception:
                        entry["corr_target"] = None
                else:
                    entry["corr_target"] = None
            else:
                entry.update({
                    "n_unique": 0, "min": None, "max": None, "mean": None,
                    "std": None, "skew": None, "kurtosis": None, "corr_target": None,
                })
        else:
            try:
                entry["n_unique"] = int(s.nunique(dropna=True))
            except Exception:
                entry["n_unique"] = None
            entry.update({"min": None, "max": None, "mean": None, "std": None,
                          "skew": None, "kurtosis": None, "corr_target": None})
        cols_out.append(entry)

    crs = None
    if hasattr(df, "crs") and df.crs is not None:
        crs = str(df.crs)

    return {
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "target": target,
        "columns": cols_out,
        "crs": crs,
    }


@router.get("/results/batch")
async def get_batch_results(
    ids: str = Query(
        ...,
        description=(
            "Comma-separated 'stage:artifact_id' pairs, "
            "e.g. '0:correlogram_results,1:gwen_results'."
        ),
    ),
):
    """Return multiple artifacts in a single round-trip."""
    from sparc.server.artifact_reader import read_batch

    pairs: list[tuple[str, str]] = []
    for token in ids.split(","):
        token = token.strip()
        if ":" in token:
            stage, artifact_id = token.split(":", 1)
            pairs.append((stage.strip(), artifact_id.strip()))
    store = deps.get_open_store()
    return await asyncio.to_thread(read_batch, store, pairs)


@router.post("/results/manifest/rescan")
async def rescan_manifest():
    """Re-walk the run directory and import any unregistered artifacts."""
    if deps.state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if deps.state.registry is None:
        from sparc.server.app import _attach_registry
        _attach_registry(deps.state.project_config)
    if deps.state.registry is None:
        raise HTTPException(503, "Run registry unavailable")
    from sparc.run.pipeline_paths import PipelinePaths
    paths = PipelinePaths.from_config(deps.state.project_config)
    n = deps.state.registry.migrate_from_disk(paths)
    return {"newly_registered": n,
            "total_artifacts": len(deps.state.registry.manifest.all_artifacts())}


@router.post("/results/manifest/repair")
async def repair_manifest(threshold_minutes: float = Query(30.0)):
    """Detect stale partial-write entries and tombstone them."""
    if deps.state.registry is None:
        raise HTTPException(503, "Run registry unavailable")
    stale = deps.state.registry.stale_partials(threshold_minutes=threshold_minutes)
    if not stale:
        return {"repaired": {}}
    repaired: dict[str, list[str]] = {}
    for entry in stale:
        entry.partial = False
        stage_key = str(entry.stage)
        repaired.setdefault(stage_key, []).append(entry.id)
        sm = deps.state.registry.manifest.stages.get(stage_key)
        if sm is not None and sm.status not in ("failed",):
            sm.status = "failed"
            sm.error = f"Partial write detected for {entry.id!r}; re-run stage {stage_key}."
    deps.state.registry.save()
    return {"repaired": repaired}


@router.get("/results/manifest")
async def get_results_manifest(
    refresh: bool = Query(False, description="Reload from disk before returning."),
    rescan: bool = Query(False, description="Re-run the migrate_from_disk scan."),
):
    """Return the full run manifest (Pydantic RunManifest)."""
    if deps.state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if deps.state.registry is None:
        from sparc.server.app import _attach_registry
        _attach_registry(deps.state.project_config)
    if deps.state.registry is None:
        raise HTTPException(503, "Run registry unavailable")

    if refresh or rescan:
        deps.state.registry.load()
        if rescan:
            from sparc.run.pipeline_paths import PipelinePaths
            paths = PipelinePaths.from_config(deps.state.project_config)
            deps.state.registry.migrate_from_disk(paths)

    return deps.state.registry.manifest.model_dump(mode="json", exclude_none=False)


@router.get("/results/manifest/{stage}")
async def get_stage_manifest(stage: str):
    """Return the per-stage manifest fragment (artifacts + status)."""
    if deps.state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if deps.state.registry is None:
        from sparc.server.app import _attach_registry
        _attach_registry(deps.state.project_config)
    if deps.state.registry is None:
        raise HTTPException(503, "Run registry unavailable")

    sm = deps.state.registry.manifest.stages.get(str(stage))
    if sm is None:
        return {"stage": str(stage), "status": "pending", "artifacts": {}}
    return sm.model_dump(mode="json", exclude_none=False)


@router.get("/results/bundle")
async def get_results_bundle():
    """Stream a ZIP of every registered artifact (data formats + manifest)."""
    _ensure_registry_attached()
    if deps.state.registry is None:
        raise HTTPException(404, "No active run registry.")

    import io
    import zipfile
    from sparc.registry.run_registry import set_active_registry, get_active_registry
    from sparc.registry.store import ArtifactStore
    from sparc.report.render import (
        RenderError,
        render_csv,
        render_geojson,
        render_json,
    )

    try:
        from sparc.report.figures import FigureRenderError, render_for_artifact
        figures_available = True
    except ImportError:
        figures_available = False

    store = ArtifactStore(deps.state.registry)
    manifest = deps.state.registry.manifest

    buf = io.BytesIO()
    prev = get_active_registry()
    set_active_registry(deps.state.registry)
    try:
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "manifest.json",
                manifest.model_dump_json(indent=2),
            )
            stage_items = list(manifest.stages.items())
            for stage_id, stage_manifest in stage_items:
                artifact_items = list(stage_manifest.artifacts.items())
                for artifact_id, entry in artifact_items:
                    base = f"stage_{stage_id}/{artifact_id}"
                    try:
                        if entry.storage_kind == "table":
                            geom = (entry.metadata or {}).get("geometry_col")
                            if geom:
                                zf.writestr(
                                    f"{base}.geojson",
                                    render_geojson(stage_id, artifact_id),
                                )
                            else:
                                zf.writestr(
                                    f"{base}.csv",
                                    render_csv(stage_id, artifact_id),
                                )
                        elif entry.storage_kind == "struct":
                            zf.writestr(
                                f"{base}.json",
                                render_json(stage_id, artifact_id),
                            )
                    except RenderError as exc:
                        zf.writestr(
                            f"{base}.SKIPPED.txt",
                            f"render failed: {exc}".encode("utf-8"),
                        )
                    if figures_available:
                        try:
                            png = render_for_artifact(
                                stage_id, artifact_id, registry=deps.state.registry
                            )
                            zf.writestr(f"{base}.png", png)
                        except FigureRenderError:
                            pass
                        except Exception as exc:  # noqa: BLE001
                            zf.writestr(
                                f"{base}.png.SKIPPED.txt",
                                f"png render failed: {exc}".encode("utf-8"),
                            )
    finally:
        set_active_registry(prev)

    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="sparc_results_bundle.zip"',
        },
    )


@router.get("/results/availability")
async def get_results_availability():
    """Report which result artifacts are present in artifacts.db (DB-only)."""
    if deps.state.project_config is None:
        raise HTTPException(400, "No project loaded")

    db_artifacts: set[tuple[str, str]] = set()
    if deps.state.registry is not None:
        try:
            man = deps.state.registry.manifest
            for stage_obj in man.stages.values():
                for art in stage_obj.artifacts.values():
                    if not art.partial:
                        db_artifacts.add((str(stage_obj.stage), art.id))
        except Exception:
            db_artifacts = set()

    endpoints: dict[str, tuple[str, str] | None] = {
        "/results/correlogram": ("0", "correlogram_results"),
        "/results/gwen": ("1", "gwen_variable_importance"),
        "/results/model_performance": ("2", "ensemble_results"),
        "/results/spatial_cv/predictions": ("2", "spatial_cv_predictions"),
        "/results/neural_pdp": None,
        "/results/pdp_curves": None,
        "/results/causal": ("3", "scenario_coefficients"),
        "/results/causal/diagnostics": ("3", "causal_diagnostics"),
        "/results/causal/dose_response": ("3", "dose_response_curves"),
        "/results/causal/cate_map": ("3", "cate_summary"),
        "/results/scenarios/nuts_summary": ("3", "nuts_summary"),
        "/results/scenarios/detail": ("4", "scenario_results"),
        "/results/kernel_field": ("0", "kernel_field"),
        "/results/causal/pdp_curves": ("3", "causal_pdp_curves"),
        "/results/causal/divergence": ("3", "cate_vs_gwr_divergence"),
        "/results/scenarios/routing_audit": ("4", "scenario_routing_audit"),
        "/dag/mc3_result": ("3", "mc3_summary"),
    }

    out: dict[str, dict] = {}
    has_neural_pdp = any(s == "2" and a.startswith("v2_neural_pdp::") for s, a in db_artifacts)
    has_gwrf = (("2", "gwrf_condition_curves") in db_artifacts) or \
               (("3", "gwrf_condition_curves") in db_artifacts)

    for endpoint, key in endpoints.items():
        if endpoint == "/results/neural_pdp":
            available = has_neural_pdp
            source = "artifacts.db:2:v2_neural_pdp::*" if available else ""
        elif endpoint == "/results/pdp_curves":
            available = has_neural_pdp or has_gwrf
            source = "artifacts.db (neural_pdp+gwrf)" if available else ""
        else:
            stage_id, art_id = key  # type: ignore[misc]
            available = (stage_id, art_id) in db_artifacts
            source = f"artifacts.db:{stage_id}:{art_id}" if available else ""
        out[endpoint] = {"available": available, "source": source}

    out["_project"] = {
        "available": True,
        "source": deps.state.project_path,
    }
    return out


@router.get("/results/artifacts")
async def list_artifacts():
    """List every registered artifact across all stages (DB-only)."""
    if deps.state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if deps.state.registry is None:
        raise HTTPException(404, "No active run registry")

    stage_labels = {
        "0": "Stage 0 — Correlogram",
        "1": "Stage 1 — GWEN",
        "2": "Stage 2 — Spatial CV",
        "3": "Stage 3 — Causal",
        "4": "Stage 4 — Scenarios",
    }

    artifacts: list[dict] = []
    for stage_key, stage_obj in list(deps.state.registry.manifest.stages.items()):
        for art_id, art in list(stage_obj.artifacts.items()):
            if getattr(art, "partial", False):
                continue
            kind = getattr(art, "kind", "")
            ext = {"table": "csv", "struct": "json", "blob": "bin"}.get(kind, "bin")
            artifacts.append({
                "stage": stage_key,
                "stage_label": stage_labels.get(str(stage_key), f"Stage {stage_key}"),
                "artifact_id": art_id,
                "filename": f"{art_id}.{ext}",
                "kind": kind,
                "extension": f".{ext}",
                "size_bytes": getattr(art, "size_bytes", None),
                "producer": getattr(art, "producer", None),
                "download_url": f"/artifacts/{stage_key}/{art_id}",
            })

    return {"artifacts": artifacts, "total": len(artifacts)}


@router.get("/results/download/{stage}/{file_path:path}")
async def download_artifact(stage: str, file_path: str):
    """Download a registered artifact in its native (or requested) format."""
    if deps.state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if deps.state.registry is None:
        raise HTTPException(404, "No active run registry")

    from sparc.registry.run_registry import set_active_registry, get_active_registry
    from sparc.report.render import (
        render_native, render_csv, render_json, render_geojson, RenderError,
    )

    leaf = file_path.rsplit("/", 1)[-1]
    if "." in leaf:
        artifact_id, _, ext = leaf.rpartition(".")
        ext = ext.lower()
    else:
        artifact_id, ext = leaf, ""

    prev = get_active_registry()
    set_active_registry(deps.state.registry)
    try:
        try:
            if ext == "csv":
                data = render_csv(str(stage), artifact_id)
                media = "text/csv"
                fname = f"{artifact_id}.csv"
            elif ext == "json":
                data = render_json(str(stage), artifact_id)
                media = "application/json"
                fname = f"{artifact_id}.json"
            elif ext == "geojson":
                data = render_geojson(str(stage), artifact_id)
                media = "application/geo+json"
                fname = f"{artifact_id}.geojson"
            elif ext == "png":
                from sparc.report.figures import (
                    FigureRenderError, render_for_artifact,
                )
                try:
                    data = render_for_artifact(
                        str(stage), artifact_id, registry=deps.state.registry,
                    )
                except FigureRenderError as exc:
                    raise HTTPException(404, str(exc))
                media = "image/png"
                fname = f"{artifact_id}.png"
            else:
                data, native_ext = render_native(str(stage), artifact_id)
                media = _RENDER_MIME.get(native_ext, "application/octet-stream")
                fname = f"{artifact_id}.{native_ext}"
        except RenderError as exc:
            raise HTTPException(404, str(exc))
    finally:
        set_active_registry(prev)

    return Response(
        content=data,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.get("/results/geopackage")
async def download_geopackage():
    """Merge spatial layers from artifacts.db into a single GeoPackage download."""
    if deps.state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if deps.state.registry is None:
        raise HTTPException(404, "No active run registry")

    import geopandas as gpd
    import tempfile
    from sparc.registry.store import ArtifactStore

    store = ArtifactStore(deps.state.registry)
    layer_specs = [
        ("predictions",     "2", "predictions"),
        ("causal_effects",  "3", "cate_summary"),
        ("scenario_deltas", "4", "scenario_results"),
    ]

    layers: dict[str, gpd.GeoDataFrame] = {}
    for layer_name, stage_id, art_id in layer_specs:
        if not store.has(stage_id, art_id):
            continue
        try:
            gdf = store.read_table(stage_id, art_id)
        except Exception:
            continue
        if not isinstance(gdf, gpd.GeoDataFrame):
            if "geometry" in gdf.columns:
                gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
            else:
                continue
        if len(gdf) > 0:
            layers[layer_name] = gdf

    if not layers:
        raise HTTPException(404, "No spatial artifacts available in artifacts.db")

    project_name = deps.state.project_config.get("project", {}).get("name", "sparc_results")
    tmp = tempfile.NamedTemporaryFile(
        suffix=".gpkg", prefix=f"{project_name}_", delete=False
    )
    tmp_path = tmp.name
    tmp.close()

    try:
        for layer_name, gdf in layers.items():
            gdf.to_file(tmp_path, layer=layer_name, driver="GPKG")
    except Exception as exc:
        raise HTTPException(500, f"GeoPackage creation failed: {exc}")

    return FileResponse(
        tmp_path,
        filename=f"{project_name}_results.gpkg",
        media_type="application/geopackage+sqlite3",
    )


@router.get("/results/{stage}/predictions")
async def get_predictions(stage: int, format: str = Query("geojson", regex="^(json|geojson)$")):
    """Return spatial predictions for a stage (DB-only).

    Mapping:
      - stage 2 -> artifact ("2", "spatial_cv_predictions")
      - stage 4 -> ScenarioBundle baseline geometry from ("4", "scenario_results")
      - other stages do not produce per-point predictions.
    """
    if stage == 2:
        store = deps.get_open_store()
        try:
            if not store.has("2", "spatial_cv_predictions"):
                raise deps.missing_artifact_response(
                    artifact_id="spatial_cv_predictions", stage="2",
                    hint="Stage 2 has not produced spatial_cv_predictions. Run Stage 2.",
                )
            gdf = store.read_any("2", "spatial_cv_predictions")
        finally:
            from sparc.registry.run_registry import set_active_registry
            set_active_registry(None)
        if format == "geojson":
            return _to_geojson(gdf)
        return _to_json(gdf)

    if stage == 4:
        raise deps.missing_artifact_response(
            artifact_id="scenario_results", stage="4",
            hint="Use /results/scenarios/detail for stage 4 scenario predictions.",
        )

    hints = {
        0: "Stage 0 (EDA) does not write per-point predictions.",
        1: "Stage 1 (GWEN) writes feature-selection results, not predictions.",
        3: "Stage 3 produces causal effects, not direct predictions. View predictions on the Stage 2 tab.",
    }
    raise deps.missing_artifact_response(
        artifact_id=f"stage_{stage}_predictions", stage=str(stage),
        hint=hints.get(stage, f"No predictions registered for stage {stage}."),
    )


@router.get("/results/{stage}")
async def get_results(stage: int, format: str = Query("json", regex="^(json|geojson)$")):
    """Return in-memory results for a completed pipeline stage (DB-only).

    For artifact-style data use the dedicated /results/<name> endpoints.
    This endpoint only exposes the live in-memory result captured during
    the current process run.
    """
    result = deps.state.get_result(stage)
    if result is None:
        raise deps.missing_artifact_response(
            artifact_id=f"stage_{stage}_result", stage=str(stage),
            hint=(
                f"No in-memory result for stage {stage}. Run the stage in this "
                "session, or query a specific artifact via `/results/<name>`."
            ),
        )
    if format == "geojson":
        return _to_geojson(result)
    return _to_json(result)
