"""Decision-support and equity routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query

from sparc.server import deps

router = APIRouter(tags=["decision"])


# ---------------------------------------------------------------------------
# Private helpers (avoid importing from sparc.server.app — circular import)
# ---------------------------------------------------------------------------

async def _get_scenario_summary() -> list[dict]:
    """Return scenario summary records from the active registry; [] on failure."""
    state = deps.state
    if state.registry is None:
        return []
    try:
        from sparc.registry.store import ArtifactStore
        from sparc.scenario import ScenarioBundle

        store = ArtifactStore(state.registry)
        bundle = ScenarioBundle.from_store(store)
        if "results" not in bundle.available or bundle.summary is None:
            return []
        return bundle.summary.to_dict(orient="records")
    except Exception:
        return []


async def _get_nuts_dict() -> dict | None:
    """Return NUTS posterior summary dict from the active registry; None on failure."""
    state = deps.state
    if state.registry is None:
        return None
    try:
        from sparc.registry.store import ArtifactStore

        store = ArtifactStore(state.registry)
        result: dict[str, Any] = {}
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
        return result if result else None
    except Exception:
        return None


def _parse_intervention_candidates(raw_candidates: list, InterventionCandidate: type) -> list:
    """Build InterventionCandidate instances from a list of raw dicts."""
    return [
        InterventionCandidate(
            name=str(r.get("name", "candidate")),
            treatment=str(r.get("treatment", r.get("name", "candidate"))),
            magnitude=float(r.get("magnitude", 0.0) or 0.0),
            mean_effect=float(r.get("mean_effect", r.get("delta", 0.0)) or 0.0),
            effect_std=float(r.get("effect_std", 0.0) or 0.0),
            cost=float(r.get("cost", 1.0) or 1.0),
            equity_weight=float(r.get("equity_weight", 1.0) or 1.0),
            notes=str(r.get("notes", "") or ""),
        )
        for r in raw_candidates
        if isinstance(r, dict)
    ]


async def _build_cate_geojson(variable: str) -> dict:
    """Build a GeoJSON FeatureCollection for the per-cell CATE of *variable*.

    Replicates the core of the inline ``get_cate_map`` handler without
    importing from ``sparc.server.app`` (which would create a circular import).
    """
    import numpy as np

    state = deps.state
    if state.registry is None:
        raise HTTPException(404, f"No CATE features for '{variable}'")

    coef: Any = None
    try:
        from sparc.registry.run_registry import set_active_registry
        from sparc.registry.store import ArtifactStore

        set_active_registry(state.registry)
        try:
            store = ArtifactStore(state.registry)
            if not store.has("3", "cate_summary"):
                raise HTTPException(404, f"No CATE features for '{variable}'")
            df = store.read_table("3", "cate_summary")
            if df is None or "treatment" not in df.columns:
                raise HTTPException(404, f"No CATE features for '{variable}'")
            sub = df[df["treatment"] == variable]
            if sub.empty or "cate_mean" not in sub.columns:
                raise HTTPException(404, f"No CATE features for '{variable}'")
            sub = sub.sort_values("cell_id")
            coef = np.asarray(sub["cate_mean"].to_numpy(), dtype=float)
        finally:
            set_active_registry(None)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"CATE store read failed: {exc}")

    if state.data is None or not hasattr(state.data, "geometry"):
        raise HTTPException(500, "Project data has no geometry; cannot build CATE map.")

    import geopandas as gpd

    src = state.data
    n = min(len(src), len(coef))
    gdf = gpd.GeoDataFrame(
        {f"cate_{variable}": coef[:n]},
        geometry=src.geometry.values[:n],
        crs=src.crs,
    )
    if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
    return gdf.__geo_interface__


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/decision/candidates")
async def get_decision_candidates():
    """Build candidate interventions from scenario + NUTS results."""
    from sparc.decision import propose_candidates_from_scenarios

    if deps.state.project_config is None:
        raise HTTPException(400, "No project loaded")

    scenarios: list[dict] = []
    nuts: dict | None = None
    try:
        scenarios = await _get_scenario_summary()
    except HTTPException:
        scenarios = []
    try:
        nuts = await _get_nuts_dict()
    except (HTTPException, Exception):
        nuts = None

    cands = propose_candidates_from_scenarios(scenarios, nuts)
    return {
        "candidates": [
            {
                "name": c.name,
                "treatment": c.treatment,
                "magnitude": c.magnitude,
                "mean_effect": c.mean_effect,
                "effect_std": c.effect_std,
                "cost": c.cost,
                "equity_weight": c.equity_weight,
                "notes": c.notes,
            }
            for c in cands
        ]
    }


@router.post("/decision/optimize")
async def post_decision_optimize(body: dict = Body(default_factory=dict)):
    """Rank intervention candidates and (optionally) pick a portfolio under a budget.

    Body fields (all optional)::

        {
          "candidates":          [InterventionCandidate, ...],   # explicit override
          "budget":              float | null,
          "robustness_lambda":   float (default 0),
          "minimise":            bool  (default false)
        }

    If ``candidates`` is omitted the endpoint derives them from the latest
    scenario + NUTS posterior outputs.
    """
    from sparc.decision import (
        InterventionCandidate,
        propose_candidates_from_scenarios,
        rank_interventions,
    )

    raw_candidates = body.get("candidates")
    if raw_candidates:
        candidates = _parse_intervention_candidates(raw_candidates, InterventionCandidate)
    else:
        if deps.state.project_config is None:
            raise HTTPException(400, "No project loaded and no candidates provided")
        scenarios: list[dict] = []
        nuts: dict | None = None
        try:
            scenarios = await _get_scenario_summary()
        except HTTPException:
            scenarios = []
        try:
            nuts = await _get_nuts_dict()
        except (HTTPException, Exception):
            nuts = None
        candidates = propose_candidates_from_scenarios(scenarios, nuts)

    if not candidates:
        raise HTTPException(404, "No intervention candidates available")

    result = rank_interventions(
        candidates,
        budget=body.get("budget"),
        robustness_lambda=float(body.get("robustness_lambda", 0.0) or 0.0),
        minimise=bool(body.get("minimise", False)),
    )
    return {
        "ranked": result.ranked,
        "settings": result.settings,
        "equity_summary": result.equity_summary,
    }


@router.post("/decision/uncertainty")
async def post_decision_uncertainty(body: dict = Body(default_factory=dict)):
    """Monte-Carlo uncertainty for the decision optimizer.

    Body fields (same as ``/decision/optimize`` plus ``n_draws`` and ``seed``)::

        {
          "candidates":         [...],   # optional, falls back to scenarios+NUTS
          "budget":             float | null,
          "robustness_lambda":  float,
          "minimise":           bool,
          "n_draws":            int (default 500),
          "seed":               int | null (default 42)
        }

    Returns selection probabilities, rank quantiles, and effect quantiles
    per candidate.
    """
    from sparc.decision import (
        InterventionCandidate,
        monte_carlo_decision,
        propose_candidates_from_scenarios,
    )

    raw_candidates = body.get("candidates")
    if raw_candidates:
        candidates = _parse_intervention_candidates(raw_candidates, InterventionCandidate)
    else:
        if deps.state.project_config is None:
            raise HTTPException(400, "No project loaded and no candidates provided")
        scenarios: list[dict] = []
        nuts: dict | None = None
        try:
            scenarios = await _get_scenario_summary()
        except HTTPException:
            scenarios = []
        try:
            nuts = await _get_nuts_dict()
        except (HTTPException, Exception):
            nuts = None
        candidates = propose_candidates_from_scenarios(scenarios, nuts)

    if not candidates:
        raise HTTPException(404, "No intervention candidates available")

    results = monte_carlo_decision(
        candidates,
        budget=body.get("budget"),
        robustness_lambda=float(body.get("robustness_lambda", 0.0) or 0.0),
        minimise=bool(body.get("minimise", False)),
        n_draws=int(body.get("n_draws", 500) or 500),
        seed=body.get("seed", 42),
    )
    return {
        "uncertainty": [
            {
                "candidate": r.candidate,
                "selection_probability": r.selection_probability,
                "rank_mean": r.rank_mean,
                "rank_p10": r.rank_p10,
                "rank_p90": r.rank_p90,
                "effect_p10": r.effect_p10,
                "effect_p50": r.effect_p50,
                "effect_p90": r.effect_p90,
            }
            for r in results
        ],
        "settings": {
            "n_draws": int(body.get("n_draws", 500) or 500),
            "robustness_lambda": float(body.get("robustness_lambda", 0.0) or 0.0),
            "budget": body.get("budget"),
            "minimise": bool(body.get("minimise", False)),
        },
    }


@router.get("/equity/census")
async def get_equity_census(
    minlon: float | None = Query(default=None),
    minlat: float | None = Query(default=None),
    maxlon: float | None = Query(default=None),
    maxlat: float | None = Query(default=None),
):
    """Auto-fetch area-wide ACS demographics for the project bbox.

    If bbox params are omitted, falls back to the loaded project's
    coordinate columns (``data.lon_column``/``lat_column``) by reading
    the source CSV. Only works for US locations (Census Geocoder).
    """
    from sparc.data.census_equity import (
        context_to_equity_layers,
        fetch_area_demographics,
    )

    state = deps.state

    # Fallback: derive bbox from the project's data file.
    if None in (minlon, minlat, maxlon, maxlat):
        if state.project_config is None:
            raise HTTPException(400, "No project loaded; provide bbox params or load a project.")
        try:
            data_cfg = state.project_config.get("data", {}) or {}
            csv_path = data_cfg.get("path") or data_cfg.get("source")
            lon_col = data_cfg.get("lon_column") or "lon"
            lat_col = data_cfg.get("lat_column") or "lat"
            if not csv_path or not Path(csv_path).exists():
                raise HTTPException(400, "Project data file not found; provide explicit bbox.")
            import pandas as pd

            df = pd.read_csv(csv_path, usecols=[lon_col, lat_col])
            minlon = float(df[lon_col].min())
            maxlon = float(df[lon_col].max())
            minlat = float(df[lat_col].min())
            maxlat = float(df[lat_col].max())
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(400, f"Could not derive bbox from project data: {exc}")

    ctx = fetch_area_demographics(
        minlon=float(minlon),  # type: ignore[arg-type]
        minlat=float(minlat),  # type: ignore[arg-type]
        maxlon=float(maxlon),  # type: ignore[arg-type]
        maxlat=float(maxlat),  # type: ignore[arg-type]
    )
    return {
        "bbox": {"minlon": minlon, "minlat": minlat, "maxlon": maxlon, "maxlat": maxlat},
        **context_to_equity_layers(ctx),
    }


@router.post("/decision/equity")
async def post_decision_equity(body: dict = Body(default_factory=dict)):
    """Combine equity layers into per-candidate weights.

    Body fields::

        {
          "candidate_names": [str, ...],
          "layers":          {layer_name: [float, ...]},   # aligned to names
          "weights":         {layer_name: float},          # optional convex weights
          "invert":          {layer_name: bool}            # flip 1−x for "advantage" layers
        }

    Returns ``{ "scores": [...], "disparity_index": float }``.
    """
    from sparc.decision import combine_equity_layers, disparity_index

    names = body.get("candidate_names") or []
    layers = body.get("layers") or {}
    if not names or not layers:
        raise HTTPException(400, "candidate_names and layers are required")

    try:
        scores = combine_equity_layers(
            layers,
            names,
            weights=body.get("weights"),
            invert=body.get("invert"),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    return {
        "scores": [
            {
                "candidate": s.candidate,
                "weight": s.weight,
                "layer_breakdown": s.layer_breakdown,
            }
            for s in scores
        ],
        "disparity_index": disparity_index([s.weight for s in scores]),
    }


@router.get("/decision/targeting")
async def get_decision_targeting(
    variable: str = Query(...),
    top_k: int = Query(50, ge=1, le=2000),
):
    """Return per-location deployment priority as GeoJSON.

    Priority = |CATE| / cost  (cost defaults to 1).  The endpoint reuses
    the spatial CATE map for the requested treatment ``variable``,
    annotates each feature with a ``priority`` property, sorts the result
    descending, and flags the top ``top_k`` features with ``selected=True``.
    """
    if deps.state.project_config is None:
        raise HTTPException(400, "No project loaded")

    cate_geo = await _build_cate_geojson(variable=variable)
    if not isinstance(cate_geo, dict):
        raise HTTPException(500, "CATE map response was not a dict")
    features = cate_geo.get("features") or []
    if not features:
        raise HTTPException(404, f"No CATE features for '{variable}'")

    cate_field = f"cate_{variable}"
    enriched: list[tuple[float, dict]] = []
    for feat in features:
        props = dict(feat.get("properties") or {})
        try:
            cate_val = float(props.get(cate_field, 0.0) or 0.0)
        except (TypeError, ValueError):
            cate_val = 0.0
        # Optional cost field; defaults to 1 for uniform cost.
        try:
            cost = float(props.get("cost", 1.0) or 1.0)
        except (TypeError, ValueError):
            cost = 1.0
        priority = abs(cate_val) / max(cost, 1e-9)
        props["priority"] = priority
        props["abs_cate"] = abs(cate_val)
        new_feat = {**feat, "properties": props}
        enriched.append((priority, new_feat))

    enriched.sort(key=lambda pair: pair[0], reverse=True)
    cutoff = min(top_k, len(enriched))
    out_features: list[dict] = []
    priorities: list[float] = []
    for rank, (priority, feat) in enumerate(enriched, start=1):
        feat["properties"]["rank"] = rank
        feat["properties"]["selected"] = rank <= cutoff
        out_features.append(feat)
        priorities.append(priority)

    return {
        "type": "FeatureCollection",
        "features": out_features,
        "summary": {
            "variable": variable,
            "n_features": len(out_features),
            "top_k": cutoff,
            "max_priority": max(priorities) if priorities else 0.0,
            "min_priority": min(priorities) if priorities else 0.0,
        },
    }


@router.get("/equity/layer")
async def get_equity_layer():
    """Return the equity layer as GeoJSON (Stage-0 artifact or on-demand Census fetch).

    Properties per cell: ``equity_score`` (0–1), ``poverty_rate``,
    ``minority_pct``, ``tract_geoid`` (when available).
    """
    state = deps.state

    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    # Try Stage-0 artifact first.
    if state.registry is not None:
        from sparc.registry.store import ArtifactStore

        store = ArtifactStore(state.registry)
        if store.has("0", "equity_layer"):
            eq_df = store.read_table("0", "equity_layer")
            if eq_df is not None and not eq_df.empty:
                return {"source": "artifact_store", "records": eq_df.to_dict(orient="records")}

    # On-demand fetch using Census TIGER.
    if state.data is None:
        raise HTTPException(
            404,
            "Equity layer not yet computed and project data is not loaded. "
            "Run Stage 0 or load a project with data to trigger the Census fetch.",
        )

    census_key = None
    try:
        from sparc.config.user_preferences import load_preferences

        prefs = load_preferences()
        census_key = (prefs.get("api_keys") or {}).get("census")
    except Exception:
        pass

    try:
        from sparc.data.census_equity import get_per_cell_equity_df

        eq_df = get_per_cell_equity_df(state.data, census_api_key=census_key)
        return {"source": "census_on_demand", "records": eq_df.to_dict(orient="records")}
    except Exception as _e:
        raise HTTPException(503, f"Census equity fetch failed: {_e}")
