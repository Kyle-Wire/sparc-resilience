"""sparc.server.routes.collect — /collect/* endpoints.

Migrated from app.py inline handlers.  Importable independently of the full
FastAPI app.  Uses a module-level CollectSession singleton (same pattern as
the original inline handlers).
"""

from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Query

import sparc.data.collect.adapters as _adapters_module  # noqa: F401 — side-effect: registers all adapters
from sparc.data.collect.session import CollectSession

router = APIRouter(tags=["collect"])

_collect_session: CollectSession = CollectSession()


@router.post("/collect/boundary")
async def collect_boundary(body: dict = Body(...)):
    """Resolve a study-area boundary from place name, file path, or drawn GeoJSON.

    Body keys (exactly one required):
      place_name : str
      file_path  : str
      geojson    : dict  (GeoJSON FeatureCollection or Feature)
    """
    from sparc.data.collect.boundary import resolve_boundary
    place_name = body.get("place_name")
    file_path  = body.get("file_path")
    geojson    = body.get("geojson")
    try:
        result = resolve_boundary(
            place_name=place_name,
            file_path=file_path,
            geojson=geojson,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))

    _collect_session.set_boundary(result)

    gdf_json = result.gdf.to_crs("EPSG:4326").to_json()  # type: ignore[union-attr]
    import json as _json
    return {
        "geojson": _json.loads(gdf_json),
        "bbox": list(result.bbox),
        "source": result.source,
        "place_name": result.place_name,
    }


@router.get("/collect/manifest")
async def collect_manifest():
    """Return the current variable manifest state."""
    return _collect_session.manifest.to_api_dict()


@router.post("/collect/fetch")
async def collect_fetch(body: dict = Body(...)):
    """Trigger a fetch for a single variable group.

    Body keys:
      group : str  — one of "landsat" | "nlcd" | "era5" | "capa" |
                             "buildings" | "equity" | "sentinel2"
      config : dict — fetch parameters (date_start, date_end, cloud_cover_max,
                      temporal_mode, enabled_indices, lidar_path, dsm_path)

    Returns the updated manifest entry for the requested group.
    """
    if _collect_session.boundary is None:
        raise HTTPException(400, "Resolve boundary first via POST /collect/boundary")

    if _collect_session.fishnet is None:
        raise HTTPException(400, "Fishnet not initialised — call /collect/boundary first")

    group = body.get("group", "")
    cfg   = body.get("config", {})

    if _collect_session.anchor_dates:
        from sparc.data.collect._temporal import TemporalWindow
        from datetime import date as _date
        ds = _date.fromisoformat(cfg.get("date_start", "2022-06-01"))
        de = _date.fromisoformat(cfg.get("date_end", "2022-08-31"))
        cfg = dict(cfg, window=TemporalWindow.from_capa_dates(
            _collect_session.anchor_dates, date_start=ds, date_end=de
        ))

    try:
        import asyncio
        from sparc.data.collect.dispatch import sync_group_fetch
        result = await asyncio.to_thread(
            sync_group_fetch,
            group,
            _collect_session.fishnet,
            _collect_session.boundary,
            _collect_session.manifest,
            cfg,
        )
        _collect_session.apply_fetch(result)
        if result.error:
            raise HTTPException(502, f"Fetch failed for group '{group}': {result.error}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Fetch failed for group '{group}': {exc}")

    return {
        "group": group,
        "manifest": _collect_session.manifest.to_api_dict(),
        "n_cells": len(_collect_session.fishnet),
    }


@router.get("/collect/capa-events")
async def collect_capa_events(city: str = ""):
    """Look up available CAPA Heat Watch campaigns for a city name.

    Query params:
      city : str — city name to look up (fuzzy-matched against the catalog)

    Returns a CapaEventsResponse with matching events, suggestions, or an
    error message when the city is not found.
    """
    from sparc.data.collect.capa_catalog import lookup_city, CAPA_CATALOG

    city = city.strip()
    if not city:
        return {
            "city": city,
            "canonical_name": None,
            "found_in_catalog": False,
            "us_city": False,
            "osf_node": None,
            "folder_hint": None,
            "events": [],
            "suggestions": [],
            "error": "No city name provided.",
        }

    canonical, entry = lookup_city(city)

    if canonical is None and entry is None:
        return {
            "city": city,
            "canonical_name": None,
            "found_in_catalog": False,
            "us_city": False,
            "osf_node": None,
            "folder_hint": None,
            "events": [],
            "suggestions": [],
            "error": f"No CAPA Heat Watch campaign found for '{city}'.",
        }

    if canonical is None and isinstance(entry, dict):
        candidates: list[str] = entry.get("ambiguous") or entry.get("suggestions") or []
        return {
            "city": city,
            "canonical_name": None,
            "found_in_catalog": False,
            "us_city": False,
            "osf_node": None,
            "folder_hint": None,
            "events": [],
            "suggestions": candidates,
            "error": (
                f"Multiple matches found for '{city}'. Did you mean one of: "
                + ", ".join(candidates) + "?"
                if candidates else f"No close match found for '{city}'."
            ),
        }

    year: int | None = entry.get("year")
    event_date: str | None = f"{year}-07-01" if year else None
    label_parts = [canonical]
    if year:
        label_parts.append(f"({year})")
    event = {
        "date": event_date,
        "label": " ".join(label_parts),
        "osf_node": entry["osf_node"],
        "folder_hint": entry.get("folder_hint"),
        "source_name": "NOAA/NIHHIS Heat Watch Campaign",
    }

    return {
        "city": city,
        "canonical_name": canonical,
        "found_in_catalog": True,
        "us_city": True,
        "osf_node": entry["osf_node"],
        "folder_hint": entry.get("folder_hint"),
        "events": [event],
        "suggestions": [],
        "error": None,
    }


@router.get("/collect/preview/{variable}")
async def collect_preview(variable: str):
    """Return a GeoJSON FeatureCollection of the fishnet coloured by *variable*.

    Used by the desktop confirmation map.  Missing values are represented
    as null in feature properties; ``has_gap`` is included as a boolean.
    """
    fishnet = _collect_session.fishnet
    if fishnet is None:
        raise HTTPException(400, "No fishnet in session — fetch data first")

    import json as _json
    gdf = fishnet.to_crs("EPSG:4326")  # type: ignore[union-attr]

    cols = ["geometry"]
    if variable in gdf.columns:
        cols.append(variable)
    if "has_gap" in gdf.columns:
        cols.append("has_gap")

    subset = gdf[cols]
    return _json.loads(subset.to_json())


@router.get("/collect/cell/{cell_id}")
async def collect_cell_inspect(cell_id: int):
    """Return all variable values for a single cell (cell-click inspector)."""
    fishnet = _collect_session.fishnet
    if fishnet is None:
        raise HTTPException(400, "No fishnet in session")

    row = fishnet[fishnet["cell_id"] == cell_id]  # type: ignore[index]
    if row.empty:
        raise HTTPException(404, f"Cell {cell_id} not found")

    props = row.drop(columns=["geometry"], errors="ignore").iloc[0].to_dict()
    return {k: (None if (isinstance(v, float) and v != v) else v)
            for k, v in props.items()}


@router.post("/collect/save-config")
async def collect_save_config(body: dict = Body(...)):
    """Persist the wizard configuration to the project.yml ``collect:`` block.

    Body keys (all optional):
      city_name       : str
      capa_event_date : str | null   (ISO date)
      capa_osf_node   : str
      variables       : dict[group, {enabled: list[str]}]
      fishnet_m       : int
      aggregation     : dict[str, str]

    Writes changes to the project.yml nearest to the server's working directory
    when one exists; otherwise returns status ``"not_persisted"``.
    """
    import re as _re
    import yaml as _yaml
    from pathlib import Path as _Path

    yml_candidates = [
        _Path("project.yml"),
        _Path("../project.yml"),
    ]
    yml_path: _Path | None = next((p for p in yml_candidates if p.exists()), None)

    city_name       = body.get("city_name", "")
    capa_osf_node   = body.get("capa_osf_node", "")
    capa_event_date = body.get("capa_event_date")
    fishnet_m       = body.get("fishnet_m", 30)

    collection_block: dict = {
        "city_name":    city_name,
        "capa_osf_node": capa_osf_node,
    }
    if capa_event_date:
        collection_block["capa_event_date"] = capa_event_date
    if fishnet_m:
        collection_block["fishnet_m"] = fishnet_m

    if yml_path is None:
        return {"status": "not_persisted", "collection": collection_block}

    try:
        text = yml_path.read_text(encoding="utf-8")

        def _set_key(t: str, key: str, value: str) -> str:
            pattern = rf"(collect:.*?\n(?:[ \t]+.*\n)*?[ \t]+{_re.escape(key)}\s*:)[^\n]*"
            replacement = rf"\g<1> {value}"
            updated, n = _re.subn(pattern, replacement, t, flags=_re.DOTALL)
            if n:
                return updated
            return _re.sub(
                r"(collect:\s*\n)",
                rf"\g<1>  {key}: {value}\n",
                t,
                count=1,
            )

        if capa_osf_node is not None:
            quote = f'"{capa_osf_node}"'
            text = _set_key(text, "capa_osf_node", quote)

        yml_path.write_text(text, encoding="utf-8")
    except Exception as exc:
        import logging as _log
        _log.getLogger(__name__).warning("save-config: could not update project.yml: %s", exc)
        return {"status": "not_persisted", "collection": collection_block}

    return {"status": "saved", "collection": collection_block}


@router.post("/collect/build")
async def collect_build(body: dict = Body(...)):
    """Run the assembler to write GeoParquet + manifest and update project.yml.

    Body keys:
      output_dir   : str   (required)
      project_yml  : str   (optional — path to project.yml to auto-update)
      temporal_mode: str   (optional — "composite" | "single" | "panel")
    """
    fishnet = _collect_session.fishnet
    boundary = _collect_session.boundary
    manifest = _collect_session.manifest

    if fishnet is None or boundary is None:
        raise HTTPException(400, "Run boundary resolution and at least one fetch group first")

    if not manifest.can_build:
        raise HTTPException(422, {
            "error": "Cannot build — required variables have errors",
            "blocking": manifest.blocking_variables,
        })

    import asyncio
    from pathlib import Path as _Path

    output_dir = _Path(body.get("output_dir", "."))
    project_yml_str = body.get("project_yml")
    temporal_mode = body.get("temporal_mode", "composite")

    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{temporal_mode}"
    geoparquet_path = output_dir / f"dataset{suffix}.parquet"
    await asyncio.to_thread(fishnet.to_parquet, str(geoparquet_path), index=False)

    manifest_path = output_dir / "data_manifest.json"
    manifest.save(manifest_path)

    if project_yml_str:
        yml_path = _Path(project_yml_str)
        if yml_path.exists():
            import re as _re
            text = yml_path.read_text(encoding="utf-8")
            text = _re.sub(r"(file_path\s*:\s*).*", lambda m: m.group(1) + f'"{geoparquet_path}"', text)
            text = _re.sub(r"(target_column\s*:\s*).*", lambda m: m.group(1) + '"aat_residual"', text)
            yml_path.write_text(text, encoding="utf-8")

    sidecar_path = geoparquet_path.with_suffix(".session.json")
    _collect_session.save(sidecar_path)

    return {
        "geoparquet_path": str(geoparquet_path),
        "manifest_path": str(manifest_path),
        "n_cells": len(fishnet),
        "can_build": True,
        "manifest": manifest.to_api_dict(),
    }
