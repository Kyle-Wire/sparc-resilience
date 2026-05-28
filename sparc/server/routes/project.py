"""sparc.server.routes.project — /project/* endpoints.

Migrated from app.py.  Importable independently of the full FastAPI app.
Routes use ``sparc.server.deps.session`` for project state.

Currently migrated:
  - /project/load
  - /project/validate
  - /project/init
  - /project/create
  - /project/detect-crs  (POST — detect and resolve CRS for a data file)
  - /project/templates
  - /project/config  (GET)
  - /project/config  (PUT)
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException, Query

from sparc.server.deps import session, state

router = APIRouter(tags=["project"])


def _find_templates_dir() -> Path:
    """Return the first accessible templates directory.

    Search order:
    1. ``SPARC_TEMPLATES_DIR`` env var — operator override.
    2. ``sparc/templates/`` bundled inside the installed package — works for
       wheel installs where the templates were co-packaged with sparc.
    3. Project-root ``templates/`` — editable (development) installs.
    4. PyInstaller ``sys._MEIPASS/templates`` — onefile sidecar builds.
    Falls back to the project-root candidate if nothing is found so that
    the ``list_templates`` endpoint can still return an informative empty
    list rather than crashing.
    """
    candidates: list[Path] = []

    env_override = os.environ.get("SPARC_TEMPLATES_DIR")
    if env_override:
        candidates.append(Path(env_override))

    # Bundled inside the sparc package (wheel installs)
    candidates.append(Path(__file__).parent.parent.parent / "templates")

    # Project root (editable / source installs)
    candidates.append(Path(__file__).parent.parent.parent.parent / "templates")

    # PyInstaller onefile
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / "templates")  # type: ignore[attr-defined]

    for p in candidates:
        if p.is_dir():
            return p

    # Fallback — return project-root path even if absent; callers check .exists()
    return candidates[-2]


# Resolve once at import time.
_TEMPLATES_DIR = _find_templates_dir()


def _resolve_safe(path: str, allow_create: bool = False) -> Path:
    """Resolve path to absolute; no path-traversal above drive root."""
    return Path(path).expanduser().resolve()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/project/load")
async def load_project(path: str = Query(..., description="Absolute path to project.yml")):
    """Load a project.yml and return its metadata."""
    import asyncio

    import yaml

    resolved = _resolve_safe(path)
    if not resolved.exists():
        raise HTTPException(404, f"Project file not found: {resolved}")

    from sparc.config.config import load_config

    with open(resolved, "r", encoding="utf-8") as fh:
        raw_yaml = yaml.safe_load(fh) or {}

    try:
        config = load_config(str(resolved))
    except Exception as exc:
        raise HTTPException(422, f"Invalid project configuration: {exc}")

    session.load(
        path=str(resolved),
        config=config,
        raw_yaml=raw_yaml,
        data=None,
        data_summary=None,
        registry=None,
    )

    # Sync to the legacy ServerState so inline app.py endpoints that still
    # read state.project_config (e.g. /dag, /run/stream) see the loaded project.
    state.project_path = str(resolved)
    state.project_config = config
    state.raw_project_yaml = raw_yaml

    # Attach the run registry so artifact and result endpoints work immediately.
    # Use a lazy import to avoid a circular dependency (routes → app → routes).
    try:
        from sparc.server.app import _attach_registry
        _attach_registry(config)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not attach registry on project load: {exc}")

    # Heavy I/O (data load + model prewarm) runs in the background so the
    # HTTP response returns quickly; the client may re-fetch columns.
    async def _background_load() -> None:
        try:
            from sparc.server.app import _load_data_into_state, _start_prewarm
            data_path = config.get("data", {}).get("file_path", "")
            if data_path and os.path.exists(data_path):
                await asyncio.to_thread(_load_data_into_state, config)
                # Sync loaded data into session so routes/data.py endpoints see it.
                session.data = state.data
                session.data_summary = state.data_summary
            _start_prewarm()
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: background project load failed: {exc}")

    asyncio.create_task(_background_load())

    return {
        "status": "loaded",
        "project": raw_yaml.get("project", {}),
        "columns": [],   # populated after background load; client may re-fetch
        "row_count": 0,
    }


@router.post("/project/validate")
async def validate_project(path: str = Query(..., description="Absolute path to project.yml")):
    """Validate a project.yml without loading it into state."""
    resolved = _resolve_safe(path)
    if not resolved.exists():
        raise HTTPException(404, f"Project file not found: {resolved}")

    from sparc.config.config import load_config

    warnings: list[str] = []
    try:
        config = load_config(str(resolved))
    except Exception as exc:
        return {"valid": False, "error": str(exc), "warnings": []}

    data_path = config["data"]["file_path"]
    if not Path(data_path).exists():
        warnings.append(f"Data file not found: {data_path}")

    return {"valid": True, "warnings": warnings}


@router.post("/project/init")
async def init_project(
    template: str = Query("blank", description="Template name"),
    output: str = Query(..., description="Output directory path"),
):
    """Scaffold a new project from a domain template."""
    source = _TEMPLATES_DIR / template
    if not source.exists():
        available = [d.name for d in _TEMPLATES_DIR.iterdir() if d.is_dir()] if _TEMPLATES_DIR.is_dir() else []
        raise HTTPException(404, f"Template '{template}' not found. Available: {available}")

    dest = _resolve_safe(output, allow_create=True)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest, dirs_exist_ok=True)

    return {
        "status": "created",
        "template": template,
        "path": str(dest),
        "project_yml": str(dest / "project.yml"),
    }


@router.post("/project/create")
async def create_project(payload: dict[str, Any] = Body(...)):
    """Create a project from wizard payload."""
    import yaml

    template = payload.get("template", "blank")
    output = payload.get("output")
    identity = payload.get("identity") or {}
    crs = payload.get("crs") or {}

    if not output:
        raise HTTPException(400, "output directory is required")
    if not identity.get("name"):
        raise HTTPException(400, "identity.name is required")
    if not crs.get("input"):
        raise HTTPException(400, "crs.input is required")
    if not (crs.get("working") or crs.get("projected")):
        raise HTTPException(400, "crs.working is required")

    source = _TEMPLATES_DIR / template
    if not source.exists():
        available = [d.name for d in _TEMPLATES_DIR.iterdir() if d.is_dir()] if _TEMPLATES_DIR.is_dir() else []
        raise HTTPException(404, f"Template '{template}' not found. Available: {available}")

    dest = _resolve_safe(str(output), allow_create=True)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest, dirs_exist_ok=True)

    yml_path = dest / "project.yml"
    if yml_path.exists():
        with open(yml_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    else:
        cfg = {}

    proj = cfg.get("project") or {}
    proj["name"] = identity["name"]
    if "description" in identity:
        proj["description"] = identity.get("description") or ""
    proj["domain"] = template
    cfg["project"] = proj

    cfg_crs = cfg.get("crs") or {}
    cfg_crs["input"] = crs["input"]
    # Accept new 'working' key; fall back to legacy 'projected' from callers
    cfg_crs["working"] = crs.get("working") or crs.get("projected") or crs["input"]
    # Remove old key if present so we don't write stale legacy keys
    cfg_crs.pop("projected", None)
    cfg_crs.pop("target_projected", None)
    cfg["crs"] = cfg_crs

    with open(yml_path, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)

    return {
        "status": "created",
        "template": template,
        "path": str(dest),
        "project_yml": str(yml_path),
    }



@router.post("/project/detect-crs")
async def detect_project_crs(payload: dict[str, Any] = Body(...)):
    """Detect and resolve the CRS for a data file.

    Request body
    ------------
    ``file_path`` : str
        Path to the data file. For spatial files (shapefile, GeoPackage,
        GeoTIFF, GeoJSON) the CRS is read from embedded metadata.  For CSV
        files the CRS cannot be auto-detected and ``detected_input_crs``
        will be ``null``.
    ``sample_xy`` : [lon, lat] (optional)
        A representative WGS-84 lon/lat coordinate from the dataset.  Used
        to select the best UTM zone when the input CRS is geographic.
        Required for accurate working-CRS selection when ``file_path`` is a
        CSV.

    Response
    --------
    ``detected_input_crs`` : str | null
        Detected input CRS from the file, or ``null`` if unavailable.
    ``working_crs`` : str
        Resolved projected CRS for spatial operations.
    ``unit`` : "m" | "ft"
        Linear unit of the working CRS.
    ``unit_label`` : "meters" | "feet"
        Human-readable unit label (for UI placeholder text).
    ``source`` : "detected" | "derived_utm" | "same_as_input"
        How the working CRS was determined.
    """
    from sparc.config.crs_resolver import (
        detect_crs_from_file,
        resolve_working_crs,
        crs_unit,
        crs_unit_label,
    )

    file_path = payload.get("file_path")
    sample_xy = payload.get("sample_xy")

    if not file_path:
        raise HTTPException(400, "file_path is required")

    # Detect input CRS from file metadata (None for CSV)
    detected = detect_crs_from_file(str(file_path))

    input_crs = detected  # may be None for CSV

    # Resolve working CRS
    if input_crs:
        sample_tuple = (
            (float(sample_xy[0]), float(sample_xy[1]))
            if sample_xy and len(sample_xy) == 2
            else None
        )
        working = resolve_working_crs(input_crs, sample_xy=sample_tuple)
        source = "same_as_input" if working == input_crs else "derived_utm"
    elif sample_xy and len(sample_xy) == 2:
        from sparc.config.crs_resolver import best_utm_zone
        working = best_utm_zone(float(sample_xy[0]), float(sample_xy[1]))
        source = "derived_utm"
    else:
        # No file CRS, no sample coords — cannot resolve
        working = None
        source = "unknown"

    result: dict[str, Any] = {
        "detected_input_crs": detected,
        "working_crs": working,
        "unit": crs_unit(working) if working else None,
        "unit_label": crs_unit_label(working) if working else None,
        "source": source,
    }
    return result


@router.get("/project/templates")
async def list_templates():
    """List available domain templates."""
    templates = []
    if _TEMPLATES_DIR.exists():
        for d in sorted(_TEMPLATES_DIR.iterdir()):
            if d.is_dir():
                yml = d / "project.yml"
                templates.append({"name": d.name, "has_project_yml": yml.exists()})
    return {"templates": templates}


@router.get("/project/config")
async def get_project_config():
    """Return the current project configuration as JSON."""
    if session.raw_project_yaml is None:
        raise HTTPException(400, "No project loaded.")
    return session.raw_project_yaml


@router.put("/project/config")
async def update_project_config(body: dict[str, Any] = Body(...)):
    """Update the project configuration and persist to disk."""
    if session.raw_project_yaml is None:
        raise HTTPException(400, "No project loaded.")
    with session._lock:
        session.raw_project_yaml.update(body)
    if session.project_path:
        import yaml

        yml_path = Path(session.project_path)
        if yml_path.is_dir():
            yml_path = yml_path / "project.yml"
        with open(yml_path, "w", encoding="utf-8") as fh:
            yaml.dump(session.raw_project_yaml, fh, default_flow_style=False, sort_keys=False)
        from sparc.config.config import load_config

        with session._lock:
            session.project_config = load_config(str(yml_path))
    return {"status": "updated"}
