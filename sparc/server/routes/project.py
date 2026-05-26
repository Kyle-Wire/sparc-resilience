"""sparc.server.routes.project — /project/* endpoints.

Migrated from app.py.  Importable independently of the full FastAPI app.
Routes use ``sparc.server.deps.session`` for project state.

Currently migrated:
  - /project/load
  - /project/validate
  - /project/init
  - /project/create
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

    return {
        "status": "loaded",
        "project": raw_yaml.get("project", {}),
        "columns": [],
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
    if not crs.get("input") or not crs.get("projected"):
        raise HTTPException(400, "crs.input and crs.projected are required")

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
    cfg_crs["projected"] = crs["projected"]
    cfg["crs"] = cfg_crs

    with open(yml_path, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)

    return {
        "status": "created",
        "template": template,
        "path": str(dest),
        "project_yml": str(yml_path),
    }


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
