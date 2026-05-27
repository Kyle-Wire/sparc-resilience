"""Reproducibility routes — provenance freeze/load/verify."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Query

from sparc.server import deps

router = APIRouter(tags=["reproduce"])


@router.get("/reproduce/provenance")
async def get_provenance(run_dir: str = Query(...)):
    """Return ``run_provenance.json`` for the given run directory (or 404)."""
    from sparc.registry.provenance import load_frozen_config, load_provenance

    p = load_provenance(run_dir)
    if p is None:
        raise HTTPException(404, f"No run_provenance.json under {run_dir}")
    cfg = load_frozen_config(run_dir)
    return {"provenance": p, "frozen_config_present": cfg is not None}


@router.post("/reproduce/freeze")
async def post_reproduce_freeze():
    """Freeze the current project: write run_provenance.json + frozen_config.json."""
    from sparc.registry.provenance import freeze_run

    state = deps.state
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")
    try:
        from sparc.run.pipeline_paths import PipelinePaths

        paths = PipelinePaths.from_config(state.project_config)
        out = paths.output_dir
    except Exception as exc:
        raise HTTPException(500, f"Cannot resolve output_dir: {exc}")
    repo_root = Path(__file__).resolve().parents[3]
    p = freeze_run(out, state.project_config, repo_root=repo_root)
    return {"path": str(p), "output_dir": str(out)}


@router.post("/reproduce/load")
async def post_reproduce_load(payload: dict = Body(...)):
    """Replace the active project_config with the frozen config from a run dir."""
    from sparc.registry.provenance import load_frozen_config, load_provenance

    state = deps.state
    run_dir = payload.get("run_dir")
    if not run_dir:
        raise HTTPException(400, "run_dir required")
    cfg = load_frozen_config(run_dir)
    if cfg is None:
        raise HTTPException(404, f"frozen_config.json not found in {run_dir}")
    state.project_config = cfg
    prov = load_provenance(run_dir)
    return {
        "loaded": True,
        "config_hash": (prov or {}).get("config_hash"),
        "data": (prov or {}).get("data"),
        "warnings": ["Re-run via /run/stream to reproduce. Then call /reproduce/verify."],
    }


@router.get("/reproduce/verify")
async def get_reproduce_verify(
    run_a: str = Query(...),
    run_b: str = Query(...),
):
    """Compare provenance between two runs (config/data/git/env)."""
    from sparc.registry.provenance import compare_runs, verify_sidecars

    cmp = compare_runs(run_a, run_b)
    cmp["sidecars_b"] = verify_sidecars(run_b)
    return cmp
