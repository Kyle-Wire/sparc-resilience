"""Run-control and pipeline-streaming routes.

HTTP routes: POST /run/cancel, GET /run/log, GET /run/events,
             GET /runs/discover, GET /runs/diff

WebSocket routes: /run/execute, /run/artifacts, /run/stream
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from sparc.server import deps
from sparc.server.stream import stream_stage

router = APIRouter(tags=["run"])


# ------------------------------------------------------------------
# POST /run/cancel
# ------------------------------------------------------------------

@router.post("/run/cancel")
async def cancel_run():
    """Mark the pipeline as idle so the frontend can restart."""
    deps.state.set_idle()
    return {"status": "cancelled"}


# ------------------------------------------------------------------
# GET /run/log
# ------------------------------------------------------------------

@router.get("/run/log")
async def get_run_log():
    """Return the persistent session log for the current project."""
    if deps.state.project_config is None:
        from fastapi import HTTPException
        raise HTTPException(400, "No project loaded.")

    project_dir = Path(deps.state.project_config["paths"]["project_root"])
    log_path = project_dir / "session_log.jsonl"

    if not log_path.exists():
        return {"entries": [], "path": str(log_path)}

    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return {"entries": entries, "path": str(log_path)}


# ------------------------------------------------------------------
# GET /run/events
# ------------------------------------------------------------------

@router.get("/run/events")
async def get_run_events():
    """Return buffered events for the current (or most recent) pipeline run."""
    return {
        "is_running": deps.state.is_running,
        "current_stage": deps.state.current_stage,
        "events": deps.state.get_buffered_events(),
    }


# ------------------------------------------------------------------
# GET /runs/discover
# ------------------------------------------------------------------

@router.get("/runs/discover")
async def get_runs_discover(search_root: str | None = Query(default=None)):
    """List candidate output directories the user could compare."""
    from sparc.evaluation.diff import discover_runs

    if search_root:
        root = Path(search_root)
    else:
        root = Path.home()
        if deps.state.project_config is not None:
            try:
                from sparc.run.pipeline_paths import PipelinePaths
                paths = PipelinePaths.from_config(deps.state.project_config)
                if paths.output_dir.parent.exists():
                    root = paths.output_dir.parent
            except Exception:
                pass
    return {"search_root": str(root), "runs": discover_runs(root)}


# ------------------------------------------------------------------
# GET /runs/diff
# ------------------------------------------------------------------

@router.get("/runs/diff")
async def get_runs_diff(
    run_a: str = Query(...),
    run_b: str = Query(...),
):
    """Diff two pipeline output directories on disk."""
    from fastapi import HTTPException
    from sparc.evaluation.diff import diff_runs

    try:
        return diff_runs(Path(run_a), Path(run_b))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Diff failed: {exc}")


# ------------------------------------------------------------------
# WebSocket /run/execute
# ------------------------------------------------------------------

@router.websocket("/run/execute")
async def run_execute(ws: WebSocket, token: str = Query(default="")):
    """Execute a pipeline stage and stream structured events over WebSocket."""
    if deps.server_token is not None and token != deps.server_token:
        await ws.close(code=4003)
        return
    await ws.accept()

    try:
        init_msg = await ws.receive_json()
    except WebSocketDisconnect:
        return

    stage_raw = init_msg.get("stage")
    if stage_raw is None:
        await ws.send_json({"type": "error", "message": "Missing required field: stage"})
        await ws.close()
        return

    stage = int(stage_raw)
    fast = bool(init_msg.get("fast", False))
    skip_gwen = bool(init_msg.get("skip_gwen", False))

    if deps.state.project_config is None:
        await ws.send_json({"type": "error", "message": "No project loaded"})
        await ws.close()
        return

    if deps.state.is_running:
        await ws.send_json({
            "type": "error",
            "message": f"Stage {deps.state.current_stage} already running",
        })
        await ws.close()
        return

    artifact_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    unsub = deps.broadcaster.subscribe(artifact_queue, loop=loop)

    async def _drain_artifacts() -> None:
        try:
            while True:
                event = await artifact_queue.get()
                await ws.send_json(event)
        except (asyncio.CancelledError, WebSocketDisconnect):
            pass

    drain_task = asyncio.create_task(_drain_artifacts())
    try:
        async for event in stream_stage(deps.state, stage, fast=fast, skip_gwen=skip_gwen):
            await ws.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        drain_task.cancel()
        unsub()
        if ws.client_state.name != "DISCONNECTED":
            await ws.close()


# ------------------------------------------------------------------
# WebSocket /run/artifacts
# ------------------------------------------------------------------

@router.websocket("/run/artifacts")
async def run_artifacts(ws: WebSocket, token: str = Query(default="")):
    """Subscribe to artifact-written events over WebSocket."""
    if deps.server_token is not None and token != deps.server_token:
        await ws.close(code=4003)
        return
    await ws.accept()

    artifact_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    unsub = deps.broadcaster.subscribe(artifact_queue, loop=loop)
    await ws.send_json({"type": "subscribed", "channel": "artifacts"})
    try:
        while True:
            event = await artifact_queue.get()
            await ws.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        unsub()
        if ws.client_state.name != "DISCONNECTED":
            await ws.close()


# ------------------------------------------------------------------
# WebSocket /run/stream
# ------------------------------------------------------------------

@router.websocket("/run/stream")
async def run_stream(ws: WebSocket, token: str = Query(default="")):
    """Subscribe to live artifact-written events over WebSocket."""
    if deps.server_token is not None and token != deps.server_token:
        await ws.close(code=4003)
        return
    await ws.accept()

    try:
        msg = await ws.receive_json()
    except WebSocketDisconnect:
        return

    channel = msg.get("subscribe", "")
    if channel != "artifacts":
        await ws.close(code=4001)
        return

    await ws.send_json({"type": "subscribed"})

    artifact_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    unsub = deps.broadcaster.subscribe(artifact_queue, loop=loop)
    try:
        while True:
            event = await artifact_queue.get()
            await ws.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        unsub()
        if ws.client_state.name != "DISCONNECTED":
            await ws.close()
