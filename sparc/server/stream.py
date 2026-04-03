"""Pipeline event streaming adapter.

Wraps existing SPARC stage executors so that structured JSON events
are yielded instead of writing to stdout.  The FastAPI WebSocket
handler in ``app.py`` pushes these events to the frontend.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import re
import sys
import threading
import time
from typing import Any, AsyncGenerator

from sparc.server.state import ServerState


# ------------------------------------------------------------------
# stdout / stderr capture → structured events
# ------------------------------------------------------------------

class _EventCapture(io.TextIOBase):
    """Captures writes to stdout, parses progress hints, and pushes
    structured events into an ``asyncio.Queue``."""

    # Patterns we try to extract from raw stdout lines
    _STAGE_RE = re.compile(r"Stage\s+(\d+)", re.IGNORECASE)
    _FOLD_RE = re.compile(r"[Ff]old\s+(\d+)")
    _METRIC_RE = re.compile(r"(r2|rmse|mae|mape)\s*[=:]\s*([\d.]+)", re.IGNORECASE)
    _PCT_RE = re.compile(r"(\d{1,3})%")

    # Phase-based progress markers (pattern → label displayed in the UI)
    _PHASE_RE: list[tuple[re.Pattern, str]] = [
        # Stage 0 — Correlogram
        (re.compile(r"Correlogram Analysis", re.IGNORECASE), "Correlogram analysis"),
        (re.compile(r"Analyzing\s+(\S+)"), "Analyzing variable"),
        (re.compile(r"Pipeline Configuration", re.IGNORECASE), "Pipeline configuration"),
        # Stage 1 — GWEN
        (re.compile(r"GWEN Variable Selection", re.IGNORECASE), "GWEN variable selection"),
        (re.compile(r"GWEN SELECTION RATIONALE", re.IGNORECASE), "GWEN results"),
        # Stage 2 — Spatial CV
        (re.compile(r"Loading and Preprocessing", re.IGNORECASE), "Loading data"),
        (re.compile(r"Loading Spatial Folds", re.IGNORECASE), "Loading spatial folds"),
        (re.compile(r"Training\s+(\S+)\s+across all folds", re.IGNORECASE), "Training model"),
        (re.compile(r"Training\s+(\S+)\s+on\s+\d+\s+samples", re.IGNORECASE), "Training model"),
        (re.compile(r"completed.*folds successful", re.IGNORECASE), "Model complete"),
        (re.compile(r"Generating OOF predictions", re.IGNORECASE), "OOF predictions"),
        (re.compile(r"Spatial autocorrelation analysis", re.IGNORECASE), "Spatial autocorrelation"),
        (re.compile(r"Deep Kriging CV", re.IGNORECASE), "Deep Kriging CV"),
        (re.compile(r"Stage 2 Complete", re.IGNORECASE), "Stage 2 complete"),
        # Stage 3 — Causal
        (re.compile(r"Causal Validation", re.IGNORECASE), "Causal validation"),
        # Stage 4 — Scenarios
        (re.compile(r"Scenario Simulation", re.IGNORECASE), "Scenario simulation"),
    ]

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self._queue = queue
        self._loop = loop
        self._buffer = ""

    # io.TextIOBase interface
    def write(self, s: str) -> int:
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self._emit(line)
        return len(s)

    def flush(self) -> None:
        pass

    def _emit(self, line: str) -> None:
        event: dict[str, Any] = {"type": "log", "message": line}

        m = self._STAGE_RE.search(line)
        if m:
            event["stage"] = int(m.group(1))

        m = self._FOLD_RE.search(line)
        if m:
            event["fold"] = int(m.group(1))

        m = self._METRIC_RE.search(line)
        if m:
            event["metric"] = m.group(1).lower()
            event["value"] = float(m.group(2))
            event["type"] = "metric"

        m = self._PCT_RE.search(line)
        if m:
            event["progress_pct"] = int(m.group(1))

        # Check for phase markers (used by the frontend progress bar)
        for pattern, label in self._PHASE_RE:
            pm = pattern.search(line)
            if pm:
                event["phase"] = label
                break

        asyncio.run_coroutine_threadsafe(self._queue.put(event), self._loop)


# ------------------------------------------------------------------
# Public streaming API
# ------------------------------------------------------------------

async def stream_stage(
    state: ServerState,
    stage: int,
    *,
    fast: bool = False,
    skip_gwen: bool = False,
) -> AsyncGenerator[dict, None]:
    """Run a pipeline stage in a background thread and yield structured
    JSON events as they are produced.

    Parameters
    ----------
    state : ServerState
        Shared server state (must have ``project_config`` loaded).
    stage : int
        Pipeline stage number (0-4).
    fast : bool
        Enable fast / reduced-precision mode.
    skip_gwen : bool
        Skip GWEN variable selection (stage 1).

    Yields
    ------
    dict
        Structured event dicts, e.g.
        ``{"type": "metric", "stage": 2, "fold": 3, "metric": "r2", "value": 0.891}``
    """
    if state.project_config is None:
        yield {"type": "error", "message": "No project loaded"}
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    state.set_running(stage)

    def _run() -> None:
        capture = _EventCapture(queue, loop)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = capture  # type: ignore[assignment]
        sys.stderr = capture  # type: ignore[assignment]
        try:
            _execute_stage(state, stage, fast=fast, skip_gwen=skip_gwen)
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "complete", "stage": stage}), loop,
            )
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "error", "stage": stage, "message": str(exc)}), loop,
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            state.set_idle()
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    while True:
        event = await queue.get()
        if event is None:
            break
        yield event


# ------------------------------------------------------------------
# Internal stage dispatch (mirrors __main__.cmd_run logic)
# ------------------------------------------------------------------

def _execute_stage(
    state: ServerState,
    stage: int,
    *,
    fast: bool = False,
    skip_gwen: bool = False,
) -> None:
    """Synchronously execute a single pipeline stage.

    Imports are deferred so the server module itself stays lightweight
    at import time.
    """
    import os
    from pathlib import Path

    config = state.project_config
    assert config is not None

    project_path = state.project_path
    os.environ["SPARC_PROJECT"] = project_path or ""

    # Lazy import to avoid pulling the full dep tree on server start
    from sparc.config.config import load_config  # noqa: F811
    from sparc.run.pipeline_paths import PipelinePaths

    paths = PipelinePaths.from_config(config)

    if stage == 0:
        print(">>> Stage 0: Correlogram Analysis")
        from sparc.run.correlogram_analysis import main as run_correlogram
        run_correlogram(fast_mode=fast)

        # Stage 0b — pipeline configuration
        print(">>> Stage 0b: Pipeline Configuration")
        from sparc.run.pipeline_configurator import PipelineConfigurator
        configurator = PipelineConfigurator(stage1_dir=str(paths.stage1_dir))
        configurator.save_pipeline_config()

    elif stage == 1 and not skip_gwen:
        print(">>> Stage 1: GWEN Variable Selection")
        from sparc.run.gwen_variable_selection import main as run_gwen
        run_gwen(config_path=project_path, fast_mode=fast)

    elif stage == 2:
        print(">>> Stage 2: Enhanced Spatial CV")
        from sparc.run.enhanced_spatial_cv import main as run_spatial_cv
        run_spatial_cv(fast_mode=fast)

    elif stage == 3:
        print(">>> Stage 3: Causal Validation")
        from sparc.run.causal_validation import main as run_causal_validation
        run_causal_validation()

    elif stage == 4:
        scenarios = config.get("scenarios", [])
        if not scenarios:
            print(">>> Stage 4: No scenarios defined — skipping.")
            return
        print(">>> Stage 4: Scenario Simulation")
        from sparc.interventions.scenario_simulator import ScenarioSimulator
        import pandas as pd

        sim = ScenarioSimulator(config)
        sim.load_models()
        csv_path = config["paths"]["raw_csv_path"]
        data = pd.read_csv(csv_path)

        dag_file = config.get("causal", {}).get("dag_file")
        if dag_file and Path(dag_file).exists():
            summary_df, results_gdf = sim.run_with_causal_dag(data, verbose=True)
        else:
            summary_df, results_gdf = sim.run(verbose=True)

        state.store_result(4, {"summary": summary_df, "spatial": results_gdf})
