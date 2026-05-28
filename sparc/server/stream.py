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
import logging
import re
import sys
import threading
import time
from typing import Any, AsyncGenerator

from sparc.server.event_classifier import classify
from sparc.server.state import ServerState


# ------------------------------------------------------------------
# stdout / stderr capture → structured events
# ------------------------------------------------------------------

class _EventCapture(io.TextIOBase):
    """Thin stdout/stderr adapter.

    Captures writes line-by-line, delegates classification to
    ``sparc.server.event_classifier.classify``, and pushes the
    resulting PipelineEvent dicts into an ``asyncio.Queue``.

    All classification logic (regex patterns, checkpoint list, model
    weight map, ETA) now lives in ``event_classifier.py`` where it
    can be tested as a pure function without asyncio or I/O.
    """

    # Keep _STAGE_RE only for the legacy ``_handle_dag_gate`` path below
    _STAGE_RE = re.compile(r"Stage\s+(\d+)", re.IGNORECASE)
    # Defensive: keep _FOLD_RE as class attr so any code that accesses
    # self._FOLD_RE (e.g. from an older installed package) doesn't raise
    # AttributeError; actual classification is done via classify() below.
    _FOLD_RE = re.compile(r"[Ff]old\s+(\d+)")

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
        event = classify(line)
        # Skip empty results (e.g. line was only whitespace after \r strip)
        if not event.get("message") and event.get("type") == "log":
            return
        asyncio.run_coroutine_threadsafe(self._queue.put(event), self._loop)


# ------------------------------------------------------------------
# Session logger — persistent JSONL log file
# ------------------------------------------------------------------

class SessionLogger:
    """Writes every pipeline event to a persistent JSONL file.

    Each line is a timestamped JSON object for post-hoc debugging
    and audit trails.
    """

    def __init__(self, log_path: str | None):
        self._path = log_path
        self._fh = None
        if log_path:
            from pathlib import Path
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(log_path, "a", encoding="utf-8")

    def log(self, event: dict) -> None:
        """Append an event to the log file."""
        if self._fh is None:
            return
        from datetime import datetime, timezone
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        self._fh.write(json.dumps(entry, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


# ------------------------------------------------------------------
# ETA estimator
# ------------------------------------------------------------------

class _ETAEstimator:
    """Tracks throughput and estimates remaining time for iterative processes."""

    def __init__(self):
        self._start_times: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._totals: dict[str, int] = {}

    def start(self, key: str, total: int) -> None:
        """Begin tracking a process with a known total."""
        self._start_times[key] = time.time()
        self._counts[key] = 0
        self._totals[key] = total

    def update(self, key: str, current: int) -> dict | None:
        """Update progress and return ETA info if trackable."""
        if key not in self._start_times:
            return None
        self._counts[key] = current
        total = self._totals.get(key, 0)
        if current <= 0 or total <= 0:
            return None
        elapsed = time.time() - self._start_times[key]
        rate = current / elapsed if elapsed > 0 else 0
        remaining = (total - current) / rate if rate > 0 else 0
        return {
            "eta_seconds": round(remaining, 1),
            "elapsed_seconds": round(elapsed, 1),
            "rate": round(rate, 2),
            "progress_fraction": round(current / total, 4),
        }

    def clear(self, key: str) -> None:
        self._start_times.pop(key, None)
        self._counts.pop(key, None)
        self._totals.pop(key, None)


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

    # Set up persistent session log
    log_path = None
    project_root = state.project_config.get("paths", {}).get("project_root")
    if project_root:
        from pathlib import Path
        log_path = str(Path(project_root) / "session_log.jsonl")

    # ETA estimator instance (shared across the run)
    eta = _ETAEstimator()

    # Memory watchdog: surface warnings/aborts as structured log events so the
    # desktop log panel renders them. The pipeline body cooperates via
    # `_memory_checkpoint` calls between stages.
    try:
        from sparc.config.hardware_profile import detect_profile
        from sparc.run.memory_watchdog import (
            MemoryWatchdog,
            MemoryPressureAbort,
            WatchdogEvent,
        )
    except Exception:
        detect_profile = None  # type: ignore[assignment]
        MemoryWatchdog = None  # type: ignore[assignment]
        MemoryPressureAbort = RuntimeError  # type: ignore[assignment,misc]
        WatchdogEvent = None  # type: ignore[assignment]

    if detect_profile is not None:
        try:
            _profile = detect_profile()
            await queue.put({
                "type": "hardware",
                "stage": stage,
                "profile": _profile.as_dict(),
                "message": _profile.banner(),
            })
        except Exception:
            pass

    def _watchdog_callback(event) -> None:
        try:
            asyncio.run_coroutine_threadsafe(
                queue.put({
                    "type": "memory_pressure",
                    "stage": stage,
                    "level": event.kind,
                    "percent": event.percent,
                    "message": event.message,
                }),
                loop,
            )
        except Exception:  # pragma: no cover - best effort
            pass

    watchdog = None
    if MemoryWatchdog is not None:
        watchdog = MemoryWatchdog(
            stage_name=f"stage{stage}",
            on_event=_watchdog_callback,
        )
        watchdog.start()

    def _run() -> None:
        capture = _EventCapture(queue, loop)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = capture  # type: ignore[assignment]
        sys.stderr = capture  # type: ignore[assignment]

        # Bridge Python logging → capture so logger.info() calls in
        # the training pipeline appear in the event stream.
        log_handler = logging.StreamHandler(capture)
        log_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)

        # Emit stage status: running
        asyncio.run_coroutine_threadsafe(
            queue.put({
                "type": "stage_status",
                "stage": stage,
                "status": "running",
                "started_at": time.time(),
            }),
            loop,
        )

        try:
            _execute_stage(state, stage, fast=fast, skip_gwen=skip_gwen)
            asyncio.run_coroutine_threadsafe(
                queue.put({
                    "type": "stage_status",
                    "stage": stage,
                    "status": "complete",
                    "completed_at": time.time(),
                }),
                loop,
            )
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "complete", "stage": stage}), loop,
            )
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            asyncio.run_coroutine_threadsafe(
                queue.put({
                    "type": "stage_status",
                    "stage": stage,
                    "status": "failed",
                    "error": str(exc),
                    "traceback": tb[-500:],  # last 500 chars of traceback
                }),
                loop,
            )
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "error", "stage": stage, "message": str(exc)}), loop,
            )
        finally:
            root_logger.removeHandler(log_handler)
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            state.set_idle()
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    session_log = SessionLogger(log_path)
    # Track epoch emission cadence per training phase for throttling
    _epoch_emit_every: dict[str, int] = {}

    try:
        while True:
            event = await queue.get()
            if event is None:
                break

            # Enrich epoch events with ETA
            if event.get("type") == "epoch_update":
                epoch = event.get("epoch", 0)
                n_epochs = event.get("n_epochs", 0)
                phase = event.get("train_phase", "cv")
                if epoch == 1 and n_epochs > 0:
                    eta.start("epoch", n_epochs)
                    # Compute emit cadence: every 25 epochs (4 updates per 100-epoch run)
                    _epoch_emit_every[phase] = max(1, n_epochs // 4)
                eta_info = eta.update("epoch", epoch)
                if eta_info:
                    event["eta_seconds"] = eta_info["eta_seconds"]
                    event["elapsed_seconds"] = eta_info["elapsed_seconds"]

                # Throttle: only yield every Nth epoch (or first/last)
                cadence = _epoch_emit_every.get(phase, 1)
                is_last = (epoch == n_epochs)
                if epoch % cadence != 0 and not is_last:
                    # Still log to file and buffer, but don't stream to frontend
                    session_log.log(event)
                    state.buffer_event(event)
                    continue

            # Log every event to persistent file
            session_log.log(event)
            state.buffer_event(event)
            yield event
    finally:
        session_log.close()
        if watchdog is not None:
            watchdog.stop()


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

    # ---- Guard against WinError 5 on Windows sidecar/service ----
    # When launched as a sidecar the USERPROFILE may resolve to
    # C:\WINDOWS\system32\config\systemprofile which is not writable.
    # Redirect to the project directory (or temp) so libraries that
    # call os.path.expanduser() or appdirs don't fail.
    _sys_prefix = os.path.join(os.environ.get("SYSTEMROOT", r"C:\WINDOWS"), "system32", "config")
    _profile = os.environ.get("USERPROFILE", "")
    if _profile.startswith(_sys_prefix) or not _profile:
        fallback_home = str(Path(project_path).parent) if project_path else os.environ.get("TEMP", "")
        os.environ["USERPROFILE"] = fallback_home
        os.environ["HOME"] = fallback_home
        os.environ["LOCALAPPDATA"] = os.path.join(fallback_home, ".local")
        print(f"[SPARC] Redirected USERPROFILE from system path to: {fallback_home}")
    # ---- End guard ----

    # Lazy import to avoid pulling the full dep tree on server start
    from sparc.config.config import load_config  # noqa: F811
    from sparc.run.pipeline_paths import PipelinePaths

    paths = PipelinePaths.from_config(config)

    # Defensive: ensure the process-global "active" registry points at this
    # run's RunRegistry so pipeline writers (`get_active_store()`) can persist
    # artifacts to artifacts.db. `_attach_registry()` in app.py also sets
    # this at /project/load time; we re-set here in case anything cleared it.
    if state.registry is not None:
        try:
            from sparc.registry.run_registry import set_active_registry
            set_active_registry(state.registry)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not (re)set active registry: {exc}")

    from sparc.run.orchestrator import RunContext
    _run_context = RunContext(
        config=config,
        paths=paths,
        registry=state.registry,
        project_path=project_path or "",
    )

    if stage == 0:
        print(">>> Correlogram Analysis")
        from sparc.run.correlogram_analysis import main as run_correlogram
        result = run_correlogram(_run_context, fast_mode=fast)
        state.store_result(0, result)

        # Pipeline configuration
        print(">>> Pipeline Configuration")
        from sparc.run.pipeline_configurator import PipelineConfigurator
        configurator = PipelineConfigurator(stage1_dir=str(paths.stage0_dir), config=config)
        configurator.save_pipeline_config()

    elif stage == 1 and not skip_gwen:
        print(">>> GWEN Variable Selection")
        from sparc.run.gwen_variable_selection import main as run_gwen
        result = run_gwen(_run_context, fast_mode=fast)
        state.store_result(1, result)

    elif stage == 2:
        print(">>> Enhanced Spatial CV")
        from sparc.run.cv_engine import SpatialCVEngine
        result = SpatialCVEngine(_run_context, fast=fast).run()
        state.store_result(2, result)

    elif stage == 3:
        print(">>> Causal Validation")
        from sparc.run.causal_validation import main as run_causal_validation

        def _dag_approval_gate(mc3_payload: dict) -> None:
            """Block the pipeline thread until the user approves the DAG."""
            import json as _json

            state.pending_mc3 = mc3_payload

            # Build a compact edge list for the WebSocket event
            node_names = mc3_payload.get("node_names", [])
            median_dag = mc3_payload.get("median_dag", {})
            edges = median_dag.get("edges", [])

            print(
                f"[DAG_GATE] MC³ complete — {len(edges)} edges above 0.50.  "
                "Awaiting user approval..."
            )
            # Emit structured event (captured by _EventCapture's write())
            print(
                "[DAG_APPROVAL_REQUESTED] "
                + _json.dumps({"n_edges": len(edges), "n_nodes": len(node_names)})
            )
            # Block until POST /dag/approve sets the event
            state.dag_approved.wait()

        result = run_causal_validation(_run_context, approval_gate=_dag_approval_gate)
        state.store_result(3, result)

    elif stage == 4:
        scenarios = config.get("scenarios", [])
        if not scenarios:
            print(">>> Stage 4: No scenarios defined — skipping.")
            return
        print(">>> Scenario Simulation")
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

    # Post-stage: scan output dirs for any artifacts produced by legacy
    # disk-fallback writers and register them so the desktop's /results/*
    # endpoints see them. Mirrors `__main__.py::_rescan_registry`.
    if state.registry is not None:
        try:
            stage_label = str(stage)
            try:
                state.registry.start_stage(stage_label)
            except Exception:  # noqa: BLE001
                pass
            n = state.registry.migrate_from_disk(paths)
            try:
                state.registry.complete_stage(stage_label, status="complete")
            except Exception:  # noqa: BLE001
                pass
            if n:
                print(f"[registry] stage {stage_label}: +{n} artifact(s) imported from disk")
        except Exception as exc:  # noqa: BLE001
            print(f"[registry] post-stage rescan failed: {exc}")
