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

    # ---- Training telemetry patterns ----
    # Capacity sweep result:  "hidden_dim=256: CV R²=0.8912"
    _CAPACITY_RE = re.compile(
        r"hidden_dim=(\d+):\s*CV\s+R[²2]=\s*([\d.]+)"
    )
    # Epoch log:  "Epoch 10/100  loss=0.1234  [mse=0.050 phys=0.020 ...]"
    _EPOCH_RE = re.compile(
        r"(?:Epoch|Retrain|SWA epoch)\s+(\d+)/(\d+)\s+loss=([\d.]+)"
    )
    # Loss components inside brackets: [mse=0.050 phys=0.020 ...]
    _COMPONENTS_RE = re.compile(
        r"\[([\w=.\s]+)\]"
    )
    # Curriculum stage marker: "[CURRICULUM] Stage B: Physics Activation"
    _CURRICULUM_RE = re.compile(
        r"\[CURRICULUM\]\s+(Stage\s+\w+):\s*(.+)"
    )
    # Convergence marker: "[CONVERGENCE] converging" or "[CONVERGENCE] converged"
    _CONVERGENCE_RE = re.compile(
        r"\[CONVERGENCE\]\s+(\w+)"
    )

    # DAG approval gate marker
    _DAG_GATE_RE = re.compile(r"\[DAG_APPROVAL_REQUESTED\]\s*(\{.*\})")

    # Structured model-level markers emitted by enhanced_spatial_cv.py
    _MODEL_START_RE = re.compile(r"\[MODEL_START\]\s+(\w+)\s+\((\d+)/(\d+)\)")
    _MODEL_DONE_RE = re.compile(r"\[MODEL_DONE\]\s+(\w+)\s+\((\d+)/(\d+)\)")

    # Stage 2 model weight map (% of total stage 2 progress)
    _MODEL_WEIGHTS: dict[str, tuple[int, int]] = {
        # model_name → (start_pct, end_pct) within stage 2
        "ols":     (5, 15),
        "gwr":     (15, 35),
        "gwrf":    (35, 55),
        "ggpgam":  (55, 70),
    }

    # Phase-based progress markers (pattern → label displayed in the UI)
    _PHASE_RE: list[tuple[re.Pattern, str]] = [
        # Correlogram
        (re.compile(r"Correlogram Analysis", re.IGNORECASE), "Correlogram analysis"),
        (re.compile(r"Analyzing\s+(\S+)"), "Analyzing variable"),
        (re.compile(r"Pipeline Configuration", re.IGNORECASE), "Pipeline configuration"),
        # GWEN
        (re.compile(r"GWEN Variable Selection", re.IGNORECASE), "GWEN variable selection"),
        (re.compile(r"GWEN SELECTION RATIONALE", re.IGNORECASE), "GWEN results"),
        # Spatial CV
        (re.compile(r"Loading and Preprocessing", re.IGNORECASE), "Loading data"),
        (re.compile(r"Loading Spatial Folds", re.IGNORECASE), "Loading spatial folds"),
        (re.compile(r"Generating Spatial Folds", re.IGNORECASE), "Generating spatial folds"),
        (re.compile(r"Training\s+(\S+)\s+across all folds", re.IGNORECASE), "Training model"),
        (re.compile(r"Training\s+(\S+)\s+on\s+\d+\s+samples", re.IGNORECASE), "Training model"),
        (re.compile(r"completed.*folds successful", re.IGNORECASE), "Model complete"),
        (re.compile(r"Generating OOF predictions", re.IGNORECASE), "OOF predictions"),
        (re.compile(r"Retraining Base Models", re.IGNORECASE), "Retraining base models"),
        (re.compile(r"Spatial autocorrelation analysis", re.IGNORECASE), "Spatial autocorrelation"),
        (re.compile(r"Spatial CV Complete", re.IGNORECASE), "Spatial CV complete"),
        # Neural meta-learner
        (re.compile(r"Neural Meta.?Learner", re.IGNORECASE), "Neural meta-learner"),
        (re.compile(r"Capacity sweep", re.IGNORECASE), "Capacity sweep"),
        # Causal
        (re.compile(r"Causal Validation", re.IGNORECASE), "Causal validation"),
        # Scenarios
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

        # Detect model-level start/done markers for checkpoint progress
        m_start = self._MODEL_START_RE.search(line)
        if m_start:
            model_name = m_start.group(1)
            model_idx = int(m_start.group(2))
            model_total = int(m_start.group(3))
            event["phase"] = f"Training {model_name.upper()}"
            event["model"] = model_name
            event["model_index"] = model_idx
            event["model_total"] = model_total
            w = self._MODEL_WEIGHTS.get(model_name)
            if w:
                event["progress_pct"] = w[0]

        m_done = self._MODEL_DONE_RE.search(line)
        if m_done:
            model_name = m_done.group(1)
            model_idx = int(m_done.group(2))
            model_total = int(m_done.group(3))
            event["phase"] = f"{model_name.upper()} complete"
            event["model"] = model_name
            event["model_index"] = model_idx
            event["model_total"] = model_total
            w = self._MODEL_WEIGHTS.get(model_name)
            if w:
                event["progress_pct"] = w[1]

        # Check for phase markers (used by the frontend progress bar)
        if "phase" not in event:
            for pattern, label in self._PHASE_RE:
                pm = pattern.search(line)
                if pm:
                    event["phase"] = label
                    break

        # ---- Training telemetry events ----

        # Capacity sweep result
        m_cap = self._CAPACITY_RE.search(line)
        if m_cap:
            event["type"] = "capacity_result"
            event["hidden_dim"] = int(m_cap.group(1))
            event["r2"] = float(m_cap.group(2))

        # Epoch / Retrain / SWA epoch update
        m_ep = self._EPOCH_RE.search(line)
        if m_ep:
            event["type"] = "epoch_update"
            event["epoch"] = int(m_ep.group(1))
            event["n_epochs"] = int(m_ep.group(2))
            event["total_loss"] = float(m_ep.group(3))
            # Determine training phase label
            if line.lstrip().startswith("Retrain"):
                event["train_phase"] = "retrain"
            elif line.lstrip().startswith("SWA"):
                event["train_phase"] = "swa"
            else:
                event["train_phase"] = "cv"
            # Parse per-component losses from brackets
            m_comp = self._COMPONENTS_RE.search(line)
            if m_comp:
                components: dict[str, float] = {}
                for pair in m_comp.group(1).split():
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        try:
                            components[k] = float(v)
                        except ValueError:
                            pass
                event["components"] = components

        # Curriculum stage transition
        m_cur = self._CURRICULUM_RE.search(line)
        if m_cur:
            event["type"] = "curriculum_stage"
            event["curriculum"] = m_cur.group(1)
            event["label"] = m_cur.group(2)

        # Convergence status
        m_conv = self._CONVERGENCE_RE.search(line)
        if m_conv:
            event["type"] = "convergence"
            event["status"] = m_conv.group(1).lower()

        # DAG approval gate
        m_gate = self._DAG_GATE_RE.search(line)
        if m_gate:
            import json as _json
            try:
                payload = _json.loads(m_gate.group(1))
            except _json.JSONDecodeError:
                payload = {}
            event["type"] = "dag_approval_requested"
            event["n_edges"] = payload.get("n_edges", 0)
            event["n_nodes"] = payload.get("n_nodes", 0)

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

    try:
        while True:
            event = await queue.get()
            if event is None:
                break

            # Enrich epoch events with ETA
            if event.get("type") == "epoch_update":
                epoch = event.get("epoch", 0)
                n_epochs = event.get("n_epochs", 0)
                if epoch == 1 and n_epochs > 0:
                    eta.start("epoch", n_epochs)
                eta_info = eta.update("epoch", epoch)
                if eta_info:
                    event["eta_seconds"] = eta_info["eta_seconds"]
                    event["elapsed_seconds"] = eta_info["elapsed_seconds"]

            # Log every event to persistent file
            session_log.log(event)
            state.buffer_event(event)
            yield event
    finally:
        session_log.close()


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

    if stage == 0:
        print(">>> Correlogram Analysis")
        from sparc.run.correlogram_analysis import main as run_correlogram
        result = run_correlogram(fast_mode=fast)
        state.store_result(0, result)

        # Pipeline configuration
        print(">>> Pipeline Configuration")
        from sparc.run.pipeline_configurator import PipelineConfigurator
        configurator = PipelineConfigurator(stage1_dir=str(paths.stage0_dir))
        configurator.save_pipeline_config()

    elif stage == 1 and not skip_gwen:
        print(">>> GWEN Variable Selection")
        from sparc.run.gwen_variable_selection import main as run_gwen
        result = run_gwen(config_path=project_path, fast_mode=fast)
        state.store_result(1, result)

    elif stage == 2:
        print(">>> Enhanced Spatial CV")
        from sparc.run.enhanced_spatial_cv import main as run_spatial_cv
        result = run_spatial_cv(fast_mode=fast)
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

        result = run_causal_validation(approval_gate=_dag_approval_gate)
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
