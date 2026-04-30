"""Regression test: pipeline worker thread must activate the run registry.

The Stage 0–4 writers in :mod:`sparc.run.*` and :mod:`sparc.interventions.*`
all call :func:`sparc.registry.store.get_active_store` and silently no-op
when it returns ``None``. The desktop server runs ``_execute_stage`` on a
background thread; if that thread does not push the active registry,
every artifact write is dropped and the Results page shows nothing.

This test mocks every stage entry point with a probe that records the
active store at call time, drives ``stream_stage`` for stages 0–4, and
asserts the probe saw a live ``ArtifactStore`` bound to ``state.registry``.
It also asserts that endpoint handlers wiping the global
``_ACTIVE_REGISTRY`` (via ``set_active_registry(None)``) on a different
thread cannot poison the worker thread's view.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pandas as pd
import pytest

from sparc.registry.run_registry import (
    RunRegistry,
    get_active_registry,
    pop_active_registry,
    push_active_registry,
    set_active_registry,
)
from sparc.registry.store import ArtifactStore, get_active_store
from sparc.server import app as app_module
from sparc.server import stream as stream_module
from sparc.server.stream import stream_stage


@pytest.fixture()
def attached_state(tmp_path, monkeypatch):
    """Wire a real ``RunRegistry`` + minimal project config onto ``state``."""
    reg = RunRegistry(tmp_path, autoload=False)
    monkeypatch.setattr(app_module.state, "registry", reg, raising=False)
    monkeypatch.setattr(
        app_module.state,
        "project_config",
        {
            "output": {"base_dir": str(tmp_path), "disk_writes": False},
            "paths": {
                "project_root": str(tmp_path),
                "output_dir": str(tmp_path),
                "raw_csv_path": str(tmp_path / "data.csv"),
            },
            "scenarios": [],
            "causal": {},
            "pipeline": {"random_seed": 42},
            "project": {"name": "pytest"},
        },
        raising=False,
    )
    monkeypatch.setattr(
        app_module.state, "project_path", str(tmp_path / "project.yml"), raising=False,
    )
    # ``_run`` calls state.set_running / set_idle / store_result; the
    # ServerState class supports these without further setup.
    yield reg
    # Defensive: the worker thread should have already cleaned up, but
    # in the event of a test failure we want a known-good baseline.
    pop_active_registry()
    set_active_registry(None)


# ---------------------------------------------------------------------------
# Unit: thread-local push/pop isolates worker thread from concurrent unset
# ---------------------------------------------------------------------------


def test_push_active_registry_is_thread_local(tmp_path):
    """A concurrent ``set_active_registry(None)`` must not poison the worker."""
    reg = RunRegistry(tmp_path, autoload=False)

    saw_store: dict[str, ArtifactStore | None] = {}
    started = threading.Event()
    can_check = threading.Event()

    def worker():
        push_active_registry(reg)
        try:
            started.set()
            # Wait until the main thread has had a chance to clobber
            # the process-wide global.
            assert can_check.wait(timeout=2.0)
            saw_store["worker"] = get_active_store()
        finally:
            pop_active_registry()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    assert started.wait(timeout=2.0)
    # Simulate a ``/results/*`` endpoint's ``finally: set_active_registry(None)``
    # running on the request thread.
    set_active_registry(None)
    can_check.set()
    t.join(timeout=2.0)

    assert saw_store["worker"] is not None, (
        "worker thread lost its registry when another thread cleared the global"
    )
    assert isinstance(saw_store["worker"], ArtifactStore)


def test_get_active_registry_thread_local_takes_precedence(tmp_path):
    """``push_active_registry`` overrides the process-wide default."""
    default_reg = RunRegistry(tmp_path / "default", autoload=False)
    override_reg = RunRegistry(tmp_path / "override", autoload=False)
    set_active_registry(default_reg)
    try:
        assert get_active_registry() is default_reg
        push_active_registry(override_reg)
        try:
            assert get_active_registry() is override_reg
        finally:
            pop_active_registry()
        assert get_active_registry() is default_reg
    finally:
        set_active_registry(None)


# ---------------------------------------------------------------------------
# Integration: stream_stage activates the registry on the worker thread
# ---------------------------------------------------------------------------


async def _drain(stage: int, attached_state) -> list[dict]:
    events = []
    async for evt in stream_stage(attached_state, stage, fast=True):
        events.append(evt)
    return events


def _probe_factory(captured: dict, key: str):
    """Return a stage-main replacement that records the active store."""

    def probe(*args, **kwargs):
        store = get_active_store()
        captured[key] = store
        # Mimic the real stage writers: write a marker artifact so we can
        # also assert it lands in the manifest end-to-end.
        if store is not None:
            store.write_struct(
                stage=key,
                artifact_id=f"probe_marker_stage_{key}",
                payload={"ok": True, "stage": key},
                producer="test_probe",
            )
        return {"probe_stage": key}

    return probe


@pytest.mark.parametrize("stage", [0, 1, 2, 3, 4])
def test_stream_stage_activates_registry_for_each_stage(
    stage, attached_state, monkeypatch,
):
    captured: dict = {}

    if stage == 0:
        # Stage 0 calls correlogram_analysis.main + pipeline_configurator;
        # we replace correlogram_analysis.main with the probe and stub the
        # configurator to a no-op.
        import sparc.run.correlogram_analysis as mod
        monkeypatch.setattr(mod, "main", _probe_factory(captured, "0"))
        import sparc.run.pipeline_configurator as cfg_mod

        class _StubConfigurator:
            def __init__(self, *a, **kw): pass
            def save_pipeline_config(self): return None

        monkeypatch.setattr(cfg_mod, "PipelineConfigurator", _StubConfigurator)
    elif stage == 1:
        import sparc.run.gwen_variable_selection as mod
        monkeypatch.setattr(mod, "main", _probe_factory(captured, "1"))
    elif stage == 2:
        import sparc.run.enhanced_spatial_cv as mod
        monkeypatch.setattr(mod, "main", _probe_factory(captured, "2"))
    elif stage == 3:
        import sparc.run.causal_validation as mod

        def _probe_with_gate(*args, approval_gate=None, **kwargs):
            return _probe_factory(captured, "3")()

        monkeypatch.setattr(mod, "main", _probe_with_gate)
    elif stage == 4:
        # Stage 4 short-circuits when ``scenarios`` is empty, so add one and
        # stub the simulator out.
        attached_state_cfg = app_module.state.project_config
        attached_state_cfg["scenarios"] = [{"name": "probe_scenario"}]

        # Provide a tiny on-disk CSV so pd.read_csv doesn't crash.
        csv_path = Path(attached_state_cfg["paths"]["raw_csv_path"])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"lon": [0.0], "lat": [0.0], "y": [1.0]}).to_csv(
            csv_path, index=False,
        )

        import sparc.interventions.scenario_simulator as mod

        class _ProbeSim:
            def __init__(self, *a, **kw):
                pass

            def load_models(self):
                return None

            def run(self, *a, **kw):
                _probe_factory(captured, "4")()
                return pd.DataFrame(), None

            def run_with_causal_dag(self, *a, **kw):
                return self.run(*a, **kw)

        monkeypatch.setattr(mod, "ScenarioSimulator", _ProbeSim)

    events = asyncio.run(_drain(stage, app_module.state))

    key = str(stage)
    assert key in captured, (
        f"stage {stage} probe was not invoked (events={events!r})"
    )
    assert captured[key] is not None, (
        f"stage {stage} ran but get_active_store() returned None — "
        "the worker thread did not activate the registry"
    )
    assert isinstance(captured[key], ArtifactStore)

    # The probe writes a marker artifact; confirm it landed in the manifest
    # via the same registry the test attached.
    marker = attached_state.manifest.lookup(key, f"probe_marker_stage_{stage}")
    assert marker is not None, (
        f"stage {stage} probe wrote to a store that is not the attached "
        "registry — push_active_registry did not bind to state.registry"
    )

    # Worker thread must restore the global on completion.
    assert get_active_registry() is None or get_active_registry() is attached_state, (
        "worker thread leaked its thread-local override to the global"
    )
