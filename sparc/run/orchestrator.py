"""
sparc/run/orchestrator.py — Pipeline orchestration module.

``RunContext`` holds the three pipeline-wide dependencies (config, paths,
registry) in one place.  ``PipelineOrchestrator`` drives stage execution
against an injectable map of stage runners — allowing tests to swap
real stage runners for lightweight fakes without touching disk or env vars.

The existing ``cmd_run`` in ``sparc/__main__.py`` is a thin adapter that
constructs a ``RunContext`` and delegates here; nothing else changes.

Interface
---------
::

    ctx = RunContext(config=config, paths=paths, registry=registry,
                    project_path=project_path)
    orch = PipelineOrchestrator(ctx)
    orch.run("all", fast=True)

    # Tests inject fake runners to avoid disk I/O:
    orch = PipelineOrchestrator(ctx, stage_runners={
        "2": lambda ctx, flags: run_minimal_cv(ctx),
    })
    orch.run("2")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

# Stage execution order — the canonical list of stage keys.
_STAGE_ORDER = ["0", "1", "2", "3", "4", "5"]


# ---------------------------------------------------------------------------
# RunContext
# ---------------------------------------------------------------------------

@dataclass
class RunContext:
    """Single holder for pipeline-wide dependencies.

    Attributes
    ----------
    config : dict
        Loaded project.yml config dict (from ``load_config``).
    paths : PipelinePaths
        Resolved output directories.
    registry : RunRegistry | None
        Active artifact registry for this run.  ``None`` when the registry
        module is unavailable or initialisation failed.
    project_path : str
        Absolute path to the ``project.yml`` used for this run.
    """

    config: Dict[str, Any]
    paths: Any              # PipelinePaths — typed as Any to avoid circular import
    registry: Any           # RunRegistry | None
    project_path: str

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_project(cls, project_path: str) -> "RunContext":
        """Build a ``RunContext`` from a ``project.yml`` path.

        Unlike the construction sequence in ``cmd_run``, this factory **does
        not** mutate any module-level globals (``_global_paths``,
        ``_ACTIVE_REGISTRY``).  It is safe to call from tests and from
        server routes without side-effects.

        Parameters
        ----------
        project_path:
            Absolute or relative path to a ``project.yml`` file.

        Returns
        -------
        RunContext

        Raises
        ------
        FileNotFoundError
            If *project_path* does not exist.
        """
        from pathlib import Path as _Path

        p = _Path(project_path).resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"RunContext.from_project: project.yml not found: {p}"
            )

        from sparc.config.config import load_config
        from sparc.run.pipeline_paths import PipelinePaths

        config = load_config(str(p))
        paths = PipelinePaths.from_config(config)

        registry = None
        try:
            from sparc.registry import RunRegistry
            registry = RunRegistry(paths.output_dir, autoload=True)
        except Exception:
            # Registry is optional; leave as None if unavailable
            pass

        return cls(
            config=config,
            paths=paths,
            registry=registry,
            project_path=str(p),
        )


# ---------------------------------------------------------------------------
# PipelineOrchestrator
# ---------------------------------------------------------------------------

class PipelineOrchestrator:
    """Drive stage execution through injectable stage runners.

    Parameters
    ----------
    context : RunContext
        The pipeline-wide dependencies for this run.
    stage_runners : dict | None
        Optional mapping from stage key (``"0"``–``"5"``) to a callable
        ``(context: RunContext, flags: dict) -> None``.  When *None*, the
        real SPARC stage runners are used (imported lazily).  Pass a custom
        dict in tests to avoid disk I/O and external dependencies.
    """

    def __init__(
        self,
        context: RunContext,
        stage_runners: Optional[Dict[str, Callable]] = None,
    ) -> None:
        self._ctx = context
        self._runners = dict(stage_runners) if stage_runners is not None else {}
        self._completed: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        stage: str,
        *,
        fast: bool = False,
        resume: bool = False,
        skip_gwen: bool = False,
    ) -> None:
        """Execute one stage or all stages.

        Parameters
        ----------
        stage : str
            A single stage key (``"0"``–``"5"``) or ``"all"``.
        fast : bool
            When *True*, passes ``fast=True`` to each stage runner.
        resume : bool
            When *True*, stages already marked complete via
            :meth:`mark_complete` are skipped.
        skip_gwen : bool
            When *True*, stage ``"1"`` (GWEN) is omitted even when
            ``stage="all"``.
        """
        flags = {"fast": fast, "skip_gwen": skip_gwen}

        if stage == "all":
            for key in _STAGE_ORDER:
                if key == "1" and skip_gwen:
                    continue
                self._run_one(key, flags, resume=resume)
        else:
            self._run_one(stage, flags, resume=resume)

    def mark_complete(self, stage_key: str) -> None:
        """Mark a stage as complete (used by ``--resume`` logic and tests)."""
        self._completed.add(stage_key)

    def is_complete(self, stage_key: str) -> bool:
        """Return True if the stage has been marked complete."""
        return stage_key in self._completed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_one(self, key: str, flags: dict, *, resume: bool) -> None:
        """Execute a single stage runner unless resume+already-complete."""
        if resume and key in self._completed:
            return

        runner = self._get_runner(key)
        if runner is None:
            return   # stage not in runner map — silently skip

        runner(self._ctx, flags)
        self._completed.add(key)

    def _get_runner(self, key: str) -> Optional[Callable]:
        """Return the runner for *key*, falling back to real stage imports."""
        if key in self._runners:
            return self._runners[key]
        # Fall back to real pipeline stages (imported lazily so tests
        # that inject all runners never trigger heavy imports).
        return self._real_runner(key)

    @staticmethod
    def _real_runner(key: str) -> Optional[Callable]:
        """Return the real stage runner for *key*, or None if unavailable."""
        _REAL_RUNNERS: Dict[str, str] = {
            "0":  "sparc.run.correlogram_analysis:main",
            "1":  "sparc.run.gwen_variable_selection:main",
            "2":  "sparc.run.enhanced_spatial_cv:main",
            "3":  "sparc.run.causal_validation:main",
        }
        spec = _REAL_RUNNERS.get(key)
        if spec is None:
            return None
        module_path, fn_name = spec.split(":")
        try:
            import importlib
            mod = importlib.import_module(module_path)
            real_fn = getattr(mod, fn_name)
            # Wrap to match (context, flags) signature, passing ctx through
            def _wrapped(ctx: RunContext, flags: dict, _fn=real_fn) -> None:
                _fn(ctx, fast_mode=flags.get("fast", False))
            return _wrapped
        except (ImportError, AttributeError):
            return None
