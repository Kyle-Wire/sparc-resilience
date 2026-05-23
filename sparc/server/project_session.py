"""sparc.server.project_session — session-scoped project state.

Holds the currently loaded project's configuration, data, and registry
under a dedicated lock.  This is intentionally separate from
:class:`~sparc.server.execution_context.ExecutionContext` so that
read-only config lookups do not contend with the streaming event lock.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from fastapi import HTTPException


class ProjectSession:
    """Thread-safe container for loaded-project state.

    All fields start as ``None`` (no project loaded).  Call :meth:`load`
    atomically to populate them, and :meth:`clear` to reset on unload.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.project_path: Optional[str] = None
        self.project_config: Optional[dict] = None
        self.raw_project_yaml: Optional[dict] = None
        self.data: Any = None
        self.data_summary: Optional[dict] = None
        self.registry: Any = None  # sparc.registry.RunRegistry

    def load(
        self,
        *,
        path: str,
        config: dict,
        raw_yaml: dict,
        data: Any,
        data_summary: Optional[dict],
        registry: Any,
    ) -> None:
        """Atomically replace all session fields."""
        with self._lock:
            self.project_path = path
            self.project_config = config
            self.raw_project_yaml = raw_yaml
            self.data = data
            self.data_summary = data_summary
            self.registry = registry

    def clear(self) -> None:
        """Reset all fields to ``None``."""
        with self._lock:
            self.project_path = None
            self.project_config = None
            self.raw_project_yaml = None
            self.data = None
            self.data_summary = None
            self.registry = None

    def get_store(self):
        """Return an :class:`~sparc.registry.store.ArtifactStore` for the active registry.

        Raises :exc:`~fastapi.HTTPException` ``400`` when no project is loaded
        or the registry is unavailable.
        """
        with self._lock:
            registry = self.registry
            config = self.project_config
        if config is None:
            raise HTTPException(400, "No project loaded")
        if registry is None:
            raise HTTPException(400, "No active run registry. Load a project first.")
        from sparc.registry.store import ArtifactStore

        return ArtifactStore(registry)
