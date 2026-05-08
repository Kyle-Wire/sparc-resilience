"""Thread-safe server state container for the SPARC FastAPI server."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from sparc.server.result_cache import ResultCache


@dataclass
class ServerState:
    """Holds loaded project data, models, and results across requests.

    All mutations go through the lock so background pipeline threads
    and concurrent HTTP handlers don't corrupt shared state.
    """

    project_path: str | None = None
    project_config: dict | None = None
    raw_project_yaml: dict | None = None  # Original project.yml dict
    data: Any = None  # gpd.GeoDataFrame once loaded
    data_summary: dict | None = None
    results: dict[int, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    registry: Any = None  # sparc.registry.RunRegistry, attached on project load
    is_running: bool = False
    current_stage: int | None = None
    event_buffer: list[dict] = field(default_factory=list)
    # DAG approval gate — set by the streaming layer, waited on by the
    # pipeline thread.  ``pending_mc3`` holds MC³ results for the frontend;
    # ``dag_approved`` is set by POST /dag/approve to unblock the gate.
    pending_mc3: dict | None = None
    dag_approved: threading.Event = field(default_factory=threading.Event, repr=False)
    result_cache: ResultCache = field(default_factory=ResultCache, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_running(self, stage: int) -> None:
        with self._lock:
            self.is_running = True
            self.current_stage = stage
            self.event_buffer.clear()
            self.pending_mc3 = None
            self.dag_approved.clear()

    def set_idle(self) -> None:
        with self._lock:
            self.is_running = False
            self.current_stage = None

    def buffer_event(self, event: dict) -> None:
        """Append an event to the in-memory buffer for reconnecting clients."""
        with self._lock:
            self.event_buffer.append(event)

    def get_buffered_events(self) -> list[dict]:
        """Return a snapshot of all buffered events."""
        with self._lock:
            return list(self.event_buffer)

    def store_result(self, stage: int, result: Any) -> None:
        with self._lock:
            self.results[stage] = result

    def get_result(self, stage: int) -> Any | None:
        with self._lock:
            return self.results.get(stage)

    def clear(self) -> None:
        with self._lock:
            self.project_path = None
            self.project_config = None
            self.data = None
            self.data_summary = None
            self.results.clear()
            self.models.clear()
            self.registry = None
            self.is_running = False
            self.current_stage = None
            self.pending_mc3 = None
            self.dag_approved.clear()
        # Flush result cache outside the main lock (ResultCache is self-locking).
        self.result_cache.clear()
