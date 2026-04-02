"""Thread-safe server state container for the SPARC FastAPI server."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


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
    is_running: bool = False
    current_stage: int | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_running(self, stage: int) -> None:
        with self._lock:
            self.is_running = True
            self.current_stage = stage

    def set_idle(self) -> None:
        with self._lock:
            self.is_running = False
            self.current_stage = None

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
            self.is_running = False
            self.current_stage = None
