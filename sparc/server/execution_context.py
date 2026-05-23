"""sparc.server.execution_context — pipeline execution state.

Holds is_running, stage, event buffer, and the DAG-approval gate under
a dedicated lock.  Intentionally separate from
:class:`~sparc.server.project_session.ProjectSession` so config reads
do not block on the streaming-event lock.
"""

from __future__ import annotations

import threading
from typing import Any, Optional


class ExecutionContext:
    """Thread-safe container for active pipeline execution state.

    All fields start idle.  Call :meth:`set_running` at pipeline start
    and :meth:`set_idle` on completion or error.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.is_running: bool = False
        self.current_stage: Optional[int] = None
        self.event_buffer: list[dict] = []
        self.pending_mc3: Optional[dict] = None
        self.dag_approved: threading.Event = threading.Event()

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def set_running(self, stage: int) -> None:
        """Mark pipeline as running; clear the event buffer and DAG gate."""
        with self._lock:
            self.is_running = True
            self.current_stage = stage
            self.event_buffer.clear()
            self.pending_mc3 = None
            self.dag_approved.clear()

    def set_idle(self) -> None:
        """Mark pipeline as idle."""
        with self._lock:
            self.is_running = False
            self.current_stage = None

    # ------------------------------------------------------------------
    # Event buffer
    # ------------------------------------------------------------------

    def buffer_event(self, event: dict) -> None:
        """Append *event* to the in-memory replay buffer."""
        with self._lock:
            self.event_buffer.append(event)

    def get_buffered_events(self) -> list[dict]:
        """Return a snapshot of buffered events (safe to mutate)."""
        with self._lock:
            return list(self.event_buffer)

    # ------------------------------------------------------------------
    # DAG approval gate
    # ------------------------------------------------------------------

    def set_dag_approval(self) -> None:
        """Signal that the user has approved the causal DAG."""
        self.dag_approved.set()

    def wait_for_dag_approval(self, timeout: float) -> bool:
        """Block until DAG is approved or *timeout* seconds elapse.

        Returns ``True`` if approved within the timeout.
        """
        return self.dag_approved.wait(timeout=timeout)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset all execution fields to their initial (idle) state."""
        with self._lock:
            self.is_running = False
            self.current_stage = None
            self.event_buffer.clear()
            self.pending_mc3 = None
            self.dag_approved.clear()
