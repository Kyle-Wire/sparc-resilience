"""Thread-safe artifact pub/sub bus for pipeline-to-server fan-out.

``ArtifactBus`` decouples the pipeline's artifact writes from any specific
server infrastructure.  Subscribers register a callable and receive a cancel
token (an unsubscribe callable).  ``publish`` fans out to all live subscribers;
failures in individual handlers are swallowed so one broken subscriber never
halts delivery to the others.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


class ArtifactBus:
    """Fan-out artifact entries to all registered handler callables.

    Usage::

        bus = ArtifactBus()

        def my_handler(entry: dict) -> None:
            print(entry["name"])

        cancel = bus.subscribe(my_handler)
        bus.publish({"name": "cv_result", "path": "/out/cv.json"})
        cancel()  # stop receiving

    Thread-safety: ``subscribe``, ``publish``, and the returned cancel tokens
    are all safe to call from any thread simultaneously.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[dict[str, Any]], None]] = []

    def subscribe(self, handler: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        """Register *handler* to receive future ``publish`` calls.

        Returns a cancel token — a zero-argument callable that removes this
        subscriber.  Calling the token more than once is safe (idempotent).
        """
        with self._lock:
            self._subscribers.append(handler)

        def _cancel() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(handler)
                except ValueError:
                    pass  # already removed — idempotent

        return _cancel

    def publish(self, entry: dict[str, Any]) -> None:
        """Deliver *entry* to all current subscribers.

        Failures in individual handlers are caught and suppressed so that
        one broken subscriber cannot prevent delivery to the others.
        """
        with self._lock:
            snapshot = list(self._subscribers)

        for handler in snapshot:
            try:
                handler(entry)
            except Exception:
                pass  # never raise — broken subscriber is isolated
