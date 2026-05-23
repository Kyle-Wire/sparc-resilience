"""Thread-safe event broadcaster for pipeline WebSocket subscribers.

``EventBroadcaster`` has a narrow interface — ``subscribe`` and ``broadcast``
— that hides all thread-safety and asyncio cross-loop wiring.  Tests can
drive it with plain ``queue.SimpleQueue`` instances; production WebSocket
handlers supply ``asyncio.Queue`` objects with a running loop.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


class EventBroadcaster:
    """Fan-out events to all subscribed queues, thread-safely.

    Each subscriber calls ``subscribe(queue, loop)`` and receives an
    unsubscribe callable.  When ``broadcast(event)`` is called from any
    thread, every subscriber's queue receives the event.

    * If ``loop`` is ``None``, ``put_nowait`` is called directly (works with
      ``queue.SimpleQueue`` for tests or synchronous callers).
    * If ``loop`` is an ``asyncio.AbstractEventLoop``, ``call_soon_threadsafe``
      is used so delivery is safe from a background pipeline thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Each entry: (queue-like, loop-or-None)
        self._subscribers: list[tuple[Any, Any]] = []

    def subscribe(self, q: Any, loop: Any) -> Callable[[], None]:
        """Register ``q`` as a subscriber.

        Returns an unsubscribe callable that removes ``q`` from future
        broadcasts.  Calling unsubscribe more than once is safe.
        """
        entry = (q, loop)
        with self._lock:
            self._subscribers.append(entry)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(entry)
                except ValueError:
                    pass  # already removed — idempotent

        return _unsubscribe

    def broadcast(self, event: dict[str, Any]) -> None:
        """Deliver ``event`` to all current subscribers.

        Failures on individual queues are swallowed so a broken subscriber
        never halts delivery to the others.
        """
        with self._lock:
            snapshot = list(self._subscribers)

        for q, loop in snapshot:
            try:
                if loop is not None and loop.is_running():
                    loop.call_soon_threadsafe(q.put_nowait, event)
                else:
                    q.put_nowait(event)
            except Exception:  # noqa: BLE001
                pass
