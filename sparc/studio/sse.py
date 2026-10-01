"""Server-Sent Events encoding (SPEC §5.7, api.md §4).

Frames are ``id: …\\nevent: <type>\\ndata: <json>\\n\\n``; frames without an
id (transient ``resource`` samples) are never replayed by the browser.
:func:`sse_response` wraps an async generator of frames in a
``StreamingResponse`` with ``Cache-Control: no-store`` and
``X-Accel-Buffering: no``; the generators send a ``: ping`` comment every
15 s of silence.

:class:`TickCoalescer` thins ``tick`` events to 4 Hz per span **on the wire
only** (the file keeps every line).  It never drops a tick with ``k == 1``
or ``k == n``, so unit accounting gives the same result on the wire as on
disk, and it never sends anything out of cursor order: a held tick is
flushed before a later event once its window has opened, otherwise it is
superseded.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

from starlette.responses import StreamingResponse

__all__ = ["SSE_HEADERS", "PING", "frame", "sse_response", "TickCoalescer"]

SSE_HEADERS = {"Cache-Control": "no-store", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
PING = b": ping\n\n"


def _dumps(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str)


def frame(event: str | None, data: Any, id: str | int | None = None) -> bytes:
    """One SSE frame.  ``data`` is JSON-encoded on a single line (JSON never contains raw newlines)."""
    parts = []
    if id is not None:
        parts.append(f"id: {id}\n")
    if event:
        parts.append(f"event: {event}\n")
    parts.append(f"data: {data if isinstance(data, str) else _dumps(data)}\n\n")
    return "".join(parts).encode("utf-8")


def sse_response(gen: AsyncIterator[bytes]) -> StreamingResponse:
    return StreamingResponse(gen, media_type="text/event-stream", headers=dict(SSE_HEADERS))


class TickCoalescer:
    """Per-span 4 Hz limiter for ``tick`` events on one stream.

    Feed every event in cursor order to :meth:`push`; it returns the events
    to send now (possibly a held tick followed by the new event).  Call
    :meth:`due` when idle to release held ticks whose window has opened.
    """

    def __init__(self, interval_s: float = 0.25, clock=time.monotonic):
        self.interval = interval_s
        self.clock = clock
        self.last_sent: dict[Any, float] = {}
        self.held: dict[Any, tuple[int, dict]] = {}      # span → (cursor, event)

    @staticmethod
    def _always(ev: dict) -> bool:
        k, n = ev.get("k"), ev.get("n")
        return k == 1 or (k is not None and k == n)

    def push(self, cursor: int, ev: dict) -> list[tuple[int, dict]]:
        now = self.clock()
        out = self._release(now, before=cursor)
        if ev.get("type") != "tick":
            out.append((cursor, ev))
            return out
        span = ev.get("span")
        if self._always(ev) or now - self.last_sent.get(span, -1e18) >= self.interval:
            self.held.pop(span, None)
            self.last_sent[span] = now
            out.append((cursor, ev))
        else:
            self.held[span] = (cursor, ev)                # supersedes an older held tick of this span
        return out

    def due(self) -> list[tuple[int, dict]]:
        """Held ticks whose window has opened (send in the returned order)."""
        return self._release(self.clock(), before=None)

    def _release(self, now: float, before: int | None) -> list[tuple[int, dict]]:
        """Held ticks that may go out now; when ``before`` is given, every other held tick is dropped
        (it cannot be sent after a later cursor)."""
        if not self.held:
            return []
        out = []
        for span, (cur, ev) in sorted(self.held.items(), key=lambda kv: kv[1][0]):
            if now - self.last_sent.get(span, -1e18) >= self.interval:
                out.append((cur, ev))
                self.last_sent[span] = now
                del self.held[span]
            elif before is not None:
                del self.held[span]
        return out

    def next_due_in(self) -> float | None:
        """Seconds until the earliest held tick may be released (None when nothing is held)."""
        if not self.held:
            return None
        now = self.clock()
        return max(0.0, min(self.interval - (now - self.last_sent.get(s, -1e18)) for s in self.held))
