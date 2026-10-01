"""Server-Sent Event streams (api.md §4, §17; SPEC §5.7).

``GET /api/stream`` - global events with ids ``g:<gseq>``; resumes from
``Last-Event-ID`` / ``?after=g:<gseq>`` out of the 10,000-event ring buffer,
or starts with ``resync {reason: "expired"}``.  A subscriber that falls
1,000 events behind gets ``resync {reason: "overflow"}``.

``GET /api/jobs/{jid}/stream`` - every line of the job's ``events.jsonl``
(``id`` = byte cursor, ``event`` = type, ``data`` = the line plus
``cursor``): backfill from the file after ``after`` / ``Last-Event-ID``,
then the live tail.  Ticks are coalesced to 4 Hz per span on the wire only
(``k == 1`` and ``k == n`` always pass); transient ``resource`` samples
carry no id; ``end {status}`` closes the stream once the job is final and
the file drained; a subscriber that falls 2,000 events behind gets
``resync {after}`` and the stream closes.  Both send ``: ping`` every 15 s.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from sparc.studio.app import StudioContext, get_ctx
from sparc.studio.errors import ApiError
from sparc.studio.events import ALL_TOPICS
from sparc.studio.jobs.tailer import read_events
from sparc.studio.schemas.common import FINAL_STATUSES
from sparc.studio.sse import PING, TickCoalescer, frame, sse_response
from sparc.studio.workspace import read_json

router = APIRouter(tags=["stream"])

_SSE = {200: {"description": "text/event-stream", "content": {"text/event-stream": {}}}}
BACKFILL_BATCH = 2000
IDLE_CHECK_S = 1.0


def _parse_gseq(v: str | None) -> int | None:
    if not v:
        return None
    v = v.strip()
    if v.startswith("g:"):
        v = v[2:]
    try:
        return int(v)
    except ValueError:
        return None


@router.get("/stream", response_class=StreamingResponse, responses=_SSE)
async def global_stream(request: Request, topics: str | None = Query(None, description=",".join(ALL_TOPICS)),
                        after: str | None = Query(None, description="resume after g:<gseq>"),
                        sctx: StudioContext = Depends(get_ctx)):
    hub = sctx.hub
    wanted = [t.strip() for t in (topics or "").split(",") if t.strip()] or None
    since = _parse_gseq(request.headers.get("last-event-id"))
    if since is None:
        since = _parse_gseq(after)
    ping = sctx.config.ping_interval_s
    sub = hub.subscribe_global(wanted)

    async def gen():
        last = -1
        try:
            yield b": connected\n\n"
            if since is not None:
                expired, records = hub.replay(since)
                if expired:
                    yield frame("resync", {"reason": "expired", "gseq": hub.gseq, "ts": round(time.time(), 3)},
                                id=f"g:{hub.gseq}")
                    last = hub.gseq
                else:
                    for rec in records:
                        if sub.wants(rec["type"]):
                            yield frame(rec["type"], rec, id=f"g:{rec['gseq']}")
                        last = rec["gseq"]
            while True:
                if sub.overflowed:
                    sub.overflowed = False
                    yield frame("resync", {"reason": "overflow", "gseq": hub.gseq, "ts": round(time.time(), 3)},
                                id=f"g:{hub.gseq}")
                    last = hub.gseq
                    continue
                rec = await sub.get(timeout=ping)
                if rec is None:
                    if sub.overflowed:
                        continue
                    if sub.closed or hub.closed:
                        while sub.items:                     # deliver what was queued before the close
                            r = sub.items.popleft()
                            if r["gseq"] > last:
                                yield frame(r["type"], r, id=f"g:{r['gseq']}")
                        return
                    yield PING
                    continue
                if rec["gseq"] <= last:
                    continue
                last = rec["gseq"]
                yield frame(rec["type"], rec, id=f"g:{rec['gseq']}")
        finally:
            sub.close()

    return sse_response(gen())


def _events_path(sctx: StudioContext, row: dict) -> Path:
    """The job's event log; a ``run.external`` pseudo-job tails ``run_state.events_path`` of its run."""
    if sctx.jobs is not None:
        return sctx.jobs.events_path(row)
    if row.get("executor") == "external" and row.get("run_id"):
        run = sctx.db.fetchone("SELECT run_dir FROM runs WHERE id = ?", (row["run_id"],))
        if run:
            st = read_json(Path(run["run_dir"]) / "run_state.json", {}) or {}
            if st.get("events_path"):
                return Path(st["events_path"])
    return Path(row["job_dir"]) / "events.jsonl"


@router.get("/jobs/{jid}/stream", response_class=StreamingResponse, responses={**_SSE, 404: {"description": "no job"}})
async def job_stream(jid: str, request: Request, after: int | None = Query(None, description="cursor"),
                     sctx: StudioContext = Depends(get_ctx)):
    row = sctx.db.fetchone("SELECT * FROM jobs WHERE id = ?", (jid,))
    if row is None:
        raise ApiError("not_found", f"no job {jid!r}")
    header = request.headers.get("last-event-id")
    if header is not None and header.strip().lstrip("-").isdigit():
        after = int(header)
    path = _events_path(sctx, row)
    hub = sctx.hub
    ping = sctx.config.ping_interval_s
    sub = hub.subscribe_job(jid)

    def status_of() -> str | None:
        r = sctx.db.fetchone("SELECT status FROM jobs WHERE id = ?", (jid,))
        return r["status"] if r else None

    def live_tailer() -> bool:
        return sctx.jobs is not None and jid in sctx.jobs.tailers

    async def gen():
        last = -1 if after is None else int(after)     # cursor of the last event sent
        pos = last                                      # cursor of the last event examined
        co = TickCoalescer()
        last_write = time.monotonic()

        async def drain():
            nonlocal last, pos
            out = []
            while True:
                evs, _nxt, eof = await asyncio.to_thread(read_events, path, None if pos < 0 else pos,
                                                         limit=BACKFILL_BATCH)
                for ev in evs:
                    pos = max(pos, ev["cursor"])
                    for c, e in co.push(ev["cursor"], ev):
                        out.append(frame(e.get("type") or "log", e, id=c))
                        last = c
                if eof or not evs:
                    break
            return out

        try:
            yield b": connected\n\n"
            for f in await drain():
                yield f
            while True:
                if sub.overflowed:
                    yield frame("resync", {"after": last})
                    return
                if not live_tailer():
                    for f in await drain():
                        yield f
                        last_write = time.monotonic()
                    status = status_of()
                    if status in FINAL_STATUSES and not live_tailer():
                        for f in await drain():
                            yield f
                        for c, e in sorted(co.held.values(), key=lambda x: x[0]):
                            yield frame(e.get("type") or "log", e, id=c)
                        co.held.clear()
                        yield frame("end", {"status": status})
                        return
                due = co.next_due_in()
                timeout = min(IDLE_CHECK_S, max(0.0, ping - (time.monotonic() - last_write)))
                if due is not None:
                    timeout = min(timeout, due)
                item = await sub.get(timeout=max(timeout, 0.01))
                for c, e in co.due():
                    yield frame(e.get("type") or "log", e, id=c)
                    last = max(last, c)
                    last_write = time.monotonic()
                if item is None:
                    if sub.closed or hub.closed:
                        return
                    if time.monotonic() - last_write >= ping:
                        yield PING
                        last_write = time.monotonic()
                    continue
                kind = item[0]
                if kind == "event":
                    cursor, ev = item[1], item[2]
                    if cursor <= pos:
                        continue
                    pos = cursor
                    for c, e in co.push(cursor, {**ev, "cursor": cursor}):
                        yield frame(e.get("type") or "log", e, id=c)
                        last = c
                        last_write = time.monotonic()
                elif kind == "resource":
                    yield frame("resource", item[1])
                    last_write = time.monotonic()
                elif kind == "end":
                    for f in await drain():
                        yield f
                    for c, e in sorted(co.held.values(), key=lambda x: x[0]):
                        yield frame(e.get("type") or "log", e, id=c)
                    co.held.clear()
                    yield frame("end", {"status": status_of() or item[1]})
                    return
        finally:
            sub.close()

    return sse_response(gen())
