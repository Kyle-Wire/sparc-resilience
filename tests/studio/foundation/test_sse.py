"""SSE streams (SPEC §5.7, api.md §4/§17) over a real socket: byte cursors, reconnects, overflow, coalescing,
resource events and ring-buffer expiry."""

from __future__ import annotations

import time

import pytest

from sparc.studio.sse import TickCoalescer
from tests.studio.conftest import read_jsonl, sse_events, wait_for

FINAL = ("succeeded", "failed", "cancelled", "interrupted")


def read_stream(client, url, *, headers=None, until=None, max_events=100_000, timeout=60.0):
    """Collect SSE events until the server closes the stream, ``until(event)`` is true, or ``max_events``."""
    out = []
    deadline = time.monotonic() + timeout
    with client.stream("GET", url, headers=headers or {}) as r:
        assert r.status_code == 200, r.read()
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["cache-control"] == "no-store" and r.headers["x-accel-buffering"] == "no"
        buf = []
        for line in r.iter_lines():
            buf.append(line)
            if line == "":
                evs = sse_events(buf)
                buf = []
                out.extend(evs)
                if evs and (until is not None and until(evs[-1]) or len(out) >= max_events):
                    break
            if time.monotonic() > deadline:
                raise AssertionError(f"stream {url} did not finish in {timeout} s")
    return out


def run_job(http, kind="test.events", **params):
    r = http.post("/api/jobs", json={"kind": kind, "params": params})
    assert r.status_code == 202, r.text
    return r.json()["id"]


def wait_final(http, jid, timeout=60):
    return wait_for(lambda: http.get(f"/api/jobs/{jid}").json()["status"] in FINAL, timeout, what=f"{jid} to end")


def line_offsets(path):
    return [c for c, _ in read_jsonl(path)]


@pytest.fixture
def server(live_server):
    return live_server(sample_interval_s=0.2)


def test_job_stream_ordered_byte_cursors(server):
    http = server.client()
    jid = run_job(http, seconds=0.5)
    wait_final(http, jid)
    evs = read_stream(http, f"/api/jobs/{jid}/stream")
    assert evs[-1]["event"] == "end" and evs[-1]["data"]["status"] == "succeeded" and "id" not in evs[-1]
    body = [e for e in evs if "id" in e]
    ids = [int(e["id"]) for e in body]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    offsets = line_offsets(server.ctx.workspace.job_dir(jid) / "events.jsonl")
    assert set(ids) <= set(offsets)                                # every id is the byte offset of a line
    for e in body:
        assert e["data"]["cursor"] == int(e["id"]) and e["event"] == e["data"]["type"]
    file_events = read_jsonl(server.ctx.workspace.job_dir(jid) / "events.jsonl")
    non_tick = [c for c, ev in file_events if ev["type"] != "tick"]
    assert [i for i in ids if i in set(non_tick)] == non_tick     # nothing but ticks may be thinned
    kn = [c for c, ev in file_events if ev["type"] == "tick" and ev["k"] in (1, ev["n"])]
    assert set(kn) <= set(ids)                                     # k == 1 and k == n always reach the wire


def test_last_event_id_reconnect_has_no_gaps_or_duplicates(server):
    http = server.client()
    jid = run_job(http, seconds=3.0)
    first = read_stream(http, f"/api/jobs/{jid}/stream", max_events=12)
    last_id = [e for e in first if "id" in e][-1]["id"]
    second = read_stream(http, f"/api/jobs/{jid}/stream", headers={"Last-Event-ID": last_id})
    assert second[-1]["event"] == "end"
    ids = [int(e["id"]) for e in first + second if "id" in e]
    assert len(ids) == len(set(ids)) and ids == sorted(ids)
    file_events = read_jsonl(server.ctx.workspace.job_dir(jid) / "events.jsonl")
    must = [c for c, ev in file_events if ev["type"] != "tick" or ev["k"] in (1, ev["n"])]
    assert [i for i in ids if i in set(must)] == must              # no gap across the reconnect
    # ?after= behaves like Last-Event-ID
    third = read_stream(http, f"/api/jobs/{jid}/stream?after={last_id}")
    assert [e.get("id") for e in third if "id" in e] == [e.get("id") for e in second if "id" in e]


def test_snapshot_cursor_then_stream(server):
    http = server.client()
    jid = run_job(http, seconds=2.0)
    wait_for(lambda: http.get(f"/api/jobs/{jid}/tracker").json()["cursor"] > 2000, 30, what="progress")
    snap = http.get(f"/api/jobs/{jid}/tracker").json()
    evs = read_stream(http, f"/api/jobs/{jid}/stream?after={snap['cursor']}")
    ids = [int(e["id"]) for e in evs if "id" in e]
    assert ids and min(ids) > snap["cursor"]
    file_events = read_jsonl(server.ctx.workspace.job_dir(jid) / "events.jsonl")
    must = [c for c, ev in file_events if c > snap["cursor"] and (ev["type"] != "tick" or ev["k"] in (1, ev["n"]))]
    assert [i for i in ids if i in set(must)] == must


def test_tick_coalescing_on_the_wire(server):
    """A writer ticking one span at 20 Hz (e.g. pool workers sharing their parent span): the file keeps every
    line, the wire carries ≤ 4 Hz for that span plus every k == 1 / k == n tick."""
    import json
    import os
    import threading

    http = server.client()
    jid = run_job(http, kind="test.sleep", seconds=60)
    wait_for(lambda: http.get(f"/api/jobs/{jid}").json()["status"] == "running", 30, what="running")
    path = server.ctx.workspace.job_dir(jid) / "events.jsonl"
    n = 40

    def writer():
        fd = os.open(path, os.O_WRONLY | os.O_APPEND)
        try:
            for k in range(1, n + 1):
                ev = {"v": 1, "type": "tick", "seq": k, "ts": round(time.time(), 3), "t_rel": 0.0, "pid": 1,
                      "job": jid, "lvl": "info", "span": "pool:1", "parent": None, "path": ["task:pool"], "ctx": {},
                      "k": k, "n": n, "unit": "replicates", "frac": k / n, "label": ""}
                os.write(fd, (json.dumps(ev) + "\n").encode())
                time.sleep(0.05)
        finally:
            os.close(fd)
        http.post(f"/api/jobs/{jid}/cancel")

    th = threading.Thread(target=writer)
    th.start()
    evs = read_stream(http, f"/api/jobs/{jid}/stream")
    th.join()
    ticks = [e["data"] for e in evs if e.get("event") == "tick" and e["data"]["span"] == "pool:1"]
    file_ticks = [ev for _, ev in read_jsonl(path) if ev["type"] == "tick" and ev["span"] == "pool:1"]
    assert len(file_ticks) == n                                       # disk keeps every line
    # the writer aims for 2 s; a loaded runner sleeps longer, so bound by the time the ticks actually took
    span_s = max(file_ticks[-1]["ts"] - file_ticks[0]["ts"], 0.05 * (n - 1))
    assert len(ticks) < min(n, 4 * span_s + 4)                         # the wire is thinned to ≤ 4 Hz …
    assert ticks[0]["k"] == 1 and ticks[-1]["k"] == n                 # … but never loses k == 1 or k == n
    mid = [t["ts"] for t in ticks if t["k"] not in (1, n)]
    assert len(mid) <= 4 * span_s + 2
    # released ≥ 250 ms apart on arrival; the recorded ts can differ by the tailer's poll jitter
    assert all(b - a >= 0.1 for a, b in zip(mid, mid[1:]))
    assert evs[-1]["event"] == "end"


def test_tick_coalescer_rules():
    now = [0.0]
    co = TickCoalescer(0.25, clock=lambda: now[0])
    t = lambda c, k, n=10, span="s": (c, {"type": "tick", "span": span, "k": k, "n": n})   # noqa: E731
    assert co.push(*t(0, 1)) == [t(0, 1)]                            # k == 1 always
    assert co.push(*t(1, 2)) == []                                   # held
    assert co.push(*t(2, 3)) == []                                   # supersedes the held one
    now[0] = 0.3
    assert co.due() == [t(2, 3)]
    assert co.push(*t(3, 4)) == []
    assert co.push(4, {"type": "metric"}) == [(4, {"type": "metric"})]   # a later event drops the held tick
    assert co.held == {}
    assert co.push(*t(5, 10)) == [t(5, 10)]                          # k == n always
    assert co.push(*t(6, 5, span="other")) == [t(6, 5, span="other")]


def test_resource_events_have_no_id(server):
    http = server.client()
    jid = run_job(http, kind="test.sleep", seconds=2.0)
    evs = read_stream(http, f"/api/jobs/{jid}/stream")
    res = [e for e in evs if e.get("event") == "resource"]
    assert res, "no resource sample arrived"
    for e in res:
        assert "id" not in e
        assert {"rss_mb", "cpu_pct", "n_procs", "threads", "ts"} <= set(e["data"])
        assert e["data"]["rss_mb"] > 0 and e["data"]["n_procs"] >= 1
    peak = server.ctx.db.fetchval("SELECT peak_rss_mb FROM jobs WHERE id = ?", (jid,))
    assert peak >= max(e["data"]["rss_mb"] for e in res)            # jobs.peak_rss_mb is maintained
    assert server.ctx.db.fetchval("SELECT COUNT(*) FROM resource_samples WHERE job_id = ?", (jid,)) >= 1


def test_job_stream_overflow_sends_resync_and_closes(server):
    http = server.client()
    hub = server.ctx.hub
    hub.job_queue = 5
    http.post("/api/queue/pause")
    jid = run_job(http, kind="test.sleep", seconds=1)                 # queued: nothing tails it yet
    got = []

    def flood():
        wait_for(lambda: hub.has_job_subscribers(jid), 10, what="the subscriber")
        items = [("event", 10_000 + i, {"type": "log", "msg": str(i)}) for i in range(50)]
        hub._loop.call_soon_threadsafe(lambda: [hub._publish_job(jid, it) for it in items])

    import threading

    th = threading.Thread(target=flood)
    th.start()
    got = read_stream(http, f"/api/jobs/{jid}/stream", timeout=20)
    th.join()
    assert got[-1]["event"] == "resync"
    assert isinstance(got[-1]["data"]["after"], int)
    http.post(f"/api/jobs/{jid}/cancel")
    http.post("/api/queue/resume")


def test_global_stream_overflow_and_ring_expiry(server):
    http = server.client()
    hub = server.ctx.hub
    # ring-buffer expiry: an id older than the 10,000-event buffer → resync{expired}
    hub._loop.call_soon_threadsafe(lambda: [hub._publish("storage.low", {"free_bytes": i, "threshold_bytes": 1})
                                            for i in range(10_050)])
    wait_for(lambda: hub.gseq >= 10_050, 10, what="the burst")
    evs = read_stream(http, "/api/stream", headers={"Last-Event-ID": "g:1"}, max_events=1)
    assert evs[0]["event"] == "resync" and evs[0]["data"]["reason"] == "expired"
    assert evs[0]["id"] == f"g:{hub.gseq}"
    # an id inside the buffer replays exactly the missed events
    start = hub.gseq - 5
    evs = read_stream(http, f"/api/stream?after=g:{start}", max_events=5)
    assert [e["id"] for e in evs] == [f"g:{start + i}" for i in range(1, 6)]
    assert all(e["event"] == "storage.low" and e["data"]["gseq"] == int(e["id"][2:]) for e in evs)
    # a client of an earlier server process (id ahead of gseq) is told to resync
    evs = read_stream(http, "/api/stream", headers={"Last-Event-ID": f"g:{hub.gseq + 100}"}, max_events=1)
    assert evs[0]["event"] == "resync"
    # overflow of a slow global subscriber → resync{overflow}
    wait_for(lambda: len(hub._global) == 0, 10, what="the earlier streams to unsubscribe")
    hub.global_queue = 10
    import threading

    def flood():
        wait_for(lambda: len(hub._global) >= 1, 10, what="the subscriber")
        hub._loop.call_soon_threadsafe(lambda: [hub._publish("storage.low", {"free_bytes": i, "threshold_bytes": 1})
                                                for i in range(50)])

    th = threading.Thread(target=flood)
    th.start()
    evs = read_stream(http, "/api/stream?topics=storage", until=lambda e: e.get("event") == "resync", timeout=20)
    th.join()
    assert evs[-1]["data"]["reason"] == "overflow"


def test_global_stream_job_events_and_topics(server):
    http = server.client()
    got = []
    import threading

    def consume():
        got.extend(read_stream(http, "/api/stream?topics=jobs",
                               until=lambda e: e.get("event") == "job.status" and e["data"]["status"] in FINAL))

    th = threading.Thread(target=consume)
    th.start()
    wait_for(lambda: len(server.ctx.hub._global) >= 1, 10, what="the subscriber")
    server.ctx.hub.publish("run.indexed", {"run_id": "r", "project_id": None, "origin": "studio"})   # other topic
    jid = run_job(http, seconds=1.0)
    th.join(60)
    types = [e["event"] for e in got]
    assert "run.indexed" not in types
    assert types[0] == "job.created" and got[0]["data"]["job"]["id"] == jid
    statuses = [e["data"]["status"] for e in got if e["event"] == "job.status"]
    assert statuses == ["starting", "running", "succeeded"]
    progress = [e["data"] for e in got if e["event"] == "job.progress"]
    assert progress and all(p["job_id"] == jid for p in progress)
    assert {"frac", "eta_s", "eta_lo", "eta_hi", "stage", "path_tail"} <= set(progress[0])
    gseqs = [int(e["id"][2:]) for e in got]
    assert gseqs == sorted(gseqs) and all(e["data"]["gseq"] == int(e["id"][2:]) for e in got)


def test_shutdown_ends_streams(live_server):
    server = live_server()
    http = server.client()
    got = []
    import threading

    th = threading.Thread(target=lambda: got.extend(read_stream(http, "/api/stream", timeout=30)))
    th.start()
    wait_for(lambda: len(server.ctx.hub._global) >= 1, 10, what="the subscriber")
    server.server.should_exit = True
    th.join(30)
    assert not th.is_alive()
    assert got and got[-1]["event"] == "server_shutdown"
    server.thread.join(30)
    assert not server.thread.is_alive()
