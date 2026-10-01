import { describe, expect, it } from "vitest";
import { BACKOFF_S, StreamManager, backoffMs, type EventSourceLike, type StreamDeps } from "../api/sse";
import type { GlobalEvent, Job, JobEvent } from "../api/types";

/** A controllable EventSource double. */
class FakeES implements EventSourceLike {
  static all: FakeES[] = [];
  onopen: ((ev: Event) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  closed = false;
  listeners = new Map<string, ((ev: MessageEvent) => void)[]>();
  url: string;
  constructor(url: string) {
    this.url = url;
    FakeES.all.push(this);
  }
  addEventListener(type: string, l: (ev: MessageEvent) => void) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), l]);
  }
  close() {
    this.closed = true;
  }
  open() {
    this.onopen?.(new Event("open"));
  }
  fail() {
    this.onerror?.(new Event("error"));
  }
  send(type: string, data: unknown, id?: string) {
    const ev = new MessageEvent(type, { data: JSON.stringify(data), lastEventId: id ?? "" });
    for (const l of this.listeners.get(type) ?? []) l(ev);
  }
}

type Timer = { at: number; cb: () => void; id: number };

/** Deterministic timers, a manual frame queue and a scripted JSON fetcher. */
function harness(json: (path: string) => unknown = () => ({ items: [] })) {
  FakeES.all = [];
  let now = 0;
  let seq = 0;
  let timers: Timer[] = [];
  const frames: (() => void)[] = [];
  const fetched: string[] = [];
  const deps: StreamDeps = {
    EventSource: FakeES,
    getJson: async <T,>(path: string) => {
      fetched.push(path);
      return json(path) as T;
    },
    setTimeout: (cb, ms) => {
      const t = { at: now + ms, cb, id: ++seq };
      timers.push(t);
      return t.id;
    },
    clearTimeout: (h) => {
      timers = timers.filter((t) => t.id !== h);
    },
    frame: (cb) => void frames.push(cb),
  };
  const advance = async (ms: number) => {
    const end = now + ms;
    for (;;) {
      const due = timers.filter((t) => t.at <= end).sort((a, b) => a.at - b.at)[0];
      if (!due) break;
      timers = timers.filter((t) => t !== due);
      now = due.at;
      due.cb();
      await Promise.resolve();
      await Promise.resolve();
    }
    now = end;
  };
  const runFrames = () => {
    while (frames.length) frames.shift()!();
  };
  return { deps, advance, runFrames, fetched, latest: () => FakeES.all[FakeES.all.length - 1], pending: () => timers.map((t) => t.at - now) };
}

const ev = (cursor: number, type = "log", extra: Record<string, unknown> = {}) => ({ v: 1, type, seq: cursor, ts: 1, t_rel: 0, pid: 1, job: "j_1", lvl: "info", span: null, parent: null, path: [], ctx: {}, cursor, ...extra });

describe("StreamManager: job stream", () => {
  it("opens with after=<snapshot cursor> and batches events per frame", () => {
    const h = harness();
    const m = new StreamManager(h.deps);
    const got: JobEvent[][] = [];
    m.openJob("j_1", { after: 1200, onEvents: (b) => got.push(b) });
    const es = h.latest();
    expect(es.url).toBe("/api/jobs/j_1/stream?after=1200");
    es.open();
    expect(m.state).toBe("live");
    es.send("tick", ev(1300, "tick", { k: 1, n: 5 }), "1300");
    es.send("log", ev(1400), "1400");
    expect(got).toHaveLength(0); // waits for the frame
    h.runFrames();
    expect(got).toHaveLength(1);
    expect(got[0].map((e) => e.cursor)).toEqual([1300, 1400]);
  });

  it("reconnects from the last cursor with backoff and drops replayed duplicates", async () => {
    const h = harness();
    const m = new StreamManager(h.deps);
    const got: number[] = [];
    m.openJob("j_1", { after: 0, onEvents: (b) => got.push(...b.map((e) => e.cursor!)) });
    const es1 = h.latest();
    es1.open();
    es1.send("log", ev(100), "100");
    es1.send("log", ev(250), "250");
    es1.fail();
    expect(es1.closed).toBe(true);
    expect(m.state).toBe("reconnecting");
    expect(h.pending()).toEqual([1000]); // first retry after 1 s
    await h.advance(1000);
    const es2 = h.latest();
    expect(es2).not.toBe(es1);
    expect(es2.url).toBe("/api/jobs/j_1/stream?after=250");
    es2.open();
    es2.send("log", ev(250), "250"); // a server replaying the boundary line is tolerated
    es2.send("log", ev(400), "400");
    h.runFrames();
    expect(got).toEqual([100, 250, 400]);
  });

  it("backs off 1 → 2 → 4 … 30 s", () => {
    expect(BACKOFF_S).toEqual([1, 2, 4, 8, 16, 30]);
    expect([1, 2, 3, 4, 5, 6, 7, 12].map(backoffMs)).toEqual([1000, 2000, 4000, 8000, 16000, 30000, 30000, 30000]);
  });

  it("falls back to polling after three failures, then returns to streaming", async () => {
    const pages: Record<string, unknown> = {
      "/api/jobs/j_1/events?after=500&limit=5000": { events: [ev(600), ev(700)], next_cursor: 700, eof: false },
      "/api/jobs/j_1/events?after=700&limit=5000": { events: [ev(800, "job.status", { status: "succeeded" })], next_cursor: 800, eof: true },
    };
    const h = harness((p) => pages[p] ?? { events: [], next_cursor: 0, eof: false });
    const m = new StreamManager(h.deps);
    const got: number[] = [];
    let ended: string | null = null;
    m.openJob("j_1", { after: 500, onEvents: (b) => got.push(...b.map((e) => e.cursor!)), onEnd: (s) => (ended = s) });
    h.latest().fail(); // 1
    await h.advance(1000);
    h.latest().fail(); // 2
    await h.advance(2000);
    expect(m.state).toBe("reconnecting");
    h.latest().fail(); // 3 → polling
    expect(m.state).toBe("polling");
    await h.advance(0); // first poll is immediate
    h.runFrames();
    expect(h.fetched[0]).toBe("/api/jobs/j_1/events?after=500&limit=5000");
    expect(got).toEqual([600, 700]);
    await h.advance(5000); // next poll after 5 s, sees the final status at eof
    h.runFrames();
    expect(h.fetched).toContain("/api/jobs/j_1/events?after=700&limit=5000");
    expect(got).toEqual([600, 700, 800]);
    expect(ended).toBe("succeeded");
    expect(m.jobStreamId).toBeNull();
  });

  it("a reconnect that opens stops polling", async () => {
    const h = harness(() => ({ events: [], next_cursor: 0, eof: false }));
    const m = new StreamManager(h.deps);
    m.openJob("j_1", { onEvents: () => {} });
    h.latest().fail();
    await h.advance(1000);
    h.latest().fail();
    await h.advance(2000);
    h.latest().fail();
    expect(m.state).toBe("polling");
    await h.advance(4000); // retry timer (4 s) fires while polling
    h.latest().open();
    expect(m.state).toBe("live");
    const before = h.fetched.length;
    await h.advance(20000);
    expect(h.fetched.length).toBe(before);
  });

  it("handles resync by reconnecting immediately from `after`", () => {
    const h = harness();
    const m = new StreamManager(h.deps);
    const got: number[] = [];
    m.openJob("j_1", { onEvents: (b) => got.push(...b.map((e) => e.cursor!)) });
    const es1 = h.latest();
    es1.open();
    es1.send("log", ev(10), "10");
    es1.send("resync", { after: 10 });
    expect(es1.closed).toBe(true);
    const es2 = h.latest();
    expect(es2).not.toBe(es1);
    expect(es2.url).toBe("/api/jobs/j_1/stream?after=10");
    expect(h.pending()).toEqual([]); // no backoff for a resync
    es2.open();
    es2.send("log", ev(20), "20");
    h.runFrames();
    expect(got).toEqual([10, 20]);
  });

  it("closes on `end` and never reconnects", async () => {
    const h = harness();
    const m = new StreamManager(h.deps);
    let ended: string | null = null;
    m.openJob("j_1", { onEvents: () => {}, onEnd: (s) => (ended = s) });
    const es = h.latest();
    es.open();
    es.send("end", { status: "cancelled" });
    expect(ended).toBe("cancelled");
    expect(es.closed).toBe(true);
    await h.advance(60000);
    expect(FakeES.all).toHaveLength(1);
  });

  it("keeps at most one job stream: opening another job replaces it", () => {
    const h = harness();
    const m = new StreamManager(h.deps);
    let replaced = "";
    m.openJob("j_1", { onEvents: () => {}, onClosed: (r) => (replaced = r) });
    const first = h.latest();
    m.openJob("j_2", { onEvents: () => {} });
    expect(first.closed).toBe(true);
    expect(replaced).toBe("replaced");
    expect(m.jobStreamId).toBe("j_2");
  });

  it("transient resource events (no id) do not move the cursor", async () => {
    const h = harness();
    const m = new StreamManager(h.deps);
    m.openJob("j_1", { onEvents: () => {} });
    const es = h.latest();
    es.open();
    es.send("log", ev(50), "50");
    es.send("resource", { type: "resource", rss_mb: 100, cpu_pct: 50, n_procs: 2, threads: 4 });
    es.fail();
    await h.advance(1000);
    expect(h.latest().url).toBe("/api/jobs/j_1/stream?after=50");
  });
});

describe("StreamManager: global stream", () => {
  it("resumes from the last g:<gseq>, delivers batches and calls resync handlers", async () => {
    const h = harness();
    const m = new StreamManager(h.deps);
    const got: GlobalEvent[] = [];
    let resyncs = 0;
    m.onGlobal((b) => got.push(...b));
    m.onResync(() => resyncs++);
    m.startGlobal(["jobs", "runs"]);
    const es = h.latest();
    expect(es.url).toBe("/api/stream?topics=jobs%2Cruns");
    es.open();
    es.send("job.progress", { gseq: 7, ts: 1, job_id: "j_1", frac: 0.42, eta_s: 60, eta_lo: 50, eta_hi: 70, stage: "S2_S3", path_tail: [] }, "g:7");
    h.runFrames();
    expect(got[0]).toMatchObject({ type: "job.progress", job_id: "j_1", frac: 0.42 });
    es.fail();
    await h.advance(1000);
    expect(h.latest().url).toBe("/api/stream?topics=jobs%2Cruns&after=g%3A7");
    h.latest().open();
    h.latest().send("resync", { gseq: 9, ts: 2, reason: "expired" }, "g:9");
    expect(resyncs).toBe(1);
    expect(got.map((e) => e.type)).toEqual(["job.progress", "resync"]);
  });

  it("polls the active jobs while the global stream is down", async () => {
    const job = { id: "j_9", status: "running" } as Job;
    const h = harness((p) => (p.startsWith("/api/jobs?status=active") ? { items: [job], next_cursor: null } : {}));
    const m = new StreamManager(h.deps);
    const polled: Job[][] = [];
    m.onPollJobs((jobs) => polled.push(jobs));
    m.startGlobal();
    h.latest().fail();
    await h.advance(1000);
    h.latest().fail();
    await h.advance(2000);
    h.latest().fail();
    expect(m.state).toBe("polling");
    await h.advance(0);
    expect(polled).toEqual([[job]]);
    await h.advance(5000);
    expect(polled).toHaveLength(2);
  });

  it("reports the combined connection state", () => {
    const h = harness();
    const m = new StreamManager(h.deps);
    const states: string[] = [];
    m.subscribeState(() => states.push(m.getState()));
    m.startGlobal();
    h.latest().open();
    m.openJob("j_1", { onEvents: () => {} });
    h.latest().open();
    h.latest().fail();
    expect(states).toEqual(["live", "connecting", "live", "reconnecting"]);
  });
});
