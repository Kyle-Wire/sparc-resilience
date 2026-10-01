// Server-Sent Events (SPEC §5.7, api.md §4 and §17).
//
// StreamManager keeps at most two EventSources per tab: the global stream (`/api/stream`)
// and one per-job stream (`/api/jobs/{jid}/stream`). For each stream it:
// - resumes from the last seen id (`after=g:<gseq>` / `after=<cursor>`), so a reconnect has
//   no gap; events at or before the last delivered cursor are dropped, so it has no
//   duplicates either;
// - reconnects with backoff 1 → 2 → 4 → 8 → 16 → 30 s (reset once a connection opens);
// - after three failures in a row, falls back to polling every 5 s (global: GET
//   /api/jobs?status=active; job: GET /api/jobs/{jid}/events?after=) and keeps retrying the
//   stream in the background; the connection pill then reads "polling";
// - handles `resync`: global → listeners refetch active jobs and open resources;
//   job → reconnect immediately from `after` (the server closed the stream on overflow);
// - closes a job stream on `end {status}`;
// - batches deliveries per animation frame, so a burst of events causes one render.
import { api } from "./client";
import {
  GLOBAL_EVENT_TYPES,
  JOB_EVENT_TYPES,
  type GlobalEvent,
  type Job,
  type JobEvent,
  type JobEventsPage,
  type JobStatus,
  type Page,
  FINAL_JOB_STATUSES,
} from "./types";

export type ConnState = "connecting" | "live" | "reconnecting" | "polling" | "closed";

/** The subset of EventSource the manager uses (a fake one is injected in tests). */
export interface EventSourceLike {
  onopen: ((ev: Event) => void) | null;
  onerror: ((ev: Event) => void) | null;
  onmessage: ((ev: MessageEvent) => void) | null;
  addEventListener(type: string, listener: (ev: MessageEvent) => void): void;
  close(): void;
}

export type EventSourceCtor = new (url: string, init?: EventSourceInit) => EventSourceLike;

export type StreamDeps = {
  EventSource: EventSourceCtor;
  getJson: <T>(path: string) => Promise<T>;
  setTimeout: (cb: () => void, ms: number) => unknown;
  clearTimeout: (h: unknown) => void;
  /** Schedule a batch flush (requestAnimationFrame in the browser). */
  frame: (cb: () => void) => void;
};

export const BACKOFF_S = [1, 2, 4, 8, 16, 30] as const;
export const POLL_MS = 5000;
export const FAILURES_BEFORE_POLLING = 3;

export function backoffMs(failures: number): number {
  const i = Math.max(0, Math.min(BACKOFF_S.length - 1, failures - 1));
  return BACKOFF_S[i] * 1000;
}

function defaultDeps(): StreamDeps {
  const raf =
    typeof window !== "undefined" && typeof window.requestAnimationFrame === "function"
      ? (cb: () => void) => void window.requestAnimationFrame(() => cb())
      : (cb: () => void) => void setTimeout(cb, 16);
  return {
    EventSource: (typeof EventSource !== "undefined" ? EventSource : undefined) as unknown as EventSourceCtor,
    getJson: <T,>(path: string) => api.get<T>(path),
    setTimeout: (cb, ms) => setTimeout(cb, ms),
    clearTimeout: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
    frame: raf,
  };
}

type GlobalListener = (events: GlobalEvent[]) => void;
type JobSubscriber = {
  onEvents: (events: JobEvent[]) => void;
  onEnd?: (status: JobStatus) => void;
  onClosed?: (reason: "replaced" | "stopped") => void;
};

/** Shared mechanics of one logical stream: connect, backoff, polling, batching. */
abstract class Channel<E> {
  protected es: EventSourceLike | null = null;
  protected failures = 0;
  protected retryTimer: unknown = null;
  protected pollTimer: unknown = null;
  protected buffer: E[] = [];
  protected flushScheduled = false;
  state: ConnState = "connecting";
  closed = false;

  protected deps: StreamDeps;
  protected onState: () => void;

  constructor(deps: StreamDeps, onState: () => void) {
    this.deps = deps;
    this.onState = onState;
  }

  protected abstract url(): string;
  protected abstract eventNames(): readonly string[];
  protected abstract handle(name: string, ev: MessageEvent): void;
  protected abstract poll(): Promise<void>;
  protected abstract deliver(batch: E[]): void;

  protected setState(s: ConnState): void {
    if (this.state !== s) {
      this.state = s;
      this.onState();
    }
  }

  connect(): void {
    if (this.closed) return;
    this.teardownSource();
    let es: EventSourceLike;
    try {
      es = new this.deps.EventSource(this.url(), { withCredentials: true });
    } catch {
      this.fail();
      return;
    }
    this.es = es;
    es.onopen = () => {
      if (this.es !== es) return;
      this.failures = 0;
      this.stopPolling();
      this.setState("live");
    };
    es.onerror = () => {
      if (this.es !== es) return;
      this.fail();
    };
    for (const name of this.eventNames()) es.addEventListener(name, (ev) => this.es === es && this.handle(name, ev));
    es.onmessage = (ev) => this.es === es && this.handle("message", ev);
  }

  protected fail(): void {
    this.teardownSource();
    if (this.closed) return;
    this.failures += 1;
    if (this.failures >= FAILURES_BEFORE_POLLING) {
      if (this.state !== "polling") {
        this.setState("polling");
        this.startPolling();
      }
    } else {
      this.setState("reconnecting");
    }
    this.retryTimer = this.deps.setTimeout(() => {
      this.retryTimer = null;
      this.connect();
    }, backoffMs(this.failures));
  }

  protected teardownSource(): void {
    if (this.es) {
      const es = this.es;
      this.es = null;
      es.onopen = es.onerror = es.onmessage = null;
      es.close();
    }
    if (this.retryTimer !== null) {
      this.deps.clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
  }

  protected startPolling(): void {
    if (this.pollTimer !== null || this.closed) return;
    const tick = async () => {
      if (this.closed || this.state !== "polling") {
        this.pollTimer = null;
        return;
      }
      try {
        await this.poll();
      } catch {
        /* keep polling; the pill already says so */
      }
      if (!this.closed && this.state === "polling") this.pollTimer = this.deps.setTimeout(() => void tick(), POLL_MS);
      else this.pollTimer = null;
    };
    this.pollTimer = this.deps.setTimeout(() => void tick(), 0);
  }

  protected stopPolling(): void {
    if (this.pollTimer !== null) {
      this.deps.clearTimeout(this.pollTimer);
      this.pollTimer = null;
    }
  }

  protected push(e: E): void {
    this.buffer.push(e);
    if (!this.flushScheduled) {
      this.flushScheduled = true;
      this.deps.frame(() => this.flush());
    }
  }

  flush(): void {
    this.flushScheduled = false;
    if (!this.buffer.length) return;
    const batch = this.buffer;
    this.buffer = [];
    this.deliver(batch);
  }

  close(): void {
    this.flush();
    this.closed = true;
    this.teardownSource();
    this.stopPolling();
    this.setState("closed");
  }
}

function parseData(ev: MessageEvent): Record<string, unknown> | null {
  try {
    const d = JSON.parse(String(ev.data)) as unknown;
    return d && typeof d === "object" ? (d as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

class GlobalChannel extends Channel<GlobalEvent> {
  lastId: string | null = null;
  private topics: string[] | null;
  private listeners: Set<GlobalListener>;
  private onResync: () => void;
  private onPollJobs: (jobs: Job[]) => void;

  constructor(
    deps: StreamDeps,
    onState: () => void,
    topics: string[] | null,
    listeners: Set<GlobalListener>,
    onResync: () => void,
    onPollJobs: (jobs: Job[]) => void,
  ) {
    super(deps, onState);
    this.topics = topics;
    this.listeners = listeners;
    this.onResync = onResync;
    this.onPollJobs = onPollJobs;
  }

  protected url(): string {
    const q = new URLSearchParams();
    if (this.topics?.length) q.set("topics", this.topics.join(","));
    if (this.lastId) q.set("after", this.lastId);
    const s = q.toString();
    return "/api/stream" + (s ? "?" + s : "");
  }

  protected eventNames(): readonly string[] {
    return GLOBAL_EVENT_TYPES;
  }

  protected handle(name: string, ev: MessageEvent): void {
    const data = parseData(ev);
    if (!data) return;
    const type = name === "message" ? String(data.type ?? "message") : name;
    if (ev.lastEventId) this.lastId = ev.lastEventId;
    else if (typeof data.gseq === "number") this.lastId = `g:${data.gseq}`;
    const event = { ...data, type } as GlobalEvent;
    this.push(event);
    if (type === "resync") {
      this.flush();
      this.onResync();
    }
  }

  protected async poll(): Promise<void> {
    const page = await this.deps.getJson<Page<Job>>("/api/jobs?status=active&limit=500");
    this.onPollJobs(page.items ?? []);
  }

  protected deliver(batch: GlobalEvent[]): void {
    for (const l of [...this.listeners]) l(batch);
  }
}

class JobChannel extends Channel<JobEvent> {
  /** Highest cursor delivered; events at or below it are duplicates. */
  lastCursor: number | null = null;
  private after: number | null;
  private pollAfter: number | null = null;
  ended: JobStatus | null = null;
  readonly jid: string;
  readonly subscribers: Set<JobSubscriber>;

  constructor(deps: StreamDeps, onState: () => void, jid: string, after: number | null, subscribers: Set<JobSubscriber>) {
    super(deps, onState);
    this.jid = jid;
    this.after = after;
    this.subscribers = subscribers;
  }

  private resumeFrom(): number | null {
    return this.lastCursor ?? this.after;
  }

  protected url(): string {
    const a = this.resumeFrom();
    return `/api/jobs/${encodeURIComponent(this.jid)}/stream` + (a !== null ? `?after=${a}` : "");
  }

  protected eventNames(): readonly string[] {
    return [...JOB_EVENT_TYPES, "resync", "end"];
  }

  private accept(data: Record<string, unknown>, idHeader: string): void {
    const cur = typeof data.cursor === "number" ? data.cursor : idHeader !== "" && Number.isFinite(Number(idHeader)) ? Number(idHeader) : null;
    if (cur !== null) {
      if (this.lastCursor !== null && cur <= this.lastCursor) return; // duplicate after a reconnect
      this.lastCursor = cur;
      if (data.cursor === undefined) data.cursor = cur;
    }
    this.push(data as unknown as JobEvent);
  }

  protected handle(name: string, ev: MessageEvent): void {
    const data = parseData(ev);
    if (!data) return;
    if (name === "resync") {
      // The server dropped queued events and closes; resume from its cursor right away.
      const after = typeof data.after === "number" ? data.after : null;
      if (after !== null && this.lastCursor === null) this.after = after;
      this.flush();
      this.teardownSource();
      this.setState("reconnecting");
      this.connect();
      return;
    }
    if (name === "end") {
      this.finish(String(data.status ?? "succeeded") as JobStatus);
      return;
    }
    const type = name === "message" ? String(data.type ?? "log") : name;
    if (data.type === undefined) data.type = type;
    this.accept(data, ev.lastEventId ?? "");
  }

  private finish(status: JobStatus): void {
    this.ended = status;
    this.flush();
    for (const s of [...this.subscribers]) s.onEnd?.(status);
    this.close();
  }

  protected async poll(): Promise<void> {
    const after = this.pollAfter ?? this.resumeFrom() ?? 0;
    const page = await this.deps.getJson<JobEventsPage>(`/api/jobs/${encodeURIComponent(this.jid)}/events?after=${after}&limit=5000`);
    let final: JobStatus | null = null;
    for (const e of page.events ?? []) {
      this.accept(e as unknown as Record<string, unknown>, "");
      if (e.type === "job.status" && FINAL_JOB_STATUSES.includes((e as { status: JobStatus }).status)) final = (e as { status: JobStatus }).status;
    }
    if (typeof page.next_cursor === "number") this.pollAfter = page.next_cursor;
    if (final && page.eof) this.finish(final);
  }

  protected deliver(batch: JobEvent[]): void {
    for (const s of [...this.subscribers]) s.onEvents(batch);
  }
}

export type JobStreamHandle = { close: () => void; readonly jid: string };

export class StreamManager {
  private deps: StreamDeps;
  private global: GlobalChannel | null = null;
  private job: JobChannel | null = null;
  private globalListeners = new Set<GlobalListener>();
  private resyncListeners = new Set<() => void>();
  private pollListeners = new Set<(jobs: Job[]) => void>();
  private stateListeners = new Set<() => void>();
  private stateCache: ConnState = "connecting";

  constructor(deps: Partial<StreamDeps> = {}) {
    this.deps = { ...defaultDeps(), ...deps };
  }

  private emitState = (): void => {
    const s = this.computeState();
    if (s !== this.stateCache) {
      this.stateCache = s;
      for (const l of [...this.stateListeners]) l();
    }
  };

  private computeState(): ConnState {
    const chans = [this.global, this.job].filter((c): c is GlobalChannel | JobChannel => !!c && !c.closed);
    if (!chans.length) return this.global ? "closed" : "connecting";
    if (chans.some((c) => c.state === "polling")) return "polling";
    if (chans.some((c) => c.state === "reconnecting")) return "reconnecting";
    if (chans.every((c) => c.state === "live")) return "live";
    return "connecting";
  }

  /** Combined connection state for the pill: polling > reconnecting > connecting > live. */
  get state(): ConnState {
    return this.stateCache;
  }

  subscribeState = (cb: () => void): (() => void) => {
    this.stateListeners.add(cb);
    return () => this.stateListeners.delete(cb);
  };

  getState = (): ConnState => this.stateCache;

  /** Open the global stream (idempotent). */
  startGlobal(topics: string[] | null = null): void {
    if (this.global && !this.global.closed) return;
    this.global = new GlobalChannel(
      this.deps,
      this.emitState,
      topics,
      this.globalListeners,
      () => {
        for (const l of [...this.resyncListeners]) l();
      },
      (jobs) => {
        for (const l of [...this.pollListeners]) l(jobs);
      },
    );
    this.global.connect();
    this.emitState();
  }

  onGlobal(cb: GlobalListener): () => void {
    this.globalListeners.add(cb);
    return () => this.globalListeners.delete(cb);
  }

  /** Called on a global `resync`: refetch active jobs and every open resource. */
  onResync(cb: () => void): () => void {
    this.resyncListeners.add(cb);
    return () => this.resyncListeners.delete(cb);
  }

  /** Called with the active job list while the global stream is in polling fallback. */
  onPollJobs(cb: (jobs: Job[]) => void): () => void {
    this.pollListeners.add(cb);
    return () => this.pollListeners.delete(cb);
  }

  /**
   * Subscribe to one job's events, starting after `after` (the tracker snapshot cursor).
   * Only one job stream exists per tab: opening a different job closes the previous one
   * (its subscribers get `onClosed("replaced")`). Several subscribers to the same job share
   * the stream; it closes when the last one unsubscribes.
   */
  openJob(jid: string, sub: JobSubscriber & { after?: number | null }): JobStreamHandle {
    if (this.job && (this.job.jid !== jid || this.job.closed)) {
      const old = this.job;
      this.job = null;
      if (!old.closed) {
        for (const s of [...old.subscribers]) s.onClosed?.("replaced");
        old.close();
      }
    }
    if (!this.job) {
      this.job = new JobChannel(this.deps, this.emitState, jid, sub.after ?? null, new Set());
      this.job.subscribers.add(sub);
      this.job.connect();
    } else {
      this.job.subscribers.add(sub);
    }
    const chan = this.job;
    this.emitState();
    return {
      jid,
      close: () => {
        chan.subscribers.delete(sub);
        if (chan.subscribers.size === 0 && !chan.closed) {
          chan.close();
          if (this.job === chan) this.job = null;
        }
        this.emitState();
      },
    };
  }

  /** The job currently streamed, if any (diagnostics and tests). */
  get jobStreamId(): string | null {
    return this.job && !this.job.closed ? this.job.jid : null;
  }

  /** Deliver buffered events now instead of waiting for the next frame (tests). */
  flush(): void {
    this.global?.flush();
    this.job?.flush();
  }

  stop(): void {
    if (this.job) {
      for (const s of [...this.job.subscribers]) s.onClosed?.("stopped");
      this.job.close();
      this.job = null;
    }
    this.global?.close();
    this.emitState();
  }
}

let shared: StreamManager | null = null;

/** The tab-wide StreamManager (created on first use). */
export function getStreams(): StreamManager {
  if (!shared) shared = new StreamManager();
  return shared;
}

/** Replace the shared manager (tests). */
export function setStreams(m: StreamManager | null): void {
  shared?.stop();
  shared = m;
}
