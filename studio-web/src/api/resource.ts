// Server state cache (SPEC §12.4): `useResource(key, fetcher, {tags, immutable})`.
//
// - One cache entry per key, shared by every component that reads it; concurrent mounts
//   share one request.
// - Tags group entries for invalidation. `invalidate("run:r1")` hits every entry tagged
//   `run:r1` or `run:r1:<anything>` (the ":" boundary matters). Entries are always tagged
//   with their own key as well.
// - Immutable entries (finished-run binaries, grids, catalogues of a settled run) are never
//   refetched by invalidation or resync.
// - Global SSE events invalidate tags (stores/jobs.ts maps them). Conventional tags:
//     run:<rid>  run:<rid>:outputs  run:<rid>:engine  run:<rid>:studies  runs
//     job:<jid>  jobs  project:<pid>  projects  scenario:<sid>  study:<stid>  engine  storage
import { useCallback, useEffect, useRef, useSyncExternalStore } from "react";
import { ApiError } from "./client";

export type ResourceStatus = "idle" | "loading" | "ready" | "error";

export type ResourceSnapshot<T> = {
  data: T | undefined;
  error: ApiError | Error | null;
  status: ResourceStatus;
  /** True while a request is in flight (first load or revalidation). */
  loading: boolean;
};

export type ResourceOptions = {
  tags?: string[];
  /** Never refetch once loaded (finished-run resources). */
  immutable?: boolean;
  /** While a new key loads, keep returning the previous key's data. */
  keepPrevious?: boolean;
};

type Fetcher<T> = (signal: AbortSignal) => Promise<T>;

type Entry = {
  key: string;
  snap: ResourceSnapshot<unknown>;
  fetcher: Fetcher<unknown> | null;
  tags: Set<string>;
  immutable: boolean;
  listeners: Set<() => void>;
  inflight: Promise<unknown> | null;
  controller: AbortController | null;
  stale: boolean;
  touched: number;
};

const cache = new Map<string, Entry>();
const MAX_IDLE_ENTRIES = 400;
let clock = 0;

const IDLE: ResourceSnapshot<never> = Object.freeze({ data: undefined, error: null, status: "idle", loading: false }) as ResourceSnapshot<never>;

function getEntry(key: string): Entry {
  let e = cache.get(key);
  if (!e) {
    e = {
      key,
      snap: IDLE,
      fetcher: null,
      tags: new Set([key]),
      immutable: false,
      listeners: new Set(),
      inflight: null,
      controller: null,
      stale: true,
      touched: ++clock,
    };
    cache.set(key, e);
    gc();
  }
  return e;
}

function gc(): void {
  if (cache.size <= MAX_IDLE_ENTRIES) return;
  const idle = [...cache.values()].filter((e) => e.listeners.size === 0 && !e.inflight).sort((a, b) => a.touched - b.touched);
  for (const e of idle.slice(0, cache.size - MAX_IDLE_ENTRIES)) cache.delete(e.key);
}

function setSnap(e: Entry, patch: Partial<ResourceSnapshot<unknown>>): void {
  e.snap = { ...e.snap, ...patch };
  for (const l of [...e.listeners]) l();
}

function toError(err: unknown): ApiError | Error {
  return err instanceof ApiError || err instanceof Error ? err : new Error(String(err));
}

function load(e: Entry): Promise<unknown> {
  if (e.inflight) return e.inflight;
  const fetcher = e.fetcher;
  if (!fetcher) return Promise.resolve(e.snap.data);
  const ctrl = new AbortController();
  e.controller = ctrl;
  e.stale = false;
  setSnap(e, { loading: true, status: e.snap.status === "ready" ? "ready" : "loading" });
  const p = fetcher(ctrl.signal).then(
    (data) => {
      if (e.controller !== ctrl) return data;
      e.inflight = null;
      e.controller = null;
      setSnap(e, { data, error: null, status: "ready", loading: false });
      return data;
    },
    (err: unknown) => {
      if (e.controller !== ctrl) return undefined;
      e.inflight = null;
      e.controller = null;
      setSnap(e, { error: toError(err), status: "error", loading: false });
      return undefined;
    },
  );
  e.inflight = p;
  return p;
}

function tagMatches(entryTag: string, tag: string): boolean {
  return entryTag === tag || entryTag.startsWith(tag + ":");
}

/**
 * Mark every non-immutable entry carrying `tag` (or a sub-tag) stale. Mounted entries
 * refetch immediately (their current data stays visible meanwhile); unmounted ones are
 * dropped so the next mount refetches.
 */
export function invalidate(tag: string): void {
  for (const e of [...cache.values()]) {
    if (e.immutable) continue;
    let hit = false;
    for (const t of e.tags) if (tagMatches(t, tag)) hit = true;
    if (!hit) continue;
    revalidateEntry(e);
  }
}

function revalidateEntry(e: Entry): void {
  if (e.listeners.size === 0) {
    e.controller?.abort();
    cache.delete(e.key);
    return;
  }
  e.stale = true;
  if (e.inflight) {
    // A request started before the change may return old data; restart it.
    e.controller?.abort();
    e.inflight = null;
    e.controller = null;
  }
  void load(e);
}

/** Refetch every non-immutable resource (after an SSE `resync`). */
export function invalidateAll(): void {
  for (const e of [...cache.values()]) if (!e.immutable) revalidateEntry(e);
}

/** Replace (or compute) an entry's data locally, e.g. after a PATCH returns the new object. */
export function mutate<T>(key: string, data: T | ((prev: T | undefined) => T)): void {
  const e = getEntry(key);
  const next = typeof data === "function" ? (data as (p: T | undefined) => T)(e.snap.data as T | undefined) : data;
  e.stale = false;
  setSnap(e, { data: next, error: null, status: "ready" });
}

/** Read the cached data for a key without subscribing. */
export function peekResource<T>(key: string): T | undefined {
  return cache.get(key)?.snap.data as T | undefined;
}

/** Fetch into the cache ahead of a mount (e.g. hovering a tab). Resolves with the data. */
export function prefetch<T>(key: string, fetcher: Fetcher<T>, opts: ResourceOptions = {}): Promise<T | undefined> {
  const e = getEntry(key);
  configure(e, fetcher, opts);
  if (e.snap.status === "ready" && !e.stale) return Promise.resolve(e.snap.data as T);
  return load(e) as Promise<T | undefined>;
}

/** Forget everything (tests, logout). */
export function clearResources(): void {
  for (const e of cache.values()) e.controller?.abort();
  cache.clear();
}

/** Number of cached entries (diagnostics and tests). */
export function resourceCount(): number {
  return cache.size;
}

function configure(e: Entry, fetcher: Fetcher<unknown>, opts: ResourceOptions): void {
  e.fetcher = fetcher;
  for (const t of opts.tags ?? []) e.tags.add(t);
  if (opts.immutable) e.immutable = true;
  e.touched = ++clock;
}

export type ResourceResult<T> = ResourceSnapshot<T> & {
  /** Refetch now (ignores `immutable`). */
  reload: () => Promise<void>;
};

/**
 * Subscribe to a server resource. `key === null` skips fetching (dependent queries).
 * The fetcher may change identity every render; the latest one is used for (re)loads.
 */
export function useResource<T>(key: string | null, fetcher: Fetcher<T>, opts: ResourceOptions = {}): ResourceResult<T> {
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;
  const stableFetcher = useCallback((s: AbortSignal) => fetcherRef.current(s) as Promise<unknown>, []);
  const tagsKey = (opts.tags ?? []).join("|");
  const immutable = !!opts.immutable;

  const subscribe = useCallback(
    (cb: () => void) => {
      if (key === null) return () => {};
      const e = getEntry(key);
      e.listeners.add(cb);
      return () => {
        e.listeners.delete(cb);
      };
    },
    [key],
  );
  const getSnapshot = useCallback(() => (key === null ? IDLE : getEntry(key).snap), [key]);
  const snap = useSyncExternalStore(subscribe, getSnapshot, getSnapshot) as ResourceSnapshot<T>;

  useEffect(() => {
    if (key === null) return;
    const e = getEntry(key);
    configure(e, stableFetcher, { tags: tagsKey ? tagsKey.split("|") : [], immutable });
    if (e.stale || e.snap.status === "idle") void load(e);
  }, [key, tagsKey, immutable, stableFetcher]);

  const prev = useRef<T | undefined>(undefined);
  if (snap.data !== undefined) prev.current = snap.data;

  const reload = useCallback(async () => {
    if (key === null) return;
    const e = getEntry(key);
    e.stale = true;
    if (e.inflight) {
      e.controller?.abort();
      e.inflight = null;
      e.controller = null;
    }
    await load(e);
  }, [key]);

  const data = snap.data !== undefined ? snap.data : opts.keepPrevious ? prev.current : undefined;
  return { ...snap, data, reload };
}
