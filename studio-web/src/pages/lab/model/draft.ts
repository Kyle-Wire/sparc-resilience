// The Lab draft (SPEC §7.4, §7.14, §12.4): the scenario being edited, its brush layers and a
// client-side undo/redo history of 50 steps, plus the autosave loop (PATCH every 2 s and on
// blur). Patching the content of a scenario that already has an exact result returns
// `409 conflict_revision`; autosave then forks a new revision with the edited doc and
// editing continues on it, so results are never orphaned.
//
// Drafts live outside React (a registry keyed by run and scenario) because the Design page
// remounts when its URL changes from /lab to /lab/s/<new id> after the first save or a fork.
import { createStore, type StoreApi } from "zustand/vanilla";
import { ApiError } from "../../../api/client";
import type { Edit, Scenario, ScenarioDoc } from "../../../api/lab";
import { BrushLayers, snapshotsEqual, type BrushSnapshot } from "./brush";
import { contentKey, emptyDoc, serializeDoc } from "./doc";

export const HISTORY_LIMIT = 50;
/** Changes with the same coalesce key within this window share one undo step (typing, brush strokes). */
export const COALESCE_MS = 800;

export type DraftSnapshot = { doc: ScenarioDoc; brush: BrushSnapshot };

export type SaveState = "idle" | "saving" | "saved" | "error";

export type DraftState = {
  /** Scenario id once saved (null for a new draft). */
  sid: string | null;
  revision: number | null;
  status: Scenario["status"] | null;
  /** The server's updated_utc of the scenario this draft last loaded or saved. */
  serverUpdated: string | null;
  doc: ScenarioDoc;
  /** Committed brush state (the live arrays may run ahead during a stroke). */
  brush: BrushSnapshot;
  past: DraftSnapshot[];
  future: DraftSnapshot[];
  /** Increments on every change; `savedVersion` is the last version the server has. */
  version: number;
  savedVersion: number;
  save: SaveState;
  saveError: string | null;
  /** The id this draft was forked from by the last conflict (for the notice). */
  forkedFrom: string | null;
  lastKey: string | null;
  lastAt: number;
};

export type DraftActions = {
  /** Apply a change to the doc (one undo step unless coalesced with the previous change). */
  change: (fn: (doc: ScenarioDoc) => ScenarioDoc, opts?: { coalesce?: string }) => void;
  /** Commit the live brush arrays as one undo step (coalesced per stroke). */
  commitBrush: (opts?: { coalesce?: string }) => void;
  undo: () => void;
  redo: () => void;
  /** Replace the draft with a loaded scenario (clears the history). */
  load: (s: Scenario) => void;
  /** Point the draft at a new scenario (first save, fork) without touching the edits. */
  rebase: (s: Scenario, forkedFrom?: string | null) => void;
  setSave: (save: SaveState, error?: string | null) => void;
  markSaved: (version: number) => void;
};

export type DraftStore = StoreApi<DraftState & DraftActions> & { brushLayers: BrushLayers; brushBlobs: Map<string, { version: number; blobId: string }> };

function clone(doc: ScenarioDoc): ScenarioDoc {
  return JSON.parse(JSON.stringify(doc)) as ScenarioDoc;
}

/**
 * A draft store. `n` is the run's row count (brush layers); `now` is injectable for tests.
 */
export function createDraftStore(opts: { n: number; doc?: ScenarioDoc; now?: () => number }): DraftStore {
  const now = opts.now ?? (() => Date.now());
  const brushLayers = new BrushLayers(opts.n);
  const store = createStore<DraftState & DraftActions>()((set, get) => {
    const pushPast = (key: string | null): Pick<DraftState, "past" | "future" | "lastKey" | "lastAt"> | null => {
      const s = get();
      const t = now();
      if (key !== null && s.lastKey === key && t - s.lastAt < COALESCE_MS && s.past.length > 0) return { past: s.past, future: [], lastKey: key, lastAt: t };
      const past = [...s.past, { doc: s.doc, brush: s.brush }];
      if (past.length > HISTORY_LIMIT) past.splice(0, past.length - HISTORY_LIMIT);
      return { past, future: [], lastKey: key, lastAt: t };
    };
    // A new change after a failed save lets the autosave loop try again.
    const retry = (s: DraftState): Partial<DraftState> => (s.save === "error" ? { save: "idle", saveError: null } : {});
    return {
      sid: null,
      revision: null,
      status: null,
      serverUpdated: null,
      doc: opts.doc ? clone(opts.doc) : emptyDoc(),
      brush: {},
      past: [],
      future: [],
      version: 0,
      savedVersion: 0,
      save: "idle",
      saveError: null,
      forkedFrom: null,
      lastKey: null,
      lastAt: 0,
      change: (fn, o = {}) => {
        const s = get();
        const next = fn(clone(s.doc));
        if (serializeDoc(next) === serializeDoc(s.doc)) return;
        set({ ...pushPast(o.coalesce ?? null), doc: next, version: s.version + 1, ...retry(s) });
      },
      commitBrush: (o = {}) => {
        const s = get();
        const snap = brushLayers.snapshot();
        if (snapshotsEqual(snap, s.brush)) return;
        set({ ...pushPast(o.coalesce ?? null), brush: snap, version: s.version + 1, ...retry(s) });
      },
      undo: () => {
        const s = get();
        const prev = s.past[s.past.length - 1];
        if (!prev) return;
        brushLayers.restore(prev.brush);
        set({ past: s.past.slice(0, -1), future: [{ doc: s.doc, brush: s.brush }, ...s.future], doc: prev.doc, brush: prev.brush, version: s.version + 1, lastKey: null });
      },
      redo: () => {
        const s = get();
        const next = s.future[0];
        if (!next) return;
        brushLayers.restore(next.brush);
        set({ future: s.future.slice(1), past: [...s.past, { doc: s.doc, brush: s.brush }].slice(-HISTORY_LIMIT), doc: next.doc, brush: next.brush, version: s.version + 1, lastKey: null });
      },
      load: (sc) => {
        brushLayers.restore({});
        const v = get().version + 1;
        set({
          sid: sc.id,
          revision: sc.revision,
          status: sc.status,
          serverUpdated: sc.updated_utc,
          doc: clone(sc.doc),
          brush: {},
          past: [],
          future: [],
          version: v,
          savedVersion: v,
          save: "idle",
          saveError: null,
          forkedFrom: null,
          lastKey: null,
        });
      },
      rebase: (sc, forkedFrom = null) => set({ sid: sc.id, revision: sc.revision, status: sc.status, serverUpdated: sc.updated_utc, forkedFrom }),
      setSave: (save, error = null) => set({ save, saveError: error }),
      markSaved: (version) => set({ savedVersion: Math.max(get().savedVersion, version), save: "saved", saveError: null }),
    };
  });
  return Object.assign(store, { brushLayers, brushBlobs: new Map<string, { version: number; blobId: string }>() });
}

export const isDirty = (s: Pick<DraftState, "version" | "savedVersion">) => s.version > s.savedVersion;
export const canUndo = (s: Pick<DraftState, "past">) => s.past.length > 0;
export const canRedo = (s: Pick<DraftState, "future">) => s.future.length > 0;

// ---------------------------------------------------------------- registry

const drafts = new Map<string, DraftStore>();

const draftKey = (rid: string, sid: string | null) => `${rid}|${sid ?? "new"}`;

/** The draft for a run and scenario (a new one when absent). */
export function getDraft(rid: string, sid: string | null, n: number): DraftStore {
  const k = draftKey(rid, sid);
  let d = drafts.get(k);
  if (!d || d.brushLayers.n !== n) {
    d = createDraftStore({ n, doc: emptyDoc("Untitled scenario", rid) });
    drafts.set(k, d);
  }
  return d;
}

/** Re-file a draft under its new scenario id (after the first save or a fork). */
export function moveDraft(rid: string, from: string | null, to: string): void {
  const d = drafts.get(draftKey(rid, from));
  if (!d) return;
  drafts.delete(draftKey(rid, from));
  drafts.set(draftKey(rid, to), d);
}

export function forgetDrafts(): void {
  drafts.clear();
}

// ---------------------------------------------------------------- autosave

export type AutosaveDeps = {
  create: (doc: ScenarioDoc) => Promise<Scenario>;
  patch: (sid: string, doc: ScenarioDoc) => Promise<Scenario>;
  fork: (sid: string, doc: ScenarioDoc) => Promise<Scenario>;
  /** Upload one lever's brush layer as an edit blob; returns the blob id. */
  uploadBrush?: (lever: string, body: Uint8Array, count: number) => Promise<string>;
};

export type AutosaveEvents = {
  /** A new draft got its id (first save). */
  onCreated?: (s: Scenario) => void;
  /** A conflict forked a new revision; editing continues on it. */
  onForked?: (s: Scenario, from: string) => void;
  onError?: (e: unknown) => void;
};

export const BRUSH_LABEL = "brush";

/**
 * The doc as saved: the editor's edits plus one per-cell edit per brushed lever, pointing at
 * the uploaded edit blob (`per_cell_ref: "blob:<id>"`, label "brush").
 */
export function docForSave(doc: ScenarioDoc, brushRefs: Record<string, string>): ScenarioDoc {
  const extra: Edit[] = Object.entries(brushRefs).map(([lever, blobId]) => ({ lever, mode: "per_cell", per_cell_ref: `blob:${blobId}`, label: BRUSH_LABEL }));
  return extra.length ? { ...doc, edits: [...doc.edits, ...extra] } : doc;
}

/** Blob refs of brushed levers whose current layer is already uploaded (for compiling the saved doc). */
export function uploadedBrushRefs(store: DraftStore): Record<string, string> {
  const refs: Record<string, string> = {};
  for (const lever of store.brushLayers.levers()) {
    const cached = store.brushBlobs.get(lever);
    if (cached && cached.version === store.brushLayers.version(lever)) refs[lever] = cached.blobId;
  }
  return refs;
}

export type Autosave = { start: () => void; stop: () => void; flush: () => Promise<void>; saving: () => boolean };

/**
 * Save the draft every `intervalMs` (2 s) while it has unsaved changes, and on `flush()`
 * (editor blur, page hide). New drafts are created, saved drafts patched, and a
 * `409 conflict_revision` forks a new revision with the edited doc.
 */
export function createAutosave(store: DraftStore, deps: AutosaveDeps, events: AutosaveEvents = {}, intervalMs = 2000): Autosave {
  let timer: ReturnType<typeof setInterval> | null = null;
  let inflight: Promise<void> | null = null;

  const brushRefs = async (): Promise<Record<string, string>> => {
    const refs: Record<string, string> = {};
    const layers = store.brushLayers;
    for (const lever of layers.levers()) {
      const v = layers.version(lever);
      const cached = store.brushBlobs.get(lever);
      if (cached && cached.version === v) {
        refs[lever] = cached.blobId;
        continue;
      }
      if (!deps.uploadBrush) continue;
      const { body, count } = layers.blobBody(lever);
      const blobId = await deps.uploadBrush(lever, body, count);
      store.brushBlobs.set(lever, { version: v, blobId });
      refs[lever] = blobId;
    }
    return refs;
  };

  const saveOnce = async (): Promise<void> => {
    const s = store.getState();
    if (!isDirty(s)) return;
    const version = s.version;
    s.setSave("saving");
    try {
      const doc = docForSave(s.doc, await brushRefs());
      if (s.sid === null) {
        const created = await deps.create(doc);
        store.getState().rebase(created);
        store.getState().markSaved(version);
        events.onCreated?.(created);
        return;
      }
      try {
        const saved = await deps.patch(s.sid, doc);
        store.getState().rebase(saved, store.getState().forkedFrom);
        store.getState().markSaved(version);
      } catch (e) {
        if (!(e instanceof ApiError && e.status === 409 && e.code === "conflict_revision")) throw e;
        const from = s.sid;
        const forked = await deps.fork(from, doc);
        store.getState().rebase(forked, from);
        store.getState().markSaved(version);
        events.onForked?.(forked, from);
      }
    } catch (e) {
      store.getState().setSave("error", e instanceof Error ? e.message : String(e));
      events.onError?.(e);
    }
  };

  const flush = (): Promise<void> => {
    if (inflight) {
      // A change made while a save was in flight is saved right after it.
      return inflight.then(() => (isDirty(store.getState()) ? flush() : undefined));
    }
    if (!isDirty(store.getState())) return Promise.resolve();
    inflight = saveOnce().finally(() => {
      inflight = null;
    });
    return inflight;
  };

  return {
    start() {
      if (timer !== null) return;
      timer = setInterval(() => {
        if (!inflight && isDirty(store.getState()) && store.getState().save !== "error") void flush();
      }, intervalMs);
    },
    stop() {
      if (timer !== null) clearInterval(timer);
      timer = null;
    },
    flush,
    saving: () => inflight !== null,
  };
}

/** True when two docs differ in evaluated content (not just name, tags or notes). */
export function contentChanged(a: ScenarioDoc, b: ScenarioDoc): boolean {
  return contentKey(a) !== contentKey(b);
}
