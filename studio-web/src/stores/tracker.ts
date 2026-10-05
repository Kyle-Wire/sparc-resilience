// Tracker of one job (Mission Control, SPEC §5.4–5.10, §12.4).
//
// `applyEvent(state, event)` is a pure reducer that mirrors the server projection
// `sparc/studio/jobs/tracker.py::reduce` rule for rule, over the same JSON state shape
// (snake_case keys): span tree, stage states, planned/done units and cost-weighted progress,
// metrics latest + series, warnings deduplicated by code + message hash, artifacts,
// checkpoints, heartbeat gaps, current path and nested study children. `toContract(state)`
// returns the canonical projection shared with Python's `to_contract` (SPEC §14.3), and
// `projectionEta` ports the live ETA (`eta.projection_eta`) with seed or host rates.
//
// Mission Control loads `GET /api/jobs/{jid}/tracker` (snapshot + cursor), turns it into a
// state with `fromSnapshot`, then streams the job's events with `after=cursor`; the stream
// manager delivers them once per animation frame and `applyEvents` folds each batch. The
// wire snapshot is a summary (spans to depth 4, no unit counters, no child sub-projections),
// so the state rebuilt from it is shown at once and then replaced by the exact one: the job's
// event log up to the snapshot cursor (`GET /api/jobs/{jid}/events`, paged) folded from the
// first line, plus the events streamed meanwhile.
//
// Purity: `applyEvents` never mutates its input. It copies containers on first write within
// a batch (a small copy-on-write draft), so untouched parts keep their identity and React
// selectors re-render only what changed.
import { useEffect } from "react";
import { create } from "zustand";
import { ApiError } from "../api/client";
import { getStreams, type JobStreamHandle, type StreamManager } from "../api/sse";
import {
  getEvents,
  getJob,
  getTracker,
  isFinalStatus,
  type ArtifactRow,
  type ChildRow,
  type CheckpointRow,
  type HeartbeatGap,
  type LogLine,
  type MetricLatest,
  type ResourceRow,
  type SeriesPoint,
  type SnapshotStage,
  type Span,
  type SpanKind,
  type SpanStatus,
  type StageUiState,
  type TrackerSnapshot,
  type UnitRate,
  type WarningRow,
} from "../api/tracking";
import type { Job, JobEvent, JobStatus, PlanNode, StageId } from "../api/types";

// ---------------------------------------------------------------- constants

export const STAGE_IDS: readonly StageId[] = ["S0", "S1", "S2_S3", "baselines", "cv_curve", "S4", "S5", "climate", "S6", "S7", "finish"];
const SERIES_NAMES = new Set(["candidate_rmse", "heldout_rmse", "mean_benefit"]);
const SERIES_PREFIXES = ["scenario.", "cv_row.", "influence."];
export const MAX_SPANS = 2000;
export const MAX_DEPTH = 4;
/** Heartbeats further apart than this render as a "system sleep?" band. */
export const GAP_S = 45;
const OBS_KEEP = 50;
const SERIES_KEEP = 2000;
const FINAL_JOB = new Set(["succeeded", "failed", "cancelled", "interrupted"]);
const STAGE_END: Record<string, StageUiState> = { ok: "done", error: "failed", cancelled: "cancelled" };
const SPAN_STATUS: Record<string, SpanStatus> = { succeeded: "ok", failed: "error", cancelled: "cancelled", interrupted: "error", ok: "ok", error: "error" };
const SETTLED = new Set(["done", "failed", "cancelled", "running"]);

/**
 * Seed rates of SPEC §5.4 (seconds per unit, full Providence on 4 cores). They are the
 * progress weights in both reducers and the ETA priors; `/api/meta.unit_costs` serves the
 * same table. `*` entries match a prefix.
 */
export const SEED_RATES: Readonly<Record<string, number>> = {
  "base_fit:mgwr": 170.0,
  "base_fit:gwrf": 23.0,
  "base_fit:gam": 6.0,
  "base_fit:physics": 4.0,
  "base_fit:ols": 0.1,
  adv_refit: 4.5,
  "stacker_fit:mean": 0.2,
  "stacker_fit:nnls": 0.2,
  "stacker_fit:residual": 14.0,
  "baseline_fit:*": 12.0,
  engine_pass: 13.0,
  "causal_step:dml": 10.0,
  "causal_step:spillover": 15.0,
  "causal_step:cate": 1.0,
  "causal_step:dr": 22.0,
  "causal_step:sens": 0.5,
  "causal_step:audit": 0.5,
  s0_load: 0.3,
  s1_influence: 3.3,
  checkpoint_save: 2.1,
  pareto: 0.5,
  climate_model: 20.0,
  remote_object: 5.0,
  "replicate:*": 254.0,
  "variant:*": 1200.0,
  unpickle: 26.25,
  mediator_fit: 1.0,
};
const DEFAULT_RATE = 1.0;
const REF_CELLS = 54_701;
const REF_THREADS = 4;
const RANGE_FRAC = 0.25;
const ALPHA: Record<string, number> = { "base_fit:mgwr": 1.3, "base_fit:gwrf": 1.15 };
const UNSCALED_PREFIXES = ["climate_model", "remote_object", "replicate:", "variant:"];

/** Seed seconds of `unit`: an exact entry, else the `<prefix>:*` entry, else 1 s. */
export function seedRate(unit: string): number {
  const exact = SEED_RATES[unit];
  if (exact !== undefined) return exact;
  const i = unit.indexOf(":");
  if (i >= 0) {
    const wild = SEED_RATES[unit.slice(0, i) + ":*"];
    if (wild !== undefined) return wild;
  }
  return DEFAULT_RATE;
}

/** Progress weight of one unit (its seed rate; identical in the Python projection). */
export function unitWeight(unit: string): number {
  return seedRate(unit);
}

// ---------------------------------------------------------------- state shape

export type RunInfo = {
  name?: string | null;
  stages?: string[] | null;
  fast?: boolean | null;
  coarse?: number | null;
  resume?: boolean | null;
  cv_curve?: boolean | null;
  run_meta?: Record<string, unknown>;
  started_ts?: number | null;
  run_dir?: string | null;
  fingerprint?: string | null;
  ended_ts?: number | null;
  elapsed_s?: number | null;
  done?: string[];
  timings_s?: Record<string, number>;
  error?: unknown;
};

export type StageRow = SnapshotStage;

export type SpanRow = Omit<Span, "started_ts"> & { started_ts: number | null; path: string[]; depth: number };

export type WarningEntry = WarningRow & { msg_hash: string };

export type ChildEntry = { key: string; label: string; run_id: string | null; job_id: string | null; state: TrackerState };

/** An open item level: a task counting items (`k` of `n`, `k` from 1) and the fraction done inside item `k`. */
export type KnLevel = { span: string | null; label: string; depth: number; k: number; n: number; frac: number };

/** The projection state (same keys and meaning as `tracker.py::new_state`). */
export type TrackerState = {
  cursor: number;
  n_events: number;
  first_ts: number | null;
  last_ts: number | null;
  run: RunInfo | null;
  plan: PlanNode[] | null;
  n_points: number | null;
  stages: Record<string, StageRow> | null;
  spans: Record<string, SpanRow>;
  running: string[];
  metrics_latest: Record<string, MetricLatest>;
  metric_series: Record<string, SeriesPoint[]>;
  warnings: Record<string, WarningEntry>;
  artifacts: Record<string, ArtifactRow>;
  checkpoints: CheckpointRow[];
  hb_last: Record<string, number>;
  heartbeat_gaps: HeartbeatGap[];
  planned_units: Record<string, number>;
  done_units: Record<string, number>;
  stage_done: Record<string, Record<string, number>>;
  /** span id → [stage, unit, fraction of one unit] of its latest k < n tick. */
  partial: Record<string, [string | null, string, number]>;
  stage_partial: Record<string, Record<string, number>>;
  unit_obs: Record<string, number[]>;
  stage_elapsed: Record<string, number>;
  progress: number | null;
  current_path: string[] | null;
  stage: string | null;
  tick: { depth: number; frac: number } | null;
  /** Open item levels (tasks with k of n, outermost first): the nested item fraction of a job without a plan. */
  kn: KnLevel[];
  status: JobStatus | null;
  exit_code: number | null;
  error: unknown;
  result: Record<string, unknown> | null;
  run_status: string | null;
  finished: boolean;
  cancel: { by: unknown; ts: number | null } | null;
  children: Record<string, ChildEntry>;
  child_of: Record<string, string>;
};

export function newTrackerState(): TrackerState {
  return {
    cursor: -1, n_events: 0, first_ts: null, last_ts: null,
    run: null, plan: null, n_points: null, stages: null,
    spans: {}, running: [],
    metrics_latest: {}, metric_series: {},
    warnings: {}, artifacts: {}, checkpoints: [],
    hb_last: {}, heartbeat_gaps: [],
    planned_units: {}, done_units: {}, stage_done: {}, partial: {}, stage_partial: {},
    unit_obs: {}, stage_elapsed: {},
    progress: null, current_path: null, stage: null, tick: null, kn: [],
    status: null, exit_code: null, error: null, result: null,
    run_status: null, finished: false, cancel: null,
    children: {}, child_of: {},
  };
}

// ---------------------------------------------------------------- small helpers

type Ev = Record<string, unknown> & { type?: unknown };
type Dict = Record<string, unknown>;

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isDict = (v: unknown): v is Dict => !!v && typeof v === "object" && !Array.isArray(v);
/** Python truthiness of a JSON value. */
const truthy = (v: unknown): boolean =>
  !(v === null || v === undefined || v === false || v === 0 || v === "" || (Array.isArray(v) && v.length === 0) || (isDict(v) && Object.keys(v).length === 0));
const orNull = <T>(v: T | undefined): T | null => (v === undefined ? null : v);
/** `str(v)` as Python prints JSON scalars (None, True, False). */
function pyStr(v: unknown): string {
  if (v === null || v === undefined) return "None";
  if (v === true) return "True";
  if (v === false) return "False";
  return String(v);
}

/** Python's `round(x, nd)`: nearest, ties to even (on the decimal shown). */
export function pyRound(x: number, nd: number): number {
  const m = 10 ** nd;
  const y = x * m;
  const r = Math.round(y);
  const tie = Math.abs(y - Math.trunc(y)) === 0.5;
  const v = tie ? (r % 2 === 0 ? r : r - 1) : r;
  return v / m;
}

/** `json.dumps(list_of_str)` with Python's defaults (", " separator, ASCII escapes). */
export function pyJsonList(items: readonly unknown[]): string {
  const enc = (s: unknown) =>
    typeof s === "string"
      ? JSON.stringify(s).replace(/[\u007f-￿]/g, (c) => "\\u" + c.charCodeAt(0).toString(16).padStart(4, "0"))
      : JSON.stringify(s ?? null);
  return "[" + items.map(enc).join(", ") + "]";
}

// SHA-1 of the UTF-8 bytes, as hex (the warning dedup key uses its first 12 characters,
// exactly like the Python projection, so a server projection and live events agree).
export function sha1Hex(text: string): string {
  const bytes = new TextEncoder().encode(text);
  const len = bytes.length;
  const words = new Uint32Array((((len + 8) >> 6) + 1) * 16);
  for (let i = 0; i < len; i++) words[i >> 2] |= bytes[i] << (24 - (i % 4) * 8);
  words[len >> 2] |= 0x80 << (24 - (len % 4) * 8);
  const bits = len * 8;
  words[words.length - 1] = bits >>> 0;
  words[words.length - 2] = Math.floor(bits / 2 ** 32);
  let h0 = 0x67452301, h1 = 0xefcdab89, h2 = 0x98badcfe, h3 = 0x10325476, h4 = 0xc3d2e1f0;
  const w = new Uint32Array(80);
  for (let off = 0; off < words.length; off += 16) {
    for (let i = 0; i < 16; i++) w[i] = words[off + i];
    for (let i = 16; i < 80; i++) {
      const x = w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16];
      w[i] = (x << 1) | (x >>> 31);
    }
    let a = h0, b = h1, c = h2, d = h3, e = h4;
    for (let i = 0; i < 80; i++) {
      let f: number, k: number;
      if (i < 20) { f = (b & c) | (~b & d); k = 0x5a827999; }
      else if (i < 40) { f = b ^ c ^ d; k = 0x6ed9eba1; }
      else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8f1bbcdc; }
      else { f = b ^ c ^ d; k = 0xca62c1d6; }
      const t = (((a << 5) | (a >>> 27)) + f + e + k + w[i]) >>> 0;
      e = d; d = c; c = ((b << 30) | (b >>> 2)) >>> 0; b = a; a = t;
    }
    h0 = (h0 + a) >>> 0; h1 = (h1 + b) >>> 0; h2 = (h2 + c) >>> 0; h3 = (h3 + d) >>> 0; h4 = (h4 + e) >>> 0;
  }
  return [h0, h1, h2, h3, h4].map((h) => h.toString(16).padStart(8, "0")).join("");
}

function tagText(v: unknown): string {
  if (v === true) return "true";
  if (v === false) return "false";
  if (typeof v === "number" && Number.isInteger(v) && Math.abs(v) < 1e15) return String(v);
  return pyStr(v);
}

/** `name` without tags, else `name{k1=v1,k2=v2}` with tags sorted by key (api.md §3 MetricKey). */
export function metricKey(name: string, tags: Record<string, unknown> | null | undefined): string {
  const keys = tags ? Object.keys(tags).sort() : [];
  if (!keys.length) return name;
  return name + "{" + keys.map((k) => `${k}=${tagText(tags![k])}`).join(",") + "}";
}

/** The stage id named in a span path (`stage:<id>[…]`), or null. */
export function stageOf(path: readonly unknown[] | null | undefined): string | null {
  for (const el of path ?? []) {
    if (typeof el === "string" && el.startsWith("stage:")) return el.slice(6).split("[", 1)[0];
  }
  return null;
}

function ancestry(ev: Ev): string[] {
  const p = ev.type === "artifact" ? ev.span_path : ev.path;
  return Array.isArray(p) ? (p as string[]) : [];
}

function childIndex(path: readonly unknown[]): number | null {
  for (let i = 1; i < path.length; i++) {
    const el = path[i];
    if (typeof el === "string" && el.startsWith("run:")) return i;
  }
  return null;
}

/** True when an event belongs to a nested run (a study child), not to the job's own run. */
export function isNested(ev: Ev): boolean {
  return childIndex(ancestry(ev)) !== null;
}

function weightSum(units: Record<string, number>): number {
  let s = 0;
  for (const [u, n] of Object.entries(units)) if (n) s += unitWeight(u) * Number(n);
  return s;
}

// ---------------------------------------------------------------- copy-on-write draft

/** Objects copied (or created) during the current batch may be mutated in place; others are copied first. */
class Draft {
  private owned = new WeakSet<object>();

  own<T extends object>(o: T): T {
    if (this.owned.has(o)) return o;
    const c = (Array.isArray(o) ? o.slice() : { ...o }) as T;
    this.owned.add(c);
    return c;
  }

  /** A newly built object: owned by this batch. */
  fresh<T extends object>(o: T): T {
    this.owned.add(o);
    return o;
  }

  /** `parent[key]`, owned (copied into the owned `parent` if needed). */
  at<T extends object>(parent: object, key: string): T {
    const p = parent as Record<string, unknown>;
    const cur = p[key] as T;
    const o = this.own(cur);
    if (o !== cur) p[key] = o;
    return o;
  }
}

// ---------------------------------------------------------------- reducer

/** Apply one event (in file order). Pure: returns a new state, `state` is left untouched. */
export function applyEvent(state: TrackerState, ev: JobEvent | Ev): TrackerState {
  return applyEvents(state, [ev]);
}

/** Apply a batch of events in order (one copy-on-write pass). Transient `resource` events are ignored. */
export function applyEvents(state: TrackerState, events: readonly (JobEvent | Ev)[]): TrackerState {
  if (!events.length) return state;
  const d = new Draft();
  const s = d.own(state);
  for (const raw of events) {
    const ev = raw as Ev;
    if (ev.type === "resource") continue;
    const cursor = isNum(ev.cursor) ? ev.cursor : null;
    reduce(d, s, ev, cursor);
  }
  return s;
}

/** Fold a whole event list from scratch (tests, the cross-language contract, replays). */
export function replay(events: readonly (JobEvent | Ev)[], state: TrackerState = newTrackerState()): TrackerState {
  return applyEvents(state, events);
}

function reduce(d: Draft, s: TrackerState, ev: Ev, cursor: number | null): void {
  if (cursor !== null) s.cursor = cursor;
  s.n_events += 1;
  const ts = ev.ts;
  if (isNum(ts)) {
    if (s.first_ts === null) s.first_ts = ts;
    s.last_ts = s.last_ts === null ? ts : Math.max(s.last_ts, ts);
  }
  const t = ev.type;
  const path = ancestry(ev);
  const idx = childIndex(path);
  if (idx !== null) {
    childEvent(d, s, ev, path, idx, cursor);
    spanEvent(d, s, ev, path);
    if (t === "warning") warning(d, s, ev, cursor);
    updateCurrent(s);
    updateProgress(d, s);
    return;
  }
  const handler = typeof t === "string" ? HANDLERS[t] : undefined;
  if (handler) handler(d, s, ev, path, cursor);
  updateCurrent(s);
  updateProgress(d, s);
}

type Handler = (d: Draft, s: TrackerState, ev: Ev, path: string[], cursor: number | null) => void;

function onRunStart(d: Draft, s: TrackerState, ev: Ev, path: string[]): void {
  spanEvent(d, s, ev, path);
  const run = s.run ? d.at<RunInfo>(s, "run") : d.fresh<RunInfo>({});
  Object.assign(run, {
    name: orNull(ev.name), stages: orNull(ev.stages), fast: orNull(ev.fast), coarse: orNull(ev.coarse), resume: orNull(ev.resume),
    cv_curve: orNull(ev.cv_curve), run_meta: truthy(ev.run_meta) ? ev.run_meta : {}, started_ts: orNull(ev.ts),
  });
  s.run = run;
  s.run_status = "running";
}

function onRunDir(d: Draft, s: TrackerState, ev: Ev): void {
  const run = s.run ? d.at<RunInfo>(s, "run") : d.fresh<RunInfo>({});
  Object.assign(run, { run_dir: orNull(ev.run_dir), fingerprint: orNull(ev.fingerprint) });
  s.run = run;
}

function stageRow(prev: StageRow | undefined, state: StageUiState, reason: string | null): StageRow {
  const st: StageRow = prev
    ? { ...prev }
    : { state: "planned", reason: null, started_ts: null, ended_ts: null, elapsed_s: null, est_s: null, progress: null };
  st.state = state;
  st.reason = reason;
  return st;
}

function skipState(reason: string | null | undefined): StageUiState {
  const r = reason ?? "";
  if (r === "checkpoint") return "cached";
  if (r === "not_requested") return "not_requested";
  if (r.startsWith("disabled_by_config")) return "disabled";
  return "skipped";
}

function planState(node: PlanNode): [StageUiState, string | null] {
  const st = node.state ?? "will_run";
  const reason = node.reason ?? null;
  if (st === "will_run") return ["planned", reason];
  if (st === "cached") return ["cached", reason || "checkpoint"];
  return [skipState(reason), reason];
}

function stagesOf(d: Draft, s: TrackerState): Record<string, StageRow> {
  if (s.stages === null) s.stages = d.fresh({});
  return d.at<Record<string, StageRow>>(s, "stages");
}

function onRunPlan(d: Draft, s: TrackerState, ev: Ev): void {
  const nodes = (Array.isArray(ev.nodes) ? ev.nodes : []).filter((n): n is PlanNode => isDict(n) && truthy((n as Dict).id)).map((n) => ({ ...n }));
  s.plan = nodes;
  if (ev.n_points !== undefined && ev.n_points !== null) s.n_points = ev.n_points as number;
  const stages = stagesOf(d, s);
  const planned: Record<string, number> = {};
  for (const node of nodes) {
    const sid = node.id;
    const st = stages[sid];
    if (st === undefined || !SETTLED.has(st.state)) {
      const [ns, reason] = planState(node);
      const row = d.fresh(stageRow(st, ns, reason));
      if (node.est_s !== undefined && node.est_s !== null) row.est_s = node.est_s;
      stages[sid] = row;
    }
    if ((node.state ?? "will_run") === "will_run") {
      for (const [u, n] of Object.entries(node.units ?? {})) planned[u] = (planned[u] ?? 0) + Number(n || 0);
    }
  }
  s.planned_units = planned;
}

function onStageStart(d: Draft, s: TrackerState, ev: Ev, path: string[]): void {
  spanEvent(d, s, ev, path);
  const sid = ev.stage;
  if (!truthy(sid)) return;
  const stages = stagesOf(d, s);
  const st = d.fresh(stageRow(stages[sid as string], "running", null));
  st.started_ts = orNull(ev.ts) as number | null;
  st.ended_ts = st.elapsed_s = null;
  if (ev.est_s !== undefined && ev.est_s !== null) st.est_s = ev.est_s as number;
  stages[sid as string] = st;
}

function onStageEnd(d: Draft, s: TrackerState, ev: Ev, path: string[]): void {
  spanEvent(d, s, ev, path);
  const sid = ev.stage as string;
  if (!truthy(sid)) return;
  const stages = stagesOf(d, s);
  const status = "status" in ev ? ev.status : "ok";
  const st = d.fresh(stageRow(stages[sid], STAGE_END[status as string] ?? "failed", null));
  st.ended_ts = orNull(ev.ts) as number | null;
  let el = orNull(ev.elapsed_s) as number | null;
  if (el === null && st.started_ts !== null && isNum(ev.ts)) el = ev.ts - st.started_ts;
  st.elapsed_s = el;
  stages[sid] = st;
  if (el !== null) d.at<Record<string, number>>(s, "stage_elapsed")[sid] = el;
  if (status === "ok" && (sid === "S0" || sid === "S1")) complete(d, s, sid, sid === "S0" ? "s0_load" : "s1_influence", el);
}

function onStageSkip(d: Draft, s: TrackerState, ev: Ev): void {
  const sid = ev.stage as string;
  if (!truthy(sid)) return;
  const stages = stagesOf(d, s);
  const cur = stages[sid];
  if (cur !== undefined && (cur.state === "done" || cur.state === "failed" || cur.state === "cancelled")) return;
  const reason = orNull(ev.reason) as string | null;
  stages[sid] = d.fresh(stageRow(cur, skipState(reason), reason));
}

function onTaskStart(d: Draft, s: TrackerState, ev: Ev, path: string[]): void {
  spanEvent(d, s, ev, path);
  const k = ev.k;
  const n = ev.n;
  if (isNum(k) && isNum(n) && n > 0 && path.length) {
    // an item k of n starts: it nests in the open item levels that are its ancestors
    const kn = s.kn.filter((lv) => lv.depth < path.length && path[lv.depth - 1] === lv.label);
    kn.push({ span: orNull(ev.span) as string | null, label: path[path.length - 1], depth: path.length, k, n, frac: 0 });
    s.kn = d.fresh(kn);
  }
}

/** Index of the innermost item level whose task contains an event at `path`. */
function knLevel(s: TrackerState, path: string[]): number | null {
  for (let i = s.kn.length - 1; i >= 0; i--) {
    const lv = s.kn[i];
    if (lv.depth <= path.length && path[lv.depth - 1] === lv.label) return i;
  }
  return null;
}

/** Nested item fraction: `(k − 1 + inner) / n` from the innermost item level out. */
function knProgress(kn: readonly KnLevel[]): number {
  let p = kn[kn.length - 1].frac;
  for (let i = kn.length - 1; i >= 0; i--) p = (Math.min(kn[i].k, kn[i].n) - 1 + p) / kn[i].n;
  return p;
}

function onTaskEnd(d: Draft, s: TrackerState, ev: Ev, path: string[], cursor: number | null): void {
  spanEvent(d, s, ev, path);
  if (ev.span !== null && ev.span !== undefined) {
    const i = s.kn.findIndex((lv) => lv.span === ev.span);
    if (i >= 0) s.kn = d.fresh(s.kn.slice(0, i + 1).map((lv, j) => (j === i && ev.status === "ok" ? { ...lv, frac: 1 } : lv)));
  }
  const span = ev.span;
  const pkey = String(span ?? null);
  if (truthy(span) && pkey in s.partial) {
    delete d.at<Dict>(s, "partial")[pkey];
    recomputePartial(d, s);
  }
  const unit = ev.unit;
  if (ev.status === "ok" && truthy(unit)) complete(d, s, stageOf(path), String(unit), ev.elapsed_s);
  const metrics = ev.metrics;
  if (isDict(metrics) && Object.keys(metrics).length) {
    const tags: Dict = { task: orNull(ev.name) };
    if (ev.key !== undefined && ev.key !== null) tags.key = ev.key;
    for (const [name, value] of Object.entries(metrics)) {
      if (value === null || value === undefined || typeof value === "number" || typeof value === "string" || typeof value === "boolean") {
        metric(d, s, name, value ?? null, null, tags, orNull(ev.ts) as number | null, cursor);
      }
    }
  }
}

function onTick(d: Draft, s: TrackerState, ev: Ev, path: string[]): void {
  const k = ev.k;
  const n = ev.n;
  const unit = ev.unit;
  const span = ev.span;
  const depth = path.length;
  let frac = ev.frac as number | null | undefined;
  if ((frac === null || frac === undefined) && isNum(k) && isNum(n) && n) frac = k / n;
  if (frac !== null && frac !== undefined && isNum(Number(frac))) {
    const cur = s.tick;
    if (cur === null || depth <= cur.depth) s.tick = { depth, frac: Math.max(0, Math.min(1, Number(frac))) };
    const i = knLevel(s, path);
    if (i !== null) s.kn = d.fresh(s.kn.slice(0, i + 1).map((lv, j) => (j === i ? { ...lv, frac: Math.max(0, Math.min(1, Number(frac))) } : lv)));
  }
  if (!truthy(unit) || !isNum(k) || !isNum(n) || n <= 0) return;
  const stage = stageOf(path);
  const pkey = String(span ?? null);
  if (k >= n) {
    if (pkey in s.partial) delete d.at<Dict>(s, "partial")[pkey];
    complete(d, s, stage, String(unit), ev.pass_s);
  } else {
    d.at<Dict>(s, "partial")[pkey] = [stage, String(unit), Math.max(0, k / n)];
  }
  recomputePartial(d, s);
}

function onMetric(d: Draft, s: TrackerState, ev: Ev, _path: string[], cursor: number | null): void {
  metric(d, s, ev.name as string, ev.value ?? null, orNull(ev.unit) as string | null, isDict(ev.tags) ? ev.tags : {}, orNull(ev.ts) as number | null, cursor);
}

function onArtifact(d: Draft, s: TrackerState, ev: Ev, path: string[]): void {
  const rel = ev.path;
  if (typeof rel !== "string") return;
  const stage = (truthy(ev.stage) ? (ev.stage as string) : null) ?? stageOf(path);
  const bytes = Math.trunc(Number(ev.bytes || 0)) || 0;
  const arts = d.at<Record<string, ArtifactRow>>(s, "artifacts");
  const row = arts[rel];
  if (row === undefined) {
    arts[rel] = d.fresh({ relpath: rel, role: truthy(ev.role) ? String(ev.role) : "", bytes, stage, ts: orNull(ev.ts) as number | null });
  } else {
    const r = d.at<ArtifactRow>(arts, rel);
    r.role = truthy(ev.role) ? String(ev.role) : r.role;
    r.bytes = bytes;
    r.stage = stage || r.stage;
    r.ts = orNull(ev.ts) as number | null;
  }
}

function onCheckpoint(d: Draft, s: TrackerState, ev: Ev, path: string[]): void {
  const action = orNull(ev.action) as string;
  d.at<CheckpointRow[]>(s, "checkpoints").push({ action, done: Array.isArray(ev.done) ? [...(ev.done as string[])] : [], bytes: orNull(ev.bytes) as number | null, ts: orNull(ev.ts) as number | null });
  if (action === "saved") complete(d, s, stageOf(path), "checkpoint_save", ev.elapsed_s);
}

function warning(d: Draft, s: TrackerState, ev: Ev, cursor: number | null): void {
  const code = truthy(ev.code) ? pyStr(ev.code) : "warning";
  const msg = truthy(ev.message) ? pyStr(ev.message) : "";
  const h = sha1Hex(msg).slice(0, 12);
  const key = `${code}|${h}`;
  const ws = d.at<Record<string, WarningEntry>>(s, "warnings");
  if (ws[key] === undefined) {
    ws[key] = d.fresh({
      code, lvl: truthy(ev.lvl) ? String(ev.lvl) : "warning", message: msg, count: 1, stage: stageOf(ancestry(ev)),
      first_cursor: cursor ?? -1, data: isDict(ev.data) ? ev.data : {}, msg_hash: h,
    });
  } else {
    d.at<WarningEntry>(ws, key).count += 1;
  }
}

function onHeartbeat(d: Draft, s: TrackerState, ev: Ev): void {
  const ts = ev.ts;
  if (!isNum(ts)) return;
  const pid = pyStr(ev.pid);
  const last = s.hb_last[pid];
  if (last !== undefined && ts - last > GAP_S) d.at<HeartbeatGap[]>(s, "heartbeat_gaps").push({ from_ts: last, to_ts: ts });
  d.at<Record<string, number>>(s, "hb_last")[pid] = ts;
}

function onRunEnd(d: Draft, s: TrackerState, ev: Ev, path: string[]): void {
  spanEvent(d, s, ev, path);
  const status = orNull(ev.status) as string | null;
  s.run_status = status;
  const run = s.run ? d.at<RunInfo>(s, "run") : d.fresh<RunInfo>({});
  Object.assign(run, {
    ended_ts: orNull(ev.ts), elapsed_s: orNull(ev.elapsed_s), done: truthy(ev.done) ? ev.done : [], timings_s: truthy(ev.timings_s) ? ev.timings_s : {},
    error: orNull(ev.error),
  });
  s.run = run;
  finishStages(d, s, status);
  if (status === "succeeded") s.finished = true;
}

function onJobStatus(d: Draft, s: TrackerState, ev: Ev): void {
  const status = orNull(ev.status) as JobStatus | null;
  s.status = status;
  if (ev.exit_code !== undefined && ev.exit_code !== null) s.exit_code = ev.exit_code as number;
  if (ev.error !== undefined && ev.error !== null) s.error = ev.error;
  if (status !== null && FINAL_JOB.has(status)) {
    s.finished = true;
    finishStages(d, s, status);
    const spanStatus = SPAN_STATUS[status] ?? "error";
    if (s.running.length) {
      const spans = d.at<Record<string, SpanRow>>(s, "spans");
      for (const sid of s.running) {
        if (spans[sid] === undefined || spans[sid].status !== "running") continue;
        const sp = d.at<SpanRow>(spans, sid);
        sp.status = spanStatus;
        sp.ended_ts = orNull(ev.ts) as number | null;
        if (sp.started_ts !== null && isNum(ev.ts)) sp.elapsed_s = pyRound(ev.ts - sp.started_ts, 4);
      }
    }
    s.running = [];
    if (Object.keys(s.partial).length) s.partial = {};
    recomputePartial(d, s);
  }
}

function onJobResult(_d: Draft, s: TrackerState, ev: Ev): void {
  if (isDict(ev.result)) s.result = ev.result;
}

function onCancelRequested(_d: Draft, s: TrackerState, ev: Ev): void {
  s.cancel = { by: orNull(ev.by), ts: orNull(ev.ts) as number | null };
}

function finishStages(d: Draft, s: TrackerState, status: string | null): void {
  if (!s.stages || !Object.keys(s.stages).length) return;
  const end: StageUiState = ({ succeeded: "done", failed: "failed", cancelled: "cancelled", interrupted: "failed" } as Record<string, StageUiState>)[status ?? ""] ?? "failed";
  const stages = d.at<Record<string, StageRow>>(s, "stages");
  for (const sid of Object.keys(stages)) {
    const st = stages[sid];
    if (st.state === "planned") d.at<StageRow>(stages, sid).state = "not_reached";
    else if (st.state === "running") d.at<StageRow>(stages, sid).state = end;
  }
}

const HANDLERS: Record<string, Handler> = {
  "run.start": onRunStart,
  "run.dir": onRunDir,
  "run.plan": onRunPlan,
  "stage.start": onStageStart,
  "stage.end": onStageEnd,
  "stage.skip": onStageSkip,
  "task.start": onTaskStart,
  "task.end": onTaskEnd,
  tick: onTick,
  metric: onMetric,
  artifact: onArtifact,
  checkpoint: onCheckpoint,
  warning: (d, s, ev, _p, cursor) => warning(d, s, ev, cursor),
  heartbeat: onHeartbeat,
  "cancel.requested": onCancelRequested,
  "run.end": onRunEnd,
  "job.status": onJobStatus,
  "job.result": onJobResult,
};

function complete(d: Draft, s: TrackerState, stage: string | null, unit: string, seconds: unknown): void {
  const done = d.at<Record<string, number>>(s, "done_units");
  done[unit] = (done[unit] ?? 0) + 1;
  if (stage) {
    const sd = d.at<Record<string, Record<string, number>>>(s, "stage_done");
    if (sd[stage] === undefined) sd[stage] = d.fresh({});
    const row = d.at<Record<string, number>>(sd, stage);
    row[unit] = (row[unit] ?? 0) + 1;
  }
  if (isNum(seconds) && seconds > 0) {
    const obsAll = d.at<Record<string, number[]>>(s, "unit_obs");
    if (obsAll[unit] === undefined) obsAll[unit] = d.fresh([]);
    const obs = d.at<number[]>(obsAll, unit);
    obs.push(pyRound(seconds, 6));
    if (obs.length > OBS_KEEP) obs.splice(0, obs.length - OBS_KEEP);
  }
  recomputePartial(d, s);
}

function recomputePartial(d: Draft, s: TrackerState): void {
  const entries = Object.values(s.partial);
  if (!entries.length && !Object.keys(s.stage_partial).length) return;
  const per: Record<string, Record<string, number>> = {};
  for (const [stage, unit, frac] of entries) {
    const row = (per[stage || ""] ??= {});
    row[unit] = (row[unit] ?? 0) + frac;
  }
  s.stage_partial = d.fresh(per);
}

function metric(d: Draft, s: TrackerState, name: unknown, value: unknown, unit: string | null, tags: Dict, ts: number | null, _cursor: number | null): void {
  if (!truthy(name)) return;
  const nm = String(name);
  const key = metricKey(nm, tags);
  d.at<Record<string, MetricLatest>>(s, "metrics_latest")[key] = { value: value as MetricLatest["value"], unit, tags: { ...tags }, ts };
  if ((SERIES_NAMES.has(nm) || SERIES_PREFIXES.some((p) => nm.startsWith(p))) && typeof value === "number") {
    const all = d.at<Record<string, SeriesPoint[]>>(s, "metric_series");
    if (all[key] === undefined) all[key] = d.fresh([]);
    const series = d.at<SeriesPoint[]>(all, key);
    series.push({ ts, value });
    if (series.length > SERIES_KEEP) series.splice(0, series.length - SERIES_KEEP);
  }
}

function newSpan(ev: Ev, kind: SpanKind, path: string[]): SpanRow {
  const name = kind === "stage" ? ev.stage : ev.name;
  const task = kind === "task";
  return {
    span_id: String(ev.span), parent_id: orNull(ev.parent) as string | null, kind, name: truthy(name) ? pyStr(name) : kind,
    key: task ? (orNull(ev.key) as string | null) : null, k: task ? (orNull(ev.k) as number | null) : null, n: task ? (orNull(ev.n) as number | null) : null,
    unit: task ? (orNull(ev.unit) as string | null) : null, status: "running", started_ts: orNull(ev.ts) as number | null, ended_ts: null, elapsed_s: null,
    ctx: truthy(ev.ctx) ? (ev.ctx as Dict) : {}, metrics: {}, path: [...path], depth: path.length,
  };
}

/** Open or close the span an `*.start` / `*.end` event describes. */
function spanEvent(d: Draft, s: TrackerState, ev: Ev, path: string[]): void {
  const t = typeof ev.type === "string" ? ev.type : "";
  const isStart = t.endsWith(".start");
  if (!isStart && !t.endsWith(".end")) return;
  const sid = ev.span;
  if (!truthy(sid)) return;
  const kind = t.split(".", 1)[0] as SpanKind;
  if (kind !== "run" && kind !== "stage" && kind !== "task") return;
  const id = String(sid);
  const spans = d.at<Record<string, SpanRow>>(s, "spans");
  if (isStart) {
    spans[id] = d.fresh(newSpan(ev, kind, path));
    d.at<string[]>(s, "running").push(id);
    return;
  }
  const sp = spans[id] === undefined ? (spans[id] = d.fresh(newSpan(ev, kind, path))) : d.at<SpanRow>(spans, id);
  const status = ev.status as string;
  sp.status = kind === "run" ? SPAN_STATUS[status] ?? "error" : status === "ok" || status === "error" || status === "cancelled" ? status : "error";
  sp.ended_ts = orNull(ev.ts) as number | null;
  let el = orNull(ev.elapsed_s) as number | null;
  if (el === null && sp.started_ts !== null && isNum(ev.ts)) el = pyRound(ev.ts - sp.started_ts, 4);
  sp.elapsed_s = el;
  if (kind === "task" && isDict(ev.metrics)) sp.metrics = ev.metrics;
  else if (kind === "stage" && isDict(ev.summary)) sp.metrics = ev.summary;
  const i = s.running.indexOf(id);
  if (i >= 0) d.at<string[]>(s, "running").splice(i, 1);
}

function sameList(a: readonly string[] | null, b: readonly string[] | null): boolean {
  if (a === b) return true;
  if (!a || !b || a.length !== b.length) return false;
  return a.every((x, i) => x === b[i]);
}

function updateCurrent(s: TrackerState): void {
  let cur: SpanRow | null = null;
  for (let i = s.running.length - 1; i >= 0; i--) {
    const sp = s.spans[s.running[i]];
    if (sp !== undefined && sp.status === "running") {
      cur = sp;
      break;
    }
  }
  const path = cur ? cur.path.map(String) : null;
  if (!sameList(s.current_path, path)) s.current_path = path;
  let stage: string | null = null;
  for (let i = s.running.length - 1; i >= 0; i--) {
    const sp = s.spans[s.running[i]];
    if (sp !== undefined && sp.kind === "stage" && childIndex(sp.path) === null) {
      stage = sp.name;
      break;
    }
  }
  s.stage = stage;
}

function updateProgress(d: Draft, s: TrackerState): void {
  if (s.status === "succeeded" || s.run_status === "succeeded") s.progress = 1.0;
  else {
    let p = ownProgress(s);
    const kids = Object.values(s.children);
    if (p === null && kids.length) p = kids.reduce((a, c) => a + (c.state.progress || 0), 0) / kids.length;
    if (p === null && s.kn.length) p = knProgress(s.kn);
    if (p === null && s.tick !== null) p = s.tick.frac;
    s.progress = p === null ? null : pyRound(Math.min(1, Math.max(0, p)), 12);
  }
  if (!s.stages || !Object.keys(s.stages).length) return;
  for (const node of s.plan ?? []) {
    const st = s.stages[node.id];
    if (st === undefined) continue;
    let v: number | null;
    if (st.state === "done") v = 1.0;
    else if (st.state === "running" || st.state === "planned") v = nodeProgress(s, node);
    else v = null;
    if (st.progress !== v) d.at<StageRow>(d.at<Record<string, StageRow>>(s, "stages"), node.id).progress = v;
  }
}

function ownProgress(s: TrackerState): number | null {
  const total = weightSum(s.planned_units);
  if (total <= 0) return null;
  const partial: Record<string, number> = {};
  for (const [, unit, frac] of Object.values(s.partial)) partial[unit] = (partial[unit] ?? 0) + frac;
  let got = 0;
  for (const [u, n] of Object.entries(s.planned_units)) {
    if (n) got += unitWeight(u) * Math.min(Number(n), (s.done_units[u] ?? 0) + (partial[u] ?? 0));
  }
  return got / total;
}

function nodeProgress(s: TrackerState, node: PlanNode): number | null {
  const units = node.units ?? {};
  const total = weightSum(units);
  if (total <= 0) return null;
  const done = s.stage_done[node.id] ?? {};
  const part = s.stage_partial[node.id] ?? {};
  let got = 0;
  for (const [u, n] of Object.entries(units)) if (n) got += unitWeight(u) * Math.min(Number(n), (done[u] ?? 0) + (part[u] ?? 0));
  return pyRound(got / total, 12);
}

// ---------------------------------------------------------------- nested runs (study children)

function childEvent(d: Draft, s: TrackerState, ev: Ev, path: string[], idx: number, cursor: number | null): void {
  const prefix = pyJsonList(path.slice(0, idx + 1));
  let key = s.child_of[prefix];
  if (key === undefined) {
    if (ev.type !== "run.start") return;
    key = childKey(s, ev, path[idx]);
    d.at<Record<string, string>>(s, "child_of")[prefix] = key;
    const meta = isDict(ev.run_meta) ? ev.run_meta : {};
    d.at<Record<string, ChildEntry>>(s, "children")[key] = d.fresh({
      key,
      label: (truthy(meta.role) ? String(meta.role) : truthy(ev.name) ? String(ev.name) : key),
      run_id: orNull(meta.studio_run_id) as string | null,
      job_id: null,
      state: d.fresh(newTrackerState()),
    });
  }
  const children = d.at<Record<string, ChildEntry>>(s, "children");
  const child = d.at<ChildEntry>(children, key);
  const cs = d.at<TrackerState>(child, "state");
  const sub: Ev = { ...ev };
  if (sub.type === "artifact") sub.span_path = path.slice(idx);
  else sub.path = path.slice(idx);
  reduce(d, cs, sub, cursor);
}

function childKey(s: TrackerState, ev: Ev, runEl: string): string {
  const ctx = isDict(ev.ctx) ? ev.ctx : {};
  const meta = isDict(ev.run_meta) ? ev.run_meta : {};
  let key: string;
  if (ctx.placebo !== undefined && ctx.placebo !== null) key = `placebo:${pyStr(ctx.placebo)}`;
  else if (ctx.variant !== undefined && ctx.variant !== null) key = `variant:${pyStr(ctx.variant)}`;
  else if (truthy(meta.role)) key = String(meta.role);
  else key = runEl.slice(4) || "child";
  const base = key;
  for (let i = 2; key in s.children; i++) key = `${base}#${i}`;
  return key;
}

function childStatus(cs: TrackerState): string {
  if (cs.run_status === "succeeded" || cs.run_status === "failed" || cs.run_status === "cancelled") return cs.run_status;
  return cs.run_status === "running" ? "running" : "planned";
}

/** Child rows of a study job (api.md `TrackerSnapshot.children`). */
export function childrenRows(s: TrackerState): ChildRow[] {
  return Object.entries(s.children).map(([key, c]) => {
    const metrics: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(c.state.metrics_latest)) {
      const val = v.value;
      if (!k.includes("{") && (typeof val === "number" || typeof val === "string" || typeof val === "boolean")) metrics[k] = val;
    }
    return { job_id: c.job_id, run_id: c.run_id, key, label: c.label, status: childStatus(c.state), progress: c.state.progress, metrics };
  });
}

// ---------------------------------------------------------------- outputs

function spanOut(sp: SpanRow): Span {
  const { path: _p, depth: _d, ...rest } = sp;
  return { ...rest, started_ts: sp.started_ts ?? 0 } as Span;
}

function hasAncestor(spans: Record<string, SpanRow>, sp: SpanRow, anc: string): boolean {
  let cur = sp.parent_id;
  for (let seen = 0; cur && seen < 64; seen++) {
    if (cur === anc) return true;
    cur = spans[cur]?.parent_id ?? null;
  }
  return false;
}

/**
 * Span rows (depth ≤ `maxDepth` below `under`). Beyond `limit` rows, spans deeper than two
 * levels are aggregated per (parent, name) into one row each (`metrics.aggregated` = count).
 */
export function spansList(s: TrackerState, opts: { under?: string | null; maxDepth?: number; limit?: number } = {}): Span[] {
  const maxDepth = opts.maxDepth ?? MAX_DEPTH;
  const limit = opts.limit ?? MAX_SPANS;
  const spans = s.spans;
  const all = Object.values(spans);
  let rows: SpanRow[];
  let base: number;
  if (opts.under) {
    const root = spans[opts.under];
    if (!root) return [];
    base = root.depth;
    rows = all.filter((sp) => sp.span_id === opts.under || (hasAncestor(spans, sp, opts.under!) && sp.depth - base <= maxDepth));
  } else {
    base = (all.length ? Math.min(...all.map((sp) => sp.depth)) : 1) - 1;
    rows = all.filter((sp) => sp.depth - base <= maxDepth);
  }
  if (rows.length <= limit) return rows.map(spanOut);
  const keep = rows.filter((sp) => sp.depth - base <= 2);
  const groups = new Map<string, Span & { metrics: { aggregated: number } }>();
  const rank: Record<string, number> = { running: 3, error: 2, cancelled: 1, ok: 0 };
  for (const sp of rows) {
    if (sp.depth - base <= 2) continue;
    const gk = `${sp.parent_id}\u0000${sp.name}`;
    let g = groups.get(gk);
    if (!g) {
      g = {
        span_id: `${sp.parent_id}/${sp.name}*`, parent_id: sp.parent_id, kind: sp.kind, name: sp.name, key: null, k: null, n: 0, unit: sp.unit,
        status: "ok", started_ts: sp.started_ts ?? 0, ended_ts: sp.ended_ts, elapsed_s: 0, ctx: {}, metrics: { aggregated: 0 },
      };
      groups.set(gk, g);
    }
    g.n = (g.n ?? 0) + 1;
    g.metrics.aggregated += 1;
    g.elapsed_s = pyRound((g.elapsed_s ?? 0) + (sp.elapsed_s ?? 0), 4);
    if (sp.started_ts !== null && sp.started_ts < g.started_ts) g.started_ts = sp.started_ts;
    if (sp.ended_ts === null || g.ended_ts === null) g.ended_ts = sp.status === "running" ? null : g.ended_ts ?? sp.ended_ts;
    else if (sp.ended_ts > g.ended_ts) g.ended_ts = sp.ended_ts;
    if ((rank[sp.status] ?? 0) > (rank[g.status] ?? 0)) g.status = sp.status;
  }
  return [...keep.map(spanOut), ...groups.values()];
}

/** Warning rows without the dedup hash (api.md `WarningRow`). */
export function warningRows(s: TrackerState): WarningRow[] {
  return Object.values(s.warnings).map(({ msg_hash: _h, ...w }) => w);
}

/**
 * Plan nodes in the wire shape of `PlanNode` (as `tracker.py::_plan_out`): a node without a label
 * is labelled with its id, an unknown state counts as skipped, a non-stage id is left out.
 */
export function planOut(plan: PlanNode[] | null): PlanNode[] | null {
  if (plan === null) return null;
  return plan
    .filter((n) => (STAGE_IDS as readonly string[]).includes(n.id))
    .map((n) => ({ ...n, label: String(n.label || n.id), state: n.state === "will_run" || n.state === "skipped" || n.state === "cached" ? n.state : n.state === undefined ? "will_run" : "skipped" }));
}

/**
 * The tracker snapshot wire shape of a state (api.md §3), as `tracker.py::snapshot_parts` builds
 * it: spans to depth 4 (aggregated beyond 2,000 rows), no unit counters, child rows only.
 */
export function toSnapshot(s: TrackerState, job: Job, resources: ResourceRow[] = []): TrackerSnapshot {
  let stages: TrackerSnapshot["stages"] = null;
  if (s.stages !== null) {
    stages = {};
    for (const [sid, st] of Object.entries(s.stages)) stages[sid as StageId] = { ...st };
  }
  return {
    job,
    cursor: s.cursor,
    plan: planOut(s.plan),
    stages,
    spans: spansList(s),
    metrics_latest: s.metrics_latest,
    metric_series: s.metric_series,
    warnings: warningRows(s),
    artifacts: Object.values(s.artifacts),
    checkpoints: s.checkpoints,
    resources,
    heartbeat_gaps: s.heartbeat_gaps,
    children: childrenRows(s),
    log_capped: false,
  };
}

export type Contract = {
  stages: Record<string, { state: string; reason: string | null }>;
  progress: number | null;
  done_units: Record<string, number>;
  warnings: { code: string; count: number }[];
  artifacts: string[];
};

const byCodePoint = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);

/**
 * Canonical projection shared with Python's `to_contract` (SPEC §14.3): stages, progress,
 * done units, warnings summed per code (sorted), artifacts (sorted).
 */
export function toContract(s: TrackerState): Contract {
  const perCode: Record<string, number> = {};
  for (const w of Object.values(s.warnings)) perCode[w.code] = (perCode[w.code] ?? 0) + Number(w.count);
  const stages: Contract["stages"] = {};
  for (const sid of Object.keys(s.stages ?? {}).sort(byCodePoint)) {
    const st = s.stages![sid];
    stages[sid] = { state: st.state, reason: st.reason ?? null };
  }
  const done: Record<string, number> = {};
  for (const u of Object.keys(s.done_units).sort(byCodePoint)) done[u] = s.done_units[u];
  return {
    stages,
    progress: s.progress,
    done_units: done,
    warnings: Object.keys(perCode).sort(byCodePoint).map((code) => ({ code, count: perCode[code] })),
    artifacts: Object.keys(s.artifacts).sort(byCodePoint),
  };
}

// ---------------------------------------------------------------- snapshot → state

/** Rebuild a span's path element (`<kind>:<name>[k/n]` or `[key]`, as core writes it). */
function pathElement(sp: Span): string {
  let el = `${sp.kind}:${sp.name}`;
  if (sp.k !== null && sp.k !== undefined && sp.n !== null && sp.n !== undefined) el += `[${sp.k}/${sp.n}]`;
  else if (sp.key !== null && sp.key !== undefined) el += `[${sp.key}]`;
  else if (sp.k !== null && sp.k !== undefined) el += `[${sp.k}]`;
  return el;
}

/**
 * A reducer state close to the server's at `snap.cursor`, rebuilt from the wire snapshot: what
 * it shows (stages, plan, spans, metrics, warnings, artifacts, checkpoints, gaps, child rows)
 * is carried over, and the counters behind progress (done units per stage, partial ticks) are
 * derived from the spans, stage progress, checkpoints and job fields. The snapshot leaves out
 * spans deeper than four levels and the child sub-projections, so this is the first paint
 * only: `openTracker` replaces it with the exact state of `replayLog` once the log is read.
 */
export function fromSnapshot(snap: TrackerSnapshot): TrackerState {
  const s = newTrackerState();
  const job = snap.job;
  s.cursor = snap.cursor;
  s.plan = snap.plan ? snap.plan.map((n) => ({ ...n })) : null;
  if (snap.stages) {
    s.stages = {};
    for (const [sid, st] of Object.entries(snap.stages)) if (st) s.stages[sid] = { ...st };
  }
  // spans and their paths (parents always precede in depth; walk the chain)
  const byId = new Map(snap.spans.map((sp) => [sp.span_id, sp]));
  const pathCache = new Map<string, string[]>();
  const pathOf = (sp: Span, guard = 0): string[] => {
    const hit = pathCache.get(sp.span_id);
    if (hit) return hit;
    const parent = sp.parent_id ? byId.get(sp.parent_id) : undefined;
    const p = [...(parent && guard < 64 ? pathOf(parent, guard + 1) : []), pathElement(sp)];
    pathCache.set(sp.span_id, p);
    return p;
  };
  for (const sp of snap.spans) {
    if (sp.span_id.endsWith("*")) continue; // aggregated rows of a huge tree
    const path = pathOf(sp);
    s.spans[sp.span_id] = { ...sp, path, depth: path.length };
  }
  s.running = Object.values(s.spans)
    .filter((sp) => sp.status === "running")
    .sort((a, b) => (a.started_ts ?? 0) - (b.started_ts ?? 0) || a.depth - b.depth)
    .map((sp) => sp.span_id);
  s.metrics_latest = { ...snap.metrics_latest };
  for (const [k, v] of Object.entries(snap.metric_series)) s.metric_series[k] = [...v];
  for (const w of snap.warnings) {
    const h = sha1Hex(w.message ?? "").slice(0, 12);
    s.warnings[`${w.code}|${h}`] = { ...w, msg_hash: h };
  }
  for (const a of snap.artifacts) s.artifacts[a.relpath] = { ...a };
  s.checkpoints = snap.checkpoints.map((c) => ({ ...c }));
  s.heartbeat_gaps = snap.heartbeat_gaps.map((g) => ({ ...g }));
  // timeline bounds
  const times = Object.values(s.spans).flatMap((sp) => [sp.started_ts, sp.ended_ts]).filter(isNum);
  s.first_ts = times.length ? Math.min(...times) : job.started_utc ? Date.parse(job.started_utc) / 1000 : null;
  s.last_ts = times.length ? Math.max(...times) : s.first_ts;
  // run and job status
  const root = Object.values(s.spans).find((sp) => sp.kind === "run" && sp.depth === 1);
  if (root) {
    s.run = { name: root.name, started_ts: root.started_ts, ended_ts: root.ended_ts, elapsed_s: root.elapsed_s };
    s.run_status = root.status === "running" ? "running" : root.status === "ok" ? "succeeded" : root.status === "cancelled" ? "cancelled" : "failed";
  }
  s.status = job.status;
  s.exit_code = job.exit_code;
  s.error = job.error;
  s.result = job.result;
  s.finished = isFinalStatus(job.status) || s.run_status === "succeeded";
  for (const [sid, st] of Object.entries(s.stages ?? {})) if (st.elapsed_s !== null && st.elapsed_s !== undefined) s.stage_elapsed[sid] = st.elapsed_s;
  // planned units from the plan's will-run nodes
  for (const node of s.plan ?? []) {
    if ((node.state ?? "will_run") !== "will_run") continue;
    for (const [u, n] of Object.entries(node.units ?? {})) s.planned_units[u] = (s.planned_units[u] ?? 0) + Number(n || 0);
  }
  rebuildCounters(s);
  // open item levels from the running own tasks that count items (the fraction inside an item arrives with its next tick)
  for (const sp of Object.values(s.spans).sort((a, b) => a.depth - b.depth)) {
    if (sp.kind !== "task" || sp.status !== "running" || childIndex(sp.path) !== null) continue;
    if (!isNum(sp.k) || !isNum(sp.n) || sp.n <= 0) continue;
    const label = sp.path[sp.path.length - 1];
    const chain = s.kn.filter((lv) => lv.depth < sp.depth && sp.path[lv.depth - 1] === lv.label);
    s.kn = [...chain, { span: sp.span_id, label, depth: sp.depth, k: sp.k, n: sp.n, frac: 0 }];
  }
  // children: their rows (sub-projections are not part of the wire snapshot)
  for (const row of snap.children) {
    const cs = newTrackerState();
    cs.progress = row.progress;
    cs.run_status = row.status === "planned" ? null : row.status;
    for (const [k, v] of Object.entries(row.metrics)) cs.metrics_latest[k] = { value: v as MetricLatest["value"], unit: null, tags: {}, ts: null };
    s.children[row.key] = { key: row.key, label: row.label, run_id: row.run_id, job_id: row.job_id, state: cs };
  }
  for (const sp of Object.values(s.spans)) {
    if (sp.kind !== "run" || sp.depth < 2) continue;
    const ctx = sp.ctx ?? {};
    const key = ctx.placebo != null ? `placebo:${pyStr(ctx.placebo)}` : ctx.variant != null ? `variant:${pyStr(ctx.variant)}` : sp.name;
    if (key in s.children) s.child_of[pyJsonList(sp.path)] = key;
  }
  updateCurrent(s);
  if (s.current_path === null && job.current_path?.length && !isFinalStatus(job.status)) s.current_path = [...job.current_path];
  const d = new Draft();
  const out = d.own(s);
  updateProgress(d, out);
  return out;
}

/**
 * Done units per stage from what the snapshot shows: unit tasks that ended ok, S0/S1 ends,
 * saved checkpoints; then the rest of each stage's reported progress is attributed to its
 * tick-counted units (engine passes), the fraction to the stage's deepest running span.
 */
function rebuildCounters(s: TrackerState): void {
  const add = (stage: string | null, unit: string, n: number) => {
    if (n <= 0) return;
    s.done_units[unit] = (s.done_units[unit] ?? 0) + n;
    if (stage) {
      const row = (s.stage_done[stage] ??= {});
      row[unit] = (row[unit] ?? 0) + n;
    }
  };
  const stageSpans = Object.values(s.spans).filter((sp) => sp.kind === "stage" && childIndex(sp.path) === null);
  const stageAt = (ts: number | null): string | null => {
    if (ts === null) return null;
    const hit = stageSpans.find((sp) => sp.started_ts !== null && sp.started_ts <= ts && (sp.ended_ts === null || ts <= sp.ended_ts));
    return hit ? hit.name : null;
  };
  for (const sp of Object.values(s.spans)) {
    if (sp.kind !== "task" || !sp.unit || sp.status !== "ok" || childIndex(sp.path) !== null) continue;
    add(stageOf(sp.path), sp.unit, 1);
    if (isNum(sp.elapsed_s) && sp.elapsed_s > 0) (s.unit_obs[sp.unit] ??= []).push(pyRound(sp.elapsed_s, 6));
  }
  for (const sid of ["S0", "S1"]) if (s.stages?.[sid]?.state === "done") add(sid, sid === "S0" ? "s0_load" : "s1_influence", 1);
  for (const c of s.checkpoints) if (c.action === "saved") add(stageAt(c.ts), "checkpoint_save", 1);
  for (const node of s.plan ?? []) {
    if ((node.state ?? "will_run") !== "will_run") continue;
    const st = s.stages?.[node.id];
    if (!st) continue;
    const units = node.units ?? {};
    const W = weightSum(units);
    if (W <= 0) continue;
    const done = (s.stage_done[node.id] ??= {});
    let target: number | null;
    if (st.state === "done") target = 1;
    else target = isNum(st.progress) ? st.progress : null;
    if (target === null) continue;
    let rest = target * W;
    for (const [u, n] of Object.entries(units)) rest -= unitWeight(u) * Math.min(Number(n), done[u] ?? 0);
    if (rest <= 1e-9) continue;
    // tick-counted kinds first (no task span completes them), then any unit with room left
    const order = Object.keys(units).sort((a, b) => Number(b === "engine_pass") - Number(a === "engine_pass"));
    for (const u of order) {
      const room = Number(units[u]) - (done[u] ?? 0);
      if (room <= 0 || rest <= 1e-9) continue;
      const w = unitWeight(u);
      const want = Math.min(room, rest / w);
      const whole = Math.floor(want + 1e-9);
      if (whole > 0) {
        add(node.id, u, whole);
        rest -= whole * w;
      }
      const frac = Math.min(room - whole, rest / w);
      if (frac > 1e-9 && whole < room && st.state !== "done") {
        const holder = Object.values(s.spans)
          .filter((sp) => sp.status === "running" && stageOf(sp.path) === node.id)
          .sort((a, b) => b.depth - a.depth)[0];
        s.partial[holder ? holder.span_id : `snapshot:${node.id}`] = [node.id, u, frac];
        rest -= frac * w;
      }
    }
  }
  const per: Record<string, Record<string, number>> = {};
  for (const [stage, unit, frac] of Object.values(s.partial)) {
    const row = (per[stage || ""] ??= {});
    row[unit] = (row[unit] ?? 0) + frac;
  }
  s.stage_partial = per;
}

// ---------------------------------------------------------------- exact state from the event log

/**
 * Event logs up to this size (the snapshot cursor is a byte offset) are replayed from the first
 * line after the snapshot loads; larger ones (debug-level logs near the 200 MB cap) keep the
 * state rebuilt from the snapshot.
 */
export const EXACT_REPLAY_MAX_BYTES = 48 * 1024 * 1024;
const REPLAY_PAGE = 5000;
/**
 * Events folded at a time while replaying: the fold yields to the browser between slices, so a
 * 20,000-event log never blocks a frame for longer than the 100 ms budget (SPEC §14.5).
 */
const REPLAY_SLICE = 500;

/** A new task, so rendering and input can run between replay slices. */
function nextTask(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * The projection at cursor `upTo`, exactly as the server's: every event of the job's log up
 * to and including that cursor (`GET /api/jobs/{jid}/events`, the same validated lines the
 * server's tailer folds), applied from the first line. Null when the log does not reach
 * `upTo` (no log file, e.g. a pseudo-job synthesised from `run_state.json`).
 */
export async function replayLog(jid: string, upTo: number, fetchEvents: typeof getEvents = getEvents, signal?: AbortSignal): Promise<TrackerState | null> {
  let state = newTrackerState();
  let after: number | null = null;
  for (;;) {
    const page = await fetchEvents(jid, { after, limit: REPLAY_PAGE }, signal);
    const upto = page.events.filter((e) => e.cursor <= upTo);
    for (let i = 0; i < upto.length; i += REPLAY_SLICE) {
      if (i) await nextTask();
      if (signal?.aborted) return null;
      state = applyEvents(state, upto.slice(i, i + REPLAY_SLICE));
    }
    if (state.cursor >= upTo || upto.length < page.events.length) break;
    if (page.eof || !page.events.length || page.next_cursor === after) break;
    after = page.next_cursor;
  }
  return state.cursor === upTo ? state : null;
}

// ---------------------------------------------------------------- ETA (port of eta.projection_eta)

/** Per-unit rate history (median and quartiles, normalised to 54,701 cells and 4 threads). */
export type RateTable = Record<string, Pick<UnitRate, "median_s" | "p25_s" | "p75_s" | "n">>;

function scaleOf(unit: string, nCells: number | null, threads: number | null): number {
  if (UNSCALED_PREFIXES.some((p) => unit.startsWith(p))) return 1;
  const n = nCells || REF_CELLS;
  const t = threads || REF_THREADS;
  return (n / REF_CELLS) ** (ALPHA[unit] ?? 1) * (REF_THREADS / Math.max(t, 1)) ** 0.7;
}

function quantile(xs: number[], q: number): number {
  const v = [...xs].sort((a, b) => a - b);
  if (v.length === 1) return v[0];
  const pos = q * (v.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, v.length - 1);
  return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

function unitSeconds(s: TrackerState, unit: string, rates: RateTable, nCells: number | null, threads: number | null): [number, number, number] {
  const r = rates[unit];
  const rate = r ? r.median_s : seedRate(unit);
  const [rlo, rhi] = r && r.n >= 3 ? [r.p25_s, r.p75_s] : [rate * (1 - RANGE_FRAC), rate * (1 + RANGE_FRAC)];
  const sc = scaleOf(unit, nCells, threads);
  const base = rate * sc;
  const obs = (s.unit_obs[unit] ?? []).filter((x) => x > 0);
  if (obs.length) {
    const mean = obs.reduce((a, b) => a + b, 0) / obs.length;
    if (obs.length >= 3) return [mean, quantile(obs, 0.25), quantile(obs, 0.75)];
    return [mean, mean * (base ? (rlo * sc) / base : 1 - RANGE_FRAC), mean * (base ? (rhi * sc) / base : 1 + RANGE_FRAC)];
  }
  return [base, rlo * sc, rhi * sc];
}

export type EtaEstimate = { eta_s: number | null; eta_lo: number | null; eta_hi: number | null; stages: Record<string, number> };

/**
 * Remaining seconds of a projection (SPEC §5.4): the plan's will-run nodes minus completed and
 * fractional units, with in-job means replacing priors and each cv_curve partition at 0.95 ×
 * the observed S2_S3 time. Without a plan it extrapolates elapsed time from progress.
 */
export function projectionEta(s: TrackerState, opts: { rates?: RateTable; nCells?: number | null; threads?: number | null; now?: number } = {}): EtaEstimate {
  const rates = opts.rates ?? {};
  const nCells = opts.nCells ?? s.n_points;
  const threads = opts.threads ?? null;
  const out: EtaEstimate = { eta_s: null, eta_lo: null, eta_hi: null, stages: {} };
  if (s.finished) return { ...out, eta_s: 0, eta_lo: 0, eta_hi: 0 };
  if (s.plan && s.plan.length) {
    let total = 0, loT = 0, hiT = 0;
    const s23 = s.plan.find((n) => n.id === "S2_S3");
    const s23el = s.stage_elapsed.S2_S3;
    for (const node of s.plan) {
      if ((node.state ?? "will_run") !== "will_run") continue;
      const st = s.stages?.[node.id];
      if (st && ["done", "failed", "cancelled", "skipped", "cached", "disabled", "not_requested"].includes(st.state)) {
        out.stages[node.id] = 0;
        continue;
      }
      const units = node.units ?? {};
      const done = s.stage_done[node.id] ?? {};
      const part = s.stage_partial[node.id] ?? {};
      const rem: Record<string, number> = {};
      for (const [u, n] of Object.entries(units)) rem[u] = Math.max(0, Number(n) - (done[u] ?? 0) - (part[u] ?? 0));
      if (node.id === "cv_curve" && s23 && s23el) {
        const ref = Object.fromEntries(Object.entries(s23.units ?? {}).filter(([u]) => u !== "checkpoint_save"));
        const common = Object.keys(ref).filter((u) => u in units && ref[u]);
        const parts = common.length ? units[common[0]] / ref[common[0]] : 1;
        const wTot = Object.entries(units).reduce((a, [u, n]) => a + unitWeight(u) * n, 0);
        const wRem = Object.entries(rem).reduce((a, [u, n]) => a + unitWeight(u) * n, 0);
        const est = 0.95 * s23el * parts * (wTot ? wRem / wTot : 0);
        total += est;
        loT += est * (1 - RANGE_FRAC);
        hiT += est * (1 + RANGE_FRAC);
        out.stages[node.id] = pyRound(est, 3);
        continue;
      }
      let est = 0, lo = 0, hi = 0;
      for (const [u, n] of Object.entries(rem)) {
        if (n <= 0) continue;
        const [a, b, c] = unitSeconds(s, u, rates, nCells, threads);
        est += n * a;
        lo += n * b;
        hi += n * c;
      }
      total += est;
      loT += lo;
      hiT += hi;
      out.stages[node.id] = pyRound(est, 3);
    }
    return { ...out, eta_s: pyRound(total, 3), eta_lo: pyRound(loT, 3), eta_hi: pyRound(hiT, 3) };
  }
  const p = s.progress;
  if (p !== null && s.first_ts !== null && p >= 0.02) {
    if (p >= 1) return { ...out, eta_s: 0, eta_lo: 0, eta_hi: 0 };
    const elapsed = Math.max(0, (opts.now ?? Date.now() / 1000) - s.first_ts);
    const est = (elapsed * (1 - p)) / p;
    return { ...out, eta_s: pyRound(est, 3), eta_lo: pyRound(est * (1 - RANGE_FRAC), 3), eta_hi: pyRound(est * (1 + RANGE_FRAC), 3) };
  }
  return out;
}

// ---------------------------------------------------------------- live store

export type TrackerPhase = "loading" | "live" | "ended" | "error";

export type TrackerEntry = {
  jid: string;
  phase: TrackerPhase;
  error: Error | null;
  job: Job | null;
  state: TrackerState;
  /** The state is the one rebuilt from the snapshot: the exact replay of the log has not replaced it (yet). */
  reconstructed: boolean;
  /** The event log is being replayed to replace the rebuilt state. */
  replaying: boolean;
  resources: ResourceRow[];
  /** The last DIAG_EVENTS events seen live (Copy diagnostics). */
  recent: JobEvent[];
  ended: JobStatus | null;
  /** Neural-residual validation loss per epoch (debug-level `tick`s with `unit: "epoch"`). */
  epochs: { ts: number; value: number }[];
  /** `log` and `warning` lines streamed after the snapshot (the Logs tab appends them). */
  logs: LogLine[];
  /** The job's event log passed the size cap: debug lines are on disk only. */
  logCapped: boolean;
};

/** A log line from a `log` or `warning` event, mapped like `GET /api/jobs/{jid}/logs`. */
export function logLineFromEvent(raw: JobEvent | Ev): LogLine | null {
  const ev = raw as Ev;
  const path = Array.isArray(ev.path) ? (ev.path as string[]) : [];
  if (ev.type === "log") {
    return { cursor: isNum(ev.cursor) ? ev.cursor : -1, ts: isNum(ev.ts) ? ev.ts : 0, level: String(ev.level ?? ev.lvl ?? "info").toLowerCase(), logger: String(ev.logger ?? ""), msg: String(ev.msg ?? ""), path };
  }
  if (ev.type === "warning") {
    return { cursor: isNum(ev.cursor) ? ev.cursor : -1, ts: isNum(ev.ts) ? ev.ts : 0, level: String(ev.lvl ?? "warning").toLowerCase(), logger: `warning:${String(ev.code ?? "")}`, msg: String(ev.message ?? ""), path };
  }
  return null;
}

export const DIAG_EVENTS = 200;
const EPOCH_KEEP = 2000;
/** Live log lines an entry keeps (the Logs tab re-reads the log when older ones were dropped). */
export const LOG_KEEP = 5000;
/** Resource samples kept client-side (15 min at the 2 s live cadence). */
const RESOURCE_KEEP = 450;

type TrackerStore = {
  entries: Record<string, TrackerEntry>;
  put: (jid: string, patch: Partial<TrackerEntry>) => void;
  drop: (jid: string) => void;
};

export const useTracker = create<TrackerStore>((set, get) => ({
  entries: {},
  put: (jid, patch) => {
    const cur = get().entries[jid];
    const base: TrackerEntry = cur ?? { jid, phase: "loading", error: null, job: null, state: newTrackerState(), reconstructed: false, replaying: false, resources: [], recent: [], ended: null, epochs: [], logs: [], logCapped: false };
    set({ entries: { ...get().entries, [jid]: { ...base, ...patch } } });
  },
  drop: (jid) => {
    const next = { ...get().entries };
    delete next[jid];
    set({ entries: next });
  },
}));

/** Fold one delivered batch into an entry (exported for tests): events, resources, job status, diagnostics ring. */
export function foldBatch(entry: TrackerEntry, batch: readonly JobEvent[]): TrackerEntry {
  const events = batch.filter((e) => e.type !== "resource");
  const res = batch.filter((e): e is Extract<JobEvent, { type: "resource" }> => e.type === "resource");
  let job = entry.job;
  for (const e of events) {
    if (e.type === "job.status" && job) {
      job = { ...job, status: e.status, exit_code: e.exit_code ?? job.exit_code, error: (e.error as Job["error"]) ?? job.error };
    } else if (e.type === "job.result" && job) {
      job = { ...job, result: e.result };
    }
  }
  const resources = res.length
    ? [...entry.resources, ...res.map((r) => ({ ts: r.ts, rss_mb: r.rss_mb, cpu_pct: r.cpu_pct, n_procs: r.n_procs, threads: r.threads }))].slice(-RESOURCE_KEEP)
    : entry.resources;
  const recent = events.length ? [...entry.recent, ...events].slice(-DIAG_EVENTS) : entry.recent;
  const ep = epochPoints(events);
  const epochs = ep.length ? [...entry.epochs, ...ep].slice(-EPOCH_KEEP) : entry.epochs;
  const lines = events.map(logLineFromEvent).filter((l): l is LogLine => l !== null);
  const logs = lines.length ? [...entry.logs, ...lines].slice(-LOG_KEEP) : entry.logs;
  return { ...entry, state: applyEvents(entry.state, events), resources, recent, job, epochs, logs };
}

function epochPoints(events: readonly (JobEvent | Ev)[]): { ts: number; value: number }[] {
  const out: { ts: number; value: number }[] = [];
  for (const raw of events) {
    const e = raw as Ev;
    if (e.type === "tick" && e.unit === "epoch" && isNum(e.val_mse)) out.push({ ts: isNum(e.ts) ? e.ts : 0, value: e.val_mse });
  }
  return out;
}

/**
 * Read the epoch losses already in a job's event log (debug-level ticks are not part of the
 * projection, so a page opened after S2_S3 started fetches them once on request).
 */
export async function loadEpochHistory(jid: string, fetchEvents: typeof getEvents = getEvents): Promise<number> {
  const out: { ts: number; value: number }[] = [];
  let after: number | null = null;
  for (let page = 0; page < 20; page++) {
    const res = await fetchEvents(jid, { after, limit: 5000, types: ["tick"], min_lvl: "debug" });
    out.push(...epochPoints(res.events));
    if (res.eof || !res.events.length || res.next_cursor === after) break;
    after = res.next_cursor;
  }
  if (useTracker.getState().entries[jid]) useTracker.getState().put(jid, { epochs: out.slice(-EPOCH_KEEP) });
  return out.length;
}

type Session = {
  refs: number;
  handle: JobStreamHandle | null;
  abort: AbortController;
  /** Events streamed while the exact state is being replayed from the log (null when not replaying). */
  backfill: JobEvent[] | null;
};
const sessions = new Map<string, Session>();

export type TrackerDeps = { streams?: StreamManager; fetchSnapshot?: typeof getTracker; fetchJob?: typeof getJob; fetchEvents?: typeof getEvents };

/**
 * Start tracking a job: load the snapshot, then (for a job that can still change) stream
 * its events from the snapshot cursor. When the server sends no projection, the event log
 * up to the cursor is replayed in the background and the exact state replaces the one rebuilt
 * from the snapshot (events streamed meanwhile are applied on top). Reference counted;
 * returns the release function.
 */
export function openTracker(jid: string, deps: TrackerDeps = {}): () => void {
  const store = useTracker.getState();
  const existing = sessions.get(jid);
  if (existing) {
    existing.refs += 1;
    return () => releaseTracker(jid);
  }
  const session: Session = { refs: 1, handle: null, abort: new AbortController(), backfill: null };
  sessions.set(jid, session);
  store.put(jid, { phase: "loading", error: null });
  const fetchSnapshot = deps.fetchSnapshot ?? getTracker;
  const fetchJob = deps.fetchJob ?? getJob;
  const fetchEvents = deps.fetchEvents ?? getEvents;
  fetchSnapshot(jid, session.abort.signal).then(
    (snap) => {
      if (sessions.get(jid) !== session) return;
      const state = fromSnapshot(snap);
      const final = isFinalStatus(snap.job.status);
      const exactable = snap.cursor >= 0 && snap.cursor <= EXACT_REPLAY_MAX_BYTES;
      useTracker.getState().put(jid, { phase: final ? "ended" : "live", job: snap.job, state, reconstructed: snap.cursor >= 0, replaying: exactable, resources: snap.resources ?? [], recent: [], ended: final ? snap.job.status : null, error: null, epochs: [], logs: [], logCapped: !!snap.log_capped });
      if (exactable) {
        session.backfill = [];
        replayLog(jid, snap.cursor, fetchEvents, session.abort.signal).then(
          (exact) => {
            const buffered = session.backfill ?? [];
            session.backfill = null;
            if (!useTracker.getState().entries[jid] || sessions.get(jid) !== session) return;
            if (exact) useTracker.getState().put(jid, { state: applyEvents(exact, buffered), reconstructed: false, replaying: false });
            else useTracker.getState().put(jid, { replaying: false });
          },
          () => {
            // keep the state rebuilt from the snapshot
            session.backfill = null;
            if (useTracker.getState().entries[jid] && sessions.get(jid) === session) useTracker.getState().put(jid, { replaying: false });
          },
        );
      }
      if (final) return;
      const streams = deps.streams ?? getStreams();
      session.handle = streams.openJob(jid, {
        after: snap.cursor,
        onEvents: (batch) => {
          const cur = useTracker.getState().entries[jid];
          if (!cur || sessions.get(jid) !== session) return;
          if (session.backfill) for (const e of batch) if (e.type !== "resource") session.backfill.push(e);
          useTracker.setState({ entries: { ...useTracker.getState().entries, [jid]: foldBatch(cur, batch) } });
        },
        onEnd: (status) => {
          if (sessions.get(jid) !== session) return;
          useTracker.getState().put(jid, { phase: "ended", ended: status });
          fetchJob(jid).then(
            (job) => sessions.get(jid) === session && useTracker.getState().put(jid, { job }),
            () => {},
          );
        },
        onClosed: () => {
          session.handle = null;
        },
      });
    },
    (e: unknown) => {
      if (sessions.get(jid) !== session) return;
      if (e instanceof ApiError && e.code === "aborted") return;
      useTracker.getState().put(jid, { phase: "error", error: e instanceof Error ? e : new Error(String(e)) });
    },
  );
  return () => releaseTracker(jid);
}

function releaseTracker(jid: string): void {
  const s = sessions.get(jid);
  if (!s) return;
  s.refs -= 1;
  if (s.refs > 0) return;
  sessions.delete(jid);
  s.abort.abort();
  s.handle?.close();
  useTracker.getState().drop(jid);
}

/** Reload a tracked job from a fresh snapshot (e.g. after a resync gap). */
export function reloadTracker(jid: string, deps: TrackerDeps = {}): void {
  const s = sessions.get(jid);
  if (!s) return;
  const refs = s.refs;
  s.refs = 1;
  releaseTracker(jid);
  for (let i = 0; i < refs; i++) openTracker(jid, deps);
}

/** The tracked state of a job (opens the tracker while mounted). */
export function useJobTracker(jid: string | null, deps?: TrackerDeps): TrackerEntry | undefined {
  // `deps` is read once per job (tests inject fakes); a new object each render must not reopen.
  useEffect(() => (jid ? openTracker(jid, deps) : undefined), [jid]);
  return useTracker((st) => (jid ? st.entries[jid] : undefined));
}
