# SPARC Studio — HTTP & SSE API contract (v1)

This document is the wire contract for `sparc/studio` (server) and `studio-web` (client). Behaviour is specified in [`SPEC.md`](SPEC.md). On wire-format questions this file wins.

Each endpoint is tagged with its **owner** work item:
- `[F]` backend-foundation
- `[P]` backend-projects
- `[R]` backend-runs
- `[E]` backend-engine
- `[S]` backend-studies-exports

The pydantic models that implement these schemas live in `sparc/studio/schemas/common.py` (shared, `[F]`) and in each owner's package. `GET /openapi.json` must describe every endpoint below. CI snapshots it.

---

## 0. Conventions

### 0.1 Base, formats, time

- **Base path:** `/api`. The only exceptions are `GET /auth`, the SPA (`/`, `/assets/*`, and any unknown non-`/api` GET, which returns `index.html`) and `GET /openapi.json`. `/openapi.json` requires auth like `/api/*`. FastAPI's `/docs` and `/redoc` are disabled. `sparc studio --dump-openapi PATH` writes the same document without a server.
- **JSON:** UTF-8 in both directions. Requests with a body send `Content-Type: application/json`, except the raw uploads in §5.1, §6.3, §7.4 and §11.
- **Timestamps:** ISO-8601 UTC strings (`"2026-10-01T14:22:33Z"`) unless named `*_ts`, which are unix seconds as float.
- **Numbers:** finite. NaN and Inf in JSON are serialised as `null`. Binary arrays use NaN.
- **Ids** are opaque strings with a prefix:

  | Prefix | Object |
  |---|---|
  | `p_` | project |
  | `j_` | job |
  | `st_` | study |
  | `sc_` | scenario |
  | `res_` | result |
  | `pl_` | plan |
  | `sw_` | sweep |
  | `cmp_` | comparison |
  | `rg_` | region |
  | `bl_` | blob |
  | `ex_` | export |
  | `fd_` | finding |

  Run ids look like `20261001-142233-full-a1b2`.
- **Pagination:** `?limit=` (default 50, max 500) and `?cursor=` (opaque). List responses are `Page<T> = {items: T[], next_cursor: string | null}` unless stated otherwise.
- **Optimistic concurrency** on config uses `If-Match: <version>`. A stale version returns `409 conflict`.

### 0.2 Auth

- `GET /auth?t=<token>[&next=<path>]` sets the cookie `sparc_studio` (HttpOnly; SameSite=Strict; Path=/; Secure when the request is https) and responds `302 Location: next` (default `/`). `next` must be a same-origin path (leading `/`, not `//`, no scheme); otherwise it is replaced by `/`. A wrong token returns `401` with an HTML message.
- Every `/api/*` route except `GET /api/health` requires either the cookie or `Authorization: Bearer <token>`. Missing auth returns `401 unauthorized`.
- The `Host` header must be in the allowlist; otherwise `400 bad_host`. Unsafe methods (POST, PUT, PATCH, DELETE) must carry an `Origin` (or `Referer`) matching the serving origin; otherwise `403 bad_origin`. Exception: a request authenticated with `Authorization: Bearer` that carries **neither** `Origin` nor `Referer` is accepted (scripts and tests; a browser cannot attach that header cross-site without a CORS preflight, which Studio never grants). A mismatching `Origin`/`Referer` is still `403`.

### 0.3 Error envelope

Every non-2xx JSON response has this shape:

```ts
type ErrorEnvelope = { error: {
  code: ErrorCode;                 // see §16
  message: string;                 // human sentence
  detail?: Record<string, unknown>;
  action?: Action;                 // one-click remedy the UI renders as a button
}};
type Action = { kind: "run_job" | "open" | "resume" | "link_config" | "build_emulator" | "open_engine" | "rerun_exact" | "fetch_input",
                label: string, method?: "POST" | "GET", path?: string, body?: unknown };
```

Validation errors (`422 validation`) put `detail.errors: {path: string, message: string, code: string}[]` in `detail`.

### 0.4 Binary arrays

- **Float32 / Uint8 / Int32 / Int64 arrays** are sent as `Content-Type: application/octet-stream`, little-endian. Response headers:
  - `X-SPARC-Dtype: float32 | uint8 | int32 | int64`
  - `X-SPARC-Length: <n>`
  - `ETag`

  Missing values in float arrays are NaN. Arrays are in **run row order** (the order of `predictions.parquet` / `data.ids`).
- **Packed arrays** (`grid.bin`) concatenate several arrays, each 8-byte aligned. The header `X-SPARC-Offsets` is JSON: `[{"name":"ix","dtype":"int32","offset":0,"length":n}, …]`.
- **`Bitset`** (selection masks) is a JSON string: base64 of a little-endian, LSB-first bit array of length n. Byte `i` holds rows `8i … 8i+7`.
- **`SparseEdit`** (brush) is a JSON object: `{idx: base64(Int32 LE), val: base64(Float32 LE)}`.

### 0.5 Caching

- Immutable resources of finished runs: `ETag` plus `Cache-Control: private, max-age=31536000, immutable`.
- Running runs and live resources: `Cache-Control: no-store`.
- `If-None-Match` → `304`.

### 0.6 Jobs

Every endpoint documented as **→ 202 Job** creates a job and returns:

```ts
type Job = {
  id: string; kind: string; lane: "heavy"|"medium"|"network"|"engine"|"none"; executor: "process"|"engine"|"external";   // "none"/"external": run.external pseudo-jobs
  label: string; status: JobStatus;
  project_id: string|null; run_id: string|null; study_id: string|null; scenario_id: string|null;
  parent_job_id: string|null; after_job_id: string|null; priority: number;
  params: Record<string, unknown>;
  created_utc: string; started_utc: string|null; finished_utc: string|null;
  progress: number|null;            // 0..1, cost-weighted
  eta_s: number|null; eta_lo: number|null; eta_hi: number|null;
  stage: string|null; current_path: string[]|null;
  exit_code: number|null; error: {type: string, message: string, traceback_tail?: string}|null;
  blocked: {reason: string, actions: Action[]}|null;
  result: Record<string, unknown>|null;   // kind-specific (§8)
  peak_rss_mb: number|null; threads: number|null;
};
type JobStatus = "queued"|"blocked"|"starting"|"running"|"cancelling"|"succeeded"|"failed"|"cancelled"|"interrupted";
```

---

## 1. Shared types

```ts
type PlanNode = { id: StageId; label: string; state: "will_run"|"skipped"|"cached";
  reason: string|null;   // not_requested | disabled_by_config:<key> | checkpoint | no_budget | no_treatments | no_responses | requires_S5 | no_partitions | required_by:S6
  units: Record<string, number>; checkpoint_key: string|null;
  est_s: number|null; est_lo: number|null; est_hi: number|null };
type StageId = "S0"|"S1"|"S2_S3"|"baselines"|"cv_curve"|"S4"|"S5"|"climate"|"S6"|"S7"|"finish";

type Issue = { level: "error"|"warn"|"info"; path: string; code: string; message: string;
  fix?: { path: string; value: unknown } };

type SelectionSpec =
  | { kind: "all" }
  | { kind: "zones"; values: (number|string)[] }
  | { kind: "polygon"; crs: "EPSG:4326"|"run_xy_m"; rings: [number, number][][] }
  | { kind: "circle"; crs: "EPSG:4326"|"run_xy_m"; center: [number, number]; radius_m: number }
  | { kind: "rect"; crs: "EPSG:4326"|"run_xy_m"; min: [number, number]; max: [number, number] }
  | { kind: "cells"; ids: (number|string)[] }
  | { kind: "blob"; blob_id: string }
  | { kind: "hex"; size_m: 250|500; keys: number[] }
  | { kind: "filter"; column: string; op: "<"|"<="|">"|">="|"=="|"between"|"in"; value: number|string|[number,number]|(number|string)[] }
  | { kind: "top"; column: string; frac?: number; k?: number; direction: "highest"|"lowest"; within?: SelectionSpec }
  | { kind: "buffer"; of: SelectionSpec; radius_m?: number; lever_range?: string }
  | { kind: "region"; id: string }
  | { op: "and"|"or"|"minus"; args: SelectionSpec[] }
  | { op: "not"; arg: SelectionSpec };
// column namespaces: predictor:<c> | layer:<c> | pred:<target|pred|resid|halfwidth|dist_train_m> | response:<var>:<c>
//                    | planner:<c> | configured:<slug> | result:<res_id>:delta

type LayerMeta = { key: string; group: string; label: string; unit: string; scale: "seq"|"div"|"cat";
  center: number|null; decimals: number; mult: number; zero_blank: boolean; labels: string[]|null;
  desc: string; sign_note: string|null; source: { file: string, column: string|null }|null;
  dtype: "float32"|"uint8";
  stats: { n: number; lo: number|null; hi: number|null; mean: number|null; p1: number|null; p2: number|null;
           p50: number|null; p98: number|null; p99: number|null } };

type GridMeta = { n: number; nx: number; ny: number; dx_m: number; x0_m: number; y0_m: number;
  crs: string|null; coord_scale: number; has_lonlat: boolean;
  bounds_lonlat: [number, number, number, number]|null;     // [west, south, east, north]
  corners: { sw: [number,number], se: [number,number], nw: [number,number], ne: [number,number] }|null; // [lat, lon]
  ids_kind: "int"|"str"; zones: (number|string)[]; n_folds: number|null; units: { target: string };
  // x0_m/y0_m: run-frame metres of the centre of cell (ix=0, iy=0) (sparc.core.grid.Grid; iy grows north).
  // corners: [lat, lon] of the corner cell centres: sw = (0, 0), se = (nx−1, 0), nw = (0, ny−1), ne = (nx−1, ny−1);
  //   clients interpolate lon/lat bilinearly between them for EPSG:4326 selections.
  // zones: the distinct zone codes; grid.bin's `zone` indexes this list.
  background: number|null; etag: string };

type Availability = "ready"|"partial"|"running"|"missing"|"stale";
type OutputEntry = { id: string; label: string; group: string; state: "present"|"stale"|"missing"|"writing"|"partial";
  produced_by: string; view: string; formats: string[];
  files: { relpath: string; bytes: number; mtime: string }[]; action: Action|null };

type Likely = { estimate: number; se: number|null; lo: number|null; hi: number|null;   // 95%: est ± 1.96·se
  confidence: "confident_cools"|"confident_warms"|"could_be_zero"|"unknown"; phrase: string };
```

---

## 2. Auth and system `[F]`

### `GET /auth`
Query: `t` (string, required), `next` (path, optional).
→ `302`, sets the cookie. Errors: `401` (HTML).

### `GET /api/health`
No auth. → `200 {ok: true, version: string, workspace: string, pid: number, started_utc: string, active_jobs: number, engine: {state: EngineHostState}}`.
`EngineHostState = "absent"|"starting"|"ready"|"busy"|"recycling"|"error"`.

### `GET /api/meta`
→ `200`:

```ts
{ version: string; schema_version: 1; event_schema_version: 1;
  stages: { id: StageId; label: string; desc: string; checkpoint_key: string|null; manifest_timing_key: string|null }[];
  job_kinds: { kind: string; label: string; lane: string; executor: string; needs_run: boolean; needs_checkpoint: boolean;
               locks_run: boolean; network_hosts: string[]; long: boolean; params_schema: object }[];
  output_catalog: { id: string; label: string; group: string; files: string[]; produced_by: string; view: string;
                    formats: string[]; manifest_key: string|null }[];
  palettes: { seqLight: string[]; seqDark: string[]; divLight: string[]; divDark: string[]; cat: string[] };
  unit_costs: Record<string, number>;       // seed rates of SPEC §5.4: progress weights and ETA priors
  modes: { id: "fast"|"coarse"|"full"; label: string; desc: string; default_coarse_m?: number }[];
  run_tab_outputs: Record<RunTabId, string[]>;   // derived: catalog output ids grouped by OutputSpec.view (informational)
  edit_modes: string[]; selection_kinds: string[]; warning_codes: { code: string; label: string; view: string|null }[] }
type RunTabId = "overview"|"data"|"accuracy"|"distance"|"influence"|"response"|"causal"|"scenarios"|"climate"|"budget"
  |"planner"|"lab"|"validation"|"uncertainty"|"provenance"|"track"|"map"|"docs"|"files";
```

Run-tab labels, groups and order belong to the frontend route modules (SPEC §12.3). The server knows only the id vocabulary.

### `GET /api/meta/event-schema`
→ `200` JSON Schema of the per-job event union (§17). Used for codegen.

### `GET /api/settings` · `PUT /api/settings`
`Settings`:

```ts
{ thread_budget: number; threads_heavy: number; engine_threads: number; heavy_slots: number; medium_slots: number;
  network_slots: number; engine_max_runs: number; engine_mem_budget_gb: number; engine_idle_min: number;
  auto_uncertainty: boolean; watch_roots: string[]; upload_max_gb: number; keep_job_logs_days: number|null;
  offline: boolean; notifications: boolean; basemap_url: string|null }
```

`PUT` takes a partial `Settings` and returns the full `Settings`.
Errors: `422 validation`, e.g. `threads_heavy + engine_threads > thread_budget + 1`, or a watch root not a directory.

### `GET /api/system`
→ `200 {cpu_count, cpu_model, mem_total_gb, mem_available_gb, disk_free_gb, workspace, workspace_bytes, host_id, versions: {python, sparc, numpy, pandas, torch: string|null, fastapi}, web_build: {src_sha256, vite, react}|null}`. `torch` is read from package metadata (`importlib.metadata`), never by importing torch.

### `POST /api/system/netcheck`
Body: `{hosts?: string[]}`. Default: every host from job kinds and settings.
→ `200 {results: {host: string, ok: boolean, ms: number|null, error: string|null}[]}`. Cached for 10 min per host.

### `GET /api/storage`
→ `200 {workspace_bytes, free_bytes, cache: {name, bytes, mtime}[], runs: {run_id, label, project_id, outputs_bytes, checkpoint_bytes}[], studies: {study_id, bytes}[], jobs_bytes}`.

### `DELETE /api/storage/cache/{name}`
→ `200 {freed_bytes}`. Errors: `404`.

### `POST /api/shutdown`
Body: `{stop_jobs?: boolean = false}` → `202 {ok: true}`. Broadcasts `server_shutdown`.

---

## 3. Jobs and tracking `[F]`

### `GET /api/jobs`
Query:
- `status`: comma list, or the alias `active` = queued, blocked, starting, running, cancelling
- `kind` (comma list)
- `project`, `run`, `study`, `parent`
- `limit`, `cursor`

→ `200 Page<Job>`, ordered by `created_utc` descending.

### `POST /api/jobs`
Body: `{kind: string, params: object, project_id?: string, run_id?: string, study_id?: string, priority?: number, after_job_id?: string}` → `202 Job`.

This is the generic creator. Params are validated against the kind's schema.

Errors:
- `404 unknown_kind`
- `422 validation`
- `409 precondition` (e.g. `needs_checkpoint`)
- `409 locked` is **not** returned: the job is queued as `blocked` instead.

### `GET /api/jobs/{jid}`
→ `200 Job`. Errors: `404`.

### `GET /api/jobs/{jid}/tracker`
→ `200 TrackerSnapshot`:

```ts
{ job: Job; cursor: number;
  plan: PlanNode[]|null;
  stages: Record<StageId, { state: "planned"|"running"|"done"|"failed"|"cancelled"|"not_reached"|"skipped"|"cached"|"disabled"|"not_requested";
                            reason: string|null; started_ts: number|null; ended_ts: number|null; elapsed_s: number|null;
                            est_s: number|null; progress: number|null }>|null;
  spans: Span[];                         // depth ≤ 4, collapsed beyond 2,000 nodes (aggregated rows)
  metrics_latest: Record<MetricKey, {value: number|string|null, unit: string|null, tags: object, ts: number}>;
  metric_series: Record<MetricKey, {ts: number, value: number}[]>;   // only: candidate_rmse, heldout_rmse, mean_benefit, scenario.*, cv_row.*, influence.*
  warnings: {code: string, lvl: string, message: string, count: number, stage: string|null, first_cursor: number, data: object}[];
  artifacts: {relpath: string, role: string, bytes: number, stage: string|null, ts: number}[];
  checkpoints: {action: string, done: string[], bytes: number|null, ts: number}[];
  resources: {ts: number, rss_mb: number, cpu_pct: number, n_procs: number}[];   // last 15 min, 10 s resolution
  heartbeat_gaps: {from_ts: number, to_ts: number}[];
  children: {job_id: string|null, run_id: string|null, key: string, label: string, status: string, progress: number|null, metrics: object}[];
  log_capped: boolean }                  // events.jsonl passed 200 MB: debug lines are kept on disk only (SPEC §5.6)
type Span = { span_id: string; parent_id: string|null; kind: "run"|"stage"|"task"; name: string; key: string|null;
  k: number|null; n: number|null; unit: string|null; status: "running"|"ok"|"error"|"cancelled";
  started_ts: number; ended_ts: number|null; elapsed_s: number|null; ctx: object; metrics: object };
type MetricKey = string;   // `name` when the metric has no tags, else `name{k1=v1,k2=v2}` with tags sorted by key,
                           // e.g. "influence.range_m{predictor=Pct_Canopy}", "candidate_rmse{candidate=0.1}"
```

Progress in the snapshot (`job.progress`, `stages[].progress`) is cost-weighted with the seed-rate weights (`/api/meta.unit_costs`) and SPEC §5.4 unit accounting. ETA fields use the calibrated per-host model.

### `GET /api/jobs/{jid}/spans`
Query: `under?` (span id), `max_depth?` (default 4) → `200 Span[]`.

### `GET /api/jobs/{jid}/events`
Query: `after?` (cursor; omitted = from the first line), `limit?` (default 1000, max 5000), `types?` (comma list), `min_lvl?`.
→ `200 {events: Event[], next_cursor: number, eof: boolean}`. Each event carries `cursor` (byte offset) in addition to the envelope.
`after=X` returns events whose cursor is greater than X. Cursor 0 is the first line's byte offset, so `after=0` skips that line; omit `after` to read from the start.

### `GET /api/jobs/{jid}/logs`
Query: `after?` (as for `/events`: cursor > X; omitted = from the first line), `level?` (min level), `logger?`, `stage?`, `q?` (substring), `limit?` (default 500).
→ `200 {lines: {cursor: number, ts: number, level: string, logger: string, msg: string, path: string[]}[], next_cursor: number}`.

### `GET /api/jobs/{jid}/logs/raw`
Query: `stream=events|stdout|stderr|text` (`text` is a rendered `log.txt`).
→ file download (`text/plain` or `application/x-ndjson`).

### `GET /api/jobs/{jid}/metrics`
Query: `names` (comma list) → `200 Record<string, {cursor: number, ts: number, value: number|null, value_text: string|null, tags: object}[]>`.

### `GET /api/jobs/{jid}/warnings`
→ `200` the `warnings` array of the tracker snapshot.

### `GET /api/jobs/{jid}/resources`
Query: `since_ts?` → `200 {ts, rss_mb, cpu_pct, n_procs, threads}[]`.

### `POST /api/jobs/{jid}/cancel`
→ `202 Job` (status `cancelling`, or `cancelled` if it was still queued).
Errors: `409 not_cancellable` (already finished, or an external run).

### `POST /api/jobs/{jid}/kill`
Force stop. → `202 Job`. Errors: `409` when the job is not `cancelling`/`running`, or when the grace period has not elapsed and `force_now` is absent. Body: `{force_now?: boolean}`.

### `POST /api/jobs/{jid}/retry`
→ `202 Job` (new job with the same kind and params; `parent_job_id` = old id). For `run.core` jobs this is a resume when a checkpoint exists. Errors: `409` while the job is still active.

### `PATCH /api/jobs/{jid}`
Body: `{priority: number}` (queued/blocked only) → `200 Job`.

### `DELETE /api/jobs/{jid}`
Allowed only for finished jobs. Query `files=true` also removes the job directory. → `200 {ok: true}`. Errors: `409 active`.

### `GET /api/queue` · `POST /api/queue/pause` · `POST /api/queue/resume`
→ `200 {paused: boolean, lanes: {lane: string, slots: number, running: string[], queued: string[]}[]}`.

### `GET /api/timings`
Query: `unit?`, `stage?`, `project?`, `limit?`.
→ `200 {unit_rates: {unit: string, median_s: number, p25_s: number, p75_s: number, n: number}[], stage_history: {run_id: string, label: string, git_commit: string|null, stage: string, seconds: number, n_points: number, mode: string, threads: number|null}[]}`.

---

## 4. Streams (SSE) `[F]`

Both endpoints:
- respond `200 text/event-stream` with `Cache-Control: no-store` and `X-Accel-Buffering: no`;
- send `: ping` every 15 s;
- frame events as `id: …\nevent: <type>\ndata: <json>\n\n`.

### `GET /api/stream`
Query: `topics=jobs,runs,engine,studies,storage` (default all). The resume point is the `Last-Event-ID` header or `?after=g:<gseq>`.
- Ids are `g:<gseq>`.
- If the requested id is older than the 10,000-event ring buffer, the first event is `resync {reason: "expired"}`.
- Payloads are in §17.1.

### `GET /api/jobs/{jid}/stream`
Query: `after=<cursor>`, or the `Last-Event-ID` header (cursor).
- Replays from the file, then tails it.
- `id` = byte cursor, `event` = event `type`, `data` = the full event JSON (envelope + fields).
- `tick` events are coalesced to 4 Hz per span, except ticks with `k == 1` or `k == n`, which are always sent.
- Transient `resource` events carry no `id`.
- For a `run.external` pseudo-job the stream tails `run_state.events_path` instead of a job dir.
- The stream closes with `event: end` (`data: {status}`) after the job reaches a final status and the file is drained.
- On subscriber overflow it sends `resync {after: <cursor>}` and closes; the client resumes from `after`.

Errors: `404`.

---

## 5. Projects `[P]`

```ts
type Project = { id: string; slug: string; name: string; dir: string; config_path: string; template: string|null;
  demo: boolean; active_run_id: string|null; archived: boolean; created_utc: string; updated_utc: string;
  report: { title: string|null; place: string|null; area: string|null };
  headline_scenario: string|null;   // configured slug or sc_ id
  cost_model: Record<string, {per_unit: number}>;
  n_runs: number; last_run: {id: string, status: string, created_utc: string, r2: number|null, has_checkpoint: boolean}|null;
  active_jobs: number; readiness_score: {done: number, total: number} };
type ReadinessRow = { key: "data"|"columns"|"levers"|"roles"|"forcing"|"climate_table"|"people_layers"|"config_valid"
  |"runs"|"emulator"|"studies"; label: string; state: "ok"|"warn"|"missing"|"n/a"; detail: string; action: Action|null };
```

### `GET /api/projects`
Query: `archived?=false` → `200 Project[]`.

### `POST /api/projects`
Body:

```ts
{ name: string;
  template: "blank"|"synthetic_demo"|"providence_example";
  options?: { seed?: number; n?: number; import_existing_runs?: boolean } }
```

→ `201 {project: Project, imported_runs: string[], warnings: string[]}`.

Errors:
- `409 conflict` (slug exists)
- `404 example_unavailable` (no repo checkout and no packaged example)
- `422 validation`

### `POST /api/projects/import`
Body: `{name?: string, config_path: string, copy_data?: boolean = false, run_dirs?: string[], study_dirs?: string[], trust_pickles?: boolean = false}`
→ `201 {project: Project, runs: RunSummary[], studies: Study[], warnings: string[]}`.

Errors: `404` (path), `422` (config invalid; `detail.errors`), `422 needs_config` (a run lacks provenance and no config fits it).

### `GET /api/projects/{pid}`
→ `200 {project: Project, readiness: ReadinessRow[], runs: RunSummary[], active_jobs: Job[], config_version: number}`.

### `PATCH /api/projects/{pid}`
Body: `{name?, active_run_id?, headline_scenario?, cost_model?, report?, archived?}` → `200 Project`.

### `DELETE /api/projects/{pid}`
Query: `files=false`. Refused while jobs are active (`409 active`). → `200 {ok: true}`.

### 5.1 Project files

**`PUT /api/projects/{pid}/files/{kind}/{filename}`** (raw upload)
- `kind ∈ data|join|layers|features|forcing|climate|other`.
- The body is the raw file bytes, streamed. `Content-Type` is any value. Header `X-Overwrite: 1` replaces an existing file.
- → `201 {path: string, bytes: number, kind: string, inspect: FileInspect|null}`.
- Errors: `409 exists`, `413 too_large`, `415 bad_suffix`.

**`GET /api/projects/{pid}/files`** → `200 {path, kind, bytes, mtime, used_by: string[]}[]`. `used_by` lists dotted config keys.

**`GET /api/projects/{pid}/files/inspect`**
Query: `path`, `rows?=50000` → `200 FileInspect`:

```ts
{ n_rows: number; n_rows_exact: boolean;
  columns: { name: string; dtype: string; n_null: number; min: number|null; max: number|null; n_unique: number|null; sample: unknown[] }[];
  preview: unknown[][] }
```

**`DELETE /api/projects/{pid}/files`**
Query: `path` → `200 {ok: true}`. Errors: `409 in_use` (`detail.used_by`).

**`POST /api/projects/{pid}/columns/suggest`**
Body: `{path: string}` → `200 {target, id, x, y, zone, coord_unit, crs_guess: string|null, predictors: string[], roles: Record<string, string>, confidence: Record<string, number>}`.

### 5.2 Data check (S0 inline) and preview

**`POST /api/projects/{pid}/data/check`**
Body: `{config_patch?: object}` (a deep-merged override, so unsaved form values can be checked). It is applied to the saved config as a JSON Merge Patch (RFC 7386): mappings merge, `null` deletes the key (so a draft that removed a lever, a `qa.clip` column, a role or a data key is checked as drafted), and any other value, lists included, replaces.
→ `200`:

```ts
{ n_points: number; n_input: number; n_dropped: number; clipped: Record<string, number>;
  grid: { nx: number; ny: number; cell_m: number; fill_fraction: number; collisions: number };
  background: { value: number; source: string }; noise_floor: number|null;
  flags: { code: string; severity: "warn"|"info"; message: string }[];
  dose_scale: Record<string, { sd: number; doses: number[]; doses_in_sd: number[]; percentile_reached: number[] }>;
  coarse: object|null; extent_m: [number, number]; columns_missing: string[]; preview_token: string;
  preview_columns: string[]; elapsed_s: number }   // the columns `{column}.bin` serves for this token
```

Errors: `422 validation` (missing target/x/y, unreadable file), `404` (data file). Runs inline, typically under 2 s; files over 2 M rows return `413 too_large_inline`.

**`GET /api/projects/{pid}/data/preview/{preview_token}/grid.bin`**
Packed `ix:int32, iy:int32, lon:float32, lat:float32` (§0.4) plus a `GridMeta` JSON in the `X-SPARC-Grid` header.

**`GET /api/projects/{pid}/data/preview/{preview_token}/{column}.bin`**
Float32 column in row order. Tokens expire after 30 min. The columns are the table's numeric columns plus, when `planner.layers` names a readable table, its numeric columns (`people`, `people_60_plus`, `people_under_5`, `lc_*`) aligned as the planner reads them: joined by id, or summed (people) and averaged (fractions) per coarse cell. A data column of the same name wins.

### 5.3 Config

**`GET /api/projects/{pid}/config`**
→ `200 {version: number, yaml: string, raw: object, effective: object, changed_from_defaults: {path: string, value: unknown, default: unknown}[], comments_preserved: false}`.

**`PUT /api/projects/{pid}/config`**
Header `If-Match: <version>`. Body: `{yaml: string, note?: string}` or `{raw: object, note?: string}`.
→ `200 {version: number, issues: Issue[], diff: string}`.

Errors:
- `409 conflict` (stale version; `detail.current_version`)
- `422 yaml_error` (`detail: {line, column}`)

Saving never fails on validation issues: they are returned as `issues`.

**`PATCH /api/projects/{pid}/config/sections/{section}`**
Header `If-Match`. Body: `{value: unknown, note?: string}` → same response as `PUT`.

`section ∈ name|data|predictors|encodings|qa|actionable|coupling|mediators|physics|influence|cv|models|stacker|response|scenarios|joint_scenarios|causal|climate|optimize|planner|report|output`.

**`POST /api/projects/{pid}/config/validate`**
Body: `{yaml?: string, raw?: object}` (omit to validate the saved config)
→ `200 {ok: boolean, issues: Issue[], fast_overrides: object, coarse_preview: object|null}`.

**`POST /api/projects/{pid}/config/impact`**
Body: `{yaml?: string, raw?: object}` → `200`:

```ts
{ changed_sections: string[];
  runs: { run_id: string; label: string; checkpoint_done: string[]; changed_sections: string[];
          refit_from: StageId|null; phrase: string }[] }
```

**`GET /api/projects/{pid}/config/history`** → `200 {version, saved_utc, note}[]`.
**`GET /api/projects/{pid}/config/history/{version}`** → `200 {version, yaml}`.

**`GET /api/config/schema`** → `200` JSON Schema of `CoreConfigModel`, with `x-ui` hints per property: `{group, advanced, unit, help, enum_labels}`.

**`POST /api/projects/{pid}/link`**
Body: `{kind: "forcing"|"climate"|"layers"|"features_join"|"features_new_project", path: string, apply?: boolean = false}`. `features_join` writes the join and `open_*` predictors (with roles) into this project, and `features_new_project` creates `<name>_open`.
→ `200 {yaml_diff: string, applied: boolean, version?: number, new_project_id?: string}`.

### 5.4 Inputs (network jobs) `[P]`

Each of these → **`202 Job`**:

**`POST /api/projects/{pid}/inputs/forcing`**
Body: `{date: "YYYY-MM-DD", hours: [number, number], tz: string, lat?: number, lon?: number, station?: string, wind_source: "auto"|"station"|"era5", link?: boolean = true}`.

**`POST /api/projects/{pid}/inputs/layers`**
Body: `{link?: boolean = true}`. Errors: `422 needs_crs`.

**`POST /api/projects/{pid}/inputs/features`**
Body: `{months?: string[], max_cloud?: number = 20, s2_tiles?: string[], target: "new_project"|"this_project" = "new_project"}`.
- `new_project` creates `<name>_open`, the old `write_open_project: true`.
- `this_project` writes the join and `open_*` predictors into the same project. It is the default UI choice in bootstrap mode, i.e. a project with no predictors (SPEC §9.3).

Errors: `422 needs_crs`.

**`POST /api/projects/{pid}/inputs/cmip6`**
Body: `{lat?: number, lon?: number, experiments?: string[], periods?: Record<string, [number, number]>, baseline?: [number, number] = [1995, 2014], months?: number[], variable?: "tasmax"|"tas", models?: string[], workers?: number = 4, link?: boolean = true}`.

**`POST /api/projects/{pid}/inputs/ghcn`**
Body: `{station?: string}`. Caches the series under `<ws>/cache`.

**`POST /api/projects/{pid}/inputs/stations`**
Body: `{}`. Downloads and caches the NOAA ISD station index (`<ws>/cache/isd-history.csv`, ≈3 MB).

Other input endpoints:

**`GET /api/projects/{pid}/inputs`** → `200`:

```ts
{ forcing: {path: string, date: string, physics: object, checks: string[], linked: boolean}|null;
  climate: {path: string, n_models: number, experiments: string[], periods: string[], linked: boolean}|null;
  layers: {path: string, n: number, people_total: number, linked: boolean}|null;
  features: {path: string, agreement: object[], linked: boolean}|null;
  ghcn: {station: string, years: [number, number]}|null }
```

**`GET /api/projects/{pid}/inputs/{kind}/view`**
`kind ∈ forcing|climate|layers|features` → `200` chart-ready model:
- forcing: `{era5: object, station: object|null, compare: {name, era5, station}[], checks}`
- climate: `{rows: {model, experiment, period, delta_K}[], summary: {experiment, period, median, p10, p90}[]}`
- layers: `{totals, columns}`
- features: `{agreement, scatter_bins}`

**`GET /api/projects/{pid}/forcing/stations`**
Query: `lat`, `lon`, `limit=10` → `200 {usaf_wban, name, lat, lon, dist_km, begin, end}[]`. It reads the cached index only. When `isd-history.csv` is not cached it returns `404 not_found` (`detail.missing: "isd-history.csv"`) with `action: {kind: "fetch_input", method: "POST", path: "/api/projects/{pid}/inputs/stations"}`. It never downloads inside the request.

---

## 6. Runs `[R]`

```ts
type RunSummary = { id: string; project_id: string|null; label: string|null; origin: "studio"|"imported"|"study_child"|"reproduction"|"external_live";
  status: "queued"|"running"|"complete"|"partial"|"failed"|"cancelled"|"interrupted"|"external_live"|"imported";
  mode: "fast"|"coarse"|"full"|"custom"; coarse_m: number|null; created_utc: string|null; finished_utc: string|null;
  duration_s: number|null; n_points: number|null; r2: number|null; rmse: number|null; coverage: number|null;
  n_scenarios: number|null; checkpoint_bytes: number|null; has_emulator: boolean; studies: string[];
  git_commit: string|null; git_dirty: boolean|null; demo: boolean; pinned: boolean; parent_run_id: string|null;
  study_id: string|null; last_job_id: string|null };
type CheckpointInfo = { present: boolean; bytes: number|null; done: string[]; saved_utc: string|null; fingerprint: string|null;
  matches_snapshot: { data: boolean, code: boolean, config: boolean }|null; changed_sections: string[];
  resumable: boolean; reuses: StageId[]; saves_s: number|null; reason: string|null };
```

### `GET /api/runs`
Query: `project?`, `status?`, `mode?`, `origin?`, `q?`, `sort?=created_desc|duration|r2`, `limit`, `cursor` → `200 Page<RunSummary>`.

### `GET /api/projects/{pid}/runs`
Same query → `200 Page<RunSummary>`.

### `POST /api/projects/{pid}/runs/plan`
Body:

```ts
{ mode: "fast"|"coarse"|"full"; coarse_m?: number; stages?: ("S0"|"S1"|"S2"|"S3"|"S4"|"S5"|"S6"|"S7")[];
  cv_curve?: boolean|null; threads?: number; resume_run_id?: string }
```

→ `200 RunPlan`:

```ts
{ nodes: PlanNode[]; total_est_s: number; est_lo: number; est_hi: number; est_peak_rss_gb: number; est_disk_gb: number;
  network_hosts: string[]; preflight: { check: string; ok: boolean; severity: "error"|"warn"|"info"; message: string; action: Action|null }[];
  issues: Issue[]; resumable: CheckpointInfo|null; threads: number }
```

Errors: `422` (config has errors; `detail.issues`).

### `POST /api/projects/{pid}/runs`
Launch. Body: plan body + `{label?: string, notes?: string, then?: ("post.planner"|"post.emulator"|"post.uncertainty"|"post.writeup"|"post.baselines")[]}`
→ `202 {run: RunSummary, job: Job, chain: Job[]}`.

Errors: `422` (config errors), `409 preflight_failed` (`detail.preflight`, `action`).

### `POST /api/runs/import`
Body: `{dir: string, project_id?: string, config_path?: string, trust_pickles?: boolean = false}` → `201 RunSummary`.

Errors: `404`, `422 needs_config`, `422 mismatch` (ids or folds differ from the config).

### `GET /api/runs/{rid}`
→ `200 RunDetail`:

```ts
{ run: RunSummary; header: { name: string; created_utc: string|null; git_commit: string|null; git_dirty: boolean|null;
    versions: object|null; n_points: number|null; grid_shape: [number,number]|null; cell_m: number|null; fast: boolean; coarse_m: number|null;
    run_dir: string; demo: boolean };
  state: object|null;                         // run_state.json
  launch: object|null;                        // launch.json (config_raw elided; GET /config for it)
  stages: { id: StageId; state: string; seconds: number|null; source: "events"|"manifest"|"run_state"; reason: string|null }[];
  checkpoint: CheckpointInfo;
  outputs_summary: { present: number; missing: number; stale: number; writing: number };
  sections: Record<string, { present: boolean; source: "manifest"|"file"|null; stale: boolean; older_code: boolean }>;
  flags: { code: string; severity: string; message: string }[];
  children: RunSummary[]; jobs: Job[]; warnings_count: number }
```

Study status rows come from `GET /api/runs/{rid}/studies` `[S]`, and engine state from `GET /api/runs/{rid}/engine` `[E]`. `RunDetail` does not embed them, because the runs item cannot import items that depend on it.

### `PATCH /api/runs/{rid}`
Body: `{label?, notes?, pinned?}` → `200 RunSummary`.

### `DELETE /api/runs/{rid}`
Query: `what=checkpoint|outputs|all` (default `all`; `outputs` keeps `studio/`).
→ `200 {freed_bytes: number}`.

Errors:
- `409 active` (a job is running on the run)
- `409 imported_in_place` (`all` on an imported run deletes the index row only, unless `force_files=true`)

### `DELETE /api/runs/{rid}/checkpoint`
→ `200 {freed_bytes}`. This also evicts the run from the engine host.

### `POST /api/runs/{rid}/resume`
Body: `{threads?: number, use_current_config?: boolean = false}` → `202 Job`.

Errors:
- `409 not_resumable` (`detail.checkpoint`; with `use_current_config`, `detail.impact`)
- `409 active`

### `POST /api/runs/{rid}/rerun`
Body: `{use_current_config?: boolean = true, label?: string}` → `202 {run: RunSummary, job: Job}`. Creates a new run with the same mode and args.

### `GET /api/projects/{pid}/status-board`
→ `200`:

```ts
{ columns: { id: string; label: string; group: "stage"|"post"|"study" }[];
  rows: { run: RunSummary; cells: Record<string, { state: "done"|"cached"|"running"|"failed"|"skipped"|"stale"|"not_run"|"disabled";
          seconds: number|null; progress: number|null; reason: string|null; job_id: string|null; study_id: string|null;
          action: Action|null }> }[] }
```

### `GET /api/runs/{rid}/timeline`
→ `200 {source: "events"|"manifest", stages: {id, start_ts: number|null, end_ts: number|null, seconds: number|null, state}[], jobs: {job_id, kind, start_ts, end_ts, status}[]}`.

### `GET /api/runs/{rid}/manifest`
→ `200` merged manifest, plus a `_sections` provenance map (same as `RunDetail.sections`).

### `GET /api/runs/{rid}/config`
→ `200 {effective: object, raw: object, yaml: string, source: "launch"|"manifest"|"import", config_dir: string, vs_project_diff: {path, run, project}[], vs_defaults: {path, value, default}[]}`.

### `GET /api/runs/{rid}/provenance`
→ `200 {provenance: object|null, git: object|null, platform: object|null, hashes: Record<string, string>, environment: string[], launch: object|null}`.

### `GET /api/runs/{rid}/environment`
Query: `diff_with?=<rid>` → `200 {packages: string[], diff?: {added: string[], removed: string[], changed: {name, a, b}[]}}`.

### 6.1 Outputs, views, docs, files `[R]`

**`GET /api/runs/{rid}/outputs`**
→ `200 {outputs: OutputEntry[], tabs: {id: RunTabId, availability: Availability, missing: {output: string, produced_by: string, action: Action|null}[]}[]}`. There is one `tabs` entry per `RunTabId`, computed by grouping catalog outputs by `OutputSpec.view`, with the fixed rules of SPEC §3.2 for tabs that have no outputs.

**`GET /api/runs/{rid}/outputs/{oid}`**
→ `200` the normalised JSON content of one catalog output (merged manifest section or file), plus `_meta: {source, stale, older_code}`. Errors: `404 output_missing` (`detail: {output, produced_by, expected_path}`, `action`).

**`GET /api/runs/{rid}/outputs/batch`**
Query: `ids=a,b,c` → `200 {results: Record<string, unknown>, missing: {id, produced_by, action}[]}`.

**`GET /api/runs/{rid}/views/{view}`**
`view ∈ overview|data|accuracy|distance|influence|response|scenarios|climate|causal|budget|planner|uncertainty|provenance`
→ `200 ViewModel`:

```ts
{ view: string; availability: Availability; missing: {output: string, produced_by: string, action: Action|null}[];
  units: { target: string; levers: Record<string, string> }; caveats: string[]; demo: boolean;
  sections: Record<string, unknown> }   // column-oriented, chart-ready; keys per view below
```

Section keys per view. Each key is present or `null`.

| View | Section keys |
|---|---|
| `overview` | `kpis, timings, flags, outputs_grid, studies, limitations, findings` |
| `data` | `qa_tiles, flags, frac_hist, dose_scale, predictor_hists, corr_matrix, zone_counts, coarse, joins` |
| `accuracy` | `models, obs_pred_bins, resid_hist, resid_by_zone, resid_by_fold, interval_honesty, stacker, physics, advection, forcing, cv_design, live` (`live: true` when computed from predictions mid-run) |
| `distance` | `curve, baselines, verdict, block_wins` |
| `influence` | `ranges, correlogram, rings, anisotropy, priors` |
| `response` | `levers: {var: {curve, shapes, effects, fold_means}}, literature` |
| `scenarios` | `rows, ladders, has_detail` |
| `climate` | `warming, models_table, exposure, offset, thresholds` |
| `causal` | `treatments: {t: {forest, audit, dr_curve, model_pd_curve, cate, cate_layer, sensitivity, controls}}, flags, dag_audit` (`model_pd_curve` and `cate_layer` are null on older runs) |
| `budget` | `kpis, pareto, caption, top_cells, status` (`status` = `"ok"` or core's `"no positive-benefit segments"`) |
| `planner` | `exposure, person_mean, hot_days, equity, plantable, zones, hex_files, sites, pairs, gis` |
| `uncertainty` | `rows, climate, sources` |
| `provenance` | `hashes, git, platform, launch` |

Errors: `404 unknown_view`.

**`GET /api/runs/{rid}/docs`**
→ `200 {id: "report"|"methods"|"model_card"|"uncertainty"|"placebo"|"multiverse"|"simcheck"|"benchmark", file: string, title: string, mtime: string|null, present: boolean, regenerable: boolean, frozen: boolean}[]`.

**`GET /api/runs/{rid}/docs/{doc}`**
→ `200 {markdown: string, mtime: string, frozen: boolean}`. Errors: `404 output_missing`.

**`GET /api/runs/{rid}/files`**
Query: `path?` (directory) → `200 {name, relpath, dir: boolean, bytes, mtime, output_id: string|null, state: string|null, in_manifest: boolean}[]`.

**`GET /api/runs/{rid}/files/raw`**
Query: `path`, `as?=csv|json|html|geojson` (conversion).
→ file stream with `Content-Disposition`. Supports `Range` for native files.

Errors:
- `404`
- `415 no_conversion`
- `413 too_many_features` (GeoJSON > 60,000)
- `422 needs_crs`

**`GET /api/runs/{rid}/files/table`**
Query: `path`, `limit?=200`, `columns?` → `200 {columns: {name, dtype}[], rows: unknown[][], n_rows: number}`.

**`GET /api/runs/{rid}/dictionary`**
→ `200 {output: string, column: string, unit: string, sign: string|null, description: string}[]`.

### 6.2 Grid, layers, cells `[R]`

**`GET /api/runs/{rid}/grid`** → `200 GridMeta`.

**`GET /api/runs/{rid}/grid.bin`**
Packed `ix:int32, iy:int32, lon:float32, lat:float32, zone:int16`. `lon`/`lat` are NaN when there is no CRS. Uses the `X-SPARC-Offsets` header.
`zone` is an index into `GridMeta.zones` for numeric and string zone codes alike (`-1` = no zone, and every row is `-1` when the config has no zone column), so codes that do not fit int16 survive. A `{kind: "zones"}` selection carries the codes themselves (`GridMeta.zones[zone]`), not the indices.

**`GET /api/runs/{rid}/grid/ids.bin`** → Int64[n]. Errors: `409 string_ids`, in which case use `GET /api/runs/{rid}/grid/ids.json` → `string[]`.

**`GET /api/runs/{rid}/layers`**
→ `200 {groups: {id: string, label: string, layers: LayerMeta[]}[]}`.

Generated from the files present. Studio result, plan and comparison layers are listed under the groups `studio_results`, `studio_plans` and `studio_compare`.

**`GET /api/runs/{rid}/layers/{key}.bin`**
→ Float32 or Uint8 (per `LayerMeta.dtype`).

Keys:
- catalog keys
- `res:<res_id>:<delta|delta_sd|extrapolation|abs|realized_<var>>`
- `plan:<plid>:<dose|planned_benefit|closed_loop_delta>`
- `cmp:<cid>:<a>__<b>`
- `sc:<slug>`, `sc_sd:<slug>`, `sc_ex:<slug>`; `<slug>` = `sparc.core.catalog.scenario_slug(name)` (SPEC §6.1)
- `cate_<t>`, `mslope_<t>`, `mslope_own_<t>` (from `causal_cells.parquet`, when present)

Errors: `404 unknown_layer`, `404 output_missing`.

**`GET /api/runs/{rid}/folds/{k}.bin`**
→ Uint8[n]: 0 train, 1 test, 2 buffer, 255 not in fold design. Errors: `404` (`k ≥ n_folds`).

**`GET /api/runs/{rid}/cells/{index}`**
`index` is the row index. Query: `scenarios?=res_…,configured:<slug>` → `200`:

```ts
{ index: number; id: number|string; lon: number|null; lat: number|null; zone: number|string|null;
  values: Record<string, number|null>;    // every catalog layer
  curves: Record<string, { model: string; A: number|null; ds: number|null; inflection: number|null; d90: number|null;
                           dmax: number|null; dose: number[]; benefit: number[] }>;
  scenarios: Record<string, number|null> }
```

**`GET /api/runs/{rid}/hex`**
Query: `size=250|500`, `layers=a,b`, `sums?=people`, `fmt?=json|csv|geojson|gpkg` (non-JSON formats return a file download; `geojson`/`gpkg` need a CRS → else `422 needs_crs`). JSON → `200 {hex: {key: number, cx: number, cy: number, lon: number|null, lat: number|null, n_cells: number, values: Record<string, number|null>}[]}`.

### 6.3 Selections, regions, blobs `[R]`

**`POST /api/runs/{rid}/selection/resolve`**
Body: `{selection: SelectionSpec}` → `200 {n_cells: number, area_km2: number, people: number|null, medians: Record<string, number|null>, mask: Bitset, portable: boolean, warnings: string[]}`.

Errors: `422 validation` (unknown column or namespace, bad geometry), `422 needs_crs` (EPSG:4326 geometry on a run without a CRS).

**`GET /api/runs/{rid}/regions`** → `200 {id, name, spec: SelectionSpec, n_cells, created_utc}[]`.
**`POST /api/runs/{rid}/regions`** Body `{name, spec}` → `201 {id, name, spec, n_cells}`.
**`DELETE /api/runs/{rid}/regions/{id}`** → `200 {ok: true}`.

Regions are stored per project and resolved per run. A region created on one run is listed on every run of the project, with a `portable` flag.

**`PUT /api/runs/{rid}/blobs`** (raw upload)
- Query: `kind=mask|edit`.
- Body for `mask`: the bitset bytes (raw, not base64).
- Body for `edit`: `Int32 idx[m]` followed by `Float32 val[m]`, with header `X-SPARC-Count: m`.
- → `201 {blob_id: string, bytes: number}`.

### 6.4 Analysis tools `[R]`

**`POST /api/runs/{rid}/stats/region`**
Body: `{selection: SelectionSpec, layers: string[], weights?: "people"|null, scenarios?: string[]}`. Scenario refs are `res_…` or `configured:<slug>`.
→ `200`:

```ts
{ n_cells: number; area_km2: number; people: number|null;
  layers: Record<string, { mean: number|null; sd: number|null; p10: number|null; p50: number|null; p90: number|null;
                           mean_outside: number|null }>;
  scenarios: Record<string, { inside: Likely; outside: Likely; has_folds: boolean }> }
```

**`POST /api/runs/{rid}/stats/breakdown`**
Body: `{value: string, by: {kind: "zone"|"quantile"|"hex"|"category"|"fold", layer?: string, q?: number, size_m?: 250|500}, weights?: "people"|null, stat: "box"|"mean"}`
→ `200 {groups: {label: string, n: number, people: number|null, mean: number|null, q: [number,number,number,number,number]|null}[]}`.

**`POST /api/runs/{rid}/stats/hexbin`**
Body: `{x: string, y: string, bins?: number = 60, selection?: SelectionSpec}`
→ `200 {x_edges: number[], y_edges: number[], counts: number[][], sel_counts: number[][]|null, spearman: number|null, binned_mean: {x: number, y: number}[]}`.

**`POST /api/runs/{rid}/stats/acf`**
Body: `{layer: string, max_lag_m?: number, n_perm?: number = 19}`
→ `200 {lags_m: number[], acf: number[], band_mean: number[], band_sd: number[]}`.

**`GET /api/runs/{rid}/export/layer/{key}`**
Query: `fmt=tif|csv|geojson|parquet` → file. Errors: `422 needs_crs`, `413 too_many_features`.

### 6.5 Compare runs `[R]`

**`GET /api/compare/runs`**
Query: `a`, `b` → `200`:

```ts
{ a: RunSummary; b: RunSummary;
  same: { data: boolean|null; config: boolean|null; code: boolean|null; grid: boolean };
  config_diff: { path: string; a: unknown; b: unknown }[];
  metrics: { key: string; a: number|null; b: number|null; delta: number|null }[];
  timings: { stage: string; a: number|null; b: number|null }[];
  scenarios: { name: string; a: Likely|null; b: Likely|null }[];
  climate: object|null; causal: object|null;
  environment: { added: string[]; removed: string[]; changed: { name: string; a: string; b: string }[] };
  outputs: { a_only: string[]; b_only: string[] } }
```

**`GET /api/compare/layer.bin`**
Query: `a`, `b`, `key` → Float32 (b − a). Errors: `409 grid_mismatch`.

**`POST /api/compare/priority`**
Body: `{a, b, layer}` → `200 {kendall_tau: number, top_decile_jaccard: number, n: number}`. Errors: `409 grid_mismatch`.

---

## 7. Scenario Lab `[E]`

### 7.1 Types

```ts
type Edit = { lever: string;
  mode: "add"|"set"|"scale"|"floor"|"ceiling"|"fill_headroom"|"to_percentile"|"per_cell";
  amount?: number; percentile?: number; paved_share?: number;
  per_cell_ref?: string;            // "plan:<plid>" | "blob:<blob_id>" | "csv:<project-relative path>"
  where?: SelectionSpec;            // default {kind:"all"}
  label?: string };
type ScenarioDoc = { name: string; notes?: string; tags?: string[]; anchor_run_id?: string|null;
  edits: Edit[]; regions?: Record<string, SelectionSpec>;
  costs?: Record<string, { per_unit: number }>;
  options?: { clip_to_support?: boolean; mediators?: boolean; expert?: boolean } };
type Scenario = { id: string; project_id: string; revision: number; parent_id: string|null; children: string[];
  doc: ScenarioDoc; content_hash: string; status: "draft"|"previewed"|"exact"|"stale"|"archived";
  created_utc: string; updated_utc: string; results: ResultSummary[] };
type ScenarioSummary = Pick<Scenario, "id"|"revision"|"parent_id"|"status"|"created_utc"|"updated_utc"> &
  { name: string; tags: string[]; latest: ResultSummary|null };
type ResultSummary = { id: string; scenario_id: string|null; run_id: string; kind: "exact"|"configured"|"plan"|"sweep_point";
  created_utc: string; stale: boolean; has_folds: boolean; city: Likely; edited: Likely|null;
  frac_extrapolated_edited: number|null; job_id: string|null };
type EngineState = "no_checkpoint"|"cold"|"queued"|"loading"|"ready"|"busy"|"incompatible"|"error";
type ItemRef = { kind: "result"; id: string } | { kind: "configured"; slug: string } | { kind: "plan"; id: string } | { kind: "baseline" };
```

### 7.2 Engine

**`GET /api/engine`**
→ `200 {state: EngineHostState, pid: number|null, rss_mb: number|null, budget_gb: number, max_runs: number, runs: {run_id: string, est_rss_mb: number, loaded_utc: string, last_used_utc: string}[], busy_job_id: string|null, queue: string[]}`.

**`POST /api/engine/restart`** → `202 {ok: true}`. Running engine jobs fail with `engine host exited`.

**`GET /api/runs/{rid}/engine`**
→ `200 {state: EngineState, progress: number|null, step: string|null, rss_mb: number|null, est_rss_mb: number, code_match: boolean|null, loaded_utc: string|null, last_used_utc: string|null, error: {type, message}|null, job_id: string|null, action: Action|null}`.

**`POST /api/runs/{rid}/engine/open`** → `202 Job` (`engine.open`). If the run is already loaded the response is `200` with the engine status.
Errors:
- `409 no_checkpoint`
- `409 engine_memory` (`detail: {needed_gb, available_gb, holders: {job_id|run_id, rss_gb}[]}`; `action` evicts or stops)
- `409 untrusted_pickle`

**`DELETE /api/runs/{rid}/engine`** → `200` engine status (the run is evicted).

### 7.3 Levers, emulator, preview, compile

**`GET /api/runs/{rid}/levers`** → `200`:

```ts
{ var: string; label: string; unit: string; min: number|null; max: number|null; direction: "increase"|"decrease";
  doses: number[]; cost_per_unit: number; design_dose: number|null; sd: number|null; headroom_available: boolean;
  role: string|null; mediator_children: string[];
  emulator: { available: boolean; trust: "good"|"rough"|"none"; patch_pass_rate: number|null; uniform_rel_err: number|null } }[]
```

**`GET /api/runs/{rid}/emulator`**
→ `200 {present: boolean, kernel_cells: number|null, levers: Record<string, {design_dose, bounds: [number, number], direction, trust, validation: object}>, action: Action|null}`. When the emulator is missing, `action` is `build_emulator`.

**`POST /api/runs/{rid}/preview`**
Body: `{edits: Edit[], brush?: Record<string, SparseEdit>, options?: ScenarioDoc["options"], request_seq: number}`

→ `200` packed binary (§0.4) with two arrays:
- `delta`: float32[n] ΔT in target units (negative = cooler);
- `edited`: uint8[ceil(n/8)], the edited-cell bitset as raw bytes (LSB-first).

Headers:
- `X-SPARC-Offsets` (as in §0.4)
- `X-SPARC-Summary: {"mean": number, "edited_mean": number, "n_edited": number, "outside_share": number, "trust": "good"|"rough"|"none", "hatched": boolean, "reasons": string[], "request_seq": number}`

The mask is in the body, not a header, so large grids cannot overflow proxy header limits.

Errors:
- `404 no_emulator` (`action: build_emulator`)
- `409 superseded` (a newer `request_seq` for this run arrived; the client ignores it)
- `422 validation`

**`POST /api/runs/{rid}/compile`**
Body: `{scenario: ScenarioDoc}` → `200`:

```ts
{ content_hash: string; portable: boolean;
  levers: Record<string, { n_cells: number; mean_requested: number; total_requested: number; predicted_mean_realised: number;
                           clipped_share: number; est_cost: number }>;
  union_cells: number; people: number|null;
  warnings: { code: string; message: string; edit_index: number|null; blocking: boolean }[];
  est_exact_s: number; emulator: { usable: boolean; hatched: boolean; reasons: string[] } }
```

### 7.4 Templates and library

**`GET /api/scenario-templates`**
→ `200 {id: string, label: string, desc: string, params_schema: object, requires: ("layers"|"canopy_role"|"impervious_role"|"albedo_role"|"crs")[]}[]`.

Template ids: `cool_roofs`, `shade_hottest`, `fill_plantable`, `depave`, `footprint_priority`, `around_sites`, `configured_package`.

**`POST /api/projects/{pid}/scenarios/from-template`**
Body: `{template: string, params: object, run_id: string}` → `201 Scenario`. Errors: `422 template_unavailable` (`detail.missing`).

**`GET /api/projects/{pid}/scenarios`**
Query: `run?`, `tag?`, `status?`, `q?`, `archived?=false` → `200 ScenarioSummary[]`.

**`POST /api/projects/{pid}/scenarios`**
Body: `{doc: ScenarioDoc}` → `201 Scenario`.

**`GET /api/scenarios/{sid}`** → `200 Scenario`.

**`PATCH /api/scenarios/{sid}`**
Body: `{doc?: ScenarioDoc, name?: string, tags?: string[], notes?: string, archived?: boolean}` → `200 Scenario`.
Errors: `409 conflict_revision` when `doc` changes `edits`/`regions`/`costs`/`options` and an exact result exists (`action`: fork).

**`POST /api/scenarios/{sid}/fork`**
Body: `{name?: string, doc?: ScenarioDoc}` → `201 Scenario` (revision + 1, `parent_id` = sid).

**`DELETE /api/scenarios/{sid}`**
Query: `results=false` → `200 {ok: true}`. Errors: `409 has_children` unless `force=true`.

**`POST /api/scenarios/{sid}/run`**
Body: `{run_id: string, force?: boolean = false}`
→ `200 {cached: ResultSummary}` on a cache hit, or `202 {job: Job}` (`engine.scenario`).
Errors: `409 no_checkpoint`, `422` (compile blocking warnings, `detail.warnings`).

**`POST /api/scenarios/run-batch`**
Body: `{run_id: string, scenario_ids: string[]}` → `202 {job: Job}` (`engine.batch`; one tick per scenario).

**`POST /api/scenarios/{sid}/ladder`**
Body: `{edit_index: number, amounts: number[], run_id?: string}` → `201 {scenarios: Scenario[], job: Job|null}`.

**`POST /api/scenarios/{sid}/across-runs/estimate`**
Body: `{run_ids: string[]}` → `200 {runs: {run_id: string, ok: boolean, reason: string|null, load_s: number, exact_s: number, rss_gb: number}[], total_s: number, peak_rss_gb: number}`.

**`POST /api/scenarios/{sid}/across-runs`**
Body: `{run_ids: string[]}` → `202 Job` (`scenario.across_runs`).

**`POST /api/scenarios/{sid}/promote`**
Body: `{apply?: boolean = false}` → `200 {eligible: boolean, reason: string|null, yaml_diff: string|null, names: string[], version: number|null}`.

**`GET /api/scenarios/{sid}/design.csv`**
Query: `run_id` → `text/csv` with columns `id,lever,change` (realised per-cell edit).

**`POST /api/runs/{rid}/designs/import`**
Body: raw CSV with columns `id,lever,change` (a per-cell increment) **or** `id,lever,value` (an absolute target; converted to `change = value − current input` on this run, which covers expert "upload an edited predictor table" use).
→ `201 {blobs: Record<string, string>, n_rows: number, unknown_ids: (number|string)[], levers: string[], mode: "change"|"value"}`. Blob ids are used as `per_cell_ref: "blob:<id>"`.

### 7.5 Results

**`GET /api/runs/{rid}/scenarios`** → `200`:

```ts
{ configured: { slug: string; name: string; city: Likely; p10: number|null; p90: number|null; frac_extrapolated: number|null;
                mean_realized: Record<string, number>; causal_linear: {delta: number, lo: number, hi: number, model_within: boolean}|null;
                has_folds: boolean; layer_key: string;
                doc: ScenarioDoc }[];          // the equivalent add-mode doc, used by "Clone to edit"
  results: ResultSummary[] }
```

`slug` = `scenario_slug(name)` (SPEC §6.1). `name` keeps the U+2212 minus.

**`POST /api/runs/{rid}/configured/{slug}/rerun-exact`** → `202 Job` (`engine.rerun_configured`).

**`GET /api/results/{res_id}`** → `200 Result`:

```ts
{ summary: ResultSummary; spec: object; scenario: { id: string, revision: number, name: string }|null;
  city: Likely; p10: number|null; p90: number|null; mean_delta_sd: number|null;
  regions: { name: string; auto: boolean; n_cells: number; mean: Likely; people_weighted: number|null; total: number;
             frac_cooled_01: number; frac_cooled_05: number }[];
  spill: { inside: number; outside: number; outside_share: number;
           rings: { r_m: number; mean: number; se: number|null; n: number }[]; lever_ranges: Record<string, number> };
  extrapolated_edited: number;
  realized: Record<string, { requested_mean: number; realized_mean: number; requested_total: number; realized_total: number; clipped_share: number }>;
  mediators: Record<string, { mean_change: number }>;
  cost: { total: number; per_lever: Record<string, number>; cooling_per_cost: number|null };
  causal_check: { delta: number; lo: number; hi: number; model_within: boolean }|null;
  uncertainty: { estimation_95: [number,number]|null; specification: [number,number]|null; attribution: [number,number]|null;
                 causal_band: [number,number]|null; envelope: [number,number]|null; envelope_excludes_zero: boolean|null; sources: string[] }|null;
  impacts: Impacts|null; preview_vs_exact: { mean_abs_err: number; rel_err: number }|null;
  plain: { headline: string; confidence: string; qualifiers: string[]; buys: string[] };
  warnings: { code: string; message: string }[]; stale: boolean; demo: boolean }
type Impacts = { thresholds: number[];
  exposure: { case: string; adapted: boolean; person_mean_temp: number; people_ge: Record<string, number>; share_people_ge: Record<string, number> }[];
  equity: Record<string, { quintiles: { quintile: number; mean_cooling: number; people: number; value_range: [number, number] }[]; concentration_index: number }>;
  hot_days: { station: string; cases: object[] }|null; hot_days_action: Action|null;
  zones: object[]; hexes: { "250": object[]; "500": object[] };
  climate_offset: { experiment: string; period: string; offset_share: number }[] };
```

**`GET /api/results/{res_id}/layers/{field}.bin`**
`field ∈ delta|delta_sd|extrapolation|abs|realized_<var>` → Float32[n].

**`POST /api/results/{res_id}/impacts`**
Body: `{thresholds?: number[], futures?: {experiment: string, period: string}[]}` → `200 Impacts` (cached by params). Errors: `422 needs_layers` (`action: fetch_input`).

**`DELETE /api/results/{res_id}`** → `200 {ok: true}`.

### 7.6 Compare

**`POST /api/runs/{rid}/compare`**
Body: `{items: ItemRef[] /* 2–4 */, regions?: string[], thresholds?: number[]}` → `201 Comparison`:

```ts
{ id: string; items: { ref: ItemRef; label: string; city: Likely; edited: Likely|null; cost: number|null; has_folds: boolean }[];
  pairs: { a: number; b: number; city: Likely & { paired: boolean }; regions: Record<string, Likely & { paired: boolean }>;
           layer_key: string }[];
  equity: Record<string, Record<string, number>>; exposure: object[]; cooling_per_cost: Record<string, number|null>;
  needs_exact: ItemRef[] }
```

**`GET /api/comparisons/{cid}`** → `200 Comparison`.

**`GET /api/runs/{rid}/comparisons`** → `200 {id, items: ItemRef[], created_utc}[]`.

**`DELETE /api/comparisons/{cid}`** → `200 {ok: true}`.

### 7.7 Climate × adaptation

**`GET /api/runs/{rid}/climate/factors`**
→ `200 {present: boolean, path: string|null, experiments: string[], periods: string[], models: string[], warming: {experiment, label, period, n_models, median, p10, p90, min, max, by_model: Record<string, number>}[], action: Action|null}`.

**`POST /api/runs/{rid}/climate/explore`**
Body: `{adaptations: ItemRef[], thresholds?: number[], experiments?: string[], periods?: string[], statistic?: "median"|"p10"|"p90"|{model: string}}`
→ `200 {present, thresholds, projections: object[], adaptation: string[], people_exposure: object[]|null, units}`. The payload follows `climate.summarize_projections`. Errors: `404 no_climate_factors` (`action: fetch_input`).

### 7.8 Sweeps

**`POST /api/runs/{rid}/sweeps`**
Body: `{lever: string, doses: number[], selection?: SelectionSpec}` → `202 {sweep_id: string, job: Job}`.

**`GET /api/sweeps/{swid}`**
→ `200 {params, status, curve: {dose: number, city: Likely, region: Likely|null, realized: number, frac_extrapolated: number}[], fit: {model: string, A: number|null, ds: number|null, d90: number|null}|null, pipeline_curve: object|null, points: string[]}`.

**`GET /api/runs/{rid}/sweeps`** → `200 {id, lever, doses: number[], status, created_utc, job_id}[]`.

**`DELETE /api/sweeps/{swid}`** → `200 {ok: true}` (also removes its `sweep_point` results). Errors: `409 active`.

### 7.9 Plans

```ts
type PlanParams = { lever: string; budget: number;
  cost: { scalar: number } | { column: string };   // a namespaced column (§1) or "csv:<project-relative path>:<column>" joined by id
  cap: { plantable: boolean; paved_share?: number; region?: SelectionSpec };
  min_dose?: number; objective: "cooling"|"people";
  equity?: { source: "share_60_plus"|"share_under_5"|"density"|"column"; column?: string /* same forms as cost.column */; focus: number };
  multipliers?: number[] };
type PlanPreview = { planned_total: number; n_cells_treated: number; mean_dose_treated: number; total_cost: number; gini: number;
  min_dose_dropped_cost: number;   // budget freed by the min_dose post-filter, not re-spent
  pareto: { budget: number; benefit: number; n_cells: number; n_segments: number; gini: number }[];
  dose: string /* base64 Float32[n] */; constraint: string; objective: string; caption: string };
```

| Endpoint | Body / query | Response | Errors |
|---|---|---|---|
| `POST /api/runs/{rid}/plans/preview` | `PlanParams` | `200 PlanPreview` (inline, < 1 s) | `422 needs_responses` (S4 missing), `422 needs_layers` |
| `POST /api/runs/{rid}/plans` | `{params: PlanParams, name: string, verify?: boolean = true}` | `201 {plan: Plan, job: Job\|null}` | — |
| `GET /api/runs/{rid}/plans` | — | `200 Plan[]` | — |
| `GET /api/plans/{plid}` | — | `200 Plan` (see below) | — |
| `POST /api/plans/{plid}/verify` | `{frontier?: boolean = false}` | `202 Job` (`engine.plan_verify` or `engine.plan_frontier`) | — |
| `GET /api/plans/{plid}/layers/{field}.bin` | `field ∈ dose\|planned_benefit\|closed_loop_delta` | Float32[n] | — |
| `POST /api/plans/{plid}/to-scenario` | — | `201 Scenario` | — |
| `POST /api/plans/{plid}/field-kit` | `{n_sites?: 30, min_spacing_m?: 400, n_pairs?: 30, min_distance_m?: 1000}` | `200 {cells: …, sites: …, pairs: …}` (see below) | — |
| `DELETE /api/plans/{plid}` | — | `200 {ok: true}` | — |

`Plan`:

```ts
{ id: string; name: string; params: PlanParams; planned: PlanPreview & { dose?: never };
  realised: { total: number, mean_treated: number, mean_all: number, result_id: string }|null;
  frontier: { budget: number, planned: number, realised: number }[]|null; created_utc: string }
```

`field-kit` response:

```ts
{ cells: {rank, id, lon, lat, zone, dose, planned_benefit, closed_loop_delta, people, plantable_pp}[];
  sites: {id, lon, lat, role, canopy, impervious, effect_sd}[];
  pairs: {treated_id, control_id, treated_lon, treated_lat, control_lon, control_lat, covariate_distance}[] }
```

---

## 8. Job kinds: params and results

Params are validated by pydantic per kind (`additionalProperties: false`). `result` is written to `result.json` and exposed as `Job.result`.

`export_id` in the `export.*` params is **server-filled**. `POST /api/exports` creates the `exports` row first and injects the id; clients never send it. Every export writes under `projects/<slug>/exports/<export_id>/`.

| Kind | Params | Result |
|---|---|---|
| `run.core` `[R]` | `{run_id: string, resume: boolean, use_current_config?: boolean, threads?: number}` (other args come from `launch.json`; `use_current_config` re-snapshots it as SPEC §4.3) | `{run_id, status, timings_s, metrics: {r2, rmse, coverage}\|null, done: string[]}` |
| `run.external` `[R]` | `{run_id}` (registry-created pseudo-job for a live CLI run; never queued, never cancellable) | `{run_id, status}` |
| `input.forcing` `[P]` | as `POST …/inputs/forcing` | `{path, sw_down, lw_net, wind: [number, number], checks: string[], linked: boolean}` |
| `input.layers` `[P]` | `{link: boolean}` | `{path, n_cells, people_total, linked}` |
| `input.features` `[P]` | as the endpoint | `{path, agreement: object[] (empty in bootstrap mode), open_project_id: string\|null, linked: boolean}` |
| `input.cmip6` `[P]` | as the endpoint (incl. `periods`, `baseline`) | `{path, n_models, experiments, periods: string[], skipped_models: string[], linked}` |
| `input.ghcn` `[P]` | `{station}` | `{path, years: [number, number]}` |
| `input.stations` `[P]` | `{}` | `{path, n_stations}` |
| `post.baselines` `[S]` | `{models?: string[]}` | `{verdict, best_baseline, rows: object}` |
| `post.planner` `[S]` | `{package?: string /* configured slug or exact name */, thresholds?: number[], hex_sizes?: number[], export?: boolean}` | `{people_total, package, files: string[], hot_days: boolean}` |
| `post.emulator` `[S]` | `{patches?: number = 8}` | `{levers: Record<string, {patch_pass_rate, uniform_rel_err}>}` |
| `post.uncertainty` `[S]` | `{multiverse_study?: string, simcheck_studies?: string[], placebo_study?: string, real_r2_gate?: boolean}` (default: attached studies) | `{n_scenarios, sources}` |
| `post.writeup` `[S]` | `{}` | `{files: string[]}` |
| `study.placebo` `[S]` | `{kinds: ("grf"\|"shift"\|"rotate")[], coarse_m: number\|null = 60, seed: number = 0, grf_range_m: number = 600}` | `{n_pass_model, n_pass_causal, n_placebos, children: string[]}` |
| `study.simcheck` `[S]` | `{design: Record<"physics"\|"additive"\|"own_only"\|"coarse_scale"\|"confounded"\|"null", number>, coarse_m: number\|null = 90, epochs: number = 200, workers: number = 1, threads: number = 1, continue_study_id?: string}` | `{n_rows, n_errors, summary: object}` |
| `study.multiverse` `[S]` | `{variants?: string[], custom_variants?: Record<string, Record<string, unknown>>, coarse_m: number\|null = 60, workers: number = 1, threads: number = 1}` | `{sign_stability_min, median_kendall_tau, children: string[]}` |
| `study.reproduce` `[S]` | `{stages?: string[] = ["S0","S1","S2","S3"], tol_r2?: number = 0.01, tol_effect?: number = 0.05}`. The server passes `config_dir` from the parent's `launch.json` or import config and `out_dir` = the child run dir; neither is a client param | `{pass: boolean, child_run_id, n_hard_fail: number}` |
| `study.benchmark` `[S]` | `{seed?: 0, ab?: true, epochs?: 150, n?: 96}` | `{runs: object}` (the kind writes `benchmark.json`/`.md` in the study dir) |
| `export.bundle` `[S]` | `{export_id, run_id, outputs?: string[], include_checkpoint?: boolean = false}` | `{export_id, path, bytes}` |
| `export.gis` `[S]` | `{export_id, run_id, layers?: string[]}` | `{export_id, path, bytes, files: string[]}` |
| `export.page` `[S]` | `{export_id, run_id, placebo_study?: string}` | `{export_id, path, bytes}` |
| `export.report` `[S]` | `{export_id, run_id, sections: string[], result_ids?: string[], plan_ids?: string[], finding_ids?: string[], format: "html"\|"md"}` | `{export_id, path, bytes}` |
| `export.findings` `[S]` | `{export_id, project_id, run_id?, ids?: string[], format: "md"\|"html"}` | `{export_id, path, bytes}` |
| `export.decision_pack` `[E]` | `{export_id, result_id, thresholds?: number[]}` | `{export_id, path, bytes, draft: boolean}` |
| `export.plan_pack` `[E]` | `{export_id, plan_id}` | `{export_id, path, bytes}` |
| `export.compare_pack` `[E]` | `{export_id, comparison_id}` | `{export_id, path, bytes}` |
| `engine.open` `[E]` | `{run_id}` | `{load_seconds: object, rss_mb, code_match}` |
| `engine.scenario` `[E]` | `{run_id, scenario_id, revision}` | `{result_id}` |
| `engine.batch` `[E]` | `{run_id, scenario_ids: string[]}` | `{result_ids: string[], failed: {scenario_id, error}[]}` |
| `engine.rerun_configured` `[E]` | `{run_id, slug}` | `{result_id}` |
| `engine.sweep` `[E]` | `{run_id, sweep_id}` | `{sweep_id, points: string[]}` |
| `engine.plan_verify` / `engine.plan_frontier` `[E]` | `{plan_id}` | `{result_id, realised_total}` / `{frontier: object[]}` |
| `scenario.across_runs` `[E]` | `{scenario_id, run_ids: string[]}` | `{rows: {run_id, city: Likely, ok, error}[], sign_stability, spread}` |
| `test.sleep` / `test.events` / `test.fail` / `test.ignore_sigterm` / `test.pool` `[F]` | `{seconds?, n_stages?, message?, workers?}` | `{ok: true}` |

---

## 9. Studies and post-run actions `[S]`

```ts
type StudyStatusRow = { kind: "baselines"|"planner"|"emulator"|"uncertainty"|"writeup"|"placebo"|"simcheck"|"multiverse"|"reproduce"|"literature"|"benchmark";
  state: "not_run"|"queued"|"running"|"done"|"stale"|"failed"; study_id: string|null; job_id: string|null; updated_utc: string|null;
  headline: string|null; estimate: { est_s: number, est_lo: number, est_hi: number }|null; attached: boolean|null; action: Action|null;
  requirements: { ok: boolean, missing: string[] } };
type Study = { id: string; project_id: string; kind: string; target_run_id: string|null; job_id: string|null; out_dir: string;
  status: string; params: object; summary: object|null; origin: "studio"|"imported"; created_utc: string; updated_utc: string;
  children: RunSummary[]; attached_runs: string[]; stale_vs: string[] };
```

### Status and launch

**`GET /api/runs/{rid}/studies`** → `200 StudyStatusRow[]`.

**`POST /api/runs/{rid}/actions/{kind}`**
`kind ∈ baselines|planner|emulator|uncertainty|writeup`. Body: the kind's params (§8) → `202 Job`.
Errors: `422 requirements` (`detail.missing`, e.g. `planner.layers`), `409 no_checkpoint` (emulator).

**`POST /api/runs/{rid}/studies/{kind}`**
`kind ∈ placebo|simcheck|multiverse|reproduce`. Body: params → `202 {study: Study, job: Job}`.
Errors:
- `422 requirements` (e.g. roles canopy and impervious missing for placebo/simcheck);
- `422 validation` when `workers × threads > threads_heavy` (simcheck, multiverse).

**`GET /api/runs/{rid}/truth`**
Synthetic (demo) projects only. It compares `truth.json` (SPEC §9.2) with the run → `200`:

```ts
{ rows: { quantity: "canopy_scenario"|"footprint_mean"|"L_m"|"influence_radius_m"|"noise_sd"; label: string;
          truth: number; recovered: number|null; se: number|null; share: number|null; unit: string;
          scenario: string|null }[] }
```

Errors: `404 not_found` (not a demo project, or `truth.json` missing).

**`POST /api/projects/{pid}/studies/benchmark`**
Body: params → `202 {study, job}`.

**`POST /api/studies/estimate`**
Body: `{kind: string, run_id?: string, params: object}` → `200 {est_s, est_lo, est_hi, est_peak_rss_gb, est_disk_gb, n_children}`.

### Reading and managing studies

**`GET /api/projects/{pid}/studies`**
Query: `kind?` → `200 Study[]`.

**`GET /api/studies/{stid}`** → `200 Study`.

**`GET /api/studies/{stid}/view`** → `200` per kind:

| Kind | Response |
|---|---|
| placebo | `{rows, layer_correlation, n_pass_model, n_pass_causal, children: {kind, run_id, status, verdict}[]}` |
| simcheck | `{design, grid: {generator, seed, status: "pending"\|"running"\|"done"\|"gate_fail"\|"error", share, ci_covers, causal_covers, seconds, gate_attempt}[], generators: object, bias_correction, eta_s}` |
| multiverse | `{variants: {name, label, status, r2, rmse, seconds, run_id}[], effects: object, priority: object, stability: object}` |
| reproduce | `{pass, checks: {check, ok, hard, detail}[], original, reproduction}` |
| benchmark | `{runs}` |

**`POST /api/studies/{stid}/resume`** → `202 Job`.

**`POST /api/studies/{stid}/attach`** · **`POST /api/studies/{stid}/detach`**
Body: `{run_id}` → `200 {study: Study, job: Job|null}`. Attaching enqueues `post.uncertainty` when `auto_uncertainty` is on.

**`POST /api/studies/simcheck/merge`**
Body: `{study_ids: string[]}` → `200 {summary: object, markdown: string}` (inline).

**`DELETE /api/studies/{stid}`**
Query: `files=false` → `200 {ok: true}`. Errors: `409 active`.

---

## 10. Exports and reports `[S]` (pack kinds `[E]`)

**`POST /api/exports`**
Body: `{kind: "bundle"|"gis"|"page"|"report"|"findings"|"decision_pack"|"plan_pack"|"compare_pack", project_id: string, params: object}`. The params are those of the `export.<kind>` job (§8), without `export_id`.
→ `202 {export: Export, job: Job}`.

The route (owned by `[S]`) creates the `exports` row, injects `export_id`, and enqueues `export.<kind>` by name. The pack kinds are owned by `[E]`; the job registry dispatches them. Files go to `projects/<slug>/exports/<export_id>/`. The `exports` row is completed by the kind's `on_finish` hook (SPEC §10.2).

```ts
type Export = { id: string; project_id: string; run_id: string|null; kind: string; ref: string|null; options: object; job_id: string;
  status: "running"|"ready"|"failed"; path: string|null; bytes: number|null; draft: boolean; created_utc: string };
```

**`GET /api/projects/{pid}/exports`** → `200 Export[]`.

**`GET /api/exports/{eid}`** → `200 Export`.

**`GET /api/exports/{eid}/download`** → file (streamed). Errors: `409 not_ready`.

**`DELETE /api/exports/{eid}`** → `200 {ok: true}`.

**`POST /api/projects/{pid}/report/preview`**
Body: `{run_id, sections: ("summary"|"accuracy"|"validation"|"scenarios"|"plans"|"climate"|"equity"|"caveats"|"limitations"|"provenance"|"findings")[], result_ids?, plan_ids?, finding_ids?}` → `200 {html: string}`.

---

## 11. Findings `[S]`

```ts
type Finding = { id: string; project_id: string; run_id: string|null; view: string; url_state: string; title: string;
  note_md: string; snapshot: object; image_url: string|null; position: number; created_utc: string; updated_utc: string };
```

| Endpoint | Body / query | Response |
|---|---|---|
| `GET /api/findings` | `project?`, `run?` | `200 Finding[]` (ordered by `position`) |
| `POST /api/findings` | `{project_id, run_id?, view, url_state, title, note_md?: "", snapshot: object}` | `201 Finding` |
| `PUT /api/findings/{id}/image` | raw `image/png` or `image/svg+xml` (≤ 10 MB) | `200 {image_url}` |
| `GET /api/findings/{id}/image` | — | the image |
| `PATCH /api/findings/{id}` | `{title?, note_md?, position?}` | `200 Finding` |
| `DELETE /api/findings/{id}` | — | `200 {ok: true}` |

---

## 12. On-disk contracts (shared between work items)

These file formats are contracts. Items read each other's files by format, not by importing each other's modules.

### 12.1 Core run-dir sidecars (written by core)

```jsonc
// run_state.json
{"schema": 1, "status": "running|succeeded|failed|cancelled", "pid": 4242, "job": "j_…", "host": "laptop",
 "started_utc": "…", "updated_utc": "…", "stage": "cv_curve", "done": ["S3","baselines"], "fingerprint": "16hex",
 "events_path": "/abs/jobs/j_…/events.jsonl", "error": null,
 "meta": {"n_points": 54701, "cell_m": 29.998, "grid_shape": [334, 287], "coarse_m": null, "subsample_window_n": null,
          "cv": {"n_folds": 5, "block_m": 2000.0, "buffer_m": 666.7, "seed": 42}}}
// checkpoint.json
{"schema": 1, "fingerprint": "16hex", "done": ["S3","baselines"], "bytes": 525000000, "saved_utc": "…", "code_sha256": "…",
 "sections": {"data": "…", "core": "…", "s4": "…", "s5": "…", "climate": "…", "s6": "…", "s7": "…", "code": "…"}}
```

`scenario_detail.npz` holds:
- `ids`
- `names` (JSON-encoded list, uint8 bytes)
- for each scenario index `i`: `f{i}` (float32 K×n), `sd{i}` (float32 n), `ex{i}` (float32 n)

### 12.2 Studio run dir (`<run_dir>/studio/` or `<ws>/imports/<run_id>/studio/`)

```
launch.json                       # SPEC §4.3
cache/grid.npz                    # ix, iy (int32), ids, lon, lat (float32), zone; keyed by "key" entry
engine/base_fold.npy              # float64 K×n; engine/base_fold.json {ckpt_key, code_sha}
results/<res_id>/spec.json        # {scenario_id, revision, content_hash, compiled: {levers: …}, run_id, ckpt_key, code_sha}
results/<res_id>/summary.json     # Result (api §7.5) minus impacts
results/<res_id>/cells.parquet    # id, delta, delta_sd, extrapolation, realized_<var>… (float32)
results/<res_id>/folds.npy        # float32 K×n
results/<res_id>/impacts_<hash>.json
results/<res_id>/warnings.json
plans/<plid>/{params.json, planned.json, dose.npy, planned_benefit.npy, realised.json}
sweeps/<swid>/{params.json, curve.json}
comparisons/<cid>/{items.json, summary.json, diff_<a>__<b>.npy}
blobs/<blob_id>.bin
```

Exports are **not** stored here. They live under the project (§12.4).

### 12.4 Project dir (`<ws>/projects/<slug>/`)

```
project.json  config.yml  data/  inputs/{forcing,climate,layers,features}/  runs/<run_id>/  studies/<study_id>/
exports/<export_id>/…                 # every export kind; never inside a run dir
scenarios/<sid>.json                  # {"schema": 1, "revisions": [ScenarioDoc + id/revision/parent_id/content_hash/status], "archived": bool}
findings/<fid>.json  findings/<fid>.png|svg   # Finding minus image_url, plus "image": filename|null
```

The scenario mirror is written by `[E]` on every create, patch or fork, and the findings mirror by `[S]` on every change. Both are atomic (tmp + rename). `sparc studio --reindex` rebuilds the `scenarios` and `findings` tables from them.

### 12.3 Job dir (`<ws>/jobs/<jid>/`)

```
job.json      {id, kind, params, project_id, run_id, study_id, scenario_id, threads, created_utc, lane, executor}
state.json    {pid, pgid, proc_create_time, cmdline_token, executor, status, started_utc}
events.jsonl  canonical event log (SPEC §5.3), byte-offset cursor
stdout.log  stderr.log
result.json   {status: "succeeded"|"failed"|"cancelled", exit_code, result: object|null, error: {type, message, traceback_tail}|null}
cancel        (presence = cancel requested)
```

---

## 13. Engine host protocol (internal, `[E]`)

**Transport.** `multiprocessing.connection` with `family="AF_UNIX"` (`AF_PIPE` on Windows), address `<ws>/engine/host.sock`, `authkey` from `engine/host.json`. Each message is a pickled dict. The host only accepts dicts of primitive types and numpy arrays from the server, and replies with JSON-compatible dicts plus file paths.

**Request:**

```py
{"op": "status"|"open"|"close"|"scenario"|"batch"|"rerun_configured"|"sweep"|"plan_verify"|"plan_frontier"|"shutdown",
 "request_id": "uuid", "job_id": "j_…"|None, "job_dir": "/abs"|None, "run_id": "…"|None, "run_dir": "/abs"|None,
 "threads": 1|2, "payload": {...}}
```

**Reply:**

```py
{"request_id": "…", "ok": True, "result": {...}}
{"request_id": "…", "ok": False, "error": {"type": "Cancelled"|"Incompatible"|"MemoryBudget"|"…", "message": "…", "traceback_tail": "…"}}
```

**Behaviour:**
- Requests are processed one at a time.
- `status` may be answered between folds by a lightweight reader thread.
- The host wraps each request in `progress.job_scope(job_id, sink=job_dir/events.jsonl, cancel_file=job_dir/cancel)` and `progress.limit_threads(threads)`.
- Results are written under the run's `studio/` directory before replying.

---

## 14. Worker protocol (internal, `[F]`)

**Command:** `python -m sparc.studio.jobs.worker <job_dir>`

**Steps:**
1. Read `job.json`.
2. Call `progress.configure_from_env()`, `install_signal_handlers()` and `set_threads(job.threads)`.
3. Import `KIND_MODULES`.
4. Call the registered function with `(JobContext, params_model)`.
5. Write `result.json`.

**Exit codes:**
- `0`: succeeded
- `1`: failed, with `error` filled in
- `130`: cancelled

**Replay runner** (`SPARC_STUDIO_RUNNER=replay:<fixture_dir>`, tests only):
- `run.core` jobs are executed by `python -m sparc.studio.jobs.replay <fixture_dir> <job_dir> --speed 20`.
- The runner copies fixture outputs into the run dir at the times their `artifact` events occurred, and writes the fixture events with fresh `job`, `pid` and `ts` values. It also rewrites `run.dir.run_dir` and the copied `run_state.json` (`events_path`, `pid`, `job`) to the new run.
- On cancel it stops at the next event boundary, writes `run_state.status = "cancelled"` and exits 130. On resume it reads the done set of the last `checkpoint{saved}` event replayed before the cancel, emits `stage.skip{checkpoint}` for those stages, and replays the remaining fixture events.
- The fixture's `FIXTURE.json` gives the `n`/`seed` the project must be created with (SPEC §14.5).

---

## 15. Versioning

- **API v1.** Changes are additive only: new endpoints, new optional fields. A breaking change requires `/api/v2`.
- **Event schema v1.** Every line carries `"v": 1`. Unknown types are stored as `log` and rendered generically. Unknown fields are kept.
- **Stored formats** (§12) carry `"schema"` where they are JSON. Readers accept missing fields, because older runs lack them.

---

## 16. Error codes

| HTTP | `code` | Meaning |
|---|---|---|
| 400 | `bad_host` | Host header not allowlisted |
| 401 | `unauthorized` | Missing or invalid cookie/token |
| 403 | `bad_origin` | Unsafe method without a matching Origin |
| 404 | `not_found` | Generic missing resource |
| 404 | `output_missing` | `detail: {output, produced_by, expected_path}` + `action` |
| 404 | `unknown_kind` / `unknown_view` / `unknown_layer` | Bad identifier |
| 404 | `no_emulator` / `no_climate_factors` / `example_unavailable` | Missing optional input, with `action` |
| 409 | `conflict` | Optimistic concurrency failure (`detail.current_version`) |
| 409 | `conflict_revision` | Scenario has exact results; fork instead |
| 409 | `active` | A job is running on the target |
| 409 | `not_cancellable` / `not_resumable` / `not_ready` | State does not permit the action (`detail` explains) |
| 409 | `no_checkpoint` | Operation needs `checkpoint.pkl` |
| 409 | `engine_memory` | Memory budget/preflight refused (`detail.needed_gb`, `available_gb`, `holders`) |
| 409 | `untrusted_pickle` | Run imported without pickle trust |
| 409 | `preflight_failed` | Launch preflight error (`detail.preflight`) |
| 409 | `grid_mismatch` / `string_ids` / `has_children` / `exists` / `in_use` / `imported_in_place` / `superseded` | Self-explanatory |
| 413 | `too_large` / `too_large_inline` / `too_many_features` | Size limits |
| 415 | `bad_suffix` / `no_conversion` | Upload or conversion type not allowed |
| 422 | `validation` | Body/query invalid (`detail.errors`) |
| 422 | `yaml_error` | YAML parse error (`detail.line`, `column`) |
| 422 | `needs_crs` / `needs_layers` / `needs_responses` / `needs_config` / `requirements` / `template_unavailable` / `mismatch` | Precondition on data/config |
| 500 | `internal` | Unexpected; `detail.trace_id` (logged server-side) |

---

## 17. Server-Sent Events

### 17.1 Global stream (`GET /api/stream`)

Every event's `id` is `g:<gseq>`. Every `data` object includes `gseq` and `ts`.

| `event` | Topic | `data` |
|---|---|---|
| `job.created` | jobs | `{job: Job}` |
| `job.status` | jobs | `{job_id, status, prev_status, exit_code, error, run_id, project_id, kind, label}` |
| `job.progress` | jobs | `{job_id, frac, eta_s, eta_lo, eta_hi, stage, path_tail: string[]}` (≤ 1 Hz per job) |
| `run.updated` | runs | `{run_id, project_id, status, fields: string[]}` |
| `run.indexed` | runs | `{run_id, project_id, origin}` |
| `output.written` | runs | `{run_id, relpath, role, output_id, stage, bytes}` |
| `engine.status` | engine | `{run_id: string\|null, state: EngineState\|EngineHostState, progress: number\|null, rss_mb: number\|null}` |
| `scenario.result` | runs | `{scenario_id: string\|null, result_id, run_id, kind}` |
| `study.updated` | studies | `{study_id, kind, status, project_id, target_run_id}` |
| `storage.low` | storage | `{free_bytes, threshold_bytes}` |
| `resync` | all | `{reason: "expired"\|"overflow"}`. The client refetches active jobs and open resources |
| `server_shutdown` | all | `{stop_jobs: boolean}` |

### 17.2 Per-job stream (`GET /api/jobs/{jid}/stream`)

`id` = byte cursor. `event` = the `type` field. `data` = the full line object (envelope + fields, SPEC §5.3) plus `cursor`.

Envelope:

```ts
{ v: 1; type: string; seq: number; ts: number; t_rel: number; pid: number; job: string; lvl: "debug"|"info"|"warning"|"error";
  span: string|null; parent: string|null; path: string[]; ctx: Record<string, string|number|boolean>; cursor: number }
```

Type-specific fields:

| `type` | Fields |
|---|---|
| `run.start` | `name: string, stages: string[], fast: boolean, coarse: number\|null, resume: boolean, cv_curve: boolean\|null, config_sha256: string, code_sha256: string, run_meta: object` |
| `run.dir` | `run_dir: string, fingerprint: string` |
| `run.plan` | `nodes: PlanNode[], total_units: Record<string, number>, n_points: number\|null` (re-emitted when nodes become cached/skipped or a count becomes known; a nested `run.plan` inside a study task describes that child only) |
| `stage.start` | `stage: StageId, label: string, est_s: number\|null` |
| `stage.end` | `stage: StageId, status: "ok"\|"error"\|"cancelled", elapsed_s: number, summary: object` |
| `stage.skip` | `stage: StageId, reason: string` |
| `task.start` | `name: string, key: string\|null, k: number\|null, n: number\|null, unit: string\|null` (`unit` set ⇒ a successful `task.end` completes one planned unit, SPEC §5.4) |
| `task.end` | `name, key, k, n, unit, status: "ok"\|"error"\|"cancelled", elapsed_s: number, metrics: object, error?: {type, message}` |
| `tick` | `k: number, n: number, unit: string, frac: number, label: string, ...metrics (scalars)`; a `k == n` tick completes one unit of `unit` (engine passes also carry `pass_s`); `k == 1` and `k == n` ticks are never throttled or coalesced |
| `metric` | `name: string, value: number\|string\|boolean\|null, unit: string\|null, tags: object` |
| `artifact` | `path: string, role: string, bytes: number, stage: string\|null` |
| `checkpoint` | `action: "saved"\|"loaded"\|"mismatch", done: string[], bytes: number\|null, elapsed_s: number\|null, fingerprint: string\|null, changed_sections: string[]\|null` |
| `warning` | `code: string, message: string, data: object` |
| `log` | `logger: string, level: string, msg: string` |
| `heartbeat` | `rss_mb: number, cpu_s: number, threads: number` |
| `cancel.requested` | `by: "user"\|"kill"\|"shutdown"` (server-appended) |
| `cancel.ack` | `at_path: string[]` |
| `run.end` | `status: "succeeded"\|"failed"\|"cancelled", elapsed_s: number, timings_s: Record<string, number>, done: string[], error: {type, message, traceback_tail}\|null` |
| `job.status` | `status: JobStatus, exit_code: number\|null, error: object\|null` (server-appended) |
| `job.result` | `result: object` |
| `resource` | `rss_mb, cpu_pct, n_procs, threads` (**transient, no id**) |
| `resync` | `after: number` (stream closes) |
| `end` | `status: JobStatus` (stream closes after drain) |

**Well-known `task.name` values** (the tracker maps these to panels):
- `fold`, `base_model`, `advection_check`, `adv_refit`, `stacker_candidate`, `stacker_fold`, `baseline_model`
- `cv_partition`, `engine_init`, `variable`, `dose`, `marginals`, `scenario`, `model_effects`, `treatment`
- `dml`, `spillover`, `cate`, `dr_curve`, `sensitivity`, `audit`
- `segments`, `allocate`, `closed_loop`, `pareto`
- `lever`, `placebo_kind`, `replicate`, `variant`, `remote_object`
- `load_run`, `unpickle`, `mediators`, `engine_init`

**Well-known metric names** (tags in braces):
- `fit_s`, `heldout_rmse`, `heldout_r2` (in `base_model` task metrics), `candidate_rmse {candidate}`
- `stacker_rmse`, `stacker_r2`, `interval_coverage`, `interval_halfwidth`
- `influence.range_m {predictor}`, `influence.anisotropy_ratio {predictor}`
- `cv_row.r2 {partition}`, `cv_row.rmse {partition}`
- `mean_benefit`, `mean_se`, `frac_extrapolated` (in `dose` task metrics)
- `scenario.mean_delta {scenario}`, `scenario.se {scenario}`, `scenario.frac_extrapolated {scenario}`
- `theta`, `theta_se`
- `planned_total`, `realised_total`
- `share`, `oof_r2`

---

## 18. Review changes (completeness pass, 2026-10-01)

Wire-level changes made with SPEC §18. The `docs` item later appends "As-built changes" after this one.

**Added**
- Endpoints:
  - `POST /api/projects/{pid}/inputs/stations` (+ kind `input.stations`);
  - `GET /api/runs/{rid}/truth`;
  - `GET /api/runs/{rid}/sweeps`, `DELETE /api/sweeps/{swid}`;
  - `GET /api/runs/{rid}/comparisons`, `DELETE /api/comparisons/{cid}`.
- Kind `run.external` (lane `none`, executor `external`).
- Params and fields:
  - `periods`/`baseline` on `input.cmip6`;
  - `target` on `input.features` (replaces `write_open_project`);
  - `threads` on `run.core`;
  - `/api/meta.unit_costs` and `run_tab_outputs`;
  - `MetricKey` format;
  - `PlanPreview.min_dose_dropped_cost`;
  - configured scenarios carry `doc`;
  - `id,lever,value` design CSVs;
  - `run.plan.n_points`;
  - `budget.status` and `causal.model_pd_curve`/`cate_layer` view sections.
- Layer keys: `sc:<slug>` (with the `scenario_slug` rule), `cate_<t>`, `mslope_<t>`, `mslope_own_<t>`.
- On-disk §12.4 (project dir, scenario and findings mirrors); the replay runner's run-dir rewriting and cancel/resume behaviour.

**Changed**
- `POST /api/runs/{rid}/preview` returns a packed body (`delta` + `edited` bitset). `X-SPARC-Edited` was removed.
- `export.*` params carry a server-filled `export_id`, and exports live only under `projects/<slug>/exports/`. They were removed from §12.2.
- `RunDetail` no longer embeds `studies` or `engine`.
- `/api/meta.run_tabs` was replaced by the `RunTabId` vocabulary plus the derived `run_tab_outputs`.
- `web_build` has no `built_utc`.
- `/openapi.json` requires auth; `/docs` and `/redoc` are disabled; `GET /auth?next=` is restricted to same-origin paths.
- `GET /api/projects/{pid}/forcing/stations` never downloads; it returns `404` with a `fetch_input` action instead.
- `state.json` carries `executor`.
- Tick coalescing never drops `k == 1` or `k == n` ticks.

## 19. As-built changes

The `docs` item extends this section when the spec is brought up to date. Entries are grouped by milestone.

### M1–M2 (browse, run & track)

- `RunSummary.studies` lists the ids of the studies **attached** to the run (`study_links.attached = 1`), in the runs endpoints and the project endpoints alike.
- Stages a finished run did not compute carry the remedy `{kind: "open", label: "Re-run with <the stage>", method: "GET", path: "/p/{pid}/launch?from=<run_id>"}` (Status Board cells and missing outputs). A resume cannot add them, because it replays the run's own launch snapshot. Launch reads `?from=<run_id>` and prefills mode, coarse cell, stages, CV curve and threads from that run's `launch.args` (`GET /api/runs/{rid}` → `launch`); query parameters already in the URL win.
- `overview.outputs_grid` lists every output the run has or should have, plus one row per post-run action that has not run and per stage the run skipped (its primary output, carrying the remedy). Optional files a planner pack or a study would add are left out.
- `overview.studies` always has eight chips, in this order: `baselines, planner, emulator, uncertainty, placebo, multiverse, simcheck, reproduce`. Each is `{kind, state: "done"|"running"|"failed"|"stale"|"not_run", headline: string|null, study_id: string|null}`, built from the run's Status Board cells. `headline` comes from the study summary's `headline`, else from the manifest or the summary (`placebo`: `n_placebos`, `n_pass_model`, `n_pass_causal`; `multiverse`: `sign_stability_min`; `simcheck`: `bias_correction.share_range`; `reproduce`: `pass`).
- `uncertainty.sources` rows are `{kind, label, study_id, attached, state}`. The folders listed in `uncertainty.json` `sources` (multiverse dir, simcheck dirs) are matched to the studies index by `studies.out_dir`, so that column must stay the study folder core writes into those sources. The project's other multiverse, simcheck and placebo studies follow. A study's `summary.label` names its row when present.
- Status Board cells have no `queued` state: a post-run job or study still waiting (e.g. in a launch's `then` chain) is a `running` cell with `reason: "queued"` (or `"blocked"`), and the board shows it as queued.
- An imported run with a manifest but no `run_state.json` gets `created_utc` = the manifest's `created_utc` (written when the run ended) less the sum of its `timings_s`. Its id keeps the manifest time.
- The runs routes return `503 not_ready` while the runs registry is still starting, as the jobs and inputs routes do while the server starts.
- A job with a cancel request (status `cancelling`, or its `cancel` file present) whose worker exits with `-15`/`143` ends `cancelled`, not `failed`. This covers SIGTERM arriving before the worker or the replay runner installed its handlers.
- `POST /api/projects/{pid}/data/check`: `config_patch` is a JSON Merge Patch, and the preview carries the `planner.layers` columns (§5.2).

### M3 (Scenario Lab)

- `POST /api/runs/{rid}/engine/evict` does what `DELETE /api/runs/{rid}/engine` does (the body is ignored), so an `Action` (POST/GET only) can evict a run. The `engine_memory` refusal's action uses it.
- `GET /api/runs/{rid}/engine` also reports `loading`, with `job_id` set to the exact request's job (`engine.scenario`, `engine.batch`, …, not only `engine.open`), while that request is loading a cold run. The Lab shows it as the same loading state.
- `PreviewRequest` has an optional `scenario_id`, and the Lab sends it once the draft is saved. It marks a `draft` scenario `previewed`. The Lab previews an edit (120 ms) before it autosaves it (2 s), so a save whose `{edits, options}` equal the scenario's latest preview keeps `previewed`. Any other content change makes it a `draft` again. The server keeps this in memory, so a restart only loses the status hint.
- `409 superseded` on `POST /api/runs/{rid}/preview` applies only while a newer `request_seq` for the run is in flight or waiting. Once the run's previews are idle, any `request_seq` is accepted again, so a reloaded page whose sequence restarts at 1 is not locked out.
- `POST /api/scenarios/{sid}/run` always returns `{cached, job}`, with the unused one `null`: `200` on a cache hit, `202` with the job otherwise.
- A cache hit on `POST /api/scenarios/{sid}/run` can be a result computed for **another** scenario with the same content (the cache key is content, run, checkpoint and code). That result is copied to a new result of `sid`, and `cached.id` is the copy, with `cached.scenario_id = sid`. The scenario then lists, inspects, compares and packs it as its own, and keeps it if the other scenario is deleted. The scenario's own result is preferred when there is one.
- Exact requests on a run that is not loaded run the engine memory preflight and can return `409 engine_memory`. These are scenario run, run-batch, ladder, configured rerun-exact, sweeps and plan verify/frontier. A cache hit is exempt. `POST /api/runs/{rid}/plans` keeps the plan and returns `job: null` when the verify job is refused.
- `DELETE /api/plans/{plid}` and `DELETE /api/scenarios/{sid}` return `409 active` while a job on them is running, as `DELETE /api/sweeps/{swid}` does.
- `POST /api/runs/{rid}/climate/explore` returns `present` as §7.7 says: `summarize_projections`' dict `{mean, share_at_or_above}` for today. It also returns `statistic` (the warming statistic asked for), and each `projections[].warming.selected` holds that statistic's warming, which drives the client-side future maps.
- Compare sign convention: `pairs[].city`, `pairs[].regions` and the `cmp:<cid>:<a>__<b>` difference layer are **A − B** (item `a` minus item `b`). A negative value means A cools more than B.
- The Lab's Compare page `/r/:rid/lab/compare` accepts `?cid=<comparison id>` next to `?items=`. `items` are encoded `res:<id>`, `configured:<slug>`, `plan:<id>` or `baseline`.
- `Project.last_run.has_checkpoint` (§5) is true once the latest run's `checkpoint.pkl` exists, which can happen while the run is still going (after S3). The project nav's Scenario Lab entry uses it to stay disabled until the run it would open has a checkpoint (SPEC §3.1).
- Engine host transport (§13): when `<ws>/engine/host.sock` is longer than `AF_UNIX` allows (about 100 bytes, as with a deeply nested workspace), the host binds a short per-workspace socket in the temp directory (`sparc-engine-<uid>-<hash>.sock`, owner-only). `host.json` `sock` is the path actually bound, and clients connect to that.
- Packs and `POST /api/exports` (contract for `[S]`, M4): the route inserts the `exports` row **before** it queues `export.decision_pack`, `export.plan_pack` or `export.compare_pack`, passing `export_id` plus `result_id`, `plan_id` or `comparison_id`. The kind's `on_finish` (`pack_on_finish`) completes that row with `status`, `path` and `bytes`, and records the pack's `draft` flag in `options_json.draft`. `Export.draft` is read from there. Until `[S]` lands, `POST /api/exports` has no route: the server answers `405` with code `not_found`, and the Lab's Decision pack, Plan pack and Compare pack buttons show that error. The pack kinds themselves work when submitted through `POST /api/jobs` with an `export_id`.
- `build_emulator` actions point at `POST /api/runs/{rid}/actions/emulator` (`post.emulator`, `[S]`, M4). Until then the Lab offers the action, the preview reports `404 no_emulator`, and exact runs work.
- `sparc/core/session.py` is a top-level core module, so it is part of the core code digest (SPEC §11). Checkpoints written before it was added report `code_match: false`, and their exact results are marked stale (`stale: true`). This is the documented consequence of adding a core module, not a fault in the run.
