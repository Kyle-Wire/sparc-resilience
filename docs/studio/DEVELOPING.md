# Developing SPARC Studio

This is the guide for people who change Studio itself. Users want [`USER_GUIDE.md`](USER_GUIDE.md). The behaviour is specified in [`SPEC.md`](SPEC.md) and the wire contract in [`api.md`](api.md) (api.md wins on wire formats); both end with an **As-built changes** section that records where the implementation differs from the original text. Keep those sections current when you change behaviour or the wire.

Commands in fenced blocks that start with `# runnable` are executed by `python docs/studio/doctest.py` in CI against a temporary workspace, and every other command is checked statically, so keep them correct ([§8](#documentation-checks)).

**Contents**

1. [Repository layout](#1-repository-layout)
2. [Processes and data flow](#2-processes-and-data-flow)
3. [Registries: how features plug in](#3-registries-how-features-plug-in)
4. [Adding a job kind](#4-adding-a-job-kind)
5. [Adding a page or a run tab](#5-adding-a-page-or-a-run-tab)
6. [Events and the progress API](#6-events-and-the-progress-api)
7. [Dev loop](#7-dev-loop)
8. [Regenerating fixtures, API types, built assets and docs checks](#8-regenerating-fixtures-api-types-built-assets-and-docs-checks)
9. [Tests: matrix and markers](#9-tests-matrix-and-markers)
10. [Rules of the road](#10-rules-of-the-road)

---

## 1. Repository layout

```text
sparc/core/                 the pipeline (S0–S7, studies). Studio-related modules:
  progress.py  runio.py       structured progress events; atomic writes, manifest updates, run locks
  pipeline.py                 run_core (run_dir, run_meta, run_state.json, checkpoint.json), plan_stages,
                              checkpoint_status, fingerprint_sections, STAGE_NODES, CHECKPOINT_KEY
  catalog.py                  the output catalog (OutputSpec per file, view, IGNORED globs, scenario_slug)
  session.py                  open_run / config_for_run: a run's fitted models in memory (the Lab engine)
  results_page/               the standalone results page builder (template.html is package data)
  synthetic.py                write_demo_project: the one generator of the demo city and the fixtures
sparc/studio/               the server (FastAPI). Never imports torch.
  cli.py  app.py  settings.py  security.py  errors.py  db.py  workspace.py  events.py  sse.py  meta.py
  schemas/common.py           shared wire models (Job, Page, ErrorEnvelope, LayerMeta, GridMeta, …)
  routes/                     one module per route group, listed in routes/__init__.py
  jobs/                       job kinds registry, JobManager, executors, worker, tailer, tracker, ETA,
                              resources, self-test kinds, replay runner
  projects/  runs/  engine/  scenarios/  studies/  exports/     feature packages
  examples/providence/        the packaged Providence inputs
  static/                     the committed production build of studio-web (index.html, assets/, BUILD_INFO.json)
studio-web/                 the web app (React 19, TypeScript strict, Vite). Source only; builds into sparc/studio/static
  src/router.ts  src/layouts/ src/api/ src/stores/ src/theme/ src/components/ui/ src/charts/ src/map/
  src/pages/<group>/          feature pages; each group declares its routes in routes.ts
  scripts/gen-api.mjs         OpenAPI → src/api/schema.gen.ts
tests/core/                 pipeline tests (no FastAPI, no Node)
tests/studio/               server tests by group (foundation, projects, runs, engine, studies), the OpenAPI
                            snapshot and reducer contract tests, fixtures/, and e2e/ (Playwright)
scripts/                    make_studio_fixtures.py, check_studio_assets.py, results_page/ (shims)
docs/studio/                SPEC.md, api.md, these guides, doctest.py, check_api_doc.py, img/
.github/workflows/studio.yml  Studio CI (every push) and the nightly job
```

The legacy app (`sparc-desktop/`, `sparc/server/`) is **not** part of Studio: Studio never imports it, and CI fails if a Studio change touches it.

## 2. Processes and data flow

```text
browser (React SPA: ≤ 2 SSE connections per tab)
   │  HTTP JSON · binary Float32 arrays · SSE
   ▼
sparc studio  (uvicorn + FastAPI, one asyncio process; starts in about a second; never imports torch)
   ├─ EventHub, JobManager scheduler, JobTailers, ResourceSampler, engine client      (asyncio tasks)
   ├─ thread pool: parquet/JSON reads, selections, preview (threadpool_limits(1)), planned optimiser, …
   ├─ job workers:  python -m sparc.studio.jobs.worker <job_dir>   (own process group each)
   │     └─ may fork process pools (simcheck, multiverse); progress.init_worker propagates the event sink
   └─ engine host:  python -m sparc.studio.engine.host              (own session; long-lived; LRU of loaded runs)
```

- **Workers never write SQLite.** A worker writes `jobs/<jid>/events.jsonl` (progress events), `stdout.log`, `stderr.log` and `result.json`. The server's tailer reads the event log, folds it into a projection (`jobs/tracker.py`), stores spans, metrics, warnings, artifacts and checkpoints in SQLite, publishes `job.progress` on the global stream, and calls the kind's server-side hooks.
- **The run directory is the source of truth.** Studio indexes runs in SQLite (`runs/registry.py`) and owns only `<run_dir>/studio/`. Pipeline files change only through core functions; post-run actions take the per-run lock (`runio.run_lock`).
- **Two streams.** `GET /api/stream` (global: jobs, runs, engine, studies, storage) and `GET /api/jobs/{jid}/stream` (one job's events, replayed from the file then tailed, byte-offset cursors, `Last-Event-ID` resume). A page opened mid-run fetches `GET /api/jobs/{jid}/tracker` (snapshot plus cursor), replays the event log to rebuild exact progress, then follows the stream from the cursor.
- **Binary grids.** Per-cell values travel as little-endian Float32 arrays in run row order (`grid.bin` packs several arrays with an `X-SPARC-Offsets` header) and are drawn on a canvas at one pixel per cell. There is no map library and no per-cell GeoJSON on the wire.

## 3. Registries: how features plug in

Shared files (app factory, registries, DB schema, router, layouts, API client, design system, chart and map kits) belong to the foundation. Feature packages plug in through registries and never edit them.

| Registry | Where | How a feature joins |
|---|---|---|
| Route modules | `sparc/studio/routes/__init__.py` `ROUTER_MODULES` | Create `sparc/studio/routes/<name>.py` with a `router = APIRouter()`; `create_app` mounts it under `/api` (and an optional `root_router` at `/`). A module that does not exist yet is skipped; any other import error is raised |
| Job kinds | `sparc/studio/jobs/kinds.py` `KIND_MODULES` and `@job_kind` | Decorate the job function in your package's `kinds.py` ([§4](#4-adding-a-job-kind)) |
| Executors | `sparc/studio/jobs/executors.py` `register_executor(name, executor)` | `process` (a worker subprocess, the default) and `engine` (requests to the engine host, registered by `sparc/studio/engine/executor.py`); `external` is the read-only pseudo-job of a live CLI run |
| Output catalog | `sparc/core/catalog.py` `OUTPUTS` | One `OutputSpec` per output file: id, label, group, files, producing stage, `view` (a run tab id or a viewer kind), formats, manifest key. Served as `/api/meta.output_catalog`; a test checks every file of the fixture runs matches a spec |
| Page routes | `studio-web/src/pages/<group>/routes.ts` | Export `routes: RouteDef[]`; the router collects every group with `import.meta.glob` ([§5](#5-adding-a-page-or-a-run-tab)) |
| Run tabs | `RouteDef.runTab` | Declared by the page that owns the tab; ids come from a fixed vocabulary |
| Project nav | `RouteDef.projectNav` | Declared only by the item that owns the target page |
| Reindex | `sparc.studio.db.reindex_hook` | Rebuild your tables from your files on `sparc studio --reindex` |

**Cross-package calls** go through lazy function contracts, imported with `importlib` when needed, so packages never import each other at module level:

| Caller | Callee | Signature |
|---|---|---|
| projects (Providence import, `POST /api/projects/import`) | `sparc.studio.runs.registry` | `import_run(dir, project_id, config_path=None, trust_pickles=False)` |
| projects | `sparc.studio.studies.service` | `import_study_dir(dir, project_id, kind=None, target_run_id=None)` |
| engine, studies (narratives, packs, reports) | `sparc.studio.runs.caveats` | `caveats_for(run_ctx) -> list[str]` |
| engine, studies | `sparc.core.catalog` | `scenario_slug(name)` |
| runs (Status Board study cells) | the `studies`, `study_links` and `jobs` tables | DB contract only |

## 4. Adding a job kind

A job kind is a function that runs in a worker process (or in the engine host), registered with `@job_kind` in a module listed in `KIND_MODULES` (`sparc.studio.{projects,runs,engine,studies,exports}.kinds`). The real `post.writeup` is a compact example:

```python
from pydantic import BaseModel, ConfigDict

from sparc.studio.jobs.kinds import job_kind


class WriteupParams(BaseModel):
    model_config = ConfigDict(extra="forbid")     # params are validated; unknown keys are refused


@job_kind("post.writeup", lane="medium", label="Methods & model card", params=WriteupParams, needs_run=True,
          locks_run=True, on_finish=_post_on_finish, estimate=_post_estimate("writeup"))
def post_writeup(ctx, params: WriteupParams) -> dict:
    from sparc.core import progress, runio                    # heavy imports inside the function
    from sparc.core.writeup import methods_markdown, model_card_markdown

    run_dir = Path(ctx.run_dir)
    m = merged_manifest(ctx)
    files = []
    for name, text in (("methods.md", methods_markdown(m)), ("model_card.md", model_card_markdown(m))):
        runio.write_text_atomic(run_dir / name, text)       # atomic: a cancelled job never leaves half a file
        progress.artifact(run_dir / name, role="docs")      # tells the tracker (and the Outputs tab) about it
        files.append(name)
    return {"files": files}                                 # becomes result.json and Job.result
```

What the decorator takes:

| Argument | Meaning |
|---|---|
| `lane` | `heavy` (1 slot), `medium` (1), `network` (2), `engine` (1, the engine host), or `none` (pseudo-jobs) |
| `executor` | `process` (default), `engine` or `external` |
| `params` | A pydantic model; `POST /api/jobs` and the feature routes validate against it, and `/api/meta.job_kinds` serves its JSON schema |
| `needs_run`, `needs_checkpoint` | Checked when the job is queued (`409 precondition` otherwise) |
| `locks_run` | Serialise with other run-mutating jobs on the same run (per-run lock row plus the core file lock) |
| `network_hosts` | Hosts the job contacts (shown in the UI, checked by the network check; hidden in offline mode) |
| `estimate(sctx, job, params)` | `{units?, n_cells?, est_s?, est_lo?, est_hi?, peak_ram_gb?, disk_bytes?}` for the launch forms and the memory/disk preflight |
| `preflight(sctx, job, params)` | Extra start-time checks: `[{reason, actions?, fatal?}]` → the job is `blocked` with actions, or fails |
| `on_event(sctx, job, event)`, `on_event_types` | Server-side hook for selected event types (default `run.dir`, `run.start`, `artifact`), e.g. registering a study's child runs live |
| `on_finish(sctx, job, result)` | Server-side hook after the final status: insert `results`/`plans`/`exports` rows, attach studies, enqueue `post.uncertainty` |
| `retry_params(sctx, job)` | The params `POST /api/jobs/{jid}/retry` uses (default: the same params) |
| `threads(settings, params)` | The thread count the job gets (default: by lane, SPEC §10.5) |
| `long` | Listed in `/api/meta.job_kinds`, so clients can tell long-running kinds apart |

The job function receives a `JobContext`: `job_id`, `job_dir`, `workspace`, `project_dir`, `run_dir`, `studio_dir`, `study_id`, `study_dir`, `threads`, `cache_dir`, a read-only `db` helper, `run_config()` (the run's **launch-snapshot** config, never the current project config) and `emit_result(dict)`.

Rules:
- **Import lazily.** The server imports every kind module to list kinds and their schemas, so anything that may pull torch (sessions, engines, models) is imported inside the function.
- **Hooks are fast and idempotent** (< 50 ms; a reattach may replay them). They run in a worker thread of the server, with the server context in `sctx` (`db`, `hub`, `workspace`, `jobs`, `services`).
- **Emit progress** with `sparc.core.progress` ([§6](#6-events-and-the-progress-api)): the worker has already configured the sink, the job id and the cancel file. Call `progress.check_cancel()` at safe points; `Cancelled` is a `BaseException`, so it passes through `except Exception`.
- **Write atomically** with `sparc.core.runio` and announce files with `progress.artifact`.
- Add a test in your package's test folder that runs the kind with the core functions faked (see `tests/studio/studies/fakes.py` and `fake_worker.py` for the pattern), and document the params and result in api.md §8.

The self-test kinds in `sparc/studio/jobs/testkinds.py` (`test.sleep`, `test.events`, `test.fail`, `test.ignore_sigterm`, `test.pool`, `test.lock`), enabled with `SPARC_STUDIO_TEST_KINDS=1`, are handy for exercising the job machinery without the pipeline.

## 5. Adding a page or a run tab

Every page group has a `routes.ts`:

```ts
import { lazy } from "react";
import type { RouteDef } from "../../router";

export const routes: RouteDef[] = [
  {
    path: "/r/:rid/validation",
    component: lazy(() => import("./Validation")),     // code-split per page
    title: "Validation",
    runTab: { id: "validation", label: "Validation", group: "Trust", order: 10 },
  },
  {
    path: "/p/:pid/studies",
    component: lazy(() => import("./StudiesHub")),
    title: "Studies",
    projectNav: { id: "studies", label: "Studies", order: 70, to: (pid) => `/p/${encodeURIComponent(pid)}/studies` },
  },
  { path: "/studies/:stid", component: lazy(() => import("./StudyPage")), title: "Study" },
];
```

- A new **page** needs only a `RouteDef` (and its component). `fullWidth: true` drops the page margins (maps, Mission Control, the Lab). The router logs a console warning (`[routes] …`) for a duplicate route, run tab or nav entry and for an unknown tab id or group.
- A new **project nav entry** is a `projectNav` on the route that owns its target. `to` returning `null` hides it; `disabledReason(project)` greys it out with a tooltip. Orders in use: Overview 10, Setup 20, Inputs 30, Launch 40, Runs 50, Scenario Lab 60, Studies 70, Exports 80, Findings 90.
- A new **run tab** is a `runTab` on its route, but tab ids are a fixed vocabulary shared with the server, because the server computes each tab's status dot. Adding one means: add the id to `RUN_TAB_IDS` in `sparc/studio/meta.py` and `sparc/studio/runs/outputs.py` and in `studio-web/src/api/types.ts`; point the relevant `OutputSpec.view` entries in `sparc/core/catalog.py` at it (or give it a fixed rule in `runs/outputs.py`, as `track`, `map`, `files`, `provenance`, `lab` and `validation` have); update the OpenAPI snapshot and the generated types ([§8](#8-regenerating-fixtures-api-types-built-assets-and-docs-checks)); and document it in SPEC §3.2. Groups are Model · Effects · Decisions · Trust · Run; orders inside a group are spaced by 10.

Data for pages comes through `useResource(key, fetcher, {immutable})` (`src/api/resource.ts`), whose tags the global stream invalidates (`run:<rid>`, `job:<jid>`, `scenario:<sid>`, `project:<pid>`). Finished-run resources are immutable and never refetched. Per-group endpoint functions and types live in `src/api/<group>.ts`; `src/api/contract.check.ts` (type-only) checks that they agree with the generated `schema.gen.ts`, so `npm run typecheck` fails when the server and the client drift.

## 6. Events and the progress API

### Emitting

`sparc/core/progress.py` is the only way pipeline code reports progress. Every call returns at once when no sink is configured, so instrumented code costs nothing in plain library use. A worker configures it from `SPARC_PROGRESS` (the event file), `SPARC_PROGRESS_LEVEL`, `SPARC_JOB_ID` and `SPARC_CANCEL_FILE`; `sparc core run --progress PATH --job-id J` does the same for CLI runs.

```python
from sparc.core import progress

with progress.stage("S4", label="Response surfaces"):           # stage.start … stage.end {status, elapsed_s}
    for i, lever in enumerate(levers, 1):
        with progress.task("lever", key=lever, k=i, n=len(levers)):
            for j, dose in enumerate(doses, 1):
                progress.check_cancel()                           # a safe point: raises Cancelled when asked
                fit(dose)
                progress.tick(j, len(doses), unit="engine_pass")  # throttled to 1/s, but k == 1 and k == n always go out
            progress.metric("response.max_effect", effect, unit="°F", lever=lever)   # tags → MetricKey "name{lever=…}"
        progress.artifact(run_dir / f"response_{lever}.parquet", role="response")
if few_cells:
    progress.warn("influence.few_cells", "fewer than 200 cells in range", predictor=name)
```

Also: `progress.skip(stage, reason)`, `progress.checkpoint(action, done=…, bytes=…)`, `progress.context(**kv)` (merged into every nested event, e.g. the CV partition or the multiverse variant), `progress.run_dir_scope(run_dir)`, `progress.limit_threads(n)` / `set_threads(n)`, `progress.wrap_context(fn)` for thread pools and `ProcessPoolExecutor(initializer=progress.init_worker, initargs=(progress.worker_env(),))` for process pools. `logging` records of the `sparc` logger are forwarded as `log` events (warnings also as `warning` events with code `log.<logger>`).

### The envelope and event types

Each event is one JSON line of at most 4 KB, written with a single `os.write` (so pool workers never interleave). The envelope:

| Field | Meaning |
|---|---|
| `v` | schema version (1) |
| `type` | event type |
| `seq`, `ts`, `t_rel` | per-process sequence, unix time, seconds since the sink was configured |
| `pid`, `job` | process, job id (`""` when none was configured) |
| `lvl` | `debug`, `info`, `warning`, `error` |
| `span`, `parent` | span ids `"<pid>:<n>"` |
| `path` | span ancestry, e.g. `["run:providence_uhi","stage:cv_curve","task:fold[4/5]"]` — except on `artifact` events, where `path` is the run-relative file and the ancestry is in `span_path` |
| `ctx` | the merged `progress.context` |

Types: `run.start`, `run.dir`, `run.plan`, `stage.start`, `stage.end`, `stage.skip`, `task.start`, `task.end`, `tick`, `metric`, `artifact`, `checkpoint`, `warning`, `log`, `heartbeat`, `cancel.ack`, `run.end` (from core); `cancel.requested` and `job.status` (appended by the server); `job.result` (the worker). The full field list is in SPEC §5.3 and api.md §17; `GET /api/meta/event-schema` serves the JSON Schema. A field that would clash with an envelope key is written as `<key>_`.

Real lines from the committed fixture run (`tests/studio/fixtures/synth_run/events.jsonl`):

```json
{"v":1,"type":"stage.start","seq":4,"ts":1790889678.197,"t_rel":0.024,"pid":4557,"job":"j_fixture","lvl":"info","span":"4557:2","parent":"4557:1","path":["run:synthetic_demo","stage:S0"],"ctx":{},"stage":"S0","label":"Data and QA","est_s":null}
{"v":1,"type":"task.end","seq":36,"ts":1790889680.225,"t_rel":2.052,"pid":4557,"job":"j_fixture","lvl":"info","span":"4557:6","parent":"4557:5","path":["run:synthetic_demo","stage:S2_S3","task:fold[1/3]","task:base_model[ols]"],"ctx":{},"name":"base_model","key":"ols","k":null,"n":null,"unit":"base_fit:ols","status":"ok","elapsed_s":0.0188,"metrics":{"fit_s":0.019,"heldout_rmse":1.3937025717870912,"heldout_r2":-1.9482167761071572}}
{"v":1,"type":"artifact","seq":32,"ts":1790889680.185,"t_rel":2.012,"pid":4557,"job":"j_fixture","lvl":"info","span":"4557:1","parent":null,"path":"influence.json","ctx":{},"role":"influence","bytes":14498,"stage":"S1","span_path":["run:synthetic_demo"]}
{"v":1,"type":"checkpoint","seq":146,"ts":1790889695.611,"t_rel":17.438,"pid":4557,"job":"j_fixture","lvl":"info","span":"4557:4","parent":"4557:1","path":["run:synthetic_demo","stage:S2_S3"],"ctx":{},"action":"saved","done":["S3"],"bytes":3497666,"elapsed_s":0.199,"fingerprint":"68c17c7b36fce387","changed_sections":null}
{"v":1,"type":"stage.skip","seq":180,"ts":1790889697.23,"t_rel":19.057,"pid":4557,"job":"j_fixture","lvl":"info","span":"4557:1","parent":null,"path":["run:synthetic_demo"],"ctx":{},"stage":"cv_curve","reason":"disabled_by_config:cv.distance_curve.enabled"}
{"v":1,"type":"run.end","seq":455,"ts":1790889709.646,"t_rel":31.474,"pid":4557,"job":"j_fixture","lvl":"info","span":"4557:1","parent":null,"path":["run:synthetic_demo"],"ctx":{},"status":"succeeded","elapsed_s":31.452,"timings_s":{"S0":0.02,"S1":1.94,"S2_S3":15.43,"baselines":1.61,"S4":5.47,"S5":1.5,"S6":5.14,"S7":0.19},"done":["S3","S4","S5","S6","baselines","climate"],"error":null}
```

### Plan, units and progress

`plan_stages(cfg, …)` (core) returns the run's plan without running it: one node per stage with its state (`will_run`, `skipped` with a reason, `cached`) and its planned **units** (`base_fit:<model>`, `engine_pass`, `causal_step:<step>`, `checkpoint_save`, …). It shares its gating predicates with `run_core`, so the plan cannot drift from what runs; a test asserts plan == emitted events. The run re-emits `run.plan` when counts become known.

A unit is completed by a `task.end{status: ok}` with that `unit`, by a `tick` with that `unit` and `k == n`, or (S0, S1, checkpoint saves) by the stage end or `checkpoint{saved}`. Ticks with `k < n` add fractional progress. Progress is weighted by the seed unit costs of `/api/meta.unit_costs` (SPEC §5.4), identically in the Python projection (`sparc/studio/jobs/tracker.py`) and the TypeScript reducer (`studio-web/src/stores/tracker.ts`); `tests/studio/test_reducer_contract.py` folds the fixture through both and compares them event by event. ETA seconds use the per-host calibrated cost model in `sparc/studio/jobs/eta.py`.

### Reading progress over HTTP

| Endpoint | Use |
|---|---|
| `GET /api/jobs/{jid}` | The job row: status, cost-weighted `progress`, `eta_s`/`eta_lo`/`eta_hi`, current stage and path |
| `GET /api/jobs/{jid}/tracker` | The projection: plan, stage states, spans, latest metrics and series, warnings, artifacts, checkpoints, resources, children, plus the `cursor` to stream from |
| `GET /api/jobs/{jid}/stream?after=<cursor>` | SSE: every event with `id` = its byte cursor; ends with `event: end` |
| `GET /api/jobs/{jid}/events?after=&types=&limit=` | The same events, paged (the polling fallback). Omit `after` to start at the first line |
| `GET /api/jobs/{jid}/logs`, `/metrics`, `/warnings`, `/resources`, `/spans` | Filtered slices |
| `GET /api/stream` | Global SSE: `job.status`, `job.progress`, run, engine, study and storage events |

## 7. Dev loop

Install the package with its development extras, and the web app's dependencies once:

```bash
pip install -e ".[studio,dev]" pytest-timeout playwright
npm --prefix studio-web ci
```

Run the server and the Vite dev server side by side. The dev page exchanges `VITE_STUDIO_TOKEN` at `/auth` once, and Vite proxies `/api`, `/auth` and `/openapi.json` (SSE included) to the server at `SPARC_STUDIO_URL` (default `http://127.0.0.1:8765`):

```bash
sparc studio --no-browser --token dev --dev-origin http://localhost:5173
VITE_STUDIO_TOKEN=dev npm --prefix studio-web run dev        # in a second terminal; open http://localhost:5173/
```

Python changes need a server restart (jobs keep running across restarts); web changes reload in the browser.

Useful switches:

| Variable | Effect |
|---|---|
| `SPARC_STUDIO_HOME` | The default workspace (use a scratch one while developing) |
| `SPARC_STUDIO_RUNNER=replay:tests/studio/fixtures/synth_run` | `run.core` jobs replay the recorded fixture run instead of fitting models: deterministic and quick. Create the demo with `n=40, seed=0` so its data matches the fixture |
| `SPARC_STUDIO_REPLAY_SPEED` | Replay time compression (the e2e replay uses 1 so it can reload mid-run; 20 finishes the 31 s fixture in about 2 s) |
| `SPARC_STUDIO_TEST_KINDS=1` | Register the `test.*` job kinds |
| `SPARC_STUDIO_ENGINE_SLACK_GB` | Memory slack added to the engine's memory estimate (default 1.0) |

Scripts and tests talk to the API with `Authorization: Bearer <token>` (no `Origin` needed). This session drives a replay run end to end over HTTP, the way the tests do:

```bash
# runnable
# A server whose run.core jobs replay the committed fixture run, 20 times faster than it was recorded.
export SPARC_STUDIO_RUNNER="replay:tests/studio/fixtures/synth_run" SPARC_STUDIO_REPLAY_SPEED=20
WS="${SPARC_STUDIO_HOME:-$HOME/sparc-studio}"
python -m sparc.studio --no-browser --port 0 --token dev &
for i in $(seq 1 90); do
  URL=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["url"])' "$WS/studio.lock.json" 2>/dev/null) \
    && curl -sf -H "Authorization: Bearer dev" "$URL/api/runs" > /dev/null && break
  sleep 1
done
python - "$URL" <<'PY'
import json, sys, time, urllib.request

base = sys.argv[1]

def call(method, path, body=None):
    req = urllib.request.Request(base + path, method=method, headers={"Authorization": "Bearer dev"})
    if body is not None:
        req.data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read() or b"null")

# the fixture was recorded on the demo city with n = 40, seed = 0 (tests/studio/fixtures/synth_run/FIXTURE.json)
project = call("POST", "/api/projects", {"name": "Replay city", "template": "synthetic_demo",
                                          "options": {"n": 40, "seed": 0}})["project"]
plan = call("POST", f"/api/projects/{project['id']}/runs/plan", {"mode": "fast"})
print("plan:", ", ".join(f"{n['id']} {n['state']}" for n in plan["nodes"]))
launch = call("POST", f"/api/projects/{project['id']}/runs", {"mode": "fast", "label": "replayed"})
job = launch["job"]
while job["status"] not in ("succeeded", "failed", "cancelled", "interrupted"):
    time.sleep(0.5)
    job = call("GET", f"/api/jobs/{job['id']}")
assert job["status"] == "succeeded", job
tracker = call("GET", f"/api/jobs/{job['id']}/tracker")
print("stages:", {k: v["state"] for k, v in tracker["stages"].items()})
tabs = call("GET", f"/api/runs/{launch['run']['id']}/outputs")["tabs"]
print("run tabs:", ", ".join(f"{t['id']}={t['availability']}" for t in tabs))
PY
curl -fsS -X POST -H "Authorization: Bearer dev" -H "Content-Type: application/json" -d '{"stop_jobs": true}' "$URL/api/shutdown"; echo
wait
```

## 8. Regenerating fixtures, API types, built assets and docs checks

### The OpenAPI document and the generated API types

The pydantic models are the one contract source. Whenever a route, a method or a wire model changes:

```bash
python tests/studio/test_openapi_snapshot.py --update        # refresh tests/studio/openapi.snapshot.json
npm --prefix studio-web run gen:api                          # regenerate studio-web/src/api/schema.gen.ts
npm --prefix studio-web run typecheck                        # contract.check.ts against the new types
```

then update api.md (and its As-built section) and run the route check below. `python -m sparc.studio --dump-openapi PATH` writes the same document without starting a server:

```bash
# runnable
python -m sparc.studio --dump-openapi "${TMPDIR:-/tmp}/openapi.json"
python docs/studio/check_api_doc.py --openapi "${TMPDIR:-/tmp}/openapi.json"
```

The generated types must match the committed ones (CI regenerates and diffs them):

```bash
# runnable (needs: node)
npm --prefix studio-web run gen:api
git diff --exit-code studio-web/src/api/schema.gen.ts
```

### The committed web build

`sparc/studio/static/` is committed so that pip users need no Node. After any change under `studio-web/`:

```bash
npm --prefix studio-web run build                 # writes sparc/studio/static (index.html, hashed assets, BUILD_INFO.json)
python scripts/check_studio_assets.py             # BUILD_INFO.src_sha256 matches the sources; assets complete
```

`BUILD_INFO.json` holds a hash of the sources and the Vite and React versions, but no timestamp, so rebuilding unchanged sources is byte-identical and CI can diff the committed build against a fresh one. The check is quick:

```bash
# runnable
python scripts/check_studio_assets.py
```

### Fixtures

| Fixture | Regenerate with | Used by |
|---|---|---|
| `tests/studio/fixtures/synth_run/` (a recorded fast run of the n = 40 demo, without its checkpoint, plus `events.jsonl` and `FIXTURE.json`) | `python scripts/make_studio_fixtures.py` (one thread; deterministic) | the replay runner and e2e, reader and catalog tests, the reducer contract |
| `tests/studio/fixtures/reducer_projection.golden.json` | `python tests/studio/test_reducer_contract.py --update` | the cross-language reducer contract |
| `tests/studio/fixtures/selftest_events.jsonl` | `python -m sparc.studio.jobs.testkinds --write-fixture tests/studio/fixtures/selftest_events.jsonl` | the tracker and ETA tests; update `selftest_projection.golden.json` with it on purpose |
| `tests/studio/openapi.snapshot.json` | `python tests/studio/test_openapi_snapshot.py --update` | the snapshot test, `gen:api` |

The pipeline's code digest covers every top-level `sparc/core/*.py` except `progress.py` and `runio.py`, so editing a core module changes the fixture's recorded `code_sha256`; regenerate the fixture when a test depends on it.

### Documentation checks

```bash
# runnable
python docs/studio/check_api_doc.py              # api.md defines every route of the OpenAPI document, and no others
python docs/studio/doctest.py --lint-only        # every command in the guides parses and names real files
```

The static check parses `sparc studio …` and `sparc core <command> …` with the real argument parsers (a `--project` config must exist), checks that scripts, test paths, npm scripts and `sparc` modules exist, and compiles `python -c` snippets and checks the names they import from `sparc`; `<placeholders>` such as `<run dir>` stand for any value. `python docs/studio/doctest.py` (no arguments) additionally runs every `# runnable` block of `USER_GUIDE.md`, `DEVELOPING.md`, `RELEASE_NOTES.md` and the README against a fresh temporary workspace, with this checkout first on `PYTHONPATH`; `--list` shows the blocks, `--only TEXT` runs a subset, `-v` prints their output. Blocks marked `# runnable (needs: node)` need `studio-web/node_modules`; they are skipped locally without it and fail in CI (`CI` set) or with `--strict`. CI runs it as step 8 of `.github/workflows/studio.yml`.

## 9. Tests: matrix and markers

| Suite | Command | Where it runs |
|---|---|---|
| Core | `python -m pytest tests/core -m "not slow and not integration"` | CI (no FastAPI or Node needed) |
| Studio backend | `python -m pytest tests/studio -m "not slow and not network and not e2e"` | CI |
| Web app | `npm --prefix studio-web run typecheck` and `npm --prefix studio-web test` (vitest, jsdom) | CI |
| Contract | `python -m pytest tests/studio/test_reducer_contract.py tests/studio/test_openapi_snapshot.py` | CI, after `npm ci` |
| Built assets | `python scripts/check_studio_assets.py`, and rebuild + `git diff --exit-code sparc/studio/static` | CI |
| End to end | `python -m pytest tests/studio/e2e/test_e2e_replay.py tests/studio/e2e/test_e2e_real_fast.py tests/studio/e2e/test_e2e_a11y_responsive.py` | CI |
| Wheel smoke | `python -m pytest tests/studio/e2e/test_wheel_smoke.py` (`SPARC_SMOKE_WHEEL`, `SPARC_SMOKE_FULL`) | CI |
| Docs | `python docs/studio/doctest.py` | CI |
| Slow | `python -m pytest tests/core tests/studio -m "slow and not network"` (engine host, sessions, real placebo, studies e2e, Providence e2e, performance budgets) | nightly |

Markers (`pyproject.toml`): `slow` (long; nightly), `network` (real downloads; never in CI), `e2e` (Playwright; every test under `tests/studio/e2e/`), `studio` and `integration`. A quick check that your environment works:

```bash
# runnable
python -m pytest tests/studio/foundation/test_meta.py -q -o addopts=""
```

Environment for the heavier suites:

| Variable | Effect |
|---|---|
| `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` | Single-threaded BLAS: reproducible numbers and no oversubscription on shared machines (CI sets them) |
| `PLAYWRIGHT_CHROMIUM` | Chromium for the e2e tests (default: the preinstalled `/opt/pw-browsers/chromium-1194/chrome-linux/chrome`). The tests never download browsers |
| `SPARC_E2E_REQUIRE_BROWSER=1` | Fail instead of skipping when no browser is available (CI) |
| `SPARC_E2E_ARTIFACTS` | Where the e2e screenshots go (light and dark per step) |
| `SPARC_CONTRACT_REQUIRE_TS=1` | Fail instead of skipping the TypeScript half of the reducer contract when Node is missing |
| `SPARC_PROVIDENCE_RUNS` | A folder of recorded Providence runs (`output/core/providence`) for the Providence import, grid and caveat tests and the Providence e2e; they skip without it. Point it at a copy: the tests never write into it |

Tests live with the code they own: `tests/studio/<group>/` per backend package, `studio-web/src/**/__tests__` and `*.test.ts(x)` for the web app. The e2e harness (`tests/studio/e2e/conftest.py`) starts `python -m sparc.studio --workspace <tmp> --port 0 --no-browser --token e2e` with this checkout first on `PYTHONPATH`, fails a test on any browser console error, and screenshots every step in both themes.

## 10. Rules of the road

- **The server never imports torch** (`tests/core/test_import_light.py`, the kind-module rule of [§4](#4-adding-a-job-kind)). It starts in about a second and stays under 400 MB.
- **Workers never write SQLite**; hooks do, in the server.
- **Core changes are minimal and additive.** Pickled classes gain no required attributes (readers use `getattr(…, default)`), so existing checkpoints keep unpickling. Editing a fingerprinted core module changes the resume fingerprint of every checkpoint: say so in the release notes.
- **Everything slower than about 2 s is a job**, never a request handler.
- **Threads are budgeted** (Settings): workers set them through the environment, `progress.set_threads` and the model-level overrides; inline request work runs under `threadpool_limits(1)`.
- **Paths from clients go through `safe_path`**, uploads are streamed `PUT` bodies with a suffix allowlist, pickles are loaded only from trusted runs, subprocesses use argument lists.
- **Exports go to `projects/<slug>/exports/<export_id>/`**; `POST /api/exports` creates the row before it queues the job.
- **The legacy app is untouched** (`sparc-desktop/`, `sparc/server/`); CI fails otherwise.
- **Document what you change**: api.md for the wire, SPEC.md for behaviour, their As-built sections for deviations, and these guides for anything a user or developer would do differently.
