# SPARC Studio — Product and Technical Specification

Status: **as built** — SPARC Studio 1.0.0, 2026-10-02 (designed 2026-10-01, revised after the completeness review, §18). §19 is the changelog: every place where the implementation differs from the design text, milestone by milestone. The user and developer guides are [`USER_GUIDE.md`](USER_GUIDE.md) and [`DEVELOPING.md`](DEVELOPING.md); [`RELEASE_NOTES.md`](RELEASE_NOTES.md) covers the core changes.
Companion: [`api.md`](api.md) is the exact HTTP/SSE contract. If this file and `api.md` disagree, `api.md` wins for wire formats and this file wins for behaviour.

SPARC Studio is a new local web app for the modern `sparc/core` pipeline (S0–S7 plus post-run studies). One user goes from a CSV of street-level temperatures to defensible, exportable cooling decisions, and always knows:
- what the machine is doing;
- how long it will take;
- what has already been produced;
- whether an interrupted run can be picked up where it stopped.

The old app (`sparc-desktop/`, `sparc/server/`) is **not modified, not imported and not deleted**. Studio uses its own package (`sparc/studio/`), its own frontend source (`studio-web/`) and its own default port (8765, never 8008).

---

## 0. Decision record

### 0.1 Base design and grafts

Three proposals were judged: tracking-first ("mission control"), analysis-first ("researcher workbench") and scenario-first ("Scenario Lab"). All three judges picked **scenario-first** as the spine. This spec is that design with the grafts and fixes below.

| From | What was adopted |
|---|---|
| Scenario-first (base) | Scenario Lab as the project home once a run exists. Declarative `ScenarioDoc` with portable selections. Server-side preview that calls core's own `sparc.core.emulator.emulate`, so there is no TypeScript port. Exact results keep per-fold deltas and are cached by content hash plus code sha, and become stale when either changes. Editing a scenario that has a result forks a revision. Decision packs come from exact results only. The launch-snapshot config (`launch.json`) is used for resume and readers. S5 per-fold deltas are persisted. `ScenarioEngine(base_fold=…)`. `equity_column` is aligned by id. Templates are named as planning actions. "Check across runs" robustness. Byte-offset event cursor. |
| Tracking-first | `sparc/core/progress.py` envelope, span tree, `Cancelled(BaseException)`. The single `FittedEnsemble.fold_predictions` hook. `plan_stages` with a plan == emitted-events test. A cost-model ETA with per-host calibration (tested within 20% of recorded Providence timings). `checkpoint.json` with section hashes and a config-edit **impact preview**. Reattach by pid + `create_time`, with "likely out of memory" labelling. External (CLI) runs shown live. Output catalog in `sparc/core/catalog.py` with a data dictionary and a completeness test. Cancel-at-safe-point → resume equality test. Mission Control live stage panels and Gantt. Stage-duration history. Self-test job kinds. |
| Analysis-first | Gating predicates shared by `run_core` and `plan_stages`. Analysis views that light up mid-run (accuracy computed from `predictions.parquet`). Pipeline **Status Board**. **Findings** notebook. Region stats, Breakdown, Relationships and correlogram tools. Ring-profile spill chart. "System sleep?" band. Event-invariant CI test. **Replay runner** for deterministic e2e. AF_UNIX RPC for the engine. |

### 0.2 Judge-raised flaws and how this spec fixes them

| Flaw | Fix (section) |
|---|---|
| A and B ported the emulator to TypeScript with a home-made FFT | Preview is server-side via core `emulate`. The browser never re-implements the model (§7.5). |
| Preview in the API threadpool can be oversubscribed or pile up | Preview runs under `threadpool_limits(1)`, single-flight and latest-wins per run (§7.5). |
| Engine RPC over stdout (A) or as a spawn child of uvicorn (C) | One long-lived **engine host** process in its own session. It holds an LRU of loaded run engines and speaks `multiprocessing.connection` over AF_UNIX (AF_PIPE on Windows) with an authkey. It survives server restarts and is reconnected to (§7.6). |
| Resume ambiguity: current config vs snapshot | Resume **always** uses `<run_dir>/studio/launch.json` (absolutised raw config plus mode args), so the fingerprint is stable. Config edits never invalidate an existing run (§5.9). |
| `load_run` needs `manifest.json`, so mid-run views fail | `run_state.json` carries the light metadata (cell size, grid shape, CV design) as soon as it is known. `RunReader` rebuilds data from `launch.json`, metrics from `predictions.parquet`, and folds from `run_state.cv` (§6.2). |
| Configured scenarios had no folds, so paired SE needed a re-run | Core writes `scenario_detail.npz` (folds, sd, extrapolation per scenario) (§11, item 12). |
| `equity_column` sliced by row position | Fixed in core: joined by id (§11, item 14). |
| Studio imported `sparc.scenario.budget` | New core `optimize.planned_allocation()` (§11, item 15). |
| "Check across runs" as serial engine-lane requests | A heavy tracked job, with time and RAM cost shown before start (§7.12). |
| Results page builder moving out of core | It stays in `sparc/core/results_page/` as package data (§6.9). |
| `python-multipart` dependency, spooled 2 GB uploads | Raw streamed `PUT` bodies (§10.8). |
| Bearer token in `sessionStorage` | One-time `/auth?t=` exchange into an HttpOnly SameSite=Strict cookie. Bearer is accepted only for scripts/tests (§10.8). |
| Global lock file collides across workspaces | Per-workspace `studio.lock.json` (§10.9). |
| gzip of `events.jsonl` breaks replay | Not done. Logs are size-capped instead (§5.6). |
| Layer algebra is a new correctness surface | Deferred (non-goal for v1) (§15). |
| Statistics jargon for planners | Plain-language result cards ("likely range", "confident it cools / could be zero") with an expert toggle (§7.7). |
| Demo city had no CRS, so no lon/lat or decision pack | The demo city is placed at a fictional location in EPSG:32619, badged DEMO (§9.2). |
| Threads oversubscribed (`cpu−1` + 2-thread engine) | A single thread budget is enforced by the scheduler (§10.5). |
| Per-host log-linear ETA overfits | Seeded unit rates plus a median of recent observations plus online refinement. No regression (§5.4). |
| Shell-wide shared selection coupling (B) | Selection is per run and URL-encoded. Views opt in (§6.5). |

---

## 1. Users and principles

### 1.1 Users

- **Planner** (parks, public works, sustainability office). Wants to know where to plant, depave or brighten, what it buys, for whom, and how sure we are. Spends most time in the Scenario Lab, Plans and Climate. Needs exports: decision packs, GIS, briefs.
- **Analyst** (city GIS or consultant). Sets up projects, fetches open data, launches and monitors runs, and checks validation. Spends time in Setup, Launch, Mission Control and the Run hub.
- **Researcher**. Reads outputs critically: accuracy by fold, zone and distance; whether the stack beats kriging; saturation; causal agreement; robustness. Uses the analysis tabs, generic tools, studies and Findings.

All three are the **same single local user** wearing different hats. There are no accounts.

### 1.2 Binding principles

1. **The run directory is the source of truth.** Studio indexes runs in SQLite and owns only `<run_dir>/studio/`. Pipeline files change only through core functions. Post-run functions that mutate the manifest run under a per-run lock. Every Studio-launched run gets a unique directory, so nothing is ever overwritten.
2. **Anything slower than about 2 s is a tracked Job**, never a request handler. A job is a subprocess in its own process group, or a request to the engine host. It has an append-only `events.jsonl`, an SSE stream, real Cancel, and Resume where core supports it. Closing the browser never affects a job. A server restart reattaches to it.
3. **Core emits structured progress** (`sparc/core/progress.py`). Studio never regex-scrapes prose logs. Plain log lines are still forwarded verbatim as `log` events.
4. **Two speeds, always labelled.** *Preview* is the linear emulator, run server-side with core numpy (20–150 ms on 54,701 cells). *Exact* is `ScenarioEngine` on the run checkpoint (≈2 s fast run, ≈13 s full run after a 40–60 s warm-up). Uncertainty, comparisons, decision packs and impact numbers come **only** from Exact. Preview carries a per-lever trust badge.
5. **Scenarios are declarative, portable JSON** with a stable id, content hash and lineage. Results store per-fold deltas, so every region mean, every A−B difference and every equity split carries a jackknife standard error.
6. **The 30 m grid is a raster.** Values travel as Float32 binary and are drawn on a canvas `ImageData` at one pixel per cell. There is no map library and no per-cell GeoJSON on the wire.
7. **One contract source.** Pydantic models produce OpenAPI, and generated TS types are checked against the hand-written client types in CI. Stage, output, layer, job-kind and palette catalogs are Python constants served by `/api/meta`.
8. **Core changes are minimal and additive.** Pickled classes gain no required attributes; readers use `getattr(..., default)`. Existing checkpoints keep unpickling.
9. **Honesty over polish.** Every number shows units and sign ("cooler"/"warmer"). Extrapolated cells are hatched. Missing outputs name the stage that produces them and offer a one-click action. Stale files are flagged.
10. **Few dependencies.**
    - Server: `fastapi`, `uvicorn`, `threadpoolctl`, plus the existing `pydantic`, `pyarrow`, `psutil`, `pyproj`, `scipy`, `matplotlib`.
    - Client runtime: `react`, `react-dom`, `zustand`, `marked`, and three `@fontsource` font packages (CSS and woff2 only).

---

## 2. User journeys

Each journey is an acceptance scenario. The e2e suite (§14.5) automates J1, J2 (partial), J4, J5, J6 and J10.

**J1 — Ten-minute tour with no data.**
1. `pip install "sparc[studio]"`, then `sparc studio`.
2. The terminal prints the workspace and the one-time link (`SPARC Studio 1.0.0 - workspace ~/sparc-studio`, then `Open http://127.0.0.1:8765/auth?t=…`) and opens the browser on that link.
3. On Home, choose **Try a synthetic city**. Studio writes the synthetic CSV and `truth.json`, plus DEMO people/land-cover layers and a DEMO CMIP6-style climate table, all at a fictional location (EPSG:32619). It also writes a runnable `config.yml`.
4. The project overview readiness spine is all green. The primary button reads **Launch first run**.
5. Launch preselects **Fast**. The plan graph shows S0…S7, with S6 "will run" (canopy treatment), and estimates "≈4–6 min on this machine".
6. Start opens Mission Control. The stage rail advances, the fold × model grid fills, S4 curves draw point by point, and the S5 scenario feed fills.
7. The Accuracy tab is usable as soon as `predictions.parquet` lands, while S4 is still running.
8. On success the primary button becomes **Open Scenario Lab**. Choose the template "Shade the hottest 10%". The preview ΔT map appears in about 100 ms.
9. **Run exact** shows 3 fold ticks. The result card reads, for example: "Cools the edited area by 0.41 °F (likely range 0.30–0.52 °F). Confident it cools. 12% of the cooling lands outside the edited cells."
10. Validation shows a "Truth vs recovered" card (planted canopy effect vs recovered effect).
11. Export a decision pack.

**J2 — New city from the analyst's CSV.**
1. New project, then upload a CSV (streamed `PUT`).
2. Header preview with per-column stats, plus suggested mappings (target, id, x/y, zone, coord unit, CRS hint).
3. **Check data** runs S0 inline (≈0.3 s for 54,701 rows). It shows QA flags as badges, grid shape, fill fraction, noise floor and a preview raster of any column.
4. Levers step: actionable toggles, bounds, doses, direction, cost. The dose-scale table warns about doses beyond 1 sd.
5. Physics step: role mapping dropdowns. Missing `canopy`/`impervious` roles are flagged with "placebo, simcheck, planner and emulator need these".
6. Inputs step: Campaign forcing (ERA5+ISD), People & land cover (HRSL+WorldCover), CMIP6 change factors. Each is a tracked network job with a host pre-check, and each ends with **Link into config**, which shows a YAML diff.
7. Validate is clean. Launch **Coarse 60 m** (est. ≈25 min), then **Full** overnight.

**J3 — Overnight full run, laptop sleeps, server dies.**
1. A FULL run with CV curve is estimated at 1 h 40 m–1 h 55 m, peak RAM ≈3 GB, disk ≈0.6 GB.
2. The lid closes and the server is killed.
3. Next morning, run `sparc studio`. The job's worker pid is gone and there is no `run.end`, so the status is **interrupted**. The last RSS is shown; if it was above 80% of RAM, the label reads "possibly out of memory".
4. The run page shows the checkpoint card: done S3 + baselines, saved 02:14, 525 MB, "matches launch snapshot". It also shows outputs already written (`predictions.parquet`, `baselines.json`), which are already viewable.
5. **Resume** plan: S0 reloads the data (0.3 s, always runs); S1/S2_S3/baselines come from the checkpoint; cv_curve, S4–S7 will run; "saves ≈26 min".
6. Resume. The new job's rail marks reused stages as **cached**. Lineage links both jobs to the run.

**J4 — Cancel, adjust, relaunch.**
1. During S2_S3 the stacker leaderboard shows the equal-weight mean winning, and a `physics.antiphysical_a` warning appears.
2. **Cancel**. The worker acknowledges at the next safe point (≤10 s: MGWR checks for cancel after every tuning score, §5.5). Status becomes cancelled; the checkpoint card says "none (S3 not reached)".
3. Edit `stacker.tune_lambda` in the config editor. The impact preview says "2 existing runs would need a refit to reflect this change (section changed: models)". Save.
4. Relaunch.

**J5 — Explore every output of a finished run.** Run hub Overview shows:
- KPIs: held-out R², RMSE, 90% interval coverage vs target, half-width vs noise floor, headline scenario, mid-century warming;
- a stage timing bar;
- data-health flags;
- an outputs availability grid with **stale** badges (e.g. an old `causal.json`).

Then the user tours Map (all themes), Accuracy, Distance, Influence, Response, Scenarios, Climate, Causal, Budget, Planner, Uncertainty, Docs, Files and Provenance. From there they:
- brush the far-distance bin of the residual histogram, which selects cells on the map;
- open Region stats and Relationships (residual vs `dist_train_m`);
- pin three Findings;
- export a GeoTIFF of package cooling and a whole-run ZIP (a tracked job; the checkpoint is excluded).

**J6 — District scenario (planner).**
1. Lab on the full run. The engine chip shows "Loading checkpoint 312/525 MB…", then ready in ≈45 s.
2. Selection: Zone 3 **and** `lc_built ≥ 0.5`. The chip reads "1,284 cells · 1.16 km² · 6,420 residents · median canopy 12%".
3. Edits: canopy **fill 50% of plantable headroom**, plus albedo **set 0.35**.
4. Preview with a trust badge. Albedo is city-wide-unreliable, but this edit is local, so it is not hatched.
5. **Run exact** (5 fold ticks, ≈13 s). The result shows:
   - edited-area and city means with likely ranges;
   - a ring-profile spill chart;
   - share of edited cells extrapolated;
   - realised vs requested dose;
   - the causal linear band;
   - residents at or above 90 °F today and under SSP2-4.5 2041–2060, with and without the scenario;
   - equity quintiles.
6. Compare against the configured "Green Infrastructure Package". The paired SE is available immediately from `scenario_detail.npz`.
7. Save, fork a variant, and promote nothing (the scenario is not expressible in YAML).

**J7 — Budget plan to field kit.**
1. Plans: canopy, a budget slider of 20,000 pp·cells, plantable cap on, objective "people", equity "share aged 60+" with focus 0.3.
2. The planned allocation and Pareto curve update in under 1 s.
3. **Verify exactly** reports planned 1,704 vs realised 1,037 °F·cells, labelled "spillover non-additivity".
4. **Plan → scenario** puts the plan into the library. **Field kit** produces a ranked list with lon/lat, logger sites and before/after pairs.
5. Export a plan pack.

**J8 — Validation dossier.** Studies hub on the full run:
- baselines (minutes);
- placebo (3 child runs, each with a mini stage rail);
- multiverse (10 variant child runs filling a heatmap);
- simcheck (generator × seed grid, resumable across days).

When attached studies finish, an `uncertainty` job runs automatically. Envelopes then appear on the configured scenarios and on exact Lab results. Finish with **Regenerate methods & model card**.

**J9 — Reproduce and compare.**
1. **Reproduce this run** produces a child run and a checklist: hard checks (cv design, R² per model, scenario effects, input data) and soft ones (code, versions).
2. Compare fast vs full shows a config diff, metrics, scenario effects, a provenance same/different chip, an environment diff, stage timings, and a difference map for same-grid pairs.

**J10 — Bring existing work in.**
1. Home → **Open Providence example**. This copies the bundled inputs (`brown4.csv`, forcing, CMIP6 table, layers).
2. From a repo checkout it optionally imports `output/core/providence/{providence_uhi, providence_uhi_fast, placebo, simcheck*, multiverse}` in place.
3. Older-code sections render as "not in this run (older code)". The fast run's Sep-30 `causal.json` and `allocation.parquet` are flagged **stale**.
4. The full run has no provenance, so Studio uses the example config (the user is told).

---

## 3. Information architecture

### 3.1 Shell (every page)

- **Left sidebar:**
  - project switcher;
  - project nav: Overview · Setup · Inputs · Launch · Runs · **Scenario Lab** (emphasised) · Studies · Exports · Findings. Each entry is declared by the route module that owns its target (§12.3 `projectNav`):

    | Entry | Target | Owner |
    |---|---|---|
    | Overview | `/p/:pid` | tracking |
    | Setup | `/p/:pid/setup/data` | projects |
    | Inputs | `/p/:pid/setup/inputs` | projects |
    | Launch | `/p/:pid/launch` | projects |
    | Runs | `/p/:pid/runs` | tracking |
    | Scenario Lab | `/r/<active run>/lab` (disabled with a reason until a run with a checkpoint exists) | lab |
    | Studies | `/p/:pid/studies` | studies |
    | Exports | `/p/:pid/exports` | studies |
    | Findings | `/p/:pid/findings` | studies |

    The config editor (`/p/:pid/config`) is reached from Setup, not from the nav.
  - at the bottom: Activity, Settings.
- **Top bar:**
  - breadcrumb (Project › Run › Tab);
  - active-run chip (mode badge, R², date, switcher);
  - **job tray** (count of running and queued jobs, mini progress bars, click → Mission Control);
  - engine status dot (cold / loading x% / ready / busy);
  - connection pill (live / reconnecting / polling);
  - command palette (Ctrl/Cmd-K: jump to run, job, layer, scenario; start actions);
  - theme toggle (system / light / dark).
- **Browser tab title** mirrors the most important running job (`▶ 42% S2_S3 · SPARC Studio`). Browser notifications on job end are opt-in.
- **URL holds all navigation state** (project, run, tab, layer, scenario ids, compare items, selection id). Every view is deep-linkable and reload-safe.

### 3.2 Routes

The owner column names the frontend work item that implements the route module (see §17).

| Route | Page | Shows | Owner |
|---|---|---|---|
| `/` | Home | Project cards (name, active run, readiness score, last activity); quickstarts **Try a synthetic city**, **Open Providence example**, **Import a config / run folder**, **New blank project**; running jobs across projects; disk and engine strip | projects |
| `/projects` | Projects | Sortable list, archive/delete | projects |
| `/p/:pid` | Project overview | Readiness spine (each row ok / warn / missing / n/a with a one-click action); **Pipeline Status Board**; latest runs; active jobs; primary CTA (Set up data → Launch run → Open Scenario Lab) | tracking |
| `/p/:pid/setup/:step` | Setup wizard | Non-linear tabs with completion dots: `data`, `levers`, `physics`, `inputs`, `scenarios`, `analysis`, `about` (§9) | projects |
| `/p/:pid/config` | Config editor | YAML textarea with line numbers and issue gutter; diff vs DEFAULTS; version history; impact preview | projects |
| `/p/:pid/launch` | Launch | Mode cards (Fast / Coarse M / Full) with time, RAM and disk estimates; stage checklist with dependency rules; CV-curve toggle; threads; "then run" chain; plan graph; preflight checks; Start | projects |
| `/p/:pid/runs` · `/runs` | Run history | Status, label, mode, started, duration, R², RMSE, coverage, scenarios, checkpoint size, studies, commit (dirty flag), origin; saved filters; select two → Compare; stage-duration history chart | tracking |
| `/p/:pid/compare?a=&b=` | Run comparison | Config diff tree, provenance chips, metrics table, scenario effects, timings, environment diff, difference map (same grid), priority τ / Jaccard | run-hub |
| `/r/:rid` | Run overview | Header, KPIs, stage timings, data health, outputs grid, studies status, caveats, run findings. If running, a compact tracker on top (the foundation `JobStrip` component fed by the global `job.progress` events; feature folders never import each other) | run-hub |
| `/r/:rid/track` | Tracker for the run's jobs | Mission Control of the latest job, with a job switcher (launch, resume, …) | tracking |
| `/r/:rid/map` | Map explorer | Full-width MapView with all themes and the analysis-tools drawer | run-hub |
| `/r/:rid/data` | Data & QA | §6.4 | run-hub |
| `/r/:rid/accuracy` | Accuracy | §6.4 | run-hub |
| `/r/:rid/distance` | Distance & baselines | §6.4 | run-hub |
| `/r/:rid/influence` | Influence | §6.4 | run-hub |
| `/r/:rid/response` | Response | §6.4 | run-hub |
| `/r/:rid/scenarios` | Configured scenarios | §6.4 | run-hub |
| `/r/:rid/climate` | Climate | §6.4 | run-hub |
| `/r/:rid/causal` | Causal audit | §6.4 | run-hub |
| `/r/:rid/budget` | Budget (S7) | §6.4 | run-hub |
| `/r/:rid/planner` | Planner pack | §6.4 | run-hub |
| `/r/:rid/uncertainty` | Uncertainty | §6.4 | run-hub |
| `/r/:rid/docs/:doc?` | Documents | report / methods / model card / uncertainty / placebo / multiverse / simcheck / benchmark markdown; Regenerate | run-hub |
| `/r/:rid/files` | Files | Inventory, data dictionary, previews, raw downloads, conversions, checkpoint card | run-hub |
| `/r/:rid/provenance` | Provenance | Hashes, git, platform, environment list and diff, effective config, launch snapshot, Reproduce | run-hub |
| `/r/:rid/validation` | Validation | Study cards with status, cost estimate, launch and result charts | studies |
| `/r/:rid/lab` | Scenario Lab — Design | Left: editor + selection builder; centre: map; right: inspector; bottom: compare tray | lab |
| `/r/:rid/lab/library` | Library | Search, tags, lineage tree, configured group, statuses | lab |
| `/r/:rid/lab/s/:sid` | Scenario + result | Editor plus the latest result on this run | lab |
| `/r/:rid/lab/plans` · `/r/:rid/lab/plans/:plid` | Budget plans | §7.10 | lab |
| `/r/:rid/lab/sweeps/:swid?` | Sweeps | §7.9 | lab |
| `/r/:rid/lab/climate` | Climate × adaptation | §7.11 | lab |
| `/r/:rid/lab/compare?items=` | Compare | §7.12 | lab |
| `/p/:pid/studies` · `/studies/:stid` | Studies hub / study page | Status matrix per run; study page with child tracking and result views | studies |
| `/p/:pid/exports` | Exports & reports | Report builder (sections, preview), export history | studies |
| `/p/:pid/findings` · `/findings` | Findings | Notebook, reorder, annotate, export MD/HTML | studies |
| `/jobs` | Activity | Queue (reorder, pause), running, history with filters, interrupted with Resume | tracking |
| `/jobs/:jid` | Mission Control | §5.10 | tracking |
| `/settings` | Settings | Workspace, thread budget and slots, engine budget and idle timeout, watch roots, cache manager, storage manager, network check, about | tracking |
| `*` | Not found | Link home | foundation |

**Run tabs are registry-driven.** Each route module can declare `runTab: {id, label, group, order}`. `RunLayout` (foundation) builds the tab bar from those declarations. Groups:
- **Model**: Overview, Data, Accuracy, Distance, Influence
- **Effects**: Response, Causal
- **Decisions**: Scenarios, Climate, Budget, Planner, Lab
- **Trust**: Validation, Uncertainty, Provenance
- **Run**: Track, Map, Docs, Files

**Run tab ids** are a fixed vocabulary shared by the frontend route modules and the server: `overview, data, accuracy, distance, influence, response, causal, scenarios, climate, budget, planner, lab, validation, uncertainty, provenance, track, map, docs, files`.

Each tab shows a status dot (ready, partial, running, missing (+ action), stale). The dot comes from `GET /api/runs/{rid}/outputs` → `tabs[]`, keyed by tab id. The server computes it by grouping catalog outputs by `OutputSpec.view` (§6.1), whose values are these tab ids or a viewer kind. Tabs with no catalog outputs of their own (`track`, `map`, `files`, `provenance`, `lab`, `validation`) use fixed rules: `track` = running if a job is active; `lab` = missing (+ "needs a checkpoint") without `checkpoint.pkl`; the others are always ready. Route modules do not declare outputs.

### 3.3 Navigation rules

- Run tabs are always reachable, even mid-run. Missing outputs render a typed empty state naming the producing stage or study, with an action button ("Resume to compute S6", "Run planner pack").
- Clicking a chart element with spatial meaning switches the map layer. Examples:
  - a scenario dot → its Δ layer;
  - a lever row → its footprint layer;
  - a fold bar → the CV fold layer.
- Selections are per run, stored in the URL (`sel=<region id>` or an inline encoded spec). Views that support selection (Map tools, histograms, Lab) read it; other views ignore it.

---

## 4. System overview

### 4.1 Processes

```
browser (React SPA, ≤2 SSE connections per tab)
   │  HTTP JSON · binary Float32 · SSE
   ▼
uvicorn + FastAPI (1 process, asyncio)  ── never imports torch; starts in ~1 s
   ├─ EventHub, JobManager scheduler, JobTailers, ResourceSampler, EngineHost client   (asyncio tasks)
   ├─ anyio thread pool: parquet/JSON reads, selection resolve, preview (threadpool_limits(1)),
   │                     planned optimiser, climate summarize, single-layer exports
   ├─ job workers: `python -m sparc.studio.jobs.worker <job_dir>`  (own session/process group each)
   │     └─ may fork ProcessPool workers (simcheck, multiverse) → progress.init_worker propagates sink/context
   └─ engine host: `python -m sparc.studio.engine.host --sock <ws>/engine/host.sock`  (own session; long-lived)
         └─ LRU of RunSession objects (ScenarioEngine + ResponseEngine + mediators + layers), one request at a time
```

### 4.2 Workspace layout

Default `~/sparc-studio`. Overridden by `$SPARC_STUDIO_HOME` or `--workspace DIR`.

```
studio.sqlite                 # index/projection; rebuildable with `sparc studio --reindex`
studio.lock.json              # {pid, port, url, version, started_utc}  (0600, per workspace)
token                         # current launch token (0600)
cache/                        # ghcn_*.csv, global_hourly_*.csv, pangeo-cmip6-*.csv, isd-history.csv — passed to every fetcher/planner_pack
engine/host.json  host.sock   # engine host pid/create_time/authkey (0600)
jobs/<jid>/                   # job.json, state.json, events.jsonl, stdout.log, stderr.log, result.json, cancel
projects/<slug>/
  project.json                # {id, name, template, demo, created_utc, report{...}, headline_scenario, cost_model}
  config.yml                  # core YAML; relative paths resolve here
  data/                       # uploaded CSV/parquet (+ join tables)
  inputs/{forcing,climate,layers,features}/
  runs/<run_id>/              # the core run_dir (run_core(run_dir=...)); studio/ inside
  studies/<study_id>/         # study outputs; child runs under children/
  exports/<export_id>/        # every export of the project (bundles, GIS, pages, packs, reports, findings) — never inside a run dir
  scenarios/<sid>.json        # durable mirror of each scenario (all revisions), rewritten on every save (§10.6)
  findings/<fid>.json  findings/<fid>.<png|svg>   # durable mirror of findings and their images
imports/<run_id>/studio/      # studio/ side folder for runs imported in place from elsewhere
```

### 4.3 Run identity and the launch snapshot

- **`run_id`** has the form `YYYYMMDD-HHMMSS-<fast|coarse60|full|custom>-<4 hex>`. The run directory is `projects/<slug>/runs/<run_id>/`, passed to core as `run_core(run_dir=...)`, so runs never collide.
- At launch Studio writes **`<run_dir>/studio/launch.json`**:

  ```json
  {"studio_version": "1.0.0", "run_id": "...", "project_id": "...",
   "config_raw": { /* user raw config with every path made absolute, before DEFAULTS merge */ },
   "config_dir": "/abs/project/dir",
   "args": {"stages": ["S0","S1","S2","S3","S4","S5","S6","S7"], "fast": false, "coarse": null,
            "cv_curve": null, "threads": 3},
   "created_utc": "...", "job_id": "j_..."}
  ```

  The worker always builds the config from this file, via `core_config_from_dict(config_raw, base_dir=config_dir)`, and calls `run_core(cfg, run_dir=…, run_meta=…, **args)`. **Resume reuses it byte-for-byte.** As a result:
  - the core fingerprint (which hashes `cfg.raw`, the args and the data stat) is identical at launch and at resume;
  - later project-config edits never invalidate an interrupted run.

  Rules for building `config_raw` at launch:
  - every path key (`data.path`, `data.join[].path`, `physics.forcing`, `climate.table`, `planner.layers`) is made absolute;
  - `climate.cache` is set to the absolute workspace cache `<ws>/cache`, because `climate_stage` writes its CMIP6 cache under `output.dir / climate.cache` and an absolute value wins the path join;
  - `output.dir` is set to the absolute `projects/<slug>/runs` (only the climate cache and study children read it; `run_dir` is always explicit).

  Two documented exceptions to "byte-for-byte":
  - `args.threads` may differ on resume (`POST /api/runs/{rid}/resume {threads}`), because threads are not part of the fingerprint;
  - **Resume with current project config** (`use_current_config: true`) renames the old file to `launch.<n>.json` and writes a new `launch.json` from the current project config. This is a deliberate refit and the dialog shows its impact first.
- **`run_meta`** (passed to every `run_core` Studio starts, stored in `manifest.run_meta` and in the `run.start` event): `{studio_run_id, project_id, origin, study_id?, parent_run_id?, role?}`. `role` is `"placebo:<kind>"`, `"variant:<name>"` or `"reproduction"` for study children. The registry links child runs from this field, never from directory names.
- Readers (`RunReader`, `open_run`, `load_run`) resolve the config in this order:
  1. `launch.json`
  2. `manifest.config` + `provenance.config_dir`
  3. an explicit config path given at import
- **Run origins:** `studio` · `imported` (CLI-made, indexed in place) · `study_child` · `reproduction` · `external_live` (a CLI run in a watch root with a fresh `run_state.json` heartbeat).

---

## 5. Pipeline tracking

### 5.1 What is tracked

Every job of every kind is tracked:
- runs and resumes;
- post-run actions: baselines, planner, emulator, uncertainty, writeup;
- studies: placebo, simcheck, multiverse, reproduce, benchmark;
- input fetches: forcing, layers, features, CMIP6, GHCN, the ISD station index;
- external CLI runs in watch roots (read-only pseudo-jobs, §5.14);
- engine requests: open, scenario, batch, sweep, plan verify, rerun configured;
- heavy scenario jobs (across runs);
- exports: bundle, GIS, page, packs, report, findings.

For each job the tracker records:
- lifecycle status;
- a span tree with timings;
- planned work units and cost-weighted progress;
- ETA with a range;
- live metrics;
- warnings (deduplicated, counted);
- artifacts written, as they appear;
- checkpoints;
- resource use;
- structured logs plus raw stdout/stderr;
- effective config and provenance;
- lineage (parent job/run/study, child runs, the scenario it serves).

### 5.2 Core emitter: `sparc/core/progress.py` (exact public API)

The emitter is stdlib-only and about 350 lines. Every call returns immediately when no sink is configured (a single module-level bool check). The module is **excluded from the resume fingerprint**.

```python
SCHEMA = 1
ENV_SINK, ENV_LEVEL, ENV_JOB, ENV_CANCEL = "SPARC_PROGRESS", "SPARC_PROGRESS_LEVEL", "SPARC_JOB_ID", "SPARC_CANCEL_FILE"

class Cancelled(BaseException): ...      # BaseException: passes through every `except Exception`

def configure(sink: str | Callable[[dict], None] | None, *, level: str = "info", job_id: str | None = None,
              cancel_file: str | None = None, heartbeat_s: float = 15.0) -> None
    # sink: path (opened O_WRONLY|O_APPEND|O_CREAT), "stderr", "fd:N", or a callable (in-process/tests)
def configure_from_env() -> None          # idempotent; reads the four env vars
def reset() -> None                       # tests
def enabled() -> bool
def emit(type: str, /, *, lvl: str = "info", **fields) -> None
@contextmanager
def span(kind: Literal["stage", "task"], name: str, *, k: int | None = None, n: int | None = None,
         unit: str | None = None, est_s: float | None = None, **fields)
    # emits "<kind>.start" then "<kind>.end" {status: ok|error|cancelled, elapsed_s, error?, metrics?}
def stage(name: str, **f); def task(name: str, **f)           # thin wrappers over span
def skip(stage: str, reason: str, **fields) -> None            # "stage.skip"
def tick(k: int, n: int, *, unit: str, label: str = "", **metrics) -> None
    # throttled to ≤1/s per span, EXCEPT ticks with k == 1 or k == n, which are always written
    # (they mark the start and completion of each counted sequence; §5.4 unit accounting relies on k == n)
def metric(name: str, value: float | int | str | bool | None, *, unit: str | None = None, **tags) -> None
def artifact(path, *, role: str, stage: str | None = None) -> None   # adds bytes; path made relative to run_dir context
def checkpoint(action: str, *, done, bytes: int | None = None, elapsed_s: float | None = None,
               fingerprint: str | None = None, changed_sections: list[str] | None = None) -> None
def warn(code: str, message: str, **data) -> None
@contextmanager
def context(**kv)                         # merged into ctx of every nested event (partition, variant, generator, seed, placebo)
@contextmanager
def run_dir_scope(run_dir)                # artifact() paths become relative to it
@contextmanager
def job_scope(*, job_id: str, sink: str, cancel_file: str | None)   # engine host: per-request reconfiguration
def check_cancel() -> None                # raises Cancelled if requested (signal flag, or cancel file — stat ≤ every 0.5 s)
def cancel_requested() -> bool
def request_cancel() -> None              # in-process/tests
def install_signal_handlers() -> None     # SIGTERM sets flag; 2nd SIGTERM/SIGINT raises Cancelled immediately
def worker_env() -> dict                  # env to hand a ProcessPool initializer
def init_worker(env: dict) -> None        # ProcessPoolExecutor(initializer=progress.init_worker, initargs=(progress.worker_env(),))
def wrap_context(fn: Callable) -> Callable   # contextvars.copy_context().run wrapper for ThreadPoolExecutor.submit
@contextmanager
def limit_threads(n: int)                 # threadpoolctl + torch.set_num_threads + OMP/MKL/OPENBLAS env (restored on exit)
def set_threads(n: int) -> None           # permanent form of limit_threads for a worker process (env + threadpoolctl + torch)
class LogBridge(logging.Handler)          # attached to the "sparc" logger when a sink exists; record → "log" event;
                                          # WARNING+ also → "warning" {code: "log.<logger>"}; logging.captureWarnings(True)
```

`limit_threads` and `set_threads` import `threadpoolctl` and `torch` lazily and skip either when absent. `torch` is never imported by any other progress function.

**Write discipline.** Each event is one `json.dumps(..., separators=(",",":"), allow_nan=False)` line, no longer than 4,096 bytes, written by a **single `os.write`** on the O_APPEND file descriptor. Lines from pool workers therefore never interleave.
- Field values are JSON scalars or small lists/dicts of scalars (a plan, a done set, a path). Never per-cell arrays.
- NaN/Inf become `null`.
- If a line would exceed 4 KB, string fields are truncated with `"…"` first; if it is still too long, the largest field is dropped and `"truncated": true` is added.

A daemon thread emits `heartbeat` every `heartbeat_s` seconds while the sink is configured. It reads RSS with psutil when importable, else `/proc/self/statm`, else `resource.getrusage` (the module stays stdlib-only).

### 5.3 Event envelope and types (schema v1)

Envelope, present on every line:

```json
{"v":1, "type":"task.end", "seq":412, "ts":1790000000.123, "t_rel":812.4, "pid":4242, "job":"j_7f3a",
 "lvl":"info", "span":"4242:57", "parent":"4242:12",
 "path":["run:providence_uhi","stage:cv_curve","task:cv_partition[2/3]","task:fold[4/5]","task:base_model[mgwr]"],
 "ctx":{"partition":"1000 m blocks"}, "...": "type-specific fields"}
```

- `seq` is monotonic per process.
- `span` has the form `"<pid>:<n>"`.
- `path` names span ancestry using `"<kind>:<name>[k/n]"`.

| type | Fields (besides the envelope) | Emitted by |
|---|---|---|
| `run.start` | `name, stages[], fast, coarse, resume, cv_curve, config_sha256, code_sha256, run_meta{}` | core `run_core` |
| `run.dir` | `run_dir, fingerprint` | core |
| `run.plan` | `nodes[PlanNode], total_units{}, n_points`. Re-emitted when nodes become cached or skipped, or when a count becomes known (e.g. CMIP6 models) | core |
| `stage.start` / `stage.end` | `stage, label, est_s?` / `stage, status, elapsed_s, summary{}` | core |
| `stage.skip` | `stage, reason` ∈ `not_requested`, `disabled_by_config:<key>`, `checkpoint`, `no_budget`, `no_treatments`, `no_responses`, `requires_S5`, `no_partitions` | core |
| `task.start` / `task.end` | `name, key?, k?, n?, unit?` / `+ status, elapsed_s, metrics{}`. `unit` is set when the task completes one planned unit (§5.4) | core |
| `tick` | `k, n, unit, frac, label, ...metrics` | core |
| `metric` | `name, value, unit?, tags{}` | core |
| `artifact` | `path` (run-relative), `role, bytes, stage?` | core |
| `checkpoint` | `action` ∈ `saved`/`loaded`/`mismatch`, `done[], bytes?, elapsed_s?, fingerprint?, changed_sections[]?` | core |
| `warning` | `code, message, data{}` | core / LogBridge |
| `log` | `logger, level, msg` | LogBridge |
| `heartbeat` | `rss_mb, cpu_s, threads` | core daemon thread |
| `cancel.requested` | `by` (`user`, `kill` or `shutdown`) | **server** (appended) |
| `cancel.ack` | `at_path[]` | core |
| `run.end` | `status` ∈ `succeeded`/`failed`/`cancelled`, `elapsed_s, timings_s{}, done[], error{type, message, traceback_tail}?` | core |
| `job.status` | `status, exit_code?, error?` | **server** (appended) |
| `job.result` | `result{}` (kind-specific summary, also in `result.json`) | worker |

**Stage ids** are `S0, S1, S2_S3, baselines, cv_curve, S4, S5, climate, S6, S7, finish`. These match the manifest `timings_s` keys, plus `climate` and `finish`. The checkpoint key for `S2_S3` is `S3`; the mapping is served in `/api/meta` (`stages[].checkpoint_key`).

**Warning codes**:

- Data and pipeline:
  - `qa.<flag>` (classed_target, albedo_scale, canopy_scale, impervious_scale, cover_overlap, dose_scale_<var>, coarse, window)
  - `cv.block_raised`, `cv.partition_skipped`
  - `checkpoint.mismatch`
  - `influence.few_cells`, `influence.constant_predictor` (emitted by `influence.py`, which is instrumented, §11 item 7)
  - `physics.missing_roles`, `physics.antiphysical_a`
- Causal:
  - `causal.hole_scale`, `causal.blp_constant`, `causal.dag_unavailable`
  - `causal.no_confounders`, `causal.controls_missing`, `causal.treatment_missing`
  - `causal.audit_flag {treatment, check, verdict}`
- Validation:
  - `baselines.stack_not_better`
  - `interval.coverage_below_target`
- Studies, post-run and inputs:
  - `reproduce.input_changed`
  - `planner.hot_days_skipped`, `planner.gpkg_skipped`
  - `layers.missing`, `layers.empty_points`
  - `forcing.lsm_unavailable`, `forcing.station_unavailable`
  - `climate.model_skipped`
  - `network.retry` (one per retried request in `climate.http_fetch` / `forcing.http_range`, with host, attempt and wait; until now these retries were silent for up to ≈30 s)
  - `log.<logger>` (bridged)

### 5.4 Plan and ETA

**`plan_stages`** lives in core (§11, item 4). It is a pure function built on the **same gating predicates** that `run_core` uses: `_wants_s4`, `_s6_runs`, `_s7_runs`, `_climate_runs`, `_baselines_list`, `_cv_curve_sizes`. Because the two share code, the plan cannot drift from what actually runs.

`PlanNode = {id, label, state: "will_run"|"skipped"|"cached", reason?: str, units: {unit_kind: n}, checkpoint_key?: str}`

`reason` is a `stage.skip` reason (§5.3) for skipped nodes, `checkpoint` for cached nodes, and `required_by:<stage>` for a will_run node forced by a later stage (S4 when only S6 or S7 is requested).

Each node's planned units:

| Node | Units |
|---|---|
| `S0` | `s0_load:1` |
| `S1` | `s1_influence:1` |
| `S2_S3` | `base_fit:<model>` × K per enabled model; `adv_refit` × K when advection is fitted; `stacker_fit:<mean\|nnls\|residual>` × K for each candidate (C = 2 + len(tune_lambda)); `checkpoint_save:1` |
| `baselines` | `baseline_fit:<model>` × K |
| `cv_curve` | P × (S2_S3 units without `checkpoint_save`), where P is the number of block sizes left after the skip rules (< 3·dx, > extent/3, equal to the main block); plus `baseline_fit` units per partition when baselines are on |
| `S4` | `engine_pass:1` for the ScenarioEngine baseline pass, plus Σ over levers of (non-zero doses + marginal passes) `engine_pass`. Marginal passes per lever (`ResponseEngine.marginals`) = 2 (base + own) + len(influence.scales) × (number of moved columns: the lever plus every mediator it feeds) + 1 when the physics model is present; with the default 3 scales and physics, canopy and impervious (which both move NDVI) take 9 and albedo 6 |
| `S5` | `engine_pass` × N_specs; plus `climate_model` × M when `climate.source == cmip6` (M = 24 until the catalogue is read, then the real count via a re-emitted `run.plan`) |
| `S6` | Σ over treatments of (8 `engine_pass` + `causal_step:{dml,spillover,cate,dr,sens,audit}`) |
| `S7` | `engine_pass:1`, `pareto:1` |

**Unit accounting** (one rule for the Python projection, the TS reducer and the calibration feed). A planned unit of kind U is completed by:
1. a `task.end` with `status: ok` and `unit == U` (one unit each). Examples: `base_model` → `base_fit:<model>`, `adv_refit`, `stacker_fold` → `stacker_fit:<mean|nnls|residual>`, `baseline_model` → `baseline_fit:<model>`, the causal steps → `causal_step:<s>`, `remote_object` → `climate_model`, `pareto`, `replicate` → `replicate:<g>`, `variant` → `variant:<name>`;
2. a `tick` with `unit == U` and `k == n` (one unit each). `engine_pass` is counted this way: `fold_predictions` ticks `k/K` with `unit="engine_pass"`, and the `k == K` tick also carries `pass_s` (seconds for the whole pass, used by the calibration feed);
3. `stage.end{ok}` of S0 / S1 for `s0_load` / `s1_influence`, and `checkpoint{saved}` for `checkpoint_save`.

Ticks with `k < n` add fractional progress (k/n of one unit) but never count as completions.

**Progress weights.** Cost-weighted progress uses the **seed rates** of the table below as weights (no host calibration, no n-scaling). This keeps the Python projection and the TS reducer identical; `/api/meta.unit_costs` serves the table. ETA seconds use the full cost model below.

**Cost model.** Implemented in `sparc/studio/jobs/eta.py`, owned by the backend foundation:

```
seconds(unit) = rate(host, unit) × (n_cells / 54_701)^α(unit) × (4 / threads)^0.7
α = 1.3 for base_fit:mgwr, 1.15 for base_fit:gwrf, 1.0 otherwise; climate_model/network units are not scaled
```

`rate(host, unit)` is the median of the last 20 observations in `unit_timings` for this host. If no history exists, it falls back to the **seed table**, measured on full Providence with 4 cores:

| Unit | Seed rate (s) |
|---|---|
| `base_fit:mgwr` | 170 |
| `base_fit:gwrf` | 23 |
| `base_fit:gam` | 6 |
| `base_fit:physics` | 4 |
| `base_fit:ols` | 0.1 |
| `adv_refit` | 4.5 |
| `stacker_fit:mean` | 0.2 |
| `stacker_fit:nnls` | 0.2 |
| `stacker_fit:residual` | 14 per fold (≈70 per candidate) |
| `baseline_fit:*` | 12 (≈60 per fold for five baselines) |
| `engine_pass` | 13 |
| `causal_step:dml` / `spillover` / `cate` / `dr` / `sens` / `audit` | 10 / 15 / 1 / 22 / 0.5 / 0.5 |
| `s0_load` | 0.3 |
| `s1_influence` | 3.3 |
| `checkpoint_save` | 2 per 500 MB |
| `pareto` | 0.5 |
| `climate_model` | 20 |
| `remote_object` (other network objects) | 5 |
| `replicate:<generator>` | 254 at 90 m |
| `variant:<name>` | 1,200 at 60 m |
| `unpickle` | 1 per 20 MB |
| `mediator_fit` | 1 |

**Reference unit counts** for the recorded full Providence run (`configs/core_providence.yml` as of the recorded run: 5 folds, the five base models, no wind so no advection refits, `tune_lambda [0, 0.1, 1]` so C = 5, no in-run baselines, CV curve with 3 partitions, 3 levers with 7/7/5 non-zero doses and 9/9/6 marginal passes, 12 scenarios, 3 treatments, table climate, S7 on). The ETA calibration test (§14.2, foundation) embeds these counts as constants, so it does not need `plan_stages`; `test_plan_stages` (core) asserts that `plan_stages` reproduces them for that config.

| Node | Units | Seed seconds |
|---|---|---|
| S0, S1 | 1 + 1 | 3.6 |
| S2_S3 | `base_fit` 5 × {mgwr, gwrf, gam, physics, ols}; `stacker_fit` mean 5, nnls 5, residual 15; `checkpoint_save` 1 | 1,229.6 |
| cv_curve | 3 × (S2_S3 without checkpoint) | 3,682.5 |
| S4 | `engine_pass` 44 (1 + 19 doses + 24 marginal) | 572 |
| S5 | `engine_pass` 12 | 156 |
| S6 | 3 × (8 `engine_pass` + 6 steps) | 459 |
| S7 | `engine_pass` 1, `pareto` 1 | 13.5 |
| **Total** | | **≈6,116 vs 6,097 recorded (+0.3%)** |

**Other job kinds** estimate through the same table (the `estimate=` callable of each kind, §10.2):
- `post.emulator`: per lever (2 + channels + 2·n_patches + 1) `engine_pass`, plus `unpickle` for the checkpoint;
- `post.baselines`: `baseline_fit:<m>` × K per model;
- `post.planner`, `post.uncertainty`, `post.writeup`: fixed 30 s / 2 s / 1 s;
- `engine.open`: `unpickle` + `mediator_fit` + 1 `engine_pass`; `engine.scenario`: 1 `engine_pass`;
- `study.placebo`: kinds × `plan_stages(S0–S6)` at the study's coarse size;
- `study.multiverse`: `variant` × variants not yet done; `study.simcheck`: `replicate` × pending pairs ÷ workers;
- `study.reproduce`: `plan_stages(stages)` of the parent's snapshot;
- `input.*`: `remote_object` × expected objects (CMIP6: `climate_model` × M).

**Online refinement:**
- After the first completed unit of a kind, the in-job mean of that kind replaces the prior rate.
- After `S2_S3` ends, each cv_curve partition is estimated at 0.95 × observed S2_S3 seconds.
- The first S4 dose sets the `engine_pass` rate for the remaining S4, S5, S6 and S7.

**Range.** p25–p75 of the per-unit rate history, propagated as a sum. With fewer than 3 observations, ±25%.

**Display.** Overall progress = Σ completed planned cost / Σ planned cost. This is smoother than a stage count. ETA is shown as "≈1 h 12 m (58 m–1 h 25 m)". The Launch screen shows the same numbers before Start, plus peak RAM and checkpoint disk:
- peak RAM ≈ 55 kB × n_cells × K^0.5 + 0.8 GB (calibrated from `resource_samples` peaks);
- checkpoint ≈ 9.6 kB × n_cells.

**Calibration test.** The seed table predicts the full Providence run's recorded `timings_s` total within 20%.

**Calibration feed.** Every completed unit (unit accounting above) writes a row to `unit_timings(host_id, unit, seconds, n_cells, threads, mode, ts)`: `elapsed_s` of the `task.end`, or `pass_s` of the completing `engine_pass` tick. `host_id` = sha1(cpu model + cpu count + total RAM)[:12].

### 5.5 Emission map in core

The target is no info-level gap longer than 15 s at full resolution. This is enforced by the event-invariant test, §14.1.

**`pipeline.run_core`**
- `run.start` → `run.plan` → `run.dir` → stage spans around each block:
  - S0 181–195. `stage.end.summary` = `{n_points, n_input, n_dropped, clipped{col: n}, grid_shape, cell_m, fill_fraction, background, noise_floor}` (feeds the S0 KPI tiles)
  - S1 202–213. After `compute_influence`, one `metric influence.range_m {predictor}` and one `metric influence.anisotropy_ratio {predictor}` per predictor; `stage.end.summary` = `{block_size_m, L_prior_m, target_resid_range_m}`
  - S2_S3 219–256
  - baselines 259–276
  - cv_curve 279–297
  - S4 306–326: `task engine_init` (mediator fit + ScenarioEngine baseline pass) before the loop, then `task variable[v/V]` at 314
  - S5 328–343, with `task scenario[s/S]` per spec, then scalar metrics `scenario.mean_delta`, `scenario.se` and `scenario.frac_extrapolated`, each tagged `{scenario: <name>}`, at 334
  - climate 344–353
  - S6 357–382, with `task model_effects[t/T]` at 365–370
  - S7 385–408
  - finish in `_finish`
- `stage.skip` at each gate.
- `artifact()` after every write.
- `checkpoint` events in `_save_checkpoint` / `_load_checkpoint`.
- Every `data.qa["flags"]` entry → `warn("qa.<code>")`.
- `run.end` from `try/except/finally`, which also writes `run_state.json`.

**`ensemble.fit_ensemble`**
- `task fold[k/K]` at 208 › `task base_model[name]` (`unit="base_fit:<name>"`) at 213, with metrics `fit_s` and `heldout_rmse` / `heldout_r2` (computed at 222 from `p[te]` vs `y[te]`).
- `task advection_check` at 178 › `task adv_refit[k/K]` (`unit="adv_refit"`) per fold.
- `task stacker_candidate[c/C]` (`key` = `mean`, `nnls` or `residual:<λ>`) at 269 › `task stacker_fold[k/K]` (`unit="stacker_fit:<mean|nnls|residual>"`) at 271.
- `metric candidate_rmse {candidate}` at 284; final metrics `stacker_rmse`, `stacker_r2`, `interval_coverage`, `interval_halfwidth` before 306.

**`FittedEnsemble.fold_predictions`** (91–99): `check_cancel()` + `tick(k+1, K, unit="engine_pass")`; the `k == K` tick carries `pass_s`. This **one hook** gives fold-level progress and a cancel point to every S4/S5/S6/S7, scenario, sweep and emulator computation.

**Debug level only:**
- `stacker.fit` eval epochs (229): `val_mse, best, bad`.
- `base_models` MGWR tuning (318–328): one tick per (name, candidate) `score()` call; GWRF anchors (393).
- `physics` multistart (226).

All of these also call `check_cancel()` **at every level** (the level only gates writing the tick). Calling it after every MGWR `score()` (a 5-iteration backfit, ≈2–6 s at full resolution) bounds cancel latency to about 10 s even inside one MGWR fold, which takes 150–365 s.

**Import rule.** Instrumented modules keep `torch` imports function-local, as they are today. A core test asserts that importing `sparc.core.pipeline`, `catalog`, `cv`, `data`, `optimize`, `emulator`, `planner`, `climate` and `baselines` leaves `torch` out of `sys.modules`. The Studio server imports these modules and must never load torch (§4.1).

**Other modules**
- `influence.compute_influence`: `warn("influence.few_cells")` (528) and `warn("influence.constant_predictor")` (862), keeping the existing log lines.
- `diagnostics.cv_distance_curve`: `task cv_partition[p/P]` inside `context(partition=label)` (91); skip → `warn("cv.partition_skipped")` (97); metrics `cv_row.r2` and `cv_row.rmse` tagged `{partition}` (114).
- `baselines.baseline_oof`: `tick` per fold; `task baseline_model[name]` (`unit="baseline_fit:<name>"`) per model and fold.
- `response.sweep`: `task dose[d/D]` + metrics `mean_benefit, mean_se, frac_extrapolated` (362). Also `task marginals` (261) and `own_only_pd` ticks (327).
- `causal.run_causal_validation`:
  - `task treatment[t/T]` (1603);
  - step tasks `dml, spillover, cate, dr_curve, sensitivity, audit`;
  - DR bootstrap tick every 10 (1249); dag_audit ticks (1491);
  - each `out["flags"]` entry → `warn("causal.audit_flag")`.
- `optimize`: tasks `segments, allocate, closed_loop, pareto`.
- Network fetchers emit one `task remote_object[name]` per remote object (`unit="climate_model"` for CMIP6 models, else `unit="remote_object"`):
  - `climate.cmip6_change_factors`: per model (ThreadPool submissions wrapped with `progress.wrap_context`)
  - `forcing.campaign_forcing`: ERA5 block, ISD file
  - `opendata.fetch_layers`: per HRSL/WorldCover window
  - `features_open`: per S2 scene and DEM tile
- `planner.planner_pack`: steps. `emulator.emulator_for_run`: `task lever[v/V]` with patch ticks.
- Studies:
  - `placebo` kinds loop (193): `task placebo_kind[kind]` in `context(placebo=kind)`;
  - `simcheck._worker` (430): `task replicate[<generator>/<seed>]` (`unit="replicate:<generator>"`) in `context(generator, seed)`; the parent emits `metric share`, `metric oof_r2` (tagged generator, seed) and a `tick(done, total, unit="replicates")` at 465/473;
  - `multiverse.run_variant` (69): `task variant[name]` (`unit="variant:<name>"`) in `context(variant)`, with a parent tick at 107;
  - **nested runs.** `run_core` called inside a study emits its own `run.start`/`run.plan`/stage spans under the study's task span. The reducers attribute a nested `run.*` subtree to the matching `children[]` entry (by `ctx.placebo`/`ctx.variant` and `run_meta`), never to the job's own stage rail. Study-job progress = mean child progress (placebo, multiverse, reproduce) or completed ÷ planned replicates (simcheck);
  - every `ProcessPoolExecutor` uses `initializer=progress.init_worker`;
  - on `Cancelled` the parent calls `executor.shutdown(wait=False, cancel_futures=True)` and re-raises. Pool workers share the process group and receive SIGTERM too.

### 5.6 Job runner and persistence

**Job directory** `<ws>/jobs/<jid>/`:
- `job.json` — `{id, kind, lane, executor, params, project_id, run_id, study_id, scenario_id, created_utc, threads}`
- `state.json` — `{pid, pgid, proc_create_time, cmdline_token, executor, status, started_utc}`
- `events.jsonl` — canonical, append-only
- `stdout.log`, `stderr.log`
- `result.json` — written by the worker before exit: `{status, exit_code, result{}}`
- `cancel` — touched to request a cancel

**Spawn** (process executor):

```
subprocess.Popen([sys.executable, "-m", "sparc.studio.jobs.worker", job_dir],
                 start_new_session=True, stdout=…, stderr=…,
                 env={…, SPARC_PROGRESS=<job_dir>/events.jsonl, SPARC_JOB_ID, SPARC_CANCEL_FILE=<job_dir>/cancel,
                      OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=<threads>, PYTHONUNBUFFERED=1})
```

The worker then:
1. calls `progress.configure_from_env()`, `progress.install_signal_handlers()` and `progress.set_threads(threads)` (threadpoolctl + torch);
2. dispatches to the **library function** registered for the kind (never the `cmd_*` printers);
3. writes `result.json`;
4. exits `0` (succeeded), `1` (failed) or `130` (cancelled).

On Windows, `CREATE_NEW_PROCESS_GROUP` is used, and `CTRL_BREAK_EVENT` stands in for SIGTERM.

**Workers never write SQLite.** The server process is the only writer. Kinds report through `events.jsonl` and `result.json`. Server-side hooks registered with the kind (§10.2) turn those into rows:
- registering study child runs on their `run.dir` event;
- inserting `results`, `plans`, `sweeps` and `exports` rows on success.

**Single ordered log per job.** The server appends `job.status` and `cancel.requested` lines to the same `events.jsonl` with a single `os.write` on its own O_APPEND file descriptor. **Cursor = byte offset of a line's first byte.** It is monotonic and unique, needs no coordination, and serves as the SSE `id`. The tailer reads only complete lines (terminated by `\n`).

**JobTailer** (one asyncio task per live job) polls the file size every 250 ms and parses new complete lines. It validates them against the pydantic union, keeping unknown fields and storing unknown types as `log`. It then:
- feeds the **projection reducer** `sparc/studio/jobs/tracker.py::reduce(state, event)`, which builds the span tree, stage states, planned/done units, progress, ETA, metrics latest + series, warnings (dedup by code + message hash, count), artifacts, checkpoints and current path;
- publishes to the EventHub;
- every 2 s and at the end, flushes the projection to SQLite (`jobs.progress/eta_*/current_path/stage/last_cursor`, plus the `spans`, `metrics`, `artifacts`, `warnings`, `checkpoints` and `unit_timings` tables).

**Log size cap.** When `events.jsonl` exceeds 200 MB, the tailer stops forwarding `debug`-level lines to SQLite (they stay on disk), and the worker's emitter is downgraded to `info` on its next stat. The UI shows a note. Nothing is compressed or rotated, so cursors stay valid.

**Retention.** Job directories are kept until deleted from Activity, or by the setting "delete finished job logs after N days" (default: keep).

### 5.7 Streaming

- **Global stream** `GET /api/stream?topics=jobs,runs,engine,studies,storage`:
  - The server keeps a ring buffer of 10,000 global events with a monotonic `gseq`. The SSE id is `g:<gseq>`.
  - Events: `job.created`, `job.status`, `job.progress` (throttled to 1 Hz per job: `{job_id, frac, eta_s, eta_lo, eta_hi, stage, path_tail}`), `run.updated`, `run.indexed`, `output.written`, `engine.status`, `scenario.result`, `study.updated`, `storage.low`, `resync`, `server_shutdown`.
  - On reconnect with `Last-Event-ID` older than the buffer, the server sends `resync`, and the client refetches `GET /api/jobs?status=active` and its open resources.
- **Per-job stream** `GET /api/jobs/{jid}/stream[?after=<cursor>]` carries every line of `events.jsonl`:
  - `event:` is the event `type`, `id:` is the byte cursor.
  - It honours `Last-Event-ID`. Backfill comes from the file, then live tail.
  - `tick` events are coalesced to 4 Hz per span **on the wire only**; disk keeps every line. Coalescing never drops a tick with `k == 1` or `k == n`, so unit accounting (§5.4) gives the same result on the wire as on disk.
  - It also carries transient `resource` events every 2 s with **no id**, so they are never replayed.
  - After the job reaches a final status and the file is drained, the server sends `end {status}` and closes the stream.
- Each subscriber has a bounded queue: 2,000 events for per-job streams, 1,000 for the global stream. On overflow, queued events are dropped, `resync {after}` is sent, and the per-job stream closes; the client reconnects with `after`.
- `: ping` comment every 15 s. Responses use `Cache-Control: no-store` and `X-Accel-Buffering: no`.
- **Client rules.** At most 2 EventSources per tab: the global stream plus at most one job stream. Reconnect backoff 1 → 30 s. If streaming fails three times in a row, fall back to polling `GET /api/jobs/{jid}/events?after=` every 5 s, and the connection pill says "polling".
- **Snapshot plus live.** `GET /api/jobs/{jid}/tracker` returns the projection plus `cursor`. The client opens the stream with `after=cursor`, so a reload mid-run reattaches with no gap.

### 5.8 State machines

**Job**

```
queued → starting → running → succeeded | failed | cancelled
                       └→ cancelling → cancelled | (grace 90 s) → killed→cancelled
starting --(no event within 30 s)--> failed("worker did not start")
running --(pid vanished, no run.end/result.json)--> interrupted
queued --(dependency/lock)--> blocked → queued
```

**Stage node (UI):**
- `planned` → `running` → one of: `done`, `failed`, `cancelled`, `not_reached`
- terminal states that skip running: `skipped(reason)`, `cached` (from checkpoint), `disabled` (by config), `not_requested`

**Run (registry):**
- `queued` (launched, job not started yet) · `running` · `complete` · `partial` (cancelled/failed/interrupted with a non-empty checkpoint done set) · `failed` · `cancelled` · `interrupted` · `external_live` · `imported`
- `partial` runs always show **Resume**.

### 5.9 Checkpoints, cancel, resume, reattach

**Sidecars written by core** (§11):

```
run_state.json   {status: running|succeeded|failed|cancelled, pid, job, host, started_utc, updated_utc, stage,
                  done[], fingerprint, events_path, error?,
                  meta: {n_points?, cell_m?, grid_shape?, coarse_m?, subsample_window_n?, cv?: {n_folds, block_m, buffer_m, seed}}}
checkpoint.json  {fingerprint, sections: {data, core, s4, s5, climate, s6, s7, code}, done[], bytes, saved_utc, code_sha256}
```

- `run_state.json` is atomically rewritten at run start, at every stage boundary and at the terminal state.
- `checkpoint.json` is written after every `_save_checkpoint`.
- Studio **never unpickles 500 MB to answer "can I resume?"**.

**Checkpoint card** (run page, tracker, Resume dialog):
- exists, bytes, done set, saved_utc;
- **matches launch snapshot** (normally yes; no if the data file mtime changed or core code changed — names which);
- what a resume would reuse ("S2_S3 + baselines, saves ≈26 min").

**Impact preview** (config editor). `core.fingerprint_sections(edited_cfg, fast)` is compared with each project run's `checkpoint.json.sections`. The result lists, per run, the changed sections and the phrase "a re-run with this config would refit from <first invalidated stage>". The preview does **not** affect resume, because resume uses the snapshot.

**Cancel**
1. `POST /api/jobs/{jid}/cancel` → the server appends `cancel.requested`, touches `cancel`, and sends SIGTERM to the process group. Status becomes `cancelling`.
2. Core raises `Cancelled` at the next `check_cancel()` safe point. There are about 30 of them:
   - between folds, models and stacker candidates/folds
   - CV partitions
   - baseline folds
   - S4 doses
   - S5 scenarios
   - S6 treatments and estimators
   - DR bootstrap
   - climate models
   - simcheck replicates and multiverse variants
   - `fold_predictions` folds
   - MGWR, GWRF and physics inner ticks
3. Core writes `run_state.status = cancelled` and emits `cancel.ack` and `run.end{cancelled}`. The worker exits 130.
4. After a 90 s grace the UI enables **Force stop** (`POST …/kill`), which sends SIGKILL to the process group, pool children included. Checkpoint writes are atomic (`.tmp` + `os.replace`), and so are artifact writes (§11, item 8). A force-stopped run is therefore always consistent with its last checkpoint.

**Resume** (`POST /api/runs/{rid}/resume`) creates a new `run.core` job:
- same `run_dir`, `resume=True`, args from `launch.json` (only `threads` may be overridden, §4.3);
- linked to the run (`jobs.run_id`) and to the previous job (`parent_job_id`);
- the rail marks loaded stages as `cached` (core emits `stage.skip{reason:"checkpoint"}`).

The dialog offers **"Resume with current project config"** only as an explicit alternative. It shows the impact preview and the replan (e.g. "will refit from S1, ≈1 h 40 m").

**Studies resume at their natural granularity:**
- simcheck skips completed (generator, seed) rows;
- multiverse skips variants with a JSON result, and child runs resume from their checkpoints;
- placebo resumes per kind (§11);
- reproduce re-runs.

**Reattach** (server start). For each job in `starting`/`running`/`cancelling`:
- If `psutil.Process(pid)` exists **and** `create_time()` equals `state.json.proc_create_time` **and** its cmdline contains the job token, a tailer and an exit watcher are attached. The exit watcher polls every 1 s, because the process is not our child. The final status comes from `result.json`, else `run.end`.
- Otherwise the job becomes `interrupted`, and the run's status is recomputed from `run_state.json` + `checkpoint.json`.

**Out-of-memory labelling:**
- An exit by SIGKILL (−9) when the last RSS sample was above 80% of available memory → `failed: "likely out of memory (peak 11.8 GB of 15.0 GB)"`.
- For interrupted reattached jobs with a high last RSS → "process vanished; possibly out of memory".

### 5.10 Mission Control (`/jobs/:jid`, and `/r/:rid/track`)

**Header**
- Project · run label · mode badge (FULL / COARSE 60 / FAST) · status pill.
- Elapsed and ETA with range.
- Cost-weighted progress bar.
- Current-path breadcrumb, e.g. `cv_curve › 1000 m blocks › fold 4/5 › mgwr › tuning 9/13`.
- Threads and pid.
- Buttons: Cancel, Force stop (after the grace period), Resume, Retry, Duplicate with changes, Open outputs, Copy diagnostics (a JSON blob with job, last 200 events, versions).

**Stage rail.** Eleven chips, S0 … finish. Each chip shows:
- a state icon and text (never colour alone);
- duration or ETA;
- a tooltip with the skip or cached reason;
- checkpoint markers (hover: done set, bytes).

Clicking a chip opens its live panel and zooms the Gantt.

**Live stage panels** (right side, driven by `metric`/`task` events):

| Stage | Panel |
|---|---|
| S0 | QA KPI tiles (cells, dropped, clipped, grid fill, background, rounding-noise floor) and flag badges as they arrive |
| S1 | Influence-range table filling per predictor; anisotropy chips |
| S2_S3 | **Fold × model heatmap**: cell = seconds, colour = held-out fold RMSE, the running cell pulses unless reduced motion is set. Physics params per fold; advection decision chip; **stacker candidate leaderboard** (OOF RMSE per candidate, winner badge, 0.1% tie rule); epoch loss sparkline at debug level; final R², RMSE and coverage vs the 0.90 target |
| baselines | ΔMSE ± 2 SE forest growing per baseline |
| cv_curve | R² vs block size, gaining a point per partition; nested fold heatmap for the active partition |
| S4 | Per-lever dose–response small multiples drawing point by point (±SE ribbon, hollow when > 20% extrapolated) |
| S5 | Scenario feed table (name, mean Δ, likely range, % extrapolated); CMIP6 model ticker when fetching |
| S6 | Treatment × step checklist (model effects 8/8, DML, spillover, CATE, DR with bootstrap ticks, sensitivity, audit), with θ ± SE and verdict chips (consistent / magnitude differs / sign conflict) |
| S7 | Planned vs realised, Pareto points |
| finish | Documents written |

**Gantt.** A virtualised SVG span tree (run › stage › task › subtask, depth 4, collapsible).
- Bars grow live.
- Warning ticks and checkpoint flags sit on the time axis.
- Gaps of more than 45 s between heartbeats render as a hatched **"system sleep?"** band.
- The time axis toggles between wall clock and relative.

**Bottom tabs**

| Tab | Contents |
|---|---|
| Logs | Virtualised list of `log` and `warning` events plus raw stderr. Level, logger, stage and text filters; follow-tail; copy; download `events.jsonl`, `stdout.log`, `stderr.log`, or a `log.txt` rendering |
| Warnings | Grouped by code with counts. Each links to the relevant output view |
| Outputs | One row per `artifact` event (file, size, stage). Clicking opens the matching run tab, even mid-run |
| Checkpoints | Saves with done sets and bytes |
| Resources | RSS, CPU% and process-count sparklines; peak; free RAM and disk; a warning banner when RSS exceeds 80% of available memory |
| Config & provenance | `launch.json`, effective config, hashes, git dirty flag |

### 5.11 Studies and child runs

Study jobs render a **child matrix** above the generic tracker:
- **placebo**: 3 columns (grf / shift / rotate), each with a mini stage rail, child run link and verdict chip.
- **multiverse**: 10+ variant rows (status, R², seconds, scenario deltas). The scenario × variant heatmap and the priority-stability (τ / Jaccard) chart fill in as variants finish. Variants already done are listed before launch.
- **simcheck**: a generator × seed grid. Cell colour is effect share (diverging around 1); grey = pending, red = error, a dot = gate redraw. A live per-generator strip plot with IQR; ETA = mean seconds × remaining ÷ workers. Already-done pairs are shown before launch.
- **reproduce**: child rail plus the check list.

Child runs are written under `studies/<sid>/children/` and registered with `origin=study_child`, `study_id` and `parent_run_id`. Core receives `run_meta` (study_id, parent) so the link is explicit, not inferred from names. Name-pattern grouping is used only for imported legacy study folders and is marked "inferred".

### 5.12 Status Board, history, compare, notifications

- **Pipeline Status Board** (project overview). Rows are runs (mode chip, label, created, status). Columns are `S0, S1, S2_S3, baselines, cv_curve, S4, S5, climate, S6, S7 | planner, emulator, uncertainty, writeup | placebo, multiverse, simcheck, reproduce`. Each cell is a status chip: done (seconds), cached, running (%), failed, skipped(reason), stale, not run (+ launch). It is computed from SQLite plus `run_state.json` / `checkpoint.json` / manifest, so it covers imported CLI runs too. Clicking a chip opens the tracker span or the analysis view.
- **Run history** (`/runs`): a stage-duration history chart across runs, grouped by commit, to spot performance regressions. `manifest.timings_detail` (§11) feeds it for external runs.
- **Compare runs** (§6.8) overlays stage timings.
- **Notifications:** a toast on job end with headline metrics and links; an opt-in browser `Notification`; a tab-title badge `(2 running)`.

### 5.13 Resource monitor

`ResourceSampler` (foundation) samples each live job's process tree every 2 s with psutil: RSS sum, CPU%, process count and threads.
- Live samples go out as transient `resource` events.
- One sample every 10 s is stored in `resource_samples`.
- `jobs.peak_rss_mb` is maintained.
- The engine host is sampled the same way and reported in `engine.status`.
- `storage.low` is broadcast when workspace free disk drops below 2 GB.

**Preflight** (launch, engine open, heavy jobs) compares the estimated peak RAM with `psutil.virtual_memory().available` minus other live jobs' RSS:
- refuse when estimate + 1 GB > available;
- offer "stop job X" or "evict engine" in the dialog.

### 5.14 External (CLI) runs

- `run_core` always writes `run_state.json` when `write=True`.
- CLI runs get `--progress PATH` and `--job-id`. Alternatively the user sets `SPARC_PROGRESS=<file> python -m sparc.core run …`. The path is recorded as `run_state.events_path`.
- Settings → **watch roots**: the registry scans them every 10 s.
- A run whose `run_state.status == running` with `updated_utc` under 2 min old shows as **external_live**. Studio tails its `events_path` when present (full Mission Control). Otherwise it shows stage-level state from `run_state.json`.
- To reuse the tracker endpoints unchanged, the registry creates a read-only **pseudo-job** for each external live run: kind `run.external`, executor `external`, lane `none`, no job dir. Its tailer reads `run_state.events_path` (or synthesises stage events from `run_state.json` every 10 s). The pseudo-job ends when `run_state.status` leaves `running`, or becomes `interrupted` when `updated_utc` is more than 2 min old and the pid is gone.
- External runs cannot be cancelled from Studio (`409 not_cancellable`). Studio shows the pid and a hint instead.

---

## 6. Outputs

### 6.1 Output catalog — `sparc/core/catalog.py`

The catalog is a single Python table imported by Studio, the results-page builder and the tests. `/api/meta` serves it, so the client never hard-codes file names.

```python
@dataclass(frozen=True)
class OutputSpec:
    id: str                 # stable id, e.g. "predictions", "response_maps:{var}", "planner_hex:250"
    files: tuple[str, ...]  # globs relative to the root below
    label: str
    group: str              # model | effects | decisions | trust | docs | state | planner | studio
    produced_by: str        # "stage:S2_S3" | "post:planner" | "study:placebo" | "studio:scenario"
    view: str               # a run tab id (§3.2) or a viewer kind: table|markdown|json|map|download
    formats: tuple[str, ...]  # native + conversions: csv|geojson|geotiff|json|html|md|zip
    manifest_key: str | None  # manifest section mirroring it, if any
    dictionary: str | None    # DATA_DICTIONARY key
    root: str = "run"         # "run" (relative to run_dir) | "study" (relative to the study dir)
OUTPUTS: list[OutputSpec]
IGNORED: tuple[str, ...]    # never catalogued or flagged: ".sparc.lock", "*.tmp", "studio/**", "__pycache__/**"
DATA_DICTIONARY: dict[str, dict[str, tuple[str, str, str]]]   # file → column → (units template, sign meaning, description)
def units_for(cfg) -> dict          # resolves templates ("{target}", "{lever:Pct_Canopy}") for a config
def scenario_slug(name: str, taken: set[str] = frozenset()) -> str
    # configured-scenario id: lower-case; "−"/"-" before a number → "minus-", "+" → "plus-"; every other run of
    # non [a-z0-9] → "-"; strip "-"; append "-2", "-3"… on collision.
    # "Impervious Decrease −10" → "impervious-decrease-minus-10"; "Albedo Increase +0.1" → "albedo-increase-plus-0-1"
```

`scenario_slug` is the single definition of configured-scenario ids. It is used for layer keys (`sc:<slug>`), `configured:<slug>` references, API paths and the project's `headline_scenario`. It lives in `catalog.py` so the runs, engine and studies items share it.

**Catalogued outputs.** The test in §14.1 asserts that every file written by `run_core`, the post-run functions and the studies on the synthetic run matches an entry or `IGNORED`.

| Source | Outputs |
|---|---|
| Run core | manifest.json; influence.json; predictions.parquet; physics.json; baselines.json; cv_distance.json; response_<var>.parquet; response_curves.json; scenarios.json; scenario_deltas.parquet; **scenario_detail.npz** (new); climate.json; causal.json; **causal_cells.parquet** (new, §11 item 21); optimize.json; allocation.parquet; report.md, methods.md, model_card.md, environment.txt; **input_frame.parquet** (new, frame-input runs only, §11 item 22) |
| Run state | checkpoint.pkl, **checkpoint.json**, **run_state.json** |
| Post-run | emulator.npz/.json; uncertainty.json/.md; placebo.json (copied); planner/ (planner.json, planner_cells.parquet, hex_250m.csv, hex_500m.csv, hexagons.gpkg, geotiff/*.tif, logger_sites.csv, before_after_pairs.csv); results.html (legacy standalone page found in imported runs; Studio itself writes pages to exports) |
| Reproduction runs (run dir of the child) | reproduce.json |
| Studies (study dir, `root="study"`) | placebo.json/.md; simcheck.jsonl, simcheck_summary.json/.md; <variant>.json, <variant>_maps.npz, multiverse_summary.json/.md; benchmark.json/.md. The library functions return dicts; the Studio study kinds write the `.json`/`.md` files the CLI would have written (§8) |
| Studio (`studio/`) | launch.json (+ `launch.<n>.json`); results/; plans/; sweeps/; comparisons/; blobs/; engine/; cache/ (§12.2 of api.md). Exports never live here |

**Data dictionary.** Conventions are surfaced in every legend, tooltip and table header. A `README.txt` sidecar with the same conventions goes into every export.
- `delta` / `ΔT`: target units; **negative = cooler**.
- `cooling` / `benefit`: target units; **positive = cooler**.
- `own_effect_per_unit`: ∂ΔT per +1 lever unit; sign as in the data.
- `footprint_effect_per_unit`: target·cells per unit.
- Totals: °F·cells.
- Lever units: from `actionable[var].unit`.
- `id`: the `data.id` column.
- `logger_sites.cell` and `before_after_pairs.treated/control`: **row indices**. Studio converts these to ids and lon/lat on export.

**Output states.** `RunReader.outputs()` returns one of:
- **present**: the file exists, and the manifest references it or it is newer than its stage start.
- **stale**: the mtime is more than 60 s older than `manifest.created_utc` and the manifest lacks the section. Example: `providence_uhi_fast/causal.json`.
- **missing**: carries `{produced_by, action}`. Examples: "Planner pack needs people layers → [Fetch HRSL + WorldCover] then [Run planner pack]"; "No emulator → [Build emulator, ≈15 min]".
- **writing**: the producing stage is running and no `artifact` event has arrived yet.
- **partial**: some files of a multi-file output exist.

### 6.2 RunReader — `sparc/studio/runs/reader.py`

A lazily loaded `RunContext` per run, cached in a byte-capped LRU (default 768 MB across runs). Keys are `(run_id, manifest mtime_ns or run_state.updated_utc)`. Its attributes:

- **`cfg`**: from `launch.json` → `manifest.config` + `provenance.config_dir` → the import-time config path.
- **`manifest`**: the manifest **merged with per-stage files**. If the manifest lacks `causal`, `optimize`, `cv_distance`, `baselines`, `climate` or `response`, they are read from `causal.json`, `optimize.json`, `cv_distance.json`, `baselines.json`, `climate.json` or `response_curves.json`. Each section is tagged `{present, source: "manifest"|"file", stale, older_code}`. Older-code detection uses feature presence: provenance, literature, interval_diagnostics, mean_delta_se, frac_sigmoid, inflection_dose, qa.flags.
- **`data`**: `load_core_data(cfg)`, applying coarse/subsample from `launch.json.args`, `run_state.meta` or `manifest.qa`. Cost: about 0.3 s for full Providence. Runs made from an in-memory frame (`provenance.input_kind == "frame"`: placebo children) are rebuilt from their `input_frame.parquet` (§11 item 22) instead of `data.path`, because the frame differs from the file (shifted, rotated or added GRF layers). Older frame-input runs without it are marked "inputs not reproducible": their predictor layers are hidden, and their prediction layers still work.
- **`grid`**: `Grid.from_points(x_m, y_m, cell=cell_m)`. `cell_m` comes from `run_state.meta.cell_m`, `manifest.qa.cell_m` or `influence.json.cell_m`. The result is persisted to `<studio_dir>/cache/grid.npz` (ix, iy, ids, lon, lat, zone), keyed by `manifest.created_utc` or the `run_state` start time. Lon/lat come from a pyproj transform `data.crs → EPSG:4326` on `x_m / coord_scale`. They are null when there is no CRS.
- **`folds`**: from the manifest `cv` section, or `run_state.meta.cv` mid-run, via `cv.make_spatial_folds` (deterministic). The result is cross-checked against `predictions.fold`. The checkpoint is **never** unpickled for browsing.
- **`metrics_live`**: when the manifest has no metrics yet, R², RMSE, MAE and bias are computed from `predictions.parquet` (`target` vs `pred`), along with coverage of `[pi_lo, pi_hi]` and per-model OOF metrics from `oof_<model>`. This is what makes **Accuracy usable about 20 minutes into a 1.7 h run**.
- **Parquet reads** use pyarrow column projection. JSON is cached by `(path, mtime_ns, size)`.

### 6.3 Layer catalog and encoding (30 m grids on canvas)

`sparc/studio/runs/layers.py` generates the catalog **from the config and the files present**. Nothing is hard-coded to Providence. Each entry is a `LayerMeta`:

```
{key, group, label, unit, scale: "seq"|"div"|"cat", center, decimals, mult, zero_blank, labels?: [str],
 desc, sign_note, source: {file, column}, stats: {n, lo, hi, mean, p1, p2, p50, p98, p99}}
```

| Theme | Layers |
|---|---|
| Temperature | `obs` (target; div about the city median); `pred` (OOF); `resid` (target − pred; div about 0); `halfwidth` (pi_hi − pred); `halfwidth_adaptive`; `dist_train_m`; `oof_<model>` and `resid_<model>` for each base model |
| Inputs | Every predictor (actionable levers and roles first, labels and units from config); `zone` (cat) |
| CV design | `fold` (cat); fold-k class arrays served separately (0 train, 1 test, 2 buffer) |
| Effects (per lever) | `fp_<v>` footprint (div, `mult` 0.01 for albedo-type units); `own_<v>`; `own_sd_<v>`; `fp_sd_<v>`; `marg_<v>`; `A_<v>`; `d90_<v>`; `ds_<v>`; `infl_<v>`; `fitr2_<v>`; `headroom_<v>`; `cls_<v>` (cat: 0 censored/insufficient grey, 1 saturating, 2 linear, 3 S-shaped) |
| Causal (per treatment, when `causal_cells.parquet` exists) | `cate_<t>` (R-learner CATE per cell, div about 0); `mslope_<t>` (the model's per-cell adoption slope); `mslope_own_<t>` (own-cell slope) |
| Scenarios (configured) | `sc:<slug>` Δ per scenario column (`scenario_slug`, §6.1); plus `sc_sd:<slug>` and `sc_ex:<slug>` when `scenario_detail.npz` exists |
| Budget | `alloc_dose` (zero_blank); `alloc_delta` |
| Climate | Computed client-side: `obs + warming(stat) + adaptation Δ`; ≥T exceedance (cat) |
| Planner & people | `people`, `people_60_plus`, `people_under_5`, `lc_*`, `plantable_pp`, every `hot_days_ge_*` column (incl. `hd_95_ssp245_mid`) |
| Studio results | `res:<res_id>:{delta, delta_sd, extrapolation, realized_<var>, abs}`; `plan:<plid>:{dose, planned_benefit, closed_loop_delta}`; `cmp:<cid>:<a>__<b>` difference |

**Wire format.**
- `GET /api/runs/{rid}/layers/{key}.bin`: little-endian Float32 of length n in run row order, NaN = no data. Categorical layers are Uint8.
- Size: about 219 kB per layer at 54,701 cells.
- Geometry comes once from `GET /api/runs/{rid}/grid.bin`: packed `ix:i32, iy:i32, lon:f32, lat:f32, zone:i16`, with offsets in the `X-SPARC-Offsets` header. Ids come from `grid/ids.bin` (Int64) or `grid/ids.json` (string ids).
- Caching: finished runs send a strong ETag plus `Cache-Control: private, max-age=31536000, immutable`. Running runs send `no-store`.
- There is no u8/u16 quantisation on the wire. That encoding (`_enc`) survives only inside the standalone results page.

**Colour.** The OKLab LUT and the four ramps are ported verbatim from the results-page template (`scripts/results_page/template.html`, which moves to `sparc/core/results_page/template.html`, §6.9). There is one definition in Python (`/api/meta.palettes`, backend foundation) and one in TS (`palette.ts`, frontend foundation). The normative values are below, so each foundation tests against **this spec** rather than against the other's code.

| Ramp | Stops |
|---|---|
| seqLight | `#cde2fb #9ec5f4 #6da7ec #3987e5 #256abf #184f95 #0d366b` |
| seqDark | `#1d2f45 #184f95 #256abf #3987e5 #6da7ec #9ec5f4 #cde2fb` |
| divLight | `#0d366b #256abf #6da7ec #f0efec #f19a8f #d6403f #8a1f22` |
| divDark | `#9ec5f4 #3987e5 #1f4f8a #383835 #8f3434 #e66767 #f6b3ab` |

`lut(stops, n=256)`: equally spaced stops; linear interpolation in OKLab (Ottosson matrices as in the template); sRGB gamma; clamp; `Math.round` (half up). Golden LUT entries, verified against the template's JS:

| Ramp | [0] | [64] | [128] | [191] | [255] |
|---|---|---|---|---|---|
| seqLight | `#cde2fb` | `#85b6f0` | `#3987e5` | `#1e5caa` | `#0d366b` |
| seqDark | `#1d2f45` | `#1e5caa` | `#3a87e5` | `#85b6f0` | `#cde2fb` |
| divLight | `#0d366b` | `#4a89d6` | `#f0eeeb` | `#e57168` | `#8a1f22` |
| divDark | `#9ec5f4` | `#2c6ab6` | `#393835` | `#b94d4d` | `#f6b3ab` |

`studio-web/src/test/fixtures/palette.json` (frontend foundation) and the backend `test_meta.py` both encode this table.

Domain rules:
- clip to the 2nd–98th percentiles;
- diverging scales are symmetric about the centre (city median for temperatures, 0 for deltas);
- `zero_blank` paints values ≤ 0 as `--nodata`;
- `mult` applies display scaling;
- the user can lock the scale across layers.

Categorical maps use at most three colours plus grey.

### 6.4 Run hub tabs

Every chart is built from the SVG kit (§12.6). Each one has a **Table view**, SVG/PNG export, CSV copy and **Pin to Findings**. Every number carries units and the cooler/warmer wording.

**Overview**
- Header: name, mode chips, created, commit plus dirty flag, versions, n cells, grid, run dir, DEMO badge.
- KPI row: held-out R²; RMSE; 90% interval coverage vs target; half-width vs noise floor; headline scenario Δ with likely range (picked by id in project settings, never by regex); package Δ with causal band; mid-century SSP2-4.5 warming; realised budget cooling.
- Data-health flags.
- Stage timing bar (from events or `timings_s`; skipped stages greyed).
- Outputs availability grid; studies status chips.
- Generated caveats (`runs/caveats.py`, the same logic as the results page) and model-card limitations.
- The run's findings.

**Data & QA**
- QA flag list (warn/info).
- KPI tiles: input, dropped, clipped per column, grid fill, collisions, background (and its source), rounding noise floor.
- Fractional-part histogram (evidence of classed targets).
- Dose-scale table (dose, dose/sd, percentile).
- Predictor histograms (small multiples) and a correlation matrix heatmap.
- Per-zone counts, coarse aggregation stats, join provenance hashes.

**Accuracy**
- Per-model table: R², RMSE, MAE, bias, blend weight as an inline bar.
- Obs-vs-pred hexbin (canvas) with 1:1 line; residual histogram (brushable into a selection); residuals by zone and by fold (box).
- **Interval honesty** small multiples: coverage by fold, distance quartile and zone; global vs adaptive; 0.9 line.
- **Stacker panel**: candidate RMSE bars (from `lambda_scores`); 100%-stacked NNLS weights per fold; "residual kept in k/K folds"; val_mse_base vs with-residual; best_epoch; a "Spatial+ on: mgwr, gam" chip from `manifest.spatial_plus`.
- **Physics card**: parameters as mean ± sd with priors and plain labels; per-fold dots (L_m, a, s, a1); fit warnings.
- **Advection verdict** with per-fold ΔRMSE dots.
- **Forcing card**: date, hours, SW↓, LW net, wind arrow, station, checks.
- **CV design card**: blocks, buffer, test size per fold; linked fold map layer.

**Distance & baselines**
- Skill vs block size on a log axis: stack line with fold min–max band, base models and baselines in grey, main design marked, random-points row annotated "leaky reference".
- Baseline forest: ΔMSE ± 2 SE, zero line, stack-better side shaded.
- Verdict banner; share of blocks the stack wins.

**Influence**
- Range bars per predictor.
- Target correlogram with permutation band and directional toggle (0/45/90/135).
- Ring-kernel β by distance (small multiples; family and significance).
- Anisotropy rose with reliability marker; table.
- L prior, block size, residual range.
- Clicking a predictor draws its influence circle on the map.

**Response** (per lever)
- Dose–response line with ±1.96 SE ribbon; hollow points when more than 20% of cells are extrapolated; realised-dose secondary axis.
- Curve-shape stacked bar.
- Own vs footprint table with ratio; fold-means dots.
- Linked maps.
- The cell inspector rebuilds that cell's fitted curve from A, ds, inflection and model.
- **Literature panel**: published ranges as bars, SPARC ± SE plus the causal point, factor-of-2 band, citation/status tooltip.

**Scenarios (configured)**
- Dot-and-whisker: mean Δ with likely range, p10–p90 whisker, hollow when > 20% extrapolated, orange diamond for the causal band (flagged when the model is outside it).
- Ladder small multiples.
- Table: tier, has-folds, realised doses.
- Row actions: "Open in Lab", "Clone to edit".

**Climate**
- Warming per SSP × period: dot-range (median, p10–p90, min–max) with per-model strip.
- Model × projection table.
- Exposure grouped bars: share of cells ≥ T, today vs futures × adaptation variants (p10–p90 whiskers); threshold toggle.
- "% of median warming offset" gauge.
- Future-temperature map.
- Link to Lab › Climate for user scenarios.

**Causal**
- Per-treatment forest of θ_own, θ_nbr and θ_sum (±1.96 SE) beside the model slopes.
- Audit verdict chips.
- DR dose–response with CI ribbon, ESS and clipped share, with the **model's own-cell partial-dependence curve overlaid** (`causal.json treatments[t].model_effects.own_pd_curve`, §11 item 21; hidden with "not in this run (older code)" otherwise); CATE quantile box with BLP calibration.
- CATE and model-slope maps (`cate_<t>`, `mslope_<t>`, linked to the Map tab) when `causal_cells.parquet` exists.
- Sensitivity: E-value and its CI, robustness value, design effect.
- Controls, basis scale, hole-scale warning, nuisance R².
- DAG-audit table when present.

**Budget (S7)**
- KPIs: cells treated, mean dose, planned vs realised (labelled spillover non-additivity), cost, Gini, constraint, objective.
- Pareto chart: labelled open-loop; `n_treated` relabelled "segments", with a note. The caption is computed (2×/1× ratio).
- Dose and closed-loop maps; top-cells table.
- `optimize.json` of the form `{status: "no positive-benefit segments"}` renders as an explained empty state ("no cell has a positive cooling footprint for this lever"), not as missing.
- "Re-plan in Lab".

**Planner**
- Exposure grouped bars and person-mean temperature table.
- Hot-days small multiples (campaign-like vs lower bound).
- Equity quintiles with concentration-index chips.
- Plantable KPIs; sortable zone table; hex choropleth mode (250 / 500 m).
- Logger sites and before/after link lines as map overlays.
- GIS gallery: GeoTIFF thumbnails, hexagons.gpkg, CSVs with lon/lat added on export.
- "Re-run planner with package…".

**Uncertainty**
- Layered interval chart per scenario (estimation, specification, attribution, causal band, envelope) around the estimate, with a zero line and "excludes 0" badges.
- Climate spread table; sources with attach/detach.

**Docs**
- Rendered markdown (`marked`, raw HTML escaped) for report, methods, model_card, uncertainty, placebo, multiverse_summary, simcheck_summary and benchmark.
- report.md is labelled "as of run end" because it cannot be re-rendered from the manifest.
- **Regenerate methods & model card** runs the writeup job.

**Files**
- Tree with size, mtime, catalog id, state badge and produced_by.
- Raw download (Range supported); conversions (parquet → CSV with lon/lat; JSON → CSV for tabular sections; md → html).
- Parquet/CSV preview (first 200 rows).
- Data dictionary table.
- **Checkpoint card** with a guarded delete ("exact scenarios and the emulator build will need a refit").

**Provenance**
- Copyable hashes (input, joins, config, code); git commit plus dirty flag; platform.
- `environment.txt` with a diff against another run.
- Effective config; diff vs project config and vs DEFAULTS; `launch.json`.
- **Reproduce this run**.

### 6.5 Map explorer and analysis tools

`MapView` (frontend foundation) is used by the run hub, the Lab and the setup preview. Its parts:
- Theme → layer picker (searchable).
- Legend with an embedded brushable histogram; layer summary (n, mean, p10/median/p90, or category shares).
- **Pinned cell inspector**: every layer value for the cell, lon/lat, zone, per-lever curve, configured and Studio scenario deltas.
- Swipe A|B, synced side-by-side, difference mode (same grid only).
- Opacity; lock scale; manual range.
- Overlays: logger sites, before/after links, zone outlines, hex grid, CV block lines, influence circle, selection outline.
- Nice scale bar (1/2/5×10^k) and north arrow.
- PNG export with legend (rendered at 4×).
- Keyboard cell cursor (arrows, Enter pins) with an aria-live readout.

**Analysis tools drawer** (run-hub Map tab, also pinnable to Findings):
- **Region stats**: for the current selection: n, area km², residents; mean/sd/p10/p50/p90 inside vs outside for chosen layers; scenario means with jackknife SE where folds exist. "Save as region" turns it into a `SelectionSpec` that scenarios can reuse.
- **Breakdown**: any layer grouped by zone, quantile of any layer, hex (250/500 m), curve class or fold. Bar, box or strip chart, optionally people-weighted.
- **Relationships**: hexbin of layer X vs layer Y, with the selection highlighted, Spearman ρ and a binned-mean line. Brushing the plot selects cells.
- **Correlogram**: FFT ACF of any layer (`influence.fft_acf`) with a 19-permutation band.

### 6.6 Selections (shared by tools and the Lab)

`SelectionSpec` (pydantic in `sparc/studio/schemas/common.py`, TS in `src/api/types.ts`):

- **Primitives:**
  - `{kind:"all"}`
  - `{kind:"zones", values:[…]}`
  - `{kind:"polygon", crs:"EPSG:4326"|"run_xy_m", rings:[[[x,y],…]]}`
  - `{kind:"circle", center:[x,y], crs, radius_m}`
  - `{kind:"rect", min:[x,y], max:[x,y], crs}`
  - `{kind:"cells", ids:[…]}`
  - `{kind:"blob", blob_id}` — a brushed bitset uploaded as binary
  - `{kind:"hex", size_m:250|500, keys:[…]}`
  - `{kind:"filter", column, op:"<"|"<="|">"|">="|"=="|"between"|"in", value}`
  - `{kind:"top", column, frac?|k?, direction:"highest"|"lowest", within?}`
  - `{kind:"buffer", of, radius_m?|lever_range?}`
  - `{kind:"region", id}`
- **Combinators:** `{op:"and"|"or"|"minus", args:[…]}`, `{op:"not", arg}`.
- **Column namespaces:**
  - `predictor:<col>`
  - `layer:<col>` (people, lc_*, plantable_pp)
  - `pred:<target|pred|resid|halfwidth|dist_train_m>`
  - `response:<var>:<col>`
  - `planner:<col>`
  - `configured:<slug>`
  - `result:<res_id>:delta`

`sparc/studio/runs/selection.py::resolve(ctx, spec) -> np.ndarray[bool]` is pure numpy and takes a few ms on 54,701 cells:
- polygons: pyproj into the run frame, then `matplotlib.path.Path.contains_points` on cell centres;
- buffer: `scipy.ndimage.distance_transform_edt` on the raster;
- hex: `planner.hex_ids`.

The wire format for masks is a bitset: little-endian, LSB-first, base64.

Polygons in lon/lat make selections **portable across runs**: fast, coarse, full, open-data and multiverse children. Without a CRS, selections stay in `run_xy_m` and are flagged non-portable.

### 6.7 Downloads and exports

| What | Formats | How |
|---|---|---|
| Any catalog file | Native, with Range | `GET /api/runs/{rid}/files/raw` |
| Per-cell parquet | CSV (id, lon, lat if CRS, values), GeoJSON points (≤ 60,000 features, otherwise refused with a hint), parquet | conversion endpoint |
| Any layer, incl. results, plans, diffs | GeoTIFF (float32, data CRS, deflate, nodata NaN), CSV, GeoJSON | `GET /api/runs/{rid}/export/layer/{key}?fmt=` via `runs/export.py::export_layer`, built on `planner.export_geotiffs` (inline, < 2 s; written to a temp file and streamed, never stored) |
| Hexes | CSV, GeoJSON (EPSG:4326), GPKG | inline |
| Charts and maps | SVG / PNG (client side) | ChartFrame / MapView |
| Whole-run bundle | ZIP of selected outputs; checkpoint excluded unless ticked; contents manifest; README with units and sign | `export.bundle` job, streamed `FileResponse` |
| GIS pack | All layers as GeoTIFF + hexagons.gpkg + sites and pairs with lon/lat | `export.gis` job |
| Standalone results page | Single `results.html` | `export.page` job → `sparc.core.results_page.build_results_page` |
| Decision / plan / compare packs | §7.13 | `export.*_pack` jobs |
| Project report | Self-contained HTML (inline SVG and PNG maps), Markdown; PDF via print stylesheet | `export.report` job |
| Findings | Markdown ZIP (with images) or self-contained HTML | `export.findings` job |

Runs without a CRS disable lon/lat, GeoTIFF, GPKG and GeoJSON, and the UI says why.

Every export job (bundle, GIS, page, report, findings and the three packs) writes to `projects/<slug>/exports/<export_id>/`. `POST /api/exports` creates the `exports` row and the id first, then passes `export_id` in the job params.

### 6.8 Compare runs (`/p/:pid/compare?a=&b=`)

- Config diff tree with effective defaults hidden.
- Provenance chips: same data, config, code, grid.
- Metrics table with deltas; stage timings side by side.
- Scenario effects by name with likely ranges.
- Climate and causal summaries.
- `environment.txt` diff (added/removed/changed).
- Outputs only in A, or only in B.

When grids match (same `cell_m`, same ids):
- a difference map for any common layer;
- Kendall τ and top-decile Jaccard of priority layers (`fp_<v>`).

### 6.9 Standalone results page

The builder moves to `sparc/core/results_page/` (`__init__.py` exposing `build_results_page(run_dir, cfg, out, placebo_path=None) -> Path`, plus `template.html` as package data). It keeps the existing `collect()` output and fixes the known defects:
- folds come from `baselines.load_run`, not by unpickling the checkpoint;
- causal, cv_distance and optimize fall back to their stage files;
- the fold bitmask is per-fold u8 (no 8-fold limit);
- the HTML shell gets `<!doctype html>`, `<meta charset>`, viewport and `lang`;
- the Pareto caption is computed from the numbers;
- `hd_95_ssp245_mid` is listed;
- lat/lon is hemisphere-aware;
- the layer catalog is generated from config roles (not Providence names).

`scripts/results_page/build_page.py` becomes a 10-line shim. `check_page.py` points at the packaged builder.

### 6.10 Findings notebook

A **pin** stores:
- `view` (route id) and `url_state` (the full query, so it can be reopened);
- `title`;
- `note_md`;
- `snapshot` (the numbers on screen, as JSON);
- an optional `image` (chart SVG or map PNG, uploaded raw).

Findings belong to a project and optionally a run. They are ordered, annotated and filtered by run.

Export produces Markdown (ZIP with images) or self-contained HTML. Both carry a provenance block (run, commit, hashes) and the generated caveats. The exported numbers are the snapshotted numbers, so the report reproduces exactly what was on screen.

---

## 7. Simulation — the Scenario Lab

### 7.1 Concepts

| Concept | Meaning |
|---|---|
| **Lever** | An actionable predictor with bounds, unit, direction, doses, cost and dose scale (sd) |
| **Selection** | A set of cells (§6.6) |
| **Edit** | lever × mode × amount × selection |
| **Scenario** | A named, revisioned list of edits, plus reporting regions and a cost model |
| **Result** | Evaluation of one scenario revision on one run. Kinds: `preview` (emulator, never stored as a result), `exact` (engine), `configured` (S5 output), `plan`, `sweep_point` |
| **Plan** | A budget-optimised single-lever allocation |
| **Comparison** | 2–4 results with paired statistics |
| **Library** | All of the above, with lineage |

Once a run with a checkpoint exists, the project's primary CTA is **Open Scenario Lab**.

### 7.2 `ScenarioDoc` (pydantic; stored in SQLite and `results/<res_id>/spec.json`)

```json
{"id":"sc_3f9c2a", "revision":3, "parent_id":"sc_77aa01", "project_id":"p_…", "name":"Downtown cool corridor",
 "notes":"", "tags":["district"], "anchor_run_id":"20261001-090000-full-a1b2",
 "edits":[{"lever":"Pct_Canopy","mode":"fill_headroom","amount":0.5,"where":{"op":"and","args":[{"kind":"zones","values":[3]},
            {"kind":"filter","column":"layer:lc_built","op":">=","value":0.5}]},"label":"street trees"},
          {"lever":"Albedo","mode":"set","amount":0.35,"where":{"kind":"region","id":"rg_12"}}],
 "regions":{"Westside":{"kind":"polygon","crs":"EPSG:4326","rings":[[[-71.43,41.82],[-71.42,41.82],[-71.42,41.83],[-71.43,41.82]]]}},
 "costs":{"Pct_Canopy":{"per_unit":1.0},"Albedo":{"per_unit":100.0}},
 "options":{"clip_to_support":true,"mediators":true,"expert":false},
 "created_utc":"…","updated_utc":"…"}
```

`content_hash` is the sha256 of the canonical JSON of `{edits, regions, costs, options}`. Names, tags and notes are excluded.

### 7.3 Edit modes (compiler: `sparc/studio/engine/compile.py`)

Each edit becomes one core `Intervention`. Edits apply in list order, as in core. The same compiler is used for preview, cost estimates and exact runs.

| Mode | Becomes |
|---|---|
| `add a` | `Intervention(var, "add", a, where=mask)` |
| `set v` | `Intervention(var, "set", v, where=mask)` |
| `scale f` | `Intervention(var, "scale", f, where=mask)` |
| `floor v` | `per_point = max(v − x, 0)` on the mask ("raise canopy to at least 30%") |
| `ceiling v` | `per_point = min(v − x, 0)` ("cap impervious at 60%") |
| `fill_headroom f` | `per_point = f × plantable_headroom(canopy, layers, paved_share)`. Canopy role only; needs `planner.layers`; paved_share is editable |
| `to_percentile p` | Increase levers: `per_point = max(q_p − x, 0)`. Decrease levers: `min(q_p − x, 0)` |
| `per_cell ref` | `per_point` from `plan:<plid>` (dose signed by direction), `blob:<id>` (brush: Int32 indices + Float32 values), or `csv:<file>` (id, change; joined by id) |

**Compile output, per lever:**
- `n_cells`;
- mean and total requested;
- **predicted clamping**, using the engine's own `_bounds` rule (never push a cell further outside its bounds than its baseline);
- clipped share;
- estimated cost.

**Guardrails:**
- **Blocked** without `options.expert`:
  - an edit to a non-actionable predictor;
  - an edit to a mediator (e.g. NDVI) when its parents are also edited, because core silently overwrites it (`mediators.py:49–63`).
- **Warned:**
  - an amount beyond 1 sd of `qa.dose_scale` ("likely extrapolation");
  - an amount beyond the 99.5th percentile;
  - an empty selection;
  - `fill_headroom` without layers;
  - a direction opposite to the configured one;
  - cell-id or blob edits applied to a run with a different grid (resampled to the nearest cell centre).

**Scenario options.** `options.clip_to_support` and `options.mediators` are applied per request: the engine host saves `engine.clip_support` and `engine.mediators`, sets them from the options (`mediators: false` → `None`), runs the scenario, and restores them in a `finally`. Both are part of `content_hash`, so they are part of the cache key. Coupling rules always come from the run's config.

### 7.4 Library

**Statuses:** `draft` → `previewed` → `exact` (current) → `stale` (run checkpoint or core code changed) → `archived`.

**Revisions.** Patching the `edits`, `regions`, `costs` or `options` of a scenario that already has an exact result returns `409 conflict_revision`. The client then calls **fork**, which creates a new revision with `parent_id`. Results are never orphaned.

**Library views:** lineage tree, tags, search, and filters by status, run and tag.

**Pin to the compare tray.**

**"Configured in this run" group.** The S5 set, read-only. It is instant because the stored deltas exist. Actions: "Clone to edit", and "Re-run exactly" when `scenario_detail.npz` is absent (older runs).

**Templates** (`GET /api/scenario-templates`) are parameterised forms that emit `ScenarioDoc`s:
- **Cool roofs on built-up cells**: albedo set 0.35 where `lc_built ≥ 0.5`.
- **Shade the hottest X%**: canopy add 20 within the top X% of observed temperature.
- **Fill X% of plantable space**: `fill_headroom`.
- **Depave the most paved blocks**: impervious ceiling 60 where impervious ≥ 80.
- **Prioritise by footprint**: top-k by `response:<canopy>:footprint_effect_per_unit`.
- **Around sites**: buffer of uploaded points.
- **Configured package**: from `joint_scenarios`.

**Ladder.** "Make ladder [5,10,20,30]" on an edit forks N child revisions tagged `ladder:<id>` and batch-runs them. The library plots Δ vs dose with a likely-range ribbon.

**Promote to config.** Allowed only for whole-city `add` edits on actionable levers. It produces a single-lever `scenarios` ladder or a `joint_scenarios` package, and shows a YAML diff and the exact generated names (U+2212 for decreases). It writes a new project config version. The current run is never touched.

### 7.5 Preview (emulator, server-side)

`POST /api/runs/{rid}/preview` (inline, in the API process):
- loads `emulator.npz`/`.json` and the grid once per run (cached, a few MB);
- computes `ΔT = Σ_levers sparc.core.emulator.emulate(em_v, grid, dx_v)`, i.e. core numpy/scipy, with exact parity;
- runs under `threadpool_limits(1)`;
- is **single-flight, latest-wins per run**: a new request supersedes a queued one, and superseded requests return `409 superseded`, which the client ignores;
- takes 20–150 ms on 54,701 cells;
- returns a **packed binary body** (api.md §0.4): `delta` (float32[n] ΔT) and `edited` (the edited-cell bitset, raw bytes), with offsets in `X-SPARC-Offsets` and a small `X-SPARC-Summary` header: `{mean, edited_mean, n_edited, outside_share, trust, hatched, reasons[], request_seq}`. The mask travels in the body, not in a header, so large grids cannot overflow proxy header limits (Node's 16 KB default in the Vite dev proxy).

**Brushing** sends sparse `{lever: {idx: Int32 b64, val: Float32 b64}}` payloads, debounced at 120 ms on the client.

**Trust per lever**, from `emulator.json` validation:
- `good`: patch_pass_rate ≥ 0.9 and uniform rel_err ≤ 0.35;
- `rough`: otherwise;
- `none`: no emulator.

**Hatching and banner.** The preview is hatched and shows a banner when any of these hold:
- edited cells > 3,000;
- a `rough` lever edits > 1,000 cells;
- a city-wide edit uses a lever whose uniform rel_err exceeds 1.0. Example banner: "Preview unreliable at this scale (uniform albedo rel. error 246%) — run exact".

The UI states plainly that the preview is linear and never saturates or clips to support. Without an emulator, the Lab suggests **Build emulator** (≈1 min fast, ≈4–5 min per lever full); Exact still works.

### 7.6 Exact engine — the engine host (long-lived worker with an LRU of loaded runs)

**Process.** `python -m sparc.studio.engine.host --sock <ws>/engine/host.sock`
- Started on first need in its own session.
- Records `engine/host.json {pid, create_time, sock, authkey_hex}` (0600).
- Listens with `multiprocessing.connection.Listener(family="AF_UNIX", authkey=…)`, or `AF_PIPE` on Windows.
- Logs go to `engine/host.log`. stdout is never used for protocol.
- Survives a server restart. On start the server reconnects if `pid` + `create_time` match. Otherwise it removes the stale socket and starts the host lazily.
- Stopped on clean shutdown, and after 30 min with no loaded engines.

**LRU of `RunSession`s** (`OrderedDict[run_id, RunSession]`). Capacity is bounded by count (`engine_max_runs`, default 2) **and** memory (`engine_mem_budget_gb`, default `min(0.4 × RAM, 6)`).
- Eviction drops the least recently used session and calls `gc.collect()`.
- If host RSS still exceeds budget + 1 GB (allocator fragmentation), the host **recycles itself**: it finishes the current request, exits, and the server restarts it lazily and re-opens the MRU run.

**Loading a run** (`open` op → `sparc.core.session.open_run(run_dir, cfg, threads, base_fold=cached)`):
1. config from `launch.json` (else manifest);
2. `load_run`;
3. unpickle the checkpoint through a byte-counting reader that ticks "312/525 MB";
4. override `GWRFModel.n_jobs = threads` on every fold;
5. `MediatorChain.fit` under `limit_threads`;
6. `ScenarioEngine(..., base_fold=…)`, using `<studio_dir>/engine/base_fold.npy` (float64 K×n, keyed by checkpoint mtime+size+code sha; written on first open, saving ≈13 s afterwards);
7. `ResponseEngine` lazily;
8. layers;
9. `causal.json`;
10. drop `cv_distance` and causal `tau_hat` arrays;
11. `code_match` compares `checkpoint.json.code_sha256` with current core code.

Load times: ≈40–60 s and 1.6–2.5 GB on full Providence; ≈2 s fast. Loading is an `engine.open` job with progress. Before loading, the server checks memory: refuse if estimated RSS (≈3.5 × checkpoint bytes + 0.3 GB) + 1 GB exceeds available memory, offering to evict or stop job X.

**Protocol.**
- Request: `{op, request_id, job_id, job_dir, run_id, payload}`.
- Ops: `status`, `open`, `close`, `scenario`, `batch`, `rerun_configured`, `sweep`, `plan_verify`, `plan_frontier`.
- Reply: `{request_id, ok, result?, error?{type, message, traceback_tail}}`.
- The host wraps each request in `progress.job_scope(job_id, sink=<job_dir>/events.jsonl, cancel_file=<job_dir>/cancel)`. The request's job file gets fold ticks from the `fold_predictions` hook, and the standard tailer handles it.
- Requests are served **one at a time**. Engine-lane jobs queue in the JobManager.
- Each request runs under `limit_threads(t)`: t = 1 while a heavy job is running, else `engine_threads` (default 2).

**Engine cancel.** Touch the request's cancel file; the host raises `Cancelled` between folds and stays alive. After a 60 s grace, **Force stop** kills the host, evicting all engines. The server marks in-flight engine jobs `failed: "engine host exited"` and restarts the host lazily.

**Incompatible checkpoints.** If unpickling or the first evaluation fails with `AttributeError`/`ImportError`/`ModuleNotFoundError`, the engine status becomes `incompatible`, with the exception text and the action "Refit S2/S3 (resume S0–S3 from snapshot)" or "Use preview only".

**Cost.** ≈2 s per scenario on a fast run, 5–10 s coarse 60, ≈13 s full (K = 5 fold passes, each a tick).

### 7.7 Exact results and the inspector

**Stored in** `<studio_dir>/results/<res_id>/`:
- `spec.json` (compiled, with requested arrays' hash)
- `summary.json`
- `cells.parquet` (id, delta, delta_sd, extrapolation, realized_<var>…)
- `folds.npy` (float32 K×n, ≈1.1 MB full)
- `warnings.json`

**Cache key** = `(content_hash, run_id, checkpoint mtime+size, core code sha)`. An identical request returns the cached result instantly. A changed code sha or checkpoint marks old results `stale`; they stay viewable with a banner.

**Statistics** (`sparc/studio/engine/stats.py`; all from `delta` and `folds`):
- **City mean Δ** with jackknife SE `std_k(mean_i Δ_ki)·√(K−1)`. This is identical to core `ScenarioResult.summary()`.
- **Regions:**
  - auto: `edited cells`, `edited + ring at lever range`, `outside`;
  - every named region and every zone.
  - For each region: n, mean Δ ± SE (same jackknife on the masked columns), people-weighted mean, total Δ (°F·cells), share of cells cooled ≥ 0.1 and ≥ 0.5.
- **Spill**: share of total cooling outside the edited cells, and a **ring profile** (mean Δ ± SE in 30 m rings from the edited set out to 2 km, with lever influence ranges marked).
- **Extrapolation**: share of **edited** cells with score > 1. This avoids the city-wide dilution in core's summary.
- **Realised vs requested** per lever (mean and total over edited cells, clipped share) and mediator moves (e.g. "NDVI follows canopy +0.04").
- **Cost**: Σ |realised| × per_unit, and cooling per cost unit.
- **Causal check**: `pipeline.causal_crosscheck([summary], causal.json)` gives a band and "model inside/outside".
- **Uncertainty envelope**: `uncertainty.scenario_uncertainty()` on this summary, with the run's attached studies (attribution from simcheck for canopy-only scenarios). Specification comes from "check across runs" (§7.12) when available.
- **Impacts** (people layers required):
  - `planner.exposure_table` (residents ≥ T today and for SSP2-4.5 / SSP5-8.5 × 2041–2060 / 2081–2100 medians, with and without the scenario);
  - `planner.benefit_by_group` (density, 60+, under-5 quintiles and concentration index);
  - `planner.hot_days` person-days avoided (GHCN cache required; otherwise an `input.ghcn` job is offered);
  - climate offset share per SSP × period.
- **Preview vs exact** error (when both exist), to calibrate trust.

**Plain-language card** (default view). This is the expert toggle's counterpart; the toggle reveals SE, jackknife, n, folds and method notes.
- "**Cools the edited area by 0.62 °F** (likely range 0.44–0.80 °F). City-wide: 0.021 °F cooler."
- Confidence line, using the 95% interval est ± 1.96·SE:
  - "**Confident it cools**" when the interval excludes 0 on the cool side;
  - "Confident it warms" on the warm side;
  - "**Could be zero**" otherwise.
- Qualifiers, appended when they apply:
  - "partly outside observed conditions (23% of edited cells)" when > 20%;
  - "independent causal check disagrees";
  - "preview only — not verified".
- "What it buys": residents moved below 90 °F today and mid-century; cooling per $1k; share of mid-century warming offset.

### 7.8 Configured scenarios and paired SE

Core writes `scenario_detail.npz` (§11, item 12), so configured scenarios have folds, sd and extrapolation without a re-run.

**Paired difference** of A and B:

`SE(A−B) = std_k(mean_i(Δ_A,ki − Δ_B,ki))·√(K−1)`

This uses the shared fold models and is usually much tighter than independent SEs. Older runs without `scenario_detail.npz` offer "Re-run exactly for paired SE" (`engine.rerun_configured`).

### 7.9 Sweeps

`engine.sweep` job:
- inputs: lever, custom doses, optional region;
- runs `engine.run` per dose with the `where` mask;
- fits `response.fit_saturation` on the region-mean benefit and neighbourhood dose (`fit_neighbourhood`), and on the requested dose (`fit`, the curve's x axis; §19);
- output: curve (mean benefit ± SE, region and city, realised dose, neighbourhood dose, % extrapolated), fit (model, A, d_s, d90, axis), overlaid on the pipeline's `response_curves`;
- each point is stored as a `sweep_point` result.

### 7.10 Budget plans

**Inputs:**
- lever (actionable, with S4 responses);
- budget (log slider);
- cost: scalar, or a cost layer (any namespaced column, §6.6, or `csv:<project-relative path>:<column>` from an uploaded CSV joined by id);
- cap: plantable headroom (canopy role + layers, editable paved share) and/or region (cap 0 outside the selection);
- minimum dose per treated cell: `planned_allocation(min_dose=…)` drops allocations below it after the greedy pass. The freed budget is not re-spent, and the result reports it as `min_dose_dropped_cost`, labelled in the UI;
- objective: cooling, or people (HRSL smoothed at the lever range via `optimize.optimizer_layers`);
- equity: score source (share 60+, under-5, density rank, or a column — namespaced or `csv:<path>:<column>` — joined **by id**, normalised 0–1) and focus 0–1;
- Pareto multipliers.

**Planned mode** (inline, < 1 s, no engine). `VariableResponse` is rebuilt from `response_<var>.parquet` + `response_curves.json` + cfg, then `sparc.core.optimize.planned_allocation(...)` (§11, item 15).
- Output: dose map, planned total, **real cells treated** (not segments), Gini, cost, Pareto curve.
- Updates live on the slider (debounced 150 ms).

**Verify** (`engine.plan_verify`, ≈13 s full). A closed-loop exact run of the allocation as `per_point` gives realised vs planned with the spillover explanation. **Verify frontier** (`engine.plan_frontier`) runs exact at each multiplier, giving a closed-loop Pareto curve.

**Plan → scenario.** Creates a `per_cell` edit with `ref = plan:<plid>`. The scenario then joins the library, compare, climate and exports. Combining single-lever plans in one package is allowed and labelled "not jointly optimal".

**Field kit** (inline):
- ranked cell list: rank, id, lon, lat, zone, dose, planned benefit, closed-loop Δ, people, plantable headroom;
- `planner.logger_sites` on the plan;
- `planner.matched_controls` (treated = dose ≥ median positive dose) as the evaluation design.

The endpoint returns JSON with ids and lon/lat; the browser renders the CSV and GeoJSON downloads from it. The plan pack (§7.13) carries the same tables as files.

### 7.11 Climate × adaptation (`/r/:rid/lab/climate`)

**Inputs:** SSPs, periods, statistic (median / p10 / p90 / a single model), thresholds (free numeric; default from config) and adaptations (any exact results plus configured scenarios).

**Computation** (inline, < 1 s): `climate.summarize_projections(observed, factors, {name: Δ}, thresholds, to_units)` on the run's CMIP6 table (`climate.table` or the cmip6 cache), plus residents ≥ T when people layers exist.

**Charts:**
- warming dot-range with per-model strip;
- exposure grouped bars (p10–p90 across models);
- offset gauges ("cancels 41% of SSP2-4.5 mid-century median warming").

**Maps.** Future temperature and ≥ T exceedance are computed **client-side** (obs + warming statistic + Δ), so they are instant.

If no factors table exists, the page offers **Fetch CMIP6 change factors** (an `input.cmip6` job).

### 7.12 Compare and robustness

**Compare** (`/r/:rid/lab/compare?items=`). Items are 2–4 of: exact results, configured scenarios, plans, baseline. The view shows:
- small multiples on a locked diverging scale;
- A−B difference map; swipe; per-cell scatter;
- KPI table with pairwise differences and **paired SE** (or a "needs exact re-run" chip);
- regional breakdown, equity side by side, exposure today/futures, cooling per cost, causal bands.

Cross-run comparison of the same scenario is summary-level. Maps are shown only when ids align.

**Check across runs** (`scenario.across_runs`, **heavy-lane subprocess job**):
- evaluates a portable scenario on selected project runs (fast, coarse, full, open-data, multiverse children with checkpoints);
- loads one engine at a time **inside the job process** (not the engine host);
- **before start**, the dialog shows estimated time (Σ load + exact per run) and peak RAM (max engine RSS), and refuses runs without checkpoints with a reason;
- output: dot plot of mean Δ ± SE per run/variant, sign stability and spread. These feed the result's **specification** band.

### 7.13 Decision, plan and compare packs

Pack jobs belong to the engine work item and are **exact only**. A pack built from preview-only scenarios is stamped **DRAFT**, watermarked and has no SEs.

**Decision pack** (`export.decision_pack`):
- `brief.html`, self-contained:
  - title and place;
  - an auto-written summary paragraph from the numbers;
  - Δ and realised-dose map PNGs, rendered server-side with the shared OKLab LUTs via `matplotlib.image.imsave`;
  - KPI table with likely ranges;
  - regions, equity, exposure, climate offset, causal check;
  - assumptions and caveats (preview vs exact, extrapolation);
  - model-card limitations;
  - provenance hashes.
- `scenario.json`, `summary.json`.
- `cells.csv`: id, lon, lat, zone, delta, delta_sd, extrapolation, realised per lever.
- `delta.tif`, `realized_<var>.tif`.
- `hex_250m.csv`, `hex_500m.csv` (+ `hexagons.gpkg`).
- `README.txt` with units and sign conventions.

**Plan pack**: the decision-pack contents plus `field_list.csv`, `logger_sites.csv`, `before_after_pairs.csv` (with ids and lon/lat), and `pareto.csv`.

**Compare pack**: per-item summaries, paired differences, difference GeoTIFFs, `brief.html`.

### 7.14 Guardrails everywhere

- Every mean shows its likely range and the word cooler/warmer.
- Extrapolated cells are hatched.
- Preview is always labelled.
- Non-actionable edits need expert mode.
- Units come from config.
- Decrease scenarios display with U+2212, but ids are slugs.
- Drafts autosave every 2 s and on blur.
- Undo/redo (50 steps) is client-side.

---

## 8. Studies and post-run actions

All of these are tracked jobs. They run against the **parent run's launch snapshot config**, never the current project config. They write outputs under `studies/<sid>/` or into the run directory (post-run actions), take the per-run write lock where they mutate the manifest, and link child runs explicitly.

| Kind | Type | Library call | Lane | Writes | Notes |
|---|---|---|---|---|---|
| `post.baselines` | action | `baselines.baselines_for_run(run_dir, cfg, models)` | medium | baselines.json; manifest via `update_manifest` | Model subset selectable |
| `post.planner` | action | `planner.planner_pack(run_dir, cfg, package, thresholds, hex_sizes, export, cache_dir=<ws>/cache)` | medium | planner/*; manifest | Requires `planner.layers`; GHCN optional (network) |
| `post.emulator` | action | `emulator.emulator_for_run(run_dir, cfg, n_patches)` | heavy | emulator.npz/.json | Requires checkpoint, checked when the job starts (so it can wait in a Launch "then" chain) |
| `post.uncertainty` | action | `uncertainty.uncertainty_report(run_dir, multiverse_dir, simcheck_dirs, placebo_path, real_r2_gate)` | medium | uncertainty.json/.md, placebo.json; manifest | **Auto-enqueued** when an attached study finishes (setting `auto_uncertainty`, default on) |
| `post.writeup` | action | `writeup.methods_markdown` / `model_card_markdown` on the merged manifest | medium | methods.md, model_card.md | report.md is not re-rendered (frozen at run end) |
| `study.placebo` | study | `placebo.run_placebo_suite(cfg, kinds, coarse, seed, grf_range_m, children_dir, resume=True, run_meta)` | heavy | study dir: placebo.json/.md (written by the kind with `placebo_markdown`, as the CLI does) + 1 child run per kind | Needs canopy/impervious roles |
| `study.simcheck` | study | `simcheck.run_simcheck(cfg, design, out_dir, coarse, epochs, workers, threads)` | heavy | simcheck.jsonl, simcheck_summary.json (by core); simcheck_summary.md (by the kind, `simcheck_markdown`) | Resumable; `simcheck_merge` across studies is inline |
| `study.multiverse` | study | `multiverse.run_multiverse(cfg, out_dir, variants, coarse, workers, threads, extra_variants, run_meta)` | heavy | <variant>.json/_maps.npz, multiverse_summary.json (by core); multiverse_summary.md (by the kind, `multiverse_markdown`) + child runs | Custom variants as dotted overrides |
| `study.reproduce` | study | `reproduce.reproduce(run_dir, stages, tol_r2, tol_effect, config_dir, out_dir, run_meta)` | heavy | child run + reproduce.json (in the child run dir) | Refuses frame-input runs. `config_dir` = the parent's `launch.json.config_dir`, else its import-time config dir (the full Providence example has no `provenance.config_dir`) |
| `study.benchmark` | study (project) | `diagnostics.run_benchmark(seed, spatial_plus_ab, epochs, n)` | heavy | benchmark.json/.md (written by the kind with `benchmark_markdown`) | No run needed |

All study and post-run writes use `runio` atomic helpers. For `study.simcheck` and `study.multiverse`, `workers × threads` must not exceed `threads_heavy` (`422 validation` otherwise), because the pool runs inside the single heavy slot.

- **Validation tab** (`/r/:rid/validation`): one card per kind. Each card shows status (not run / queued / running / done / stale vs run / failed), estimated cost, a launch form, and the result chart:
  - baselines forest;
  - placebo verdict table and Δ per sd vs real;
  - simcheck share strip, coverage bars, false positives, bias-correction card;
  - multiverse heatmaps and R² by variant;
  - uncertainty layered intervals;
  - literature panel;
  - reproduce checklist;
  - benchmark effect shares;
  - for synthetic projects, **Truth vs recovered** (`GET /api/runs/{rid}/truth`). It compares `truth.json` (§9.2) with the run, one row each:
    - canopy +d city-mean ΔT: `derived.true_scenarios[name]` vs the configured scenario's mean ± SE, with share = model ÷ truth;
    - mean canopy footprint per pp: `derived.true_footprint_mean` vs `mean(footprint_effect_per_unit)`;
    - relaxation length: `L` vs the physics `L_m` mean ± sd;
    - canopy influence radius: `influence_radius_90` vs the S1 canopy range;
    - noise floor: `noise_sd` vs the held-out stack RMSE (a sanity bound: RMSE should not fall below the planted noise; the synthetic target is not rounded, so `qa.target_rounding_noise_sd` does not apply).
- **Attach/detach.** A study can be attached to a run, which feeds its uncertainty. Placebo, simcheck and multiverse studies launched from a run are attached to it when they are created. Attaching a finished study re-enqueues `post.uncertainty`, and so does detaching one when the run already has an uncertainty report (so the report drops it).
- **Stale.** A study is stale vs its run when the run's checkpoint fingerprint differs from the one recorded when the study ran.

---

## 9. Project setup wizard

### 9.1 Creating a project

`POST /api/projects` with `template`:

- **`blank`** — creates an empty config with sensible DEFAULTS.
- **`synthetic_demo`** (§9.2).
- **`providence_example`**:
  - copies `brown4.csv`, `configs/forcing/providence_2020-07-29.json`, `configs/climate/providence_cmip6_tasmax_jja.csv` and `configs/layers/providence_layers.parquet` into the project, and rewrites `configs/core_providence.yml` paths relative to the project dir;
  - sources: a repo checkout when present, else the packaged copy in `sparc/studio/examples/providence/` (`brown4.csv.gz` ≈1.6 MB, decompressed on create; the other files as-is);
  - option `import_existing_runs` (default true when `output/core/providence/` exists): registers `providence_uhi`, `providence_uhi_fast` and the `placebo`, `simcheck*` and `multiverse` study folders **in place** (`origin=imported`). The projects item does this through the cross-item contracts of §10.2: `sparc.studio.runs.registry.import_run(...)` and `sparc.studio.studies.service.import_study_dir(...)`, imported lazily. When either module is not installed yet, the project is still created and the response carries a warning ("run import unavailable in this build").
- **`existing_config`** (`POST /api/projects/import`): imports a YAML with paths absolutised, and optionally `copy_data`, run dirs and study dirs.

### 9.2 Synthetic demo city (with a CRS)

- **One generator for everything.** All files below come from the new core function `sparc.core.synthetic.write_demo_project(out_dir, n=96, seed=0) -> {config_path, files, truth}` (§11 item 20). The `synthetic_demo` template, `scripts/make_studio_fixtures.py` and the tests all call it. A demo project created with the same `n` and `seed` as the committed fixture is therefore byte-identical in its data, which the replay e2e relies on (§14.5).
- Data: `make_synthetic_city(n, seed)` (default `n=96`; the committed fixture uses `n=40`; the real-run e2e uses `n=48`).
- Location: coordinates offset to a **fictional location** (EPSG:32619, x + 300,000 m, y + 4,630,000 m). The project is flagged `demo: true`, and every page shows a **DEMO DATA** badge.
- Files written:
  - `data/city.csv` and `data/city.truth.json`;
  - `inputs/layers/demo_layers.parquet`: people, people_60_plus, people_under_5 and lc_* derived deterministically from canopy/impervious;
  - `inputs/climate/demo_cmip6.csv`: 6 pseudo-models × 4 SSPs × 3 periods in the `cmip6_change_factors` schema;
  - `config.yml`, built from `synthetic_city_config()` plus: `data.path`, `data.crs`, physics roles; actionable canopy/impervious/albedo; one ladder per lever plus a package; `causal.treatments: [canopy]`; climate (table); optimize (canopy, budget); planner.layers; report title "Synthetic city (DEMO)".
- `truth.json` = the generator's `truth` dict plus `derived: {true_scenarios: {<configured canopy scenario name>: city-mean ΔT from SyntheticCity.true_response(d)}, true_footprint_mean, n, seed}`. It drives the "Truth vs recovered" validation card (§8).
- Generation is inline (< 2 s).

### 9.3 Steps (`/p/:pid/setup/:step`)

Steps are non-linear tabs with completion dots computed from server-side readiness.

1. **data**
   - Drop a CSV or parquet. Upload is a streamed `PUT` to `data/`.
   - Header preview: per-column dtype, nulls, min/max, unique count, sample.
   - **Suggested mapping** (`columns/suggest`): target, id, x/y, zone, coord unit (feet vs metres from the coordinate range), CRS hint (EPSG search box with a centroid inset).
   - Further settings: target units; background (median / number / column); subsample/coarse; join tables (path, on, right_on).
   - **Check data** runs S0 inline (`data/check`). It shows QA flags as badges, grid shape, cell, fill fraction, dropped/clipped counts, background, noise floor and a preview raster of any column via `MapView`.
2. **levers**
   - Predictors (multi-select of numeric columns), encodings (categorical / circular), `qa.clip`.
   - Per predictor, an actionable toggle with min/max, unit, dose chips, direction and cost.
   - **Dose-scale table** from the check: dose / sd and percentile reached; > 1 sd is amber.
   - Coupling sums.
   - Mediators mini-editor: parents, context, monotone signs.
3. **physics**
   - Role mapping dropdowns for the six roles. Missing canopy or impervious raises a warning listing the features that need them.
   - Window, SW↓, LW net, wind.
   - Forcing file link showing its checks and values.
   - Advanced drawer: shade_form, priors, albedo_map, tau_s, L_max_m, v_max_m, fit/select advection, max_iter, num_threads.
4. **inputs** — four cards. Each shows status, file summary and the network hosts used, with a host pre-check (`/api/system/netcheck`). Each launches a tracked network job and ends with **Link into config** (YAML diff preview):
   - **Campaign forcing**: date, hours, tz, station (nearest ISD list from the cached `isd-history.csv`), wind source. If the station index is not cached yet, the picker offers **Fetch station list** (an `input.stations` network job, ≈3 MB); it never downloads inside a request. The result shows ERA5 vs station side by side and the checks as warnings. Links `physics.forcing`.
   - **People & land cover**: HRSL + WorldCover. Shows total residents and a preview map. Links `planner.layers`.
   - **Open predictors**: months, max cloud, S2 tiles. Per-scene progress. Shows the agreement table and scatter (only when the project already maps physics roles). **Create _open project** creates a new project `<name>_open` whose config joins the parquet (`data.join` with `right_on: id`) and uses the `open_*` predictors with roles mapped to them.
     - **Bootstrap mode (any city).** A project with only target, id, x/y and a CRS (no predictors yet) can still run this job. The kind loads S0 geometry with an in-memory copy of the config whose `predictors` is `[<target>]` and whose roles are empty. That copy is used only to build the grid and ids and is never saved. Agreement is skipped. With **Use as this project's predictors**, the join and `open_*` predictors are written into the same project instead. This closes the "features needs predictors" gap for a brand-new city.
   - **CMIP6 change factors**: site (default `climate.site` or the data centroid), SSPs, periods (default 2021–2040, 2041–2060, 2081–2100), baseline (default 1995–2014), months, variable, models, workers. Per-model progress list. Links `climate.table`, and `climate.periods` when non-default periods were fetched.
5. **scenarios** — configured ladders and joint packages, showing the exact generated names (U+2212). This step also notes that richer scenarios live in the Lab.
6. **analysis** — causal (treatments, confounders, excluded mediators, contrasts, DAG audit), climate (enabled, table vs live, site pin, SSP and period chips, months, thresholds, adaptation picker populated from scenario names), optimize (variable, budget, cost, plantable, objective, equity column, focus), planner (layers, paved share, GHCN station). Advanced: influence, cv (block, buffer, seed, distance curve, baselines list), models (toggles, Spatial+), stacker (epochs, tune_lambda, coverage), `response.clip_to_support`.
7. **about** — report title, place, area, limitations, caveats; headline scenario (by slug); cost model defaults.

**Readiness spine** (`GET /api/projects/{pid}` `readiness[]`). Each row has ok / warn / missing / n/a plus an action:
- data file
- columns mapped
- levers (n)
- physics roles (k/6)
- campaign forcing
- climate table
- people layers
- config valid
- runs (fast / coarse / full)
- emulator on active run
- studies (baselines, placebo, multiverse, simcheck, uncertainty)

### 9.4 Config editor and validation (`/p/:pid/config`)

- **YAML round trip** keeps the `core:` block. Comments are **not** preserved; the editor says so, and every save writes a `config_versions` row. Saves use optimistic concurrency (`If-Match: <version>` → 409 on conflict).
- **Pydantic `CoreConfigModel`** (`sparc/studio/projects/config_schema.py`):
  - covers DEFAULTS **plus** keys missing from DEFAULTS: `planner.*`, `report.*`, `response.clip_to_support`, physics extras, extended influence keys, the `dag_audit` dict form and `output.scenario_detail`;
  - uses `extra="allow"` (unknown keys pass through, with an info notice);
  - adds `x-ui` hints (group, advanced, unit, help, enum);
  - a test diffs it against `sparc.core.config.DEFAULTS`.
- **`validate_deep`** returns issues `{level: error|warn|info, path, code, message, fix?}`. Checks:
  - files exist; header columns exist and are numeric (sample of 50k rows);
  - target/id/x/y set; coord_unit valid;
  - CRS present when climate site fallback, open data, GeoTIFF or the people objective need it;
  - predictors non-empty;
  - levers are predictors; doses within bounds and including 0;
  - direction valid; mediator names and parents are predictors;
  - roles map to predictors;
  - scenario and joint variables are actionable;
  - causal treatments, confounders and exclusions are predictors;
  - `climate.table` exists when `source=table`;
  - `optimize.variable` is actionable; `objective=people` or `plantable` needs `planner.layers`;
  - `equity_column` exists in the data;
  - forcing JSON parses; thresholds are numeric;
  - fast/coarse override preview.
- **Diff vs DEFAULTS**, **history**, and the **impact preview** (§5.9).

---

## 10. Backend architecture

### 10.1 Package layout

New package `sparc/studio/`. It imports nothing from `sparc/server` or the legacy pipeline. The work-item owner of each file is given in brackets (§17).

```
sparc/studio/
  __init__.py  __main__.py                         [foundation]  python -m sparc.studio
  cli.py          main(argv) / add_studio_arguments  [foundation]
  app.py          create_app(settings)               [foundation]  lifespan, middleware, static SPA, router registry
  settings.py     StudioSettings                     [foundation]
  security.py     token, cookie, Host/Origin, safe_path, upload allowlist   [foundation]
  errors.py       ApiError + envelope handlers       [foundation]
  db.py           sqlite3 WAL, full DDL (§10.6), migrations, DAO helpers     [foundation]
  workspace.py    paths, ids, slugs, atomic writes   [foundation]
  events.py       pydantic event union v1, EventHub, global ring buffer      [foundation]
  sse.py          StreamingResponse SSE encoder       [foundation]
  schemas/common.py  Job, Page, ErrorEnvelope, LayerMeta, GridMeta, SelectionSpec, PlanNode   [foundation]
  meta.py         stages, palettes, job-kind listing, output catalog (lazy)  [foundation]
  routes/__init__.py  ROUTER_MODULES registry         [foundation]
  routes/system.py    auth, health, meta, settings, system, storage, netcheck, shutdown   [foundation]
  routes/jobs.py      jobs CRUD, tracker, events, logs, metrics, cancel/kill/retry, queue, timings   [foundation]
  routes/stream.py    global + per-job SSE            [foundation]
  jobs/kinds.py       JobKind registry + KIND_MODULES [foundation]
  jobs/manager.py     JobManager (queue, lanes, locks, spawn, cancel, reattach, chains)   [foundation]
  jobs/executors.py   Executor protocol, ProcessExecutor, register_executor   [foundation]
  jobs/worker.py      subprocess entry point            [foundation]
  jobs/tailer.py      JobTailer                         [foundation]
  jobs/tracker.py     reduce(state, event) projection   [foundation]
  jobs/eta.py         cost model + calibration          [foundation]
  jobs/resources.py   ResourceSampler, preflight memory/disk   [foundation]
  jobs/testkinds.py   test.* kinds (enabled by SPARC_STUDIO_TEST_KINDS=1)   [foundation]
  jobs/replay.py      replay runner (SPARC_STUDIO_RUNNER=replay:<dir>)    [foundation]
  projects/  service.py config_service.py config_schema.py validate.py templates.py readiness.py files.py inputs.py kinds.py   [backend-projects]
  examples/providence/  core_providence.yml brown4.csv.gz forcing/… climate/… layers/…   [backend-projects]
  routes/projects.py  routes/inputs.py                                          [backend-projects]
  runs/      registry.py reader.py layers.py grid.py selection.py stats.py compare.py caveats.py export.py kinds.py statusboard.py   [backend-runs]
  routes/runs.py  routes/layers.py  routes/compare.py                          [backend-runs]
  engine/    host.py client.py executor.py compile.py preview.py stats.py store.py kinds.py   [backend-engine]
  scenarios/ library.py templates.py plans.py climate.py sweeps.py impacts.py packs.py   [backend-engine]
  routes/lab.py  routes/plans.py                                                [backend-engine]
  studies/   service.py kinds.py views.py                                       [backend-studies-exports]
  exports/   bundle.py gis.py report.py narrative.py findings.py kinds.py        [backend-studies-exports]
  routes/studies.py  routes/exports.py  routes/findings.py                      [backend-studies-exports]
  static/    committed Vite build (index.html, assets/*-<hash>.js|css|woff2, BUILD_INFO.json)   [integration]
sparc/core/progress.py  sparc/core/runio.py                                      [core-progress]
sparc/core/session.py                                                           [backend-engine]
sparc/core/catalog.py                                                           [backend-runs]
sparc/core/results_page/{__init__.py, template.html}                            [backend-studies-exports]
```

### 10.2 Registries (how parallel items plug in without editing shared files)

**Routers.** `sparc/studio/routes/__init__.py` defines

```python
ROUTER_MODULES = ["system", "jobs", "stream", "projects", "inputs", "runs", "layers", "compare", "lab",
                  "plans", "studies", "exports", "findings"]
```

`create_app` imports each `sparc.studio.routes.<name>` and mounts its `router` under `/api`.
- A module that does not exist yet (`ModuleNotFoundError` whose `.name` is exactly that module) is skipped with a debug log.
- Any other import error is raised.

**Job kinds.** `jobs/kinds.py` defines `KIND_MODULES = ["sparc.studio.projects.kinds", "sparc.studio.runs.kinds", "sparc.studio.engine.kinds", "sparc.studio.studies.kinds", "sparc.studio.exports.kinds"]`, loaded with the same tolerance rule. A feature item registers its kinds with:

```python
@job_kind("post.planner", lane="medium", executor="process", label="Planner pack", params=PlannerParams,
          needs_run=True, needs_checkpoint=False, locks_run=True, network_hosts=("noaa-ghcn-pds.s3.amazonaws.com",),
          long=True, estimate=planner_estimate)
def run_planner(ctx: JobContext, params: PlannerParams) -> dict: ...
```

`JobContext` provides: `job_id, job_dir, workspace, project_dir, run_dir, studio_dir, study_id, study_dir, threads, cache_dir, db (read-only helper), run_config() (the run's launch-snapshot CoreConfig, §4.3 resolution order), emit_result(dict)`.

**Kind modules import lazily.** The server imports every kind module to list kinds and params schemas. A kind module therefore imports heavy libraries (`sparc.core.session`, scenario/response engines, anything that may pull torch) inside the job function, never at module level.

**Server-side hooks.** Workers never write SQLite (§5.6). `@job_kind` takes optional hooks that the foundation calls **in the server process**:
- `on_event(sctx, job, event)` for selected event types (`run.dir`, `run.start`, `artifact`), used to register study child runs live;
- `on_finish(sctx, job, result)` after the final status, used to insert `results` / `plans` / `sweeps` / `exports` rows, attach studies and enqueue `post.uncertainty`.

`sctx` carries the writer DB handle, the EventHub and the registry. Hooks must be fast (< 50 ms) and idempotent, because reattach may replay them.

**Cross-item function contracts** (called lazily with `importlib`; a missing module is tolerated with a logged warning, so parallel items never block each other):

| Caller | Callee (owner) | Signature |
|---|---|---|
| projects (Providence import, `POST /api/projects/import`) | `sparc.studio.runs.registry` (runs) | `import_run(dir, project_id, config_path=None, trust_pickles=False) -> RunSummary` |
| projects | `sparc.studio.studies.service` (studies) | `import_study_dir(dir, project_id, kind=None, target_run_id=None) -> Study` |
| runs (Status Board study/post cells) | the `studies`, `study_links` and `jobs` tables | DB contract only |
| engine, studies (narratives, packs, reports) | `sparc.studio.runs.caveats` (runs) | `caveats_for(run_ctx) -> list[str]` |
| engine, studies | `sparc.core.catalog.scenario_slug` (runs) | §6.1 |

**Executors.** `jobs/executors.py` defines `Executor` (`async submit(job)`, `async cancel(job)`, `async kill(job)`, `async reattach(job)`). Two implementations:
- `ProcessExecutor` (foundation)
- `EngineExecutor` (backend-engine), registered at import via `register_executor("engine", EngineExecutor(...))`

**Meta.** `/api/meta.output_catalog` comes from `sparc.core.catalog.OUTPUTS`, imported lazily, or `[]`. `stages` comes from `sparc.core.pipeline.STAGE_NODES`, imported lazily, with a static fallback list.

### 10.3 Job kinds (full list)

Params schemas are in `api.md` §8.

| Kind | Lane | Executor | Owner | Locks run |
|---|---|---|---|---|
| `test.sleep`, `test.events`, `test.fail`, `test.ignore_sigterm`, `test.pool` | heavy | process | foundation | — |
| `input.forcing`, `input.layers`, `input.features`, `input.cmip6`, `input.ghcn`, `input.stations` | network | process | projects | — |
| `run.core` (launch and resume) | heavy | process | runs | ✓ |
| `run.external` (read-only pseudo-job for a live CLI run, §5.14; never queued or cancellable) | none | external | runs | — |
| `post.baselines`, `post.planner`, `post.uncertainty`, `post.writeup` | medium | process | studies | ✓ |
| `post.emulator` | heavy | process | studies | ✓ |
| `study.placebo`, `study.simcheck`, `study.multiverse`, `study.reproduce`, `study.benchmark` | heavy | process | studies | — |
| `export.bundle`, `export.gis`, `export.page`, `export.report`, `export.findings` | medium | process | studies | — |
| `engine.open`, `engine.scenario`, `engine.batch`, `engine.rerun_configured`, `engine.sweep`, `engine.plan_verify`, `engine.plan_frontier` | engine | engine | engine | — |
| `scenario.across_runs` | heavy | process | engine | — |
| `export.decision_pack`, `export.plan_pack`, `export.compare_pack` | medium | process | engine | — |

### 10.4 Scheduler

- **Queue.** Persisted in `jobs`. Order is priority (desc), then `created_utc`. The queue can be paused.
- **Lanes:**
  - heavy: 1 slot (setting `heavy_slots`)
  - medium: 1
  - network: 2
  - engine: 1 (the host serves one request at a time)
  - Inline endpoints are not jobs.
- **Per-run write lock.** Kinds flagged `locks_run` are serialised per run (`run_locks` table plus core `runio.run_lock` fcntl on `<run_dir>/.sparc.lock`). A blocked job shows `blocked` with the reason "waiting for post.planner on this run".
- **Chains.** A job may have `after_job_id`. It starts when that job **succeeds** and is cancelled if it fails. Launch uses this for the "then run" chain: planner → emulator → uncertainty → writeup.
- **Auto jobs.** When an attached study finishes and the setting `auto_uncertainty` is on, `post.uncertainty` is enqueued.
- **Preflight** before a `heavy` or `engine.open` job transitions to starting:
  - memory (§5.13);
  - disk (checkpoint estimate + 1 GB);
  - for kinds with `network_hosts`: an optional host check (cached 10 min);
  - `needs_checkpoint` → `checkpoint.pkl` exists.
  - A failure leaves the job `blocked` with `{reason, actions[]}`, or fails it when not recoverable.

### 10.5 Thread budget

The total budget is `cpu_count` (setting `thread_budget`).

| Consumer | Threads |
|---|---|
| Heavy job | `threads_heavy` (default `max(1, cpu_count − 1)`) |
| Medium jobs | 1 |
| Network jobs | 1 |
| Engine requests | `engine_threads` (default 2), reduced to 1 while a heavy job runs |
| Preview and inline ops | 1 (`threadpool_limits(1)`) |

Workers apply the thread count through env vars, `progress.set_threads`, torch, and the GWRF `n_jobs` override. The scheduler refuses a configuration whose concurrent sum exceeds the budget by more than one thread. This avoids the measured 82–368 s mediator fits under oversubscription.

### 10.6 SQLite schema (owned by the foundation; one file, `PRAGMA user_version` migrations)

`studio.sqlite` runs in WAL mode with `synchronous=NORMAL`. A single writer connection is guarded by an asyncio lock and used via `to_thread`; readers use per-thread connections. All timestamps are ISO-8601 UTC strings. All JSON columns are TEXT.

```sql
CREATE TABLE projects (id TEXT PRIMARY KEY, slug TEXT UNIQUE NOT NULL, name TEXT NOT NULL, dir TEXT NOT NULL,
  config_path TEXT NOT NULL, template TEXT, demo INTEGER DEFAULT 0, active_run_id TEXT, archived INTEGER DEFAULT 0,
  meta_json TEXT DEFAULT '{}', created_utc TEXT, updated_utc TEXT);
CREATE TABLE config_versions (project_id TEXT, version INTEGER, yaml TEXT NOT NULL, note TEXT, saved_utc TEXT,
  sections_json TEXT, PRIMARY KEY (project_id, version));
CREATE TABLE runs (id TEXT PRIMARY KEY, project_id TEXT, run_dir TEXT UNIQUE NOT NULL, studio_dir TEXT NOT NULL,
  origin TEXT NOT NULL, parent_run_id TEXT, study_id TEXT, label TEXT, mode TEXT, coarse_m REAL, stages_json TEXT,
  status TEXT NOT NULL, created_utc TEXT, finished_utc TEXT, n_points INTEGER, r2 REAL, rmse REAL, coverage REAL,
  checkpoint_bytes INTEGER, checkpoint_done_json TEXT, fingerprint TEXT, code_sha TEXT, config_sha TEXT, input_sha TEXT,
  git_commit TEXT, git_dirty INTEGER, has_emulator INTEGER DEFAULT 0, manifest_mtime REAL, last_job_id TEXT,
  pinned INTEGER DEFAULT 0, notes TEXT, demo INTEGER DEFAULT 0);
CREATE TABLE jobs (id TEXT PRIMARY KEY, kind TEXT NOT NULL, lane TEXT NOT NULL, executor TEXT NOT NULL, label TEXT,
  project_id TEXT, run_id TEXT, study_id TEXT, scenario_id TEXT, parent_job_id TEXT, after_job_id TEXT,
  params_json TEXT NOT NULL, priority INTEGER DEFAULT 0, status TEXT NOT NULL, blocked_json TEXT,
  pid INTEGER, pgid INTEGER, proc_create_time REAL, job_dir TEXT NOT NULL, threads INTEGER,
  created_utc TEXT, started_utc TEXT, finished_utc TEXT, exit_code INTEGER, error_json TEXT, result_json TEXT,
  progress REAL, eta_s REAL, eta_lo REAL, eta_hi REAL, current_path TEXT, stage TEXT, last_cursor INTEGER DEFAULT 0,
  peak_rss_mb REAL, host_id TEXT);
CREATE INDEX jobs_status ON jobs(status); CREATE INDEX jobs_run ON jobs(run_id);
CREATE TABLE spans (job_id TEXT, span_id TEXT, parent_id TEXT, kind TEXT, name TEXT, key TEXT, k INTEGER, n INTEGER,
  unit TEXT, status TEXT, started_ts REAL, ended_ts REAL, elapsed_s REAL, ctx_json TEXT, metrics_json TEXT,
  PRIMARY KEY (job_id, span_id));
CREATE TABLE metrics (job_id TEXT, cursor INTEGER, span_id TEXT, name TEXT, value REAL, value_text TEXT, unit TEXT,
  tags_json TEXT, ts REAL);
CREATE INDEX metrics_job_name ON metrics(job_id, name);
CREATE TABLE artifacts (job_id TEXT, run_id TEXT, relpath TEXT, role TEXT, stage TEXT, bytes INTEGER, ts REAL);
CREATE TABLE warnings (job_id TEXT, code TEXT, msg_hash TEXT, lvl TEXT, message TEXT, data_json TEXT, stage TEXT,
  first_cursor INTEGER, count INTEGER DEFAULT 1, PRIMARY KEY (job_id, code, msg_hash));
CREATE TABLE checkpoints (job_id TEXT, run_id TEXT, action TEXT, done_json TEXT, bytes INTEGER, ts REAL);
CREATE TABLE unit_timings (host_id TEXT, unit TEXT, seconds REAL, n_cells INTEGER, threads INTEGER, mode TEXT,
  job_id TEXT, ts REAL);
CREATE INDEX unit_timings_hu ON unit_timings(host_id, unit);
CREATE TABLE stage_timings (run_id TEXT, stage TEXT, seconds REAL, n_points INTEGER, mode TEXT, threads INTEGER,
  host_id TEXT, git_commit TEXT, source TEXT, PRIMARY KEY (run_id, stage));
CREATE TABLE resource_samples (job_id TEXT, ts REAL, rss_mb REAL, cpu_pct REAL, n_procs INTEGER, threads INTEGER);
CREATE TABLE run_locks (run_id TEXT PRIMARY KEY, job_id TEXT, acquired_utc TEXT);
CREATE TABLE studies (id TEXT PRIMARY KEY, project_id TEXT, kind TEXT, target_run_id TEXT, job_id TEXT, out_dir TEXT,
  status TEXT, params_json TEXT, summary_json TEXT, run_fingerprint TEXT, origin TEXT DEFAULT 'studio',
  created_utc TEXT, updated_utc TEXT);
CREATE TABLE study_links (run_id TEXT, study_id TEXT, attached INTEGER DEFAULT 1, PRIMARY KEY (run_id, study_id));
CREATE TABLE scenarios (id TEXT PRIMARY KEY, project_id TEXT, revision INTEGER, parent_id TEXT, name TEXT,
  doc_json TEXT NOT NULL, content_hash TEXT NOT NULL, tags_json TEXT DEFAULT '[]', status TEXT, anchor_run_id TEXT,
  archived INTEGER DEFAULT 0, created_utc TEXT, updated_utc TEXT);
CREATE TABLE results (id TEXT PRIMARY KEY, scenario_id TEXT, run_id TEXT NOT NULL, kind TEXT NOT NULL,
  content_hash TEXT, code_sha TEXT, ckpt_key TEXT, dir TEXT, summary_json TEXT, has_folds INTEGER, stale INTEGER DEFAULT 0,
  job_id TEXT, created_utc TEXT);
CREATE INDEX results_lookup ON results(run_id, content_hash, ckpt_key, code_sha);
CREATE TABLE plans (id TEXT PRIMARY KEY, project_id TEXT, run_id TEXT, name TEXT, params_json TEXT, summary_json TEXT,
  dir TEXT, verified_result_id TEXT, created_utc TEXT);
CREATE TABLE sweeps (id TEXT PRIMARY KEY, run_id TEXT, params_json TEXT, dir TEXT, job_id TEXT, summary_json TEXT, created_utc TEXT);
CREATE TABLE comparisons (id TEXT PRIMARY KEY, run_id TEXT, items_json TEXT, dir TEXT, summary_json TEXT, created_utc TEXT);
CREATE TABLE regions (id TEXT PRIMARY KEY, project_id TEXT, run_id TEXT, name TEXT, spec_json TEXT, n_cells INTEGER, created_utc TEXT);
CREATE TABLE blobs (id TEXT PRIMARY KEY, run_id TEXT, kind TEXT, path TEXT, bytes INTEGER, created_utc TEXT);
CREATE TABLE exports (id TEXT PRIMARY KEY, project_id TEXT, run_id TEXT, kind TEXT, ref TEXT, options_json TEXT,
  job_id TEXT, path TEXT, bytes INTEGER, status TEXT, created_utc TEXT);
CREATE TABLE findings (id TEXT PRIMARY KEY, project_id TEXT, run_id TEXT, view TEXT, url_state TEXT, title TEXT,
  note_md TEXT, snapshot_json TEXT, image_path TEXT, position REAL, created_utc TEXT, updated_utc TEXT);
CREATE TABLE settings (key TEXT PRIMARY KEY, value_json TEXT);
```

**Ids.** Generated with a prefix and 8 lowercase base32 characters: `p_`, `j_`, `st_`, `sc_`, `res_`, `pl_`, `sw_`, `cmp_`, `rg_`, `bl_`, `ex_`, `fd_`. Run ids follow §4.3.

**Reindex.** `sparc studio --reindex` drops and rebuilds these tables from disk:
- `runs`, `studies`, `stage_timings`, `artifacts` and `checkpoints`;
- `jobs`, from `jobs/*/job.json` + `state.json` + `result.json`;
- `results`, `plans`, `sweeps` and `comparisons`, from the run `studio/` dirs (api.md §12.2);
- `scenarios` and `findings`, from the project mirrors `projects/<slug>/scenarios/*.json` and `findings/*.json`. The engine item writes the scenario mirror on every save; the studies item writes the findings mirror.

`events.jsonl`, run directories and the project mirrors remain the source of truth. Only `settings`, `config_versions` history, `regions` and `blobs` rows are lost if the database is deleted.

### 10.7 Caching

| Cache | Rule |
|---|---|
| `RunContext` | LRU by bytes (768 MB) |
| JSON | Keyed by `(path, mtime_ns, size)` |
| Layer arrays | Float32 LRU (256 MB) |
| Grid | `<studio_dir>/cache/grid.npz` |
| Emulator arrays | Per run |
| HTTP | ETag = sha1(run_id, relpath, mtime_ns, size, key). Finished runs: `immutable`. Running runs: `no-store` |
| SPA | `index.html` is `no-store`; `/assets/*` is `immutable` |
| Server process | Never imports torch; stays < 400 MB |

### 10.8 Security

- **Token.**
  - Per-launch `secrets.token_urlsafe(32)`, written to `<ws>/token` (0600).
  - The CLI opens `GET /auth?t=<token>[&next=/path]`, which sets the cookie `sparc_studio=<session>` (HttpOnly, SameSite=Strict, Path=/; Secure when served over https) and responds 302.
  - Scripts and tests may send `Authorization: Bearer <token>`.
  - Exempt from auth: `/api/health`, `/auth`, static assets.
  - `--token T` fixes the token (dev/e2e).
- **Host allowlist.** Accepts `127.0.0.1:<port>`, `localhost:<port>`, plus `--public-host` values. Anything else → 400.
- **Origin check.** Unsafe methods require `Origin` (or `Referer`) to match → otherwise 403. CORS is disabled. Dev mode allows `--dev-origin`.
- **Remote binding.** Binding to a non-loopback host requires `--allow-remote`. The URL with the token is printed once.
- **`safe_path(raw, roots)`** is used for every path parameter. It resolves symlinks and rejects `..`, absolute paths and anything outside the allowed roots: the workspace, plus run, study or config directories explicitly registered at import.
- **Uploads** are streamed `PUT` bodies written in 1 MB chunks to a temp file, then renamed.
  - Suffix allowlist: `.csv .parquet .json .yml .yaml .tif .tiff .gpkg .geojson`.
  - Size cap 2 GB (setting).
  - No `python-multipart`.
- **Pickles.**
  - Unpickled only from runs under the workspace, or from runs imported with explicit confirmation. The import dialog names the risk: "checkpoint files execute code when loaded; import only folders you produced".
  - The engine host never loads a checkpoint the registry has not marked trusted, and neither does any job that unpickles one (`post.emulator`).
- **Subprocesses** use argument lists, never `shell=True`.
- **Other routes.**
  - `GET /openapi.json` requires auth like `/api/*`.
  - FastAPI's `/docs` and `/redoc` are disabled (`docs_url=None`, `redoc_url=None`): they load scripts from a CDN, which would break offline use and widen the surface.
  - `GET /auth?next=` accepts only same-origin paths: a leading `/`, not `//`, no scheme. Anything else redirects to `/`.
- **Raw image uploads** (findings, `PUT /api/findings/{id}/image`) are validated by content type (`image/png`, `image/svg+xml`) and size (≤ 10 MB), not by the file-suffix allowlist. SVGs are served with `Content-Security-Policy: sandbox` and are never inlined into Studio pages.

### 10.9 Startup, shutdown, CLI

```
sparc studio [--workspace DIR] [--host 127.0.0.1] [--port 8765|0] [--no-browser] [--allow-remote]
             [--public-host H ...] [--token T] [--dev-origin URL] [--reindex] [--stop-jobs-on-exit] [--log-level info]
sparc studio --dump-openapi PATH      # write the OpenAPI JSON without starting a server (used by `npm run gen:api`)
python -m sparc.core studio …   ≡   python -m sparc.studio …   ≡   sparc core studio …   ≡   sparc studio …
```

**Delegation** (core-progress). `sparc studio …`, `sparc core studio …` and `python -m sparc.core studio …` hand the remaining argv to `sparc.studio.cli.main(argv)` **before argparse runs**, so `--help` and every flag reach the studio parser verbatim. The `studio` subparsers still exist, with `add_help=False`, so they show up in the parent's help. Two cases print `pip install "sparc[studio]"` and exit 2: an `ImportError` from `sparc.studio`, or one from any of its server dependencies (`fastapi`, `starlette`, `uvicorn`, `threadpoolctl`). The core CI installs only `requirements-core.txt`, so this second case is the one tested there.

**Start:**
1. Resolve the workspace.
2. If `studio.lock.json` names a live pid (psutil + `create_time`) and its `/api/health` answers, print and open that URL, then exit 0. If that pid is live but `/api/health` does not answer within 10 s (still starting, or hung), print why and **exit 1** rather than start a second server (and scheduler) on the same workspace.
3. Otherwise **bind the socket in the CLI** (preferred port, falling back to an ephemeral port if busy; `--port 0` asks for an ephemeral one) and hand the bound socket to uvicorn. The real port is known before the lock is written. **The occupant is never killed.**
4. Write the lock (with the real port), generate the token, run uvicorn (`log_level=warning`) on the bound socket, and open `/auth?t=…` unless `--no-browser`.

**Lifespan startup:** migrate the DB → load settings → `Registry.scan()` (workspace + watch roots) → `JobManager.start()` + reattach → reconnect to the engine host if alive → start ResourceSampler.

**Shutdown** (Ctrl-C or `POST /api/shutdown`):
- broadcast `server_shutdown`;
- flush projections;
- stop the engine host;
- **jobs keep running** in their own process groups and are reattached on next start, unless `--stop-jobs-on-exit` or `{stop_jobs: true}`, which cancels them (grace 30 s, then kill).

**Exit codes** propagate: `sparc studio` returns the server's code.

### 10.10 Settings (`settings` table; `GET/PUT /api/settings`)

| Setting | Default |
|---|---|
| `thread_budget` | `cpu_count` |
| `threads_heavy` | `max(1, cpu_count − 1)` |
| `engine_threads` | 2 |
| `heavy_slots` | 1 |
| `medium_slots` | 1 |
| `network_slots` | 2 |
| `engine_max_runs` | 2 |
| `engine_mem_budget_gb` | `min(0.4·RAM, 6)` |
| `engine_idle_min` | 30 |
| `auto_uncertainty` | true |
| `watch_roots` | `[]` |
| `upload_max_gb` | 2 |
| `keep_job_logs_days` | null |
| `offline` | false (hides network actions) |
| `notifications` | false |
| `basemap_url` | null (off; reserved) |

---

## 11. Core changes (minimal, additive)

These are the core changes. The old app is not touched, and pickled classes gain no required attributes.

**Known consequence.** Editing any fingerprinted core module changes the resume fingerprint. Checkpoints written before this release therefore cannot be *resumed*, but they still unpickle for browsing and for the engine. The release notes state this. `progress.py` and `runio.py` are excluded from the fingerprint glob so future instrumentation edits do not repeat it.

**Owner `core-progress`:**

1. **`sparc/core/progress.py`** (§5.2), including `set_threads`.
2. **`sparc/core/runio.py`**:
   - `write_json_atomic(path, obj)`: `_jsonable` maps NaN/inf → null, `allow_nan=False`, tmp + `os.replace`.
   - `write_text_atomic`, `write_parquet_atomic(df, path)`, `write_npz_atomic(path, **arrays)`.
   - `@contextmanager run_lock(run_dir, timeout=600)`: `fcntl.flock` on `<run_dir>/.sparc.lock`; `msvcrt.locking` best-effort on Windows.
   - `update_manifest(run_dir, sections: dict, *, source: str) -> dict`: under `run_lock`, read, merge sections, append `post_run[] {section, at_utc, source}`, write atomically.
3. **CLI** (`sparc/core/cli.py`, `sparc/__main__.py`):
   - Add `--progress PATH` and `--job-id` to every subcommand.
   - Every `cmd_*` calls `progress.configure_from_env()` (after applying the flags) and `install_signal_handlers()`, and returns **130** on `Cancelled`.
   - Studio delegation as in §10.9:
     - `cli.main(argv)` and `sparc/__main__.main()` intercept `studio` (and `core studio`) **before** argparse and call `sparc.studio.cli.main(rest)`, imported lazily;
     - the `studio` subparsers are registered with `add_help=False` only so they are listed in help;
     - a missing `sparc.studio` or a missing server dependency prints `pip install "sparc[studio]"` and exits 2.
   - `sparc/__main__.py` uses `sys.exit(args.func(args) or 0)` for `core` and `studio` only, so exit codes propagate. Legacy commands are unchanged.

**Owner `core-instrumentation`:**

4. **`pipeline.py` planning API.**
   - `STAGE_NODES`, `CHECKPOINT_KEY` (S2_S3 → S3).
   - Gating predicates factored out of `run_core` and used by both `run_core` and `plan_stages`: `_wants_s4(stages)`, `_s6_runs(cfg, stages, responses_available)`, `_s7_runs(cfg, stages, responses)`, `_climate_runs(cfg, stages)`, `_baselines_list(cfg)`, `_cv_curve_sizes(cfg, block_m, dx, extent)`.
   - `apply_mode_overrides(cfg, fast=False, coarse=None, cv_curve=None, frame_given=False)`, factored from lines 163–175.
   - `plan_stages(cfg, stages=ALL_STAGES, fast=False, coarse=None, cv_curve=None, done=frozenset(), n_points=None, extent=None, dx=None) -> list[dict]`, a pure function. When n_points, extent or dx are unknown it runs S0 on demand (0.3 s), or the caller passes them from `data/check`.
5. **`run_core(..., run_dir: Path | None = None, run_meta: dict | None = None)`**:
   - computes `run_dir` before S0;
   - emits all events (§5.5);
   - writes `run_state.json` atomically at start, at each stage boundary and in a `try/finally` terminal state;
   - fills `run_state.meta` (n_points, cell_m, grid_shape, coarse_m, subsample_window_n, cv) as soon as known;
   - calls `check_cancel()` at stage boundaries;
   - stores `run_meta` in `manifest.run_meta`.
6. **Checkpoint sidecar.** `_save_checkpoint` also writes `checkpoint.json {fingerprint, sections, done, bytes, saved_utc, code_sha256}` and emits `checkpoint saved`. `_load_checkpoint` emits `loaded`/`mismatch` with `changed_sections`. Also adds:
   - `fingerprint_sections(cfg, fast, frame=None) -> dict[str, str]`, with sections:
     - `data`: identity
     - `core`: data/qa/encodings/predictors/influence/cv minus `distance_curve.enabled`, plus models/physics/stacker
     - `s4`: actionable/response/mediators/coupling
     - `s5`: scenarios/joint_scenarios
     - `climate`
     - `s6`: causal
     - `s7`: optimize/planner
     - `code`
   - `checkpoint_status(run_dir, cfg=None, fast=None) -> {present, done, bytes, saved_utc, matches: {data, code, config}, changed_sections}`, which never unpickles.
   - `_fingerprint` skips `progress.py` and `runio.py`. Resume semantics are otherwise unchanged; stage-scoped resume is phase 2.
   - `STAGE_NODES`, the gating predicates, `apply_mode_overrides`, `plan_stages`, `fingerprint_sections` and `checkpoint_status` stay importable without torch (the import rule of §5.5).
7. **Instrumentation** at the sites in §5.5, in `pipeline, ensemble, stacker, base_models, physics, influence, diagnostics, baselines, response, causal, optimize, climate, forcing, opendata, features_open, planner, emulator, placebo, simcheck, multiverse`.
   - ThreadPool submissions are wrapped with `progress.wrap_context`.
   - ProcessPools use `initializer=progress.init_worker`.
   - On `Cancelled`, pools call `shutdown(wait=False, cancel_futures=True)` and re-raise.
8. **Atomic writes everywhere:**
   - `pipeline._write_json` → `runio.write_json_atomic`; parquet and markdown writes via runio.
   - `baselines.py:196`, `planner.py:461–463` and `uncertainty.py:107–113` use `runio.update_manifest`.
   - `emulator.py:218–219` writes `emulator.npz` / `emulator.json` with `runio.write_npz_atomic(..., compressed=True)` / `write_json_atomic`, and records a small `emulator` summary section (per-lever trust numbers) via `update_manifest`. It currently writes no manifest section.
   - `run_lock` opens a fresh file descriptor per acquisition and also holds a process-local `threading.Lock`, so it serialises threads of one process as well as processes.
9. **Manifest durability** in `_finish`:
   - adds `schema_version: 2`, `stages_run[]`, `timings_detail` (from `ens.timings`, per-fold model seconds, physics fit seconds) and `run_meta`;
   - preserves `POST_RUN_KEYS = (planner, uncertainty, simcheck, multiverse, placebo, emulator, post_run)` and a post-run `baselines` from an existing manifest when this invocation did not recompute them.
10. **`FittedEnsemble.fold_predictions`** gets the tick + `check_cancel` hook (§5.5).
11. **`ScenarioEngine.__init__(..., base_fold: np.ndarray | None = None)`** skips the K-fold baseline pass when given and exposes `self.base_fold`.
12. **S5 detail.** When `output.scenario_detail` is true (default), `pipeline` writes `scenario_detail.npz` via `runio.write_npz_atomic`:
    - `ids`, `names` (JSON list);
    - per scenario `i`: `f{i}` (float32 K×n `delta_folds`), `sd{i}` (float32), `ex{i}` (float32 extrapolation).
    - The mean over folds equals `scenario_deltas.parquet`.
13. **Study parameters:**
    - `run_placebo_suite(..., children_dir=None, resume=False, run_meta=None)`: each child gets `run_dir = children_dir / f"{name}_placebo_{kind}_coarse{M:g}"` and `run_meta = {**run_meta, "role": f"placebo:{kind}"}`.
    - `run_multiverse(..., extra_variants: dict[str, dict] | None = None, run_meta=None)`; children follow `cfg.output.dir`, which Studio sets to `studies/<sid>/children`, with `role = "variant:<name>"`.
    - `reproduce(..., config_dir=None, out_dir=None, run_meta=None)` (`config_dir` already exists): the child `run_dir` = `out_dir`, with `role = "reproduction"`.
    - `run_simcheck` unchanged (already resumable; replicates use `write=False`, so no child runs).
14. **Equity alignment.** `optimize.equity_column` stops being read from the raw CSV and sliced `[:data.n]` (pipeline.py:392–395). Instead `data.py` carries it through S0 as an **auxiliary column**: `CoreData.aux: dict[str, np.ndarray] | None = None`, filled for `optimize.equity_column` when present. It is filtered with the same non-finite mask as the frame and averaged onto coarse cells like predictors. It is not a predictor, so it never enters the models or the fingerprinted predictor list (it is fingerprinted under `s7`). This needs `data.py` (core-instrumentation).
15. **`optimize.planned_allocation(vr, budget, cost_per_unit=1.0, equity_scores=None, equity_focus=0.0, multipliers=(0.25,0.5,1.0,2.0), cap=None, benefit_weight=None, min_dose=0.0) -> dict`**:
    - returns `{dose[n], planned_benefit[n], planned_total_cooling, n_cells_treated (cells), mean_dose_treated, total_cost, gini, min_dose_dropped_cost, pareto{points[{budget, total_benefit, n_cells, n_segments, gini}]}}`;
    - `min_dose` is a post-filter: allocations below it are set to 0 after the greedy pass, and the freed budget is reported, not re-spent;
    - `optimise_allocation` becomes `planned_allocation` + closed loop.
    - `optimizer_layers` (in `pipeline.py`) is made public (the private alias is kept).
16. **Lazy exports.** `sparc/core/__init__.py` lazily exports `plan_stages` and `open_run`.

**Owner `backend-engine`:**

17. **`sparc/core/session.py`**:
    - `config_for_run(run_dir, fallback=None) -> CoreConfig`, resolving in the order of §4.3.
    - `open_run(run_dir, cfg=None, *, threads=2, base_fold=None, drop=("cv_distance","causal_arrays"), progress_cb=None) -> RunSession`, with fields `data, folds, manifest, predictions, ensemble, influence, mediators, engine, resp (lazy property), responses, layers, causal, code_match, load_seconds`.
    - A byte-counting checkpoint reader ticks unpickle progress.
    - `emulator_for_run` keeps its own loader in v1. A test asserts `open_run` gives identical scenario deltas.

**Owner `backend-runs`:**

18. **`sparc/core/catalog.py`** (§6.1).

**Owner `backend-studies-exports`:**

19. **`sparc/core/results_page/`** (§6.9), plus the `scripts/results_page/` shims.

**Owner `core-instrumentation` (additions from the completeness review, §18).** These need three files outside the original list: `sparc/core/synthetic.py`, `sparc/core/influence.py` and `sparc/core/data.py`.

20. **`synthetic.write_demo_project(out_dir, n=96, seed=0) -> dict`**. It writes the whole synthetic demo of §9.2:
    - `data/city.csv` with the EPSG:32619 offset, and `data/city.truth.json` with `derived.true_scenarios` from `SyntheticCity.true_response`;
    - `inputs/layers/demo_layers.parquet` and `inputs/climate/demo_cmip6.csv`;
    - `config.yml` (relative paths).

    It returns the paths and the truth dict. It is deterministic in `(n, seed)`: same bytes for CSV, parquet and YAML. `scripts/make_studio_fixtures.py` and the `synthetic_demo` template both use it.
21. **Causal per-cell outputs.** S6 keeps what `_strip_arrays` drops, in two places:
    - **`causal_cells.parquet`**: `id`, `cate:<t>` (R-learner `tau_hat`), `mslope:<t>` (model `per_point_slope`) and `mslope_own:<t>` (`per_point_own_slope`), float32, one column per treatment;
    - **`causal.json treatments[t].model_effects`**: `{adoption_slope, adoption_se, own_slope, own_se, own_pd_curve: {t, y, se}}` (7 points).

    `causal.json` itself stays array-free otherwise.
22. **Frame-input runs are re-readable.** `run_core(frame=…, write=True)` writes the frame to **`input_frame.parquet`** (via runio) and records `provenance.input_frame = "input_frame.parquet"`. `baselines.load_run` (and so `RunReader` and `open_run`) loads data from it when present. This affects placebo children only; simcheck uses `write=False`.
23. **`influence.py`**: the two warnings of §5.5, as `progress.warn` calls next to the existing log lines.
24. **Torch-free import guard**: `tests/core/test_import_light.py` asserts the import rule of §5.5.

---

## 12. Frontend architecture

### 12.1 Stack and dependencies

React 19 + TypeScript (strict) + Vite 8. The source lives in **`studio-web/`** at the repo root. `sparc-desktop/` is untouched.

- **Runtime:** `react`, `react-dom`, `zustand`, `marked`, and `@fontsource/archivo`, `@fontsource/public-sans`, `@fontsource/ibm-plex-mono` (OFL; CSS and woff2 bundled locally, so it works offline).
- **Dev:** `vite`, `@vitejs/plugin-react`, `typescript`, `vitest`, `jsdom`, `@types/react`, `@types/react-dom`, `openapi-typescript`.
- **Not used:** router libraries, chart libraries, map libraries, CSS frameworks, UI kits, d3.

### 12.2 Source tree and ownership

```
studio-web/
  package.json package-lock.json vite.config.ts tsconfig.json index.html          [foundation]
  scripts/gen-api.mjs                                                            [foundation]
  src/main.tsx  src/App.tsx                                                      [foundation]
  src/router.ts                  History-API router (~150 lines) + route registry via import.meta.glob("./pages/*/routes.ts")   [foundation]
  src/layouts/{AppShell,ProjectLayout,RunLayout}.tsx  (tab bar & project nav built from route metadata)   [foundation]
  src/api/client.ts              typed fetch, ApiError{status,code,message,detail,action}, getBin(dtype), putRaw   [foundation]
  src/api/sse.ts                 StreamManager (global + 1 job stream, Last-Event-ID, backoff, polling fallback)   [foundation]
  src/api/binary.ts              Float32/Uint8/Int32 views, bitset codec, offsets header parsing   [foundation]
  src/api/resource.ts            useResource(key, fetcher, {immutable}), tags, invalidate   [foundation]
  src/api/types.ts               common types: Job, JobStatus, ApiErrorBody, LayerMeta, GridMeta, SelectionSpec, PlanNode, events   [foundation]
  src/api/<group>.ts             per-group typed endpoint functions + types (projects, runs, lab, studies, …)   [feature items]
  src/stores/{jobs,ui,selection}.ts   global job list (from global SSE), theme/toasts/palette, per-run selection   [foundation]
  src/theme/{tokens.css,base.css,fonts.ts,palette.ts,format.ts}                   [foundation]
  src/components/ui/*            Button IconButton Seg Select NumberField Slider Chips Card Kpi Pill Badge StatusChip Tabs
                                 Table(sortable, virtualised>500, CSV) VirtualList Dialog Drawer Toast Tooltip ProgressBar
                                 EmptyState(output_missing→action) Markdown CodeArea Diff FileDrop CommandPalette JobTray
                                 JobStrip(compact progress of one job from stores/jobs: bar, stage, ETA range, link)
                                 ConnectionPill PlainResult(likely-range card)   [foundation]
  src/charts/*                   scales ticks Axis ChartFrame(title, caption, units, Table view, SVG/PNG/CSV, Pin) LineBand
                                 Bars(h/v, grouped, stacked, 100%) DotRange Forest Heatmap Histogram(brushable)
                                 HexbinScatter(canvas) BoxStrip Gantt Sparkline Pareto Rose Gauge IntervalStack RingProfile
                                 SmallMultiples   [foundation]
  src/map/*                      GridCanvas MapView LayerPicker Legend Inspector Overlay SwipeCompare useMapView
                                 tools/{pan,brush,rect,circle,polygon,pick}.ts colour.ts domain.ts   [foundation]
  src/pages/NotFound.tsx  src/pages/core/routes.ts                               [foundation]
  src/pages/projects/**  src/api/projects.ts  src/api/inputs.ts                  [frontend-projects]
  src/pages/tracking/**  src/stores/tracker.ts  src/api/tracking.ts             [frontend-tracking]
  src/pages/runhub/**    src/api/runs.ts  src/api/analysis.ts                   [frontend-run-hub]
  src/pages/lab/**       src/api/lab.ts                                         [frontend-lab]
  src/pages/studies/**   src/api/studies.ts  src/api/exports.ts  src/api/findings.ts   [frontend-studies-exports]
  src/api/contract.check.ts  src/api/schema.gen.ts                               [integration]
```

### 12.3 Router and registry

`router.ts`:
- `matchRoute`, `navigate`, `useRoute()` (params + query);
- `<Link>`;
- query codecs (`useUrlState(key, codec)`);
- scroll restoration.

Each `src/pages/<group>/routes.ts` exports

```ts
type RouteDef = { path: string; component: React.LazyExoticComponent<any>; title: string;
  runTab?: { id: RunTabId; label: string; group: "Model"|"Effects"|"Decisions"|"Trust"|"Run"; order: number };
  projectNav?: { id: string; label: string; order: number; to: (pid: string, project: Project) => string | null;
                 disabledReason?: (project: Project) => string | null } };
export const routes: RouteDef[] = [
  { path: "/r/:rid/accuracy", component: lazy(() => import("./Accuracy")), title: "Accuracy",
    runTab: { id: "accuracy", label: "Accuracy", group: "Model", order: 30 } },
];
```

- `RunTabId` is the fixed vocabulary of §3.2. The tab's status dot is read from `/outputs.tabs[id]`, so route modules do not list outputs.
- `projectNav` entries are declared only by the item that owns the target route (table in §3.1). `to` returning `null` hides the entry; `disabledReason` greys it out with a tooltip.

The foundation collects them with `import.meta.glob("./pages/*/routes.ts", { eager: true })`. Groups that do not exist yet simply contribute nothing. Each group is code-split: the initial JS budget is ≤ 200 kB gzip, and the Lab chunk is ≤ 140 kB gzip.

### 12.4 State and data flow

- **Server state** goes through `useResource`. Keys are tagged: `run:<rid>`, `job:<jid>`, `scenario:<sid>`, `project:<pid>`.
- **Finished-run resources are immutable**: never refetched.
- **Global SSE** events update `stores/jobs` and invalidate tags. For example, `output.written` → `run:<rid>:outputs`, and `scenario.result` → `scenario:<sid>`.
- **Mission Control** fetches `/tracker` (snapshot + cursor), then opens the job stream with `after=cursor`. Events are applied by the pure `applyEvent` reducer in `stores/tracker.ts`, batched per `requestAnimationFrame`.
- **localStorage** holds conveniences only (theme, last tab, dock state), each access wrapped in try/catch.
- **Lab drafts** PATCH every 2 s and on blur. Undo/redo is local.

### 12.5 Canvas grid renderer (`GridCanvas`)

**Inputs:**
- `GridMeta`;
- `cellToPt` (Int32Array of `nx·ny`; −1 = empty; built once from `grid.bin` as `(ny−1−iy)·nx + ix`, north-up);
- a values typed array;
- `LayerMeta`;
- a LUT;
- an optional selection bitset (tint);
- an optional hatch mask (extrapolation > 1, approximate preview).

**Rendering:**
- `colorize()` writes a `Uint32Array` view of an offscreen `nx×ny` `ImageData`, taking about 1–2 ms for 54,701 cells. NaN is transparent; `zero_blank` uses `--nodata`.
- The viewport canvas (devicePixelRatio-aware) draws the offscreen rasters with a `{scale, tx, ty}` transform and `imageSmoothingEnabled = false`. Redraws happen only on change, via rAF.
- Overlays are SVG in the same transform.

**Interaction:**
- wheel zoom about the cursor (1–32 px per cell);
- drag to pan (space + drag while a tool is active);
- `+`/`−`/`0` keys; arrow-key cell cursor; Enter pins the inspector.
- Hit test: inverse transform → cell → `cellToPt`.

**Tools** produce `SelectionSpec`s or edit arrays:
- **brush**: radius 60/120/250/500 m, add/erase. It writes Float32 per-lever edit arrays clamped against exact Float32 inputs (never u8), and strokes overwrite idempotently.
- **rect**, **circle**, **polygon**: produce lon/lat when the grid has a CRS.
- **pick**: zone or hex.

**Swipe and side-by-side.** Two rasters clipped at a draggable, keyboard-movable divider. Side-by-side panes share `useMapView`.

**Basemap.** None by default (offline-safe).

**Budgets:**
- layer switch < 50 ms including decode from cache;
- recolour < 10 ms;
- pan/zoom at 60 fps.

### 12.6 Charts

Hand-built SVG components following the dataviz rules:
- at most three categorical colours plus grey;
- diverging blue (cooler) → red (warmer), centred on the city median or 0;
- hollow mark = extrapolated; orange diamond = independent causal check;
- tabular numerals;
- Unicode minus;
- units in every axis title.

`ChartFrame` provides title, caption (computed, never canned), a Table view `<details>`, SVG/PNG/CSV export, **Pin to Findings**, and keyboard-focusable points with `aria-label`s and an aria-live readout.

### 12.7 Design system

**Tokens** are ported from `template.html`:
- surfaces and ink: `--page`, `--surface`, `--ink`, `--ink-2`, `--muted`, `--grid`, `--axis`, `--line`;
- accents: `--accent`, `--accent-ink`, `--s1`, `--s2`, `--s3`, `--gray-mark`;
- semantic: `--good`, `--warning`, `--serious`, `--critical` and their `*-ink` variants;
- `--chip`.

**New tokens:**
- `--nodata`, `--hatch`;
- status tokens `--st-queued`, `--st-running`, `--st-done`, `--st-cached`, `--st-skipped`, `--st-failed`, `--st-cancelled`, `--st-interrupted`, `--st-stale`, `--st-blocked`;
- `--demo`.

**Theming.** Light and dark use `prefers-color-scheme`, guarded by `:root:not([data-theme=light])` and `:root[data-theme=dark]`, plus a manual toggle. `body` has an explicit background. Colour LUTs are rebuilt when the theme changes.

**Type.**
- Archivo (wdth 87) for display;
- Public Sans 15 px / 1.55 for body;
- IBM Plex Mono for eyebrows, numbers and logs.

**Components.** Cards (radius 10), KPI tiles, seg pills, pills with icons, tables with `.hl` rows.

**Layout.**
- Content width 1120–1440 px. Mission Control, Map and the Lab use full width.
- The Lab is three panes (360 px · flex · 340 px), each collapsible.
- Single column below 860 px with a 16 px gutter and no horizontal page scroll.

### 12.8 Accessibility

- Proper HTML shell: doctype, `lang`, charset, viewport.
- Skip link and landmarks.
- `:focus-visible` rings; `aria-pressed` on toggles.
- Keyboard map cursor with an aria-live readout.
- Focusable chart points; tooltips on focus and on hover.
- aria-live for job status changes.
- `prefers-reduced-motion` disables pulsing.
- Status is never shown by colour alone: icons, text, hatching and shape carry it too.
- Contrast is checked in both themes.

---

## 13. Packaging and developer workflow

### 13.1 `pyproject.toml` changes

These belong to the backend foundation item.

- **Install extra:**

  ```toml
  [project.optional-dependencies]
  studio = ["fastapi>=0.111,!=0.136.3", "uvicorn>=0.29", "threadpoolctl>=3", "httpx>=0.27"]
  ```

  `httpx` is listed because the TestClient needs it. `fastapi` 0.136.3 is excluded because it was flagged as malicious.
- **Package data:**

  ```toml
  [tool.setuptools.package-data]
  "sparc.studio" = ["static/**/*", "examples/**/*"]
  "sparc.core" = ["results_page/*.html"]
  ```

- **Test markers:** add `e2e` and `studio` to `[tool.pytest.ini_options].markers`.

### 13.2 Built assets

`npm --prefix studio-web ci && npm --prefix studio-web run build` writes hashed assets to **`sparc/studio/static/`**:
- `index.html`
- `assets/*-<hash>.{js,css,woff2}`
- `BUILD_INFO.json`: `{src_sha256` over `studio-web/src` + `package-lock.json` + `vite.config.ts` + `index.html`, `vite, react}`. It has **no timestamp**, so rebuilding unchanged sources gives byte-identical output and the CI rebuild-diff can pass.
  - `src_sha256` is computed exactly as `sourceHash()` in `studio-web/vite.config.ts`: every file under `studio-web/src` (skipping names that start with `.`), plus `package-lock.json`, `vite.config.ts` and `index.html`; POSIX paths relative to `studio-web/`, sorted by code point; SHA-256 over `relpath + "\0" + bytes + "\0"` per file. `scripts/check_studio_assets.py` reimplements it.

The built assets are **committed**, so pip users need no Node.

`scripts/check_studio_assets.py` (integration) fails when `BUILD_INFO.src_sha256` does not match the current source. CI also rebuilds and runs `git diff --exit-code sparc/studio/static`.

If `static/index.html` is missing (fresh checkout before the first build), the server serves a minimal built-in page: "Frontend not built — run `npm --prefix studio-web run build`".

### 13.3 Development

- **Backend:** `sparc studio --no-browser --token dev --dev-origin http://localhost:5173`.
- **Frontend:** `npm --prefix studio-web run dev`. Vite on 5173 proxies `/api` and `/auth`, with SSE passthrough. The dev page reads `VITE_STUDIO_TOKEN=dev` and calls `/auth?t=dev` once.
- **Typed API:** `npm --prefix studio-web run gen:api` runs `python -m sparc.studio --dump-openapi <tmp>` (no server needed) and then `openapi-typescript` to write `src/api/schema.gen.ts`. `src/api/contract.check.ts` (type-only) asserts that the hand-written group types are assignable to and from the generated ones. `npm run typecheck` therefore fails on drift. `schema.gen.ts` and `contract.check.ts` are both created by the integration item, so the foundation's `typecheck` has nothing to check against before then; `gen-api.mjs` (foundation) only needs the `--dump-openapi` flag (backend foundation).

### 13.4 Old app

`sparc-desktop/` and `sparc/server/` stay as they are and keep their tests. Studio does not import either.

---

## 14. Testing

Test runners:
- core and backend: pytest; the backend uses FastAPI `TestClient` (httpx);
- frontend logic: vitest;
- e2e: Python Playwright with Chromium at `/opt/pw-browsers/chromium-1194/chrome-linux/chrome`, or `PLAYWRIGHT_CHROMIUM` when set.

Markers are `slow`, `network`, `e2e` and `studio`. CI runs everything except `network`. `slow` and the Providence e2e run nightly.

### 14.1 Core (`tests/core/`)

**`test_progress.py`** (core-progress):
- Calls are no-ops when unconfigured (zero sink calls).
- Spans nest across contextvars, threads (`wrap_context`) and spawn ProcessPools (`init_worker`).
- Tick throttle keeps the first and last tick.
- Lines are ≤ 4 KB and valid JSON with `allow_nan=False`.
- `Cancelled` survives `except Exception`.
- The signal handler sets a flag; a second signal raises.
- `LogBridge` forwards verbatim.
- The heartbeat thread starts and stops.
- **Disabled cost.** With no sink, 1e6 calls each of `tick`, `metric` and `emit` (with keyword arguments) write nothing. The median over 5 repeats is ≤ 2× the time of 1e6 calls to a no-op function with the same signature. (A literal "< 1% of an empty loop" is impossible in CPython: one function call already costs more than an empty loop iteration.)
- Ticks with `k == 1` and `k == n` survive throttling.

**`test_runio.py`** (core-progress):
- Atomic writes.
- NaN is written as null.
- Concurrent `update_manifest` from 8 threads plus 2 processes loses no section.

**`test_cli_studio.py`** (core-progress):
- `python -m sparc.core studio --help` and `sparc studio --help` reach `sparc.studio.cli.main(["--help"])` verbatim (monkeypatched stub).
- With `sparc.studio` unimportable, **and separately with only `fastapi` unimportable**, they print the install hint and exit 2.
- Exit codes propagate through `sparc core`.
- 130 on cancel.

**`test_progress_pipeline.py`** (core-instrumentation; the **event-invariant test**). Uses `write_demo_project(n=40, seed=0)` run in fast mode with all stages (causal on canopy, budget set, table climate, demo layers) and `SPARC_PROGRESS` set. It asserts:
- every enabled stage has start and end;
- disabled stages have `stage.skip` with the expected reason;
- every file in the run dir has an `artifact` event, except `run_state.json`, `checkpoint.pkl` and `checkpoint.json` (covered by `checkpoint` events) and the catalog's `IGNORED` globs;
- `checkpoint` events match `checkpoint.json`;
- `run_state.json` ends `succeeded`;
- no info-level gap exceeds 15 s;
- for every unit kind, the completions counted by the unit-accounting rule (§5.4) equal the `run.plan` units.

`scripts/make_studio_fixtures.py` (owned by core-instrumentation) runs the same synthetic run and saves `events.jsonl` plus the run dir **without `checkpoint.pkl`** as the committed fixture `tests/studio/fixtures/synth_run/` (≤ 3 MB). It also writes `tests/studio/fixtures/synth_run/FIXTURE.json` `{n: 40, seed: 0, mode: "fast", generator: "write_demo_project"}`, which the replay e2e reads (§14.5). **Determinism** means: re-running it yields the same file list; numerically identical arrays in every parquet/npz (`np.array_equal`); identical JSON apart from the volatile keys (`created_utc`, `timings_s`, `timings_detail`, `git_commit`, `provenance.git`, `pid`, `ts`, `elapsed_s`, `seconds`, `fit_s`, `pass_s`, `saved_utc`, `updated_utc`, `started_utc`, `bytes` of pickles); and the same sequence of `(type, name, key, stage, k, n, unit)` in `events.jsonl` after removing `heartbeat`, `log` and throttled `k < n` ticks. A test runs the script twice in temp dirs and compares under these rules (marked `slow`). Tests that need an engine build a fresh run in a temp dir (marked `slow`).

**`test_plan_stages.py`**:
- A gating matrix (stage subsets × fast × coarse × cv_curve × climate/causal/optimize toggles). The `plan_stages` node states must equal the `stage.start`/`stage.skip` events of real `write=False` runs on the n=40 demo. Six representative combinations run unmarked; the full matrix is `slow`.
- A config-only case: `plan_stages(configs/core_providence.yml with the recorded-run settings, n_points=54701, …)` reproduces the reference unit counts of §5.4 without running anything.

**`test_cancel_resume.py`**:
- Inject a cancel at parametrised safe points through a callable sink that calls `request_cancel()` when it sees the trigger event: S1 end, fold 1 MGWR tick (`SPARC_PROGRESS_LEVEL=debug`), stacker candidate 2, S4 dose 2, S6 treatment 1.
- Assert exit 130, `run_state.cancelled`, and that the `checkpoint.json` done set matches the pickle.
- `resume=True` completes, and its metrics equal an uninterrupted run within 1e-9. Both runs use `threads=1` (`progress.set_threads(1)`) and fixed seeds, so torch and BLAS reductions are reproducible.

**`test_demo_project.py`**: `write_demo_project` is deterministic (byte-identical CSV, parquet and YAML for the same `(n, seed)`), has a CRS, and `truth.json.derived.true_scenarios` matches `true_response`.

**`test_causal_cells.py`**: `causal_cells.parquet` is row-aligned with `predictions.parquet`. `model_effects.own_pd_curve` has 7 points. `causal.json` stays free of per-cell arrays.

**`test_input_frame.py`**: a placebo-style `run_core(frame=…, write=True)` writes `input_frame.parquet`, and `baselines.load_run` rebuilds identical ids and folds from it.

**`test_import_light.py`**: the torch-free import rule of §5.5.

**`test_checkpoint_sidecar.py`**:
- `fingerprint_sections` pinpoints changed sections.
- `checkpoint_status` never unpickles (monkeypatched `pickle.load` raises).
- Editing `progress.py` or `runio.py` content (monkeypatched) leaves the fingerprint unchanged.

**`test_manifest_durability.py`**: `_finish` preserves post-run sections; `schema_version`, `stages_run` and `timings_detail` are present.

**`test_scenario_detail.py`**: the npz fold mean equals `scenario_deltas.parquet` within 1e-6.

**`test_equity_alignment.py`**: with dropped non-finite rows and subsample, equity is aligned by id.

**`test_planned_allocation.py`**: equals `optimise_allocation`'s planned totals; `n_cells_treated` counts cells.

**`test_study_params.py`** (slow): placebo `children_dir`/`resume`; multiverse `extra_variants`; reproduce `out_dir`; `run_meta` recorded.

### 14.2 Studio backend (`tests/studio/`)

Test files live in `tests/studio/<group>/` (`foundation`, `projects`, `runs`, `engine`, `studies`), so each work item owns its own directory. Shared fixtures are in `tests/studio/conftest.py` (foundation): a temp-workspace app, an authenticated client, and fixture loaders. The ETA calibration test embeds the recorded Providence numbers as constants, because `output/` is gitignored.

| Test | Owner | What it checks |
|---|---|---|
| `test_security.py` | foundation | 401 without auth; cookie exchange; wrong Host → 400; cross-Origin POST → 403; `safe_path` rejects traversal, absolute paths and symlink escapes; upload allowlist and cap; streamed upload does not buffer (memory assertion); `--allow-remote` required |
| `test_jobs_manager.py` | foundation | Uses test kinds: queue order and lane limits; per-run lock serialisation; `after_job_id` chains; cancel → 130 → cancelled; `test.ignore_sigterm` → kill → pgid gone incl. `test.pool` children (psutil); `starting` timeout; result.json |
| `test_reattach.py` | foundation | Start a job, drop the app, create a new app → reattached and tails to completion. Kill the worker while the app is down → `interrupted`. Mismatched `create_time` → not reattached |
| `test_sse.py` | foundation | Ordered cursors; reconnect with Last-Event-ID gives no gaps or duplicates; overflow → resync; tick coalescing on the wire; resource events carry no id; global ring buffer resync |
| `test_tracker_reducer.py` | foundation | Replaying the foundation's own `tests/studio/fixtures/selftest_events.jsonl` (written by `test.events`, schema-complete: run/stage/task spans, ticks, metrics, artifacts, checkpoints, warnings, run.end) gives the committed golden projection; monotonic progress |
| `test_eta.py` | foundation | The cost model applied to the **reference unit counts of §5.4** (embedded as constants; `n_cells` 54,701, threads 4) predicts the recorded Providence `timings_s` total (6,096.6 s, also a constant) within 20%. Online refinement converges after fold 0 on the selftest fixture. No dependency on `plan_stages` |
| `test_meta.py` | foundation | `/api/meta.palettes` LUT entries equal the golden table of §6.3; `unit_costs` equals the seed table of §5.4; stage list falls back to the static list when `sparc.core.pipeline.STAGE_NODES` is absent |
| `test_projects.py` | projects | Templates blank / synthetic_demo (CRS present, DEMO flag, files byte-identical to `write_demo_project`) / providence_example (packaged and repo sources); YAML round trip keeps `core:`; `validate_deep` catches the listed errors; `If-Match` 409; data/check; columns/suggest; impact preview flags a changed `core` section; schema vs DEFAULTS diff test. The `import_existing_runs` case is skipped unless `sparc.studio.runs.registry` is importable (§10.2 contracts) |
| `test_inputs.py` | projects | Input kinds with fetchers monkeypatched (no network), incl. `input.stations` and features bootstrap mode on a predictor-less project; link diffs; network-marked live tests |
| `test_runs_api.py` | runs | Launch with the replay runner; plan endpoint = `plan_stages`; import `providence_uhi_fast` (skip if absent); stale detection of its Sep-30 `causal.json`; manifest/file merge; status board cells |
| `test_reader_midrun.py` | runs | Partially copied fixture run (no manifest) → accuracy view, grid and folds work from `run_state` + predictions |
| `test_layers.py` | runs | Catalogue keys match files; Float32 length n; NaN round trip; grid shapes (100, 86) and (334, 287); fold classes match `make_spatial_folds`; ETag/immutable headers |
| `test_selection.py` | runs | Polygon in lon/lat; circle area within 3%; buffer equals brute force; hex keys; zones; filters; top-k within; and/or/minus/not laws; bitset codec |
| `test_stats_tools.py` | runs | Region SE equals manual jackknife; breakdown; hexbin counts; ACF matches `influence.fft_acf` |
| `test_catalog.py` | runs | Every file of the synthetic run + post-run + studies fixture matches an `OutputSpec` |
| `test_compile.py` | engine | Each mode yields the expected `per_point`; clamping prediction equals `ScenarioEngine.apply` realised values on the synthetic run; guardrails (mediator-with-parents blocked; non-actionable needs expert) |
| `test_preview.py` | engine | Preview equals `sum(emulate(...))` within 1e-6; latest-wins supersede; `threadpool_limits(1)` in effect |
| `test_engine_host.py` (slow) | engine | Host open on the synthetic run; scenario job emits K fold ticks; cache hit; cancel between folds keeps the host alive; LRU eviction by count and budget; recycle; reconnect after server restart; incompatible checkpoint state |
| `test_results_stats.py` | engine | Region "all" equals `ScenarioResult.summary()`; paired SE(A−A) = 0; paired ≤ independent on correlated folds; spill shares sum to 1; plain-language confidence wording |
| `test_plans_climate.py` | engine | Planned mode equals `planned_allocation`; region cap zeroes cells outside; climate explore equals `summarize_projections`; field kit uses ids + lon/lat |
| `test_session.py` (slow) | engine | `open_run` delta equals the `emulator_for_run` path and the S5 stored delta (1e-9); `base_fold` passthrough identical; GWRF `n_jobs` override |
| `test_studies.py` | studies | Each kind's dispatch with library functions monkeypatched (expected `cache_dir`, `children_dir`, `config_dir`, `run_meta`, launch-snapshot config); the kind writes `placebo.json/.md`, `multiverse_summary.md`, `simcheck_summary.md`, `benchmark.json/.md`; child-run registration via the `on_event` hook; attach → auto uncertainty; stale vs fingerprint; `workers × threads` validation; `GET /runs/{rid}/truth` on the demo fixture |
| `test_exports.py` | studies | GeoTIFF readable by rasterio with correct CRS and transform; bundle ZIP contents and README; findings MD/HTML; results page renders and passes `check_page.py` |

### 14.3 Frontend (vitest, `studio-web`)

**Foundation:**
- router matching and query codecs;
- `client` errors map to `ApiError`;
- `sse` reconnect, `after`, resync and polling fallback with a fake EventSource;
- bitset and offsets codecs;
- OKLab LUT matches the golden table of §6.3 (`src/test/fixtures/palette.json` is hand-written from that table, not generated from Python);
- domain rules (symmetric diverging, zero_blank, mult);
- `format` (Unicode minus, units, durations, bytes);
- GridCanvas `cellToPt` and colorize on a 3×3 grid;
- chart scales and ticks.

**Tracking:**
- `applyEvent` replay of the item's own `__fixtures__/events.sample.jsonl` gives the expected stage states, progress and warnings;
- stage rail state mapping.

**Integration (cross-language contract):** `tests/studio/test_reducer_contract.py` runs the Python `jobs/tracker.reduce` and the TS `applyEvent` (via `studio-web/src/contract/reducer.contract.test.ts`) over `tests/studio/fixtures/synth_run/events.jsonl`. It asserts identical stage states, progress (±1e-9), warnings and artifacts.
- Both sides expose `to_contract(state)` / `toContract(state)` returning the same canonical JSON: `{stages: {id: {state, reason}}, progress, done_units: {unit: n}, warnings: [{code, count}] (sorted), artifacts: [relpath] (sorted)}`.
- Progress uses the seed-rate weights (§5.4), so neither side needs host calibration.
- The vitest file writes its projection to the path in `SPARC_CONTRACT_OUT` (a pytest temp file, so no gitignore entry is needed). The Python test runs `npm --prefix studio-web exec vitest run src/contract` with that variable set (skipped when Node is absent) and compares the file with its own projection.

**Feature items** test their pure logic with `fetch` mocked:
- projects: column suggestion display, plan graph layout;
- run hub: view-model formatting;
- lab: draft store undo/redo, brush clamping idempotence, ScenarioDoc serialisation, plain-language wording;
- studies: simcheck grid colouring.

### 14.4 Contract and assets (integration)

- `npm run gen:api` + `contract.check.ts` typecheck.
- `pytest tests/studio/test_openapi_snapshot.py`: committed `openapi.json` snapshot; changes need an explicit update.
- `scripts/check_studio_assets.py`.

### 14.5 End-to-end (Playwright Python, `tests/studio/e2e/`, marker `e2e`)

The harness starts `python -m sparc.studio --workspace <tmp> --port 0 --no-browser --token e2e` as a subprocess and reads the URL from `studio.lock.json`. **Any console error fails a test.** Screenshots are taken per step in light and dark themes.

1. **`test_e2e_replay.py`** (fast, deterministic, every CI run; `SPARC_STUDIO_RUNNER=replay:tests/studio/fixtures/synth_run`, replayed at ×1 — at the runner's default ×20 the 31 s fixture ends in under 2 s, too fast to reload mid-run):
   - create the synthetic demo **with the fixture's `n` and `seed` from `FIXTURE.json`** (`options: {n: 40, seed: 0}`). The project's data is then byte-identical to the data the fixture was produced from, so `RunReader` id and fold checks pass on the replayed outputs. Walk the setup pages; launch Fast;
   - the replay runner rewrites `job`, `pid` and `ts` (as in api.md §14) and also `run.dir.run_dir` and `run_state.json.events_path`/`pid` to the new run's values;
   - the plan graph matches the fixture; the rail reaches done in order; the fold × model heatmap fills;
   - the Outputs feed lights up; Accuracy opens mid-replay;
   - reload mid-run → no gap in log cursors;
   - visit every run tab and every map layer (as `check_page.py` does);
   - brush a histogram → the selection count shows on the map and in Region stats;
   - pin a finding → export findings HTML.
2. **`test_e2e_real_fast.py`** (CI; synthetic city n=48, real `--fast` run, ≈3–6 min):
   - launch; cancel during S2_S3 → cancelled within 30 s; resume → cached stages shown → succeeded;
   - open the Lab → engine loads; draw a circle with +10 canopy → preview appears; Run exact → fold ticks → plain-language card with likely range;
   - save, fork, compare with a configured scenario (paired SE shown);
   - climate explore with threshold change;
   - plan preview slider updates the Pareto curve; verify;
   - export a decision pack (zip has `brief.html`, `cells.csv`, `delta.tif`, `README.txt`) and a GeoTIFF.
3. **`test_e2e_studies.py`** (nightly): placebo `kinds=["shift"]` at coarse 60 m on the `n=96` synthetic city (≈1,600 coarse cells; at 120 m the n=40/48 cities shrink to under 200 cells and MGWR tuning is skipped) → child run and verdict; benchmark (`n=48`).
4. **`test_e2e_providence.py`** (nightly; skipped if absent): open the Providence example with run import; every tab of `providence_uhi_fast` renders; stale badges shown; engine opens; one scenario; emulator preview parity hook.
5. **`test_e2e_a11y_responsive.py`**:
   - tab through the shell;
   - map keyboard cursor updates the aria-live readout;
   - every button has an accessible name;
   - at 390 px width there is no horizontal scroll.

**Performance checks** (nightly) on a 54,701-cell synthetic fixture:

| Check | Target |
|---|---|
| Layer endpoint, warm | < 50 ms |
| Layer switch | < 50 ms |
| Recolour | < 10 ms |
| Preview | < 150 ms |
| Selection resolve | < 30 ms |
| Run list (200 runs) | < 100 ms |
| Tailer throughput | ≥ 2,000 events/s |
| Tracker with 20,000 events | No frame > 100 ms |
| SPA initial bundle | ≤ 200 kB gzip |

### 14.6 CI (`.github/workflows/studio.yml`, owned by integration)

1. `ruff check sparc/studio sparc/core/progress.py sparc/core/runio.py`
2. `pytest tests/core -m "not slow and not integration"` (instrumentation tests included)
3. `pytest tests/studio -m "not slow and not network and not e2e"`
4. `npm --prefix studio-web ci`, then `typecheck`, `test` and `build`, then `git diff --exit-code sparc/studio/static`
5. `python scripts/check_studio_assets.py`
6. `pytest tests/studio/e2e/test_e2e_replay.py tests/studio/e2e/test_e2e_real_fast.py`
7. Wheel smoke test: build, install into a clean venv, run `sparc studio --no-browser --port 0`, and `GET /` returns index.html with hashed assets.
8. If `docs/studio/doctest.py` exists (added later by the docs item), run it. The integration item adds this step with an existence guard, so the docs item never edits the workflow.
9. Guard for the old app: the `sparc-desktop` and `sparc/server` trees at HEAD must equal those at the pinned baseline commit `STUDIO_BASELINE` (the last commit before Studio work). See §19.

**Existing workflow.** `.github/workflows/tests.yml` already has a `legacy` job that runs `pytest tests --ignore=tests/core` with the `[server]` extra, which includes FastAPI. Without a change it would collect `tests/studio/**`, including slow, e2e and Node-dependent tests. The integration item therefore owns one edit to that file: add `--ignore=tests/studio` to the legacy job. Its `core` job installs only `requirements-core.txt`, so the new `tests/core` files must not need FastAPI, psutil or Node (the studio delegation test uses the install-hint path).

**Nightly:** `-m slow`, studies e2e, Providence e2e and performance checks.

---

## 15. Non-goals (v1)

Explicitly out of scope:

- **Product scope:**
  - accounts, multi-user, sharing or cloud hosting (single local user; a cloud container works with `--allow-remote`);
  - the legacy pipeline (`sparc run`), legacy stages 0–4, or any change to `sparc-desktop/` and `sparc/server/`;
  - a Tauri or desktop wrapper (it may wrap this server later);
  - editing model internals beyond config;
  - multi-lever joint budget optimisation (single-lever plans can be combined but are labelled "not jointly optimal");
  - PDF generation on the server (use the browser print stylesheet);
  - AI chat.
- **Pipeline behaviour:**
  - stage-scoped resume and per-partition/per-variable partial checkpoints (phase 2; `checkpoint.json` sections already prepare for it);
  - a manifest-only `report.md` renderer (Studio's views and the report builder replace it).
- **Frontend:**
  - a layer-algebra expression language;
  - a browser-side emulator or any client-side re-implementation of model maths;
  - basemap tiles by default (reserved setting).
- **Platforms:** Windows/macOS process semantics are designed for (process groups, AF_PIPE, msvcrt) but CI-tested on Linux only.

---

## 16. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Memory: a warm full engine (1.6–2.5 GB) plus a full run (≈3 GB) plus OS on 8–16 GB | Engine host LRU bounded by count and memory budget; host recycles itself on fragmentation; preflight refuses and offers evict or stop; estimates shown in the launch and open dialogs; idle reaper |
| Pickle drift: core class changes break unpickling; instrumentation invalidates resume once | No new required attributes; `getattr` defaults; `incompatible` state with a refit action; `progress.py`/`runio.py` excluded from the fingerprint; release note |
| Thread oversubscription (measured 82–368 s vs 1 s) | Global thread budget; env + threadpoolctl + torch + GWRF `n_jobs` override in every worker and engine request; preview limited to 1 thread |
| Slow cancel inside long native calls | Inner MGWR, GWRF and physics ticks; 90 s grace then process-group SIGKILL; atomic checkpoint and artifact writes |
| Server restart, sleep or container recycle | Jobs and engine host in their own sessions; reattach by pid + `create_time`; `interrupted` → Resume from snapshot; heartbeat "sleep?" bands |
| Manifest races and section loss | `runio.update_manifest` under `run_lock`; scheduler per-run locks; `_finish` preserves post-run keys; reader merges stage files and flags stale ones |
| Emulator over-trust (uniform rel. error: albedo 246%) | Trust badges; hatching rules; impacts and packs from exact results only; preview-vs-exact error shown |
| ETA quality on new machines | Seed rates plus per-host medians plus online refinement; ranges, not points; calibration test |
| Event volume (debug ticks, simcheck with 84 nested runs) | Info level by default; ≤ 1 tick/s per span; 4 Hz wire coalescing; ≤ 4 KB lines; 200 MB debug cutoff |
| Network fragility of input jobs | Host pre-check; per-object ticks plus heartbeats; workspace cache; clear `missing input` readiness; offline mode |
| Disk growth (525 MB checkpoints, multiverse children) | Storage manager with sizes and guarded deletes; bundles exclude checkpoints; free-space preflight; `storage.low` event |
| Committed assets drift | `BUILD_INFO` source hash plus CI rebuild-diff; wheel smoke test |
| Contract drift between Python and TS | OpenAPI snapshot; generated types plus `contract.check.ts`; shared replay fixture for both reducers |
| Scope | Registry pattern lets 9 feature items proceed in parallel; milestone order M1 browse → M2 run+track → M3 Lab core → M4 setup + studies + exports |
| Security of a local server | Always-on token, HttpOnly cookie, Host/Origin checks, `--allow-remote`, `safe_path`, trusted-pickle registry |

---

## 17. Build plan

### 17.1 Work items

The parallel work breakdown is returned to the orchestrator. Ownership is exclusive by file. Shared files (app factory, router/kind registries, DB schema, frontend router, layouts, API client, design system, canvas and chart kits) belong to the two **foundation** items. Feature items plug in through registries (§10.2, §12.3) and never edit foundation files.

| Id | Layer | Depends on | Owns (summary) |
|---|---|---|---|
| `core-progress` | core | — | `sparc/core/progress.py`, `runio.py`, `cli.py`, `sparc/__main__.py` studio dispatch, their tests |
| `core-instrumentation` | core | core-progress | All other edited `sparc/core/*.py` (pipeline, ensemble, stacker, base_models, physics, **influence**, **data**, **synthetic**, diagnostics, baselines, response, causal, optimize, scenarios, climate, forcing, opendata, features_open, planner, emulator, placebo, simcheck, multiverse, reproduce, uncertainty, report, config, `__init__`), their tests, `scripts/make_studio_fixtures.py`, `tests/studio/fixtures/synth_run/` |
| `backend-foundation` | backend | core-progress | `sparc/studio/` core modules (app, settings, security, errors, db, workspace, events, sse, meta, schemas/common, routes/{__init__,system,jobs,stream}, jobs/*), `pyproject.toml`, foundation tests |
| `frontend-foundation` | frontend | — | `studio-web/` scaffolding, router, layouts, api/{client,sse,binary,resource,types}, stores/{jobs,ui,selection}, theme, components/ui, charts, map |
| `backend-projects` | backend | backend-foundation, core-instrumentation | `sparc/studio/projects/**`, `examples/**`, routes/{projects,inputs} |
| `backend-runs` | backend | backend-foundation, core-instrumentation | `sparc/core/catalog.py`, `sparc/studio/runs/**`, routes/{runs,layers,compare} |
| `backend-engine` | backend | backend-runs | `sparc/core/session.py`, `sparc/studio/engine/**`, `scenarios/**`, routes/{lab,plans} |
| `backend-studies-exports` | backend | backend-runs | `sparc/core/results_page/**`, `scripts/results_page/**`, `sparc/studio/studies/**`, `exports/**`, routes/{studies,exports,findings} |
| `frontend-projects` | frontend | frontend-foundation | `studio-web/src/pages/projects/**`, `api/{projects,inputs}.ts` |
| `frontend-tracking` | frontend | frontend-foundation | `studio-web/src/pages/tracking/**`, `stores/tracker.ts`, `api/tracking.ts` |
| `frontend-run-hub` | frontend | frontend-foundation | `studio-web/src/pages/runhub/**`, `api/{runs,analysis}.ts` |
| `frontend-lab` | frontend | frontend-foundation | `studio-web/src/pages/lab/**`, `api/lab.ts` |
| `frontend-studies-exports` | frontend | frontend-foundation | `studio-web/src/pages/studies/**`, `api/{studies,exports,findings}.ts` |
| `integration-e2e` | test | all of the above | `tests/studio/e2e/**`, contract fixtures, `sparc/studio/static/**`, contract check, `.github/workflows/studio.yml`, **the one-line `--ignore=tests/studio` edit to `.github/workflows/tests.yml`**, `scripts/check_studio_assets.py` |
| `docs` | docs | integration-e2e | `docs/studio/**` (incl. updating this spec to as-built, `doctest.py` and the api-doc check script `check_api_doc.py`), README section |

**Ownership notes:**
- **No file is owned by two items.** Shared directories are split by file: `tests/studio/fixtures/` holds the foundation's selftest files, core-instrumentation's `synth_run/` and integration's reducer golden.
- **`sparc/core/__init__.py`** (core-instrumentation) lazily exports `open_run` from `sparc/core/session.py` (backend-engine) through a module-level `__getattr__`, so it does not fail before that file exists.
- **Feature folders never import each other.** The compact tracker on the run overview is the foundation's `JobStrip`, and cross-item backend calls go through the lazy contracts of §10.2.

### 17.2 Milestones

| Milestone | Delivers | Items involved |
|---|---|---|
| **M1 — Browse** | Open the Providence example; import runs; browse every output tab and map layer | core-progress, backend-foundation, backend-runs, frontend-foundation, frontend-run-hub |
| **M2 — Run & track** | Launch, track, cancel, resume and reattach a fast run; Status Board; Activity | + core-instrumentation, backend-projects (minimal), frontend-tracking, frontend-projects (Launch) |
| **M3 — Lab core** | Selection, compile, preview, exact, inspector, library, compare, plans, climate, decision pack | backend-engine, frontend-lab |
| **M4 — Setup, studies, exports** | Full setup wizard and inputs; studies hub; reports; findings | backend-projects, backend-studies-exports, frontend-projects, frontend-studies-exports, integration-e2e, docs |

---

## 18. Review changes (completeness pass, 2026-10-01)

A completeness review checked this spec and `api.md` against the operation, output, simulation, tracking, results-page and legacy-server catalogs, and against the work breakdown. The changes it made are listed below. The `docs` item later appends an "As-built changes" section after this one.

**Gaps closed against the catalogs**
- **Features bootstrap for a brand-new city** (§9.3): a project with only target, id, x/y and a CRS can build open predictors.
- **Study files the CLI used to write** (§8): Studio writes `placebo.json/.md`, `multiverse_summary.md`, `simcheck_summary.md` and `benchmark.json/.md` itself, because the library functions only return dicts.
- **`study.reproduce` passes `config_dir`** (§8), so runs without `provenance.config_dir` can be reproduced.
- **CMIP6 cache path** (§4.3): `climate.cache` is absolutised to `<ws>/cache` in `launch.json`; `climate_stage` would otherwise write under `output.dir`.
- **Model-vs-causal outputs** (§11 item 21, §6.3, §6.4): the per-cell CATE and model slopes, plus the model's own-cell PD curve, are persisted (`causal_cells.parquet`, `model_effects`). This enables the overlay on the DR curve and the CATE maps.
- **Placebo children are readable** (§11 item 22, §6.2): `input_frame.parquet` is written for frame-input runs.
- **`influence.py` is instrumented** (§5.5, §11 item 23). The `influence.*` warning codes and the S1 per-predictor metrics had no emitter.
- **More CLI options exposed**: CMIP6 `periods`/`baseline`; the design CSV accepts absolute `value`s (expert full-table edits); scenario options `clip_to_support`/`mediators` are applied per request (§7.3).
- **`results.html` and `reproduce.json` placement** (§6.1): both are catalogued, and the catalog's `IGNORED` globs are defined.
- **New job kinds**: `input.stations` (no network calls inside a request) and the `run.external` pseudo-job (Mission Control for CLI runs, §5.14).
- **New endpoints** (api.md): `GET /api/runs/{rid}/truth`, sweep and comparison listings, and `DELETE /api/sweeps/{swid}`.
- **The "Truth vs recovered" card has an exact definition** (§8, §9.2), and the demo writes `truth.json.derived`.

**Contradictions resolved**
- **Exports location**: always `projects/<slug>/exports/<export_id>/`. The run `studio/exports/` was removed, and `export_id` is passed in the job params.
- **Run tabs**: one fixed id vocabulary, with tab status computed by the server from `OutputSpec.view`. The `outputs` field on route modules and the duplicate `run_tabs` source were removed (§3.2, §12.3).
- **Project nav**: Launch was added, and each nav entry has one owner (§3.1). The frontend-projects work item claimed "Overview", which belongs to tracking.
- **Event vocabulary**:
  - `cancel.requested.by` includes `shutdown`;
  - the per-job stream sends `end` and closes on overflow;
  - `job.progress` uses `job_id`;
  - `scenario.summary` is split into scalar metrics (`scenario.mean_delta`, `scenario.se`, `scenario.frac_extrapolated`);
  - `cv_row` is split into `cv_row.r2` and `cv_row.rmse`;
  - `cv.partition_skipped` is a registered code;
  - the PlanNode reason `required_by:<stage>` is documented.
- **Run status `queued`** (api.md) was added to the run state machine (§5.8). The fields of `job.json` and `state.json` were aligned.
- **`min_dose`** is a core post-filter (`planned_allocation`), not a Studio-only filter (§7.10, §11 item 15).
- **Preview mask**: it moved from a response header to the packed body (§7.5), because a 54k+ cell bitset can exceed proxy header limits.
- **Resume "byte-for-byte"**: the two documented exceptions are `threads` and the explicit current-config resume (§4.3).
- **`RunDetail` no longer embeds `studies` or `engine`** (api.md §6). Those come from endpoints owned by later items, which the runs item cannot import.
- **`BUILD_INFO.json` has no timestamp**, so the committed-asset rebuild-diff can pass (§13.2).

**Tracking made testable**
- **Unit accounting** (§5.4): units are counted by `task.end` with a `unit`, or by `k == n` ticks. Neither the core throttle nor the wire coalescing ever drops `k == 1` or `k == n`. Progress weights are the seed rates, identical in Python and TS.
- **Reference unit counts for full Providence** (§5.4): the ETA test no longer needs `plan_stages` (a foundation item cannot depend on core-instrumentation). The counts were corrected against `ResponseEngine.marginals`.
- **Nested runs inside studies** are attributed to the job's `children[]`, not to the job's own stage rail (§5.5).
- **Workers never write SQLite** (§5.6). Server-side `on_event`/`on_finish` hooks were added, plus lazy cross-item function contracts (§10.2).

**Untestable or fragile acceptance criteria rewritten**
- **Progress overhead**: "< 1% of an empty loop" is impossible in CPython; it is now "≤ 2× a no-op call with the same signature, nothing written" (§14.1).
- **Fixture determinism** is defined field by field (§14.1).
- **The cancel → resume equality test** runs with `threads=1` (§14.1).
- **Palette tests** use golden LUT values printed in this spec (§6.3) instead of one foundation generating fixtures from the other's code.
- **The replay e2e** creates the demo with the fixture's `n=40, seed=0`, from the one shared generator `synthetic.write_demo_project` (§9.2, §11 item 20). Otherwise the replayed outputs would fail the id and fold checks.
- **Studio CLI delegation** happens before argparse (so `--help` reaches Studio), and the install hint also fires when only FastAPI is missing, which is what the core CI job has (§10.9, §11 item 3).
- **The legacy CI job** would have collected `tests/studio`; the integration item adds `--ignore=tests/studio` (§14.6).
- **The docs doctest** runs through a guarded CI step that the integration item adds (§14.6).
- **Server never imports torch**: a torch-free import test now guards that requirement (§5.5, §11 item 24).

**Work-breakdown changes this implies** (the breakdown itself lives with the orchestrator):
- core-instrumentation also owns `sparc/core/influence.py`, `data.py` and `synthetic.py`, plus the new tests `test_demo_project.py`, `test_causal_cells.py`, `test_input_frame.py` and `test_import_light.py`;
- backend-foundation adds `--dump-openapi` and `tests/studio/foundation/test_meta.py`;
- frontend-foundation adds `JobStrip`;
- backend-projects adds the `input.stations` kind and features bootstrap mode;
- backend-runs adds the `run.external` pseudo-job and `scenario_slug` in `catalog.py`;
- backend-studies-exports adds `GET /api/runs/{rid}/truth` and writes the study `.md`/`.json` files listed in §8;
- backend-engine adds the sweep and comparison listings and the scenario mirror files;
- integration-e2e also owns the `tests.yml` one-line edit;
- frontend-projects declares nav entries for Setup, Inputs and Launch (not Overview).

## 19. As-built changes

This is the changelog of the specification after the completeness review (§18). It lists, milestone by milestone, every behaviour that differs from the text above as built, including the deviations each work item reported during implementation and review, and ends with the gaps still open at release. Wire-level details are in `api.md` §19. Where the body of this spec was brought up to date (the status line, J1 step 2, §8's emulator and attach rules, §9.1, §11 item 15, §14.5 item 1, §17.1), the entry here records the change.

### Foundation (core progress, core instrumentation, backend and frontend foundations)

**Core progress (`progress.py`, `runio.py`, CLI)**
- `span()` also takes `kind="run"` (`run.start`/`run.end`), so `run:<name>` path elements exist; span handles expose `.metrics`, `.summary` and `.set(**fields)` for closing fields. `tick()` and `metric()` take `lvl=` so debug-level ticks and metrics are gated by level (§5.5).
- `artifact` events carry the run-relative path in `path` and the span ancestry in `span_path`; any other clashing field is written `<key>_`; `job` is `""` without a configured job id (api.md §19).
- `limit_threads`/`set_threads` **never import torch**: they call `torch.set_num_threads` only if torch is already imported and otherwise rely on `OMP/MKL/OPENBLAS_NUM_THREADS` (checked to hold for a torch imported later). This keeps the server torch-free (§4.1, §5.5). A torch first imported inside a `limit_threads` block keeps that count after the block.
- `job_scope(job_id, *, sink, cancel_file=None)` takes the job id positionally or by keyword.
- Additive helpers: `progress.sink_path()`, `runio.atomic_open`, `runio.write_bytes_atomic`, `cli.tracked_command`, `cli.run_studio`, `EXIT_CANCELLED`, `STUDIO_DEPS`, `STUDIO_INSTALL_HINT`. The install hint prints "SPARC Studio is not available (…)" and "Install it with: pip install "sparc[studio]"".
- `sparc core reproduce` on a folder without `manifest.json` exits 2 with a message (it raised), and a failed reproduction's exit status now propagates through `sparc core`.
- `LogBridge` does not change logger levels; the host decides verbosity (the CLI and the job worker configure `logging` at INFO).
- `runio.update_manifest` on a folder without `manifest.json` starts an empty manifest instead of raising (documented; parallel items relied on it).
- `progress.py` is about 900 lines (mostly docstrings) against the estimated 350.

**Core instrumentation (`pipeline.py` and the instrumented modules)**
- `checkpoint_save` units are planned on **every** node that saves a checkpoint (S2_S3, baselines, cv_curve, S4, S5, climate, S6), not only S2_S3 as the §5.4 table shows, so planned units equal observed units.
- `climate_model` units sit on the `climate` node (its own stage span, where the downloads happen), not on S5; `run.plan` is re-emitted with the real model count through an `on_models` callback.
- `CHECKPOINT_KEY` maps S1 → S3 as well as S2_S3 → S3 (S1's influence result is restored from the S3 checkpoint); both become cached when S3 is done.
- The engine's baseline pass (`engine_init`) runs lazily in the first stage that needs the engine (normally S4), whose timing includes it; a fully cached resume skips it. S6 plans its 8 engine passes and audit step only for treatments that have an S4 response.
- Extended signatures: `plan_stages(…, *, block_m, write)`, `fingerprint_sections(…, *, coarse, cv_curve)`; `checkpoint_status` also returns `fingerprint` and `fingerprint_match` and reads the mode args from `studio/launch.json` or the manifest when `fast is None`; `multiverse.run_variant(…, changes, run_meta)`; `run_core` reads an optional `optimize.min_dose` (default 0).
- A `plan_stages` call made before S1 (Launch, the first `run.plan` with `cv.block_m: auto`) cannot know the main CV block, so it may plan one extra CV-curve partition; `run_core` corrects it with a re-emitted `run.plan` after S1.
- A distance curve whose partitions are all skipped is `stage.skip{no_partitions}` plus `cv.partition_skipped` warnings (it used to write a `cv_distance.json` holding only the main-block row).
- `optimize.json` Pareto points are `{budget, total_benefit, n_cells, n_segments, gini}` (`n_treated` counted segments).
- `run.plan` events omit `null` `est_*` fields and may drop labels to fit one 4 KB line; `plan_stages()` returns full nodes.
- GWRF forests use `n_jobs` capped by `OMP_NUM_THREADS` at fit and predict time, so one-thread predictions are bit-reproducible.
- Missing equity scores are replaced by the mean score before the optimiser.
- Atomic writes also cover the CMIP6 catalogue cache, the GHCN and ISD caches, planner GeoTIFFs and the GeoPackage.
- The fixture's run files sit at the top level of `tests/studio/fixtures/synth_run/` (next to `events.jsonl` and `FIXTURE.json`) so the replay runner copies them by artifact path; `scripts/make_studio_fixtures.py` works in a fixed folder with a fixed data mtime, so recorded paths and the fingerprint are stable.
- Python and numpy warnings still reach the `LogBridge` as `log.*` warning events.
- The code digest covers every top-level `sparc/core/*.py` except `progress.py` and `runio.py`, so adding `catalog.py` (M1) and `session.py` (M3) changed it again, as §11 anticipates; the committed fixture's `code_sha256` follows the code it was recorded with.

**Backend foundation**
- Bearer-authenticated unsafe requests without `Origin`/`Referer` are accepted (§10.8 "scripts and tests may send Bearer"); a mismatching `Origin`/`Referer` is still 403.
- An omitted `after` on the event and log endpoints reads from the first line (cursor 0 is the first line's offset).
- `TrackerSnapshot.log_capped` reports the 200 MB debug cut-off (§5.6).
- The CLI exits 1 instead of starting a second server when the lock names a live Studio process that does not answer `/api/health` within 10 s (§10.9 now says so).
- The terminal prints `SPARC Studio <version> - workspace <dir>` and `Open <url>/auth?t=…` (J1 step 2 updated).
- The network host check of the scheduler preflight (§10.4) is advisory, not a blocking check; `/api/system/netcheck` and the offline setting cover the user-facing side.
- `--reindex` empties the rebuildable tables, and each feature package registers a reindex hook for its own (runs, studies, exports, findings, scenarios, engine results); project rows are registered from `project.json` at reindex and at every start.
- `@job_kind` also takes `preflight`, `retry_params` and `threads` hooks.
- On a one-CPU machine `engine_threads` defaults to 1, so the defaults satisfy `threads_heavy + engine_threads ≤ thread_budget + 1`.

**Frontend foundation**
- TypeScript is pinned to 5.9.3: `openapi-typescript` 7.13 needs TypeScript 5.
- `@fontsource/archivo` has no width axis, so the display face renders at normal width (`font-stretch: 87%` has no effect).
- The `GridCanvas` < 10 ms check times the recolour step (`colorize()` into the pixel buffer); canvas drawing cannot run in jsdom.
- `HexbinScatter` draws rectangular 2-D bins, because `/stats/hexbin` returns `x_edges`, `y_edges` and `counts`.
- `RouteDef` has an optional `fullWidth`; Mission Control, the run track, map and Lab routes use it.
- The tab title reads `(2 running) ▶ 42% S2_S3 · SPARC Studio` with several running jobs.
- `layouts/resources.ts` defines the shell's `ProjectDetail` and `RunDetailHead` types and hooks (cache keys `project:<pid>`, `run:<rid>`, `run:<rid>:outputs`); `contract.check.ts` checks them too.
- The final-status toast fetches `GET /api/jobs/{jid}` for its headline numbers.
- `grid.bin`'s `zone` is an index into `GridMeta.zones` (`-1` = none).
- The chart kit's `Bars` and `DotRange` have no reference-line prop, so the Accuracy interval-honesty panels show the 0.90 target through outlined groups, the caption and the legend instead of a line (§6.4).

### M1–M2 (browse, run & track)

**Projects and setup**
- `POST /api/projects` takes `options.trust_pickles` (default false) for the Providence run import; imported checkpoints stay untrusted otherwise (§10.8). The Lab's trust dialog re-imports a run with trust when the user confirms.
- Slug collisions are `409 conflict` (no automatic suffix) except the `<name>_open` project; a project deleted without its files keeps its folder, marked deleted, so its slug stays taken.
- Config saves: `If-Match` is optional; an identical save returns the current version. The wizard saves with one atomic `PUT /config {raw}` (never a half-saved config), not a `PATCH` per section.
- Input jobs that link into the config edit `config.yml` in the worker under the project lock and leave `.config_note.json`; the server records the version from the job's events (workers still never write SQLite). The UI starts input jobs with `link: false` (the CMIP6 and forcing cards have an opt-in "Link into the config when done"), so each ends with **Link into config** showing the YAML diff first, as J2 step 6 describes.
- `input.cmip6` also links `climate.experiments` when the fetched SSPs differ; `input.ghcn` sets `planner.ghcn_station` only when unset. `features_join` on a project that has predictors appends the `open_*` predictors and maps only unmapped roles; bootstrap mode replaces predictors and roles; neither creates levers.
- `Project.report` overrides are stored in `project.json`, not in `config.yml`.
- File kinds map to folders `data|join → data/`, `layers|features|forcing|climate → inputs/<kind>/`, `other → other/`.
- The impact preview compares in non-fast mode at the top level and uses each run's own mode arguments per run (a fast run shows no `core` change for a `tune_lambda` edit).
- Join tables use core's keys `key`/`right_key` (not "on / right_on").
- Inputs has its own route `/p/:pid/setup/inputs` (same Setup page) so it can carry its nav entry; `/p/:pid/setup` redirects to the data step.
- Setup's completion dots for scenarios, analysis and about come from the draft and its validation (the readiness spine has no rows for them).
- The CRS picker is a curated offline EPSG list (WGS84, Web Mercator, UTM zones on WGS84/NAD83/ETRS89, common State Plane and national grids) and accepts any typed EPSG code; there is no EPSG search endpoint.
- The data check applies `config_patch` as a JSON Merge Patch and its preview carries the `planner.layers` columns, so the People & land cover card can draw its map.
- The packaged `brown4.csv.gz` is 1.6 MB (§9.1 updated).
- `GET /api/projects?archived=true` includes archived projects with the active ones.
- The input cards show a job strip and links to Mission Control and the job tray rather than redrawing per-object progress (CMIP6 models, Sentinel-2 scenes) inside the card.

**Runs, run hub and tracking**
- `RunSummary.studies` = attached study ids; stages a finished run skipped carry **Re-run with <stage>** (Launch prefilled with `?from=<run_id>`), because a resume replays the run's own snapshot and cannot add a stage; `overview.outputs_grid` lists expected outputs plus pending post-run actions and skipped stages; `overview.studies` always has eight chips; uncertainty sources are matched to studies by `studies.out_dir`; Status Board cells show queued post-run jobs as running with reason `queued`; the runs routes answer `503 not_ready` while the registry starts; a cancel-requested job that exits on SIGTERM ends `cancelled` (api.md §19 has the wire details).
- A Studio run that completes becomes its project's active run when the project has none; deleting the active run clears it.
- Resume (§5.9) is refused only for a run without a launch snapshot (a CLI run), a complete run, or while a job runs on it. A checkpoint that no longer matches the snapshot does not block it: the checkpoint card says "a resume refits", and the resume reuses nothing.
- Imported runs with a manifest but no `run_state.json` get `created_utc` = manifest time − Σ `timings_s`.
- Skipped stages read from a manifest carry the `plan_stages` reasons; Status Board cells show `disabled` for config-disabled stages; `overview.timings` includes `finish`.
- `DELETE /api/runs/{rid}/checkpoint` works on runs imported in place (an explicit checkpoint-card action).
- The output catalog adds an `optional` flag, a `studio` root and extra ignored globs.
- The view sections' inner shapes are defined by the run hub's TypeScript types (`studio-web/src/api/runs.ts`), which the server builds (api.md §6.1 fixes only the keys).
- Hexagon mode is computed in the browser; map-legend and Relationships brushes become uploaded mask blobs (a copied URL works within the same workspace and run); the Accuracy histogram brush is a portable filter.
- Interval honesty (Accuracy) outlines groups more than 5 points under the 0.90 target, and states the target in the caption and legend (see the frontend foundation entry on reference lines). Response shows the realised dose as a second "requested vs realised dose" chart next to the dose–response line, because the line-and-band chart has no secondary axis.
- A missing causal DAG audit reads "No DAG audit in this run" (it is off unless configured), not "older code".
- "Open in Lab" on a configured scenario opens `/r/:rid/lab/library?configured=<slug>`; "Clone to edit" creates a Lab scenario from the configured scenario's `doc` and opens it.
- Tracking: a page opened mid-run replays the job's event log (up to about 48 MB) for exact progress, otherwise it rebuilds approximate counters from the snapshot and says so (no exact reducer state in the snapshot). The baselines forest plots per-fold baseline MSE minus the stack's pooled out-of-fold MSE; the S7 panel shows totals and links to the Budget view (Pareto points are not events); timeline warning ticks come from `GET /events?types=warning`; epoch losses are kept beside the reducer state; jobs without a plan show only the stages seen; per-stage estimates on the rail are the client cost split scaled to the header ETA; **Duplicate with changes** opens Launch with `?from=<run_id>`.

### M3 (Scenario Lab)

- **Core code fingerprint.** `sparc/core/session.py` (`open_run`, `config_for_run`) is a top-level core module, so adding it changed the core code digest (§11). Checkpoints whose `checkpoint.json` was written before it report `code_match: false` (checkpoints without the sidecar, from before the core instrumentation, report `null`: the code that wrote them is unknown), and exact results computed before the change read as stale; results computed afterwards record the new digest and are current. This is the documented consequence of adding a core module.
- **Scenario status** (§7.4): a preview marks a draft `previewed`. The Lab previews each edit before its autosave, so a save whose `{edits, options}` match the latest preview stays `previewed`. Other content changes make the scenario a `draft` again.
- **Shared exact cache** (§7.4, §7.7): when a scenario's exact run is a cache hit on a result made for another scenario with identical content, the result is copied as this scenario's own. It is then listed, opened in the inspector, compared and packed like any other.
- **Engine actions**: the memory refusal's evict action is `POST /api/runs/{rid}/engine/evict`, because actions are POST/GET only. Exact requests on a run that is not loaded run the memory preflight. While such a request loads a cold run, the run's engine state is `loading`, carrying that request's job id. Plans, scenarios and sweeps cannot be deleted while a job on them runs (`409 active`).
- **Engine host socket** (§7.6): on POSIX the host binds `<ws>/engine/host.sock`. When that path is longer than `AF_UNIX` allows, it binds a short owner-only socket in the temp directory instead, and `host.json` records the path it actually bound. The Windows `AF_PIPE` address is still untested.
- **Brush clamping** (§12.5): the map kit's brush clamps each cell to `[min(lo, base), max(hi, base)]`, the engine's `_bounds` rule. A cell already outside the lever bounds is never pushed further out, and an add never becomes a decrease.
- **Saved brush strokes** stay a per-cell `blob:` edit of their lever (label "brush"). A reopened scenario cannot load them back into the brush layer, because there is no blob read endpoint. New strokes become another per-cell edit that adds to the saved one, and the Lab says so.
- **Scenario Lab nav entry** (§3.1): `Project.last_run.has_checkpoint` keeps the entry disabled until the run it opens has a checkpoint. That run is the active run, else the latest one.
- **Compare** (§7.12): every pair is read A − B (item a minus item b), in the table, the paired SE and the difference layer.
- **Climate explorer** (§7.11): the explore reply carries today's exposure as `present` (the `summarize_projections` dict). Each projection carries the selected statistic's warming, which drives the client-side future maps.
- **Decision packs** (§7.13): realised-change maps show the size of the change (lightest = unchanged), and the narrative states likely ranges from the smaller to the larger value. Pack downloads go through `POST /api/exports`, which arrived in M4; until then the Lab's pack buttons reported the missing route.
- **Emulator**: the `build_emulator` action targets `POST /api/runs/{rid}/actions/emulator` (`post.emulator`, added in M4). Without an emulator the preview reports `no_emulator` with that action, and exact runs work.
- **Spill** is the total change inside and outside the edited cells plus `outside_share`; rings step by `max(30 m, cell size)` out to 2 km.
- **Engine host**: it disables progress-tick throttling while serving a request (every fold ticks); at server shutdown it keeps running while an engine job is live, unless jobs are being stopped; a host that recycles itself writes `engine/recycle.json` and the executor re-opens the most recently used run at most once per 300 s; `SPARC_STUDIO_ENGINE_SLACK_GB` (default 1.0) is added to memory needs.
- **Incompatible checkpoints** (§7.6): the state is stored in `<run>/studio/engine/status.json` keyed on the checkpoint, and its action is **Refit S2/S3** = a re-run of the run from its launch snapshot (`POST /api/runs/{rid}/rerun`). The untrusted-checkpoint action re-imports the run with `trust_pickles`.
- **Packs** are zip files; a decision pack is a draft when its scenario has no exact result, a plan pack when the plan is unverified (`exports.options_json.draft`).
- **Compare** regions are region ids or `zone:<code>`, and results always include `all`.
- **Hatching** on the design map: edited cells are always tinted and are hatched only when the preview is flagged unreliable (J6: a local edit is not hatched); result layers hatch cells with extrapolation > 1.
- **Sweep fits**: only the saturating fit has a closed form from the reported A and d_s; the linear fit is drawn as the least-squares line through the origin, and the sigmoid fit shows only its d90 line.
- **Climate warming** values are in target units, as `summarize_projections` produces them.

### M4 (studies, exports, findings, results page, integration)

**Studies and post-run actions (backend)**
- Studio placebo, simcheck and multiverse studies are **attached** to their target run at creation, so a finished study feeds that run's uncertainty report without a separate attach (§8 updated). Detaching a finished study also re-enqueues `post.uncertainty` when the run already has a report.
- `post.emulator` is registered without `needs_checkpoint` and checks the checkpoint in a start-time preflight, so it can wait in a Launch "then" chain; `POST /actions/emulator` still refuses a run without a checkpoint (§8 updated).
- `post.writeup` uses the merged manifest (stage-file sections included), as §8 says; the first build read raw `manifest.json`.
- A simcheck parent folder whose replicates sit in sub-folders (Providence `simcheck/{null,null_direct,physics}`) is imported as **one study per sub-folder**, each matching its `uncertainty.json` source; an empty folder (`simcheck_a`) is refused and reported as a warning by the Providence example.
- `Study.status` uses the job statuses (`succeeded` for imported folders). A reproduction's child keeps origin `reproduction`; placebo and multiverse children are `study_child`.
- When a study ends, every child it has is re-derived, so a cancelled or failed pass never leaves its fitting child marked running.
- A fitting study child has a live `run_state.json` and no `run.core` job of its own. The runs registry reads such a row (origin `study_child` or `reproduction`, a `study_id`, or a manifest whose `run_meta` names a study) as `running`, never `external_live`, and its watcher never starts a `run.external` pseudo-job for it, so a full rescan (a server restart) while a child fits leaves it alone. A row an older server flipped to `external_live` is put back to `study_child` on the next pass. Study-child ids use `run_state.started_utc` before the manifest time, so a child indexed while it ran keeps its id after `--reindex`. The studies hooks still re-derive child rows at `run.dir`, `run.end` and the end of the study.
- Study requirement codes are config keys or run files (`manifest`, `predictions`, `checkpoint`, `planner.layers`, `physics.roles.canopy`, `physics.roles.impervious`, `data.path`, `manifest.config`, `config_dir`).
- `study.reproduce` gets `config_dir` from `launch.json`, else the folder of the config given at import, else the manifest's `provenance.config_dir`.

**Exports, reports and findings (backend)**
- The results-page builder lives in `sparc/core/results_page/` with `build_results_page()`, its template and `python -m sparc.core.results_page`; `scripts/results_page/build_page.py` is a short shim, the old template is gone, and `check_page.py` targets the packaged builder and gained `--build`, an HTML-shell check and a walk through every CV fold. On Providence the page differs from the old one only by the computed Pareto caption and the added `hd_95_ssp245_mid`; the null-artefact caveat, the single-generator wording and the uncertainty "vs null" column are unchanged. Corners of runs with `data.reproject_to` are computed in the reprojected frame.
- `POST /api/exports` validates everything it can before creating the row (ids in the project, outputs, layers, CRS, placebo study) and refuses a client-sent `export_id`; each export keeps an `export.json` record, and a `running` row whose job ended is reconciled when read.
- The GIS pack is **one zip**, and a Markdown findings export is a zip with its images, because `Export.path` is a single file; the Markdown report embeds its images as data URIs.
- Bundles never follow symlinks out of the run folder.
- Reports: the sections are `summary, accuracy, validation, scenarios, plans, climate, equity, caveats, limitations, provenance, findings` in that order (`SECTIONS`, their titles and one builder per section in `_BUILDERS` of `sparc/studio/exports/report.py`); every number is written from the run's files (no canned text); images are data-URI `<img>` elements, at most six maps of about 360 px; SVG is never inlined; climate sentences without warming numbers are left out instead of failing.
- Findings: a run of another project is refused; image uploads are staged and content-checked before they replace the current image; the findings reindex keeps only bare image names.
- The report preview checks its ids against the project, as the export does.

**Studies, exports and findings (web app)**
- Report and findings export requests always send their id lists (possibly empty), so the selection is explicit; the bundle always sends its output list and needs at least one output (an empty list would mean "all").
- The Findings notebook loads a project's findings and filters by run in the browser, so reordering under a run filter keeps hidden findings' positions; on `/findings` reordering needs a chosen project.
- The literature card has no launch form (the response stage computes the panel).
- A finished simcheck replicate without a share (null generators) is drawn in the no-data colour with "–", so it is not mistaken for pending; gate failures are hatched.
- The report preview is made inert before it reaches the sandboxed iframe: scripts, frames, `on*` handlers, `javascript:` URLs, `<meta http-equiv=refresh>` and `<base>` are removed and a CSP meta forbidding scripts is added.
- Running studies refresh their views every 10 s (status events only fire on status changes); the Studies hub matrix refreshes while any cell runs.
- The multiverse form launches every built-in variant by default; custom variant keys are validated as the server does; reproduction tolerances must be above zero; **Run again** starts from the last study's parameters.
- The emulator card reads the manifest's `emulator` section (falling back to `emulator.json`).
- Only studies the server accepts are offered: finished placebo studies for the results page, finished studies as uncertainty sources, Studio simchecks to continue; imported studies are never offered for file deletion.

**Integration (contract, end to end, CI)**
- The replay journey plays the fixture at ×1 (§14.5 updated) and creates its demo through the API with the fixture's `n = 40, seed = 0` (the one-click demo uses n = 96).
- The real fast-run journey cancels twice (during S2_S3, and again once S4 runs) to prove that resume reuses cached stages, then walks the Lab, compare, climate, plans and a decision pack.
- `contract.check.ts` checks request bodies for type and nullability only. Its `Gate<…>` allowances for known client nullability differences fail the typecheck once they no longer apply; at release the list is empty, because the client types of tracking, projects, the run hub and the Lab were made nullable where the server sends `null` (api.md §19).
- The accessibility test also requires an accessible name on every link. Every page fits 390 px: the children of `.stack` and `.card` have `min-width: 0` (foundation), so Mission Control's tab strip, the job picker, log lines and the Docs tab's documents scroll or wrap inside their own boxes.
- The performance fixture has 54,901 cells (the synthetic generator at n = 281, real predictions). The tracker folds a replayed event log in slices of 500 events and yields to the browser between slices, so opening a finished 20,000-event job meets the 100 ms frame budget, as the live-streaming case does.
- The Providence journey uses `$SPARC_PROVIDENCE_RUNS` or `output/core/providence`, copying a run outside the repository to a temp folder before importing it.
- The wheel smoke test builds from a hard-linked copy of the sources (no `build/` left in the tree); locally it installs with `--no-deps` plus the server dependencies, and CI installs the full `[studio]` extra with CPU torch.
- `tests.yml`'s legacy job passes `--ignore=tests/core --ignore=tests/studio`. CI sets `SPARC_E2E_REQUIRE_BROWSER=1` and `SPARC_CONTRACT_REQUIRE_TS=1`, so a missing browser or Node fails instead of skipping; the old-app guard compares the `sparc-desktop` and `sparc/server` trees at HEAD with a pinned baseline commit (`STUDIO_BASELINE`, the last commit before Studio work), because `pi-jepa-dev` shares no history with `main` and the merge-base/push-range version failed in CI. Update the baseline when the old app is changed on purpose.

### Review fixes (2026-10-02)

- **Trusted pickles** (§10.8): the scenario emulator (`post.emulator`) unpickles the run's checkpoint, so it follows the engine's rule: a run imported without trusting its checkpoint is refused (`409 untrusted_pickle` with the trust action) when the emulator is asked for, and the job's start-time preflight refuses it too.
- **Readiness spine** (§9.3): a config section saved with the wrong type (`climate: on`, `data: 5`, `physics.roles` as a list) counts as empty for the spine, and the error shows in the *Config valid* row; the project list and project pages stay readable so the config can be fixed in Studio. `Project.report` shows numbers in the config's `report` block as text and other non-strings as empty.
- **Import with `copy_data`** (§9.1): inputs that share a file name (the data table `a/data.csv` and a join `b/data.csv`) are copied side by side (`data/data.csv`, `data/data-2.csv`) with a warning instead of the later overwriting the earlier.
- **Imported run ids** (§4.3, §9.1): an imported folder's `studio/launch.json` is untrusted input; its `run_id` names the side folder `imports/<run_id>/` only when it is a plain id that no other run folder holds (a copied Studio run gets its own id), and a refused import removes only the side folder it created.
- **Deleting a run, its checkpoint or its outputs** waits for the engine host to evict the run in a worker thread, so a long engine request on another run no longer freezes the server.
- **Config validation never fails a save** (§9.4): the checks are robust to values of the wrong type (they are the schema's type issues), so a save that is already written always reports its new version, and one bad config no longer takes the project list down. `POST /config/validate` and the readiness spine stay available; an unexpected failure of the checks is one `validation_failed` error.
- **Impact preview of a blank project** (§9.4): a config without `data.path` compares as "no data" (a name edit changes nothing, adding the data file changes the input data); a config core cannot fingerprint shows every section as changed instead of failing the review dialog.
- **Data check of an unusable config** (§9.3): values S0 cannot use (a column name that is a list, `predictors` that is not a list of names, a non-positive cell size, a CRS pyproj cannot read with `reproject_to`, `actionable.canopy: 10`, `qa.clip` values that are not `[lo, hi]`) are `422 validation` with the offending config paths, not a server error.
- **A CRS pyproj cannot read** (`data.crs: garbage`, `EPSG 32619`) is a `bad_crs` validation error and counts as no CRS for a run: no lon/lat on maps and hexagons, and the formats that need earth coordinates (GeoTIFF, GeoJSON, GeoPackage, `EPSG:4326` selections, the GIS pack) are refused with `needs_crs`.
- **Deleting a run** (§4.3) takes the rows that hang off it with its folder: engine results, plans, sweeps, comparisons, blobs and saved regions. A scenario whose exact result lived in the deleted run is no longer `exact` (it goes back to `draft` when no exact result is left) and can be edited again; exports and findings stay with the project.
- **One run job at a time** (§4.3, §5.9): a resume checks the run and queues its job under a per-run lock, so a double click or a second tab gets `409 active` instead of a second resume that re-ran S0, S7 and finish over the completed run (and renamed the launch snapshot twice). Retrying an old `run.core` job is a resume with the same guards, and a resume that would start on a run completed meanwhile fails before it touches the run.
- **Chains** (§10.3): a job whose dependency **succeeded** still starts when that finished dependency is deleted (from Activity or by retention); only a failed or cancelled dependency cancels it.
- **Killed writers** (§5.13, §11): a worker killed while it saved a file (Force stop, the OOM killer during a 525 MB checkpoint) left a hidden `.<file>.<pid>.<thread>.tmp` that nothing removed. Core's next atomic write of that file removes the temporaries of writers that no longer exist; Studio sweeps the run folder when a `run.core` job ends, and *Delete checkpoint* frees them too.
- **"Being computed"** (§3.2): a run tab is shown as being computed only while a job that makes its outputs runs (the run itself for stage outputs, a planner pack for the Planner tab, …); building the emulator no longer marks Planner and Uncertainty as being computed. Track follows any job on the run.
- **Study children** (§4.3, §5.11): a placebo or multiverse refit and a reproduction are not the project's "Last run" (Home, the Scenario Lab entry) and do not count in the readiness spine; they are still listed with their origin in the run lists.
- **Commit of a run** (§11): `manifest.git_commit` and the report's commit line name the commit of the code that ran (git in the core package's folder, as `provenance.git`), not of the worker's working directory, so Studio runs no longer report `commit None`; decision packs read `provenance.git.commit` first. `sparc/core/report.py` is fingerprinted, so checkpoints written before this fix report `code_match: false` and their exact results are marked stale (§11 item 6).
- **Report headline** (§6.7, §9.3 about): the report's summary leads with the headline scenario chosen in Setup → About (stored as its slug), as the run Overview does; before, the report matched names only and fell back to its automatic pick.
- **Budget top cells** (§6.4): the table lists the treated cells that cool most (closed loop), in that order, ties by dose; before, it ranked by dose alone, so cells sharing a dose came in row order and the table did not match its "by cooling" caption.
- **Literature panel** (§6.4 Response): the SPARC rows carry the extrapolated share of the scenario they scale from.
- **Findings export** (§6.10): a pinned chart's numbers are exported as a table with every row and the column labels and units as written (`City-mean ΔT (°F)`), not as JSON cut at 400 characters; other nested snapshot values are written in full.
- **People-objective plans** (§7.10): the people objective ranks cells by cooling weighted by the residents around them; the plan's planned cooling, its Pareto curve, the planned-benefit map and the field kit report the allocation's actual (unweighted) cooling in °·cells, and the caption gives the weighted total separately. Before, the weighted sums were shown as °F·cells of cooling, so a people plan appeared to cool more than the cooling-optimal plan.
- **Extrapolation not computed** (§7.7, §7.13): a DRAFT pack (emulator preview) has no extrapolation scores; its brief now says the share of edited cells outside the observed conditions was "not computed (preview)" instead of stating 0%, and the result's `extrapolated_edited` is null rather than 0.
- **Unverified plans** (§7.10–§7.13): the planned benefit of a plan is a footprint total on each treated cell (the cooling its dose brings to its neighbourhood, °·cells), never a per-cell ΔT. Compare shows an unverified plan's city mean, cost and cooling per cost only (no edited-area mean, regions, equity, exposure or difference map, with a note to verify the plan); Climate × adaptation refuses it; its plan pack is a DRAFT built from the emulator preview of its doses (refused without an emulator); the planned-benefit map is labelled in °·cells. Before, compare, climate and the draft pack read the footprint totals as ΔT, overstating the cooling on treated cells 3–4× and showing none around them.
- **Sweep fits** (§7.9): a sweep is fitted twice, on the requested dose (drawn as the overlay, its d90 line and caption, on the chart's dose axis) and on the selection's mean neighbourhood dose (quoted in the caption in its own units, with a neighbourhood-dose column in the points table). Before, the neighbourhood-dose fit was drawn and captioned on the requested-dose axis, so a regional sweep claimed to saturate several times sooner than its points do.
- **Check across runs** (§7.12): the spread across runs is the specification band of the scenario version it evaluated. An exact result that already exists takes the band as soon as the check finishes (the inspector's envelope and "excludes zero" update), a check of a draft that is then edited does not carry over to the new version, and the Across runs panel calls the spread the result's band only when the check is of the result's version.
- **Compare pair wording** (§7.7, §7.12): the plain-language line of a pairwise difference compares the two items ("A cools the city 0.97 °F less than B … Confident there is a difference."; each item's own effect when one cools and the other warms) instead of reusing the single-result phrase, which read "Warms A vs B by 0.97 °F … Confident it warms" when both scenarios cool.
- **DEMO packs** (§7.13, §9.2): packs of the synthetic city carry the DEMO DATA marking in the brief's title, a banner, the README and `summary.json`, and give its place as "a fictional location" rather than the coordinates of the fictional placement.
- **Equity wording** (§7.7, §7.13): the concentration index divides by the mean cooling, so it has the same sign for a cooling and a warming of the same shape. Equity results carry the resident-weighted mean cooling, and the pack, the Lab caption and the report say whether the cooling or the warming concentrates in the higher or lower quintiles, instead of calling a warming scenario's concentration a benefit.
- **Progress of jobs without a plan** (§5.4, §5.10): an engine sweep, batch, frontier or check across runs has no `run.plan`, so its progress came from the latest shallowest tick: the fold ticks of the current item, which went 0 → 1 on every dose (a 40-dose sweep read 100% at dose 4, and the bar went backwards). Both reducers (Python and TS) now use the nested item fraction when tasks that count items (`k` of `n`, `k` from 1) are open: each gives `(k − 1 + inner) / n` from the innermost out, with `inner` the latest tick inside the innermost item (1 once it ended ok), so the bar moves once from 0 to 1 over the items. Ticks before the first item (loading a cold run) still show, so a sweep on a cold run starts over once when its first dose begins. Plan-driven progress (core runs) and study progress (mean child progress) are unchanged; the reducer state has a new `kn` list (the open item levels).
- **Recycle is transparent** (§7.6): a host that recycles or stops when idle removes its `host.json` as it closes its socket and refuses work it receives meanwhile ("HostStopping", nothing run); the server waits for that host to exit and sends the request to a new host. Before, an exact scenario queued behind one that triggered a recycle failed with "engine host exited". Host processes that exit by themselves are reaped at once (no zombie children of the server).
- **Stopping jobs on exit** (§7.6, §10): when the server stops its jobs, a running engine job ends "cancelled" like any other job: it is cancelled before the engine host is stopped. Before, stopping the host first made it "failed: engine host exited", and a sweep or plan it belonged to was marked failed.
- **Unfinished runs and the engine** (§3, §7.6): a run that stopped after S3 without finishing (cancelled, failed, interrupted) has a checkpoint but no manifest, and the engine cannot rebuild its data and folds without the manifest. Its engine chip says so and offers **Resume the run to finish it** instead of **Open engine**, and opening the engine or running a scenario exactly on it is refused with the same remedy. A run that is still running says the engine opens once it finishes. Before, "Open engine" failed every time with a missing `manifest.json` and was offered again.
- **Recovery caveat without a simulation check** (§6.4 Overview, §6.7, §6.9): a run without a simulation check no longer says the model "recovers about two-thirds of the true canopy effect" (a number never computed for the run; the demo recovers 80–91% of its planted canopy effect and other seeds overstate it). The caveat, in Studio and on the results page, says the recovered share is not computed until a simulation check runs and that scenario magnitudes may be too small or too large.

### Documentation

- `docs/studio/doctest.py` runs every fenced shell block marked `# runnable` in the user and developer guides, the release notes and the README against a fresh temporary workspace, and also checks every other shell command statically (`sparc studio` and `sparc core <command>` flags parse with the real parsers and `--project` configs exist, scripts and test paths exist, npm scripts and `sparc` modules exist, `python -c` snippets compile and import names that exist; `<placeholders>` are not checked). The runnable examples call the API with `curl -fsS`, so an HTTP error fails the block. `# runnable (needs: node)` marks blocks that need `studio-web/node_modules`; unmet needs fail in CI.
- `docs/studio/check_api_doc.py` diffs api.md's endpoint definitions against the generated OpenAPI document (191 operations at release, none missing or extra); api.md's route spellings now use the server's parameter names, and `POST /api/runs/{rid}/engine/evict` has its own definition.
- The guides' screenshots (`docs/studio/img/`) come from the end-to-end runs; Home, the project overview, Activity, Settings and the Studies hub were captured with the same harness on the replay runner, and the exact-result inspector comes from the M3 milestone check (the e2e journeys do not scroll to it).

### Known gaps at release

These spec points are not met by 1.0.0; each is tracked with its owner.
- **ETA after resume**: the estimate does not discount cached stages at first (a resumed fast run showed ≈16 min for a 40 s resume).
- **Uncertainty sources and relative paths**: `uncertainty.json` sources written as relative paths (Providence: `output/core/providence/simcheck/null/`) are matched against the server's working directory, so they match the imported study rows only when the server runs from the repository root.
- **0.90 reference line** on the Accuracy interval-honesty charts needs a `refLines` prop on the chart kit's `Bars` (§6.4); the target is conveyed by outlines, caption and legend.
- **Brush strokes** saved in an earlier session reload as per-cell edits that can be removed but not repainted (no blob read endpoint).
- **Tests**: the Providence tests skip unless `SPARC_PROVIDENCE_RUNS` (or `output/core/providence`) is available.
- **Test coverage**: some paths are exercised only through typecheck and shared helpers, not a dedicated page test (Lab promote, ladder, batch run, sweep creation and design-CSV import); the run hub's `views.json` fixture was generated once by a script that is not in the repository (the server's `runs/views.py` now builds the same shapes); the long-running study views (simcheck grid, multiverse, reproduce) were checked live only through a placebo study.
- **Platforms**: the engine host's Windows `AF_PIPE` transport and Windows/macOS process handling are implemented but untested; the engine's mid-load `loading` progress and the idle-shutdown watchdog have no automated test.
