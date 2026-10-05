# Release notes — SPARC Studio

## Since 1.0: heat stress, verdicts, Windows and macOS

- **Heat tab** (run hub, Decisions): the US National Weather Service heat index of every cell with the campaign dewpoint, residents in each NWS category (Caution, Extreme caution, Danger, Extreme danger) today, with the adaptation package and in each CMIP6 future, as a range between constant dewpoint and constant relative humidity; a five-part plain brief (how hot, who, as the climate warms, what helps, how sure); a what-if dewpoint control. See [Heat stress and verdicts](USER_GUIDE.md#heat-stress-and-verdicts).
- **Verdicts**: every configured scenario reads **Robust**, **Direction only** or **Not established**, with the reasons, from its uncertainty envelopes and the no-effect simulation (Scenarios, Uncertainty and Heat tabs; results page).
- **Heat risk of a design**: an exact Lab result's impacts lead with the residents it moves out of Extreme caution or worse, today and in each future; the decision pack and the report export ("Heat stress" section) carry it. Impacts cached before this are recomputed once.
- **Map**: "Heat stress" layers (heat index and NWS category, today and with the package) in warm category colours.
- **Results page**: a Heat stress section and verdict pills.
- **Windows and macOS**: CI runs the start-to-end self-check (`python -m sparc.studio.selfcheck`) and the fast suites on both. Fixed: on Windows the Scenario Lab's engine host never exited after Studio stopped (a named-pipe wait that closing does not interrupt), and Studio could not kill it afterwards; atomic file replacement retries through Windows sharing violations.
- **No change to checkpoints**: the heat code sits in `sparc/core/heat/` (outside the resume fingerprint), so existing runs resume as before.

---

# SPARC Studio 1.0

SPARC Studio 1.0.0 ships in the `sparc` package (1.0.14) as the `studio` extra. Date: 2026-10-02.

- **New**: SPARC Studio, a local web app for the modern pipeline (`sparc/core`): set up a city, launch and track runs, browse every output, design and verify cooling scenarios, run the validation studies, and export packs, GIS layers, reports and findings. Install with `pip install "sparc[studio]"` and start with `sparc studio`; see the [user guide](USER_GUIDE.md).
- **Read first if you have existing runs**: checkpoints written before this release **cannot be resumed** (a one-time change of the resume fingerprint, [below](#one-time-change-checkpoints-from-earlier-releases-cannot-be-resumed)). Their outputs stay readable.
- **Core pipeline**: structured progress events, run and checkpoint sidecars, a planning API, persisted per-fold scenario detail, in-memory run sessions, a budget planner API and a catalogue of outputs ([new core APIs](#new-core-apis-and-files)); and a fix to how the equity column lines up with the data ([fixes](#fixes)).
- **Unchanged**: the legacy desktop app (`sparc-desktop/`) and its server (`sparc/server/`) are untouched and remain available until Studio reaches parity with them. `sparc run` (the legacy pipeline) is unchanged.

---

## Install and upgrade

```bash
pip install -U "sparc[studio]"        # or, from a checkout: pip install -e ".[studio]"
sparc studio
```

Studio needs no Node: the web app is pre-built inside the package. The `studio` extra adds `fastapi` (not 0.136.3, which was a malicious upload), `uvicorn`, `threadpoolctl` and `httpx`. The core pipeline's own dependencies are unchanged, and `sparc core …` works without the extra; `sparc studio` without it prints `pip install "sparc[studio]"` and exits with status 2.

---

## One-time change: checkpoints from earlier releases cannot be resumed

A run's `checkpoint.pkl` is reused by `--resume` (and by Studio's Resume) only when the run's **fingerprint** still matches. The fingerprint hashes the effective config, the mode arguments, the data file's identity and a **digest of the pipeline code**: every top-level `sparc/core/*.py` (sub-packages such as `sparc/core/results_page/` are not part of it). This release instruments those modules for progress reporting and adds new ones (`catalog.py`, `session.py`), so the code digest of every earlier checkpoint differs.

What this means:

- **Resume refits from the start.** `sparc core run --resume` on an older run prints "checkpoint … is from a different config/data/code version — ignoring it", emits a `checkpoint.mismatch` warning and runs every stage. In Studio, runs made before this release come from the command line and have no launch snapshot, so they cannot be resumed there either: relaunch them from Launch. For any Studio run whose checkpoint no longer matches, the checkpoint card says "the checkpoint does not match the launch snapshot (code changed): a resume refits", and a resume reuses nothing.
- **Everything else keeps working.** Old checkpoints still unpickle: no pickled class gained a required attribute. Every output of an older run stays viewable in Studio, the run can be imported, and the Scenario Lab can still load its checkpoint (once you trust it). Earlier checkpoints have no `checkpoint.json` sidecar, so the engine cannot compare their code and reports `code_match: null` (unknown); exact results computed from them now are ordinary results.
- **From now on a code change is visible.** Checkpoints written by this release carry `checkpoint.json` with their code digest. After a later edit of a fingerprinted core module (or an upgrade that brings one), their engine reports `code_match: false`, and exact results computed before the change are marked **stale** (they are still shown; **Run exact** again to refresh them).
- **Future instrumentation edits will not repeat this.** `sparc/core/progress.py` and `sparc/core/runio.py` are now excluded from the code digest, so changes to progress reporting or atomic I/O no longer invalidate checkpoints. Editing any other core module changes the digest, as before.

Check a run without loading its checkpoint (never unpickles, so it is instant even for a 500 MB checkpoint). Pass the config a resume would use to compare the data and config too:

```bash
python -c "from sparc.core.pipeline import checkpoint_status; print(checkpoint_status('output/core/providence/providence_uhi_fast', 'configs/core_providence.yml'))"
```

A checkpoint written by earlier code has no `checkpoint.json`: the answer has `present: True` but `fingerprint`, `done`, `matches.code` and `fingerprint_match` are all `None` (unknown), and `sparc core run --resume` will refit it. A checkpoint written by this release carries the sidecar, so the answer is definite: `matches.code` is `false` after a code change, and with a config `changed_sections` names what differs (`data`, `core`, `s4`, `s5`, `climate`, `s6`, `s7`, `code`) and `fingerprint_match` says whether `--resume` would reuse the checkpoint.

---

## New core APIs and files

All additions are backwards compatible; existing callers of `run_core`, the CLI and the study functions keep working unchanged.

### Progress, atomic I/O and the CLI

| API | What it is |
|---|---|
| `sparc.core.progress` | Structured progress events (JSON lines, schema v1): spans for runs, stages and tasks, throttled ticks, metrics, artifacts, checkpoints, warnings, heartbeats, and cooperative cancellation (`check_cancel`, `Cancelled`, signal handlers). Every call is a no-op unless a sink is configured. Thread and process pools propagate the context (`wrap_context`, `init_worker`); `limit_threads` / `set_threads` cap BLAS, OpenMP and torch threads |
| `sparc.core.runio` | `write_json_atomic` (NaN written as null), `write_text_atomic`, `write_parquet_atomic`, `write_npz_atomic`, `run_lock(run_dir)` (a per-run file lock) and `update_manifest(run_dir, sections, source=…)`, which merges post-run sections under the lock and records them in `post_run[]` |
| CLI | Every `sparc core` subcommand takes `--progress PATH` and `--job-id`, or the environment variables `SPARC_PROGRESS`, `SPARC_PROGRESS_LEVEL`, `SPARC_JOB_ID`, `SPARC_CANCEL_FILE`. A cancelled command exits with status **130**. `sparc studio …` and `sparc core studio …` hand their arguments to Studio. `sparc core reproduce` on a folder without `manifest.json` now exits 2 with a message (it used to raise), and a failed reproduction now propagates its exit status |

All pipeline writes (JSON, parquet, Markdown, NPZ, checkpoints, the CMIP6/GHCN/ISD caches, planner GeoTIFFs and the GeoPackage) go through atomic temporary-file-then-rename writes, so a cancelled or killed run never leaves half a file.

### Planning and run sidecars

| API or file | What it is |
|---|---|
| `pipeline.plan_stages(cfg, stages, fast, coarse, cv_curve, done, n_points, extent, dx, *, block_m, write)` | The run's plan without running it: one node per stage (`S0, S1, S2_S3, baselines, cv_curve, S4, S5, climate, S6, S7, finish`) with its state (`will_run`, `skipped` + reason, `cached`) and planned work units. It shares its gating predicates with `run_core`, so the plan and the run cannot drift. Also exported lazily as `sparc.core.plan_stages` |
| `pipeline.STAGE_NODES`, `CHECKPOINT_KEY`, `apply_mode_overrides` | The stage list, the checkpoint key per stage (S1 and S2_S3 → `S3`), and the fast/coarse/CV-curve config overrides as a function |
| `run_core(…, run_dir=None, run_meta=None)` | An explicit run folder (Studio gives every run its own) and caller metadata stored in `manifest.run_meta` and the `run.start` event |
| `run_state.json` | Written at the start, at every stage boundary and at the end: status, pid, stage, done set, fingerprint, events path, and light metadata (cells, grid shape, cell size, CV design) as soon as it is known — so a run can be read while it is still going |
| `checkpoint.json` | Written with every checkpoint: fingerprint, per-section hashes, done set, size, time, code digest. `pipeline.checkpoint_status(run_dir, cfg=None, fast=None)` reads it (never the pickle); `pipeline.fingerprint_sections(cfg, fast, …)` gives the section hashes of a config, which powers Studio's config-edit impact preview |
| `manifest.json` | `schema_version: 2`, `stages_run`, `timings_detail` (per fold and model) and `run_meta`; post-run sections (planner, uncertainty, studies, emulator, baselines) are preserved when a run is resumed |

### Results, sessions and planning

| API or file | What it is |
|---|---|
| `scenario_detail.npz` | Per configured scenario: the per-fold deltas (K × n), their sd and the extrapolation score, written by S5 (`output.scenario_detail`, default on). Its fold mean equals `scenario_deltas.parquet`. Studio uses it for paired comparisons without re-running anything |
| `sparc.core.session.open_run(run_dir, cfg=None, *, threads=2, base_fold=None, drop=…, progress_cb=None, studio_dir=None)` and `config_for_run(run_dir, fallback=None)` | A run's data, folds, fitted ensemble, mediators and scenario engine in memory, from its launch snapshot or manifest; unpickling reports byte progress. Also exported lazily as `sparc.core.open_run` |
| `ScenarioEngine(…, base_fold=None)` | Skips the K-fold baseline pass when the cached baseline predictions are given; `FittedEnsemble.fold_predictions` ticks one `engine_pass` per fold and checks for cancel |
| `optimize.planned_allocation(vr, budget, cost_per_unit=1.0, equity_scores=None, equity_focus=0.0, multipliers=…, cap=None, benefit_weight=None, min_dose=0.0)` | The open-loop budget allocation on its own (`optimise_allocation` is now this plus the closed loop): dose per cell, planned benefit, cells treated (cells, not segments), cost, Gini, the cost dropped by `min_dose`, and the Pareto points. `pipeline.optimizer_layers` (cap, weight and objective layers) is public |
| `sparc.core.catalog` | The catalogue of every output file a run can contain (`OUTPUTS`, the `IGNORED` globs, `scenario_slug`), with the stage that produces it and where it is shown |
| `sparc.core.results_page` | The standalone results page builder, moved from `scripts/results_page/` into the package: `build_results_page(run_dir, cfg, out, placebo_path=None)` and `python -m sparc.core.results_page`. `scripts/results_page/build_page.py` and `check_page.py` keep working |
| `sparc.core.synthetic.write_demo_project(out_dir, n=96, seed=0)` | The synthetic demo city as a ready project (data with a CRS, truth, DEMO layers and climate table, config), deterministic in `(n, seed)` |
| `causal_cells.parquet`, `causal.json treatments[t].model_effects` | Per-cell CATE and model slopes, and the model's own-cell dose curve, for the causal maps and the model-versus-causal overlay |
| `input_frame.parquet` | Runs fed an in-memory table (placebo children) save it, so they can be read back like any other run |
| Study parameters | `run_placebo_suite(…, children_dir, resume, run_meta)`, `run_multiverse(…, extra_variants, run_meta)`, `reproduce(…, config_dir, out_dir, run_meta)`: explicit child-run folders and links |

A quick look at the planning API on the demo city:

```bash
# runnable
python - <<'EOF'
import tempfile
from pathlib import Path

from sparc.core.config import load_core_config
from sparc.core.pipeline import plan_stages
from sparc.core.synthetic import write_demo_project

demo = write_demo_project(Path(tempfile.mkdtemp()) / "demo", n=40, seed=0)
cfg = load_core_config(demo["config_path"])
for node in plan_stages(cfg, fast=True):
    units = ", ".join(f"{unit} × {n}" for unit, n in node["units"].items())
    print(f'{node["id"]:<10} {node["state"]:<9} {node["reason"] or "":<45} {units[:70]}')
EOF
```

And the results page builder in its new home, on the committed fixture run (recorded on that same demo city):

```bash
# runnable
OUT="${TMPDIR:-/tmp}/sparc-results-page"
python - "$OUT" <<'EOF'
import sys
from sparc.core.synthetic import write_demo_project
write_demo_project(sys.argv[1] + "/demo", n=40, seed=0)
EOF
python -m sparc.core.results_page tests/studio/fixtures/synth_run "$OUT/demo/config.yml" --out "$OUT/results.html"
python scripts/results_page/build_page.py tests/studio/fixtures/synth_run "$OUT/demo/config.yml" --out "$OUT/results_shim.html"
```

---

## Fixes

- **Equity alignment (S7).** `optimize.equity_column` was read from the raw CSV and cut to the first *n* rows (`[:data.n]`), so equity scores were attached to the wrong cells whenever S0 dropped rows (non-finite values), subsampled, or averaged the data onto coarse cells. The column now travels through S0 with the data as an auxiliary column (`CoreData.aux`): it is filtered with the same mask, averaged onto coarse cells like a predictor, and joined **by id**. It is not a predictor, so the models are unchanged; it is fingerprinted with the S7 section. Missing scores are replaced by the mean score. **Allocations from earlier runs that used `equity_column` on data with dropped rows or in coarse mode should be recomputed** (re-run S7, or relaunch).
- **Budget Pareto points** in `optimize.json` are now `{budget, total_benefit, n_cells, n_segments, gini}`. The old `n_treated` counted segments, not cells. The results page reads only `budget` and `total_benefit`.
- **Skill-versus-distance curve.** When every CV partition is skipped, the stage now reports `stage.skip{no_partitions}` with `cv.partition_skipped` warnings, instead of writing a `cv_distance.json` that held only the main-block row.
- **Reproducible GWRF predictions.** Random-forest `n_jobs` follows `OMP_NUM_THREADS` at fit and predict time, so with one thread predictions are bit-reproducible (parallel tree sums differed at the 1e-16 level). Fitted trees are unchanged.
- **Multiverse rank agreement** no longer divides by zero on a variant whose marginal-benefit map is all NaN (too few doses for the saturation fit); it returns NaN and the summary medians skip it.
- **Results page** (now `sparc.core.results_page`): folds are rebuilt with `baselines.load_run` instead of unpickling the checkpoint; causal, CV-distance and budget sections fall back to their stage files; more than eight CV folds are supported (one mask per fold); the page has a proper HTML shell (doctype, charset, viewport, language); the Pareto caption is computed from the numbers; every hot-days case is listed (including `hd_95_ssp245_mid`); latitude and longitude labels know the hemisphere, and reprojected runs (`data.reproject_to`) get correct corners; the layer list comes from the config's roles and levers instead of Providence's column names.
- **Instrumented warnings.** `influence.py` now reports `influence.few_cells` and `influence.constant_predictor`, and network retries in the CMIP6 and forcing downloads report `network.retry` (they used to be silent for up to about 30 s).

---

## SPARC Studio 1.0 at a glance

- **Projects and setup**: a synthetic demo city, the Providence example (optionally with its recorded runs and studies), import of existing configs and run folders, a non-linear setup wizard with an inline data check, open-data inputs as tracked downloads with "Link into config", and a YAML editor with validation, history and a config-edit impact preview.
- **Runs and tracking**: Launch with per-machine time, memory and disk estimates and a stage plan; Mission Control with a stage rail, live stage panels, a span timeline and logs; real Cancel, Force stop and Resume from the launch snapshot; reattach after a server restart and "out of memory" labelling; a Pipeline Status Board; CLI runs followed live from watch roots.
- **Outputs**: nineteen run tabs, a canvas map explorer with analysis tools (region statistics, breakdowns, relationships, correlograms), mid-run views, stale and older-code badges, downloads and conversions.
- **Scenario Lab**: declarative scenarios with portable selections, a server-side emulator preview, exact results with likely ranges and plain-language wording, a scenario library with revisions, paired comparisons, sweeps, budget plans with closed-loop verification and a field kit, climate × adaptation, and decision, plan and compare packs.
- **Studies**: baselines, planner pack, emulator, uncertainty report, methods and model card, placebo tests, simulation check, multiverse, reproduction and the effect benchmark, each tracked, with child runs, attach/detach and an automatically rebuilt uncertainty report.
- **Exports**: run bundles, GIS packs, the standalone results page, a project report builder (HTML with a print stylesheet, or Markdown), and a findings notebook.

The design is in [`SPEC.md`](SPEC.md), the HTTP contract in [`api.md`](api.md); both end with the list of as-built deviations.

## Known issues

- The ETA shown when a run is resumed does not discount the stages that will come from the checkpoint, so it can overstate a resume until those stages finish.
- Brush strokes saved in an earlier session reopen as per-cell edits that can be removed but not repainted.
- The engine host's Windows named-pipe transport and, generally, Windows and macOS process handling are implemented but not tested in CI.
