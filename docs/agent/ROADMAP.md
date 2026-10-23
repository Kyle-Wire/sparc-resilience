# SPARC steward roadmap

This is the working plan for the `sparc-steward` Managed Agent (config in
`ops/managed-agents/sparc-steward/`). The agent reads this file at the start of
every run and records progress in `docs/agent/STATUS.md`. Edit this file to
steer it; it is the agent's source of truth for scope and gates.

**Vision.** An ROI planner whose numbers rest on measured effects, used by
climate planners everywhere. Extreme heat (canopy cooling) comes first. Other
fields come only after the heat work is proved (Phase 3 gate), and only through
a written proposal the owner approves (see "Beyond heat" at the end).

---

# Plan: canopy effects from real traverses, Providence → many cities → ROI planner

## Context

The map SPARC trains on (CAPA's interpolated Heat Watch raster) cannot identify canopy's cooling effect. In the identification lab, every design run on the map reports a canopy effect in 50–58% of no-effect worlds. The street-difference design on the raw vehicle traverses passed the lab at block scale (within 100 m), as "direction only" at 300 m, and as a validated one-sided floor for the city-wide effect. It has **never been run on the real traverses** (`docs/results/canopy_identification.md`).

The goal is the original vision: an ROI planner whose canopy numbers rest on measured effects. The path:

1. Providence on real traverses.
2. A second city.
3. Many cities pooled into a cross-city effect model that predicts canopy's effect for cities without traverses, validated leave-one-city-out.
4. That model feeds the scenario and budget engine.
5. JEPA last, retargeted to traverse readings.

"Train multi-city" means pooling the per-city traverse estimates into an effect model, not JEPA. JEPA predicts temperature maps, not intervention effects. It is still trained on the interpolated rasters with ERA5-matched (circular) dates, so it comes last and only after its labels are switched to traverse readings.

Facts established in exploration:
- OSF is reachable: `api.osf.io/v2/nodes/wu9v7` returns 200. The Providence traverses are inside `All Data_Heat Watch RI_100620.zip` (38 MB).
- `identify estimate` needs no finished core run. It needs the config's input table (layout and target), `data.crs`, `physics.roles.canopy`/`impervious` and the forcing JSON.
- The `lab` uses the real target's statistics for realism, so each city needs a real target column. CAPA's `aat_midday` raster is fine for that role.
- The lab takes about 10 min on 4 cores / 15 GB RAM. `output/` is gitignored, so results are summarised in committed docs.
- No cross-city pooling code exists anywhere.

---

## Phase 1: Providence on real traverses (no code changes expected)

1. **Download** the All Data zip to `output/core/providence/identify/raw/`. The reader (`sparc/core/identify/traverses.py:146-207`, `scan_traverses`) unpacks nested zips and shapefile sidecars itself.
2. **Inspect:** `python -m sparc.core.identify inspect -p configs/core_providence.yml --traverses <raw>`. Expect 4 runs (`am_trav`, `af_trav`, `pm_trav`, `mi_trav`), 6 sensors and about 15k readings per run, read on the local clock (`dttm_lc`).
   - If runs land at 10–11, 19–20 and 23–24 h, add `--utc`.
   - Check `share_on_grid`.
3. **Lab:** `lab --workers 4` (12 reps × 6 worlds).
4. **Kilometre lab:** `kmlab --winds <the real per-run winds that estimate prints>`.
5. **Estimate:** `estimate --window all`, with winds fetched from the KPVD station in `configs/forcing/providence_2020-07-29.json`.
6. **If the real files trip the reader** (column names, time zone, sensor ids), make minimal fixes in `traverses.py` and `sparc/data/collect/capa.py:565-682`, each with a regression test in `tests/core/test_identify.py` built from a small synthetic table in the observed layout.
7. **Write up** in `docs/results/canopy_identification.md`, in a new section "Real traverses (Providence, 2020-07-29)": QA; block-scale effect per run with its CI and lab verdict; the 300 m reading; the city-wide floor; the kilometre wind-shift result. Update the uncertainty and canopy lines in `docs/results/providence_core.md`.

**Gate:** the reader handles the real files; the lab verdicts reproduce the documented ones (street_100 trustworthy); a result is reported either way (a null block-scale effect is a finding, not a failure). A results-page rebuild needs the 3 h core run directory, so it is optional and can be done on the owner's machine.

---

## Phase 2: A second city (make the path general)

Code changes:

- **`sparc/core/identify/__main__.py:19-23`.** Default output becomes `cfg.output_dir/identify`, using the full name instead of `cfg.name.split("_")[0]` (today `new_orleans_uhi` maps to "new").
- **`sparc/core/identify/validate.py:76-91`.** Key the `_TRUTH` cache by layout as well as world, so two cities in one process don't share truths.
- **`sparc/core/identify/traverses.py:542`.** Keep `se` in each `runs[]` entry (dropped today); pooling needs per-run SEs.
- **`sparc/data/collect/capa.py`.** Add `fetch_capa_raw(node_id, dest, folder_hint)`: downloads a project's "All Data"/traverse zips and shapefile sets to disk, reusing `_scan_storage` (251-290) and the Waterbutler fetch, and lets `scan_traverses` parse. It replaces the unused, unverified `download_capa_traverses` route, whose name filter rejects `am_trav` and misses "All Data" bundles.
- **`sparc/data/collect/capa_catalog.py:52-55`.** Add `traverse_node: "wu9v7"` (2020) for Providence; the entry currently points at the 2019 `tdsy7` project.
- **New `scripts/setup_core_city.py <city>`.** Builds a core config for a city from its `configs/multicity_pilot.yml` entry and `CAPA_CATALOG`, reusing:
  - `scripts/collect_cities.py:63-90` (`build_fishnet`, 30 m UTM) and `capa._sample_rasters_to_fishnet` (406-503) to write a CSV with `id,x,y,aat_midday` plus placeholder predictors;
  - `sparc core features` (`features_open.py`) for the six `open_*` predictors;
  - `sparc core forcing` (`forcing.py:393-447`) for the campaign afternoon, station from the lookup in `scripts/fetch_training_city_airport_bg.py:60`;
  - `sparc core layers` for the planner;
  - a core YAML from the `configs/core_providence_open.yml` pattern (`crs: <utm_epsg>`, `coord_unit: m`, open roles, forcing path, `output.dir: ../output/core/<slug>`).

**Choosing the city:** single-tile (not one of the 19 multi-tile cities, `features_open.py:183-199`); has its own OSF project with traverse files; contrasting climate. List candidate OSF projects first (read-only), e.g. Albuquerque (arid), Atlanta, Raleigh or Houston, and take the first whose listing shows traverses. Then `inspect` → `lab` → `kmlab` → `estimate` with `--out output/core/<slug>/identify`, and write `docs/results/<slug>_canopy.md`.

**Gate:** a second CAPA file layout reads cleanly, the lab passes on that city's layout, and the setup script works unattended.

---

## Phase 3: Many cities and the pooled effect model

- **Batch driver** `scripts/run_canopy_cities.py`: for each catalog city, setup → fetch → lab → estimate. Failures logged per city (`output/core/pooled/city_status.csv`), never abort the batch. Commit a summary of the status table to `docs/results/canopy_pooled.md`, since `output/` does not persist.
- **Multi-tile support:** mosaic WorldCover and DEM tiles in `sparc/core/opendata.py:152-153` and `features_open.py:183-199`, unlocking the 19 blocked pilot cities.
- **New `sparc/core/identify/pool.py`** with a `pool` subcommand:
  - random-effects meta-analysis (REML, DerSimonian–Laird fallback) of `street_100` per time window: pooled effect, τ², prediction interval;
  - pooled city-wide floor and pooled kilometre wind-shift test (Stouffer's combination of each city's `wind_shift` z or rotation p);
  - meta-regression of the city effect on covariates from each city's own files: background canopy and impervious means; campaign dewpoint and wind (forcing JSON); Köppen class; time window;
  - leave-one-city-out: predict each held-out city's effect from the others, score interval coverage and error against its own traverse estimate, compare against the plain pooled mean.
- **Outputs:** `output/core/pooled/pool.json`, `pool.md`, and `docs/results/canopy_pooled.md`.

**Gate:** leave-one-city-out prediction intervals cover close to nominal. If meta-regression doesn't beat the pooled mean, report the pooled mean with τ.

---

## Phase 4: Into the ROI planner (scoped after Phase 3)

Canopy effects in scenarios, the emulator and the S7 budget come from the pooled model instead of the map. Candidate routes: a prior on the physics shading parameter `s` (`sparc/core/physics.py`), or calibrating the scenario canopy response so the city-wide edit matches the predicted effect and floor. Pick once Phase 3 shows the size and spread of the effect; write a short plan in `docs/agent/` first.

## Phase 5: JEPA (optional, last)

In `scripts/train_multicity_jepa.py`: swap raster labels (`:1080-1090`, `:1957-1971`) for per-cell traverse means from `traverses.to_campaign` (sparse labels + validity mask); take campaign dates from traverse timestamps (removes the ERA5 circularity flagged in `configs/multicity_pilot.yml:6-21`); score leave-one-city-out against held-out traverse readings. Keep only if it beats the core stack at traverse points. JEPA stays behind flags (`PI_JEPA_AGENT_SPEC.md` §0).

---

## Verification (every phase)

- `pytest tests/core/test_identify.py tests/core -m "not slow"` after every change.
- New tests: per-run SE present; output-dir default uses the full name; truth cache keyed per layout; `fetch_capa_raw` against a mocked OSF listing with `All Data_*.zip` and `am_trav.*` sidecars; `pool.py` on synthetic cities with known effect and τ (recovers both; LOCO coverage near nominal).
- End to end per city: `identify simulate --world additive` → `estimate --window all` recovers the planted effect (the round trip at `test_identify.py:212`), then the real-traverse run.
- Phase 1: lab verdicts match the documented table; `inspect` QA matches the documented run and sensor counts.
- Raw data and `output/` stay out of git.

---

## Beyond heat (only after the Phase 3 gate passes)

Write `docs/agent/proposals/<field>.md` for the next field (candidates in
`docs/roadmap/SPARC_Domain_Atlas.md`, `docs/roadmap/AIR_QUALITY_PLAN.md`): what
would be measured, which public ground-truth data exists, which identification
test would prove the effect, and the cost. Mark it `Status: awaiting owner
approval` in `STATUS.md` and do not build it until the owner changes that line.
