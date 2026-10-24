# SPARC steward status

Updated by the `sparc-steward` agent at the end of every run. The owner can
edit any line; the agent treats owner edits as instructions.

**Owner decisions (2026-10-24):**
1. **Phase 2 gate accepted** with the Albuquerque caveat. Phase 3 may start.
2. **Pool-entry rule accepted:** a city enters the Phase 3 pool only if street_100 is Trustworthy on its own
   real routes (`lab --routes`). Cities that fail are reported, not pooled.
3. **Real-route labs always use 36 replicates per world.** This is fixed in advance for every city and run; do
   not change the replicate count or the verdict rules after seeing a result.
4. **Before starting Phase 3, re-run Albuquerque's real-route labs (both runs) at 36 replicates**, so both
   cities face the same test. Update `docs/results/albuquerque_nm_canopy.md` and the pool status with the
   result, whatever it is.
5. **Proposals:** write them up and wait for approval in this file before building them. A proposal counts
   as approved only when the owner writes so here.
6. **Sandbox install:** add `rasterio` and `mgrs` to `requirements-core.txt` (with the versions the sandbox
   used), since `setup_core_city.py` needs them.

**Current phase:** 3 (many cities and the pooled effect model). Phase 2 gate accepted by the owner on
2026-10-24 (decisions above).
**Phase gate (Phase 1):** **met** on 2026-10-23.
- **Reader:** it handles the real files with no reader change: 4 shapefiles read, 24 rasters skipped with
  reasons, 62,084 readings, 4 runs × 6 sensors on the local clock.
- **Lab:** the verdicts reproduce the documented table value for value. street_100 is **Trustworthy**
  (null −0.015 °F, 8%), street_300 is Direction only, and both floors are validated.
- **Result reported:** the block-scale null, the evening's measured cooling, the 300 m sign, the 0.05 °F floor
  and the underpowered kilometre test. See `docs/results/canopy_identification.md`, section "Real traverses
  (Providence, 2020-07-29)".
- **Not done:** the optional results-page rebuild. It needs the 3 h core run directory and is left to the
  owner's machine.

**Phase 2 gate: met on 2026-10-23.** Evidence, with detail in `docs/results/albuquerque_nm_canopy.md`:
1. **A second CAPA file layout reads cleanly.** Albuquerque (`np4q9`, 2021) needed three reader fixes, each
   with a regression test:
   - the time is in a shapefile field cut to `datetim`;
   - sensor files are named by a bare number (`1023B_…`);
   - 1,423 readings carry CAPA's −9999 no-data value.

   After the fixes `inspect` shows 70,412 readings in 2 runs (06–07 and 15–16 MDT, 11 and 17 sensors), and 87%
   land on the grid.
2. **The lab passes on that city's layout.** On idealised campaigns over Albuquerque's 439,342-cell layout,
   street_100 is **Trustworthy** (null −0.014 °F, 0%; bias ≤ 0.024 °F in all five effect worlds), and both
   floors are validated.
3. **The setup script works unattended.** `python scripts/setup_core_city.py albuquerque_nm --force` ran every
   step with no input in 3 min 18 s (`setup.json`: all six steps ok) and wrote a grid identical to the
   previous run's. Two pipeline fixes were needed first:
   - the forcing falls back to ERA5 when the station reports only variable wind;
   - GDAL 3.10 needs `GDAL_CURL_CA_BUNDLE` behind the sandbox's TLS egress.
- **Caveat for the owner.** On Albuquerque's *real* routes and noise (`lab --routes`, 12 replicates), street_100
  is **Not trustworthy** in both runs. The real estimate is a wide null: −0.17 °F per +10 pp within 100 m
  (−0.45 to +0.12). The city has almost no street canopy (2.4% at the visited cells), and its street air is
  noisy. This does not block the gate as written, but it means Albuquerque's numbers should not be used.
  Phase 3 pooling needs the real-route check per city. Suggestion: make "street_100 trustworthy on the city's
  real routes" the condition for a city to enter the pool.

**Phase 2 items**

| Item | Status |
|---|---|
| `identify` default output: `<output.dir>/identify`, or `<output.dir>/<full name>/identify` on the shared default folder | done `ff9931a` |
| `_TRUTH` cache keyed by layout | done `ff9931a` |
| `se` kept in each `runs[]` entry | done `ff9931a` |
| `fetch_capa_raw(node_id, dest, folder_hint)`, replacing `download_capa_traverses` | done `ff9931a`; live on `wu9v7` and `np4q9` |
| `capa_catalog`: Providence `traverse_node: "wu9v7"` | done `ff9931a` |
| P1 real-route lab (approved by the owner in session) | done `4903bd5` |
| Reader: the Albuquerque layout | done `5285305`, `26a206a` |
| `scripts/setup_core_city.py <slug>` | done `bc3cd34` |
| City chosen | Albuquerque (`np4q9`). It is the first in the roadmap's list whose OSF listing shows traverses, single-tile and arid. Atlanta `r74h2`, Raleigh `rwjnx` and Houston `yqh5u` also hold traverses. |
| Albuquerque: `inspect` → `lab` → `lab --routes` → `kmlab` → `estimate`; `docs/results/albuquerque_nm_canopy.md` | done (this commit) |

**Owner decisions: done.**

| Decision | What was done |
|---|---|
| 3 and 4 | Albuquerque's real-route labs re-run at 36 replicates (`fd4618f`): afternoon **Direction only**, morning **Partial**. Not pooled. |
| 6 | `rasterio>=1.4.4` and `mgrs>=1.6.0` added to `requirements-core.txt` (`e767165`). |
| 5 | Proposal P2 is written and was not built. A trial version was reverted before commit when decision 5 was read. |

**Phase 3 progress**
- **Batch driver:** `scripts/run_canopy_cities.py` (`116569f`). It runs setup → lab → estimate → 36-replicate
  real-route labs → kmlab → estimate. It writes one status row per city and per run and never aborts.
  Tests: `tests/core/test_run_canopy_cities.py`.
- **Third city:** Burlington, VT (`tezb9`), at the owner's request for a northern, leafy city.
  - `setup_core_city.py` built it unattended on the first try in 1 min 55 s.
  - **Results** (`docs/results/burlington_vt_canopy.md`): a tight block-scale null, −0.012 °F per +10 pp
    within 100 m (−0.047 to +0.022).
  - **Idealised lab:** Direction only (it overstates).
  - **Real-route labs** (36 reps): Conservative, Not trustworthy, Conservative (they understate `own_only`).
  - **Pooled:** no.
- **Pool status** (`docs/results/canopy_pooled.md`): only Providence's morning, evening and night runs pass the
  pool-entry rule. That is too few for `pool.py`'s meta-analysis.
- **Phase 3 gate** (leave-one-city-out coverage): not started. It needs several pooled cities.

**Next step:** depends on P2.
- **If the owner approves P2:**
  1. Build it.
  2. Re-run the real-route labs for Providence and Burlington, and the idealised labs, at 36 replicates with
     unchanged verdict rules.
  3. Update the pool status whatever the result.
- **Either way:** add more single-tile northern cities through `run_canopy_cities.py`: Seattle `mz79p` (2020
  "All Data" bundle) and Scranton `8cnq4` (2023). Portland `e7js9` returned HTTP 401.
- **Lab cost per city:** 15 min (small, like Burlington) to 2 h (like Albuquerque) on 4 cores.

**Blockers:** none.
**Proposals awaiting owner approval:**
- **P2:** score the lab's street designs against the streets the campaign drives
  (`docs/agent/proposals/P2_lab_truth_at_sampled_cells.md`). Not built. The Phase 2 gate and the pool-entry
  rule were accepted on 2026-10-24.

**For the owner:** done 2026-10-24: the Oracle storage host is now in
`ops/managed-agents/sparc-steward/environment.yaml`, matching the live environment. `rasterio` and `mgrs`
go into `requirements-core.txt` per owner decision 6.

## Findings the roadmap should know about

1. **OSF downloads.** Run 1 found that per-file downloads redirect to an Oracle Cloud storage host the sandbox
   could not reach. The Waterbutler folder zip (`https://files.osf.io/v1/resources/<node>/providers/osfstorage/?zip=`)
   worked. Since the owner's environment change, per-file downloads work, and `fetch_capa_raw` uses them.
2. **The real campaign is not the lab's idealised one.**
   - Only 3 vehicles drive inside the study area, on irregular routes.
   - The afternoon is noisy (SE 0.077 °F against the idealised lab's 0.034 °F).
   - Along-pass variation is autocorrelated: difference variance at about 300 m is 0.06 °F² at night and
     0.82 °F² in the afternoon.
   - The lab's planted worlds are much rougher cell to cell than real air (0.70 °F² at 30 m against
     0.006–0.06 °F²), because `validate.plant` adds an independent per-cell term.
   - The real-route lab (P1) now checks verdicts on the campaign as driven: see the results below. A world
     generator with realistic short-range smoothness is a possible later refinement, not a gate change.
3. **The real winds differ from the defaults in the docs.**
   - KPVD gives morning 240° at 0.8 m/s, afternoon 170° at 7.7, evening 190° at 5.0, night 200° at 2.1.
   - That spans only 70°, so the kilometre wind-shift test is underpowered on this campaign: `kmlab` shows
     0% false alarms and 0% detection.
4. **Output folders.** The roadmap asked for `cfg.output_dir/identify` "using the full name".
   - **Implemented:** `<output.dir>/identify` for a config with its own `output.dir`, so Providence's
     documented path is unchanged. A config left on the shared default `output/core` gets
     `output/core/<full name>/identify`.
   - **Remaining overlap:** `core_providence.yml` and `core_providence_open.yml` share `../output/core/providence`,
     so their identify outputs would overwrite each other. Give the open config its own `output.dir` if both
     are used.

## Phase 1 results (summary; detail in `docs/results/canopy_identification.md`)

| Reading (+10 pp canopy) | Estimate (95% CI) | Lab verdict |
|---|---|---|
| Within 100 m, all runs pooled | −0.03 °F (−0.07 to +0.02) | Trustworthy |
| Within 100 m, afternoon 15–16 | +0.02 °F (−0.13 to +0.17) | Trustworthy (idealised lab); Direction only on its real routes |
| Within 100 m, morning / evening / night | −0.03 (−0.09, +0.04) / **−0.08 (−0.11, −0.04)** / −0.02 (−0.04, +0.01) | Trustworthy on each run's real routes (36 reps) |
| Within 300 m, evening | **−0.17 °F (−0.26 to −0.07)** | Trustworthy on the evening's real routes |
| Within 300 m, all runs pooled | −0.11 °F (−0.17 to −0.04) | Direction only |
| City-wide floor (from 300 m) | cools by at least 0.05 °F | Validated floor |
| Kilometre wind shift (4 runs) | +0.08 °F (−0.01 to +0.17), rotation p = 0.56 | Underpowered under these winds |

## Run log

Newest first. One entry per run: date, what was done, commit(s), result,
next step.

- **2026-10-24 (session 3, after the owner's decisions):** owner decisions 3, 4, 5 and 6 carried out; Phase 3
  started.
  - **Burlington:** the third city, set up and analysed.
  - **Fix:** a calm or variable station wind takes ERA5's wind for the run (`83eb106`).
  - **Lab scoring:** a truth-population mismatch was diagnosed and written up as proposal P2 (not built).
  - **Phase 3:** batch driver, pool status.
  - **Tests:** `pytest tests/core/test_identify.py tests/core -m "not slow"` gave 310 passed, 9 deselected
    (10 min 48 s).
  - **Commits:** `83eb106`, `e767165`, `4cd39cf`, `fd4618f`, `116569f`, COMMIT_S3.

- **2026-10-23 (run 1, session 2, continued):** setup script and Albuquerque.
  - **Setup:** `bc3cd34` adds `scripts/setup_core_city.py` and its tests, the forcing fallback to ERA5 on
    variable station wind, and the GDAL CA bundle.
  - **Reader:** `5285305` and `26a206a` fix the Albuquerque layout: `datetim`, bare-number sensors, −9999
    no-data, and the variable-wind run direction.
  - **Albuquerque runs:** idealised lab (27 min), real-route labs (20 and 14 min), `kmlab` (6 min),
    `estimate`. `setup_core_city.py --force` ran unattended in 3 min 18 s.
  - **Tests:** 307 passed, 9 deselected (8 min 53 s, CPU idle). One earlier run that overlapped a lab failed
    the 15-second timing test `test_progress_pipeline::test_no_info_level_gap_exceeds_15_s` (16.0 s): a CPU
    contention artefact. It passes with the CPU idle.
  - **Commits:** `5285305`, `bc3cd34`, `26a206a`, `88d6d3b`.
- **2026-10-23 (run 1, session 2, after the owner's go-ahead):** Phase 2 code and the P1 real-route lab.
  - **Phase 2 code** (`ff9931a`):
    - the output-dir default;
    - the truth cache keyed per layout;
    - per-run SEs;
    - `fetch_capa_raw`, which replaces `download_capa_traverses` and its unused helpers; their tests are
      removed from `tests/test_fix_era5_capa.py`;
    - the Providence `traverse_node`.

    Each change has a regression test that fails without it.
  - **P1** (`4903bd5`): `lab --routes`. The idealised lab still reproduces: the identical seed stream gives
    values within 1e-9.
  - **Real-route labs:** 12 replicates, then 36, for each run. The results are in the write-up (this commit).
  - **Second city:** OSF listings checked read-only; Albuquerque chosen.
  - **Tests:** `pytest tests/core/test_identify.py tests/core -m "not slow"` gave 297 passed, 9 deselected
    (about 9.2 min). `tests/test_fix_era5_capa.py` gave 21 passed.
  - **Commits:** `ff9931a`, `4903bd5`, `c012045`.
  - **Next:** `scripts/setup_core_city.py`.
- **2026-10-23 (run 1):** Phase 1, steps 1–7, done; gate met.
  - **Steps:** downloaded `wu9v7` via the Waterbutler folder zip; `inspect` (reader unchanged); `lab` (3 min);
    `kmlab` with the real KPVD winds (1.5 min); `estimate --window all` and the afternoon headline (20 s each).
  - **Write-up:** `docs/results/canopy_identification.md` ("Real traverses") and `docs/results/providence_core.md`
    (S5 uncertainty line, caveats).
  - **Code fix:** "underpowered" had no `STATUS_NOTE` in `traverses.py`, so the estimate printed an empty sentence
    ("…wind. ."). The regression test is `test_estimate_markdown_explains_an_underpowered_kilometre_verdict`.
  - **Tests:** `pytest tests/core/test_identify.py tests/core -m "not slow"` gave 291 passed, 9 deselected
    (baseline before the fix: 290 passed, 9 deselected), about 9.5 min.
  - **Commits:** `7693584` (steps 1–2 status), then `ad78f41` (results, fix, test, status).
  - **Next:** Phase 2, the output-dir default.
- 2026-10-23: roadmap and agent config added on branch `sparc-steward` (set up by hand, no agent run yet).
