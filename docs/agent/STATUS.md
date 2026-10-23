# SPARC steward status

Updated by the `sparc-steward` agent at the end of every run. The owner can
edit any line; the agent treats owner edits as instructions.

**Current phase:** 2 (a second city). Phase 2 code is done except `scripts/setup_core_city.py`.
**Phase gate (Phase 1):** **met** on 2026-10-23. Evidence follows.
- **Reader:** it handles the real files with no reader change: 4 shapefiles read, 24 rasters skipped with
  reasons, 62,084 readings, 4 runs × 6 sensors on the local clock.
- **Lab:** the verdicts reproduce the documented table value for value. street_100 is **Trustworthy**
  (null −0.015 °F, 8%), street_300 is Direction only, and both floors are validated.
- **Result reported:** the block-scale null, the evening's measured cooling, the 300 m sign, the 0.05 °F floor
  and the underpowered kilometre test. See `docs/results/canopy_identification.md`, section "Real traverses
  (Providence, 2020-07-29)".
- **Not done:** the optional results-page rebuild. It needs the 3 h core run directory and is left to the
  owner's machine.

**Phase 2 progress (gate not yet met: needs a second city read, lab-passed, and the setup script run unattended)**

| Phase 2 item | Status |
|---|---|
| `identify` default output: `<output.dir>/identify`, or `<output.dir>/<full name>/identify` on the shared default folder | done `ff9931a` |
| `_TRUTH` cache keyed by layout | done `ff9931a` |
| `se` kept in each `runs[]` entry | done `ff9931a` |
| `fetch_capa_raw(node_id, dest, folder_hint)`, replacing `download_capa_traverses` | done `ff9931a`; live on `wu9v7`: 2.6 s, md5 matches, `inspect` reads 62,084 readings |
| `capa_catalog`: Providence `traverse_node: "wu9v7"` | done `ff9931a` |
| `scripts/setup_core_city.py <city>` | **next** |
| Choose the city | **Albuquerque** (`np4q9`; `traverses_chw_albuquerque_100421.zip`). It is the first in the roadmap's list whose listing shows traverses, it is single-tile (WorldCover N33W108, DEM N35 W107), and it is arid. The other candidates also hold traverses: Atlanta `r74h2`, Raleigh `rwjnx`, Houston `yqh5u` (as an "All Data" bundle). |
| P1 real-route lab (approved by the owner in session, 2026-10-23) | done `4903bd5`; results in the write-up |

**Next step:** Phase 2, write `scripts/setup_core_city.py`.
- **Needs:** `rasterio` (not in `requirements-core.txt`; `pip install rasterio`) and the open-data hosts.
- **Then for Albuquerque:** `fetch_capa_raw('np4q9', ...)` → setup → `inspect` → `lab` → `lab --routes` per run →
  `kmlab` with the real winds → `estimate --window all`. Write `docs/results/albuquerque_nm_canopy.md`.

**Blockers:** none.
**Proposals awaiting owner approval:** none.
**For the owner:** per-file OSF downloads now work in the sandbox, after your environment change. But
`ops/managed-agents/sparc-steward/environment.yaml` in the repo still lists only `osf.io`, `api.osf.io`,
`files.osf.io`. If you added the Oracle storage host
(`idvyorynw6xy.compat.objectstorage.us-ashburn-1.oraclecloud.com`) outside the repo, add it to the file too,
so a later `ant apply` does not remove it.

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
  - **Commits:** `ff9931a`, `4903bd5`, COMMIT_DOCS.
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
