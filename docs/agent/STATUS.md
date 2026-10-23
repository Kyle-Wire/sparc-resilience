# SPARC steward status

Updated by the `sparc-steward` agent at the end of every run. The owner can
edit any line; the agent treats owner edits as instructions.

**Current phase:** 1 complete; Phase 2 (a second city) is next.
**Phase gate (Phase 1):** **met** on 2026-10-23. Evidence follows.
- **Reader:** it handles the real files with no reader change: 4 shapefiles read, 24 rasters skipped with
  reasons, 62,084 readings, 4 runs × 6 sensors on the local clock.
- **Lab:** the verdicts reproduce the documented table value for value. street_100 is **Trustworthy**
  (null −0.015 °F, 8%), street_300 is Direction only, and both floors are validated.
- **Result reported:** the block-scale null, the 300 m sign, the 0.05 °F floor and the underpowered kilometre
  test. See `docs/results/canopy_identification.md`, section "Real traverses (Providence, 2020-07-29)".
- **Not done:** the optional results-page rebuild. It needs the 3 h core run directory and is left to the
  owner's machine.

**Next step:** Phase 2, first code change: the default output dir in `sparc/core/identify/__main__.py`
should use the full `cfg.name` under `cfg.output_dir` (with a regression test). Then the `_TRUTH` cache keyed
by layout, `se` kept in `runs[]`, and `fetch_capa_raw` (see the OSF note below).
**Blockers:** none.
**Proposals awaiting owner approval:** P1 below (a real-route lab). It is optional; Phase 2 does not wait for it.

## Findings the roadmap should know about

1. **The OSF per-file download is unreachable from the sandbox; the folder zip works.**
   - `osf.io/download/<guid>` and `files.osf.io/v1/.../<file_id>` answer with a 302 to
     `*.compat.objectstorage.us-ashburn-1.oraclecloud.com`. That host is not in `environment.yaml` and does not
     resolve.
   - The Waterbutler folder zip `https://files.osf.io/v1/resources/<node>/providers/osfstorage/?zip=` streams
     through `files.osf.io` and works (129 MB for `wu9v7`, about 5 s).
   - **Proposed change to the Phase 2 `fetch_capa_raw` step:** download through the `?zip=` folder route, or add
     the Oracle host to `environment.yaml`. Otherwise it will fail in the sandbox the same way.
2. **The real campaign is noisier than the lab's idealised one.**
   - The afternoon block-scale SE is 0.077 °F against the lab's 0.034 °F, with similar pair and cluster counts.
   - Only 3 of the 6 vehicles drive inside the brown4 study area, on irregular routes.
   - The lab's verdicts (bias and coverage) were earned on simulated 10-vehicle street-grid campaigns with
     white sensor noise.
3. **The real winds differ from the defaults in the docs.**
   - KPVD gives morning 240° at 0.8 m/s, afternoon 170° at 7.7, evening 190° at 5.0, night 200° at 2.1.
   - That spans only 70°, so the kilometre wind-shift test is underpowered on this campaign: `kmlab` shows
     0% false alarms and 0% detection.

### P1 (proposal, awaiting owner approval): a real-route lab

**What.** Add an option to `lab` that replays the real campaign's sampling over the planted worlds:

- the real cells, passes, vehicles and times from `to_campaign`;
- sensor noise scaled to each run's real residual variance;
- an optional AR(1) along-pass noise term.

The lab would then score street_100/300 bias, coverage and power on the campaign as it was driven, one table
per run.

**Why.** It checks that the "Trustworthy" verdict and the 0.05 °F floor hold for the real sparse, noisy
afternoon sampling, not only for the idealised grid. It also gives the real power of each run.

**Cost.** About 60 lines in `validate.py` and `campaign.py`, plus a test. Lab time is about 3 min per run on
4 cores.

**Where.** It fits as Phase 1 step 3b, or as a Phase 2 item. The same check would then run for every new
city. It is not a gate change.

## Phase 1 results (summary; detail in `docs/results/canopy_identification.md`)

| Reading (+10 pp canopy) | Estimate (95% CI) | Lab verdict |
|---|---|---|
| Within 100 m, all runs pooled | −0.03 °F (−0.07 to +0.02) | Trustworthy |
| Within 100 m, afternoon 15–16 | +0.02 °F (−0.13 to +0.17) | Trustworthy |
| Within 100 m, morning / evening / night | −0.03 (−0.09, +0.04) / **−0.08 (−0.11, −0.04)** / −0.02 (−0.04, +0.01) | Trustworthy (afternoon-validated) |
| Within 300 m, all runs pooled | −0.11 °F (−0.17 to −0.04) | Direction only |
| City-wide floor (from 300 m) | cools by at least 0.05 °F | Validated floor |
| Kilometre wind shift (4 runs) | +0.08 °F (−0.01 to +0.17), rotation p = 0.56 | Underpowered under these winds |

## Run log

Newest first. One entry per run: date, what was done, commit(s), result,
next step.

- **2026-10-23 (run 1):** Phase 1, steps 1–7, done; gate met.
  - **Steps:** downloaded `wu9v7` via the Waterbutler folder zip; `inspect` (reader unchanged); `lab` (3 min);
    `kmlab` with the real KPVD winds (1.5 min); `estimate --window all` and the afternoon headline (20 s each).
  - **Write-up:** `docs/results/canopy_identification.md` ("Real traverses") and `docs/results/providence_core.md`
    (S5 uncertainty line, caveats).
  - **Code fix:** "underpowered" had no `STATUS_NOTE` in `traverses.py`, so the estimate printed an empty sentence
    ("…wind. ."). The regression test is `test_estimate_markdown_explains_an_underpowered_kilometre_verdict`.
  - **Tests:** `pytest tests/core/test_identify.py tests/core -m "not slow"` gave 291 passed, 9 deselected
    (baseline before the fix: 290 passed, 9 deselected), about 9.5 min.
  - **Commits:** `7693584` (steps 1–2 status), then COMMIT_MAIN (results, fix, test, status).
  - **Next:** Phase 2, the output-dir default.
- 2026-10-23: roadmap and agent config added on branch `sparc-steward` (set up by hand, no agent run yet).
