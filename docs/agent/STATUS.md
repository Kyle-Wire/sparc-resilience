# SPARC steward status

Updated by the `sparc-steward` agent at the end of every run. The owner can
edit any line; the agent treats owner edits as instructions.

**Current phase:** 1 (Providence on real traverses)
**Phase gate:** not yet met (reader handles the real files: yes; lab and estimate: in progress this run)
**Next step:** Phase 1, step 3: `lab --workers 4` (running).
**Blockers:** none
**Proposals awaiting owner approval:** none

## Phase 1 progress

- Step 1 (download): done 2026-10-23. Per-file OSF downloads redirect to an Oracle Cloud storage host the
  sandbox cannot resolve; the Waterbutler folder zip (`https://files.osf.io/v1/resources/wu9v7/providers/osfstorage/?zip=`,
  129 MB) streams through `files.osf.io` and works. The inner `All Data_Heat Watch RI_100620.zip`
  (38,297,340 B) matches OSF's md5 `4a0d015ac43149a044586d97a9eccfd6`.
- Step 2 (inspect): done 2026-10-23. The reader handles the real files without changes: 4 shapefiles read
  (`af_trav`, `am_trav`, `mi_trav`, `pm_trav`), 24 raster files skipped with reasons, 62,084 readings; 4 runs
  at 06:00–07:00, 15:00–16:00, 19:00–19:59 (07-29) and 00:00–00:59 (07-30) local, 6 sensors each
  (CAPA1061/1079/1080/1082/1083/1085), 14,999–16,329 readings per run. Local clock (`dttm_lc`) used, so no
  `--utc`. 44% of readings land on the project grid; only 3 of the 6 vehicles drive inside the study area.

## Run log

Newest first. One entry per run: date, what was done, commit(s), result,
next step.

- 2026-10-23: roadmap and agent config added on branch `sparc-steward` (set up by hand, no agent run yet).
