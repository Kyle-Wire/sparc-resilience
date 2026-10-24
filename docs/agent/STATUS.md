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
5. *(Replaced by decision 10.)* ~~Proposals wait for approval in this file before being built.~~
6. **Sandbox install:** add `rasterio` and `mgrs` to `requirements-core.txt` (with the versions the sandbox
   used), since `setup_core_city.py` needs them.

**Owner decisions (2026-10-24, second set):**
7. **P2 approved, as written.** Re-run every city's labs under the new scoring (idealised and real-route, 36
   replicates, unchanged verdict rules), Providence included, and report verdicts before and after in
   `docs/results/canopy_pooled.md` whatever they are. State in the pooled docs that the pooled effect is the
   effect at the streets campaigns drive.
8. **Investigate the rest of Burlington's idealised-lab gap** (the part P2 does not explain; suspects: the
   canopy knots set on Providence, the per-cell noise in `validate.plant`). Fix it if the evidence supports a
   fix, under decision 10's rules; do not change verdict rules.
9. **P3 not approved for now.** Skip cities without timestamps or sensor ids; prioritise cities that have
   them (Atlanta, Raleigh, Houston, then others). You may revisit P3 under decision 10 if many cities turn
   out to need it.
10. **You decide; no owner approval needed.** This branch is yours, so make the wise call yourself and keep
    going instead of waiting. That covers proposals, method changes, the order of work, Phase 4 and, once
    the Phase 3 gate is met, fields beyond heat. The owner reviews the branch and merges. Rules that still
    hold:
    - **Write the decision down before acting:** a short entry under "Agent decisions" below with what,
      why, and what result would show it was wrong. Rules that judge results (verdict rules, replicate
      counts, pool-entry, gates) may change only this way, before the re-run, and every affected city is
      re-run and reported before and after, whatever the outcome.
    - **Never weaken a test, gate or verdict rule to pass a result,** and report nulls and failures as
      findings.
    - **Stay on `sparc-steward`;** never push to `main` or `pi-jepa-dev`, never force-push.
    - **Ask the owner (a line under "Questions for the owner") only** for things outside the repo: money,
      accounts, private data, publishing, or anything that can't be undone.
11. **Budget:** runs are scheduled daily until the month's $100 is spent. Spend it on the highest-value next
    step; there's no need to stop early to save money.

**Owner decisions (2026-10-24, third set):**
12. **Leave-one-city-out sensitivity for every pooled result.** Report each pooled estimate (both pools, every
    window) with each city left out in turn, next to the full estimate. Start with the evening street_100
    pool: Burlington's idealised no-effect world comes out at −0.066 °F, the same size as the pooled effect,
    so show whether the effect holds on Providence and Seattle alone. Keep this table current as cities are
    added.
13. **As many cities as possible.** Growing the pool is the top priority. That includes:
    - every catalog city with timestamped traverses through `run_canopy_cities.py`;
    - the roadmap's multi-tile support (mosaicking WorldCover and DEM tiles), which unlocks the 19 blocked
      pilot cities;
    - CAPA/Heat Watch campaigns not yet in the catalog (other years, other cities) where their OSF projects
      are public;
    - making the per-city lab cheaper where that does not change what it measures (decide under decision 10,
      with a before/after check on a city already done).

    Ask the owner only if a city needs a new data host.
14. **Beyond canopy: every physical variable, then new ones.** The ROI planner compares interventions, so it
    needs measured effects for each lever, not only trees. This is a Phase 3 track that runs alongside the
    city work and uses the same campaigns. The roadmap section "Phase 3b" sets out the method and its rules.

## Agent decisions

Newest first. One entry per decision made under owner decision 10: date, decision, why, what would show it
was wrong.

- **2026-10-24, A8 (decision 14, Phase 3b): the variable list and primary analyses, fixed before any real-data
  run.**
  - **Levers, in this order:**
    1. **impervious**: −10 pp within 100 m and within 300 m of a street point;
    2. **albedo**: +0.05 at the cell and within 100 m.

    `water_distance` and `elevation` stay controls: they are not levers a planner can move. `ndvi` stays out
    as a lever, because it is a mediator of canopy and impervious.
  - **Joint design.** Each lever is read from the same street-difference fit as canopy, which already
    differences canopy rings, impervious rings, albedo, elevation and water distance. Each estimate is
    therefore mutually adjusted.
  - **Lab worlds.** The planted worlds will get a second axis: impervious (and later albedo) with no effect,
    a local effect, and a correlated confounder, each crossed with the canopy worlds. Correlations are kept
    as in each city's real layout.
  - **Pool entry.** A lever's reading is reported as an effect only if its design is Trustworthy on the
    city's real routes at 36 replicates.
  - **Primary set:** impervious street_100 and albedo street_100, each in the afternoon and evening windows,
    pooled across cities. That is 4 tests, adjusted with Holm. Everything else run (300 m, other windows,
    single cities) is reported as secondary and exploratory.
  - **What would show this wrong:** a lever whose lab verdict cannot be Trustworthy on any city's routes. It
    is then reported as not identifiable from street differences, with no fallback design.
  - **Not built yet:** the impervious and albedo worlds and estimands. They come first in the next run's
    Phase 3b work.

- **2026-10-24, A7 outcome: P3 fails its check and is not used. The untimed cities stay out.**
  - **The check.** Providence's and Atlanta's real layers were stripped to `t_f`, `rh`, `hi_f` and positions
    and rebuilt in row order. The per-run street_100 and street_300 estimates differ from the timed ones by
    −1.52 to +0.69 SE. The criterion set in advance was 0.25 SE.

    | Run | street_100 difference (in SE) | street_300 difference (in SE) |
    |---|---|---|
    | Providence afternoon | −1.01 | −1.52 |
    | Atlanta afternoon | +0.69 | +0.62 |
    | the four other runs | −0.45 to +0.36 | −0.50 to +0.09 |
  - **Why it fails.** The rows lose the time gaps that split passes. Providence's runs fall from 19–29
    passes to 3–4, Atlanta's from 85–142 to 27–45. Most gaps over 30 s show no position jump (only 4% of
    Providence's and 27% of Atlanta's do), so row order cannot recover them.
  - **What was kept.** The reader change was reverted before commit. The 19 untimed cities, Boston's 2019
    `time_n` layer among them, remain out of the pool.
  - **What could revive it:** the timestamps themselves. CAPA's raw per-sensor CSVs, if published elsewhere,
    or a request to CAPA. That is a question for the owner (see below).
- **2026-10-24, A7 (decision 9 allows revisiting P3): build P3, reading untimed traverse layers in row order,
  but use it only after it passes a check on cities whose layers do carry time.**
  - **Why now.** A survey of every pilot city's OSF files found 19 of 43 untimed: 2022–2023 layers with only
    `t_f`/`temp_f`, `rh`, `hi_f` and positions (Philadelphia, Chicago, Milwaukee, Brooklyn, Charlotte,
    Jacksonville, Columbia SC, Oklahoma City, Boulder, Las Vegas, Brockton, Scranton, Kansas City KS, Omaha,
    Columbia MO, Iowa City, Cedar Rapids, San Francisco). Boston 2019 has a `time_n` field, HHMMSS in UTC
    with no date. Growing the pool is the top priority (decision 13).
  - **Evidence for the assumption.** In every timed layer checked (Providence 2020, Burlington 2020, Atlanta
    2021, Raleigh 2021), rows come in one contiguous block per sensor, in time order, mostly at 1 s steps
    (median step 1 s; 88–91% exactly 1 s, except Burlington's duplicated timestamps). The untimed layers look
    like the same export with the time and sensor columns dropped. Consecutive rows are about 12 m apart
    (Scranton).
  - **Method.**
    - A file with no usable time gets time = row index at 1 s, its run from the file's window
      (`am`/`af`/`pm`), and one vehicle per file.
    - Passes break at jumps over 200 m. Sensor blocks begin elsewhere in the city, so a new sensor starts a
      new pass.
    - The run is flagged "time from row order".
    - No station wind is fetched, because the date is unknown.
  - **Check, before any untimed city is run.**
    - Strip the time and sensor columns from Providence's and Atlanta's real layers.
    - Rebuild them with P3 and run `estimate` on both versions.
    - P3 is adopted only if every run's street_100 and street_300 estimates agree with the timed ones to
      within 0.25 of their SE.
    - It is also adopted only if one run's real-route lab (36 replicates) gives the same street_100 verdict
      both ways.
  - **What would show it wrong:** a larger disagreement on either city. P3 is then not used, and the
    untimed cities stay out.

- **2026-10-24, A6 outcome: confirmed.** Burlington's evening real-route lab (36 replicates) was re-run with
  the traverse designs only. All 314 numbers match the full lab to within 4e-16, and every verdict is the
  same. A replicate's cost drops from about 21 s to about 3 s (Atlanta's evening routes, profiled).
- **2026-10-24, A6 (decision 13: a cheaper lab): skip the forest-made map in campaigns whose designs do not
  use it.** The driver's real-route labs request only the traverse designs.
  - **What.** The idealised lab, which reports the map designs, is unchanged.
  - **Why.** On Atlanta's evening routes a replicate takes about 21 s. About 18 s of that is the
    random-forest map product and the two map designs, which the real-route labs do not need: the
    pool-entry rule reads street_100 only. The traverse designs take 1–2 s.
  - **It cannot change what the lab measures.** The product draws its random numbers after the campaign's
    samples, so the planted field, the samples and every traverse design's estimate are the same.
  - **Check.** Burlington's evening real-route lab is re-run both ways and compared value for value.
  - **What would show it wrong:** any difference in a traverse design's numbers or verdict.

- **2026-10-24, A5: the Phase 3 gate is judged only in a time window with at least 8 pooled cities.** Until
  then, leave-one-city-out coverage is reported as uninformative.
  - **Why.** The roadmap's gate is "leave-one-city-out prediction intervals cover close to nominal". With
    k cities, each held-out prediction uses k − 1 cities and a t distribution with k − 3 degrees of freedom.
    - With 4 cities that is t(1). The 95% intervals are ±0.9 °F, so they cover anything. "100% of 4" is no
      evidence.
    - With 8 cities (t(5)) a miss rate far from 5% starts to be visible.
  - **This tightens the gate, it does not weaken it.**
  - **What would show it wrong:** if the owner reads the roadmap's gate as already met by any coverage
    figure, this delays Phase 4. In that case the owner can say so here.

- **2026-10-24, A4: add a second pool, for street_300 (canopy within 300 m), under the same entry rule. It is
  applied to its own verdict and sits beside the street_100 pool, which it does not replace.**
  - **What.** A run's street_300 reading enters the street_300 pool only if street_300 is Trustworthy on
    that run's own real routes at 36 replicates. The rules are the same and so is every city.
  - **Why.**
    - In the leafy cities the block-scale design tends to give the 100–300 m ring's cooling to the 0–100 m
      ring. The two rings' canopy moves together, and the idealised labs overstate `additive` and
      `confounded` in Burlington, Seattle and Atlanta.
    - street_300 sums both rings, so it is immune to that split. It is also the scale at which planting
      programmes act.
  - **Caution.** I saw street_300's real-route verdicts before writing this. To guard against having chosen
    it for the answer, both pools are reported for every city, and street_100 remains the primary pool and
    the one the Phase 3 gate is judged on.
  - **What would show it wrong:**
    - street_300's leave-one-city-out coverage falls clearly below nominal once 4 or more cities are pooled;
      or
    - its pooled effect is driven by a city whose street_300 idealised lab is Not trustworthy.

    Either is reported if it happens.

- **2026-10-24, A3 outcome: candidate rejected; the knots stay at (15, 40).** Screen results: 12 replicates,
  P2 truth, all six worlds.

  | City | street_100, knots (15, 40) → (25, 50, 75) | street_300 |
  |---|---|---|
  | Burlington (development) | Direction only → **Trustworthy**; floors become valid | Not trustworthy → Conservative |
  | Providence (held out) | Trustworthy → Trustworthy (`own_only` bias +0.024 → +0.032 °F) | Direction only → Direction only |
  | Seattle (held out) | Conservative → Conservative, but the miss moves to `own_only`: bias +0.022 → +0.052 °F, coverage 92% → 33% | Conservative → **Not trustworthy**; floor becomes invalid |

  - **Why rejected.** Seattle meets the rejection condition written in advance: its bias grows in a world,
    and its 300 m verdict worsens.
  - **Albuquerque's screen was stopped** once Seattle had settled the decision.
  - **What this means.** Knot placement trades error between worlds, city by city. No fixed knots fix
    Burlington without harming a held-out city. Burlington's remaining gap is reported as open
    (`docs/results/canopy_pooled.md`).
  - **Possible later work:** a canopy response that does not depend on knots, such as a monotone spline
    chosen by cross-validation inside each campaign. It would be screened the same way.
- **2026-10-24, A3 (decision 8): a candidate fix for the rest of Burlington's idealised-lab gap, screened on
  held-out cities before adoption.**
  - **Evidence so far.** Burlington, 12 replicates, P2 truth. Base street_100 overstates `additive` by
    −0.133 °F and `confounded` by −0.161 °F.
    - Removing `validate.plant`'s per-cell noise does not help (−0.121 °F).
    - Knots (15, 40, 70) do not help.
    - Knots (25, 50, 75) cut the bias to −0.046 and −0.076 °F, within tolerance, and the SE from 0.138 to
      0.082 °F.
  - **Candidate.** Change the flexible canopy knots from (15, 40) to (25, 50, 75).
  - **Why test before adopting.** The candidate was found on Burlington, so Burlington cannot judge it.
  - **Screen.** The idealised street_100 verdict on Providence, Seattle and Albuquerque: all six worlds, 12
    replicates, the same seeds, P2 truth, current knots against candidate knots. The verdict rules are
    unchanged.
  - **Adopt only if** no held-out city's verdict gets worse and Burlington's improves. Then every city's
    idealised and real-route labs are re-run at 36 replicates and reported before and after.
  - **What would show it wrong:** a held-out city's verdict worsens, or its bias grows in any world. In that
    case the knots stay at (15, 40), and the Burlington gap is reported as open.

- **2026-10-24, A2 outcome: confirmed.** Rescoring and a true 36-replicate re-run under P2 agree to within
  9e-16 on every number (432 per lab), and the verdicts are identical. Checked on Providence's 4 real-route
  labs and Burlington's afternoon one.
- **2026-10-24, A2: the "after" side of decision 7 for the real-route labs is a deterministic rescoring of the
  36-replicate labs already run, checked against true re-runs.**
  - **What.** The replicate's random stream feeds the planted field, the campaign and the estimate. The
    truth is computed separately (its own fixed seed). So under P2 a re-run would produce exactly the same
    estimates and standard errors; only the truth columns change. Recomputing the truths for the new street
    population and re-summarising the stored rows therefore gives the re-run's results. It skips the
    estimated 4 h of lab time (Albuquerque and Seattle take most of it).
  - **Check.** Every real-route lab of Providence and one of Burlington's is also truly re-run under P2, and
    the two results are compared value for value.
  - **What would show it wrong:** any difference between a rescored and a re-run summary beyond floating-point
    noise (1e-9). In that case every lab is truly re-run.
- **2026-10-24, A1: decision 7's idealised labs are re-run at 36 replicates under P2.** "Before" stays the
  12-replicate labs already reported. To separate the two changes, the 12-replicate labs are also rescored
  under P2 and reported.
  - **Why.** Decision 7 asks for 36 replicates, and 12 proved too noisy to judge in Providence's real-route
    labs.
  - **The verdict rules are unchanged.**
  - **What would show it wrong:** nothing about this choice can bias a verdict towards passing. Both counts
    are reported, and the idealised lab is not part of the pool-entry rule.

## Questions for the owner

- **Untimed Heat Watch layers (2022–2023).** 19 pilot cities publish traverses without timestamps or sensor
  ids. Agent decision A7 shows they cannot be rebuilt reliably from row order. If CAPA (or NOAA's NIHHIS
  program) could share the per-sensor raw CSVs with timestamps, these cities could enter the pool. Asking
  them is outside the repo, so it is your call.

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

**Phase 3 progress (session 4, later: owner decisions 12–14)**
- **Cities run:** Providence, Albuquerque, Burlington, Seattle, Atlanta, Raleigh, Charleston, Arlington,
  Richmond, Las Cruces. The queue continues with San Diego, Austin, Bronx, Jersey City, Virginia Beach,
  Asheville, Salt Lake City and New Orleans; Seattle's and Albuquerque's 36-replicate idealised labs come
  last. Status and pool are in `docs/results/canopy_pooled.md`, regenerated by `scripts/pool_doc.py`.
  Per-city notes after the first seven are in `docs/results/canopy_cities.md`.
- **Evening street_100 pool:** 6 cities, −0.066 °F per +10 pp within 100 m (−0.093 to −0.039), τ = 0,
  prediction interval −0.104 to −0.028. No single city carries it (decision 12's leave-one-out table). The
  Phase 3 gate is judged at 8 or more cities (A5).
- **Afternoon pool:** 5 cities, −0.006 °F (−0.050 to +0.038): a null.
- **Decision 12:** done (`pool.py` leave-one-out, `--exclude`).
- **Decision 13:**
  - multi-tile mosaics built and used live (Charleston's 2 DEM tiles);
  - the real-route labs are about 7 times cheaper (A6);
  - the city survey found 19 untimed cities, and P3 failed its check (A7). They stay out; see "Questions
    for the owner".
  - Two campaigns are shared by two pilot cities (Manhattan/Bronx, Jersey City/Newark). One city per
    campaign is run, so the same traverses are not pooled twice.
- **Decision 14:** variable list and primary analyses fixed (A8). Not built yet.
- **Known issue:** the estimate's city-wide floor uses the idealised lab's floor validity. In near-treeless
  Las Cruces it printed a "floor" of 0.98 °F from a design no lab trusts. The fix is to validate the floor on
  the real routes, per run. To do.

**Phase 3 progress (session 4, 2026-10-24, after owner decisions 7–11)**
- **P2 is built** (`26f41f6`). The labs now score street designs at the streets campaigns drive. `lab
  --rescore` re-scores stored replicates, which is exactly equal to a re-run (A2, checked).
- **Pool after P2:** Providence 3/4 runs, Burlington 2/3 (afternoon, evening), Seattle 1/3 (evening),
  Albuquerque 0/2. Before and after tables are in `docs/results/canopy_pooled.md`.
- **First cross-city pooled estimate:** evening, 3 cities: **−0.066 °F per +10 pp within 100 m (−0.098 to
  −0.035)**, τ = 0. Phase 3 gate: not met (it needs 4 or more cities in a window).
- **Burlington's remaining idealised gap (decision 8):** per-cell noise is ruled out. A knots fix was found
  but rejected on held-out Seattle (A3). The gap stays open.
- **Next cities (decision 9):** Atlanta and Raleigh are set up; their labs are queued. Houston is set up
  (641,785 cells: slow).
- **Still to do for decision 7:** the idealised labs at 36 replicates under P2. Providence and Burlington
  first, then Seattle and Albuquerque after the new cities.

**Phase 3 progress (earlier)**
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
- **Fourth city:** Seattle, WA (`mz79p`), run end to end by the batch driver
  (`docs/results/seattle_wa_canopy.md`).
  - **Estimate:** −0.045 °F per +10 pp within 100 m (−0.122 to +0.033). Within 300 m, −0.124 (−0.231 to
    −0.017). City-wide floor 0.03 °F.
  - **Real-route labs:** Conservative in all three runs; the only miss is `own_only`. **Pooled:** no.
- **Scranton, PA (`8cnq4`):** blocked. Its 2023 traverses have no timestamps or sensor ids. Proposal P3.
- **Pool status** (`docs/results/canopy_pooled.md`): only Providence's morning, evening and night runs pass the
  pool-entry rule. That is too few for `pool.py`'s meta-analysis.
  - **Why the leafy cities fail:** Burlington and Seattle fail only through the lab's truth-population
    mismatch, which P2 addresses (P2 now has the Seattle evidence too).
  - **Decision for the owner:** P2 is the decision that most changes what can be pooled.
- **`pool.py`:** `sparc/core/identify/pool.py` plus the `python -m sparc.core.identify pool` subcommand.
  - **What it does:** REML random effects (DerSimonian–Laird fallback), the prediction interval for a new
    city, leave-one-city-out scoring and Stouffer's combination, per time window.
  - **Synthetic tests:** it recovers μ and τ, and LOCO coverage is 95.6% with 12 cities.
  - **On real data:** today it pools only Providence's runs.
  - **Not built:** the meta-regression and the pooled floor.
- **Phase 3 gate** (leave-one-city-out coverage near nominal): not met. It needs at least 4 pooled cities per
  window; there is 1.

**Next step:** depends on P2.
- **If the owner approves P2:**
  1. Build it.
  2. Re-run the real-route labs for Providence and Burlington, and the idealised labs, at 36 replicates with
     unchanged verdict rules.
  3. Update the pool status whatever the result.
- **Either way:** add more single-tile northern cities through `run_canopy_cities.py`.
  - **Scranton (`8cnq4`)** was tried. Its 2023 traverses carry only `t_f`, `rh`, `hi_f` and positions (no
    time, no sensor), so no runs form. It is recorded as "failed at setup", and P3 is written.
  - **Seattle (`mz79p`, 2020 "All Data" bundle)** is done.
  - **Portland (`e7js9`)** returned HTTP 401.
- **Lab cost per city:** 15 min (small, like Burlington) to 2 h (like Albuquerque) on 4 cores.

**Blockers:** none.
**Proposals awaiting owner approval:** none. P2 approved and P3 deferred (decisions 7 and 9); from now on
the agent decides under decision 10.

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
    (10 min 48 s). On the final tree, after Seattle and the cumulative-status fix: 311 passed, 9 deselected
    (10 min 54 s).
    - That run also logged 16,600 scikit-learn `delayed`/`Parallel` UserWarnings in
      `test_models_pipeline.py`, against 33 warnings in earlier runs.
    - They appeared after the sandbox restart. No code they touch changed this session.
    - They did not recur. With `pool.py` added: 318 passed, 9 deselected, 33 warnings (10 min 15 s).
  - **Commits:** `83eb106`, `e767165`, `4cd39cf`, `fd4618f`, `116569f`, `7b8500b`, `b3c7d49`, `307d20f`, `c126354`.
  - **Later in the session:**
    - Seattle ran through the driver.
    - Scranton is blocked (P3).
    - The driver's status table is now cumulative, so running one city no longer erases the others; a test
      covers it.
  - **Sandbox:** the VM restarted once mid-lab, after a tool call slept about 5 min. The driver resumed: it
    skips finished steps.

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
