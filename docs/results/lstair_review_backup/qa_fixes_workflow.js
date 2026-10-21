export const meta = {
  name: 'lstair-qa-fixes',
  description: 'Implement the city-review fixes in four parallel streams (data reader/build, Landsat, evaluation design, continuous US land cover), each followed by an adversarial review-and-repair pass',
  phases: [
    { title: 'Implement', detail: 'data (reader, then drift/flags), Landsat, evaluation, NLCD land cover' },
    { title: 'Review', detail: 'one adversarial reviewer per stream: verify against the spec, run tests, fix confirmed defects' },
  ],
}

const S = args.scratch
const COMMON = (wt) => `
You are implementing fixes in the SPARC repository, in the git worktree ${wt} (branch already checked out; work ONLY
inside ${wt}; do not touch /home/user/sparc-resilience or other worktrees). Python: /home/user/.venv311/bin/python;
run tests from ${wt} with \`PYTHONPATH=. /home/user/.venv311/bin/python -m pytest <paths> -q -p no:cacheprovider\`.
Never run "pip install". Real data live read-only in /home/user/data/capa/lstair (cities/<slug>/meta.json,
table.parquet, rasters.npz; raw/<slug>/ = OSF downloads). They are untrusted downloads: any script that reads them
runs with \`/home/user/.venv311/bin/python -I\` and inserts \`sys.path.insert(0, "${wt}")\` itself; keep such scripts
under ${S}/qa_work/<your-label>/. Never write into /home/user/data. Tests must be offline and synthetic.
Background: the cross-city LST pilot (sparc/core/lstair/: dataset.py builds each CAPA Heat Watch city; evaluate.py
holds cities out; docs/results/lst_air_pilot.md is the PRE-REGISTERED analysis - read it, it is the spec for the
evaluation). A review of the 31 real cities found the faults listed below; the full review evidence is in
${S}/qa_details.txt (read the parts relevant to you). Match the surrounding code's style: comment density, naming,
docstrings that cite evidence, no new dependencies. Commit your work in ${wt} with clear messages ending with the two
lines:
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CCf2ib1fspSM5Y7QAARoDC
Do not push. Before finishing, run the tests you touched plus \`tests/core -m "not slow" -x -q\` (it takes ~20 min;
run it once at the end) and report the counts.`

const SHARED_IFACE = `
Interfaces shared between streams (implement exactly; other streams rely on them):
- table.parquet gains a string column \`qa_exclude\`: "" for a usable row, else the reason the row's RUN is excluded
  by default (e.g. "single-vehicle run: drift not identifiable", "pedestrian platform (median 4.9 km/h)",
  "qa registry: rain over 4 of 7 routes"). evaluate.py drops rows whose qa_exclude != "" unless asked to include
  flagged runs; a table without the column means all rows usable.
- meta.json runs[i] gains "qa_exclude" (same string) and qa["median_speed_kmh"], "platform" ("vehicle"|"pedestrian").
- The continuous US land-cover stream adds, post hoc, columns tcc, tcc_f100, tcc_f300, tcc_f1000, nlcd_imp,
  nlcd_imp_f100, nlcd_imp_f300, nlcd_imp_f1000 (percent 0-100) to table.parquet and the same keys to rasters.npz,
  with meta["landcover_us"] provenance. evaluate.py's \`--land-cover nlcd\` swaps canopy/impervious (and their focal
  means) for these when every scored city has them.`

const DATA_A = `${COMMON('/home/user/wt/qa-data')}
${SHARED_IFACE}
Your part (reader; files sparc/core/identify/traverses.py, sparc/data/collect/capa.py, new sparc/core/lstair/exclusions.py
+ its data file; tests in tests/core/test_identify.py and new tests/core/test_lstair_exclusions.py):
R1. Sensor id hidden by a row-index ID: in traverses._standardise replace
    \`if "file" in norm and not any(n in norm for n in ID_COLUMNS):\` with \`if "file" in norm and _id_column(df) is None:\`
    (a row-unique 'ID' rejected by _id_column must not block the sensor id parsed from 'file'; a valid id column still
    wins). Asheville, Longmont, Salt Lake City, Sedona 2023 tables. Test: row-unique ID + file -> per-sensor vehicles.
R2. Second-resolution clock: CAPA hasFix tables whose local clock (dttm_lc) is minute-truncated (seconds == 0 for
    > 90% of rows) while the GPS 'time' field is HHMMSS[.sss] UTC with seconds (Houston, Sacramento, Fort Lauderdale).
    Rebuild the UTC time as dttm_tc floored to the minute plus the seconds of 'time', ONLY where dttm_tc's HH:MM equals
    time's HHMM (>= 95% of rows, else leave the clock alone), never from the GPS 'date' field (corrupt values like
    '180919M'; runs cross UTC midnight), then convert with the file's tz column / zone as the code already does for
    dttm_tc. Record the clock note. Test with a synthetic table crossing UTC midnight.
R3. Within-file exact duplicates: _drop_duplicates also drops exact (vehicle_id, time, lat, lon, temp_f) repeats
    within one file (Burlington CAPA1034 pm: 3,093 of 6,074); count them with the cross-file ones in the report.
R4. Helper \`non_vehicle_sensors(points) -> np.ndarray[bool]\` in traverses.py: True where the sensor id (vehicle_id,
    else the part of vehicle_key after the last ':') matches r"bicycle|bike|\\bwalk|pedestrian" case-insensitively
    (Sacramento's 'Bicycle_CAPA56'). Do not apply it here; the build applies it.
R5. QA registry: sparc/core/lstair/exclusions.py with a small, documented data file (YAML or JSON next to it;
    whichever the repo already reads - check for yaml usage) of verified faults, each entry: slug, kind ("readings"
    or "run"), sensor (id, for readings), date (local, YYYY-MM-DD), window ("morning"|"midday"|"evening"|"night"),
    from_local / until_local (HH:MM, either may be absent), reason, evidence (one line). Functions:
    \`reading_mask(points, slug) -> (keep: bool array, records: list[dict])\` and \`run_exclusions(slug) -> dict[run_id,
    reason]\` where run_id is "<date> <window>" as in meta runs. Entries (from the review, all verified by a skeptic):
      - asheville_nc readings, sensor CAPA2037, 2023-07-24 midday, from 15:05: rain-cooled route (RH 40->68%, -5.8 C).
      - los_angeles_ca readings, sensor CAPA47, 2019-09-15 morning, until 06:16: sensor warm-up (flat 23.9-24.2 C over
        7 km, then 18.4 C); and 2019-09-15 evening until 19:11: warm-up transient (27.0 -> 21.6 C).
      - fort_lauderdale_fl run 2019-09-18 morning: rain over 5 of 7 routes (51% of readings at RH >= 99%).
    Sensor matching must use the sensor id as parsed (vehicle_id / the key suffix), and times the local clock.
    Tests: masks and run ids on synthetic points.
Report what you changed and the test counts.`

const DATA_B = `${COMMON('/home/user/wt/qa-data')}
${SHARED_IFACE}
A previous agent in this same worktree has just committed the reader part (R1-R5: sensor-id fix, second-resolution
clock, within-file duplicates, traverses.non_vehicle_sensors, sparc/core/lstair/exclusions.py). Read its commits
(\`git log -5\`, \`git show\`). Your part is the build: sparc/core/lstair/dataset.py (adjust_visits, build_city, era5_run,
settings) with tests in tests/core/test_lstair_dataset.py:
D1. Within-vehicle protocol line: adjust_visits' protocol detrend fits temp ~ vehicle dummies + P(t) and removes
    P @ slope (Virginia Beach af: routes ending early aliased +2.29 C/h of route composition into time; within-vehicle
    +1.06). It reduces to the current line with one vehicle. Update drift_per_h / drift_coef.
D2. Vehicle offsets only from co-located near-simultaneous visits: two vehicles are linked only by cells both visited
    within |dt| <= 10 min (at least 5 such cells); offsets are estimated per connected component of that graph and
    vehicles without such a link keep offset 0. Rationale (verified): offsets fitted from single short links 52-57 min
    apart were wrong (Los Angeles am CAPA34/CAPA39 +-1.4 C; Richmond pm 1102), while near-simultaneous co-visits
    corroborate real ones (Fort Lauderdale af CAPA70 ~ +1.2 C). Keep the cell fixed effects for the drift.
D3. FE drift guard: with >= 2 vehicles, accept the fe drift only if the >= 15 min repeat cells fall in >= 10 distinct
    300 m blocks AND no single vehicle holds more than 80% of them; otherwise use the protocol line (D1). Cluster the fe
    drift SE by vehicle (or block). When |fe - protocol| > 0.75 C/h, add a warning (not a fallback). Cases: Fort
    Lauderdale 2019-09-18 midday (all 22 repeats from CAPA80: fe -2.16 vs protocol +0.20, injected ramps), New Orleans
    2020-08-18 midday (34 of 35 repeats from CAPA1050). Single-vehicle runs are exempt from the vehicle-share rule.
D4. Single-vehicle runs: when one vehicle holds >= 90% of a run's visits (count real vehicles after R1) and the drift
    is the protocol line, set the run's qa_exclude = "single-vehicle run: drift not identifiable" (Lynchburg am/af,
    Fort Lauderdale 09-21/09-23 make-up runs, Boston if it is one key per file - check what Boston really is and say).
D5. Platform: per run, the median speed of each vehicle's consecutive raw readings (dt <= 30 s, haversine), stored as
    qa["median_speed_kmh"]; below 10 km/h set platform "pedestrian", a warning, and qa_exclude = "pedestrian platform
    (median X km/h)" (Farmville 4.7-5.7, Abingdon ~5; cars >= ~30).
D6. Apply traverses.non_vehicle_sensors before gridding: drop those readings with a warning naming the sensor ids.
D7. Apply exclusions.reading_mask before gridding (warning with counts) and exclusions.run_exclusions to qa_exclude.
D8. ERA5: era5_run returns the time-weighted mean over the run's span of the hourly fields linearly interpolated (Sedona
    pm ended 19:27 but averaged 19 h and 20 h: bg-ERA5 +5.4 C), request the bracketing hours.
D9. Feature reuse: rebuilding a city whose previous ok build has the same grid (crs, x0, y0, dx, nx, ny), campaign date
    and feature provider may reuse that build's feature rasters (FEATURES keys from rasters.npz) instead of calling
    open_features (the Sentinel-2 composite takes 5-25 min per city); record meta["features_reused_from"] (built_utc).
    LST and ERA5 are always recomputed. A different grid recomputes. Read the old rasters before overwriting them.
D10. Bump a dataset/QA version in the settings key so every existing city cache is reported stale and rebuilt.
D11. Write qa_exclude into table.parquet (per row, from its run) and meta runs, as the shared interface says.
Tests for D1-D9 on synthetic data (known drift and offsets: the fe/protocol choices, offsets from co-visits only,
single-vehicle and pedestrian flags, exclusions applied, feature reuse used/not used). Report changes and test counts.`

const LST = `${COMMON('/home/user/wt/qa-lst')}
Your part: sparc/data/collect/landsat.py (campaign_lst and scene selection) and how sparc/core/lstair/dataset.py's
landsat_lst calls it (only if needed; another stream edits dataset.py heavily, so keep changes there minimal and say
exactly what you changed), tests in tests/test_landsat_c2.py:
L1. Cloud-edge buffer: LST next to cloud masks is cool-biased (verified: land-cover residual of LST -0.75 to -0.95 C
    within 60 m of masked pixels, -0.32 at 60-120 m, ~0 beyond 120-240 m, inside 2 km blocks; Charleston, Virginia
    Beach). Mask LST within 150 m (5 Landsat 30 m pixels) of the cloud, dilated-cloud, cirrus and cloud-shadow QA bits
    (NOT fill, so scene edges are not eaten). clear_fraction keeps using the undilated mask. Record the buffer in the
    result ("mask_buffer_m": 150).
L2. One overpass split across WRS rows: candidates with the same path and date (consecutive rows, seconds apart) are
    mosaicked onto the city grid before the clear fraction is computed and a scene chosen (Albuquerque: row 35 ended
    inside the grid, row 36 of the same overpass never inspected; Atlanta rows 36/37 of 2021-08-29 at 0.60/0.63 each).
    Each row is QA-masked on its own; overlaps take any clear value. The chosen "scene" may then be a list of ids;
    keep the result schema readable by existing code (scenes list with ids, date, clear_fraction).
L3. Footprint vs cloud: report per candidate the share of the study area inside its footprint (not fill) apart from
    the clear share, so a scene that misses part of the city is not mistaken for a cloudy one (Seattle: the east of
    the grid lies outside path 047's footprint).
L4. Choice when no candidate reaches prefer_clear: take the candidate nearest in time whose clear fraction is within
    0.10 of the clearest (and >= min_clear), rather than the clearest alone (Arlington: 06-30 at 0.72, -15 d, chosen
    over 07-16 at 0.64, +1 d). Document the rule next to the existing one.
Tests on synthetic QA/ST arrays (buffer excludes fill, mosaic of two rows, footprint share, L4 choice).
Report changes and test counts.`

const EVAL = `${COMMON('/home/user/wt/qa-eval')}
${SHARED_IFACE}
Your part: sparc/core/lstair/evaluate.py and the evaluate subcommand in sparc/core/lstair/cli.py, tests in
tests/core/test_lstair_evaluate.py and tests/core/test_lstair_synthetic.py. Implement the PRE-REGISTERED analysis in
docs/results/lst_air_pilot.md exactly (read it first; where this list and the document differ, the document wins and
you say so):
E1. Model N (no anomaly: zero after centring). Per held-out unit and run, skill SS_M = 1 - MSE_M / MSE_N, averaged over
    the unit's runs of the window; the primary comparison is dSS = SS_B - SS_A per unit (and SS_C - SS_A, SS_C - SS_C0),
    tested with the existing one-sided sign-flip (exact up to 16 units, else random) and a 95% t-interval. Absolute
    dMSE is kept as a secondary row. Claim rule: Holm-adjusted p <= 0.025 over the 3 windows (am, af, pm) x 2 claims
    (B-A, C-A), CI above 0, mean dSS >= 0.05, and mean SS_B > 0 (resp. SS_C > 0). The primary window is af; others are
    secondary but get Holm-adjusted p too. Report per window the number of held-out units with r(A, dTa) < 0.
E2. Training weights: sample weight proportional to 1/(n_city * var_city(dTa)) from training labels only (mean 1);
    Settings.weights = "variance" (default) | "n" (old) for the sensitivity. Apply in the inner CV and the final fits.
E3. campaign_groups: the node+date link only when the grids lie within SAME_NODE_KM = 50 km (compute gx, gy first;
    keep the 10 km neighbour link). Setting/flag to restore the old node+date link for the "Virginia as one group"
    sensitivity. Update the docstring (the old merge cost power; it was conservative, not anti-conservative).
E4. Centring on scored cells: apply the predictor filter (cells with every A and B predictor) BEFORE design(), so the
    land-cover anomalies, LST anomalies and target are all centred on the scored cells (Seattle impervious median 53 vs
    69 on scored cells).
E5. Spline extrapolation "constant" (clamped) instead of "linear" (LA's elevation anomaly SD 140 m vs 6-46 m elsewhere).
E6. Rows with qa_exclude != "" are dropped by default (list them in results); Settings/flag include_flagged=True keeps
    them (sensitivity). Old tables without the column: all rows usable.
E7. Leave-one-unit-out influence on each claim: recompute mean dSS, CI, p and verdict without each unit; report the
    range and whether any single unit flips the verdict.
E8. Sensitivity variants (Settings fields + CLI flags), each a full rerun of evaluate():
    - lst_mode = "full" (default) | "local": B uses lst - lst_f3000, lst_f100 - lst_f3000, lst_f300 - lst_f3000, where
      lst_f3000 is a NaN-aware Gaussian focal mean (sigma 3000 m, as dataset.focal_mean) computed from the city's lst
      raster at load time (no rebuild);
    - regional = False (default) | True: A and B also get elevation_f3000 and impervious_f3000 anomalies (computed from
      rasters likewise);
    - land_cover = "worldcover" (default) | "nlcd" (shared interface; refuse with a clear error if a scored city lacks
      the columns);
    - physics_transfer = "median" (default, per-parameter medians) | "medoid": the training city whose fitted parameter
      vector (log of positive parameters, standardised) is closest to all others, used whole;
    - physics_cell_m and physics_L_max_m flags (existing Settings fields; add CLI flags --physics-L-max).
    Also report per window how many physics fits sit within 5% of the L bounds (lower = cell size, upper = L_max).
E9. A \`sparc core lstair sensitivity --data DIR --out DIR [--variants ...]\` subcommand (or an evaluate flag) that runs
    the variants above (absolute statistic needs no rerun) sequentially, reusing the physics fit cache where the key
    allows, and writes sensitivity.json + sensitivity.md (one row per variant: per-window mean dSS B-A and C-A, CI,
    Holm p, verdict).
E10. results.md: the primary verdict first (af, as registered), then the other windows, the per-unit table (SS_N=0,
    SS_A, SS_B, SS_C, SS_C0, r), influence, exclusions; secondary absolute dMSE.
E11. Keep the synthetic validation honest: update sparc/core/lstair/synthetic.py and its tests if the statistic change
    needs it, and run \`sparc core lstair validate\` (small seed count is fine) to show the new rule still claims in the
    informative worlds and not in the null/redundant ones; report the rates.
Report changes, test counts and the validation rates.`

const NLCD = `${COMMON('/home/user/wt/qa-nlcd')}
${SHARED_IFACE}
Your part: continuous US land cover as a sensitivity for "LST beyond land cover" (the critic's point: WorldCover's
tree and built CLASS fractions miss street trees inside built pixels; canopy is exactly 0 on 55-98% of driven cells in
17 of 31 cities). All 31 cities are in CONUS.
N1. Find an openly reachable source (the network is open; probe with short timeouts) for (a) USFS/NLCD Tree Canopy
    Cover (percent, 30 m; 2016/2019/2021) and (b) NLCD Percent Developed Imperviousness (30 m; 2016/2019/2021): e.g.
    MRLC's GeoServer WCS/WMS (www.mrlc.gov/geoserver/...), MRLC direct COGs, AWS/Azure/Planetary Computer open-data
    mirrors. Prefer a windowed read (WCS GetCoverage by bbox, or a COG via GDAL /vsicurl/) over whole-CONUS downloads.
    Record what works (URL patterns, layer names, years) in the module docstring.
N2. New module sparc/core/lstair/landcover_us.py: fetch both layers for a city grid (CityGrid in dataset.py: crs, x0, y0,
    dx, nx, ny; grid cells are 30 m in a UTM CRS), reproject/average onto the grid, nearest NLCD year to the campaign
    date, and compute focal means at 100/300/1000 m with dataset.focal_mean. Add a CLI subcommand in
    sparc/core/lstair/cli.py: \`sparc core lstair add-landcover --data DIR [--cities ...]\` that augments each ok city
    post hoc (atomic writes of table.parquet and rasters.npz; meta["landcover_us"] provenance: source URLs, years,
    layer names, coverage share). Do not edit dataset.py build_city (another stream rewrites it); a later rebuild
    would drop the columns, so the command must be idempotent and rerunnable.
N3. Offline tests with a mocked fetch (synthetic rasters): grid alignment, units, focal means, idempotence, schema.
N4. Do NOT write into /home/user/data. To check the real source end to end, run the fetch for one small city into a
    COPY under ${S}/qa_work/nlcd/ (copy /home/user/data/capa/lstair/cities/farmville_va there first) and report the
    tcc/nlcd_imp ranges and their correlation with WorldCover canopy/impervious on driven cells.
If no source is reachable, say exactly what you tried and stop after the code + tests with a mocked source.
Report changes, test counts and the real-data check.`

const REVIEW = (wt, what) => `${COMMON(wt)}
${SHARED_IFACE}
You are the adversarial reviewer of the work just committed in ${wt} (compare with f79e934: \`git diff f79e934 --stat\`,
\`git log f79e934..\`). The specification it had to meet:
---
${what}
---
Check each numbered item: implemented as specified? correct (trace a realistic input through it; edge cases: one
vehicle, no LST, NaNs, empty runs, old caches without new columns)? tested by a test that would fail without it? Any
regression in existing behaviour or tests? Then FIX every confirmed defect yourself in ${wt} (small, focused commits)
and add the missing tests. Do not rewrite working code for taste. Finally run the touched tests and
\`tests/core -m "not slow" -q\` once IN THE FOREGROUND (background commands do not survive a container restart) and
report: a list of findings (item, defect, fixed y/n), test counts. The implementer may have stopped before its own
tests finished: treat nothing as tested until you have run it.`

const RESUME = `IMPORTANT - RESUMING: a container restart interrupted the previous agent on this task. Its work so far is in
the worktree (commits after f79e934 and/or uncommitted changes). FIRST inspect \`git log f79e934..\`, \`git status\` and
\`git diff\`; continue from that state and never discard or reset it. Finish the remaining items, commit, and report.
Test-run override: run only the test files you touched (the reviewer runs the full tests/core suite); background
commands do not survive a restart, so run tests in the foreground and wait for them.
`
const NOFULL = `Test-run override: run only the test files you touched (the reviewer runs the full tests/core suite), in the
foreground.
`
const streams = [
  { key: 'data', wt: '/home/user/wt/qa-data', impl: [DATA_A, NOFULL + DATA_B], spec: DATA_A + '\n' + DATA_B },
  { key: 'lst', wt: '/home/user/wt/qa-lst', impl: [LST], spec: LST },
  { key: 'eval', wt: '/home/user/wt/qa-eval', impl: [RESUME + EVAL], spec: EVAL },
  { key: 'nlcd', wt: '/home/user/wt/qa-nlcd', impl: [RESUME + NLCD], spec: NLCD },
]

const out = await pipeline(
  streams,
  async (s) => {
    const reports = []
    for (let i = 0; i < s.impl.length; i++) {
      reports.push(await agent(s.impl[i], { label: `impl:${s.key}${s.impl.length > 1 ? ':' + (i + 1) : ''}`, phase: 'Implement' }))
    }
    return reports
  },
  async (reports, s) => {
    const rev = await agent(REVIEW(s.wt, s.spec), { label: `review:${s.key}`, phase: 'Review' })
    return { stream: s.key, worktree: s.wt, implement: reports, review: rev }
  },
)
return out.filter(Boolean)
