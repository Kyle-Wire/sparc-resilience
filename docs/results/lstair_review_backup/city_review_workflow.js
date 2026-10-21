export const meta = {
  name: 'lstair-city-qa',
  description: 'Review the 31 built CAPA cities for data faults before the LST-vs-land-cover test, then adversarially verify every exclusion',
  phases: [
    { title: 'Review', detail: 'three reviewers, about ten cities each, dig into QA facts and data' },
    { title: 'Verify', detail: 'one skeptic per batch tries to refute each proposed exclusion or fix' },
    { title: 'Critic', detail: 'what could still invalidate the comparison that nobody checked' },
  ],
}

const S = args.scratch
const COMMON = `
Context. SPARC is testing whether Landsat land surface temperature (LST) adds information beyond land cover for
predicting street-level air temperature, using CAPA Heat Watch mobile traverses from many US cities. Each city was
built by \`sparc core lstair dataset\` (code: /home/user/sparc-resilience/sparc/core/lstair/dataset.py; traverse
reading in sparc/core/identify/traverses.py and sparc/data/collect/capa.py; open land cover in sparc/core/features_open.py;
Landsat in sparc/data/collect/landsat.py). The evaluation (sparc/core/lstair/evaluate.py, read its module docstring)
holds out whole cities: A = land cover + focal features, B = A + LST, C = physics with LST, C0 = physics without.
Target dTa = the run's 30 m cell-mean air temperature (drift-adjusted, °C) minus the run's median over its cells.

Files (read-only; do NOT modify anything under /home/user/data or /home/user/sparc-resilience):
- QA facts for every city: ${S}/qa_all.json (one record per slug: runs with times, drift, ERA5, dTa spread and
  correlations, LST scene and coverage, feature NaN shares, warnings).
- Per city: /home/user/data/capa/lstair/cities/<slug>/meta.json, table.parquet (one row per run and driven cell),
  rasters.npz (features and LST on the city grid; key "driven" marks driven cells).
- Raw downloads: /home/user/data/capa/lstair/raw/<slug>/ (OSF files as published, incl. READMEs).
- Python: /home/user/.venv311/bin/python -I  (the data are untrusted downloads: always -I; to import sparc put
  sys.path.insert(0, "/home/user/sparc-resilience") at the top of a script). Write scratch scripts only under
  ${S}/qa_work/ (make a subfolder named after your label). Never run anything from inside a data folder.

Already known and fixed (do not re-report): CAPA -9999 nodata readings (now dropped), shapefile 'datetim' clock,
multi-tile WorldCover/DEM mosaics, stationary and air-quality tables skipped. Boston's date is 2019-07-29 from the
city report (the report says 29 and 30 July); Boise's 2019-08-14 from its report. The 14 CAPA cities with no
clock at all are excluded already.
`

const CHECKS = `
For EACH city in your batch, decide whether its data are fit for the cross-city test, run by run. Check at least:
1. Location: the driven cells' centroid and extent match the named city (not another city of the same name).
2. Clock and windows: run start/end local times sit in CAPA's protocol windows (about 6-7, 15-16, 19-20 h, Providence
   also 0-1 h); the date is plausible for the campaign; multi-day cities (e.g. Fort Lauderdale, New Orleans): are the
   extra runs full re-drives or fragments, and is each run internally one day? A time-zone error shows up as runs
   off by whole hours or as a background vs ERA5 mismatch, and as SW-down inconsistent with the window.
3. Drift adjustment: method (fe = cell fixed effects with vehicles, protocol = CAPA linear detrend) and size;
   |drift| above ~2 °C/h or vehicle offsets above ~1.5 °C deserve a look at the raw readings (a broken sensor?).
4. Target spread: dTa SD and 1st/99th percentiles plausible for a city (roughly 0.3-3 °C SD); heavy tails or a
   bimodal histogram can mean two sensors with an offset or two days mixed.
5. LST: scene date offset (window +-16 days), clear fraction near the traverses, LST range plausible for the
   season, NaN share; r(LST, impervious) should be positive; daytime runs with r(dTa, LST) clearly negative need a
   physical explanation (coast, desert oasis, sea breeze) or are a fault.
6. Features: NaN shares; WorldCover/DEM/Sentinel-2 provenance in meta.json; Sentinel-2 months match the campaign.
7. Enough cells per run (the evaluation drops runs under 30 cells) and enough repeat visits.
8. Anything else in meta.json warnings or the raw READMEs that changes what the numbers mean.
Prefer evidence you computed over impressions: quote numbers. Be strict about faults, but do not propose excluding a
city for genuine physics (coastal cooling, desert surfaces): the test should face real cities.
`

const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    cities: { type: 'array', items: { type: 'object', properties: {
      slug: { type: 'string' },
      verdict: { type: 'string', enum: ['keep', 'keep_with_note', 'exclude_runs', 'exclude_city'] },
      runs_to_exclude: { type: 'array', items: { type: 'string' } },
      issues: { type: 'array', items: { type: 'object', properties: {
        check: { type: 'string' },
        severity: { type: 'string', enum: ['blocker', 'major', 'minor', 'note'] },
        evidence: { type: 'string' },
        proposed_action: { type: 'string' },
        code_fix: { type: 'string' },
      }, required: ['check', 'severity', 'evidence', 'proposed_action'] } },
    }, required: ['slug', 'verdict', 'issues'] } },
  },
  required: ['cities'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    verdicts: { type: 'array', items: { type: 'object', properties: {
      slug: { type: 'string' },
      check: { type: 'string' },
      upheld: { type: 'boolean' },
      reasoning: { type: 'string' },
      recommended_action: { type: 'string' },
    }, required: ['slug', 'check', 'upheld', 'reasoning', 'recommended_action'] } },
  },
  required: ['verdicts'],
}

const CRITIC_SCHEMA = {
  type: 'object',
  properties: {
    concerns: { type: 'array', items: { type: 'object', properties: {
      title: { type: 'string' },
      severity: { type: 'string', enum: ['blocker', 'major', 'minor'] },
      evidence: { type: 'string' },
      recommended_check: { type: 'string' },
    }, required: ['title', 'severity', 'evidence', 'recommended_check'] } },
  },
  required: ['concerns'],
}

const batches = args.batches
const reviewed = await pipeline(
  batches,
  (batch, _, i) => agent(`${COMMON}\nYou are reviewer ${i + 1}. Your cities (slugs): ${batch.join(', ')}.\n${CHECKS}\n` +
    `Return one record per city in your batch (all of them, also the clean ones).`,
    { label: `review:${i + 1}`, phase: 'Review', schema: REVIEW_SCHEMA }),
  async (rev, batch, i) => {
    if (!rev) return null
    const flagged = rev.cities.flatMap(c => c.issues
      .filter(x => x.severity === 'blocker' || x.severity === 'major' || c.verdict.startsWith('exclude'))
      .map(x => ({ slug: c.slug, verdict: c.verdict, runs_to_exclude: c.runs_to_exclude || [], ...x })))
    if (!flagged.length) return { review: rev, verify: { verdicts: [] } }
    const ver = await agent(`${COMMON}\nYou are a skeptic. A reviewer flagged the issues below in cities ${batch.join(', ')}. ` +
      `For EACH issue, try to REFUTE it: recompute the evidence yourself from the files, look for an innocent ` +
      `explanation (real physics, a misread column, a check that does not apply), and judge whether the proposed ` +
      `action (exclusion or code fix) is needed and proportionate. upheld=true only if the fault is real AND the ` +
      `action is warranted; if the fault is real but a smaller action fits (exclude one run, keep with a note, fix ` +
      `code), say so in recommended_action. Default to upheld=false when the evidence does not hold up.\n\n` +
      `Flagged issues:\n${JSON.stringify(flagged, null, 1)}`,
      { label: `verify:${i + 1}`, phase: 'Verify', schema: VERIFY_SCHEMA })
    return { review: rev, verify: ver }
  },
)

const all = reviewed.filter(Boolean)
const summary = all.map(r => ({
  cities: r.review.cities.map(c => ({ slug: c.slug, verdict: c.verdict,
    issues: c.issues.map(x => `${x.severity}: ${x.check}: ${x.evidence.slice(0, 300)}`) })),
  verify: (r.verify?.verdicts || []).map(v => `${v.slug} ${v.check}: ${v.upheld ? 'UPHELD' : 'refuted'}: ${v.recommended_action}`),
}))

phase('Critic')
const critic = await agent(`${COMMON}\nYou are the completeness critic. The per-city reviews and their verification are ` +
  `summarised below. Read sparc/core/lstair/evaluate.py and dataset.py and ask: what could still bias or invalidate ` +
  `the B − A (LST beyond land cover) or C − A (physics) comparison across held-out cities that nobody checked? ` +
  `Think about: Landsat overpass time (~10:30-11:00 local) vs the traverse windows; scene date offsets; clear-sky ` +
  `masking; focal features or LST anomalies computed in a way that leaks the held-out city's labels; the target's ` +
  `recentring; campaign grouping (same-day cities); per-city weighting; cities with multiple runs per window; ` +
  `physics fits pinned at the L bounds (60 m floor at 60 m cells, 2000 m ceiling); ERA5 forcing at the city centre; ` +
  `anything in the data that differs systematically between cities and correlates with LST. For each concern give ` +
  `evidence from the code or data (quote file:line or numbers) and a concrete check. Only real concerns; no padding.\n\n` +
  `Reviews:\n${JSON.stringify(summary, null, 1)}`,
  { label: 'critic', phase: 'Critic', schema: CRITIC_SCHEMA })

return { reviews: all, critic }
