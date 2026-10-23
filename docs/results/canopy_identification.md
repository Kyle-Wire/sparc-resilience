# Isolating the causal effect of canopy (Providence)

**Question.** How much does tree canopy cool the afternoon air, as a cause, not as a correlate of
everything else that comes with leafy streets?

**Answer, in short.**

| Canopy raised by +10 pp... | Can one heat-watch campaign isolate the effect? | How |
|---|---|---|
| within 100 m of a point (about a city block) | **Yes.** The design recovers the true effect in every simulated world (bias ≤ 0.02 °F) and stays at zero when canopy does nothing. | Street differences on the raw traverse readings |
| within 300 m | **Direction only.** The sign is reliable; the size depends on how far the cooling of a tree spreads. | Same design |
| city-wide (+10 pp everywhere) | **A floor, yes.** If canopy never warms the air at a distance, a city-wide edit cools a street at least as much as the same edit within 300 m of it. The one-sided 95% floor held in ≥ 92% of simulated campaigns in every world. | Same design |
| within 1 km, or the city-wide value itself | **No, not from one city's campaign.** A kilometre-scale canopy effect and a kilometre-scale confounder that tracks canopy leave the same traces. Four designs were built to break this (below); none can with one city's data. | Many cities' campaigns pooled |
| the part of kilometre-scale cooling the wind carries | **Testable, but one day is too little.** The same streets are compared across runs under different winds. The test never raised a false alarm in the lab, including with a confounder whose effect changes through the day, but it detects real advection in only 8% of one-day campaigns. | Same streets, different winds |
| anything, from the interpolated map | **No.** The design that works on the measurements reports a canopy effect in 58% of worlds where there is none when run on the map. | — |

So canopy's causal effect *can* be isolated at the scale of a block, and bounded from one side at the
scale of the city, but only from the vehicle traverses, not from the gridded temperature map. The
Providence traverses are on OSF (project `wu9v7`). The method is built, validated on Providence's own
layout, and runs on the downloaded folder as it is ([below](#running-it-on-the-real-traverses)).
It has now run on them: no block-scale effect is detected, the city-wide floor is 0.05 °F of cooling per
+10 pp, and the kilometre test has no power under that day's winds
([Real traverses](#real-traverses-providence-2020-07-29)).

Code: `sparc/core/identify/`. Lab output: `output/core/providence/identify/identify_lab.{json,md}` and
`identify_kmlab.{json,md}`.
Results page: section "Can canopy's own effect be isolated?".

## Why the map cannot answer it

`AAT_z` is a product, not a measurement. CAPA drove vehicles with sensors along routes during the
afternoon hour, then a random forest on land-cover predictors filled every 30 m cell. 72% of its
values are whole °F.

Where canopy and paving move together, the forest learns canopy as a stand-in for whatever it cannot
see. The map therefore carries a canopy dependence even where canopy has no effect at all:

- **The simulation check.** It plants no effect, emulates the product and re-runs the full pipeline.
  It finds −0.19 °F for "+10 pp canopy", the same size as the real estimate (25% false positives;
  8% without the forest step).
- **The identification lab** (below), on the map:
  - the levels regression reports an effect in 50% of no-effect worlds;
  - neighbour differences report one in 58%.

  Run on the traverses the forest learned from, the identical neighbour-difference estimator reports one
  in 0%.

No model of the map can remove this, because the map is itself a function of canopy.

The real Providence map behaves as a land-cover interpolation would: it shows no street-scale canopy
effect at all.

| Design on the real map (`python -m sparc.core.identify map`) | +10 pp canopy | 95% CI |
|---|---|---|
| Neighbour differences, canopy within 100 m | −0.01 °F | −0.08 to +0.06 |
| Neighbour differences, canopy within 300 m | +0.01 °F | −0.12 to +0.14 |
| Levels regression with a smooth spatial basis | −0.65 °F | −2.09 to +0.79 |

## The approach: go back to the measurements

**1. Street differences along each pass.** A vehicle pass samples a street every few seconds.
Compare samples of the same pass up to 300 m apart, seconds to a minute apart:

- the afternoon's warming drift cancels, and a Δtime term absorbs the rest;
- the vehicle's sensor offset cancels;
- anything that varies smoothly over a few hundred metres cancels: neighbourhood wealth, building
  age, distance to the bay, elevation trends.

What remains is how temperature changes along a street as the canopy around it changes. Street trees
come and go while the road stays paved. Impervious surface, albedo, elevation and distance to water
(piecewise) are differenced alongside as controls. Standard errors are clustered on 1 km blocks.

**2. Canopy by reach.** Canopy enters as means over distance rings around each point:

- the cell itself;
- 0–100 m;
- 100–300 m;
- 300 m–1 km.

Raising canopy in every ring out to R is exactly a disk edit of radius R around the point, so the
coefficients add up to *the effect of canopy within R*. One fit gives three readings: within 100 m,
300 m and 1 km.

**3. A flexible near-field response.** Shade saturates. A straight line fitted across 0–100% canopy
is dominated by the largest canopy contrasts and understates the next 10 pp at low canopy, where most
streets are. A linear version recovered only ~78% of the truth in every world.

So canopy in the cell and the 100 m ring enters through a piecewise-linear basis (knots at 15% and
40%). The effect is the fitted model applied to the exact +10 pp edit, clipped at 100%. Wider rings
average many cells and stay linear: the extra terms cost more precision there than they gain.

**4. The wind signature (a test, not an estimate).** Air carries the cooling of trees downwind, so a
physical effect makes upwind canopy matter more than downwind canopy. Confounding by land use and an
isotropic interpolator have no direction. The forcing file gives the wind (from 170°): the design
reports the 1 km upwind − downwind canopy contrast.

## Validation: the identification lab

Every design is run on simulated campaigns over Providence's real layout. Each campaign has:

- street routes every 300 m, cut at unpaved stretches (~500 passes);
- 10 vehicles over one afternoon hour;
- warming drift of 0.5–1.5 °F/h;
- a vehicle sensor offset (sd 0.2 °F) and sensor noise (sd 0.3 °F);
- the forest-made map built from those traverses.

The campaigns run over six worlds where the true effect is known: the simulation check's generators,
each with a spatially correlated residual matched to the real target. That is 12 campaigns per world,
72 in all.

Each design is scored against **its own estimand**, computed exactly from the generator as the mean
change at street points under disk edits of each radius. How far cooling reaches differs by world.
True effect of +10 pp (°F):

| World | Within 100 m | Within 300 m | Within 1 km | Everywhere |
|---|---|---|---|---|
| `additive` — cooling within ~150 m of the trees | −0.165 | −0.268 | −0.269 | −0.269 |
| `own_only` — cooling only in the tree's own cell | −0.275 | −0.275 | −0.275 | −0.275 |
| `physics` — cooling carried downwind over ~1 km (advection–diffusion) | −0.029 | −0.109 | −0.239 | −0.272 |
| `coarse_scale` — cooling spread over ~1 km | −0.006 | −0.047 | −0.224 | −0.254 |
| `confounded` — local cooling plus a hidden 2 km factor that tracks canopy | −0.165 | −0.268 | −0.269 | −0.269 |
| `null` — canopy does nothing | 0 | 0 | 0 | 0 |

In the advected `physics` world only 40% of a tree's cooling lands within 300 m of it.

**Results** (mean estimate / truth; for `null`, the share of 95% intervals wrongly excluding zero):

| Design | null | additive | own_only | physics | coarse | confounded | Verdict |
|---|---|---|---|---|---|---|---|
| Street differences, within 100 m | −0.015 · 8% | −0.18 / −0.17 | −0.29 / −0.28 | −0.03 / −0.03 | −0.02 / −0.01 | −0.18 / −0.17 | **Trustworthy** |
| Street differences, within 300 m | −0.010 · 0% | −0.26 / −0.27 | −0.26 / −0.28 | −0.06 / −0.11 | −0.09 / −0.05 | −0.24 / −0.27 | Direction only |
| Street differences, within 1 km | −0.053 · 8% | −0.31 / −0.27 | −0.24 / −0.28 | −0.27 / −0.24 | −0.29 / −0.22 | **−0.03 / −0.27** | Not trustworthy |
| Map neighbour differences, within 300 m | −0.034 · **58%** | −0.22 / −0.23 | −0.23 / −0.23 | −0.11 / −0.09 | −0.11 / −0.05 | −0.13 / −0.23 | Not trustworthy |
| Levels regression on traverses (footprints + basis) | +0.22 · 50% | −0.92 / −0.27 | −0.81 / −0.28 | −0.28 / −0.27 | +0.48 / −0.25 | −0.11 / −0.27 | Not trustworthy |
| Levels regression on the map | +0.09 · 50% | −0.74 / −0.23 | −0.64 / −0.23 | −0.42 / −0.24 | −0.05 / −0.25 | −0.15 / −0.23 | Not trustworthy |

The wind signature flags advection in 25% of `physics` campaigns and in up to 17% of worlds without
it. It is not trustworthy with one campaign: the expected contrast (about −0.04 °F at 1 km) is far
smaller than its standard error (about 0.1 °F).

Verdict rules (`validate.design_verdict`):

- **Trustworthy**:
  - with no effect, the mean is within 0.05 °F and intervals exclude zero at most 1 in 8 times;
  - in every effect world, the bias is within a third of the truth (at least 0.03 °F) and the 95%
    intervals cover the truth ≥ 80% of the time.
- **Direction only**: clean with no effect and the right sign everywhere.

**Precision.** The within-100 m design has a standard error of about 0.034 °F per campaign. One
campaign therefore detects a block-scale effect of about 0.1 °F per +10 pp or more (80% power). It
cannot tell the physics world's −0.03 °F from zero (17% power), though it is unbiased there.

**From streets to the city.** The design identifies the effect at street points. Applied to every
cell it extrapolates, and the extrapolation runs 10–18% stronger than the truth in the local worlds:
streets have less canopy than the city, and shade saturates.

## What can and cannot be identified

Comparing nearby readings removes everything that varies smoothly across a neighbourhood. That is
what makes the block-scale answer trustworthy, and also why it cannot see further.

A kilometre-scale cooling effect and a kilometre-scale factor that tracks canopy are observationally
equivalent within one campaign. Older, leafier and wealthier districts are such a factor. In the lab,
the 1 km reading recovers 129% of a real kilometre-scale effect (`coarse_scale`) but 12% of the truth
when such a factor is present (`confounded`).

### What one campaign does identify at the city scale: a floor

The city-wide edit equals the edit within 300 m plus the edit beyond it. If canopy never warms the air
at a distance, the part beyond 300 m can only add cooling. A city-wide +10 pp therefore cools a street
at least as much as +10 pp within 300 m of it.

The one-sided 95% bound of the street design (estimate + 1.645 SE) is that floor. In the lab it lay
above the true city-wide change at street points in:

- 100% of campaigns in every world with an effect;
- 92% of campaigns with no effect.

It is tight where cooling is local (it reaches 89–97% of the truth in the local worlds) and loose
where cooling spreads (22% in the advected world). The real-data report gives it as "street air cools
by at least … °F".

The assumption is physical, not statistical. A tree shades and transpires; neither warms the air a few
hundred metres away. Sheltering from the wind could in principle, which is why the floor is stated with
its assumption.

### What was tried for the kilometre value, and why none works on one city

| Design | Idea | Lab result |
|---|---|---|
| 1 km rings in the street differences | Read canopy out to 1 km | Unbiased in four worlds; collapses to 12% of the truth with a canopy-tracking 2 km factor |
| Physics-constrained extrapolation | Fit the advection–diffusion kernel to the near-field shape (100 m vs 300 m, upwind vs downwind) and extrapolate | Precise only when cooling is local. With wide kernels, 95% intervals ran from −10 to +7 °F, and a no-effect world returned −9.5 °F |
| Same streets, different winds (shipped) | Compare each street with itself across runs. The wind changes which canopy is upwind while nothing about the street changes. Run-specific smooth controls absorb a confounder whose effect changes through the day. Inference rotates every run's wind together | Never a false alarm, in any of 6 no-advection worlds (shore cooling carried by the wind included). Finds real advection in 8% of one-day campaigns; four days do not help, because the run-specific controls that remove the time-varying confounder also absorb most of the advected signal |
| Before/after a planting programme | Difference-in-differences on canopy change (8 zones of 500 m, +30 pp, 3 runs before and 3 after) | The 1 km and 3 km readings scatter by ±0.3 to ±1.6 °F |

They all fail for the same reason: a city holds only a handful of independent kilometre-sized
patches, and weather and confounding vary at exactly that scale. More of the same campaign does not
fix it.

Two things would:

- **Many cities' campaigns pooled.** CAPA has run Heat Watch in dozens of cities, and SPARC's open-data
  features can build the layers for any of them. Each city adds independent kilometre-sized patches,
  and the floor and the wind-shift test pool directly.
- **A very large canopy change.** A change spanning several kilometres and many neighbourhoods,
  measured before and after on the same routes, would also identify it.

The city-wide numbers the scenarios report rest on the model and its physics prior, not on identified
variation. The uncertainty report already flags canopy as **Not established**. The floor is the part a
planner can lean on.

## Running it on the real traverses

The Providence traverses are on OSF: <https://osf.io/wu9v7/>. Download the project's files ("Download
as zip" on the Files tab), unzip them, and point SPARC at the folder as it is. Zips inside it are
unpacked; rasters, boundary polygons and summary tables are skipped and listed.

In PowerShell, from the repository with the environment active (`.venv\Scripts\Activate.ps1`):

```powershell
# 1. What is in the download: files read or skipped, the runs of the day, overlap with the grid
python -m sparc.core.identify inspect  -p configs/core_providence.yml --traverses "C:\Users\<you>\Downloads\wu9v7-osfstorage-archive"

# 2. Validate the designs on Providence's layout (3–10 minutes on 4 cores, once)
python -m sparc.core.identify lab      -p configs/core_providence.yml

# 3. The answer: every run, the floor and the kilometre test
python -m sparc.core.identify estimate -p configs/core_providence.yml --traverses "C:\Users\<you>\Downloads\wu9v7-osfstorage-archive"

# 4. The kilometre test's power under the campaign's real winds (step 3 prints them)
python -m sparc.core.identify kmlab    -p configs/core_providence.yml --winds 240:0.8 170:7.7 190:5.0 200:2.1

# 5. Show it on the results page ("From traverses" column)
python -m sparc.core.results_page output/core/providence/providence_uhi configs/core_providence.yml
```

- **Step 3: winds.** Step 3 reads each run's wind from the Providence airport station in the forcing
  file (NOAA Global Hourly, so it needs internet). Without internet, give the winds yourself, e.g.
  `--wind morning=240:0.8 midday=170:7.7 evening=190:5.0 night=200:2.1` (KPVD for 2020-07-29/30).
- **Step 3: timestamps.** If the timestamps carry a time zone, add `--timezone America/New_York`.
  If `inspect` shows runs at 10–11 h, 19–20 h and 23–24 h instead of 6–7, 15–16 and 19–20, the times
  are UTC without saying so: add `--utc` to `inspect` and `estimate`.
- **Outputs.** Everything is written to `output/core/providence/identify/`:
  `identify_estimate.md` is the readable answer, `identify_estimate.json` the full record.

What the estimate reports:

- **The headline run** (the afternoon by default; `--window morning|evening|night|all`): the effect
  of +10 pp canopy within 100 m, 300 m and 1 km of a street point, each with its 95% interval and its
  lab verdict.
- **The city-wide floor.**
- **By time of day**: the block-scale effect in each run.
- **The kilometre scale**: the wind-shift test across runs, with each run's wind and the rotation
  p-value.

The Providence download holds four runs as shapefiles: `am_trav` (06–07 EDT), `af_trav` (15–16),
`pm_trav` (19–20) and `mi_trav` (00–01 the next night). Each run has six sensors (CAPA1061 … CAPA1085)
and about 15,000 one-second readings. A shapefile is several files with the same name: `.shp` (the
positions), `.dbf` (the readings), `.shx` and `.prj`. Keep them together; a `.dbf` on its own has no
positions and is reported as such.

Reading the files:

- CSV, GeoJSON, GeoPackage, shapefile and zip archives are accepted, nested zips included. Every
  file is either read or listed as skipped with the reason.
- CAPA tables carry the local clock (`dttm_lc`) next to the GPS clock in UTC (`dttm_tc`, and `date` +
  `time` such as `290720` / `100001`). The local clock is used.
- The sensor named in `file` (e.g. `CAPA1061_290720_095119.093_0.csv`) is the vehicle.
- Columns are detected by name:
  - time: `datetime`, `timestamp`, `date` + `time`, ...;
  - position: `lat`/`lon`, or the geometry of a point layer in any coordinate system, reprojected;
  - temperature: `T_F`, `temp_f`, `TempF`, ... in °F, or the °C equivalents, converted;
  - vehicle: `car`, `vehicle`, `route`, `sensor`, ..., otherwise the source file.
- A run is a cluster of readings with no gap over 1.5 h, named by its local time. An overnight run
  becomes its own "night" run, not part of the morning.
- Points are projected into the model's frame (EPSG:3438 in metres) and snapped to the nearest cell.
- A pass breaks at gaps over 30 s or jumps over 200 m.
- Samples of one pass in the same cell are averaged: a 1 Hz logger takes about three per 30 m cell.

To try the whole path first:

```
python -m sparc.core.identify simulate -p configs/core_providence.yml --world additive --out sim.csv
python -m sparc.core.identify estimate -p configs/core_providence.yml --traverses sim.csv --window all
```

`simulate` writes a campaign in the CAPA layout and prints the true effect. The round trip is tested
in `tests/core/test_identify.py`.

## Real traverses (Providence, 2020-07-29)

The designs have now run on CAPA's published traverses (OSF `wu9v7`, `All Data_Heat Watch RI_100620.zip`,
md5 `4a0d015a…eccfd6`). Run by the `sparc-steward` agent on 2026-10-23. Outputs (not committed):
`output/core/providence/identify/identify_{lab,kmlab,estimate,estimate_midday}.{json,md}` and
`identify_lab_routes_{morning,midday,evening,night}.{json,md}`.

**In short.** Averaged over the day, no block-scale canopy effect is detected (canopy within 100 m of a
street point): −0.07 to +0.02 °F per +10 pp, pooled over the four runs. The afternoon run is too noisy to say
much (−0.13 to +0.17 °F), and on its own routes the lab trusts only its direction. **The evening run measures
cooling:** −0.08 °F within 100 m (−0.11 to −0.04) and −0.17 °F within 300 m (−0.26 to −0.07). The lab replayed
the evening's real routes and noise, and rates both readings Trustworthy there. Wider readings are negative in
three of four runs. They give a validated city-wide floor of 0.05 °F of cooling per +10 pp. Under the day's
real winds the kilometre test has no power. All of these are findings: the afternoon and day-averaged nulls
are measured, not missing.

### QA: what was read

| Check | Result |
|---|---|
| Files | 4 point shapefiles read (`am_trav`, `af_trav`, `pm_trav`, `mi_trav`, nested 3 zips deep); 24 raster files skipped with reasons. No reader change was needed. |
| Clock | `dttm_lc` (local) used; `dttm_tc` and `date`/`time` are the UTC GPS clock. Runs land at 06:00–07:00, 15:00–16:00, 19:00–20:00 and 00:00–00:59 (07-30), so no `--utc`. |
| Vehicles | 6 sensors per run (CAPA1061, 1079, 1080, 1082, 1083, 1085), taken from the `file` column |
| Readings | 62,084 at 1 Hz: morning 14,999, afternoon 15,153, evening 15,603, night 16,329 |
| On the project grid | 44% (`share_on_grid` 0.42–0.44 per run). The campaign covers more of Rhode Island than the Brown study area; only 3 of the 6 vehicles drive inside it. |
| Per run on the grid | 2,069–2,481 cell visits on 13–29 passes, 2,056–2,466 cells (4% of the grid), median step 8 m. The street design uses 15,800–19,400 within-pass pairs in 44–52 one-kilometre clusters (lab campaigns: about 43,000 pairs in 63 clusters). |
| Winds (KPVD, NOAA Global Hourly) | morning from 240° at 0.8 m/s; afternoon 170° at 7.7; evening 190° at 5.0; night 200° at 2.1 |

### The lab on this layout

`lab --workers 4` (12 replicates × 6 worlds) reproduces the documented table above value for value. Street
differences within 100 m: **Trustworthy** (null −0.015 °F, 8% false positives; bias ≤ 0.017 °F in all five
effect worlds). Within 300 m: **Direction only**. Within 1 km, the wind signature, the levels regressions
and the map designs: **Not trustworthy**. The 100 m and 300 m floors are both validated (they hold in ≥ 92% of
campaigns in every world). The gate condition "street_100 trustworthy" is met.

### Block scale, by run (+10 pp canopy within 100 m of a street point)

| Run | Within 100 m (95% CI) | SE | Within 300 m (95% CI) | One-sided floor on city-wide cooling (from 300 m) |
|---|---|---|---|---|
| Morning 06–07 | −0.03 °F (−0.09 to +0.04) | 0.033 | **−0.24 °F (−0.36 to −0.11)** | at least 0.13 °F |
| **Afternoon 15–16** (the run the lab simulates) | +0.02 °F (−0.13 to +0.17) | 0.077 | +0.09 °F (−0.14 to +0.31) | none |
| Evening 19–20 | **−0.08 °F (−0.11 to −0.04)** | 0.019 | **−0.17 °F (−0.26 to −0.07)** | at least 0.09 °F |
| Night 00–01 | −0.02 °F (−0.04 to +0.01) | 0.014 | **−0.09 °F (−0.14 to −0.03)** | at least 0.04 °F |
| **All four runs pooled** (`--window all`) | **−0.03 °F (−0.07 to +0.02)** | 0.024 | **−0.11 °F (−0.17 to −0.04)** | **at least 0.05 °F** |
| Lab verdict | Trustworthy | | Direction only | Validated floor |

- **Block scale: a null.** The pooled interval includes zero. It excludes block-scale cooling stronger than
  0.07 °F per +10 pp averaged over the day. The afternoon interval excludes cooling stronger than 0.13 °F.
  The lab's local worlds plant −0.17 to −0.28 °F, so an afternoon effect that large is ruled out.
- **The afternoon run is imprecise.** Its SE (0.077 °F) is more than twice the lab's (0.034 °F), though it
  has a similar number of pairs and clusters. Street air is noisier at 15–16 h than the lab's sensor noise
  (sd 0.3 °F) assumes. Turbulent mixing under a 7.7 m/s afternoon breeze is one plausible cause. The calmer runs
  are the precise ones: night SE 0.014, evening 0.019.
- **Time of day.** Evening and night differ by −0.06 °F (z ≈ 2.6). Only the evening run detects cooling
  within 100 m. The lab validated the design on an afternoon hour; the other runs use the same design and
  sampling, but their noise was not simulated.
- **300 m: the sign.** The 300 m reading is negative and excludes zero in morning, evening, night and
  pooled. The lab trusts its sign, not its size. Read this as "canopy around a street cools it, more in
  the morning and evening". Since the 100 m reading is near zero, most of that cooling would come from
  canopy 100–300 m away. A kilometre-scale confounder that tracks canopy could produce the same pattern; the
  lab's `confounded` world keeps the 300 m sign right, but not every confounder shape was simulated.
- **The 1 km reading** (−0.26 °F, −0.47 to −0.04, pooled) is **Not trustworthy**: with a canopy-tracking
  2 km factor the lab recovers only 12% of the truth. It is reported, not used.

### The lab on the real routes

**Why.** The idealised lab drives 10 vehicles on a regular street grid with white sensor noise. The real
campaign is different on three counts:

- only 3 vehicles drive on the grid, on irregular routes;
- the afternoon noise is larger;
- real along-street variation is autocorrelated. The variance of temperature differences along a pass grows
  roughly linearly with lag: at 30 m it is 0.006 °F² at night and 0.06 °F² in the afternoon; at 300 m, 0.06
  and 0.82 °F².

**How.** `lab --routes <raw> --window <run>` replays one real run: its own cells, passes, vehicles and times.
It plants the same six worlds and replaces white noise with the run's own along-pass deviations from each
pass mean. Each pass's deviations are circularly shifted by a random lag. That keeps their roughness and
autocorrelation but breaks their alignment with the streets and their canopy.

The test is conservative on noise. The deviations still carry the real streets' own temperature pattern,
and the planted world adds its own on top. The lab's standard errors therefore run above the real run's:
0.104 against 0.077 °F in the afternoon, 0.059 against 0.019 °F in the evening.

**Replicates.** I first ran 12 replicates per world, as the idealised lab does. On the afternoon routes,
street_100 then flagged an effect in 2 of 12 no-effect campaigns (17%), which rated it **Not trustworthy**.
Two in twelve is too few to judge, so every run was repeated with 36 replicates. The verdict rules were not
changed. The verdicts below are from 36 replicates per world (216 campaigns per run, about 3 minutes on 4 cores).

| Run (real routes) | Within 100 m | Within 300 m | Floor (from 300 m) | Null: 100 m mean, false positives, sd / mean SE |
|---|---|---|---|---|
| Morning | **Trustworthy** | Direction only | valid | +0.010 °F, 3%, 0.067 / 0.073 |
| Afternoon | Direction only (misses `physics` −0.035 and `coarse_scale` +0.035, just past the 0.03 °F tolerance) | Conservative | valid | −0.010 °F, 11%, 0.105 / 0.104 |
| Evening | **Trustworthy** | **Trustworthy** | valid | −0.004 °F, 3%, 0.058 / 0.059 |
| Night | **Trustworthy** | Conservative | valid | +0.008 °F, 3%, 0.042 / 0.049 |

- **The block-scale design holds up on the campaign as driven,** except the afternoon run. That run is
  clean with no effect, but too noisy to size the effect in the two kilometre-scale worlds. The afternoon
  reading (+0.02 °F) stays a null, rated Direction only.
- **The standard errors are calibrated.** With no effect, the spread of estimates matches the mean reported
  SE in every run (ratio 0.86–1.01). Clustering on 1 km blocks is enough even with autocorrelated real noise.
- **The evening's 300 m reading is sized,** not only signed: −0.17 °F (−0.26 to −0.07), Trustworthy on the
  evening routes.
- **The evening's 1 km reading** (−0.57 °F, −0.96 to −0.17) rates **Conservative** on the evening routes.
  It understates the cooling in the `confounded` world (27% of it) and recovers it in the other four. That
  is weaker evidence than for 100 m and 300 m. It rests on the confounder shapes the lab simulates.
- **The floors** (from 300 m) are valid on every run's routes. The pooled `--window all` estimate mixes runs
  and is not replayed by this lab.
- **What the evening result means.** At 19–20 h, a street with 10 pp more canopy within 100 m is about
  0.08 °F cooler, and 0.17 °F with 10 pp more within 300 m. The afternoon cannot confirm or rule out a
  similar effect: its interval runs from −0.13 to +0.17 °F.

### City-wide floor

From the validated 300 m reading (estimate + 1.645 SE, assuming canopy never warms the air at a distance), a
city-wide +10 pp cools street air by **at least 0.05 °F**, pooled over the day. By run: at least 0.13 °F in
the morning, 0.09 °F in the evening and 0.04 °F at night. The afternoon run sets no floor above zero. In the
lab this bound held in ≥ 92% of campaigns in every world. It is tight only where cooling is local.

### Kilometre scale: same streets, different winds

- **Estimate:** +0.08 °F upwind − downwind per +10 pp (95% CI −0.01 to +0.17). Rotation p = 0.56;
  one-sided for cooling p = 0.75. 2,480 cells across the four runs enter the comparison.
- **Not detected.** The sign is the opposite of advected cooling.
- **Underpowered under the real winds.** `kmlab --winds 240:0.8 170:7.7 190:5.0 200:2.1` finds:
  - no false alarm in any of 6 no-advection worlds;
  - advected cooling detected in 0% of `physics` campaigns.
- **Why.** The day's winds span only 70° (170–240°), and the morning is nearly calm. The doc's sea-breeze
  default spans 120° and gave 8% power. The real noise is also higher than the lab's: the rotation null sd
  is 0.10 °F on the real data against about 0.045 °F in the lab.
- **Verdict:** this campaign cannot test advection.

### What this does and does not change

- **The map's canopy effect is not confirmed at block scale.** The model's −0.19 ± 0.14 °F for +10 pp
  (city-wide, from the map) is not contradicted by the floor (0.05 °F). The traverses show no block-scale
  effect to support it.
- **Planners can lean on three things:**
  - the sign of the 300 m reading;
  - the 0.05 °F city-wide floor;
  - the evening's sized block-scale cooling: 0.08 °F per +10 pp within 100 m, 0.17 °F within 300 m.
- **Still unidentified:** the city-wide size. It needs pooled campaigns (Phase 3).
- **Verdicts on the campaign as driven.** The idealised lab's verdicts were checked on each run's real
  routes and noise ([above](#the-lab-on-the-real-routes)). They hold, except that the afternoon block-scale
  reading drops to Direction only.

Reproduce (data from `https://files.osf.io/v1/resources/wu9v7/providers/osfstorage/?zip=`, which streams
the project's files as one zip; take `All Data_Heat Watch RI_100620.zip` out of it):

```
python -m sparc.core.identify inspect  -p configs/core_providence.yml --traverses <raw>
python -m sparc.core.identify lab      -p configs/core_providence.yml --workers 4          # ~3 min on 4 cores
python -m sparc.core.identify kmlab    -p configs/core_providence.yml --workers 4 --winds 240:0.8 170:7.7 190:5.0 200:2.1
python -m sparc.core.identify estimate -p configs/core_providence.yml --traverses <raw> --window all
python -m sparc.core.identify estimate -p configs/core_providence.yml --traverses <raw> --name identify_estimate_midday
python -m sparc.core.identify lab      -p configs/core_providence.yml --workers 4 --reps 36 --routes <raw> --window evening   # and morning, midday, night
```

`<raw>` can also be fetched directly: `python -c "from sparc.data.collect.capa import fetch_capa_raw; fetch_capa_raw('wu9v7', '<raw>')"`.

## Limits

- **The worlds are SPARC's own generators.** A real effect with a shape none of them has could behave
  differently. Several shapes are covered (own cell, ~150 m, advected ~1 km, smooth ~1 km, with and
  without a kilometre-scale confounder), and only the within-100 m design passes all of them.
- **Simulated campaigns are idealised.** Routes are a regular street grid and drift is linear in time.
  Real campaigns have irregular routes, traffic stops and possibly non-linear drift. Short-lag
  differences are insensitive to the last of these. `lab --routes` replays a real run's routes and noise
  instead.
- **The estimand is the effect at street points.** City-wide extrapolation from streets is an
  assumption: streets have less canopy than the rest of the city.
- **Time of day.** Canopy's effect differs between morning, afternoon and evening. The estimate
  reports each run; the lab validates the afternoon's design.
- **Street-scale confounders do not cancel.** Something that changes along a street together with
  canopy within 100 m would bias the estimate. Impervious surface, albedo and terrain are differenced
  as controls. Traffic and building shade are not observed.
