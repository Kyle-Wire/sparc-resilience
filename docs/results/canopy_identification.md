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

# 2. Validate the designs on Providence's layout (about 10 minutes on 4 cores, once)
python -m sparc.core.identify lab      -p configs/core_providence.yml

# 3. The answer: every run, the floor and the kilometre test
python -m sparc.core.identify estimate -p configs/core_providence.yml --traverses "C:\Users\<you>\Downloads\wu9v7-osfstorage-archive"

# 4. The kilometre test's power under the campaign's real winds (step 3 prints them)
python -m sparc.core.identify kmlab    -p configs/core_providence.yml --winds 290:2.0 170:7.7 210:4.0

# 5. Show it on the results page ("From traverses" column)
python -m sparc.core.results_page output/core/providence/providence_uhi configs/core_providence.yml
```

- **Step 3: winds.** Step 3 reads each run's wind from the Providence airport station in the forcing
  file (NOAA Global Hourly, so it needs internet). Without internet, give the winds yourself, e.g.
  `--wind morning=290:2.0 midday=170:7.7 evening=210:4.0`.
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

Reading the files:

- CSV, GeoJSON, GeoPackage, shapefile and zip archives are accepted, nested zips included. Every
  file is either read or listed as skipped with the reason.
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

## Limits

- **The worlds are SPARC's own generators.** A real effect with a shape none of them has could behave
  differently. Several shapes are covered (own cell, ~150 m, advected ~1 km, smooth ~1 km, with and
  without a kilometre-scale confounder), and only the within-100 m design passes all of them.
- **Simulated campaigns are idealised.** Routes are a regular street grid and drift is linear in time.
  Real campaigns have irregular routes, traffic stops and possibly non-linear drift. Short-lag
  differences are insensitive to the last of these.
- **The estimand is the effect at street points.** City-wide extrapolation from streets is an
  assumption: streets have less canopy than the rest of the city.
- **Time of day.** Canopy's effect differs between morning, afternoon and evening. The estimate
  reports each run; the lab validates the afternoon's design.
- **Street-scale confounders do not cancel.** Something that changes along a street together with
  canopy within 100 m would bias the estimate. Impervious surface, albedo and terrain are differenced
  as controls. Traffic and building shade are not observed.
