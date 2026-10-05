# Isolating the causal effect of canopy (Providence)

**Question.** How much does tree canopy cool the afternoon air, as a cause, not as a correlate of
everything else that comes with leafy streets?

**Answer, in short.**

| Canopy raised by +10 pp... | Can one heat-watch campaign isolate the effect? | How |
|---|---|---|
| within 100 m of a point (about a city block) | **Yes.** The design recovers the true effect in every simulated world (bias ≤ 0.02 °F) and stays at zero when canopy does nothing. | Street differences on the raw traverse readings |
| within 300 m | **Direction only.** The sign is reliable; the size depends on how far the cooling of a tree spreads. | Same design |
| within 1 km, or city-wide | **No.** A kilometre-scale canopy effect and a kilometre-scale confounder that tracks canopy leave the same traces in one campaign. | Needs repeated campaigns (different winds, before/after canopy change) or many cities |
| anything, from the interpolated map | **No.** The design that works on the measurements reports a canopy effect in 58% of worlds where there is none when run on the map. | — |

So canopy's causal effect *can* be isolated at the scale of a block, but only from the vehicle
traverses, not from the gridded temperature map. Providence's raw traverses (CAPA Heat Watch, OSF
project `tdsy7`) could not be downloaded from the development environment. The method is built,
validated on Providence's own layout, and ready to run on them ([below](#running-it-on-the-real-traverses)).

Code: `sparc/core/identify/`. Lab output: `output/core/providence/identify/identify_lab.{json,md}`.
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

No spatial-contrast design on one campaign can separate the two. The city-wide numbers the scenarios
report therefore rest on the model and its physics prior, not on identified variation. The
uncertainty report already flags canopy as **Not established**.

What would identify the kilometre scale:

1. **Repeat campaigns under different winds.** The same street has different upwind canopy on
   different days, while its confounders stay put. Street fixed effects then identify the advected
   part.
2. **Before/after canopy change.** Campaigns years apart around plantings or tree losses give a
   difference-in-differences on canopy change.
3. **Many cities.** Pooling the wind signature over CAPA's campaigns (dozens of cities) gives it
   power that one campaign lacks.

## Running it on the real traverses

The raw traverses are on OSF, next to the map: <https://osf.io/tdsy7/files>. They are archives of
CSVs or shapefiles for the morning, afternoon and evening runs.

1. Download the afternoon archive (or all of them) into a folder, for example `data/capa/providence/`.
2. In PowerShell (Windows) or bash, from the repository:

   ```
   python -m sparc.core.identify lab      -p configs/core_providence.yml --reps 12 --workers 4
   python -m sparc.core.identify estimate -p configs/core_providence.yml --traverses data/capa/providence --window midday --timezone America/New_York
   ```

   The lab takes about 10 minutes on 4 cores and writes the verdicts the estimate is judged by.
3. The answer is written to `output/core/providence/identify/identify_estimate.{json,md}`. It gives the
   effect within 100 m, 300 m and 1 km with 95% intervals, each with its lab verdict, plus the wind
   signature and QA: points on the grid, passes, vehicles, duration.
4. Rebuild the results page to show it in the "From traverses" column:

   ```
   python -m sparc.core.results_page output/core/providence/providence_uhi configs/core_providence.yml
   ```

Reading the files:

- CSV, GeoJSON, shapefile and zip archives are accepted, nested zips included.
- Columns are detected by name:
  - time: `datetime`, `timestamp`, ...;
  - position: `lat`/`lon`;
  - temperature: `T_F` (°F, or a °C name converted to °F);
  - vehicle: `car`, `vehicle`, `route`, `sensor`, ..., otherwise the source file.
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
- **One time window.** Canopy's effect differs between morning, afternoon and evening. Run `--window`
  for each.
- **Street-scale confounders do not cancel.** Something that changes along a street together with
  canopy within 100 m would bias the estimate. Impervious surface, albedo and terrain are differenced
  as controls. Traffic and building shade are not observed.
