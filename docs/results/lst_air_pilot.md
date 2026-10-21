# Does land surface temperature tell us more about street air than land cover does?

**Question (if → then).** If Landsat land surface temperature (LST) carries information about the
surface–air coupling that land cover does not, then adding LST to a land-cover model makes its
predictions of street-level air temperature better **in cities the model has never seen**.

**Why it matters.** If it does, LST can stand in for air temperature where nobody has driven a
traverse, and the physics can use it as a calibration of the surface heat source. If it does not, land cover and
the physics carry the signal, and LST is a surface diagnostic only. The literature points both ways:
surface and air heat islands differ several-fold (Venter, Chakraborty & Lee 2021: 1.45 °C surface against
0.26 °C air, globally), and LST explains little of daily-maximum air temperature on its own (Venter et al.
2020), yet machine-learned air-temperature maps lean on it (Ho et al. 2014; Shandas et al. 2019, CAPA's
own method).

**Status.** Pre-registered (2026-10-20), run 2026-10-21. **Answer: no.** In the afternoon, LST adds no
detectable skill beyond land cover in held-out cities ([Result](#result)).

## Data

**Measurements.** CAPA Heat Watch mobile traverses (OSF), one-second car-mounted readings, gridded to
30 m cells per run. Each run's target is the drift-adjusted cell mean minus the run's median over its
cells (dTa, °C).

**Cities used.** 31 cities whose readings carry a clock (2019–2023):

> Abingdon, Albuquerque, Arlington, Asheville, Atlanta, Austin, Boise, Boston, Burlington, Charleston,
> Farmville, Fort Lauderdale, Houston, Las Cruces, Longmont, Los Angeles, Lynchburg, New Orleans,
> Petersburg, Providence, Raleigh, Richmond, Sacramento, Salem, Salt Lake City, San Bernardino,
> San Diego, Seattle, Sedona, Victorville, Virginia Beach.

**Cities not used.**
- **No clock or sensor id.** CAPA's 2022–24 releases for Boulder, Brockton, Brooklyn, Charlotte, Chicago,
  Columbia MO, Columbia SC, Jacksonville, Kansas City KS, Las Vegas, Milwaukee, Oklahoma City, Omaha and
  Philadelphia have neither, so no run can be formed and no drift can be removed.
- **No files.** About 15 catalog entries have no files on OSF, and Portland and Wilmington DE are private.

**Predictors.**
- **Land cover:** ESA WorldCover 2021 tree and built fractions, Sentinel-2 NDVI and albedo, Copernicus
  DEM and distance to water, plus their 100/300/1000 m focal means.
- **LST:** Landsat 8/9 Collection 2 surface temperature from the scene nearest the campaign day
  (±16 days), cloud-masked.
- **Forcing:** ERA5 (for the physics only).

### Data checks

Every city was screened by script, then reviewed by three independent reviewers and a skeptic per
reviewer who tried to refute each proposed exclusion (2026-10-20). Faults found and their treatment
(the first four were fixed before the review; the rest are implemented before the full run):

| Fault | Cities | Treatment |
|---|---|---|
| CAPA's −9999 "removed outlier" read as a temperature | 9 (VA, Albuquerque) | fixed in the reader |
| shapefile clock named `datetim` | Albuquerque | fixed in the reader |
| study area across WorldCover/DEM tile edges | Abingdon, Boston, Burlington | tiles mosaicked |
| stationary-sensor and other-city air-quality tables | Columbia SC, Las Vegas, Oklahoma City | skipped |
| row-index `ID` hides the sensor id in `file` | Asheville, Longmont, Salt Lake City, Sedona | fixed in the reader |
| one sensor's route rained on (RH 40→68%, −5.8 °C) | Asheville afternoon | that sensor's readings after 15:05 dropped |
| sensor warm-up (flat 24 °C, then 18 °C) | Los Angeles morning | that sensor's readings before 06:16 dropped |
| rain over 4 of 7 routes (51% of readings at RH ≥ 99%) | Fort Lauderdale 09-18 morning | run excluded |
| bicycle on an off-street parkway mixed into car files | Sacramento | non-vehicle sensors dropped |
| vehicle offsets from a single short link far apart in time | Los Angeles am, Richmond pm, Fort Lauderdale 09-18 af, New Orleans 08-18 af | offsets only from co-located visits ≤ 10 min apart |
| drift aliased from route composition | Virginia Beach af | drift line fitted within vehicles |
| one car drives the whole run (drift not identifiable) | Lynchburg am/af, Fort Lauderdale make-up runs | excluded (sensitivity: included) |
| walking campaign, not car-mounted | Farmville, Abingdon | excluded (sensitivity: included) |
| LST cool-biased within ~150 m of cloud masks | Charleston, Virginia Beach and others | cloud mask buffered 150 m |
| one overpass split across two Landsat rows | Albuquerque, Atlanta | rows of one overpass mosaicked |
| no Landsat scene ≥ 50% clear | Abingdon | out of the LST test (recorded) |

Real physics was **kept**, not excluded: Los Angeles's 3.5 °C afternoon coast-to-inland contrast,
Virginia Beach's sea breeze, Boise's dawn inversion, desert surfaces whose LST runs against the air
(Las Cruces, San Diego mornings).

## Pre-registered analysis

**Models.**
- **N:** predicts no anomaly (zero after centring). This is the floor any model must clear.
- **A, land cover:** an additive spline ridge or gradient boosting, chosen by inner leave-city-out CV on
  the training cities. Splines are clamped outside the training range (no linear extrapolation).
- **B, A + LST:** A plus the LST anomaly and its 100/300 m focal means.
- **C, physics with LST:** SPARC's advection–diffusion–relaxation model with LST in the heat source,
  transferred to the held-out city by the median physical parameters of the training cities.
- **C0:** C without LST.

**Held-out unit.** A city.
- **Grouped:** catalog cities that are one campaign (same OSF node and date, grids within 50 km) or
  neighbours (grids within 10 km) are held out together.
- **Sensitivity:** the eight Virginia cities of 15 July 2021 as one group.

**Scoring.** Scale-free, so that no single city decides the result.
- Per held-out unit and run, skill SS = 1 − MSE/MSE_N (MSE_N is the target's variance on the scored
  cells), averaged over the unit's runs.
- The comparison is ΔSS = SS_B − SS_A per unit.
- Absolute ΔMSE (°C²) is reported as a secondary measure.

**Training weights.** Each training city weighs equally *relative to its own variance*: sample weight ∝
1/(n · var(dTa)), from training labels only. Sensitivity: weights ∝ 1/n.

**Centring.** Every predictor anomaly, the LST anomaly and the target are centred on the cells that are
scored (cells with every A and B predictor). Centring uses positions, never labels.

**The claim "LST adds information beyond land cover"** is made only if, in the **afternoon window
(15–16 h, primary)**, all of these hold:
1. the one-sided sign-flip test over held-out units gives p ≤ 0.025 after Holm's correction across the
   three windows (am, af, pm) and the two claims (B − A, C − A);
2. the 95% t-interval of mean ΔSS lies above 0;
3. mean ΔSS ≥ 0.05, i.e. B removes at least 5% of the target variance more than A does;
4. B beats N on average (mean SS_B > 0).

The physics claim "C beats A" uses the same rule with ΔSS = SS_C − SS_A. C − C0 is reported as what LST
adds inside the physics. The other windows (am, pm, and Providence's night run) are secondary.

**Runs scored.** Car-mounted runs with an identifiable drift, after the data checks above.

**Sensitivities (reported, never used for the claim).**
- Absolute ΔMSE.
- Weights ∝ 1/n.
- Virginia cities as one group.
- Including walking campaigns and single-car runs.
- LST split into a regional part (3 km focal) and a local residual: B with the local residual only.
- A with regional controls (3 km focal elevation and impervious).
- Continuous land cover in place of WorldCover classes, if obtainable (NLCD tree canopy and percent
  impervious).
- Physics at 30 m cells, and with a coherent parameter vector in place of per-parameter medians.
- Leave-one-unit-out: does the verdict survive dropping any single unit?

**Seen before registering.** A smoke run on 7 cities, afternoon only, under the earlier design
(absolute MSE, weights ∝ 1/n, no N):
- B − A: mean ΔMSE −0.60 °C² (95% CI −1.94 to +0.73). Of that, −3.82 of the summed −4.22 came from Los
  Angeles alone; B was better in 3 cities and worse in 4.
- C − A: −0.40 (CI −0.78 to −0.03), better in 6 of 7.

These findings shaped the scoring rule above, and the full run was stopped unread when the review
showed the earlier scoring was dominated by one city.

## Result

Full run 2026-10-21, under the rules above.
- **Data.** 31 cities built with every data fix, NLCD land cover added, about 2.3 h on 4 CPUs.
- **Units.** 27 held-out units in the afternoon window: Petersburg and Richmond are 19 km apart and are held out together.
- **Not scored.** Abingdon (no clear Landsat scene); walked and single-car runs (Farmville, Lynchburg am/af,
  Fort Lauderdale make-up runs).
- **Output.** `sparc core lstair evaluate`, written to `results.md` in the run folder.

**Answer: no.** In cities the models never saw, LST does not make afternoon street air temperature better
predicted than land cover alone.

| afternoon (primary) | mean ΔSS | 95% CI | Holm p | units better / worse | verdict |
|---|---|---|---|---|---|
| **B − A** (LST beyond land cover) | **+0.006** | −0.006 to +0.018 | 0.78 | 16 / 11 | **not shown** |
| C − A (physics vs land cover) | −0.083 | −0.135 to −0.031 | 1 | 7 / 20 | not shown (worse) |
| C − C0 (LST inside the physics) | +0.111 | +0.062 to +0.159 | 0.00025* | 22 / 5 | LST helps the physics |

\* unadjusted; C − C0 is outside the claim family.

- **Tight null.** Whatever LST adds lies below 2% of a city's afternoon anomaly variance, against the 5%
  the claim needed.
- **Robust.** No single unit changes the verdict: mean ΔSS stays between +0.004 and +0.008 when any one unit
  is dropped.
- **Absolute errors agree.** B − A is −0.027 °C² (−2%), CI −0.071 to +0.017.

**The other windows (secondary).**
- **Morning (06–07 h):** nothing. ΔSS −0.001, CI −0.004 to +0.003.
- **Evening (19–20 h):** a small, consistent gain. ΔSS +0.027, CI +0.008 to +0.045, Holm p 0.018, better in 22
  of 27 units, and robust to dropping any unit. It falls short of the 5% the claim needed, so it is not claimed.
  It fits the physics: evening air is warmed by heat the surface stored during the day, and LST at the
  10:30 overpass measures that surface.

**How well anything transfers.**
- **Land cover alone, in the afternoon:** it explains about 15% of a held-out city's street-level anomaly
  variance (mean SS_A 0.15).
- **Same city, held-out blocks:** about 37% (interpolation-prone).
- **Worse than predicting nothing:** Los Angeles and San Diego, where coast-to-inland marine air dominates and
  no predictor here carries it.

**The physics.** As transferred (per-parameter medians of the other cities' fits), C is worse than the
land-cover model in every window.
- **Afternoon:** SS 0.07 against 0.15.
- **Morning:** SS −0.87, worse than predicting nothing. Several fits switch the source off (amplitude ≤ 0),
  and the medians of fits that disagree do not transfer.
- **Inside the physics:** LST does help (C − C0). The fixed-form land-cover source misses what LST carries.
- **L at its bounds:** in the afternoon, 6 of 28 fits sit at the 60 m cell size and 6 at the ceiling.

So the cross-city weakness is the parameter transfer and the resolution, not the equation alone.

**What this means for SPARC.**
- **LST as an estimator:** a satellite LST map is not a stand-in for afternoon street air temperature in a
  new city. It adds nothing a land-cover model does not already know. To map a city's air, use the physics
  with land cover, and calibrate it on that city's traverses or stations where they exist.
- **LST in the physics:** LST is worth keeping inside the physics, as a surface-heat source or diagnostic,
  and as an evening predictor.
- **Next for the physics:** a coherent parameter transfer, finer cells, and a marine-air or sea-breeze term
  for coastal cities.

**Sensitivities.** Run on the primary window and reported in `sens_af/sensitivity.md`; results are added
below when they finish:
- weights ∝ 1/n;
- walked and single-car runs included;
- NLCD canopy and impervious in place of WorldCover;
- the medoid city's whole physics parameter vector.

Not run, to save compute:
- **LST local-only, and regional controls:** both guard against LST getting credit it does not deserve, and
  LST earned none here.
- **Physics at 30 m cells:** it costs four times the cells.
- **Virginia as one group:** it only merges units, which is conservative.

## Extension studies (pre-registered 2026-10-21, before any of them is run)

The primary test closed one question: LST is not a substitute for air temperature in a new city. These
studies ask what *does* make a held-out city's street-level heat map good.

**Shared rules.** Every study uses the primary analysis's machinery unchanged:
- the same 31 cities, data checks and held-out units (city; node+date link within 50 km);
- the same scoring: skill SS against N, ΔSS paired over units, one-sided sign-flip, 95% t-interval;
- the same claim thresholds: p ≤ 0.025 after Holm *within the study*, CI above 0, mean ΔSS ≥ 0.05, mean SS of
  the better model above 0;
- the afternoon window (15–16 h) is primary in every study; am and pm are secondary.

Each study is its own family, and none of them reopens the primary verdict.

### S1. Calibrating with local stations

**If** a city has a few point measurements of air temperature at the hour, **then** spreading their residuals
(regression kriging; Hengl, Heuvelink & Rossiter 2007) makes the held-out map better.

- **Base.** A (the best uncalibrated model of the primary test).
- **Residual field.** The station residuals (observed minus predicted anomaly, minus their mean) are
  interpolated by simple kriging. The exponential covariance's range and nugget come from the *training*
  cities' residual variograms (pooled per window), never from the held-out city.
- **Simulated stations (primary).** k ∈ {3, 10, 30} driven cells per run are taken as perfect stations, at
  least 1 km apart, in 20 random draws. Scoring is on the driven cells more than 500 m from every station,
  averaged over draws.
- **Comparison.** ΔSS = SS(A calibrated) − SS(A), with Holm over the three k.
- **Real stations (secondary).**
  - NOAA Global Hourly (ISD/ASOS) stations inside the grid, at the run's hours; at least 2 stations.
  - Scoring is on all driven cells.
  - Airport sites stand for open ground, not streets, so this tests the network a city already has.

### S2. Marine and lake air

**If** sea or lake breezes cool the neighbourhoods near large water, **then** distance to large water improves
the transfer, above all in coastal cities.

- **Predictor.** `coast_km`, the distance to the nearest water body of at least 100 km², from WorldCover water
  over the grid plus a 30 km margin. Its log goes into A and B as a within-run anomaly.
- **Comparison.** A_coast − A (primary).
- **Secondary.** C_marine − C: the physics with a marine sink, q −= c_m·exp(−d/λ_m), with λ_m fitted in 1–30 km
  and c_m ≥ 0 transferred like the other parameters.

### S3. JEPA representation

**If** a self-supervised representation of the predictor maps captures neighbourhood context the hand-made
focal features miss, **then** adding it improves the transfer.

- **Representation.** SPARC's JEPA components (EMA trunk and latent predictor with spatial patch masking)
  are pretrained on the predictor rasters of all 31 cities: land cover, NDVI, albedo, elevation, water
  distance. Neither LST nor labels are used, so a held-out city's predictors may be seen, as they would be in
  any new city.
- **Embedding.** It is reduced to 8 principal components over all cities' cells and added to A as anomalies.
- **Comparisons.** A_jepa − A (primary) and B_jepa − A_jepa (LST beyond JEPA land cover), with Holm over the
  two.

### S4. Physics transfer

**If** the physics fails across cities because per-parameter medians mix incompatible fits and L is
grid-limited, **then** these help C:
- a coherent parameter vector (the medoid city's fit);
- a 30 m physics cell (L floor 30 m, ceiling 5 km).

C_medoid − C and C_30 − C are compared, with Holm over the two.

### S5. Afternoon LST (when an Earthdata token is available)

**If** the 10:30 Landsat overpass is too early to reflect the afternoon surface–air coupling, **then** ECOSTRESS
LST from 13–17 h local (±16 days, ≥ 50% clear) adds what Landsat did not.

The comparison is B_eco − A, on the cities that have such a scene (at least 8 units, or the study is not
testable).

## References

- Ho, H.C., Knudby, A., Sirovyak, P., Xu, Y., Hodul, M. & Henderson, S.B. (2014). Mapping maximum urban air
  temperature on hot summer days. *Remote Sens. Environ.* 154, 38–45.
- Hengl, T., Heuvelink, G.B.M. & Rossiter, D.G. (2007). About regression-kriging: from equations to case
  studies. *Comput. Geosci.* 33, 1301–1315.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scand. J. Stat.* 6, 65–70.
- Murphy, A.H. (1988). Skill scores based on the mean square error and their relationships to the
  correlation coefficient. *Mon. Weather Rev.* 116, 2417–2424.
- Roberts, D.R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical,
  or phylogenetic structure. *Ecography* 40, 913–929.
- Shandas, V., Voelkel, J., Williams, J. & Hoffman, J. (2019). Integrating satellite and ground
  measurements for predicting locations of extreme urban heat. *Climate* 7(1), 5.
- Venter, Z.S., Brousse, O., Esau, I. & Meier, F. (2020). Hyperlocal mapping of urban air temperature
  using remote sensing and crowdsourced weather data. *Remote Sens. Environ.* 242, 111791.
- Venter, Z.S., Chakraborty, T. & Lee, X. (2021). Crowdsourced air temperatures contrast satellite
  measures of the urban heat island and its mechanisms. *Sci. Adv.* 7(22), eabb9569.
