# SPARC Air Quality: Plan

**Status:** proposal, nothing implemented · **Date:** 2026-10-13 · **Scope:** neighbourhood-scale NO₂ and PM2.5 around facilities, with data centres as the first application · **Data:** public only (ground monitors and satellites)

> Builds on [`CORE_ROADMAP.md`](CORE_ROADMAP.md). The core pipeline (`sparc/core/`) is the starting point; the legacy `air_quality` template (`sparc/templates/air_quality/`) is for the old `sparc run` pipeline and is not reused.

---

## TL;DR

- **Can SPARC give trustworthy air-quality results?** Yes, after four structural changes. The biggest is **time**: the core models one snapshot on a filled raster, while air quality is a few sensors measured every hour.
- **Can it show where pollution comes from?** For **local, primary** pollution (NO₂, diesel PM, black carbon near a road or facility): yes, as a per-source contribution map checked against a model-free wind-direction analysis. For **regional or secondary PM2.5** (most of a city's PM2.5): no, and the tool should say so.
- **Can it emulate emission changes?** Yes. Once emissions enter the physics model as a source term, concentration is linear in each source's emissions (NO₂ chemistry aside), so "cut this facility by 40%", "Tier 4 engines instead of Tier 2" or "add a new data centre with these permit limits" are exact, cheap re-runs with uncertainty.
- **Data centres:** the signal is real but uneven.
  - **Detectable:** continuous on-site generation, such as the gas turbines at xAI's Memphis site, which is permitted for up to 87 tons of NOx a year from 15 turbines ([TechCrunch](https://techcrunch.com/2025/07/03/xai-gets-permits-for-15-natural-gas-generators-at-memphis-data-center)).
  - **Hard to detect:** backup diesel generators, which run in short testing windows and grid emergencies. Their annual average at one site is usually below what sparse sensors can see. The honest output there is a **minimum detectable effect** and an emissions-based estimate, not a claim that nothing is happening.
- **Public data only, from two kinds of observation:**
  - **Ground monitors:** EPA regulatory monitors and public low-cost sensors (PurpleAir). Accurate at a point, hourly, but sparse.
  - **Satellites:** NASA TEMPO (hourly daytime NO₂ over North America, public since May 2024), TROPOMI (daily NO₂ since 2018), MAIAC aerosol optical depth (daily, 1 km, a proxy for PM2.5), plus Sentinel-2 and Landsat imagery to date construction. They cover everywhere, but they measure the whole air column at kilometre scale, not what people breathe.
  - **Fused:** the satellites set the spatial pattern and the pre/post timeline, and the monitors pin down surface concentrations. §4.11 describes how.
- **Trust comes from two things this repo already does well:**
  - **Pipeline validation:** before any real result is reported, the pipeline must recover planted effects, and find nothing where nothing was planted, on this site's real sensor layout, satellite footprints and weather (§4.7).
  - **Pre-registration:** freezing the analysis before looking at post-construction data. For an environmental tool, this decides whether opponents can dismiss the results.

---

## 1. Questions the tool answers

| Question | Answer type | Feasible? |
|---|---|---|
| What are concentrations across the neighbourhood, hour by hour, with uncertainty? | Map ± interval | Yes |
| How much does facility *F* add at each home? | Per-source contribution map | Yes for NO₂, diesel PM and black carbon near the source |
| Did concentrations change after *F* started operating? | Pre/post causal estimate with CI, or a minimum detectable effect | Yes, if public sensors near *F*, or satellite pixels over it, cover both periods |
| What if *F*'s emissions change (fewer hours, cleaner engines, more turbines)? | Scenario map ± interval, residents above thresholds, health impact | Yes |
| What would a *proposed* data centre add? | Scenario from permit-level emissions | Yes, as a forward estimate, labelled as such |
| What share of the city's PM2.5 comes from *F*? | Share of local increment only | Partly: regional and secondary PM2.5 is out of scope |
| What about the power plants feeding *F*? | Regional emission change | Separate module (EPA AVERT), not the neighbourhood model |

---

## 2. Data centre emissions: what to look for

| Mode | Pollutants | Timing | Who can see it |
|---|---|---|---|
| **Construction** | Diesel PM, NOx, dust (PM10) | Months to years, working hours | Nearby PM sensors; truck routes |
| **Backup diesel generator testing** | NOx, diesel PM, CO | Short scheduled runs, usually daytime | Only sensors downwind within ~1–2 km during the run; hourly data dilutes it |
| **Generators during grid emergencies** | Same, many units at once | Rare, hot days, coincides with high ozone | Regional monitors if many sites run together |
| **On-site turbines / prime power** | NOx, formaldehyde, CO | Continuous | Nearby NO₂ sensors; TEMPO and TROPOMI for large sites |
| **Traffic and diesel deliveries** | NO₂, black carbon | Daily | Roadside sensors |
| **Upstream power plants** | NOx, SO₂, PM2.5 | Continuous | Regional; handled by AVERT, not this model |
| **Heat** (bonus) | Air temperature | Continuous | The existing UHI pipeline already models canopy and impervious changes |

Published estimates show how much assumptions matter. For Northern Virginia, estimated premature deaths range from about 3 to 177 a year, depending mostly on whether generators emit 10% or 100% of their permitted limits ([arXiv 2412.06288](https://arxiv.org/pdf/2412.06288), [VCU](https://www.news.vcu.edu/article/northern-virginia-data-center-air-pollution-rivals-power-plant-emissions), [Sierra Club](https://www.sierraclub.org/coal/secret-coal-plant-dc-suburbs)). **That gap is SPARC's opening:** measurements calibrate how much of the permitted emissions actually reach people.

---

## 3. Data inventory

**Public data only.** Every input below is free to download, or is a public record. Anything that has to be requested, rather than downloaded, is marked as such. Each source gets a fetcher in `sparc/core/opendata.py` (which already fetches WorldCover, HRSL and Sentinel-2) or in a new `sparc/core/satellite.py`, so a run can be rebuilt from scratch by anyone.

**Ground monitors**

| Need | Public source | Notes |
|---|---|---|
| **Regulatory monitors** | EPA AQS (archive) / AirNow (recent), hourly NO₂, PM2.5, O₃, CO | Accurate, but usually too few and too far from the facility. They are the calibration anchor for everything else. |
| **Low-cost sensors** | PurpleAir public map / API (PM2.5) | Needs the EPA US-wide correction (Barkjohn et al. 2021), QA, and calibration against AQS monitors at the same site. Check API terms for bulk history. |
| **Mobile monitoring** | Public campaign datasets where they exist (e.g. Google/Aclima Oakland, Apte et al. 2017) | City-dependent. Same structure as Heat Watch traverses, so `sparc/core/identify/` mostly carries over. |

**Satellites**

| Product | What it measures | Resolution | Coverage | Use in SPARC |
|---|---|---|---|---|
| **NASA TEMPO** NO₂ (L2/L3) | Tropospheric NO₂ column | ~2 × 4.75 km, **hourly** in daylight | North America, Aug 2023 → | Main satellite signal: hourly pre/post at sites that started after late 2023 (e.g. Memphis, 2024), wind-sorted plumes |
| **TROPOMI** (Sentinel-5P) NO₂, HCHO | Tropospheric columns | ~5.5 × 3.5 km, daily early afternoon | Global, May 2018 → | Long pre-period baseline; plume and emission estimates by averaging many days |
| **MAIAC** aerosol optical depth (MODIS) | Total aerosol in the column | 1 km, daily | Global, 2000 → | Spatial PM2.5 proxy, calibrated against monitors; weak for small sources |
| **Sentinel-2 / Landsat** | Surface imagery | 10–30 m, 5–16 days | Global | Construction start, building footprint, turbine and generator yards, land-cover change |
| **VIIRS** night lights | Light at night | ~500 m, daily | Global | Independent check on when a site came online |

Satellite caveats the plan has to handle:
- **They measure the column, not the air people breathe.** Converting to surface concentration needs a ratio, calibrated against monitors (§4.11).
- **Clouds and snow blank them out,** and the gaps are not random: cloudy days have different weather. Pre/post comparisons must use matched weather, not raw averages.
- **TEMPO NO₂ is still labelled "provisional"** and has had processing-version changes. Use one consistent version across pre and post, never mix near-real-time and standard files.
- **A pixel covers several km².** A single backup-generator site will usually not show up in it; a 100+ MW continuous plant can.

**Sources, timeline, weather and people**

| Need | Public source | Notes |
|---|---|---|
| **Point sources** | EPA NEI (every 3 years), EPA ECHO, state air permits | Permits give units, kW, permitted hours and tons. Actual run-hour reports are public records but usually need a records request. |
| **Facility timeline** | Building permits, interconnection filings, satellite imagery, news | Construction start, commissioning, turbine arrival: the "treatment dates" |
| **Roads** | HPMS / state AADT counts, truck share | Line sources for traffic |
| **Meteorology** | ASOS hourly wind (critical); ERA5 / HRRR boundary-layer height | `sparc/core/forcing.py` already fetches ERA5 and stations |
| **People and equity** | HRSL (already used), CDC PLACES asthma, census | EJScreen was taken down from EPA's site in 2025; use archived copies |
| **Health** | BenMAP-CE concentration–response functions, baseline rates | For avoided deaths, asthma and ER visits, and dollars |

---

## 4. What changes in the code

Effort key from the core roadmap: S ≈ days, M ≈ 1–2 weeks, L ≈ 3+ weeks.

### 4.1 Time axis and sparse observations (L): the biggest change
- **Today:** `CoreData` holds one target value per lattice cell (`sparc/core/data.py`), and `Grid` assumes points on or near a lattice (`sparc/core/grid.py`).
- **Change:** keep the 30 m raster for *covariates and sources*, and add a long observation table: `(sensor_id, x, y, t, value, instrument, qc_flag)`.
  - New `sparc/core/obs.py`: sensor QA (PurpleAir A/B channel agreement, humidity correction, drift, collocation calibration against AQS), timestamps in UTC with time zones handled (the core roadmap's DST lesson).
  - **Target:** the local increment above regional background, `ΔC = C − C_bg(t)`, where C_bg comes from upwind or regional monitors. This mirrors ΔT = T − background.
  - **Unit of analysis:** hourly, aggregated to daily and annual for reporting against standards.

### 4.2 Physics: emissions instead of energy balance (L)
- **Today:** `sparc/core/physics.py` drives the advection–diffusion–relaxation operator with a sensible-heat source built from albedo, canopy and impervious cover.
- **Change:** the source becomes emissions: `q = Σ_roads E_r·line_r + Σ_facilities E_f·point_f`.
  - **Ground-level line sources (traffic):** the existing 2D operator is a reasonable approximation, with hourly wind `v = u·τ` and dilution scaled by mixing height.
  - **Elevated point sources (stacks, turbines, generator exhausts):** the 2D operator is wrong near the source because plume rise and stack height matter. Use **AERMOD**, EPA's regulatory dispersion model, which is free. Run it once per source at unit emission rate for each hour's meteorology, and store the result as a transfer kernel `K_f(x, t)`. Then `ΔC_phys = Σ_f θ_f·E_f·K_f + roads term`, where θ_f are calibration factors fitted from measurements and bounded by the permits.
  - **Why AERMOD and not our own:** regulators and courts already accept it. Re-implementing it gains nothing and costs credibility. SPARC's value is calibrating it against what sensors measure and testing it causally.
  - **NO₂ chemistry:** the NOx → NO₂ conversion uses AERMOD's ARM2 or ozone-limiting method. This is the one non-linearity in emission scenarios.
  - **Fallback** if AERMOD is not available in the environment: a Gaussian plume with Briggs dispersion, labelled as screening-level.

### 4.3 Cross-validation in space and time (M)
- **Today:** `sparc/core/cv.py` builds square spatial blocks with a buffer, which is fine for 54,000 cells but not for 10–50 sensors.
- **Change:**
  - **Leave-one-sensor-cluster-out**, with a distance buffer.
  - **Leave-one-time-block-out**: whole weeks, because hourly data is strongly autocorrelated.
  - **Report both, plus the combination.** A model that only works on known sensors at known times is an interpolator, which is the same lesson as the 0.944 score.
- **Stacker** (`stacker.py`): unchanged. With thousands of sensor-hours, the gated neural residual may finally beat the blend. If it doesn't, the gate keeps it out.

### 4.4 Area of influence (S–M)
- `sparc/core/influence.py` currently runs FFT correlograms on a filled raster, which is not possible with sparse sensors.
- **Keep** the distributed-lag "kernel-shape" ranges: road and emission density within r, with r learned from sensor data.
- **Add wind-conditioned focal features**: emissions upwind of each sensor in each hour. This is the strongest single predictor in the literature, and it comes almost for free from the operator.
- **Saturation** (`response.py`): mostly irrelevant, because concentration is roughly linear in emissions. Repurpose it to distance-decay curves ("how far does facility *F*'s effect reach?").

### 4.5 Scenarios become emission edits (M)
- `sparc/core/scenarios.py` already edits the predictor frame and re-runs physics and the stacker. New intervention types:
  - `scale_source(F, factor)`: fewer operating hours, or a curtailment;
  - `replace_emission_factor(F, from=Tier2, to=Tier4)`: cleaner engines (EPA emission factors);
  - `add_source(location, permit_emissions, stack)`: a proposed data centre;
  - `schedule(F, hours)`: testing at night vs midday, which matters through mixing height;
  - `traffic(route, Δtrucks)`.
- **Two emission levels for every scenario: permitted, and actual as estimated from satellites and monitors** (§4.6), or as reported where those reports are public. The gap between them is often the whole policy argument.
- **Uncertainty:** emission-rate uncertainty × posterior of the calibration factor × fold spread.

### 4.6 Source attribution (M)
- **Model side:** a per-source contribution map `θ_f·E_f·K_f`, plus each source's share of the local increment at every home.
- **Model-free side:** a conditional bivariate polar analysis per sensor, showing concentration by wind direction and speed (the openair `polarPlot` method). A facility shows up as a lobe pointing at it.
- **Satellite side:** average many days of TEMPO or TROPOMI NO₂, each rotated so the wind blows the same way, and the plume from a large source emerges from the noise. Fitting its shape gives an emission rate, independent of the permit (the "exponentially modified Gaussian" or flux-divergence methods in the satellite literature). For a site like Memphis, this is a public-data check on what was actually emitted, compared with what was permitted.
- **Gate:** report a source attribution only when the sides agree on direction and order of magnitude. This mirrors the existing model-vs-causal audit in `causal.py`.

### 4.7 Causal: pre/post designs (L)
These are new estimators in `sparc/core/causal.py` or a new `sparc/core/identify/facility.py`:

1. **Wind-direction difference-in-differences.** At each sensor, compare hours when it is downwind of *F* with hours when it is upwind, before vs after *F* starts operating. Weather, season, regional pollution and the sensor's own bias cancel; only something that blows from *F* survives. This is the air-quality equivalent of the canopy street-difference design.
2. **Distance-ring event study:** change after start-up by distance band, with a smooth spatial basis. The effect should decline with distance.
3. **Synthetic control:** far-away sensors weighted to match the near-site sensors before construction.
4. **Run-event study:** if public records give generator run dates (state records requests; grid-emergency days are public from the grid operator), compare run days with matched non-run days, downwind only.

5. **Satellite version of each design.** The same wind-sorted difference-in-differences and distance-ring designs, run on TEMPO and TROPOMI pixels instead of sensors (§4.11). This gives an estimate where there are no ground sensors near the site, and a second, independent estimate where there are.

**Pipeline validation (runs automatically before any real result is reported).** These checks don't produce findings; they test whether the pipeline works as designed on *this* site's data. If one fails, the run reports the failure instead of an effect.
- **Planted-effect check** (`simcheck.py` / `identify/kmlab.py`). Plant a facility with known emissions into the real sensor layout, real satellite pixel footprints and cloud gaps, and real hourly wind. Run every design and confirm that it recovers the planted effect.
- **Nothing-planted check.** The same run with zero emissions must report ~0 at least 90% of the time. This catches designs that turn weather, seasons or sensor drift into a fake effect.
- **Minimum detectable effect** comes out of these two: "with these sensors, these satellite pixels and these dates, an increase smaller than X µg/m³ would not be detected 80% of the time." This is how to report backup generators honestly.
- **Real-data timing check** (optional, off by default). The same design with the start-up date moved a year earlier, before anything was built, should find nothing. It checks the pipeline on the real data rather than simulated data, catching problems the simulation didn't anticipate (a sensor swapped that year, a satellite processing change). It is reported in the methods section, not as a result. `placebo.py` has the machinery.
- **E-value / sensitivity**, already in `causal.py`, says how strong an unmeasured cause would have to be to explain the estimate away.

### 4.8 Planner outputs (M)
`sparc/core/planner.py` and the Studio results page get these outputs:
- residents above thresholds: EPA NAAQS (PM2.5 annual 9 µg/m³; NO₂ 1-hour 100 ppb, annual 53 ppb) and WHO 2021 guidelines;
- the facility's contribution at schools, clinics and homes;
- health impact through BenMAP concentration–response functions, with dollar values;
- an equity split (`optimize.equity_column` already exists).

New Studio tabs: **Sources** (contribution maps, polar plots) and **Facility impact** (pre/post estimate, minimum detectable effect, scenarios).

### 4.9 Synthetic test city (M): do this first
- Extend `sparc/core/synthetic.py` with roads, one or two point sources with known emissions, hourly wind from a real ASOS record, a realistic sensor layout (a few regulatory monitors and dozens of low-cost sensors with noise and drift), and a known start-up date.
- **Every stage must recover the planted truth before any real data is touched.** This is P0's rule in the core roadmap, applied again.

### 4.10 Pre-registration (S)
- New command `sparc core preregister -p <config>`. It hashes the config, code, design choices and the *pre-period* data into a signed manifest before post-period data is loaded. `provenance.py` and `reproduce.py` already have most of the machinery.
- The report states whether the analysis matched its pre-registration. This is the single strongest defence against "you tuned it until it showed harm".

### 4.11 Satellite fusion (L)
Satellites are a second observation layer with its own footprint, not just another predictor.
- **Observation model.** A satellite pixel sees the *column average over its footprint*: `Ω_pixel(t) = ∫ w(x)·[ΔC(x,t)·H(t) + C_bg(t)·H_bg] dx + noise`. H is the effective mixing depth, which turns surface concentration into a column, and is fitted per hour from ERA5 boundary-layer height plus a calibration against ground monitors. Neighbouring pixels then constrain the model's 30 m surface field, instead of being smeared into it.
- **New module** `sparc/core/satellite.py`:
  - fetchers for TEMPO (NASA ASDC / Earthdata), TROPOMI and MAIAC, with versions recorded in the manifest;
  - quality filtering (cloud fraction, the products' own QA flags, solar angle);
  - pixel footprints as polygons, so each pixel maps onto the 30 m grid as a weighted average, not a point;
  - day-by-day matching to hourly weather.
- **Fitting.** The physics model and the stacker are fitted jointly to monitors (point, surface) and pixels (area, column), with a weight for each based on its error. Monitors pin the level; satellites fill the space between them and extend coverage back to 2018 (TROPOMI) or hourly from late 2023 (TEMPO).
- **Testing.** Satellites get their own held-out test: hold out whole days of pixels, and report skill separately for monitors and pixels. A model that fits pixels but misses monitors (or the reverse) is flagged.
- **What it can't do:** see a single backup-generator test, or anything smaller than about one pixel near a large city's own NO₂. The pipeline validation in §4.7 states which sites are above that limit.

---

## 5. Phased plan

### P0: Synthetic facility city and pipeline validation (M)
- §4.9 and the pipeline-validation part of §4.7. The synthetic city also produces fake satellite pixels (column averages over real TEMPO/TROPOMI footprints, with cloud gaps), so the satellite path is tested from day one.
- **Exit:** on the synthetic city:
  - contribution maps recover the planted emissions within their intervals;
  - the wind-direction difference-in-differences recovers the planted step, from sensors and from pixels;
  - it reports ~0 for a facility with no emissions in at least 90% of runs.

### P1: Public observations and time axis (L)
- §4.1 and §4.3, plus the satellite fetchers and QA from §4.11. Load AQS, PurpleAir, TEMPO, TROPOMI and MAIAC for one metro area, 2018 to now.
- **Exit:**
  - sensor QA report, and calibration error against AQS monitors;
  - satellite coverage report: usable pixels per day, cloud gaps by season, one consistent processing version across the whole period;
  - baselines (land-use regression, kriging, nearest AQS monitor) scored on held-out sensors and held-out weeks.

### P2: Emission physics (L)
- §4.2 and §4.4, starting with roads and NEI point sources.
- **Exit:**
  - the physics model alone beats land-use regression on held-out sensors;
  - the stack beats both;
  - 90% intervals cover 85–95% of held-out sensor-hours.

### P3: Satellite fusion (L)
- §4.11: the joint fit to monitors and pixels, plus the satellite designs in §4.6 and §4.7.
- **Exit:**
  - adding satellites improves skill on held-out *monitors*, not just on pixels;
  - on the synthetic city, the satellite plume method recovers planted emission rates within its interval.

### P4: Pilot site, pre-registered (M)
1. Pick the site (see §6).
2. Assemble the timeline (satellite imagery, permits, news), the permits and the public sensors.
3. **Pre-register** (§4.10).
4. Run the pipeline validation on the real layout, real pixels and real weather to get the minimum detectable effect *before* estimating anything.

- **Exit:** a pre-registration manifest and a minimum-detectable-effect table, published before any results.

### P5: Pilot estimate, attribution and scenarios (M)
- §4.5–§4.8.
- **Exit:** the facility-impact report, with ground and satellite estimates side by side, the source-attribution gate, and scenarios at both permitted and satellite-estimated emission levels.

### P6: Second site and write-up (M)
- Rerun on a backup-generator site, where nothing changes in the code and only the config does. Then write the paper.

---

## 6. Pilot site

| Option | For | Against |
|---|---|---|
| **Memphis, xAI Colossus** | Continuous turbines, so the largest and most detectable signal. A sharp start date (2024), after TEMPO began (Aug 2023), so there are hourly satellite data before and after, and TROPOMI back to 2018. Public city testing and a University of Tennessee satellite analysis already exist ([Tennessee Lookout](https://tennesseelookout.com/briefs/naacp-others-appeal-xai-turbine-permits-for-memphis-data-center/), [Inside Climate News](https://insideclimatenews.org/news/17072025/elon-musk-xai-data-center-gas-turbines-memphis/)). | It's in active litigation, so anything we publish will be scrutinised. Pre-registration is mandatory. The number of turbines over time is disputed. |
| **Northern Virginia (Loudoun / Prince William)** | Hundreds of facilities, about 10,000 generators, state permits that list generator counts, dense PurpleAir coverage. Answers the cumulative question. | Effects are small and spread out, and run times are uncertain, so the likely answer is a minimum detectable effect plus an emissions-based estimate. |

**Recommendation:** use Memphis to prove the method, because the signal is big enough to detect. Then use Northern Virginia for the cumulative backup-generator question, where the honest answer will be more limited.

---

## 7. A run we can trust: checklist

A run is reported as valid only if all of these hold. Each one is a manifest field, and the report states any that fail.

- [ ] Every sensor passes QA; calibration error against AQS monitors is reported per instrument type.
- [ ] Satellite inputs use one processing version across the whole period, and their QA filters and cloud gaps are reported.
- [ ] Skill is reported on held-out sensors, held-out weeks and both together, and the stack beats land-use regression and AERMOD alone.
- [ ] Adding satellites improves skill on held-out monitors, or the satellite layer is reported as not used.
- [ ] 90% interval coverage on held-out sensor-hours is within 85–95%.
- [ ] Pipeline validation passes on *this* sensor layout, *these* satellite pixels and *this* weather: planted effects recovered, nothing found when nothing is planted.
- [ ] The minimum detectable effect is reported next to every facility estimate.
- [ ] Model-based and wind-direction (model-free) attribution agree, or the attribution is withheld.
- [ ] Model-implied facility effects agree with the pre/post causal estimate within its interval.
- [ ] Emission inputs are cited (permit number, NEI year), and scenarios are given at both permitted and estimated-actual levels.
- [ ] The analysis matches its pre-registration, or every deviation is listed.

---

## 8. What this will not do

- **Regional and secondary PM2.5** (sulphate, nitrate, organic aerosol formed downwind) cannot be attributed to one facility with this model. Chemical source apportionment (EPA speciation network plus PMF) is a different study.
- **Sub-hourly generator tests** will mostly be invisible in hourly data. Report the minimum detectable effect, and say that a short campaign with fast sensors near the site would close the gap.
- **Upstream power-plant emissions** are regional and belong to EPA AVERT, not this neighbourhood model.
- **Ozone** forms over hours, regionally, and needs a photochemical model (CMAQ). Out of scope.
- **"No effect detected" is not "no effect."** Every null result comes with its minimum detectable effect.

---

## 9. Open questions

1. **Which public sensors and satellite versions cover the pilot site before and after start-up?** P1 answers this; it decides the minimum detectable effect.
2. **Who is the first user?** A community group, a law clinic, a health department or a journalist. Each changes the report format, not the method.
3. **Is AERMOD acceptable as a dependency?** It's a Fortran binary from EPA. The alternative is a screening-level Gaussian plume.
4. **Will we publish nulls?** The plan assumes yes. It is the condition for the tool being trusted by anyone other than people who already agree with it.

## Sources

- [TechCrunch: xAI gets permits for 15 natural gas generators at Memphis data center (2025-07-03)](https://techcrunch.com/2025/07/03/xai-gets-permits-for-15-natural-gas-generators-at-memphis-data-center)
- [Tennessee Lookout: NAACP and others appeal xAI turbine permits](https://tennesseelookout.com/briefs/naacp-others-appeal-xai-turbine-permits-for-memphis-data-center/)
- [Inside Climate News: Colossus operated gas turbines without appropriate permits, residents claim](https://insideclimatenews.org/news/17072025/elon-musk-xai-data-center-gas-turbines-memphis/)
- [arXiv 2412.06288: Health-Informed Computing, the public health impact of data centers](https://arxiv.org/pdf/2412.06288)
- [VCU: Northern Virginia data center air pollution rivals power plant emissions](https://www.news.vcu.edu/article/northern-virginia-data-center-air-pollution-rivals-power-plant-emissions)
- [Sierra Club: A secret coal plant in the DC suburbs](https://www.sierraclub.org/coal/secret-coal-plant-dc-suburbs)
- [NASA Earthdata: TEMPO NRT version 02 Level 1, 2, 3 data released](https://www.earthdata.nasa.gov/data/alerts-outages/tempo-nrt-version-02-level-1-2-3-data-released)
- [NASA ASDC: TEMPO Level 2/3 trace gas and clouds user guide](https://asdc.larc.nasa.gov/documents/tempo/guide/TEMPO_Level-2-3_trace_gas_clouds_user_guide_V1.2.pdf)
