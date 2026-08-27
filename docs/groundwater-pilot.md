# Groundwater Pilot — High Plains Aquifer

**Status:** proposed
**Role:** validation testbed for the core stack, ahead of the heat EWS pilot
**Parent:** [`plan.md`](../plan.md)

---

## 1. Why this domain

This is not a softer target chosen to get an easy win. It is the domain where more
of SPARC runs, with less new code, against better ground truth, than anywhere else
the pipeline has been pointed.

### 1.1 The physics maps exactly

Steady-state groundwater flow is `∇·(K∇h) = −R`. With locally constant K:

```
K ∇²h = −R
```

`sparc/physics/pde_solver.py` solves:

```
α ∇²T = S
```

| SPARC symbol | Hydrogeology | Units |
|---|---|---|
| `alpha` | K — hydraulic conductivity | m/day |
| `T` | h — hydraulic head | m above datum |
| `source` | −R — recharge | m/day |
| `h` (grid spacing) | grid spacing | m |

**`poisson_solve` runs unmodified.** Tier 2 of the scenario engine becomes live in
this domain with zero new physics written. Today it is exercised only in the heat
domain, where its steady-state assumption is a stretch; here the steady-state
assumption is the standard hydrogeologic idealisation.

### 1.2 α becomes independently verifiable

This is the scientific headline and the reason to run this pilot at all.

`ProcessRateNet` learns α as a spatially varying process-rate field. In the heat
domain nobody measures urban thermal diffusivity, so α is unfalsifiable — it is
whatever the network needs it to be. In hydrogeology, **K is a real, named,
independently measured physical parameter** that practitioners estimate from pump
tests and specific-capacity data, and USGS publishes regional estimates (High Plains
K ranges from under 1 to over 100 ft/day, averaging roughly 60 ft/day in Colorado
and New Mexico).

So the pilot can ask a question the heat work cannot: **does the learned α field
correlate with independently known K?** A positive answer is a publishable result
and the strongest possible evidence for the learned-parameter-field thesis. A
negative answer is equally valuable and tells you the α field is a fitting device,
not a physical quantity — which you need to know before building anything else on it.

### 1.3 What else it exercises

| Capability | Heat pilot | Groundwater pilot |
|---|---|---|
| Stage 0 correlogram / Matérn | ✓ | ✓ — smooth field, ideal case |
| Anisotropy | statistical only | **physically real** (bedding planes, paleochannels) |
| GWR per-cell β | ✓ | ✓ |
| Neural trunk + α | ✓ | ✓ **and α is falsifiable** |
| Stage 3 causal | ✓ | ✓ — pumping → decline is a clean treatment |
| **Tier 2 forward solve** | assumption stretched | **✓ native** |
| Cross-region transfer | cities | **basins** — reuses LOCO machinery |
| Ground-truth density | ~5–15 ASOS stations/city | **8,000–9,400 wells** |

### 1.4 Honest limitations

- **Not an evacuation hazard.** The EWS framing here is depletion, well failure and
  subsidence — real decisions on annual timescales, not emergency response. If the
  goal is specifically the emergency-response story, this pilot validates the
  machinery but not the product.
- **Sparse observations.** ~8,000–9,400 wells over ~450,000 km² is roughly one well
  per 50 km². This is a *sparse-observation, dense-covariate* problem — the opposite
  regime from UHI's 54,701 points. See §7.4: **do not expect the neural meta-learner
  to win here**, and treat that as information rather than failure.
- **Steady state is an idealisation.** A depleting aquifer is by definition not in
  equilibrium. §6.6 addresses this by targeting the change field rather than pretending
  otherwise.

---

## 2. Fix this before anything else

**The template's target variable is the wrong physical quantity.**

`templates/groundwater/project.yml` sets:

```yaml
target_column: "gw_level_m"   # Depth to groundwater (m below ground surface)
```

Depth-to-water does **not** satisfy the groundwater flow equation. The flow equation
is written in hydraulic head `h`, and:

```
h = land_surface_elevation − depth_to_water
```

Depth-to-water is dominated by topography — it is essentially the land surface minus
the (much smoother) water table. Feeding it to a Laplacian operator produces a
residual driven by terrain, not by flow. Every PDE term, the α field, and Tier 2
would all be fitting the wrong surface.

**Do:** change the target to `head_m` computed as `dem_elevation_m − depth_to_water_m`,
using 3DEP for land surface elevation at the well collar. Keep depth-to-water as a
derived reporting column, since that is what water managers actually read.

This is a one-line config change plus a derivation step in the collector, and it is
the single highest-leverage correction in this document.

---

## 3. Scope

### 3.1 Region

**Primary:** the full USGS High Plains aquifer extent — eight states (CO, KS, NE, NM,
OK, SD, TX, WY), roughly 450,000 km².

**Why the full extent rather than one district:** the observation network is sparse
per unit area, so a single groundwater management district yields only a few hundred
wells — too few for the neural path and marginal for spatial block CV. The full
network gives N ≈ 8,000–9,400, which is the same order as the ForceSMIP run (71,498
cells) and workable throughout.

**Dense core:** Kansas, where KGS and KDA-DWR measure ~1,400 wells every winter and
publish through the WIZARD database. Use Kansas for any analysis needing high local
density.

**Transfer holdout:** Nebraska. It sits in a genuinely different hydrogeologic regime
— higher recharge, Sandhills recharge zone, far less depletion than the southern High
Plains. Kansas GMD3 alone shows −1.69 ft/yr average decline and −35.4 ft cumulative
over 1996–2017. Training on the depleting south and predicting the recharging north
is a real generalisation test, not a reshuffle.

This directly reuses the leave-one-city-out machinery in
`scripts/train_multicity_jepa.py` with basins substituted for cities. See §9.

### 3.2 Time

**Primary target:** a single year cross-section — 2017, chosen because USGS publishes
a water-level-change raster through 2017 that serves as gridded validation (§7.3).

**Secondary target:** the predevelopment (~1950) → 2017 water-level *change* field.
This is the depletion signal, it is the actual hazard, and USGS publishes it as a
validated raster — giving gridded truth rather than points alone.

Start with the cross-section. Add the change field only after §7 clears.

### 3.3 Analysis frame

| Setting | Value | Rationale |
|---|---|---|
| CRS | EPSG:5070 (NAD83 CONUS Albers) | equal-area, USGS standard for CONUS work |
| Grid resolution | **1 km** | ~450,000 cells; well spacing is ~7 km so finer is meaningless |
| Training points | well locations (N ≈ 8–9k) | model trains on points, predicts to grid |

> ⚠️ The template ships `grid_resolution_m: 30`. At the High Plains extent that is
> ~500 million cells. Change it to 1000 before running anything.

---

## 4. Data inventory

Every layer, with source and current status. Nothing here requires a licence or an
API key.

| Layer | Role | Source | Res. | Notes |
|---|---|---|---|---|
| Water levels | **target** | USGS NWIS groundwater levels service; USGS High Plains WLMS | point | 8–9k wells, annual, mandated reporting to Congress |
| Water levels (dense core) | target | KGS WIZARD | point | ~1,400 KS wells, winter campaign |
| Land surface elevation | **head derivation** + predictor | 3DEP | 1/3 arc-sec | 99% CONUS coverage as of FY25 |
| Saturated thickness | predictor | ScienceBase `631405d0d34e36012efa33aa` (2009) | raster | also a 1996–97 version, `631405cdd34e36012efa3338` |
| Aquifer base elevation | predictor / BC | USGS High Plains geometry datasets | raster | needed for unconfined thickness |
| Water-level change 1950→2017 | **validation** + secondary target | USGS Science Data Catalog | raster | gridded truth |
| Irrigation extent | **pumping proxy** | LANID (30 m, annual, 1997–2017) | 30 m | aggregate to 1 km fraction-irrigated |
| Irrigation extent (alt) | pumping proxy | MIrAD-US | 250 m | 5-year intervals; coarser fallback |
| Precipitation | recharge driver | PRISM or gridMET | 4 km / 1 km | annual + antecedent multi-year |
| Hydraulic conductivity | **α validation** | USGS regional K estimates; High Plains model parameters | raster/zonal | <1 to >100 ft/day |
| NDVI | ET proxy | Landsat / MODIS | 30 m / 250 m | already in `sparc/data/collect/landsat.py` |
| Distance to river | predictor | NHD flowlines | vector | compute distance surface |

### 4.1 Collector work

Add `sparc/data/collect/nwis.py` alongside the existing modules, following the
pattern in `era5.py` and `capa.py`.

```
sparc/data/collect/nwis.py
  fetch_gw_levels(bbox|state, start, end) -> DataFrame
  derive_head(levels_df, dem)             -> adds head_m  (see §2)
  to_fishnet(levels_df, grid)             -> GeoDataFrame
```

**Caution:** USGS is mid-migration. The legacy service lives at
`waterservices.usgs.gov/nwis/gwlevels/` and modernised OGC-style endpoints are
rolling out at `api.waterdata.usgs.gov` through 2025, with a published migration
guide. Verify parameter names and response schema against current docs before
writing the client — do not copy an older example. Build against the modern endpoint
if it is stable, with the legacy one as fallback.

Reuse `sparc/data/collect/http_client.py` and the existing caching/session helpers
rather than writing new HTTP code.

---

## 5. Prerequisites from `plan.md`

This pilot **must not start before** the following, or its results are unreadable:

- **Phase 0.1 — F1 leakage fix.** Every number below is a comparison. Comparisons
  against a leaking baseline are meaningless.
- **Phase 0.4 — re-baseline.** Needed so "the neural path underperforms here" can be
  distinguished from "the neural path underperforms everywhere."
- **§2 above — head vs depth-to-water.** Non-negotiable.

Strongly recommended but not blocking:

- **Phase 2.1 — conformal calibration.** With sparse observations, interval quality
  matters more than usual, and kriging (the baseline) produces principled variance
  for free. Going in without calibrated intervals means losing the uncertainty
  comparison by default.
- **Phase 1.5 — the MGWR rename.** Hydrogeologists will check for back-fitting.

---

## 6. Stage-by-stage

### 6.1 Stage 0 — correlogram and anisotropy

Expect this to be the best-behaved run the pipeline has had. Water table surfaces are
smooth and strongly autocorrelated — precisely the regime Matérn models were built for.

- Expect long correlation ranges (tens of km) and a smooth ν selection, probably
  ν = 1.5 or 2.5.
- **Anisotropy is physically meaningful here.** Aquifers are anisotropic along bedding
  planes and buried paleochannels. If the fitted ellipse orientation aligns with known
  paleochannel trends in the Ogallala, that is an independent validation of the
  anisotropy module — worth checking explicitly against published hydrogeologic maps.
- Watch the harmonic-mean marginal (Phase 1 item, `correlogram_matern_fit.py:220`).
  With a smooth field and few lags this estimator is at its least reliable. Record the
  per-ν marginals and sanity-check the selection by hand.

**Failure mode:** with ~8k irregularly spaced points, lag binning may be unstable at
short distances. Check the lag histogram before trusting the fit.

### 6.2 Stage 1 — GWEN

**Skipped.** Being removed per `plan.md` §1.3. Predictors here are chosen by
hydrogeology, not by predictive screening, and several — K, saturated thickness — are
confounders that must not be dropped for low predictive importance.

### 6.3 Stage 2 — models

Run with the reduced stack from `plan.md` §1.2: GWRF as predictor, GWR for per-cell β,
trunk + head as the neural path. No surrogates, no GGPGAM, no OLS ensemble member.

- **Regime warning.** N ≈ 8–9k training points with ~10 predictors is a small-data
  regime for a SIREN trunk with spatial attention. Expect GWR/GWRF/kriging to be
  competitive or better. **This is a finding, not a failure** — it maps the boundary
  of where the neural architecture applies, which you currently do not know.
- Spatial block CV: with ~8k points across 450,000 km², check fold populations
  carefully. `spatial_kfold_enhanced` warns below 50 test points per fold; verify no
  fold is degenerate.
- The α field is the output that matters most here. Persist it at full resolution.

### 6.4 Stage 3 — causal

The causal story is unusually clean, which is part of why this domain is a good test.

- **Treatment:** irrigation pumping (proxied by LANID fraction-irrigated).
- **Outcome:** head, or head change.
- **Confounders:** K, saturated thickness, aquifer base elevation, distance to river,
  land surface elevation.
- **Mediator:** recharge.

The template DAG at `templates/groundwater/causal/dag.yml` is already written and
cites Darcy (1856), Theis (1935), Healy (2010), Winter et al. (1998). Review it
against the data actually collected rather than assuming it fits.

- **Sign check is free here.** Pumping must lower head. If the estimated effect has
  the wrong sign, something upstream is broken — this is a much stronger correctness
  signal than the heat domain provides.
- **Magnitude check is also available.** The Theis solution gives an analytical
  drawdown for a given pumping rate, transmissivity and storativity. Compare the
  estimated effect against Theis for representative parameters. **No other SPARC
  domain offers an analytical answer to check against.**
- MC³ with the retuned prior (`plan.md` §1.9) — with a physically well-established DAG,
  this is the ideal place to demonstrate prior→posterior movement.

### 6.5 Stage 4 — Tier 2 forward solve

**The main structural test.** For the first time Tier 2 runs on physics it natively
describes.

- Scenario: reduce pumping by X% over a management district; solve for the new
  equilibrium head field.
- Boundary conditions: `poisson_solve` holds boundary points at `T_init`
  (quasi-Dirichlet). For an aquifer this is a *specified-head* boundary — defensible
  at the aquifer margins and along major rivers, less so elsewhere. Document the
  choice; it is a real modelling assumption, not an implementation detail.
- Convergence: the solver caps at `max_iter=500` with `tol=1e-4` and Jacobi/SOR. Over
  450,000 cells Jacobi will converge slowly. Expect to need `omega` in (1, 2) and
  possibly a higher iteration cap. **Check `info['converged']` — do not consume a
  non-converged field.**
- **Benchmark honestly:** MODFLOW is the validated standard for this problem, and the
  orchestrate-don't-own principle applies exactly as it does to HEC-RAS. The claim
  here is *not* "we replace MODFLOW." It is "a learned α field plus a simple solve
  reproduces the regional pattern," which is a much smaller and defensible claim.
  If the pilot succeeds, MODFLOW becomes the Phase 5 orchestration target.

### 6.6 On steady state

A depleting aquifer is not in equilibrium, so `K∇²h = −R` is an approximation. Two
honest ways to handle it, in order of preference:

1. **Target the change field.** Model the 1950→2017 water-level change as a
   quasi-steady response to a sustained pumping stress. This is closer to what the
   equation supports and it is the hazard-relevant quantity.
2. **Treat the cross-section as quasi-steady** and state the assumption. Acceptable
   for a single year, but say so.

Do not silently use the steady-state solver on a transient system and present it as
physics.

---

## 7. Baselines and metrics

### 7.1 The arms

The baseline here is **strong**, and pretending otherwise wastes the pilot. Ordinary
kriging is the standard method for water table mapping, and the literature is explicit
that kriging with external drift using land surface elevation outperforms it further —
plus geostatistical methods incorporating the groundwater flow equation already exist.
Do not benchmark against a strawman.

| Arm | Method | Tests |
|---|---|---|
| **A** | Ordinary kriging of head | the classical floor |
| **B** | Kriging with external drift (elevation as covariate) | **the real null** — this is what a hydrogeologist would do |
| **C** | GWRF + full covariate set, no PDE | does the covariate stack beat geostatistics |
| **D** | Full SPARC: trunk + α + PDE + conformal | does the physics add anything over C |

Arm B is the one that matters. Beating A and losing to B means the pipeline is an
expensive kriging.

### 7.2 Point metrics

Held-out wells under spatial block CV:

- RMSE and MAE on head (m)
- R² — but report it alongside RMSE; with a smooth strongly-autocorrelated field R²
  will look flattering and mean little
- **Interval coverage** at 90% nominal, empirical, from conformal (arm D) and kriging
  variance (arms A/B). Kriging gives principled variance for free — this is a fair
  fight and SPARC should not lose it.

### 7.3 Gridded validation

Independent of the point metrics: compare the predicted change surface against the
**published USGS water-level-change raster** (predevelopment → 2017). This is gridded
truth from an authoritative source, which no other SPARC domain has. Report spatial
correlation and a difference map.

### 7.4 α ↔ K — the headline

Correlate the learned α field against independently known K:

- Spearman correlation of α vs published regional K, at well locations and by zone
- Does α recover the known coarse-grained structure — high K in the central Nebraska
  paleochannels, lower K in the southern fine-grained facies?
- Report honestly. α is dimensionless-ish inside the loss (normalised by its detached
  mean at `training/loss.py`), so the comparison is about **spatial pattern, not
  absolute magnitude**. State this rather than implying a calibrated K estimate.

**If α correlates with K:** that is the strongest result available from this pilot and
is worth a paper on its own.
**If it does not:** α is a fitting device, and the "learned parameter fields for
physical solvers" thesis needs rethinking before Phase 5 depends on it.

### 7.5 Kill criteria

State before running.

| Condition | Meaning |
|---|---|
| Arm D loses to arm B on held-out RMSE | the pipeline is an expensive kriging — stop and reconsider |
| Arm D does not beat arm C | the physics adds nothing; keep the covariate model, drop the PDE claim |
| Empirical coverage misses nominal by >10 pts while kriging is calibrated | the UQ story is not competitive |
| Pumping → head effect has the wrong sign | something upstream is broken; do not proceed |
| Tier 2 does not converge on the real domain | the solver needs work before Phase 5 |
| α shows no relationship to K | record it; it constrains Phase 5 scope |

---

## 8. Sequence

| Step | Work | Depends on |
|---|---|---|
| G1 | Fix target: head, not depth-to-water (§2). Fix `grid_resolution_m`. | — |
| G2 | Write `sparc/data/collect/nwis.py`; verify against current API docs | G1 |
| G3 | Assemble covariate stack — 3DEP, saturated thickness, LANID, PRISM, NHD | G2 |
| G4 | Build the 1 km analysis frame in EPSG:5070; join wells | G3 |
| G5 | Arms A and B — kriging baselines. **Do these first.** | G4 |
| G6 | Stage 0; check anisotropy against paleochannel maps | G4 |
| G7 | Stage 2 arms C and D | G6, plan.md 0.1 + 0.4 |
| G8 | α ↔ K analysis (§7.4) | G7 |
| G9 | Stage 3; Theis sign and magnitude checks | G7 |
| G10 | Stage 4 Tier 2 forward solve; convergence audit | G7 |
| G11 | Gridded validation against USGS change raster | G7 |
| G12 | Basin transfer — train south, hold out Nebraska | G7 |
| G13 | Write up, including negative results | all |

**Do G5 before G7.** Establishing the kriging baseline first means the neural result
gets interpreted against a known number rather than rationalised after the fact.

---

## 9. Basin transfer

Reuse `scripts/train_multicity_jepa.py` with basins substituted for cities. The
existing machinery maps over directly:

| Multi-city | Groundwater |
|---|---|
| city | groundwater management district / state sub-region |
| Köppen zone weighting | hydrogeologic setting (unconfined / confined, facies) |
| `phase1_only` cities (no CAPA labels) | sub-regions with covariates but sparse wells |
| LOCO holdout (Philadelphia) | Nebraska |
| ERA5 → UHI anomaly calibration | regional recharge → head calibration |

This is also the honest place to re-test PI-JEPA (`plan.md` §1.4a): sub-regions with
full covariate rasters and few measured wells are exactly the unlabeled-abundant
regime, and the LOCO metric is uncontaminated.

---

## 10. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Neural path loses to kriging | **high** | expected; §7.5 treats it as information. The pilot's value does not depend on winning |
| NWIS API migration breaks the collector | medium | build against modern endpoint, keep legacy fallback, pin response schema in a test |
| Sparse wells → degenerate CV folds | medium | check fold populations; fall back to leave-one-out or larger blocks |
| Steady-state assumption criticised | medium | §6.6 — target the change field, state the assumption |
| K reference data too coarse for α validation | medium | fall back to zonal comparison rather than per-cell; report at the resolution the reference supports |
| Scope creep into MODFLOW coupling | **high** | explicitly Phase 5. Not this pilot |

---

## 11. What this de-risks

Whatever the outcome, the heat EWS pilot in `plan.md` §3 becomes cheaper and better
posed:

- **Tier 2 gets exercised on native physics** before being asked to do anything harder.
- **The α field gets falsified or validated** against an independent measurement,
  before Phase 5 stakes anything on learned parameter fields.
- **The conformal calibration path gets tested** against kriging variance — a fair,
  well-understood competitor.
- **The transfer thesis gets a second, independent test** in a different domain.
- **The stack gets characterised in a sparse-data regime**, which is currently a blind
  spot: everything to date has been dense-N.
- The causal stage gets **sign and magnitude checks against an analytical solution**,
  which no other domain provides.

That last point is worth restating. Nothing else in the SPARC portfolio lets you
check a causal estimate against a closed-form answer. If Stage 3 cannot reproduce
Theis drawdown to order of magnitude on a domain this clean, that is something you
want to discover here rather than in front of a reviewer.
