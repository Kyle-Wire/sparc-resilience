# SPARC — Remediation and Early-Warning Pilots

**Status:** proposed
**Branch:** `claude/sparc-resilience-review-eklzi2`
**Baseline commit:** `0a04b7a`

This plan does three things, because they are not independent: it clears the defects
found in review, it removes the accreted complexity those reviews exposed, and it runs
two scoped experiments to test whether SPARC can serve as the last mile of an
early-warning system.

The two pilots are ordered deliberately. **Groundwater comes first** — it is cheaper,
it exercises more of the pipeline (including Tier 2, on physics it natively
describes), it has far denser ground truth, and it is the only domain where the
learned α field can be checked against an independently measured physical parameter.
It de-risks the heat pilot rather than competing with it.

Two items from the defect list sit directly on the critical path to both pilots:

- **F1 (leakage)** — until it is fixed, no measurement in the repo can be trusted,
  including every ablation this plan proposes.
- **F10 (calibration)** — an early-warning system's product *is* a calibrated
  probability. Without coverage validation there is no pilot, only a demo.

So the sequence below is not "chores first, then the fun part." Phase 0 and Phase 2
are prerequisites of Phases 3 and 4.

---

## Table of contents

- [Phase 0 — Unblock measurement](#phase-0--unblock-measurement)
- [Phase 1 — Subtract](#phase-1--subtract)
- [Phase 2 — Calibration](#phase-2--calibration)
- [Phase 3 — Groundwater pilot](#phase-3--groundwater-pilot)
- [Phase 4 — Heat EWS pilot](#phase-4--heat-ews-pilot)
- [Phase 5 — Response layer](#phase-5--response-layer)
- [Phase 6 — Simulation cascade (design note only)](#phase-6--simulation-cascade-design-note-only)
- [Not doing](#not-doing)
- [Finding index](#finding-index)

---

## Phase 0 — Unblock measurement

Nothing else is worth doing until these land. Every number in the README and every
ablation below is measured against this baseline.

### 0.1 — Fix the OOF leakage (F1)

**Where:** `sparc/run/enhanced_spatial_cv.py:2107–2115`

`_base_oof` is built from `base_fitted_values`, which Stage 2b collects from models
refit on the *full* dataset — the code says so itself: *"These are in-sample
predictions (not OOF)."* Those values are then sliced `[train_idx]` at
`sparc/run/v2_neural_training.py:730` and used as surrogate pre-training targets. A
GWR fitted value at a training point is a kernel-weighted function of neighbouring
`y`, so with a `block/3` buffer it reaches into the held-out block.

**Do:** pass the genuine OOF predictions instead. They already exist in the same
function as `base_predictions` (built at `enhanced_spatial_cv.py:2049–2056`).

**Done when:** Stage 2 re-runs clean and the R² delta is recorded in this file. Do
not hide the delta — publish it. The difference between 0.902 and the corrected
number is the honest size of the leak, and reporting it is a stronger position than
quietly restating a lower figure.

### 0.2 — pytest in CI (F11)

**Where:** `.github/workflows/` currently contains only `build-desktop.yml`, on tags.

486 tests exist; 411 pass under a partial dependency set. Nothing gates a merge.

**Do:** add a `pytest` job on push and PR. Start with the subset that runs without
GPU or geospatial deps (~400 tests). Add `torch`/`geopandas` jobs later if runner
minutes allow.

**Done when:** a PR with a deliberately broken assertion fails CI.

### 0.3 — Make `--resume` work (F2)

**Where:** `sparc/__main__.py:444` is the only `_orch.mark_complete()` call in the file.

Stages 2–5 never mark completion. The calls at 544/561/586/647 are
`_sp.stage_done()` — that is `StageProgress`, the progress bar, a different object
with a near-identical method name. `orchestrator.stage_done()`
(`sparc/run/orchestrator.py:238–254`) also has no disk-sentinel fallback despite the
comment at `__main__.py:398–399` promising one, so the `.gwen_complete` file written
at line 530 is never read.

**Do:** add `mark_complete` after each successful stage and give `stage_done` the
documented fallback. Longer term, route `cmd_run` through
`PipelineOrchestrator.run()` — `_run_one` at `orchestrator.py:260–268` already gets
this right.

**Done when:** `sparc run -s all` then `sparc run -s all --resume` skips Stages 0–4.

### 0.4 — Re-baseline

Re-run UHI and ForceSMIP end to end on the fixed code. Record corrected OOF R² for
every base model and the meta-learner in a `docs/baselines.md`. **Every ablation in
Phase 1 is measured against this table, not against the README.**

---

## Phase 1 — Subtract

The pipeline's complexity came from adding capabilities without removing what they
superseded. Each item here is a deletion justified by a measurement — run the
measurement first.

### 1.1 — Trunk-only ablation (gates 1.2)

**Experiment:** train `SharedTrunk + CityHead` on physics features and coordinates
only. No base-model inputs, no surrogates.

**Rationale:** the published stack scores 0.902 against GWRF's 0.898 on UHI, and
0.640 against GWRF's 0.642 on ForceSMIP — i.e. on ForceSMIP the ensemble was *worse*
than one of its own inputs. A SIREN field with spatial attention already occupies
GWR's hypothesis class.

**Decision rule:** if trunk-only lands within noise (±0.01 R²) of the full stack on
the 0.4 baseline, proceed to 1.2. If it loses materially, keep the base inputs and
record why.

### 1.2 — Delete the superseded stack

Conditional on 1.1. Roughly 5,700 LOC across:

| Component | Path | LOC | Reason |
|---|---|---|---|
| GWEN | `sparc/models/gwen.py`, `sparc/run/gwen_*.py` | ~1,890 | see 1.3 |
| Surrogates | `sparc/models/surrogates.py` | 779 | fidelity term already inactive |
| GGPGAM | `sparc/models/ggpgam.py` | 575 | subsumed by GWRF; `lam=0.6` hard-coded |
| OLS as ensemble member | `sparc/models/ols.py` | 204 | R² 0.294 — keep as printed baseline |
| Stage 2 JEPA | see 1.4 | ~1,400 | wrong data regime |
| cross_correlogram | `sparc/run/cross_correlogram.py` | 828 | see 1.5 |

**Keep GWR.** Not as a predictor (0.828) but as the only source of per-cell β, which
Stage 3 NUTS consumes as an informed prior and Stage 4 Tier 1 samples from. That is a
real job with real consumers.

### 1.3 — Drop GWEN

The reason is stronger than "not critical." GWEN selects predictors by *predictive*
importance and hands the survivors to a causal stage that needs a fixed DAG with
named confounders. **Dropping a confounder because it has low predictive importance
biases every downstream causal estimate** — confounders earn their place by sitting
on a backdoor path, not by improving R². Running it before Stage 3 is backwards.

It also owns the approval gate that blocks the pipeline, which is expensive while
0.3 is unfixed.

**Keep:** the VIF and condition-number diagnostics at
`sparc/run/gwen_variable_selection.py:532–546`. Re-home them in Stage 0 as a
collinearity report that informs but does not gate.

### 1.4 — Move PI-JEPA to where it belongs

**Cut from Stage 2.** `sparc/run/v2_neural_training.py:2768` pretrains for 20 epochs
over `np.arange(N_pt)` — every point in the dataset, all of which have labels the
supervised loss uses moments later. Self-supervised pretraining on exactly the
labeled set is close to a no-op. It also runs *once before the fold loop* and loads
into every fold (`:610`) and the final model (`:3138`), so the OOF R² you would use
to judge it is itself contaminated by it.

**Keep in `scripts/train_multicity_jepa.py`.** `configs/multicity_pilot.yml` carries
cities flagged `phase1_only` — full predictor rasters, no CAPA labels. Those cities
contribute to Phase 1 representation learning and nothing else. That is the
abundant-unlabeled / scarce-labeled regime JEPA exists for.

**Do:** set `jepa.enable: false` in every single-city template; delete the Stage 2
pretraining block; describe PI-JEPA as the multi-city trunk strategy, not a
per-project pipeline component.

**Ablation (1.4a):** run the leave-one-city-out evaluation with `pretrain_epochs: 20`
and `0`. The RMSE delta is the honest value of PI-JEPA, and LOCO is the only clean
place to measure it — the Stage 2 metric is confounded by construction.

### 1.5 — It is not MGWR

**Where:** `sparc/models/gwr.py:335` and `:351`

`fit` makes one pass of local regressions; there is no back-fitting loop anywhere in
the file. True MGWR (Fotheringham, Yang & Kang 2017) requires back-fitting because
per-covariate bandwidths cannot be expressed as a single weighted least-squares
solve. And the per-predictor bandwidths are collapsed anyway:

```python
# gwr.py:335 — the max sets the neighbour radius
max_bandwidth = max(self.variable_bandwidths.values())
# gwr.py:351 — the mean becomes the one kernel bandwidth for every row
bandwidth = np.mean(list(self.variable_bandwidths.values())) + 1e-10
```

The V×V effective-range matrix from `cross_correlogram.py` — 828 LOC — reduces to two
scalars. The anisotropic branch is better (geometric mean of per-predictor Matérn
kernels) but its own docstring concedes it keeps "a single weight per row."

**Do:** rename to GWR everywhere — including `bayesian_mgwr_ensemble.py`, the
`mgwr_local_coefficients_unconstrained.csv` artifact, the README's "Bayesian MGWR
priors," and the `mgwr_causal_anchored` coefficient source in the published ForceSMIP
table. Delete `cross_correlogram.py` and use a single correlogram-derived bandwidth.

**Keep the anisotropy.** That path does real geometric work in both GWR
(`gwr.py:277`) and GWRF (`gwrf.py:191`).

### 1.6 — Surface silent failures

**Where:** `sparc/models/gwr.py:746` and the batch handler below it; `gwrf.py:226`.

A failed local fit is recorded as `β = 0, intercept = ȳ`; a failed local RF becomes
`None`. For a tool emitting per-cell policy advice, "the fit failed here" and "canopy
has no effect here" must not be the same output.

**Do:** count failures, write the count to the artifact store, and fail loudly above
a threshold (say 1% of cells).

### 1.7 — Consolidate Stage 4

The scenario subsystem is ~9,940 LOC with two complete engines. `_resolve_auto_scenario_mode`
picks a mode from whichever artifacts happen to exist, then `_try_run_with_v4_engine`
may return `None` and silently fall through to the legacy `ScenarioSimulator`. A user
cannot tell which engine produced their numbers.

Three `run_*` methods have zero call sites: `run_with_uncertainty`,
`run_with_diffusion_posterior`, `run_bayesian_scenarios`.

**Do:** delete those three and `scenario_diffuser.py`. Pick v4 (it has the mode system
and the tests) and delete the legacy fallback. Keep the four tiers *inside* one
engine.

### 1.8 — Small fixes

- `lambda_surrogate` survives only as a CMA-ES search dimension
  (`sparc/training/cma_es.py:39`) with no effect on the loss. Remove it — the
  optimiser is currently spending budget on an inert knob.
- `mc3.py:373` — `penalty_acyclic` does not penalise cycles; it penalises low-prior
  edges. Rename.
- Fix the `cmd_run` docstring (`__main__.py:263–269`): stages 0 and 1 are swapped.
- README: `numpyro` is not a dependency and appears nowhere. `sparc/causal/nuts.py` is
  a hand-written PyTorch NUTS. Claim it.
- README: retire or reproduce the "0.944 with Laplacian" figure. Laplacian eigenmaps
  are off by default (`enhanced_spatial_cv.py:2280–2284`) and were computed over all
  coordinates including held-out blocks.
- PDE loss carries 12 terms; README says 10; only 8 appear in `_ACTIVATION_SCHEDULE`.
  Reconcile, and ablate the sheaf and fractional Laplacian terms before defending them.
- Move ~250 MB of committed `.gpkg` run outputs out of `templates/forcesmip/output/`.

### 1.9 — Retune MC³

**Where:** `sparc/causal/mc3.py:390–393`

The prior assigns 0.8 to expert-DAG edges and 0.1 to everything else; edges are pruned
at posterior 0.30 (`v2_bayesian_causal.py:1270`). An edge starting at 0.8 essentially
cannot fall below 0.30, so the gate is close to inert and the reported edge
probabilities largely restate the DAG that was fed in.

**Do:** move to 0.5 / 0.15 and report **prior → posterior movement** per edge rather
than the posterior alone. That delta is the actual evidence and it is currently
invisible.

**Also:** with Gaussian observational data and a BGe score, structure learning
identifies the Markov equivalence class — edge *direction* comes from the prior, not
the likelihood. Present MC³ as a coherence check on the expert DAG, not as discovery.

---

## Phase 2 — Calibration

This is the bridge. It closes three findings and it is the precondition for Phases 3 and 4.

### 2.1 — Wire spatial conformal prediction (F10)

**Where:** `sparc/evaluation/conformal.py` — exported from `__init__.py:11`,
referenced nowhere else in the pipeline.

`SpatialConformalPredictor` is already written, correctly motivated (Barber et al.
2023 on conformal prediction beyond exchangeability), and weights calibration
residuals by proximity using the Stage 0 κ. The API is clean:

```python
from sparc.evaluation.conformal import SpatialConformalPredictor

pred = SpatialConformalPredictor(kappa=stage0_kappa)
pred.calibrate(cal_coords, np.abs(cal_residuals))
intervals = pred.prediction_interval(test_coords, y_hat)
empirical = intervals.coverage(y_true)   # ← the number that matters
```

**Do:** hold out one spatial block as a calibration set inside Stage 2, calibrate on
it, and report empirical coverage at a 90% nominal target on the test blocks. Write
`coverage_report` to the artifact store.

**Done when:** every run prints nominal vs empirical coverage, and the README quotes
the empirical number instead of describing MC-Dropout output as "credible intervals."

MC-Dropout gives epistemic uncertainty only — the docstring at `neural_meta.py:447`
says so correctly — and is well documented as overconfident relative to deep
ensembles. Keep it as the point-uncertainty signal; let conformal own the intervals.

### 2.2 — Fix the causal intervals (F3, F4)

- **Bootstrap (F3):** `causal_validation.py:1849–1867` refits a plain
  `LinearRegression`, not the DML that produced the ATE it is reported beside. That
  is why the README's Impervious row shows ATE `+0.018` with a "bootstrap 95% CI" of
  `[+0.020, +0.021]` — an interval excluding its own point estimate. It is also an
  i.i.d. bootstrap over 54,701 autocorrelated cells. **Do:** bootstrap the actual
  estimator, resampling spatial blocks (reuse the geometry from
  `spatial_kfold_enhanced`). Until then, drop the column.
- **E-values (F4):** two implementations, each breaking a different half of
  VanderWeele & Ding. `causal_validation.py:2044` standardises correctly but fits an
  *unadjusted* univariate OLS; `wager2025_addons.py:480` uses the adjusted ATE but
  skips standardisation entirely. **Do:** route both through
  `sparc/causal/sensitivity.py` with the adjusted effect and `d = |ATE·SD(T)|/SD(Y)`.
- **Spatial cross-fitting (F6):** `counterfactual_engine.py:678–691` drops the spatial
  splitter in the sklearn fallback, and `spatial_cate.py:222` (`cv=3`) never had one.
  Thread it through both; record in the artifact when the fallback fires.

### 2.3 — Real thresholds

**Where:** `project.yml:256` — `exceedance_thresholds: [0.25, 0.50, 0.75]`

Those are z-scores. `neural_meta.py:227` builds one sigmoid head per threshold
emitting P(outcome > τ) — the right object, pointed at the wrong numbers. Make the
thresholds physical and health-relevant so the head output means something to a user.

---

## Phase 3 — Groundwater pilot

**Full detail: [`docs/groundwater-pilot.md`](docs/groundwater-pilot.md).**

Validate the core stack on the High Plains aquifer before taking on forecast
integration. This was inserted ahead of the heat pilot deliberately — it is cheaper,
it exercises *more* of the pipeline, and it de-risks Phase 4.

Why it comes first:

- **Tier 2 runs natively.** Steady-state groundwater flow is `K∇²h = −R`; the solver
  in `sparc/physics/pde_solver.py` is `α∇²T = S`. The mapping is exact (α↔K, T↔h,
  S↔−R) and `poisson_solve` runs unmodified. Tier 2 has never been exercised on
  physics it natively describes.
- **α becomes falsifiable.** Nobody measures urban thermal diffusivity, so the learned
  α field cannot be checked in the heat domain. K is a real, independently measured
  parameter with published regional estimates. This is the one place the
  learned-parameter-field thesis can be tested before Phase 6 depends on it.
- **Ground truth is dense and free.** 8,000–9,400 USGS monitoring wells measured
  annually, plus a published gridded water-level-change raster. Compare with ~5–15
  ASOS stations per city.
- **Stage 3 gets an analytical check.** The Theis solution gives closed-form drawdown
  for a given pumping rate. No other SPARC domain lets you check a causal estimate
  against a known answer.
- **New data regime.** ~8k sparse points is the opposite of UHI's 54,701. Expect the
  neural path to lose to kriging here — that maps the boundary of where the
  architecture applies, which is currently unknown.

Two things to fix before starting, both in `templates/groundwater/`:

1. **The target is the wrong quantity.** `target_column: "gw_level_m"` is depth below
   ground surface. The flow equation is in hydraulic head, `h = surface_elevation −
   depth_to_water`. Depth-to-water is dominated by topography and does not satisfy the
   Laplacian. Every PDE term would fit the wrong surface.
2. `grid_resolution_m: 30` over the High Plains extent is ~500 million cells. Use 1000.

Kill criteria, baselines, data inventory and the full stage-by-stage walkthrough are
in the linked document. The headline: **arm B is kriging-with-external-drift, and that
is the real null.** Beating naive kriging while losing to KED means the pipeline is an
expensive interpolator.

---

## Phase 4 — Heat EWS pilot

**Thesis:** we do not forecast the atmosphere. NOAA does that, and we will not beat
them. Our job is the last mile — downscale their forecast to a 30 m grid using learned
land-cover physics, and convert it to calibrated exceedance probability.

**The null hypothesis we have to beat:** bilinearly interpolated HRRR. This is a
genuinely hard baseline — HRRR runs a WRF urban canopy scheme, so it is *not*
UHI-blind, as your own `scripts/compare_era5_hrrr.py` notes. If we cannot beat
bilinear HRRR at station locations, the thesis is dead and we should say so.

### 4.1 — Scope

One city, one season, one hazard. **Philadelphia** — you already have
`output/cities/philadelphia_pa/stations.parquet` and the collection tooling.

> ⚠️ Philadelphia is currently the LOCO holdout in `configs/multicity_pilot.yml`. If
> you want to keep using it for transfer evaluation, either rotate the LOCO holdout to
> another city first, or accept that the EWS pilot burns it. Decide before 3.4.

### 4.2 — Data

| Layer | Source | Status |
|---|---|---|
| Forecast | HRRR 3 km, hourly, 18 h — AWS `noaa-hrrr-pds` via NODD, using `herbie-data`, or the `hrrrzarr` Zarr mirror | **new** |
| Ground truth | ASOS via Iowa Environmental Mesonet | already built — `scripts/fetch_open_meteo_stations.py` |
| Static predictors | canopy, impervious, NDVI, albedo, elevation | already built — `sparc/data/collect/` |
| Threshold basis | NWS/CDC **HeatRisk** (0–4, CONUS gridded, experimental since spring 2024) | **new** |

Use HeatRisk rather than an ad-hoc heat-index cutoff: it already folds in
climatological unusualness, duration, overnight temperatures, and CDC health-impact
data — which is precisely the framing an EWS needs, and it makes the output legible
to the agencies who would use it.

### 4.3 — Event catalogue

Select 3–5 historical heat events for Philadelphia from the NWS event archive and IEM,
plus a matched set of hot-but-not-extreme control days. **Do not pick events by
eyeballing the data you will validate on** — define the selection rule (e.g. HeatRisk
≥ 3 over ≥ 2 consecutive days) and take whatever it returns.

### 4.4 — Model

Change the signature from `(features, coords) → T` to
`(features, coords, HRRR fields at t+h) → P(exceed at t+h)`.

- Widen `neural_meta.py:148` — `nn.Embedding(3, time_embed_dim)` is hardcoded to three
  slots (morning/midday/night). Replace with a continuous or Fourier time encoding.
- Promote `sparc/run/spatiotemporal_cv.py` (210 LOC, currently peripheral) to the real
  fold factory. **Blocking must be spatial *and* temporal** — adjacent hours are far
  more correlated than adjacent cells, and a random temporal split will produce a
  beautiful, meaningless score.
- Populate `nocturnal_dT_dt`. It is threaded through `joint_loss.py:180` →
  `loss.py:243` → `pde_loss.py:129` and **never supplied by any call site**, so the
  nocturnal radiative-cooling term is dead. HRRR hourly gives you the overnight
  cooling slope. This is also the term that makes α identifiable — in `α∇²T − S`, α has
  units of m²/s, i.e. thermal diffusivity, and diurnal cooling curves are the classical
  way to estimate it.

### 4.5 — Metrics

Accuracy is not a metric here — a classifier that always says "no" scores well on a
rare event and is worthless.

- **Discrimination:** POD, FAR, CSI at each lead time (6 / 12 / 24 h)
- **Calibration:** Brier score, Brier skill score vs climatology, reliability diagram
- **Headline:** the lead-time vs false-alarm-ratio curve — this is the plot that
  decides whether anyone can use it
- **Spatial:** does skill hold at the block-group level, or only city-wide?

Compare three arms:

1. bilinear HRRR at station points (**the null**)
2. HRRR + static land cover, no physics constraint
3. full SPARC downscaling with PDE constraint and conformal calibration

Arm 2 is the important control. It separates "land cover helps" from "our physics
helps," and without it a win over arm 1 tells you nothing about whether the PDE
machinery earns its place.

### 4.6 — Kill criteria

State these before running, and honour them.

- Arm 3 does not beat arm 1 on CSI at 12 h → **thesis dead**, write it up, stop.
- Arm 3 does not beat arm 2 → land cover is doing the work, the physics is not. Keep
  the downscaling, delete the PDE claim.
- Empirical coverage misses nominal by more than 10 points → not shippable as a
  warning system at any skill level.

### 4.7 — Cheap precursor (do this first)

Before touching HRRR: pull the hourly ERA5 you *already fetch* and currently collapse
to a daily mean (`sparc/data/collect/era5.py` — `era5_t2m` is the daily mean), feed the
overnight cooling slope into `nocturnal_dT_dt`, and see whether the α field sharpens
against station observations. That is roughly a day of work and it tells you whether
the diurnal axis buys anything at all before you take on a gridded-forecast
integration.

---

## Phase 5 — Response layer

**Conditional on Phase 4 clearing its kill criteria.** This is the differentiator —
every EWS on the market stops at the alert.

Compose three things that already exist:

- exceedance heads (`neural_meta.py:227`) → who crosses the threshold
- `sparc/data/census_equity.py` (720 LOC, per-cell tract-level ACS vulnerability) →
  who is exposed
- the budget-constrained allocator in Stage 4 → what to do about it

Target output:

> Heat emergency likely Thursday 2–7pm across 40 block groups (P = 0.82, empirical
> coverage 0.89 at nominal 0.90). 11,000 residents over 65 without air conditioning
> inside that footprint. Opening these 6 cooling centers covers 78% of them.

Note that this is also the **only** place the causal stack earns its keep in an EWS.
Prediction does not need causality; the response recommendation does. Stage 3 is
otherwise idle in this product, and that is fine — just do not claim otherwise.

---

## Phase 6 — Simulation cascade (design note only)

**Not active work.** Recorded so the architectural decision is pinned while it is
fresh, and so it does not get rebuilt wrong later. Nothing here starts before Phase 3
and Phase 4 report.

The idea: a calibrated trigger fires on a footprint, and a high-fidelity forward
simulation runs on *just that footprint* — flood inundation, fire spread — producing
a far more precise picture than the screening model can.

### 6.1 — The decision: orchestrate, don't own

`sparc/physics/pde_solver.py` is a 126-line **steady-state** Jacobi/SOR solver for
`α∇²T = S`. It has no time-stepping. Flood spread is 2D shallow water (Saint-Venant);
fire spread is Rothermel rate-of-spread with a level-set or cellular front. Those are
moving-front, time-evolving problems — different mathematics, not a parameter change.

Mature validated open solvers already exist: LISFLOOD-FP and HEC-RAS 2D for flood;
ELMFIRE, Cell2Fire, WRF-SFIRE and ForeFire for fire. **Do not write our own.** Two
reasons, and the second is decisive:

1. We will not beat them on fidelity.
2. **Agencies require validated models.** HEC-RAS is the USACE standard for regulatory
   floodplain work. A bespoke solver, however good, has no path into the market this
   product would sell to.

### 6.2 — What SPARC contributes to the cascade

The solvers are free. The unserved parts are the layers around them:

| Layer | SPARC provides |
|---|---|
| Where / when to run | calibrated exceedance heads + conformal — the trigger |
| Initial & boundary conditions | downscaled to the analysis grid, not the forecast grid |
| **Which runs to make** | parameter posterior → solver ensemble, not one deterministic run |
| Solver parameters | learned spatially varying fields (α ↔ Manning's *n*, fuel moisture) |
| What it means | exposure + budget allocation on the output |

Row three is the differentiator. People do run these solvers as ensembles — there is
published work sweeping dozens of LISFLOOD-FP configurations over roughness, channel
width and discharge — but the perturbations are ad hoc. Nobody drives them from a
calibrated posterior. The product is *a calibrated ensemble wrapper around validated
physics*.

Row four is why Phase 3 matters: the α↔K result in `docs/groundwater-pilot.md` §7.4
either supports or kills the learned-parameter-field claim this row depends on.

### 6.3 — Prove the cascade on heat first

Before swapping in any external solver, demonstrate the *pattern* end to end using
physics already in the repo:

> exceedance head trips on a block group → Tier 2 `poisson_solve` fires on just that
> footprint at high resolution → ensemble over the NUTS posterior → exposure map

That exercises triggering, sub-domain extraction, boundary conditions from the coarse
field, ensemble orchestration, and hand-off to the exposure layer — with no new
physics. If it works, swapping `poisson_solve` for an ELMFIRE or LISFLOOD-FP
subprocess is an adapter, not a rewrite.

### 6.4 — On drone-tasked survey

The "fly LiDAR at the emergency and simulate" version does not survive contact:

- **3DEP is at 99% coverage as of FY25.** The baseline LiDAR already exists nationally.
- **Timing.** Task → fly → process point cloud → bare-earth DEM → simulate is not a
  minutes-scale loop. Fire and flash flood move faster.
- **Airspace.** Wildfires get TFRs, and drone incursions ground air tankers.
- **Liability.** "Who NEEDS to evacuate" is the highest-stakes output in the space. A
  model that says *you don't need to go* and is wrong kills someone. Evacuation orders
  are a legal authority held by emergency managers; the defensible output is a ranked,
  uncertainty-aware prioritisation supporting that decision — never a binary, never
  with the human removed.

**The version that does survive: value-of-information survey tasking.** Don't task
surveys during the event — task them *before the season, where the model is most
uncertain*. Three existing instruments already identify those locations:

- `sparc/interventions/extrapolation_guard.py` — Mahalanobis distance, where we are
  outside the training distribution
- conformal interval width — where predictions are least certain
- the CATE-vs-GWR divergence audit — where identification itself is shaky

Survey the top-N uncertainty cells, re-run, repeat. The model says where to look,
looking reduces uncertainty, the better model says where to look next. Novel,
fundable, and not life-safety-critical. It also applies directly to groundwater —
"where should we drill the next monitoring well" is the same question asked by people
with budgets, and Phase 3 gives a cheap place to test it.

---

## Not doing

Explicitly out of scope, recorded so it does not get relitigated:

- **Hazard detection from imagery.** GOES already ships operational fire detection;
  NWS already issues heat warnings. A CNN beats this pipeline at "is the thing there,"
  and those products are free.
- **Building a weather forecast model.** We downscale someone else's.
- **Geostationary LST for intra-urban UHI.** ABI thermal bands are 2 km; the UHI work
  is 30 m. Landsat/GOES fusion at 30 m/5 min exists in the literature (~2 K RMSE) but
  it is a research project, not a pilot. Revisit only if Phase 4 succeeds and the
  bottleneck turns out to be temporal resolution rather than skill.
- **Real MGWR via back-fitting.** At 54,701 points it is likely infeasible without a
  fitting-point subsample. See 1.5 — we rename instead.
- **Writing our own flood or fire solver.** See Phase 6.1. Validated open solvers
  exist, we will not beat them, and agencies require validated models. If the cascade
  is ever built we orchestrate.
- **Drone-tasked LiDAR during an event.** See Phase 6.4. 3DEP is already at 99%
  coverage, the timing does not close, and wildfire TFRs ground drones. The
  value-of-information tasking variant stays open.
- **Any product that outputs a binary evacuate / do-not-evacuate.** The defensible
  output is ranked and uncertainty-aware, supporting an emergency manager's legal
  authority rather than replacing it.
- **Operational deployment.** An EWS is a service with uptime and liability. That is a
  different company than a desktop research app, and it is a decision for after the
  pilot reports.

---

## Finding index

Defect IDs referenced above, from the review at `0a04b7a`.

| ID | Summary | Phase |
|---|---|---|
| F1 | Surrogate pre-training targets are in-sample fits, sliced inside the fold | 0.1 |
| F2 | `--resume` only skips Stage 0 | 0.3 |
| F3 | "Bootstrap 95% CI" comes from a different estimator than its ATE | 2.2 |
| F4 | Both E-value paths invalid, in complementary ways | 2.2 |
| F5 | √ taper amplifies interventions; infinite marginal return at the knee | see note |
| F6 | Spatial cross-fitting lost in DML fallback; absent in CATE path | 2.2 |
| F7 | Block IDs collide on the domain's upper edge | 1.8 |
| F8 | Feature scaler and `y_mean`/`y_std` fit on full dataset | document |
| F9 | `stratify_y` balances folds by outcome mean | document |
| F10 | Conformal prediction and coverage modules never called | 2.1 |
| F11 | Test suite never runs in CI | 0.2 |
| F12 | README/code drift (numpyro, Laplacian, stage docstring, MC-Dropout count) | 1.8 |

**Note on F5.** `scenario_simulator.py:349–371` implements
`effective(Δ) = τ + √(Δ−τ)·√τ`, whose derivative at the knee is `+∞` and which
returns `effective(Δ) > Δ` for all `τ < Δ < 2τ` — with the default `τ = 10`, a +11 pp
canopy increase is simulated as +13.2 pp. A C¹-continuous concave replacement with
slope exactly 1 at the knot is `τ + 2τ(√(1 + (Δ−τ)/τ) − 1)`.

This is not scheduled above because it belongs to the intervention-scenario product,
not the EWS path. Fix it before publishing any further Stage 4 dose–response numbers.
Separately, the √ shape contradicts Ziter et al. (2019, *PNAS*) — the paper cited in
`physics_priors.py:112–120` for the canopy prior — which found daytime cooling to be
nonlinear with the *greatest* cooling above 40% canopy cover, i.e. accelerating rather
than diminishing.
