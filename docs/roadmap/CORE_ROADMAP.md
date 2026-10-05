# SPARC Core Roadmap

**Status:** P0–P6 implemented in `sparc/core/` (single city; see Appendix C) · P7 not started · **Date:** 2026-09-30 · **Scope:** the predictive core (urban air temperature / cooling)
**Line references are pinned to `pi-jepa-dev` @ `0a04b7a`.**

> Supersedes `docs/roadmap/SPARC_Integration_Status.md` as the source of truth for what is wired.
> Related: `docs/adr/0002-jepa-rebuild-scope.md`.

---

## TL;DR

- The core approach is sound. What drifted is the **wiring** and the **validation**:
  - the stages are computed but not connected;
  - several pieces are silently broken;
  - the recent PI-JEPA work evolved into a second, disconnected system.
- **The main reorder:** the trained physics-informed model becomes the scenario engine. Causal inference moves after it as a validator; it no longer runs as a parallel engine that feeds coefficients into scenarios.
- **Sequence:** one city until the pipeline works, then few-shot, then zero-shot. PI-JEPA is rebuilt as a real spatial JEPA and is **gated**: it is kept only if it beats simpler neighbourhood features on held-out cities.
- **The target core is about 12–15k lines,** down from about 36k core lines inside a roughly 131k-line repo.

## Goals (unchanged)

1. **Area of influence.** An anisotropic correlogram gives the spatial range and direction over which each variable influences air temperature.
2. **Prediction.** Geographically weighted base models feed a **physics-informed neural network** that makes the final prediction, with PI-JEPA as an optional representation layer.
3. **Saturation.** Per-variable dose-response curves: where cooling benefit saturates, and where it is largest.
4. **Scenarios.** Predicted air temperature under adaptation strategies and new conditions.

---

## 1. Diagnosis: what actually happens today

The full file:line index is in [Appendix A](#appendix-a--defect-index).

### 1.1 Two disconnected systems
- **(A) `sparc run`** (`sparc/__main__.py`): Stage 0 correlogram → Stage 2 GW base models + neural stacker + PDE loss → Stage 3 causal → Stage 4 scenarios. It runs on the Brown/Providence data (`brown4.csv`).
- **(B) `scripts/train_multicity_jepa.py`** (2,737 lines): multi-city JEPA trunk with per-city heads, evaluated with Philadelphia left out.
  - It uses **no GW base-model predictions and no PDE loss**.
  - It runs Stage 0/2/3 per city, but ignores the outputs and swallows errors.
- The only links between them are adapters (`sparc/inference/trunk.py`, `sparc/causal/jepa_trunk_adapter.py`), and those are not wired into training.

### 1.2 Goal 1, area of influence: computed, then dropped
- **Only one quadrant of offsets is used.** The FFT correlogram keeps offsets with dx, dy ≥ 0, so directional estimates are invalid. The 1/√pairs standard error also marks nearly every lag significant.
- **"Range" means three different things:** Moran's-I zero-crossing, Matérn κ, and cross-range peak.
  - GWR bandwidths use each predictor's *own* autocorrelation range, not its range of influence on temperature.
  - The cross-range only fills gaps.
- **Anisotropic weighting is effectively off.**
  - GWR's anisotropic weighting silently collapses to uniform during CV: numpy inputs produce `feature_i` names that don't match the kernel field.
  - GWRF's anisotropy branch is dead code.
- **No feature uses a variable-specific radius.** Spatial lag is a fixed k=8 nearest-neighbour average, and the CV block size is a hard-coded 300 m user value that overrides the correlogram.

### 1.3 Goal 2, base models → NN stacker: leakage and unit bugs
- **The stacker does not stack out-of-fold predictions.** Differentiable surrogates are trained toward **in-sample full-data fits**, so the neural out-of-fold R² (0.944 in the README) is likely optimistic.
- **Coordinate units are mixed.** The full-data refit uses coordinates in **feet** (EPSG:3438), while CV and the bandwidths use **metres**. The GWR that is evaluated (uniform weights) is not the GWR that is deployed (anisotropic weights).
- **Failures are silent.** Failed folds are filled with `mean(y)`, and `n_splits=5` is hard-coded.

### 1.4 Physics: mostly inert
- **The live constraint is weak.** It is a Poisson equation with a free source, `α∇²T − S = 0`, on **z-scored** T. `SourceTermNet` is a free MLP, so it can absorb the constraint.
- **α is effectively constant.** It is divided by its own mean. The learned α sits at 0.21–0.40 against configured bounds of 0.5–12.
- **There is no advection term.** ERA5 wind never enters the physics.
- **Several loss terms are always zero or wrong:**
  - the "directional" term is always zero;
  - the "anisotropy" term *penalizes* anisotropy;
  - the "gradient-flux" term is plain smoothing.
- **Written but never used:**
  - `energy_balance_residual` has no caller, and its sensible-heat formula is actually ground conduction;
  - the transient, nocturnal and fractional terms are never activated;
  - the sheaf term has a shape mismatch.

### 1.5 PI-JEPA: not a spatial JEPA
- **No spatial information flows.**
  - The trunk is a **per-pixel MLP**.
  - Masked rows are set to zero, and each pixel predicts its **own** unmasked embedding.
  - Batches are 4,096 random pixels drawn across 22 cities.
- **Consequences:**
  - Mask shape cannot matter, so the I1/I2/B6 verdicts and ADR-0001 only hold for this architecture (see ADR-0002).
  - The "physics-informed" part is an auxiliary head predicting `1 − albedo`, which is already an input.
- **Evaluation problems:**
  - The keep-threshold σ = 0.024 came from `--skip-pretrain` runs, which measure head-seed noise only. It was then used to judge trunk changes, whose own σ is about 0.34 (B6).
  - Few-shot uses random, non-blocked pixel splits on CAPA *area-wide* rasters, which CAPA interpolated with a random forest.
  - About 50 `trainNN` iterations were tuned against Philadelphia, so it can no longer serve as an unbiased test.
- **Data problems:**
  - Campaign dates were chosen by matching ERA5 to CAPA, which is circular. Philadelphia's own 2022-09-21 date is suspect and could explain the persistent −8 °F morning bias.
  - `land_cover` is fed as a number.
  - Aspect and wind direction are treated as linear, not circular.
  - `cdc_svi` (a social index) is used as a physical predictor.
  - `lst` and `ndvi` are *mediators* of the interventions, so they cannot be held fixed in scenarios.
  - ERA5 hour windows ignore daylight saving time.

### 1.6 Goal 3, saturation: assumed, not learned
- **The README's "diminishing returns beyond ~15 pp" is a hand-set √ taper** (`caps.yml` thresholds), not a fitted curve.
- **The one real curve fitter (GWRF PDP) is disabled** (`skip_pdp = True`).
- **The causal PDP is linear by construction,** `response = mean(τ)·(dose − median)`, so it can never find a knee.
- **The v4 saturation clip has a unit mismatch:** it compares an absolute dose with an increment.
- **Per-cell curves are thrown away,** so there is no map of *where* saturation happens.

### 1.7 Goal 4, scenarios: bypass the trained model, and crash
- **Scenarios don't use the model.** They multiply causal coefficients by heuristics. Re-prediction exists only in legacy modes 3/4, using the V1 base-model consensus, **never the physics-informed NN**.
- **`sparc run` Stage 4 raises `NameError`:** `args` is not in scope.
- **The default mode cannot run:** `mode_5` can never dispatch.
- **An artifact name mismatch blocks re-prediction:** the code looks for `standard_meta_ensemble`, but Stage 2 writes `final_meta_ensemble`.

### 1.8 Scope and hygiene
- **Dead and duplicated code.** About 6k lines of modules have zero importers. There are four bandwidth resolvers, three KernelField builders and three Stage-4 entry points.
- **Duplicated templates and tracked junk.** `templates/` is a byte-identical duplicate of `sparc/templates/` and adds about 520 MB of tracked ForceSMIP outputs. `.vs/` and `.vite/` are tracked.
- **Tests don't cover the core.** 54 test files exist, but no CI runs them. The correlogram, GWR math, saturation and scenario math have no tests.

---

## 2. Target architecture: one pipeline, reordered

```
S0 Data + QA        independently verified campaign dates, traverse-point labels,
                    ΔT target (°C, vs ERA5 background), ERA5 radiation / wind / BLH
S1 Influence        full-plane directional cross-correlogram T↔X → per-variable influence kernel
                    (range r_j, anisotropy behind a quality gate) → focal features X_j^(r_j)
                    + physics length-scale prior
S2 Base models      OLS | MGWR | GWRF | GAM | PHYSICS model → true spatial-block out-of-fold predictions
S3 Stacker          NN(out-of-fold base preds, focal features, [JEPA embedding slot]) + consistent PDE loss
                    → final ΔT + uncertainty (fold ensemble + spatial conformal)
S4 Response         ICE sweeps through S1 re-featurisation → per-cell saturation curves → maps
S5 Scenarios        perturb inputs → recompute focal features (spillover) → re-predict S3 → ΔT maps ± UQ
S6 Causal validate  spatial DML / CATE / doubly-robust dose-response / spillover / sensitivity
                    → gates S4–S5 outputs
S7 Optimize         budget + equity allocation on validated S5 benefit surfaces
```

### 2.1 Physics formulation (replaces the free-source Poisson equation)

This is a linearized canopy-layer heat budget for the anomaly ΔT = T_air − T_bg, where T_bg is the ERA5 2 m temperature for the window:

```
u·∇ΔT − K∇²ΔT + ΔT/τ = Q_H(x) / (ρ c_p h)

Q_H  = Q* + Q_F − Q_E − ΔQ_S
Q*   = (1 − albedo)·SW↓·(1 − shade(canopy, SVF)) + LW_net(SVF)        ERA5 ssrd / strd
Q_E  = EF(ndvi, canopy, impervious) · (Q* − ΔQ_S)
ΔQ_S = OHM: a1·Q* + a2·dQ*/dt + a3, by surface type                   night: stored-heat release
```

- **Learned:** a few scalars per city and window. These are K (horizontal eddy diffusivity), τ (vertical-exchange relaxation time), the canopy-layer wind fraction, and the OHM/EF coefficients, each bounded by literature values.
- **From ERA5:** u (10 m wind vector) and h (boundary-layer height).
- **Solution method.** With constant coefficients, ΔT = **G ⊛ Q_H**, where G is a wind-skewed Bessel-K₀ Green's function with length scale √(Kτ). It runs as an FFT convolution on the raster, so it is cheap and differentiable.
- **Why this ties the goals together:**
  - **Goal 1 ↔ physics.** By the Whittle–Matérn (ν = 1) SPDE link, the correlogram range of the target residual gives a prior on √(Kτ). Anisotropy direction should align with the wind, which is a testable check that replaces the unreliable θ estimates.
  - **A label-free physics base model** in S2, which is the foundation for zero-shot.
  - **The spillover mechanism** in S5: a local change in canopy or albedo changes Q_H locally, and G spreads the cooling to neighbours.
- **PDE loss in S3.** The residual of the *same* operator on the NN output, with fitted K and τ and a known Q_H. Units are consistent and there is no free source.

---

## 3. Phased roadmap

**Effort key:** S ≈ days, M ≈ 1–2 weeks, L ≈ 3+ weeks. **Every phase ships with tests and CI.**

### P0: Stabilize and establish ground truth (M) — *nothing downstream is trustworthy until this is done*
1. **Freeze and tag** `main` and `pi-jepa-dev` as reference.
2. **Dev city (decision D1).**
   - Pick a CAPA city with morning, midday and evening windows and traverse points, whose campaign date can be confirmed **independently of ERA5**. Candidates are Chicago (2023-07-28) and Raleigh (2021-06-17); both are currently only ERA5-matched.
   - Document the provenance of `brown4.csv` and its z-score definition, then keep it as a legacy benchmark.
   - Create a **locked final-test set** of Philadelphia plus 2 more cities, never used for tuning.
3. **Labels.**
   - Supervise on CAPA **traverse points**, not the random-forest-interpolated area-wide rasters.
   - Take dates from CAPA metadata or traverse timestamps, never from ERA5 matching, and re-verify Philadelphia.
   - The target is ΔT in °C, one model per time window, with timestamps DST-correct.
4. **Features.**
   - One-hot or embed `land_cover`; use sin/cos for aspect and wind direction.
   - Move `cdc_svi` out of the predictors and into equity weighting in S7.
   - Tag `lst` and `ndvi` as mediators: exclude them from the scenario model, or model them as a chain.
   - Add ERA5 `ssrd`, `strd`, `blh` and `u10`/`v10`.
5. **Evaluation protocol.** Write it down before any modelling:
   - Spatial block CV with block size ≥ the target-residual range, plus a buffer; nested CV for hyperparameters.
   - Metrics: RMSE/MAE (°C), R², spatial correlation, and interval coverage.
   - σ from ≥ 3 seeds with **full retrain**.
   - One declared primary metric per phase.
6. **CI** runs pytest on every push and includes a **synthetic-city fixture**: an analytic field with a planted influence range, anisotropy and saturation, which every stage must recover.

### P1: Area of influence, goal 1 (M)
- **One correlogram module.** Full-plane FFT ACF/CCF, directional bins at 0°/45°/90°/135°, and standard errors from a spatial block bootstrap.
- **One definition of influence range.** r_j is where the target↔X_j cross-correlogram of **model residuals** decays to noise, reported with a CI. The Matérn fit is kept only for the target residual, where it becomes the physics prior.
- **Anisotropy quality gate.** θ is used only if its CI is narrower than 30° and it is consistent with the wind direction; otherwise the variable is treated as isotropic.
- **Outputs:**
  - an influence-kernel table (r_j, eccentricity, θ, CIs);
  - **focal features**: kernel-weighted means at {r_j/2, r_j, 2r_j};
  - the S2 block size.
- **Exit criteria:** recovers the planted ranges on the fixture, and focal features beat raw per-pixel features in S2 CV.
- **Port** `cross_correlogram.py`, `correlogram_matern_fit.py`, `anisotropy.py`. **Drop** `scale_hierarchy`, `memory_efficient_spatial_analysis`, `matern_fitter`, `bandwidth_advisor`, `pipeline_configurator`, and the duplicate KernelField builders.

### P2: Base models with honest stacking, goal 2a (M)
- **Base models:**
  - OLS on focal features;
  - **real MGWR** (the `mgwr` library is already a dependency) with per-variable bandwidths initialised from r_j and metres everywhere;
  - GWRF;
  - GAM;
  - the **physics model** (§2.1).
- **Stacking inputs:** true spatial-block out-of-fold predictions only. **Remove the differentiable surrogates.**
- **A failed fold is a hard error.**
- **Exit criteria:** per-model out-of-fold metrics on the dev city, plus a report of the variance the physics model explains with zero in-city labels.

### P3: Physics-informed NN stacker, goal 2b (M–L)
- **Inputs:** out-of-fold base predictions, focal features, and an optional embedding slot that stays empty until P7.
- **Output:** **ΔT = physics prediction + NN residual**, with the residual regularized toward zero. A small MLP or gated mixture is enough.
- **Loss:** the data term plus the §2.1 PDE residual. **Delete** the no-op, inverted, smoothing and never-activated PDE terms, and the SIREN, sheaf and fractional code.
- **Uncertainty:** a deep ensemble across CV folds plus split-conformal calibration on spatial blocks (`sparc/evaluation/conformal.py`).
- **Freeze:** CMA-ES, EWC/continual/optimal-transport, meta-λ, exceedance heads and VSBA.
- **Exit criteria:** beats the best base model out-of-fold by more than seed σ, with 90% interval coverage between 0.85 and 0.95.

### P4: Saturation surfaces, goal 3 (M)
1. **Per-cell curves.** For each actionable variable (canopy, impervious, albedo, building coverage/height, SVF), sweep the dose **through S1 re-featurisation**, so neighbours' focal features update too. Re-predict with S3 to get per-cell ICE curves.
2. **Fit a saturating curve** per cell (or per cluster where cells are noisy): ΔT(d) = A·(1 − e^(−(d − d₀)/d_s)). Fall back to logistic or piecewise-linear if either fits better.
3. **Maps:**
   - **max achievable cooling** A;
   - **saturation dose** d_sat (the dose reaching 90% of A);
   - **headroom** d_sat − current dose;
   - **marginal benefit** ∂ΔT/∂d at the current dose.

   Together these answer *where each variable is most beneficial*.
4. **Remove** the √ taper, the v4 clip and the linear causal PDP. Guardrails become physical bounds plus an extrapolation flag.
5. **Exit criteria:** recovers the planted saturation on the fixture, with CIs on d_sat that are stable across the fold ensemble.

### P5: Scenario engine and optimizer, goal 4 (M)
- **One engine, about 500 lines.** It replaces `scenario_simulator` (4.2k lines), v4, `causal_stack` and the selector.
  - Pipeline: spec → constrained input edits (bounds, canopy + impervious ≤ 100%) → recompute focal features (spillover) → S3 plus physics → ΔT map with UQ plus a Mahalanobis extrapolation flag (reuse `extrapolation_guard.py`).
- **New weather conditions** mean swapping the ERA5 forcing. These are flagged **extrapolative in single-city mode**, because one campaign day cannot identify a weather response. Real weather generalization arrives in P7.
- **Optimizer.** Port `sparc/scenario/budget.py` (greedy/MILP, Pareto sweep) and the equity weighting (with SVI). It runs on validated S5 surfaces.
- **Exit criteria:** S5 results equal the S4 curves at matching doses.

### P6: State-of-the-art causal validation (M–L)
**Core methods:**
- **DML** (EconML) with **spatially blocked cross-fitting** and spatial-confounding adjustment. `spatial_residualizer.py` belongs here, consistent with the runlog's A3 conclusion.
- **CATE** with `CausalForestDML`, to show *where* effects differ.
- **Doubly-robust continuous-treatment dose-response** (Kennedy et al. 2017; Colangelo & Lee 2020). These give **causal saturation curves with CIs**, compared against the model-based curves from P4.
- **Spillover estimation** via exposure mapping on the S1 kernels. This validates the physics kernel; reuse ideas from `interference.py`.
- **Sensitivity:** E-values plus omitted-variable-bias bounds (Cinelli & Hazlett 2020), and the DoWhy refutations.
- **Model-vs-causal audit** (evolved from `divergence_audit.py`). When the NN's implied effect disagrees with DML/CATE beyond its CI, the S4/S5 output is **flagged**. This is the gate.

**DAG structure learning:**
- MC³, DiBS and order-MCMC are three solvers for the *same* problem, a Bayesian posterior over DAGs. Running all three adds redundancy, not rigour.
- With 6–10 spatially autocorrelated variables the structure is weakly identified, and physics largely dictates the DAG anyway.
- **Keep one sampler** (MC³, which is already wired, or order-MCMC) as a **DAG audit** on spatial-block bootstrap resamples. It flags unexpected or missing edges against the expert DAG but does not drive predictions.
- Archive DiBS and the rest. NUTS per-edge posteriors become an optional Bayesian cross-check.

### P7: Multi-city — few-shot, then zero-shot, then PI-JEPA, gated (L)
1. **Scale S0–S6** to the verified cities, using the same code and a city ID.
2. **Few-shot.**
   - A global stacker trained on the other cities, plus the physics model (no labels needed).
   - **GW models act as the local residual correction** fitted on the few labels. This is their clean multi-city role.
   - N-curves use spatially blocked splits.
3. **Zero-shot.**
   - Physics model plus the global NN, plus a city-offset regression on ERA5 and city morphology (a generalization of Option-C).
   - Evaluate with rotating leave-one-city-out on at least 5 cities; score the locked test set once per milestone.
4. **PI-JEPA, rebuilt** (see ADR-0002).
   - **Encoder and objective:** a ViT/CNN on raster tiles (one city per batch) with I-JEPA context → target prediction, positional mask tokens, and loss only on masked targets.
   - **Pretraining data:** unlabeled rasters from **100+ US cities** (NLCD, Landsat, Sentinel-2 and buildings are nationwide). This scale is JEPA's real advantage.
   - **Where the physics comes in:**
     - (a) target-block size is scaled to the S1 range and elongated along the wind;
     - (b) auxiliary heads predict **non-input** physics fields: the physics-model ΔT (free pseudo-labels in unlabeled cities) and held-out LST.
   - **Plugs into** the S3 embedding slot.
   - **Kept only if** it beats focal features in rotating leave-one-city-out by more than full-retrain σ.
5. **ANP** (optional) is used for station conditioning, trained *with* coordinates and a Matérn prior, and only with in-city (non-airport) sensors.

---

## 4. Keep / rewrite / freeze

| Area | Action |
|---|---|
| `data/collect` (landsat, nlcd, sentinel2, dem, buildings, era5, capa, boundary, assembler_multicity, http_client) | **Port and fix**: traverse points, radiation vars, DST, date provenance |
| Correlogram (`cross_correlogram`, `correlogram_matern_fit`, `anisotropy`, `spatial_autocorr_comprehensive`) | **Rewrite** into one `influence/` module (~1.5k lines) |
| `models/ols, gwr, gwrf, ggpgam` | **Port**: real MGWR in metres; fix the CV feature-name bug; remove dead anisotropy code |
| `physics/` | **Rewrite**: energy balance + advection–diffusion–relaxation kernel/solver; keep the 5-point stencils, add upwind advection |
| `v2_neural_training` (4.6k), `neural_meta`, `surrogates`, `process_rate_net`, `training/loss` | **Rewrite** as a slim stacker (~800 lines) without the surrogates |
| `interventions/*` (9.8k), `scenario_engine_selector`, `causal_pdp` | **Replace** with S4 response + S5 engine (~1k lines); port `extrapolation_guard` and the caps |
| `sparc/scenario/budget.py`, `sparc/decision/equity.py` | **Port** |
| Causal: `cate_validation`, `spatial_cate`, `spatial_residualizer`, `sensitivity`, refutations, `divergence_audit`, one DAG sampler | **Port and upgrade** (P6) |
| DiBS, the other DAG samplers, `iv`, `panel`, `dynamic`, `wager2025_addons`, `mediation` (revisit), `nuts` (optional) | **Freeze** |
| JEPA/ANP/continual/transfer (`training/jepa_*`, `ewc`, `replay`, `inference/*`, `scripts/train_multicity_jepa.py`) | **Freeze** as reference; rebuild in P7 |
| GWEN, VSBA, CMA-ES, meta-λ, `scale_hierarchy`, `latent_rollout`, `scenario_diffuser` | **Freeze** |
| `server/`, `sparc-desktop/`, `supabase/`, `report/`, 12 non-UHI templates, `.agents/` | **Freeze** (re-attach later through an adapter) |
| `registry/` (artifacts.db) | **Simplify** to run directories (parquet + JSON manifest) unless the desktop app returns |
| `.vs/`, `.vite/`, duplicate `templates/` + ~520 MB outputs, root `brown4.csv` | **Delete or move** (data goes in `data/` or a release asset) |

## 5. Repo decision (open, decision D2)

The evidence favours a **new lean repo** (for example `sparc-core`):
- `pi-jepa-dev` already has no shared git history with `main`.
- Most of the code is peripheral or dead.
- Hundreds of MB of tracked outputs would otherwise come along.
- Nearly every core module needs a rewrite rather than a patch.

**If you go with a new repo:** port module by module in phase order P0 → P6, each with tests and the synthetic fixture, and keep this repo read-only as reference.

**If you prune in place instead:** use a `core-cleanup` branch cut from `main` and fix these first:
1. the Stage-4 `NameError`;
2. the stacker leakage;
3. the feet/metre mismatch;
4. the correlogram quadrant bug.

## 6. Open decisions

| ID | Decision | Recommendation |
|---|---|---|
| D1 | Development city | A CAPA city with traverse points and an ERA5-independent date (Chicago or Raleigh); Philadelphia goes to the locked test set |
| D2 | New repo vs prune in place | New lean repo |
| D3 | Which single DAG sampler to keep | Whichever of MC³ or order-MCMC has better tests on the synthetic fixture |

---

## Appendix A — Defect index

All references are at `0a04b7a`.

| # | Area | Location | Defect |
|---|---|---|---|
| A1 | Correlogram | `sparc/run/spatial_autocorr_comprehensive.py:748` | FFT ACF sliced `[:ny, :nx]` keeps only offsets with dx, dy ≥ 0 |
| A2 | Correlogram | `sparc/run/spatial_autocorr_comprehensive.py:791-796` | SE = 1/√pairs ignores autocorrelation, so nearly every lag is "significant" |
| A3 | Bandwidth | `sparc/run/gwr_bandwidth.py:39-50`, `sparc/models/gwr.py:176-181` | Bandwidth taken from a predictor's own range; cross-range only fills gaps via `setdefault` |
| A4 | GWR | `sparc/models/gwr.py:660`, `:271-293` | numpy input → `feature_i` names → kernel-field lookup fails → uniform weights in CV |
| A5 | GWRF | `sparc/models/gwrf.py:163-175` | Anisotropy branch checks a nonexistent `.kernels` attribute; dead |
| A6 | Spatial CV | `sparc/run/enhanced_spatial_cv.py:309-317`, `project.yml:260-262` | User's 300 m block size overrides the correlogram-derived size |
| A7 | Features | `sparc/run/enhanced_spatial_cv.py:1444-1450` | Spatial lag uses fixed k=8 nearest neighbours, not a variable-specific radius |
| A8 | Stacking | `sparc/run/enhanced_spatial_cv.py:2026-2041`, `sparc/run/v2_neural_training.py:724-755` | Surrogate targets are in-sample full-data fits (named `base_oof_predictions`), so information leaks into the neural out-of-fold score |
| A9 | Units | `sparc/run/enhanced_spatial_cv.py:1848` vs `:1434-1437` | Full refit uses raw config coordinates (feet); CV uses projected metres |
| A10 | CV | `sparc/run/enhanced_spatial_cv.py:1113, 1143, 1227, 2167` | Failed folds and predictions silently filled with `mean(y)` |
| A11 | CV | `sparc/run/enhanced_spatial_cv.py:1508, 1659` | `n_splits=5` hard-coded |
| A12 | Physics | `sparc/training/loss.py:87-89` | α divided by its own mean; only the relative pattern survives |
| A13 | Physics | `sparc/physics/pde_loss.py:233` | Directional residual `d²x + d²y − ∇²` is zero with the same stencil |
| A14 | Physics | `sparc/physics/pde_loss.py:254` | "Anisotropy" term penalizes `|d²x − d²y|`, i.e. enforces isotropy |
| A15 | Physics | `sparc/physics/pde_loss.py:273` | Gradient-flux term is α‖∇T‖², plain smoothing |
| A16 | Physics | `sparc/physics/energy_balance.py:68-88` | Sensible heat computed as −k∇²T·d (ground conduction); `energy_balance_residual` has no caller |
| A17 | Physics | `sparc/physics/pde_loss.py:399-402` | Sheaf operator is built on the full graph but applied to batch-sized predictions |
| A18 | Physics | `scripts/pde_diagnostics_output.txt` | Learned α in [0.21, 0.40] vs configured bounds [0.5, 12]; script then crashes on a Unicode print |
| A19 | JEPA | `scripts/train_multicity_jepa.py:162-176`, `:805-814` | Per-pixel MLP trunk; masked rows zeroed; each pixel predicts its own embedding, with no cross-pixel flow |
| A20 | JEPA | `scripts/train_multicity_jepa.py:819-828` | "Energy-balance" head predicts `1 − albedo`, an input feature |
| A21 | JEPA eval | `scripts/train_multicity_jepa.py:1920-1960` | Few-shot train/test split is random pixels, not spatially blocked |
| A22 | Labels | `sparc/data/collect/capa.py:74-76`, `:427-444` | Labels are CAPA area-wide (random-forest/Ranger interpolated) rasters, not traverse points |
| A23 | Labels | `configs/multicity_pilot.yml` (`campaign_date_override`) | Dates chosen by ERA5↔CAPA matching, which is circular with the ERA5 features |
| A24 | ERA5 | `sparc/data/collect/era5.py:276-280` | Window hours from longitude/15 (standard time); DST ignored |
| A25 | Saturation | `sparc/interventions/scenario_simulator.py:348-390`, `sparc/templates/uhi/physics/caps.yml:127-137` | Diminishing returns are a hand-set √ taper |
| A26 | Saturation | `sparc/run/enhanced_spatial_cv.py:1964` | `skip_pdp = True` disables the GWRF curve fitter (`sparc/models/gwrf.py:721-834`) |
| A27 | Saturation | `sparc/causal/causal_pdp.py:207-211` | Response = mean(τ)·(dose − median), linear by construction |
| A28 | Saturation | `sparc/interventions/scenario_engine_v4.py:621-675` | Knee dose (absolute) compared with scenario increment |
| A29 | Scenarios | `sparc/__main__.py:786` | `args` not in scope in `_run_scenarios` → `NameError` in `sparc run` Stage 4 |
| A30 | Scenarios | `sparc/run/scenario_engine_selector.py:195`, `:272-277` | Ensemble predictor built only for modes 3/4; mode_5 (default) raises |
| A31 | Scenarios | `sparc/interventions/scenario_simulator.py:895` vs `sparc/run/enhanced_spatial_cv.py:2470` | Reads `standard_meta_ensemble.pkl`; Stage 2 writes `final_meta_ensemble.pkl` |
| A32 | Scenarios | `sparc/interventions/causal_stack.py:539-602` | Scenario Δ is causal β × heuristics; no re-prediction through the trained NN |
| A33 | Docs | `docs/roadmap/SPARC_Integration_Status.md` | Cites nonexistent `correlogram_runner.py`, `gwen_runner.py`, `spatial_cv_runner.py`, `mgwr_runner.py` |
| A34 | Repo | `templates/` vs `sparc/templates/`, `.vs/`, `.vite/` | Byte-identical duplicate templates plus ~520 MB of tracked outputs and IDE caches |
| A35 | Scenarios | `sparc/interventions/scenario_simulator.py` (canopy + impervious ≤ 100 at three sites) | Constraint hard-enforced although `caps.yml` says `enforcement: warning`. On Brown 35% of cells already exceed 100% (canopy overhangs pavement), so every canopy scenario silently removed impervious cover there and inflated cooling |

## Appendix C — Implementation status (2026-09-30)

### The new core: `sparc/core/`

The core is a lean, self-contained package (numpy, scipy, pandas, scikit-learn, torch, pyproj). It imports nothing from `sparc.models`.

- **Run it:** `sparc core run --project configs/core_providence.yml [--fast] [--stages S0,S1,...]`, or `python -m sparc.core run ...`.
- **Tests:** `pytest tests/core` (runs in CI), covering a synthetic city with planted truths and a Providence (`brown4.csv`) integration run.
- **Results:** `docs/results/providence_core.md`.

| Stage | Module | Status |
|---|---|---|
| S0 data, grid, QA | `data.py`, `grid.py`, `config.py` | Done |
| S1 area of influence | `influence.py` | Done: full-plane FFT ACF/CCF, permutation noise band, kernel-shape distributed-lag ranges, bootstrap-gated anisotropy, focal features at {r/2, r, 2r} |
| S2 base models | `base_models.py`, `cv.py` | Done: ridge OLS, MGWR (FFT backfitting, per-variable bandwidths from S1 tuned on an inner split), GRF-lite, GAM, physics; one shared spatial-block partition with buffers |
| S3 physics + stacker | `physics.py`, `operators.py`, `stacker.py`, `ensemble.py` | Done: steady advection–diffusion–relaxation operator with an energy-balance source (variable projection + L-BFGS); cross-fitted physics-informed MLP stacker with an operator penalty (λ tuned out of fold); cross-conformal intervals |
| S4 saturation and marginal effects | `response.py` | Done: neighbourhood-adoption sweeps; per-cell saturating fit against realised neighbourhood dose, with censoring; own-only vs footprint marginals (analytic chain rule through the focal kernels and the physics Green's function, mediator-aware) |
| S5 scenarios | `scenarios.py`, `mediators.py` | Done: add/set/scale, bounds, optional baseline-relative coupling, mediator abduction, re-featurisation through the trained stack, fold-spread uncertainty, extrapolation flags |
| S6 causal validation | `causal.py` | Done: spatial DML; exposure-mapping spillover (θ_own, θ_nbr); R-learner CATE with BLP calibration; Kennedy DR dose-response; E-value and Cinelli–Hazlett RV; model-vs-causal audit on matched estimands; optional MC³ DAG audit |
| S5c climate futures | `climate.py` | Done: CMIP6 change factors (24 models × SSP1-2.6/2-4.5/3-7.0/5-8.5 × 2021–2040/2041–2060/2081–2100, June–August daily highs, land-weighted at the site) read straight from the AWS Pangeo archive; future = observed + model warming + adaptation Δ; heat-threshold exposure with model ranges and the share of warming each package offsets |
| S7 budget and equity | `optimize.py` | Done: concave segments from footprint × saturation into `sparc.scenario.budget`; closed-loop re-prediction of the chosen allocation |
| P7 multi-city / PI-JEPA | — | Not started. Needs the multi-city dataset and a GPU; the specification is ADR-0002 |

### Appendix A items

- **Fixed in place:** the legacy `sparc run` path, each with a regression test in `tests/test_fix_*.py`.
- **Superseded:** the core pipeline does it correctly; legacy code is left as is.
- **Frozen:** untouched until the P7 gate (ADR-0002).
- **Needs data:** the code is fixed and mock-tested, but live verification needs network or data not available here.

| # | Status | Note |
|---|---|---|
| A1, A2 | Fixed | Full-plane ACF with signed offsets; permutation null for z-scores |
| A3 | Fixed | Cross-range bandwidths override own-range; only explicit `manual_parameters.bandwidths` win |
| A4 | Fixed | `GWRModel.fit(feature_names=...)`; CV workers pass names |
| A5 | Fixed | GWRF anisotropy via `KernelField.anisotropic_distance`, with an eccentricity gate |
| A6, A7 | Superseded | Core derives block size from the S1 target-residual range and uses variable-specific focal features |
| A8 | Fixed | `train_neural_meta(base_full_fitted=...)`; fold path uses the true Stage-2a out-of-fold predictions |
| A9 | Fixed | Stage 2b and `ScenarioSimulator` use projected metres (`data_utils.project_coords`) |
| A10 | Fixed | Failed folds are NaN and dropped from stacking, not `mean(y)` |
| A11 | Fixed | `cfg.n_splits` honoured; stale cached folds invalidated |
| A12 | Superseded | Core physics fits an absolute gain `a` with a sign constraint |
| A13–A15 | Fixed | The inert or wrong terms default to weight 0 and can be configured through `physics.pde_weights` |
| A16 | Fixed | Renamed `ground_conduction_proxy` (deprecated alias kept); dead residual removed |
| A17 | Fixed | Sheaf term skipped unless the operator matches the batch |
| A18 | Fixed | Configured α bounds reach `ProcessRateNet`; diagnostics script UTF-8 safe |
| A19, A20 | Frozen | ADR-0002 |
| A21 | Fixed | Few-shot split drops test pixels within a buffer of training pixels; `block` sampler added |
| A22 | Needs data | `download_capa_traverses()` + parser, tested on a synthetic ZIP (OSF blocked here) |
| A23 | Needs data | Per-city `campaign_date_source` provenance + warning; real dates need traverse timestamps |
| A24 | Needs data | ERA5 requested in local time (`timezone=`), CAPA-protocol windows, SW↓/BLH/u10/v10 added; mock-tested (Open-Meteo blocked here) |
| A25 | Fixed | √ taper only when a threshold is explicitly configured; skipped when a condition curve exists |
| A26 | Fixed | GWRF condition curves exported by default |
| A27 | Fixed | Causal PDP integrates E[τ | T] over dose; DR variant via `sparc.core.causal` |
| A28 | Fixed | Knee compared against baseline + increment per cell |
| A29–A31 | Fixed | Stage 4 delegates to `stage4_runner`; mode_5 builds the ensemble; meta model optional |
| A32 | Superseded | Core scenarios re-predict through the trained stack |
| A33 | Fixed | Stale banner on `SPARC_Integration_Status.md` |
| A34 | Fixed | Single template source in `sparc/templates/`; root `templates/`, `.vs/`, `.vite/` removed |
| A35 | Fixed | `enforcement` honoured; when enforced the cap is relative to the baseline, so a zero scenario is an identity |

### Found while fixing

- **The legacy neural stacker never ran.** `train_neural_meta` raised `UnboundLocalError` on `target_lambdas` before its first epoch, so Stage 2 silently fell back to a weighted average of base models.
  - The README's 0.944 R² therefore came from that fallback, evaluated with the leakage in A8/A9.
  - It is fixed, but no legacy number from before this date should be cited.
- **Other latent crashes:** undefined names in `v2_neural_training.py` (`get_active_store`, `model`, `coords_t`, `self`), `decision_stage.main`, `scenarios.benefit` and the mediation logger.
- **`AAT_z` in `brown4.csv` is air temperature in °F** (81–93.5), not a z-score, and 72% of values are whole degrees. RMSE below about 0.29 °F is below the rounding noise.
- **The core's own CV was leaking until 2026-09-30.** The block-size cap used the ΔT target range instead of the y-coordinate range, which gave 4 m blocks: random-point CV in disguise. Under it:
  - the stack scored R² 0.975;
  - the interpolating forest dominated;
  - the canopy footprint came out with the wrong sign.

  With the fix (2 km blocks, 667 m buffer, 23 blocks), the stack scores **R² 0.56 / RMSE 1.14 °F** on unseen neighbourhoods. A reporting-only skill-vs-distance table (`--cv-curve`) shows the decay:

  | Held-out unit | R² |
  |---|---|
  | random cells | 0.975 |
  | 500 m blocks | 0.74 |
  | 1 km blocks | 0.66 |
  | 2 km blocks | 0.56 |

- **Under honest CV the neural residual does not earn its place, on Brown or on the synthetic city.**
  - The stack is now a non-negative blend of base models. A gated MLP residual competes with it out of fold at each λ_PDE, and the residual helps only under leaky random-cell CV.
  - On Brown the blend is GAM 0.59 + physics 0.35. The physics model alone scores R² 0.51 on unseen blocks (L ≈ 240 m, no advection).
  - On the synthetic city the blend puts about 0.88 on physics, which generated the data.
  - `physics_mode: feature` stays the default. Forcing physics as the backbone overstated canopy and albedo effects against the causal estimates.
- **Fitted effects are attenuated, less so now (tracked by `sparc core benchmark`).** On the synthetic city with planted truths, the effect-recovery benchmark gives:

  | | Share of planted canopy effect | Spatial correlation | Stack held-out R² |
  |---|---|---|---|
  | Before | 53% (footprint map) | 0.87 | 0.855 |
  | Now | 63% (footprint map), 67% (uniform +5 pp scenario) | 0.95 | 0.878 |

  Per model (uniform +5 pp scenario):

  | Model | Share |
  |---|---|
  | physics | 72% (was 63%) |
  | MGWR | 50% |
  | OLS | 47% |
  | GAM | 35% |
  | GRF | 31% |

  What worked, what didn't:
  - **Saturating canopy shade in the physics source.** 1 − s·(1 − e^{−c/κ})/(1 − e^{−1/κ}) with κ learned under a weak prior. κ is recovered (0.16 vs 0.15 planted), and the physics model both de-attenuates and predicts better. On Brown κ fits 1.8–5 (near-linear) and accuracy is unchanged.
  - **Spatial+ (Dupont et al. 2022) is implemented but opt-in** (`models.spatial_plus: [mgwr]`):
    - On the synthetic city it raises MGWR from 50% to 69% at the same accuracy.
    - On the full Providence data it lowers MGWR's held-out R² from 0.39 to 0.31, with no stack gain. Covariate–temperature relations there are scale-dependent, which violates Spatial+'s premise.
    - A two-stage variant (refit the intercept against the full covariates) exploded on smooth focal features and was dropped.
  - **Spatial+ does not help the GAM** (35% → 36%). Its attenuation is the prediction-tuned ridge shrinkage: with no spatial basis at all it still recovers 36%.
  - **Stacker candidates now include an equal-weight mean**, alongside the NNLS blend and the gated residual (the forecast-combination puzzle). It is chosen only when fitted weights do not transfer across blocks; on full Providence the NNLS blend still wins.

  Remaining levers:
  - effect-aware ridge penalties for the GAM;
  - reporting the causal-linear scenario band next to every model scenario. This is now done: `causal_linear` in `scenarios.json`.
- **On Brown the S6 audit agrees on the effects that drive scenarios.** The model's neighbourhood-adoption slopes match the causal θ_own + θ_nbr:

  | Treatment | Model | Causal |
  |---|---|---|
  | Canopy (per pp) | −0.026 | −0.011 ± 0.008 |
  | Impervious (per pp) | +0.065 | +0.044 ± 0.012 |
  | Albedo (per +0.1) | −0.79 | −0.63 ± 0.28 |

  The one flag is canopy's own-cell effect: a small warming association once neighbours are controlled (+0.007 ± 0.001 per pp), against about 0 in the model. Canopy's cooling acts through the neighbourhood (θ_nbr −0.018).
- **Albedo scenarios on Brown extrapolate.** Observed albedo is narrow:

  | Albedo change | Edited cells beyond observed conditions |
  |---|---|
  | +0.05 | 8% |
  | +0.1 | 23% |
  | +0.15 or more | essentially all |

  The core flags these, and the large-change numbers should not be used.
- **Results:** `docs/results/providence_core.md`. The interactive page is built by `scripts/results_page/build_page.py` from any run directory.

## Appendix D — Trust, planner and any-city round (2026-10-01)

**Goal of the round.** A researcher meeting SPARC for the first time should find:
- the evidence to trust it;
- the outputs a planner needs;
- a way to run it on another city from open data.

Findings that changed the code are marked ⚑.

**Correctness and provenance**
- Decision quantities are fold-averaged (no 2 km seams), and their uncertainty is the delete-a-group jackknife.
- Intervals are reported per fold and per zone, with a distance-adaptive alternative.
- Coarse full-extent mode (`--coarse 60`) gives the validation studies 13.9k cells over the same 23 blocks.
- S0 data findings:
  - the target is a classed product (72% whole degrees);
  - albedo is not broadband: mean 0.40, where Sentinel-2 broadband gives 0.155;
  - canopy overlaps pavement on 38% of cells;
  - dose scale.
- ⚑ **Campaign date.** The RI Heat Watch traverses ran on **2020-07-29**, not 07-18. The target matches the 15–16 EDT traverse.
  - ERA5 forcing: SW↓ 636 W/m², clear-sky index 0.87.
  - KPVD: 87.5 °F, wind 7.7 m/s from S.

**Researcher trust**
- **Baselines on the same folds** (2 km blocks), against stack RMSE 1.14 °F. The stack beats every baseline by more than 2 block-clustered SE:

  | Baseline | RMSE (°F) | Held-out R² |
  |---|---|---|
  | Boosting on the stack's own neighbourhood features | 1.37 | 0.36 |
  | Boosting + x,y | 1.45 | — |
  | Regression-kriging | 1.58 | — |
  | Boosting without location | 1.75 | — |
  | IDW | 1.88 | — |

  - The neighbourhood features account for part of the gain. The geographically weighted and physics models plus the stacker account for the rest (R² 0.36 → 0.56).
- ⚑ **Placebos exposed a prior-driven canopy effect.** With the old physics priors (shading strength s ~ N(0.6, 0.1)), a canopy layer moved half the map away still got −0.44 °F per sd (45% of the real effect). Decomposition:
  - The physics model held s ≈ 0.53 for the placebo: about 2/3 of the spurious effect.
  - The statistical models also picked up chance large-scale correlation of the smooth layer.
  - The causal check (DML with a spatial basis) passed.

  Physics-only test (`s` and the effect of +1 sd canopy):

  | Prior sd | Real canopy | Shifted | Rotated |
  |---|---|---|---|
  | 0.1 | s 0.56, −1.30 °F | s 0.53, −0.69 °F | s 0.51, −0.94 °F |
  | 0.3 | s 0.34, −1.03 °F | s 0.09, −0.27 °F | s 0.13, −0.50 °F |
  | 1.0 | s 0.03, −0.27 °F | s 0.01, −0.11 °F | s 0.04, −0.25 °F |

  - Held-out R² was 0.533, 0.548 and 0.543 respectively.
  - Priors are now sd 0.3 and configurable (`physics.priors`).
  - The canopy-specific shading channel is weakly identified from one afternoon; vegetation cooling is shared with NDVI.
- **Literature panel.**
  - Canopy: 0.14 °C per +0.10 cover. Ziter et al. 2019 report 0.07–0.15; the Krayenhoff et al. 2021 model review reports 0.3.
  - Albedo: 0.44 °C per +0.10, inside 0.2–0.6 (Krayenhoff) and 0.3–0.9 (Santamouris 2014), with the scale caveat above.
- **Reproducibility.**
  - Hashes and `environment.txt` in every run.
  - `sparc core reproduce` reproduces a fast run bit for bit.
  - Golden synthetic numbers in CI.
  - `Dockerfile.core`.
- **Documentation:** auto-written `methods.md` and `model_card.md`; `docs/references.bib`; `docs/planner_guide.md`; `docs/results/reconciliation.md`.

**Validation studies**
- ⚑ **Simulation check.** The first batch was much noisier than Providence (held-out R² ≈ 0.1 against 0.56). Generators now match the real explained-variance share.
  - Physics generator (20 runs): recovered share 1.13 (IQR 0.90–1.46), jackknife CI covers the truth 80%, per-cell rank correlation 0.27.
- ⚑ **Null simulation found a product-made canopy effect.** With no planted canopy effect, the pipeline reports −0.19 ± 0.05 °F for canopy +10 pp (25% flagged significant; causal 42%), the size of the real estimate.
  - A control without the product's land-cover forest (`--product direct`) cuts it to −0.06 ± 0.04 °F with an 8% false-positive rate: two thirds of the artefact is written into the target by the interpolation, not by SPARC.
  - The uncertainty report now flags canopy estimates that are not distinguishable from the null artefact (all of Providence's are). Settling canopy needs raw traverse points.
- **Multiverse** (8 variants, 60 m): every lever's sign is stable in 100% of variants. Sizes rest on the physics model (canopy +10 pp −0.35 → −0.06 °F without it). Priority maps are not robust (top-decile overlap mostly < 0.6), so they are for shortlisting, not cell picking.
  - ⚑ The uncertainty report applied multiverse spread as offsets, which flipped canopy's sign across resolutions; it now uses ratios to the multiverse baseline.
- The uncertainty report keeps estimation, specification, attribution and the null check separate. The page shows them.

**Planner pack.** HRSL residents (all, 60+, under 5) and WorldCover land cover give:
- exposure today and in each future, with and without the package;
- equity quintiles and a concentration index;
- hot afternoons per summer from KPVD 1995–2014 with CMIP6 deltas;
- plantable space, which caps S7;
- zone and hexagon tables;
- GeoTIFF and GeoPackage export;
- logger sites and before/after pairs.

Not done: quantile delta mapping with daily CMIP6 (the seasonal delta is used) and the nClimGrid cross-check.

**Design tool.** A linear emulator of the fold-averaged model (own, neighbourhood channels, physics Green's kernel; secant at a design dose) runs in the browser.
- Fast-run validation, canopy patches: median error 0.02 °F (13%).
- Large uniform edits overshoot, because saturation is not emulated. The page says so.

**Any-city builder.** `sparc core features` builds all six predictors from WorldCover, Sentinel-2 (S3 listing, cloud-masked median) and the Copernicus DEM in about a minute. Agreement with brown4:

| Feature | Pearson r |
|---|---|
| Elevation | 0.98 |
| NDVI | 0.89 |
| Canopy | 0.77 |
| Impervious | 0.76 |
| Albedo | 0.63 |
| Water distance | 0.46 |

Canopy and impervious fall short of the 0.8 / 0.7 R² targets. The usual remedy is the Meta 1 m canopy-height map, but its bucket refuses listing from here.

The refit on open features only (`configs/core_providence_open.yml`, 60 m) meets the transfer target: held-out R² 0.564 against 0.603 with brown4's layers, canopy +10 pp −0.36 ± 0.24 (vs −0.35), impervious −10 pp −0.28 ± 0.16 (vs −0.39). The stack's margin over boosting on neighbourhood features shrinks to 1.6 SE.

**Deferred:**
- multi-city transfer (needs the other cities' CAPA data);
- the multi-intervention cost optimiser;
- a night-time model.

## Appendix E — SPARC Studio 1.0 (2026-10-01 to 10-05)

**What it is.** A local web app for the core pipeline, from a CSV of street temperatures to exported cooling decisions. `sparc studio` starts a FastAPI server on 127.0.0.1:8765 and opens the prebuilt React app (`sparc/studio/static`). It replaces the old frontend for the core pipeline. `sparc-desktop/` and `sparc/server/` are untouched and stay until parity is confirmed.

User guide: `docs/studio/USER_GUIDE.md`. Developer guide: `docs/studio/DEVELOPING.md`. Specification and wire contract: `docs/studio/SPEC.md` and `api.md`; §19 of each lists the as-built changes.

**What a user can do:**
- **Set up a project** from a CSV, the bundled Providence example or a synthetic demo. This covers the data check with a preview map, a config editor with validation and an impact preview, and open-data input jobs (forcing, layers, features, CMIP6, GHCN, stations).
- **Track runs.** Mission Control shows the stage rail, the fold × model grid, ETA, logs, warnings and resources. Runs can be cancelled, resumed from the last checkpoint and reattached after a server restart, and CLI runs are followed live. There is also an Activity list, a Status Board and run history.
- **Read outputs.** The run hub has 17 tabs: overview, data, accuracy, distance, influence, response, causal, scenarios, climate, budget, planner, uncertainty, provenance, track, map, docs and files. The map explorer has analysis tools (region stats, breakdown, relationships, correlogram), plus run compare.
- **Simulate in the Scenario Lab.**
  - Design: edits by lever with a selection builder or brush, a live emulator preview with reliability badges, and compile checks.
  - Exact results: computed on a long-lived engine with fold-jackknife ranges, spill rings, residents and equity, and climate offset.
  - Around the results: a library with forks, compare, budget plans, sweeps, the climate explorer and decision packs.
- **Validate and export.**
  - Validation: post-run actions and studies (baselines, planner, emulator, uncertainty, placebo, simulation check, multiverse, reproduction, benchmark), launched and tracked from Studio.
  - Exports: run bundle, GIS pack, standalone results page and report builder, plus the Findings notebook.

**Core changes it needed** (additive; existing runs still load):
- `sparc/core/progress.py`: structured events, cancellation and thread limits.
- `runio.py`: atomic writes and a locked manifest.
- Planning and run state: `plan_stages`, `run_state.json`, `checkpoint.json`.
- New outputs: `scenario_detail.npz`, `causal_cells.parquet`, `input_frame.parquet`.
- New modules: `catalog.py` and `session.py`.
- The results-page builder moved to `sparc/core/results_page`; `scripts/results_page` keeps working through shims.
- One-time effect: the new core files change the resume fingerprint, so checkpoints made before this update re-run S2–S3 on resume.

**How it was built and checked.**
- **Design.** Six readers catalogued the pipeline. Three designs were proposed and scored by three judges; the scenario-first design won and borrowed the tracking- and analysis-first designs' best parts. A critic then closed 21 gaps and 19 contradictions.
- **Build.** 15 work items were built in four milestones: browse; run & track; Scenario Lab; studies, exports, end-to-end tests and docs. Each item had an implementer, an adversarial reviewer that fixed what it found, and fix rounds. Each milestone ended with an integration pass and a live Chromium walkthrough.
- **Whole-app review.** Five reviewers looked at correctness, security, job lifecycle, the user journey and scientific integrity, and raised 48 findings.
  - Each finding had to be reproduced; high and critical ones needed two of three skeptics to agree.
  - The 30 most severe were verified this way, 28 were confirmed, and the 18 lower-severity ones were reproduced before any fix.
  - Result: 44 fixed, each with a regression test, and 4 refuted as documented behaviour. Two of the 44 were the same comparison-wording defect, reported through two lenses.

  Fixes that mattered most:
  - **Security (critical):** an untrusted checkpoint could be unpickled through the emulator action. Run import could also delete or write folders outside the run.
  - **Wrong numbers:** unverified budget plans overstated local cooling 3–4×, and the sweep fit was drawn on the wrong dose axis.
  - **Unearned or wrong claims:** draft packs claimed "0% extrapolated" without computing it, and a caveat quoted an uncomputed canopy recovery share. Pairwise comparisons said "warms" when both scenarios cooled.
- **Final state** (d90e6a9):
  - core 254, Studio 579 and web 679 tests pass, plus 70 end-to-end browser tests;
  - the OpenAPI document matches api.md (191 operations), the docs doctest passes, and the committed web build is reproducible.

**Known gaps** (SPEC §19, "Known gaps at release"):
- the ETA after a resume does not discount cached stages;
- relative source paths in `uncertainty.json` match study rows only when the server runs from the repository root;
- no 0.90 reference line on the interval-honesty bars;
- brush strokes from an earlier session reload as per-cell edits;
- some Lab paths are covered only by typecheck;
- Windows and macOS are untested.

## Appendix F — Heat stress, verdicts and portability (2026-10-05)

**Why.** Air temperature alone understates danger, and a single number with an interval is easy to over-read. This round turns the pipeline's temperatures into the quantity health guidance uses (the NWS heat index and its categories), counts the people exposed, and gives every scenario a plain verdict.

**What was built.**
- `sparc/core/heat` (outside the code fingerprint; checkpoints unaffected):
  - the NWS heat index (Rothfusz with the low- and high-humidity adjustments, parity with `forcing.heat_index_F`) and its categories;
  - the campaign dewpoint (station, else ERA5);
  - residents by category today, with the package and per CMIP6 future, as a constant-dewpoint to constant-RH band;
  - `effect_verdict`: Robust / Direction only / Not established, from the envelope, sign stability, the rf null artefact and the extrapolated share (above 50% is Not established).
- Results page (v9): a Heat stress section, verdict pills, heat layers, and live heat risk in the Design emulator.
- SPARC Studio:
  - a Heat tab: the server writes the brief, the dewpoint is a what-if control, futures carry the humidity band widened by the models' p10–p90 warming;
  - heat layers with a warm category palette;
  - verdicts on the Scenarios and Uncertainty tabs;
  - the heat risk of each Lab design (exact impacts, decision brief, report section);
  - a live heat-risk preview while designing that names levers whose scenarios are not established.
- Portability:
  - a start-to-end self-check (`python -m sparc.studio.selfcheck`) on Windows and macOS CI;
  - Windows fixes: the engine host now exits (named-pipe wake) and is killed by the PID Studio asked to stop; atomic replace retries on sharing violations;
  - UTF-8-safe CI reports.

**Providence.**
- 36.7k residents (21%) were at Extreme caution or worse on the campaign afternoon; 3.7k with the Green Infrastructure Package.
- SSP2-4.5 mid-century: 143k–174k; with the package, 55k–172k.
- Verdicts: impervious −10/−20/−30 Robust; the package Robust (26% extrapolated); canopy Not established (indistinguishable from the null artefact); albedo +0.2 Not established (fully extrapolated).

**Limits.**
- One dewpoint applies to the whole city.
- The heat index is steep near category edges, so counts near an edge move a lot with small changes.
- Futures shift today's afternoon by the median warming (delta method); they do not model future humidity or heat-wave frequency.
- The live preview is the linear emulator; exact results verify it.

## Appendix G — Isolating canopy's causal effect (2026-10-05)

**Why.** Canopy is the one lever the uncertainty report calls Not established: the map is a forest-made product, and its canopy dependence is as large when the simulation plants no effect as the real estimate. The question was whether any design can isolate canopy's effect, and at what scale.

**What was built** (`sparc/core/identify`, outside the code fingerprint):
- simulated heat-watch campaigns on the real layout (routes, vehicles, drift, sensor offsets) and the map a forest makes from them;
- design layers: canopy in distance rings with a flexible near-field response and the exact +10 pp edit, wind sectors, piecewise geography;
- estimators: street differences along each pass, read as the effect of canopy within 100 m / 300 m / 1 km; the upwind–downwind signature; levels and map foils;
- the identification lab: every design on six worlds × 12 campaigns, scored against its own estimand (exact disk-edit reach profiles), with verdicts;
- the real-data path: CAPA traverse files placed on the grid (projection, snapping, passes, 1 Hz collapse) and estimated with the lab's verdicts; `python -m sparc.core.identify lab | estimate | map | simulate`;
- a results-page section, "Can canopy's own effect be isolated?".

**Findings** ([canopy_identification.md](../results/canopy_identification.md)):
- Street differences within 100 m are trustworthy: bias ≤ 0.02 °F in every world, including the advected and confounded ones; 8% false positives with no effect.
- Within 300 m: direction only. Within 1 km and city-wide: not identified by one campaign (a kilometre-scale effect and a canopy-tracking kilometre-scale confounder are observationally equivalent).
- The same estimator on the forest-made map reports an effect in 58% of no-effect worlds (0% on the traverses): the product, not the method, breaks identification. The real map shows no street-scale canopy effect (−0.01 °F, CI −0.08 to +0.06).

**Kilometre scale (second round).**
- Four designs were built and tested for the kilometre-scale value:
  - 1 km rings;
  - a physics-constrained extrapolation of the near-field kernel;
  - the same streets compared across runs under different winds, with a rotation test;
  - before/after a planting programme.
- None identifies it from one city's campaign: a city holds only a handful of independent kilometre-sized patches, and weather and confounding vary at that scale.
- What is identified is a floor: city-wide cooling is at least the 300 m effect. The one-sided 95% bound held in ≥ 92% of lab campaigns in every world.
- The wind-shift test is shipped as a valid test: 0% false alarms in six no-advection worlds. It is underpowered on one day (8% power).
- The real-data path now reads an OSF download as it is (nested zips, point shapefiles in any CRS, °C, junk files listed), splits the day into runs by time gaps, reports each run, and fetches each run's wind from the forcing station.
- New commands: `inspect` and `kmlab`.

**Open.**
- Run `estimate` on Providence's traverses (OSF `wu9v7`).
- Pool the floor and the wind-shift test across many cities' Heat Watch campaigns: the only route to the kilometre-scale value.

## Appendix B — References

- **Assran et al. (2023):** *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA).
- **Chernozhukov et al. (2018):** *Double/debiased machine learning for treatment and structural parameters.*
- **Cinelli & Hazlett (2020):** *Making sense of sensitivity: extending omitted variable bias.*
- **Colangelo & Lee (2020):** *Double debiased machine learning nonparametric inference with continuous treatments.*
- **Dupont, Wood & Augustin (2022):** *Spatial+: a novel approach to spatial confounding.*
- **Grimmond, Cleugh & Oke (1991):** *An objective urban heat storage model* (OHM).
- **Kennedy, Ma, McHugh & Small (2017):** *Non-parametric methods for doubly robust estimation of continuous treatment effects.*
- **Lindgren, Rue & Lindström (2011):** *An explicit link between Gaussian fields and Gaussian Markov random fields: the SPDE approach.*
