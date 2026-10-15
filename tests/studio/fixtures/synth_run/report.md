# SPARC core run — synthetic_demo

*2026-10-15T15:48:45+00:00* · commit `f8284e5` · 1120 points · **fast mode**

## S0 — Data

- grid [42, 42] at 30.00 m, fill 0.63
- background (ΔT reference): 87.92 degF (field_median)
- target values that are whole numbers: 0.00 (rounding noise ≈ 0.01 degF)
- QA clips: none

**Data findings** (read before using the results):

- ℹ canopy + impervious exceeds 100% on 13% of cells (canopy overhanging pavement), so no sum constraint is imposed between them.
- ℹ albedo doses 0.2 exceed 3 sd of the layer (sd 0.0602); their effects lean on extrapolation (see the scenarios' extrapolated share).
- ℹ Fast window: only 1,120 points around the centre — a smoke-test extent, too small for the spatial-block CV to mean what it does on the full area.

## S1 — Area of influence

Target residual range 630 m → physics length prior L₀ = 223 m; CV block 630 m.

| variable | influence range (m) | anisotropy b/a | θ (°) | reliable |
|---|---|---|---|---|
| canopy | 315 | 0.28 | 128 | False |
| impervious | 315 | 0.64 | 162 | False |
| albedo | 60 | 0.68 | 152 | False |
| ndvi | 315 | 0.00 | 135 | False |
| elevation | 306 | 0.95 | 123 | False |
| water_dist | 60 | 0.47 | 20 | False |

## S2/S3 — Out-of-fold performance (spatial blocks, cross-fitted stacking)

3 folds of square blocks (390 m, 16 blocks with data); training points within 130 m of a test point are dropped.

| model | RMSE (degF) | MAE | R² |
|---|---|---|---|
| ols | 1.048 | 0.776 | -0.739 |
| mgwr | 0.931 | 0.716 | -0.373 |
| gwrf | 0.470 | 0.380 | 0.650 |
| gam | 0.515 | 0.429 | 0.581 |
| physics | 0.319 | 0.278 | 0.839 |
| base_mean | 0.508 | 0.419 | 0.592 |
| stacker | 0.339 | 0.296 | 0.818 |

Stacker: convex (NNLS) blend of base models (out-of-fold RMSE by candidate: {'mean': 0.5077083816327378, 'off': 0.3394922332170534, '0.0': 0.4066576923608064, '0.1': 0.432644776514148}); 90% cross-conformal interval coverage 0.891 (mean half-width 0.501 degF).

**Interval honesty** — pooled coverage is nearly guaranteed by construction, so coverage is also shown per fold, by distance from training data and per zone (global vs distance-adaptive cross-conformal intervals):

| group | n | global coverage | adaptive coverage | global ± (degF) | adaptive ± (degF) |
|---|---|---|---|---|---|
| all | 1120 | 0.891 | 0.886 | 0.50 | 0.51 |
| fold 0 | 373 | 0.997 | 0.995 | 0.54 | 0.54 |
| fold 1 | 375 | 0.808 | 0.797 | 0.47 | 0.48 |
| fold 2 | 372 | 0.868 | 0.866 | 0.49 | 0.50 |
| distance 0-180 m | 315 | 0.940 | 0.917 | 0.50 | 0.50 |
| distance 180-240 m | 254 | 0.894 | 0.890 | 0.50 | 0.50 |
| distance 240-330 m | 323 | 0.808 | 0.808 | 0.50 | 0.50 |
| distance 330-577 m | 228 | 0.939 | 0.947 | 0.50 | 0.54 |

**Stacker base weights (mean over folds):** ols 0.01, mgwr 0.02, gwrf 0.05, gam 0.08, physics 0.85; neural residual kept in 0/3 folds (gated off where it did not beat the convex base on held-out inner blocks).

**Physics (mean ± sd over folds):** L_m = 145.414 ± 26.132, vx_m = 0.000 ± 0.000, vy_m = 0.000 ± 0.000, a = 34.999 ± 9.069, gamma = 0.010 ± 0.004

**Forcing:** generic (SW↓ 800 W/m², net LW -100 W/m², wind [0.03333333333333333, 0.0]); run `sparc core forcing` for the campaign day.

**Advection:** dropped — held-out RMSE with − without advection = 0.0302 ± 0.0372 degF (kept only if lower by > 1 SE).

### Stack vs standard baselines (same folds, paired by CV block)

| baseline | RMSE | R² | ΔRMSE vs stack | ΔMSE ± SE (block-clustered) | blocks where stack better |
|---|---|---|---|---|---|
| gradient boosting, covariates + x,y | 0.675 | 0.278 | +0.336 | +0.341 ± 0.124 | 75% |
| gradient boosting on the stack's inputs (covariates + neighbourhood features) | 0.470 | 0.651 | +0.130 | +0.105 ± 0.059 | 88% |
| inverse-distance interpolation (no covariates) | 0.644 | 0.343 | +0.305 | +0.300 ± 0.162 | 81% |

Positive Δ = the stack is better. **Verdict:** stack not distinguishable from hgb_focal (|ΔMSE| ≤ 2 SE).

## S4 — Saturation and marginal effects

| variable | saturating | censored | linear | median d90 | median max cooling | own effect/unit | footprint/unit |
|---|---|---|---|---|---|---|---|
| canopy | 1.00 | 0.00 | 0.00 | 31.84 | 1.854 | -0.0047 | -0.1359 |
| impervious | 0.37 | 0.12 | 0.48 | 17.68 | 1.346 | 0.0004 | 0.0195 |
| albedo | 0.25 | 0.20 | 0.75 | — | 4.091 | -0.2444 | -4.3858 |

*Own effect*: only the cell itself changes. *Footprint*: total change summed over the neighbourhood (drives the optimiser). Saturation is fitted to neighbourhood-adoption sweeps against the realised neighbourhood dose; 'censored' = no knee within the tested doses/headroom.

## Consistency with published effect sizes

| quantity | published | conditions | source (status) | SPARC (°C per +0.10) | ratio | within ×2 |
|---|---|---|---|---|---|---|
| canopy | ≈0.3 °C cooling per +0.10 tree canopy cover | systematic review of numerical models (47 higher-quality studies); afternoon, clear sky, summer; near-surface air | Krayenhoff et al. 2021, Environ. Res. Lett. 16, 053007 (abstract) | 0.57 ± 0.15 | 1.91 | yes |
| canopy | 0 → 100% canopy: −0.7 °C (10 m radius), −1.3 °C (30 m), > −1.5 °C (60–90 m); nonlinear, cooling accelerates above ~40% canopy at block scale | bicycle transects, Madison WI, summer daytime; per-0.10 range is the linear average of the 10 m and 60–90 m totals (the real curve is steeper above 40%) | Ziter et al. 2019, PNAS 116, 7575–7580 (abstract) | 0.57 ± 0.15 | 5.22 | no |
| albedo | ≈0.2–0.6 °C cooling per +0.10 neighbourhood albedo | numerical models; afternoon, clear sky, summer | Krayenhoff et al. 2021 (abstract) | 0.24 ± 0.04 | 0.61 | yes |
| albedo | city-wide +0.1 albedo: ≈0.3 K mean and ≈0.9 K peak ambient temperature decrease | review of simulation studies; peak (≈ afternoon) is the comparable figure | Santamouris 2014, Solar Energy 103, 682–703 (abstract) | 0.24 ± 0.04 | 0.41 | yes |

SPARC column: city-wide mean cooling of the closest uniform scenario, scaled to +0.10 cover / albedo (fold-averaged ± jackknife SE). Status *abstract* = value from the paper's abstract or an indexed summary; check the full text before citing. Other cities, methods and scales: agreement within ×2 is what consistency can mean here. Albedo in this dataset may not be broadband (see Data findings).

## S5 — Scenarios

| scenario | mean Δ (degF) | p10 | p90 | SE of mean (jackknife) | extrapolated | linear causal Δ (95%) |
|---|---|---|---|---|---|---|
| Canopy Increase +5 | -0.625 | -1.042 | -0.307 | 0.162 | 0.486 | -0.202 (-0.270 to -0.134) ⚑ |
| Canopy Increase +10 | -1.033 | -1.696 | -0.508 | 0.279 | 0.728 | -0.403 (-0.539 to -0.267) ⚑ |
| Canopy Increase +20 | -1.506 | -2.440 | -0.751 | 0.451 | 1.000 | -0.806 (-1.078 to -0.534) ⚑ |
| Impervious Decrease −10 | -0.184 | -0.247 | -0.120 | 0.153 | 0.342 | — |
| Impervious Decrease −20 | -0.353 | -0.493 | -0.217 | 0.305 | 0.683 | — |
| Albedo Increase +0.05 | -0.220 | -0.318 | -0.129 | 0.040 | 0.224 | — |
| Albedo Increase +0.1 | -0.440 | -0.640 | -0.261 | 0.076 | 0.537 | — |
| Cooling package | -1.310 | -2.068 | -0.685 | 0.373 | 0.749 | — |

Linear causal Δ: the S6 own + neighbour effect × the mean realised change (a local-slope extrapolation); ⚑ = the model's mean Δ lies outside its 95% band.

## Climate projections (CMIP6 delta method)

Change in 6-7-8 mean daily tasmax vs 1995-2014, 6 CMIP6 models at 41.821°, -71.390° (land-weighted bilinear).  Future = observed + model warming (+ adaptation Δ).

| pathway | period | warming median (10–90%) | share ≥ 95 degF: no adaptation |  |
|---|---|---|---|
| today | — | — | 0.0% |
| SSP1-2.6 | 2021-2040 | 1.70 (1.34 to 2.19) | 0.0% |
| SSP1-2.6 | 2041-2060 | 2.46 (1.93 to 3.16) | 0.0% |
| SSP1-2.6 | 2081-2100 | 2.83 (2.23 to 3.64) | 0.0% |
| SSP2-4.5 | 2021-2040 | 1.89 (1.49 to 2.43) | 0.0% |
| SSP2-4.5 | 2041-2060 | 3.21 (2.52 to 4.13) | 0.0% |
| SSP2-4.5 | 2081-2100 | 4.91 (3.86 to 6.32) | 2.3% |
| SSP3-7.0 | 2021-2040 | 1.89 (1.49 to 2.43) | 0.0% |
| SSP3-7.0 | 2041-2060 | 3.78 (2.97 to 4.86) | 0.0% |
| SSP3-7.0 | 2081-2100 | 6.99 (5.49 to 8.99) | 51.6% |
| SSP5-8.5 | 2021-2040 | 2.08 (1.63 to 2.67) | 0.0% |
| SSP5-8.5 | 2041-2060 | 4.54 (3.56 to 5.83) | 0.4% |
| SSP5-8.5 | 2081-2100 | 8.69 (6.83 to 11.18) | 99.5% |

Delta method: the adaptation effect is assumed not to change with background warming; daily extremes may warm more than the seasonal mean of daily maxima.

## S6 — Causal validation

| treatment | DML θ ± SE | θ_own | θ_nbr | θ_own+θ_nbr | audit |
|---|---|---|---|---|---|
| canopy | -0.0197 ± 0.0037 | -0.0154 | -0.0249 | -0.0403 | adoption_vs_theta_sum: magnitude differs; own_vs_theta_own: magnitude differs; own_pd_vs_dr_curve: consistent |

The audit compares the model's implied effects with the causal estimates on matched estimands: neighbourhood-adoption slope ↔ θ_own + θ_nbr, own-only slope ↔ θ_own, own-only partial dependence ↔ doubly-robust dose-response.

## S7 — Budget allocation

canopy: budget 2000 → 232 cells treated (mean dose 8.62); planned total cooling 586.73, closed-loop realised 626.81 (degF·cells). Constraint: plantable space (WorldCover open land + 20% of built-up area); objective: total cooling.

## Timings (s)

S0: 0.1, S1: 2.76, S2_S3: 24.89, baselines: 2.09, S4: 6.45, S5: 1.64, S6: 6.89, S7: 0.2
