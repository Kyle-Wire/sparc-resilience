# Methods — synthetic_demo

*Draft generated from the run manifest; numbers are this run's.*

## Data

The response is `T` (degF) on 1,120 cells of 30 m. Temperatures are analysed as differences from the field median (87.92 degF). Predictors: `canopy`, `impervious`, `albedo`, `ndvi`, `elevation`, `water_dist`. Physics forcing: generic daytime forcing (SW↓ 800 W m⁻², net LW -100 W m⁻²).

Data checks raised: .

## Area of influence

Anisotropic empirical correlograms of the target's residual (after the predictors) and of each predictor give the distances over which they vary together. The target residual range (630 m) sets the cross-validation block size and the physics length prior (L₀ = 223 m). Predictor influence ranges: canopy 315 m, impervious 315 m, albedo 60 m, ndvi 315 m, elevation 306 m, water_dist 60 m. Neighbourhood (focal) means of each predictor at ½, 1 and 2 × its range enter the models.

## Models

Base models: ridge-regularised linear regression on the predictors and their neighbourhood means; multiscale geographically weighted regression (Fotheringham et al. 2017); geographically weighted random forest; generalised additive model with a spatial smooth (Wood 2017); a physics model: a surface-energy source term (shortwave absorption, canopy shading, impervious heat storage, evapotranspiration) propagated by an advection–diffusion–relaxation operator. Their out-of-fold predictions are stacked: candidates are the equal-weight mean, a non-negative convex blend, and the blend plus a gated neural residual with a physics (PDE) penalty (λ ∈ [0.0, 0.1]); the candidate with the lowest outer out-of-fold RMSE is kept (convex (NNLS) blend of base models).

The physics source terms carry weakly informative priors (mean, sd): canopy shading s ~ N(0.6, 0.3), impervious storage a₁ ~ N(0.3, 0.3); a tighter shading prior (sd 0.1) held s at its prior even for a displaced placebo canopy layer.

Advection by the campaign wind was dropped after an out-of-fold test (ΔRMSE with − without = 0.030 ± 0.037 degF).

## Validation

Spatial-block cross-validation (Roberts et al. 2017): 16 square blocks of 390 m in 3 folds, with training points within 130 m of a test block removed. Every model, the stacker choice and the prediction intervals use these folds; base-model predictions are cross-fitted so the stacker never sees in-fold predictions. Prediction intervals are cross-conformal (nominal 90%), globally and scaled by distance to the nearest training data.

Held-out skill: RMSE 0.34 degF, R² 0.81, interval coverage 0.89.

Standard baselines were fitted on the same folds and compared with the stack by a block-clustered paired difference in squared error: gradient boosting, covariates + x,y RMSE 0.67 (ΔMSE +0.334 ± 0.121); gradient boosting on the stack's inputs (covariates + neighbourhood features) RMSE 0.48 (ΔMSE +0.110 ± 0.062); inverse-distance interpolation (no covariates) RMSE 0.64 (ΔMSE +0.298 ± 0.161). Verdict: stack not distinguishable from hgb_focal (|ΔMSE| ≤ 2 SE).

## Effects

Decision quantities (scenario changes, own-cell and footprint sensitivities, saturation curves) use the mean over the fold models, so maps have no fold seams; their uncertainty is the delete-a-group jackknife over folds (SE = sd·√(K−1)). Scenario inputs are clipped to the observed support of each model input except in the physics term; the share of cells needing extrapolation is reported. Saturation is fitted per cell to neighbourhood-adoption dose sweeps (linear, saturating or sigmoid, chosen by AIC).

## Causal audit

Each treatment's effect is re-estimated without the predictive models by double/debiased machine learning (Chernozhukov et al. 2018) with spatial-block cross-fitting, a spatial basis for unmeasured smooth confounding, own-cell and neighbourhood (spillover) exposures, a doubly robust dose-response curve, and sensitivity analysis (E-values, VanderWeele & Ding 2017; robustness values, Cinelli & Hazlett 2020). The model's effects are compared with these estimates.

## Climate futures

Summer (June–August) mean daily maximum temperature change from 6 CMIP6 models (Eyring et al. 2016) between 1995-2014 and future periods at the site, land-weighted, added to the observed field (delta method), with and without the adaptation scenarios.

## Reproducibility

Code commit `358200a0233a`, core code SHA-256 `f12ff6797b8c`, input SHA-256 `4bfb0ca0ba3b`, config SHA-256 `58f0d8cabbf3`; package versions in `environment.txt`. `sparc core reproduce <run dir>` re-runs and compares.
