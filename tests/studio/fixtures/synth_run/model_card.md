# Model card — synthetic_demo

*Generated 2026-10-01T21:21:49+00:00 from the run manifest.*

## Intended use

- Screening and ranking where neighbourhood-scale cooling (tree canopy, less impervious cover, reflective surfaces) would lower afternoon air temperature in this study area, and by roughly how much.
- Conditions like the campaign day: generic daytime forcing (SW↓ 800 W m⁻², net LW -100 W m⁻²).
- Comparing scenarios against each other; locating where effects saturate; framing climate futures.

## Not intended for

- Single-cell or site-design decisions: effects are learned from neighbourhood-scale variation; read cell values as part of a neighbourhood average.
- Night-time temperatures, other weather, or other cities without refitting.
- Health or mortality estimates, or regulatory compliance.

## Performance (held-out spatial blocks)

- RMSE 0.34 degF, R² 0.81, 90% interval coverage 0.89.

- Against standard baselines on the same folds: stack not distinguishable from hgb_focal (|ΔMSE| ≤ 2 SE).

## Effect checks

- Causal audit (model vs DML/spillover/DR estimates): canopy: adoption vs theta sum magnitude differs, own vs theta own magnitude differs, own pd vs dr curve consistent.
- Causal flags: 2 (see report).
- Published effect sizes: Krayenhoff et al. 2021, Environ. Res. Lett. 16, 053007 canopy — SPARC 0.57 °C per +0.10 vs 0.3 °C; Ziter et al. 2019, PNAS 116, 7575–7580 canopy — SPARC 0.57 °C per +0.10 vs 0.07–0.15 °C; Krayenhoff et al. 2021 albedo — SPARC 0.24 °C per +0.10 vs 0.2–0.6 °C; Santamouris 2014, Solar Energy 103, 682–703 albedo — SPARC 0.24 °C per +0.10 vs 0.3–0.9 °C.

## Noise floor

- Target rounding noise ≈ 0.01 degF (0% whole-degree values).
- City-wide scenario differences smaller than ≈ 0.45 degF (2 × median jackknife SE) are not distinguishable from fold-to-fold variation.

## Limitations

- Daytime (afternoon) model of one campaign day; it is never applied to night.
- Effects are associations adjusted for the measured predictors and smooth spatial confounding; unmeasured local factors (building shade, irrigation, traffic heat) can bias them.
- Scenario cells outside the observed combination of predictors are extrapolations (share reported).
- Physics forcing is generic, not the campaign day's.

## Reproducibility

- commit `5a04f4472c57`, code `9ef8fac0551a`, input `4bfb0ca0ba3b`, config `58f0d8cabbf3`; `sparc core reproduce` re-runs and compares.
