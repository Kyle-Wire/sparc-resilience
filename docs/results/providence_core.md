# Providence (brown4.csv): core pipeline results

**Run:** `sparc core run --project configs/core_providence.yml --cv-curve` · 2026-10-01 · commit `d0d6cc9` · 102 min on 3 CPU threads (61 of them for the optional distance table).

**Interactive version:** the "Providence Heat Model" results page (maps, charts and tables) is built from the same run directory. See [Regenerating](#regenerating).

These results supersede every Providence number produced by the legacy pipeline (see `CORE_ROADMAP.md` Appendix C).

## Data

| Item | Value |
|---|---|
| Cells | 54,701 on an exact 30 m lattice (grid 334 × 287, 57% filled) |
| Target | `AAT_z`: air temperature in **°F** (81–93.5), not a z-score |
| Rounding | 72% of values are whole degrees: an interpolated, rounded product. Errors below about 0.3 °F are noise. |
| Background | Field median, 88.0 °F. ΔT is relative to it; there is no ERA5 record for this campaign. |
| QA | 4 albedo values above 0.9 clipped |
| Land cover | Canopy + impervious exceeds 100% on 35% of cells (canopy overhangs pavement), so no sum constraint is imposed |

## Cross-validation design

Every accuracy number below is out of fold under **5-fold spatial-block CV**:

- square blocks of 2,000 m, with 23 blocks containing data;
- each fold holds out whole blocks, about 10,900 cells;
- training cells within 667 m (block ÷ 3) of any test cell are dropped, which removes 22–37% of the training set per fold.

**Block size.** The block size is the S1 target-residual correlation range. It reached the correlogram's 2 km cap, so the true range is at least 2 km.

**What this measures.** This is the standard design for spatially autocorrelated data (Roberts et al. 2017). It measures prediction for an unseen neighbourhood.

**Why the scores are lower than a random split.** It is deliberately stricter than in-city gap filling, which is why scores sit well below random-split numbers. Wadoux et al. (2021) discuss why block CV is pessimistic for mapping when the sample covers the area.

**One partition is shared by everything:** base models, the stacker, the conformal intervals and causal cross-fitting.

### Skill vs distance from training data (reporting only)

| Held-out unit | Buffer (m) | Blocks | Stacked R² | Stacked RMSE (°F) | Fold R² range | Stacker chosen |
|---|---|---|---|---|---|---|
| Random cells (leaky reference) | 0 | 54,137 | 0.975 | 0.27 | 0.97–0.98 | blend + neural residual |
| 500 m blocks | 167 | 245 | 0.743 | 0.87 | 0.59–0.80 | blend only |
| 1,000 m blocks | 333 | 73 | 0.660 | 1.00 | 0.41–0.81 | blend only |
| **2,000 m blocks (main)** | **667** | **23** | **0.556** | **1.14** | **0.26–0.74** | **blend only** |

**Reading the curve:**
- **Random-cell CV mostly measures interpolation.** Every test cell has training neighbours 30 m away. That is how the earlier pipeline reached 0.94–0.98.
- **Skill decays smoothly with distance from training data.**
- **The neural residual helps only in the leaky setting.** That is evidence it learns location rather than physics or land cover.

## Models (main 2 km blocks)

| Model | R² | RMSE (°F) | Blend weight (mean over folds) |
|---|---|---|---|
| Ridge regression (raw + focal features) | 0.30 | 1.43 | 0.00 |
| MGWR (FFT backfitting) | 0.39 | 1.34 | 0.04 |
| Geographically weighted random forest | 0.42 | 1.30 | 0.01 |
| GAM + spatial smooth | 0.55 | 1.15 | 0.59 |
| Physics (heat transport) | 0.50 | 1.21 | 0.35 |
| **Stacked model** | **0.556** | **1.14** | — |

- **Stacker.** Chosen on the held-out blocks: a non-negative (NNLS) blend of the base models (RMSE 1.14). The other candidates were an equal-weight mean (1.19) and the blend plus a neural residual at λ_PDE ∈ {0, 0.1, 1} (1.23, 1.27, 1.30).
- **Intervals.** 90% cross-conformal intervals cover 89.7% of held-out cells, with a mean half-width of ±1.86 °F.
- **Physics.** The steady advection–diffusion–relaxation operator is driven by an energy-balance source, with a canopy shade term that can saturate with cover. Across folds:
  - L = 240 ± 22 m and gain a = 36 ± 8;
  - the shade-saturation scale κ = 3.4 ± 1.1, so on Brown the shade response stays close to linear;
  - no advection, because there is no wind record.

  On its own it scores R² 0.50 on unseen blocks, second only to the GAM, and it carries 19–50% of the blend.
- **Spatial+** is available (`models.spatial_plus: [mgwr]`) but off here. It lowered MGWR's held-out R² from 0.39 to 0.31 on these data with no stack gain.

## S1: Area of influence

| Variable | Influence range | Notes |
|---|---|---|
| Distance to water | 1,462 m | |
| Impervious cover | 618 m | |
| NDVI | 447 m | |
| Tree canopy | 60 m (floor) | Acts through the cell and its neighbours |
| Albedo | 60 m (floor) | |
| Elevation | 60 m (floor) | |
| Temperature residual | ≥ 2,000 m | Reached the cap |

No anisotropy passed the bootstrap gate.

## S4: Saturation and marginal effects

| Intervention | Saturating | Linear | Censored | Median dose for 90% of max | Own-cell effect / unit | Footprint / unit |
|---|---|---|---|---|---|---|
| Canopy added (pp) | 56% | 37% | 5% | 11.4 pp | −0.0001 °F | −0.022 °F |
| Impervious removed (pp) | 9% | 78% | 2% | 38.9 pp | +0.0032 °F per +1 pp | +0.052 °F per +1 pp |
| Albedo added | 98% | 1% | 2% | 0.24 | −0.34 °F per unit | −7.4 °F per unit (−0.074 per +0.01) |

**The footprint is about 15–150× the own-cell effect.** Interventions cool their surroundings far more than the treated cell, which is why S1's area of influence matters for the optimiser.

City-wide mean cooling by neighbourhood dose:

| Dose | +5 | +10 | +15 | +20 | +30 | +40 | +50 |
|---|---|---|---|---|---|---|---|
| Canopy (pp) | 0.11 | 0.24 | 0.37 | 0.50 | 0.75 | 1.00 | 1.19 |
| Impervious removed (pp) | 0.21 | 0.41 | 0.61 | 0.83 | 1.34 | 1.92* | 2.51* |

\* Impervious doses of 40 pp or more leave 41–71% of cells beyond observed conditions. Albedo is extrapolated from +0.15 upwards.

## S5: Scenarios (mean ΔT, °F)

The last column is an independent cross-check from S6: the causal own + neighbour effect × the scenario's mean realised change, with a 95% band. It is a straight-line extrapolation, so it cannot follow saturation. ⚑ marks a model mean outside the band.

| Scenario | Model mean | 10th–90th pct (model) | Beyond observed | Linear causal estimate (95%) |
|---|---|---|---|---|
| Canopy +5 / +10 pp | −0.11 / −0.24 | −0.45 to +0.02 (at +10) | 6% | −0.05 (−0.12 to +0.02) / −0.10 (−0.25 to +0.04) |
| Canopy +20 / +30 pp | −0.50 / −0.75 | — | 6–7% | −0.20 (−0.48 to +0.08) ⚑ / −0.30 (−0.71 to +0.12) ⚑ |
| Impervious −5 / −10 pp | −0.21 / −0.41 | −0.65 to −0.14 (at −10) | 4–5% | −0.20 / −0.40 (−0.61 to −0.18) |
| Impervious −20 / −30 pp | −0.83 / −1.34 | — | 9–18% | −0.78 / −1.15 (−1.77 to −0.52) |
| Albedo +0.05 / +0.1 | −0.39 / −0.79 | −1.10 to −0.50 (at +0.1) | 8% / 23% | −0.32 / −0.63 (−1.18 to −0.09) |
| Albedo +0.2 | −1.32 | — | **100%: do not use** | −1.27 |
| Green package (canopy +15, impervious −15, albedo +0.1) | −1.71 | −2.21 to −1.22 | 26% | −1.38 (−2.04 to −0.71) |

**Impervious and albedo:** the model and the causal estimate agree throughout.

**Canopy:** the model's cooling is larger than the causal straight line from +20 pp upwards. The causal canopy effect is itself weakly identified (its band includes zero), so canopy magnitudes are the least certain part of the scenario set.

## Climate futures (CMIP6, delta method)

**Method:**
- **Warming:** change in June–August mean daily maximum near-surface air temperature (`tasmax`) between 1995–2014 and each IPCC AR6 period. It comes from **24 CMIP6 models**, one member each, read from the public AWS archive.
- **Location:** the four model grid cells around Providence (41.826°N, 71.403°W), weighted by land fraction.
- **Future maps:** today's measured field plus each model's warming, plus an adaptation scenario's re-predicted change.
- **Per-model values:** `configs/climate/providence_cmip6_tasmax_jja.csv`.

| Pathway | 2021–2040 | 2041–2060 | 2081–2100 |
|---|---|---|---|
| SSP1-2.6 | +1.8 °F (1.1–2.8) | +2.6 (1.5–4.2) | +2.4 (1.5–4.5) |
| SSP2-4.5 | +1.8 (1.2–3.3) | **+3.0 (2.3–4.9)** | +4.6 (3.3–7.0) |
| SSP3-7.0 | +1.9 (1.2–3.6) | +3.6 (2.1–6.7) | +7.8 (4.6–11.6) |
| SSP5-8.5 | +2.0 (1.2–3.6) | +4.1 (3.0–6.5) | +9.4 (5.8–13.6) |

The table shows the median across models, with the 10th–90th percentile in brackets.

**Heat exposure:** share of the study area at or above 90 °F. Today the share is 13% (and 0% reach 95 °F). Adaptation packages are each variable's largest in-support dose.

| Pathway · period | No adaptation | Canopy +30 pp | Impervious −30 pp | Albedo +0.05 |
|---|---|---|---|---|
| SSP2-4.5 · 2041–2060 | 76% | 60% | 41% | 69% |
| SSP2-4.5 · 2081–2100 | 90% | 88% | 77% | 90% |
| SSP5-8.5 · 2041–2060 | 90% | 82% | 69% | 84% |
| SSP5-8.5 · 2081–2100 | 100% (90% at ≥ 95 °F) | 100% | 100% | 100% |

**Share of the median mid-century SSP2-4.5 warming each package offsets:**
- impervious −30 pp: 44%;
- canopy +30 pp: 25%;
- albedo +0.05: 13%.

By 2081–2100 under SSP3-7.0 or SSP5-8.5, no single land-cover package keeps the area below 90 °F.

**Assumptions (delta method):**
- Land-cover effects are assumed unchanged as the background warms.
- The hottest days may warm more than the seasonal average of daily highs.
- Campaign-day conditions are treated as representative.

The CMIP6 output is used under the CMIP6 terms of use (CC BY 4.0 for most modelling groups).

## S6: Causal validation

Spatial DML with the same 2 km blocks for cross-fitting (22 block degrees of freedom), plus exposure-mapping spillover. "Model" is the model's own estimate of the matching quantity.

| Treatment | Check | Model | Causal | Verdict |
|---|---|---|---|---|
| Canopy (per +1 pp) | Neighbourhood adoption vs θ_own + θ_nbr | −0.026 | −0.011 ± 0.008 | consistent |
| | Own cell vs θ_own | −0.0001 | +0.0071 ± 0.0014 | **sign conflict** |
| Impervious (per +1 pp) | Neighbourhood adoption vs θ_own + θ_nbr | +0.065 | +0.044 ± 0.012 | consistent |
| | Own cell vs θ_own | +0.003 | +0.000 ± 0.003 | consistent |
| Albedo (per +0.1) | Neighbourhood adoption vs θ_own + θ_nbr | −0.79 | −0.63 ± 0.28 | consistent |

**Robustness to hidden confounding:**

| Treatment | DML total effect | E-value | Robustness value | Strength |
|---|---|---|---|---|
| Impervious | +0.0099 ± 0.0043 per pp | 1.32 | 14.7% | Moderately robust |
| Albedo | −0.38 ± 0.15 per +0.1 | 1.84 | 10.4% | Moderately robust |
| Canopy (own-cell-only) | −0.0009 ± 0.0030 | 1.08 | 1.2% | Not distinguishable from zero |

**Interpretation:**
- **Canopy cools through its neighbourhood.** The causal estimate of the neighbour effect (θ_nbr −0.018) and the model's footprint agree on that.
- **The own-cell canopy sign conflict is small.** It is a slight warming association for a cell's own canopy once its neighbours are controlled. It is more likely an artefact of the interpolated temperature product or residual confounding than real physics, and it does not change any scenario conclusion.

## S7: Canopy budget plan

- **Budget:** 20,000 dose units (+1 pp on one cell = 1 unit), placed on **864 cells** at a mean of +23.1 pp each.
- **Planned vs re-predicted cooling:** planned 1,704 °F·cells. Re-predicting the whole plan at once gives 1,037 °F·cells (61%), because neighbouring plantings share footprints.
- **Cooling achieved:** 0.15 °F mean in treated cells; 0.019 °F city-wide.
- **Scaling:** doubling the budget raises planned cooling 1.77×.

## Validation of the method (synthetic city with planted truths)

- **Stack choice:** the stack puts most of its weight on physics, which generated the data, and beats the best base model (held-out R² 0.878 vs 0.872).
- **Effect recovery** (`sparc core benchmark`): the stacked footprint recovers 63% of the planted canopy effect, with spatial correlation 0.95. Before the saturating physics shade it was 53% and 0.87. Effects are still attenuated, mostly by the prediction-tuned GAM and forest, so treat scenario magnitudes as conservative.

## Caveats

- Rounded, interpolated target: about 0.3 °F noise floor.
- Accuracy varies strongly by area (fold R² 0.26–0.74).
- No wind and no time-of-day record; daytime forcing is assumed.
- Albedo's observed range is narrow, so large albedo edits extrapolate.

## Regenerating

```bash
sparc core climate --lat 41.826 --lon -71.403 --out configs/climate/providence_cmip6_tasmax_jja.csv   # CMIP6 change factors (~5 min)
sparc core run --project configs/core_providence.yml --cv-curve        # add --resume after an interruption
sparc core benchmark                                                   # synthetic-city effect recovery
python scripts/results_page/build_page.py output/core/providence/providence_uhi configs/core_providence.yml \
       --out output/core/providence/providence_uhi/results.html
```
