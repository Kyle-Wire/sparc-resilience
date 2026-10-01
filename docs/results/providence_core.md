# Providence (brown4.csv): core pipeline results

**Run:** `sparc core run --project configs/core_providence.yml --cv-curve` · 2026-10-01 · commit `1d361e0` · 103 min on 3 CPU threads.

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
| **2,000 m blocks (main)** | **667** | **23** | **0.557** | **1.14** | **0.26–0.74** | **blend only** |

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
| Physics (heat transport) | 0.51 | 1.20 | 0.35 |
| **Stacked model** | **0.557** | **1.14** | — |

- **Stacker.** The stacker is a non-negative blend of the base models. A neural residual was scored out of fold at λ_PDE ∈ {0, 0.1, 1}, with RMSE 1.23, 1.27 and 1.30. It lost to the blend alone (1.14), so it is switched off.
- **Intervals.** 90% cross-conformal intervals cover 89.7% of held-out cells, with a mean half-width of ±1.86 °F.
- **Physics.** The steady advection–diffusion–relaxation operator is driven by an energy-balance source. Across folds it gives L = 242 ± 22 m and gain a = 37 ± 8. There is no advection, because there is no wind record. On its own it scores R² 0.51 on unseen blocks, second only to the GAM, and it carries 20–50% of the blend.

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
| Canopy added (pp) | 54% | 40% | 6% | 7.4 pp | −0.0001 °F | −0.020 °F |
| Impervious removed (pp) | 9% | 78% | 2% | 38.9 pp | +0.0031 °F per +1 pp | +0.052 °F per +1 pp |
| Albedo added | 98% | 1% | 1% | 0.24 | −0.33 °F per unit | −7.6 °F per unit (−0.076 per +0.01) |

**The footprint is about 15–150× the own-cell effect.** Interventions cool their surroundings far more than the treated cell, which is why S1's area of influence matters for the optimiser.

City-wide mean cooling by neighbourhood dose:

| Dose | +5 | +10 | +15 | +20 | +30 | +40 | +50 |
|---|---|---|---|---|---|---|---|
| Canopy (pp) | 0.10 | 0.23 | 0.35 | 0.49 | 0.74 | 0.99 | 1.20 |
| Impervious removed (pp) | 0.21 | 0.41 | 0.61 | 0.83 | 1.35 | 1.92* | 2.51* |

\* Impervious doses of 40 pp or more leave 41–71% of cells beyond observed conditions. Albedo is extrapolated from +0.15 upwards.

## S5: Scenarios (mean ΔT, °F)

| Scenario | Mean | 10th–90th pct | Fold spread | Beyond observed |
|---|---|---|---|---|
| Canopy +5 / +10 / +20 / +30 pp | −0.10 / −0.23 / −0.49 / −0.74 | −0.45 to +0.02 (at +10) | ±0.10 (at +10) | 6–7% |
| Impervious −5 / −10 / −20 / −30 pp | −0.21 / −0.41 / −0.83 / −1.35 | −0.65 to −0.14 (at −10) | ±0.15 (at −10) | 4–18% |
| Albedo +0.05 / +0.1 | −0.40 / −0.80 | −1.10 to −0.50 (at +0.1) | ±0.16 | 8% / 23% |
| Albedo +0.2 | −1.35 | — | — | **100%: do not use** |
| Green package (canopy +15, impervious −15, albedo +0.1) | −1.72 | −2.21 to −1.22 | ±0.28 | 26% |

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

- **Budget:** 20,000 dose units (+1 pp on one cell = 1 unit), placed on **783 cells** at a mean of +25.5 pp each.
- **Planned vs re-predicted cooling:** planned 1,702 °F·cells. Re-predicting the whole plan at once gives 981 °F·cells (58%), because neighbouring plantings share footprints.
- **Cooling achieved:** 0.15 °F mean in treated cells; 0.018 °F city-wide.
- **Scaling:** doubling the budget raises planned cooling 1.73×.

## Validation of the method (synthetic city with planted truths)

- **Stack choice:** the stack puts about 0.88 weight on physics, which generated the data, and matches or beats the best base model.
- **Effect recovery:** it recovers 53% of the planted canopy footprint, with spatial correlation 0.87. Effects are attenuated, so treat scenario magnitudes as conservative.

## Caveats

- Rounded, interpolated target: about 0.3 °F noise floor.
- Accuracy varies strongly by area (fold R² 0.26–0.74).
- No wind and no time-of-day record; daytime forcing is assumed.
- Albedo's observed range is narrow, so large albedo edits extrapolate.

## Regenerating

```bash
sparc core run --project configs/core_providence.yml --cv-curve        # add --resume after an interruption
python scripts/results_page/build_page.py output/core/providence/providence_uhi configs/core_providence.yml \
       --out output/core/providence/providence_uhi/results.html
```
