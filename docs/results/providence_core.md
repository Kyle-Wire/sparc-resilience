# Providence (brown4.csv): core pipeline results

**Run:** `sparc core run --project configs/core_providence.yml --cv-curve`, 2026-10-01, commit `5e5c79b`.
- 173 min on 2 CPU threads; 101 of those minutes are the optional distance table.
- Post-run commands: `planner`, `emulator`, `uncertainty`, `writeup`.

**Interactive version:** the "Providence Heat Model" results page. It holds:
- maps, charts and tables;
- a Planner view, a Methods & validation section, and a live Design tool.

It is built from the same run directory; see [Regenerating](#regenerating).

These results supersede every earlier Providence number:
- the legacy pipeline (see [`reconciliation.md`](reconciliation.md));
- the previous core run, which used generic forcing and stronger physics priors.

## Data

| Item | Value |
|---|---|
| Cells | 54,701 on an exact 30 m lattice (grid 334 × 287, 57% filled) |
| Target | `AAT_z`: air temperature in **°F** (81–93.5), not a z-score. It is CAPA's area-wide surface for the **Rhode Island Heat Watch afternoon traverse, 2020-07-29, 15–16 EDT**. |
| Product | 72% of values are whole degrees: a classed, modelled product. Traverse points are extended to every cell with land-cover predictors, so fitted land-cover effects partly describe that model. Rounding noise is about 0.25 °F. |
| Background | Field median, 88.0 °F; ΔT is relative to it |
| Forcing (physics) | ERA5 for 2020-07-29, 15–16 EDT:<br>• SW↓ 636 W/m², clear-sky index 0.87<br>• net LW −71 W/m²<br>• 2 m temperature 87.8 °F, against 87.5 °F at the KPVD airport station<br><br>KPVD wind: 7.7 m/s from the south |
| Albedo | Mean 0.40, while Sentinel-2 broadband albedo averages 0.155 here. The layer is not broadband, so albedo effects are per unit of this layer. |
| Land cover | Canopy + impervious exceeds 100% on 37% of cells (canopy overhangs pavement), so no sum constraint is imposed |
| QA | 4 albedo values above 0.9 clipped |

## Cross-validation design

Every accuracy number below is out of fold under **5-fold spatial-block CV**:
- square blocks of 2,000 m, with 23 blocks containing data;
- each fold holds out whole blocks, about 10,900 cells;
- training cells within 667 m (block ÷ 3) of any test cell are dropped.

The block size is the S1 target-residual correlation range, which reached the correlogram's 2 km cap. One partition is shared by:
- the base models and the stacker;
- the conformal intervals;
- causal cross-fitting;
- the baselines.

### Skill vs distance from training data

| Held-out unit | Buffer (m) | Blocks | Stacked R² | RMSE (°F) | Fold R² range | Best baseline R² |
|---|---|---|---|---|---|---|
| Random cells (leaky reference) | 0 | 54,137 | 0.975 | 0.27 | 0.97–0.98 | IDW 0.989 ▲ |
| 500 m blocks | 167 | 245 | 0.743 | 0.87 | 0.59–0.80 | HGB + x,y 0.679 ▼ |
| 1,000 m blocks | 333 | 73 | 0.659 | 1.00 | 0.41–0.81 | HGB on stack inputs 0.548 ▼ |
| **2,000 m blocks (main)** | **667** | **23** | **0.551** | **1.15** | **0.26–0.73** | **HGB on stack inputs 0.361 ▼** |

▼ means the stack is better by more than 2 block-clustered SE; ▲ means the baseline is better.

**Random-cell CV measures interpolation.** Inverse-distance interpolation of the neighbours (no covariates at all) beats every model there. That is how earlier SPARC numbers reached 0.94–0.98.

**On unseen neighbourhoods the stack beats every baseline:**

| Baseline | RMSE (°F) | Held-out R² | ΔMSE vs stack |
|---|---|---|---|
| Gradient boosting on the same neighbourhood features | 1.37 | 0.36 | +0.56 ± 0.22 |
| Boosting + coordinates | 1.45 | 0.29 | — |
| Regression-kriging | 1.58 | 0.15 | — |
| Boosting without location | 1.75 | −0.05 | — |
| Inverse-distance interpolation | 1.88 | −0.21 | — |

Where the stack's gain comes from:
- the area-of-influence features raise held-out R² from −0.05 to 0.36;
- the geographically weighted and physics models plus the stacker raise it from 0.36 to 0.55.

## Models (main 2 km blocks)

| Model | R² | RMSE (°F) | Blend weight (mean over folds) |
|---|---|---|---|
| Ridge regression (raw + focal features) | 0.30 | 1.43 | 0.00 |
| MGWR (FFT backfitting) | 0.39 | 1.34 | 0.06 |
| Geographically weighted random forest | 0.42 | 1.30 | 0.02 |
| GAM + spatial smooth | 0.55 | 1.15 | 0.63 |
| Physics (heat transport) | 0.49 | 1.22 | 0.29 |
| **Stacked model** | **0.551** | **1.146** | — |

**Stacker.** A non-negative blend (NNLS), chosen on the held-out blocks. Held-out RMSE by candidate:

| Candidate | RMSE (°F) |
|---|---|
| Non-negative blend (chosen) | 1.146 |
| Equal-weight mean | 1.195 |
| Blend + neural residual, λ_PDE = 0 | 1.223 |
| Blend + neural residual, λ_PDE = 0.1 | 1.199 |
| Blend + neural residual, λ_PDE = 1 | 1.312 |

**Intervals.** 90% cross-conformal intervals cover 89.5% of held-out cells, with a mean half-width of ±1.88 °F.
- Per fold, coverage ranges from 0.80 to 0.98.
- By distance from training data it is 0.88–0.93; the distance-adaptive version gives 0.89–0.91.
- Zone 4 is under-covered (0.64, 298 cells).

**Physics.** Fitted across folds:
- L = 256 ± 20 m and gain a = 45 ± 5;
- canopy shading s = 0.37 ± 0.09 under a weakly informative prior N(0.6, 0.3²);
- the shade-saturation scale κ = 2.1 ± 0.3.

Advection by the campaign wind was tested out of fold and dropped: ΔRMSE was +0.003 ± 0.017 °F, so it did not help.

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
| Canopy added (pp) | 60% | 12% | 8% | 44 pp | +0.0001 °F | −0.018 °F |
| Impervious removed (pp) | 6% | 51% | 22% | 43 pp | +0.0032 °F per +1 pp | +0.053 °F per +1 pp |
| Albedo added | 98% | 1% | 0% | 0.22 | −0.31 °F per unit | −8.0 °F per unit (−0.08 per +0.01) |

The remaining cells, 20% for canopy, follow the accelerating (S-shaped) curve.

City-wide mean cooling by neighbourhood dose (°F):

| Dose | +5 | +10 | +15 | +20 | +30 | +40 | +50 |
|---|---|---|---|---|---|---|---|
| Canopy (pp) | 0.09 | 0.19 | 0.29 | 0.39 | 0.59 | 0.77 | 0.93 |
| Impervious removed (pp) | 0.20 | 0.40 | 0.60 | 0.83 | 1.33 | 1.89* | 2.47* |

\* Impervious doses of 40 pp or more leave 41–71% of cells beyond observed conditions. Albedo is extrapolated from +0.15 upwards, for 98–100% of cells.

## S5: Scenarios (mean ΔT, °F)

| Scenario | Model mean ± SE (jackknife) | Beyond observed | Linear causal estimate (95%) |
|---|---|---|---|
| Canopy +10 pp | −0.19 ± 0.14 | 6% | −0.10 (−0.25 to +0.04) |
| Canopy +30 pp | −0.59 ± 0.40 | 7% | −0.30 (−0.71 to +0.12) |
| Impervious −10 pp | −0.40 ± 0.19 | 5% | −0.40 (−0.61 to −0.18) |
| Impervious −30 pp | −1.33 ± 0.39 | 18% | −1.15 (−1.77 to −0.52) |
| Albedo +0.05 / +0.1 | −0.42 ± 0.28 / −0.85 ± 0.50 | 8% / 23% | −0.32 / −0.63 (−1.18 to −0.09) |
| Albedo +0.2 | −1.44 | **100%: do not use** | −1.27 |
| Green package (canopy +15, impervious −15, albedo +0.1) | −1.72 ± 0.55 | 26% | −1.38 (−2.04 to −0.71) |

**The model sits inside the causal band for every scenario.**

**Canopy is now the most uncertain lever.** Its 95% jackknife interval includes zero at every dose, as does the causal estimate. With a weakly informative physics prior, the data do not pin the canopy effect down as tightly as before. The earlier tighter estimate was partly prior-driven (see [Validation](#validation-studies)).

**Uncertainty components** (from `sparc core uncertainty`):
- Estimation 95% intervals: canopy +10 pp −0.46 to +0.08; impervious −10 pp −0.78 to −0.03; package −2.80 to −0.65.
- The multiverse specification range is added when that study completes.

## Climate futures (CMIP6, delta method)

**Method:**
- **Warming:** June–August mean daily maximum (`tasmax`) between 1995–2014 and each IPCC AR6 period, from 24 CMIP6 models (one member each).
- **Location:** land-weighted at Providence.
- **Future maps:** today's measured field plus each model's warming, plus an adaptation scenario.

| Pathway | 2021–2040 | 2041–2060 | 2081–2100 |
|---|---|---|---|
| SSP1-2.6 | +1.8 °F (1.1–2.8) | +2.6 (1.5–4.2) | +2.4 (1.5–4.5) |
| SSP2-4.5 | +1.8 (1.2–3.3) | **+3.0 (2.3–4.9)** | +4.6 (3.3–7.0) |
| SSP3-7.0 | +1.9 (1.2–3.6) | +3.6 (2.1–6.7) | +7.8 (4.6–11.6) |
| SSP5-8.5 | +2.0 (1.2–3.6) | +4.1 (3.0–6.5) | +9.4 (5.8–13.6) |

The table shows the median across models, with the 10th–90th percentile in brackets.

**Heat exposure:** share of the study area at or above 90 °F on an afternoon like the campaign's. Today it is 13%.

| Pathway · period | No adaptation | Canopy +30 pp | Impervious −30 pp | Albedo +0.05 |
|---|---|---|---|---|
| SSP2-4.5 · 2041–2060 | 76% | 67% | 42% | 68% |
| SSP2-4.5 · 2081–2100 | 90% | 89% | 78% | 90% |
| SSP5-8.5 · 2041–2060 | 90% | 84% | 71% | 83% |
| SSP5-8.5 · 2081–2100 | 100% (90% ≥ 95 °F) | 100% (86%) | 100% (74%) | 100% (87%) |

**Share of the median mid-century SSP2-4.5 warming each package offsets:**
- impervious −30 pp: 44%;
- canopy +30 pp: 19%;
- albedo +0.05: 14%.

**Assumptions (delta method):**
- Land-cover effects are assumed unchanged as the background warms.
- The hottest days may warm more than the seasonal average of daily highs.

## People and hot days (planner pack)

The planner pack combines HRSL residents (174k in the study area), ESA WorldCover 2021, and KPVD daily maxima for 1995–2014.

| Case | Residents in cells ≥ 90 °F | Hot afternoons ≥ 90 °F per summer (cell mean) |
|---|---|---|
| Today | 31,900 (18%) | 10.5 |
| Today, with the green package | 1,600 (1%) | — |
| SSP2-4.5, 2041–2060 | 160,800 (93%) | 20.1 |
| … with the package | 79,500 (46%) | 13.9 |
| SSP2-4.5, 2081–2100 | 170,300 (98%) | 27.2 → 19.7 with the package |
| SSP5-8.5, 2081–2100 | all | 51.9 → 42.7 with the package |

**How hot afternoons are counted.** Each cell's count is the airport's daily maximum plus the cell's measured campaign-afternoon difference from the airport.
- Halving that difference (a lower bound for less clear, calm days) changes today's count to 9.5.
- Futures add each model's warming; the median over models is shown.

**Who benefits.** The package's cooling is shared almost evenly across population-density and age-group quintiles: concentration indices are 0.00 for density, −0.004 for residents aged 60+, and +0.012 for under-5s.

**Plantable space.** Open land plus a fifth of built-up area leaves room for 20 pp more canopy per cell on average; 13% of cells have none. The budget plan respects this cap.

## S6: Causal validation

Spatial DML with the 2 km blocks for cross-fitting, plus exposure-mapping spillover. These numbers do not depend on the physics model, so they match the previous run.

| Treatment | Check | Model | Causal | Verdict |
|---|---|---|---|---|
| Canopy (per +1 pp) | Neighbourhood adoption vs θ_own + θ_nbr | — | −0.011 ± 0.008 | consistent |
| | Own cell vs θ_own | +0.0001 | +0.0071 ± 0.0014 | magnitude differs |
| Impervious (per +1 pp) | Neighbourhood adoption vs θ_own + θ_nbr | — | +0.044 ± 0.012 | consistent |
| Albedo (per +0.1) | Neighbourhood adoption vs θ_own + θ_nbr | — | −0.63 ± 0.28 | consistent |

**Robustness to hidden confounding:**

| Treatment | E-value | Robustness value |
|---|---|---|
| Impervious | 1.32 | 14.7% |
| Albedo | 1.84 | 10.4% |
| Canopy (own cell) | 1.08 | 1.2%, not distinguishable from zero |

## S7: Canopy budget plan

- **Budget:** 20,000 dose units (+1 pp on one cell = 1 unit), now capped by plantable space. It is placed on **1,348 cells** at a mean of +14.8 pp each.
- **Planned vs re-predicted cooling:** planned 1,374 °F·cells; re-predicting the whole plan at once gives 1,046 (76%).
- **Cooling achieved:** 0.12 °F mean in treated cells.
- **Scaling:** doubling the budget raises planned cooling 1.74×.

## Validation studies

**Baselines:** see the distance table above. The stack beats every standard tool on unseen neighbourhoods.

**Placebos** (`sparc core placebo`, 60 m re-fits). Effects should be about zero:

| Placebo | Model effect per sd | Causal effect per sd | Verdict |
|---|---|---|---|
| Random smooth layer | +0.02 ± 0.05 °F | +0.03 ± 0.11 °F | both pass |
| Canopy shifted half the map away | −0.30 ± 0.10 °F (33% of the real −0.89 ± 0.49) | −0.03 ± 0.14 °F | **model fails**, causal passes |
| Canopy rotated 180° | −0.25 ± 0.22 °F | — | model passes |
| Impervious shifted | −0.72 ± 0.52 °F | — | model passes |
| Impervious rotated | −0.17 ± 0.25 °F | −0.26 ± 0.10 °F | model passes, **causal fails** |

The model passes 4 of 5 and the causal check 4 of 5.

**The first placebo run exposed a prior-driven canopy effect.** It used physics priors with sd 0.1, and the model passed only 3 of 5 (shifted canopy −0.44 °F, 45% of the real effect).

| Shading prior sd | Real canopy | Shifted placebo | Physics-only held-out R² |
|---|---|---|---|
| 0.1 | s 0.56, −1.30 °F per sd | s 0.53, −0.69 °F per sd | 0.533 |
| 0.3 (now used) | s 0.34, −1.03 °F per sd | s 0.09, −0.27 °F per sd | 0.548 |

With sd 0.1 the shading strength stayed at the prior whether or not the layer was real. At sd 0.3 the data separate the real canopy from the placebo, and held-out skill improves slightly.

The remaining spurious effect comes from the statistical models. Any smooth layer can borrow chance large-scale correlation, so modelled canopy effects should be read against a placebo floor of about 0.3 °F per sd.

**Simulation check** (`sparc core simcheck`, 90 m). A known canopy effect is planted on the city's real layout; the target product is imitated; the full pipeline is re-run. Physics generator, 20 replicates:

| Measure | Result |
|---|---|
| Recovered effect share | median 1.13 (IQR 0.90–1.46) |
| 95% jackknife interval covers the truth | 80% |
| Causal interval covers the truth | 65% |
| Per-cell rank correlation | 0.27 |
| Prediction-interval coverage | 0.90 |

- The simulated worlds are somewhat easier than reality: held-out R² is 0.6–0.8, against 0.55.
- The 5-fold jackknife intervals run narrower than nominal, so read them as lower bounds on uncertainty.
- Other generators (null, additive, …) are added as they complete.

**Literature:**
- Canopy: 0.11 ± 0.08 °C per +0.10 cover. Ziter et al. 2019 report 0.07–0.15; the Krayenhoff et al. 2021 model review reports 0.3.
- Albedo: 0.47 ± 0.28 °C per +0.10, inside Krayenhoff's 0.2–0.6 and Santamouris 2014's 0.3–0.9 (with the scale caveat).

**Open-data predictors** (`sparc core features`). Agreement with brown4's own layers:

| Feature | Pearson r |
|---|---|
| Elevation | 0.98 |
| NDVI | 0.89 |
| Canopy | 0.77 |
| Impervious | 0.76 |
| Albedo | 0.63 |
| Water distance | 0.46 |

A refit on open data only (`configs/core_providence_open.yml`) is in progress.

**Design-tool emulator vs the full model:**
- canopy patches: 100% pass, median error 0.003 °F;
- impervious: 88% pass;
- albedo: 44% pass (median error 0.05 °F), because albedo edits quickly leave the observed range.

The in-browser emulator matches the Python reference to within 1e-4 °F.

## Caveats

- **Product target.** Classed and modelled; about 0.25 °F noise floor.
- **Uneven accuracy.** It varies strongly by area (fold R² 0.26–0.73).
- **One afternoon only.** There is no night-time model.
- **Albedo scale.** The layer is not broadband, so albedo effects are relative to it.
- **Canopy effects are uncertain.** Read them with the placebo floor and the simulation check in mind.

## Regenerating

```bash
sparc core forcing -p configs/core_providence.yml --date 2020-07-29 --hours 15-16 --station 72507014765 \
       --out configs/forcing/providence_2020-07-29.json
sparc core layers -p configs/core_providence.yml --out configs/layers/providence_layers.parquet
sparc core run -p configs/core_providence.yml --cv-curve                 # --resume after an interruption
R=output/core/providence/providence_uhi
sparc core planner $R -p configs/core_providence.yml
sparc core emulator $R -p configs/core_providence.yml
sparc core placebo -p configs/core_providence.yml --coarse 60
sparc core simcheck -p configs/core_providence.yml --design physics=20,null=12,additive=12 --out output/core/providence/simcheck/all
sparc core multiverse -p configs/core_providence.yml --out output/core/providence/multiverse
sparc core uncertainty $R --multiverse output/core/providence/multiverse --simcheck output/core/providence/simcheck/* \
       --placebo output/core/providence/providence_uhi_placebo/placebo.json
sparc core writeup $R
python scripts/results_page/build_page.py $R configs/core_providence.yml --out $R/results.html
```
