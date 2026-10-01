# Reconciling earlier Providence numbers with the core pipeline

Earlier SPARC material reported Providence numbers that differ from the core pipeline's. These include:
- the legacy README's "Results: Urban Heat Island" section;
- statements that it improves on the 2025 *Urban Climate* paper (doi:10.1016/j.uclim.2025.102671).

This note explains each difference, so readers can tell what changed in the data and what changed in the evaluation.

The current numbers come from the full core run recorded in [`providence_core.md`](providence_core.md): 54,701 cells at 30 m, with 2 km spatial blocks. Refresh this note when that file is updated.

## Skill: the evaluation changed, not the data

| quantity | earlier report | core pipeline | why they differ |
|---|---|---|---|
| best out-of-fold R² | 0.944 ("enhanced meta-ensemble"); 0.902 standard | **0.56** (2 km blocks held out); 0.74 (500 m); 0.98 (random cells) | Evaluation design (below) |
| RMSE | 0.42 "z-units" | 1.14 °F (2 km), 0.27 °F (random cells) | Units, plus evaluation design |
| neural stacker | credited for the 0.944 | a convex blend is chosen; the gated neural residual does not beat it out of fold | The legacy stacker crashed before training and fell back to a weighted average (CORE_ROADMAP Appendix C) |
| units | "z-units" | °F | `AAT_z` is air temperature in °F (81–93.5) |

**Evaluation design.** The core run scores the same model under four cross-validation partitions:
- **Random cells:** R² 0.98. Every test cell has training neighbours 30 m away, so the score mostly measures interpolation. The earlier 0.90–0.94 figures came from this kind of evaluation, plus the leakage documented in Appendix C. They match what random-cell CV gives for any reasonable model here.
- **2 km blocks with a 667 m buffer:** R² 0.56. This answers the question a planner asks: how well does the model predict a neighbourhood it has never seen? It is the honest number.
- **500 m blocks:** R² 0.74.

On the same 2 km folds, the stack beats every standard baseline by more than 2 block-clustered SE:

| baseline | RMSE (°F) |
|---|---|
| gradient boosting on the same neighbourhood features | 1.37 |
| boosting with coordinates | 1.45 |
| regression-kriging | 1.58 |
| inverse-distance interpolation | 1.88 |

## Effects: broadly consistent, now with uncertainty

| quantity | earlier report | core pipeline | note |
|---|---|---|---|
| +10 pp canopy, city mean | −0.26 °F | −0.24 °F (realised +9.6 pp; model, fold-averaged) | Same size. The causal (spillover DML) estimate is about half: −0.11 °F per +10 pp (95% CI ± 0.15) |
| canopy per pp (DML) | −0.022 °F/pp | θ_sum −0.011 ± 0.008 °F/pp. Split: own cell +0.007, neighbourhood −0.018 | The earlier DML had no spatial cross-fitting and no spillover term. The core finds canopy's cooling comes from the surrounding area |
| impervious per pp (DML) | +0.022 °F/pp | θ_sum +0.044 ± 0.012 °F/pp. Model: −0.41 °F for −10 pp | The core estimate is larger and almost entirely a neighbourhood effect |
| albedo per unit (DML) | −2.76 °F | θ_sum −6.3 ± 2.8 °F (−0.63 per +0.1). Model: −0.79 °F per +0.1 | The albedo layer is not broadband (mean 0.40), so effects are per unit of this layer |
| canopy scenarios | removed impervious where canopy + impervious > 100% | no sum constraint | 35–38% of cells have canopy overhanging pavement |

## What carries over

- The ranking of levers per unit of change is the same as before: impervious reduction ≥ canopy, with albedo strong but scale-uncertain.
- The sign of every effect is unchanged.
- The cooling share from neighbourhood rather than own-cell change is new. Planting decisions should be read at the neighbourhood scale, not cell by cell.

## What should not be cited any more

Do not cite the following:
- any R² above about 0.6 for "new neighbourhoods";
- "z-units";
- the claim that a neural meta-learner produced the 0.944;
- legacy scenario tables that coupled canopy and impervious.

Use [`providence_core.md`](providence_core.md) and the run's `model_card.md` instead.
