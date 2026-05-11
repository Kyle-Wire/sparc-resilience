# Reading SPARC Results

Here's the thing about data science outputs: they're easy to produce and hard to interpret. A model can spit out 47 metrics, and if you don't know what each one is *for*, you end up either trusting everything blindly or ignoring the numbers entirely and going with your gut anyway.

Neither is right. This guide is about knowing what each number is actually telling you — and equally importantly, what it *isn't* telling you.

For the full technical reference, see the [Manual](MANUAL.md). For a conceptual introduction before you dive in here, start with the [Concepts Guides](concepts/README.md).

---

## Table of Contents

- [Stage 2 — Did the Model Learn Your City?](#stage-2--did-the-model-learn-your-city)
- [Stage 3 — Did the Model Learn the Right Causes?](#stage-3--did-the-model-learn-the-right-causes)
- [Stage 4 — What Would Actually Happen?](#stage-4--what-would-actually-happen)
- [How Configuration Choices Propagate](#how-configuration-choices-propagate)
- [Quick Checklist Before Presenting Results](#quick-checklist-before-presenting-results)

---

## Stage 2 — Did the Model Learn Your City?

Stage 2 is where SPARC trains its models. Before you trust any causal estimate or scenario prediction, you need to know: did the model actually capture the spatial patterns in your data?

### R² — The Headline Number

Imagine plotting every observation in your study area — one dot per location, with the actual temperature on one axis and the model's predicted temperature on the other. R² tells you how tightly that cloud of dots clusters around a perfect diagonal line.

An R² of 0.94 means 94% of the spatial variation in temperature is explained by the model. The remaining 6% is noise, measurement error, or patterns the model couldn't capture.

**What these ranges mean in practice:**
- **0.0–0.3:** The model barely knows more than the city average. Check your data quality and predictor selection.
- **0.3–0.6:** Useful for exploration. Not reliable for policy decisions.
- **0.6–0.8:** Solid. Typical for coarser-resolution or more complex domains.
- **0.8–1.0:** Excellent. The model has genuinely learned the spatial structure.

**The critical detail:** SPARC computes R² on *held-out* data from spatially-separated folds — not on the data it was trained on. A model that memorizes training data can look perfect on paper and be useless for prediction. SPARC's spatial cross-validation keeps training and test locations physically separated. A 0.85 R² from spatially-buffered folds is worth far more than a 0.99 R² from random cross-validation on the same dataset.

### RMSE — The Human-Readable Error

R² tells you the fraction of variance explained. RMSE tells you the typical error in the same units as your outcome variable.

If your target is temperature in °F and the RMSE is 0.5, the model's predictions are typically off by about half a degree. If the RMSE is 2.0, you're off by two degrees on average — a very different situation for a heat vulnerability analysis. Lower is always better.

### The Model Comparison Table — Understanding the Ensemble

SPARC trains four fundamentally different models and then blends them. Each model sees the problem differently.

| Model | What it is | When it's unusually strong |
|-------|-----------|--------------------------|
| **OLS** | One global equation, applied everywhere | When relationships are roughly uniform across the whole study area |
| **GWR** | Locally-fitted linear models — the equation changes by location | When spatial nonstationarity is strong (effects differ by neighborhood) |
| **GWRF** | Local random forest at each location — captures non-linear local effects | Complex heterogeneous landscapes with interaction effects |
| **GGPGAM** | Smooth non-linear functions of predictors, spatially adjusted | Gradual non-linear trends without assuming a functional form |
| **Meta-ensemble** | All of the above, combined with physics constraints | Almost always the best — it inherits the strengths of each component |

If the meta-ensemble is substantially weaker than one of its components, something unusual is happening — worth investigating before moving to Stage 3.

---

## Stage 3 — Did the Model Learn the Right Causes?

Stage 3 is where SPARC goes from "the model fits the data" to "here is what actually drives the outcome, and by how much." This is where the causal structure you specified gets tested against reality.

### Structural Coefficients — The Causal Story in Numbers

A structural coefficient is not a correlation. It's an estimate of a *causal* effect — the change in the outcome you'd expect if you *set* the treatment to a new value, holding everything else constant.

For example: "Pct_Canopy → AAT_z = −0.022" means that each additional percentage point of tree canopy *causally* reduces temperature by 0.022 z-score units — after accounting for confounders like elevation and distance from water.

**What to look for:**
- **Sign:** Does the direction match physics? Canopy should cool (negative), impervious should warm (positive). A sign flip needs investigation.
- **Magnitude:** Is the size plausible given domain knowledge? A 1% canopy increase cooling the whole city by 5°F would be physically implausible.
- **Confidence interval:** Wide CI means uncertain estimate — either weak signal in data, or multicollinearity.

### ATE and CATE — Average vs. Spatially-Varying Effects

The **ATE (Average Treatment Effect)** answers "what is the typical effect of this treatment across the study area?"

The **CATE (Conditional Average Treatment Effect)** answers the more useful question: "where does this treatment work best?" CATE is estimated at every location, producing a map of effect strength. High CATE variability (large standard deviation) means the intervention has very different effects in different neighborhoods.

If CATE mean ≈ ATE, the effect is relatively uniform across space. If CATE standard deviation is large relative to the mean, spatial variation is too important to ignore — map it before making targeting decisions.

### E-value — A Number That Earns Its Place

The E-value might be the most honest metric in SPARC's output. It answers: *how strong would an unmeasured confounder need to be to completely explain away this causal finding?*

An E-value of 3.0 means a hidden variable would need to be associated with both the treatment and the outcome by a factor of 3× or more to nullify the result. An E-value of 1.2 means a fairly weak unmeasured confounder could flip the conclusion.

| E-value | What it means |
|---------|--------------|
| > 3.0 | Robust. A substantial hidden confounder would be needed to explain it away. |
| 2.0–3.0 | Reasonably robust. Interpret with standard caution. |
| 1.5–2.0 | Moderately sensitive. Consider what unmeasured variables might exist. |
| < 1.5 | Sensitive. The finding could be explained by a relatively weak confounder. |

E-values are sensitivity bounds, not proof. They tell you how *robust* a finding is, not whether it is *true*.

### Refutation Tests — The Adversarial Check

SPARC runs four adversarial tests for each causal edge:

| Test | What It Does | Pass Criterion |
|------|-------------|----------------|
| **Placebo treatment** | Replaces the real treatment with random noise | Estimated effect ≈ 0 (not significantly different from zero) |
| **Random common cause** | Adds a random confounder to the model | Original estimate does not change significantly |
| **Data subset** | Re-estimates on a random 80% subset | Estimate remains stable |
| **Unobserved confounding** | Simulates a hidden confounder correlated with treatment and outcome | Estimate remains in the same direction |

**How to read:** All four tests should pass. If a test fails, it suggests the causal estimate for that treatment may not be reliable — investigate the DAG structure and potential missing confounders.

### Causal Discovery (Diagnostic)

SPARC optionally runs automated structure learning algorithms (PC-stable, LiNGAM, GES) and compares the discovered graph to your expert DAG. The **edge F1 score** measures agreement:
- **F1 > 0.8:** Strong agreement — your DAG is well-supported by the data.
- **F1 0.5–0.8:** Moderate — some edges differ. Review the discovered graph for insights.
- **F1 < 0.5:** Weak — consider revising the DAG or investigating data quality.

This is a diagnostic, not a replacement for domain expertise. Use it to sanity-check your DAG, not to auto-generate one.

---

## Stage 4 — What Would Actually Happen?

This is the payoff stage: physics-constrained, causally-grounded predictions of what specific interventions would do to your outcome variable.

### Understanding the Three Prediction Modes

SPARC computes scenario predictions three ways and reports all of them. They're complementary, not competing.

**Mode 1 — Model re-prediction:** The trained ensemble is re-run with treatment variables set to their new values. This captures the full non-linear model response, including learned interaction effects. Best for understanding total effects including interactions the DAG might miss.

**Mode 2 — Causal pathway propagation:** Structural coefficients from Stage 3 are applied through spatially-varying weights, tracing effects through mediator pathways in your DAG. Lets you decompose the total effect into direct and indirect components (e.g., how much of canopy's cooling is direct shading vs. mediated through NDVI?).

**Mode 3 — Monte Carlo uncertainty propagation:** Coefficients are sampled from their estimated distributions hundreds of times. Each draw produces a full prediction surface. The result is a credible interval over outcomes — not a single number, but an honest range.

Large disagreement between Mode 1 and Mode 2 is worth investigating: it often means the model has learned non-linear effects the DAG doesn't represent, or there's a confounding pathway the DAG missed.

### Physics Guardrails — What's Protecting Your Predictions

These aren't warnings — they're enforced constraints that run on every scenario before results are shown.

| Guardrail | What it does |
|-----------|-------------|
| **Variable bounds** | Canopy can't exceed 100% or go below 0%. Physically impossible inputs are clipped. |
| **Diminishing returns** | Large interventions are tapered. Planting twice as many trees doesn't cool twice as much. This is a feature, not a bug — it reflects how physical systems actually behave. |
| **Sign enforcement** | If you've told SPARC that canopy can only cool, it enforces that. Physically implausible sign-reversals can't sneak through. |
| **Combined constraints** | Canopy + impervious surface can't together exceed 100% of land cover. Multi-variable limits are respected. |
| **Extrapolation guard** | If the scenario pushes feature values beyond what the training data covers, predictions are flagged. Numbers still appear, but with an explicit caution that you're in extrapolation territory. |

### Reading the Scenario Output Tables

| Column | What it's telling you |
|--------|----------------------|
| **Scenario name** | The intervention (e.g., "Canopy +10 pp") |
| **Avg. Actual Change** | The change *after* physics constraints are applied. Less than requested means a physical bound was hit. |
| **Mean Δ** | Average predicted effect across all spatial observations |
| **Std** | How much the effect varies across space — high Std means some areas benefit much more than others |
| **MC 5th / 50th / 95th** | The credible interval under causal uncertainty |

If the 5th and 95th percentiles have the same sign, the direction of effect is robust even under substantial uncertainty. If they straddle zero, you can't be confident the intervention helps — and this is important to know before committing resources.

---

## How Configuration Choices Propagate

The choices you make in `project.yml` ripple forward through the whole pipeline.

**Data and predictors:** More predictors can improve fit but risks collinearity. GWEN (Stage 1) selects spatially stable features. Higher data resolution generally improves GWR and GWRF. Use a projected metric CRS — not EPSG:4326 (degrees).

**Physics configuration:** Priors pull causal estimates toward literature-known values when data evidence is weak. Caps clip scenario predictions at physical bounds — tighter caps reduce apparent scenario effects but make them more reliable. Monotone constraints enforce directional physics and prevent physically absurd predictions.

**DAG structure:** Adding or removing edges changes both causal estimates and scenario predictions. Adding a mediator decomposes the total effect into direct and indirect pathways — useful for understanding mechanisms but requires confidence in the proposed pathway. Changing the estimator (OLS → DML → HGB) affects coefficient magnitude and confidence intervals; DML is recommended for most applications.

---

## Quick Checklist Before Presenting Results

- [ ] **Check R²** — is it high enough to trust model predictions for policy use? (Generally ≥ 0.7 for scenario work)
- [ ] **Verify all refutation tests passed** — failed tests mean treat that edge's scenario predictions with extra caution
- [ ] **Check the sign of structural coefficients** — unexpected signs need investigation before presenting
- [ ] **Compare Mode 1 vs. Mode 2** — large divergence means non-linear effects or DAG gaps worth exploring
- [ ] **Check MC percentiles** — if 5th and 95th percentiles have different signs, the intervention direction is uncertain
- [ ] **Look for diminishing returns** — for large interventions, actual predicted effect will be less than linear extrapolation
- [ ] **Check extrapolation flags** — scenarios in extrapolation territory should be flagged when presented
- [ ] **Report E-values alongside estimates** — this is what separates honest causal reporting from naive correlation

---

*Further reading: [Pipeline Guide](PIPELINE_GUIDE.md) · [Manual](MANUAL.md) · [Concepts](concepts/README.md)*
