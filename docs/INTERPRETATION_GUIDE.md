# SPARC Results Interpretation Guide

This guide explains — in plain language — what the numbers in SPARC's output mean, how to read scenario tables, and how different inputs affect results. For the full technical reference, see the [Manual](MANUAL.md) and [Pipeline Guide](PIPELINE_GUIDE.md).

---

## Table of Contents

- [Stage 2: Model Performance Metrics](#stage-2-model-performance-metrics)
- [Stage 3: Causal Validation Metrics](#stage-3-causal-validation-metrics)
- [Stage 4: Scenario Output Metrics](#stage-4-scenario-output-metrics)
- [How Inputs Affect Outputs](#how-inputs-affect-outputs)
- [Quick Checklist for Reading Scenario Tables](#quick-checklist-for-reading-scenario-tables)

---

## Stage 2: Model Performance Metrics

### R² (Coefficient of Determination)

**What it is:** The fraction of spatial variance in your target variable that the model explains. An R² of 0.915 means the model captures 91.5% of the spatial pattern.

**How to read it:**
- **0.0–0.3:** Poor — the model explains very little. Check your data quality, predictor selection, and CRS.
- **0.3–0.6:** Moderate — useful for exploratory analysis but not reliable for scenario predictions.
- **0.6–0.8:** Good — typical for coarser-resolution data or complex domains.
- **0.8–1.0:** Excellent — strong predictive power. Common with dense, high-resolution spatial data.

**Important:** R² is computed on out-of-fold predictions (spatial cross-validation), not on training data. This guards against overfitting and spatial leakage. A high R² on spatially-buffered folds is far more credible than a high training R².

### RMSE (Root Mean Square Error)

**What it is:** The average magnitude of prediction error, in the same units as your target variable.

**How to read it:** Lower is better. Compare RMSE across models to see which one predicts most accurately. For example, if your target is temperature in °F, an RMSE of 0.50 means the model is typically off by about half a degree.

### Model Comparison

SPARC trains four base models and a meta-ensemble:

| Model | Strengths | When It Shines |
|-------|-----------|---------------|
| **OLS** | Simple, interpretable, fast | Baseline reference; useful when relationships are globally linear |
| **GWR** | Spatially varying coefficients | When the effect of a predictor changes across space |
| **GWRF** | Non-linear + spatial | Complex interactions, heterogeneous landscapes |
| **GGPGAM** | Semi-parametric flexibility | Smooth non-linear trends with spatial structure |
| **Meta-ensemble** | Stacks all base models | Almost always the best; uses Optuna-tuned LightGBM with monotonic physics constraints |

The meta-ensemble typically matches or exceeds the best individual model. If it doesn't, it usually means one model dominates (as in the ForceSMIP case, where GWRF ≈ meta-ensemble).

---

## Stage 3: Causal Validation Metrics

### Structural Coefficients

**What they are:** The estimated causal effect of each edge in your DAG, after controlling for confounders. For example, "Pct_Canopy → AAT_z = −0.022" means each 1 percentage-point increase in tree canopy is associated with a 0.022 z-unit decrease in temperature, holding other variables constant.

**How they're estimated:** SPARC uses one of three estimators:
- **DML (Double Machine Learning):** Doubly-robust; best for most applications.
- **HGB (Histogram Gradient Boosting):** Non-parametric; good for non-linear relationships.
- **OLS:** Simple linear; interpretable but assumes linearity.

Coefficients are blended with literature priors via adaptive shrinkage when priors are provided. This pulls data-driven estimates toward known physics without overriding the data.

### ATE (Average Treatment Effect)

**What it is:** The average causal effect of a treatment across all spatial observations, estimated via the backdoor criterion. For example, "ATE of Pct_Canopy = −0.015" means increasing canopy by 1 pp cools the average location by 0.015 z-units.

### CATE (Conditional Average Treatment Effect)

**What it is:** How the causal effect varies across space. SPARC estimates CATE using EconML's CausalForestDML, producing a per-observation treatment effect. The CATE mean and standard deviation tell you:
- **Mean ≈ ATE:** The effect is relatively uniform across space.
- **Large Std:** The effect varies significantly — some locations respond much more than others.

### E-value

**What it is:** A sensitivity measure for unmeasured confounding. The E-value answers: "How strong would an unmeasured confounder need to be — in terms of its association with both the treatment and the outcome — to fully explain away the observed causal effect?"

**How to read it:**
- **E-value > 2.0:** Strong. An unmeasured confounder would need to double the risk of both treatment and outcome to nullify the finding.
- **E-value 1.5–2.0:** Moderate. Some robustness, but a reasonably strong confounder could explain the effect.
- **E-value < 1.5:** Weak. The causal claim is sensitive to unmeasured confounding — interpret with caution.

**Important:** E-values are sensitivity bounds, not proof. They tell you how *robust* a finding is, not whether it is *true*.

### Refutation Tests

SPARC runs four refutation tests from DoWhy for each treatment:

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

## Stage 4: Scenario Output Metrics

### Three Prediction Modes

SPARC simulates interventions using three complementary approaches:

#### Mode 1 — Ensemble Re-prediction

The meta-ensemble re-predicts the outcome after modifying the treatment variable in the input data. This captures the full non-linear model response, including all learned interactions.

**Best for:** Understanding the total effect including non-linear interactions the DAG might miss.

#### Mode 2 — DAG Coefficients × Local Spatial Weights

The structural causal coefficients from Stage 3 are applied through the MGWR spatial weight matrices, producing location-specific effect estimates. Indirect effects through mediators (e.g., Canopy → NDVI → Temperature) are traced through the DAG.

**Best for:** Decomposing effects by pathway and understanding spatial heterogeneity in treatment effects.

#### Mode 3 — Monte Carlo Uncertainty Propagation

Coefficients are drawn from their estimated distributions (mean ± standard error) multiple times. Each draw produces a full prediction surface, yielding percentile-based credible intervals.

**Best for:** Quantifying uncertainty. The 5th and 95th percentiles bracket the plausible range of effects.

### Physics Constraints Applied in Scenarios

| Constraint | What It Does | Example |
|-----------|-------------|---------|
| **Variable bounds (caps)** | Prevents variables from exceeding physical limits | Canopy cannot exceed 100% or go below 0% |
| **Diminishing returns (√ taper)** | Compresses large deltas to reflect decreasing marginal effectiveness | +50 pp canopy doesn't cool 5× as much as +10 pp |
| **Sign enforcement (monotone)** | Ensures interventions move outcomes in the physically correct direction | More canopy can only cool, never warm |
| **Combined constraints** | Enforces multi-variable physical limits | Canopy + Impervious ≤ 100% |
| **Extrapolation guards** | Flags observations pushed beyond training data range via Mahalanobis distance | Large values → predictions may be unreliable |

### Reading the Scenario Tables

Each scenario table row shows:

| Column | Meaning |
|--------|---------|
| **Scenario name** | The intervention applied (e.g., "Canopy +10 pp") |
| **Avg. Actual Change** | The change after physics constraints are applied — may be less than requested if caps were hit |
| **Mean Δ (z-units or native)** | Average predicted change across all spatial observations |
| **Std** | Standard deviation of the change across space — higher means more spatial heterogeneity |
| **MC 5th / 50th / 95th** | Credible interval from Monte Carlo draws — the range of plausible effects |

---

## How Inputs Affect Outputs

Understanding how your configuration choices propagate through the pipeline helps you make better modeling decisions.

### Data and Predictors

- **Adding more predictors** can improve R² but risks overfitting if they are collinear. GWEN (Stage 1) helps by selecting only spatially-stable features.
- **Data resolution** matters: denser data (more observations per unit area) generally improves geographically-weighted model performance. Coarse data limits GWR/GWRF's ability to detect local variation.
- **CRS choice** affects distance calculations. Use a projected (metric) CRS appropriate for your study area — not EPSG:4326 (degrees).

### Physics Configuration

- **Priors (priors.yml):** Providing literature-based coefficient priors pulls the causal estimates toward known physics via adaptive shrinkage. Stronger priors have more influence when data evidence is weak.
- **Caps (caps.yml):** Variable-level minimum/maximum bounds prevent scenarios from producing physically impossible values. Tight caps may reduce apparent scenario effects.
- **Monotone constraints:** Enforcing that canopy can only cool (−1) or impervious can only warm (+1) improves plausibility but assumes the sign is universal across your study area.

### DAG Structure

- **Adding an edge** introduces a causal pathway. The coefficient will be estimated and used in Mode 2 scenario propagation.
- **Removing an edge** drops that pathway — effects previously attributed to it will be redistributed or absorbed by other paths.
- **Adding a mediator** (e.g., Canopy → NDVI → Temperature) decomposes the total effect into direct and indirect components.
- **Changing the estimator** (OLS → DML → HGB) affects coefficient magnitude and confidence intervals. DML is recommended for most applications.

### Scenario Definition

- **Intervention magnitude** directly scales the predicted effect (approximately linearly for small changes, with √ taper for large changes).
- **Joint scenarios** (multiple variables changed simultaneously) capture interaction effects and combined constraints that single-variable scenarios miss.

---

## Quick Checklist for Reading Scenario Tables

Use this checklist when reviewing SPARC scenario outputs:

- [ ] **Did physics constraints cap the intervention?** Compare "Avg. Actual Change" to the requested change. If actual < requested, bounds were hit.
- [ ] **Is the direction of effect physically plausible?** More canopy should cool, more impervious should warm. If not, investigate the DAG and data.
- [ ] **Compare Mode 1 vs. Mode 2.** Large disagreement may indicate non-linear effects the DAG doesn't fully capture.
- [ ] **Check the Std column.** High standard deviation means the effect varies significantly across space — consider mapping CATE to see where effects are strongest.
- [ ] **Review MC percentiles.** If the 5th and 95th percentiles have the same sign, the direction of effect is robust under uncertainty.
- [ ] **Look for diminishing returns.** For large interventions (e.g., +50 pp canopy), the √ taper compresses gains. This is intentional — don't extrapolate linearly.
- [ ] **Check extrapolation flags.** If many observations are flagged by the Mahalanobis guard, the scenario may be pushing beyond the data's support.
- [ ] **Cross-reference with refutation tests.** If a treatment failed refutation tests in Stage 3, treat its scenario predictions with extra caution.

---

## Further Reading

- [**Manual**](MANUAL.md) — Complete configuration reference and model documentation
- [**Pipeline Guide**](PIPELINE_GUIDE.md) — Step-by-step execution walkthrough with troubleshooting
- [**Contributing**](CONTRIBUTING.md) — Developer guide for adding models and domain templates
