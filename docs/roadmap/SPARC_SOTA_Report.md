# SPARC Labs — State of the Art Report

**SPARC Labs LLC | May 2026**
**Spatial Predictive Analytics for Resilience and Causality**

---

## What SPARC Is

SPARC is a spatial analysis and causal inference engine built for practitioners who need to answer two questions that no existing tool answers simultaneously:

> *Where does this phenomenon occur, and why — and what happens if we intervene?*

It combines physics-informed machine learning, Bayesian causal inference, and transfer learning into a single end-to-end pipeline. A user brings spatial observations (sensor readings, survey data, satellite-derived measurements). SPARC produces probabilistic spatial predictions, causal attribution with uncertainty, and counterfactual scenario maps — all in a single reproducible run.

---

## The Five-Stage Pipeline

SPARC executes five stages in sequence. Each stage produces traceable artifacts consumed by the next.

### Stage 0 — Spatial Structure Discovery

Before any model is fit, SPARC characterizes the spatial autocorrelation structure of the outcome variable through correlograms, Moran's I statistics, and Bayesian Matérn kernel fitting. It also detects anisotropy — whether spatial dependence is stronger in one compass direction than another (e.g., along prevailing wind directions for air quality, along river valleys for groundwater).

The output is an automatically configured `KernelField` that informs all downstream spatial smoothing. No other pipeline tool automatically wires kernel structure from empirical correlogram fitting into the model training loop.

### Stage 1 — Geographically Weighted Variable Selection (GWEN)

SPARC employs a Geographically Weighted Elastic Net (GWEN) to rank predictors by their spatially-varying relevance. Unlike global variable selection, GWEN identifies that a variable may be highly predictive in one part of a domain and irrelevant in another. This produces spatially-resolved feature importance maps before the neural model is trained, guiding both interpretability and feature engineering.

### Stage 2 — Spatial Cross-Validation and Neural Meta-Learning

Four base models (OLS, GWR, GWRF, GAM) are trained and evaluated under spatial block cross-validation. Their outputs feed a neural meta-learner — `SPARCMetaLearner` — that combines them into a single physics-informed prediction.

**The `SPARCMetaLearner` architecture is the technical core of SPARC:**

- **SharedTrunk:** A physics encoder built on SIREN (sinusoidal representation networks) that ingests physics features — elevation, land cover fractions, solar forcing variables — through a learnable spatially-varying process-rate field $\alpha(\mathbf{x})$. The trunk encodes physics knowledge that transfers across cities, domains, and datasets.

- **CityHead:** A lightweight city-specific output layer that adapts the trunk's representation to a particular deployment context. Only the CityHead is retrained when moving to a new city; the trunk transfers.

- **SparseSpatialAttention:** A spatial attention mechanism over observation locations, allowing the model to weight nearby observations more strongly when spatial heterogeneity is high.

- **MC-Dropout:** Uncertainty quantification through Monte Carlo sampling, producing posterior mean and credible interval maps rather than point predictions.

- **PDE Curriculum:** A 10-term physics loss enforces that predictions satisfy the governing partial differential equation of the domain. The curriculum progressively activates physics terms during training — starting with smoothness and energy balance, then introducing transient and cross-domain terms — preventing physics constraints from overwhelming data-driven signals early in training.

### Stage 3 — Bayesian Causal Inference

SPARC runs a full causal inference pipeline on the trained spatial model:

- **MC³ DAG Search:** Markov Chain Monte Carlo over the space of directed acyclic graphs to find the most probable causal structure consistent with the data, respecting user-supplied domain knowledge constraints.
- **NUTS Posterior Sampling:** No-U-Turn Sampler over edge weights in the inferred DAG, producing full posterior distributions over causal effect sizes rather than point estimates.
- **Double Machine Learning (DML) / CATE:** Debiased causal effect estimation with spatial heterogeneity — treatment effects that vary continuously across geography.
- **DoWhy Refutation Suite:** Four automated robustness checks (placebo treatment, random confounder, data subset, unobserved confounding) that must all pass before a causal estimate is reported.
- **Sensitivity Analysis and E-values:** Every causal estimate is accompanied by an E-value quantifying how strong an unmeasured confounder would need to be to reverse the finding.

### Stage 4 — Physics-Constrained Scenario Simulation

Given the causal structure and trained model, SPARC simulates counterfactual interventions: "what would the spatial field look like if we changed these inputs?" A four-tier delta computation applies interventions at the feature level, propagates them through the physics model, and produces scenario maps with uncertainty bounds. Budget allocation optimization identifies which set of interventions maximizes a user-defined objective within resource constraints.

---

## Key Differentiators

### 1. Physics-Informed Spatial Machine Learning

SPARC is not a generic neural network applied to spatial data. The governing physics of each domain is encoded as a loss function that shapes the model's representation during training. This means predictions are constrained to be physically plausible — they cannot, for example, produce a spatial temperature field that violates energy conservation or has discontinuities inconsistent with the thermal diffusion equation. This physics constraint also dramatically reduces the data required for accurate prediction, because the model cannot fit physically impossible solutions even when data is sparse.

No other spatial prediction tool combines neural network flexibility with hard physics constraints in a curriculum-learning framework.

### 2. Transfer Learning Across Cities and Domains

The SharedTrunk / CityHead architecture enables physics knowledge learned in one location to transfer to another. A trunk trained on urban heat dynamics in Providence, Rhode Island encodes generalizable physical priors — how vegetation fraction relates to cooling, how building density affects thermal storage, how elevation interacts with solar forcing. That prior accelerates training in Boston, Phoenix, and eventually Lagos — a city where no local calibration data exists at all.

The continual learning infrastructure (Elastic Weight Consolidation + Experience Replay) prevents the trunk from forgetting earlier cities as new ones are added, enabling a progressively improving global prior.

### 3. Uncertainty Quantification at Every Step

Most spatial models produce a point prediction. SPARC produces a full posterior at every stage: Bayesian Matérn kernel fitting in Stage 0, MC-Dropout spatial predictions in Stage 2, NUTS-sampled causal effect posteriors in Stage 3, and uncertainty-propagated scenario bounds in Stage 4. Every map SPARC produces has a companion uncertainty map. Every causal estimate comes with credible intervals and refutation results.

### 4. Causal Inference as First-Class Output

Most spatial models stop at prediction. SPARC is designed from the ground up to answer causal questions: does adding tree canopy *cause* temperature reduction, or do cooler areas simply have more trees? The MC³ + NUTS pipeline answers this with posterior uncertainty rather than a binary yes/no. DML removes confounding from continuous treatments. The refutation suite stress-tests every finding against alternative explanations.

SPARC treats spatial prediction as a prerequisite to causal inference, not as the end product.

### 5. Modifiable Areal Unit Problem (MAUP) Awareness

The MAUP is a fundamental problem in spatial statistics: the same underlying phenomenon can produce wildly different statistical results depending on how spatial units are defined (census tracts vs. blocks vs. grid cells). SPARC's spatially-continuous modeling approach — predicting at arbitrary point locations rather than aggregated units — makes it structurally resistant to MAUP artifacts. Causal estimates derived from SPARC are more stable across spatial scales than those derived from areal aggregation methods. No other causal inference tool in the spatial domain explicitly addresses this.

### 6. Bayesian MGWR Ensemble

SPARC's Bayesian Multiscale Geographically Weighted Regression (MGWR) ensemble operates at multiple bandwidth scales simultaneously, capturing both local and regional spatial patterns in the same model. The ensemble is calibrated with Bayesian posterior weights rather than fixed coefficients, so broader spatial patterns do not drown out local variation in areas with strong local dependence.

### 7. Integrated Scenario Simulation with Physics Constraints

Counterfactual scenario analysis is not bolted on after prediction — it is integrated into the physics model. Interventions are applied at the feature level and propagated through the same PDE physics that governed the prediction. This means a scenario that plants trees in a neighborhood doesn't just shift a coefficient — it changes the land cover fraction, which changes the thermal diffusivity field $\alpha(\mathbf{x})$, which changes the steady-state temperature distribution in a physically consistent way.

### 8. Native Desktop Application — Zero Cloud, Zero Data Egress

SPARC ships as a native desktop application (Tauri v2 + React) running the full pipeline locally. No cloud service. No API calls for the core pipeline. Data never leaves the user's machine. For domains where data sensitivity is critical — healthcare, criminal justice, defense — this is a hard requirement that rules out SaaS alternatives.

---

## Competitive Landscape

| Tool | Spatial | Causal | Physics | Transfer | Uncertainty | Scenarios |
|---|---|---|---|---|---|---|
| **SPARC** | ✅ Full | ✅ Full | ✅ PDE-constrained | ✅ SharedTrunk | ✅ Full posterior | ✅ Physics-consistent |
| DoWhy | ❌ | ✅ Causal graphs | ❌ | ❌ | Partial | ❌ |
| EconML | ❌ | ✅ CATE/DML | ❌ | ❌ | Partial | ❌ |
| GeoDa | ✅ Spatial stats | ❌ | ❌ | ❌ | ❌ | ❌ |
| PySAL | ✅ Spatial stats | ❌ | ❌ | ❌ | Partial | ❌ |
| GWR4 | ✅ GWR only | ❌ | ❌ | ❌ | ❌ | ❌ |
| MODIS/LST | ✅ Remote sensing | ❌ | ❌ | ❌ | ❌ | ❌ |
| UrbanSim | ✅ Urban | Partial | Partial | ❌ | ❌ | ✅ Land use |

SPARC is the only tool in this space that is simultaneously spatial, causal, physics-constrained, transferable, and uncertainty-quantified.

---

## Academic Foundation

SPARC's methodology draws from and extends peer-reviewed work across spatial statistics, physics-informed machine learning, and causal inference:

- **Gaussian Process / Matérn kernels:** Stein (1999), Rasmussen & Williams (2006)
- **Geographically Weighted Regression:** Fotheringham, Brunsdon & Charlton (2002)
- **SIREN networks:** Sitzmann et al. (2020)
- **Physics-Informed Neural Networks:** Raissi, Perdikaris & Karniadakis (2019)
- **Elastic Weight Consolidation:** Kirkpatrick et al. (2017)
- **Double Machine Learning:** Chernozhukov et al. (2018)
- **MC³ structure search:** Madigan & York (1995)
- **NUTS sampler:** Hoffman & Gelman (2014)
- **E-values:** VanderWeele & Ding (2017)
- **JEPA / VICReg:** LeCun (2022), Bardes, Ponce & LeCun (2022)
- **V-JEPA 2:** Assran et al. (2025)

SPARC has been applied in published urban heat island research (DOI: 10.1016/j.uclim.2025.102671) and the ForceSMIP climate forcing attribution benchmark.
