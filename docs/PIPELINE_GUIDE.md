# SPARC Pipeline Guide

**Step-by-step walkthrough of the five-stage pipeline.**

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Stage 0 — Correlogram Analysis](#stage-0--correlogram-analysis)
- [Stage 1 — GWEN Variable Selection](#stage-1--gwen-variable-selection)
- [Stage 2 — Spatial Cross-Validation & Model Training](#stage-2--spatial-cross-validation--model-training)
- [Stage 3 — Causal Validation](#stage-3--causal-validation)
- [Stage 4 — Scenario Simulation](#stage-4--scenario-simulation)
- [Output Directory Structure](#output-directory-structure)
- [Running the Pipeline](#running-the-pipeline)
- [Troubleshooting](#troubleshooting)

---

## Overview

SPARC runs five stages in sequence. Each stage reads from the previous stage's output, producing a fully traceable chain from raw data to policy-relevant scenario maps.

> **Run registry (since v3):** every artifact written by every stage is recorded in
> `<output_dir>/artifacts_manifest.json` (mirrored to `artifacts.db`). Server endpoints,
> the desktop **Results** view, the **Decision Support** wizard, and the report generator
> all resolve files through this registry — no path is hard-coded outside `PipelinePaths`.
> A run-final pass merges every spatial layer into `<output_dir>/master.gpkg`.
> The desktop **Export PNG** buttons (top-right of every chart/map) snapshot the current
> view, persist it under `<output_dir>/exports/`, and register it as an `export::*` artifact.

```
Stage 0   Correlogram Analysis     →  bandwidths, block sizes, kernel selection
     │
Stage 0b  Pipeline Configuration   →  auto-wires correlogram results into config
     │
Stage 1   GWEN Variable Selection  →  ranked predictors, selection frequencies (optional)
     │
Stage 2   Spatial Cross-Validation →  4 base models + meta-ensemble + Deep Kriging
     │
Stage 3   Causal Validation        →  structural coefficients, refutation tests, CATE
     │
Stage 4   Scenario Simulation      →  physics-constrained "what-if" predictions
```

All stages are driven by a single `project.yml` configuration file. See the [MANUAL](MANUAL.md) for the full configuration reference.

---

## Prerequisites

1. **Python 3.10+** with SPARC installed:
   ```powershell
   pip install -e .
   ```
2. A valid `project.yml` (scaffold one from a template):
   ```powershell
   sparc init --template uhi --output ./my_project
   ```
3. Your CSV data file with the target variable, predictor columns, and coordinate columns.
4. (Optional) Physics configuration files: `priors.yml`, `caps.yml`
5. (Optional) Causal DAG file: `dag.yml`

### Validate before running

```powershell
sparc validate --project ./my_project/project.yml
```

This checks that the data file exists, required columns are present, CRS codes are valid, and referenced physics/DAG files exist.

---

## Stage 0 — Correlogram Analysis

**Module:** `sparc.run.correlogram_analysis`
**Purpose:** Quantify spatial autocorrelation and auto-detect optimal model parameters.

### What it does

1. Loads your data and reprojects to the target CRS.
2. Computes **Moran's I** at multiple distance lags (default 15) for every variable (target + all predictors).
3. Identifies the **first zero-crossing** distance — the range beyond which spatial autocorrelation is negligible.
4. Derives optimal parameters:

| Parameter | Derivation | Used By |
|-----------|-----------|---------|
| **GWR bandwidth** | First zero-crossing distance | GWR model |
| **GGPGAM bandwidth** | 0.8 × zero-crossing | GGPGAM model |
| **GWRF bandwidth** | 1.2 × zero-crossing | GWRF model |
| **CV block size** | Zero-crossing of target variable | Spatial fold generation |
| **Best kernel** | Decay pattern (exponential / gaussian / spherical) | GWR, GGPGAM |

5. For large datasets (>5,000 points), intelligent sub-sampling is performed automatically.
6. Generates a pipeline configuration file that downstream stages consume.

### Key configuration

```yaml
# project.yml — these are auto-detected but can be overridden
models:
  spatial_cv:
    block_size: 300           # Override auto-detected block size (meters)
    buffer_size: 300          # Buffer zone around test folds (meters)
    block_size_source: user   # "user" to use your value, "correlogram" for auto
```

### Outputs

| File | Location | Description |
|------|----------|-------------|
| `{variable}_correlogram.png` | `Stage_0_Correlogram/` | 2-panel plot: correlogram curve + block-size quartiles |
| `variogram_analysis_results.json` | `Stage_0_Correlogram/` | All bandwidth and block-size recommendations |
| `pipeline_config.json` | `Stage_0_Correlogram/` | Auto-wired pipeline configuration |
| `dataset_profile.json` | `Stage_0_Correlogram/` | Dataset size tier and parameter recommendations |

### CLI

```powershell
sparc run -p project.yml -s 0
```

### Tips

- If your data has < 100 observations, correlogram results may be unstable — consider setting bandwidths manually.
- The `max_distance` is auto-set to 30% of the bounding-box diagonal. For irregularly shaped study areas, you may want to adjust.
- Review the correlogram plots: a smooth monotonic decay → gaussian kernel; sharp initial decay → exponential kernel.

---

## Stage 1 — GWEN Variable Selection

**Module:** `sparc.run.gwen_variable_selection`
**Purpose:** Identify which predictors have spatially stable relationships with the target.

> **Optional.** Skip with `sparc run -s all --skip-gwen` or set `flags.use_gwen_selection: false` in `project.yml`.

### What it does

1. Fits a **Geographically Weighted Elastic Net** at sampled locations across the study area.
2. At each location, ElasticNetCV selects features and estimates local coefficients using adaptive spatial weighting.
3. Aggregates results to compute per-variable **selection frequency** (fraction of locations where each variable was selected).
4. Variables exceeding the threshold (default ≥ 10%) are flagged as selected.
5. Generates diagnostic plots and a human checkpoint.

### Key configuration

```yaml
# project.yml
flags:
  use_gwen_selection: true

gwen:
  sample_size: 5000           # Number of locations to fit local models
  k_neighbors: 500            # Adaptive bandwidth (k-nearest neighbors)
  cv_folds: 5                 # CV folds for ElasticNet
  selection_threshold: 0.1    # Min selection frequency to include variable
  l1_ratios: [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
  n_alphas: 100
  quick_mode: false           # Reduce sample_size for faster runs
```

### Stability metrics reported per variable

| Metric | Description |
|--------|-------------|
| `selection_frequency` | Fraction of local models selecting the variable |
| `mean_abs_coefficient` | Mean |β| when selected |
| `coefficient_sign_stability` | Fraction where sign matches dominant direction |
| `spatial_variability` | Standard deviation of |coefficient| across space |
| `coefficient_range` | Min–max of non-zero coefficients |

### Outputs

| File | Location | Description |
|------|----------|-------------|
| `gwen_results.json` | `{output}/` | Complete results (params, importance, summary) |
| `gwen_variable_importance.csv` | `{output}/` | Per-variable importance table |
| `selected_features.txt` | `{output}/` | Simple list of selected feature names |
| `gwen_diagnostics.png` | `{output}/` | 4-panel diagnostic plot |
| `gwen_diagnostics.json` | `{output}/` | Tuning parameters and stability metrics |

### Human checkpoint

When running `sparc run --stage all`, the pipeline pauses after GWEN to allow manual review. Create the approval file to continue:

```powershell
# Review gwen_variable_importance.csv, then approve:
echo "approved" > output/gwen_approved.txt

# Resume:
sparc run -p project.yml --stage all --resume
```

### CLI

```powershell
sparc run -p project.yml -s 1
```

---

## Stage 2 — Spatial Cross-Validation & Model Training

**Module:** `sparc.run.enhanced_spatial_cv`
**Purpose:** Train all base models and the meta-ensemble using spatially-buffered cross-validation.

### What it does

1. Generates **spatial block-CV folds** using the block size from Stage 0 (or user-specified). Methods: grid-based blocks or k-means spatial clustering.
2. Applies **buffer zones** around test blocks to prevent spatial leakage.
3. Trains **four base models** on each fold:

| Model | Class | Type | Key Feature |
|-------|-------|------|-------------|
| **OLS** | `OLSModel` | Global linear | Optional Laplacian eigenmaps as spatial features |
| **GWR** | `GWRModel` | Local linear | Spatially varying coefficients, adaptive bandwidth |
| **GWRF** | `GWRFModel` | Local non-linear | Geographically weighted random forest, PDP condition curves |
| **GGPGAM** | `GGPGAM_SVC` | Semi-parametric | Penalised splines + spatial tensor product |

4. Collects **out-of-fold (OOF) predictions** from all four models.
5. Trains a **LightGBM meta-ensemble** that stacks OOF predictions + original features + optional Laplacian PCA. Tuned via Optuna with monotone constraints from the physics config.
6. Optionally applies **Deep Kriging V2** residual correction — a multi-scale spatial neural network that corrects the meta-ensemble when residual spatial autocorrelation is detected.

### Laplacian eigenmaps

When enabled (`flags.use_laplacian_eigenmaps_in_ols: true`), the pipeline computes eigenmaps from a spatial weights matrix. These capture smooth spatial variation and are used as features by OLS and as PCA inputs by the meta-ensemble.

```yaml
laplacian:
  n_eigenmaps: 150            # Number of eigenvectors to extract
  k_for_swm: 50               # k-nearest neighbors for spatial weights
```

### Key configuration

```yaml
models:
  spatial_cv:
    block_size: 300
    buffer_size: 300
    block_size_source: correlogram    # or "user"
    method: block                     # "block" or "kmeans"
    stratify_y: true
  gwr:
    kernel: adaptive
    reg_lambda: 0.01
  gwrf:
    n_estimators: 200
    min_samples_leaf: 5
  ggpgam:
    n_splines: 25
    lam: 0.6
  meta_ensemble:
    algorithm: neural
    include_base_features: true
    include_laplacian_pca: true

pipeline:
  n_spatial_folds: 5
  random_seed: 42
```

### Parallelisation

Models run sequentially; folds within each model run in parallel via `ProcessPoolExecutor`. Hardware is auto-detected (CPU cores, RAM).

### Outputs

| File | Location | Description |
|------|----------|-------------|
| `optimized_oof_predictions.csv` | `Stage_2_Spatial_CV/` | OOF predictions per model |
| `optimized_meta_model.txt` | `Stage_2_Spatial_CV/` | Meta-ensemble artifact |
| `laplacian_features.pkl` | `Stage_2_Spatial_CV/` | Laplacian eigenmaps |
| `laplacian_eigenvalues.pkl` | `Stage_2_Spatial_CV/` | Eigenvalue spectrum |
| `spatial_weights.pkl` | `Stage_2_Spatial_CV/` | Spatial weights matrix |
| `feature_scaler.pkl` | `Stage_2_Spatial_CV/` | Pre-processing scaler (used by Stage 4) |
| `gwrf_condition_curves.json` | `spatial_intelligence/gwrf_pdp/` | GWRF PDP curves (used by Stage 4) |
| Model `.pkl` files | `Stage_2_Spatial_CV/` | Serialised model artifacts |

### CLI

```powershell
sparc run -p project.yml -s 2
```

### Tips

- Use `fast_mode: true` in `pipeline` for quicker exploratory runs (reduced Optuna trials, smaller sample sizes).
- If GWR fails on a fold due to singular matrices, the pipeline auto-adjusts the bandwidth upward and retries.
- Review OOF predictions to diagnose spatial bias — are residuals randomly distributed?

---

## Stage 3 — Causal Validation

**Module:** `sparc.run.causal_validation`
**Purpose:** Estimate structural causal coefficients and validate with refutation tests.

### What it does

1. **Loads the DAG** from `dag.yml` and classifies nodes into treatments, mediators, confounders, and outcomes.
2. **Causal discovery** (optional): Runs data-driven structure learning algorithms (PC-stable, LiNGAM, GES, PCMCI) and compares learned edges against your expert DAG. Reports edge F1 score.
3. **DAG diagnostics**: Checks positivity (overlap of treatment distributions) and SUTVA (spatial autocorrelation of treatment variables).
4. **Structural coefficient estimation**: For every edge in the DAG:
   - Computes the backdoor adjustment set (Pearl 2009)
   - Fits the chosen estimator (OLS, HGB, or DML)
   - **Shrinks toward physics priors** with adaptive shrinkage based on prior confidence
   - Applies scale harmonisation (literature resolution → your data resolution)
   - Enforces monotone constraints (sign consistency)
   - Clamps within ±50% of the prior (max-deviation guard)
5. **Refutation tests** (DoWhy): Placebo treatment, random common cause, subset validation, unobserved confounding (E-value).
6. **Treatment effect estimation**: ATE (backdoor), bootstrap CI, and optional CATE via EconML's CausalForestDML.

### DAG file format (`dag.yml`)

```yaml
nodes:
  - name: Pct_Canopy
    type: treatment
    description: "Tree canopy coverage percentage"
  - name: NDVI
    type: mediator
    description: "Normalised difference vegetation index"
  - name: Elevation_m
    type: confounder
    description: "Elevation above sea level"
  - name: AAT_z
    type: outcome
    description: "Ambient air temperature z-score"

edges:
  - parent: Pct_Canopy
    child: AAT_z
    mechanism: "Evapotranspiration cooling"
  - parent: Pct_Canopy
    child: NDVI
    mechanism: "Trees increase vegetation"
  - parent: Elevation_m
    child: AAT_z
    mechanism: "Lapse rate cooling"
```

**Node types:** `treatment` (variables you intervene on), `outcome` (target), `mediator` (intermediate pathway), `confounder` (common cause).

### Estimators

| Estimator | Key | Description |
|-----------|-----|-------------|
| **OLS** | `ols` | Linear regression with backdoor adjustment |
| **HGB** | `hgb` | HistGradientBoosting with monotone constraints — default |
| **DML** | `dml` | Debiased / Orthogonal ML with K-fold cross-fitting, optional spatial blocks |

### Key configuration

```yaml
causal:
  dag_file: causal/dag.yml
  estimator: hgb                     # ols | hgb | dml
  dml_shrinkage: 0.25                # Shrinkage toward physics priors (DML)
  dml_cv_folds: 5                    # Cross-fitting folds for DML
  spatial_block_size: 300            # Spatial blocking for DML
  dag_blend_weight: 0.7             # Blend weight for structural coefficients
  estimate_cate: true                # Heterogeneous treatment effects
  actionable_variables:              # Variables that can be intervened on
    - Pct_Canopy
    - Pct_Impervious
    - Albedo
    - NDVI
  fixed_variables:                   # Confounders (not intervened)
    - Elevation_m
    - Distance_from_water_m
  discovery:
    enabled: true
    methods: [pc_stable, lingam, ges]
    alpha: 0.05
    compare_expert: true
    consensus_threshold: 0.5
```

### Outputs

| File | Location | Description |
|------|----------|-------------|
| `scenario_coefficients.json` | `Stage_3_Causal_Validation/` | Structural coefficients consumed by Stage 4 |
| `causal_validation_summary.txt` | `Stage_3_Causal_Validation/` | Human-readable summary |
| `dag_discovery_report.json` | `Stage_3_Causal_Validation/` | Learned DAG comparison (if discovery enabled) |
| Per-treatment diagnostics | `Stage_3_Causal_Validation/` | Positivity, SUTVA checks |
| CATE results | `Stage_3_Causal_Validation/` | Spatial heterogeneous effects (if enabled) |

### CLI

```powershell
sparc run -p project.yml -s 3
```

### Tips

- If causal discovery disagrees with your DAG, review the F1 report — it may suggest missing or spurious edges.
- E-values > 2.0 indicate robust results. E-values < 1.5 suggest sensitivity to unobserved confounding.
- Stage 3 is optional: if no DAG file is provided, Stage 4 falls back to physics-prior-only coefficients.

---

## Stage 4 — Scenario Simulation

**Module:** `sparc.interventions.scenario_simulator`
**Purpose:** Predict outcomes under user-defined "what-if" interventions with physics constraints.

### What it does

1. Loads trained models from Stage 2 and structural coefficients from Stage 3.
2. For each defined scenario, applies the intervention delta to the baseline data.
3. Computes predicted change using a **four-tier delta method**:

| Tier | Method | When Used |
|------|--------|-----------|
| **1** | MGWR local coefficients | GWR produced reliable spatially-varying β — optionally anchored to DML structural coefficient |
| **2** | Saturation curve | GWRF PDP condition curve available with R² ≥ 0.5 — non-linear response |
| **3** | MGWR blend | MGWR has weak signal — blend with physics literature |
| **4** | Physics literature | Pure fallback when no reliable model signal |

4. Propagates **indirect effects** through DAG mediator pathways (e.g., Canopy → NDVI → Temperature).
5. Applies **physics constraints**:

| Constraint | Description |
|-----------|-------------|
| **Diminishing returns** | Square-root taper beyond per-variable threshold |
| **Hard caps** | Min/max bounds and delta limits from `caps.yml` |
| **Monotone constraints** | Sign enforcement (e.g., more canopy → never warmer) |
| **Combined constraints** | Cross-variable bounds (e.g., Canopy + Impervious ≤ 100%) |
| **Extrapolation guard** | Mahalanobis distance flags points far from training distribution |

6. Optionally runs **Monte Carlo uncertainty propagation** — perturbs coefficients, re-simulates, and reports credible intervals (5th / 50th / 95th percentile).

### Scenario types

**Single-variable scenarios:**
```yaml
scenarios:
  - name: canopy_increase
    variable: Pct_Canopy
    direction: increase
    increments: [5, 10, 15, 20, 30, 50]
    min_val: 0
    max_val: 100
    unit: "percentage points"
```

**Joint (multi-variable) scenarios:**
```yaml
joint_scenarios:
  - name: green_infrastructure_package
    auto_propagate_dag: true
    interventions:
      - variable: Pct_Canopy
        delta: 15
      - variable: Albedo
        delta: 0.05
      - variable: Pct_Impervious
        delta: -10
```

When `auto_propagate_dag: true`, the DAG is used to automatically propagate changes through mediators (e.g., +15 pp Canopy → +0.045 NDVI via the Canopy→NDVI edge).

### Physics files

**`priors.yml`** — Literature-derived coefficients:
```yaml
Pct_Canopy:
  coefficient: -0.280
  units: "°F per +10 pp"
  uncertainty: 0.20
  confidence: high
  source: "Ziter et al. (2019) PNAS"
```

**`caps.yml`** — Hard bounds and guardrails:
```yaml
Pct_Canopy:
  min: 0
  max: 100
  delta_cap: 50
  diminishing_threshold: 15
  combined_constraints:
    - partner: Pct_Impervious
      max_sum: 100
```

### Key configuration

```yaml
physics:
  priors_file: physics/priors.yml
  caps_file: physics/caps.yml
  monotone_constraints:
    Pct_Canopy: -1              # Negative relationship expected
    Pct_Impervious: 1           # Positive relationship expected
    Albedo: -1
    NDVI: -1

pipeline:
  run_mc_uncertainty: true       # Enable Monte Carlo
  n_mc_draws: 50                 # Number of MC simulations
```

### Outputs

| File | Location | Description |
|------|----------|-------------|
| `scenario_summary.csv` | `Stage_4_Scenarios/` | Summary table (scenario, increment, mean delta, std) |
| `scenario_results.gpkg` | `Stage_4_Scenarios/` | Full spatial results as GeoPackage |
| Per-scenario maps | `Stage_4_Scenarios/` | Spatial delta visualisations |

### CLI

```powershell
sparc run -p project.yml -s 4

# Or run specific scenarios:
sparc scenario -p project.yml --scenario canopy_increase
```

---

## Output Directory Structure

After a full pipeline run, the output directory looks like this:

```
{output_base}/
├── Stage_0_Correlogram/
│   ├── {variable}_correlogram.png          # Per-variable correlogram
│   ├── variogram_analysis_results.json     # Bandwidth & block-size recommendations
│   ├── pipeline_config.json                # Auto-wired configuration
│   └── dataset_profile.json                # Dataset tier assessment
│
├── Stage_1_GWEN/
│   ├── gwen_results.json                   # GWEN feature selection results
│   ├── selected_features.txt               # Selected predictors
│   ├── gwen_variable_importance.csv        # Per-variable importance
│   └── gwen_diagnostics.png                # GWEN diagnostic plot
│
├── Stage_2_Spatial_CV/
│   ├── optimized_oof_predictions.csv       # Out-of-fold predictions
│   ├── optimized_meta_model.txt            # Meta-ensemble model
│   ├── laplacian_features.pkl              # Laplacian eigenmaps
│   ├── feature_scaler.pkl                  # Feature scaler
│   └── *.pkl                               # Additional model artifacts
│
├── Stage_3_Causal_Validation/
│   ├── scenario_coefficients.json          # Structural coefficients
│   ├── causal_validation_summary.txt       # Summary report
│   └── dag_discovery_report.json           # DAG comparison (if enabled)
│
├── Stage_4_Scenarios/
│   ├── scenario_summary.csv                # Results table
│   └── scenario_results.gpkg              # Spatial results
│
├── Final_Interpretation_Results/
│   ├── final_ensemble_results.json         # Final model performance
│   ├── final_ensemble_predictions.csv      # Full predictions
│   └── Performance_Analysis/
│       └── model_performance_metrics.csv   # Per-model metrics
│
└── spatial_intelligence/
    └── gwrf_pdp/
        └── gwrf_condition_curves.json      # GWRF partial dependence curves
```

---

## Running the Pipeline

### Full run

```powershell
sparc run --project project.yml --stage all
```

### Stage-by-stage

```powershell
sparc run -p project.yml -s 0      # Correlogram analysis
sparc run -p project.yml -s 1      # GWEN variable selection
sparc run -p project.yml -s 2      # Model training
sparc run -p project.yml -s 3      # Causal validation
sparc run -p project.yml -s 4      # Scenario simulation
```

### Useful flags

| Flag | Description |
|------|-------------|
| `--fast` | Reduced-precision mode for faster exploration |
| `--skip-gwen` | Skip GWEN variable selection (Stage 1) |
| `--resume` | Resume from last completed stage |

### Desktop App

Launch the SPARC desktop application:

```powershell
python -m sparc desktop
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'sparc'` | Run `pip install -e .` from the repo root |
| `CRS validation failed` | Verify EPSG codes at [epsg.io](https://epsg.io) |
| GWR singular matrix error | Pipeline auto-adjusts bandwidth; if persistent, increase `gwr.reg_lambda` |
| GWEN pauses pipeline | Create the approval file: `echo "approved" > output/gwen_approved.txt`, then `--resume` |
| Stage 3 fails (no DoWhy) | Install: `pip install dowhy econml`. Stage 4 will fall back to physics-only. |
| Correlogram shows flat line | Data may lack spatial structure. Check coordinate columns and CRS projection. |
| Out of memory in Stage 2 | Use `fast_mode: true`, reduce `n_spatial_folds`, or reduce `gwen.sample_size` |
| Scenario deltas are zero | Check `monotone_constraints` signs and `caps.yml` bounds. Review `scenario_coefficients.json`. |
