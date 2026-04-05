# SPARC v2.1 User Manual

**Spatial Analysis and Research Core — SPARC Labs LLC**

---

## Table of Contents

1.  [Introduction](#1-introduction)
2.  [Installation](#2-installation)
3.  [Quick Start](#3-quick-start)
4.  [Project Configuration (`project.yml`)](#4-project-configuration)
    - [project](#41-project)
    - [data](#42-data)
    - [crs](#43-crs)
    - [predictors](#44-predictors)
    - [physics](#45-physics)
    - [causal](#46-causal)
    - [output](#47-output)
    - [pipeline](#48-pipeline)
    - [flags](#49-flags)
    - [models](#410-models)
    - [gwen](#411-gwen)
    - [laplacian](#412-laplacian)
    - [scenarios](#413-scenarios)
    - [joint_scenarios](#414-joint_scenarios)
    - [temporal](#415-temporal)
    - [benchmark_metrics](#416-benchmark_metrics)
    - [fingerprint](#417-fingerprint)
5.  [Domain Templates](#5-domain-templates)
6.  [Physics Configuration](#6-physics-configuration)
7.  [Causal DAG Configuration](#7-causal-dag-configuration)
8.  [Pipeline Stages](#8-pipeline-stages)
9.  [CLI Reference](#9-cli-reference)
10. [Streamlit UI](#10-streamlit-ui)
11. [Models Reference](#11-models-reference)
12. [Advanced Topics](#12-advanced-topics)
13. [FAQ](#13-faq)

---

## 1. Introduction

SPARC is a physics-constrained spatial machine-learning pipeline that:

- **Trains** an ensemble of geographically-weighted models (OLS, GWR, GWRF, GGPGAM) plus a stacking meta-ensemble.
- **Validates** causal relationships via directed acyclic graphs (DAGs) with DoWhy refutation tests.
- **Simulates** "what-if" intervention scenarios with physics guardrails and uncertainty quantification.

It is domain-agnostic: urban heat islands, climate forcing attribution, groundwater, air quality, and more — all driven by a single `project.yml` configuration file.

---

## 2. Installation

**Requirements:** Python 3.10+

```powershell
cd GW3C_v2.0/GW3C_v2.1
pip install -e .
```

Verify:

```powershell
sparc --help
```

### Core dependencies

| Category | Packages |
|----------|----------|
| Scientific | numpy, pandas, scipy, scikit-learn |
| Spatial | geopandas, libpysal, esda, mgwr, pyproj |
| Modeling | lightgbm, optuna, pygam, torch |
| Causal | dowhy, networkx, econml |
| Utilities | matplotlib, seaborn, joblib, pyyaml, jsonschema |

### Optional dependencies

| Package | Required For |
|---------|-------------|
| `econml` | CATE estimation in Stage 3 |
| `tensorflow` / `keras` | Deep Kriging V2 residual correction |
| `streamlit` | Interactive UI |

---

## 3. Quick Start

```powershell
# 1. Scaffold from a template
sparc init --template uhi --output ./my_project

# 2. Edit project.yml (data path, CRS, predictors)
notepad ./my_project/project.yml

# 3. Validate
sparc validate --project ./my_project/project.yml

# 4. Run the full pipeline
sparc run --project ./my_project/project.yml --stage all
```

Or run stage-by-stage:

```powershell
sparc run -p project.yml -s 0      # Correlogram analysis
sparc run -p project.yml -s 1      # GWEN variable selection
sparc run -p project.yml -s 2      # Model training
sparc run -p project.yml -s 3      # Causal validation
sparc run -p project.yml -s 4      # Scenario simulation
```

---

## 4. Project Configuration

Every SPARC project is driven by a single `project.yml` file. Required sections are marked with ★.

### 4.1 project ★

Project-level metadata.

```yaml
project:
  name: "My Analysis"                  # Required
  description: "..."                    # Optional
  domain: uhi                           # Optional — matches template name
  version: "2.1.0"                      # Optional
  author: "Your Name"                   # Optional
  response_units: "°F"                  # Optional — used in reports
```

### 4.2 data ★

Input data specification.

```yaml
data:
  file_path: "data/my_data.csv"         # Required — relative to project.yml or absolute
  target_column: "AAT_z"                # Required — response variable column name
  identifier_column: "OBJECTID"         # Optional — unique row ID
  coord_columns: ["POINT_X", "POINT_Y"] # Required — [Easting, Northing] column names
  areas_of_interest_file: "data/AOI.csv" # Optional — zone/district mapping
```

**Path resolution:** All file paths are resolved relative to the `project.yml` file location unless given as absolute paths.

### 4.3 crs ★

Coordinate reference systems.

```yaml
crs:
  input: "EPSG:3438"                    # Required — CRS of the raw coordinates
  projected: "EPSG:26919"               # Required — Projected CRS for spatial analysis (must be in meters)
```

The `input` CRS describes your raw data coordinates. The `projected` CRS is used for all spatial operations (distance calculations, bandwidth selection, block sizes). Use an equal-area or UTM projection in meters.

Verify your codes at [epsg.io](https://epsg.io).

### 4.4 predictors ★

List of feature column names to include.

```yaml
predictors:
  - Pct_Canopy
  - Pct_Impervious
  - NDVI
  - Albedo
  - Elevation_m
  - Distance_from_water_m
```

At least one predictor is required. GWEN (Stage 1) can further refine this list if enabled.

### 4.5 physics

Physics-based domain constraints. Optional but recommended for scenario accuracy.

```yaml
physics:
  priors_file: "physics/priors.yml"     # Literature-derived coefficients
  caps_file: "physics/caps.yml"         # Hard bounds, delta caps, diminishing returns
  monotone_constraints:                 # Per-predictor sign constraints
    Pct_Canopy: -1                      # -1 = negative relationship (more canopy → cooler)
    Pct_Impervious: 1                   # +1 = positive relationship (more impervious → warmer)
    NDVI: -1
    Albedo: -1
    Elevation_m: 0                      #  0 = unconstrained
    Distance_from_water_m: 0
```

See [Section 6: Physics Configuration](#6-physics-configuration) for `priors.yml` and `caps.yml` format.

### 4.6 causal

Causal inference configuration. Optional — without it, Stage 4 uses physics priors only.

```yaml
causal:
  dag_file: "causal/dag.yml"            # Path to DAG definition
  estimator: "hgb"                      # Estimation method (see table below)

  # Shrinkage & blending
  dml_shrinkage: 0.25                   # DML shrinkage toward physics priors (0–1)
  dag_blend_weight: 0.7                 # Weight for structural coefficients in Stage 4

  # DML-specific
  dml_cv_folds: 5                       # Cross-fitting folds
  spatial_block_size: 300               # Spatial blocking for DML folds (meters)

  # CATE (heterogeneous treatment effects)
  estimate_cate: true                   # Spatial CATE via CausalForestDML (default: true)

  # Multi-estimator cross-checks
  ipw_enabled: false                    # Inverse propensity weighting
  gps_enabled: false                    # Generalised propensity score
  matching_enabled: false               # Matching estimator
  doubly_robust: false                  # LinearDML doubly-robust
  dose_response: false                  # Dose-response curves
  bootstrap_n: 100                      # Bootstrap samples for CIs

  # Variable classification
  actionable_variables:                 # Variables that can be intervened on
    - Pct_Canopy
    - Pct_Impervious
    - Albedo
    - NDVI
  fixed_variables:                      # Confounders (not intervened)
    - Elevation_m
    - Distance_from_water_m

  # Causal discovery (structure learning)
  discovery:
    enabled: true
    methods: [pc_stable, lingam, ges]   # Algorithms: pc_stable, lingam, ges, pcmci
    alpha: 0.05                         # Significance level
    compare_expert: true                # Compare learned DAG to your expert DAG
    consensus_threshold: 0.5            # Fraction of algorithms agreeing on edge
```

#### Estimator options

| Key | Name | Description |
|-----|------|-------------|
| `ols` | OLS | Linear regression with backdoor adjustment |
| `hgb` | HistGradientBoosting | Non-linear, monotone-constrained — **default** |
| `dml` | Debiased ML | Orthogonal ML with K-fold cross-fitting (most robust) |

### 4.7 output

Output directory structure.

```yaml
output:
  base_dir: "output"                    # Base output directory (default: "output")
  stage_dirs:
    stage_0: "Stage_0_Correlogram"
    stage_1: "Stage_1_GWEN"
    stage_2: "Stage_2_Spatial_CV"
    stage_3: "Stage_3_Causal_Validation"
    stage_4: "Stage_4_Scenarios"
    final:   "Final_Interpretation_Results"
```

### 4.8 pipeline

Global pipeline parameters.

```yaml
pipeline:
  random_seed: 42                       # Reproducibility seed
  n_spatial_folds: 5                    # Number of spatial CV folds
  fast_mode: false                      # Reduced-precision mode (fewer trials, smaller samples)
  overwrite_outputs: false              # Overwrite existing stage outputs
  run_mc_uncertainty: true              # Monte Carlo uncertainty in Stage 4
  n_mc_draws: 50                        # Number of MC simulations
```

### 4.9 flags

Feature flags to enable/disable pipeline components.

```yaml
flags:
  use_gwen_selection: true              # Run GWEN variable selection (Stage 1)
  include_gwrf_in_ensemble: true        # Include GWRF in model training
  use_laplacian_eigenmaps_in_ols: false # Add Laplacian eigenmaps to OLS features
```

### 4.10 models

Per-model hyperparameters. All are optional — sensible defaults are used when omitted.

```yaml
models:
  gwr:
    bandwidth: null                     # null = auto from correlogram
    kernel: "gaussian"                  # gaussian | exponential | bisquare
    alpha: 0.1                          # L2 regularisation
    min_points: 50                      # Minimum points for local regression

  gwrf:
    n_estimators: 100                   # Number of trees
    k_neighbors: 100                    # Adaptive bandwidth
    min_samples_leaf: 5
    n_jobs: 1
    subsample_fraction: null            # Fraction of data per local model
    subsample_n: null                   # Absolute number per local model

  ggpgam:
    n_splines: 5                        # Number of spline bases per feature
    n_spatial_bases: 10                 # Spatial tensor product bases
    lam: 0.6                            # Smoothing penalty
    max_iter: 100

  meta_ensemble:
    algorithm: "lightgbm"               # lightgbm | xgboost | catboost | linear
    n_optuna_trials: 25                 # Hyperparameter tuning trials
    include_base_features: true         # Include original features in stacking
    include_laplacian_pca: true         # Include Laplacian PCA in stacking

  deep_kriging:
    enabled: true                       # Enable neural residual correction
    version: 2                          # V2 architecture (multi-scale)
    hidden_layers: [64, 32, 16]
    dropout_rate: 0.2
    epochs: 100
    learning_rate: 0.001
    batch_size: 256

  spatial_cv:
    block_size: 300                     # Block size in meters
    buffer_size: 300                    # Buffer zone between folds (meters)
    block_size_source: "user"           # "user" | "correlogram"
    method: "block"                     # "block" | "kmeans"
    stratify_y: true                    # Stratify folds by response distribution
```

### 4.11 gwen

GWEN variable selection parameters.

```yaml
gwen:
  sample_size: 5000                     # Locations to fit local models
  k_neighbors: 500                      # Adaptive bandwidth (k-nearest)
  cv_folds: 5                           # ElasticNet CV folds
  selection_threshold: 0.1              # Min selection frequency (0–1)
  l1_ratios: [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
  n_alphas: 100
  quick_mode: false                     # Reduce sample_size for speed
  stability_folds: null                 # Stability assessment folds
  auto_tune: false                      # Auto-tune gwen parameters
  report_format: null                   # Report output format
```

### 4.12 laplacian

Laplacian eigenmap parameters.

```yaml
laplacian:
  n_eigenmaps: 150                      # Number of eigenvectors to extract
  k_for_swm: 10                         # k-nearest for spatial weights matrix
```

Laplacian eigenmaps capture smooth spatial variation and are used as features by OLS and as PCA components by the meta-ensemble.

### 4.13 scenarios

Single-variable intervention scenarios for Stage 4.

```yaml
scenarios:
  - name: "Canopy Increase"
    variable: Pct_Canopy                # Column name to modify
    direction: increase                 # increase | decrease
    increments: [5, 10, 15, 20, 30, 50] # List of delta values to test
    min_val: 0                          # Hard minimum bound
    max_val: 100                        # Hard maximum bound
    unit: "percentage points"           # Display unit

  - name: "Albedo Increase"
    variable: Albedo
    direction: increase
    increments: [0.05, 0.10, 0.20, 0.30]
    min_val: 0.05
    max_val: 0.70
    unit: "reflectance"
```

Each scenario tests one predictor at multiple increment levels. Physics constraints (diminishing returns, caps, monotone signs) are applied automatically.

### 4.14 joint_scenarios

Multi-variable intervention scenarios.

```yaml
joint_scenarios:
  - name: "Green Infrastructure Package"
    auto_propagate_dag: true            # Auto-propagate through DAG mediators
    interventions:
      - variable: Pct_Canopy
        direction: increase
        increment: 15
      - variable: Pct_Impervious
        direction: decrease
        increment: 15
      - variable: Albedo
        direction: increase
        increment: 0.10

  - name: "Cool-Roof Only"
    auto_propagate_dag: false           # Direct effect only
    interventions:
      - variable: Albedo
        direction: increase
        increment: 0.20
```

When `auto_propagate_dag: true`, the DAG's cross-predictor edges automatically propagate induced changes to downstream variables (e.g., +15 pp Canopy → computed NDVI change via the Canopy→NDVI edge). Combined constraints (e.g., Canopy + Impervious ≤ 100%) are enforced.

### 4.15 temporal

Temporal/spatiotemporal configuration. Optional — for panel or time-series data.

```yaml
temporal:
  enabled: false
  time_column: "date"
  frequency: "monthly"                  # daily | weekly | monthly | annual
  training_window: null
  validation_window: null
  forecast_horizon: null
  lag_features: []                      # e.g. ["precipitation_mm"]
  temporal_cv: false
  aggregation: null
```

When `temporal.enabled: true`, the pipeline adds lag features and supports temporal cross-validation.

### 4.16 benchmark_metrics

Climate benchmarking configuration (primarily for ForceSMIP-type analyses).

```yaml
benchmark_metrics:
  enabled: false
  area_weighted: true                   # Weight by grid cell area
  lat_column: "lat"                     # Latitude column for area weighting
```

### 4.17 fingerprint

Fingerprinting configuration for climate forcing attribution.

```yaml
fingerprint:
  enabled: false
  pattern_source: "ensemble_mean"       # ensemble_mean | eof | both
  eof_n_modes: 3                        # Number of EOF modes
  blend_alpha: 0.5                      # Blend weight
  truth_column: null                    # Known forced-response column
```

---

## 5. Domain Templates

SPARC ships with 13 pre-configured domain templates. Each contains a `project.yml`, physics priors, constraint caps, and a causal DAG.

| Template | Domain | Use Case |
|----------|--------|----------|
| `uhi` | Urban Heat Island | Ambient air temperature vs. land-cover |
| `forcesmip` | Climate Forcing | GCM forced-response attribution |
| `groundwater` | Hydrogeology | Groundwater level / contaminant modeling |
| `air_quality` | Air Quality | Pollutant concentration spatial analysis |
| `water_quality` | Water Quality | Surface water quality indicators |
| `stormwater` | Stormwater | Runoff and stormwater management |
| `coastal` | Coastal Engineering | Coastal erosion, sea-level response |
| `geotechnical` | Geotechnical | Soil properties and ground deformation |
| `seismic` | Seismic Hazard | Ground motion and seismic response |
| `noise` | Noise / Acoustics | Environmental noise modeling |
| `drought` | Drought | Drought index and vegetation stress |
| `wildfire` | Wildfire | Fire risk and burn severity |
| `blank` | Custom | Empty skeleton — bring your own config |

### Using a template

```powershell
sparc init --template uhi --output ./my_uhi_project
```

This copies the template's `project.yml`, physics files, and DAG into your project directory. Edit the data path, CRS, and predictors for your study area, then run:

```powershell
sparc validate --project ./my_uhi_project/project.yml
sparc run --project ./my_uhi_project/project.yml --stage all
```

### Creating a custom template

1. Start from the `blank` template: `sparc init --template blank --output ./custom`
2. Edit `project.yml` with your domain's configuration.
3. Add `physics/priors.yml` with literature-derived coefficients.
4. Add `physics/caps.yml` with variable bounds and constraints.
5. Add `causal/dag.yml` with your domain's causal structure.
6. Place the directory under `templates/my_domain/` to make it available as `sparc init --template my_domain`.

---

## 6. Physics Configuration

### 6.1 Priors file (`priors.yml`)

Defines literature-derived coefficients for each predictor. Used for shrinkage in Stage 3 and as fallback in Stage 4.

```yaml
Pct_Canopy:
  coefficient: -0.280                   # Effect per unit change
  units: "°F per +10 pp"               # Human-readable units
  uncertainty: 0.20                     # Fractional uncertainty (±20%)
  source_scale: "100m-500m"            # Resolution of literature source
  confidence: high                      # high | medium | low
  literature_source: "Ziter et al. (2019) PNAS"

Pct_Impervious:
  coefficient: 0.330
  units: "°F per +10 pp"
  uncertainty: 0.20
  source_scale: "100m-1km"
  confidence: high
  literature_source: "Li et al. (2020)"

Albedo:
  coefficient: -0.230
  units: "°F per +0.1"
  uncertainty: 0.20
  source_scale: "500m-2km"
  confidence: medium
  literature_source: "Santamouris et al. (2011)"
```

**Confidence levels affect adaptive shrinkage:**

| Confidence | Shrinkage Behavior |
|-----------|-------------------|
| `high` | Strong pull toward prior — data coefficient must be very different to override |
| `medium` | Balanced — prior and data contribute roughly equally |
| `low` | Weak pull — data coefficient dominates |

**Scale harmonisation:** When `source_scale` differs from your data resolution, SPARC's `ScaleHarmonizer` applies a power-law correction factor (finer resolution data shows weaker per-unit effects due to spatial averaging).

### 6.2 Caps file (`caps.yml`)

Defines hard bounds, delta limits, and diminishing-return thresholds.

```yaml
Pct_Canopy:
  min: 0                                # Absolute minimum value
  max: 100                              # Absolute maximum value
  delta_cap: 50                         # Maximum allowed delta per scenario
  diminishing_threshold: 15             # Beyond this, apply √ taper
  combined_constraints:
    - partner: Pct_Impervious
      max_sum: 100                      # Canopy + Impervious ≤ 100%

Albedo:
  min: 0.05
  max: 0.70
  delta_cap: 0.40
  diminishing_threshold: 0.15

Pct_Impervious:
  min: 0
  max: 100
  delta_cap: 50
  diminishing_threshold: 15
```

**Diminishing returns:** For deltas beyond `diminishing_threshold`, the effective delta is scaled by √(delta / threshold). This models the physical reality that marginal returns decrease (e.g., the 50th tree per hectare cools less than the 10th).

**Combined constraints:** Cross-variable bounds ensure physical consistency. For example, `Pct_Canopy + Pct_Impervious ≤ 100%` because they are fractions of the same land area.

---

## 7. Causal DAG Configuration

### 7.1 DAG file format (`dag.yml`)

```yaml
nodes:
  - name: Pct_Canopy
    type: treatment
    description: "Tree canopy coverage (%)"
  - name: Pct_Impervious
    type: treatment
    description: "Impervious surface coverage (%)"
  - name: NDVI
    type: mediator
    description: "Normalised difference vegetation index"
  - name: Elevation_m
    type: confounder
    description: "Elevation (m)"
  - name: Distance_from_water_m
    type: confounder
    description: "Distance to nearest water body (m)"
  - name: Albedo
    type: treatment
    description: "Surface albedo (reflectance)"
  - name: AAT_z
    type: outcome
    description: "Ambient air temperature z-score"

edges:
  - parent: Pct_Canopy
    child: AAT_z
    mechanism: "Evapotranspiration cooling"
  - parent: Pct_Canopy
    child: NDVI
    mechanism: "Trees increase greenness"
  - parent: Pct_Canopy
    child: Pct_Impervious
    mechanism: "Land-cover trade-off"
  - parent: Pct_Impervious
    child: AAT_z
    mechanism: "Heat absorption + reduced evapotranspiration"
  - parent: NDVI
    child: AAT_z
    mechanism: "Vegetation cooling via transpiration"
  - parent: Albedo
    child: AAT_z
    mechanism: "Surface reflectance reduces absorbed radiation"
  - parent: Elevation_m
    child: AAT_z
    mechanism: "Environmental lapse rate"
  - parent: Distance_from_water_m
    child: AAT_z
    mechanism: "Proximity cooling from water bodies"
```

### 7.2 Node types

| Type | Role | Stage 4 Behavior |
|------|------|-----------------|
| `treatment` | Variables you intervene on | Direct + indirect effects computed |
| `outcome` | Response variable | Predicted change reported |
| `mediator` | Intermediate pathway variable | Indirect effects propagated via mediator edges |
| `confounder` | Common cause | Held fixed during intervention |

### 7.3 Temporal edges

For temporal/panel data, add time-lagged edges:

```yaml
temporal_edges:
  - source: precipitation_mm
    target: gw_level_m
    lag: 3                              # 3 time-step lag
```

This internally creates a `precipitation_mm_t-3` node to maintain DAG acyclicity.

---

## 8. Pipeline Stages

See the [Pipeline Guide](PIPELINE_GUIDE.md) for a detailed walkthrough of each stage with configuration examples, outputs, and tips.

| Stage | Name | Module | Required |
|-------|------|--------|----------|
| 0 | Correlogram Analysis | `sparc.run.correlogram_analysis` | Yes |
| 0b | Pipeline Configuration | `sparc.run.pipeline_configurator` | Auto (runs with 0) |
| 1 | GWEN Variable Selection | `sparc.run.gwen_variable_selection` | Optional |
| 2 | Spatial Cross-Validation | `sparc.run.enhanced_spatial_cv` | Yes |
| 3 | Causal Validation | `sparc.run.causal_validation` | Optional |
| 4 | Scenario Simulation | `sparc.interventions.scenario_simulator` | Optional |

---

## 9. CLI Reference

### `sparc init`

Scaffold a new project from a template.

```
sparc init --template <name> --output <dir>
```

| Argument | Description |
|----------|-------------|
| `--template`, `-t` | Template name (default: `blank`). See `templates/` directory. |
| `--output`, `-o` | Output directory for the new project (required). |

### `sparc validate`

Validate a project configuration.

```
sparc validate --project <path>
```

Checks: data file exists, required columns present, CRS codes valid, physics/DAG files exist.

### `sparc run`

Run the pipeline (all stages or a specific one).

```
sparc run --project <path> [--stage 0|1|2|3|4|all] [--fast] [--skip-gwen] [--resume]
```

| Argument | Description |
|----------|-------------|
| `--project`, `-p` | Path to `project.yml` (required). |
| `--stage`, `-s` | Stage to run: `0` (Correlogram), `1` (GWEN), `2` (Spatial CV), `3` (Causal), `4` (Scenarios), `all` (default). |
| `--fast` | Reduced-precision mode for faster exploration. |
| `--skip-gwen` | Skip GWEN variable selection (Stage 1). |
| `--resume` | Resume from last completed stage (skip completed markers). |

### `sparc scenario`

Run scenario simulation independently.

```
sparc scenario --project <path> [--scenario <name>]
```

| Argument | Description |
|----------|-------------|
| `--project`, `-p` | Path to `project.yml` (required). |
| `--scenario`, `-n` | Name of a specific scenario to run (default: all). |

### `sparc report`

Generate a final interpretation report.

```
sparc report --project <path>
```

---

## 10. Streamlit UI

SPARC includes an interactive Streamlit UI for guided project configuration and execution.

### Launch

```powershell
# Windows
Start_SPARC.bat

# Any platform
streamlit run sparc/ui/app.py
```

### Pages

| Page | Purpose |
|------|---------|
| 1 — Project Setup | Project metadata, template selection, working directory |
| 2 — Data | Data file path, target column, identifier, coordinates |
| 3 — Variables | Predictor selection, AOI file |
| 4 — CRS | Input and projected coordinate reference systems |
| 5 — DAG Builder | Visual causal DAG construction |
| 6 — Physics | Physics priors, caps, monotone constraints |
| 7 — Scenarios | Single-variable and joint scenario definitions |
| 8 — Models | Model hyperparameters and flags |
| 9 — Run Pipeline | Generate config files, validate, execute pipeline |
| 10 — Results | View outputs and diagnostics |

The sidebar shows completion status for key configuration steps. All settings are stored in session state and exported to `project.yml` via the YAML generator.

---

## 11. Models Reference

### 11.1 OLS (`sparc.models.ols`)

Global linear regression baseline. Optionally includes Laplacian eigenmaps as spatial features. Uses scikit-learn + statsmodels. Fits on standardised features.

### 11.2 GWR (`sparc.models.gwr`)

Geographically Weighted Regression with adaptive bandwidth and configurable kernel (gaussian, exponential, bisquare). Produces spatially-varying local coefficients (`coefficients_`). L2 regularisation (`reg_lambda`) for numerical stability. Physics sign constraints can be applied to local coefficients.

### 11.3 GWRF (`sparc.models.gwrf`)

Geographically Weighted Random Forest. Fits a local random forest at each prediction location using k-nearest neighbors bandwidth. Supports sub-sampling for large datasets. Exports partial dependence (PDP) condition curves used by Stage 4's saturation-aware scenario simulation.

### 11.4 GGPGAM (`sparc.models.ggpgam`)

Geographically Guided Penalised GAM with Spatially Varying Coefficients. Uses `pygam.LinearGAM` with penalised splines for feature effects and spatial tensor product terms for local variation.

### 11.5 Meta-Ensemble (`sparc.models.meta_ensemble`)

LightGBM-based stacking ensemble. Inputs: out-of-fold predictions from all base models + original features + Laplacian PCA. Hyperparameters tuned via Optuna. Monotone constraints from physics config are enforced. SHAP explainability available.

### 11.6 Deep Kriging V2 (`sparc.models.deep_kriging_v2`)

Multi-scale spatial residual correction network (TensorFlow/Keras). Three branches:
1. **Wendland RBF coordinate encoder** — learnable radial basis functions for spatial position
2. **Laplacian spectral encoder** — eigenmap embeddings for smooth spatial structure
3. **Local neighbourhood context** — aggregated features from spatial neighbours

Applied when residual spatial autocorrelation is detected in the meta-ensemble predictions. The output is the meta-ensemble prediction + neural residual correction.

### 11.7 GWEN (`sparc.models.gwen`)

Geographically Weighted Elastic Net for variable selection. Fits ElasticNetCV at sampled locations with adaptive spatial weighting. Reports selection frequency, coefficient stability, and spatial variability per feature.

### 11.8 Fingerprint Projector (`sparc.models.fingerprint`)

Projects SPARC predictions onto known forced spatial patterns (ensemble mean or EOF). For climate signal detection (ForceSMIP-type analyses). Blends raw predictions with projected field using configurable alpha.

---

## 12. Advanced Topics

### 12.1 Scale harmonisation

Physics priors from literature are typically derived at coarser resolutions (100m–1km) than your data (e.g., 30m). SPARC's `ScaleHarmonizer` applies a power-law correction to translate coefficients:

$$\beta_{target} = \beta_{source} \times \left(\frac{res_{target}}{res_{source}}\right)^{\gamma}$$

where $\gamma$ depends on the spatial structure of the variable. This is applied automatically in Stage 3.

### 12.2 Adaptive shrinkage

When estimating structural coefficients (Stage 3), data-driven estimates are shrunk toward physics priors:

$$\hat{\beta} = w \cdot \beta_{prior} + (1 - w) \cdot \beta_{data}$$

The shrinkage weight $w$ adapts based on:
- **Prior confidence** (`high` → stronger shrinkage)
- **Data sample size** (more data → less shrinkage)
- **Estimator type** (DML uses lower default shrinkage = 0.25)

A max-deviation guard clamps the final coefficient within ±50% of the prior.

### 12.3 Four-tier delta computation

Stage 4 uses a hierarchy of methods to compute scenario deltas:

1. **MGWR local coefficients** — spatially varying β from GWR, optionally anchored to DML structural coefficients. Preferred when GWR produced reliable results (mean |β| ≥ threshold).
2. **Saturation curve** — non-linear PDP response from GWRF condition curves. Used when PDP has R² ≥ 0.5.
3. **MGWR blend** — weak MGWR signal blended with physics literature.
4. **Physics literature** — pure fallback using priors.yml coefficients.

The tier selection is automatic per variable, per spatial point.

### 12.4 Extrapolation guards

When a scenario pushes data points far from the training distribution, SPARC flags these using Mahalanobis distance. Points where the modified feature vector is more than 3 standard deviations from the training distribution centroid are flagged as extrapolation-risk.

### 12.5 Spatial multiplier

Scenario deltas are not spatially uniform. A spatial multiplier (0.5×–1.5×) is derived from:
- Baseline target variable z-scores (warmer areas respond more to cooling interventions)
- Inter-model prediction disagreement (areas with less model agreement get attenuated deltas)

This preserves physics sign consistency while capturing local heterogeneity.

### 12.6 CATE estimation

When `causal.estimate_cate: true`, EconML's `CausalForestDML` estimates spatially heterogeneous treatment effects. This produces a per-observation treatment effect estimate (rather than a single ATE), revealing where interventions are most / least effective.

---

## 13. FAQ

**Q: Do I need to provide physics priors?**
A: No. Physics priors are optional but recommended. Without them, Stage 3 uses purely data-driven coefficients and Stage 4 uses the meta-ensemble for scenario prediction.

**Q: What if Stage 3 (causal) fails?**
A: Stage 4 automatically falls back to physics-prior-only or ensemble-based prediction. You can also omit the `causal` section entirely.

**Q: How large can my dataset be?**
A: SPARC handles datasets from hundreds to 70,000+ observations. Large datasets are automatically sub-sampled for computationally intensive steps (GWEN, GWR). The `fast_mode: true` flag reduces sample sizes and tuning trials for exploratory runs.

**Q: Can I add my own model?**
A: Yes. Implement a model class in `sparc/models/` with `fit(X, y, coords)` and `predict(X, coords)` methods, then register it in the spatial CV pipeline.

**Q: What CRS should I use?**
A: Use a projected CRS in meters appropriate for your study area. UTM zones work well globally. Avoid geographic (lat/lon) CRS for the `projected` field — distance calculations require meters.

**Q: How do I interpret E-values?**
A: E-values represent the minimum confounding strength needed to explain away the observed effect. E-value > 2.0 is considered robust; < 1.5 suggests sensitivity to unobserved confounding.

**Q: Can I use SPARC for temporal / panel data?**
A: Yes, with the `temporal` configuration section. Set `temporal.enabled: true`, specify the time column and frequency, and optionally add lag features. Temporal cross-validation is supported.

**Q: What does `block_size_source: correlogram` do?**
A: It tells Stage 2 to use the block size automatically determined by Stage 0's correlogram analysis (the first zero-crossing of Moran's I for the target variable) instead of a user-specified value.
