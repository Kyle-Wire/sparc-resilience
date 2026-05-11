# SPARC Labs — Future Roadmap

**SPARC Labs LLC | May 2026**
**Engineering Vision: Self-Sufficient Spatial Reasoning Engine**

---

## North Star

> A practitioner types a city name, draws a boundary, or provides a single weather station reading. SPARC produces a probabilistic spatial field, a causal attribution map, and a counterfactual intervention analysis — with no local training data, no ground sensor network, and no domain-specific configuration required.

This is a zero-shot spatial reasoning engine. Everything below is the path to it.

---

## Phase Overview

| Phase | Name | Core Deliverable | Status |
|---|---|---|---|
| **Phase 1** | Finish V3 Wiring | EWC + replay + temporal + JEPA active in training loop | Pending |
| **Phase 2** | Central Registry | Cloud-hosted model registry with privacy-preserving sync | Pending |
| **Phase 3** | Satellite + Climate Ingestion | Any city on Earth → SPARC feature set from free satellite data | Pending |
| **Phase 4** | Zero-Shot Inference | Predict spatial fields for unseen cities, no ground data | Pending |
| **Phase 5** | Causal Inference Leadership | Best-in-field spatial causal engine | Ongoing |
| **Phase 6** | Emerging Field Integration | PCMCI+, causal representation learning, transportability | Future |

---

## Phase 1 — Finish V3 Wiring

**Goal:** Multi-city continual training works end-to-end. Run `sparc continual` and get a global trunk that measurably improves with each city. All V3 modules active.

### 1.1 — EWC + Replay in Epoch Loop
Wire `ewc_penalty()` and `compute_replay_loss()` into `train_neural_meta()` in `sparc/training/v2_neural_training.py`. The functions exist; the `loss +=` lines are missing. Refer to `SPARC_Integration_Status.md` Gap 1 for exact implementation spec.

### 1.2 — Temporal Features in Training Runner
Wire `compute_diurnal_features()` and `get_snapshot_time_indices()` into the data preparation path. Pass `time_idx` to `model.forward()` to activate the 3-way time embedding and transient PDE terms 9–10. Refer to Integration Status Gap 2.

### 1.3 — JEPA Pretraining Pass
Before supervised training, run a JEPA self-supervised pretraining phase using the EMA trunk as a momentum encoder and `jepa_loss()` with curriculum scheduling. This shapes the trunk's latent geometry before the physics supervision kicks in. Refer to Integration Status Gap 3.

### 1.4 — Post-Training Artifact Extraction
After training, `train_neural_meta()` must return `fisher_matrix` and `trunk_state_dict` so `train_continual()` can populate the registry with real artifacts. Refer to Integration Status Gap 4.

### 1.5 — Validation: Providence → Boston Transfer
Run the transfer validation with real city data. Success criteria: warm-start R² ≥ cold-start R², faster convergence, positive `r2_improvement` in `transfer_comparison.json`.

### 1.6 — Validation: Continual Learning (2 Cities)
Run `sparc continual --cities providence.yml,boston.yml`. Success criteria: Providence R² does not degrade after Boston training (EWC working); Boston R² ≥ cold-start (transfer helping).

---

## Phase 2 — Central Registry Infrastructure

**Goal:** A cloud-hosted registry that user runs automatically feed into, building a progressively better global trunk.

### 2.1 — Registry Sync Protocol
Implement `sparc/registry/sync.py` with `push_city()`, `pull_global_trunk()`, and `pull_registry_manifest()`. Only model artifacts leave the machine (trunk weights, Fisher matrix, 400-point standardized coreset, Welford state, aggregate metrics). Raw data never leaves.

### 2.2 — Central Registry Server
Deploy `sparc/registry/server.py` as a FastAPI service on SPARC Labs infrastructure. Endpoints: `POST /v1/cities/{name}/upload`, `GET /v1/trunk/latest`, `GET /v1/manifest`. Backed by S3/Azure Blob/GCS for artifact storage.

### 2.3 — Automated Post-Run Upload
After a successful `sparc run`, if `registry.remote_url` is configured, prompt the user to upload artifacts. This is the organic growth mechanism for the central registry.

### 2.4 — Global Trunk Recomputation
`sparc/registry/retrain.py`: nightly job that merges Fisher matrices from all registered cities, runs continual training through city coresets in sequence, validates against held-out coreset points from each city, publishes the new trunk if it outperforms the previous.

### 2.5 — CLI Integration
Add `sparc push`, `sparc pull`, and `sparc registry --list` subcommands.

**Privacy architecture:** Eight cities across four Köppen climate groups is the minimum viable cross-climate global trunk. The system is useful before that point; it becomes zero-shot capable at that threshold.

---

## Phase 3 — Satellite + Climate Ingestion

**Goal:** Any city on Earth → SPARC feature set from freely available satellite imagery, with no local data collection.

### 3.1 — Satellite Ingestion Pipeline
`sparc/data/satellite_ingest.py`: standardized ingestion from Sentinel-2 L2A (10m, every 5 days) and Landsat 8/9 Collection 2 (30m, every 16 days). Derive: NDVI, NDBI, MNDWI, BSI, albedo, percent canopy/impervious/water, morphological derivatives (∇²NDVI, |∇NDBI|), SRTM topography (elevation, slope, aspect).

### 3.2 — ERA5 Climate Forcing
`sparc/data/era5_fetch.py`: pull ERA5 hourly reanalysis from Copernicus CDS. Fields: 2m air temperature (fallback thermal anchor), surface solar radiation, 10m wind u/v, humidity. Regrid from 0.25° to target resolution via bilinear interpolation.

### 3.3 — Weather Station Anchor
`sparc/data/station_anchor.py`: resolve the thermal boundary condition from the nearest WMO GSOD station or fall back to ERA5. Each scenario has a quantified uncertainty: ±0.3°C (station inside domain) → ±2.1°C (ERA5 fallback). The station anchor becomes the Dirichlet boundary condition for the PDE.

### 3.4 — Climate Zone Encoder (Full)
Expand `sparc/models/climate_encoder.py` from 30-class Köppen one-hot encoding to a richer conditioning: mean annual temperature, mean annual precipitation, annual solar irradiance, humidity index. Output is an additive modulation on the trunk's output — teaching the model that the same vegetation fraction cools differently in a humid subtropical climate vs. a hot desert.

### 3.5 — Unified Satellite Feature Set
Expand `SatelliteFeatureSet` in `sparc/data/satellite_types.py` to include all fields needed for zero-shot: spectral indices, topography, ERA5 forcing scalars, thermal anchor, climate normals, and metadata.

---

## Phase 4 — Zero-Shot Inference

**Goal:** `sparc predict --city "Lagos, Nigeria" --date 2026-07-15` produces a probabilistic spatial temperature field. No local training data required.

### 4.1 — Zero-Shot Inference Engine
Implement `sparc/inference/zero_shot.py`. Flow:

```
SatelliteFeatureSet + global trunk
  → load global_trunk.pt from registry
  → encode climate zone → trunk conditioning vector
  → resolve thermal anchor (station or ERA5)
  → compute morphological derivatives
  → run trunk forward: satellite features → physics representation
  → initialize lightweight city adapter from trunk defaults (no training)
  → apply PDE boundary condition: shift predictions to satisfy thermal anchor
  → MC Dropout (500 samples) → posterior mean + uncertainty
  → ZeroShotPrediction (T_mean, T_std, credible intervals)
```

The model predicts spatial structure relative to the thermal anchor — "given that today is X°C at the station, how does temperature distribute across this land cover configuration." The PDE bridges the scalar anchor to the spatial field.

### 4.2 — Few-Shot Fine-Tuning
Implement `sparc/inference/few_shot.py`. With 3–10 ground observations, fine-tune only the city adapter (2 linear layers) while keeping the trunk frozen. 50 epochs on a handful of points takes seconds on CPU. Expected uncertainty reduction vs. zero-shot:

| Stations | Uncertainty Reduction |
|---|---|
| 3 | ~35% |
| 5 | ~55% |
| 10 | ~70% |

### 4.3 — Scenario Simulation from Satellite Inputs
Extend `sparc/interventions/scenario_simulator.py` to accept satellite-derived feature sets as input. `sparc scenario --city "Lagos, Nigeria" --intervention "pct_canopy += 0.20"` works without any local training data.

### 4.4 — Output as GeoTIFF Stack
Output is a GeoTIFF stack (mean, std, lower_90, upper_90, dT_from_station, uhi_intensity) that opens directly in ArcGIS, QGIS, or any GIS tool.

### 4.5 — Global Coverage Validation

| Mode | RMSE Target | R² Target | 90% CI Coverage |
|---|---|---|---|
| Zero-shot (1 station) | < 1.8°C | 0.70–0.75 | 88–92% |
| Few-shot (5 stations) | < 1.1°C | 0.82–0.87 | 88–92% |
| Few-shot (10 stations) | < 0.9°C | 0.88–0.92 | 88–92% |
| Calibrated (full network) | < 0.8°C | 0.90+ | 88–92% |

---

## Phase 5 — Causal Inference Leadership

**Goal:** SPARC is the definitive spatial causal inference engine. Four pillars define this standard.

### Pillar 1 — Spatial Heterogeneity in Causal Effects

SPARC already has `spatial_cate.py`. The goal is a continuous CATE surface: treatment effects that vary as smooth functions of geography, not just binned demographic groups.

**Path forward:**
- Gaussian Process regression over the CATE surface, using the same Matérn kernel from Stage 0
- Uncertainty-quantified CATE maps where credible intervals widen appropriately in data-sparse areas
- Moran's I test on CATE residuals to verify spatial autocorrelation has been captured

**Outcome:** A practitioner can identify which neighborhoods see the largest cooling benefit from tree planting — and which see essentially none — with posterior uncertainty.

### Pillar 2 — Physical Identifiability

The PDE is not just a prediction constraint — it is a causal constraint. If two causal structures produce predictions that are physically indistinguishable (both satisfy the governing equation equally well), the PDE provides no additional identification. If one structure violates the PDE, it can be ruled out without data.

**Path forward:**
- Formal integration of PDE residuals as a constraint in the MC³ structure search: candidate DAGs that produce predictions violating the PDE are penalized
- A "physical plausibility score" for every causal hypothesis in the posterior
- Documentation of the identifiability conditions for each domain's governing PDE

**Outcome:** Causal claims are grounded not just in data fit but in physical law. The claim "tree canopy reduces temperature" is supported both statistically and thermodynamically.

### Pillar 3 — Longitudinal / Temporal Causal Inference

`panel.py` currently supports panel data models. The expansion covers:

- **Difference-in-differences (DiD)** with spatial spillovers: treated and control units are geographically clustered, and spillover between them must be modeled (connects to `interference.py`)
- **Synthetic Control** for spatial policy evaluation: construct a synthetic counterfactual city/region from a donor pool
- **Interrupted Time Series (ITS)** with spatial structure: detect and estimate the causal effect of a discrete policy change on a spatially distributed outcome
- **Event study designs** with spatial fixed effects

**Outcome:** SPARC can estimate the causal impact of a policy (tree planting program, zoning change, infrastructure investment) on a spatial outcome over time.

### Pillar 4 — Automated Causal Discovery

Today, users supply a DAG in `causal/dag.yml`. The future is ML-assisted DAG construction:

**Near-term:** LLM-assisted DAG proposal. The Claude integration in the desktop setup wizard is extended to reason about causal structure from domain text. The user describes their domain; the LLM proposes a DAG skeleton; the user edits and confirms.

**Medium-term:** Data-driven structure refinement. MC³ already searches over DAG structure. Add a gradient-based structure learning option (DCDI / ENCO / NOTEARS) as an alternative to MC³ for high-dimensional settings.

**Long-term:** Causal representation learning. Train the SharedTrunk with identifiability constraints (Schölkopf et al. causal representation learning) so the latent dimensions correspond to independent causal mechanisms. This makes the trunk's latent space causally interpretable, not just predictively useful.

---

## Phase 6 — Emerging Field Integration

The following eight emerging research areas have concrete integration paths in SPARC's architecture. Tiered by development horizon.

---

### Near-Term (1–2 releases)

#### LLM-Assisted DAG Construction
**What it is:** Use a large language model to propose a causal DAG skeleton from a plain-text domain description, existing literature, or a dataset schema. The user reviews and edits; the LLM explains its reasoning.

**SPARC integration:** Extend the Claude setup assistant in the desktop app. After project configuration, offer a "DAG Suggestion" step where the user describes their domain and the LLM proposes a `dag.yml`. SPARC validates the proposed DAG against data and flags issues.

**Why it matters:** The biggest adoption barrier for causal inference is DAG specification. Making this interactive and AI-assisted dramatically lowers the barrier.

---

#### PCMCI+ / DYNOTEARS for Spatio-Temporal Panels
**What it is:** Causal discovery algorithms designed specifically for time-lagged spatial data. PCMCI+ (Runge et al.) recovers time-lagged causal links between spatially distributed variables. DYNOTEARS extends NOTEARS to dynamical systems.

**SPARC integration:** Extend `sparc/causal/panel.py` with a `discover_spatiotemporal_causal_structure()` function that wraps PCMCI+ (via the `tigramite` library) for spatial panels. Output feeds directly into the MC³ prior.

**Why it matters:** Many critical spatial phenomena — disease spread, crime diffusion, pollutant transport — are fundamentally spatio-temporal causal processes. Current SPARC panel models assume a known causal structure; this discovers it from data.

---

### Medium-Term (2–4 releases)

#### Causal Representation Learning
**What it is:** Train neural encoders such that the latent dimensions correspond to independent causal mechanisms, not arbitrary statistical correlates (Schölkopf et al. 2021, Locatello et al. 2019).

**SPARC integration:** Add an identifiability regularizer to the SharedTrunk training objective. When the trunk is pretrained with JEPA (Phase 1), an additional loss term encourages the latent dimensions to be causally disentangled. This means that intervening on feature $k$ changes only the corresponding trunk dimension, not all of them.

**Why it matters:** A causally disentangled trunk produces causal effect estimates that are more robust to distribution shift — critical for zero-shot deployment in cities whose data distribution differs from training cities.

---

#### Transportability / Do-Calculus Generalization
**What it is:** Formally, a causal estimate is "transportable" from a source population to a target population if the causal structure is invariant and the distribution shift is accounted for (Bareinboim & Pearl 2016). For SPARC: does the tree-cooling causal effect estimated in Providence generalize to Lagos?

**SPARC integration:** Implement a `transport_causal_estimate()` function in `sparc/causal/` that takes a causal estimate from the source city, the do-calculus transport conditions, and the target city's selection diagram, and returns a transported causal estimate with uncertainty. Connect to the climate zone encoder: cities in the same Köppen zone are assumed to have transportable causal structures by default; cross-zone transport requires explicit justification.

**Why it matters:** Zero-shot scenario simulation (Phase 4.3) implicitly assumes causal transportability. Making this explicit and testable is the difference between a rigorous tool and an overconfident one.

---

#### Causal Fairness / Path-Specific Effects
**What it is:** Decompose causal effects along equity-relevant pathways. For heat exposure: how much of the disparity in heat burden between high-income and low-income neighborhoods is mediated by tree canopy (a fixable pathway) vs. explained by land use history (a structural pathway)?

**SPARC integration:** Extend `sparc/causal/mediation.py` with path-specific effect decomposition. Add a `fairness_audit()` function that computes natural direct and indirect effects stratified by a protected attribute (income decile, race/ethnicity, age group). Output a decomposition table alongside every causal report.

**Why it matters:** Every SPARC domain — heat, air quality, crime, health, flooding — has a disparity dimension. Path-specific effects give practitioners actionable leverage: which interventions address the most tractable pathways.

---

### Long-Horizon (4+ releases)

#### Causal Bandits for Active Intervention Design
**What it is:** Sequential experimental design where the spatial model acts as an oracle, and an agent selects which interventions to test in order to maximize information about the true causal effect with minimum cost (Lattimore et al. 2016).

**SPARC integration:** The scenario simulator becomes a reward model. A bandit agent (Thompson sampling or UCB-style) selects which neighborhoods to instrument, which interventions to test, and in what sequence, to optimally reduce posterior uncertainty about the causal effect surface. This turns SPARC from a passive analysis tool into an active experiment designer.

**Why it matters:** Reduces the data required to achieve a given causal precision. Critical for domains where data collection is expensive — seismic monitoring, clinical environmental health studies, precision agriculture.

---

#### Neural Causal Models (DCDI / ENCO)
**What it is:** Fully differentiable causal structure learning — gradient-based optimization over the space of DAGs (Brouillard et al. 2020, Lippe et al. 2021). Unlike MC³ (sampling-based), these methods can scale to hundreds of variables.

**SPARC integration:** Add a `neural_structure_learning()` option in `sparc/causal/causal_discovery.py` that runs DCDI or ENCO as an alternative to MC³ for high-dimensional variable sets. MC³ remains the default for interpretability; neural structure learning is the option for feature-rich datasets (satellite-derived features + social indicators + infrastructure features).

**Why it matters:** As SPARC expands into domains with richer feature sets (epidemiology, economics), MC³ sampling over large DAG spaces becomes computationally intractable. Neural structure learning scales where MC³ cannot.

---

#### MAUP-Aware Causal Estimation (Unique SPARC Differentiator)
**What it is:** The Modifiable Areal Unit Problem — causal estimates that flip or change magnitude depending on the spatial unit of analysis — is structurally present in every areal aggregation method. SPARC's continuous spatial modeling is resistant to MAUP by design, but this needs to be formalized as a diagnostic and guarantee.

**SPARC integration:** Implement a `maup_sensitivity_analysis()` function that reruns the causal inference at multiple spatial resolutions (point, 100m grid, 500m grid, census block) and reports the stability of causal estimates across scales. A "MAUP robustness score" is reported alongside every causal estimate: 1.0 means the estimate is scale-invariant; < 0.8 means it should be interpreted cautiously.

**Why it matters:** This is a unique SPARC capability. No other causal tool in the spatial domain explicitly diagnoses MAUP sensitivity. Publishing this as a methodology would be a significant academic contribution.

---

## Dependency Graph

```
Phase 1: Finish V3 Wiring
  1.1 EWC + Replay wired
  1.2 Temporal features wired
  1.3 JEPA pretraining pass
  1.4 Post-training artifact extraction
  1.5 Transfer validation
  1.6 Continual validation
        ↓
Phase 2: Central Registry
  2.1 Sync protocol           ← depends on 1.6
  2.2 Registry server         ← parallel with 2.1
  2.3 Auto post-run upload    ← parallel with 2.2
  2.4 Global trunk retrain    ← depends on 2.2
  2.5 CLI push/pull
        ↓
Phase 3: Satellite Ingestion  ← can start parallel with Phase 2
  3.1 Satellite ingest
  3.2 ERA5 fetch
  3.3 Station anchor
  3.4 Climate zone encoder    ← parallel with 3.1–3.3
  3.5 Satellite feature set
        ↓
Phase 4: Zero-Shot Inference  ← depends on Phase 2 + Phase 3
  4.1 Zero-shot engine
  4.2 Few-shot fine-tuning
  4.3 Scenario from satellite
  4.4 GeoTIFF output
  4.5 Global validation (≥8 cities)

Phase 5: Causal Leadership   ← parallel with Phase 2–4
  Pillar 1: Spatial CATE surface
  Pillar 2: Physical identifiability
  Pillar 3: Longitudinal / temporal
  Pillar 4: Automated discovery

Phase 6: Emerging Fields     ← near-term starts with Phase 5
  Near: LLM-DAG, PCMCI+
  Medium: Causal representation, transportability, causal fairness
  Long: Causal bandits, neural causal models, MAUP-aware estimation
```

---

## End State

A user opens SPARC Desktop, types "Dhaka, Bangladesh", selects "Urban Heat" and "July 2026", and clicks Run.

SPARC:
1. Pulls the latest global trunk from the central registry
2. Fetches Sentinel-2 imagery and ERA5 fields for Dhaka from open APIs
3. Resolves a thermal anchor from the nearest WMO station
4. Produces a probabilistic spatial temperature map with uncertainty bounds
5. Attributes the spatial pattern to land cover, building density, and vegetation fraction causally — with posteriors, refutations, and E-values
6. Simulates a counterfactual: "what if Dhaka planted trees in the 20 highest-burden neighborhoods?"
7. Outputs a GeoTIFF, a causal report, and a scenario comparison map

No local sensors. No ground truth network. No domain configuration required.

That is a global equity tool.
