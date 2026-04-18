# SPARC V4 Roadmap — From Transfer Learning to Zero-Shot Prediction

**SPARC Labs LLC | April 2026**
**Status: Planning**

---

## Overview

This roadmap describes the full path from where SPARC stands today (V2 production pipeline + V3 transfer/continual learning infrastructure) to a globally deployable zero-shot spatial prediction system.

The end state: a user types in a city name, draws a boundary, or provides a weather station reading — and SPARC produces a probabilistic spatial temperature field with uncertainty bounds, causal attribution, and scenario simulation capability. No ground-truth sensor network required.

This requires four phases:

| Phase | Name | Core Deliverable |
|-------|------|------------------|
| **Phase 1** | Finish V3 Wiring | EWC + replay losses active in training loop; multi-city training works end-to-end |
| **Phase 2** | Central Registry Infrastructure | Cloud-hosted model registry that user runs feed into automatically |
| **Phase 3** | Satellite + Climate Ingestion | Any city on Earth → SPARC feature set from free satellite imagery |
| **Phase 4** | Zero-Shot & Few-Shot Inference | Predict spatial fields for unseen cities from satellite data alone |

---

## What Already Exists

### V2 (Production)
- Full 5-stage pipeline: Correlogram → GWEN → Spatial CV → Causal → Scenarios
- `SPARCMetaLearner` with SharedTrunk + CityHead architecture
- 10-term PDE physics loss with staged curriculum
- `ProcessRateNet` learning spatially-varying thermal diffusivity
- MC³ + NUTS Bayesian causal inference
- Scenario simulator with 4-tier delta computation

### V3 (Infrastructure — Built, Partially Wired)
- **Temporal features**: `compute_diurnal_features()`, `get_snapshot_time_indices()` in `temporal.py`
- **Time embedding**: 3-way learned embedding (morning/midday/night) in `neural_meta.py`
- **Transient PDE loss**: Terms 9–10 (transient consistency + nocturnal calibration) in `pde_loss.py`
- **Transfer training**: `train_cold_start()`, `train_warm_start()`, `train_warm_start_finetune()` in `transfer_training.py`
- **Transfer validation**: A/B comparison orchestrator in `transfer_validation.py`
- **City Registry**: `CityRegistry` class managing trunk checkpoints, Fisher matrices, coresets, Welford states in `registry/city_registry.py`
- **EWC**: `compute_fisher_matrix()` + `ewc_penalty()` in `training/ewc.py`
- **Experience Replay**: `CoresetSelector` (greedy K-medoids) + `compute_replay_loss()` in `training/replay.py`
- **Welford Scaler**: Online standardization with `partial_fit()`, `merge()` in `data/welford.py`
- **Continual orchestrator**: `train_continual()` in `run/continual_training.py`
- **CLI**: `sparc transfer` and `sparc continual` subcommands
- **V4 stubs**: `SatelliteFeatureSet`, `ClimateZoneEncoder`, `zero_shot_predict()`, `few_shot_predict()` interfaces

### What's Not Wired Yet
The V3 modules exist but the training loop in `v2_neural_training.py` does not yet consume `config["_continual"]` to add EWC and replay losses to the gradient step. The orchestrator passes the config, the loss functions exist, but the actual `loss += lambda_ewc * ewc_penalty(...)` line inside the epoch loop is missing. This is Phase 1.

---

## Phase 1 — Finish V3 Wiring

**Goal:** Multi-city continual training works end-to-end. Run `sparc continual --cities providence.yml,boston.yml` and get a global trunk that demonstrably improves with each city.

### Step 1.1 — Wire EWC + Replay into Training Loop

**File:** `sparc/training/v2_neural_training.py`

Inside the `train_neural_meta()` epoch loop, check for `config["_continual"]` and add:

```python
# After computing base loss
if "_continual" in config:
    cont = config["_continual"]
    if cont["ewc_lambda"] > 0 and cont["fisher_matrices"]:
        loss += cont["ewc_lambda"] * ewc_penalty(model, cont["fisher_matrices"], ...)
    if cont["replay_lambda"] > 0 and cont["previous_coresets"]:
        for cs in cont["previous_coresets"]:
            loss += cont["replay_lambda"] * compute_replay_loss(model, cs["X"], cs["y"])
```

Also wire `config["_transfer"]` to call `model.load_trunk()` / `model.freeze_trunk()` at model initialization, and `model.unfreeze_trunk()` at the configured epoch.

### Step 1.2 — Wire Temporal Features into Training

**File:** `sparc/training/v2_neural_training.py`

When `config["temporal"]["snapshots"]` is present:
- Call `compute_diurnal_features()` during data prep
- Call `get_snapshot_time_indices()` to build `time_idx` tensor
- Pass `time_idx` to `model.forward()` and temporal kwargs to `sparc_joint_loss()`

### Step 1.3 — Post-Training Fisher + Trunk Extraction

After `train_neural_meta()` completes, compute and return:
- `result["fisher_matrix"] = compute_fisher_matrix(model, data_loader, _TRUNK_KEYS)`
- `result["trunk_state_dict"] = extract_trunk_params(model, _TRUNK_KEYS)`

These are what `train_continual()` expects but currently receives as empty dicts.

### Step 1.4 — Validation: Providence → Boston Transfer

Run the transfer validation with real data:
```bash
sparc transfer --source-project examples/providence_ri_uhi/project.yml \
               --target-project examples/boston_ma_uhi/project.yml \
               --output transfer_results/
```

**Success criteria:**
- Warm-start R² ≥ cold-start R² (trunk carries useful physics)
- Warm-start converges in fewer epochs
- `transfer_comparison.json` shows positive `r2_improvement`

### Step 1.5 — Validation: Continual Learning (2 Cities)

```bash
sparc continual --cities providence.yml,boston.yml --registry ./sparc_registry/
```

**Success criteria:**
- Providence R² does not degrade after Boston training (EWC works)
- Boston R² ≥ cold-start baseline (transfer helps)
- Registry contains both cities with all artifacts

---

## Phase 2 — Central Registry Infrastructure

**Goal:** A cloud-hosted registry that user runs automatically feed into, building a progressively better global trunk.

### Step 2.1 — Registry Sync Protocol

**New file:** `sparc/registry/sync.py`

Define a protocol for syncing local registry artifacts to a central registry:

```python
def push_city(local_registry, remote_url, city_name, api_key):
    """
    Upload a city's artifacts to the central registry.
    
    Uploads: trunk_checkpoint.pt, fisher_matrix.pt, coreset.npz,
             welford_state.pkl, metrics.json
    Does NOT upload raw data — only model artifacts and summary stats.
    """

def pull_global_trunk(remote_url, local_path, api_key):
    """
    Download the latest global trunk from the central registry.
    """

def pull_registry_manifest(remote_url, api_key):
    """
    Get the list of all cities and their metadata from central.
    """
```

**Privacy constraint:** Only model weights, Fisher matrices, coresets (400 points, standardized), and aggregate metrics leave the user's machine. Raw data never leaves. The coreset is too small and too transformed to reconstruct the original dataset.

### Step 2.2 — Central Registry Server

**New module:** `sparc/registry/server.py` (or separate repo)

A lightweight API server (FastAPI) that:
- Accepts city artifact uploads via authenticated POST
- Stores artifacts in structured cloud storage (S3 / Azure Blob / GCS)
- Maintains a global manifest of all registered cities
- Periodically recomputes the global trunk by merging Fisher matrices and retraining
- Serves the latest global trunk via GET
- Provides a public manifest of available cities and aggregate performance

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/cities/{name}/upload` | Upload city artifacts |
| GET | `/v1/trunk/latest` | Download latest global trunk |
| GET | `/v1/manifest` | List all registered cities + metadata |
| GET | `/v1/cities/{name}/metrics` | Get a city's training metrics |
| POST | `/v1/trunk/retrain` | Trigger global trunk recomputation (admin) |

### Step 2.3 — Automated Post-Run Upload

**File:** `sparc/run/run_enhanced_pipeline.py` + `sparc/__main__.py`

After a successful `sparc run` with `registry.path` configured:
1. Compute Fisher matrix and select coreset (same as continual does)
2. Register city locally
3. If `registry.remote_url` is set, prompt user: "Upload artifacts to central registry? (y/n)"
4. On confirmation, call `push_city()`

This makes the central registry grow organically as users run SPARC on new cities.

### Step 2.4 — Global Trunk Recomputation

**File:** `sparc/registry/retrain.py`

When new cities arrive at the central registry, recompute the global trunk:

1. Pull all city coresets + Fisher matrices from storage
2. Merge Welford scalers across all cities
3. Run continual training through all cities in sequence (or by climate zone clusters)
4. Evaluate on held-out coreset points from each city
5. If new trunk outperforms previous on aggregate metrics, publish as latest
6. Version the trunk: `global_trunk_v{N}.pt`

This can run nightly or on-demand when a new city is added.

### Step 2.5 — CLI Integration

```bash
# After a run, push to central
sparc push --registry ./sparc_registry/ --city providence --remote https://registry.sparclabs.io

# Pull latest global trunk before a new city run
sparc pull --remote https://registry.sparclabs.io --output ./sparc_registry/

# Check what's in the central registry
sparc registry --remote https://registry.sparclabs.io --list
```

---

## Phase 3 — Satellite + Climate Ingestion

**Goal:** Any city on Earth can be converted to a SPARC feature set from freely available satellite imagery and climate reanalysis data, with no local data collection.

### Step 3.1 — Satellite Ingestion Pipeline

**New file:** `sparc/data/satellite_ingest.py`

Standardized ingestion from Sentinel-2 and Landsat 8/9:

- **Spectral indices:** NDVI, NDBI, MNDWI, BSI, Albedo
- **Spectral unmixing:** Derive Pct_Canopy, Pct_Impervious, Pct_Water from indices
- **Morphological derivatives:** ∇²NDVI, |∇NDBI|, thermal_stress_laplacian (same features the PDE loss uses)
- **Topography:** SRTM 30m DEM → elevation, slope, aspect

Data sources (all free, global coverage):
- Sentinel-2 L2A: 10m multispectral, every 5 days, via Copernicus Open Access Hub
- Landsat 8/9 Collection 2: 30m multispectral, every 16 days, via USGS EarthExplorer
- SRTM: 30m DEM, static, via OpenTopography

### Step 3.2 — ERA5 Climate Forcing

**New file:** `sparc/data/era5_fetch.py`

Pull ERA5 reanalysis fields from Copernicus Climate Data Store:
- 2m air temperature (T2m) — the no-station fallback thermal anchor
- Surface solar radiation downward (SSRD) — source term S for PDE
- 10m wind u/v — advection term (optional, improves accuracy)
- Humidity — latent heat flux modulation

ERA5 provides global hourly fields at 0.25° (~28km) resolution. Regrid to target resolution using bilinear interpolation. This is the "truly zero ground infrastructure" fallback.

### Step 3.3 — Weather Station Anchor

**New file:** `sparc/data/station_anchor.py`

Resolve the thermal boundary condition:

| Scenario | Source | Uncertainty |
|----------|--------|-------------|
| Station inside domain | Direct observation | ±0.3°C |
| Station 0–25km outside | Nearest WMO GSOD | ±0.8°C |
| Station 25–50km outside | Nearest WMO GSOD | ±1.5°C |
| No station (ERA5 fallback) | ERA5 T2m | ±2.1°C |

Data source: NOAA GSOD (Global Summary of the Day) — free, global, daily back to 1929.

The station reading (or ERA5 mean) becomes the Dirichlet boundary condition for the PDE. It anchors the domain-level thermal baseline. The PDE then propagates spatial structure from the satellite-derived land cover.

### Step 3.4 — Climate Zone Encoder (Full Implementation)

**File:** `sparc/models/climate_encoder.py` (expand existing stub)

Upgrade from the current 30-class Köppen embedding to a richer climate conditioning:

```python
class ClimateZoneEncoder(nn.Module):
    """
    Inputs (all from ERA5/CHELSA climate normals):
        - Köppen-Geiger zone label (0–29)
        - Mean annual temperature
        - Mean annual precipitation
        - Annual solar irradiance
        - Humidity index
    
    Output:
        - climate_shift: (trunk_dim,) additive modulation on trunk output
    """
```

The encoder learns that "high NDVI in Cfa (humid subtropical, Providence) cools differently than high NDVI in BWh (hot desert, Phoenix)" — same vegetation fraction, different cooling physics due to latent heat and background temperature.

### Step 3.5 — Unified Satellite Feature Set

**File:** `sparc/data/satellite_types.py` (expand existing stub)

Expand `SatelliteFeatureSet` to include all fields needed for zero-shot:

```python
@dataclass
class SatelliteFeatureSet:
    coords: np.ndarray           # (N, 2) projected coordinates
    bands: dict                  # spectral indices + land cover fractions
    resolution_m: float
    
    # Topography
    elevation_m: np.ndarray
    slope_deg: np.ndarray | None
    
    # ERA5 forcing
    solar_forcing_wm2: float
    t2m_era5_K: float            # fallback thermal anchor
    wind_u_ms: float | None
    wind_v_ms: float | None
    
    # Thermal anchor
    T_station: float | None      # None → ERA5 fallback
    station_id: str | None
    station_dist_km: float | None
    
    # Climate
    koppen_zone: str
    climate_normals: dict        # MAT, MAP, solar, humidity
    
    # Metadata
    date: str
    city: str
    crs: str
```

---

## Phase 4 — Zero-Shot & Few-Shot Inference

**Goal:** `sparc predict --city "Lagos, Nigeria" --date 2026-07-15` produces a spatial temperature field with uncertainty — no local training data.

### Step 4.1 — Zero-Shot Inference Engine

**File:** `sparc/inference/zero_shot.py` (implement existing stub)

```
Input: SatelliteFeatureSet + global trunk from registry
                    ↓
    1. Load global_trunk.pt from central registry
    2. Encode climate zone → trunk conditioning vector
    3. Resolve thermal anchor (station or ERA5)
    4. Compute morphological derivatives (∇²NDVI, etc.)
    5. Run trunk forward: satellite features → physics representation
    6. Initialize lightweight city adapter from trunk defaults (no training)
    7. Apply PDE boundary condition: shift predictions to satisfy anchor
    8. MC Dropout (500 samples) → posterior mean + uncertainty
                    ↓
Output: ZeroShotPrediction with T_mean, T_std, credible intervals
```

The model doesn't predict absolute temperature from scratch. It predicts "given that today is 87°F at the station, how does temperature distribute spatially across this land cover configuration." The PDE is the bridge between the scalar anchor and the spatial field.

### Step 4.2 — Few-Shot Fine-Tuning

**File:** `sparc/inference/few_shot.py` (implement existing stub)

When 3–10 ground observations are available, fine-tune only the city adapter (2 linear layers) while keeping the trunk frozen. 50 epochs on a handful of points takes seconds on CPU.

Expected uncertainty reduction vs zero-shot:
- 3 stations: ~35% reduction in posterior std
- 5 stations: ~55% reduction
- 10 stations: ~70% reduction

This is the practical sweet spot: a traditional UHI study deploys 20–50 sensors for weeks. Few-shot SPARC achieves comparable spatial prediction from 3–5 stations deployed for a single day.

### Step 4.3 — Scenario Simulation from Satellite Inputs

**File:** `sparc/interventions/scenario_simulator.py` (extend)

Enable counterfactual scenarios on zero-shot predictions:

```bash
sparc scenario --city "Lagos, Nigeria" --intervention "pct_canopy += 0.20"
```

The trunk + PDE physics know how canopy changes affect temperature (learned from Providence, Boston, Phoenix, etc.). The satellite features give the current land cover. The scenario shifts the relevant features and re-runs the forward pass.

This is the key differentiator from any existing satellite temperature product: not just "here is the temperature" but "here is what the temperature would be if you planted trees in these neighborhoods."

### Step 4.4 — Global Coverage Validation

**File:** `sparc/validation/global_coverage.py`

Before claiming zero-shot capability, validate against completely held-out cities:

1. Run zero-shot prediction using only satellite + nearest station
2. Compare to observed station network
3. Compute: RMSE, MAE, Moran's I of residuals, 90% CI coverage rate
4. Compare to baselines:
   - T = T_station everywhere (no spatial structure)
   - T = linear regression on NDVI + NDBI (no physics)
   - MODIS LST (land surface temperature — not air temperature)

**Target performance:**

| Mode | RMSE Target | R² Target | 90% CI Coverage |
|------|------------|-----------|-----------------|
| Zero-shot (1 station) | < 1.8°C | 0.70–0.75 | 88–92% |
| Few-shot (5 stations) | < 1.1°C | 0.82–0.87 | 88–92% |
| Few-shot (10 stations) | < 0.9°C | 0.88–0.92 | 88–92% |
| Calibrated (full network) | < 0.8°C | 0.90+ | 88–92% |

**Minimum cities for reliable zero-shot by climate zone:**

| Climate Group | Example Cities | Minimum Needed |
|---------------|---------------|----------------|
| Temperate (C) | Providence, Boston, London | 2 |
| Continental (D) | Chicago, Minneapolis | 2 |
| Tropical (A) | Miami, Houston, Bangkok | 3 |
| Arid (B) | Phoenix, Las Vegas | 2 |
| Polar/Alpine (E) | Denver, Anchorage | 1 |

Eight cities across four climate groups is the minimum viable cross-climate trunk.

### Step 4.5 — Operational CLI

```bash
# Zero-shot: satellite imagery + date → temperature map
sparc predict \
    --city "Lagos, Nigeria" \
    --date 2026-07-15 \
    --satellite sentinel2 \
    --output lagos_heat_map.tif

# Few-shot: add ground observations
sparc predict \
    --city "Lagos, Nigeria" \
    --date 2026-07-15 \
    --satellite sentinel2 \
    --stations stations.csv \
    --output lagos_heat_map.tif

# Scenario from satellite inputs
sparc scenario \
    --city "Lagos, Nigeria" \
    --date 2026-07-15 \
    --intervention "pct_canopy += 0.20" \
    --output lagos_scenario.tif
```

Output is a GeoTIFF stack (mean, std, lower_90, upper_90, dT_from_station, uhi_intensity) that opens directly in ArcGIS or QGIS.

---

## Dependency Graph

```
Phase 1: Finish V3 Wiring
    1.1  Wire EWC + replay into training loop
    1.2  Wire temporal features into training
    1.3  Post-training Fisher + trunk extraction
    1.4  Transfer validation (Providence → Boston)
    1.5  Continual validation (2 cities)
            ↓
Phase 2: Central Registry
    2.1  Registry sync protocol                ← depends on 1.5
    2.2  Central registry server
    2.3  Automated post-run upload             ← parallel with 2.2
    2.4  Global trunk recomputation            ← depends on 2.2
    2.5  CLI push/pull commands
            ↓
Phase 3: Satellite + Climate
    3.1  Satellite ingestion pipeline          ← can start parallel with Phase 2
    3.2  ERA5 climate forcing fetch
    3.3  Weather station anchor
    3.4  Climate zone encoder (full)           ← parallel with 3.1–3.3
    3.5  Unified satellite feature set
            ↓
Phase 4: Zero-Shot Inference
    4.1  Zero-shot inference engine            ← depends on Phase 2 + Phase 3
    4.2  Few-shot fine-tuning
    4.3  Scenario simulation from satellite
    4.4  Global coverage validation            ← requires ≥ 8 cities in registry
    4.5  Operational CLI
```

---

## The Central Model Concept

The central registry is the backbone of the zero-shot system. Here is how data flows:

```
User A (Providence)                    Central Registry
    sparc run → trained model    →    push trunk + Fisher + coreset
                                              ↓
User B (Boston)                       Recompute global trunk
    sparc pull ← global trunk    ←    (nightly or on new city)
    sparc run → trained model    →    push trunk + Fisher + coreset
                                              ↓
User C (Phoenix)                      Global trunk now informed by
    sparc pull ← global trunk    ←    Providence + Boston
    sparc run → trained model    →    push trunk + Fisher + coreset
                                              ↓
                                      After 8+ cities, 4+ climate zones:
                                              ↓
Anyone (Lagos, no sensors)            Zero-shot capable
    sparc predict ← global trunk ←    Satellite + station → prediction
```

**What gets uploaded:** Model weights (trunk checkpoint), Fisher information matrix, 400-point coreset (standardized features), Welford scaler state, aggregate metrics (R², RMSE, Moran's I). Approximately 5–15 MB per city.

**What never leaves the user's machine:** Raw sensor data, coordinates of individual observations, personally identifiable information, client project details.

**Who controls the central registry:** SPARC Labs hosts and maintains it. Users opt in to contributing. The global trunk is available to all users regardless of whether they contribute. Contributing improves the trunk for everyone including the contributor's future runs.

---

## Why This Matters

Urban heat is the leading cause of weather-related mortality globally. The burden falls disproportionately on cities in low- and middle-income countries — cities least likely to have spatial temperature monitoring networks, and most likely to have rapid urban expansion increasing heat exposure.

A meaningful urban heat analysis currently requires a ground monitoring network: tens of thousands of dollars and weeks of fieldwork. Most cities can't afford it.

SPARC at Phase 4 completion means any city with Sentinel-2 coverage — the entire land surface of Earth, every 5 days, for free — can get a probabilistic spatial air temperature analysis from a single daily station reading. The uncertainty is higher than a calibrated analysis, but it is honest, quantified, and physically grounded.

Not just "here is the temperature" but "here is what the temperature would be if you planted trees in these neighborhoods" — with uncertainty, with causal attribution, from satellite imagery alone.

That is a global equity tool. The cities that need urban heat analysis most are exactly the cities this system makes newly accessible.
