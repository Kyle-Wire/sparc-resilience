# Track D — Zero-Shot Satellite Temperature Prediction
**SPARC V2 · SPARC Labs · Kyle Wire · 2026**

---

## What This Track Is

Track D is the destination that makes everything else in the roadmap worthwhile at scale.

Tracks A–C build a model that learns transferable urban heat physics across cities. Track D is the application of that accumulated knowledge to predict spatial air temperature fields from satellite imagery alone — no ground temperature network required. Any city on Earth with Landsat or Sentinel-2 coverage becomes analyzable.

This is not a new model. It is the natural endpoint of the transfer learning infrastructure: a shared trunk that has seen enough climate diversity that it can propagate a single weather station anchor across a full city domain using only satellite-derived land surface inputs and PDE physics.

---

## The Two Prediction Regimes

| Regime | Ground Data Required | When Available |
|--------|---------------------|----------------|
| **Calibrated** (current) | Full spatial temperature network | Now — city-by-city training |
| **Few-shot** | 3–10 weather station readings | After Track B (transfer learning) |
| **Zero-shot** | 1 weather station OR ERA5 mean | After Track D (this track) |

The progression is not binary. Each stage reduces the ground data requirement while maintaining physically defensible predictions. Zero-shot is the limit — one number (today's temperature from the nearest station) anchors the boundary condition, and the PDE propagates the spatial structure from satellite inputs.

---

## The Full Zero-Shot Input Pipeline

Every input in this pipeline is globally available with no local ground infrastructure:

```
Sentinel-2 / Landsat 8-9  (30m multispectral, global, every 5 days)
        ↓
Spectral indices:
    NDVI   — vegetation fraction
    NDBI   — built-up intensity
    MNDWI  — water bodies
    BSI    — bare soil index
    Albedo — surface reflectance (MCD43A3 or Landsat-derived)
        ↓
Spectral unmixing:
    Pct_Canopy      — from NDVI + classification
    Pct_Impervious  — from NDBI + spectral unmixing
    Pct_Water       — from MNDWI mask
        ↓
Topography  (SRTM / 3DEP, global 30m):
    Elevation_m
    Slope, aspect  (optional — improves solar forcing estimate)
        ↓
Morphological derivatives  (Step 3, Track A):
    ∇²NDVI, |∇NDBI|, thermal_stress_laplacian, cooling_potential_laplacian
        ↓
Climate normals  (ERA5 / CHELSA, global hourly):
    Mean annual temperature   — climate zone conditioning
    Humidity index            — latent heat flux modulation
    Solar irradiance          — source term S
    Wind u, v components      — advection term QA (optional but significant)
        ↓
Boundary condition anchor:
    T_station = nearest WMO weather station daily mean
    OR T_era5  = ERA5 2m air temperature (truly no-station fallback)
        ↓
SPARC shared trunk  (pretrained across N cities, Track C registry)
        ↓
City adapter  (zero-shot: initialized from trunk, no fine-tuning)
             (few-shot: fine-tuned on 3–10 station observations)
        ↓
Predicted air temperature surface + uncertainty + causal effects
```

---

## What the Model Actually Does

The station reading (or ERA5 mean) sets the Dirichlet boundary condition — the domain-level thermal baseline for that day:

```python
# Today's thermal anchor
T_station = 87.0   # °F — nearest WMO station daily mean

# Dirichlet BC: anchors the overall temperature level
BoundarySpec(
    type=BoundaryType.DIRICHLET,
    value=T_station,
    direction='domain_mean'
)
```

The PDE then propagates that anchor across the domain, modulated by land cover, diffusivity, and source terms:

```
∇²T = S(T_station, albedo, solar_forcing) / α(NDVI, NDBI, Pct_Water)
```

The model does not predict "what is the temperature today" from scratch — it predicts "given that today is 87°F at the station, how does that temperature distribute spatially across the city, given its land cover structure." That is a physically well-posed problem. The PDE is the bridge between the scalar boundary anchor and the spatial field.

The output is not a single number but a full spatial prediction:

```
"Block X: 93.2°F ± 1.4°F  (6.2°F above station)"
"Block Y: 84.1°F ± 1.1°F  (2.9°F below station)"
"Park Z:  81.3°F ± 0.9°F  (5.7°F below station)"
```

Every prediction comes with a credible interval. Uncertainty is higher in zero-shot mode (no local ground truth) but is honest and quantified rather than absent.

---

## Track D Steps

---

### D1 — Satellite Ingestion Pipeline

**File:** `sparc/data/satellite_ingest.py`
**Depends on:** Track A complete, Track C registry operational

Build a standardized ingestion pipeline that takes raw satellite imagery and produces the SPARC feature set. This pipeline must be city-agnostic — it should work identically for Providence, Nairobi, and Jakarta.

**Key functions:**

```python
def ingest_sentinel2(scene_path, aoi_bounds, resolution=10):
    """
    Ingest Sentinel-2 L2A scene, compute spectral indices.
    Returns: NDVI, NDBI, MNDWI, BSI, Albedo at target resolution
    """

def ingest_landsat(scene_path, aoi_bounds, resolution=30):
    """
    Ingest Landsat 8/9 Collection 2 L2, compute spectral indices.
    Same output schema as Sentinel-2 path.
    """

def spectral_unmixing(ndvi, ndbi, mndwi, method='linear'):
    """
    Decompose spectral indices into fractional land cover.
    Returns: Pct_Canopy, Pct_Impervious, Pct_Water, Pct_Soil
    """

def compute_albedo_landsat(red, nir, swir1, swir2):
    """
    Liang (2001) narrowband-to-broadband conversion.
    Validated against MODIS MCD43A3 albedo product.
    """

def fetch_era5_forcing(lat_bounds, lon_bounds, date, variables):
    """
    Pull ERA5 reanalysis fields for a given date and domain.
    Variables: t2m (air temp), u10, v10 (wind), ssrd (solar), sp (pressure)
    Returns: dict of spatial fields at 0.25° resolution, regridded to target res
    """

def fetch_srtm(lat_bounds, lon_bounds, resolution=30):
    """
    Pull SRTM 1-arcsecond DEM, compute elevation, slope, aspect.
    """
```

**Standardized output schema:**

```python
@dataclass
class SatelliteFeatureSet:
    # Spectral indices
    ndvi:            np.ndarray   # (H, W)
    ndbi:            np.ndarray
    mndwi:           np.ndarray
    albedo:          np.ndarray

    # Fractional land cover
    pct_canopy:      np.ndarray
    pct_impervious:  np.ndarray
    pct_water:       np.ndarray

    # Topography
    elevation_m:     np.ndarray
    slope_deg:       np.ndarray   # optional

    # ERA5 forcing scalars (spatially uniform or regridded)
    solar_forcing_wm2:  float
    t2m_era5_K:         float     # fallback if no station
    wind_u_ms:          float     # optional
    wind_v_ms:          float     # optional

    # Boundary anchor
    T_station:       float | None  # None triggers ERA5 fallback
    station_id:      str   | None
    station_dist_km: float | None  # distance to nearest station

    # Metadata
    date:            str
    city:            str
    crs:             str
    resolution_m:    float
    coords:          np.ndarray    # (N, 2) projected coordinates
```

**Deliverable:** Any city on Earth can be ingested from Sentinel-2 or Landsat into a `SatelliteFeatureSet` in a single function call. ERA5 fetch works as no-station fallback. Unit tests against known Providence scene.

---

### D2 — Climate Zone Conditioning

**File:** `sparc/models/climate_encoder.py`
**Depends on:** D1, Track B (shared trunk architecture)

The shared trunk learned urban heat physics across cities — but cities in different climate zones have systematically different relationships between land cover and temperature. A tropical city and a continental city with identical NDVI values have different canopy cooling effects because of humidity, solar angle, and background temperature.

Climate zone conditioning adds a lightweight embedding that shifts the trunk's representations based on the city's climate context — without requiring any local temperature observations.

```python
class ClimateZoneEncoder(nn.Module):
    """
    Encodes climate context as a latent shift vector applied to the shared trunk.

    Inputs (all from ERA5 / CHELSA climate normals — no ground stations):
        - Mean annual temperature (MAT)
        - Mean annual precipitation (MAP)
        - Annual solar irradiance
        - Humidity index (MAP / potential evapotranspiration)
        - Climate zone label (Köppen-Geiger, 0-29)

    Output:
        - climate_shift: (trunk_dim,) vector that additively modulates trunk output
        - climate_embed: (climate_dim,) for diagnostic logging
    """

    def __init__(self, climate_dim=16, trunk_dim=128):
        self.koppen_embed  = nn.Embedding(30, climate_dim)   # 30 Köppen zones
        self.scalar_enc    = nn.Linear(4, climate_dim)       # MAT, MAP, solar, humidity
        self.shift_proj    = nn.Linear(climate_dim * 2, trunk_dim)
```

**Köppen zone coverage requirements before zero-shot is reliable:**

| Climate Group | Example Cities | Minimum Cities Needed |
|---------------|---------------|----------------------|
| Temperate (C) | Providence, Boston, London, Seattle | 2 (already have) |
| Continental (D) | Chicago, Minneapolis, Moscow | 2 |
| Tropical (A) | Miami, Houston, Bangkok, Lagos | 3 |
| Arid (B) | Phoenix, Las Vegas, Riyadh | 2 |
| Polar/Alpine (E) | Denver (altitude), Anchorage | 1 |

Eight cities across four climate groups is the minimum viable cross-climate trunk. Below this threshold, zero-shot predictions outside the training climate zones are extrapolations the model cannot defend.

**Deliverable:** `ClimateZoneEncoder` integrated into shared trunk. Climate shift vector logged per city during training. Trunk representations visualized by climate zone (t-SNE) showing separation — confirms the encoder is learning climate-conditioned representations.

---

### D3 — Weather Station Anchor Integration

**File:** `sparc/data/station_anchor.py`
**Depends on:** D1, Track A Step 7 (boundary conditions)

This is the operational component that makes "today is 87°F" work as a boundary condition. It needs to handle three scenarios gracefully:

**Scenario 1 — Station available, inside domain**
Use station observation directly as Dirichlet BC at station coordinates. PDE propagates outward.

**Scenario 2 — Station available, outside domain (nearest WMO station)**
Use station as domain-mean Dirichlet BC. Adds uncertainty proportional to station distance.

**Scenario 3 — No station (ERA5 fallback)**
Use ERA5 2m air temperature field as spatially distributed BC. Higher uncertainty, but still physically grounded.

```python
def resolve_thermal_anchor(
    domain_bounds,
    date,
    station_obs=None,      # (lat, lon, T_obs) if available
    max_station_dist_km=50,
    era5_fallback=True
) -> ThermalAnchor:
    """
    Returns a ThermalAnchor with:
        - T_anchor: float (°C)
        - anchor_type: 'station_interior' | 'station_exterior' | 'era5'
        - anchor_uncertainty: float (°C std dev — increases with distance)
        - bc_spec: BoundarySpec for the PDE solver
    """

def build_station_dirichlet_bc(T_anchor, anchor_uncertainty):
    """
    Converts station reading to BoundarySpec.
    Uncertainty propagates into prediction credible intervals.
    """

def fetch_wmo_nearest(lat, lon, date, max_dist_km=50):
    """
    Query nearest WMO GSOD station for daily mean temperature.
    Falls back to ERA5 if no station within max_dist_km.
    NOAA GSOD is free, global, daily back to 1929.
    """
```

**Uncertainty propagation rule:**
- Station inside domain: anchor uncertainty = ±0.3°C (instrument error only)
- Station 0–25km outside: ±0.8°C
- Station 25–50km outside: ±1.5°C
- ERA5 fallback: ±2.1°C (ERA5 2m temperature validation error vs. stations)

These uncertainties feed directly into the prediction credible intervals — zero-shot predictions are wider than calibrated predictions by a known, principled amount.

**Deliverable:** `resolve_thermal_anchor()` working for all three scenarios. Uncertainty propagation tested against held-out Providence stations (treat them as missing, compare predicted credible intervals to actual observations).

---

### D4 — Zero-Shot Inference Engine

**File:** `sparc/inference/zero_shot.py`
**Depends on:** D1, D2, D3, Track C registry with ≥4 cities

This is the inference engine that puts everything together. It takes a city, a date, and satellite data and produces a predicted temperature field with uncertainty — no local training required.

```python
def predict_zero_shot(
    city_name:       str,
    date:            str,
    satellite_data:  SatelliteFeatureSet,
    registry_path:   str,
    station_obs:     tuple | None = None,   # (lat, lon, T_obs) optional
    n_samples:       int = 500,             # posterior samples for uncertainty
) -> ZeroShotPrediction:
    """
    Full zero-shot prediction pipeline.

    Steps:
        1. Load global_trunk from registry
        2. Encode climate zone from ERA5 normals
        3. Resolve thermal anchor (station or ERA5 fallback)
        4. Compute morphological derivatives from satellite features
        5. Initialize city adapter from trunk (no fine-tuning)
        6. Run forward pass: satellite features → trunk → adapter → T_pred
        7. Apply PDE BC: shift T_pred to satisfy Dirichlet anchor
        8. Sample posterior: propagate anchor uncertainty + model uncertainty
        9. Return spatial field with credible intervals
    """
```

**Output schema:**

```python
@dataclass
class ZeroShotPrediction:
    T_mean:      np.ndarray   # (H, W) predicted temperature field
    T_std:       np.ndarray   # (H, W) posterior standard deviation
    T_lower_90:  np.ndarray   # (H, W) 5th percentile
    T_upper_90:  np.ndarray   # (H, W) 95th percentile

    # Relative fields (station-anchored)
    dT_from_station: np.ndarray   # how much hotter/cooler than station
    uhi_intensity:   np.ndarray   # local deviation from domain mean

    # Diagnostics
    anchor_type:        str        # 'station_interior' | 'era5'
    anchor_uncertainty: float
    climate_zone:       str        # predicted Köppen zone
    w_source:           float      # PDE blend weight for this city
    registry_cities:    list[str]  # which cities contributed to trunk

    # Metadata
    city:        str
    date:        str
    crs:         str
    resolution_m: float
```

**Deliverable:** `predict_zero_shot()` producing spatial T field for a city not in the training set. Validated against held-out city where ground truth is available. Prediction credible intervals capture observed station temperatures at the correct coverage rate (90% CI should contain ~90% of station observations).

---

### D5 — Few-Shot Fine-Tuning

**File:** `sparc/inference/few_shot.py`
**Depends on:** D4

Zero-shot uses the trunk as-is with no local data. Few-shot adds a small number of ground observations (3–10 weather stations or rooftop sensors) to fine-tune the city adapter, dramatically reducing prediction uncertainty without requiring a full spatial network.

```python
def predict_few_shot(
    city_name:      str,
    date:           str,
    satellite_data: SatelliteFeatureSet,
    ground_obs:     list[tuple],   # [(lat, lon, T_obs), ...] 3-10 points
    registry_path:  str,
    n_finetune_epochs: int = 50,
) -> FewShotPrediction:
    """
    Fine-tunes city adapter on sparse ground observations.

    The trunk is frozen. Only the city adapter (2 linear layers) updates.
    50 epochs on 3-10 points takes seconds on CPU.

    Expected uncertainty reduction vs zero-shot:
        3 stations:  ~35% reduction in posterior std
        5 stations:  ~55% reduction
        10 stations: ~70% reduction
    """
```

**Why this matters practically:**

A standard urban heat island study deploys 20–50 temperature sensors across a city for 2–4 weeks. That's weeks of fieldwork and thousands of dollars. Few-shot SPARC achieves comparable spatial prediction quality from 3–5 rooftop weather stations deployed for a single day. The satellite data carries the spatial structure; the ground observations anchor the thermal level and reduce uncertainty. The marginal value of each additional station follows a diminishing returns curve that the model can quantify in advance — telling a client exactly how many stations they need to achieve a target uncertainty level.

**Deliverable:** Few-shot fine-tuning validated on Providence and Boston using 3, 5, and 10 station subsets. Uncertainty reduction curve plotted. Comparison against zero-shot baseline shows expected improvement.

---

### D6 — Global Coverage Validation

**File:** `sparc/validation/global_coverage.py`
**Depends on:** D4, D5, Track C with ≥8 cities across ≥4 climate zones

Before claiming zero-shot capability, validate against cities that were withheld from training entirely — true out-of-sample validation in the strongest possible sense.

**Validation protocol:**

```
For each held-out city:
    1. Run zero-shot prediction using only satellite data + nearest station
    2. Compare predicted T_mean to observed station network
    3. Compute: RMSE, MAE, Moran's I of residuals, coverage rate of 90% CI
    4. Compare to naive baselines:
        - Baseline 1: T_everywhere = T_station (no spatial structure)
        - Baseline 2: T = linear regression on NDVI + NDBI (no physics)
        - Baseline 3: MODIS LST (land surface temperature, not air temperature)
    5. Report skill score: how much better than each baseline
```

**Target performance benchmarks:**

| Metric | Zero-Shot Target | Few-Shot (5 stations) Target |
|--------|-----------------|------------------------------|
| RMSE vs station network | < 1.8°C | < 1.1°C |
| Moran's I of residuals | < 0.15 | < 0.10 |
| 90% CI coverage rate | 88–92% | 88–92% |
| Skill vs. station-only baseline | > 40% RMSE reduction | > 65% RMSE reduction |

The Moran's I of residuals is the critical diagnostic — it tells you whether the model is capturing the spatial structure correctly. If residuals are spatially autocorrelated, the model is missing systematic spatial patterns that the PDE or satellite features should be catching.

**Deliverable:** Validation report across held-out cities. Performance benchmarks met or documented with explanation. Comparison to MODIS LST baseline (this is the key comparison for publication — SPARC predicts air temperature, not land surface temperature, which is what satellites directly measure and what human health outcomes depend on).

---

### D7 — Operational Inference Interface

**File:** `sparc/inference/predict.py` + CLI
**Depends on:** D4, D5, D6

Package the zero-shot and few-shot pipelines into a clean interface that a non-expert operator can run. This is what makes SPARC useful to a city government, an NGO, or a public health agency that doesn't have ML expertise.

```bash
# Zero-shot: satellite imagery + date → temperature map
sparc predict \
    --city "Lagos, Nigeria" \
    --date 2026-07-15 \
    --satellite sentinel2 \
    --output lagos_heat_map.tif

# Few-shot: add 5 station observations
sparc predict \
    --city "Lagos, Nigeria" \
    --date 2026-07-15 \
    --satellite sentinel2 \
    --stations stations.csv \   # lat, lon, T_obs
    --output lagos_heat_map.tif

# Scenario: what if canopy increases by 20%?
sparc scenario \
    --city "Lagos, Nigeria" \
    --date 2026-07-15 \
    --intervention "pct_canopy += 0.20" \
    --output lagos_scenario.tif
```

The scenario capability is the key differentiator from any existing satellite-based temperature product. Not just "here is the temperature" but "here is what the temperature would be if you planted trees in these neighborhoods" — with uncertainty, with causal attribution, from satellite imagery alone.

**Deliverable:** CLI working end-to-end for zero-shot and few-shot prediction. Scenario simulation operational from satellite inputs. Output is a GeoTIFF stack (mean, std, lower_90, upper_90, dT_from_station, uhi_intensity) that can be opened directly in ArcGIS or QGIS.

---

## Why the Air Temperature vs. LST Distinction Matters

Satellites measure **land surface temperature (LST)** — the temperature of the ground surface as seen from space. What matters for human health, building energy demand, and urban planning is **air temperature** — the temperature of the air people breathe at ~2m height.

The relationship between LST and air temperature is nonlinear, spatially variable, and time-of-day dependent. LST can be 15–30°C higher than air temperature on a sunny day over asphalt. The gap is smaller over vegetated surfaces. It varies with wind speed. It collapses at night.

SPARC predicts air temperature directly from satellite-derived land cover inputs, calibrated against air temperature observations. This is fundamentally different from — and more useful than — any existing satellite LST product. The MODIS LST product (MOD11A1) is the current standard for satellite-based urban heat analysis. It measures the wrong thing. SPARC predicts the right thing.

This distinction is the headline for any publication arising from Track D. It is also the reason Track D is scientifically non-trivial: going from satellite inputs to air temperature requires the PDE physics to bridge the gap between surface and air temperature, and it requires the transfer learning infrastructure to generalize that bridge across climate zones.

---

## Full Track D Sequencing

```
Track A complete (PDE physics)
Track B complete (transfer learning, ≥2 cities)
Track C complete (registry, ≥4 cities)
        ↓
D1   Satellite ingestion pipeline
        ↓
D2   Climate zone conditioning          ← parallel with D3
D3   Weather station anchor integration ← parallel with D2
        ↓
D4   Zero-shot inference engine
        ↓
D5   Few-shot fine-tuning
        ↓
D6   Global coverage validation         ← requires ≥8 cities, ≥4 climate zones
        ↓
D7   Operational inference interface
```

---

## The Global Public Health Case

Urban heat is the leading cause of weather-related mortality globally. The burden falls disproportionately on cities in low- and middle-income countries — the cities least likely to have spatial temperature monitoring networks, and most likely to have rapid unplanned urban expansion that increases heat exposure.

Right now, a meaningful urban heat analysis requires a ground monitoring network. That costs tens of thousands of dollars and weeks of fieldwork. Most cities can't afford it.

SPARC at Track D completion means any city with Sentinel-2 coverage — which is the entire land surface of Earth, every 5 days, for free — can get a probabilistic spatial air temperature analysis from a single daily station reading and satellite imagery. The uncertainty is higher than a calibrated analysis, but it is honest, quantified, and physically grounded.

That is a global equity tool. The cities that need urban heat analysis most are exactly the cities that Track D makes newly accessible. That is worth building.

---

## R² and Performance Progression

| Stage | Mode | Expected RMSE | Expected R² |
|-------|------|--------------|-------------|
| Track A–C complete | Calibrated (full network) | ~0.8°C | 0.90+ |
| Track D4 | Zero-shot (1 station) | ~1.8°C | 0.70–0.75 |
| Track D5 | Few-shot (5 stations) | ~1.1°C | 0.82–0.87 |
| Track D5 | Few-shot (10 stations) | ~0.9°C | 0.88–0.92 |
| Long-term (12+ cities) | Zero-shot | ~1.3°C | 0.78–0.83 |

The zero-shot R² will improve as more cities are added to the registry — the trunk gets better at the satellite→temperature mapping with each new climate zone it sees. The few-shot R² converges toward calibrated performance as the number of stations increases, following a predictable diminishing returns curve that the model can quantify in advance.