# PRD: SPARC Data Collection Page

**Date:** 2026-05-21
**Status:** Approved — ready for implementation
**Scope:** `sparc-desktop` (React/TypeScript), `sparc/server` (FastAPI sidecar), `sparc/data/collect` (new Python module)

---

## Problem Statement

SPARC researchers working on urban heat and other geospatial domains currently have no guided path from "study area boundary" to "model-ready spatial dataset." The existing Processing page requires researchers to pre-assemble raster files manually, brings no domain-aware variable defaults, and produces a CSV that discards geometry. There is no way to confirm spatial data quality visually before committing to a pipeline run. Researchers waste hours sourcing, harmonizing, and debugging datasets before the first model ever trains.

---

## Solution

Replace the existing Processing page with a unified **Data Collection** page that guides researchers through five steps: define the study boundary, configure the variable set and temporal window, fetch data from free public APIs (with local file fallback per variable), visually confirm spatial coverage and data gaps on a 30m grid, and export a GeoParquet that the pipeline ingests directly. The page auto-updates `project.yml` with the output path and target column, handing off cleanly to the Variables page for predictor selection.

---

## User Stories

1. As a UHI researcher, I want to type a city name and have the study boundary populated automatically, so that I can start a new project without hunting for shapefiles.
2. As a researcher, I want to upload a custom shapefile or GeoJSON as my study boundary, so that I can use non-standard or sub-city study areas.
3. As a researcher, I want to draw a polygon directly on the map, so that I can define an ad-hoc boundary quickly for exploratory work.
4. As a researcher, I want the UHI template to pre-populate the variable list with sensible defaults, so that I can start a heat analysis without knowing every available index.
5. As a researcher, I want to see all available spectral indices with plain-language descriptions, so that I can make an informed decision about which ones to include.
6. As a researcher, I want to configure a date range and season filter for Landsat acquisition, so that I capture only the relevant peak-heat period.
7. As a researcher, I want to set a cloud cover threshold (0–100%) for Landsat scene filtering, so that I exclude low-quality acquisitions.
8. As a researcher, I want to choose between single best scene, median composite, and multi-scene panel outputs, so that I get the temporal structure appropriate for my modeling approach.
9. As a researcher, I want each variable row in the manifest table to show its data source (API name), status (pending / fetching / complete / failed), and spatial coverage percentage, so that I can see at a glance whether my dataset is complete.
10. As a researcher, I want to override any variable's source from API to a local file I upload, so that I can substitute proprietary LiDAR or custom rasters where they exist.
11. As a researcher, I want Landsat LST and all 13 spectral indices (NDVI, NDBI, MNDWI, NDWI, SAVI, EVI, NDBaI, NBR, NDMI, UI, BSI, Albedo, LST) computed from the same scene download, so that I don't trigger redundant API calls.
12. As a researcher, I want NLCD impervious surface, canopy cover, and land cover fetched from the MRLC WCS automatically and clipped to my boundary, so that I don't need to download national rasters.
13. As a researcher, I want NOAA CAPA air temperature (morning, midday, night) fetched and aligned only to dates with a valid Landsat overpass in my study window, so that the temporal alignment is enforced automatically.
14. As a researcher, I want a diurnal air temperature variable (`diurnal_AAT` = afternoon CAPA minus night CAPA) computed automatically, so that heat stress timing is represented in my dataset.
15. As a researcher, I want ERA5 background temperature fetched from Open-Meteo and subtracted from CAPA air temperature to produce an `AAT_residual` target column, so that the model learns the local urban heat anomaly rather than the regional climate signal.
16. As a researcher, I want NOAA weather station observations included as soft loss anchor points (not predictor columns), so that the spatial interpolation is constrained by observed ground truth.
17. As a researcher, I want building footprints and heights fetched automatically using a fallback chain (LiDAR upload → Microsoft MLBuildings → OSHB → constant default), so that I always get the best available building height estimate for my study area.
18. As a researcher, I want Sky View Factor computed per 30m cell using the H:W ratio proxy from building footprints and heights, so that urban canyon geometry is represented without requiring a full DSM.
19. As a researcher, I want to optionally provide a DSM file to override the H:W SVF proxy with a true radiative SVF calculation, so that high-accuracy canyon geometry is available when LiDAR data is present.
20. As a researcher, I want HOLC redlining grades (A–D encoded as 4–1) joined to the grid as a numeric covariate, so that historic disinvestment patterns are included as a model input.
21. As a researcher, I want CDC SVI and EJScreen scores joined to the grid as numeric covariates, so that contemporary environmental justice conditions are represented.
22. As a researcher, I want HOLC polygons rendered as a toggleable overlay on the confirmation map, so that I can visually inspect the equity geography without it affecting the model inputs.
23. As a researcher, I want the confirmation map to show a choropleth of whichever variable row I have selected in the manifest table, so that I can inspect spatial distributions one variable at a time.
24. As a researcher, I want grid cells with one or more missing variable values rendered in a distinct gap color on the map, so that data coverage problems are immediately visible.
25. As a researcher, I want to click any 30m grid cell and see a popup listing all variable values for that cell, with missing values flagged clearly, so that I can diagnose exactly which sources failed for a given location.
26. As a researcher, I want a summary histogram shown alongside the map for the selected variable, so that I can check for implausible distributions before building the dataset.
27. As a researcher, I want a "Build Dataset" button that is locked until all required variables have a status of complete, so that I cannot accidentally run the pipeline with missing inputs.
28. As a researcher, I want the page to write a GeoParquet (named by temporal mode: `_composite`, `_panel`, or `_best_scene`) and update `project.yml` with the new file path and target column automatically, so that the pipeline picks up the new dataset without manual config edits.
29. As a researcher, I want the page to also write a `data_manifest.json` sidecar recording the source, version, fetch timestamp, and coverage percentage for every variable, so that I have a reproducibility record for the dataset.
30. As a researcher, I want predictor column selection to remain in the Variables page (GWEN), so that I can still use automated variable selection after the dataset is built.
31. As a researcher, I want the local file upload option still available for any variable, so that I can use pre-processed rasters or proprietary datasets without changing my workflow.
32. As a researcher working outside the US, I want the equity layer fetch to gracefully degrade (mark as optional/skipped) when HOLC or EJScreen coverage does not exist for my study area, so that the pipeline is not blocked on US-only datasets.

---

## Implementation Decisions

### Modules to build

**New: `sparc/data/collect/` package** — each sub-module is a deep module with a simple, testable interface. No sub-module imports another; the assembler composes them.

- **`boundary.py`** — accepts a place name string, a local vector file path, or a drawn GeoJSON geometry; returns a GeoDataFrame of the clipped boundary polygon in EPSG:4326. Place-name resolution uses the Census TIGER/Geocoder API (same base URL already used in `census_equity.py`).

- **`landsat.py`** — accepts boundary GeoDataFrame + date range + cloud threshold + temporal mode; queries USGS Landsat Collection 2 STAC, downloads COG tiles clipped to boundary, computes all 13 spectral indices and LST per scene, returns a GeoDataFrame at 30m resolution. Temporal mode determines whether the return is a single row-per-cell composite or a multi-date panel.

- **`nlcd.py`** — accepts boundary GeoDataFrame; fetches imperviousness, canopy cover, and land cover from the MRLC WCS; reclassifies land cover string codes to numeric; clips and reprojects to the 30m fishnet.

- **`capa.py`** — accepts boundary GeoDataFrame + list of Landsat overpass dates; fetches NOAA CAPA air temperature at morning, midday, and night windows on those specific dates only; spatially interpolates observations to the 30m grid; computes `diurnal_AAT` = midday minus night.

- **`era5.py`** — accepts boundary GeoDataFrame + date range; fetches hourly 2m temperature from Open-Meteo ERA5 (no API key); aggregates to daily means; bilinearly downscales from ~0.25° to the 30m grid; produces the background temperature field subtracted from CAPA to yield `AAT_residual`.

- **`buildings.py`** — accepts boundary GeoDataFrame; tries sources in order: (1) local LiDAR file path, (2) Microsoft MLBuildings Azure Blob tiles, (3) OSHB via Planetary Computer, (4) constant default from `project.yml`; returns building footprints GeoDataFrame with a height column. Computes H:W SVF proxy per 30m cell: `SVF = W / sqrt(W² + H²)` where W is cell width (30m) and H is mean building height in cell. Optionally accepts a DSM path to run true SVF instead.

- **`equity.py`** — accepts boundary GeoDataFrame; fetches HOLC polygons from Mapping Inequality API; fetches CDC SVI CSV; fetches EJScreen REST API; joins all three to the 30m grid. HOLC grades encoded A=4, B=3, C=2, D=1. Gracefully marks as skipped where coverage is absent.

- **`manifest.py`** — accepts a dict of variable fetch results; produces a `data_manifest.json` with `{variable, source, tier, fetch_timestamp, coverage_pct, status}` per entry. `coverage_pct` = fraction of grid cells with non-null value. Required variables with `coverage_pct < 1.0` are flagged as warnings; `coverage_pct = 0` is a blocking error.

- **`assembler.py`** — composes all fetch results onto the 30m fishnet; writes `{name}_{temporal_mode}.parquet` (GeoParquet) as the pipeline input; calls `manifest.py` to write the sidecar JSON; updates `project.yml` `data.file_path` and `data.target_column` in-place.

### Modifications to existing modules

- **`sparc/server/app.py`** — add new endpoint group `/collect/*`:
  - `POST /collect/boundary` — resolves boundary from place name / file / GeoJSON; returns GeoJSON + bbox
  - `POST /collect/fetch` — triggers async fetch for one variable; streams progress via SSE; returns when variable is complete
  - `GET /collect/manifest` — returns current manifest state (all variables, statuses, coverage %)
  - `GET /collect/preview/{variable}` — returns GeoJSON of the 30m grid colored by the specified variable (for map QA)
  - `POST /collect/build` — runs assembler; writes GeoParquet + manifest; updates project.yml; returns output paths
  - Existing `/data/fishnet` and `/data/zonal_stats` endpoints are retained unchanged for the local-file fallback path.

- **`sparc/config/config.py`** — accept `.parquet` / `.geoparquet` in addition to `.csv` for `data.file_path`; recognize `AAT_residual` as a valid target column; no breaking changes to existing config keys.

- **`sparc-desktop/src/components/pages/DataCollectionPage.tsx`** (replaces `ProcessingPage.tsx`) — five-step stepper:
  - Step 1: Boundary — place-name search field + local file picker + draw-on-map toggle
  - Step 2: Variables & Temporal — manifest table with per-row source selector; date range picker; season filter (month checkboxes); cloud cover slider; temporal mode radio (single / composite / panel)
  - Step 3: Fetch — progress view per variable with status icons; cancel/retry per row
  - Step 4: Confirm — full-width map (choropleth reacts to selected manifest row); histogram sidebar; HOLC boundary toggle; gap-color cells; cell-click popup listing all variable values
  - Step 5: Build — summary of coverage, cell count, date range, temporal mode; "Build Dataset" button (disabled until all required vars complete); on success: shows output file path

- **`sparc-desktop/src/components/data/VariableManifestTable.tsx`** (new) — controlled table component; each row: variable name, description tooltip, source badge (API name or "Local File"), status indicator, coverage %, source override dropdown.

- **`sparc-desktop/src/components/data/BoundarySelector.tsx`** (new) — tabs for place search / file upload / draw; emits an `onBoundaryChange` GeoJSON callback; renders boundary outline on map.

- **`sparc-desktop/src/components/data/CellInspector.tsx`** (new) — popup panel bound to map click; renders a two-column key/value list of all variable values for the clicked cell; missing values shown as `—` with a warning badge.

- **`sparc-desktop/src/components/map/SpatialMap.tsx`** — add `gapField` prop: when provided, cells where `gapField` is truthy render in the configured gap color (default `#ef4444`); add `onFeatureClick` prop already exists — wire it to the CellInspector.

- **`sparc-desktop/src/lib/api.ts`** — add typed wrappers for all `/collect/*` endpoints; add SSE stream helper for `/collect/fetch`.

- **`App.tsx`** — rename "Processing" nav item to "Data Collection"; route to `DataCollectionPage`.

### Data contracts

- The 30m fishnet grid is the canonical spatial unit. All fetched variables are zonal-aggregated or resampled to this grid before joining. Grid cells are identified by `cell_id` (integer) + `cell_x`, `cell_y` centroid coordinates in the project projected CRS.
- The output GeoParquet schema includes: `cell_id`, `geometry` (polygon), `cell_x`, `cell_y`, `AAT_residual` (target), all fetched predictor columns, `has_gap` (boolean: any null predictor). The pipeline reads predictors from the columns present in the file; `project.yml` `predictors` list is left empty and auto-populated by the Variables page.
- `data_manifest.json` schema: `{ "generated_at": ISO8601, "boundary": {...}, "temporal": {...}, "variables": [ { "name", "source_name", "source_tier", "fetch_timestamp", "coverage_pct", "status", "notes" } ] }`.
- The station anchor points are written to a separate `stations.parquet` alongside the main GeoParquet. The physics training loop reads this file when present and adds the soft station residual loss term.

### API data sources

| Variable group | Endpoint / dataset | Auth |
|---|---|---|
| Landsat spectral + LST | USGS Landsat C2 STAC (`landsatonaws.com`) | None |
| NLCD imperviousness, canopy, land cover | MRLC WCS (`mrlc.gov/geoserver/wcs`) | None |
| NOAA CAPA air temperature | NOAA CAPA reanalysis (public endpoint) | None |
| ERA5 background temperature | Open-Meteo historical API | None |
| NOAA weather stations | NOAA GHCN-Daily ISD | None |
| Building footprints + heights | Microsoft MLBuildings Azure Blob | None |
| OSHB building heights | Microsoft Planetary Computer STAC | None |
| HOLC redlining | Mapping Inequality API | None |
| CDC SVI | CDC static download (CSV) | None |
| EJScreen | EPA EJScreen REST API | None |

---

## Testing Decisions

A good test exercises the module's public interface — what it returns given specific inputs — not its internal steps. Tests should not mock deep internals; mock only at network/filesystem boundaries.

### Modules with tests

- **`boundary.py`** — given a known place name ("Providence, RI"), assert the returned GeoDataFrame has a valid polygon geometry covering the expected lat/lon range. Given a local GeoJSON file path, assert geometry passes through unmodified.
- **`manifest.py`** — given a dict of variable results with known coverage percentages, assert the manifest JSON contains the correct `status` values ("complete", "warning", "error") and that required variables with zero coverage raise a blocking error.
- **`assembler.py`** — given a pre-built dict of small synthetic GeoDataFrames (one per variable), assert the output GeoParquet contains the expected columns, has no duplicate `cell_id` values, and that `project.yml` `data.file_path` and `data.target_column` are updated correctly.
- **`era5.py`** — given a mocked Open-Meteo response, assert `AAT_residual` values are correctly computed as (CAPA obs) − (ERA5 background) at matching spatial cells.
- **`buildings.py`** — given a boundary with no local LiDAR and a mocked MLBuildings tile response, assert the fallback chain terminates at MLBuildings and SVF values are in [0, 1]. Given a boundary where MLBuildings returns empty, assert the chain falls through to OSHB.
- **`VariableManifestTable.tsx`** — given a manifest array with one failed and one pending variable, assert the "Build Dataset" button is absent (or disabled). Given all variables complete, assert the button is present and enabled.
- **`CellInspector.tsx`** — given a feature properties object with two null values, assert two warning badges render and the missing values display as `—`.

Prior art for server-side tests: `tests/test_artifact_store.py`, `tests/test_config.py`. Prior art for component tests: `tests/test_render.py`.

---

## Out of Scope

- MODIS-derived diurnal LST (`diurnal_LST`) — deferred; will be added as an optional variable once the CAPA-only diurnal path is validated.
- Stratified model runs by HOLC grade — HOLC is a covariate in this release; stratum-split analysis belongs in the Scenarios engine.
- Non-US equity layers (EU Urban Atlas, etc.) — architecture supports graceful skip; sourcing non-US layers is future work.
- True radiative SVF via UMEP/WhiteboxTools — the DSM override path is specified but the UMEP/WhiteboxTools integration is deferred pending dependency audit.
- Cloud hosting of the fetch pipeline — all fetches run on the local machine in this release.
- Tile-based streaming map (PMTiles / MVT) for very large grids — current GeoJSON-on-demand approach is sufficient for city-scale 30m grids; tiling is a follow-on optimization.
- ERA5 variables beyond 2m air temperature (wind, humidity) — scoped to temperature only in this release.
- Any changes to Stage 0–3 pipeline logic — the output GeoParquet is a drop-in for the existing `data.file_path` contract.

---

## Further Notes

- The `stations.parquet` soft-anchor mechanism is new to the physics training loop. The loss term `L_station = λ · Σ(ŷ_i - y_i)²` over station cells should be additive to the existing physics residual loss, with `λ` configurable in `physics/priors.yml` (default 0.1). This requires a small modification to the training engine but zero changes to the pipeline DAG.
- The existing `SatelliteBand` enum in `sparc/data/satellite_types.py` should be extended to include all 13 spectral indices as canonical band names. The `SatelliteFeatureSet` dataclass is the right container for Landsat fetch results.
- Land cover reclassification (NLCD string → numeric) should be implemented as a standalone lookup table in `nlcd.py` and unit-tested independently — this is a known source of silent errors in spatial pipelines.
- The `census_equity.py` module already handles Census API calls and caching. The boundary place-name lookup in `boundary.py` should reuse the same `_cache_dir()` and HTTP timeout patterns from that module rather than duplicating them.
- Waterbodies must be retained as a discrete land cover class (not masked) — they are thermally significant and their omission would bias LST and diurnal range estimates near rivers and coastlines.
