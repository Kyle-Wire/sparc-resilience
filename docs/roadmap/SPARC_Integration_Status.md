# SPARC Integration Status

**SPARC Labs LLC | May 2026**
**Engineering Reference — What Is Built, What Is Wired, What Is Stubbed**

---

## Overview

This document provides a precise, file-level accounting of every advanced capability in SPARC's architecture. Status is one of three categories:

| Status | Meaning |
|---|---|
| ✅ **Built + Wired** | Module exists, is called from the main pipeline, and produces output in a normal `sparc run` |
| ⚠️ **Built + Not Wired** | Module is fully implemented and tested in isolation, but the pipeline does not yet invoke it |
| 🔲 **Stub Only** | Interface and data structures defined, implementation deferred |

---

## Status Table

### Core Pipeline (Stage 0–4)

| Capability | Module | Status | Notes |
|---|---|---|---|
| Bayesian Matérn correlogram | `sparc/run/correlogram_runner.py` | ✅ Built + Wired | Fully active in Stage 0 |
| Moran's I autocorrelation | `sparc/run/correlogram_runner.py` | ✅ Built + Wired | |
| Anisotropy detection | `sparc/run/correlogram_runner.py` | ✅ Built + Wired | Auto-wires KernelField |
| KernelField configuration | `sparc/models/` | ✅ Built + Wired | |
| GWEN variable selection | `sparc/run/gwen_runner.py` | ✅ Built + Wired | Stage 1, optional |
| Spatial block cross-validation | `sparc/run/spatial_cv_runner.py` | ✅ Built + Wired | Stage 2 |
| OLS / GWR / GWRF / GAM base models | `sparc/models/` | ✅ Built + Wired | Stage 2 base models |
| `SPARCMetaLearner` (SharedTrunk + CityHead) | `sparc/models/neural_meta.py` | ✅ Built + Wired | |
| SIREN physics encoder | `sparc/models/spatial_attention.py` | ✅ Built + Wired | Part of SharedTrunk |
| `ProcessRateNet` (α(x) field) | `sparc/models/neural_meta.py` | ✅ Built + Wired | Spatially-varying diffusivity |
| SparseSpatialAttention | `sparc/models/spatial_attention.py` | ✅ Built + Wired | |
| MC-Dropout uncertainty | `sparc/models/neural_meta.py` | ✅ Built + Wired | |
| 10-term PDE curriculum loss | `sparc/physics/pde_loss.py` | ✅ Built + Wired | |
| Exceedance probability heads | `sparc/models/neural_meta.py` | ✅ Built + Wired | |
| Bayesian MGWR ensemble | `sparc/run/mgwr_runner.py` | ✅ Built + Wired | Stage 2 optional backend |
| MC³ DAG structure search | `sparc/causal/mc3.py` | ✅ Built + Wired | Stage 3 |
| NUTS edge posteriors | `sparc/causal/nuts.py` | ✅ Built + Wired | Stage 3 |
| DML / CATE | `sparc/causal/cate_validation.py` | ✅ Built + Wired | Stage 3 |
| Spatial CATE | `sparc/causal/spatial_cate.py` | ✅ Built + Wired | Stage 3 |
| DoWhy refutation suite | `sparc/causal/dag_definition.py` | ✅ Built + Wired | Stage 3 |
| E-values | `sparc/causal/sensitivity.py` | ✅ Built + Wired | Stage 3 |
| Causal PDP | `sparc/causal/causal_pdp.py` | ✅ Built + Wired | Stage 3 |
| Mediation analysis | `sparc/causal/mediation.py` | ✅ Built + Wired | Stage 3 |
| IV estimation | `sparc/causal/iv.py` | ✅ Built + Wired | Stage 3 |
| Panel data causal models | `sparc/causal/panel.py` | ✅ Built + Wired | Stage 3 |
| Interference / spillover | `sparc/causal/interference.py` | ✅ Built + Wired | Stage 3 |
| Divergence audit | `sparc/causal/divergence_audit.py` | ✅ Built + Wired | Stage 3 |
| 4-tier scenario simulator | `sparc/interventions/scenario_simulator.py` | ✅ Built + Wired | Stage 4 |
| Budget allocation optimizer | `sparc/scenario/` | ✅ Built + Wired | Stage 4 |
| Artifact store + manifest | `sparc/run/result_store.py` | ✅ Built + Wired | |
| Desktop app (Tauri + React) | `sparc-desktop/` | ✅ Built + Wired | |

---

### V3 Transfer & Continual Learning

| Capability | Module | Status | Gap Description |
|---|---|---|---|
| `train_cold_start()` | `sparc/run/transfer_training.py` | ✅ Built + Wired | |
| `train_warm_start()` | `sparc/run/transfer_training.py` | ✅ Built + Wired | |
| `train_warm_start_finetune()` | `sparc/run/transfer_training.py` | ✅ Built + Wired | |
| Transfer A/B validation | `sparc/run/transfer_validation.py` | ✅ Built + Wired | |
| `CityRegistry` (local) | `sparc/registry/city_registry.py` | ✅ Built + Wired | |
| Welford online scaler | `sparc/data/welford.py` | ✅ Built + Wired | |
| `compute_fisher_matrix()` | `sparc/training/ewc.py` | ✅ Built + Wired | |
| `ewc_penalty()` | `sparc/training/ewc.py` | ✅ Built + Wired | Computed but **not added to gradient step** |
| `CoresetSelector` (K-medoids) | `sparc/training/replay.py` | ✅ Built + Wired | |
| `compute_replay_loss()` | `sparc/training/replay.py` | ✅ Built + Wired | Computed but **not added to gradient step** |
| `train_continual()` orchestrator | `sparc/run/continual_training.py` | ✅ Built + Wired | Passes config, but training loop doesn't consume it |
| **EWC wired into epoch loop** | `sparc/training/v2_neural_training.py` | ⚠️ **Not Wired** | `loss += ewc_lambda * ewc_penalty(...)` line missing |
| **Replay wired into epoch loop** | `sparc/training/v2_neural_training.py` | ⚠️ **Not Wired** | `loss += replay_lambda * compute_replay_loss(...)` line missing |
| **Fisher + trunk extraction post-training** | `sparc/training/v2_neural_training.py` | ⚠️ **Not Wired** | `train_neural_meta()` doesn't return `fisher_matrix` or `trunk_state_dict` |
| CLI: `sparc transfer` | `sparc/__main__.py` | ✅ Built + Wired | |
| CLI: `sparc continual` | `sparc/__main__.py` | ✅ Built + Wired | Wired but training loop gap means EWC/replay are no-ops |

---

### Temporal / Spatio-Temporal Features

| Capability | Module | Status | Gap Description |
|---|---|---|---|
| `compute_diurnal_features()` | `sparc/data/temporal.py` | ✅ Built + Wired | Function exists |
| `get_snapshot_time_indices()` | `sparc/data/temporal.py` | ✅ Built + Wired | Function exists |
| 3-way time embedding (morning/midday/night) | `sparc/models/neural_meta.py` | ✅ Built + Wired | Architecture exists |
| Transient PDE terms (terms 9–10) | `sparc/physics/pde_loss.py` | ✅ Built + Wired | Loss terms exist |
| **Temporal features called in training** | `sparc/training/v2_neural_training.py` | ⚠️ **Not Wired** | `config["temporal"]["snapshots"]` is not checked; `compute_diurnal_features()` is never called from the training runner |
| **`time_idx` passed to `model.forward()`** | `sparc/training/v2_neural_training.py` | ⚠️ **Not Wired** | The temporal embedding path in the model is never activated |
| PCMCI+ spatio-temporal causal discovery | — | 🔲 **Stub Only** | Would extend `sparc/causal/panel.py`; no implementation |
| DYNOTEARS graph discovery on panels | — | 🔲 **Stub Only** | — |

---

### JEPA Self-Supervised Pretraining

| Capability | Module | Status | Gap Description |
|---|---|---|---|
| JEPA loss (VICReg: alignment + variance + covariance) | `sparc/training/jepa_loss.py` | ✅ Built + Wired | Full implementation |
| `jepa_curriculum_weight()` scheduler | `sparc/training/jepa_loss.py` | ✅ Built + Wired | |
| JEPA blend in causal stack | `sparc/interventions/causal_stack.py` | ⚠️ **Partially Wired** | Phase 2.f blend weight exists; coefficient γ defaults to 0 (inactive) |
| `EMATrunk` (momentum encoder) | `sparc/models/neural_meta.py` | ✅ Built + Wired | Architecture exists |
| `LatentPredictor` | `sparc/models/neural_meta.py` | ✅ Built + Wired | Architecture exists |
| **JEPA pretraining pass in training runner** | `sparc/training/v2_neural_training.py` | ⚠️ **Not Wired** | No pretraining phase before supervised training; `jepa_loss()` is never called from `train_neural_meta()` |

---

### Zero-Shot and Few-Shot Inference

| Capability | Module | Status | Gap Description |
|---|---|---|---|
| `ZeroShotPrediction` dataclass | `sparc/inference/zero_shot.py` | ✅ Built + Wired | Interface defined |
| `zero_shot_predict()` | `sparc/inference/zero_shot.py` | 🔲 **Stub Only** | Raises `NotImplementedError` |
| `few_shot_predict()` | `sparc/inference/few_shot.py` | 🔲 **Stub Only** | Interface defined; no implementation |
| Latent rollout inference | `sparc/inference/latent_rollout.py` | 🔲 **Stub Only** | |
| `SatelliteFeatureSet` dataclass | `sparc/data/satellite_types.py` | ✅ Built + Wired | Data structure defined |
| Satellite ingestion (Sentinel-2 / Landsat) | — | 🔲 **Stub Only** | `sparc/data/satellite_ingest.py` not yet created |
| ERA5 climate forcing fetch | — | 🔲 **Stub Only** | `sparc/data/era5_fetch.py` not yet created |
| Weather station anchor | — | 🔲 **Stub Only** | `sparc/data/station_anchor.py` not yet created |
| `ClimateZoneEncoder` (full) | `sparc/models/climate_encoder.py` | ⚠️ **Partially Wired** | 30-class Köppen embedding exists; richer ERA5/CHELSA conditioning not implemented |

---

### Central Registry (Cloud)

| Capability | Module | Status | Gap Description |
|---|---|---|---|
| Local city registry | `sparc/registry/city_registry.py` | ✅ Built + Wired | SQLite-backed, fully functional |
| Registry sync protocol | — | 🔲 **Stub Only** | `sparc/registry/sync.py` not yet created |
| Central registry server | — | 🔲 **Stub Only** | `sparc/registry/server.py` not yet created |
| Global trunk recomputation | — | 🔲 **Stub Only** | `sparc/registry/retrain.py` not yet created |
| CLI: `sparc push` / `sparc pull` | `sparc/__main__.py` | 🔲 **Stub Only** | Subcommands not yet registered |

---

### LLM-Assisted Tooling

| Capability | Module | Status | Gap Description |
|---|---|---|---|
| Claude-assisted project setup (desktop) | `sparc-desktop/` | ✅ Built + Wired | Claude API optional; guides config creation |
| LLM-assisted DAG construction | — | 🔲 **Stub Only** | No LLM call for proposing causal graph structure from domain text |

---

## Priority Wiring Gaps

The following four gaps have the highest leverage — resolving them unlocks multiple downstream capabilities:

### Gap 1 — EWC + Replay in Training Loop
**File:** `sparc/training/v2_neural_training.py`
**What's needed:** Inside the epoch loop in `train_neural_meta()`, check for `config["_continual"]` and add:
```python
if "_continual" in config:
    cont = config["_continual"]
    if cont["ewc_lambda"] > 0:
        loss += cont["ewc_lambda"] * ewc_penalty(model, cont["fisher_matrices"])
    if cont["replay_lambda"] > 0:
        loss += cont["replay_lambda"] * compute_replay_loss(model, cont["previous_coresets"])
```
**Unlocks:** Multi-city continual training, central registry, zero-shot prerequisite.

### Gap 2 — Temporal Features in Training Runner
**File:** `sparc/training/v2_neural_training.py`
**What's needed:** Check `config["temporal"]["snapshots"]`, call `compute_diurnal_features()` during data prep, build `time_idx` tensor, pass to `model.forward()` and `sparc_joint_loss()`.
**Unlocks:** Diurnal prediction, multi-snapshot analysis, transient PDE terms 9–10.

### Gap 3 — JEPA Pretraining Pass
**File:** `sparc/training/v2_neural_training.py`
**What's needed:** Before the supervised training loop, if `config["jepa"]["enabled"]` is true, run a pretraining phase where the EMA trunk generates targets for the online trunk via `jepa_loss()` with `jepa_curriculum_weight()` scheduling.
**Unlocks:** Richer trunk representations, better zero-shot generalization, self-supervised pretraining from satellite imagery.

### Gap 4 — Post-Training Fisher + Trunk Extraction
**File:** `sparc/training/v2_neural_training.py`
**What's needed:** After `train_neural_meta()` completes, compute and return `result["fisher_matrix"]` and `result["trunk_state_dict"]`. These are what `train_continual()` expects but currently receives as empty dicts.
**Unlocks:** Gap 1 becomes fully functional; city registry is populated with real artifacts.
