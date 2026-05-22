# SPARC — Intellectual DNA

**Updated:** May 2026

This document is the canonical reference for SPARC's intellectual identity. The research agent reads this first on every session to orient itself before identifying gaps or cross-pollination opportunities.

---

## What SPARC Is

SPARC is a physics-constrained spatial causal inference engine. It takes environmental or infrastructure spatial observations and produces:
1. Probabilistic spatial predictions (where does this phenomenon occur, and with what uncertainty?)
2. Causal attribution (what variables *cause* it, with posterior distributions over effect sizes?)
3. Counterfactual scenarios (what would change if we intervened, with budget constraints?)

The distinguishing claim: **prediction and causal inference are unified under shared physics constraints**, not bolted together post-hoc.

---

## The Five-Stage Pipeline

```
Stage 0   Correlogram Analysis
          Moran's I + Bayesian Matérn + anisotropy detection
          → auto-wired KernelField (bandwidth, kernel type, anisotropy angle)

Stage 1   GWEN Variable Selection  (optional)
          Geographically-weighted elastic net
          → spatially-varying feature importance maps

Stage 2   Spatial CV + Neural Meta-Learner
          4 base models (OLS, GWR, GWRF, GAM) under spatial block cross-validation
          → SPARCMetaLearner (SharedTrunk + CityHead) with PDE curriculum loss

Stage 3   Bayesian Causal Inference
          MC³ DAG search + NUTS edge posteriors + DML/CATE + DoWhy refutations
          → causal effect posteriors with E-values and sensitivity analysis

Stage 4   Physics-Constrained Scenario Simulation
          4-tier delta computation + budget-optimized allocation
          → counterfactual scenario maps with propagated uncertainty
```

---

## Core Technical Components

### SPARCMetaLearner (sparc/models/neural_meta.py)
- **SharedTrunk:** SIREN-based (sinusoidal representation networks) physics encoder. Ingests physics features through a learnable spatially-varying process-rate field α(x) produced by `ProcessRateNet`. Portable across cities and domains.
- **CityHead:** Lightweight city-specific output layer. Only this is retrained per deployment; the trunk transfers.
- **SparseSpatialAttention:** Spatial attention over observation locations. Bandwidth seeded from Stage 0 correlogram (Stage0→Stage2 circular loop).
- **MC-Dropout:** Posterior mean + credible interval maps via Monte Carlo sampling.
- **10-term PDE curriculum loss** (`sparc/physics/pde_loss.py`): Physics terms activated progressively — smoothness and energy balance first, then transient and cross-domain terms. Prevents physics from overwhelming data-driven signal early in training.

### Spatial Statistics Stack
- **Bayesian Matérn correlogram** (Stage 0): Fits ν, ρ, σ² from empirical semi-variogram. Auto-detects anisotropy (geometric + zonal).
- **Moran's I:** Global and local spatial autocorrelation diagnostics.
- **GWR / GWRF / GAM / MGWR:** Geographically-weighted base models capturing spatial nonstationarity.
- **MAUP awareness:** Spatially-continuous prediction is structurally resistant to areal aggregation artifacts.

### Bayesian Causal Inference Stack (Stage 3)
- **MC³:** Markov Chain Monte Carlo over DAG space. Produces a posterior distribution over graph structure.
- **NUTS:** No-U-Turn Sampler over edge weights in the winning DAG structure.
- **DML / CATE / Spatial CATE:** Double Machine Learning for debiased causal effects; spatially-varying CATE surfaces.
- **DoWhy refutation suite:** Placebo, random confounder, data subset, unobserved confounding — all four must pass.
- **E-values + sensitivity analysis:** Every estimate is accompanied by an E-value quantifying unmeasured confounder strength required to reverse findings.
- **Mediation, IV, panel, interference modules:** All built and wired.

### Transfer and Continual Learning (V3 — partially wired)
- **SharedTrunk / CityHead architecture:** Physics knowledge transfers; only CityHead is retrained per city.
- **EWC (Elastic Weight Consolidation):** `compute_fisher_matrix()` + `ewc_penalty()` in `sparc/training/ewc.py`. *Built but not wired into the training loop epoch step.*
- **Experience Replay:** `CoresetSelector` (greedy K-medoids, 400 points) + `compute_replay_loss()` in `sparc/training/replay.py`. *Built but not wired.*
- **Welford online scaler:** `sparc/data/welford.py`. Incremental standardization for streaming city data.
- **CityRegistry:** `sparc/registry/city_registry.py`. Manages trunk checkpoints, Fisher matrices, coresets, Welford states.
- **Temporal features:** `compute_diurnal_features()`, 3-way time embedding (morning/midday/night), transient PDE terms 9–10. *Built but not wired into training runner.*
- **JEPA pretraining:** EMA trunk momentum encoder + `jepa_loss()`. *Phase 1.3 — not yet implemented.*

### Artifact Infrastructure
- **Artifact store + manifest:** `sparc/registry/store.py` + `artifacts.db`. Every artifact written by every stage is registered; the desktop app resolves all files through this registry.
- **`master.gpkg`:** All spatial layers merged into a single GeoPackage at run end.

---

## Research Frontiers (from SPARC_Future_Roadmap.md)

### Near-Term (Phase 1 — active work)
- Wire EWC + replay into the training loop (`v2_neural_training.py`)
- Wire temporal features and time embedding into training runner
- JEPA self-supervised pretraining pass before supervised training
- Post-training artifact extraction (Fisher matrix + trunk state dict)
- Transfer validation: Providence → Boston
- Continual learning validation: 2-city end-to-end

### Near-Term (Phase 5 + Phase 6 emerging)
- LLM-assisted DAG construction (desktop setup wizard)
- PCMCI+ / DYNOTEARS for spatio-temporal causal discovery (`tigramite` already in deps)
- Spatial CATE surface with GP regression over treatment effects
- Physical identifiability: PDE residuals as causal constraint in MC³ search

### Medium-Term
- Causal representation learning: identifiability regularizer on SharedTrunk (Schölkopf et al.)
- Transportability / do-calculus: `transport_causal_estimate()` between cities
- Causal fairness: path-specific effect decomposition in `mediation.py` + `fairness_audit()`
- Satellite ingestion pipeline (Sentinel-2, Landsat, ERA5)
- Central model registry with privacy-preserving sync

### Long-Horizon
- Zero-shot spatial inference: `sparc predict --city "Lagos, Nigeria"`
- MAUP-aware causal estimation: `maup_sensitivity_analysis()` diagnostic
- Causal bandits: sequential intervention design with the scenario simulator as reward model
- Neural causal models (DCDI/ENCO) for high-dimensional DAG learning

---

## Adjacent Fields to Cross-Pollinate

These have not been fully exploited in SPARC and are worth exploring each session:

| Field | Connection to SPARC |
|---|---|
| **Riemannian geometry** | Anisotropic spatial weighting as geodesic distance on a Riemannian manifold |
| **Spectral graph theory** | Laplacian eigenmaps already in SPARC; Chebyshev polynomials for irregular coastal geometry |
| **Optimal transport** | Wasserstein distance for comparing spatial distributions across cities/scenarios |
| **Information geometry** | Fisher information already computed for EWC; natural gradient updates for spatial optimization |
| **Topological data analysis (TDA)** | Persistent homology of flood connectivity, heat island topology |
| **Diffusion maps** | Nonlinear dimensionality reduction of hazard fields; alternative to Laplacian PCA |
| **Sheaf theory** | Cellular sheaves over spatial graphs for multi-scale consistency constraints |
| **Causal representation learning** | Schölkopf et al. identifiable latent causal models; disentangled trunk dimensions |
| **Neural operators (FNO, DeepONet)** | PDE solution operators as alternatives to SIREN for physics encoding |
| **Score-based / diffusion generative models** | Uncertainty-aware scenario generation; posterior sampling over spatial fields |
