# SPARC Roadmap — Reference Guide

## ROI Scoring Guide

### User ROI Scale (1–5)

| Score | Meaning | Example |
|-------|---------|---------|
| **5** | Unlocks a capability that currently doesn't exist at all | Zero-shot prediction for data-poor cities |
| **4** | Saves hours per run or dramatically improves decision quality | Satellite auto-ingestion removes data prep burden |
| **3** | Meaningful improvement to existing workflow | Faster training, better uncertainty calibration |
| **2** | Nice-to-have improvement, marginal user impact | Additional output artifact format |
| **1** | Internal quality / tech debt, invisible to users | Code refactor, test coverage |

### Spatial World Model Lift Scale (1–5)

| Score | Meaning | Example |
|-------|---------|---------|
| **5** | Core new capability toward the zero-shot North Star | Zero-shot inference engine, global trunk registry |
| **4** | Significant advance in spatial generalization or cross-domain transfer | Cross-city continual learning wired end-to-end |
| **3** | Strengthens the spatial physics model or causal world model | Sheaf Laplacian MAUP guarantee, PDE identifiability |
| **2** | Incremental improvement to existing spatial reasoning | Better kernel fitting, additional domain template |
| **1** | No spatial WM relevance | UI polish, report export format |

### Effort Weights

| Label | Time | Weight |
|-------|------|--------|
| **S** (Small) | < 4 hours | 1 |
| **M** (Medium) | Half-day to 1 day | 2 |
| **L** (Large) | 2–5 days | 4 |
| **XL** (Extra Large) | 1+ weeks | 8 |

---

## Current SPARC Status Summary

### What is built AND wired (production-ready)
- Stage 0: Bayesian Matérn correlogram → KernelField auto-wiring
- Stage 2: SPARCMetaLearner (SharedTrunk + CityHead) with sparse spatial attention, MC-Dropout
- Stage 2: 10-term PDE curriculum loss (terms 1–10 including transient)
- Stage 2: Temporal feature embedding (time_embed_dim wired as of May 2026)
- Stage 2: JEPA pretraining pass (wired May 2026)
- Stage 3: MC³ DAG search + NUTS edge posteriors
- Stage 3: DML/CATE + SpatialCATEEstimator + BayesianSpatialCATE
- Stage 3: DoWhy refutation suite + E-values
- Stage 3: Wager 2025 causal gaps (10/10 wired)
- Stage 3: GP regression over CATE surface (wired May 2026)
- Stage 3: MAUP sensitivity analysis (wired May 2026)
- Stage 3: Fairness audit in mediation (wired May 2026)
- Stage 4: 4-tier scenario simulation + budget-constrained allocation + Pareto frontier
- EWC penalty: built and **wired** into training loop (May 2026)

### What is built but NOT yet wired / validated
- Experience Replay (`compute_replay_loss()`) — interface mismatch, requires 2-phase fix
- Transfer validation (Providence → Boston): requires real city data
- Continual learning 2-city validation loop

### What is NOT YET built (major roadmap items)
- Registry sync protocol (`sparc/registry/sync.py`)
- Central registry server
- Satellite ingestion pipeline (Sentinel-2, Landsat)
- ERA5 climate forcing ingestion
- Zero-shot inference engine (`sparc/inference/zero_shot.py`)
- Few-shot fine-tuning (`sparc/inference/few_shot.py`)
- Wasserstein trunk alignment (research derivative)
- Sheaf Laplacian MAUP loss (research derivative)

---

## Spatial World Model — Conceptual Framework

SPARC's path to becoming a spatial world model parallels how language models achieved zero-shot generalization:

| Language Model Capability | SPARC Analog | Status |
|--------------------------|--------------|--------|
| Shared token embedding across domains | SharedTrunk (SIREN + physics features) | Built |
| Pre-training on diverse corpora | JEPA self-supervised pretraining on city data | Wired |
| Continual learning without forgetting | EWC + experience replay | EWC wired; replay blocked |
| Zero-shot generalization | Zero-shot inference engine | Not built |
| Few-shot adaptation | Few-shot fine-tuning of CityHead | Not built |
| Scale-up with more data | Central registry + global trunk | Not built |
| Physics grounding (beyond language) | 10-term PDE curriculum + α-field | Built and wired |
| Causal reasoning (beyond prediction) | MC³ + NUTS + DML | Built and wired |

**SPARC's unique moat vs. language models:** physics constraints + causal structure. A language model produces plausible text. SPARC produces physically consistent, causally attributed spatial fields. This is the core differentiator that justifies the "world model" framing.

---

## End-User ROI Opportunities — By Persona

### Urban Planner / Climate Analyst
- **Highest ROI:** Scenario simulation is already production-quality. The bottleneck is data preparation. Satellite auto-ingestion (Phase 3) removes the single largest barrier to adoption.
- **Secondary ROI:** Better uncertainty communication in the desktop app (credible interval maps, not just point estimates).

### Academic Researcher
- **Highest ROI:** Publishable methodological novelty. Top candidates: Sheaf Laplacian (formal MAUP guarantee), PDE identifiability scores on causal edges, Wasserstein trunk alignment.
- **Secondary ROI:** Reproducible run exports for peer review (already partially implemented via artifact manifest).

### City Government / Resilience Officer
- **Highest ROI:** Equity audit (fairness_audit() is wired). The ROI unlock is surfacing this prominently in the desktop app and report exports.
- **Secondary ROI:** Budget allocation with explicit equity constraints (Gini coefficient already computed — needs UI prominence).

### Consultant / Domain Specialist
- **Highest ROI:** 13 domain templates + single project.yml removes setup time. The unlock is more templates and better template auto-configuration from satellite data.
- **Secondary ROI:** Multi-city comparison reports.

### Zero-Data City (Future Persona)
- **Highest ROI:** Zero-shot inference (Phase 4). This is the biggest market expansion opportunity and the North Star.
- **Dependency chain:** Phase 3 (satellite ingestion) → Phase 2 (registry) → Phase 4 (zero-shot) — sequential, cannot shortcut.

---

## Breakthrough Research Directions

Pulled from `docs/research/derivatives.md` and external literature:

### Tier A — High SWM Lift, Implementation Path Clear
1. **Wasserstein Trunk Alignment** — Replaces EWC quadratic penalty with optimal-transport geometry preservation. Better cross-city calibration. Builds on existing K-medoids coresets.
2. **Sheaf Laplacian MAUP Loss** — Formal multi-scale consistency. Publishable differentiator. Builds on existing cardinal neighbor graph.
3. **PDE-Residual Physical Plausibility in MC³** — Causal edges scored by physical plausibility. Dual-axis scoring: statistical + physical. Unique in causal inference literature.

### Tier B — High SWM Lift, Requires Infrastructure First
4. **Zero-Shot Inference Engine** — The North Star feature. Requires Phase 2 (registry) + Phase 3 (satellite ingestion) first.
5. **Cross-Climate Global Trunk** — Requires 8+ cities across 4 Köppen zones. Phase 2 registry infrastructure is the prerequisite.

### Tier C — Exploratory / Novel
6. **Spatial Diffusion Prior for CATE** — Use denoising diffusion to model the prior distribution of treatment effect surfaces. Analogous to how image diffusion models learned natural image priors; this would learn "what CATE surfaces look like" across cities.
7. **Neural Operator (FNO/DeepONet) PDE Surrogate** — Replace forward Poisson/advection-diffusion solve in Stage 4 Tier 2 with a trained Fourier Neural Operator. 1000× faster PDE solves; enables real-time scenario simulation.
8. **Causal Transportability via Do-Calculus** — Formal external validity: given a causal model for City A, what can be transported to City B? Pearl's do-calculus + Bareinboim transportability theory. Would make Stage 3 results formally generalizable.
