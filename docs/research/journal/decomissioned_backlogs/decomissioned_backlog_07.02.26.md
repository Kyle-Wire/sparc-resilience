# SPARC — Research Backlog (ARCHIVED 2026-07-02)

**Archived by:** research agent, 2026-07-03 session  
**Reason:** Fresh start for train22→train23 phase (head covariate shift + stochastic convergence investigation)  
**Replaced by:** `docs/research/backlog.md` (new)

---

## Archive Summary — What Was Open at Time of Archive

All items below were `[ ]` (not yet started) when this backlog was archived.
Completed items existed in the original but are omitted here — see git history or the full journal entries for implementation notes.

### Phase 1 Continuation

- [ ] **1.5 Transfer validation: Providence → Boston** — complexity: **medium**
  - Run `transfer_validation.py`; confirm warm-start R² ≥ cold-start R².

### JEPA Deep Integration

- [ ] **JD-1 Stage 2 OOF predictions as DML `model_y` nuisance** — complexity: **low** (~25 lines, 2 files)
  - Gap: `CounterfactualEngine._fit_edge_dml()` refits HGB from scratch; Stage 2 `oof_preds` are discarded.
  - Files: `sparc/run/v2_bayesian_causal.py`, `sparc/causal/counterfactual_engine.py`

- [ ] **JD-2 SpatialResidualizer — JEPA trunk residualization for DML treatment features** — complexity: **medium**
  - Files: new `sparc/causal/spatial_residualizer.py`, `sparc/run/v2_bayesian_causal.py`
  - Depends on: JD-3

- [ ] **JD-3 `jepa.enable: true` schema default** — complexity: **low**
  - File: `sparc/config/project_schema.json` — change default false → true

- [ ] **JD-4 Multi-step latent rollout (mode: latent)** — complexity: **medium**
  - File: `sparc/inference/latent_rollout.py`

- [ ] **JD-5 Multi-step re-encode with physics cascade table (mode: reencode)** — complexity: **medium**
  - Files: new `sparc/inference/feature_perturbation.py`, `sparc/inference/latent_rollout.py`, all 13 caps.yml files
  - Depends on: JD-4

- [ ] **Causal transportability: transport ATE from source to target city** — complexity: **high**
  - Files: new `sparc/causal/transportability.py`, optional call in `v2_bayesian_causal.py`

### GPU / CUDA Acceleration (remaining)

- [ ] **CU-9 CUDA Graph capture for epoch step** — complexity: **high** (CU-9a and CU-9b done; full end-to-end capture remains)
- [ ] **CU-10 Multi-GPU fold-parallel DDP** — complexity: **high** (CU-10a–e sub-tasks all done; end-to-end validation remains)

### Stage 3 Structure Learning Extensions

- [ ] **S3-11b Full Cholesky leapfrog for correlated posterior geometry** — complexity: **medium** (~80 lines, 1 file)
  - File: `sparc/causal/nuts.py` only
  - Depends on: S3-11 (diagonal mass matrix — done)

- [ ] **S3-14 SVGD-NOTEARS warm-start for MC³ parallel chains** — complexity: **medium** (~90 lines, 3 files)
  - Files: new `sparc/causal/dag_warm_start.py`, `sparc/causal/mc3.py`, `sparc/config/causal_defaults.py`

- [ ] **Sheaf diffusion confounder recovery from NUTS residuals** — complexity: **high** (blue-sky)
- [ ] **Normalizing flows over DAG posterior** — complexity: **high** (blue-sky)

### Continual Learning

- [ ] **Prov-2 Provenance-aware trunk transfer** — complexity: **medium** (~45 lines, 3 files)
  - Files: `sparc/data/versioning.py`, `sparc/registry/city_registry.py`, `sparc/run/transfer_training.py`
  - Depends on: P4-6 (step hashes — done), Wasserstein alignment (wired)

### Blue-Sky Candidates

- [ ] **BS-1 Energy-Based Coreset Selection** — complexity: **medium**
  - Files: `sparc/training/replay.py`, `sparc/run/continual_training.py`

- [ ] **BS-3 Spatial Foundation Model via Physics-Guided Mask Token Pretraining** — complexity: **high**
  - Files: `sparc/models/neural_meta.py`, `sparc/training/jepa_loss.py`, `sparc/run/v2_neural_training.py`

- [ ] **BS-4 Road Network Graph GNN for Directional Urban Heat Transport** — complexity: **high**
  - Files: new `sparc/data/road_network.py`, `sparc/models/spatial_attention.py`, `sparc/run/v2_neural_training.py`

---

## Completed Items Summary (last sprint)

Session 2026-07-02 completed:
- D1: Extract preprocessing pipeline to `sparc/data/preprocessing.py`
- E1: Extract ScenarioExecutor to `sparc/scenario/executor.py`

Session 2026-06-09 completed:
- C1–C5: Data collection architecture (adapters, CollectSession, assembler callback, app.py wiring)

Session 2026-06-02 completed:
- Candidate A (gate feedback EMA), B (CityReplayState / replay loss), C (alpha-class curriculum), D (LambdaOptimizer meta-lambda), E (FoldTrainer facade)
- 1.1-b replay interface redesign resolved

See full journal entries in `docs/research/journal/` for implementation details.
