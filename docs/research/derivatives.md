# SPARC — Novel Research Derivatives

**Last updated by research agent:** *(not yet run)*

This file is maintained by the research agent. Each session, the agent appends its top 3–5 novel cross-pollination ideas here. The synthesis agent reads this file and converts entries into concrete proposals in `backlog.md`.

---

## Format

Each entry follows this structure:

```
### [Short title]
**Source fields:** [SPARC component] × [adjacent field]
**Core idea:** One paragraph description of the novel connection.
**Potential impact:** methodological / performance / interpretability / new capability
**Relevant SPARC modules:** list of files
**Status:** new | under-synthesis | in-backlog | implemented
```

---

## Derivatives

---

### Wasserstein Trunk Alignment for Cross-City Continual Learning
**Source fields:** `sparc/training/ewc.py` × Optimal Transport (Villani 2008; Peyré & Cuturi 2019)
**Core idea:** EWC penalizes *parameter* changes to prevent catastrophic forgetting between cities. An optimal-transport alternative penalizes *distributional* divergence between trunk activation distributions: `λ_OT * W₂(trunk_activations_B, coreset_activations_A)` replaces the EWC quadratic penalty. This preserves the latent *geometry* (not just parameter magnitude) across cities, producing better-calibrated uncertainty on held-out cities. The existing K-medoids coresets in `CityRegistry` are already representative 400-point samples — natural empirical distributions for Wasserstein comparison.
**Potential impact:** methodological (principled latent-space geometry preservation), performance (better zero-shot calibration), new capability (cross-climate-zone transfer with formal geometry guarantee)
**Relevant SPARC modules:** `sparc/training/ewc.py`, `sparc/training/replay.py`, `sparc/registry/city_registry.py`
**Status:** in-backlog

---

### Sheaf Laplacian for Multi-Scale MAUP-Resistant Spatial Consistency
**Source fields:** `sparc/physics/pde_operators.py` × Sheaf theory / topological signal processing (Hansen & Ghrist 2020; Bodnar et al. 2022)
**Core idea:** A cellular sheaf over the spatial graph formalizes the multi-scale consistency condition: predictions at finer resolution must aggregate consistently to coarser resolutions. The sheaf coboundary operator becomes a new physics loss term — `sheaf_restriction_loss()` — that penalizes scale-inconsistent predictions. This gives SPARC a formal MAUP-resistance guarantee rather than a structural claim. The cardinal-neighbor graph already built in `_build_cardinal_neighbors()` is the natural sheaf base space; the existing PDE Laplacian generalizes directly to a sheaf Laplacian.
**Potential impact:** methodological (formal MAUP guarantee — unique SPARC differentiator), interpretability (multi-scale decomposition), new capability (publishable MAUP robustness score)
**Relevant SPARC modules:** `sparc/physics/pde_operators.py`, `sparc/run/v2_neural_training.py` (`_build_cardinal_neighbors`), `sparc/causal/spatial_cate.py`
**Status:** in-backlog

---

### Physical Identifiability: PDE Residuals as Causal Plausibility Scores in MC³
**Source fields:** `sparc/causal/mc3.py` × `sparc/physics/pde_loss.py` × Physics-informed causal inference (Peters et al. 2017; Mooij et al. 2020)
**Core idea:** MC³ currently scores candidate DAGs by data likelihood (BIC) alone. PDE residuals computed by the neural meta-learner can act as a second scoring axis: DAGs whose implied spatial fields violate the governing equation are physically implausible and can be penalized or ruled out without additional data. A new `sparc/causal/pde_identifiability.py` module computes `physical_plausibility_score(dag, surrogate, physics_feats)` and a `lambda_pde`-scaled term is added to the MC³ log-score. Every causal edge posterior then carries a "physical plausibility" co-score.
**Potential impact:** methodological (causal claims grounded in physical law, not just statistical fit), interpretability (per-edge physical plausibility), new capability (publishable formal identifiability result)
**Relevant SPARC modules:** `sparc/causal/mc3.py`, `sparc/physics/pde_loss.py`, new `sparc/causal/pde_identifiability.py`
**Status:** in-backlog

---

### GP Regression for Continuous CATE Surface with Stage-0 Matérn Kernel
**Source fields:** `sparc/causal/spatial_cate.py` × `sparc/run/correlogram_matern_fit.py` × Gaussian Process regression (Rasmussen & Williams 2006)
**Core idea:** SPARC produces discrete CATE estimates at observed locations. A GP with Matérn-5/2 kernel — seeded from the Stage 0 ν and ρ parameters — interpolates a continuous CATE surface with proper posterior uncertainty. Credible intervals widen appropriately in data-sparse areas. A Moran's I test on GP residuals verifies that spatial autocorrelation in treatment effects has been captured. The Stage 0 kernel parameters are already stored in the artifact store; `_load_kernel_field()` fetches them.
**Potential impact:** performance (continuous uncertainty-quantified CATE maps), interpretability (neighborhood-level intervention response with CIs), new capability (spatial heterogeneity in causal effects as a first-class output)
**Relevant SPARC modules:** `sparc/causal/spatial_cate.py`, `sparc/run/correlogram_matern_fit.py`, `sparc/registry/store.py`
**Status:** in-backlog  *(duplicate of existing backlog item "GP regression over CATE surface" — no new proposal added)*

---

### Minimum Effective Range (MER) Self-Supervised Block Size for Spatial CV
**Source fields:** `sparc/run/enhanced_spatial_cv.py` × Stage 0 `effective_range_matrix` × blockCV (Valavi et al. 2019)
**Core idea:** Stage 2 block size is currently derived from the spatial autocorrelation range of the labeled target `y`. The Stage 0 Matérn fit already produces an `effective_range_matrix` with per-variable-pair spatial ranges stored in the artifact store. The minimum effective range (MER) — the 10th-percentile of the range distribution across all predictor pairs — is the most conservative self-supervised block size: any larger block will decorrelate all variable pairs without requiring a single label. Adding a 12-line read of `effective_range_matrix` inside `get_block_size_from_config()` removes the only label dependency from fold construction.
**Potential impact:** methodological (label-free CV fold construction), new capability (Stage 2 runs without any ground truth, prerequisite for Phase 3 zero-shot mode)
**Relevant SPARC modules:** `sparc/run/enhanced_spatial_cv.py` (`get_block_size_from_config`, `spatial_kfold_enhanced`), Stage 0 artifact `effective_range_matrix`
**Status:** in-backlog

---

### OOF Spatial Intelligence Extraction as JEPA Pretrain Seed
**Source fields:** `sparc/run/enhanced_spatial_cv.py` (`EXTRACT_OOF_INTELLIGENCE` flag) × `sparc/training/jepa_loss.py` × GWR local coefficients (Fotheringham et al. 2002)
**Core idea:** GWR's spatially-varying local β-coefficients (one vector per test location per fold) encode *how the spatial process varies across the domain* — without using the label magnitude as a value, only as a fitting signal. These β-maps are a self-supervised spatial representation: the JEPA trunk should learn to predict β-map structure in masked geographic regions from unmasked context. `EXTRACT_OOF_INTELLIGENCE = False` with a wired but absent `oof_extraction_hooks.py` is the current blocking gap. Implementing the module (~80 lines) and flipping the flag creates a feature bank of Stage 2 spatial patterns that seed the JEPA pretrain trunk, closing the Stage 2 → JEPA loop that currently only runs in one direction (JEPA → Stage 2).
**Potential impact:** performance (JEPA trunk seeded by real spatial process structure, not random masks), methodological (GWR β-maps become first-class spatial representation artifacts), new capability (closes Stage 2 → neural meta-learner feedback loop)
**Relevant SPARC modules:** `sparc/run/enhanced_spatial_cv.py`, new `sparc/run/oof_extraction_hooks.py`, `sparc/run/v2_neural_training.py` (JEPA pretrain phase)
**Status:** in-backlog

---

### Spatial Contrastive Learning for Cross-City CV Block Invariance
**Source fields:** `sparc/registry/city_registry.py` × SimCLR contrastive learning (Chen et al. 2020) × `sparc/models/kernel_field.py`
**Core idea:** In multi-city continual learning, two geographic blocks with similar land cover (dense urban core in different cities) should be close in the model's representation space; blocks with dissimilar land cover should be far apart. A spatial contrastive loss `L_NCE(z_block_A, z_block_B, negative_blocks)` — applied as a Stage 2 self-supervised pretext — trains a block encoder that is content-invariant to geography but sensitive to process type. Positive pairs are defined by KernelField bandwidth similarity; negative pairs are drawn from different bandwidth clusters. The encoder shares the JEPA trunk. Applied across cities via the CityRegistry coresets, this creates base model initializations that generalize to unseen cities without any label alignment.
**Potential impact:** new capability (cross-city Stage 2 warm-start), performance (better few-shot generalization), feeds Phase 2 Central Registry (contrastive alignment becomes the federated aggregation mechanism)
**Relevant SPARC modules:** `sparc/registry/city_registry.py` (coresets), `sparc/models/kernel_field.py` (positive pair definition), `sparc/training/jepa_loss.py` (loss infrastructure), `sparc/run/enhanced_spatial_cv.py`
**Status:** in-backlog

---

### `torch.compile` Epoch Step for Kernel Fusion (E-Perf-A)
**Source fields:** `sparc/run/v2_neural_training.py` (epoch loop) × PyTorch 2.0 compilation (Ansel et al. 2024)
**Core idea:** The training epoch dispatches 3 sequential surrogate forwards + SPARCMetaLearner + PDE loss as separate Python calls. Wrapping a `_epoch_step(batch_dict) -> loss` function with `torch.compile(mode="reduce-overhead")` fuses GELU/LayerNorm pairs, eliminates Python dispatch cost, and produces a single GPU kernel for the `cat → fusion → regression_head` chain. Prerequisite: `_remap_indices_to_local` must be made numpy-free (replaced with `torch.zeros().scatter_()`) before graph capture works. Gap E1 (vectorize local_map build) is a necessary first step.
**Potential impact:** performance (1.5–3× epoch throughput on CPU, 3–5× on GPU/MPS through kernel fusion)
**Relevant SPARC modules:** `sparc/run/v2_neural_training.py` (epoch loop), `sparc/models/neural_meta.py`, `sparc/run/v2_neural_training.py:_forward_surrogates`
**Status:** in-backlog

---

### Precomputed Sparse Laplacian Matrix for PDE Loss (E-Perf-B)
**Source fields:** `sparc/physics/pde_operators.py` × Sparse linear algebra (LeVeque 2007)
**Core idea:** `compute_pde_loss` reconstructs the Laplacian `∇²T` via 4 uncoalesced index-gather ops per evaluation (N/S/E/W gathers from the `(N,4)` cardinal index). The same Laplacian operator can be precomputed once as a `torch.sparse_coo_tensor L ∈ R^(N×N)` (at most 4N non-zeros, built from `_build_cardinal_neighbors` output which is constant across epochs). `lap_T = L @ T_pred` is then a single GPU-compatible sparse matmul replacing the per-epoch gather pattern. The sparse matrix is built once in `_prepare_tensors` and reused across all epochs and all three training loops.
**Potential impact:** performance (single sparse matmul replaces 4 gathers per epoch per training loop; GPU-friendly; particularly valuable for N > 5 000)
**Relevant SPARC modules:** `sparc/physics/pde_operators.py` (`laplacian()`), `sparc/physics/pde_loss.py`, `sparc/run/v2_neural_training.py` (`_prepare_tensors`, `_build_cardinal_neighbors`)
**Status:** in-backlog

---

### Variational Spatial Block Autoencoder (VSBA) for Label-Free Fold Quality Scoring
**Source fields:** `sparc/models/neural_meta.py` (`ProcessRateNet`, `EMATrunk`) × VAE (Kingma & Welling 2013) × Matérn GP prior
**Core idea:** A VAE with a Matérn GP prior on the latent space — trained on the feature distribution of each CV block — learns a posterior over spatial fields. The ELBO evaluated on a held-out block, conditioned on the training blocks' posterior, provides a label-free fold quality score: "how likely is this test block under the spatial field model trained on context?" High ELBO = in-distribution; low ELBO = OOD. The Stage 0 Matérn kernel parameters are the GP prior, coupling the self-supervised fold score to the same spatial covariance structure as the correlogram. The `ProcessRateNet` α(x) field is a natural spatially-varying latent for the encoder; the Laplacian prior on latent space is already computed in `pde_operators.py`.
**Potential impact:** methodological (calibrated label-free fold quality, adapts to spatial autocorrelation of domain), new capability (honest generalization estimate for zero-shot regions), feeds Phase 4 (label-free confidence in zero-shot predictions)
**Relevant SPARC modules:** `sparc/models/neural_meta.py`, `sparc/physics/pde_operators.py`, `sparc/run/enhanced_spatial_cv.py`, Stage 0 Matérn artifacts
**Status:** in-backlog

---

### Streaming Step-Level Data Provenance for Reproducibility
**Source fields:** `sparc/data/data_utils.py` preprocessing pipeline × data provenance research (Buneman et al. 2001; Halevy 2001) × MLflow-style experiment tracking
**Core idea:** The 8-step CSV preprocessing pipeline (`/data/preprocess`) transforms a dataframe through sequential stages. If each SSE event also emits a SHA-256 hash of the output dataframe at that step, the artifact store can record a full transformation DAG: `raw_sha256 → after_reproject_sha256 → ... → final_sha256`. This is zero user effort — hashing `pd.util.hash_pandas_object(df).sum()` after each step is ~1 ms per call. The lineage record enables: (1) cross-machine reproducibility verification — same raw file + config must produce same final hash; (2) upstream-change detection — if the raw CSV is replaced, a hash mismatch at step 1 triggers a re-run prompt; (3) provenance-aware transfer learning — if two cities share identical hashes through step 4, their post-imputation representations are directly comparable without re-alignment. The step hashes can also serve as cache keys: if step 3 hash matches a prior run, steps 4–8 can be loaded from cache rather than re-executed.
**Potential impact:** interpretability (full data lineage in the UI with zero effort), new capability (cross-machine reproducibility verification, upstream-change detection), methodological (provenance-aware transfer — cities with matching preprocessing genealogy can share trunk representations without re-alignment)
**Relevant SPARC modules:** `sparc/data/data_utils.py`, `sparc/data/versioning.py`, `sparc/registry/store.py`, `sparc/server/app.py` (new `/data/preprocess` endpoint), `sparc-desktop/src/components/pages/ProcessingPage.tsx`
**Status:** new
