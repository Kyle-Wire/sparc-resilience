# SPARC — Novel Research Derivatives

**Last updated by research agent:** 2026-06-02
**Last synthesized by synthesis agent:** 2026-05-22c

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
**Status:** in-backlog  *(backlog: P4-7 — upstream CSV change detection; Prov-2 — provenance-aware trunk transfer)*

---

### Latent Diffusion for Posterior Scenario Sampling
**Source fields:** Score-based generative models (Song et al. 2021 *Score-Based Generative Modeling*; Ho et al. 2020 DDPM) × `sparc/models/neural_meta.py` JEPA trunk latent space × `sparc/interventions/scenario_simulator.py`
**Core idea:** The JEPA trunk encodes spatial physics observations into a latent field h(x) ∈ ℝ^{hidden_dim}. Currently the scenario simulator produces a single point-estimate counterfactual map. A lightweight DDPM conditioned on the physics features — 3 denoising UNet layers over the N×hidden_dim trunk latent field — would let SPARC *sample* a posterior over plausible counterfactual spatial fields. Each sample is a spatially coherent field that satisfies the PDE constraints implicit in the trunk's training. The result: instead of "cooling estimate: −1.2°C", SPARC reports "posterior mean −1.2°C, 90% credible interval [−2.1°C, −0.4°C]" with the full distribution available for risk analysis. The JEPA VICReg variance/covariance loss terms already regularize the trunk latent space to be near-Gaussian and non-collapsed — analogous to the LDM first-stage KL regularizer (Rombach et al. 2022). Given an observed h(x), add noise, train a UNet to denoise conditioned on physics features. A practical implementation would first project the trunk's 256-dim latent to a 32-dim bottleneck before diffusion. The existing `jepa_loss.py` VICReg terms provide the necessary latent regularization for free.
**Potential impact:** new capability (full posterior over scenarios — a major differentiator for insurance/planning use cases: uncertainty maps per scenario not just point estimates), methodological (first physics-constrained diffusion model for spatial urban intervention scenarios, publishable), interpretability (spatial credible interval maps for every intervention scenario)
**Relevant SPARC modules:** `sparc/models/neural_meta.py` (`EMATrunk.encode_target` as denoising target), `sparc/training/jepa_loss.py` (VICReg latent regularization), `sparc/interventions/scenario_simulator.py` (scenario conditioning), `sparc/physics/pde_loss.py` (physics constraint on generated samples)
**Status:** in-backlog

---

### CUDA Graph Capture for Zero-Overhead Epoch Step
**Source fields:** `s~parc/run/v2_neural_training.py` (epoch loop) × CUDA Graphs (NVIDIA 2020; PyTorch `torch.cuda.CUDAGraph`) × `_maybe_compile()` (already wired)
**Core idea:** `torch.compile` has already traced the computation graph. CUDA Graphs can go further: capture the entire forward + backward + optimizer step as a single replay-able GPU kernel sequence. The `torch.cuda.make_graphed_callables` API handles first-step warmup automatically. The spatial minibatch sampler produces fixed-size batches for all but the final batch, which is the prerequisite for CUDA Graph's static shape requirement. Captured graphs eliminate ~50 kernel launch overheads per batch step (~0.1–0.5 ms per batch on modern hardware). This is complementary to AMP (CU-2) — a CUDA Graph captures a BF16 autocast region equally well. The `_maybe_compile` pattern already in place is the natural insertion point for `make_graphed_callables` after compile, guarded by `torch.cuda.is_available() and hasattr(torch.cuda, "make_graphed_callables")`.
**Potential impact:** performance (15–30% additional throughput on A100/H100 on top of AMP + torch.compile; effectively free throughput with zero functional change), new capability (enables sub-millisecond batch processing for real-time city digital twin updates)
**Relevant SPARC modules:** `sparc/run/v2_neural_training.py` (`_maybe_compile`, epoch loop), `sparc/training/optimizer.py` (`training_step`)
**Status:** in-backlog

---

### Multi-GPU Fold-Parallel Training via `DistributedDataParallel`
**Source fields:** `sparc/run/v2_neural_training.py` (CV fold loop) × PyTorch `DistributedDataParallel` (DDP) × `HardwareProfile.gpu_count` (once CU-1 adds this field)
**Core idea:** Each CV fold is fully independent — no cross-fold communication during training, only OOF collection afterward. With K GPUs, fold `i` can be assigned to GPU `i`, giving K× training speedup for the CV phase. `torch.multiprocessing.spawn` with DDP init and one process per GPU follows the standard PyTorch multi-GPU pattern. The `gpu_count` field in `HardwareProfile` (once CU-1 is implemented) provides the worker count. For the full retrain, standard DDP with `batch_size * world_size` linear scaling. The spatial minibatch sampler needs a partitioned-N version for DDP (each rank processes a disjoint spatial partition — the natural partition is already geographic blocks from `spatial_kfold_enhanced`). EWC and OT alignment penalties need gradient reduction across ranks. The JEPA pretraining loop is also trivially data-parallel (same architecture, separate random masks per rank). This gives SPARC a clear path to K× training speedup for large cities (N > 10k) with zero accuracy change.
**Potential impact:** performance (K× training speedup for K-GPU machines; 2-GPU = 2× for free; enables running 10-fold CV in same wall-clock as 1-fold CV), new capability (enables training on large-N cities that are memory-constrained per GPU but feasible with tensor parallel split), methodological (unlock distributed training as a first-class pipeline mode)
**Relevant SPARC modules:** `sparc/run/v2_neural_training.py` (fold loop, full retrain), `sparc/config/hardware_profile.py` (`gpu_count` field from CU-1), `sparc/training/optimizer.py`, `sparc/training/ewc.py` (gradient reduction for EWC penalty)
**Status:** in-backlog

---

### Causal Bandits for Sequential Intervention Design
**Source fields:** Multi-armed bandits with Thompson sampling (Thompson 1933; Russo et al. 2018 *A Tutorial on Thompson Sampling*) × `sparc/decision/policy_learning.py` (`EmpiricalWelfareMaximizer`) × `sparc/causal/spatial_cate.py` (`CATEGPSurface`) × `sparc/interventions/scenario_simulator.py`
**Core idea:** `EmpiricalWelfareMaximizer` learns a one-shot policy: given AIPW scores, choose treatments to maximize welfare under budget. But urban heat interventions are sequential — planting canopy in year 1 changes the effective treatment response for impervious reduction in year 2 (interaction effects, carry-over). A Thompson sampling bandit treats each intervention type (canopy, albedo, impervious) as an arm. The posterior over arm rewards is the CATE GP surface from `spatial_cate.py` — a Matérn GP fitted with Stage 0 covariance parameters. At each "decision round" (year), the bandit samples from the CATE posterior via `sklearn.GaussianProcessRegressor.sample_y()`, executes the highest-posterior-sample arm, and updates the GP posterior with the observed spatial field change (simulated via `scenario_simulator.py` as a cheap oracle). After T rounds, the optimal multi-year intervention sequence is revealed. The infrastructure is nearly complete: add `sample_posterior()` to `CATEGPSurface` (~5 lines exposing `gpr.sample_y()`) and write a `CausalBandit` class (~80 lines). `EmpiricalWelfareMaximizer` becomes the one-shot baseline to beat. Provably sublinear regret (GP-UCB; Srinivas et al. 2012).
**Potential impact:** new capability (optimal multi-year intervention planning under uncertainty — directly relevant for municipal climate adaptation with budget constraints), methodological (bridges causal inference and sequential decision-making, publishable as GP-UCB / Thompson sampling over causal posteriors), performance (sublinear regret vs. greedy one-shot allocation from `EmpiricalWelfareMaximizer`)
**Relevant SPARC modules:** `sparc/causal/spatial_cate.py` (`CATEGPSurface` — add `sample_posterior()`), `sparc/decision/policy_learning.py` (`EmpiricalWelfareMaximizer` — one-shot baseline), `sparc/interventions/scenario_simulator.py` (bandit oracle), `sparc/causal/sensitivity.py` (E-values as risk bounds on arm selection)
**Status:** in-backlog

---

### tqdm-Based CLI Progress Layer with Dataset-Tier ETA

**Source fields:** `sparc/__main__.py` (cmd_run stage dispatch) × `sparc/run/v2_neural_training.py` (epoch loop) × tqdm (Casper da Costa-Luis 2016) × FastAI/PyTorch Lightning training callbacks

**Core idea:** The five-stage SPARC pipeline provides no user-visible progress feedback beyond one-line `print()` stage markers. For Stage 2 — the dominant wall-clock step (60–200 epochs × 3–5 folds = minutes to tens of minutes) — the epoch loop uses `logger.info()` which is invisible during a default `sparc run`. A minimal three-layer progress system uses only `tqdm` (already in requirements): (1) a pipeline-level bar wrapping all stage dispatches in `cmd_run` with overall `[N/6]` count and elapsed/remaining; (2) a fold-level bar inside `train_neural_meta` for the `enumerate(folds)` loop; (3) an epoch-level bar inside `_exec_cv_fold` replacing the bare `for epoch in range(n_epochs)` — populated with `loss`, `r2`, and curriculum stage via `set_postfix`. All bars guard with `disable=not sys.stdout.isatty()` for CI/pipe safety. Additionally: after Stage 0's `dataset_profile.json` (size tier) and Stage 0's hardware detection are available, a one-line ETA hint is printed — "Stage 2 estimated ~8 min (medium dataset, CPU)" — from a calibrated lookup table in `sparc/run/progress.py`. An EWM (exponential weighted mean) over per-epoch duration inside the epoch bar provides a dynamically-updating remaining-epoch ETA after the first epoch completes.

**Potential impact:** user experience (silent multi-minute training loops → live progress with ETA), new capability (stage-level timing written to artifact store enables historical ETA learning per machine), prerequisite for desktop live-progress streaming (Direction 2 blue-sky)

**Relevant SPARC modules:** new `sparc/run/progress.py` (~70 lines), `sparc/__main__.py` `cmd_run` (~20 line change), `sparc/run/v2_neural_training.py` (`_exec_cv_fold` epoch loop ~10 lines, `train_neural_meta` fold loop ~8 lines)

**Status:** in-backlog

---

### Causal CATE Posterior Surface via NUTS-Conditioned Spatial GP (2026-05-21b)
**Source fields:** `sparc/causal/spatial_cate.py` (`CausalForestDML`) × `sparc/run/v2_bayesian_causal.py` (NUTS β posterior) × Gaussian Process regression conditioned on stochastic nuisance estimates (Kennedy & O'Hagan 2001; Stein 2012)
**Core idea:** `CausalForestDML` produces a frequentist point-estimate CATE surface. NUTS already produces a full posterior `β ~ P(β|Y)` stored as `("3", "nuts_beta")`. For each posterior draw of β, a GP with Matérn-5/2 kernel — seeded from Stage 0 ν and ρ — interpolates a continuous CATE surface. The envelope of all GP draws is a credible-interval CATE map. Unlike `CausalForestDML`, this propagates both coefficient uncertainty (NUTS) and spatial interpolation uncertainty (GP) into the final CATE estimate. The output is a per-pixel `(CATE_mean, CATE_ci5, CATE_ci95)` raster. `_load_kernel_field()` already retrieves Stage 0 kernel parameters from the artifact store.
**Potential impact:** new capability (credible-interval CATE maps — full posterior uncertainty in spatial policy recommendations), methodological (first SPARC output combining Bayesian posterior sampling with GP spatial interpolation), interpretability (uncertainty widens appropriately in data-sparse areas)
**Relevant SPARC modules:** `sparc/causal/spatial_cate.py`, `sparc/run/correlogram_matern_fit.py`, `sparc/run/v2_bayesian_causal.py`, Stage 0 artifact `effective_range_matrix`
**Status:** in-backlog  *(backlog: S3-5)*

---

### Structural Causal Model Intervention Distributions via Full DAG Posterior (2026-05-21b)
**Source fields:** `sparc/causal/mc3.py` (edge inclusion probabilities) × `sparc/run/v2_bayesian_causal.py` (NUTS β) × Pearl's do-calculus (Pearl 2000; Bareinboim & Pearl 2016)
**Core idea:** Stage 3 uses the *median probability DAG* for intervention propagation — discarding structural uncertainty from MC³ edge inclusion probabilities. Sampling DAG structures proportional to their MC³ acceptance probability and propagating interventions through each sampled DAG produces `P(Y | do(X=x))` with credible intervals from both coefficient uncertainty (NUTS) and graph uncertainty (MC³). The median-DAG path remains as a fast approximate path; the full-posterior path is opt-in via `causal.inference: "full_dag_posterior"`. All infrastructure exists: MC³ `edge_probs` matrix, NUTS β posterior, `scenario_simulator.py` intervention propagation.
**Potential impact:** methodological (causal claims carry formal uncertainty from structure and parameters — publishable as a complete Bayesian causal inference pipeline), new capability (credible intervals on scenario outputs rather than point predictions)
**Relevant SPARC modules:** `sparc/causal/mc3.py`, `sparc/run/v2_bayesian_causal.py`, `sparc/interventions/scenario_simulator.py`, `sparc/run/causal_validation.py`
**Status:** in-backlog  *(backlog: S3-4)*

---

### WebSocket Live Progress Streaming to Desktop App

**Source fields:** `sparc/server/` (FastAPI) × `sparc-desktop/src/` (Tauri + React) × Server-Sent Events / WebSocket protocol × `sparc/run/progress.py` (proposed)

**Core idea:** When the user invokes `sparc run` from the desktop application, the Tauri frontend receives no pipeline feedback — only a spinner. The proposed `StageProgress` module (tqdm-based CLI layer) can optionally emit structured progress events over a WebSocket endpoint (`/ws/pipeline/progress`) if the `SPARC_SERVER_URL` env variable is set (the desktop sets this automatically when launching a run). The FastAPI server broadcasts incoming progress events to all connected WebSocket clients. The React frontend subscribes and renders a `PipelineProgressPanel` component with stage bars, epoch bars, and ETA display — identical state to the CLI experience but in the desktop UI. The existing `sparc/run/result_store.py` artifact store can persist progress snapshots so the desktop can reconstruct progress state on reconnect. This creates a unified progress model: the CLI and desktop show the same information from the same source.

**Potential impact:** new capability (live run monitoring in desktop without polling), user experience (users can watch Stage 2 training progress in the desktop app rather than switching to terminal), methodological (structured progress events become a first-class run artifact, enabling post-hoc performance analysis in the registry)

**Relevant SPARC modules:** `sparc/server/` (new `ws/pipeline/progress` endpoint), `sparc/run/progress.py` (event emitter), `sparc-desktop/src/` (new `PipelineProgressPanel` component), `sparc/registry/store.py` (progress snapshot persistence)

**Status:** in-backlog  *(backlog: P4-8 — PipelineProgressPanel desktop component)*

*Note (2026-05-20d self-grill): `sparc/server/stream.py` already implements `_EventCapture` (stdout/stderr/logging redirect → structured events), `_EPOCH_RE` (epoch progress parsing), `_PCT_RE` (tqdm percentage parsing → `progress_pct` events), and `stream_stage()` wires it into the `/run/stream` WebSocket endpoint. The infrastructure is ~80% built for the desktop path. The remaining gap is stage-level `progress_pct` events from `StageProgress` output (tqdm `N%|` already parsed by `_PCT_RE`). No new WebSocket endpoint needed — the tqdm bars (UX-1/UX-2/UX-3) directly serve both CLI and desktop.*

---

### Riemannian HMC via Dense Mass Matrix for Correlated Treatment Posteriors (2026-05-21c)
**Source fields:** `sparc/causal/nuts.py` × Information geometry / Riemannian manifold Monte Carlo (Girolami & Calderhead 2011)
**Core idea:** NUTS currently uses a diagonal mass matrix (`inv_mass_diag`) which cannot capture correlations between β coefficients. When spatial treatments are anticorrelated by land-use type (e.g., tree canopy and impervious surface), the posterior of β is a banana-shaped ridge. The treatment covariance matrix `X_raw.T @ X_raw / n` is already computed in `_run_nuts_sampling()` and is the natural empirical Fisher at the likelihood mode — the right seed for a low-rank mass matrix. Replacing `inv_mass_diag` with an `inv_mass_chol` Cholesky factor in `_leapfrog()` and `_kinetic_energy()` enables the sampler to step along the posterior's natural geometry with ~5–20× ESS improvement for correlated posteriors.
**Potential impact:** performance (ESS ×5–20 for correlated multi-treatment models), methodological (theoretically grounded sampler for correlated spatial priors)
**Relevant SPARC modules:** `sparc/causal/nuts.py` (`_leapfrog`, `_kinetic_energy`, `run_nuts`), `sparc/run/v2_bayesian_causal.py` (`_run_nuts_sampling`)
**Status:** in-backlog  *(backlog: S3-11b — full Cholesky leapfrog)*

---

### DiBS — SVGD over DAG Latent Space (2026-05-22b)
**Source fields:** `sparc/causal/` × Differentiable structure learning (Lorch et al. 2021, NeurIPS)
**Core idea:** Bayesian structure learning via Stein Variational Gradient Descent over a continuous DAG latent space Z (Lorch et al. 2021). K=20 particles in Z-space, A_k = sigmoid(Z_k/τ), temperature annealing (τ: 1.0 → 0.05 geometric), SVGD joint update, BGe score + physics prior soft-BCE + acyclicity penalty. Returns `DiBSResult` compatible with `MC3Results`. Wired behind `inference_backend: "dibs"` in `v2_bayesian_causal.py`.
**Potential impact:** new capability (continuous posterior over DAGs via variational inference, richer uncertainty than MC³ point samples), methodological (gradient-based structure learning enables physics prior gradients to propagate directly into graph structure)
**Relevant SPARC modules:** `sparc/causal/dibs.py` (new), `sparc/run/v2_bayesian_causal.py` (dispatch), `sparc/config/causal_defaults.py` (config block)
**Status:** implemented

---

### Order-MCMC — MCMC over Topological Orderings (2026-05-22b)
**Source fields:** `sparc/causal/` × Friedman & Koller (2003) Order-MCMC
**Core idea:** MCMC over topological orderings σ rather than DAG adjacency matrices. For each ordering, score(σ) = Σ_j [best BGe conditional for j from its predecessors], enumerate all subsets of size ≤ k_max_parents using `itertools.combinations`. MH acceptance, post-burnin edge frequency accumulation. BGe cache makes repeated lookups O(1) after first warmup. Returns `OrderMCMCResult` compatible with `MC3Results`. Wired behind `inference_backend: "order_mcmc"`.
**Potential impact:** new capability (theoretical mixing guarantee in ordering space, avoids MC³'s MEC-collapse problem on large-n peaked posteriors), performance (BGe cache with O(1) lookup after warmup makes large p tractable)
**Relevant SPARC modules:** `sparc/causal/order_mcmc.py` (new), `sparc/run/v2_bayesian_causal.py` (dispatch), `sparc/config/causal_defaults.py` (config block)
**Status:** implemented

---

### Causal Transportability from Source to Target City (2026-05-22b)
**Source fields:** `sparc/causal/mc3.py` (DAG posterior) × Bareinboim & Pearl (2014, PNAS) transportability theory × Phase 4 zero-shot inference
**Core idea:** Given a causal DAG learned in Providence, transport the causal estimates to Lagos without ground-truth labels. The transport formula requires identifying which population-specific mechanisms differ across domains (selection diagram nodes). SPARC's satellite feature space (land cover, climate zone) defines the selection variables. If the DAG is identified across these selection variables, ATE in the target city = Σ_z E_source[Y(t)|Z=z] · P_target(Z=z) where Z = climate/land-cover covariates from `SatelliteFeatureSet`. New module `sparc/causal/transportability.py`: `build_selection_diagram(dag, selection_vars)`, `compute_transport_formula(dag, treatment, outcome, selection_vars, source_data, target_covariates)`. This directly bridges Phase 4 zero-shot and Phase 5 causal leadership.
**Potential impact:** new capability (causal estimates for unseen cities without labels — a differentiating claim over all existing spatial causal tools), methodological (publishable: transportability theory applied to spatial causal inference)
**Relevant SPARC modules:** new `sparc/causal/transportability.py`, `sparc/causal/mc3.py`, `sparc/run/v2_bayesian_causal.py`, `sparc/data/satellite_types.py` (selection variables)
**Status:** in-backlog  *(backlog: Causal Transportability item in Phase 5)*

---

### Energy-Based Coreset Selection for Smarter Anti-Forgetting Replay (2026-06-02)
**Source fields:** `sparc/training/replay.py` (`CoresetSelector`) × Energy-based models / Langevin dynamics (Du & Mordatch 2019; Grathwohl et al. 2020)
**Core idea:** `CoresetSelector` uses greedy K-medoids (facility location in feature space) — optimal for feature-space coverage but ignorant of where the model is currently wrong. An energy-based selector defines `E(z) = ‖model(z) − y‖²` as the selection energy and uses Langevin-dynamics sampling to find points near the model's current failure modes. Each city's coreset is thus weighted toward the OOD frontier rather than the feature centroid — the most valuable anti-forgetting signal. The `CityReplayState` infrastructure (added 2026-06-02) already provides the interface; only `CoresetSelector._greedy_kmedoids()` needs a gradient-aware alternative.
**Potential impact:** performance (dramatically better coreset quality for replay in OOD regions where forgetting is worst), methodological (replaces O(N·k) greedy heuristic with a principled EBM criterion)
**Relevant SPARC modules:** `sparc/training/replay.py` (`CoresetSelector`, `CityReplayState`), `sparc/run/continual_training.py`
**Status:** new

---

### Sheaf-Coboundary Operator as MAUP-Resistant Topology Loss (2026-06-02)
**Source fields:** `sparc/physics/pde_operators.py` × Topological signal processing (Bodnar et al. 2022, *CW-Networks*; Hansen & Ghrist 2020)
**Core idea:** The cardinal-neighbor graph built in `_build_cardinal_neighbors()` is a natural 1-complex. A cellular sheaf assigns a vector space (local prediction field) to each node and edge with restriction maps encoding the expected N/S/E/W prediction relationships. The sheaf coboundary operator `δ: C^0 → C^1` generalizes the graph Laplacian; `‖δT‖²` is zero iff adjacent predictions are consistent across all scales — a formal MAUP-resistance condition. The existing `sheaf_delta` parameter in `sparc_joint_loss()` already anticipates this integration. The architecture seam is in place; only the operator construction is missing.
**Potential impact:** methodological (formal MAUP robustness proof — publishable differentiator for SPARC vs. all existing spatial models), new capability (multi-scale consistency certificate per prediction)
**Relevant SPARC modules:** `sparc/physics/pde_operators.py`, new `sparc/physics/sheaf_operators.py`, `sparc/training/loss.py` (`sheaf_delta` param already present)
**Status:** new

---

### Normalizing Flows over DAG Posterior (2026-05-22b blue-sky)
**Source fields:** `sparc/causal/dibs.py` × Continuous normalizing flows (Rezende & Mohamed 2015; Chen et al. 2018, Neural ODE)
**Core idea:** DiBS does particle-based variational inference over Z-space (K fixed particles). The natural extension is a continuous normalizing flow (CNF) that learns p(Z | D, physics) directly — a flow-based distribution over Z-matrices rather than point masses. A CNF's adjoint ODE replaces the SVGD kernel step. This gives (1) direct density estimation of the DAG posterior, (2) richer exploration through ODE-parameterized flow trajectories, (3) exact log-likelihood of any candidate DAG under the posterior. Connects to the DiBS paper (Lorch et al. 2021) follow-up line and directly extends `sparc/causal/dibs.py`.
**Potential impact:** methodological (publishable: physics-constrained CNF for DAG posterior), new capability (exact posterior log-likelihood enables model comparison and marginal likelihood estimation for physics prior tuning)
**Relevant SPARC modules:** `sparc/causal/dibs.py` (base), potential new `sparc/causal/dag_flow.py`
**Status:** in-backlog  *(backlog: Normalizing Flows item in Phase S3 structure learning extensions)*

---

### Spatial Causal Confounder Recovery via Latent Sheaf Diffusion (2026-05-22b blue-sky)
**Source fields:** `sparc/physics/pde_operators.py` × Sheaf neural networks (Bodnar et al. 2022, NeurIPS) × `sparc/causal/sensitivity.py` (E-values)
**Core idea:** Latent spatial confounders (wind patterns, unmeasured urban morphology) manifest as spatially smooth residual fields — fields with low sheaf coboundary energy. A sheaf diffusion layer over the KNN graph learns to extract latent confounders from NUTS residuals and feeds them back as additional DAG nodes, iteratively deconfounding the causal graph. `sparc/physics/pde_operators.py` has `_build_cardinal_neighbors()` — the natural sheaf base space. `sparc/causal/sensitivity.py` E-values already quantify unmeasured confounders as a scalar; this gives them an explicit spatial representation. The resulting confounder nodes can be input to DoWhy placebo refutation.
**Potential impact:** new capability (spatially-explicit confounder recovery — first method for this in spatial causal inference), methodological (publishable: sheaf diffusion for spatial confounder identification), interpretability (maps of spatially-varying confounders alongside causal maps)
**Relevant SPARC modules:** `sparc/physics/pde_operators.py`, `sparc/causal/nuts.py`, `sparc/causal/sensitivity.py`, potential new `sparc/causal/sheaf_confounder.py`
**Status:** in-backlog  *(backlog: Sheaf Diffusion Confounder Recovery item in Phase S3 structure learning extensions)*

---

### Stein Variational Gradient Descent as MC³ Warm-Start (2026-05-21c)
**Source fields:** `sparc/causal/mc3.py` × Stein variational inference (Liu & Wang 2016) × NOTEARS differentiable DAG search (Zheng et al. 2018)
**Core idea:** MC³ initializes from the empty DAG and relies on random walk exploration to find high-posterior regions, yielding 2–3% acceptance with the current defaults. SVGD over a differentiable BGe relaxation (via NOTEARS sigmoid adjacency `W = σ(θ)`) can rapidly locate diverse high-score DAG candidates in ≤ 500 gradient steps. These warm-start the parallel tempering chains. The SPARC `BGeSuffStats` scatter matrix and PyTorch backend already support automatic differentiation; adding NOTEARS requires only a `h(W) = tr(e^{W∘W}) − p` acyclicity penalty as a soft constraint during the gradient pre-search phase.
**Potential impact:** performance (warm-started chains begin near high-posterior regions; acceptance rate 2–3% → 15–30% immediately), methodological (SVGD finds diverse modes so parallel chains start at different high-quality DAGs rather than all at the empty graph)
**Relevant SPARC modules:** `sparc/causal/mc3.py` (`run_mc3`, `BGeSuffStats`), new `sparc/causal/dag_warm_start.py`
**Status:** in-backlog  *(backlog: S3-14 SVGD-NOTEARS MC³ warm-start)*

---

### JEPA Trunk Residualization for Causal Deconfounding (2026-05-22c)
**Source fields:** `sparc/training/jepa_loss.py` (JEPA trunk) × `sparc/causal/counterfactual_engine.py` (DML) × Schölkopf et al. (2021) *Toward Causal Representation Learning* × Peters et al. (2016) *Causal and Anticausal Learning*
**Core idea:** The JEPA SharedTrunk encodes physics-consistent, spatially-smooth latent representations h(x) ∈ ℝ^256 per spatial point. Spatial autocorrelation — a known confounder for causal discovery — is precisely the structure the trunk learns to represent via VICReg + spatial patch masking. Projecting raw treatment features onto the trunk embedding space (PCA-16 → Ridge probe) and taking the residuals removes the spatially-structured confounding from the features before DML treatment nuisance estimation. The DML `model_t` (predicting treatment T from confounders W) on residualized features is less biased by the spatially autocorrelated component that confounds both treatment assignment and outcome. This is a nonlinear generalization of spatial error models (Anselin 1988) applied to causal discovery. Simultaneously, Stage 2's `oof_preds` are the best available `model_y` for the DML outcome nuisance — the ensemble's R² on the full dataset is an upper bound on any nuisance model HGB could achieve from confounders alone, providing a free-win bias reduction on the Y side.
**Potential impact:** methodological (nonlinear spatial deconfounding of causal estimates — publishable as first integration of self-supervised representation learning into spatial DML), performance (lower DML bias → tighter ATE credible intervals), interpretability (residualized features have lower Moran's I → conditional independence tests are more reliable)
**Relevant SPARC modules:** `sparc/models/neural_meta.py` (`encode()`), `sparc/causal/counterfactual_engine.py` (`_fit_edge_dml`), new `sparc/causal/spatial_residualizer.py`, `sparc/run/v2_bayesian_causal.py`
**Status:** in-backlog  *(backlog: JD-1, JD-2)*

---

### JEPA as Multi-Step World Model for Sequential Intervention Planning (2026-05-22c)
**Source fields:** `sparc/inference/latent_rollout.py` (V-JEPA 2-AC single step) × V-JEPA 2 (Assran et al. 2025, arXiv:2506.09985) × Model Predictive Control (Camacho & Bordons 2007) × Physics cascade tables (domain knowledge)
**Core idea:** The current `latent_rollout.py` applies a single action-conditioned predictor step. Urban climate interventions are sequential: planting tree canopy in year 1 changes the effective response landscape for impervious reduction in year 2 (carry-over, interaction effects). A multi-step rollout API extends V-JEPA 2-AC to sequences: `multi_step_latent_rollout(..., actions=[(treatment, Δx, Δt), ...])` chains `h_{t+1} = predictor(h_t, action_embed(actions[t]))` and decodes at each or the final step. Two rollout modes: (1) **latent chain** — pure predictor chaining in embedding space (fast, small drift risk, capped at 5 steps); (2) **re-encode** — apply a physics cascade table `{treatment: {feature: scale}}` per step to produce physically perturbed feature values, then re-encode via trunk for each step (physically grounded, requires domain knowledge). The physics cascade table is per-template YAML — auditable by domain experts, directly connects to SPARC's physical identifiability claim. This architecture is the precursor to a full MPC-style causal bandit: the world model simulates the sequential state trajectory, the bandit optimizes over intervention sequences.
**Potential impact:** new capability (compound multi-year intervention planning — "tree canopy now, cool roofs year 2, permeable pavement year 3" with outcome trajectory), methodological (first physics-cascade-constrained JEPA world model for urban climate, publishable), prerequisite for causal bandit / MPC planning (which uses this world model as the oracle)
**Relevant SPARC modules:** `sparc/inference/latent_rollout.py` (extend to `multi_step_latent_rollout`), new `sparc/inference/feature_perturbation.py` (`PhysicsCascade`), `sparc/models/latent_predictor.py`, `sparc/models/action_embedding.py`, domain template `caps.yml` files
**Status:** in-backlog  *(backlog: JD-4, JD-5)*
