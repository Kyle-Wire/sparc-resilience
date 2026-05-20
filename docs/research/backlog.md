# SPARC — Research Backlog

**Maintained by:** synthesis agent
**Last updated:** 2026-05-20 (research scan + Wager runner dead-code finding)

Items are ranked by impact/effort. Improvement agent picks the top `[ ]` item with complexity **low** or **medium**.

---

## Legend
- `[ ]` — todo
- `[~]` — in progress
- `[x]` — done
- `[!]` — blocked (reason noted inline)

Complexity: **low** = < 1 hour of focused edits | **medium** = half-day | **high** = multi-day / architectural

---

## Queue

### Phase 1 — Finish V3 Wiring (from SPARC_Future_Roadmap.md)

- [x] **1.1-a Wire EWC penalty + Fisher extraction + import fix** — complexity: low (~50 lines, 2 files) — *Done 2026-05-12: fixed import in continual_training.py, added optimal_params_list to _continual dict, wired ewc_penalty into all 3 training loops in v2_neural_training.py, added fisher_matrix + trunk_state_dict to return dict.*
  - Files: `sparc/run/v2_neural_training.py`, `sparc/run/continual_training.py`
  - **Self-grill finding:** `ewc_penalty()` requires 4 args: `model, fisher_matrices, optimal_params_list, trunk_keys`. The `_continual` dict (set by `train_continual()`) needs `optimal_params_list` populated via `extract_trunk_params()` after each city. Also fix the broken import in `continual_training.py` line 208: `from sparc.training.v2_neural_training` → `from sparc.run.v2_neural_training` (file does not exist at the training path).
  - Add `fisher_matrix` + `trunk_state_dict` to `train_neural_meta()` return dict (Gap 1.4 — bundle in same PR).
  - Success criterion: `sparc continual --cities a.yml,b.yml` runs without `ImportError`; second city's loss log shows `ewc_penalty > 0`.

- [!] **1.1-b Wire replay loss into training epoch loop** — complexity: medium — **BLOCKED: interface mismatch (2-phase fix)**
  - **Phase 1 — Interface redesign:** `compute_replay_loss()` calls `model(features=X)` but `SPARCMetaLearner.forward()` requires `(base_preds, physics_feats, X_spatial, coords, knn_index, alpha)`. Must redesign `compute_replay_loss()` signature to accept a SPARC-compatible dict.
  - **Phase 2 — Coreset schema extension:** Spatial encodings can be recomputed from `coords` cheaply at replay time (no schema change needed). Surrogate predictions must be cached: either (a) load saved surrogate checkpoints at replay time (surrogates already persisted to disk — safe), or (b) cache surrogate outputs in the registry at city-registration time (faster but requires CityRegistry schema change). Option (a) recommended.
  - **Do not bundle with 1.1-a.** Fix the interface + cache the surrogate outputs, then wire.
  - **Self-grill 2026-05-20:** Both phases confirmed required. Complexity is medium (not low). Cannot be done in a single pass without first testing the 2-city continual loop.

- [x] **1.2 Wire temporal features into training runner** — complexity: low (~30 lines, 1 file) — *Done 2026-05-12: added `time_embed_dim = neural_cfg.get("time_embed_dim", 0)`, passed it to both SPARCMetaLearner constructors; built time_idx tensor in `_prepare_tensors` via `get_snapshot_time_indices()`; passed `time_idx` to all 3 forward call sites (CV fold, main retrain, SWA). Guarded by `if tensors["time_idx"] is not None`.*
  - File: `sparc/run/v2_neural_training.py`
  - **Self-grill finding:** `SPARCMetaLearner` is always instantiated with `time_embed_dim=0` (hardcoded). Must also read `neural_cfg.get("time_embed_dim", 0)` and pass it to the constructor, otherwise `time_idx` passing is a no-op.
  - Call `compute_diurnal_features()` and `get_snapshot_time_indices()` in data prep when `config["temporal"]["snapshots"]` is set. Pass `time_idx` to `model.forward()`. Regression-safe: guarded by `if self.time_embed is not None`.
  - Success criterion: training log shows `time_embed` weight in param count; PDE loss dict includes non-zero `pde_transient`.

- [x] **1.3 JEPA pretraining pass** — complexity: medium — *Done 2026-05-12: Added standalone JEPA self-supervised pretraining loop before CV fold loop, gated by `config["jepa"]["pretrain_epochs"] > 0 AND jepa_enable`. Builds temporary SPARCMetaLearner + EMATrunk + LatentPredictor, trains with masked-context alignment + VICReg on full dataset, then transfers trunk weights to each fold model and the full-retrain model via `load_state_dict(strict=False)`. Added `jepa_loss` to the function import.*
  - File: `sparc/run/v2_neural_training.py` (new pre-training phase before CV loop)
  - **Self-grill finding:** inline JEPA is functional and produces valid gradients. This is a quality improvement (better trunk geometry), not a correctness fix. Lower priority than 1.1-a and 1.2.
  - Add a standalone pretraining loop when `config["jepa"]["pretrain_epochs"] > 0`, before the CV fold loop.

- [ ] **1.5 Transfer validation: Providence → Boston** — complexity: medium
  - Requires real city data. Run `transfer_validation.py` and confirm warm-start R² ≥ cold-start R².
  - Success criterion: positive `r2_improvement` in `transfer_comparison.json`.

### Phase 5 — Causal Inference Leadership

- [ ] **W-1 Wire `run_wager2025_gaps()` into Stage 3 pipeline** — complexity: **low** (~15 lines, 1 file)
  - **Gap:** `sparc/run/wager2025_addons.py::run_wager2025_gaps()` is a complete auto-runner for all 10 Wager (2025) audit gaps. All 10 underlying estimator modules are implemented and individually tested (`_audit.py` + `GAP_N_IMPLEMENTED` flags). The function is **never called** — no import or call site exists in `v2_bayesian_causal.py` or `run_enhanced_pipeline.py`. All 10 Wager gaps are dead code during a default `sparc run`.
  - **File:** `sparc/run/v2_bayesian_causal.py` — after NUTS block completes (~line 300), add:
    ```python
    from sparc.run.wager2025_addons import run_wager2025_gaps as _run_wager
    from sparc.causal.utils import dag_to_networkx, get_node_roles
    _G = dag_to_networkx(dag_def)
    _roles = get_node_roles(_G)
    run_wager2025_gaps(
        data=data, config=config, dag_def=dag_def,
        graph=_G, roles=_roles,
        output_dir=str(output_dir / "wager2025"),
        nuts_results=nuts_results,
    )
    ```
  - **Why safe:** Every gap is wrapped in `try/except` — a failing estimator logs a warning and continues, never aborting Stage 3. Strictly additive: no existing logic changes.
  - **Success criterion:** `sparc run --stage 3` produces `Stage_3_Causal/wager2025/` with `overlap_diagnostics.json`, `cbps_balance.json`, `policy_learning.json` etc. `_audit.report()` shows 10/10 gaps addressed.
  - **Academic source:** Wager (2025) *Introduction to Causal Inference* — each gap is a documented estimator from the text.

- [x] **GP regression over CATE surface** — complexity: medium — *Done 2026-05-12: Added `CATEGPSurface` class and `_moran_i()` helper to `sparc/causal/spatial_cate.py`. `CATEGPSurface` uses `sklearn.gaussian_process.GaussianProcessRegressor` with a `ConstantKernel × Matérn` kernel seeded from the Stage 0 correlogram payload (`nu`, `kappa.mean`) via `CATEGPSurface.from_correlogram()`. Fits on sub-sampled CATE estimates (default 2 000 rows, O(N³) GPR) and exposes `predict(coords_norm) → (mean, std)`. Moran's I (normality-assumption variance) on GP residuals reports `I`, `z_score`, `p_value`, and a one-line interpretation. Wired into `SpatialCATEEstimator.fit_gp_surface()` which loops over all treatments, returns `{treatment: {"gp_mean", "gp_std", "moran_i", "log_ml", "kernel_params"}}`, and stores per-treatment `CATEGPSurface` in `self._gp_surfaces`.*

- [x] **PCMCI+ integration for spatio-temporal panels** — complexity: medium — *Done 2026-05-13: Added `discover_spatiotemporal_causal_structure()` to `sparc/causal/panel.py`. Wraps `tigramite.pcmci.PCMCI.run_pcmciplus()` with selectable CI tests (`parcorr`/`cmiknn`/`gpdc`), panel-to-mean-time-series collapsing, NaN forward/back-fill, and a T-length guard. Builds `edge_prob_matrix` (p×p float in [0,1]) from per-lag p-values blended with a 0.1 base prior — ready for `PhysicsInformedGraphPrior(edge_probs=...)`. Returns `graph`, `val_matrix`, `p_matrix`, `edge_prob_matrix`, `variables`, `summary`. Module docstring updated. 15 causal_gaps + causal_discovery tests passed.*

- [x] **PDE-informed MC³ prior edge probabilities** — complexity: medium *(revised from high)* — *Done 2026-05-13: Added `pde_residual_edge_probs(data, node_names, coords, knn_k, alpha_base) -> np.ndarray` to `sparc/causal/mc3.py`. For each ordered pair (i,j) fits an OLS ANM in both directions and scores each direction by the spatial Laplacian energy of its residuals — the direction whose residuals are more spatially homogeneous (lower Laplacian energy) gets higher edge probability via a sigmoid. Added `PhysicsInformedGraphPrior.from_pde_residuals()` classmethod wrapping the above. Extended `MC3Results` with `physical_plausibility_score: dict[str,float]` field; `run_mc3()` accepts optional `pde_coords` and `pde_knn_k` params — if supplied it computes per-winning-edge plausibility at completion. New `TestPDEMC3` class (5 tests) uses fully synthetic data and all pass.*

- [x] **`maup_sensitivity_analysis()` diagnostic** — complexity: medium — *Done 2026-05-13: Created `sparc/causal/maup_sensitivity.py`. Implements `maup_sensitivity_analysis(data, treatment, outcome, confounders, coord_cols, cell_sizes, ...)` which aggregates point data onto regular grids at each requested cell size, runs cross-fit DML (HistGradientBoosting nuisance models, `KFold` splitting) at each scale, and returns a `MAUPSensitivityResult` with per-scale `ScaleResult` objects and a scalar `maup_robustness_score` ∈ [0, 1] = 1 − CV(ATE across scales). Score ≥ 0.9 = excellent; < 0.8 = interpret cautiously; sign instability flagged separately. Auto-derives default cell sizes from coordinate range when `cell_sizes=None`. Added to `sparc/causal/__init__.py`. 11 synthetic-data tests all pass (`test_maup_sensitivity.py`).*

- [x] **`fairness_audit()` in mediation.py** — complexity: medium — *Done 2026-05-14: Added `fairness_audit(data, treatment, mediator, outcome, confounders, protected_attr, n_strata=10) -> FairnessAuditResult` method to `MediationDecomposer` in `sparc/causal/mediation.py`. Categorical protected attrs use unique levels as strata; continuous attrs are binned into n_strata equal-frequency quantiles. Strata with <30 rows are skipped with a warning. `FairnessAuditResult` dataclass tracks `stratum_results`, exposes `nde_disparity` / `nie_disparity` / `cte_disparity` (max−min across strata) and `disparity_ratio` (max|NDE|/min|NDE|). `summary_table()` returns a DataFrame with a `_disparity_` footer row. 13 tests in `tests/test_fairness_audit.py` all pass.*

### From Synthesis Agent (cross-pollination ideas)

<!-- Populated 2026-05-12. Ranked by impact × (1/effort). -->

- [x] **Add `wasserstein_trunk_alignment()` as OT-based continual learning penalty** — complexity: medium — *Done 2026-05-14: Added `wasserstein_trunk_alignment(trunk_activations_new, coreset_activations, blur=0.01) -> Tensor` to `sparc/training/ewc.py`. Uses `geomloss.SamplesLoss` when available; falls back to a pure-PyTorch sliced-Wasserstein approximation (64 random projections, sorted 1-D Wasserstein) when geomloss is not installed. Wired into `v2_neural_training.py` at all three training loops (CV fold, full retrain, SWA) via `trunk_fusion` forward hook capturing batch activations; guard `if _ot_active` reads `_cont.ot_lambda` and `_cont.coreset_activations`. `geomloss>=0.2` added to `requirements.txt` as optional commented dep. 10 tests in `tests/test_wasserstein_alignment.py` all pass.*

- [x] **Wire `pde_dag_score()` into MC³ log-score via new `pde_identifiability.py`** — complexity: medium — *Done 2026-05-14: Created `sparc/causal/pde_identifiability.py` (~220 lines). `compute_dag_pde_plausibility(adj, node_names, data, coords, knn_k, lambda_pde) -> float` scores a full DAG by the summed Laplacian energy of OLS-ANM residuals at each node with parents — log-score penalty = −lambda_pde × Σ L(ε_j). `PDEDagScorer` dataclass wraps this as a callable compatible with `run_mc3(pde_scorer=...)`. Added optional `pde_scorer` parameter to `run_mc3()` in `sparc/causal/mc3.py`; scorer result is added to `prop_score` before the MH acceptance step. Module added to `sparc/causal/__init__.py`. 13 tests in `tests/test_pde_identifiability.py` all pass.*

### Efficiency (from 2026-05-14 session — zero functional change, same outputs)

- [x] **E1 Vectorize `_remap_indices_to_local`** — complexity: **low** (~8 lines, 1 file) — *Done 2026-05-14: replaced Python enumerate loop with `local_map[batch_idx] = np.arange(len(batch_idx))` vectorized write. Identical output; ~50× faster per invocation.*
  - **File:** `sparc/run/v2_neural_training.py` — `_remap_indices_to_local()` lines 211–218
  - **What's needed:** Replace `for local_i, global_i in enumerate(batch_idx): local_map[global_i] = local_i` with `local_map[batch_idx] = np.arange(len(batch_idx))`. Drop-in — identical output.
  - **Why:** Called 3×/batch × ~50 batches/epoch × 100+ epochs = ~15 000 invocations; O(batch_size) Python loop in each. Vectorized numpy fancy-index is ~50× faster.
  - **Prerequisite for:** E1-torch — full `torch.compile` migration requires an additional step: replace `.cpu().numpy()` round-trips in the same function with `torch.zeros().scatter_()` + `torch.where()` (medium-complexity follow-up, listed separately).
  - **Success criterion:** All existing tests pass; `_remap_indices_to_local` profiling time drops >10× on any N > 100 batch.

- [x] **E2 Precompute KNN and Cardinal in capacity sweep closures** — complexity: **low** (~8 lines moved, 1 file) — *Done 2026-05-14: hoisted `_sweep_knn`, `_sweep_card`, `_sweep_knn_full` out of `_sweep_train`/`_sweep_eval` closures; built once after `sweep_epochs` setup. Eliminates 3+ redundant cKDTree builds per capacity sweep.*
  - **File:** `sparc/run/v2_neural_training.py` — inside capacity sweep block (~lines 1041–1180)
  - **What's needed:** Hoist `_build_knn_index(coords[sweep_train_idx], ...)` and `_build_cardinal_neighbors(coords[sweep_train_idx], ...)` out of `_sweep_train` and `_sweep_eval` closures. Both are called with the same fixed `sweep_train_idx` on every CMA-ES trial; they produce the same result every time.
  - **Why:** Each `_build_knn_index` is an O(N log N) cKDTree build; `_build_cardinal_neighbors` does 4 tree queries. With 3+ capacity candidates × both closures = 6+ redundant builds.
  - **Success criterion:** `hidden_dim` selection unchanged; log confirms KNN/cardinal built exactly once before sweep.

- [x] **E3 Vectorize `calculate_fold_spatial_autocorr` Moran's I** — complexity: **low** (~15 lines replaced, 1 file) — *Done 2026-05-14: replaced O(N×K) double Python loop with vectorized `yc_nb = yc[indices[:,1:]]; cross = (yc[:,None]*yc_nb).sum()`. Mathematically identical row-standardised Moran's I.*
  - **File:** `sparc/run/enhanced_spatial_cv.py` — `calculate_fold_spatial_autocorr()` lines 258–265
  - **What's needed:** Replace the double for-loop with vectorized numpy: `yc_nb = yc[indices[:, 1:]]; cross = (yc[:, None] * yc_nb).sum()`. Row-standardized equal weights mean the formula simplifies cleanly.
  - **Why:** N=2 000, K=8 → 16 000 Python iterations, called 20× per run (5 folds × 4 models). Numpy vectorized version is O(N·K) in C.
  - **Success criterion:** Moran's I output within 1e-10 of current for same input arrays; all autocorrelation tests pass.

- [x] **E4 Shared spatial index in `score_model_spatial_consistency`** — complexity: **low** (~25 lines, 1 file) — *Done 2026-05-14: added `_precomputed` param to `_spatial_smooth` and `_precomputed_indices` to `calculate_fold_spatial_autocorr`; `score_model_spatial_consistency` builds NearestNeighbors once and threads (distances, indices) through both helpers.*
  - **File:** `sparc/run/enhanced_spatial_cv.py` — `score_model_spatial_consistency`, `_spatial_smooth`, `calculate_fold_spatial_autocorr`
  - **What's needed:** `score_model_spatial_consistency` calls `_spatial_smooth` then `calculate_fold_spatial_autocorr` on the same `coords` — two `NearestNeighbors.fit()` calls. Refactor `_spatial_smooth` to accept an optional precomputed `indices` array; build once in `score_model_spatial_consistency` and thread through both helpers.
  - **Why:** `NearestNeighbors.fit` is O(N log N); called twice for identical coords. Config flag `self_supervised_hparam_scoring: true` triggers this path per model per run.
  - **Success criterion:** Single `NearestNeighbors.fit` per `score_model_spatial_consistency` call; spatial consistency scores identical to previous.

- [x] **E5 `predict_with_uncertainty` — `torch.inference_mode` + training-state restoration** — complexity: **low** (~8 lines, 1 file) — *Done 2026-05-14: saves `was_training = self.training` before `self.train()`, replaced `torch.no_grad()` with `torch.inference_mode()`, restores eval state via `if not was_training: self.eval()` after MC loop.*
  - **File:** `sparc/models/neural_meta.py` — `predict_with_uncertainty()`
  - **What's needed:** (a) Replace `torch.no_grad()` with `torch.inference_mode()` (strictly faster — skips version tracking). (b) Save `was_training = self.training` before `self.train()` and restore with `if not was_training: self.eval()` after the loop.
  - **Why:** `torch.no_grad()` does not suppress view tracking in all PyTorch versions; `inference_mode` is always faster. State leak: if `predict_with_uncertainty` is called while model is in eval mode, the model is left in train mode afterward — unexpected stochastic behavior in downstream calls.
  - **Success criterion:** MC Dropout statistics identical (same seed); `model.training` is `False` after `model.eval(); predict_with_uncertainty(model, ...)`. No test regressions.

- [x] **E1-torch Full `_remap_indices_to_local` torch-native migration** — complexity: **medium** (~20 lines, 1 file) — *Done 2026-05-14: replaced all numpy operations with pure-torch: `local_map` built as `torch.full((map_size,), -1)` + `local_map[batch_t] = torch.arange(len(batch_idx))`, then moved to device; `neighbor_tensor` never leaves device — no `.cpu().numpy()` round-trip. `torch.where(valid, local_map[nb_safe], full_like(nb,-1))` replaces the numpy scatter-gather. Equivalence confirmed on 4 test cases including edge cases. 35 tests pass.*
  - **File:** `sparc/run/v2_neural_training.py` — `_remap_indices_to_local()` entire body
  - **What's needed:** After E1, replace the remaining `.cpu().numpy()` round-trips with `torch.zeros().scatter_()` (build local_map as a LongTensor), `torch.full_like(neighbor_tensor, -1)` for remapped init, and `torch.where(valid_mask, local_map_tensor[nb_clamped], minus_one_tensor)` for the scatter-gather. Result: entirely pure-torch, no numpy, no CPU round-trip on GPU/MPS. This is the actual prerequisite for `torch.compile` graph capture over the epoch step.
  - **Why:** The `.cpu().numpy()` calls in the current body cross the Python/C++ boundary and break `torch.compile` graph capture even after E1. This migration eliminates the last Python/numpy dependency in the hot batch loop.
  - **Success criterion:** Identical output on all inputs; `torch.compile` on a wrapped epoch step function no longer raises graph-break on `_remap_indices_to_local`.

- [x] **E-Perf-B Precomputed sparse Laplacian in `_prepare_tensors` for PDE loss** — complexity: **medium** (~40 lines, 3 files) — *Done 2026-05-14: Added `build_sparse_laplacian(cardinal_idx, h)` to `sparc/physics/pde_operators.py` — vectorised build using `valid_idx.unsqueeze(1).expand(M,4)` + sign-correct convention (`L[i,i]=-4/h²`, `L[i,j]=+1/h²`). Stored `sparse_laplacian` + `valid_laplacian_mask` in `_prepare_tensors` dict. Added optional `sparse_laplacian`/`valid_laplacian_mask` params to `compute_pde_loss` (all 5 laplacian call sites converted to `_laplacian_of()` dispatcher) and threaded through `sparc_joint_loss`. Sparse path activates only when `T_pred.shape[0] == sparse_L.shape[0]` (full-N context); batch training falls back to 4-gather path. Max numerical diff vs. dense: 4.66e-10. 55 tests pass.*
  - **Files:**
    - `sparc/physics/pde_operators.py` — new `build_sparse_laplacian(cardinal: np.ndarray, N: int, device) -> torch.Tensor` that converts the `(N,4)` cardinal index array into a `torch.sparse_coo_tensor` with ≤4N non-zeros (diagonal = degree, off-diagonal = −1 per valid neighbor)
    - `sparc/physics/pde_loss.py` — in `compute_pde_loss`, replace the 4 per-direction index-gather ops for `lap_T` with `lap_T = torch.sparse.mm(sparse_L, T_pred.unsqueeze(-1)).squeeze(-1)`
    - `sparc/run/v2_neural_training.py` — in `_prepare_tensors`, after `_build_cardinal_neighbors`, call `build_sparse_laplacian` and store result as `tensors["sparse_laplacian"]`; thread it into `compute_pde_loss` call sites
  - **Sketch:**
    ```python
    # pde_operators.py
    def build_sparse_laplacian(cardinal: np.ndarray, N: int, device) -> torch.Tensor:
        rows, cols, vals = [], [], []
        for i in range(N):
            nb = cardinal[i][cardinal[i] >= 0]
            if len(nb) == 0: continue
            rows += [i] * (len(nb) + 1); cols += [i] + nb.tolist()
            vals += [float(len(nb))] + [-1.0] * len(nb)
        idx = torch.tensor([rows, cols], dtype=torch.long)
        v = torch.tensor(vals, dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, v, (N, N), device=device).coalesce()

    # pde_loss.py — replace gathers:
    lap_T = torch.sparse.mm(sparse_L, T_pred.unsqueeze(-1)).squeeze(-1)
    ```
  - **Why:** Replaces 4 uncoalesced scatter-gather ops per Laplacian evaluation with a single precomputed sparse matmul; GPU/MPS-compatible; the cardinal structure is constant across epochs so `L` is built exactly once per run.
  - **Depends on:** none
  - **Success criterion:** PDE loss values within 1e-5 of current implementation on identical inputs; all PDE loss tests pass; profiling shows ≥2× speedup on Laplacian step for N > 5 000.

- [x] **E-Perf-A Wire `torch.compile(mode="reduce-overhead")` onto epoch step** — complexity: **medium** (~20 lines, 1 file) — *Done 2026-05-14: Added `_maybe_compile(mod)` helper (graceful no-op on torch < 2.0 or compile failure). Applied to `model`, `process_net`, `source_net`, and all three surrogates after each creation site — once in the CV fold setup and once in the full-retrain setup. `torch.compile` modifies nn.Module in-place so optimizer parameter collection, OT forward hooks, and JEPA EMA wrapping all remain unaffected. 55 tests pass.*
  - **File:** `sparc/run/v2_neural_training.py` — extract the per-batch forward+loss body into a standalone `_epoch_step(batch_dict) -> loss` function; apply `torch.compile(_epoch_step, mode="reduce-overhead")` once after the function is defined; thread the compiled callable through the CV fold, full-retrain, and SWA training loops
  - **Sketch:**
    ```python
    def _make_epoch_step(model, surrogates, pde_tensors, device):
        def _epoch_step(batch_dict):
            base_preds = _forward_surrogates(surrogates, batch_dict)
            T_pred, pde_feats, _ = model(base_preds, batch_dict["physics"],
                                         batch_dict["X_spatial"], batch_dict["coords"],
                                         batch_dict["knn"], batch_dict["alpha"])
            loss = criterion(T_pred, batch_dict["y"]) + compute_pde_loss(T_pred, pde_feats, pde_tensors)
            return loss
        if torch.__version__ >= "2.0":
            return torch.compile(_epoch_step, mode="reduce-overhead")
        return _epoch_step
    ```
  - **Why:** `torch.compile` fuses GELU/LayerNorm kernel pairs across surrogates + SPARCMetaLearner, eliminates Python dispatch overhead on each `nn.Module.forward()`, and produces a single GPU kernel for the `cat → fusion → regression_head` sequence; 1.5–3× CPU, 3–5× GPU/MPS epoch throughput.
  - **Depends on:** E1-torch (pure-torch `_remap_indices_to_local` migration — eliminates last numpy call that breaks graph capture)
  - **Success criterion:** `sparc run` completes without `torch._dynamo` graph-break errors or fallback warnings; epoch wall-clock time reduced ≥ 30% on a 3 000-point reference dataset on CPU; behavior identical to uncompiled run (same loss curve, same final R²).

### Phase 2 / Phase 3 — Self-Supervised Stage 2 (from 2026-05-13 session)

- [x] **2.SS-1 MER block size from `effective_range_matrix` (self-supervised fold construction)** — complexity: low (~20 lines, 1 file) — *Done 2026-05-13: Added path #3 to `EnhancedSpatialCV.get_block_size_from_config()` that reads `effective_range_matrix` from the Stage 0 store, flattens the 2D `range_matrix`, computes `np.percentile(ranges, 10)`, floors at 500m, and logs "Self-supervised block size from MER (p10 of effective ranges): Xm". Strictly additive — user config and correlogram paths take priority; path #3 only fires when both are absent. 20 ERM+correlogram tests pass.*
  - **File:** `sparc/run/enhanced_spatial_cv.py` — `EnhancedSpatialCV.get_block_size_from_config()` and `spatial_kfold_enhanced()` fallback path
  - **What's needed:** In `get_block_size_from_config()`, after the Stage 0 `optimal_block_size` scalar read fails, read `effective_range_matrix` from the store (`_safe_read("0", "effective_range_matrix")`), extract `effective_ranges` dict values, compute `np.percentile(ranges, 10)`, floor at 500m, return. This is strictly additive — existing paths return first if available.
  - **Why:** Removes the last `y`-dependency from fold construction. Prerequisite for Phase 3 zero-shot mode (no ground labels).
  - **Self-grill verdict:** Confirmed — zero regression risk. Falls through to existing y-based fallback if `effective_range_matrix` artifact absent.
  - **Success criterion:** In a project with Stage 0 artifacts but without user-specified block size, the log line reads `Self-supervised block size from MER (p10 of effective ranges): Xm` rather than `Estimating spatial autocorrelation range...`.

- [x] **2.SS-2 `oof_extraction_hooks.py` — OOF spatial intelligence extraction** — complexity: medium (~80 lines, new module + flag flip) — *Done 2026-05-13: Created `sparc/interventions/oof_extraction_hooks.py` (~180 lines). `extract_from_fitted_model()` handles GWR (nearest-training-point β-map via `model.nn_.kneighbors`) and GWRF (nearest-subsample-center importances via BallTree on `model.subsample_coords`). `aggregate_oof_intelligence()` stitches per-fold dicts into full `(N, n_features)` arrays and writes to Stage 2 artifact store (`oof_gwr_beta_map`, `oof_gwrf_importance_map`) with .npy fallback. Flipped `EXTRACT_OOF_INTELLIGENCE = True` in `enhanced_spatial_cv.py` and cleaned up import path (removed sys.path trick). Added ~20 lines to `v2_neural_training.py`: loads `oof_gwr_beta_map` from Stage 2 store before JEPA pretrain block, z-scores it, creates a frozen `nn.Linear` projector, injects the β-map into the EMA target encoding inside `torch.no_grad()`. Import check + 24 EMA/causal-stack tests pass; inline extraction logic verified with mock GWR model.*
  - **Files:** New `sparc/run/oof_extraction_hooks.py`; `sparc/run/enhanced_spatial_cv.py` line 118 (`EXTRACT_OOF_INTELLIGENCE = False` → `True`); `sparc/run/v2_neural_training.py` JEPA pretrain phase (~15 lines to load OOF intelligence from artifact store and condition JEPA targets on β-maps)
  - **What's needed:** `extract_from_fitted_model(model, model_name, X_test, coords_test, feature_names, test_idx)` returns:
    - GWR: `{"type": "gwr_local_coefficients", "beta_map": model.coefficients_[model.nn_.kneighbors(coords_test)[1][:,0]], ...}` — nearest-neighbor β-map at test locations
    - GWRF: `{"type": "gwrf_spatial_importances", "importance_map": importances[BallTree(model.subsample_coords).query(coords_test)[1][:,0]], ...}` — local RF feature importances at nearest subsample centers
  - **Dependency:** Requires `GWRModel.nn_` (BallTree on training coords) — confirmed present. Requires `GWRFModel.subsample_coords` — confirmed present.
  - **Self-grill verdict:** API confirmed. Note: β-maps and RF importances are derived from labeled fitting, not purely label-free — they encode *spatial process structure* (how coefficients vary geographically), not label values. Valid as JEPA prediction targets.
  - **Success criterion:** After Stage 2, artifact store contains `oof_gwr_beta_map.npz` with shape `(n, n_features)` and `oof_gwrf_importance_map.npz` with shape `(n, n_features)`; JEPA pretrain log shows `oof_intelligence loaded: gwr_local_coefficients (n=XXXX)`.

- [x] **2.SS-3 `latent_guided_spatial_kfold()` — Area of Applicability fold quality** — complexity: medium (~60 lines, 1 file) — *Done 2026-05-13: Added `latent_guided_spatial_kfold()` (~95 lines) to `sparc/run/enhanced_spatial_cv.py` wrapping `spatial_kfold_enhanced()`. Added `_load_prior_trunk()` to `EnhancedSpatialCV` (infers architecture from state-dict weight shapes, loads via `CityRegistry`). Wired into `generate_optimized_oof_predictions()` call site. Single-city runs: pure pass-through. Multi-city: embeds all N points via `prior_trunk.encode(physics_t, alpha_t)`, computes NNCV D_i = min_d(test→train) / mean_d(train,train), logs OOD-heavy folds (>30% D_i>1.5). 46 tests pass, OOD warning verified with identity-mock trunk.*
  - **File:** `sparc/run/enhanced_spatial_cv.py` — new function wrapping `spatial_kfold_enhanced()` with AoA validation
  - **What's needed:** When a prior-city trunk checkpoint is available from `CityRegistry`, embed all N points via `trunk.encode_physics()`, compute NNCV dissimilarity index D_i (Meyer et al. 2021: `min_d(test, train) / mean_d(train, train)`), log folds where >30% of test points have D_i > 1.5 as "OOD-heavy" folds.
  - **Dependency:** Prior-city trunk in CityRegistry. **Active only in multi-city continual mode; degrades gracefully to geographic blocking otherwise.**
  - **Self-grill verdict:** Cannot use current city's trunk (produced after fold assignment). Requires prior city's trunk from registry. Single-city runs unaffected.
  - **Success criterion:** In 2-city continual run, log shows `Fold 3: 42% OOD test points (AoA D > 1.5) — flagged as out-of-distribution`; no regression on single-city runs.

- [x] **2.SS-4 `score_model_spatial_consistency()` — label-free hyperparameter scoring** — complexity: medium (~55 lines, 1 file) — *Done 2026-05-13: Added `_spatial_smooth(predictions, coords, bandwidth, k)` (Gaussian IDW, ~20 lines) and `score_model_spatial_consistency(predictions, coords, bandwidth) -> float` (~15 lines) to `sparc/run/enhanced_spatial_cv.py`. Score = Moran's I on `predictions − _spatial_smooth(predictions)`. Wired into `generate_optimized_oof_predictions()`: when `models.spatial_cv.self_supervised_hparam_scoring: true`, logs each model's score after OOF assembly. Uses `get_block_size_from_config()` as the default bandwidth. 46 tests pass, inline correctness check confirmed smooth predictions differ from noisy ones.*
  - **File:** `sparc/run/enhanced_spatial_cv.py` — new function; config key `models.spatial_cv.self_supervised_hparam_scoring: true`
  - **What's needed:** `score_model_spatial_consistency(model, X_test, coords_test) -> float` computes `pred - _spatial_smooth(pred, coords_test, bandwidth=500)` as a pseudo-residual (spatially smooth deviation), returns Moran's I on the pseudo-residuals via `calculate_fold_spatial_autocorr()`. Lower Moran's I = spatially consistent predictions = better model. Used as the primary criterion for bandwidth/k-neighbors tuning when `self_supervised_hparam_scoring: true`.
  - **Dependency:** Requires new `_spatial_smooth(predictions, coords, bandwidth)` helper (~15 lines): inverse-distance-weighted mean of k-nearest neighbors' predictions.
  - **Self-grill verdict:** Conceptually sound for label-free hyperparameter selection. Primarily useful for Phase 3 zero-shot mode. Does not replace labeled R² for normal supervised runs.
  - **Success criterion:** Config flag activates; log shows `Bandwidth 800m: spatial_consistency_score = 0.12 (lower = better)`; selected bandwidth differs from RMSE-optimal by <20% on test datasets.

- [x] **2.SS-5 `SpatialContrastiveLoss` for cross-city block encoder pretext** — complexity: high
  - **Implemented:** `sparc/training/spatial_contrastive.py` — `SpatialContrastiveLoss` (InfoNCE), `mine_positive_pairs` (log-bandwidth proximity), `query_coresets_by_bandwidth_cluster` (registry helper), `log_bandwidth_from_payload`, `spatial_contrastive_pretext` (trunk pretext training loop); `sparc/registry/city_registry.py` — `query_coresets_by_bandwidth_cluster()` method; `sparc/run/v2_neural_training.py` — contrastive pretext block gated by `continual.contrastive_pretext`; `tests/test_spatial_contrastive.py` — 20 tests, all pass.
  - **Files:** New `sparc/training/spatial_contrastive.py` — `SpatialContrastiveLoss(z_anchor, z_positive, z_negatives, temperature)` with InfoNCE loss; positive pair mining from `KernelField` bandwidth cluster membership; `sparc/run/v2_neural_training.py` — optional contrastive pretext phase before supervised CV loop (gated by `config["continual"]["contrastive_pretext"]`); `sparc/registry/city_registry.py` — `query_coresets_by_bandwidth_cluster()` helper
  - **Sketch:** `L_NCE = -log[exp(sim(a,p)/τ) / Σ exp(sim(a,n_i)/τ)]`. Positive pairs: coreset blocks from any registered city where `|log(bw_A) − log(bw_B)| < 0.3`; negatives: blocks from different KernelField bandwidth clusters. Encoder = JEPA trunk (shared). Loss applied as pretext phase over CityRegistry coresets before per-city supervised training.
  - **Why:** Trains a block encoder that is content-invariant to geography but sensitive to spatial process type; becomes the Phase 2 Central Registry federated alignment mechanism without requiring label alignment across cities.
  - **Depends on:** 1.5 (transfer validation Providence → Boston confirmed), `wasserstein_trunk_alignment` (wired, confirms cross-city trunk sharing is stable)
  - **Success:** In 2-city continual run, same-bandwidth-cluster coreset blocks have cosine similarity > 0.7 in JEPA latent space; cross-cluster pairs < 0.3; single-city runs unaffected.

- [x] **2.SS-6 `VariationalSpatialBlockAutoencoder` (VSBA) for label-free fold quality** — complexity: high
  - **Implemented:** `sparc/models/vsba.py` — `VariationalSpatialBlockAutoencoder` (encoder→μ/logvar, decoder, Matérn K-NN sparse prior, `elbo_loss`, `fit`, `elbo_score`, `from_correlogram` classmethod); `sparc/run/enhanced_spatial_cv.py` — `_vsba_fold_quality_score()` helper + `vsba_scoring/corr_payload/vsba_n_epochs` params on `latent_guided_spatial_kfold`; `sparc/run/v2_neural_training.py` — VSBA pretext block before CV loop, gated by `models.spatial_cv.vsba_fold_scoring`; `tests/test_vsba.py` — 18 tests, all pass.
  - **Files:** New `sparc/models/vsba.py` — `VariationalSpatialBlockAutoencoder` with Matérn GP prior on latent space (`ν`, `ρ` seeded from Stage 0 artifacts), `elbo_score(block_feats, coords) -> float` method; `sparc/run/enhanced_spatial_cv.py` — optional `_vsba_fold_quality_score()` hook called per fold, config key `models.spatial_cv.vsba_fold_scoring: true`; `sparc/run/v2_neural_training.py` — VSBA pretext training (20 epochs) before CV loop when enabled
  - **Sketch:** Encoder: `X_block → μ, σ²` (latent dim 16). GP prior: `KL(q(z) || p(z)) = KL(N(μ, σ²) || N(0, K_Matérn(coords, ν, ρ)))` — Matérn kernel built from Stage 0 `correlogram_payload` (`nu`, `kappa.mean`). Decoder: `z → X̂`. ELBO = `E[log p(X|z)] − KL`. Fold quality = mean ELBO over test block; low ELBO = OOD fold. `ProcessRateNet` α(x) field is natural spatially-varying latent for the encoder; Laplacian prior from `pde_operators.py`.
  - **Why:** Provides a calibrated label-free fold quality score that adapts to domain spatial autocorrelation; prerequisite for Phase 4 honest zero-shot confidence scores without any ground truth.
  - **Depends on:** 2.SS-1 (self-supervised block size, so VSBA trains on label-free fold boundaries), Stage 0 Matérn artifacts (for GP prior)
  - **Success:** On a run with ground truth available, VSBA ELBO correlates r > 0.6 with labeled fold R²; geographic holdout folds score ≤ 10th-percentile ELBO relative to in-distribution folds.

- [x] **Add `sheaf_restriction_loss()` as PDE curriculum term 11 for formal MAUP-resistance** — complexity: high
  - **Implemented:** `sparc/physics/pde_operators.py` — `build_sheaf_laplacian(knn_idx, stalk_dim=2, restriction_maps=None)` returning sparse δ⁰ coboundary; `sparc/physics/pde_loss.py` — Term 11 `pde_sheaf` in `compute_pde_loss` (gated by `sheaf_delta`, stage-activated at pde_epoch≥20), `PDELossWeights.sheaf=0.03`; `sparc/run/v2_neural_training.py` — `_prepare_tensors` builds `sheaf_delta` from KNN index, passed to all 3 `compute_pde_loss` call sites; `tests/test_sheaf_laplacian.py` — 14 tests, all pass.
  - Files: `sparc/physics/pde_operators.py` — `build_sheaf_laplacian(spatial_graph, restriction_maps, stalk_dim)`; `sparc/physics/pde_loss.py` — add term 11 `sheaf_restriction_loss(predictions_multiscale, sheaf_laplacian)`; `sparc/run/v2_neural_training.py` — build multi-resolution prediction targets and pass to `compute_pde_loss()`
  - Sketch:
    ```python
    # pde_operators.py
    def build_sheaf_laplacian(spatial_graph, restriction_maps, stalk_dim):
        """Coboundary operator δ⁰ : C⁰(F) → C¹(F); enforces multi-scale consistency."""
        ...
    # pde_loss.py — term 11:
    def sheaf_restriction_loss(preds_100m, preds_500m, sheaf_laplacian):
        """Penalize scale-inconsistent predictions via sheaf coboundary."""
        stacked = torch.cat([preds_100m, preds_500m], dim=-1)
        return (sheaf_laplacian @ stacked).pow(2).mean()
    ```
  - Why: Gives SPARC a formal MAUP-resistance guarantee via the sheaf coboundary constraint; multi-scale predictions that don't aggregate consistently are penalized as a physics term — unique differentiator for publication.
  - Depends on: Multi-resolution prediction targets (requires architectural addition); prerequisite for `maup_sensitivity_analysis()` stub in roadmap
  - Success: `sheaf_restriction_loss` non-zero in training log; 100m predictions aggregate to ±1% of 500m predictions on test grid; `maup_robustness_score ≥ 0.95`.

---

## Completed

*(Populated as items are implemented)*

### Sprint log — 2026-05-14

Implemented 3 medium-complexity backlog items in this session (continuing from 2026-05-13):

**`fairness_audit()` in mediation.py**  
Added `MediationDecomposer.fairness_audit(data, treatment, mediator, outcome, confounders, protected_attr, n_strata=10) -> FairnessAuditResult` to `sparc/causal/mediation.py`. Categorical protected attributes use unique levels as strata; continuous attributes are binned via `pd.qcut`. Strata with <30 rows are skipped with a warning. `FairnessAuditResult` exposes `nde_disparity`, `nie_disparity`, `cte_disparity` (max−min across strata), `disparity_ratio`, `summary_table()`, `as_dict()`. 13 tests in `tests/test_fairness_audit.py` all pass.

**`wasserstein_trunk_alignment()` in ewc.py**  
Added `wasserstein_trunk_alignment(trunk_activations_new, coreset_activations, blur=0.01) -> Tensor` to `sparc/training/ewc.py`. Tries `geomloss.SamplesLoss("sinkhorn", p=2)` first; when absent falls back to sliced-Wasserstein (64 random unit projections, average 1-D sorted Wasserstein). Wired into `v2_neural_training.py` at all three training loops (CV fold, full retrain, SWA) via `register_forward_hook` on `trunk_fusion`. OT config read from `_continual.ot_lambda` and `_continual.coreset_activations`. `geomloss>=0.2` added as commented optional dep in `requirements.txt`. 10 tests in `tests/test_wasserstein_alignment.py` all pass.

**PDE identifiability module (`pde_identifiability.py`)**  
Created `sparc/causal/pde_identifiability.py` with `compute_dag_pde_plausibility(adj, node_names, data, coords, knn_k, lambda_pde) -> float` and `PDEDagScorer` callable wrapper. Scores a full DAG by summed Laplacian energy of OLS-ANM residuals (log-score = −λ Σ L(ε_j)). `PDEDagScorer.from_data(...)` creates scorer; passing it as `pde_scorer=` to `run_mc3()` adds the PDE term to every MH proposal score. `run_mc3()` signature extended with `pde_scorer: Any | None = None`. Module added to `sparc/causal/__init__.py`. 13 tests in `tests/test_pde_identifiability.py` all pass.

**Remaining eligible items**: Only `sheaf_restriction_loss()` (complexity: **high**) remains — skipped per backlog rules.

---

### Phase 4 — Desktop Processing Page (from 2026-05-14b session)

- [ ] **P4-1 Add `/data/preprocess` endpoint and fix `handleApplyAll`** — complexity: medium
  - **Gap:** `handleApplyAll` in `ProcessingPage.tsx` calls `prepareData({ raster_paths: [] })` which hits `/data/prepare` (the spatial fishnet builder). That endpoint ignores the call and does nothing. All 8 step cards are marked done immediately with no real processing. `sparc/data/data_utils.py::prepare_data(df, config)` exists as the correct CSV transformation function but has no API endpoint.
  - **Files:**
    - `sparc/server/app.py` — new `POST /data/preprocess` endpoint that loads the project dataframe, calls `data_utils.prepare_data(df, config)`, streams SSE step events (`{"step": "<name>", "done": true}`), persists result as a new version via `versioning.save_version()`
    - `sparc-desktop/src/lib/api.ts` — new `preprocessData(onStep)` function consuming the SSE stream
    - `sparc-desktop/src/components/pages/ProcessingPage.tsx` — `handleApplyAll` must call `preprocessData` (not `prepareData`), toggling each step's `done` flag on receipt of its SSE event
  - **Pre-implementation check:** Read `sparc/data/data_utils.py` lines 306–420 to enumerate exactly which of the 8 steps `prepare_data` already covers. Steps not covered need to be composed from `processing.py`, `welford.py`, and a new Arrow cache helper.
  - **Success criterion:** User clicks Apply All → 8 step cards light up sequentially with real processing delays → `project_dir/versions/` shows a new version entry → data summary refreshes with the transformed dataframe.

- [ ] **P4-2 Surface `/data/validate` on Processing page** — complexity: low
  - **Gap:** `POST /data/validate` → `sparc/data/validation.py::validate_dataset()` is fully implemented (critical / warning / info tiers) but is never called from ProcessingPage. Users have no data quality feedback.
  - **Files:**
    - `sparc-desktop/src/components/pages/ProcessingPage.tsx` — add **Validate** button; call `validateData()` API function; render issue panel beneath step cards; disable Apply All when `validationResult.critical.length > 0` with tooltip "Fix critical issues first"
    - `sparc-desktop/src/lib/api.ts` — confirm `validateData()` exists (endpoint already wired); if not, add it
  - **Success criterion:** Validate button renders structured issue list within ~1 s; critical issues show in red; Apply All disabled when any critical issue is present.

- [ ] **P4-3 Wire Welford scaler into "Standardize (z-score)" step** — complexity: low
  - **Gap:** `sparc/data/welford.py` implements an online Welford scaler that is not imported or called anywhere in the server. The "Standardize" step currently does nothing.
  - **Files:**
    - `sparc/server/app.py` (new `/data/preprocess` endpoint) — in step 6 (`standardize`), call `WelfordScaler.fit_transform(df[numeric_cols])`, serialize fitted scaler state to `project_dir/artifacts/welford_scaler.pkl`
    - `sparc/run/v2_neural_training.py` or inference runner — load `welford_scaler.pkl` and apply `transform()` to new input batches at inference time
  - **Success criterion:** After Apply All, `project_dir/artifacts/welford_scaler.pkl` exists with valid `mean_` / `var_` / `n_samples_seen_` fields. Running Apply All twice on same data produces identical z-scores.

- [ ] **P4-4 Arrow cache write for "Write cached arrow" step** — complexity: low
  - **Gap:** Step 8 "Write cached arrow" has no backend implementation. No Parquet cache is written anywhere in the processing pipeline.
  - **Files:**
    - `sparc/server/app.py` (new `/data/preprocess` endpoint) — as last step, call `df.to_parquet(project_dir / "data_cache.parquet", engine="pyarrow", index=False)`
    - `sparc/run/v2_neural_training.py` — when loading dataset, check for `data_cache.parquet` first and use it; log "loaded from parquet cache"; CSV fallback if absent
  - **Success criterion:** After Apply All, `project_dir/data_cache.parquet` exists. `POST /run` log shows "loaded from parquet cache".

- [ ] **P4-5 Full version picker in Processing page UI** — complexity: medium
  - **Gap:** `GET /data/versions` and `POST /data/select_version` are fully implemented. The UI exposes only "Revert to Original" (always restores version 0). Users cannot restore intermediate states.
  - **Files:**
    - `sparc-desktop/src/components/pages/ProcessingPage.tsx` — replace "Revert to Original" button with a Versions dropdown (`<select>`) populated from `getDataVersions()`; selecting a version calls `selectDataVersion({ version_id })`; a badge on step cards shows active version label
    - `sparc-desktop/src/lib/api.ts` — confirm `getDataVersions()` and `selectDataVersion()` exist; add if missing
  - **Success criterion:** After 3 Apply All runs, Versions dropdown shows 3 entries. Selecting entry 1 restores that exact dataframe and refreshes the data summary canvas.

- [ ] **P4-6 Data provenance hashing in preprocessing SSE stream** — complexity: low — *(derivative from 2026-05-14b session)*
  - **Gap:** No step-level provenance is recorded. There is no way to verify cross-machine reproducibility or detect upstream data changes.
  - **Files:**
    - `sparc/server/app.py` (new `/data/preprocess` endpoint) — after each step, compute `hash_val = int(pd.util.hash_pandas_object(df).sum()) % (2**32)` and include it in the SSE event: `{"step": "<name>", "done": true, "sha": "<hash_val>"}`
    - `sparc/data/versioning.py` — persist step hashes alongside version metadata
    - `sparc-desktop/src/components/pages/ProcessingPage.tsx` — display per-step hash badge (truncated 8 chars) in the step card; clicking copies full hash to clipboard
  - **Depends on:** P4-1
  - **Success criterion:** Step cards show 8-char hash badges after Apply All. Re-running Apply All on unchanged data produces identical hashes. Replacing the raw CSV triggers different hash at step 1.

---

### Session Digest — 2026-05-20

**Focus:** Codebase scan after push; Wager (2025) audit dead-code finding; replay interface analysis.

**No implementations this session** — research/synthesis only.

**Key finding — Wager runner is dead code:** All 10 Wager (2025) audit gap modules are implemented (`dynamic.py`, `balancing.py`, `policy_learning.py`, `counterfactual_engine.py`, etc.) and the `run_wager2025_gaps()` auto-runner exists in `wager2025_addons.py`. However, no call site was found in `v2_bayesian_causal.py` or `run_enhanced_pipeline.py`. All 10 gaps are built but never exercised during `sparc run`. Added `W-1` to backlog (complexity: **low**).

**Replay interface mismatch confirmed (1.1-b):** `compute_replay_loss()` in `replay.py` lines 158–165 calls `model(**{"features": X})` which will throw `TypeError` against `SPARCMetaLearner.forward(base_preds, physics_feats, X_spatial, coords, knn_index, alpha)`. Updated 1.1-b to document the 2-phase fix (interface redesign + surrogate cache strategy).

**Phase 1 wiring audit:** All Phase 1 items confirmed wired except 1.1-b (replay, blocked). EWC at all 3 loops confirmed at lines 2116–2119, 2746–2749, 2951–2954. Fisher matrix + trunk_state_dict returned at lines 3288–3300. Temporal, JEPA, contrastive pretext, VSBA, sparse Laplacian, sheaf Laplacian, torch.compile — all confirmed wired.

**2 blue-sky derivatives added to derivatives.md:** (1) Latent Diffusion for Posterior Scenario Sampling (DDPM over JEPA trunk latent), (2) Causal Bandits for Sequential Intervention Design (Thompson sampling over CATE GP surface with `scenario_simulator.py` as oracle).

**Streaming Provenance derivative promoted:** `under-synthesis` (was `new`).

**Priority recommendation for next implementation session:** `W-1` (wire Wager runner, complexity low, zero regression risk, 15 lines).

**Focus:** Desktop Processing page correctness and data-quality tooling.

**Critical bug confirmed:** `ProcessingPage.tsx::handleApplyAll` calls `prepareData({ raster_paths: [] })` → `POST /data/prepare` (spatial fishnet builder). Called with empty rasters it does nothing. All 8 step cards are immediately marked done with no actual CSV transformation. The 8-step UI is scaffolding only.

**Root cause:** `sparc/data/data_utils.py::prepare_data(df, config)` is the correct CSV transformation function but has no `POST /data/preprocess` API endpoint.

**Unreachable backend utilities identified:** `validation.py::validate_dataset` (endpoint exists, never called from UI), `welford.py` (not wired anywhere), Arrow/Parquet cache write (missing entirely), full versioning picker (only revert-to-original exposed).

**5 backlog items added:** P4-1 through P4-5 (P4-6 is a low-complexity derivative follow-on). Priority order: P4-1 (pipeline broken) → P4-2 (validate button, endpoint exists) → P4-3 (welford scaler) → P4-4 (arrow cache) → P4-5 (version picker).
