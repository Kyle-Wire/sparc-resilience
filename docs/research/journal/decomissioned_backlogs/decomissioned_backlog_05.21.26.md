# SPARC — Research Backlog

**Maintained by:** synthesis agent
**Last updated:** 2026-05-21c (S3-3 through S3-9 added — Stage 3 MC³ / NUTS performance fixes; 1 CRITICAL bug, 2 parameter fixes, 2 structural improvements, 2 medium-complexity enhancements)

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

- [x] **W-1 Wire `run_wager2025_gaps()` into Stage 3 pipeline** — complexity: **low** (~15 lines, 1 file) — *Done: Added Wager2025 call block in `v2_bayesian_causal.py` before `return`. Imports `dag_to_networkx`, `get_node_roles` from `sparc.causal.dag_definition`; builds graph + roles; calls `run_wager2025_gaps()`; adds `wager2025_summary` to return dict. Wrapped in try/except — non-fatal.*
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

- [x] **P4-1 Add `/data/preprocess` endpoint and fix `handleApplyAll`** — complexity: medium — *Done: Added `POST /data/preprocess` SSE endpoint in app.py (8-step pipeline with per-step hash events). Added `preprocessData(onStep)` streaming function in api.ts. Updated `handleApplyAll` in ProcessingPage.tsx to call `preprocessData` and update step status in real time.*
  - **Gap:** `handleApplyAll` in `ProcessingPage.tsx` calls `prepareData({ raster_paths: [] })` which hits `/data/prepare` (the spatial fishnet builder). That endpoint ignores the call and does nothing. All 8 step cards are marked done immediately with no real processing. `sparc/data/data_utils.py::prepare_data(df, config)` exists as the correct CSV transformation function but has no API endpoint.
  - **Files:**
    - `sparc/server/app.py` — new `POST /data/preprocess` endpoint that loads the project dataframe, calls `data_utils.prepare_data(df, config)`, streams SSE step events (`{"step": "<name>", "done": true}`), persists result as a new version via `versioning.save_version()`
    - `sparc-desktop/src/lib/api.ts` — new `preprocessData(onStep)` function consuming the SSE stream
    - `sparc-desktop/src/components/pages/ProcessingPage.tsx` — `handleApplyAll` must call `preprocessData` (not `prepareData`), toggling each step's `done` flag on receipt of its SSE event
  - **Pre-implementation check:** Read `sparc/data/data_utils.py` lines 306–420 to enumerate exactly which of the 8 steps `prepare_data` already covers. Steps not covered need to be composed from `processing.py`, `welford.py`, and a new Arrow cache helper.
  - **Success criterion:** User clicks Apply All → 8 step cards light up sequentially with real processing delays → `project_dir/versions/` shows a new version entry → data summary refreshes with the transformed dataframe.

- [x] **P4-2 Surface `/data/validate` on Processing page** — complexity: low — *Done: Added Validate button to ProcessingPage header; `handleValidate` calls `validateData()` API; validation issue panel renders beneath StatGrid with severity colour coding; Apply All disabled when n_critical > 0.*
  - **Gap:** `POST /data/validate` → `sparc/data/validation.py::validate_dataset()` is fully implemented (critical / warning / info tiers) but is never called from ProcessingPage. Users have no data quality feedback.
  - **Files:**
    - `sparc-desktop/src/components/pages/ProcessingPage.tsx` — add **Validate** button; call `validateData()` API function; render issue panel beneath step cards; disable Apply All when `validationResult.critical.length > 0` with tooltip "Fix critical issues first"
    - `sparc-desktop/src/lib/api.ts` — confirm `validateData()` exists (endpoint already wired); if not, add it
  - **Success criterion:** Validate button renders structured issue list within ~1 s; critical issues show in red; Apply All disabled when any critical issue is present.

- [x] **P4-3 Wire Welford scaler into "Standardize (z-score)" step** — complexity: low — *Done: Step 6 of `/data/preprocess` calls `WelfordScaler.partial_fit()` + `transform()` on all numeric columns; fitted scaler saved via `scaler.save(artifacts_dir / "welford_scaler.pkl")`.*
  - **Gap:** `sparc/data/welford.py` implements an online Welford scaler that is not imported or called anywhere in the server. The "Standardize" step currently does nothing.
  - **Files:**
    - `sparc/server/app.py` (new `/data/preprocess` endpoint) — in step 6 (`standardize`), call `WelfordScaler.fit_transform(df[numeric_cols])`, serialize fitted scaler state to `project_dir/artifacts/welford_scaler.pkl`
    - `sparc/run/v2_neural_training.py` or inference runner — load `welford_scaler.pkl` and apply `transform()` to new input batches at inference time
  - **Success criterion:** After Apply All, `project_dir/artifacts/welford_scaler.pkl` exists with valid `mean_` / `var_` / `n_samples_seen_` fields. Running Apply All twice on same data produces identical z-scores.

- [x] **P4-4 Arrow cache write for "Write cached arrow" step** — complexity: low — *Done: Step 8 of `/data/preprocess` writes `project_dir/data_cache.parquet` via `df.to_parquet(..., engine="pyarrow", index=False)`.*
  - **Gap:** Step 8 "Write cached arrow" has no backend implementation. No Parquet cache is written anywhere in the processing pipeline.
  - **Files:**
    - `sparc/server/app.py` (new `/data/preprocess` endpoint) — as last step, call `df.to_parquet(project_dir / "data_cache.parquet", engine="pyarrow", index=False)`
    - `sparc/run/v2_neural_training.py` — when loading dataset, check for `data_cache.parquet` first and use it; log "loaded from parquet cache"; CSV fallback if absent
  - **Success criterion:** After Apply All, `project_dir/data_cache.parquet` exists. `POST /run` log shows "loaded from parquet cache".

- [x] **P4-5 Full version picker in Processing page UI** — complexity: medium — *Done: Added `versions` state populated via `getDataVersions()` on mount. "Revert" button replaced by `<select>` dropdown when >1 versions exist; selecting a version calls `handleSelectVersion()` → `selectDataVersion()`.*
  - **Gap:** `GET /data/versions` and `POST /data/select_version` are fully implemented. The UI exposes only "Revert to Original" (always restores version 0). Users cannot restore intermediate states.
  - **Files:**
    - `sparc-desktop/src/components/pages/ProcessingPage.tsx` — replace "Revert to Original" button with a Versions dropdown (`<select>`) populated from `getDataVersions()`; selecting a version calls `selectDataVersion({ version_id })`; a badge on step cards shows active version label
    - `sparc-desktop/src/lib/api.ts` — confirm `getDataVersions()` and `selectDataVersion()` exist; add if missing
  - **Success criterion:** After 3 Apply All runs, Versions dropdown shows 3 entries. Selecting entry 1 restores that exact dataframe and refreshes the data summary canvas.

- [x] **P4-6 Data provenance hashing in preprocessing SSE stream** — complexity: low — *Done: Each SSE step event includes `"sha": "<8-hex>"` computed via `pd.util.hash_pandas_object`. Step cards display `sha·<8hex>` badge in the `detail` field. Step hashes persisted in `save_versioned()` settings.*
  - **Gap:** No step-level provenance is recorded. There is no way to verify cross-machine reproducibility or detect upstream data changes.
  - **Files:**
    - `sparc/server/app.py` (new `/data/preprocess` endpoint) — after each step, compute `hash_val = int(pd.util.hash_pandas_object(df).sum()) % (2**32)` and include it in the SSE event: `{"step": "<name>", "done": true, "sha": "<hash_val>"}`
    - `sparc/data/versioning.py` — persist step hashes alongside version metadata
    - `sparc-desktop/src/components/pages/ProcessingPage.tsx` — display per-step hash badge (truncated 8 chars) in the step card; clicking copies full hash to clipboard
  - **Depends on:** P4-1
  - **Success criterion:** Step cards show 8-char hash badges after Apply All. Re-running Apply All on unchanged data produces identical hashes. Replacing the raw CSV triggers different hash at step 1.

---

### GPU / CUDA Acceleration (from 2026-05-20b session)

These items are ordered by implementation priority. CU-1 and CU-3 are prerequisites for CU-2.
All changes are guarded by `torch.cuda.is_available()` or new profile fields — zero regression risk on CPU-only machines.

- [x] **CU-5 `torch.cuda.empty_cache()` between CV folds** — complexity: **low** (~4 lines, 1 file)
  - **Gap:** Each CV fold allocates new model objects `.to(device)`. At fold end, Python `del model` releases references but CUDA's caching allocator retains memory blocks. On low-VRAM GPUs (4–8 GB) over 5+ folds with large N, allocator fragmentation can cause OOM even when instantaneous usage is within bounds.
  - **File:** `sparc/run/v2_neural_training.py` — after `del model, surrogates, process_net, source_net` at fold end (~L2250) and before full retrain (~L2284):
    ```python
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ```
  - **Why safe:** `empty_cache()` is a no-op on CPU and MPS; does not affect correctness; zero functional change.
  - **Success criterion:** On 8 GB VRAM GPU with 5 CV folds + large N (>5k): no CUDA OOM error; `torch.cuda.memory_reserved()` returns closer to baseline after each fold.

- [x] **CU-1 Add GPU fields to `HardwareProfile`; decouple `force_cpu` from RAM tier** — complexity: **low** (~40 lines, 1 file)
  - **Gap:** `hardware_profile.py` line 100: `force_cpu=True` is hardcoded for the "low" RAM tier (< 12 GB), regardless of GPU presence. A machine with 8 GB RAM + RTX 3070 (8 GB VRAM) always trains on CPU. `HardwareProfile` has no `gpu_available`, `gpu_vram_gb`, `gpu_count`, or `gpu_name` fields. `/api/hardware` returns no GPU info — the desktop cannot show GPU status.
  - **File:** `sparc/config/hardware_profile.py` — add `_detect_gpu() -> tuple[bool, int, float, str]` helper; add 4 fields to `HardwareProfile` dataclass; fix `_build_profile()` to set `force_cpu = not gpu_avail` for low-RAM tier (decouple from RAM):
    ```python
    @dataclass(frozen=True)
    class HardwareProfile:
        ...
        gpu_available: bool = False
        gpu_count: int = 0
        gpu_vram_gb: float = 0.0
        gpu_name: str = ""
    ```
  - **The server endpoint** (`/api/hardware`) already returns `effective.as_dict()` — GPU fields appear automatically once added to the dataclass. No server changes needed.
  - **Success criterion:** On 8 GB RAM + RTX 3070 machine: `detect_profile().force_cpu == False`; `/api/hardware` returns `{"gpu_available": true, "gpu_vram_gb": 8.0, "gpu_name": "NVIDIA GeForce RTX 3070"}`; desktop can show GPU badge.

- [x] **CU-3 GPU-VRAM-aware batch size in `HardwareProfile`** — complexity: **low** (~15 lines, 2 files)
  - **Gap:** `HardwareProfile.batch_size` (512/2048/4096) was sized for CPU RAM. On an RTX 3090 (24 GB VRAM), batch 4096 leaves the GPU at ~20–30% SM occupancy. The `spatial_minibatch_sampler` in `optimizer.py` clamps to `_profile.batch_size` (CPU-tuned). Optimal GPU batch is `min(N, floor(gpu_vram_gb * 256))` (conservative; ~256 rows per GB VRAM based on ~12 KB per-row activation footprint for hidden_dim=256).
  - **Depends on:** CU-1 (requires `gpu_available` and `gpu_vram_gb` fields)
  - **Files:**
    - `sparc/config/hardware_profile.py` — add `gpu_batch_size: int = 0` field; in `_build_profile()`: `gpu_bs = int(min(max(gpu_vram * 256, 512), 32768))` when GPU present
    - `sparc/run/v2_neural_training.py` (~5 lines) — after device selection at L949, override `batch_size` upward: `if device.type == "cuda": batch_size = max(batch_size, _hp.gpu_batch_size)`
  - **Success criterion:** On RTX 3090 (24 GB): `_hp.gpu_batch_size == 6144`; training log shows "GPU batch size override: 6144"; `nvtop` shows ≥ 50% SM occupancy vs. ~25% with CPU-tuned default.

- [x] **CU-4 Remove `fold_cardinal.cpu().numpy()` GPU→CPU sync in training loop** — complexity: **low** (~5 lines, 1 file) — *(housekeeping)*
  - **Gap:** `v2_neural_training.py` L1873: `spatial_minibatch_sampler(coords[train_idx], fold_cardinal.cpu().numpy(), ...)` forces a GPU→CPU sync every epoch. `fold_cardinal` is an `(N_train, 4)` LongTensor on device. The fix: keep a CPU numpy copy in `_prepare_tensors()` alongside the device tensor (device version still needed for PDE loss).
  - **Note:** Impact is low for N < 20k (sync is `n_epochs` times, ~96 KB per transfer, ~0.01 ms). Housekeeping for correctness and future CUDA Graph compatibility.
  - **File:** `sparc/run/v2_neural_training.py` — add `"cardinal_idx_cpu": cardinal_np` to `_prepare_tensors()` return dict; use it directly in the fold loop (`fold_cardinal_np = tensors["cardinal_idx_cpu"][train_idx]`)
  - **Success criterion:** Zero `cudaMemcpy(DeviceToHost)` calls during epoch loop in `nvprof`; training throughput unchanged.

- [x] **CU-2 Automatic Mixed Precision (AMP) training** — complexity: **medium** (~40 lines, 2 files)
  - **Gap:** All forward/backward passes are FP32. On Ampere/Hopper GPUs (RTX 3080+, A100), BF16 Tensor Core operations run 1.5–3× faster for the GEMM-heavy paths in SIREN, spatial attention, and fusion layers. `torch.compile` is already wired; AMP stacks on top for additional gain.
  - **Depends on:** CU-1 + CU-3 (larger batch sizes increase Tensor Core utilization for AMP; speedup scales with batch size)
  - **Critical dependency — sparse Laplacian guard (CU-6):** `sparc_joint_loss` calls `torch.sparse.mm(sparse_L, T_pred)`. COO sparse tensors do not support FP16 in all PyTorch versions; must wrap the entire `sparc_joint_loss` call in `autocast(enabled=False)` to keep PDE terms in FP32.
  - **Files:**
    - `sparc/run/v2_neural_training.py` — add `_use_amp` flag and `GradScaler`; wrap batch forward with `torch.autocast("cuda", dtype=torch.bfloat16, enabled=_use_amp)`; guard `sparc_joint_loss` with inner `autocast(enabled=False)`; replace `loss.backward()` with `_scaler.scale(loss).backward()` + `_scaler.unscale_` + `_scaler.step` + `_scaler.update`. Apply to CV fold, full-retrain, and SWA loops.
    - `sparc/training/optimizer.py::training_step` — add optional `scaler: GradScaler | None` param
  - **Success criterion:** On CUDA machine: training log shows `AMP active (bfloat16)`; epoch wall-clock drops ≥ 30% vs. FP32 on same hardware; `oof_predictions` R² within 0.5% of FP32 run (same seed); no `RuntimeError: expected scalar type Float but found Half`.

- [x] **CU-7 Stage 0 Matérn NUTS hardcodes `device="cpu"`** — complexity: **low** (1 line, 1 file)
  - **Gap:** `correlogram_matern_fit.py` line 307: `run_nuts(..., device="cpu")` — hardcoded. Stage 3's NUTS already has the correct pattern (`if torch.cuda.is_available() and str(device) == "cpu": device = torch.device("cuda")`). Stage 0's Matérn fitting runs 2 chains × 3 ν values = 6 NUTS runs; each with N=L (lag count, typically 20–50) and 3000 samples. These are small enough that CUDA overhead may not help, but the fix is trivial and consistent.
  - **File:** `sparc/run/correlogram_matern_fit.py` — before the `run_nuts` call at L297, add:
    ```python
    _nuts_device = "cuda" if torch.cuda.is_available() else "cpu"
    ```
    then pass `device=_nuts_device` instead of `device="cpu"`.
  - **Success criterion:** On CUDA machine: Stage 0 log shows "Matérn NUTS: using cuda"; `run_nuts` completes without error. On CPU-only machine: no change.

- [x] **CU-8 Stage 1 GWEN: cuML optional GPU path for ElasticNet + KNN** — complexity: **medium** (~30 lines, 1 file)
  - **Gap:** `GWENModel` uses `sklearn.ElasticNetCV` (CPU) and `sklearn.neighbors.NearestNeighbors` (CPU) for local model fitting. For N > 5k, the O(N²) local weight matrix and N local ElasticNet fits dominate Stage 1 runtime. `cuML` provides GPU-accelerated equivalents (`cuml.linear_model.ElasticNet`, `cuml.neighbors.NearestNeighbors`) with near-identical APIs. On a 10k-row dataset, cuML KNN is typically 20–50× faster than sklearn.
  - **Note:** `cuml` is an optional dep (requires RAPIDS, not always available). Must gate entirely on import availability.
  - **File:** `sparc/models/gwen.py` — in `GWENModel.fit()`, attempt `from cuml.neighbors import NearestNeighbors as _NearestNeighbors` + `from cuml.linear_model import ElasticNet as _ElasticNet`; fall back to sklearn silently. The `NearestNeighbors` interface is identical; ElasticNet loses `CV` (use fixed alpha from global model's `alpha_` — already done in `quick_mode`).
  - **Success criterion:** With `cuml` installed: Stage 1 log shows "GWEN: cuML GPU path active"; ElasticNet + KNN fit time drops ≥ 10× for N > 5k; `gwen_results.json` identical to sklearn path (same seed).

- [x] **BF-2 Replace Unicode math symbols in Stage 0 print statements with ASCII equivalents** — complexity: **low** (~6 lines, 2 files) — *Done 2026-05-22: Replaced `Matérn`→`Matern`, `κ`→`kappa`, `ν`→`nu` in correlogram_analysis.py print calls; replaced `–`→`-`, `±`→`+/-`, `°`→`deg` in anisotropy.py `_theta_to_compass` and `_make_dominant_direction_hint`. Root cause was PowerShell `2>&1` re-encoding CP1252 bytes through CP437 decoder (e.g. `\xe9` for `é` → CP437 Θ → UTF-8 `\xce\x98`). 15/15 tests pass.*
  - **Gap:** `correlogram_analysis.py` uses `é` (U+00E9), `κ` (U+03BA), and `ν` (U+03BD) in `print()` calls. On Windows, when output is redirected with PowerShell's `2>&1`, the CP1252 byte `\xe9` (for `é`) is decoded by PowerShell's OEM code page (CP437, where `\xe9` = Θ) and re-encoded as UTF-8 `\xce\x98`. Result: the output file shows `MatΘrn` instead of `Matérn`. Similarly `κ`/`ν` show as `?` (CP1252 replacement). Root cause: confirmed by `> file 2>&1` round-trip test — CP1252 byte `\xe9` → CP437 decode → UTF-8 re-encode → `\xce\x98` (Θ). Fix: replace with plain ASCII equivalents.
  - **File:** `sparc/run/correlogram_analysis.py` — in the two `print()` calls that output Matérn fit and anisotropy summaries: `Matérn` → `Matern`, `κ` → `kappa`, `ν` → `nu`
  - **Why safe:** Pure print-string substitution. No logic, no API, no artifact change. Output remains readable on all encodings.
  - **Success criterion:** On Windows with `sparc run -s 0 --fast > out.txt 2>&1`: `out.txt` shows `Matern fit (bayes): kappa=0.001891 [nu=1.5]` (no replacement chars or Θ); all `test_stage0_migration.py` tests pass.

- [x] **BF-1 Windows CP437/CP1252 encoding fix for Stage 0 summary lines** — complexity: **low** (12 lines, 1 file) — *Done 2026-05-22: Added `UnicodeEncodeError` fallback in `_VerbosityFilter._emit()` in `sparc/run/console.py`. When the downstream stream can't encode Unicode math symbols (κ, ν, ε₀), the line is re-encoded with `errors='replace'` (κ→?) so it appears rather than being silently dropped. Also removed an earlier `reconfigure(encoding='utf-8')` attempt that caused double-encoding in PowerShell pipes. Root cause: Windows CP437/CP1252 streams can't encode Greek Unicode; the old `except Exception: pass` silently swallowed UnicodeEncodeError, making the Matérn fit summary line invisible at NORMAL verbosity. 4/4 stage0 migration tests + 11 render tests pass.*
  - **File:** `sparc/run/console.py` — `_VerbosityFilter._emit()`: replace bare `except Exception: pass` with specific `except UnicodeEncodeError:` handler that retries `_downstream.write(safe)` where `safe = line.encode(enc, errors="replace").decode(enc)`.
  - **Success criterion:** On Windows: `sparc run -s 0 --fast` shows per-variable Matérn fit and anisotropy summary lines (with `?` for κ/ν); no silent line drops; 4/4 `test_stage0_migration.py` tests pass.

- [ ] **CU-9 CUDA Graph capture for the epoch step** — complexity: **high** *(synthesized sub-tasks below)*
  - [x] **CU-9a** `valid_mask` parameter in `sparc_joint_loss` — complexity: **low** — *Done 2026-05-21: Added `valid_mask: torch.Tensor | None = None` to `sparc_joint_loss`; masking applied in MSE, cross-entropy, alpha_prior, and neighborhood terms; PDE/physics terms unchanged; backward compat preserved; 20/20 tests pass.*
  - [x] **CU-9b** Static-batch padding helper + CUDA Graph capture in training loops — complexity: **medium** — *Done 2026-05-21: Added `_pad_batch_to_size(b_idx, target_size, device)` helper and `_capture_cuda_graph(mod, sample_inputs)` helper to `v2_neural_training.py`. Added `cuda_graphs: bool = False` field to `HardwareProfile` (overridable). Added `_use_cuda_graphs` flag after AMP setup. Applied padding + `valid_mask` in all 3 training loops (CV fold, full-retrain, SWA); `bsize`/`bsz` uses valid count for correct epoch-loss weighting. Gated behind `cuda_graphs=True` — zero behavior change by default. 20/20 tests pass.*
  - **Source:** derivatives.md "CUDA Graph Capture for Zero-Overhead Epoch Step"
  - **Gap:** `torch.compile` is applied to individual models but the full forward→loss→backward→step sequence still has ~50 kernel launches per batch due to Python dispatch overhead between models. `torch.cuda.make_graphed_callables` can capture the entire sequence as a single replay kernel, eliminating launch overhead (~15–30% additional throughput on top of AMP + compile).
  - **Blocker — static shapes required:** `spatial_minibatch_sampler` produces variable-size batches (the last batch per epoch is smaller than `batch_size`). CUDA Graphs require static input shapes. Strategy: pad last batch to full `batch_size` with a boolean `valid_mask` tensor; mask the loss summation. This adds 5–10 lines to the batch loop but is unavoidable.
  - **Depends on:** CU-1 (CUDA must be enabled), CU-2 (AMP — graphs capture BF16 regions correctly), CU-4 (no CPU sync inside graphed region)
  - **Files:**
    - `sparc/run/v2_neural_training.py` — extend `_maybe_compile` to optionally call `torch.cuda.make_graphed_callables` after compile; add `_pad_batch_to_size(batch_dict, target_size) -> (dict, mask)` helper; apply valid_mask in loss computation
    - `sparc/training/optimizer.py` — `training_step` accepts optional `valid_mask` for loss masking
  - **Success criterion:** `nsys profile` or `nvprof` shows single compound kernel per batch step vs. ~50 individual kernels; epoch wall-clock drops ≥ 15% on top of AMP baseline; output identical to non-graphed run (same seed).
  - **Note:** High complexity. Do not attempt before CU-1 + CU-2 + CU-4 are complete.

- [ ] **CU-10 Multi-GPU fold-parallel training via `DistributedDataParallel`** — complexity: **high** *(synthesized sub-tasks below)*
  - [x] **CU-10a** DDP spawn entrypoint — complexity: **medium** — Add `_ddp_fold_worker(rank, world_size, fold_idx, shared_tensors, result_queue)` in `v2_neural_training.py`; wrap fold loop with `torch.multiprocessing.spawn` when `gpu_count > 1`. Gate behind `cfg.ddp_enabled: bool`.
  - [x] **CU-10b** `dist.all_reduce` in EWC penalty accumulation — complexity: **low** — In `sparc/training/ewc.py`, wrap Fisher penalty accumulation with `dist.all_reduce(fisher, op=dist.ReduceOp.SUM)` when `dist.is_initialized()`.
  - [x] **CU-10c** `dist.all_reduce` in OT alignment loop — complexity: **low** — Same pattern as CU-10b for the optimal-transport penalty in the optimizer step; add `if dist.is_initialized(): dist.all_reduce(ot_loss, ...)` guard.
  - [x] **CU-10d** JEPA EMA broadcast from rank 0 — complexity: **low** — After each EMA trunk update, broadcast state dict from rank 0 to all ranks: `for p in ema_trunk.parameters(): dist.broadcast(p.data, src=0)`. Done: added `dist.broadcast` guard at all 3 EMA trunk update sites in `v2_neural_training.py` (cv fold, JEPA pretrain, full retrain).
  - [x] **CU-10e** Partitioned spatial minibatch sampler — complexity: **medium** — *Done 2026-05-22: Added `rank: int = 0` and `world_size: int = 1` to `spatial_minibatch_sampler` in `optimizer.py`. Global batch counter (`global_count`) increments for every accepted batch; rank `r` yields only batches where `global_count % world_size == r`, interleaving geographic regions across ranks. Safety cap scaled by `world_size`. Added `rank`/`world_size` to `_exec_cv_fold` signature; `_ddp_fold_worker` passes its rank/world_size through. Sequential fold loop uses defaults (rank=0, world_size=1) — zero behaviour change. 7 new tests added to `tests/test_spatial_minibatch_sampler.py`; all pass. Broader suite: 306/307 pass (pre-existing CMA-ES missing-package failure).*
  - **Source:** derivatives.md "Multi-GPU Fold-Parallel Training via DistributedDataParallel"
  - **Gap:** Each CV fold is fully independent (no cross-fold communication during training). With K GPUs, assigning fold `i` to GPU `i` gives K× training speedup for the CV phase. `HardwareProfile.gpu_count` (from CU-1) provides the worker count. For the full retrain, standard DDP with `batch_size * world_size` linear LR scaling.
  - **Key complications:**
    - EWC and OT alignment penalties need `dist.all_reduce` across ranks before the optimizer step
    - JEPA EMA trunk update must be synchronized (`dist.broadcast` from rank 0)
    - The spatial minibatch sampler needs a partitioned-N version (each rank receives a disjoint geographic block, natural from `spatial_kfold_enhanced` fold indices)
    - `torch.multiprocessing.spawn` entrypoint must be added to `v2_neural_training.py`
  - **Depends on:** CU-1 (`gpu_count` field), CU-5 (`empty_cache` between folds — critical in multi-process context)
  - **Files:**
    - `sparc/run/v2_neural_training.py` — add `_ddp_fold_worker(rank, world_size, fold_idx, shared_tensors, result_queue)` function; wrap fold loop with `torch.multiprocessing.spawn` when `gpu_count > 1`; collect OOF predictions from `result_queue`
    - `sparc/config/hardware_profile.py` — `gpu_count` (from CU-1)
    - `sparc/training/ewc.py` — add `dist.all_reduce` around Fisher penalty accumulation
    - `sparc/training/optimizer.py` — DDP-aware `clip_grad_norm_` (operates on local replica; all_reduce handled by DDP hooks)
  - **Success criterion:** On 2-GPU machine: CV training wall-clock ≤ 55% of single-GPU time (accounting for DDP init overhead); OOF R² within 0.2% of single-GPU run (same seed); both GPU processes show ≥ 60% utilization in `nvidia-smi`.
  - **Note:** High complexity. Do not attempt before CU-1 + CU-5 + CU-2 are complete and stable.



### Session Digest — 2026-05-20b (CUDA Research)

**Focus:** CUDA/GPU acceleration — full scan of `v2_neural_training.py`, `hardware_profile.py`, `optimizer.py`, and `server/app.py`.

**No implementations this session** — research/synthesis only.

**Key findings:**

**What works:** Basic CUDA device detection at L949; all models + tensors `.to(device)`; `torch.compile` wired; NUTS has an explicit CUDA switch in `v2_bayesian_causal.py`; `/api/hardware` endpoint exists.

**Critical gap — `HardwareProfile` ignores GPU (CU-1):** `force_cpu=True` for low RAM tier (< 12 GB) is unconditional. A machine with 8 GB RAM + RTX 3070 always trains on CPU. No `gpu_available`, `gpu_vram_gb`, or `gpu_name` fields anywhere in `hardware_profile.py`. Desktop cannot show GPU status.

**No AMP anywhere (CU-2):** Zero `autocast` / `GradScaler` usage in all training loops. All arithmetic is FP32 on CUDA. 1.5–3× throughput gain available on Ampere+ hardware. Requires sparse Laplacian guard (`autocast(enabled=False)` around `sparc_joint_loss`) to avoid COO sparse FP16 incompatibility.

**CPU-tuned batch sizes on GPU (CU-3):** `HardwareProfile.batch_size` sized for RAM (512/2048/4096). GPU SM occupancy is ~20–30% at batch 4096 on a 24 GB VRAM device.

**Minor items:** `fold_cardinal.cpu().numpy()` GPU→CPU sync every epoch (CU-4); no `empty_cache()` between folds (CU-5).

**Implementation order:** CU-5 → CU-1 → CU-3 → CU-4 → CU-2

**Priority recommendation for next implementation session:** `CU-5` (4 lines, zero risk) bundled with `CU-1` (40 lines, prerequisite for CU-2/CU-3) in a single PR.

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

---

### Session Digest — 2026-05-20 (Processing Page)

**Focus:** Desktop Processing page correctness and data-quality tooling.

**Critical bug confirmed:** `ProcessingPage.tsx::handleApplyAll` calls `prepareData({ raster_paths: [] })` → `POST /data/prepare` (spatial fishnet builder). Called with empty rasters it does nothing. All 8 step cards are immediately marked done with no actual CSV transformation. The 8-step UI is scaffolding only.

**Root cause:** `sparc/data/data_utils.py::prepare_data(df, config)` is the correct CSV transformation function but has no `POST /data/preprocess` API endpoint.

**Unreachable backend utilities identified:** `validation.py::validate_dataset` (endpoint exists, never called from UI), `welford.py` (not wired anywhere), Arrow/Parquet cache write (missing entirely), full versioning picker (only revert-to-original exposed).

**5 backlog items added:** P4-1 through P4-5 (P4-6 is a low-complexity derivative follow-on). Priority order: P4-1 (pipeline broken) → P4-2 (validate button, endpoint exists) → P4-3 (welford scaler) → P4-4 (arrow cache) → P4-5 (version picker).

---

### CLI Progress Tracking (from 2026-05-20d session)

**Focus:** User experience — understanding pipeline state and ETA during `sparc run`.

**Current gap confirmed:** Stage announcements are bare `print()` calls with no timing. Training epoch loop (`_exec_cv_fold`) logs via `logger.info()` — invisible during default `sparc run`. Fold loop has no tqdm bar. No stage-level ETA or elapsed time summary. `tqdm>=4.65` is already in `requirements.txt`.

**5 backlog items added (UX-1 through UX-5):**

- [x] **UX-1 Epoch-level tqdm bar in `_exec_cv_fold`** — complexity: **low** — *Done 2026-05-20: Added `_tqdm` epoch bar with EWM ETA (`_ema_epoch_s`), `set_postfix(loss, eta)` at end of each epoch, `disable=not sys.stdout.isatty()` for CI safety. Added `import sys` + `from tqdm import tqdm as _tqdm` to module-level imports.*
  - **What:** Replace bare `for epoch in range(n_epochs)` with `tqdm(range(n_epochs), desc=f"Fold {fold_idx+1} epochs", disable=not sys.stdout.isatty(), leave=False, unit="ep")`. Call `bar.set_postfix(loss=..., r2=..., curriculum=...)` each epoch.
  - **Also bundle (3 lines):** EWM-based per-epoch ETA: `ema_epoch_s = ema_epoch_s * 0.7 + epoch_dur * 0.3; bar.set_postfix(..., eta=f"{ema_epoch_s*(n_epochs-epoch-1)/60:.1f}m")` — dynamically-updating minutes remaining.
  - **File:** `sparc/run/v2_neural_training.py` — epoch loop inside `_exec_cv_fold` (line ~866)
  - **CI safe:** `disable=not sys.stdout.isatty()` — no output to pipes/CI runners
  - **Regression risk:** Zero — the loop body is unchanged; `tqdm` wraps the iterator only
  - **Success criterion:** During `sparc run`, terminal shows `Fold 1 epochs:  45%|████▌     | 45/100 [01:23<01:42, loss=0.0231, eta=1.7m]`

- [x] **UX-2 Fold-level tqdm bar in `train_neural_meta`** — complexity: **low** — *Done 2026-05-20: Replaced bare `enumerate(folds)` with `_tqdm(enumerate(folds), total=len(folds), ...)` in the sequential `else:` branch. `set_description` updates per fold. `set_postfix(r2=...)` shows per-fold R² (try/except guarded). DDP branch unchanged — bars are disabled there via `disable=not sys.stdout.isatty()`.*
  - **What:** Replace bare `for fold_idx, (train_idx, test_idx) in enumerate(folds)` with a `tqdm(enumerate(folds), total=len(folds), desc="CV Folds", ...)` outer bar. `bar.set_postfix(r2=f"{fold_r2:.3f}")` after each fold.
  - **File:** `sparc/run/v2_neural_training.py` — sequential fold loop (line ~2650)
  - **Success criterion:** Terminal shows `CV Folds:  67%|██████▋   | 2/3 [04:12<02:06, r2=0.847]`

- [x] **UX-3 `StageProgress` module (`sparc/run/progress.py`)** — complexity: **low** — *Done 2026-05-20: Created `sparc/run/progress.py` with `StageProgress(total_stages)` class (`stage_start`, `stage_done`, `finish`, `print_eta_hint`). Wired into `sparc/__main__.py` `cmd_run`: `_sp_stages` list computed from stage arg + flags; `_sp.stage_start/done` wraps each stage dispatch block; `_sp.finish()` replaces final print. 345 tests pass, no regressions.*
  - **What:** New `sparc/run/progress.py` — `StageProgress` class with `stage_start(key, label)`, `stage_done(key)`, and `finish()` methods. tqdm outer bar in TTY, timestamped `print()` in pipes. Displays `[N/6]` stage count, overall elapsed time, and per-stage completion time.
  - **Also:** Wire into `sparc/__main__.py` `cmd_run` — call `StageProgress.stage_start/done` around each stage dispatch block.
  - **Files:** new `sparc/run/progress.py`, `sparc/__main__.py` (~20 line change)
  - **CI safe:** `disable=not sys.stdout.isatty()` on the tqdm bar; `print()` falls through when not TTY
  - **Success criterion:** Terminal shows pipeline bar `SPARC | Stage 2: Spatial CV + Neural Training  ████░░░░  2/6 [00:45<03:20]`; after run: `Pipeline complete — total 248.3s (4.1 min)` with per-stage breakdown

- [x] **UX-4 Dataset-tier-aware pre-run ETA hint** — complexity: **low** — *Done 2026-05-20: Added `print_eta_hint(size_tier, has_gpu)` call in `__main__.py` Stage 0b block immediately after `print(f"  Dataset tier: ...")`. GPU check uses `torch.cuda.is_available() or torch.backends.mps.is_available()` (try/except guarded). `_ETA_TABLE` already present in `progress.py` from UX-3.*
  - **What:** After Stage 0 `dataset_profile.json` is read (the tier already printed as `Dataset tier: MEDIUM`), call `print_eta_hint(size_tier, has_gpu)` from a lookup table in `sparc/run/progress.py`. E.g.: `[ETA hint] Stage 2 estimated ~8 min (medium dataset, CPU)`.
  - **Lookup table:** `{("small",False):90, ("small",True):30, ("medium",False):480, ("medium",True):120, ("large",False):1800, ("large",True):360, ("xlarge",False):5400, ("xlarge",True):900}` — conservative empirical estimates, adjustable.
  - **Files:** `sparc/run/progress.py` + `sparc/__main__.py` Stage 0b block
  - **Success criterion:** After Stage 0b, terminal shows `[ETA hint] Stage 2 estimated ~8 min (medium dataset, CPU)`

- [x] **UX-5 Stage timing persistence to artifact store** — complexity: **low** — *Done 2026-05-20: Added `timing_sink: Callable[[str, float], None] | None` parameter to `StageProgress.__init__`; called from `stage_done()` with `(key, elapsed_s)`. In `__main__.py`, `_stage_timing_sink()` closure writes `{stage, elapsed_s, timestamp}` to artifact store via `write_struct(key, "stage_timing", ...)`. All 7 stages auto-persist elapsed times to `artifacts.db`.*
  - **What:** In `StageProgress.stage_done()` (or `_mark_stage_done()`), write `{stage, elapsed_s, timestamp}` to artifact store via `_store.write_struct(stage_key, "stage_timing", {...})`. This persists stage runtimes alongside completion status — viewable in desktop registry, enables future per-machine ETA refinement.
  - **File:** `sparc/__main__.py` `_mark_stage_done()` helper or `sparc/run/progress.py`
  - **Success criterion:** After `sparc run`, `artifacts.db` contains `stage_timing` structs for all completed stages; `sparc registry --list` shows per-stage runtimes

**Priority order:** UX-1 (epoch bar, highest pain point) → UX-2 (fold bar) → UX-3 (stage wrapper module) → UX-4 (ETA hint) → UX-5 (timing persistence)

**Self-grill verdict:** All five items use only `tqdm` (already required) with zero new dependencies. UX-1 and UX-2 are the highest-pain-point / lowest-risk changes. CI safety is guaranteed by `disable=not sys.stdout.isatty()` throughout. No regression risk on any existing test.

---

### Decision Intelligence (from 2026-05-20 synthesis)

**Source derivatives:** `Causal Bandits for Sequential Intervention Design` (new → in-backlog) and `Latent Diffusion for Posterior Scenario Sampling` (new → in-backlog).

**Priority ranking:** CB-1 (Score 4.0) → CB-2 (Score 2.0) → LD-1 (Score 1.0, high complexity — do not attempt before CB-1 + CB-2 stable)

- [x] **CB-1 Add `sample_posterior()` to `CATEGPSurface`** — complexity: **low** — *Done 2026-05-20: Added 5-line `sample_posterior(n_samples, coords_norm)` method to `CATEGPSurface` after `predict()` in `sparc/causal/spatial_cate.py`. Delegates to `self._gpr.sample_y(...)`. Two tests added to `tests/test_bayesian_cate.py`: shape check + pre-fit RuntimeError guard. All 7 tests passed.*
  - Files: `sparc/causal/spatial_cate.py` — add `sample_posterior(n_samples, coords_norm) -> np.ndarray` method to `CATEGPSurface` after the `predict()` method
  - Sketch:
    ```python
    def sample_posterior(self, n_samples: int, coords_norm: np.ndarray) -> np.ndarray:
        """Draw n_samples from the GP posterior at coords_norm. Shape: (N, n_samples)."""
        if self._gpr is None:
            raise RuntimeError("fit() must be called before sample_posterior()")
        coords_norm = np.atleast_2d(np.asarray(coords_norm, dtype=np.float64))
        return self._gpr.sample_y(coords_norm, n_samples=n_samples, random_state=None)
    ```
  - Why: Exposes Thompson sampling from the fitted CATE GP posterior — the arm reward distribution for `CausalBandit`.
  - Depends on: none (`CATEGPSurface` is already built and wired in Stage 3)
  - Success: `surf.sample_posterior(10, coords_norm).shape == (N, 10)`; sample values span > 1 posterior std; test added to `tests/test_bayesian_cate.py`

- [x] **CB-2 Implement `CausalBandit` for multi-year intervention planning** — complexity: **medium** — *Done 2026-05-20: Created `sparc/decision/causal_bandit.py` with `CausalBandit` class (Thompson-sampling via `sample_posterior`). Wired into `sparc/decision/__init__.py` export. 6 tests added to `tests/test_causal_bandit.py`: API shape, history consistency, arm-total accounting, weak-regret check, empty-surfaces guard, package import. All 6 passed.*
  - Files: new `sparc/decision/causal_bandit.py` — `CausalBandit` class; `sparc/decision/__init__.py` — add export
  - Sketch:
    ```python
    class CausalBandit:
        """Thompson-sampling bandit over CATE GP surfaces for multi-round intervention design."""
        def __init__(self, cate_surfaces: dict[str, CATEGPSurface],
                     simulator: ScenarioSimulator, budget: float, n_rounds: int):
            self._surfaces = cate_surfaces
            self._simulator = simulator
            self._budget = budget
            self._n_rounds = n_rounds
            self._history: list[dict] = []

        def run(self, coords_norm: np.ndarray) -> dict:
            welfare_accum = {arm: 0.0 for arm in self._surfaces}
            for t in range(self._n_rounds):
                # Thompson sample: draw one posterior realization per arm
                arm_rewards = {
                    arm: float(surf.sample_posterior(1, coords_norm)[:, 0].mean())
                    for arm, surf in self._surfaces.items()
                }
                chosen = max(arm_rewards, key=arm_rewards.__getitem__)
                welfare_accum[chosen] += arm_rewards[chosen]
                self._history.append({"round": t, "arm": chosen, "sampled_reward": arm_rewards[chosen]})
            return {
                "history": self._history,
                "arm_totals": welfare_accum,
                "recommended_sequence": [h["arm"] for h in self._history],
                "baseline_welfare": self._greedy_baseline(coords_norm),
            }

        def _greedy_baseline(self, coords_norm):
            """One-shot greedy pick (EmpiricalWelfareMaximizer equivalent)."""
            means = {arm: float(surf.predict(coords_norm)[0].mean())
                     for arm, surf in self._surfaces.items()}
            return max(means, key=means.__getitem__)
    ```
  - Why: Achieves sublinear cumulative regret (GP-UCB; Srinivas et al. 2012) vs. greedy one-shot `EmpiricalWelfareMaximizer`; enables optimal multi-year climate intervention sequences under uncertainty.
  - Depends on: CB-1 (`sample_posterior()` on `CATEGPSurface`)
  - Success: `CausalBandit(surfaces, simulator, budget=1e6, n_rounds=5).run(coords)` returns ranked arm sequence and `baseline_welfare`; `tests/test_causal_bandit.py` verifies cumulative regret < greedy over 20 rounds on synthetic CATE surface (5 arms, known optimum)

- [x] **LD-1a `LatentScenarioDiffuser` DDPM model class** — complexity: **low** (~110 lines, 1 new file). Done: `sparc/models/scenario_diffuser.py` created with DDPM class; `tests/test_scenario_diffuser.py` — 5/5 unit tests pass.
  - Files: new `sparc/models/scenario_diffuser.py` — full DDPM class; new `tests/test_scenario_diffuser.py` — unit tests
  - Sketch: `LatentScenarioDiffuser(trunk_dim=256, bottleneck=32, cond_dim=0, T=200)`. `cond_dim=0` = unconditional mode (disables `cond_proj`). `loss(z0, cond=None)`: project trunk to bottleneck, forward diffuse with sampled `t`, predict noise with 3-layer SiLU MLP, return MSE. `sample(cond, n_samples)`: reverse DDPM loop, project back up via `proj_up`, return `(n_samples, trunk_dim)`. Registers `betas` and `alpha_bars` as buffers.
  - Why: Self-contained DDPM prior over the trunk latent space — zero codebase dependencies enables isolated testing before wiring into training or inference
  - Depends on: none
  - Success: `loss()` is scalar and finite; `sample()` shape `(n_samples, trunk_dim)`; loss decreases after 5 gradient steps; unconditional and conditional modes both pass unit tests

- [x] **LD-1b Wire diffuser training hook into `v2_neural_training.py`** — complexity: **low** (~55 lines, 1 file modified). Done: `_pretrain_diffuser()` added to `v2_neural_training.py`; gated by `pretrain_epochs>0`; saves `.pt` + sidecar `scenario_diffuser_config.json`.
  - Files: `sparc/run/v2_neural_training.py` — add private `_pretrain_diffuser(model, tensors, config, device, artifact_dir, store)` function + single call site after the checkpoint save block (~line 3493); no changes to existing training paths
  - Sketch: Gated by `config.get("diffuser", {}).get("pretrain_epochs", 0) > 0`. Uses `final_model.encode(physics_feats_t, alpha_t)` (already exists on `SPARCMetaLearner`) to get `h_trunk (N, hidden_dim)`. Builds `LatentScenarioDiffuser(trunk_dim=model.hidden_dim, cond_dim=n_physics_features)`. Trains for N epochs with Adam. `cond_feats = physics_feats_t.mean(0, keepdim=True).expand(N, -1)`. Saves: `torch.save(state, artifact_dir / "scenario_diffuser.pt")` + `store.write_blob("2", "v2_diffuser_state", state, serializer="torch")`.
  - Why: Re-uses `model.encode()` — no new trunk extraction infrastructure needed; the entire hook is dead code when `pretrain_epochs=0`
  - Depends on: LD-1a
  - Success: With `pretrain_epochs=2`, `scenario_diffuser.pt` is created and loadable; with `pretrain_epochs=0` (default), no file is written; no changes to existing test suite

- [x] **LD-1c `ScenarioSimulator.run_with_diffusion_posterior()`** — complexity: **medium** (~55 lines, 1 file modified). Done: method added to `scenario_simulator.py`; reads sidecar config to reconstruct correct architecture before `load_state_dict`.
  - Files: `sparc/interventions/scenario_simulator.py` — new public method appended after `run_with_uncertainty` (line ~3721)
  - Sketch: `run_with_diffusion_posterior(self, data, n_samples=50) -> dict`. Raises `FileNotFoundError` with clear message if `artifact_dir / "scenario_diffuser.pt"` absent. Prepares tensors once via existing helper. `cond = model.encode(physics_feats_t, alpha_t).mean(0, keepdim=True).expand(N, -1)`. `trunk_samples = diffuser.sample(cond, n_samples)` → `(n_samples, hidden_dim)`. For each `i`: expand to `(N, hidden_dim)`, call `model.decode(h_trunk_i, base_preds, X_spatial, coords, knn_index)` → `T_pred_i (N,)` (de-normalize). Stack → `(n_samples, N)` → `{mean, p05, p95}` via `np.percentile`.
  - Why: Spatially coherent posterior credible intervals vs. independent per-coefficient MC draws in `run_with_uncertainty`; `model.decode()` already exists on `SPARCMetaLearner`
  - Depends on: LD-1a, LD-1b
  - Success: Returns `{mean, p05, p95}` each shape `(N,)`; `(p95 - p05).mean() > 0` (non-degenerate); `tests/test_diffuser_integration.py` passes with mock model + synthetic diffuser checkpoint

- [x] **LD-1d Config schema + end-to-end smoke test** — complexity: **low** (~30 lines, 2 files). Done: `diffuser` defaults added to both config paths in `config.py`; `tests/test_diffuser_integration.py` — 6/6 integration tests pass.
  - Files: `sparc/config/` (default config dict) — add `"diffuser": {"pretrain_epochs": 0, "bottleneck": 32, "T": 200}` to default config so `sparc run` without any diffuser config never errors; `tests/test_diffuser_integration.py` — full chain smoke test
  - Sketch: Smoke test builds tiny `SPARCMetaLearner(hidden_dim=32, n_base_models=2, n_physics_features=4, d_spatial=8)`, fabricates random trunk features, runs `_pretrain_diffuser` for 2 epochs into a `tmp_path`, loads checkpoint, calls `run_with_diffusion_posterior` on mocked `ScenarioSimulator`, asserts output shapes and that `p95 > p05` at mean.
  - Why: Proves the full chain (train hook → checkpoint → simulator method) wires correctly before a real `sparc run` touches it
  - Depends on: LD-1a, LD-1b, LD-1c
  - Success: `sparc run` with no `diffuser` key in config runs without `KeyError`; smoke test passes; `scenario_diffuser.pt` is written and reloaded correctly in the test

---

### Session Digest — 2026-05-21b (Stage 3 Audit)

**Focus:** Root-cause analysis of Stage 3 appearing to "hang" after "Discovery report saved";
DML appropriateness audit; four targeted bug fixes implemented.

**Root cause:** Stage 3 does NOT hang — it runs correctly but produces **zero visible output** at
NORMAL verbosity between "Discovery report saved" and pipeline exit. Every MC³ progress line,
NUTS warmup/sampling banner, structural coefficient header, and DAG diagnostics header fell
through all patterns in `console.py::classify_line()` to **default DEBUG**.

**Latent bug found:** `counterfactual_engine.py::_fit_edge_dml()` called
`self._fit_edge_dml_sklearn(...)` where `self` was a `CausalValidator` instance — which does not
inherit `CounterfactualEngine`. The econml fallback path would raise `AttributeError` on any
machine without the econml Cython extension.

**DML verdict:** DML is appropriate for the 54 k-row Brown University dataset. Per-edge HGB
fitting with 5-fold CV + 7 DAG edges ≈ 3–5 min total — well within acceptable range. `estimator`
switched back to `"dml"` in `project.yml`.

**4 fixes implemented (this session):**
1. `sparc/run/console.py` — added 12 CHECKPOINT patterns for MC³/NUTS/Stage3 milestones + 1 explicit DEBUG pattern for per-sample NUTS ticks
2. `sparc/run/v2_bayesian_causal.py` — added MC³ completion print, NUTS completion print, per-edge NUTS progress counter; reduced NUTS defaults (15k→6k, 3k→1.5k samples)
3. `sparc/causal/counterfactual_engine.py` — fixed DML fallback AttributeError; lightened HGB nuisance model (200/4 → 100/3 iter/depth)
4. `project.yml` — `estimator: "ols"` → `"dml"`

**3 new backlog items added (S3-1 through S3-3 below).**

**Priority recommendation for next implementation session:** `S3-1` (spatial block CV for DML,
complexity low, highest-impact improvement, ~15 lines; prerequisite: confirm Stage 0
`effective_range_matrix` artifact is present).

---

### Phase S3 — Stage 3 Quality Improvements (from 2026-05-21b session)

- [x] **S3-1 Spatial block CV for DML folds** — complexity: **low** (~15 lines, 1 file) — *Done 2026-05-21: Added auto-detection block in `fit()` (counterfactual_engine.py lines 341-362): when estimator is DML and `causal.spatial_block_size` is not configured, loads `effective_range_matrix` from active ArtifactStore, flattens non-None values, takes np.percentile(range_values, 10) as block_size, logs INFO. Graceful fallback to random K-fold when artifact unavailable.*
  - **Gap:** `counterfactual_engine.py::_fit_edge_dml()` passes `spatial_block_size=None` by
    default → random K-fold ignores spatial autocorrelation. DML coefficient standard errors
    are anti-conservative when spatial autocorrelation inflates train/test dependence.
  - **Academic source:** Valavi et al. (2019) "blockCV"; Chernozhukov et al. (2018) DML —
    cross-fitting requires independent folds.
  - **Implementation:** In `_fit_edge_dml`, load `effective_range_matrix` from artifact store
    (`_get_store().read_struct("0", "effective_range_matrix")` if store is available); set
    `spatial_block_size = np.percentile(range_values, 10)` (MER heuristic); pass to
    `SpatialBlockKFold` (already wired in `enhanced_spatial_cv.py`).
  - **File:** `sparc/causal/counterfactual_engine.py` — `_fit_edge_dml()` (~15 lines)
  - **Dependency:** Stage 0 must have been run to populate `effective_range_matrix`. Degrades
    gracefully to random K-fold when artifact store is unavailable (existing behavior).
  - **Success criterion:** With Stage 0 artifact present, DML log shows "spatial block CV
    activated (block_size=NNm)"; DML coefficients change by < 5% from random K-fold
    (correctness check); no regression on existing DML tests.

- [x] **S3-2 Per-edge NUTS ESS adaptive budget** — complexity: **low** (~20 lines, 1 file) — *Done 2026-05-21: Inserted 38-line adaptive block in `_run_per_edge_nuts_sampling()` (v2_bayesian_causal.py) immediately after ESS is computed. When `ess < 200` and `n_samples < 4000`, logs WARNING, prints `[WARN]` banner, re-runs with `min(n_samples*2, 4000)` draws (seed+1), and overwrites res/beta_chain/rh/es. Falls through to summary_rows.append with updated stats.*
  - **Gap:** `_run_per_edge_nuts_sampling()` uses a fixed per-edge budget regardless of chain
    ESS. Collinear edges (NDVI ↔ Pct_Canopy) can have high autocorrelation → low ESS even
    at 1,500 samples → misleading narrow posteriors in `scenario_coefficients.json`.
  - **Academic source:** Hoffman & Gelman (2014) NUTS; Vehtari et al. (2021) rank-normalized R-hat
  - **Implementation:** After `run_nuts()` returns, check `ess = np.mean([v.ess for v in res.diagnostics])`. If `ess < 200` and `n_samples < 4000`, re-run with `n_samples=min(n_samples*2, 4000)` and log a `[WARN]`.
  - **File:** `sparc/run/v2_bayesian_causal.py` — inside `_run_per_edge_nuts_sampling()` (~20 lines)
  - **Dependency:** None — purely additive.
  - **Success criterion:** On a synthetic collinear dataset, log shows "Per-edge NUTS: low ESS
    (N), doubling budget for {parent}→{child}"; final ESS > 200; no regression.

- [x] **S3-3 DML–NUTS sign coherence check** — complexity: **low** (~35 lines, 2 files) — *Done 2026-05-21: Added 52-line sign coherence block in `run_bayesian_causal()` (v2_bayesian_causal.py) after `_run_nuts_sampling()` returns. Reads `all_structural_coefficients` from `output_dir/scenario_coefficients.json`; reads NUTS per-edge samples from `_store.read_blob("3","nuts_edge_samples")`; computes `P(wrong_sign|NUTS)` per edge; logs WARNING and prints `[WARN]` CHECKPOINT banner for edges where `p_wrong > 0.10`; writes `("3","sign_coherence")` struct via ArtifactStore. `causal_validation.py` required no changes — coefficients already in `scenario_coefficients.json`.*
  - **Gap:** `validator.fit()` produces a frequentist DML point estimate per edge; NUTS produces
    a Bayesian posterior. These are never compared. Sign discordance (DML β < 0, NUTS mean > 0)
    would silently produce incoherent scenario outputs downstream.
  - **Academic source:** Dawid (1982) coherence; practical: `P(sign discordance) = P(DML * NUTS_mean < 0)`.
  - **Implementation:** After `_run_per_edge_nuts_sampling()` returns, for each edge with both
    DML and NUTS posteriors, compute `P(β < 0 | NUTS)` (fraction of posterior < 0). If DML
    coefficient is positive but `P(β < 0 | NUTS) > 0.10`, log `[WARN sign discordance]` and
    write result to `("3", "sign_coherence")`.
  - **Files:** `sparc/run/v2_bayesian_causal.py` (~30 lines), `sparc/run/causal_validation.py`
    (~5 lines to pass DML coefficients to `run_bayesian_causal()`).
  - **Dependency:** DML coefficients must be available in artifact store by time NUTS completes.
    Currently they are: `validator.fit()` runs before `run_bayesian_causal()` and writes to
    `("3", "structural_coefficients")`.
  - **Success criterion:** On a dataset with induced sign discordance, `[WARN sign discordance]`
    appears in log for the affected edge; `sign_coherence.json` present in Stage 3 output;
    no regression on normal data.

- [x] **S3-4 Full DAG posterior intervention distributions** — complexity: **medium** (~80 lines, 2 files) — *Done 2026-05-21: Added `sample_dags_from_edge_probs()` to `sparc/causal/mc3.py` (~35 lines, Bernoulli sampling + Kahn cycle check). Added `_compute_dag_posterior_effects()` module-level helper to `v2_bayesian_causal.py` (~33 lines, outer-product of DAG edge indicator × thinned β draws). Added `"_beta_chain"` and `"_target_col"` keys to `nuts_summary` return dict. Wired call site in `run_bayesian_causal()` after sign-coherence block; writes `("3","dag_posterior_effects")` struct. Entire block non-fatal (try/except). 34/34 tests pass.*
  - **Source derivative:** "Structural Causal Model Intervention Distributions via Full DAG Posterior" (2026-05-21b)
  - **Gap:** Stage 3 uses the single *median probability DAG* (edge inclusion prob ≥ 0.5) for all downstream scenario propagation and NUTS coefficient estimation. Structural uncertainty — the probability mass assigned to alternative graph structures by MC³ — is silently discarded. A user asking "what is the effect of 10% canopy increase?" gets a point estimate (beta_mean), not a credible interval that reflects both coefficient uncertainty (NUTS) *and* structural uncertainty (MC³).
  - **Academic source:** Pearl (2000) *Causality* do-calculus; Bareinboim & Pearl (2016) causal inference and the data-fusion problem; Madigan & Raftery (1994) Bayesian model averaging over graphical models.
  - **Files to touch:**
    - `sparc/causal/mc3.py` — new standalone function `sample_dags_from_edge_probs(edge_probs, node_names, n_samples=50, max_attempts=2000, seed=42) -> list[np.ndarray]`; uses independent Bernoulli sampling per edge + Kahn's cycle detection O(V+E) — returns list of valid DAG adjacency matrices
    - `sparc/run/v2_bayesian_causal.py` — (1) add `_beta_chain` key to `_run_nuts_sampling()` return dict (`beta_chain_orig` already in scope); (2) new helper `_compute_dag_posterior_effects(sampled_dags, beta_chain, available_cols, treatments, target_col) -> dict` (cross-products all DAG samples × thinned beta draws); (3) call after NUTS block, save `("3", "dag_posterior_effects")` to store; wrap in try/except — non-fatal
  - **Implementation sketch:**
    ```python
    # mc3.py
    def sample_dags_from_edge_probs(
        edge_probs: np.ndarray, node_names: list[str],
        n_samples: int = 50, max_attempts: int = 2000, seed: int = 42,
    ) -> list[np.ndarray]:
        from collections import deque
        rng = np.random.default_rng(seed)
        p = edge_probs.shape[0]; dags = []
        for _ in range(max_attempts):
            if len(dags) >= n_samples: break
            adj = rng.random((p, p)) < edge_probs
            np.fill_diagonal(adj, False)
            in_deg = adj.sum(axis=0).copy()
            q = deque(i for i in range(p) if in_deg[i] == 0); visited = 0
            while q:
                u = q.popleft(); visited += 1
                for v in np.where(adj[u])[0]:
                    in_deg[v] -= 1
                    if in_deg[v] == 0: q.append(v)
            if visited == p: dags.append(adj.astype(bool))
        return dags

    # v2_bayesian_causal.py — _compute_dag_posterior_effects
    def _compute_dag_posterior_effects(sampled_dags, beta_chain, cols, treatments, target_col):
        target_idx = cols.index(target_col)
        n_thin = max(1, len(beta_chain) // 200)  # thin to ~200 beta draws
        results = {}
        for t in treatments:
            if t not in cols: continue
            t_idx = cols.index(t)
            dag_t_probs = [float(d[t_idx, target_idx]) for d in sampled_dags]
            betas = beta_chain[::n_thin, treatments.index(t)]
            # Outer product: n_dags × n_beta_draws — each cell = dag_present * beta
            effects = np.outer(dag_t_probs, betas).ravel()
            results[t] = {"mean": float(effects.mean()), "ci5": float(np.percentile(effects, 5)),
                          "ci95": float(np.percentile(effects, 95)),
                          "edge_inclusion_mean": float(np.mean(dag_t_probs)),
                          "n_dag_samples": len(sampled_dags)}
        return results
    ```
  - **Why it improves SPARC:** First SPARC output that propagates both structural uncertainty (MC³ graph posterior) and coefficient uncertainty (NUTS β posterior) into a single credible interval — makes the causal claim formally complete as a Bayesian SCM.
  - **Dependencies:** None. Purely additive — the median-DAG path is unchanged; the new path is additive output. `edge_probs` matrix and `beta_chain_orig` are both available at the insertion point.
  - **Success criterion:** After `sparc run`, artifact `("3", "dag_posterior_effects")` exists with per-treatment `{"mean", "ci5", "ci95", "edge_inclusion_mean", "n_dag_samples"}`; CI width > 0 on any dataset with NUTS convergence; no change to existing scenario simulator output; pipeline runtime increase < 30 s (DAG sampling + outer product are fast).

- [x] **S3-5 NUTS-conditioned GP CATE posterior surface** — complexity: **medium** (~70 lines, 2 files) — *Done 2026-05-21: Added `nuts_conditioned_cate_surface()` to `sparc/causal/spatial_cate.py` (~85 lines). Wired in `run_bayesian_causal()` (~55 lines, after DAG posterior block). Reads `_beta_chain` from nuts_results (added S3-4). Normalises coords, thins to 50 β draws, fits CATEGPSurface per draw, stacks → {cate_mean, cate_ci5, cate_ci95}. Optionally seeds GP kernel from ("0","correlogram_matern_fit"). Writes ("3","nuts_gp_cate") as pickle blob. Non-fatal. 34/34 tests pass.*
  - **Source derivative:** "Causal CATE Posterior Surface via NUTS-Conditioned Spatial GP" (2026-05-21b)
  - **Gap:** The existing `CATEGPSurface` (fitted in `fit_gp_surface()`) uses only the *posterior-mean* CATE as fitting targets — discarding all coefficient uncertainty from NUTS. The existing `BayesianSpatialCATE` captures spatial uncertainty via random-RBF features but is a separate estimator path. Neither path propagates *structural* NUTS β uncertainty (from the `_run_nuts_sampling` global β model) through GP spatial smoothing to give a per-pixel credible interval CATE map that combines parameter uncertainty + spatial interpolation uncertainty.
  - **Academic source:** Kennedy & O'Hagan (2001) Bayesian calibration; Stein (2012) interpolation of spatial data; Chernozhukov et al. (2018) DML.
  - **Files to touch:**
    - `sparc/causal/spatial_cate.py` — new module-level function `nuts_conditioned_cate_surface(nuts_beta_samples, treatment_values, coords, treatment_names, corr_payload=None, n_posterior_draws=50, max_fit_rows=2000, seed=42) -> dict[str, dict]`; ~65 lines; uses existing `CATEGPSurface` machinery
    - `sparc/run/v2_bayesian_causal.py` — after NUTS block: add `"_beta_chain": beta_chain_orig` to `_run_nuts_sampling()` return dict; call `nuts_conditioned_cate_surface()` gated on `len(coord_cols) >= 2`; save `("3", "nuts_gp_cate")` to store; wrap in try/except — non-fatal
  - **Implementation sketch:**
    ```python
    # spatial_cate.py
    def nuts_conditioned_cate_surface(
        nuts_beta_samples: np.ndarray,   # (D, n_treatments)
        treatment_values: np.ndarray,    # (N, n_treatments) raw treatment cols
        coords: np.ndarray,              # (N, 2)
        treatment_names: list[str],
        corr_payload: dict | None = None,
        n_posterior_draws: int = 50,
        max_fit_rows: int = 2_000,
        seed: int = 42,
    ) -> dict[str, dict[str, np.ndarray]]:
        rng = np.random.default_rng(seed)
        D = len(nuts_beta_samples)
        draw_idx = np.arange(0, D, max(1, D // n_posterior_draws))[:n_posterior_draws]
        coords_min = coords.min(0); coords_range = coords.max(0) - coords_min
        coords_range[coords_range == 0] = 1.0
        coords_norm = (coords - coords_min) / coords_range
        results = {}
        for t_idx, tname in enumerate(treatment_names):
            T_col = treatment_values[:, t_idx]
            surfaces = np.empty((len(draw_idx), len(T_col)))
            for k, d in enumerate(draw_idx):
                beta_d = float(nuts_beta_samples[d, t_idx])
                cate_d = beta_d * T_col   # local structural CATE at each point
                gps = (CATEGPSurface.from_correlogram(corr_payload, max_fit_rows=max_fit_rows)
                       if corr_payload else CATEGPSurface(max_fit_rows=max_fit_rows))
                gps.fit(cate_d, coords_norm)
                surfaces[k], _ = gps.predict(coords_norm)
            results[tname] = {"cate_mean": surfaces.mean(0),
                              "cate_ci5": np.percentile(surfaces, 5, 0),
                              "cate_ci95": np.percentile(surfaces, 95, 0),
                              "n_posterior_draws": len(draw_idx)}
        return results
    ```
  - **Why it improves SPARC:** Produces per-pixel `(CATE_mean, CATE_ci5, CATE_ci95)` continuous rasters that propagate both NUTS coefficient uncertainty and GP spatial interpolation uncertainty — the output is a genuine posterior predictive credible interval map for each treatment's local effect.
  - **Dependencies:** S3-4 shares the `_beta_chain` return value addition in `_run_nuts_sampling` — implement the `_beta_chain` dict key as part of either S3-4 or S3-5 (both reference the same 1-line change).
  - **Success criterion:** After `sparc run`, artifact `("3", "nuts_gp_cate")` exists with per-treatment `{"cate_mean": (N,), "cate_ci5": (N,), "cate_ci95": (N,)}`; `(cate_ci95 - cate_ci5).mean() > 0` (non-degenerate CI); `cate_mean` correlates r > 0.9 with `beta_mean * T_values` (sanity check that GP posterior mean ≈ structural prediction); pipeline runtime increase < 2 min for N ≤ 5 000.

