# Tasks: Reasoning Engine Spine + Stage 2 Surrogate Inner Loop

Related PRD: [docs/prd/prd-reasoning-engine-spine.md](../prd/prd-reasoning-engine-spine.md)
Branch: `feat/reasoning-engine-spine`

State legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked

---

## Phase 0 — Setup

- [ ] **0.1** Add `scikit-optimize>=0.9` to `requirements.txt` and `pyproject.toml`; verify install succeeds in the dev venv.
- [ ] **0.2** Verify `pydantic>=2` is already required (used by `sparc/registry/manifest.py`) — no action if true.

## Phase 1 — Decision schema (`sparc/run/decisions.py`)

- [ ] **1.1** Create `sparc/run/decisions.py` with the Pydantic models from PRD §Technical Approach: `MetricEvaluation`, `DecisionRecord`, `NextAction`, `ProvenanceInfo`, `StageDecision`.
- [ ] **1.2** Add helpers: `StageDecision.to_artifact_struct()` and `StageDecision.from_artifact_struct()` for round-trip with `artifacts.db`.
- [ ] **1.3** Add `tests/test_stage_decision_schema.py`:
  - schema validation rejects bad input
  - JSON round-trip preserves all fields
  - `schema_version` defaults to 1
  - `metrics["r2"].passed` flips correctly when threshold crossed

## Phase 2 — Orchestrator skeleton (`sparc/run/orchestrator.py`)

- [ ] **2.1** Create `sparc/run/orchestrator.py` with `run_stage(stage, config, *, fast, skip_gwen, max_revisions=2) -> StageDecision`.
- [ ] **2.2** Move the stage if/elif dispatch out of `sparc/server/stream.py::_execute_stage` into `orchestrator.py::_dispatch_stage`. Keep behavior identical (no scoring yet).
- [ ] **2.3** After dispatch, auto-populate `StageDecision.metrics` from each stage's return dict:
  - Stage 0: spatial extent, n predictors analyzed, mean κ
  - Stage 1: n features selected, GWEN best λ
  - Stage 2: per-model OOF R², RMSE, residual Moran's I; meta-learner R²/RMSE
  - Stage 3: MC³ acceptance rate, NUTS R-hat max, n divergences
  - Stage 4: scenario count, mean predicted delta
- [ ] **2.4** Persist `StageDecision` as a struct artifact in `artifacts.db` via `get_active_store().write_struct(stage, f"stage_{stage}_decision", ...)`.
- [ ] **2.5** Modify `sparc/server/stream.py::_execute_stage` to delegate: `decision = orchestrator.run_stage(...)` and return.
- [ ] **2.6** Modify `sparc/__main__.py::cmd_run` to delegate to the orchestrator for each stage in the loop.
- [ ] **2.7** Add `tests/test_orchestrator_dispatch.py`:
  - With feedback off, orchestrator produces equivalent results to legacy dispatch on a small synthetic project
  - `StageDecision` artifact is written and queryable from `artifacts.db`
  - `metrics` dict is non-empty for stages that ran

## Phase 3 — Narrow surrogate inner loop (`sparc/run/inner_loops/bandwidth_search.py`)

- [ ] **3.1** Create `sparc/run/inner_loops/__init__.py` and `bandwidth_search.py`.
- [ ] **3.2** Implement `_composite_objective(predictions, y, residuals, coords, lambdas) -> float` — OOF MSE + λ_phys·PDE_residual + λ_mono·monotonicity + λ_spatial·|Moran's I|. Reuse existing `calculate_fold_spatial_autocorr` from `enhanced_spatial_cv.py` for the Moran's I term.
- [ ] **3.3** Implement `_kappa_posterior_to_bo_prior(kappa_mean, kappa_hdi) -> (search_space, prior_mean)` — converts the correlogram κ posterior into a `skopt` `Real` search dim with bounds = HDI, log-transform, and a Gaussian initial mean.
- [ ] **3.4** Implement `_evaluate_surrogate(hparams, surrogate, X, y, coords, folds, lambdas) -> float` — fast surrogate-based composite score (no full base-model fit).
- [ ] **3.5** Implement `_verify_full_model(hparams, full_model_factory, X, y, coords, folds, lambdas) -> float` — trust-region verification with a real base-model fit.
- [ ] **3.6** Implement `run_narrow_bandwidth_search(...)` per PRD §Technical Approach with `gp_minimize`, top-K verification, and improvement gate.
- [ ] **3.7** The function returns `(best_hparams, decision_record, provenance)` where `provenance.search_log` contains the full BO trace and verified scores.
- [ ] **3.8** Add `tests/test_bandwidth_search_gate.py`:
  - **promotion path:** stub a candidate that beats baseline by ≥1% → result is the candidate, decision action is `accept`
  - **fallback path:** stub a candidate within ±1% of baseline → result is correlogram defaults, decision action is `fallback`
- [ ] **3.9** Add `tests/test_bandwidth_search_smoke.py`:
  - Synthetic 200-point spatial dataset; tiny surrogate; n_trials=5; verify the search runs end-to-end without exception and returns a valid `DecisionRecord`.

## Phase 4 — Wire into Stage 2 (`sparc/run/enhanced_spatial_cv.py`)

- [ ] **4.1** Add `pipeline.use_surrogate_search` config flag (default `false`).
- [ ] **4.2** In `EnhancedSpatialCV.run_enhanced_spatial_cv()`, after `get_kernel_field()` and before `generate_optimized_oof_predictions()`:
  - Skip if `use_surrogate_search` is false (preserve byte-identical behavior).
  - Otherwise: hoist surrogate training to here (warm-start from correlogram defaults).
  - Compute correlogram-baseline composite score with one full base-model fit per model.
  - Call `run_narrow_bandwidth_search` for `gwr`, `gwrf`, `ggpgam` sequentially.
  - Apply the returned hparams (winner or fallback) to the model construction in `create_optimized_models()`.
- [ ] **4.3** Aggregate the three per-model `DecisionRecord`s into the Stage 2 `StageDecision` returned by the orchestrator.
- [ ] **4.4** Confirm Stage 2's `_source_geodataframe` and downstream artifact contracts are unchanged (the search only changes hparams, not artifact shapes).

## Phase 5 — Validation

- [ ] **5.1** Run full pipeline on `my_project/brown4.csv` with `use_surrogate_search: false` — confirm byte-identical model outputs vs. main (compare `final_ensemble_results.json` and `optimized_oof_predictions.csv`).
- [ ] **5.2** Run full pipeline on `my_project/brown4.csv` with `use_surrogate_search: true` — confirm:
  - Pipeline completes without exception
  - StageDecision artifacts present for all stages
  - Stage 2 decision contains three per-model search records
  - Provenance correctly records whether each model used search-winner or correlogram-fallback hparams
- [ ] **5.3** Compare R² and composite-objective deltas between the two runs; record the result in the PR description.
- [ ] **5.4** Run full existing test suite: `pytest tests/` — must pass with no modifications.

## Phase 6 — PR

- [ ] **6.1** Squash to clean commits along phase boundaries.
- [ ] **6.2** PR description includes: PRD link, acceptance-criteria checklist, before/after metrics from Phase 5.
- [ ] **6.3** Tag a release candidate (do not bump the main version yet — gated behind config flag).

---

## Status

- Branch created: `feat/reasoning-engine-spine` ✓
- PRD: written ✓
- Implementation: not started

## Out of Scope (tracked for follow-up PRs)

- S3→S2 refutation-gated retrain loop (PRD: pending)
- Wide joint surrogate loop over the neural meta-learner (PRD: pending)
- Desktop UI "Reasoning Trace" panel rendering (PRD: pending)
- Parallel per-model search (refactor of Phase 4)
