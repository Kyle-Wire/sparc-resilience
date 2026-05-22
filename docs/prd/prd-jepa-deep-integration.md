# PRD: JEPA Deep Integration — Causal Deconfounding and Multi-Step World Model

**SPARC Labs LLC | May 2026**
**Status: Ready for Implementation**

---

## Problem Statement

The JEPA (Joint Embedding Predictive Architecture) stack in SPARC is more complete than its current utilisation suggests. Phase 1 (masked spatial context self-supervised pretraining with EMA target trunk and VICReg anti-collapse) and Phase 2 (action-conditioned FiLM predictor for scenario distillation) are both fully wired into the Stage 2 neural training loop. Yet the trunk's physics-consistent spatial representations never reach Stage 3, and the scenario rollout machinery is limited to a single intervention step.

This produces three concrete gaps:

1. **DML outcome nuisance is suboptimal.** `CounterfactualEngine._fit_edge_dml()` refits a `HistGradientBoostingRegressor` as `model_y` (the outcome nuisance) from scratch per causal edge, using only the observed confounders. Stage 2 has already computed out-of-fold ensemble predictions (`oof_preds`) that are strictly better — these span the full SPARCMetaLearner ensemble under spatial cross-validation and have never been passed to Stage 3.

2. **DML treatment nuisance is spatially confounded.** Treatment features entering Stage 3 carry the same spatial autocorrelation structure that confounds the causal discovery step. The JEPA trunk has learned a physics-consistent low-dimensional spatial basis for this structure, but no path exists from the trunk to the DML treatment nuisance estimator.

3. **Scenario simulation is single-step only.** The latent rollout module supports one action-conditioned forward step. Compound multi-year interventions — "plant canopy cover in year 1, add cool roofs in year 2" — cannot be modelled, and there is no way to specify how a primary intervention cascades into correlated feature changes (tree planting → NDVI up, albedo down, ET up).

---

## Solution

Close all three gaps in a coordinated set of five changes:

- **JD-1** passes Stage 2 OOF predictions directly into DML as a precomputed outcome nuisance, replacing the fresh HGB fit per edge with a better estimate that is already available in the artifact store. No JEPA dependency.
- **JD-2** introduces a `SpatialResidualizer` pre-processing stage between Stage 2 and Stage 3 that uses the JEPA trunk embedding to remove spatially structured confounding from treatment features before DML estimation.
- **JD-3** flips the `jepa.enable` schema default to `true` so the trunk is available in all future runs. Ships alongside JD-2.
- **JD-4** extends the latent rollout module with a multi-step recurrent latent chain so compound interventions can be simulated as sequences of action-conditioned predictor steps.
- **JD-5** adds a physics cascade table per domain template and a `mode="reencode"` path to the multi-step rollout that re-encodes through the trunk at each step using physically plausible feature co-changes.

---

## User Stories

1. As a SPARC pipeline operator, I want Stage 3 DML to automatically use the Stage 2 ensemble predictions as its outcome nuisance model, so that Average Treatment Effect estimates are less biased by spatial structure captured by the ensemble but not by a fresh HGB fit.
2. As a SPARC pipeline operator, I want the DML outcome nuisance substitution to be config-controlled and to fall back gracefully when Stage 2 OOF predictions are unavailable, so that existing runs are not broken.
3. As a SPARC pipeline operator, I want a log message confirming that Stage 2 OOF predictions were used as the DML outcome nuisance, including the effective R² of those predictions, so that I can verify the improvement path in the run log.
4. As a SPARC pipeline operator, I want the `SpatialResidualizer` to automatically remove JEPA trunk-encoded spatial confounding from treatment features before DML estimation, so that treatment nuisance models are fitted on features with lower Moran's I.
5. As a SPARC pipeline operator, I want the `SpatialResidualizer` to degrade gracefully to a no-op when no JEPA trunk checkpoint is available in the artifact store, so that Stage 3 runs cleanly on projects that have not enabled JEPA.
6. As a SPARC pipeline operator, I want a log message indicating which treatment features were residualized and the Moran's I before and after, so that I can assess the spatial deconfounding effect.
7. As a SPARC pipeline operator, I want JEPA to be enabled by default for new projects without requiring an explicit `jepa:` config block, so that the trunk is available for spatial residualization in all standard runs.
8. As a SPARC pipeline operator, I want projects with explicit `jepa.enable: false` to continue training without the JEPA pretraining pass, so that existing opt-out configurations are respected.
9. As a domain expert, I want to simulate a compound multi-year intervention as a sequence of treatment steps — for example, canopy planting followed by cool roof installation — and receive a trajectory of outcome predictions for each step.
10. As a domain expert, I want the multi-step rollout to support a `mode="latent"` path that chains predictor steps purely in embedding space without re-encoding, so that fast approximate trajectory estimates are available.
11. As a domain expert, I want the multi-step rollout to enforce a configurable maximum number of steps to prevent unbounded latent drift.
12. As a domain expert, I want to receive a `MultiStepRolloutResult` containing both the intermediate step results and the final decoded outcome, so that I can inspect the trajectory at each stage.
13. As a domain expert working with the UHI template, I want a `mode="reencode"` path that consults a physics cascade table to determine how my primary intervention (e.g., tree planting) changes correlated features (NDVI, albedo, ET) before re-encoding, so that multi-step simulations respect physical co-dependency.
14. As a domain expert, I want the physics cascade table to be defined as a YAML section in the domain template's caps file, so that domain scientists can audit and amend the cascade assumptions without reading code.
15. As a domain expert, I want all 13 domain templates (UHI, wildfire, coastal, groundwater, stormwater, noise, drought, air quality, geotechnical, water quality, seismic, blank, forcesmip) to ship with initial `treatment_cascades` tables, so that re-encode rollouts are immediately available across all hazard types.
16. As a researcher, I want the `SpatialResidualizer` to also store residualized-feature columns alongside original columns in the Stage 3 working DataFrame so that I can compare DAG structure discovered on raw vs. residualized features.
17. As a researcher, I want single-action calls to `multi_step_latent_rollout` to produce results matching the existing `latent_rollout` function for regression compatibility.
18. As a researcher, I want a `PhysicsCascade` utility class that can be instantiated from a caps YAML path, so that cascade application can be tested independently of the full rollout loop.
19. As a researcher, I want the `PhysicsCascade` to write both original and cascaded feature columns to its output DataFrame so that the magnitude of co-changes can be inspected.
20. As a researcher, I want the two rollout modes (`latent` and `reencode`) to share a single API surface with a `mode` parameter, so that A/B comparisons between modes are trivially scriptable.

---

## Implementation Decisions

### JD-1 — Stage 2 OOF as DML Outcome Nuisance

**Problem:** `_fit_edge_dml()` fits a fresh `HistGradientBoostingRegressor` as `model_y` for each causal edge. The Stage 2 ensemble already has better outcome predictions in the artifact store.

**Change to `CounterfactualEngine._fit_edge_dml()`:** Add an optional `precomputed_outcome_preds` parameter. When provided (a 1-D array matching `data` length), skip `model_y.fit()` and compute the outcome residual as `Ỹ = Y - precomputed_outcome_preds` directly. The treatment nuisance `model_t` is unchanged — it is still fitted from scratch because it predicts treatment from confounders, not outcome from features.

**Change to Stage 3 dispatch (`v2_bayesian_causal.py`):** Before calling `CounterfactualEngine.fit()`, attempt to read `("2", "oof_predictions")` from the artifact store. If found, pass it as `precomputed_outcome_preds`. If absent, proceed without it (standard HGB path).

**New config flag:** `causal.use_stage2_outcome_nuisance` (boolean, default `true`). When `false`, reverts to the HGB fit regardless of artifact availability.

**Log message on success:** `DML model_y: using Stage 2 oof_preds (stage2_r2=X.XX) for edge {parent}→{child}`.

**Fallback condition:** If the OOF array shape does not match the input data (possible when the pipeline is run on a different subset), log a warning and fall back to HGB.

---

### JD-2 — SpatialResidualizer

**New module:** A thin preprocessing stage that accepts the Stage 3 working DataFrame, loads the JEPA trunk from the artifact store (`("2", "jepa_trunk_state_dict")`), encodes physics features via `SPARCMetaLearner.encode()`, reduces the trunk embedding to 16 principal components, then for each treatment column fits a Ridge regression to predict the treatment from the PCA scores and writes the residuals as `{col}_resid` into the DataFrame alongside the original column.

**Graceful degradation:** If the trunk checkpoint is absent from the artifact store, the module logs `[INFO] No JEPA trunk found — skipping spatial residualization` and returns the DataFrame unchanged. Stage 3 proceeds without residualization.

**Integration point:** The residualizer runs as a named pre-step in the Stage 3 dispatch, gated by `causal.use_spatial_residuals` (boolean, default `false`). When enabled, the DML treatment nuisance (`model_t`) is fitted on `{col}_resid` columns; the original columns remain available for all other Stage 3 operations.

**PCA dimensionality:** Fixed at 16. This is a configurable constant in the residualizer but not surfaced in the project schema (internal detail, not a user-facing knob).

**Ridge regularisation:** Scikit-learn `Ridge(alpha=1.0)` — standard default, no per-template tuning.

---

### JD-3 — JEPA Default-On

**Change:** The `jepa.enable` property in `project_schema.json` changes its `"default"` value from `false` to `true`.

**Description string:** Updated to remove "Off by default".

**Constraint:** This change ships in the same commit as JD-2. Enabling JEPA by default before the `SpatialResidualizer` exists would add training cost with no visible benefit and should not be shipped standalone.

**Backward compatibility:** Existing `project.yml` files that explicitly set `jepa.enable: false` continue to behave correctly. The schema default only affects projects that omit the `jepa` section entirely.

---

### JD-4 — Multi-Step Latent Rollout (mode: latent)

**New API function:** `multi_step_latent_rollout(*, model, predictor, action_embed, physics_feats, base_preds, X_spatial, coords, knn_index, alpha, actions, mode="latent", max_steps=5, y_mean=0.0, y_std=1.0)`.

**`actions` parameter:** A list of `(treatment: str, delta_x: float, delta_t: float)` tuples. Each tuple describes one intervention step. The list must not exceed `max_steps`; a `ValueError` is raised if it does.

**New dataclass:** `MultiStepRolloutResult(steps: list[LatentRolloutResult], final: LatentRolloutResult, n_steps: int)`. The `steps` list contains one `LatentRolloutResult` per intermediate latent state (using the pre-decode `h_t` tensor decoded at each step for trajectory inspection), and `final` is the decoded result at the last step.

**Recurrence in `mode="latent"`:** `h_{t+1} = predictor(h_t, action_embed(actions[t]))`. No re-encoding. Baseline `T_baseline` in every step is the decoded `h_0` (the unperturbed initial latent). `delta` is `T_pred_t - T_baseline`.

**Regression guard:** A single-action call to `multi_step_latent_rollout` must produce a `final` that matches the existing `latent_rollout()` output to within floating-point tolerance. This is enforced by a test.

---

### JD-5 — Multi-Step Re-encode with Physics Cascade (mode: reencode)

**New module:** `PhysicsCascade`, instantiated from the domain template's caps YAML. It reads a `treatment_cascades` section structured as `{treatment_name: {feature_name: scale_factor}}`. The `apply(df, treatment, delta_x)` method returns a copy of the input DataFrame with `df[treatment] += delta_x` and `df[feature] += scale_factor * delta_x` for each correlated feature listed under that treatment. Both original-column names and cascaded-column names exist in the output — the original columns are preserved.

**Cascade YAML section example (UHI template):**
```yaml
treatment_cascades:
  pct_canopy:
    ndvi:       +0.80
    albedo:     -0.05
    et_flux:    +0.30
  pct_impervious:
    albedo:     +0.04
    et_flux:    -0.15
```

**`mode="reencode"` in `multi_step_latent_rollout`:** Requires an additional `cascade` parameter (a `PhysicsCascade` instance). Each step: (1) apply cascade to produce `df_perturbed`; (2) construct `physics_feats_perturbed` from `df_perturbed`; (3) `h_t = model.encode(physics_feats_perturbed, alpha)`; (4) `h_{t+1} = predictor(h_t, action_embed(actions[t]))`; (5) decode `h_{t+1}`. In `mode="reencode"` without a `cascade`, the function raises `ValueError`.

**Template updates:** All 13 domain templates get a `treatment_cascades` section in their `caps.yml`. For templates where cross-feature physics is underdetermined (e.g., seismic, noise), the section is present but empty: `treatment_cascades: {}`.

**Future extension:** A learned cascade model (small network predicting ΔX_all from action) is explicitly noted as a future item in the module docstring and the backlog. The `PhysicsCascade` interface is designed to be swappable.

---

### Schema Changes

- `project_schema.json`: `jepa.enable.default` changes from `false` to `true`.
- `project_schema.json`: New optional `causal.use_stage2_outcome_nuisance` boolean (default `true`).
- `project_schema.json`: New optional `causal.use_spatial_residuals` boolean (default `false`).
- All 13 template `caps.yml` files: New `treatment_cascades` section (dict, may be empty).

---

## Testing Decisions

### What makes a good test here

Tests should verify observable behaviour at module boundaries — the shape and values of returned arrays, the presence or absence of log messages, and the fallback behaviour when artifacts are absent. Tests should not assert on internal Ridge coefficients, PCA loadings, or the number of sklearn `fit()` calls made internally.

### Modules to test

**JD-1 — DML outcome nuisance substitution**
- `CounterfactualEngine._fit_edge_dml()` with and without `precomputed_outcome_preds`.
- Verify residual array shapes match in both paths.
- Verify that when OOF preds are provided, the fitted DML model's `model_y` is not invoked.
- Prior art: `tests/test_counterfactual_engine.py` (existing DML fit tests).

**JD-2 — SpatialResidualizer**
- `SpatialResidualizer.fit_transform()` with a mock trunk checkpoint: output DataFrame has both `{col}` and `{col}_resid` columns.
- Graceful no-op when trunk checkpoint is absent: output DataFrame is identical to input.
- Residualized columns have lower or equal Moran's I than raw columns on a synthetic spatial dataset.
- Prior art: `tests/test_counterfactual_engine.py`, `tests/test_causal_stack.py`.

**JD-3 — Schema default**
- Load a minimal `project.yml` with no `jepa:` section; verify that `config["jepa"]["enable"]` is `True`.
- Prior art: `tests/test_config.py`.

**JD-4 — Multi-step latent rollout (latent chain)**
- `multi_step_latent_rollout` with a single action produces `final` matching `latent_rollout` output.
- Two-step call returns `n_steps=2`, `len(steps)==2`, both `LatentRolloutResult` shapes correct.
- Call with `len(actions) > max_steps` raises `ValueError`.
- Prior art: existing `latent_rollout` usage in `tests/test_gwrf_predict.py`, `scripts/run_jepa_standalone.py`.

**JD-5 — PhysicsCascade and re-encode rollout**
- `PhysicsCascade.apply()` arithmetic: given a cascade config, verify `df[treatment]` shifts by `delta_x` and `df[feature]` shifts by `scale_factor * delta_x`.
- `PhysicsCascade.from_caps()` loads correctly from a minimal YAML fixture.
- `multi_step_latent_rollout(..., mode="reencode")` produces intermediate `LatentRolloutResult` that differ from `mode="latent"` on the same action sequence (re-encoding produces different trunk states than pure latent chaining).
- `multi_step_latent_rollout(..., mode="reencode")` without `cascade` raises `ValueError`.
- Prior art: `tests/test_causal_stack.py`, `tests/test_config.py`.

---

## Out of Scope

- **Learned cascade model** — predicting ΔX_all from action via a small network trained on observational feature covariance. Noted as a future item in the `PhysicsCascade` module.
- **Model Predictive Control / planning optimizer** — using the multi-step world model to optimise over intervention sequences. The `CausalBandit` stub in the backlog is the intended follow-on.
- **Zero-shot city transfer via trunk** — using the trunk for `latent_rollout` in cities that lack training data. Depends on Phase 4 zero-shot infrastructure not yet built.
- **Desktop UI for multi-step rollout** — visualising the step trajectory in the intervention panel. Deferred to a separate UX PRD.
- **Moran's I reporting in the SPARC run report** — the spatial deconfounding metrics from `SpatialResidualizer` are logged but not yet included in the Stage 3 summary table or the HTML report.

---

## Further Notes

- **Shipping order:** JD-1 should ship first as it has no JEPA dependency and establishes a baseline against which JD-2's deconfounding lift can be measured. JD-3 must ship alongside JD-2. JD-5 depends on JD-4's `MultiStepRolloutResult` dataclass.
- **JEPA training cost:** Enabling JEPA by default adds approximately 25–35% training time per fold. This is acceptable for the current single-operator deployment.
- **Graceful degradation as a first-class contract:** Both JD-1 and JD-2 must degrade silently when artifacts are absent. This is a hard requirement — Stage 3 must remain runnable on any SPARC project regardless of whether Stage 2 produced OOF predictions or a JEPA trunk checkpoint.
- **Frisch-Waugh-Lovell basis for JD-1:** Passing precomputed `ĝ(X)` as `model_y` in DML is a direct application of the Frisch-Waugh-Lovell theorem — the partialled-out residual `Ỹ = Y - ĝ(X)` is orthogonal to confounders iff `ĝ` is a consistent estimator of `E[Y|X]`. Stage 2's ensemble OOF predictions are the most consistent estimator available in the pipeline and replacing the per-edge HGB fit with them is strictly correct.
- **Spatial residualization basis for JD-2:** The `SpatialResidualizer` implements a linear spatial control approach related to Schölkopf et al. (2021) causal representation learning. The Ridge probe on JEPA trunk PCA components removes the spatially-structured variation from treatment features — variation that correlates with both treatment assignment and outcome, constituting spatial confounding. This complements the existing Stage 0 spatial autocorrelation range detection, which informs DML fold sizes but does not adjust the features themselves.
