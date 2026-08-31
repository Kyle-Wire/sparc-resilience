# Pipeline Remediation — Implementation Spec

**Status:** in progress
**Parent:** [`plan.md`](../plan.md) Phases 0–2
**Baseline commit:** `0a04b7a`

Implementation-level detail for the review findings. `plan.md` says *what* and *why*;
this document says *where* and *how*, with the exact code at each site.

Each change carries: the site, the current behaviour, the replacement, how it is
verified, and whether it is reversible. Items marked **✅ applied** have landed on this
branch; the rest are specified but not yet written.

---

## Change ledger

| ID | Change | Risk | Verified by | Status |
|---|---|---|---|---|
| R1 | Surrogate anchor: OOF instead of in-sample fits (F1) | **high value** | re-baseline delta ⚠️ | ✅ applied |
| R2 | Diminishing-return taper is C¹ and monotone (F5) | low | 9 unit tests | ✅ applied |
| R5 | Block index clipped at domain edge (F7) | trivial | 2 unit tests | ✅ applied |
| R7 | `lambda_surrogate` removed from CMA-ES search space | trivial | unit test | ✅ applied |
| R8 | `penalty_acyclic` → `penalty_low_prior` | trivial | unit test (needs torch) | ✅ applied |
| R3 | Single E-value implementation, correctly fed (F4) | low | unit test | **not started** |
| R4 | `--resume` marks stage completion (F2) | low | manual + unit test | **not started** |
| R6 | Local-fit failures counted, not silently zeroed | low | unit test | **not started** |
| R9 | Groundwater template: head not depth; grid 1 km | **blocking G1** | schema test | **not started** |
| R10 | pytest in CI (F11) | none | CI run | **not started** |
| R11 | Spatial bootstrap + DML-consistent CI (F3) | medium | unit test | specified |
| R12 | Spatial splitter through DML fallback + CATE (F6) | medium | unit test | specified |
| R13 | Wire `SpatialConformalPredictor` into Stage 2 (F10) | medium | coverage report | specified |
| R14 | MGWR → GWR rename (F1.5) | low, wide | tests pass | specified |
| R15 | README reconciliation (F12) | none | — | specified |

**⚠️ R1 is applied but not yet validated.** The code change is in; the re-baseline
that measures the leakage delta requires a full Stage 2 run with `torch`, which has
not happened. Until that runs, treat R1 as *believed correct, unmeasured*.

Applied changes are covered by `tests/test_remediation.py` — 12 passing, 1 skipped
(the `mc3` case needs `torch`). The rest of the suite is unchanged at 411 passed /
75 failed / 24 skipped / 6 errors, identical to the pre-change baseline; every
failure and error is a missing optional dependency (`torch`, `geopandas`, `mgwr`,
`httpx`), not a defect.

---

## R1 — Surrogate anchor (F1)

**Site:** `sparc/run/enhanced_spatial_cv.py`, the `_base_oof` construction feeding
`train_neural_meta`.

**Before.** `_base_oof` was built from `base_fitted_values`, which Stage 2b collects by
refitting each base model on the **full** dataset. The code was explicit about it:

```python
# Build base-model fitted values dict for surrogate pretraining
# Use full-model fitted values (Stage 2b) instead of OOF predictions
_base_oof = {}
for _mn in ('gwr', 'gwrf', 'ggpgam'):
    if _mn in base_fitted_values:
        ...
```

Those values are sliced `[train_idx]` at `v2_neural_training.py:730` and used as
surrogate pre-training targets. A GWR fitted value *at a training point* is a
kernel-weighted function of neighbouring `y`, and with a `block/3` buffer that
neighbourhood reaches into the held-out block. The parameter is named
`base_oof_predictions`; the values were not out-of-fold.

**After.** `base_predictions` — built at the same scope from the joined OOF table —
is used by default. The old behaviour is retained behind an explicit config key so the
delta can be measured rather than argued about:

```yaml
models:
  surrogate_anchor: "oof"   # "oof" (default, leak-free) | "full" (legacy, leaks)
```

**Verify.** Re-run Stage 2 both ways and record both numbers in `docs/baselines.md`.
The difference is the honest size of the leak. Publish it.

**Reversible:** yes, via config.

**Note.** `base_predictions` is `{}` on the `skip_stage_2_base_models` path, in which
case the anchor is absent and surrogates train against `y` directly — the pre-existing
behaviour for that path, unchanged.

---

## R2 — Diminishing-return taper (F5)

**Site:** `sparc/interventions/scenario_simulator.py`, `_diminishing_return`.

**Before.** `effective(Δ) = τ + √(Δ−τ)·√τ` for `Δ > τ`. Two defects:

- derivative at `Δ = τ⁺` is `+∞` (measured 316 at `Δ = τ + 1e-4`)
- `effective(Δ) > Δ` for all `τ < Δ < 2τ` — with the default `τ = 10`, a +11 pp
  intervention is simulated as **+13.2 pp**

A function whose stated purpose is diminishing returns had its steepest marginal
return exactly at the knee, and *amplified* interventions across the range planners
care about.

**After.** A C¹-continuous concave form with slope exactly 1 at the knot:

```
effective(Δ) = τ + 2τ(√(1 + (Δ−τ)/τ) − 1)
```

Properties, all unit-tested in `tests/test_remediation.py`:

| Property | Check |
|---|---|
| continuity at knot | `effective(τ) == τ` |
| C¹ at knot | slope → 1 as `Δ → τ⁺` |
| never amplifies | `effective(Δ) ≤ Δ` for all `Δ ≥ 0` |
| monotone increasing | `d effective/dΔ > 0` |
| concave | marginal return strictly decreasing above the knot |
| odd symmetry | `effective(−Δ) == −effective(Δ)` |

**Impact.** Every Stage 4 dose–response number above the threshold changes. Do not
publish Stage 4 curves computed before this change.

**Still open (not a code change).** The √-family shape contradicts Ziter et al. (2019,
*PNAS*), cited in `physics_priors.py` for the canopy prior, which found daytime
cooling *accelerates* above ~40% canopy cover. Fixing the function does not fix the
functional form. Prefer fitting the dose–response from Stage 3's
`CausalDoseResponseCurve` objects over imposing any taper; failing that, set
per-variable `diminishing_return_thresholds` in `caps.yml` from literature rather than
falling back to a neutral 10.0.

---

## R3 — E-values (F4)
> **Status: not started.** Specified below; the code change has not been written.


**Sites:** `sparc/run/causal_validation.py` (`_compute_e_value`) and
`sparc/run/wager2025_addons.py`.

**Before.** Two implementations, each breaking a different half of VanderWeele & Ding
(2017):

| Path | Effect used | Standardisation |
|---|---|---|
| `causal_validation` | **unadjusted univariate OLS** — `lr.fit(t.reshape(-1,1), y)` | correct: `|β·SD(T)|/SD(Y)` |
| `wager2025_addons` | correct: adjusted ATE | **none** — raw ATE passed as Cohen's *d* |

An E-value is by construction the sensitivity of a *confounder-adjusted* estimate to
an *unmeasured* confounder. Computed from a marginal correlation it answers a
different question; computed from an unstandardised coefficient it is not on the
`exp(0.91·d)` scale at all.

**After.** Both routed through `sparc.causal.sensitivity.e_value_continuous`, which was
already correct, with a new helper that applies the standardisation:

```python
e_value_for_effect(ate, t_values, y_values, label=...)
    # d = |ate * SD(T)| / SD(Y)  →  RR = exp(0.91 d)  →  E = RR + sqrt(RR(RR-1))
```

`causal_validation._compute_e_value` now takes the **adjusted** estimate rather than
refitting a bivariate regression.

**Verify (planned).** A test should pin the arithmetic against hand-computed values
and asserts both call sites agree for the same input.

**Impact.** Every reported E-value changes. The README table is stale.

---

## R4 — `--resume` (F2)
> **Status: not started.** Specified below; the code change has not been written.


**Site:** `sparc/__main__.py`, `sparc/run/orchestrator.py`.

**Before.** `_orch.mark_complete()` was called exactly once, for Stage 0. The calls
that looked like completion markers for Stages 2–5 were `_sp.stage_done()` — that is
`StageProgress`, the progress bar, a different object with a near-identical method
name. `orchestrator.stage_done()` also had no disk-sentinel fallback despite a comment
promising one, so the `.gwen_complete` file Stage 1 writes was never read.

Net effect: `--resume` re-ran Stage 2 — the multi-hour GPU training — from scratch,
including via the GWEN approval gate that tells the user to do exactly that.

**After.**

- `_orch.mark_complete("<n>")` after each successful stage.
- `orchestrator.stage_done()` gains the documented disk-sentinel fallback via a new
  `sentinel_paths` mapping, so legacy on-disk markers are honoured.

**Verify.** `sparc run -s all` then `sparc run -s all --resume` skips 0–4.
A test should cover the store and sentinel paths with fakes.

---

## R5 — Block index at the domain edge (F7)

**Site:** `sparc/run/spatial_fold_factory.py`.

**Before.** `block_y = int((y − min_y) / block_size)` evaluates to `n_blocks_y` for
points exactly on the maximum edge, and `block_x * n_blocks_y + block_y` then collides
with the id of `(block_x + 1, 0)`. A handful of edge points join a spatially distant
block, breaking the contiguity invariant blocking depends on.

**After.** Both indices clipped to `[0, n_blocks − 1]`.

**Verify.** `tests/test_remediation.py` places points on the exact maximum edge and
asserts fold partitioning stays clean and blocks stay spatially contiguous.

---

## R6 — Failed local fits (F1.6)
> **Status: not started.** Specified below; the code change has not been written.


**Sites:** `sparc/models/gwr.py`, `sparc/models/gwrf.py`.

**Before.** A failed local regression was recorded as `β = 0, intercept = ȳ`; a failed
local RF appended `None`. For a tool emitting per-cell policy advice, "the fit failed
here" and "this variable has no effect here" must not be the same output — and there
was no count.

**After.** Both models track `n_failed_fits_` and expose `failure_rate_`. A warning is
emitted above 1% of points, and the count is written to the artifact store so it
surfaces in reports.

The imputed values are unchanged — this is about visibility, not behaviour. Changing
the imputation is a separate decision.

**Verify (planned).** A test should force failures and assert the counter.

---

## R7 — `lambda_surrogate`

**Site:** `sparc/training/cma_es.py`.

The surrogate-fidelity loss term is documented as inactive in `training/loss.py`
(*"accepted but inactive — surrogates learn end-to-end via the main loss terms"*), and
`lambda_surrogate` appears nowhere in the loss. It survived only as a dimension of the
CMA-ES search space, so the optimiser was spending budget searching an inert knob.

**After.** Removed from `HyperparamSpec`. Also removed from shipped `project.yml`
templates, where it was documented as a tunable.

---

## R8 — `penalty_acyclic`

**Site:** `sparc/causal/mc3.py`.

The parameter does not penalise cycles. It applies an extra penalty when an edge's
*prior* probability is below 0.5. Renamed to `penalty_low_prior`, with the old name
accepted as a deprecated alias for one release.

---

## R9 — Groundwater template (blocking `docs/groundwater-pilot.md` G1)
> **Status: not started.** Specified below; the code change has not been written.


**Site:** `templates/groundwater/project.yml` and `sparc/templates/groundwater/project.yml`.

Two defects, both blocking the pilot:

1. **Wrong target quantity.** `target_column: "gw_level_m"` is depth below ground
   surface. The groundwater flow equation is written in hydraulic head,
   `h = surface_elevation − depth_to_water`. Depth-to-water is dominated by
   topography and does not satisfy the Laplacian, so every PDE term, the α field and
   Tier 2 would all fit the wrong surface.
   → target is now `head_m`, with `depth_to_water_m` retained as a derived reporting
   column and `elevation_m` documented as required for the derivation.
2. **Grid resolution.** `grid_resolution_m: 30` over the High Plains extent is roughly
   500 million cells. → `1000`.

---

## R10 — pytest in CI (F11)
> **Status: not started.** Specified below; the code change has not been written.


**Site:** `.github/workflows/tests.yml` (new).

486 tests existed with nothing gating a merge; the only workflow built the desktop app
on tags. The new workflow runs on push and PR against the core dependency set — the
~400 tests that need neither GPU nor geospatial stack — with `--continue-on-collection-errors`
so optional-dependency modules skip rather than abort the run.

Heavier jobs (torch, geopandas, mgwr) are left as a follow-up once runner cost is
understood.

---

## Specified, not yet applied

### R11 — Spatial bootstrap (F3)

`causal_validation.run_enhanced_bootstrap` refits a plain `LinearRegression`, not the
DML that produced the ATE it is reported beside — which is why the README's Impervious
row shows ATE `+0.018` with a "bootstrap 95% CI" of `[+0.020, +0.021]`, an interval
excluding its own point estimate. It is also an i.i.d. bootstrap over 54,701
spatially autocorrelated cells.

**Do:** resample **spatial blocks** (reuse the geometry from `spatial_kfold_enhanced`)
and refit the actual estimator. Until then the column should be dropped rather than
published; `const_marginal_effect_inference().stderr` is already captured and is the
more defensible interval.

Deferred because it needs the DML path exercised end to end, which needs `econml`.

### R12 — Spatial splitter through the fallback (F6)

`counterfactual_engine._fit_edge_dml` builds spatial folds and hands them to econml,
but the `except` branch calls `_fit_edge_dml_sklearn(T, Y, W, model_y, model_t, n_splits)`
— coordinates and splitter both dropped — reverting to random `KFold`. The comment
above it explains the branch exists because `econml.tree._utils` is known to break, so
it is live. `spatial_cate.py` (`cv=3`, `KFold(shuffle=True)`) never had a spatial
splitter at all.

**Do:** thread `cv_splitter` into the sklearn fallback and into `CausalForestDML`;
record in the artifact when the fallback fires so downstream reports can label the
estimate.

### R13 — Conformal calibration (F10)

`SpatialConformalPredictor` is written, correctly motivated (Barber et al. 2023), and
called from nowhere. It is the precondition for both pilots.

**Do:** hold out one spatial block inside Stage 2 as a calibration set, calibrate,
report empirical coverage at 90% nominal on the test blocks, write `coverage_report`
to the store.

```python
pred = SpatialConformalPredictor(kappa=stage0_kappa)
pred.calibrate(cal_coords, np.abs(cal_residuals))
intervals = pred.prediction_interval(test_coords, y_hat)
empirical = intervals.coverage(y_true)
```

Deferred because it needs a full Stage 2 run to validate.

### R14 — MGWR → GWR rename (F1.5)

`gwr.py` has no back-fitting loop and collapses per-predictor bandwidths to their
arithmetic mean, so the estimator is GWR with a correlogram-derived bandwidth, not
MGWR. The label propagates to `bayesian_mgwr_ensemble.py`, the
`mgwr_local_coefficients_unconstrained.csv` artifact, the README, and the
`mgwr_causal_anchored` coefficient source in published results.

Deferred because it touches artifact names, which changes on-disk layouts and
downstream readers. Wants its own commit and a migration note.

### R15 — README reconciliation (F12)

- numpyro is not a dependency; `sparc/causal/nuts.py` is a hand-written PyTorch NUTS
- the "0.944 with Laplacian" headline is not reproducible from a default run —
  eigenmaps are off by default and were computed over all coordinates including
  held-out blocks
- `cmd_run` docstring has Stages 0 and 1 swapped
- MC-Dropout default is `n_samples=100`, not 500
- PDE loss carries 12 terms; README says 10; 8 appear in `_ACTIVATION_SCHEDULE`

Deferred until R1's re-baseline produces the corrected numbers, so the README is
edited once rather than twice.

---

## Verification status

Run in this environment with a partial dependency set (no `torch`, `geopandas`,
`mgwr`, `econml`):

- `python -m compileall` clean across `sparc/`, `tests/`, `scripts/`
- `tests/test_remediation.py`: 12 passed, 1 skipped (needs `torch`)
- full suite: 411 passed / 75 failed / 24 skipped / 6 errors — **identical to
  the pre-change baseline**; all failures are missing optional dependencies
- The full pipeline has **not** been run here — it requires the GPU and geospatial
  stack. R1's re-baseline and R13's coverage report must be produced locally.
