# PRD: Reasoning Engine Spine + Stage 2 Surrogate Inner Loop

**Status:** Planning
**Author:** SPARC Labs LLC
**Date:** May 2026
**Related:** First step toward SPARC as an autonomous spatial causal reasoning engine.

---

## Problem Statement

SPARC today is a linear, fixed-order pipeline. Each of the five stages runs once, in sequence, and reports its results. Diagnostic metrics that already exist — out-of-fold R², residual Moran's I, MC³ acceptance rate, NUTS HDI width, DoWhy refutation p-values — are computed and printed, but **no stage acts on them**. If Stage 2 produces a poor R², the pipeline does not retry with different hyperparameters. If Stage 3 fails its refutation tests, the pipeline does not drop confounded predictors and retrain. The user, not the system, is the only feedback loop.

This is the gap between SPARC as a *spatial ML pipeline* and SPARC as a *spatial reasoning engine*. A reasoning engine observes evidence, branches based on it, and revises decisions — even within a single run.

The bandwidth selection in Stage 0's correlogram is a separate but related issue. The correlogram is a brilliant *prior* — it estimates the effective range of each variable's spatial autocorrelation via a Bayesian Matérn fit — but it is performance-blind. The bandwidth that minimizes the Moran's I zero-crossing residual is not the same as the bandwidth that minimizes the trained model's out-of-fold composite loss. The two agree on smooth Gaussian fields and diverge on data with feature interactions, non-stationary autocorrelation, or strong physics constraints — which is most real spatial data.

## Goals

1. Establish a versioned, machine-readable **decision protocol** that every pipeline stage emits, so reasoning becomes inspectable, testable, and durable.
2. Introduce a **central orchestrator** (`sparc/run/orchestrator.py`) that owns stage dispatch, reads decision files, and applies feedback actions. Both the desktop WebSocket entry point and the CLI entry point delegate to it.
3. Ship the **first concrete feedback loop** — a Stage 2 surrogate-assisted bandwidth and k-neighbors search — that demonstrably improves base model performance over the correlogram-only baseline.
4. Guarantee the new system **cannot regress** today's pipeline quality: an improvement gate compares the search winner against the correlogram baseline and falls back to defaults if the win is below a configurable threshold.
5. Produce an auditable reasoning trace per stage that the desktop UI can render in a follow-up PR.

## Non-Goals

- LLM integration or natural language scenario specification — explicitly deferred.
- Inter-stage feedback (S3→S2 refutation-gated retraining) — second PR.
- Wide joint surrogate search over the neural meta-learner — third PR.
- S2→S1 GWEN re-run — deprioritized; users can manually toggle.
- New UI panels — the reasoning trace is queryable via existing `/results/*` endpoints in this PR; rendering is a follow-up.
- Replacement of the correlogram. Stage 0 is preserved unchanged and provides the prior the BO search consumes.

## Design Decisions

### D1. Decision protocol is a versioned Pydantic schema

Every stage produces a `StageDecision` object validated by Pydantic, persisted as a struct artifact in the existing `artifacts.db`. The schema has a `schema_version` field so future loops can extend it without breaking older readers.

Rejected alternatives:
- Free-form JSON (no programmatic reasoning, no UI contract)
- Minimal `{stage, status, action}` schema (forces breaking changes for every new loop)

### D2. Single orchestrator, two entry points delegate to it

`sparc/run/orchestrator.py` exposes `run_stage(stage, config, *, fast, skip_gwen, max_revisions=2) -> StageDecision`. Both `sparc/server/stream.py::_execute_stage` and `sparc/__main__.py::cmd_run` are slimmed down to a delegation call. The legacy `sparc/run/run_enhanced_pipeline.py` is left untouched.

Rejected alternatives:
- Evolve `_execute_stage` and `cmd_run` independently (two divergent reasoners — the bug we're trying to avoid)
- Subprocess + decision-file orchestrator (loses the rich return dict signal that's already available in Python)

### D3. Bayesian Optimization with `scikit-optimize`

The narrow inner loop uses `gp_minimize` from `scikit-optimize`. The correlogram's per-variable κ posterior mean and HDI seed the BO's Gaussian prior over log-bandwidth. ~50 trials per model.

Rejected alternatives:
- Grid search (discards the κ-posterior prior; no sample efficiency for the wider search later)
- BoTorch (heavier dependency; unnecessary at this scale)
- CMA-ES (better for the wider joint search later, but overkill for 2–3 dims)

### D4. Composite objective matches the training loss

The BO minimizes a composite of OOF MSE, PDE residual, monotonicity penalty, and absolute residual Moran's I — with λ weights borrowed from `sparc_joint_loss()` defaults. This aligns the search target with the actual training objective and explicitly closes the loop on the residual-autocorrelation diagnostic Stage 2 already prints but never acts on.

$$\mathcal{L}(\theta) = \text{MSE}_{\text{OOF}}(\theta) + \lambda_{\text{phys}} \cdot R_{\text{PDE}}(\theta) + \lambda_{\text{mono}} \cdot P_{\text{mono}}(\theta) + \lambda_{\text{spatial}} \cdot \lvert I_{\text{Moran,resid}}(\theta) \rvert$$

Rejected: pure OOF MSE (would optimize for the wrong target; produces models that look good on R² but violate physics).

### D5. Three-layer safety stack

Surrogate-induced bias is the central failure mode in surrogate-assisted optimization. The system guards against it with:

1. **Trust-region verification** — top-3 BO candidates re-fit with the *full* base model on spatial folds. Surrogate scores never determine the winner; verified scores do.
2. **Improvement gate** — promote new hyperparameters only if the verified composite score beats the correlogram-only baseline by ≥1% (configurable). Otherwise fall back to correlogram defaults. **This guarantees the search cannot regress today's pipeline quality.**
3. **Provenance tagging** — every artifact records whether it was correlogram-derived or surrogate-search-derived; full BO trace and verification scores attach to the `StageDecision`.

### D6. Sequential per-model search

GWR → GWRF → GGPGAM, each with its own ~50-trial BO run on a 2–3 dim space. Each base model has its own surrogate (`DifferentiableGWR`, `DifferentiableGWRF`, `DifferentiableGGPGAM`), so they're naturally independent. Parallelization is a small later refactor reusing existing per-model parallel scaffolding.

Rejected:
- Joint BO over all model hyperparams (search space inflates without strong cross-model couplings; surrogate quality drops)
- Parallel from day one (harder to debug; complicates decision-file ordering for negligible wall-time win on the first PR)

### D7. Correlogram is the prior, not the competition

Stage 0 is preserved unchanged. The κ posterior produced per variable becomes the BO prior's mean and the search bounds (HDI). The correlogram-derived bandwidths are computed as the **baseline** that the search must beat by ≥1% to promote. If the search fails to beat the baseline, the pipeline produces correlogram-only artifacts — equivalent to today's behavior.

## Technical Approach

### Module layout

```
sparc/run/
├── decisions.py                    # NEW — Pydantic StageDecision schema
├── orchestrator.py                 # NEW — central reasoner; entry-point shim
├── inner_loops/
│   ├── __init__.py                 # NEW
│   └── bandwidth_search.py         # NEW — narrow BO inner loop
└── enhanced_spatial_cv.py          # MODIFIED — calls inner loop, emits StageDecision

sparc/server/stream.py              # MODIFIED — _execute_stage delegates to orchestrator
sparc/__main__.py                   # MODIFIED — cmd_run delegates to orchestrator
requirements.txt                    # MODIFIED — add scikit-optimize
pyproject.toml                      # MODIFIED — add scikit-optimize
```

### `decisions.py` — schema

```python
class MetricEvaluation(BaseModel):
    name: str
    value: float
    threshold: float | None = None
    passed: bool | None = None
    direction: Literal["higher_is_better", "lower_is_better"]

class DecisionRecord(BaseModel):
    rule: str                              # e.g. "improvement_gate"
    evidence: dict[str, Any]               # e.g. {"baseline": 0.812, "candidate": 0.829, "delta_pct": 2.1}
    action: Literal["accept", "reject", "fallback", "retry", "escalate"]

class NextAction(BaseModel):
    kind: Literal["none", "retry_stage", "rerun_stage", "escalate"]
    stage: int | None = None
    reason: str
    config_overrides: dict[str, Any] = Field(default_factory=dict)

class ProvenanceInfo(BaseModel):
    source: str                            # "correlogram" | "surrogate_search" | "user_override"
    search_log: dict[str, Any] | None = None
    baseline_comparison: dict[str, Any] | None = None

class StageDecision(BaseModel):
    schema_version: int = 1
    stage: int
    status: Literal["passed", "failed", "needs_revision"]
    metrics: dict[str, MetricEvaluation]
    decisions: list[DecisionRecord]
    next_actions: list[NextAction]
    provenance: ProvenanceInfo
```

### `orchestrator.py` — shape

```python
def run_stage(
    stage: int,
    config: dict,
    *,
    fast: bool = False,
    skip_gwen: bool = False,
    max_revisions: int = 2,
) -> StageDecision:
    """Dispatch a stage, score its metrics, write the StageDecision artifact,
    and execute any next_actions (capped by max_revisions)."""
```

Phase-1 implementation: dispatch identical to today's `_execute_stage`, plus auto-population of `metrics` from each stage's return dict (R², RMSE, residual Moran's I for Stage 2; MC³ acceptance, NUTS R-hat for Stage 3; etc.). `next_actions` is empty for all stages **except** Stage 2, where the inner loop emits its own decisions.

### `inner_loops/bandwidth_search.py` — shape

```python
def run_narrow_bandwidth_search(
    *,
    model_name: Literal["gwr", "gwrf", "ggpgam"],
    surrogate: torch.nn.Module,           # already-trained DifferentiableXXX
    full_model_factory: Callable[..., Any], # builds the actual base model from hparams
    X: np.ndarray, y: np.ndarray, coords: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    correlogram_prior: dict,              # {bandwidth_mean, bandwidth_hdi, ...}
    correlogram_baseline_score: float,    # composite-objective value at correlogram defaults
    config: dict,
    n_trials: int = 50,
    n_verify: int = 3,
    improvement_threshold_pct: float = 1.0,
) -> tuple[dict, DecisionRecord, ProvenanceInfo]:
    """Run BO against the surrogate, verify top-K against the full model,
    apply improvement gate, return (best_hparams or correlogram_defaults,
    decision_record, provenance)."""
```

### Integration into `enhanced_spatial_cv.py`

After the correlogram-derived hyperparameters are loaded (~`get_variable_bandwidths` / `get_kernel_field`) and **before** the final spatial-CV fit:

1. Train the differentiable surrogates on a quick warm-start (this is already done as part of the V2 neural training path; we hoist surrogate fitting earlier when the feature flag is on).
2. Compute the correlogram-baseline composite score by running one full base-model fit at the correlogram-recommended hyperparams.
3. Call `run_narrow_bandwidth_search` per model.
4. Use the returned hyperparams (either search winner or correlogram fallback) for the actual Stage 2 fit.
5. Emit a `StageDecision` aggregating all per-model decisions.

The whole inner loop is gated by `pipeline.use_surrogate_search: bool` in `project.yml`, defaulting to `false` for the first PR so existing runs are byte-identical.

### Configuration surface (`project.yml`)

```yaml
pipeline:
  use_surrogate_search: false        # off by default in v1
  surrogate_search:
    n_trials: 50
    n_verify: 3
    improvement_threshold_pct: 1.0
    composite_lambdas:               # default to sparc_joint_loss() values
      phys: null                     # null = inherit from training config
      mono: null
      spatial: null
    bo_random_seed: 42
```

### Data flow

```
Stage 0 correlogram ──┬──> κ posterior + HDI ──> BO prior ──┐
                      └──> baseline hparams ──> verified baseline score ──┤
                                                                          ├──> BO search (50 trials)
Differentiable surrogates ────────────────────────────────────────────────┘
                                                                          │
                                                              top-K BO candidates
                                                                          │
                                                               trust-region verification
                                                                          │
                                                              improvement gate (+1%?)
                                                                          │
                                                          ┌───────────────┴───────────────┐
                                                  YES: promote                    NO: fallback to correlogram
                                                          │                               │
                                                          └────────┬──────────────────────┘
                                                                   ▼
                                                           Stage 2 fit + StageDecision
```

## Acceptance Criteria

1. `sparc/run/decisions.py` exists with the full Pydantic schema and round-trips through JSON without loss.
2. `sparc/run/orchestrator.py` exposes `run_stage()`; both `_execute_stage` and `cmd_run` delegate to it.
3. Every stage emits a valid `StageDecision` registered as a struct artifact in `artifacts.db` (queryable as `("decisions", f"stage_{n}_decision")`).
4. With `pipeline.use_surrogate_search: false`, a full pipeline run produces byte-identical model outputs to today (exception: the new decision artifacts).
5. With `pipeline.use_surrogate_search: true`, the Stage 2 inner loop runs end-to-end on at least one reference project (`my_project/brown4.csv`) without exception.
6. The improvement gate is exercised by both paths in tests:
   - **promotion path:** an injected synthetic candidate that beats baseline by ≥1% gets promoted, with provenance recording the win
   - **fallback path:** an injected candidate within ±1% of baseline triggers fallback to correlogram defaults, with provenance recording the rejection
7. Sequential per-model search produces three independent search-log sub-records in the StageDecision (one each for GWR, GWRF, GGPGAM).
8. New tests:
   - `tests/test_stage_decision_schema.py` — schema validation, version round-trip
   - `tests/test_orchestrator_dispatch.py` — orchestrator produces equivalent results to legacy dispatch when feedback is off
   - `tests/test_bandwidth_search_gate.py` — improvement gate promotion + fallback paths
   - `tests/test_bandwidth_search_smoke.py` — narrow BO end-to-end on a small synthetic dataset
9. `scikit-optimize` is added as a dependency in both `requirements.txt` and `pyproject.toml`.
10. Existing test suite passes with no modifications.

## Open Questions

1. **Surrogate hoisting cost.** The differentiable surrogates are currently trained as part of the V2 neural path (later in Stage 2). The inner loop needs them earlier. Estimated cost is one extra forward pass per surrogate; needs to be measured to confirm it doesn't dominate the search budget.
2. **Improvement threshold tuning.** 1% is a defensible default but may be too tight for noisy small datasets. Should it scale with the size of the κ HDI (wider HDI → looser threshold)? Defer to empirical results from the first reference-project runs.
3. **Trust-region size.** Top-3 verification is the proposed default. Top-5 would be safer but doubles cost. Defer until we see how often the BO's surrogate-best matches its full-model-best in practice.
4. **What to do when no surrogate is available** (e.g., the V2 neural path is disabled). Should the inner loop be silently skipped, or should it train a quick disposable surrogate just for the search? Default in v1: silently skip with a clear log message and `provenance.source = "correlogram"`.
5. **Where the decision UI panel lives.** Out of scope for this PR but the schema must support the eventual UI contract. Current schema is designed to be UI-friendly.
