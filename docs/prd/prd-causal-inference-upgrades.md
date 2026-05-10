# PRD: Causal Inference Upgrades

**SPARC Labs LLC | May 2026**
**Status: Ready for Implementation**

---

## Problem Statement

SPARC's causal inference stack covers the Wager (2025) audit gaps and produces defensible ATEs and CATEs. However, five concrete gaps remain: (1) there is no mediation decomposition — the total effect through a mediator cannot be split into direct and indirect components; (2) causal discovery findings are write-only — PC/LiNGAM/GES run and report discrepancies but those discrepancies never reach NUTS, which always runs on the original expert DAG regardless of what the data shows; (3) sensitivity analysis is E-value only with no bounding of the *range* of true effects under partial unmeasured confounding; (4) spatial spillover is first-order only, missing second-ring and distance-decay models; (5) dynamic treatment regime optimization is correctly deferred as out of scope for land-cover interventions.

---

## Goals

1. Add mediation decomposition (NDE/NIE/CTE) — both linear and nonlinear paths.
2. Fix the discovery → NUTS gap: user reviews, approves or rejects, and re-runs Stage 3 with full information.
3. Add Rosenbaum partial-identification bounds to sensitivity analysis.
4. Extend `NetworkInterferenceModel` with second-order and kernel-weighted spillover modes.

## Non-Goals

- Optimal dynamic treatment regimes / Q-learning (deferred — not relevant to land-cover interventions).
- Auto-accepting DAG changes without user review.
- Modifying the expert DAG YAML file on disk at any point.
- Wind-aware / directional spillover (deferred pending meteorological input integration).

---

## Design Decisions

### 1. Mediation Decomposition

**New file:** `sparc/causal/mediation.py`

**Class:** `MediationDecomposer`

**Approach:** Dual-path estimation:
- **Linear path** — product-of-coefficients. `NDE = α`, `NIE = β·γ` where `α` is the direct T→Y coefficient with M controlled, `β` is T→M, `γ` is M→Y. Bootstrap CIs (1000 draws). Fast — runs always.
- **Nonlinear path** — g-computation via Monte Carlo integration. Uses the same cross-fit nuisance learners already fitted by `CounterfactualEngine._fit_edge_dml_sklearn()` — no duplicated fitting. Resamples counterfactual outcomes under `do(T=t, M=m*)` and `do(T=t*, M=m*)` where `m*` is the natural value of M under each treatment arm. 500 MC draws by default.

**Integration:** `CounterfactualEngine.fit()` calls `MediationDecomposer` after DML fit completes, once per mediator node identified in `dag_definition.get_node_roles()["mediators"]`. Results added to `stage3/causal_results.json` under `mediation`.

**Output per mediator path:**
```
{
  "treatment": "Pct_Canopy",
  "mediator": "NDVI",
  "outcome": "LST",
  "NDE_linear": float,
  "NIE_linear": float,
  "CTE_linear": float,
  "NDE_linear_ci": [lower, upper],
  "NIE_linear_ci": [lower, upper],
  "NDE_nonlinear": float,
  "NIE_nonlinear": float,
  "NDE_nonlinear_ci": [lower, upper],
  "NIE_nonlinear_ci": [lower, upper]
}
```

### 2. Discovery → NUTS Bug Fix

**Root cause confirmed:** `run_bayesian_causal()` builds `gate_payload` with only MC³ edge probabilities and `median_dag`. The `discovery_report` from `run_causal_discovery()` (stored on `CausalValidator.discovery_report`) is never included. NUTS always receives the original unmodified `dag_def`.

**Fix — two changes:**

**Change A — `v2_bayesian_causal.py`:** Include discovery findings in the gate payload:
```python
gate_payload = {
    "node_names": available_cols,
    "edge_probs": edge_probs.tolist(),
    "mc3_summary": mc3_summary,
    "median_dag": median_dag,
    # NEW: include discovery discrepancies for user review
    "discovery_report": discovery_report,  # passed as parameter
}
```

`run_bayesian_causal()` gains an optional `discovery_report: dict | None = None` parameter, threaded from `causal_validation.py` which already holds `self.discovery_report`.

**Change B — gate semantics remain simple:** Gate is approve/reject only. If rejected, Stage 3 halts with a clear message: "DAG discrepancies require review. Update your dag.yml and re-run Stage 3." No in-flight DAG patching. No structured edge accept/reject lists. The user makes changes in the UI/YAML and re-runs — keeping the expert DAG as ground truth throughout.

### 3. Sensitivity Bounds (Rosenbaum Partial-ID)

**File:** `sparc/causal/sensitivity.py` — add `sensitivity_bounds()` function alongside existing `e_value_continuous()`.

**Signature:**
```python
def sensitivity_bounds(
    effect: float,
    se: float,
    gamma: float,  # max odds ratio for unmeasured confounding strength
    *,
    alpha: float = 0.05,
) -> SensitivityBounds:
```

**Method:** Rosenbaum sensitivity analysis adapted for continuous outcomes. Under a confounder with maximum odds ratio `gamma`, the true effect is bounded:
```
lower = effect - SE · z_{1-α/2} · sqrt(gamma)
upper = effect + SE · z_{1-α/2} · sqrt(gamma)
```
This gives the range of true effects consistent with the observed estimate under confounders of strength ≤ `gamma`. Returns `SensitivityBounds(lower, upper, gamma, null_included: bool)`.

**Integration:** Stage 3 calls `sensitivity_bounds()` for each treatment ATE with a `gamma` configurable via `project.yml → causal.sensitivity.gamma` (default: 1.5, meaning a confounder that changes treatment odds by 50%).

**Output added to `scenario_coefficients.json`** under `sensitivity` per treatment.

### 4. Higher-Order Spillover

**File:** `sparc/causal/interference.py`

**Change:** Add `spillover_order` parameter to `NetworkInterferenceModel.__init__()`:

- `spillover_order=1` — current behavior, backward-compatible. `H_i = (W_i, mean(W_{N_i}))`.
- `spillover_order=2` — adds second-ring regressor. `H_i = (W_i, mean(W_{N_i^1}), mean(W_{N_i^2}))` where `N_i^2` is neighbors-of-neighbors excluding first ring. Requires `k` neighbors for ring 1 and `k` neighbors of each ring-1 neighbor for ring 2. Second-ring lookup uses the same `cKDTree`.
- `spillover_order='kernel'` — replaces mean-aggregation with Matérn distance-decay weighted sum: `H_i = (W_i, Σ K_ν(d_{ij}) · W_j / Σ K_ν(d_{ij}))` where `K_ν` uses the Matérn kernel from `kernel_field.py`. Requires `kernel_field` argument to be passed. Falls back to `order=1` if `kernel_field` is None.

HAC variance computation and SUTVA spatial-permutation test unchanged.

**Config:** `project.yml → causal.interference.spillover_order` (default: `1`).

---

## Technical Approach

### File Changes

| File | Change Type | Summary |
|------|-------------|---------|
| `sparc/causal/mediation.py` | **New** | `MediationDecomposer` — linear + nonlinear NDE/NIE/CTE |
| `sparc/causal/counterfactual_engine.py` | Edit | Call `MediationDecomposer` post-fit, include results in output |
| `sparc/causal/sensitivity.py` | Edit | Add `sensitivity_bounds()` + `SensitivityBounds` dataclass |
| `sparc/causal/interference.py` | Edit | Add `spillover_order` param + ring-2 + kernel-weighted paths |
| `sparc/run/v2_bayesian_causal.py` | Edit | Add `discovery_report` param, include in gate payload |
| `sparc/run/causal_validation.py` | Edit | Pass `self.discovery_report` to `run_bayesian_causal()` |

### Data Flow After Fix

```
Stage 3 start
  → run_causal_discovery()          stores self.discovery_report
  → run_bayesian_causal(
        discovery_report=self.discovery_report
    )
      → MC³ runs
      → gate_payload = {mc3, median_dag, discovery_report}
      → approval_gate(gate_payload)  ← user sees BOTH MC³ + discovery
      → if rejected: halt with message
      → if approved: NUTS runs on dag_def (expert DAG, unchanged)
  → CounterfactualEngine.fit()
      → DML nuisance fit per edge
      → MediationDecomposer.decompose() per mediator path
  → sensitivity_bounds() per treatment ATE
  → NetworkInterferenceModel(spillover_order=cfg) per treatment
```

---

## Acceptance Criteria

1. **Mediation:** `sparc/causal/mediation.py` passes unit tests with known linear DGP where NDE + NIE = total effect (within bootstrap CI).
2. **Discovery → NUTS fix:** Running Stage 3 with a discovery discrepancy present shows the discrepancy in the approval gate UI payload. NUTS receives unmodified `dag_def` whether approved or rejected (gate is information-only, not DAG-modifying).
3. **Sensitivity bounds:** `sensitivity_bounds(effect=0.5, se=0.1, gamma=2.0)` returns `null_included=False` for a strong effect; `sensitivity_bounds(effect=0.15, se=0.1, gamma=2.0)` returns `null_included=True`.
4. **Spillover:** `spillover_order=2` produces a `SpilloverDecomposition` with three components (direct, ring-1, ring-2). `spillover_order='kernel'` uses Matérn-weighted neighbor mean and produces the same dataclass shape as `order=1`.
5. All existing Stage 3 tests pass unchanged (backward compatibility).

## Open Questions

- None — all major decisions resolved.
