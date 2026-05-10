# Tasks: Causal Inference Upgrades

Related PRD: docs/prd/prd-causal-inference-upgrades.md

---

## Tasks

### Bug Fix (ship first — unblocks Stage 3 accuracy)

- [ ] **C1** — Fix discovery → NUTS gap in `v2_bayesian_causal.py`
  - Add `discovery_report: dict | None = None` param to `run_bayesian_causal()`
  - Include it in `gate_payload` dict
  - Update call site in `causal_validation.py` to pass `self.discovery_report`
  - Confirm NUTS still receives original unmodified `dag_def` regardless of gate outcome

### New Module

- [ ] **C2** — Implement `sparc/causal/mediation.py`
  - `MediationDecomposer` class with `decompose(treatment, mediator, outcome, data, nuisance_models)` method
  - Linear path: product-of-coefficients with bootstrap CIs (1000 draws)
  - Nonlinear path: g-computation (500 MC draws) using passed-in nuisance learners
  - Output: `MediationResult` dataclass with NDE/NIE/CTE + CIs for both paths

- [ ] **C3** — Wire `MediationDecomposer` into `CounterfactualEngine`
  - Call after DML fit, once per mediator node from `get_node_roles()["mediators"]`
  - Add `mediation` key to counterfactual engine output dict
  - Ensure `stage3/causal_results.json` includes mediation results

### Edits to Existing Modules

- [ ] **C4** — Add `sensitivity_bounds()` to `sparc/causal/sensitivity.py`
  - New `SensitivityBounds` dataclass: `lower, upper, gamma, null_included`
  - `sensitivity_bounds(effect, se, gamma, *, alpha=0.05)` implementation
  - Wire into Stage 3 call chain: one call per treatment ATE
  - Read `gamma` from `project.yml → causal.sensitivity.gamma` (default 1.5)
  - Add results to `scenario_coefficients.json` under `sensitivity`

- [ ] **C5** — Extend `NetworkInterferenceModel` in `sparc/causal/interference.py`
  - Add `spillover_order: int | str = 1` param (1, 2, or `'kernel'`)
  - Implement ring-2 neighbor lookup via second-pass `cKDTree` query
  - Implement kernel-weighted path using `matern_kernel_weights` from `kernel_field.py`
  - Kernel path: add `kernel_field=None` param; fall back to `order=1` if None
  - `SpilloverDecomposition` dataclass gains optional `ring2_effect` / `ring2_se` fields
  - Read `spillover_order` from `project.yml → causal.interference.spillover_order`

### Tests

- [ ] **C6** — Unit tests
  - `test_mediation.py`: known linear DGP, assert NDE + NIE ≈ total effect within CI
  - `test_sensitivity_bounds.py`: assert `null_included` logic for strong vs. weak effects
  - `test_interference_orders.py`: assert ring-2 produces 3-component decomposition; kernel path produces valid weights summing ≈ 1
  - `test_causal_discovery_gate.py`: assert `discovery_report` present in gate payload; assert `dag_def` unmodified after gate call
