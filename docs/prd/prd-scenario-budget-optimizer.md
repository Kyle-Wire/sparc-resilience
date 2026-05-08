# PRD: Scenario Simulation + Budget-Constrained Spatial Optimizer

## Problem Statement

SPARC produces rich causal estimates (NUTS β(s), PDP dose-response curves, physics priors) but exposes them only as static pipeline outputs. There is no way for a user to interactively say "increase tree canopy 25%, reduce impervious 15%, I have $2M — where do I spend it?" and get a spatially-explicit, equity-informed allocation map.

Additionally, scenarios are currently hardcoded in `project.yml` and cannot be created at runtime by the user.

## Goals

1. **Unified "final function":** Combine the three causal evidence tiers into a single per-cell effect estimator: `effect_i = β(s_i) × f_PDP(Δ_i) × physics_guard`.
2. **User-defined runtime scenarios:** Remove hardcoded `scenarios:` from `project.yml`. Accept scenario specs at runtime via `POST /scenarios/run` body.
3. **Budget-constrained spatial allocator:** Given a dollar budget and $/unit-of-treatment cost per variable, find the optimal set of cells to intervene on (greedy ROI-first).
4. **Equity layer from Census TIGER:** Auto-fetch ACS 5-year tract data (poverty + minority %) and spatial-join to project grid → composite equity score 0–1 stored as Stage-0 artifact.
5. **Equity-modulated allocation:** A 0–1 equity slider that scales each cell's score by `(1−α)·benefit/cost + α·equity_score·benefit/cost`, where α is the slider value.
6. **Decision Support destination:** The `DecisionSupportPage` becomes the full "Scenario → Budget → Allocation" workflow: allocation map, Pareto frontier, summary stats, GeoPackage export.

## Non-Goals

- Replacing the existing causal pipeline stages (0–3); this is purely a Stage-4 / decision-layer feature.
- External SVI/HOLC data sources (only Census ACS via TIGER for now).
- Real-time re-running of Stages 1–3 when a scenario changes.
- Multi-objective optimization beyond the equity-weighted ROI scalar.

## Design Decisions

### D1 — Tier Combination (Option A)
The three evidence tiers combine multiplicatively rather than as a sequential waterfall:

```
effect_i = β(s_i) · f_PDP(Δ_i) · physics_guard(v, effect_i)
```

- **β(s_i):** NUTS posterior mean per-cell spatial CATE from `cate_summary` (Stage 3). This is the primary causal estimator.
- **f_PDP(Δ_i):** Non-linear saturation shape from the dose-response / GWRF condition curve. Computed as a dimensionless shape modifier: `f_PDP = |PDP_slope(Δ)| / mean(|PDP_slope|)` over active cells. Applied whenever a condition curve exists for the treatment variable with R² ≥ threshold.
- **physics_guard:** Sign check + ±3σ magnitude cap from the physics literature prior. Acts as a bounding constraint, not a coefficient.

When β(s_i) is unavailable (Stage 3 not run), the current fallback chain (saturation curve → physics literal) continues to operate unchanged.

This combination replaces the existing `pde_alpha_field_pdp` Tier 0 path, which was the closest prior art but only activated when a PDE alpha field was present. The new combined path activates whenever β(s_i) exists.

### D2 — Runtime Scenario API
`POST /scenarios/run` gains an optional `scenarios` body parameter:

```json
{
  "scenarios": [
    {
      "name": "Green Infrastructure",
      "interventions": { "Pct_Canopy": 25, "Pct_Impervious": -15 },
      "unit_costs":    { "Pct_Canopy": 500, "Pct_Impervious": 200 },
      "budget":        2000000,
      "equity_focus":  0.6
    }
  ]
}
```

When a body is provided, it overrides the config. When no body is provided (legacy call), the config is used **only if** scenarios exist there; otherwise a 400 is returned asking the user to define scenarios in the builder.

The `scenarios:` block in `project.yml` is deprecated and stripped from all domain templates. Any existing project.yml with scenarios still runs as before (backward compatible).

### D3 — Multi-Intervention Joint Scenarios
When a scenario contains multiple variables (e.g. canopy +25% AND impervious −15%), they are run jointly through `run_with_causal_dag`. The DAG handles:
- Indirect pathway interactions (e.g. canopy → NDVI → temperature)
- Co-intervention skip_variables to prevent double-counting
- Per-cell effect: `Σ_v [β(s_i,v) × f_PDP(Δ_i,v) × guard_v]` + DAG indirect sum

### D4 — Equity Layer Fetch
`sparc/data/census_equity.py` is upgraded from county-level to **tract-level** with spatial join:

1. Derive bounding box from project grid (lat/lon extents from data).
2. Fetch ACS 5-year tract geometries from TIGER: `https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_Current/MapServer/8/query`
3. Fetch ACS demographics per tract: poverty rate (`B17001_002E / B17001_001E`) + minority % (`(B01003_001E - B03002_003E) / B01003_001E`).
4. Spatial join: for each project grid cell, find the covering tract and assign its composite score.
5. Composite: `equity_score_i = 0.5 × poverty_rate_i + 0.5 × minority_pct_i`, min-max normalised to [0, 1].
6. Store as `("0", "equity_layer")` in the artifact store.

Census API key: optional, stored in user preferences under `api_keys.census`. HTTP requests include the key when present. Rate limit 429 responses trigger a `warnings.warn` and a 5-second retry (one attempt). Fetch is cached to `~/.sparc/cache/census/` by bounding-box hash; TTL 7 days.

### D5 — Budget Model
Each intervention variable has a `unit_cost: float` ($/unit of change per grid cell). Budget is in the same currency. The per-cell cost for variable `v` at cell `i` with allocation `x_i ∈ [0, 1]` is:

```
cost_i = unit_cost_v × |Δ_v| × x_i
```

For joint scenarios, total per-cell cost = `Σ_v unit_cost_v × |Δ_v| × x_i` (single allocation scalar since all variables are applied together).

The `optimize()` function in `budget.py` receives:
- `benefits[i]` = equity-modulated per-cell effect (see D6)
- `costs[i]` = total per-cell cost for full intervention
- `budget` = dollar budget from the scenario
- Returns allocation `x_i ∈ [0, 1]` per cell.

### D6 — Equity-Modulated Scoring
The equity slider value `α ∈ [0, 1]` is applied at the scoring step before the greedy solver:

```
score_i = [(1 - α) + α · equity_score_i] · benefit_i / cost_i
```

At α=0: pure ROI. At α=1: equity fully modulates ROI (cells with equity_score=0 get zero score regardless of benefit). At α=0.5: equal blend. ROI ordering is preserved by design — equity amplifies the score of disadvantaged cells rather than bypassing the efficiency constraint.

### D7 — Output and Page Routing
`DecisionSupportPage` is expanded to a 4-step wizard:
1. **Scenario Builder** — pick variables, set magnitudes (sliders), name the scenario. Links to from ScenariosPage.
2. **Budget & Costs** — dollar budget, $/unit cost per variable, equity slider.
3. **Equity Layer** — show fetched ACS map, allow manual equity_focus override.
4. **Allocation** — allocation map (cells colored by x_i intensity), Pareto frontier (benefit vs. budget sweep), summary card (total projected Δ, cost, N cells treated, equity Gini), GeoPackage export.

`ScenariosPage` retains its exploration role (NUTS posteriors, per-scenario Δ histograms, scenario library). An "Optimize →" button passes the active scenario to DecisionSupportPage.

## Technical Approach

### Layer 1 — `sparc/interventions/scenario_simulator.py`
- Refactor `_compute_mgwr_direct_delta` to apply the Option A combination whenever β(s_i) is available: compute PDP shape modifier from condition curve, multiply β(s_i) × f_PDP × effective_change, apply physics guard.
- New method `compute_combined_effect(variable, effective_change, baseline_values, modified_values) → (delta_array, source_label)` that implements the tier combination cleanly.

### Layer 2 — `sparc/server/app.py`
- `POST /scenarios/run`: Accept `Optional[List[ScenarioSpec]]` body. Build simulator config from runtime spec. Deprecate reading scenarios from project config in this endpoint.
- New `POST /scenarios/optimize` endpoint: accepts scenario spec + budget params → runs simulator → runs budget allocator → returns allocation GeoJSON + Pareto points + summary.
- `GET /equity/layer`: Returns cached equity layer as GeoJSON, fetching from Census if not in store.
- `GET /api/preferences`: Add `api_keys.census` field to the preferences schema.

### Layer 3 — `sparc/data/census_equity.py`
- Upgrade from county-level to tract-level with TIGER spatial join.
- Add API key support (`census_api_key` parameter).
- Rate limit handling (429 → warn + single retry).
- Return per-cell equity score array aligned to project grid.

### Layer 4 — `sparc/scenario/budget.py`
- `optimize()` already accepts benefit/cost arrays. Add `equity_scores` parameter and `equity_focus` float to apply D6 scoring before solving.

### Layer 5 — `sparc-desktop/src/components/pages/DecisionSupportPage.tsx`
- Expand to 4-step wizard per D7.
- Equity layer map panel (uses `/equity/layer`).
- Pareto curve chart.
- Allocation map layer in SpatialMap.
- GeoPackage download button.

### Layer 6 — Templates
- Remove `scenarios:` and `joint_scenarios:` blocks from all domain template `project.yml` files (14 templates).
- Update `project.yml` schema documentation.

## Acceptance Criteria

1. User can define a scenario (Pct_Canopy +25%, Pct_Impervious −15%) in the DecisionSupportPage builder with no `project.yml` edit.
2. Running the scenario produces per-cell deltas using `β(s_i) × f_PDP(Δ_i) × physics_guard` when NUTS results exist; falls back gracefully otherwise.
3. Setting a $1M budget and $500/cell unit cost produces an allocation map with ≤ 2000 cells treated, sorted by ROI.
4. Setting equity_focus=1.0 concentrates allocation visibly toward high equity_score cells vs equity_focus=0.0.
5. Census equity layer is fetched automatically on first use; subsequent loads use the 7-day cache.
6. All domain template `project.yml` files no longer contain a `scenarios:` block.
7. Legacy projects with `scenarios:` in `project.yml` still run without error.
8. Pareto frontier chart shows benefit at 50/75/100/125/150% of budget.
9. GeoPackage export of allocated cells works and includes `equity_score`, `allocation`, `projected_delta`, `estimated_cost` columns.

## Open Questions

None — all major design decisions resolved in grill-me session.
