# Tasks: Scenario Simulation + Budget-Constrained Spatial Optimizer

Related PRD: docs/prd/prd-scenario-budget-optimizer.md

## Tasks

### T1 — Tier Combination: `_compute_combined_effect` in scenario_simulator.py
- [ ] Implement `compute_combined_effect(variable, effective_change, baseline_values, modified_values)` method on `ScenarioSimulator`
- [ ] Logic: if β(s_i) available AND condition curve available → `β(s_i) × f_PDP(Δ_i) × physics_guard`; if only β(s_i) → `β(s_i) × effective_change × physics_guard`; fallback chain unchanged
- [ ] Replace the `_compute_mgwr_direct_delta` call sites in `run_with_causal_dag` with `compute_combined_effect`
- [ ] PDP shape modifier: `f_PDP = |PDP_slope(Δ)| / mean(|PDP_slope|[active])` — compute from condition curve, clamp to [0.1, 10.0], apply to β(s_i) × effective_change
- [ ] Physics guard: reuse existing sign-check + ±3σ cap logic; apply as final step
- [ ] Unit test: verify tier-combination path is used when β(s) + condition curve both present

### T2 — Runtime Scenario API: `POST /scenarios/run` body
- [ ] Add `Optional[List[ScenarioSpec]]` Pydantic model to `app.py`; fields: `name`, `interventions: Dict[str, float]`, `unit_costs: Dict[str, float]`, `budget: float`, `equity_focus: float`
- [ ] When body provided: build simulator config from runtime spec (variable, direction inferred from sign of value, increment = abs value, single-increment list)
- [ ] When body absent: use project config scenarios if present; return 400 with helpful message if not
- [ ] Backward compat: existing calls with no body and a config with `scenarios:` still work

### T3 — New endpoint: `POST /scenarios/optimize`
- [ ] Accept `ScenarioSpec` + budget params (same body as T2, single scenario)
- [ ] Run `ScenarioSimulator.run_with_causal_dag()` to get per-cell deltas
- [ ] Fetch equity layer (from store or fetch on-demand via `census_equity`)
- [ ] Compute equity-modulated benefit vector: `benefit_i = [(1−α) + α·equity_i] · |delta_i|`
- [ ] Compute per-cell cost vector: `cost_i = Σ_v unit_cost_v × |Δ_v|`
- [ ] Call `budget.optimize(benefits, budget, costs)` → allocation vector
- [ ] Return: allocation GeoJSON (cell geometry + allocation + projected_delta + equity_score + estimated_cost), Pareto points, summary dict
- [ ] Run Pareto sweep at [0.5, 0.75, 1.0, 1.25, 1.5] × budget

### T4 — Equity Layer: upgrade `census_equity.py` to tract-level
- [ ] Add `fetch_tract_equity(bbox, census_api_key=None) -> pd.DataFrame` returning `{tract_geoid, geometry_wkt, poverty_rate, minority_pct, equity_score}`
- [ ] TIGER WMS tract geometry fetch: `https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_Current/MapServer/8/query` with spatial filter on bbox
- [ ] ACS poverty + minority fetch from `api.census.gov` (B17001, B01003, B03002)
- [ ] Spatial join: point-in-polygon each grid cell centroid to tract → assign equity_score
- [ ] API key support: pass `&key=<key>` on ACS requests when key is provided
- [ ] 429 handling: `warnings.warn(rate_limit_msg)` + single 5s retry
- [ ] Cache by `sha256(bbox_str)[:12]` to `~/.sparc/cache/census/`; TTL 7 days
- [ ] New function `get_per_cell_equity(data_df, census_api_key=None) -> np.ndarray` that takes the project DataFrame and returns a 0–1 array aligned to rows

### T5 — Equity as Stage-0 artifact
- [ ] In `sparc/run/` Stage-0 runner: after correlogram, call `get_per_cell_equity(data)` and register result as `("0", "equity_layer")` in the artifact store (columns: cell_id, equity_score, poverty_rate, minority_pct, tract_geoid)
- [ ] Add `GET /equity/layer` endpoint to `app.py`: reads `("0", "equity_layer")` from store; if missing, fetches on-demand; returns GeoJSON with equity_score property
- [ ] Add `api_keys.census` field to user preferences schema in `sparc/config/user_preferences.py`
- [ ] Expose `GET /api/preferences` response `api_keys` field (already exists; just add census key)
- [ ] Add Census API key input to SettingsPage.tsx

### T6 — Budget: equity-modulated scoring in `budget.py`
- [ ] Add `equity_scores: np.ndarray | None = None` and `equity_focus: float = 0.0` params to `optimize()`
- [ ] When equity_scores provided: apply `effective_benefits = benefits * [(1−α) + α·equity_scores]` before solver dispatch
- [ ] Expose through `budget_sweep()` as well
- [ ] Unit test: α=0 → same result as no equity; α=1 → cells with equity_score=0 never selected

### T7 — Strip hardcoded scenarios from templates
- [ ] Remove `scenarios:` and `joint_scenarios:` blocks from all 13 domain template `project.yml` files:
  - air_quality, blank, coastal, drought, forcesmip, geotechnical, groundwater, noise, seismic, stormwater, uhi, water_quality, wildfire
- [ ] Keep backward compat: `project.yml` loader should not error if scenarios block is absent
- [ ] Update `ScenarioSimulator.__init__` fallback: `self.scenarios = config.get("scenarios", [])` stays, but `run_with_causal_dag` and `run` no longer error if empty (they simply return empty results when no scenarios given at runtime)

### T8 — DecisionSupportPage: 4-step wizard expansion
- [ ] Step 1 (Scenario Builder): variable selector (from `getCateMapVariables()`), magnitude sliders, scenario name input. Replaces current "Candidates" step.
- [ ] Step 2 (Budget & Costs): dollar budget input, per-variable unit cost inputs, equity_focus slider (0–1)
- [ ] Step 3 (Equity Layer): `GET /equity/layer` → choropleth map showing equity_score; show fetch status + cache age; "Re-fetch" button
- [ ] Step 4 (Allocation): call `POST /scenarios/optimize`; show allocation choropleth on SpatialMap (cells colored by x_i), Pareto curve chart (canvas/SVG), summary card (Δ, cost, N cells, Gini), GeoPackage export button
- [ ] "Optimize →" button on ScenariosPage passes active scenario to DecisionSupportPage via query param or store
- [ ] Persist wizard state to localStorage under `sparc:decision-support:wizard:v2`

### T9 — GeoPackage export endpoint
- [ ] `GET /scenarios/optimize/export?format=gpkg` (or POST with same body): returns a GeoPackage file with columns `cell_id, equity_score, allocation, projected_delta, estimated_cost, tract_geoid`
- [ ] If geopandas + fiona available: write real GPKG; else fall back to GeoJSON download
- [ ] Trigger from DecisionSupportPage "Export Allocation" button

### T10 — Integration tests
- [ ] Test `compute_combined_effect`: β(s) + PDP + guard path vs. fallback paths
- [ ] Test `POST /scenarios/run` with body (no project.yml scenarios)
- [ ] Test `POST /scenarios/optimize` returns allocation + Pareto + summary
- [ ] Test `fetch_tract_equity` mocked HTTP → correct equity_score output
- [ ] Test `optimize()` with equity_focus=0 vs 1
