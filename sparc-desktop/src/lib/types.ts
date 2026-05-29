/**
 * Shared TypeScript types matching the FastAPI server's response shapes.
 */

export interface HealthResponse {
  status: string;
  project_loaded: boolean;
  is_running: boolean;
  current_stage: number | null;
  /** True if the run registry was successfully attached to the loaded project. */
  manifest_loaded?: boolean;
}

// ------------------------------------------------------------------
// Run registry (mirrors `sparc/registry/manifest.py`)
// ------------------------------------------------------------------

export type ArtifactFormat =
  | "json" | "csv" | "parquet" | "gpkg" | "npy" | "pkl"
  | "pt" | "png" | "txt" | "html" | "joblib" | "yaml"
  | "geojson" | "other";

/** Mirrors ``StorageKind`` in ``sparc/registry/manifest.py``. */
export type StorageKind =
  | "table" | "struct" | "blob_inline" | "blob_external" | "legacy_path";

export type StageStatus =
  | "pending" | "running" | "complete" | "failed" | "partial";

export interface ArtifactEntry {
  id: string;
  stage: string;
  name: string;
  /** Path relative to the project's `output_dir`. */
  path: string;
  format: ArtifactFormat;
  storage_kind?: StorageKind | null;
  /** SQLite table name when storage_kind is 'table'. */
  data_table?: string | null;
  /** FK into internal_blobs when storage_kind is 'blob_inline'. */
  blob_id?: number | null;
  /** Content hash for external blobs (storage_kind 'blob_external'). */
  blob_sha256?: string | null;
  producer?: string | null;
  consumers: string[];
  size_bytes?: number | null;
  sha256?: string | null;
  row_count?: number | null;
  written_at?: string | null;
  write_seconds?: number | null;
  partial: boolean;
  metadata: Record<string, unknown>;
}

export interface StageManifest {
  stage: string;
  status: StageStatus;
  started_at?: string | null;
  finished_at?: string | null;
  duration_seconds?: number | null;
  error?: string | null;
  artifacts: Record<string, ArtifactEntry>;
}

export interface RunManifest {
  schema_version: number;
  run_id?: string | null;
  project_name?: string | null;
  config_hash?: string | null;
  pipeline_version?: string | null;
  started_at?: string | null;
  updated_at?: string | null;
  stages: Record<string, StageManifest>;
}

/** Structured 404 body returned by registry-aware result endpoints. */
export interface MissingArtifactDetail {
  error: "missing_artifact";
  missing_artifact: string;
  produced_by_stage: string;
  expected_path?: string | null;
  candidate_paths: string[];
  hint: string;
}

export interface ProjectMeta {
  name?: string;
  description?: string;
  domain?: string;
  author?: string;
}

export interface ProjectLoadResponse {
  status: string;
  project: ProjectMeta;
  columns: string[];
  row_count: number;
}

export interface ValidationResponse {
  valid: boolean;
  error?: string;
  warnings: string[];
}

export interface InitResponse {
  status: string;
  template: string;
  path: string;
  project_yml: string;
}

export interface TemplateInfo {
  name: string;
  has_project_yml: boolean;
}

export interface DataSummary {
  row_count: number;
  column_count: number;
  columns: string[];
  dtypes: Record<string, string>;
  numeric_summary: Record<string, { mean: number; median: number; std: number; min: number; max: number; count?: number; q25?: number; q75?: number; "25%"?: number; "75%"?: number; [key: string]: number | undefined }>;
  crs?: string;
  bbox?: { minx: number; miny: number; maxx: number; maxy: number };
}

export interface DataPreview {
  rows: Record<string, unknown>[];
  total: number;
}

/** Structured event from the /run/execute WebSocket. */
export interface PipelineEvent {
  type: "log" | "metric" | "complete" | "error"
    | "capacity_result" | "epoch_update" | "curriculum_stage" | "convergence"
    | "dag_approval_requested" | "stage_status" | "training_health"
    | "run_progress" | "checkpoint" | "model_result" | "fold_start" | "fold_complete";
  message?: string;
  stage?: number;
  fold?: number;
  metric?: string;
  value?: number;
  /** Progress percentage (0–100) on run_progress events and checkpoint events with model weights. */
  pct?: number;
  /** Human-readable phase label emitted by the server for progress display. */
  phase?: string;
  /** Model name for model-level checkpoint events (e.g. "gwr", "ols"). */
  model?: string;
  /** 1-based index of current model in the sequence. */
  model_index?: number;
  /** Total number of models being trained. */
  model_total?: number;

  // ---- Training telemetry fields ----
  /** Capacity sweep: hidden dimension tested. */
  hidden_dim?: number;
  /** Capacity sweep: cross-validated R² for that dimension. */
  r2?: number;
  /** Epoch update: current epoch (1-based). */
  epoch?: number;
  /** Epoch update: total epochs in this training phase. */
  n_epochs?: number;
  /** Epoch update: total loss value for the epoch. */
  total_loss?: number;
  /** Epoch update: cv | retrain | swa. */
  train_phase?: "cv" | "retrain" | "swa";
  /** Epoch update: per-component loss breakdown. */
  components?: Record<string, number>;
  /** Curriculum stage: stage label (e.g. "Stage A"). */
  curriculum?: string;
  /** Curriculum stage: human-readable description. */
  label?: string;
  /** Convergence status: training | converging | converged. */
  status?: string;
  /** DAG gate: number of MC³ edges above threshold. */
  n_edges?: number;
  /** DAG gate: number of nodes in MC³ analysis. */
  n_nodes?: number;

  // ---- Phase 2: Status & ETA fields ----
  /** Stage status: running | complete | failed. */
  started_at?: number;
  completed_at?: number;
  error?: string;
  traceback?: string;
  /** ETA: estimated seconds remaining (from server throughput tracking). */
  eta_seconds?: number;
  /** ETA: elapsed seconds since this process started. */
  elapsed_seconds?: number;

  // ---- Phase 3: Training health fields ----
  /** Training health warning type: loss_explosion | physics_stagnation | gradient_spike. */
  warning?: string;
  /** Training health: which loss component triggered the warning. */
  component?: string;
  /** Training health: detail message. */
  detail?: string;

  // ---- Frontend-only fields (added on receipt, not from server) ----
  /** Wall-clock timestamp (ms since epoch) when the event was received by the frontend. */
  receivedAt?: number;
  // Allow additional event types from the server not yet defined here
  [key: string]: unknown;
}

/**
 * SPARC v4 DAG node types. The new types extend the original four:
 *   - `instrument`: exogenous variable affecting the outcome only via
 *     a treatment (IV identification).
 *   - `proxy_confounder`: an imperfect measurement of an unobserved
 *     confounder; flagged so the editor can warn about residual
 *     confounding.
 *   - `selection`: a variable that drives sample inclusion / dropout;
 *     conditioning on it can introduce collider bias.
 */
export type DagNodeType =
  | "treatment"
  | "mediator"
  | "confounder"
  | "outcome"
  | "instrument"
  | "proxy_confounder"
  | "selection";

export interface DagNode {
  name: string;
  type: DagNodeType;
  description?: string;
}

// ------------------------------------------------------------------
// Data validation
// ------------------------------------------------------------------
export interface ValidationItem {
  id: string;
  category: "missing" | "crs" | "duplicate" | "distribution" | "type" | "column";
  severity: "critical" | "warning" | "info";
  column?: string;
  message: string;
  detail?: string;
  value?: unknown;
}

export interface ValidationReport {
  passed: boolean;
  n_critical: number;
  n_warning: number;
  n_info: number;
  items: ValidationItem[];
  summary?: string;
}

// ------------------------------------------------------------------
// Data versioning
// ------------------------------------------------------------------
export interface DataVersion {
  version: number;
  filename: string;
  csv_filename?: string;
  path: string;
  csv_path?: string;
  timestamp: string;
  n_rows: number;
  n_cols: number;
  columns?: string[];
  description?: string | null;
  settings?: Record<string, unknown> | null;
  file_exists?: boolean;
}

/**
 * SPARC v4 DAG edge attributes.
 *   - `kind`: typing for the edge — `direct` (default), `instrumental`
 *     (an IV → treatment edge), `mediated` (channeled through a
 *     mediator), `time_lagged` (set automatically when `lag` is set).
 *   - `lag`: integer time-lag in pipeline time-steps. Replaces the
 *     legacy `temporal_edges:` block.
 *   - `sign_prior`: hypothesised direction (+ / − / 0). Drives edge
 *     colour in the editor and is checked against estimated effects.
 *   - `confidence`: 0..1 — how strongly the assumption is held. Drives
 *     edge opacity / stroke width in the editor.
 */
export interface DagEdge {
  parent: string;
  child: string;
  mechanism?: string;
  kind?: "direct" | "instrumental" | "mediated" | "time_lagged";
  lag?: number;
  sign_prior?: "+" | "-" | "0";
  confidence?: number;
}

/**
 * Top-level identification assumptions the user explicitly endorses.
 * The validator surfaces these alongside the d-separation / positivity
 * checks so reviewers can see what was claimed vs. what was tested.
 */
export interface DagAssumptions {
  conditional_exchangeability?: boolean;
  positivity?: boolean;
  consistency?: boolean;
  no_interference?: boolean;
  notes?: string;
}

export interface DagDefinition {
  nodes: DagNode[];
  edges: DagEdge[];
  assumptions?: DagAssumptions;
}

export interface DagValidation {
  valid: boolean;
  n_nodes: number;
  n_edges: number;
  error: string | null;
}

/** AI-suggested DAG edge from /dag/suggest-edges */
export interface DagEdgeSuggestion {
  parent: string;
  child: string;
  score: number;
  partial_r: number;
  toward_outcome: boolean;
  reason: string;
}

/** MC³ edge inclusion probabilities returned by /dag/mc3_result */
export interface MC3Result {
  node_names: string[];
  edge_probs: number[][];
  mc3_summary: {
    n_accepted: number;
    n_total: number;
    acceptance_rate: number;
    best_score: number;
    node_names: string[];
  };
  median_dag: {
    nodes: string[];
    edges: { source: string; target: string; probability: number }[];
  };
}

/** Full project configuration (mirrors project.yml structure) */
export interface ProjectConfig {
  project?: {
    name?: string;
    description?: string;
    domain?: string;
    version?: string;
    author?: string;
    response_units?: string;
  };
  data?: {
    file_path?: string;
    target_column?: string;
    identifier_column?: string;
    coord_columns?: string[];
    areas_of_interest_file?: string;
  };
  crs?: {
    input?: string;
    /** New canonical key for the projected working CRS. */
    working?: string;
    /** Legacy key — still accepted by the server; prefer `working`. */
    projected?: string;
  };
  predictors?: string[];
  physics?: {
    priors_file?: string;
    caps_file?: string;
    monotone_constraints?: Record<string, number>;
    literature_weight?: number;
    variable_bounds?: Record<string, { min: number | null; max: number | null }>;
  };
  causal?: {
    dag_file?: string;
    estimator?: string;
    estimate_cate?: boolean;
    actionable_variables?: string[];
    fixed_variables?: string[];
    dag_blend_weight?: number;
    bootstrap_n?: number;
    [key: string]: unknown;
  };
  models?: Record<string, Record<string, unknown>>;
  pipeline?: {
    random_seed?: number;
    n_spatial_folds?: number;
    fast_mode?: boolean;
    overwrite_outputs?: boolean;
    [key: string]: unknown;
  };
  gwen?: Record<string, unknown>;
  flags?: Record<string, boolean>;
  scenarios?: unknown;
  interaction_scenarios?: unknown;
  output?: Record<string, unknown>;
}

// ------------------------------------------------------------------
// Data Validation
// ------------------------------------------------------------------
export interface ValidationIssue {
  id: string;
  category: string;
  severity: "critical" | "warning" | "info";
  column?: string | null;
  message: string;
  detail?: string;
  value?: unknown;
}

/** Structured JSON actions returned by Claude in chat responses. */
export type ClaudeAction =
  | { action: "suggest_template"; template: string; predictors: string[]; reasoning: string }
  | { action: "propose_dag_edges"; edges: DagEdge[] }
  | { action: "suggest_physics"; monotonic_constraints: Record<string, number>; variable_bounds: Record<string, [number, number]>; combined_constraints?: unknown[] }
  | { action: "suggest_predictors"; predictors: string[]; reasoning: string };

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  actions?: ClaudeAction[];
}

// ------------------------------------------------------------------
// Correlogram
// ------------------------------------------------------------------
export interface CorrelogramVariableResult {
  variable: string;
  optimal_bandwidth: number;
  effective_range: number;
  optimal_block_size: number;
  best_kernel: string;
  max_moran_i: number;
  significant_lags: number;
  correlogram_results: {
    lag_distances: number[];
    morans_i_values: number[];
    z_scores: number[];
    p_values: number[];
  };
  spatial_parameters?: Record<string, unknown>;
}

export interface CorrelogramData {
  metadata?: Record<string, unknown>;
  individual_results: Record<string, CorrelogramVariableResult>;
  model_bandwidths?: Record<string, number | null>;
  spatial_cv_configuration?: Record<string, unknown>;
}

// ------------------------------------------------------------------
// Phase C — KernelField, causal PDP, divergence, scenario routing
// ------------------------------------------------------------------

export interface KernelFieldPredictor {
  name: string;
  kappa?: number | null;
  kappa_x?: number | null;
  kappa_y?: number | null;
  theta_rad?: number | null;
  is_anisotropic?: boolean;
  nu?: number | null;
  sigma2?: number | null;
  bandwidth_to_outcome?: number | null;
  kappa_posterior_samples?: number[] | null;
}

export interface KernelFieldPair {
  i: number;
  j: number;
  i_name: string;
  j_name: string;
  peak_sym: number;
  peak_lag_sym_m: number;
  peak_anti: number;
  peak_lag_anti_m: number;
  peak_angle_anti_deg: number | null;
  effective_range_sym_m: number | null;
  p_sym: number | null;
  p_anti: number | null;
  significant_sym: boolean;
  significant_anti: boolean;
}

export interface BandwidthMismatchWarning {
  pair: string;
  range_pair_m?: number | null;
  marginal_i_m?: number | null;
  marginal_j_m?: number | null;
  ratio_max?: number | null;
}

export interface KernelFieldData {
  outcome_name?: string;
  predictors?: KernelFieldPredictor[];
  variable_names?: string[];
  range_matrix?: (number | null)[][];
  bandwidth_mismatch_warnings?: Array<string | BandwidthMismatchWarning>;
  matern_default_nu?: number;
  /** Discriminator: "kernel_field" (legacy MGWR) | "cross_correlogram". */
  source?: string;
  // ----- cross_correlogram fallback fields -----
  n_angle_bins?: number;
  lag_distances?: number[];
  angle_centers_deg?: number[];
  pairs?: Record<string, KernelFieldPair>;
  antisymmetric_flags?: string[];
  n_pairs_per_bin?: number[][];
}

export interface CausalPDPCurve {
  treatment: string;
  dose_grid: number[];
  response_mean: number[];
  response_hdi_lo: number[];
  response_hdi_hi: number[];
  saturation_dose?: number | null;
  saturation_index?: number | null;
  peak_marginal_slope?: number;
  knee_marginal_slope?: number;
  source: "bayesian" | "frequentist" | "error" | "unknown";
  diagnostics?: Record<string, unknown>;
}

export interface CausalPDPData {
  curves: CausalPDPCurve[];
}

export interface DivergenceReport {
  treatment: string;
  cell_count: number;
  flagged_count: number;
  flagged_fraction: number;
  mean_abs_divergence: number;
  median_abs_divergence: number;
  max_abs_divergence: number;
  pearson_correlation: number;
  sign_agreement_rate: number;
  threshold_used: number;
  per_cell_divergence?: number[];
  flagged_mask?: boolean[];
  diagnostics?: Record<string, unknown>;
}

export interface AnisotropyEntry {
  name: string;
  is_anisotropic: boolean;
  kappa_x?: number | null;
  kappa_y?: number | null;
  theta_rad?: number | null;
  theta_deg?: number | null;
  range_x?: number | null;
  range_y?: number | null;
  aspect_ratio?: number;
  eccentricity?: number;
  bandwidth_to_outcome?: number | null;
  nu?: number | null;
  sigma2?: number | null;
}

export interface ScenarioRoutingAuditData {
  variables: string[];
  routing: Record<string, {
    source: "causal_cate" | "heuristic_disagreement" | "length_mismatch" | "missing";
    n_cells?: number;
    mean_multiplier?: number;
    std_multiplier?: number;
    min_multiplier?: number;
    max_multiplier?: number;
    n_cells_expected?: number;
    n_cells_actual?: number;
  }>;
  anisotropy: AnisotropyEntry[];
  kernel_field_outcome?: string | null;
  summary: {
    total_variables: number;
    causal_routed: number;
    correlational_routed: number;
    missing: number;
    anisotropic_predictor_count: number;
  };
}

// ------------------------------------------------------------------
// GWEN
// ------------------------------------------------------------------
export interface GwenRow {
  [key: string]: unknown;
}

// ------------------------------------------------------------------
// Causal
// ------------------------------------------------------------------
export interface SensitivityBounds {
  method: string;
  gamma: number;
  effect: number;
  se: number;
  lower_bound: number;
  upper_bound: number;
  null_included: boolean;
  interpretation: string;
}

export interface DirectEffect {
  structural_coeff: number;
  ate_backdoor?: number;
  cate_mean?: number;
  cate_std?: number;
  ate_ipw?: number;
  ate_gps?: number;
  att_matching?: number;
  ate_doubly_robust?: number;
  monotone_constraint?: number;
  placebo_pass?: boolean;
  rcc_pass?: boolean;
  data_subset_pass?: boolean;
  ucc_pass?: boolean;
  e_value?: number;
  bootstrap_ci_lower?: number;
  bootstrap_ci_upper?: number;
  bootstrap_se?: number;
  estimator_agreement?: boolean;
  max_discrepancy_pct?: number;
  elasticity?: number;
  relative_importance?: number;
  /** Rosenbaum partial-identification bounds (C4). Present when bootstrap_se > 0. */
  sensitivity_bounds?: SensitivityBounds;
}

export interface MediationDecomposition {
  direct_effect: number;
  direct_se?: number;
  indirect_total: number;
  indirect_se?: number;
  total_effect: number;
  total_se?: number;
  indirect_paths?: {
    mediator: string;
    beta_T_M: number;
    beta_M_Y: number;
    indirect_effect: number;
    sobel_se?: number;
    significant?: boolean;
  }[];
  mediation_proportion?: number;
}

export interface CausalResults {
  metadata?: Record<string, unknown>;
  direct_effects?: Record<string, DirectEffect>;
  mediator_propagation?: Record<string, Record<string, number>>;
  all_structural_coefficients?: Record<string, number>;
  propensity_diagnostics?: Record<string, Record<string, unknown>>;
  assumption_diagnostics?: Record<string, Record<string, unknown>>;
  mediation_decomposition?: Record<string, MediationDecomposition>;
  dose_response?: Record<string, Record<string, unknown>>;
  discovery_summary?: Record<string, unknown>;
}

// ------------------------------------------------------------------
// PDP Curves
// ------------------------------------------------------------------
export interface PdpVariable {
  pdp?: {
    grid_values: number[];
    pdp_values: number[];
    pdp_std?: number[];
  };
  curve_fit?: {
    curve_type: string;
    parameters: Record<string, number>;
    saturation_point?: number;
    r2?: number;
  };
  grid_values?: number[];
  pdp_values?: number[];
}

export type PdpSource = "neural_pde" | "gwrf" | "causal_dose_response";

/** Metadata injected by GET /results/pdp_curves to indicate which model
 *  sources have curve data and provide per-source curve maps. */
export interface PdpMeta {
  available_sources?: PdpSource[];
  by_source?: Record<PdpSource, Record<string, PdpVariable>>;
}

/** Full response shape from GET /results/pdp_curves.
 *  Top-level keys are variable names (PdpVariable values) plus the
 *  reserved `_meta` key (PdpMeta) that drives source selection. */
export interface PdpResponse {
  _meta?: PdpMeta;
  [variable: string]: PdpVariable | PdpMeta | undefined;
}

/** @deprecated Use PdpResponse. Kept for callers that only need the variable map. */
export type PdpCurves = Record<string, PdpVariable>;

// ------------------------------------------------------------------
// Dose-Response
// ------------------------------------------------------------------
export interface DoseResponseTreatment {
  dose_levels: number[];
  marginal_effects: number[];
  dose_means?: number[];
  bandwidth?: number;
  nonlinearity_ratio?: number;
  is_nonlinear?: boolean;
}

export type DoseResponseData = Record<string, DoseResponseTreatment>;

// ------------------------------------------------------------------
// Causal Diagnostics
// ------------------------------------------------------------------
export interface CausalDiagnosticsTreatment {
  cumulative_effect_curve?: {
    fractions: number[];
    cum_effects: number[];
    random_baseline?: number;
    area_ratio?: number;
  };
  calibration?: {
    bin_cate_mean: number[];
    bin_ate: number[];
    is_monotone?: boolean;
    rank_correlation?: number;
    rank_pvalue?: number;
  };
  rate_score?: number;
}

export type CausalDiagnostics = Record<string, CausalDiagnosticsTreatment>;

// ------------------------------------------------------------------
// Scenario Detail
// ------------------------------------------------------------------
export interface ScenarioSummaryRow {
  [key: string]: unknown;
}

export interface ScenarioDetail {
  geojson: GeoJsonData;
  summary: ScenarioSummaryRow[];
  /** Active mode-variant artifact id (base, dag, hybrid, or reprediction). */
  results_artifact_id?: string | null;
  summary_artifact_id?: string | null;
}

/** Mode-variant of the active scenario_results table. */
export type ScenarioVariant = "base" | "dag" | "hybrid" | "reprediction";

export interface GeoJsonFeature {
  type: "Feature";
  geometry: { type: string; coordinates: unknown };
  properties: Record<string, unknown>;
}

export interface GeoJsonData {
  type: "FeatureCollection";
  features: GeoJsonFeature[];
}

// ------------------------------------------------------------------
// Report
// ------------------------------------------------------------------
export interface ReportPayload {
  project?: Record<string, unknown>;
  data_summary?: Record<string, unknown>;
  predictors?: string[];
  causal?: Record<string, unknown>;
  physics?: Record<string, unknown>;
  pipeline?: Record<string, unknown>;
  correlogram?: Record<string, Record<string, unknown>>;
  gwen?: Record<string, unknown>[];
  spatial_cv_models?: string[];
  causal_results?: CausalResults;
  scenario_summary?: ScenarioSummaryRow[];
  /** Active scenario_results artifact id chosen by ScenarioBundle precedence. */
  scenario_results_artifact_id?: string | null;
  /** Active scenario_summary artifact id chosen by ScenarioBundle precedence. */
  scenario_summary_artifact_id?: string | null;
  plots?: Record<string, { name: string; filename: string; path: string; stage: number }[]>;
}
