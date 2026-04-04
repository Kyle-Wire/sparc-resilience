/**
 * Shared TypeScript types matching the FastAPI server's response shapes.
 */

export interface HealthResponse {
  status: string;
  project_loaded: boolean;
  project_path: string | null;
  is_running: boolean;
  current_stage: number | null;
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

/** Structured event from the /run/stream WebSocket. */
export interface PipelineEvent {
  type: "log" | "metric" | "complete" | "error";
  message?: string;
  stage?: number;
  fold?: number;
  metric?: string;
  value?: number;
  progress_pct?: number;
  /** Human-readable phase label emitted by the server for progress display. */
  phase?: string;
  /** Model name for model-level checkpoint events (e.g. "gwr", "ols"). */
  model?: string;
  /** 1-based index of current model in the sequence. */
  model_index?: number;
  /** Total number of models being trained. */
  model_total?: number;
}

export interface DagNode {
  name: string;
  type: "treatment" | "mediator" | "confounder" | "outcome";
  description?: string;
}

export interface DagEdge {
  parent: string;
  child: string;
  mechanism?: string;
}

export interface DagDefinition {
  nodes: DagNode[];
  edges: DagEdge[];
}

export interface DagValidation {
  valid: boolean;
  n_nodes: number;
  n_edges: number;
  error: string | null;
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
// GWEN
// ------------------------------------------------------------------
export interface GwenRow {
  [key: string]: unknown;
}

// ------------------------------------------------------------------
// Causal
// ------------------------------------------------------------------
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
}

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
  plots?: Record<string, { name: string; filename: string; path: string; stage: number }[]>;
}
