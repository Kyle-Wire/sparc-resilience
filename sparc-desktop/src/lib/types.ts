/**
 * Shared TypeScript types matching the FastAPI server's response shapes.
 */

export interface HealthResponse {
  status: string;
  project_loaded: boolean;
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
  numeric_summary: Record<string, { mean: number; std: number; min: number; max: number }>;
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
