/**
 * vizRegistry — central catalog mapping (stage_id, artifact_id) → viz kind.
 *
 * Mirrors the server-side `FIGURES_REGISTRY` (sparc/report/figures.py) so the
 * desktop knows what to render for each artifact. The `kind` field selects a
 * React component; `endpoint` is the API call (relative path) for the data.
 *
 * Used by DownloadMenu (Phase 6) and small-multiples grids (Phase 5) to ask
 * "what visualization should this artifact get?".
 */

export type VizKind =
  | "data_histogram"
  | "dag"
  | "correlogram"
  | "predictions_map"
  | "cate"
  | "pdp"
  | "sensitivity_tornado"
  | "dose_response"
  | "posteriors"
  | "posterior_trace"
  | "scenario_map"
  | "uncertainty_band_map"
  | "attribution_waterfall"
  | "scenario_trajectory"
  | "scenario_library_timeline"
  | "decision_allocation";

export interface VizEntry {
  stage: string;
  artifactId: string;
  kind: VizKind;
  /** Optional API endpoint to fetch the underlying data. */
  endpoint?: string;
  /** Human label for menus / tooltips. */
  label: string;
}

export const VIZ_REGISTRY: VizEntry[] = [
  // Stage 0 — EDA
  { stage: "0", artifactId: "correlogram_results", kind: "correlogram", endpoint: "/results/correlogram", label: "Correlogram" },
  // Stage 2 — Spatial CV / ensemble
  { stage: "2", artifactId: "spatial_cv_predictions", kind: "predictions_map", endpoint: "/results/spatial_cv/predictions", label: "Predictions map" },
  { stage: "2", artifactId: "ensemble_predictions", kind: "predictions_map", endpoint: "/results/spatial_cv/predictions", label: "Ensemble predictions map" },
  // Stage 3 — Causal validation
  { stage: "3", artifactId: "median_probability_dag", kind: "dag", endpoint: "/dag", label: "DAG (median probability)" },
  { stage: "3", artifactId: "edge_inclusion_probs", kind: "dag", endpoint: "/dag", label: "Edge inclusion probabilities" },
  { stage: "3", artifactId: "cate_summary", kind: "cate", endpoint: "/results/causal/cate_map", label: "CATE summary" },
  { stage: "3", artifactId: "spatial_cate_maps", kind: "predictions_map", endpoint: "/results/causal/cate_map", label: "Spatial CATE map" },
  { stage: "3", artifactId: "dose_response_curves", kind: "dose_response", endpoint: "/results/causal/dose_response", label: "Dose-response curves" },
  { stage: "3", artifactId: "parameter_posteriors", kind: "posteriors", endpoint: "/results/scenarios/nuts_summary", label: "Parameter posteriors" },
  // Stage 4 — Scenario contract
  { stage: "4", artifactId: "scenario_results", kind: "scenario_map", endpoint: "/results/scenarios", label: "Scenario map" },
  { stage: "4", artifactId: "scenario_results_dag", kind: "scenario_map", endpoint: "/results/scenarios", label: "Scenario map (DAG path)" },
  { stage: "4", artifactId: "scenario_results_reprediction", kind: "scenario_map", endpoint: "/results/scenarios", label: "Scenario map (re-prediction)" },
  { stage: "4", artifactId: "scenario_results_hybrid", kind: "scenario_map", endpoint: "/results/scenarios", label: "Scenario map (hybrid)" },
  { stage: "4", artifactId: "scenario_mc_uncertainty", kind: "uncertainty_band_map", endpoint: "/results/scenarios/uncertainty", label: "Uncertainty bands" },
];

/**
 * Prefix-keyed entries for per-feature artifact families. Used when the
 * artifact id is suffixed with a variable name (e.g. "v2_neural_pdp::ndvi").
 * Matched on longest-prefix in :func:`getVizEntry` if no exact key hits.
 */
export const VIZ_REGISTRY_PREFIX: VizEntry[] = [
  { stage: "2", artifactId: "v2_neural_pdp::", kind: "pdp", endpoint: "/results/pdp_curves", label: "Partial-dependence plot" },
  { stage: "2", artifactId: "pdp::", kind: "pdp", endpoint: "/results/pdp_curves", label: "Partial-dependence plot" },
];

const BY_KEY = new Map<string, VizEntry>(
  VIZ_REGISTRY.map((e) => [`${e.stage}__${e.artifactId}`, e]),
);

export function getVizEntry(stage: string, artifactId: string): VizEntry | undefined {
  const exact = BY_KEY.get(`${stage}__${artifactId}`);
  if (exact) return exact;
  // Longest-prefix fallback for per-feature families.
  let best: VizEntry | undefined;
  for (const entry of VIZ_REGISTRY_PREFIX) {
    if (entry.stage !== stage) continue;
    if (!artifactId.startsWith(entry.artifactId)) continue;
    if (!best || entry.artifactId.length > best.artifactId.length) {
      best = entry;
    }
  }
  return best;
}

export function getVizByStage(stage: string): VizEntry[] {
  return VIZ_REGISTRY.filter((e) => e.stage === stage);
}
