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
  { stage: "0", artifactId: "data_histogram", kind: "data_histogram", endpoint: "/data/histogram", label: "Data histogram" },
  { stage: "1", artifactId: "dag_structure", kind: "dag", endpoint: "/dag", label: "DAG" },
  { stage: "1", artifactId: "correlogram", kind: "correlogram", endpoint: "/results/correlogram", label: "Correlogram" },
  { stage: "2", artifactId: "predictions_map", kind: "predictions_map", endpoint: "/results/spatial_cv/predictions", label: "Predictions map" },
  { stage: "3", artifactId: "cate_map", kind: "cate", endpoint: "/results/causal/cate_map", label: "CATE map" },
  { stage: "3", artifactId: "pdp", kind: "pdp", endpoint: "/results/causal", label: "Partial-dependence plot" },
  { stage: "3", artifactId: "sensitivity_tornado", kind: "sensitivity_tornado", endpoint: "/results/causal/sensitivity", label: "Sensitivity tornado" },
  { stage: "3", artifactId: "dose_response", kind: "dose_response", endpoint: "/results/causal/dose_response", label: "Dose-response curve" },
  { stage: "3", artifactId: "posteriors", kind: "posteriors", endpoint: "/results/scenarios/nuts_summary", label: "Posterior densities" },
  { stage: "3", artifactId: "posterior_trace", kind: "posterior_trace", endpoint: "/results/scenarios/nuts_summary", label: "Posterior trace" },
  { stage: "4", artifactId: "scenario_results", kind: "scenario_map", endpoint: "/results/scenarios", label: "Scenario map" },
  { stage: "4", artifactId: "scenario_uncertainty", kind: "uncertainty_band_map", endpoint: "/results/scenarios/uncertainty", label: "Uncertainty bands" },
  { stage: "4", artifactId: "scenario_attribution", kind: "attribution_waterfall", endpoint: "/results/scenarios/attribution", label: "Attribution waterfall" },
  { stage: "4", artifactId: "scenario_trajectory", kind: "scenario_trajectory", endpoint: "/results/scenarios/trajectory", label: "Scenario trajectory" },
  { stage: "4", artifactId: "scenario_library", kind: "scenario_library_timeline", endpoint: "/scenarios/library", label: "Scenario library" },
  { stage: "5", artifactId: "decision_allocation", kind: "decision_allocation", endpoint: "/decision/optimize", label: "Decision allocation" },
];

const BY_KEY = new Map<string, VizEntry>(
  VIZ_REGISTRY.map((e) => [`${e.stage}__${e.artifactId}`, e]),
);

export function getVizEntry(stage: string, artifactId: string): VizEntry | undefined {
  return BY_KEY.get(`${stage}__${artifactId}`);
}

export function getVizByStage(stage: string): VizEntry[] {
  return VIZ_REGISTRY.filter((e) => e.stage === stage);
}
