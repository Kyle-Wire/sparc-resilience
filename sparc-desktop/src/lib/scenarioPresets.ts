/**
 * Domain-aware scenario presets surfaced in the Scenarios page library
 * when no project scenarios are defined yet.  Each preset is intentionally
 * coarse (fractional standardized intervention magnitudes) so they apply
 * regardless of the actual variable scale.
 */
export interface ScenarioPreset {
  id: string;
  name: string;
  description: string;
  domain: string | "*";
  /** Map of variable substring → standardized delta (-1..+1). */
  interventionPattern: Record<string, number>;
}

export const SCENARIO_PRESETS: ScenarioPreset[] = [
  {
    id: "preset-tree-canopy-10",
    name: "+10% tree canopy",
    description: "Increase canopy cover by 10% across the study area.",
    domain: "uhi",
    interventionPattern: { canopy: 0.10, tree: 0.10, ndvi: 0.05 },
  },
  {
    id: "preset-impervious-down-10",
    name: "−10% impervious",
    description: "Reduce impervious surface fraction by 10%.",
    domain: "uhi",
    interventionPattern: { impervious: -0.10, paved: -0.10 },
  },
  {
    id: "preset-cool-roof-25",
    name: "Cool-roof retrofit (25%)",
    description: "Raise albedo on 25% of roofs.",
    domain: "uhi",
    interventionPattern: { albedo: 0.05, building: 0.0 },
  },
  {
    id: "preset-pumping-down-20",
    name: "−20% pumping",
    description: "Reduce well pumping rates 20% basin-wide.",
    domain: "groundwater",
    interventionPattern: { pumping: -0.20, well: -0.20, withdrawal: -0.20 },
  },
  {
    id: "preset-recharge-up-15",
    name: "+15% recharge",
    description: "Increase managed recharge by 15%.",
    domain: "groundwater",
    interventionPattern: { recharge: 0.15, infiltration: 0.10 },
  },
  {
    id: "preset-emissions-down-30",
    name: "−30% NOx emissions",
    description: "Reduce mobile + point NOx by 30%.",
    domain: "air_quality",
    interventionPattern: { nox: -0.30, traffic: -0.20, emission: -0.30 },
  },
  {
    id: "preset-traffic-quiet-20",
    name: "−20% traffic noise",
    description: "Calm traffic; reduce arterial volumes 20%.",
    domain: "noise",
    interventionPattern: { traffic: -0.20, vehicle: -0.20 },
  },
  {
    id: "preset-baseline",
    name: "Baseline (no intervention)",
    description: "Counterfactual baseline — all variables held at observed values.",
    domain: "*",
    interventionPattern: {},
  },
];

/**
 * Apply a preset's interventionPattern to the project's predictor list.
 * Returns the concrete interventions dict (variable -> delta).
 */
export function applyPresetToPredictors(
  preset: ScenarioPreset,
  predictors: string[],
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const pred of predictors) {
    const lower = pred.toLowerCase();
    for (const [pattern, delta] of Object.entries(preset.interventionPattern)) {
      if (lower.includes(pattern)) {
        out[pred] = delta;
        break;
      }
    }
  }
  return out;
}

/** Filter presets relevant to a given domain (plus the universal baseline). */
export function presetsForDomain(domain: string | undefined): ScenarioPreset[] {
  const d = (domain ?? "").toLowerCase();
  return SCENARIO_PRESETS.filter((p) => p.domain === "*" || p.domain === d);
}
