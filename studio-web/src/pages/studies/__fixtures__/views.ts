// Study view fixtures (`GET /api/studies/{stid}/view`, api.md §9) built from the core outputs
// they summarise: placebo.json rows with verdicts, multiverse_summary.json effects and
// priority agreement, reproduce.json checks, benchmark.json runs; plus `GET /api/runs/{rid}/truth`.
import type { BenchmarkView, MultiverseView, PlaceboView, ReproduceView, TruthRow } from "../../../api/studies";

const model = (d1: number, se: number) => [
  { dose_sd: 0.5, mean_delta: d1 / 2, se: se / 2, frac_extrapolated: 0 },
  { dose_sd: 1, mean_delta: d1, se, frac_extrapolated: 0 },
];

export function placeboView(): PlaceboView {
  return {
    rows: [
      { kind: "grf", variable: "canopy", placebo: false, sd: 12, model: model(-0.9, 0.08), causal_theta_sum_per_sd: -0.8, causal_se_per_sd: 0.1, footprint_per_sd: -0.7 },
      {
        kind: "grf",
        variable: "placebo_grf",
        placebo: true,
        sd: 1,
        model: model(-0.02, 0.03),
        causal_theta_sum_per_sd: 0.01,
        causal_se_per_sd: 0.04,
        footprint_per_sd: null,
        verdict: { delta_1sd: -0.02, se_1sd: 0.03, ratio_to_real: 0.022, model_within_2se: true, model_below_10pct_of_real: true, model_pass: true, causal_ci_covers_zero: true },
        reference: "canopy",
      },
      {
        kind: "shift",
        variable: "canopy",
        placebo: true,
        sd: 12,
        model: model(-0.3, 0.05),
        causal_theta_sum_per_sd: -0.25,
        causal_se_per_sd: 0.05,
        footprint_per_sd: null,
        verdict: { delta_1sd: -0.3, se_1sd: 0.05, ratio_to_real: 0.33, model_within_2se: false, model_below_10pct_of_real: false, model_pass: false, causal_ci_covers_zero: false },
        reference: "canopy",
      },
    ],
    layer_correlation: { "shift:canopy": 0.04 },
    n_pass_model: 1,
    n_pass_causal: 1,
    n_placebos: 2,
    children: [
      { kind: "grf", run_id: "child-grf", status: "complete", verdict: "model ✓ · causal ✓" },
      { kind: "shift", run_id: "child-shift", status: "complete", verdict: "model ✗ · causal ✗" },
    ],
  };
}

export function multiverseView(): MultiverseView {
  return {
    variants: [
      { name: "baseline", label: "baseline", status: "complete", r2: 0.81, rmse: 0.34, seconds: 120, run_id: "mv-base" },
      { name: "blocks_1km", label: "1 km CV blocks", status: "complete", r2: 0.78, rmse: 0.36, seconds: 118, run_id: "mv-b1" },
      { name: "half_km", label: "half_km", status: "running", r2: null, rmse: null, seconds: null, run_id: null },
    ],
    effects: {
      "canopy +10": { baseline: -0.9, min: -1.0, max: -0.85, sd_across: 0.05, sign_stability: 1, values: { baseline: -0.9, blocks_1km: -1.0 } },
      "albedo +0.1": { baseline: 0.05, min: -0.02, max: 0.05, sd_across: 0.04, sign_stability: 0.5, values: { baseline: 0.05, blocks_1km: -0.02 } },
    },
    priority: { blocks_1km: { canopy: { kendall_tau: 0.92, top_decile_jaccard: 0.81 } } },
    stability: { sign_stability_min: 0.5, median_kendall_tau: 0.92, median_top_decile_jaccard: 0.81 },
  };
}

export function reproduceView(): ReproduceView {
  return {
    pass: false,
    checks: [
      { check: "cv design", ok: true, hard: true, detail: "same folds" },
      { check: "stack R²", ok: false, hard: true, detail: "0.81 vs 0.77 (tol 0.01)" },
      { check: "code version", ok: false, hard: false, detail: "commit differs" },
    ],
    original: "20261001-142233-fast-ab12",
    reproduction: "20261002-090000-fast-ff00",
  };
}

export function benchmarkView(): BenchmarkView {
  return {
    runs: {
      standard: { models: { rf: { share: 0.6, corr: 0.7 }, gwrf: { share: 0.7, corr: 0.75 } }, stack: { share: 0.72, corr: 0.8 }, footprint: { share: 0.9, corr: 0.85 } },
      spatial_plus: { models: { rf: { share: 0.65, corr: 0.72 }, gwrf: { share: 0.74, corr: 0.77 } }, stack: { share: 0.78, corr: 0.82 }, footprint: null },
    },
  };
}

export function truthRows(): TruthRow[] {
  return [
    { quantity: "canopy_scenario", label: "Canopy +10 pp, city mean ΔT", truth: -1.0, recovered: -0.85, se: 0.05, share: 0.85, unit: "°F", scenario: "canopy +10" },
    { quantity: "footprint_mean", label: "Mean canopy footprint per pp", truth: -0.1, recovered: -0.06, se: null, share: 0.6, unit: "°F/pp", scenario: null },
    // the server reports a share for the noise floor too (recovered ÷ truth); it is a bound, not a recovery
    { quantity: "noise_sd", label: "Noise floor", truth: 0.3, recovered: 0.34, se: null, share: 0.34 / 0.3, unit: "°F", scenario: null },
  ];
}
