import { useState, useEffect, useMemo } from "react";
import {
  getCausalResults,
  getPdpCurves,
  getDoseResponseCurves,
  getCausalDiagnostics,
  getCateMapVariables,
  getCateMap,
} from "@/lib/api";
import type {
  CausalResults,
  PdpCurves,
  DoseResponseData,
  CausalDiagnostics,
  GeoJsonData,
} from "@/lib/types";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Area,
  AreaChart,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
  ScatterChart,
  Scatter,
} from "recharts";

type CausalTab = "effects" | "pdp" | "dose_response" | "mediation" | "diagnostics" | "cate_map";

export default function CausalView() {
  const [causal, setCausal] = useState<CausalResults | null>(null);
  const [pdp, setPdp] = useState<PdpCurves | null>(null);
  const [doseResponse, setDoseResponse] = useState<DoseResponseData | null>(null);
  const [diagnostics, setDiagnostics] = useState<CausalDiagnostics | null>(null);
  const [cateVariables, setCateVariables] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<CausalTab>("effects");

  useEffect(() => {
    // Fetch all causal data in parallel
    Promise.allSettled([
      getCausalResults(),
      getPdpCurves(),
      getDoseResponseCurves(),
      getCausalDiagnostics(),
      getCateMapVariables(),
    ]).then(([causalRes, pdpRes, drRes, diagRes, cateVarRes]) => {
      if (causalRes.status === "fulfilled") setCausal(causalRes.value);
      else setError("Causal results not available. Run Stage 3 first.");
      if (pdpRes.status === "fulfilled") setPdp(pdpRes.value);
      if (drRes.status === "fulfilled") setDoseResponse(drRes.value);
      if (diagRes.status === "fulfilled") setDiagnostics(diagRes.value);
      if (cateVarRes.status === "fulfilled") setCateVariables(cateVarRes.value.variables);
    });
  }, []);

  const treatments = useMemo(
    () => Object.keys(causal?.direct_effects ?? {}),
    [causal],
  );

  const hasDoseResponse = doseResponse && Object.keys(doseResponse).length > 0;
  const hasDiagnostics = diagnostics && Object.keys(diagnostics).length > 0;
  const hasMediation =
    causal?.mediation_decomposition &&
    Object.keys(causal.mediation_decomposition).length > 0;

  const TABS: { value: CausalTab; label: string; show: boolean }[] = [
    { value: "effects", label: "Effects Overview", show: true },
    { value: "cate_map", label: "CATE Map", show: cateVariables.length > 0 },
    { value: "pdp", label: "PDP Curves", show: !!pdp },
    { value: "dose_response", label: "Dose-Response", show: !!hasDoseResponse },
    { value: "mediation", label: "Mediation", show: !!hasMediation },
    { value: "diagnostics", label: "Diagnostics", show: !!hasDiagnostics },
  ];

  if (error && !causal) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-sparc-gray-600">{error}</p>
      </div>
    );
  }

  if (!causal) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-sparc-gray-500">Loading causal analysis…</p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      {/* Tab bar */}
      <div className="flex gap-1 border-b border-sparc-gray-100 px-4 py-2">
        {TABS.filter((t) => t.show).map((t) => (
          <button
            key={t.value}
            onClick={() => setTab(t.value)}
            className={`rounded px-3 py-1.5 text-xs font-medium transition-colors ${
              tab === t.value
                ? "bg-sparc-purple text-white"
                : "border border-sparc-gray-300 hover:bg-sparc-gray-100"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-auto p-4">
        {tab === "effects" && <EffectsOverview causal={causal} treatments={treatments} />}
        {tab === "cate_map" && <CateMapView variables={cateVariables} />}
        {tab === "pdp" && pdp && <PdpView pdp={pdp} />}
        {tab === "dose_response" && doseResponse && <DoseResponseView data={doseResponse} />}
        {tab === "mediation" && causal.mediation_decomposition && (
          <MediationView data={causal.mediation_decomposition} />
        )}
        {tab === "diagnostics" && diagnostics && <DiagnosticsView data={diagnostics} causal={causal} />}
      </div>
    </div>
  );
}

// ── Effects Overview Tab ──────────────────────────────────────────

function EffectsOverview({
  causal,
  treatments,
}: {
  causal: CausalResults;
  treatments: string[];
}) {
  if (treatments.length === 0) {
    return <p className="text-sm text-sparc-gray-600">No treatment effects found.</p>;
  }

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold">Treatment Effects</h3>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        {treatments.map((t) => {
          const e = causal.direct_effects![t];
          const sign = (e.structural_coeff ?? 0) < 0 ? "cooling" : (e.structural_coeff ?? 0) > 0 ? "warming" : "neutral";
          const signColor =
            sign === "cooling"
              ? "border-blue-300 bg-blue-50"
              : sign === "warming"
                ? "border-red-300 bg-red-50"
                : "border-sparc-gray-200 bg-white";

          return (
            <div key={t} className={`rounded-lg border-2 p-4 space-y-3 ${signColor}`}>
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-bold">{t}</h4>
                {e.monotone_constraint != null && (
                  <span className="rounded bg-sparc-gray-200 px-1.5 py-0.5 text-[9px] font-mono">
                    constraint: {e.monotone_constraint > 0 ? "↑" : e.monotone_constraint < 0 ? "↓" : "—"}
                  </span>
                )}
              </div>

              {/* Main coefficient */}
              <div>
                <p className="text-[10px] text-sparc-gray-500">Structural Coefficient</p>
                <p className="text-xl font-bold tabular-nums">
                  {e.structural_coeff?.toFixed(4) ?? "—"}
                </p>
                {e.bootstrap_ci_lower != null && e.bootstrap_ci_upper != null && (
                  <p className="text-[9px] text-sparc-gray-400">
                    95% CI: [{e.bootstrap_ci_lower.toFixed(4)}, {e.bootstrap_ci_upper.toFixed(4)}]
                  </p>
                )}
              </div>

              {/* Key metrics */}
              <div className="grid grid-cols-2 gap-2 text-[10px]">
                {e.elasticity != null && (
                  <div>
                    <span className="text-sparc-gray-500">Elasticity</span>
                    <p className="font-mono font-medium">{e.elasticity.toFixed(4)}</p>
                  </div>
                )}
                {e.relative_importance != null && (
                  <div>
                    <span className="text-sparc-gray-500">Rel. Importance</span>
                    <p className="font-mono font-medium">{(e.relative_importance * 100).toFixed(1)}%</p>
                  </div>
                )}
                {e.e_value != null && (
                  <div>
                    <span className="text-sparc-gray-500">E-value</span>
                    <p className="font-mono font-medium">{e.e_value.toFixed(2)}</p>
                  </div>
                )}
                {e.max_discrepancy_pct != null && (
                  <div>
                    <span className="text-sparc-gray-500">Max Discrepancy</span>
                    <p className="font-mono font-medium">{e.max_discrepancy_pct.toFixed(1)}%</p>
                  </div>
                )}
              </div>

              {/* Refutation tests */}
              <div className="flex flex-wrap gap-1.5">
                {[
                  { key: "placebo_pass", label: "Placebo" },
                  { key: "rcc_pass", label: "RCC" },
                  { key: "data_subset_pass", label: "Subset" },
                  { key: "ucc_pass", label: "UCC" },
                ].map(({ key, label }) => {
                  const val = (e as any)[key];
                  if (val == null) return null;
                  return (
                    <span
                      key={key}
                      className={`rounded px-1.5 py-0.5 text-[9px] font-medium ${
                        val
                          ? "bg-green-100 text-green-700"
                          : "bg-red-100 text-red-700"
                      }`}
                    >
                      {val ? "✓" : "✗"} {label}
                    </span>
                  );
                })}
                {e.estimator_agreement != null && (
                  <span
                    className={`rounded px-1.5 py-0.5 text-[9px] font-medium ${
                      e.estimator_agreement
                        ? "bg-green-100 text-green-700"
                        : "bg-yellow-100 text-yellow-700"
                    }`}
                  >
                    {e.estimator_agreement ? "✓" : "~"} Agreement
                  </span>
                )}
              </div>

              {/* ATE comparison */}
              <div className="text-[9px] text-sparc-gray-400 space-y-0.5">
                {e.ate_backdoor != null && <p>Backdoor ATE: {e.ate_backdoor.toFixed(4)}</p>}
                {e.ate_ipw != null && <p>IPW ATE: {e.ate_ipw.toFixed(4)}</p>}
                {e.ate_doubly_robust != null && <p>DR ATE: {e.ate_doubly_robust.toFixed(4)}</p>}
                {e.cate_mean != null && (
                  <p>
                    CATE: {e.cate_mean.toFixed(4)}
                    {e.cate_std != null && ` ± ${e.cate_std.toFixed(4)}`}
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* All structural coefficients */}
      {causal.all_structural_coefficients && (
        <div className="mt-6">
          <h3 className="mb-2 text-sm font-semibold">All DAG Edge Coefficients</h3>
          <div className="overflow-auto rounded border border-sparc-gray-200">
            <table className="w-full text-left text-xs">
              <thead className="bg-sparc-gray-50">
                <tr>
                  <th className="px-3 py-2 font-medium">Edge</th>
                  <th className="px-3 py-2 font-medium">Coefficient</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(causal.all_structural_coefficients).map(([edge, coeff]) => (
                  <tr key={edge} className="border-t border-sparc-gray-100">
                    <td className="px-3 py-1.5 font-mono">{edge}</td>
                    <td className="px-3 py-1.5 tabular-nums font-mono">
                      {typeof coeff === "number" ? coeff.toFixed(4) : String(coeff)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// ── PDP Curves Tab ────────────────────────────────────────────────

function PdpView({ pdp }: { pdp: PdpCurves }) {
  const variables = Object.keys(pdp);
  const [selectedVar, setSelectedVar] = useState(variables[0] ?? "");

  const chartData = useMemo(() => {
    const entry = pdp[selectedVar];
    if (!entry) return [];
    const grid = entry.pdp?.grid_values ?? entry.grid_values ?? [];
    const vals = entry.pdp?.pdp_values ?? entry.pdp_values ?? [];
    const std = entry.pdp?.pdp_std;
    return grid.map((x, i) => ({
      x,
      pdp: vals[i],
      upper: std ? vals[i] + (std[i] ?? 0) : undefined,
      lower: std ? vals[i] - (std[i] ?? 0) : undefined,
    }));
  }, [pdp, selectedVar]);

  const curveFit = pdp[selectedVar]?.curve_fit;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <h3 className="text-sm font-semibold">Partial Dependence Plots</h3>
        <select
          value={selectedVar}
          onChange={(e) => setSelectedVar(e.target.value)}
          className="rounded border border-sparc-gray-200 px-2 py-1 text-xs"
        >
          {variables.map((v) => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>
      </div>

      {/* Curve fit info */}
      {curveFit && (
        <div className="flex gap-3">
          <span className="rounded bg-sparc-purple/10 px-2 py-1 text-[10px] font-mono text-sparc-purple">
            Fit: {curveFit.curve_type}
          </span>
          {curveFit.r2 != null && (
            <span className="rounded bg-sparc-gray-100 px-2 py-1 text-[10px] font-mono">
              R² = {curveFit.r2.toFixed(3)}
            </span>
          )}
          {curveFit.saturation_point != null && (
            <span className="rounded bg-sparc-gray-100 px-2 py-1 text-[10px] font-mono">
              Saturation: {curveFit.saturation_point.toFixed(2)}
            </span>
          )}
        </div>
      )}

      <ResponsiveContainer width="100%" height={400}>
        <AreaChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis
            dataKey="x"
            tick={{ fontSize: 10 }}
            label={{ value: selectedVar, position: "insideBottom", offset: -4, fontSize: 11 }}
          />
          <YAxis
            tick={{ fontSize: 10 }}
            label={{ value: "Partial Effect", angle: -90, position: "insideLeft", fontSize: 11 }}
          />
          <Tooltip
            contentStyle={{ fontSize: 11 }}
            formatter={(v) => typeof v === "number" ? v.toFixed(4) : String(v)}
          />
          {chartData[0]?.upper !== undefined && (
            <Area
              type="monotone"
              dataKey="upper"
              stroke="none"
              fill="#a44eb4"
              fillOpacity={0.15}
            />
          )}
          {chartData[0]?.lower !== undefined && (
            <Area
              type="monotone"
              dataKey="lower"
              stroke="none"
              fill="#a44eb4"
              fillOpacity={0.15}
            />
          )}
          <Line
            type="monotone"
            dataKey="pdp"
            stroke="#602468"
            strokeWidth={2.5}
            dot={false}
          />
          {curveFit?.saturation_point != null && (
            <ReferenceLine
              x={curveFit.saturation_point}
              stroke="#e74c3c"
              strokeDasharray="6 3"
              label={{ value: "Saturation", fontSize: 10, fill: "#e74c3c" }}
            />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Dose-Response Tab ─────────────────────────────────────────────

function DoseResponseView({ data }: { data: DoseResponseData }) {
  const treatments = Object.keys(data);
  const [selectedT, setSelectedT] = useState(treatments[0] ?? "");

  const entry = data[selectedT];
  const chartData = useMemo(() => {
    if (!entry) return [];
    return entry.dose_levels.map((dose, i) => ({
      dose,
      effect: entry.marginal_effects[i],
      mean: entry.dose_means?.[i],
    }));
  }, [entry]);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <h3 className="text-sm font-semibold">Dose-Response Curves</h3>
        <select
          value={selectedT}
          onChange={(e) => setSelectedT(e.target.value)}
          className="rounded border border-sparc-gray-200 px-2 py-1 text-xs"
        >
          {treatments.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
        {entry?.is_nonlinear != null && (
          <span
            className={`rounded px-2 py-1 text-[10px] font-medium ${
              entry.is_nonlinear
                ? "bg-yellow-100 text-yellow-700"
                : "bg-green-100 text-green-700"
            }`}
          >
            {entry.is_nonlinear ? "Non-linear" : "Linear"}
            {entry.nonlinearity_ratio != null && ` (ratio: ${entry.nonlinearity_ratio.toFixed(2)})`}
          </span>
        )}
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis
            dataKey="dose"
            tick={{ fontSize: 10 }}
            label={{ value: `${selectedT} Level`, position: "insideBottom", offset: -4, fontSize: 11 }}
          />
          <YAxis
            tick={{ fontSize: 10 }}
            label={{ value: "Marginal Effect", angle: -90, position: "insideLeft", fontSize: 11 }}
          />
          <Tooltip contentStyle={{ fontSize: 11 }} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line
            type="monotone"
            dataKey="effect"
            stroke="#602468"
            strokeWidth={2.5}
            dot={{ r: 3 }}
            name="Marginal Effect"
          />
          {chartData[0]?.mean !== undefined && (
            <Line
              type="monotone"
              dataKey="mean"
              stroke="#a44eb4"
              strokeDasharray="5 5"
              strokeWidth={1.5}
              dot={false}
              name="Mean Outcome"
            />
          )}
          <ReferenceLine y={0} stroke="#999" strokeDasharray="4 4" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Mediation Tab ─────────────────────────────────────────────────

function MediationView({
  data,
}: {
  data: Record<string, { direct_effect: number; indirect_total: number; total_effect: number; mediation_proportion?: number; indirect_paths?: { mediator: string; indirect_effect: number; significant?: boolean }[] }>;
}) {
  const treatments = Object.keys(data);

  return (
    <div className="space-y-6">
      <h3 className="text-sm font-semibold">Mediation Decomposition</h3>
      {treatments.map((t) => {
        const m = data[t];
        const barData = [
          { name: "Direct", value: m.direct_effect },
          ...(m.indirect_paths ?? []).map((p) => ({
            name: `via ${p.mediator}`,
            value: p.indirect_effect,
          })),
        ];

        return (
          <div key={t} className="space-y-2">
            <div className="flex items-center gap-3">
              <h4 className="text-sm font-bold">{t}</h4>
              {m.mediation_proportion != null && (
                <span className="rounded bg-sparc-gray-100 px-2 py-0.5 text-[10px] font-mono">
                  {(m.mediation_proportion * 100).toFixed(1)}% mediated
                </span>
              )}
              <span className="text-[10px] text-sparc-gray-400">
                Total: {m.total_effect?.toFixed(4)}
              </span>
            </div>

            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={barData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 10 }} width={120} />
                <Tooltip contentStyle={{ fontSize: 11 }} />
                <ReferenceLine x={0} stroke="#999" />
                <Bar
                  dataKey="value"
                  fill="#602468"
                  radius={[0, 3, 3, 0]}
                />
              </BarChart>
            </ResponsiveContainer>

            {/* Indirect path details */}
            {m.indirect_paths && m.indirect_paths.length > 0 && (
              <div className="flex flex-wrap gap-2 text-[9px]">
                {m.indirect_paths.map((p, i) => (
                  <span
                    key={i}
                    className={`rounded px-2 py-1 font-mono ${
                      p.significant
                        ? "bg-sparc-purple/10 text-sparc-purple"
                        : "bg-sparc-gray-100 text-sparc-gray-500"
                    }`}
                  >
                    {t} → {p.mediator}: {p.indirect_effect.toFixed(4)}
                    {p.significant ? " *" : ""}
                  </span>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Diagnostics Tab ───────────────────────────────────────────────

function DiagnosticsView({
  data,
  causal,
}: {
  data: CausalDiagnostics;
  causal: CausalResults;
}) {
  const treatments = Object.keys(data);
  const [selectedT, setSelectedT] = useState(treatments[0] ?? "");
  const entry = data[selectedT];

  // Cumulative effect curve data
  const cumData = useMemo(() => {
    if (!entry?.cumulative_effect_curve) return [];
    const c = entry.cumulative_effect_curve;
    return c.fractions.map((f, i) => ({
      fraction: f,
      cum_effect: c.cum_effects[i],
      random: c.random_baseline ?? 0,
    }));
  }, [entry]);

  // Calibration scatter data
  const calData = useMemo(() => {
    if (!entry?.calibration) return [];
    const c = entry.calibration;
    return c.bin_cate_mean.map((pred, i) => ({
      predicted: pred,
      observed: c.bin_ate[i],
    }));
  }, [entry]);

  // Assumption diagnostics from causal results
  const assumptions = causal.assumption_diagnostics?.[selectedT];
  const propensity = causal.propensity_diagnostics?.[selectedT];

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <h3 className="text-sm font-semibold">Causal Diagnostics</h3>
        <select
          value={selectedT}
          onChange={(e) => setSelectedT(e.target.value)}
          className="rounded border border-sparc-gray-200 px-2 py-1 text-xs"
        >
          {treatments.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
      </div>

      {/* Assumption check cards */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {assumptions && (
          <>
            {(assumptions as any).positivity && (
              <div className="rounded border border-sparc-gray-200 p-3">
                <p className="text-[10px] font-medium text-sparc-gray-500">Positivity</p>
                <p
                  className={`text-sm font-bold ${
                    (assumptions as any).positivity.positivity_ok ? "text-green-600" : "text-red-600"
                  }`}
                >
                  {(assumptions as any).positivity.positivity_ok ? "OK" : "Violation"}
                </p>
                <p className="text-[9px] text-sparc-gray-400">
                  Rate: {((assumptions as any).positivity.violation_rate * 100)?.toFixed(1)}%
                </p>
              </div>
            )}
            {(assumptions as any).sutva && (
              <div className="rounded border border-sparc-gray-200 p-3">
                <p className="text-[10px] font-medium text-sparc-gray-500">SUTVA</p>
                <p
                  className={`text-sm font-bold ${
                    !(assumptions as any).sutva.sutva_concern ? "text-green-600" : "text-yellow-600"
                  }`}
                >
                  {(assumptions as any).sutva.sutva_concern ? "Concern" : "OK"}
                </p>
                <p className="text-[9px] text-sparc-gray-400">
                  Moran&apos;s I: {(assumptions as any).sutva.morans_i?.toFixed(3)}
                </p>
              </div>
            )}
          </>
        )}
        {propensity && (
          <>
            <div className="rounded border border-sparc-gray-200 p-3">
              <p className="text-[10px] font-medium text-sparc-gray-500">Propensity Score</p>
              <p className="text-sm font-bold tabular-nums">
                {(propensity as any).ps_mean?.toFixed(3) ?? "—"}
              </p>
              <p className="text-[9px] text-sparc-gray-400">
                σ = {(propensity as any).ps_std?.toFixed(3)}
              </p>
            </div>
            <div className="rounded border border-sparc-gray-200 p-3">
              <p className="text-[10px] font-medium text-sparc-gray-500">Extreme PS</p>
              <p className="text-sm font-bold tabular-nums">
                {((propensity as any).extreme_fraction * 100)?.toFixed(1) ?? "—"}%
              </p>
              <p className="text-[9px] text-sparc-gray-400">
                Trimmed: {(propensity as any).n_trimmed ?? 0}
              </p>
            </div>
          </>
        )}
        {entry?.rate_score != null && (
          <div className="rounded border border-sparc-gray-200 p-3">
            <p className="text-[10px] font-medium text-sparc-gray-500">RATE Score</p>
            <p className="text-sm font-bold tabular-nums">{entry.rate_score.toFixed(4)}</p>
          </div>
        )}
      </div>

      {/* Cumulative effect curve */}
      {cumData.length > 0 && (
        <div>
          <h4 className="mb-2 text-xs font-semibold">Cumulative Effect Curve</h4>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={cumData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis
                dataKey="fraction"
                tick={{ fontSize: 10 }}
                label={{ value: "Fraction of Population", position: "insideBottom", offset: -4, fontSize: 10 }}
              />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ fontSize: 11 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line
                type="monotone"
                dataKey="cum_effect"
                stroke="#602468"
                strokeWidth={2}
                dot={false}
                name="Cum. Effect"
              />
              <Line
                type="monotone"
                dataKey="random"
                stroke="#999"
                strokeDasharray="5 5"
                strokeWidth={1}
                dot={false}
                name="Random Baseline"
              />
            </LineChart>
          </ResponsiveContainer>
          {entry?.cumulative_effect_curve?.area_ratio != null && (
            <p className="mt-1 text-[10px] text-sparc-gray-400">
              Area ratio: {entry.cumulative_effect_curve.area_ratio.toFixed(3)} ({">"} 1 = CATE ordering is informative)
            </p>
          )}
        </div>
      )}

      {/* CATE Calibration scatter */}
      {calData.length > 0 && (
        <div>
          <h4 className="mb-2 text-xs font-semibold">
            CATE Calibration
            {entry?.calibration?.is_monotone != null && (
              <span
                className={`ml-2 rounded px-1.5 py-0.5 text-[9px] font-normal ${
                  entry.calibration.is_monotone
                    ? "bg-green-100 text-green-700"
                    : "bg-yellow-100 text-yellow-700"
                }`}
              >
                {entry.calibration.is_monotone ? "Monotone ✓" : "Non-monotone"}
              </span>
            )}
          </h4>
          <ResponsiveContainer width="100%" height={280}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis
                dataKey="predicted"
                name="Predicted CATE"
                tick={{ fontSize: 10 }}
                label={{ value: "Predicted CATE (bin mean)", position: "insideBottom", offset: -4, fontSize: 10 }}
              />
              <YAxis
                dataKey="observed"
                name="Observed ATE"
                tick={{ fontSize: 10 }}
                label={{ value: "Within-bin ATE", angle: -90, position: "insideLeft", fontSize: 10 }}
              />
              <Tooltip contentStyle={{ fontSize: 11 }} />
              <Scatter data={calData} fill="#602468" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

// ── CATE Map Tab ──────────────────────────────────────────────────

function CateMapView({ variables }: { variables: string[] }) {
  const [selectedVar, setSelectedVar] = useState(variables[0] ?? "");
  const [geojson, setGeojson] = useState<GeoJsonData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedVar) return;
    setLoading(true);
    setError(null);
    getCateMap(selectedVar)
      .then((data) => setGeojson(data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [selectedVar]);

  // Detect the CATE property name in the GeoJSON
  const cateField = useMemo(() => {
    if (!geojson || geojson.features.length === 0) return undefined;
    const props = geojson.features[0].properties;
    return Object.keys(props).find((k) => k.startsWith("cate_"));
  }, [geojson]);

  // Compute summary stats
  const stats = useMemo(() => {
    if (!geojson || !cateField) return null;
    const vals = geojson.features
      .map((f) => f.properties[cateField])
      .filter((v): v is number => typeof v === "number" && isFinite(v));
    if (vals.length === 0) return null;
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const sorted = [...vals].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const std = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length);
    return { n: vals.length, mean, median, std, min: sorted[0], max: sorted[sorted.length - 1] };
  }, [geojson, cateField]);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <h3 className="text-sm font-semibold">Spatial CATE Map</h3>
        <select
          value={selectedVar}
          onChange={(e) => setSelectedVar(e.target.value)}
          className="rounded border border-sparc-gray-200 px-2 py-1 text-xs"
        >
          {variables.map((v) => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>
      </div>

      {/* Summary stats */}
      {stats && (
        <div className="flex flex-wrap gap-3 text-[10px]">
          <span className="rounded bg-sparc-gray-100 px-2 py-1 font-mono">
            N = {stats.n}
          </span>
          <span className="rounded bg-sparc-gray-100 px-2 py-1 font-mono">
            Mean: {stats.mean.toFixed(4)}
          </span>
          <span className="rounded bg-sparc-gray-100 px-2 py-1 font-mono">
            Median: {stats.median.toFixed(4)}
          </span>
          <span className="rounded bg-sparc-gray-100 px-2 py-1 font-mono">
            SD: {stats.std.toFixed(4)}
          </span>
          <span className="rounded bg-sparc-gray-100 px-2 py-1 font-mono">
            Range: [{stats.min.toFixed(4)}, {stats.max.toFixed(4)}]
          </span>
        </div>
      )}

      {loading && (
        <p className="text-xs text-sparc-gray-500">Loading CATE map…</p>
      )}
      {error && (
        <p className="text-xs text-red-600">{error}</p>
      )}

      {geojson && !loading && (
        <CateMapCanvas geojson={geojson} colorField={cateField} />
      )}
    </div>
  );
}

/**
 * Inline deck.gl map for the CATE tab — follows the same pattern as SpatialMap
 * but is self-contained to avoid a circular import.
 */
function CateMapCanvas({
  geojson,
  colorField,
}: {
  geojson: GeoJsonData;
  colorField?: string;
}) {
  // Lazy-import deck.gl + maplibre only when this tab is rendered
  const [DeckGL, setDeckGL] = useState<any>(null);
  const [MapGL, setMapGL] = useState<any>(null);
  const [layers, setLayers] = useState<any>(null);

  useEffect(() => {
    // Dynamic imports keep the bundle chunk lazy
    Promise.all([
      import("@deck.gl/react"),
      import("@deck.gl/layers"),
      import("react-map-gl/maplibre"),
    ]).then(([deckMod, layersMod, mapMod]) => {
      setDeckGL(() => deckMod.DeckGL);
      setMapGL(() => mapMod.default);
      setLayers({ GeoJsonLayer: layersMod.GeoJsonLayer, ScatterplotLayer: layersMod.ScatterplotLayer });
    });
  }, []);

  // Compute bounding box
  const bbox = useMemo(() => {
    let minLng = 180, maxLng = -180, minLat = 90, maxLat = -90;
    for (const f of geojson.features) {
      const c = f.geometry.type === "Point"
        ? [f.geometry.coordinates as number[]]
        : (f.geometry.coordinates as number[][]);
      for (const pt of c) {
        const [lng, lat] = pt as number[];
        if (lng < minLng) minLng = lng;
        if (lng > maxLng) maxLng = lng;
        if (lat < minLat) minLat = lat;
        if (lat > maxLat) maxLat = lat;
      }
    }
    const longitude = (minLng + maxLng) / 2;
    const latitude = (minLat + maxLat) / 2;
    const span = Math.max(maxLng - minLng, maxLat - minLat);
    const zoom = span > 0 ? Math.max(1, Math.min(16, Math.log2(360 / span) - 0.5)) : 10;
    return { longitude, latitude, zoom };
  }, [geojson]);

  const [viewState, setViewState] = useState({ ...bbox, pitch: 0, bearing: 0 });

  useEffect(() => {
    setViewState((prev) => ({ ...prev, ...bbox }));
  }, [bbox]);

  // Color ramp (diverging: blue → white → red for + / − CATE)
  const domain = useMemo<[number, number]>(() => {
    if (!colorField) return [0, 1];
    let min = Infinity, max = -Infinity;
    for (const f of geojson.features) {
      const v = f.properties[colorField];
      if (typeof v === "number" && isFinite(v)) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    return min < max ? [min, max] : [0, 1];
  }, [geojson, colorField]);

  const divergingColor = (t: number): [number, number, number, number] => {
    // -1 → blue, 0 → white, +1 → red
    const clamped = Math.max(-1, Math.min(1, t));
    if (clamped < 0) {
      const a = 1 + clamped; // 0→1
      return [Math.round(80 * (1 - a)), Math.round(80 + 175 * a), Math.round(200 + 55 * (1 - a)), 200];
    }
    return [Math.round(200 + 55 * clamped), Math.round(80 + 175 * (1 - clamped)), Math.round(80 * (1 - clamped)), 200];
  };

  if (!DeckGL || !MapGL || !layers) {
    return <p className="text-xs text-sparc-gray-500">Loading map renderer…</p>;
  }

  // Build a symmetric domain around zero for diverging palette
  const absMax = Math.max(Math.abs(domain[0]), Math.abs(domain[1]));
  const symDomain: [number, number] = absMax > 0 ? [-absMax, absMax] : [-1, 1];

  const deckLayers = [
    new layers.ScatterplotLayer({
      id: "cate-scatter",
      data: geojson.features,
      getPosition: (d: any) => {
        const c = d.geometry.coordinates;
        return d.geometry.type === "Point" ? c : c[0];
      },
      getRadius: 100,
      radiusMinPixels: 4,
      radiusMaxPixels: 20,
      getFillColor: (d: any) => {
        if (!colorField) return [96, 36, 104, 200];
        const v = d.properties[colorField];
        if (typeof v !== "number") return [200, 200, 200, 100];
        const t = v / (symDomain[1] || 1);
        return divergingColor(t);
      },
      pickable: true,
    }),
  ];

  return (
    <div className="relative overflow-hidden rounded-lg border border-sparc-gray-200" style={{ height: "480px" }}>
      <DeckGL
        viewState={viewState}
        onViewStateChange={({ viewState: vs }: any) => setViewState(vs)}
        layers={deckLayers}
        controller
        getTooltip={({ object }: any) => {
          if (!object) return null;
          const props = object.properties ?? object;
          const lines = Object.entries(props)
            .filter(([k]) => !k.startsWith("_") && k !== "geometry")
            .slice(0, 6)
            .map(([k, v]) => `${k}: ${typeof v === "number" ? v.toFixed(4) : v}`);
          return { text: lines.join("\n"), style: { fontSize: "11px", fontFamily: "monospace" } };
        }}
      >
        <MapGL
          mapStyle="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
          attributionControl={false}
        />
      </DeckGL>

      {/* Diverging legend */}
      {colorField && (
        <div className="absolute bottom-4 right-4 rounded-md bg-white/90 p-2 shadow-sm backdrop-blur-sm">
          <p className="mb-1 text-[10px] font-medium text-sparc-gray-600">{colorField}</p>
          <div className="flex items-center gap-1.5">
            <span className="text-[9px] text-sparc-gray-500">{symDomain[0].toFixed(3)}</span>
            <div
              className="h-2.5 w-24 rounded-sm"
              style={{
                background: "linear-gradient(to right, rgb(0,80,200), rgb(255,255,255), rgb(255,80,0))",
              }}
            />
            <span className="text-[9px] text-sparc-gray-500">{symDomain[1].toFixed(3)}</span>
          </div>
        </div>
      )}
    </div>
  );
}
