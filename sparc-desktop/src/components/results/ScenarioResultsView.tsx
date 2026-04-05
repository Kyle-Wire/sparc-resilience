import { useState, useEffect, useMemo, useCallback } from "react";
import { getScenarioDetail, getScenarioVariables, getScenarioIncrement } from "@/lib/api";
import type { ScenarioDetail } from "@/lib/types";
import type { ScenarioVariableInfo } from "@/lib/api";
import SpatialMap, { type MapMode } from "@/components/map/SpatialMap";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
} from "recharts";

type ScenarioTab = "explorer" | "overview";

export default function ScenarioResultsView() {
  const [tab, setTab] = useState<ScenarioTab>("explorer");
  const [detail, setDetail] = useState<ScenarioDetail | null>(null);
  const [scenVars, setScenVars] = useState<Record<string, ScenarioVariableInfo> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.allSettled([getScenarioDetail(), getScenarioVariables()]).then(
      ([detailRes, varsRes]) => {
        if (detailRes.status === "fulfilled") setDetail(detailRes.value);
        else setError("Scenario results not available. Run Stage 4 first.");
        if (varsRes.status === "fulfilled") setScenVars(varsRes.value.variables);
      },
    );
  }, []);

  const hasExplorer = scenVars && Object.keys(scenVars).length > 0;

  if (error && !detail) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-sparc-gray-600">{error}</p>
      </div>
    );
  }

  if (!detail && !scenVars) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-sparc-gray-500">Loading scenario results…</p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      {/* Tab bar */}
      <div className="flex gap-1 border-b border-sparc-gray-100 px-4 py-2">
        {hasExplorer && (
          <button
            onClick={() => setTab("explorer")}
            className={`rounded px-3 py-1.5 text-xs font-medium transition-colors ${
              tab === "explorer"
                ? "bg-sparc-purple text-white"
                : "border border-sparc-gray-300 hover:bg-sparc-gray-100"
            }`}
          >
            Scenario Explorer
          </button>
        )}
        <button
          onClick={() => setTab("overview")}
          className={`rounded px-3 py-1.5 text-xs font-medium transition-colors ${
            tab === "overview"
              ? "bg-sparc-purple text-white"
              : "border border-sparc-gray-300 hover:bg-sparc-gray-100"
          }`}
        >
          Overview
        </button>
      </div>

      <div className="flex-1 overflow-hidden">
        {tab === "explorer" && scenVars && (
          <ScenarioExplorer variables={scenVars} />
        )}
        {tab === "overview" && detail && <OverviewTab detail={detail} />}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// Scenario Explorer — interactive per-variable slider + map
// ═══════════════════════════════════════════════════════════════════

function ScenarioExplorer({
  variables,
}: {
  variables: Record<string, ScenarioVariableInfo>;
}) {
  const varNames = useMemo(() => Object.keys(variables), [variables]);
  const [selectedVar, setSelectedVar] = useState(varNames[0] ?? "");
  const info = variables[selectedVar];
  const increments = info?.increments ?? [];

  // Start at the middle increment
  const [incIdx, setIncIdx] = useState(Math.floor(increments.length / 2));
  const currentIncrement = increments[incIdx] ?? 0;

  const [geojson, setGeojson] = useState<any>(null);
  const [summaryRow, setSummaryRow] = useState<Record<string, unknown>[]>([]);
  const [loading, setLoading] = useState(false);
  const [mapMode, setMapMode] = useState<MapMode>("scatter");

  // Reset slider when variable changes
  useEffect(() => {
    const newIncrements = variables[selectedVar]?.increments ?? [];
    setIncIdx(Math.floor(newIncrements.length / 2));
  }, [selectedVar, variables]);

  // Fetch spatial data for the chosen variable + increment
  const fetchIncrement = useCallback(
    (variable: string, inc: number) => {
      setLoading(true);
      getScenarioIncrement(variable, inc)
        .then((res) => {
          setGeojson(res.geojson ?? null);
          setSummaryRow(res.summary ?? []);
        })
        .catch(() => {
          setGeojson(null);
          setSummaryRow([]);
        })
        .finally(() => setLoading(false));
    },
    [],
  );

  useEffect(() => {
    if (selectedVar && increments.length > 0) {
      fetchIncrement(selectedVar, increments[incIdx] ?? 0);
    }
  }, [selectedVar, incIdx, increments, fetchIncrement]);

  // Determine the delta column name from the GeoJSON
  const deltaField = useMemo(() => {
    if (!geojson?.features?.length) return undefined;
    const props = Object.keys(geojson.features[0].properties ?? {});
    return props.find((p) => p.startsWith("delta_"));
  }, [geojson]);

  // Compute summary stats from the delta column
  const stats = useMemo(() => {
    if (!geojson?.features?.length || !deltaField) return null;
    const vals = geojson.features
      .map((f: any) => f.properties?.[deltaField])
      .filter((v: unknown): v is number => typeof v === "number" && isFinite(v));
    if (vals.length === 0) return null;
    const mean = vals.reduce((a: number, b: number) => a + b, 0) / vals.length;
    const sorted = [...vals].sort((a: number, b: number) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    return { n: vals.length, mean, median, min: sorted[0], max: sorted[sorted.length - 1] };
  }, [geojson, deltaField]);

  const sign = info?.sign === "minus" ? "−" : "+";

  return (
    <div className="flex h-full flex-col gap-3 p-4 overflow-auto">
      {/* Controls row */}
      <div className="flex flex-wrap items-center gap-4">
        {/* Variable selector */}
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium text-sparc-gray-600">Variable:</label>
          <select
            value={selectedVar}
            onChange={(e) => setSelectedVar(e.target.value)}
            className="rounded border border-sparc-gray-200 px-2 py-1 text-xs"
          >
            {varNames.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </div>

        {/* Map mode */}
        <div className="flex items-center gap-1.5">
          {(["scatter", "choropleth", "heatmap"] as MapMode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMapMode(m)}
              className={`rounded px-2 py-1 text-[10px] font-medium capitalize ${
                mapMode === m
                  ? "bg-sparc-purple text-white"
                  : "border border-sparc-gray-300 hover:bg-sparc-gray-100"
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {/* Slider */}
      {increments.length > 1 && (
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <label className="text-xs font-semibold text-sparc-gray-700">
              {selectedVar}{" "}
              <span className="font-mono text-sparc-purple">
                {sign}{Math.abs(currentIncrement)}
              </span>
            </label>
            <span className="rounded bg-sparc-gray-100 px-2 py-0.5 text-[10px] font-mono tabular-nums">
              {incIdx + 1} / {increments.length}
            </span>
          </div>

          <input
            type="range"
            min={0}
            max={increments.length - 1}
            value={incIdx}
            onChange={(e) => setIncIdx(Number(e.target.value))}
            className="w-full accent-sparc-purple"
          />

          {/* Tick marks */}
          <div className="flex justify-between px-0.5">
            {increments.map((inc, i) => (
              <button
                key={i}
                onClick={() => setIncIdx(i)}
                className={`text-[8px] tabular-nums ${
                  i === incIdx ? "font-bold text-sparc-purple" : "text-sparc-gray-400"
                }`}
              >
                {sign}{Math.abs(inc)}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Summary stats strip */}
      {stats && !loading && (
        <div className="flex flex-wrap gap-3 text-[10px]">
          <span className="rounded bg-sparc-gray-100 px-2 py-1 font-mono">
            N = {stats.n.toLocaleString()}
          </span>
          <span className="rounded bg-blue-50 px-2 py-1 font-mono text-blue-700">
            Mean Δ: {stats.mean.toFixed(4)}
          </span>
          <span className="rounded bg-sparc-gray-100 px-2 py-1 font-mono">
            Median Δ: {stats.median.toFixed(4)}
          </span>
          <span className="rounded bg-sparc-gray-100 px-2 py-1 font-mono">
            Range: [{stats.min.toFixed(4)}, {stats.max.toFixed(4)}]
          </span>
          {summaryRow.length > 0 && summaryRow[0].Avg_Temp_Delta != null && (
            <span className="rounded bg-indigo-50 px-2 py-1 font-mono text-indigo-700">
              Avg Temp Δ: {Number(summaryRow[0].Avg_Temp_Delta).toFixed(4)}
            </span>
          )}
        </div>
      )}

      {/* Map */}
      {loading && (
        <div className="flex items-center justify-center rounded-lg border border-sparc-gray-200"
          style={{ minHeight: 420 }}>
          <p className="text-xs text-sparc-gray-500">Loading scenario map…</p>
        </div>
      )}

      {geojson && !loading && (
        <div
          className="rounded-lg border border-sparc-gray-200 overflow-hidden"
          style={{ minHeight: 420 }}
        >
          <SpatialMap
            geojson={geojson}
            colorField={deltaField}
            mode={mapMode}
            height="420px"
          />
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// Overview Tab — the existing view (all scenarios at once)
// ═══════════════════════════════════════════════════════════════════

function OverviewTab({ detail }: { detail: ScenarioDetail }) {
  const [mapMode, setMapMode] = useState<MapMode>("scatter");
  const [showDelta, setShowDelta] = useState(false);
  const [selectedField, setSelectedField] = useState<string>("");

  // Discover pred/delta columns from the GeoJSON features
  const { predColumns, deltaColumns, baselineCol } = useMemo(() => {
    if (!detail?.geojson?.features?.length) {
      return { predColumns: [] as string[], deltaColumns: [] as string[], baselineCol: "" };
    }
    const props = Object.keys(detail.geojson.features[0].properties);
    const baseline = props.find((p) => p === "pred_baseline") ?? "";
    const preds = props.filter((p) => p.startsWith("pred_") && p !== baseline);
    const deltas = props.filter((p) => p.startsWith("delta_"));
    return { predColumns: preds, deltaColumns: deltas, baselineCol: baseline };
  }, [detail]);

  // Set default field once data loads
  useEffect(() => {
    if (!selectedField) {
      if (predColumns.length) setSelectedField(predColumns[0]);
      else if (baselineCol) setSelectedField(baselineCol);
    }
  }, [predColumns, baselineCol, selectedField]);

  // All fields available for the field dropdown
  const fieldOptions = useMemo(() => {
    if (!detail?.geojson?.features?.length) return [] as string[];
    const props = Object.keys(detail.geojson.features[0].properties);
    return props.filter((p) => p !== "geometry");
  }, [detail]);

  // Summary chart comparing scenarios
  const summaryChartData = useMemo(() => {
    if (!detail?.summary?.length) return [];
    return detail.summary.map((row) => {
      const entry: Record<string, unknown> = { scenario: String(row.scenario ?? row.name ?? "?") };
      for (const [k, v] of Object.entries(row)) {
        if (k !== "scenario" && k !== "name" && typeof v === "number") {
          entry[k] = v;
        }
      }
      return entry;
    });
  }, [detail]);

  const summaryMetricKeys = useMemo(() => {
    if (!summaryChartData.length) return [] as string[];
    return Object.keys(summaryChartData[0]).filter(
      (k) => k !== "scenario" && typeof summaryChartData[0][k] === "number",
    );
  }, [summaryChartData]);

  // Update displayed field when toggling delta mode
  useEffect(() => {
    if (showDelta && selectedField.startsWith("pred_") && selectedField !== baselineCol) {
      const matching = selectedField.replace("pred_", "delta_");
      if (deltaColumns.includes(matching)) setSelectedField(matching);
    } else if (!showDelta && selectedField.startsWith("delta_")) {
      const matching = selectedField.replace("delta_", "pred_");
      if (predColumns.includes(matching)) setSelectedField(matching);
    }
  }, [showDelta, deltaColumns, predColumns, baselineCol, selectedField]);

  const COLORS = ["#602468", "#a44eb4", "#e0a0b0", "#fbdd46", "#3b82f6", "#10b981"];

  return (
    <div className="flex flex-col gap-4 p-4 overflow-auto h-full">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium text-sparc-gray-600">Display Variable:</label>
          <select
            value={selectedField}
            onChange={(e) => setSelectedField(e.target.value)}
            className="rounded border border-sparc-gray-200 px-2 py-1 text-xs"
          >
            {fieldOptions.map((f) => (
              <option key={f} value={f}>{f}</option>
            ))}
          </select>
        </div>

        {deltaColumns.length > 0 && (
          <label className="flex items-center gap-1.5 text-xs cursor-pointer">
            <input
              type="checkbox"
              checked={showDelta}
              onChange={(e) => setShowDelta(e.target.checked)}
              className="rounded"
            />
            Show Delta (change from baseline)
          </label>
        )}

        <div className="flex items-center gap-1.5">
          {(["scatter", "choropleth", "heatmap"] as MapMode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMapMode(m)}
              className={`rounded px-2 py-1 text-[10px] font-medium capitalize ${
                mapMode === m
                  ? "bg-sparc-purple text-white"
                  : "border border-sparc-gray-300 hover:bg-sparc-gray-100"
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {/* Map */}
      {detail.geojson && (
        <div className="rounded-lg border border-sparc-gray-200 overflow-hidden" style={{ minHeight: 400 }}>
          <SpatialMap
            geojson={detail.geojson as any}
            colorField={selectedField}
            mode={mapMode}
            height="400px"
          />
        </div>
      )}

      {/* Summary section */}
      {detail.summary && detail.summary.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-sm font-semibold">Scenario Summary</h3>

          {/* Table */}
          <div className="overflow-auto rounded border border-sparc-gray-200">
            <table className="w-full text-left text-xs">
              <thead className="bg-sparc-gray-50">
                <tr>
                  {Object.keys(detail.summary[0]).map((k) => (
                    <th key={k} className="px-3 py-2 font-medium whitespace-nowrap">
                      {k}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {detail.summary.map((row, i) => (
                  <tr key={i} className={`border-t border-sparc-gray-100 ${i % 2 ? "bg-sparc-gray-50/50" : ""}`}>
                    {Object.values(row).map((v, j) => (
                      <td key={j} className="px-3 py-1.5 tabular-nums whitespace-nowrap">
                        {typeof v === "number" ? v.toFixed(4) : String(v ?? "")}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Cross-scenario comparison chart */}
          {summaryChartData.length > 1 && summaryMetricKeys.length > 0 && (
            <div>
              <h4 className="mb-2 text-xs font-semibold">Cross-Scenario Comparison</h4>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={summaryChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis dataKey="scenario" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={{ fontSize: 11 }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <ReferenceLine y={0} stroke="#999" />
                  {summaryMetricKeys.slice(0, 6).map((key, i) => (
                    <Bar
                      key={key}
                      dataKey={key}
                      fill={COLORS[i % COLORS.length]}
                      radius={[3, 3, 0, 0]}
                    />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
