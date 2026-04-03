import { useState, useEffect, useMemo } from "react";
import { getResults, getPredictions, getStagePlots, stagePlotUrl } from "@/lib/api";
import SpatialMap, { type MapMode } from "@/components/map/SpatialMap";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
} from "recharts";

const STAGES = [
  { value: 0, label: "Stage 0 — Correlogram" },
  { value: 1, label: "Stage 1 — GWEN" },
  { value: 2, label: "Stage 2 — Spatial CV" },
  { value: 3, label: "Stage 3 — Causal" },
  { value: 4, label: "Stage 4 — Scenarios" },
];

const MAP_MODES: { value: MapMode; label: string }[] = [
  { value: "scatter", label: "Scatter" },
  { value: "heatmap", label: "Heatmap" },
  { value: "choropleth", label: "Choropleth" },
];

export default function ResultsView() {
  const [stage, setStage] = useState(0);
  const [data, setData] = useState<Record<string, unknown>[] | null>(null);
  const [geojson, setGeojson] = useState<any>(null);
  const [plots, setPlots] = useState<{ name: string; filename: string; path: string }[]>([]);
  const [plotIdx, setPlotIdx] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"plots" | "map" | "table" | "chart">("plots");
  const [mapMode, setMapMode] = useState<MapMode>("scatter");
  const [colorField, setColorField] = useState<string>("");

  useEffect(() => {
    setData(null);
    setGeojson(null);
    setPlots([]);
    setPlotIdx(0);
    setError(null);

    // Fetch tabular results, geospatial results, and plot list in parallel
    Promise.allSettled([
      getResults(stage),
      getPredictions(stage),
      getStagePlots(stage),
    ]).then(([tabRes, geoRes, plotRes]) => {
      if (tabRes.status === "fulfilled") {
        const r = tabRes.value as any;
        if (r?.rows) setData(r.rows);
      }
      if (geoRes.status === "fulfilled") {
        const g = geoRes.value as any;
        if (g?.type === "FeatureCollection") setGeojson(g);
      }
      if (plotRes.status === "fulfilled") {
        const p = plotRes.value as any;
        if (p?.plots) setPlots(p.plots);
      }
      if (tabRes.status === "rejected" && geoRes.status === "rejected" && plotRes.status === "rejected") {
        setError("No results available for this stage yet. Run the stage first.");
      }
    });
  }, [stage]);

  // Numeric columns for color field and chart axes
  const numericCols = useMemo(() => {
    if (!data || data.length === 0) return [];
    return Object.keys(data[0]).filter((k) => typeof data[0][k] === "number");
  }, [data]);

  // Auto-select first numeric col as color field
  useEffect(() => {
    if (numericCols.length > 0 && !colorField) {
      // Prefer prediction/target columns
      const preferred = numericCols.find(
        (c) => c.includes("pred") || c.includes("fitted") || c.includes("target"),
      );
      setColorField(preferred ?? numericCols[0]);
    }
  }, [numericCols, colorField]);

  // Summary metrics
  const metrics = useMemo(() => {
    if (!data || data.length === 0) return null;
    const nums: Record<string, number[]> = {};
    for (const row of data) {
      for (const [k, v] of Object.entries(row)) {
        if (typeof v === "number") {
          if (!nums[k]) nums[k] = [];
          nums[k].push(v);
        }
      }
    }
    return Object.fromEntries(
      Object.entries(nums).map(([k, arr]) => {
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        const std = Math.sqrt(arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length);
        return [k, { mean, std, min: Math.min(...arr), max: Math.max(...arr), n: arr.length }];
      }),
    );
  }, [data]);

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-sparc-gray-200 px-4 py-3">
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-bold">Results</h1>
          <div className="flex gap-1">
            {STAGES.map((s) => (
              <button
                key={s.value}
                onClick={() => { setStage(s.value); setColorField(""); }}
                className={`rounded px-3 py-1.5 text-xs font-medium transition-colors ${
                  stage === s.value ? "bg-sparc-purple text-white" : "border border-sparc-gray-300 hover:bg-sparc-gray-100"
                }`}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>
        <div className="flex gap-1">
          {(["plots", "map", "table", "chart"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`rounded px-3 py-1.5 text-xs font-medium capitalize ${
                tab === t ? "bg-sparc-gray-900 text-white" : "border border-sparc-gray-300 hover:bg-sparc-gray-100"
              }`}
            >
              {t === "plots" ? `Plots (${plots.length})` : t}
            </button>
          ))}
        </div>
      </div>

      {error && <p className="border-b border-red-200 bg-red-50 px-4 py-2 text-xs text-red-600">{error}</p>}

      {/* Metric cards */}
      {metrics && (
        <div className="flex gap-3 overflow-x-auto border-b border-sparc-gray-100 px-4 py-3">
          {Object.entries(metrics).slice(0, 6).map(([k, v]) => (
            <div key={k} className="min-w-[120px] rounded-md border border-sparc-gray-200 bg-white px-3 py-2">
              <p className="truncate text-[10px] font-medium text-sparc-gray-500">{k}</p>
              <p className="text-sm font-bold tabular-nums">{v.mean.toFixed(4)}</p>
              <p className="text-[9px] text-sparc-gray-400">σ {v.std.toFixed(3)} · n={v.n}</p>
            </div>
          ))}
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {tab === "plots" && (
          <div className="flex h-full flex-col">
            {plots.length === 0 ? (
              <div className="flex flex-1 items-center justify-center">
                <p className="text-sm text-sparc-gray-600">No plots available for this stage yet.</p>
              </div>
            ) : (
              <>
                {/* Navigation bar */}
                <div className="flex items-center justify-between border-b border-sparc-gray-100 px-4 py-2">
                  <button
                    onClick={() => setPlotIdx((i) => Math.max(0, i - 1))}
                    disabled={plotIdx === 0}
                    className="rounded border border-sparc-gray-300 px-3 py-1 text-xs font-medium disabled:opacity-30 hover:bg-sparc-gray-100"
                  >
                    ← Prev
                  </button>
                  <div className="flex items-center gap-2">
                    <select
                      value={plotIdx}
                      onChange={(e) => setPlotIdx(Number(e.target.value))}
                      className="rounded border border-sparc-gray-200 px-2 py-1 text-xs"
                    >
                      {plots.map((p, i) => (
                        <option key={p.path} value={i}>
                          {p.name}
                        </option>
                      ))}
                    </select>
                    <span className="text-[10px] text-sparc-gray-500">
                      {plotIdx + 1} / {plots.length}
                    </span>
                  </div>
                  <button
                    onClick={() => setPlotIdx((i) => Math.min(plots.length - 1, i + 1))}
                    disabled={plotIdx >= plots.length - 1}
                    className="rounded border border-sparc-gray-300 px-3 py-1 text-xs font-medium disabled:opacity-30 hover:bg-sparc-gray-100"
                  >
                    Next →
                  </button>
                </div>
                {/* Plot image */}
                <div className="flex flex-1 items-center justify-center overflow-auto bg-white p-4">
                  <img
                    src={stagePlotUrl(stage, plots[plotIdx].path)}
                    alt={plots[plotIdx].name}
                    className="max-h-full max-w-full object-contain"
                  />
                </div>
              </>
            )}
          </div>
        )}

        {tab === "map" && (
          <div className="flex h-full flex-col">
            {/* Map controls */}
            <div className="flex items-center gap-3 border-b border-sparc-gray-100 px-4 py-2">
              <label className="text-xs text-sparc-gray-600">Mode:</label>
              <div className="flex gap-1">
                {MAP_MODES.map((m) => (
                  <button
                    key={m.value}
                    onClick={() => setMapMode(m.value)}
                    className={`rounded px-2 py-1 text-[10px] ${
                      mapMode === m.value ? "bg-sparc-purple text-white" : "border border-sparc-gray-200 hover:bg-sparc-gray-50"
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
              {numericCols.length > 0 && (
                <>
                  <label className="ml-2 text-xs text-sparc-gray-600">Color:</label>
                  <select
                    value={colorField}
                    onChange={(e) => setColorField(e.target.value)}
                    className="rounded border border-sparc-gray-200 px-2 py-1 text-xs"
                  >
                    {numericCols.map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </>
              )}
            </div>
            <div className="flex-1">
              <SpatialMap geojson={geojson} colorField={colorField} mode={mapMode} />
            </div>
          </div>
        )}

        {tab === "table" && (
          <div className="h-full overflow-auto p-4">
            {data && data.length > 0 ? (
              <div className="overflow-auto rounded border border-sparc-gray-200">
                <table className="w-full text-left text-sm">
                  <thead>
                    <tr className="border-b border-sparc-gray-200 bg-sparc-gray-100">
                      {Object.keys(data[0]).map((col) => (
                        <th key={col} className="px-3 py-2 text-xs font-medium">{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.slice(0, 200).map((row, i) => (
                      <tr key={i} className="border-b border-sparc-gray-100">
                        {Object.values(row).map((val, j) => (
                          <td key={j} className="px-3 py-1.5 text-xs tabular-nums">
                            {val == null ? "—" : typeof val === "number" ? val.toFixed(4) : String(val)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-sm text-sparc-gray-600">No tabular results for this stage.</p>
            )}
          </div>
        )}

        {tab === "chart" && (
          <div className="h-full overflow-auto p-4">
            {data && numericCols.length >= 1 ? (
              <div className="space-y-8">
                {/* Bar chart of means */}
                {metrics && (
                  <div>
                    <h3 className="mb-2 text-sm font-semibold">Variable Means</h3>
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart
                        data={Object.entries(metrics).map(([k, v]) => ({ name: k, mean: v.mean }))}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                        <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
                        <YAxis tick={{ fontSize: 10 }} />
                        <Tooltip contentStyle={{ fontSize: 11 }} />
                        <Bar dataKey="mean" fill="#602468" radius={[3, 3, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Scatter: first two numeric cols */}
                {numericCols.length >= 2 && (
                  <div>
                    <h3 className="mb-2 text-sm font-semibold">
                      {numericCols[0]} vs {numericCols[1]}
                    </h3>
                    <ResponsiveContainer width="100%" height={280}>
                      <ScatterChart>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                        <XAxis dataKey={numericCols[0]} name={numericCols[0]} tick={{ fontSize: 10 }} />
                        <YAxis dataKey={numericCols[1]} name={numericCols[1]} tick={{ fontSize: 10 }} />
                        <Tooltip contentStyle={{ fontSize: 11 }} />
                        <Scatter data={data.slice(0, 500)} fill="#a44eb4" fillOpacity={0.6} />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-sm text-sparc-gray-600">No chart data for this stage.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
