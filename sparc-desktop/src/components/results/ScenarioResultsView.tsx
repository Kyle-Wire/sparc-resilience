import { useState, useEffect, useMemo } from "react";
import { getScenarioDetail } from "@/lib/api";
import type { ScenarioDetail } from "@/lib/types";
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

export default function ScenarioResultsView() {
  const [detail, setDetail] = useState<ScenarioDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mapMode, setMapMode] = useState<MapMode>("scatter");
  const [showDelta, setShowDelta] = useState(false);
  const [selectedField, setSelectedField] = useState<string>("");

  useEffect(() => {
    getScenarioDetail()
      .then(setDetail)
      .catch(() => setError("Scenario results not available. Run Stage 4 first."));
  }, []);

  // Discover pred/delta columns from the GeoJSON features
  const { predColumns, deltaColumns, baselineCol } = useMemo(() => {
    if (!detail?.geojson?.features?.length) {
      return { predColumns: [] as string[], deltaColumns: [] as string[], baselineCol: "", scenarioNames: [] as string[] };
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
    // Show pred/delta + any observed/identifier columns, skip geometry
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

  // Detect numeric summary keys for chart
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

  if (error) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-sparc-gray-600">{error}</p>
      </div>
    );
  }

  if (!detail) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-sparc-gray-500">Loading scenario results…</p>
      </div>
    );
  }

  const COLORS = ["#602468", "#a44eb4", "#e0a0b0", "#fbdd46", "#3b82f6", "#10b981"];

  return (
    <div className="flex h-full flex-col gap-4 p-4 overflow-auto">
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
