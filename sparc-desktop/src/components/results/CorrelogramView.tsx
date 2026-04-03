import { useState, useEffect, useMemo } from "react";
import { getCorrelogramData } from "@/lib/api";
import type { CorrelogramData } from "@/lib/types";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
} from "recharts";

export default function CorrelogramView() {
  const [data, setData] = useState<CorrelogramData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedVar, setSelectedVar] = useState<string>("");
  const [compareVars, setCompareVars] = useState<string[]>([]);

  useEffect(() => {
    getCorrelogramData()
      .then((d) => {
        setData(d);
        const vars = Object.keys(d.individual_results ?? {});
        if (vars.length > 0) setSelectedVar(vars[0]);
      })
      .catch((e) => setError(e.message));
  }, []);

  const variables = useMemo(
    () => Object.keys(data?.individual_results ?? {}),
    [data],
  );

  const selectedResult = data?.individual_results?.[selectedVar];

  const chartData = useMemo(() => {
    if (!selectedResult?.correlogram_results) return [];
    const cr = selectedResult.correlogram_results;
    return cr.lag_distances.map((lag, i) => ({
      lag,
      morans_i: cr.morans_i_values[i],
      z_score: cr.z_scores?.[i],
      p_value: cr.p_values?.[i],
    }));
  }, [selectedResult]);

  // Find first zero-crossing for bandwidth annotation
  const zeroCrossing = useMemo(() => {
    if (!selectedResult?.correlogram_results) return null;
    const { lag_distances, morans_i_values } = selectedResult.correlogram_results;
    for (let i = 1; i < morans_i_values.length; i++) {
      if (morans_i_values[i - 1] > 0 && morans_i_values[i] <= 0) {
        // Linear interpolation for precise crossing
        const frac =
          morans_i_values[i - 1] / (morans_i_values[i - 1] - morans_i_values[i]);
        return lag_distances[i - 1] + frac * (lag_distances[i] - lag_distances[i - 1]);
      }
    }
    return selectedResult.optimal_bandwidth ?? null;
  }, [selectedResult]);

  // Multi-variable overlay data
  const overlayData = useMemo(() => {
    if (compareVars.length === 0 || !data?.individual_results) return null;
    const allVars = [selectedVar, ...compareVars.filter((v) => v !== selectedVar)];
    const maxLen = Math.max(
      ...allVars.map(
        (v) => data.individual_results[v]?.correlogram_results?.lag_distances?.length ?? 0,
      ),
    );
    const rows: Record<string, unknown>[] = [];
    for (let i = 0; i < maxLen; i++) {
      const row: Record<string, unknown> = {};
      for (const v of allVars) {
        const cr = data.individual_results[v]?.correlogram_results;
        if (cr && i < cr.lag_distances.length) {
          row.lag = cr.lag_distances[i];
          row[v] = cr.morans_i_values[i];
        }
      }
      if (row.lag !== undefined) rows.push(row);
    }
    return { rows, vars: allVars };
  }, [data, selectedVar, compareVars]);

  const COLORS = ["#602468", "#a44eb4", "#f0a0b0", "#fbdd46", "#4a90d9", "#e74c3c"];

  if (error) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-sparc-gray-600">{error}</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-sparc-gray-500">Loading correlogram data…</p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      {/* Controls */}
      <div className="flex items-center gap-3 border-b border-sparc-gray-100 px-4 py-2">
        <label className="text-xs text-sparc-gray-600">Variable:</label>
        <select
          value={selectedVar}
          onChange={(e) => setSelectedVar(e.target.value)}
          className="rounded border border-sparc-gray-200 px-2 py-1 text-xs"
        >
          {variables.map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>

        <label className="ml-4 text-xs text-sparc-gray-600">Compare:</label>
        <select
          multiple
          value={compareVars}
          onChange={(e) =>
            setCompareVars(Array.from(e.target.selectedOptions, (o) => o.value))
          }
          className="h-7 rounded border border-sparc-gray-200 px-2 text-[10px]"
        >
          {variables
            .filter((v) => v !== selectedVar)
            .map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
        </select>
        {compareVars.length > 0 && (
          <button
            onClick={() => setCompareVars([])}
            className="text-[10px] text-sparc-gray-400 hover:text-sparc-gray-600"
          >
            Clear
          </button>
        )}
      </div>

      {/* Metric card */}
      {selectedResult && (
        <div className="flex gap-3 border-b border-sparc-gray-100 px-4 py-3">
          <div className="rounded-md border border-sparc-gray-200 bg-white px-3 py-2 min-w-[140px]">
            <p className="text-[10px] font-medium text-sparc-gray-500">
              Optimal Bandwidth
            </p>
            <p className="text-sm font-bold tabular-nums">
              {zeroCrossing != null ? `${Math.round(zeroCrossing).toLocaleString()} m` : "—"}
            </p>
            <p className="text-[9px] text-sparc-gray-400">First zero-crossing</p>
          </div>
        </div>
      )}

      {/* Chart */}
      <div className="flex-1 overflow-auto p-4">
        {compareVars.length > 0 && overlayData ? (
          /* Multi-variable overlay */
          <div>
            <h3 className="mb-2 text-sm font-semibold">
              Spatial Autocorrelation — Multi-variable Comparison
            </h3>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={overlayData.rows}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis
                  dataKey="lag"
                  tick={{ fontSize: 10 }}
                  label={{ value: "Lag Distance (m)", position: "insideBottom", offset: -4, fontSize: 11 }}
                />
                <YAxis
                  tick={{ fontSize: 10 }}
                  label={{ value: "Moran's I", angle: -90, position: "insideLeft", fontSize: 11 }}
                />
                <Tooltip contentStyle={{ fontSize: 11 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <ReferenceLine y={0} stroke="#999" strokeDasharray="4 4" />
                {overlayData.vars.map((v, i) => (
                  <Line
                    key={v}
                    type="monotone"
                    dataKey={v}
                    stroke={COLORS[i % COLORS.length]}
                    strokeWidth={v === selectedVar ? 2.5 : 1.5}
                    dot={false}
                    name={v}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          /* Single variable */
          <div>
            <h3 className="mb-2 text-sm font-semibold">
              Spatial Autocorrelation — {selectedVar}
            </h3>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis
                  dataKey="lag"
                  tick={{ fontSize: 10 }}
                  label={{ value: "Lag Distance (m)", position: "insideBottom", offset: -4, fontSize: 11 }}
                />
                <YAxis
                  tick={{ fontSize: 10 }}
                  label={{ value: "Moran's I", angle: -90, position: "insideLeft", fontSize: 11 }}
                />
                <Tooltip
                  contentStyle={{ fontSize: 11 }}
                  formatter={(value: any, name: any) => {
                    if (name === "morans_i") return [value?.toFixed(4), "Moran's I"];
                    if (name === "z_score") return [value?.toFixed(2), "Z-score"];
                    if (name === "p_value") return [value?.toFixed(4), "p-value"];
                    return [value, name];
                  }}
                  labelFormatter={(label) => `Lag: ${Number(label).toLocaleString()} m`}
                />
                <ReferenceLine y={0} stroke="#999" strokeDasharray="4 4" label={{ value: "Moran's I = 0", fontSize: 10, fill: "#999" }} />
                {zeroCrossing != null && (
                  <ReferenceLine
                    x={zeroCrossing}
                    stroke="#602468"
                    strokeDasharray="6 3"
                    label={{
                      value: `Bandwidth: ${Math.round(zeroCrossing).toLocaleString()} m`,
                      fontSize: 10,
                      fill: "#602468",
                      position: "top",
                    }}
                  />
                )}
                <Line
                  type="monotone"
                  dataKey="morans_i"
                  stroke="#602468"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: "#602468" }}
                  activeDot={{ r: 5 }}
                  name="morans_i"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}
