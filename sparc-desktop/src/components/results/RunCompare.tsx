import { useState, useEffect, useMemo } from "react";
import { getResults, generateReport } from "@/lib/api";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const COMPARE_STAGES = [
  { value: 2, label: "Stage 2 – Spatial CV", color: "#602468" },
  { value: 3, label: "Stage 3 – Causal", color: "#a44eb4" },
  { value: 4, label: "Stage 4 – Scenarios", color: "#fbdd46" },
];

interface StageData {
  stage: number;
  rows: Record<string, unknown>[];
  metrics: Record<string, { mean: number; std: number; min: number; max: number }>;
}

function computeMetrics(rows: Record<string, unknown>[]) {
  const nums: Record<string, number[]> = {};
  for (const row of rows) {
    for (const [k, v] of Object.entries(row)) {
      if (typeof v === "number" && isFinite(v)) {
        if (!nums[k]) nums[k] = [];
        nums[k].push(v);
      }
    }
  }
  return Object.fromEntries(
    Object.entries(nums).map(([k, arr]) => {
      const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
      const std = Math.sqrt(arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length);
      return [k, { mean, std, min: Math.min(...arr), max: Math.max(...arr) }];
    }),
  );
}

export default function RunCompare() {
  const [stageData, setStageData] = useState<StageData[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedMetric, setSelectedMetric] = useState("");
  const [reportMd, setReportMd] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    setLoading(true);
    Promise.allSettled(
      COMPARE_STAGES.map((s) =>
        getResults(s.value).then((r: any) => ({
          stage: s.value,
          rows: r?.rows ?? [],
          metrics: computeMetrics(r?.rows ?? []),
        })),
      ),
    ).then((results) => {
      const loaded = results
        .filter((r): r is PromiseFulfilledResult<StageData> => r.status === "fulfilled")
        .map((r) => r.value)
        .filter((d) => d.rows.length > 0);
      setStageData(loaded);
      setLoading(false);
    });
  }, []);

  // All shared metric keys across stages
  const sharedKeys = useMemo(() => {
    if (stageData.length === 0) return [];
    const allKeys = stageData.map((d) => new Set(Object.keys(d.metrics)));
    const shared = [...allKeys[0]].filter((k) => allKeys.every((s) => s.has(k)));
    return shared;
  }, [stageData]);

  useEffect(() => {
    if (sharedKeys.length > 0 && !selectedMetric) {
      setSelectedMetric(sharedKeys[0]);
    }
  }, [sharedKeys, selectedMetric]);

  // Chart data: one entry per stage for the selected metric
  const chartData = useMemo(() => {
    if (!selectedMetric) return [];
    return COMPARE_STAGES.map((s) => {
      const sd = stageData.find((d) => d.stage === s.value);
      return {
        name: s.label,
        mean: sd?.metrics[selectedMetric]?.mean ?? null,
        std: sd?.metrics[selectedMetric]?.std ?? null,
        min: sd?.metrics[selectedMetric]?.min ?? null,
        max: sd?.metrics[selectedMetric]?.max ?? null,
      };
    }).filter((d) => d.mean !== null);
  }, [stageData, selectedMetric]);

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-sparc-gray-500">Loading results across stages…</p>
      </div>
    );
  }

  if (stageData.length === 0) {
    return (
      <div className="p-6">
        <h2 className="text-lg font-bold">Run Comparison</h2>
        <p className="mt-2 text-sm text-sparc-gray-600">
          No completed stages with results. Run the pipeline first.
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold">Run Comparison</h2>
          <p className="text-xs text-sparc-gray-500">
            Compare metrics across pipeline stages
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={async () => {
              setGenerating(true);
              try {
                const res = await generateReport("markdown");
                setReportMd(res.markdown);
              } catch {}
              setGenerating(false);
            }}
            disabled={generating}
            className="rounded border border-sparc-gray-300 px-3 py-1.5 text-xs font-medium hover:bg-sparc-gray-100 disabled:opacity-50"
          >
            {generating ? "Generating…" : "Export Report"}
          </button>
          {sharedKeys.length > 0 && (
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
              className="rounded border border-sparc-gray-200 px-3 py-1.5 text-xs"
            >
              {sharedKeys.map((k) => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
          )}
        </div>
      </div>

      {/* Report modal */}
      {reportMd && (
        <div className="rounded-lg border border-sparc-gray-200 bg-sparc-gray-50 p-4">
          <div className="mb-2 flex items-center justify-between">
            <h3 className="text-sm font-semibold">Pipeline Report</h3>
            <div className="flex gap-2">
              <button
                onClick={() => {
                  navigator.clipboard.writeText(reportMd);
                }}
                className="rounded border border-sparc-gray-300 px-2 py-1 text-[10px] hover:bg-white"
              >
                Copy
              </button>
              <button
                onClick={() => setReportMd(null)}
                className="rounded border border-sparc-gray-300 px-2 py-1 text-[10px] hover:bg-white"
              >
                Close
              </button>
            </div>
          </div>
          <pre className="max-h-80 overflow-auto whitespace-pre-wrap text-xs font-mono text-sparc-gray-700">
            {reportMd}
          </pre>
        </div>
      )}

      {/* Stage summary cards */}
      <div className="grid grid-cols-3 gap-4">
        {COMPARE_STAGES.map((s) => {
          const sd = stageData.find((d) => d.stage === s.value);
          return (
            <div
              key={s.value}
              className="rounded-lg border-2 p-4"
              style={{ borderColor: sd ? s.color : "#e0e0e0" }}
            >
              <p className="text-xs font-semibold" style={{ color: s.color }}>{s.label}</p>
              {sd ? (
                <>
                  <p className="mt-1 text-2xl font-bold tabular-nums">
                    {sd.metrics[selectedMetric]?.mean?.toFixed(4) ?? "—"}
                  </p>
                  <p className="text-[10px] text-sparc-gray-500">
                    {sd.rows.length} rows · {Object.keys(sd.metrics).length} metrics
                  </p>
                </>
              ) : (
                <p className="mt-1 text-sm text-sparc-gray-400">No data</p>
              )}
            </div>
          );
        })}
      </div>

      {/* Line chart of selected metric across stages */}
      {chartData.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold">
            {selectedMetric} across stages
          </h3>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ fontSize: 11 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="mean" stroke="#602468" strokeWidth={2} dot={{ r: 5 }} />
              <Line type="monotone" dataKey="min" stroke="#f0a0b0" strokeDasharray="5 5" dot={{ r: 3 }} />
              <Line type="monotone" dataKey="max" stroke="#fbdd46" strokeDasharray="5 5" dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Detailed metric comparison table */}
      <div>
        <h3 className="mb-2 text-sm font-semibold">All Metrics</h3>
        <div className="overflow-auto rounded border border-sparc-gray-200">
          <table className="w-full text-left text-xs">
            <thead className="bg-sparc-gray-50">
              <tr>
                <th className="px-3 py-2 font-medium">Metric</th>
                {COMPARE_STAGES.map((s) => (
                  <th
                    key={s.value}
                    className="px-3 py-2 font-medium"
                    style={{ color: s.color }}
                  >
                    {s.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sharedKeys.map((k) => (
                <tr
                  key={k}
                  className={`border-t border-sparc-gray-100 ${
                    k === selectedMetric ? "bg-sparc-purple/5" : ""
                  }`}
                  onClick={() => setSelectedMetric(k)}
                  style={{ cursor: "pointer" }}
                >
                  <td className="px-3 py-1.5 font-mono">{k}</td>
                  {COMPARE_STAGES.map((s) => {
                    const sd = stageData.find((d) => d.stage === s.value);
                    const m = sd?.metrics[k];
                    return (
                      <td key={s.value} className="px-3 py-1.5 tabular-nums">
                        {m ? `${m.mean.toFixed(4)} ± ${m.std.toFixed(3)}` : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
