import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import type { CapacityResult } from "@/hooks/PipelineProvider";

interface Props {
  results: CapacityResult[];
}

export function CapacitySweepView({ results }: Props) {
  if (results.length === 0) return null;

  const bestDim = results.reduce((best, r) => (r.r2 > best.r2 ? r : best), results[0]).hidden_dim;

  return (
    <div className="rounded-lg border border-zinc-700 bg-zinc-900 p-4">
      <h3 className="mb-2 text-sm font-semibold text-zinc-300">Capacity Sweep</h3>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={results} margin={{ top: 4, right: 12, bottom: 4, left: 12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
            <XAxis
              dataKey="hidden_dim"
              tick={{ fill: "#a1a1aa", fontSize: 11 }}
              label={{ value: "Hidden Dim", position: "insideBottom", offset: -2, fill: "#a1a1aa", fontSize: 11 }}
            />
            <YAxis
              tick={{ fill: "#a1a1aa", fontSize: 11 }}
              label={{ value: "CV R²", angle: -90, position: "insideLeft", fill: "#a1a1aa", fontSize: 11 }}
              domain={["auto", "auto"]}
            />
            <Tooltip
              contentStyle={{ backgroundColor: "#18181b", border: "1px solid #3f3f46", borderRadius: 6 }}
              labelStyle={{ color: "#a1a1aa" }}
              formatter={(v: number) => [v.toFixed(4), "R²"]}
              labelFormatter={(dim: number) => `hidden_dim=${dim}`}
            />
            <Bar dataKey="r2" radius={[4, 4, 0, 0]}>
              {results.map((r) => (
                <Cell key={r.hidden_dim} fill={r.hidden_dim === bestDim ? "#22c55e" : "#6366f1"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-1 text-center text-xs text-zinc-500">
        Best: <span className="text-green-400 font-mono">{bestDim}</span> (R² = {results.find((r) => r.hidden_dim === bestDim)?.r2.toFixed(4)})
      </p>
    </div>
  );
}
