import { useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";
import type { EpochEntry } from "@/hooks/PipelineProvider";

interface Props {
  epochHistory: EpochEntry[];
  curriculumStage: string | null;
  curriculumLabel: string | null;
}

// Consistent colors for each loss component
const COMPONENT_COLORS: Record<string, string> = {
  mse: "#60a5fa",       // blue
  physics: "#f472b6",   // pink
  neighborhood: "#34d399", // green
  cross_entropy: "#fbbf24", // amber
  pde_total: "#a78bfa",  // violet
  bc_total: "#fb923c",   // orange
  alpha_prior: "#e879f9", // fuchsia
  surrogate: "#38bdf8",  // sky
};

export function EpochLossChart({ epochHistory, curriculumStage, curriculumLabel }: Props) {
  if (epochHistory.length === 0) return null;

  // Build data rows and detect all component keys seen
  const { data, componentKeys, transitions } = useMemo(() => {
    const keys = new Set<string>();
    const trans: { epoch: number; label: string }[] = [];
    let prevPhase = "";

    const rows = epochHistory.map((e, idx) => {
      const row: Record<string, number> = {
        idx,
        epoch: e.epoch,
        total_loss: e.total_loss,
      };
      if (e.components) {
        for (const [k, v] of Object.entries(e.components)) {
          row[k] = v;
          keys.add(k);
        }
      }
      // Detect phase transitions
      if (e.train_phase !== prevPhase && idx > 0) {
        const phaseLabel = e.train_phase === "retrain" ? "Retrain" : e.train_phase === "swa" ? "SWA" : "CV";
        trans.push({ epoch: idx, label: phaseLabel });
      }
      prevPhase = e.train_phase;
      return row;
    });

    return { data: rows, componentKeys: Array.from(keys), transitions: trans };
  }, [epochHistory]);

  return (
    <div className="rounded-lg border border-zinc-700 bg-zinc-900 p-4">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-zinc-300">Training Loss</h3>
        {curriculumStage && (
          <span className="rounded bg-indigo-900/60 px-2 py-0.5 text-xs text-indigo-300">
            {curriculumStage}{curriculumLabel ? `: ${curriculumLabel}` : ""}
          </span>
        )}
      </div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 12, bottom: 4, left: 12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis
              dataKey="idx"
              tick={{ fill: "#a1a1aa", fontSize: 10 }}
              label={{ value: "Epoch", position: "insideBottom", offset: -2, fill: "#a1a1aa", fontSize: 10 }}
            />
            <YAxis
              scale="log"
              domain={["auto", "auto"]}
              allowDataOverflow
              tick={{ fill: "#a1a1aa", fontSize: 10 }}
              label={{ value: "Loss (log)", angle: -90, position: "insideLeft", fill: "#a1a1aa", fontSize: 10 }}
            />
            <Tooltip
              contentStyle={{ backgroundColor: "#18181b", border: "1px solid #3f3f46", borderRadius: 6, fontSize: 11 }}
              labelStyle={{ color: "#a1a1aa" }}
              formatter={(v: number, name: string) => [v.toFixed(4), name]}
              labelFormatter={(idx: number) => `Epoch ${data[idx]?.epoch ?? idx}`}
            />
            <Legend wrapperStyle={{ fontSize: 10 }} />

            {/* Total loss — bold white */}
            <Line
              type="monotone"
              dataKey="total_loss"
              stroke="#e4e4e7"
              strokeWidth={2}
              dot={false}
              name="Total"
            />

            {/* Per-component lines */}
            {componentKeys.map((k) => (
              <Line
                key={k}
                type="monotone"
                dataKey={k}
                stroke={COMPONENT_COLORS[k] ?? "#888"}
                strokeWidth={1}
                dot={false}
                name={k}
              />
            ))}

            {/* Phase transition markers */}
            {transitions.map((t) => (
              <ReferenceLine
                key={`phase-${t.epoch}`}
                x={t.epoch}
                stroke="#fbbf24"
                strokeDasharray="4 4"
                label={{ value: t.label, fill: "#fbbf24", fontSize: 10, position: "top" }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
