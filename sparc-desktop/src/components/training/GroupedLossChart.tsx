import { useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";
import type { EpochEntry } from "@/hooks/PipelineProvider";

// Group loss components into logical tracks
const LOSS_GROUPS: Record<string, { label: string; color: string; components: string[] }> = {
  core: {
    label: "Core",
    color: "#60a5fa",
    components: ["mse", "cross_entropy"],
  },
  physics: {
    label: "Physics",
    color: "#f472b6",
    components: ["physics", "pde_total", "bc_total", "alpha_prior"],
  },
  regularization: {
    label: "Regularization",
    color: "#34d399",
    components: ["neighborhood", "surrogate"],
  },
};

const COMPONENT_COLORS: Record<string, string> = {
  mse: "#60a5fa",
  cross_entropy: "#fbbf24",
  physics: "#f472b6",
  pde_total: "#a78bfa",
  bc_total: "#fb923c",
  alpha_prior: "#e879f9",
  neighborhood: "#34d399",
  surrogate: "#38bdf8",
};

interface Props {
  epochHistory: EpochEntry[];
  curriculumStage: string | null;
  curriculumLabel: string | null;
}

export function GroupedLossChart({ epochHistory, curriculumStage, curriculumLabel }: Props) {
  const [enabledGroups, setEnabledGroups] = useState<Set<string>>(
    new Set(Object.keys(LOSS_GROUPS))
  );
  const [showTotal, setShowTotal] = useState(true);
  const [showComponents, setShowComponents] = useState(false);

  if (epochHistory.length === 0) return null;

  const { data, componentKeys, transitions, groupTotals } = useMemo(() => {
    const keys = new Set<string>();
    const trans: { epoch: number; label: string }[] = [];
    let prevPhase = "";
    const gTotals = new Set<string>();

    const rows = epochHistory.map((e, idx) => {
      const row: Record<string, number> = {
        idx,
        epoch: e.epoch,
        total_loss: e.total_loss,
      };

      if (e.components) {
        // Individual components
        for (const [k, v] of Object.entries(e.components)) {
          row[k] = v;
          keys.add(k);
        }

        // Compute group subtotals
        for (const [gk, group] of Object.entries(LOSS_GROUPS)) {
          let sum = 0;
          let found = false;
          for (const comp of group.components) {
            if (comp in e.components) {
              sum += e.components[comp];
              found = true;
            }
          }
          if (found) {
            row[`_group_${gk}`] = sum;
            gTotals.add(gk);
          }
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

    return {
      data: rows,
      componentKeys: Array.from(keys),
      transitions: trans,
      groupTotals: Array.from(gTotals),
    };
  }, [epochHistory]);

  const toggleGroup = (gk: string) => {
    setEnabledGroups((prev) => {
      const next = new Set(prev);
      if (next.has(gk)) next.delete(gk);
      else next.add(gk);
      return next;
    });
  };

  // Determine which lines to show
  const visibleComponents = showComponents
    ? componentKeys.filter((k) => {
        for (const [gk, group] of Object.entries(LOSS_GROUPS)) {
          if (group.components.includes(k) && enabledGroups.has(gk)) return true;
        }
        return false;
      })
    : [];

  const visibleGroupLines = groupTotals.filter((gk) => enabledGroups.has(gk));

  return (
    <div className="rounded-lg border border-zinc-700 bg-zinc-900 p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-zinc-300">Loss Breakdown</h3>
          {curriculumStage && (
            <span className="rounded bg-indigo-900/60 px-2 py-0.5 text-xs text-indigo-300">
              {curriculumStage}{curriculumLabel ? `: ${curriculumLabel}` : ""}
            </span>
          )}
        </div>

        {/* Group toggle buttons */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setShowTotal(!showTotal)}
            className={`rounded px-2 py-0.5 text-[10px] font-medium transition-colors ${
              showTotal
                ? "bg-zinc-700 text-zinc-100"
                : "bg-zinc-800 text-zinc-500 hover:text-zinc-300"
            }`}
          >
            Total
          </button>
          {Object.entries(LOSS_GROUPS).map(([gk, group]) => (
            <button
              key={gk}
              onClick={() => toggleGroup(gk)}
              className={`rounded px-2 py-0.5 text-[10px] font-medium transition-colors ${
                enabledGroups.has(gk)
                  ? "text-white"
                  : "bg-zinc-800 text-zinc-500 hover:text-zinc-300"
              }`}
              style={enabledGroups.has(gk) ? { backgroundColor: group.color + "40" } : undefined}
            >
              {group.label}
            </button>
          ))}
          <span className="mx-1 h-3 w-px bg-zinc-700" />
          <button
            onClick={() => setShowComponents(!showComponents)}
            className={`rounded px-2 py-0.5 text-[10px] font-medium transition-colors ${
              showComponents
                ? "bg-zinc-600 text-zinc-100"
                : "bg-zinc-800 text-zinc-500 hover:text-zinc-300"
            }`}
          >
            Detail
          </button>
        </div>
      </div>

      <div className="h-72">
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
              formatter={(v: unknown, name: unknown) => [Number(v).toFixed(4), String(name)]}
              labelFormatter={(idx: unknown) => `Epoch ${data[Number(idx)]?.epoch ?? idx}`}
            />
            <Legend wrapperStyle={{ fontSize: 10 }} />

            {/* Total loss */}
            {showTotal && (
              <Line
                type="monotone"
                dataKey="total_loss"
                stroke="#e4e4e7"
                strokeWidth={2}
                dot={false}
                name="Total"
              />
            )}

            {/* Group subtotal lines */}
            {visibleGroupLines.map((gk) => (
              <Line
                key={`group_${gk}`}
                type="monotone"
                dataKey={`_group_${gk}`}
                stroke={LOSS_GROUPS[gk].color}
                strokeWidth={2}
                dot={false}
                name={LOSS_GROUPS[gk].label}
              />
            ))}

            {/* Individual component lines (when Detail mode is on) */}
            {visibleComponents.map((k) => (
              <Line
                key={k}
                type="monotone"
                dataKey={k}
                stroke={COMPONENT_COLORS[k] ?? "#888"}
                strokeWidth={1}
                strokeDasharray="3 3"
                dot={false}
                name={k}
              />
            ))}

            {/* Phase transitions */}
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
