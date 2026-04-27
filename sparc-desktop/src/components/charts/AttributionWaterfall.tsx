/**
 * AttributionWaterfall — horizontal waterfall chart for a single scenario.
 *
 * Renders contributions per variable as positive/negative bars stacked from
 * a baseline value. Pure SVG, no external deps. Mirrors the matplotlib
 * server-side renderer in `sparc/report/figures.py`.
 */
import { useMemo } from "react";
import type { AttributionRecord } from "@/lib/api";

interface AttributionWaterfallProps {
  records: AttributionRecord[];
  scenarioId?: number | string;
  /** Optional baseline value at the left of the waterfall. */
  baseline?: number;
  width?: number;
  height?: number;
  positiveColor?: string;
  negativeColor?: string;
}

export default function AttributionWaterfall({
  records,
  scenarioId,
  baseline = 0,
  width = 520,
  height = 320,
  positiveColor = "#7a3a3a",
  negativeColor = "#3a5e7a",
}: AttributionWaterfallProps) {
  const data = useMemo(() => {
    const filtered = scenarioId !== undefined
      ? records.filter((r) => String(r.scenario_id) === String(scenarioId))
      : records;
    return filtered
      .map((r) => ({ variable: r.variable, contribution: Number(r.contribution) || 0 }))
      .filter((d) => isFinite(d.contribution));
  }, [records, scenarioId]);

  const layout = useMemo(() => {
    const left = 130, right = 16, top = 16, bottom = 32;
    const w = width - left - right;
    const h = height - top - bottom;
    const rowH = data.length ? h / data.length : 0;
    let running = baseline;
    const bars = data.map((d) => {
      const start = running;
      const end = running + d.contribution;
      running = end;
      return { ...d, start, end };
    });
    const total = running;
    const allVals = bars.flatMap((b) => [b.start, b.end]).concat([baseline, total]);
    const lo = Math.min(...allVals);
    const hi = Math.max(...allVals);
    const xRange = hi - lo || 1;
    const toX = (v: number) => left + ((v - lo) / xRange) * w;
    return { left, right, top, bottom, w, h, rowH, bars, lo, hi, total, toX };
  }, [data, width, height, baseline]);

  if (!data.length) {
    return (
      <div style={{ padding: 20, fontSize: 12, color: "#888" }}>
        No attribution records.
      </div>
    );
  }

  const { left, top, rowH, bars, lo, hi, total, toX } = layout;
  const zeroX = toX(baseline);
  const totalX = toX(total);

  return (
    <svg width={width} height={height} role="img" aria-label="Attribution waterfall">
      {/* Axis */}
      <line x1={left} x2={width - 16} y1={height - 24} y2={height - 24} stroke="#999" strokeWidth={1} />
      <text x={toX(lo)} y={height - 8} fontSize={10} fill="#666" textAnchor="start">{lo.toFixed(2)}</text>
      <text x={toX(hi)} y={height - 8} fontSize={10} fill="#666" textAnchor="end">{hi.toFixed(2)}</text>
      {/* Baseline marker */}
      <line x1={zeroX} x2={zeroX} y1={top} y2={height - 24} stroke="#bbb" strokeDasharray="3,2" />
      {/* Total marker */}
      <line x1={totalX} x2={totalX} y1={top} y2={height - 24} stroke="#7a3a3a" strokeWidth={1.5} />
      {/* Bars */}
      {bars.map((b, i) => {
        const y = top + i * rowH + 2;
        const barH = Math.max(2, rowH - 4);
        const x0 = toX(b.start);
        const x1 = toX(b.end);
        const xLeft = Math.min(x0, x1);
        const w = Math.abs(x1 - x0);
        const color = b.contribution >= 0 ? positiveColor : negativeColor;
        return (
          <g key={`${b.variable}-${i}`}>
            <rect x={xLeft} y={y} width={w} height={barH} fill={color} fillOpacity={0.85} />
            <text x={left - 6} y={y + barH / 2 + 3} fontSize={10} fill="#333" textAnchor="end">
              {b.variable.length > 20 ? `${b.variable.slice(0, 18)}…` : b.variable}
            </text>
            <text
              x={x1 + (b.contribution >= 0 ? 4 : -4)}
              y={y + barH / 2 + 3}
              fontSize={9}
              fill="#555"
              textAnchor={b.contribution >= 0 ? "start" : "end"}
            >
              {b.contribution >= 0 ? "+" : ""}{b.contribution.toFixed(3)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
