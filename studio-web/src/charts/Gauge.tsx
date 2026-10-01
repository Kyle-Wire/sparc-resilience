// Half-circle gauge (SPEC §6.4 Climate: "% of median warming offset"). Text carries the value,
// so the arc is never the only cue.
import { ChartFrame, useChart, type ChartTable, type FrameOptions } from "./ChartFrame";
import { fmtNum } from "../theme/format";

export type GaugeProps = FrameOptions & {
  value: number | null;
  min?: number;
  max?: number;
  label: string;
  format?: (v: number) => string;
  /** Threshold ticks on the arc (e.g. 1 = fully offset). */
  ticks?: number[];
  tone?: "accent" | "good" | "warn";
  size?: number;
};

export function gaugeTable(p: Pick<GaugeProps, "value" | "min" | "max" | "label">): ChartTable {
  return { columns: [{ key: "l", label: "Measure" }, { key: "v", label: "Value" }, { key: "min", label: "Scale min" }, { key: "max", label: "Scale max" }], rows: [[p.label, p.value, p.min ?? 0, p.max ?? 1]] };
}

function arc(cx: number, cy: number, r: number, t0: number, t1: number): string {
  const a0 = Math.PI * (1 - t0);
  const a1 = Math.PI * (1 - t1);
  const p0 = [cx + r * Math.cos(a0), cy - r * Math.sin(a0)];
  const p1 = [cx + r * Math.cos(a1), cy - r * Math.sin(a1)];
  return `M${p0[0].toFixed(2)},${p0[1].toFixed(2)}A${r},${r} 0 0 1 ${p1[0].toFixed(2)},${p1[1].toFixed(2)}`;
}

function Plot(p: GaugeProps) {
  useChart();
  const S = p.size ?? 220;
  const cx = S / 2;
  const cy = S / 2 + 6;
  const r = S / 2 - 18;
  const min = p.min ?? 0;
  const max = p.max ?? 1;
  const t = p.value === null || !Number.isFinite(p.value) ? 0 : Math.min(1, Math.max(0, (p.value - min) / (max - min || 1)));
  const text = p.value === null || !Number.isFinite(p.value) ? "—" : p.format ? p.format(p.value) : fmtNum(p.value, 2);
  const color = p.tone === "good" ? "var(--good)" : p.tone === "warn" ? "var(--warning)" : "var(--s1)";
  return (
    <svg className="chart" viewBox={`0 0 ${S} ${S / 2 + 30}`} role="img" aria-label={`${p.label}: ${text}`} style={{ maxWidth: S }}>
      <path d={arc(cx, cy, r, 0, 1)} fill="none" style={{ stroke: "var(--chip)" }} strokeWidth={14} strokeLinecap="round" />
      {t > 0 ? <path d={arc(cx, cy, r, 0, t)} fill="none" style={{ stroke: color }} strokeWidth={14} strokeLinecap="round" /> : null}
      {(p.ticks ?? []).map((v) => {
        const tt = Math.min(1, Math.max(0, (v - min) / (max - min || 1)));
        const a = Math.PI * (1 - tt);
        return <line key={v} x1={cx + (r - 11) * Math.cos(a)} y1={cy - (r - 11) * Math.sin(a)} x2={cx + (r + 11) * Math.cos(a)} y2={cy - (r + 11) * Math.sin(a)} style={{ stroke: "var(--ink)" }} strokeWidth={1.5} />;
      })}
      <text className="t-strong" x={cx} y={cy - 8} textAnchor="middle" style={{ fontSize: 26 }}>
        {text}
      </text>
      <text className="t-label" x={cx} y={cy + 16} textAnchor="middle">
        {p.label}
      </text>
    </svg>
  );
}

export function Gauge(props: GaugeProps) {
  return (
    <ChartFrame {...props} table={gaugeTable(props)}>
      <Plot {...props} />
    </ChartFrame>
  );
}
