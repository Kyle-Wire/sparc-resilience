// Anisotropy rose (SPEC §6.4 Influence): one petal per direction, drawn on both sides
// (axial data), length ∝ value; unreliable directions are dashed outlines.
import { ChartFrame, markProps, useChart, type ChartTable, type FrameOptions } from "./ChartFrame";
import { fmtNum } from "../theme/format";

export type RosePetal = { angle_deg: number; value: number | null; label?: string; reliable?: boolean };

export type RoseProps = FrameOptions & {
  petals: RosePetal[];
  unit?: string;
  valueLabel?: string;
  max?: number;
  /** Angular width of each petal in degrees (default: 180 / number of petals). */
  spread?: number;
  decimals?: number;
  size?: number;
};

export function roseTable(p: Pick<RoseProps, "petals" | "unit" | "valueLabel">): ChartTable {
  return {
    columns: [{ key: "a", label: "Direction", unit: "° from north" }, { key: "v", label: p.valueLabel ?? "Value", unit: p.unit }, { key: "r", label: "Reliable" }, { key: "l", label: "Label" }],
    rows: p.petals.map((q) => [q.angle_deg, q.value, q.reliable === undefined ? "" : q.reliable ? "yes" : "no", q.label ?? ""]),
  };
}

/** Wedge path centred on bearing `deg` (clockwise from north). */
function wedge(cx: number, cy: number, r: number, deg: number, spread: number): string {
  const a0 = ((deg - spread / 2 - 90) * Math.PI) / 180;
  const a1 = ((deg + spread / 2 - 90) * Math.PI) / 180;
  return `M${cx},${cy}L${(cx + r * Math.cos(a0)).toFixed(2)},${(cy + r * Math.sin(a0)).toFixed(2)}A${r},${r} 0 0 1 ${(cx + r * Math.cos(a1)).toFixed(2)},${(cy + r * Math.sin(a1)).toFixed(2)}Z`;
}

function Plot(p: RoseProps) {
  const ctx = useChart();
  const S = p.size ?? 260;
  const c = S / 2;
  const R = S / 2 - 22;
  const max = p.max ?? Math.max(1e-12, ...p.petals.map((q) => q.value ?? 0));
  const spread = p.spread ?? Math.min(40, 180 / Math.max(1, p.petals.length));
  const d = p.decimals ?? 2;
  return (
    <svg className="chart" viewBox={`0 0 ${S} ${S}`} role="group" aria-label={p.title} style={{ maxWidth: S }}>
      {[0.25, 0.5, 0.75, 1].map((f) => (
        <circle key={f} cx={c} cy={c} r={R * f} fill="none" className="gridline" aria-hidden="true" />
      ))}
      {[0, 45, 90, 135].map((a) => {
        const rad = ((a - 90) * Math.PI) / 180;
        return <line key={a} className="gridline" x1={c - R * Math.cos(rad)} y1={c - R * Math.sin(rad)} x2={c + R * Math.cos(rad)} y2={c + R * Math.sin(rad)} aria-hidden="true" />;
      })}
      {(["N", "E", "S", "W"] as const).map((t, i) => {
        const rad = ((i * 90 - 90) * Math.PI) / 180;
        return (
          <text key={t} className="t-axis" x={c + (R + 12) * Math.cos(rad)} y={c + (R + 12) * Math.sin(rad)} dy="0.32em" textAnchor="middle" aria-hidden="true">
            {t}
          </text>
        );
      })}
      {p.petals.map((q, i) => {
        if (q.value === null || !Number.isFinite(q.value)) return null;
        const r = (Math.max(0, q.value) / max) * R;
        const unreliable = q.reliable === false;
        const st = { fill: unreliable ? "none" : "var(--s1)", fillOpacity: 0.5, stroke: "var(--s1)", strokeDasharray: unreliable ? "3 3" : undefined };
        const label = `${q.label ?? `${fmtNum(q.angle_deg, 0)}°`}: ${fmtNum(q.value, d)}${p.unit ? " " + p.unit : ""}${unreliable ? ", not confirmed by bootstrap" : q.reliable ? ", reliable" : ""}`;
        return (
          <g key={i} {...markProps(ctx, label)}>
            <path d={wedge(c, c, r, q.angle_deg, spread)} style={st} />
            <path d={wedge(c, c, r, q.angle_deg + 180, spread)} style={st} />
          </g>
        );
      })}
    </svg>
  );
}

export function Rose(props: RoseProps) {
  const legend = props.legend ?? (props.petals.some((q) => q.reliable === false) ? <span>dashed petal = direction not confirmed by the bootstrap</span> : undefined);
  return (
    <ChartFrame {...props} legend={legend} table={roseTable(props)}>
      <Plot {...props} />
    </ChartFrame>
  );
}
