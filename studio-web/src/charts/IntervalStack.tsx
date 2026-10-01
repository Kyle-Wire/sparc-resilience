// Layered intervals per row around one estimate (SPEC §6.4 Uncertainty): estimation,
// specification, attribution, causal band, envelope… Narrow-to-wide layers get lighter and
// thinner; a zero line and "excludes 0" badges show which layers rule out no effect.
import { Axis, axisTitle } from "./Axis";
import { ChartFrame, markProps, useChart, useChartWidth, type ChartTable, type FrameOptions } from "./ChartFrame";
import { bandScale, extent, linearScale, niceDomain } from "./scales";
import { fmtSigned } from "../theme/format";

export type IntervalLayer = { id: string; label: string; lo: number | null; hi: number | null };
export type IntervalRow = { id: string; label: string; estimate: number | null; layers: IntervalLayer[] };

export type IntervalStackProps = FrameOptions & {
  rows: IntervalRow[];
  valueLabel: string;
  unit?: string;
  decimals?: number;
  width?: number;
};

export const excludesZero = (l: { lo: number | null; hi: number | null }) => l.lo !== null && l.hi !== null && (l.lo > 0 || l.hi < 0);

export function intervalTable(p: Pick<IntervalStackProps, "rows" | "valueLabel" | "unit">): ChartTable {
  const rows: ChartTable["rows"] = [];
  for (const r of p.rows) for (const l of r.layers) rows.push([r.label, r.estimate, l.label, l.lo, l.hi, excludesZero(l) ? "yes" : "no"]);
  return {
    columns: [{ key: "row", label: "Scenario" }, { key: "est", label: p.valueLabel, unit: p.unit }, { key: "layer", label: "Interval" }, { key: "lo", label: "Low", unit: p.unit }, { key: "hi", label: "High", unit: p.unit }, { key: "x0", label: "Excludes 0" }],
    rows,
  };
}

function Plot(p: IntervalStackProps) {
  const ctx = useChart();
  const auto = useChartWidth(640);
  const W = p.width ?? auto;
  const M = { top: 10, right: 92, bottom: 44, left: 170 };
  const H = M.top + M.bottom + p.rows.length * 34;
  const e = extent([0, ...p.rows.flatMap((r) => [r.estimate, ...r.layers.flatMap((l) => [l.lo, l.hi])])]) ?? [-1, 1];
  const sx = linearScale(niceDomain(e[0], e[1]), [M.left, W - M.right]);
  const band = bandScale(p.rows.map((r) => r.id), [M.top, H - M.bottom], 0.3, 0.05);
  const d = p.decimals ?? 2;
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="group" aria-label={p.title}>
      <Axis scale={sx} orient="bottom" at={H - M.bottom} grid={H - M.top - M.bottom} title={axisTitle(p.valueLabel, p.unit)} signed />
      <line className="zero" x1={sx(0)} x2={sx(0)} y1={M.top} y2={H - M.bottom} />
      {p.rows.map((r) => {
        const cy = band(r.id) + band.bandwidth / 2;
        // Widest first, so narrower (darker, thicker) layers paint on top.
        const layers = [...r.layers].filter((l) => l.lo !== null && l.hi !== null).sort((a, b) => (b.hi! - b.lo!) - (a.hi! - a.lo!));
        const n = layers.length;
        const outer = layers[0];
        return (
          <g key={r.id}>
            <text className="t-label" x={M.left - 8} y={cy} dy="0.32em" textAnchor="end" aria-hidden="true">
              {r.label.length > 26 ? r.label.slice(0, 25) + "…" : r.label}
            </text>
            {layers.map((l, k) => {
              const thick = 4 + (k / Math.max(1, n - 1)) * 10;
              const label = `${r.label}, ${l.label}: ${fmtSigned(l.lo, d)} to ${fmtSigned(l.hi, d)}${p.unit ? " " + p.unit : ""}${excludesZero(l) ? ", excludes zero" : ", includes zero"}`;
              return (
                <rect
                  key={l.id}
                  x={sx(l.lo!)}
                  width={Math.max(1, sx(l.hi!) - sx(l.lo!))}
                  y={cy - thick / 2}
                  height={thick}
                  rx={2}
                  style={{ fill: "var(--s1)", fillOpacity: 0.18 + (0.55 * k) / Math.max(1, n - 1) }}
                  {...markProps(ctx, label)}
                />
              );
            })}
            {r.estimate !== null ? <line x1={sx(r.estimate)} x2={sx(r.estimate)} y1={cy - 10} y2={cy + 10} style={{ stroke: "var(--ink)" }} strokeWidth={2} aria-hidden="true" /> : null}
            {outer && excludesZero(outer) ? (
              <text className="t-strong" x={W - M.right + 8} y={cy} dy="0.32em" style={{ fontSize: 10.5 }}>
                excludes 0
              </text>
            ) : layers.some(excludesZero) ? (
              <text className="t-axis" x={W - M.right + 8} y={cy} dy="0.32em">
                {layers.filter(excludesZero).length}/{n} exclude 0
              </text>
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}

export function IntervalStack(props: IntervalStackProps) {
  const names = [...new Set(props.rows.flatMap((r) => r.layers.map((l) => l.label)))];
  const legend = props.legend ?? <span>from darkest to lightest: {names.join(" → ")} · black tick = estimate · "excludes 0" when the widest interval does</span>;
  return (
    <ChartFrame {...props} legend={legend} table={intervalTable(props)}>
      <Plot {...props} />
    </ChartFrame>
  );
}
