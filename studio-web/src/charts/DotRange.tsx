// Dot-and-whisker rows: estimate with a likely range (thick) and an outer range (thin, e.g.
// p10–p90 or min–max), hollow when extrapolated, plus an orange diamond for the independent
// causal check (SPEC §6.4 Scenarios, Climate).
import { Axis, axisTitle } from "./Axis";
import { ChartFrame, markProps, useChart, useChartWidth, type ChartTable, type FrameOptions } from "./ChartFrame";
import { bandScale, extent, linearScale, niceDomain } from "./scales";
import { fmtNum, fmtSigned } from "../theme/format";

export type DotRangeRow = {
  id: string;
  label: string;
  est: number | null;
  lo?: number | null;
  hi?: number | null;
  /** Outer range (thin whisker). */
  lo2?: number | null;
  hi2?: number | null;
  hollow?: boolean;
  muted?: boolean;
  /** Independent check (orange diamond with its band). */
  check?: { est: number; lo?: number | null; hi?: number | null; label?: string } | null;
  /** Individual points behind the dot (e.g. per-model warming strip). */
  strip?: number[];
};

export type DotRangeProps = FrameOptions & {
  rows: DotRangeRow[];
  valueLabel: string;
  unit?: string;
  decimals?: number;
  /** Signed values (temperature changes): ticks and labels get +/−, a zero line is drawn. */
  signed?: boolean;
  rangeLabel?: string;
  outerLabel?: string;
  width?: number;
  onRowClick?: (id: string) => void;
};

export function dotRangeTable(p: Pick<DotRangeProps, "rows" | "valueLabel" | "unit" | "rangeLabel" | "outerLabel">): ChartTable {
  return {
    columns: [
      { key: "label", label: "Row" },
      { key: "est", label: p.valueLabel, unit: p.unit },
      { key: "lo", label: `${p.rangeLabel ?? "Likely range"} low`, unit: p.unit },
      { key: "hi", label: `${p.rangeLabel ?? "Likely range"} high`, unit: p.unit },
      { key: "lo2", label: `${p.outerLabel ?? "Outer range"} low`, unit: p.unit },
      { key: "hi2", label: `${p.outerLabel ?? "Outer range"} high`, unit: p.unit },
      { key: "check", label: "Independent check", unit: p.unit },
      { key: "extrap", label: "Extrapolated" },
    ],
    rows: p.rows.map((r) => [r.label, r.est, r.lo ?? null, r.hi ?? null, r.lo2 ?? null, r.hi2 ?? null, r.check?.est ?? null, r.hollow ? "yes" : "no"]),
  };
}

function Plot(p: DotRangeProps) {
  const ctx = useChart();
  const auto = useChartWidth(640);
  const W = p.width ?? auto;
  const M = { top: 8, right: 20, bottom: 44, left: 170 };
  const H = M.top + M.bottom + p.rows.length * 28;
  const vals = p.rows.flatMap((r) => [r.est, r.lo, r.hi, r.lo2, r.hi2, r.check?.est, r.check?.lo, r.check?.hi, ...(r.strip ?? [])]);
  const e = extent(vals) ?? [0, 1];
  const dom = niceDomain(p.signed ? Math.min(0, e[0]) : e[0], p.signed ? Math.max(0, e[1]) : e[1]);
  const sx = linearScale(dom, [M.left, W - M.right]);
  const band = bandScale(p.rows.map((r) => r.id), [M.top, H - M.bottom], 0.3, 0.05);
  const d = p.decimals ?? 2;
  const f = (v: number | null | undefined) => (p.signed ? fmtSigned(v, d) : fmtNum(v, d));
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="group" aria-label={p.title}>
      <Axis scale={sx} orient="bottom" at={H - M.bottom} grid={H - M.top - M.bottom} title={axisTitle(p.valueLabel, p.unit)} signed={p.signed} />
      {p.rows.map((r) => (
        <text key={`t${r.id}`} className="t-label" x={M.left - 8} y={band(r.id) + band.bandwidth / 2} dy="0.32em" textAnchor="end" aria-hidden="true">
          {r.label.length > 26 ? r.label.slice(0, 25) + "…" : r.label}
        </text>
      ))}
      {dom[0] < 0 && dom[1] > 0 ? <line className="zero" x1={sx(0)} x2={sx(0)} y1={M.top} y2={H - M.bottom} /> : null}
      {p.rows.map((r) => {
        const cy = band(r.id) + band.bandwidth / 2;
        const color = r.muted ? "var(--gray-mark)" : "var(--s1)";
        const label = `${r.label}: ${f(r.est)}${p.unit ? " " + p.unit : ""}${r.lo !== undefined && r.lo !== null && r.hi !== undefined && r.hi !== null ? `, ${p.rangeLabel ?? "likely range"} ${f(r.lo)} to ${f(r.hi)}` : ""}${r.hollow ? ", extrapolated" : ""}${r.check ? `, independent check ${f(r.check.est)}` : ""}`;
        return (
          <g key={r.id}>
            {(r.strip ?? []).map((v, i) => (
              <circle key={`s${i}`} cx={sx(v)} cy={cy} r={2} style={{ fill: "var(--gray-mark)" }} opacity={0.7} aria-hidden="true" />
            ))}
            {r.lo2 !== undefined && r.lo2 !== null && r.hi2 !== undefined && r.hi2 !== null ? (
              <line x1={sx(r.lo2)} x2={sx(r.hi2)} y1={cy} y2={cy} style={{ stroke: color }} strokeWidth={1.2} aria-hidden="true" />
            ) : null}
            {r.lo !== undefined && r.lo !== null && r.hi !== undefined && r.hi !== null ? (
              <line x1={sx(r.lo)} x2={sx(r.hi)} y1={cy} y2={cy} style={{ stroke: color }} strokeWidth={4} strokeLinecap="round" opacity={0.55} aria-hidden="true" />
            ) : null}
            {r.check ? (
              <g aria-hidden="true">
                {r.check.lo !== undefined && r.check.lo !== null && r.check.hi !== undefined && r.check.hi !== null ? (
                  <line x1={sx(r.check.lo)} x2={sx(r.check.hi)} y1={cy + 7} y2={cy + 7} style={{ stroke: "var(--s2)" }} strokeWidth={1.5} />
                ) : null}
                <path d={`M${sx(r.check.est)},${cy + 2}l5,5l-5,5l-5,-5z`} style={{ fill: "var(--s2)" }} />
              </g>
            ) : null}
            {r.est !== null && Number.isFinite(r.est) ? (
              <circle
                cx={sx(r.est)}
                cy={cy}
                r={5}
                style={{ fill: r.hollow ? "var(--surface)" : color, stroke: color }}
                strokeWidth={2}
                {...markProps(ctx, label, p.onRowClick ? () => p.onRowClick!(r.id) : undefined)}
              />
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}

export function DotRange(props: DotRangeProps) {
  const legend = props.legend ?? (
    <>
      <span>
        <i style={{ background: "var(--s1)", height: 4 }} />
        {props.rangeLabel ?? "Likely range"}
      </span>
      {props.rows.some((r) => r.lo2 !== undefined && r.lo2 !== null) ? (
        <span>
          <i style={{ background: "var(--s1)" }} />
          {props.outerLabel ?? "Outer range"}
        </span>
      ) : null}
      {props.rows.some((r) => r.hollow) ? <span>○ hollow = more than 20% of cells extrapolated</span> : null}
      {props.rows.some((r) => r.check) ? (
        <span>
          <i className="box" style={{ background: "var(--s2)", transform: "rotate(45deg)" }} />
          independent causal check
        </span>
      ) : null}
    </>
  );
  return (
    <ChartFrame {...props} legend={legend} table={dotRangeTable(props)}>
      <Plot {...props} />
    </ChartFrame>
  );
}
