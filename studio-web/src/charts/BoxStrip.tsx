// Box plots per group (whiskers p10–p90 or min–max, box q1–q3, median line) with an optional
// strip of individual points. Residuals by zone/fold, CATE quantiles, Breakdown tool.
import { Axis, BandAxis, axisTitle } from "./Axis";
import { ChartFrame, markProps, useChart, useChartWidth, type ChartTable, type FrameOptions } from "./ChartFrame";
import { bandScale, extent, linearScale, niceDomain } from "./scales";
import { fmtInt, fmtNum } from "../theme/format";

export type BoxGroup = {
  label: string;
  /** [low whisker, q1, median, q3, high whisker]. */
  q: [number, number, number, number, number] | null;
  n?: number | null;
  mean?: number | null;
  points?: number[];
  muted?: boolean;
};

export type BoxStripProps = FrameOptions & {
  groups: BoxGroup[];
  valueLabel: string;
  unit?: string;
  groupLabel?: string;
  whiskerLabel?: string;
  decimals?: number;
  zeroLine?: boolean;
  width?: number;
  height?: number;
  onGroupClick?: (label: string) => void;
};

export function boxTable(p: Pick<BoxStripProps, "groups" | "valueLabel" | "unit" | "groupLabel" | "whiskerLabel">): ChartTable {
  const w = p.whiskerLabel ?? "whisker";
  return {
    columns: [
      { key: "g", label: p.groupLabel ?? "Group" },
      { key: "n", label: "n" },
      { key: "lo", label: `Low ${w}`, unit: p.unit },
      { key: "q1", label: "Q1", unit: p.unit },
      { key: "med", label: "Median", unit: p.unit },
      { key: "q3", label: "Q3", unit: p.unit },
      { key: "hi", label: `High ${w}`, unit: p.unit },
      { key: "mean", label: "Mean", unit: p.unit },
    ],
    rows: p.groups.map((g) => [g.label, g.n ?? null, ...(g.q ?? [null, null, null, null, null]), g.mean ?? null]),
  };
}

function Plot(p: BoxStripProps) {
  const ctx = useChart();
  const auto = useChartWidth(640);
  const W = p.width ?? auto;
  const H = p.height ?? 280;
  const M = { top: 12, right: 16, bottom: p.groups.length > 8 ? 70 : 46, left: 58 };
  const vals = p.groups.flatMap((g) => [...(g.q ?? []), ...(g.points ?? []), g.mean ?? null]);
  const e = extent(vals) ?? [0, 1];
  const dom = niceDomain(p.zeroLine ? Math.min(0, e[0]) : e[0], p.zeroLine ? Math.max(0, e[1]) : e[1]);
  const sy = linearScale(dom, [H - M.bottom, M.top]);
  const band = bandScale(p.groups.map((g) => g.label), [M.left, W - M.right], 0.35, 0.1);
  const d = p.decimals ?? 2;
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="group" aria-label={p.title}>
      <Axis scale={sy} orient="left" at={M.left} grid={W - M.left - M.right} title={axisTitle(p.valueLabel, p.unit)} />
      <BandAxis scale={band} orient="bottom" at={H - M.bottom} title={p.groupLabel} />
      {p.zeroLine && dom[0] < 0 && dom[1] > 0 ? <line className="zero" x1={M.left} x2={W - M.right} y1={sy(0)} y2={sy(0)} /> : null}
      {p.groups.map((g) => {
        const x = band(g.label);
        const w = band.bandwidth;
        const cx = x + w / 2;
        const color = g.muted ? "var(--gray-mark)" : "var(--s1)";
        const pts = g.points ?? [];
        const label = g.q
          ? `${g.label}: median ${fmtNum(g.q[2], d)}, quartiles ${fmtNum(g.q[1], d)} to ${fmtNum(g.q[3], d)}, ${p.whiskerLabel ?? "whiskers"} ${fmtNum(g.q[0], d)} to ${fmtNum(g.q[4], d)}${p.unit ? " " + p.unit : ""}${g.n !== undefined && g.n !== null ? `, n ${fmtInt(g.n)}` : ""}`
          : `${g.label}: no data`;
        return (
          <g key={g.label}>
            {pts.slice(0, 400).map((v, i) => (
              <circle key={i} cx={cx + (((i * 7919) % 100) / 100 - 0.5) * w * 0.8} cy={sy(v)} r={1.6} style={{ fill: "var(--gray-mark)" }} opacity={0.6} aria-hidden="true" />
            ))}
            {g.q ? (
              <g {...markProps(ctx, label, p.onGroupClick ? () => p.onGroupClick!(g.label) : undefined)}>
                <line x1={cx} x2={cx} y1={sy(g.q[0])} y2={sy(g.q[4])} style={{ stroke: color }} strokeWidth={1.2} />
                <line x1={cx - w * 0.2} x2={cx + w * 0.2} y1={sy(g.q[0])} y2={sy(g.q[0])} style={{ stroke: color }} />
                <line x1={cx - w * 0.2} x2={cx + w * 0.2} y1={sy(g.q[4])} y2={sy(g.q[4])} style={{ stroke: color }} />
                <rect x={x + w * 0.1} width={w * 0.8} y={sy(g.q[3])} height={Math.max(1, sy(g.q[1]) - sy(g.q[3]))} rx={2} style={{ fill: color, fillOpacity: 0.22, stroke: color }} />
                <line x1={x + w * 0.1} x2={x + w * 0.9} y1={sy(g.q[2])} y2={sy(g.q[2])} style={{ stroke: "var(--ink)" }} strokeWidth={2} />
                {g.mean !== undefined && g.mean !== null ? <circle cx={cx} cy={sy(g.mean)} r={2.6} style={{ fill: "var(--surface)", stroke: "var(--ink)" }} /> : null}
              </g>
            ) : (
              <text className="t-axis" x={cx} y={(M.top + H - M.bottom) / 2} textAnchor="middle" {...markProps(ctx, label)}>
                no data
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

export function BoxStrip(props: BoxStripProps) {
  const legend = props.legend ?? (
    <span>
      box = quartiles · line = median · whiskers = {props.whiskerLabel ?? "10th–90th percentile"}
      {props.groups.some((g) => g.mean !== undefined && g.mean !== null) ? " · ○ = mean" : ""}
    </span>
  );
  return (
    <ChartFrame {...props} legend={legend} table={boxTable(props)}>
      <Plot {...props} />
    </ChartFrame>
  );
}
