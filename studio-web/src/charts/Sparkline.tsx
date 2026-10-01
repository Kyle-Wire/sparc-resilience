// Small inline line (resources, epoch loss, metric trends). Framed by default like every kit
// chart; `bare` renders just the SVG for table cells and tiles.
import { ChartFrame, markProps, useChart, useChartWidth, type ChartTable, type FrameOptions } from "./ChartFrame";
import { extent, linearScale } from "./scales";
import { fmtNum } from "../theme/format";

export type SparklineProps = FrameOptions & {
  values: (number | null)[];
  x?: number[];
  unit?: string;
  decimals?: number;
  area?: boolean;
  /** Horizontal reference (e.g. 80% of RAM). */
  ref?: number;
  width?: number;
  height?: number;
};

export function sparklineTable(p: Pick<SparklineProps, "values" | "x" | "unit" | "title">): ChartTable {
  return { columns: [{ key: "x", label: p.x ? "x" : "Index" }, { key: "v", label: p.title, unit: p.unit }], rows: p.values.map((v, i) => [p.x?.[i] ?? i, v]) };
}

function Plot(p: SparklineProps) {
  const ctx = useChart();
  const auto = useChartWidth(140);
  const W = p.width ?? (p.bare ? 140 : auto);
  const H = p.height ?? (p.bare ? 32 : 64);
  const xs = p.x ?? p.values.map((_, i) => i);
  const e = extent([...p.values, ...(p.ref !== undefined ? [p.ref] : [])]) ?? [0, 1];
  const xe = extent(xs) ?? [0, 1];
  const sx = linearScale(xe, [2, W - 4]);
  const sy = linearScale(e[0] === e[1] ? [e[0] - 1, e[1] + 1] : e, [H - 3, 3]);
  let d = "";
  let pen = false;
  let lastI = -1;
  p.values.forEach((v, i) => {
    if (v === null || !Number.isFinite(v)) {
      pen = false;
      return;
    }
    d += `${pen ? "L" : "M"}${sx(xs[i]).toFixed(1)},${sy(v).toFixed(1)}`;
    pen = true;
    lastI = i;
  });
  const last = lastI >= 0 ? (p.values[lastI] as number) : null;
  const label = `${p.title}: ${last === null ? "no data" : `latest ${fmtNum(last, p.decimals ?? 2)}${p.unit ? " " + p.unit : ""}`}, range ${fmtNum(e[0], p.decimals ?? 2)} to ${fmtNum(e[1], p.decimals ?? 2)}`;
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} style={p.bare ? { width: W, height: H } : undefined}>
      {p.area && d ? <path d={`${d}L${sx(xs[lastI]).toFixed(1)},${H - 2}L${sx(xs[0]).toFixed(1)},${H - 2}Z`} style={{ fill: "var(--s1)", fillOpacity: 0.15 }} aria-hidden="true" /> : null}
      {p.ref !== undefined ? <line className="zero" x1={0} x2={W} y1={sy(p.ref)} y2={sy(p.ref)} aria-hidden="true" /> : null}
      <path d={d} fill="none" style={{ stroke: "var(--s1)" }} strokeWidth={1.5} aria-hidden="true" />
      {last !== null ? <circle cx={sx(xs[lastI])} cy={sy(last)} r={2.2} style={{ fill: "var(--s1)" }} {...markProps(ctx, label)} /> : null}
    </svg>
  );
}

export function Sparkline(props: SparklineProps) {
  return (
    <ChartFrame {...props} table={sparklineTable(props)}>
      <Plot {...props} />
    </ChartFrame>
  );
}
