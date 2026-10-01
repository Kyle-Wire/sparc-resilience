// Pareto / budget curve (SPEC §6.4 Budget, §7.10 Plans): planned benefit vs budget (labelled
// open-loop), optional closed-loop (realised) points and a selected budget marker.
import { Axis, axisTitle } from "./Axis";
import { ChartFrame, markProps, useChart, useChartWidth, type ChartTable, type FrameOptions } from "./ChartFrame";
import { extent, linearScale, padDomain } from "./scales";
import { fmtNum } from "../theme/format";

export type ParetoPoint = { x: number; y: number; label?: string };

export type ParetoProps = FrameOptions & {
  points: ParetoPoint[];
  realised?: ParetoPoint[];
  selected?: number | null;
  xLabel: string;
  yLabel: string;
  xUnit?: string;
  yUnit?: string;
  curveLabel?: string;
  realisedLabel?: string;
  decimals?: number;
  width?: number;
  height?: number;
  onPointClick?: (index: number) => void;
};

export function paretoTable(p: Pick<ParetoProps, "points" | "realised" | "xLabel" | "yLabel" | "xUnit" | "yUnit" | "curveLabel" | "realisedLabel">): ChartTable {
  const rows: ChartTable["rows"] = p.points.map((q) => [p.curveLabel ?? "planned (open-loop)", q.x, q.y, q.label ?? ""]);
  for (const q of p.realised ?? []) rows.push([p.realisedLabel ?? "realised (closed-loop)", q.x, q.y, q.label ?? ""]);
  return { columns: [{ key: "s", label: "Series" }, { key: "x", label: p.xLabel, unit: p.xUnit }, { key: "y", label: p.yLabel, unit: p.yUnit }, { key: "l", label: "Note" }], rows };
}

function Plot(p: ParetoProps) {
  const ctx = useChart();
  const auto = useChartWidth(640);
  const W = p.width ?? auto;
  const H = p.height ?? 300;
  const M = { top: 14, right: 18, bottom: 44, left: 64 };
  const all = [...p.points, ...(p.realised ?? [])];
  const sx = linearScale(padDomain(extent(all.map((q) => q.x)), 0.03, [0]), [M.left, W - M.right]);
  const sy = linearScale(padDomain(extent(all.map((q) => q.y)), 0.06, [0]), [H - M.bottom, M.top]);
  const sorted = [...p.points].map((q, i) => ({ ...q, i })).sort((a, b) => a.x - b.x);
  const d = p.decimals ?? 1;
  const line = sorted.map((q, k) => `${k ? "L" : "M"}${sx(q.x).toFixed(1)},${sy(q.y).toFixed(1)}`).join("");
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="group" aria-label={p.title}>
      <Axis scale={sy} orient="left" at={M.left} grid={W - M.left - M.right} title={axisTitle(p.yLabel, p.yUnit)} />
      <Axis scale={sx} orient="bottom" at={H - M.bottom} title={axisTitle(p.xLabel, p.xUnit)} />
      <path d={line} fill="none" style={{ stroke: "var(--s1)" }} strokeWidth={2} aria-hidden="true" />
      {sorted.map((q) => (
        <circle
          key={`p${q.i}`}
          cx={sx(q.x)}
          cy={sy(q.y)}
          r={q.i === p.selected ? 6 : 3.5}
          style={{ fill: q.i === p.selected ? "var(--surface)" : "var(--s1)", stroke: "var(--s1)" }}
          strokeWidth={q.i === p.selected ? 2.5 : 1}
          {...markProps(ctx, `${p.curveLabel ?? "Planned (open-loop)"}: ${p.xLabel} ${fmtNum(q.x, d)}${p.xUnit ? " " + p.xUnit : ""}, ${p.yLabel} ${fmtNum(q.y, d)}${p.yUnit ? " " + p.yUnit : ""}${q.label ? ` (${q.label})` : ""}${q.i === p.selected ? ", selected" : ""}`, p.onPointClick ? () => p.onPointClick!(q.i) : undefined)}
        />
      ))}
      {(p.realised ?? []).map((q, i) => (
        <rect
          key={`r${i}`}
          x={sx(q.x) - 4}
          y={sy(q.y) - 4}
          width={8}
          height={8}
          style={{ fill: "var(--s2)" }}
          {...markProps(ctx, `${p.realisedLabel ?? "Realised (closed-loop)"}: ${p.xLabel} ${fmtNum(q.x, d)}, ${p.yLabel} ${fmtNum(q.y, d)}${p.yUnit ? " " + p.yUnit : ""}${q.label ? ` (${q.label})` : ""}`)}
        />
      ))}
    </svg>
  );
}

export function Pareto(props: ParetoProps) {
  const legend = props.legend ?? (
    <>
      <span>
        <i style={{ background: "var(--s1)" }} />
        {props.curveLabel ?? "planned (open-loop)"}
      </span>
      {props.realised?.length ? (
        <span>
          <i className="box" style={{ background: "var(--s2)" }} />
          {props.realisedLabel ?? "realised (closed-loop)"}
        </span>
      ) : null}
    </>
  );
  return (
    <ChartFrame {...props} legend={legend} table={paretoTable(props)}>
      <Plot {...props} />
    </ChartFrame>
  );
}
