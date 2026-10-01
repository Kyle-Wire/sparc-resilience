// Matrix heatmap on the map ramps (sequential or diverging about a centre); missing cells
// are grey with a dot. Used for correlation matrices, fold × model grids and study grids.
import { ChartFrame, markProps, useChart, type ChartTable, type FrameOptions } from "./ChartFrame";
import { extent } from "./scales";
import { fmtNum } from "../theme/format";
import { hexToRgb01, rampColor } from "../theme/palette";

/** Perceived lightness of a hex colour (0 dark … 1 light), to pick label ink. */
function luminance(hex: string): number {
  if (!hex.startsWith("#")) return 1;
  const [r, g, b] = hexToRgb01(hex);
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

export type HeatmapProps = FrameOptions & {
  rows: string[];
  cols: string[];
  /** values[r][c]; null = missing/pending. */
  values: (number | null)[][];
  scale?: "seq" | "div";
  center?: number;
  domain?: [number, number];
  unit?: string;
  valueLabel?: string;
  decimals?: number;
  /** Print values inside cells. */
  labels?: boolean;
  /** Optional per-cell marker text (e.g. "•" for a gate redraw, "!" for an error). */
  marks?: (string | null)[][];
  rowLabel?: string;
  colLabel?: string;
  cellSize?: number;
  onCellClick?: (row: number, col: number) => void;
};

export function heatmapTable(p: Pick<HeatmapProps, "rows" | "cols" | "values" | "unit" | "valueLabel" | "rowLabel" | "colLabel">): ChartTable {
  const rows: ChartTable["rows"] = [];
  p.rows.forEach((r, i) => p.cols.forEach((c, j) => rows.push([r, c, p.values[i]?.[j] ?? null])));
  return { columns: [{ key: "row", label: p.rowLabel ?? "Row" }, { key: "col", label: p.colLabel ?? "Column" }, { key: "v", label: p.valueLabel ?? "Value", unit: p.unit }], rows };
}

/** Position on the ramp in [0, 1] (diverging: centre → 0.5, symmetric). */
export function heatT(v: number, scale: "seq" | "div", lo: number, hi: number, center: number): number {
  if (scale === "div") {
    const span = Math.max(Math.abs(lo - center), Math.abs(hi - center)) || 1e-12;
    return Math.min(1, Math.max(0, 0.5 + (0.5 * (v - center)) / span));
  }
  return hi > lo ? Math.min(1, Math.max(0, (v - lo) / (hi - lo))) : 0.5;
}

function Plot(p: HeatmapProps) {
  const ctx = useChart();
  const cs = p.cellSize ?? (p.cols.length > 20 ? 18 : 34);
  const labW = Math.min(170, 8 + 6.5 * Math.max(4, ...p.rows.map((r) => Math.min(24, r.length))));
  const labH = Math.min(110, 10 + 5.5 * Math.max(4, ...p.cols.map((c) => Math.min(18, c.length))));
  const W = labW + p.cols.length * cs + 12;
  const H = labH + p.rows.length * cs + 8;
  const scale = p.scale ?? "seq";
  const e = p.domain ?? extent(p.values.flat()) ?? [0, 1];
  const center = p.center ?? 0;
  const d = p.decimals ?? 2;
  const rotate = p.cols.some((c) => c.length > 3) || p.cols.length > 12;
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="grid" aria-label={p.title} style={{ maxWidth: W }}>
      {p.cols.map((c, j) => (
        <text
          key={`c${j}`}
          className="t-axis"
          transform={`translate(${labW + j * cs + cs / 2},${labH - 6})${rotate ? " rotate(-50)" : ""}`}
          textAnchor={rotate ? "start" : "middle"}
          aria-hidden="true"
        >
          {c.length > 18 ? c.slice(0, 17) + "…" : c}
        </text>
      ))}
      {p.rows.map((r, i) => (
        <g key={`r${i}`} role="row">
          <text className="t-label" x={labW - 6} y={labH + i * cs + cs / 2} dy="0.32em" textAnchor="end" aria-hidden="true">
            {r.length > 24 ? r.slice(0, 23) + "…" : r}
          </text>
          {p.cols.map((c, j) => {
            const v = p.values[i]?.[j] ?? null;
            const missing = v === null || !Number.isFinite(v);
            const t = missing ? 0 : heatT(v, scale, e[0], e[1], center);
            const fill = missing ? "var(--nodata)" : rampColor(scale, t, ctx.dark);
            const mark = p.marks?.[i]?.[j] ?? null;
            const label = `${r}, ${c}: ${missing ? "no value" : fmtNum(v, d) + (p.unit ? " " + p.unit : "")}${mark ? ` (${mark})` : ""}`;
            const x = labW + j * cs;
            const y = labH + i * cs;
            return (
              <g key={`${i}-${j}`} role="gridcell">
                <rect x={x + 0.5} y={y + 0.5} width={cs - 1} height={cs - 1} rx={2} fill={fill} {...markProps(ctx, label, p.onCellClick ? () => p.onCellClick!(i, j) : undefined)} />
                {missing ? <circle cx={x + cs / 2} cy={y + cs / 2} r={1.6} style={{ fill: "var(--muted)" }} aria-hidden="true" /> : null}
                {p.labels && !missing && cs >= 30 ? (
                  <text x={x + cs / 2} y={y + cs / 2} dy="0.32em" textAnchor="middle" className="t-axis" style={{ fill: luminance(fill) < 0.4 ? "#ffffff" : "#0b0b0b" }} aria-hidden="true">
                    {fmtNum(v, Math.min(d, 2))}
                  </text>
                ) : null}
                {mark ? (
                  <text x={x + cs - 4} y={y + 9} textAnchor="end" className="t-strong" style={{ fontSize: 9 }} aria-hidden="true">
                    {mark}
                  </text>
                ) : null}
              </g>
            );
          })}
        </g>
      ))}
    </svg>
  );
}

export function Heatmap(props: HeatmapProps) {
  const dark = useChart().dark;
  const scale = props.scale ?? "seq";
  const e = props.domain ?? extent(props.values.flat()) ?? [0, 1];
  const center = props.center ?? 0;
  const lo = scale === "div" ? center - Math.max(Math.abs(e[0] - center), Math.abs(e[1] - center)) : e[0];
  const hi = scale === "div" ? center + Math.max(Math.abs(e[0] - center), Math.abs(e[1] - center)) : e[1];
  const legend = props.legend ?? (
    <span className="row" style={{ gap: 6 }}>
      <span className="num">{fmtNum(lo, props.decimals ?? 2)}</span>
      <svg width="120" height="10" aria-hidden="true">
        {Array.from({ length: 24 }, (_, i) => (
          <rect key={i} x={i * 5} width={5.5} height={10} fill={rampColor(scale, i / 23, dark)} />
        ))}
      </svg>
      <span className="num">
        {fmtNum(hi, props.decimals ?? 2)}
        {props.unit ? " " + props.unit : ""}
      </span>
      <span>· grey with a dot = no value</span>
    </span>
  );
  return (
    <ChartFrame {...props} legend={legend} table={heatmapTable(props)}>
      <Plot {...props} />
    </ChartFrame>
  );
}
