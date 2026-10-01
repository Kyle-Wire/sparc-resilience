// Lines with optional uncertainty ribbons and points (dose–response ±1.96 SE, skill vs block
// size on a log axis, CV curves, sparkline-like series). Hollow points mark extrapolation.
import { Axis, axisTitle } from "./Axis";
import { ChartFrame, markProps, useChart, useChartWidth, type ChartTable, type FrameOptions } from "./ChartFrame";
import { extent, linearScale, logScale, padDomain } from "./scales";
import { fmtNum } from "../theme/format";

export type LineSeries = {
  id: string;
  label: string;
  x: number[];
  y: (number | null)[];
  lo?: (number | null)[];
  hi?: (number | null)[];
  /** CSS colour; defaults to the categorical palette (muted series are grey). */
  color?: string;
  dashed?: boolean;
  points?: boolean;
  /** Per-point hollow marker (extrapolated). */
  hollow?: boolean[];
  muted?: boolean;
  emphasis?: boolean;
};

export type RefLine = { axis: "x" | "y"; value: number; label?: string };

export type LineBandProps = FrameOptions & {
  series: LineSeries[];
  xLabel: string;
  yLabel: string;
  xUnit?: string;
  yUnit?: string;
  xLog?: boolean;
  /** Include these values in the y domain (e.g. 0 or 0.9). */
  yInclude?: number[];
  yDomain?: [number, number];
  xDomain?: [number, number];
  refLines?: RefLine[];
  decimals?: number;
  xDecimals?: number;
  width?: number;
  height?: number;
  onPointClick?: (seriesId: string, index: number) => void;
};

const M = { top: 12, right: 16, bottom: 44, left: 58 };

function segments(xs: number[], ys: (number | null)[], sx: (v: number) => number, sy: (v: number) => number): string {
  let d = "";
  let pen = false;
  for (let i = 0; i < xs.length; i++) {
    const y = ys[i];
    if (y === null || y === undefined || !Number.isFinite(y) || !Number.isFinite(xs[i])) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${sx(xs[i]).toFixed(2)},${sy(y).toFixed(2)}`;
    pen = true;
  }
  return d;
}

function ribbon(xs: number[], lo: (number | null)[], hi: (number | null)[], sx: (v: number) => number, sy: (v: number) => number): string[] {
  const parts: string[] = [];
  let cur: number[] = [];
  const flush = () => {
    if (cur.length > 1) {
      const top = cur.map((i) => `${sx(xs[i]).toFixed(2)},${sy(hi[i] as number).toFixed(2)}`);
      const bot = [...cur].reverse().map((i) => `${sx(xs[i]).toFixed(2)},${sy(lo[i] as number).toFixed(2)}`);
      parts.push(`M${top.join("L")}L${bot.join("L")}Z`);
    }
    cur = [];
  };
  for (let i = 0; i < xs.length; i++) {
    const a = lo[i];
    const b = hi[i];
    if (a === null || b === null || a === undefined || b === undefined || !Number.isFinite(a) || !Number.isFinite(b)) flush();
    else cur.push(i);
  }
  flush();
  return parts;
}

const CAT = ["var(--s1)", "var(--s2)", "var(--s3)"];

/** Series colours: explicit, else grey for muted, else the next categorical colour. */
export function seriesColors(series: { color?: string; muted?: boolean }[]): string[] {
  let k = 0;
  return series.map((s) => s.color ?? (s.muted ? "var(--gray-mark)" : CAT[k++ % 3]));
}

export function lineBandTable(p: Pick<LineBandProps, "series" | "xLabel" | "yLabel" | "xUnit" | "yUnit">): ChartTable {
  const columns: ChartTable["columns"] = [{ key: "series", label: "Series" }, { key: "x", label: p.xLabel, unit: p.xUnit }, { key: "y", label: p.yLabel, unit: p.yUnit }];
  const withBand = p.series.some((s) => s.lo && s.hi);
  if (withBand) columns.push({ key: "lo", label: "Lower", unit: p.yUnit }, { key: "hi", label: "Upper", unit: p.yUnit });
  const rows: ChartTable["rows"] = [];
  for (const s of p.series)
    s.x.forEach((x, i) => {
      const r: ChartTable["rows"][number] = [s.label, x, s.y[i] ?? null];
      if (withBand) r.push(s.lo?.[i] ?? null, s.hi?.[i] ?? null);
      rows.push(r);
    });
  return { columns, rows };
}

function Plot(p: LineBandProps) {
  const ctx = useChart();
  const auto = useChartWidth(640);
  const W = p.width ?? auto;
  const H = p.height ?? 300;
  const xs = p.series.flatMap((s) => s.x);
  const ys = p.series.flatMap((s) => [...s.y, ...(s.lo ?? []), ...(s.hi ?? [])]);
  const refY = (p.refLines ?? []).filter((r) => r.axis === "y").map((r) => r.value);
  const xDom = p.xDomain ?? (extent(xs) ?? [0, 1]);
  const yDom = p.yDomain ?? padDomain(extent([...ys, ...refY]), 0.06, p.yInclude ?? []);
  const sx = p.xLog ? logScale(xDom, [M.left, W - M.right]) : linearScale(xDom, [M.left, W - M.right]);
  const sy = linearScale(yDom, [H - M.bottom, M.top]);
  const colors = seriesColors(p.series);
  const d = p.decimals ?? 2;
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="group" aria-label={p.title}>
      <Axis scale={sy} orient="left" at={M.left} grid={W - M.left - M.right} title={axisTitle(p.yLabel, p.yUnit)} />
      <Axis scale={sx} orient="bottom" at={H - M.bottom} title={axisTitle(p.xLabel, p.xUnit)} />
      {(p.refLines ?? []).map((r, i) =>
        r.axis === "y" ? (
          <g key={`r${i}`}>
            <line className="zero" x1={M.left} x2={W - M.right} y1={sy(r.value)} y2={sy(r.value)} />
            {r.label ? (
              <text className="t-axis" x={W - M.right} y={sy(r.value) - 4} textAnchor="end">
                {r.label}
              </text>
            ) : null}
          </g>
        ) : (
          <g key={`r${i}`}>
            <line className="zero" x1={sx(r.value)} x2={sx(r.value)} y1={M.top} y2={H - M.bottom} />
            {r.label ? (
              <text className="t-axis" x={sx(r.value) + 4} y={M.top + 10}>
                {r.label}
              </text>
            ) : null}
          </g>
        ),
      )}
      {p.series.map((s, si) =>
        s.lo && s.hi ? ribbon(s.x, s.lo, s.hi, sx, sy).map((dd, j) => <path key={`b${si}-${j}`} d={dd} style={{ fill: colors[si], fillOpacity: 0.16 }} aria-hidden="true" />) : null,
      )}
      {p.series.map((s, si) => (
        <path
          key={`l${s.id}`}
          d={segments(s.x, s.y, sx, sy)}
          fill="none"
          style={{ stroke: colors[si] }}
          strokeWidth={s.emphasis ? 2.5 : s.muted ? 1.2 : 1.8}
          strokeDasharray={s.dashed ? "5 4" : undefined}
          aria-hidden="true"
        />
      ))}
      {p.series.map((s, si) =>
        s.points !== false
          ? s.x.map((x, i) => {
              const y = s.y[i];
              if (y === null || y === undefined || !Number.isFinite(y)) return null;
              const hollow = !!s.hollow?.[i];
              const band = s.lo && s.hi && s.lo[i] !== null && s.hi[i] !== null ? ` (${fmtNum(s.lo[i], d)} to ${fmtNum(s.hi[i], d)})` : "";
              const label = `${s.label}: ${p.xLabel} ${fmtNum(x, p.xDecimals ?? 2)}${p.xUnit ? " " + p.xUnit : ""}, ${p.yLabel} ${fmtNum(y, d)}${p.yUnit ? " " + p.yUnit : ""}${band}${hollow ? ", extrapolated" : ""}`;
              return (
                <circle
                  key={`p${s.id}-${i}`}
                  cx={sx(x)}
                  cy={sy(y)}
                  r={s.muted ? 2.5 : 3.5}
                  style={{ fill: hollow ? "var(--surface)" : colors[si], stroke: colors[si] }}
                  strokeWidth={1.5}
                  {...markProps(ctx, label, p.onPointClick ? () => p.onPointClick!(s.id, i) : undefined)}
                />
              );
            })
          : null,
      )}
    </svg>
  );
}

export function LineBand(props: LineBandProps) {
  const table = lineBandTable(props);
  const colors = seriesColors(props.series);
  const legend =
    props.legend ??
    (props.series.length > 1 ? (
      <>
        {props.series.slice(0, 8).map((s, i) => (
          <span key={s.id}>
            <i style={{ background: colors[i] }} />
            {s.label}
          </span>
        ))}
      </>
    ) : undefined);
  return (
    <ChartFrame {...props} legend={legend} table={table}>
      <Plot {...props} />
    </ChartFrame>
  );
}
