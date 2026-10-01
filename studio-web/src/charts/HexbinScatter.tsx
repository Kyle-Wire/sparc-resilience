// Binned scatter on a canvas (SPEC §6.4 obs-vs-pred, §6.5 Relationships): counts per 2-D bin
// on the sequential ramp (log scale), the current selection overlaid in orange, optional 1:1
// line and binned-mean line; drag a rectangle to brush. Axes and overlays are SVG; the
// raster is embedded when the chart is exported.
import { useEffect, useRef, useState, type PointerEvent } from "react";
import { Axis, axisTitle } from "./Axis";
import { ChartFrame, useChart, useChartWidth, type ChartTable, type FrameOptions } from "./ChartFrame";
import { extent, linearScale, padDomain } from "./scales";
import { fmtInt, fmtNum } from "../theme/format";
import { getLuts } from "../theme/palette";

/** 2-D bins; counts[i][j] is x bin i, y bin j (numpy.histogram2d order). */
export type Bins2D = { x_edges: number[]; y_edges: number[]; counts: number[][]; sel_counts?: number[][] | null };

/** Bin raw points into an nx × ny grid over their extent (or the given domains). */
export function bin2d(x: ArrayLike<number>, y: ArrayLike<number>, nx = 60, ny = 60, xDom?: [number, number], yDom?: [number, number], sel?: ArrayLike<number>): Bins2D {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < x.length; i++) if (Number.isFinite(x[i]) && Number.isFinite(y[i])) (xs.push(x[i]), ys.push(y[i]));
  const [x0, x1] = xDom ?? extent(xs) ?? [0, 1];
  const [y0, y1] = yDom ?? extent(ys) ?? [0, 1];
  const wx = (x1 - x0 || 1) / nx;
  const wy = (y1 - y0 || 1) / ny;
  const counts = Array.from({ length: nx }, () => new Array<number>(ny).fill(0));
  const selCounts = sel ? Array.from({ length: nx }, () => new Array<number>(ny).fill(0)) : null;
  for (let i = 0; i < x.length; i++) {
    const a = x[i];
    const b = y[i];
    if (!Number.isFinite(a) || !Number.isFinite(b) || a < x0 || a > x1 || b < y0 || b > y1) continue;
    const ix = Math.min(nx - 1, Math.floor((a - x0) / wx));
    const iy = Math.min(ny - 1, Math.floor((b - y0) / wy));
    counts[ix][iy]++;
    if (selCounts && sel?.[i]) selCounts[ix][iy]++;
  }
  return {
    x_edges: Array.from({ length: nx + 1 }, (_, i) => x0 + i * wx),
    y_edges: Array.from({ length: ny + 1 }, (_, i) => y0 + i * wy),
    counts,
    sel_counts: selCounts,
  };
}

export type HexbinScatterProps = FrameOptions & {
  bins?: Bins2D;
  points?: { x: ArrayLike<number>; y: ArrayLike<number>; selected?: ArrayLike<number> };
  nBins?: number;
  xLabel: string;
  yLabel: string;
  xUnit?: string;
  yUnit?: string;
  /** Draw y = x (obs vs pred). */
  diagonal?: boolean;
  meanLine?: { x: number; y: number }[];
  spearman?: number | null;
  decimals?: number;
  onBrush?: (box: { x: [number, number]; y: [number, number] } | null) => void;
  width?: number;
  height?: number;
};

export function hexbinTable(b: Bins2D, p: Pick<HexbinScatterProps, "xLabel" | "yLabel" | "xUnit" | "yUnit">, cap = 5000): ChartTable {
  const rows: ChartTable["rows"] = [];
  const hasSel = !!b.sel_counts;
  for (let i = 0; i < b.counts.length && rows.length < cap; i++)
    for (let j = 0; j < b.counts[i].length && rows.length < cap; j++) {
      const c = b.counts[i][j];
      if (!c) continue;
      const r: ChartTable["rows"][number] = [b.x_edges[i], b.x_edges[i + 1], b.y_edges[j], b.y_edges[j + 1], c];
      if (hasSel) r.push(b.sel_counts![i][j]);
      rows.push(r);
    }
  const columns: ChartTable["columns"] = [
    { key: "x0", label: `${p.xLabel} from`, unit: p.xUnit },
    { key: "x1", label: `${p.xLabel} to`, unit: p.xUnit },
    { key: "y0", label: `${p.yLabel} from`, unit: p.yUnit },
    { key: "y1", label: `${p.yLabel} to`, unit: p.yUnit },
    { key: "n", label: "Cells" },
  ];
  if (hasSel) columns.push({ key: "sel", label: "Selected cells" });
  return { columns, rows };
}

function Plot(p: HexbinScatterProps & { b: Bins2D }) {
  const ctx = useChart();
  const auto = useChartWidth(520);
  const W = p.width ?? auto;
  // Close to square, so a 1:1 line reads as 45°.
  const H = p.height ?? Math.round(Math.min(460, Math.max(260, W * 0.8)));
  const M = { top: 10, right: 14, bottom: 44, left: 58 };
  const b = p.b;
  const xDom: [number, number] = [b.x_edges[0], b.x_edges[b.x_edges.length - 1]];
  const yDom: [number, number] = [b.y_edges[0], b.y_edges[b.y_edges.length - 1]];
  const sx = linearScale(xDom, [M.left, W - M.right]);
  const sy = linearScale(yDom, [H - M.bottom, M.top]);
  const pw = W - M.left - M.right;
  const ph = H - M.top - M.bottom;
  const canvas = useRef<HTMLCanvasElement>(null);
  const svg = useRef<SVGSVGElement>(null);
  const [box, setBox] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const start = useRef<{ x: number; y: number } | null>(null);
  const d = p.decimals ?? 2;

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const g = c.getContext("2d");
    if (!g) return; // no canvas support (tests): the table view still carries the data
    const nx = b.counts.length;
    const ny = nx ? b.counts[0].length : 0;
    c.width = Math.max(1, nx);
    c.height = Math.max(1, ny);
    g.clearRect(0, 0, c.width, c.height);
    let max = 1;
    for (const col of b.counts) for (const v of col) if (v > max) max = v;
    const lut = getLuts(ctx.dark).seq;
    const img = g.createImageData(c.width, c.height);
    const sel = ctx.colors.select;
    const sr = parseInt(sel.slice(1, 3), 16);
    const sg = parseInt(sel.slice(3, 5), 16);
    const sb = parseInt(sel.slice(5, 7), 16);
    for (let i = 0; i < nx; i++)
      for (let j = 0; j < ny; j++) {
        const v = b.counts[i][j];
        if (!v) continue;
        const k = ((ny - 1 - j) * nx + i) * 4; // y grows upward
        const s = b.sel_counts?.[i]?.[j] ?? 0;
        if (s > 0) {
          img.data[k] = sr;
          img.data[k + 1] = sg;
          img.data[k + 2] = sb;
          img.data[k + 3] = 140 + Math.round((115 * s) / v);
        } else {
          const t = Math.round((Math.log1p(v) / Math.log1p(max)) * 255);
          img.data[k] = lut[t * 3];
          img.data[k + 1] = lut[t * 3 + 1];
          img.data[k + 2] = lut[t * 3 + 2];
          img.data[k + 3] = b.sel_counts ? 110 : 255;
        }
      }
    g.putImageData(img, 0, 0);
  }, [b, ctx.dark, ctx.colors.select]);

  const toData = (e: PointerEvent): { x: number; y: number } | null => {
    const r = svg.current?.getBoundingClientRect();
    if (!r || !r.width) return null;
    const px = ((e.clientX - r.left) / r.width) * W;
    const py = ((e.clientY - r.top) / r.height) * H;
    return { x: sx.invert(Math.min(W - M.right, Math.max(M.left, px))), y: sy.invert(Math.min(H - M.bottom, Math.max(M.top, py))) };
  };
  const binAt = (pt: { x: number; y: number }) => {
    let i = 0;
    while (i < b.x_edges.length - 2 && b.x_edges[i + 1] <= pt.x) i++;
    let j = 0;
    while (j < b.y_edges.length - 2 && b.y_edges[j + 1] <= pt.y) j++;
    return { i, j };
  };

  const diag = p.diagonal ? [Math.max(xDom[0], yDom[0]), Math.min(xDom[1], yDom[1])] : null;
  const mean = p.meanLine?.filter((m) => Number.isFinite(m.x) && Number.isFinite(m.y)) ?? [];
  return (
    <div style={{ position: "relative" }}>
      <canvas
        ref={canvas}
        data-chart-layer="bins"
        data-x={M.left}
        data-y={M.top}
        data-w={pw}
        data-h={ph}
        aria-hidden="true"
        style={{ position: "absolute", left: `${(M.left / W) * 100}%`, top: `${(M.top / H) * 100}%`, width: `${(pw / W) * 100}%`, height: `${(ph / H) * 100}%`, imageRendering: "pixelated" }}
      />
      <svg
        ref={svg}
        className="chart"
        viewBox={`0 0 ${W} ${H}`}
        role="group"
        aria-label={`${p.title}${p.spearman !== undefined && p.spearman !== null ? `, Spearman rho ${fmtNum(p.spearman, 2)}` : ""}`}
        style={{ position: "relative" }}
      >
        <Axis scale={sy} orient="left" at={M.left} title={axisTitle(p.yLabel, p.yUnit)} />
        <Axis scale={sx} orient="bottom" at={H - M.bottom} title={axisTitle(p.xLabel, p.xUnit)} />
        {diag && diag[1] > diag[0] ? <line className="zero" x1={sx(diag[0])} y1={sy(diag[0])} x2={sx(diag[1])} y2={sy(diag[1])} aria-hidden="true" /> : null}
        {mean.length > 1 ? (
          <path d={mean.map((m, i) => `${i ? "L" : "M"}${sx(m.x).toFixed(1)},${sy(m.y).toFixed(1)}`).join("")} fill="none" style={{ stroke: "var(--ink)" }} strokeWidth={1.8} aria-hidden="true" />
        ) : null}
        {p.spearman !== undefined && p.spearman !== null ? (
          <text className="t-strong" x={W - M.right - 4} y={M.top + 14} textAnchor="end">
            ρ = {fmtNum(p.spearman, 2)}
          </text>
        ) : null}
        <rect
          x={M.left}
          y={M.top}
          width={pw}
          height={ph}
          fill="transparent"
          tabIndex={0}
          role="img"
          aria-label={`${fmtInt(b.counts.flat().reduce((a, c) => a + c, 0))} cells binned; ${p.onBrush ? "drag to select a rectangle" : "hover for bin counts"}`}
          style={{ cursor: p.onBrush ? "crosshair" : "default" }}
          onPointerDown={(e) => {
            if (!p.onBrush) return;
            const pt = toData(e);
            if (!pt) return;
            start.current = pt;
            setBox({ x0: pt.x, y0: pt.y, x1: pt.x, y1: pt.y });
            (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
          }}
          onPointerMove={(e) => {
            const pt = toData(e);
            if (!pt) return;
            if (start.current) {
              setBox({ x0: start.current.x, y0: start.current.y, x1: pt.x, y1: pt.y });
              return;
            }
            const { i, j } = binAt(pt);
            const c = b.counts[i]?.[j] ?? 0;
            ctx.tip(`${p.xLabel} ${fmtNum(b.x_edges[i], d)}–${fmtNum(b.x_edges[i + 1], d)}, ${p.yLabel} ${fmtNum(b.y_edges[j], d)}–${fmtNum(b.y_edges[j + 1], d)}: ${fmtInt(c)} cells`, e.clientX, e.clientY);
          }}
          onPointerLeave={() => ctx.untip()}
          onPointerUp={() => {
            if (!start.current || !box) return;
            start.current = null;
            const bx: [number, number] = [Math.min(box.x0, box.x1), Math.max(box.x0, box.x1)];
            const by: [number, number] = [Math.min(box.y0, box.y1), Math.max(box.y0, box.y1)];
            setBox(null);
            if (Math.abs(sx(bx[1]) - sx(bx[0])) < 3 && Math.abs(sy(by[0]) - sy(by[1])) < 3) p.onBrush?.(null);
            else p.onBrush?.({ x: bx, y: by });
          }}
          onKeyDown={(e) => {
            if (e.key === "Escape" && p.onBrush) p.onBrush(null);
          }}
        />
        {box ? (
          <rect className="brush" x={sx(Math.min(box.x0, box.x1))} y={sy(Math.max(box.y0, box.y1))} width={Math.abs(sx(box.x1) - sx(box.x0))} height={Math.abs(sy(box.y0) - sy(box.y1))} pointerEvents="none" aria-hidden="true" />
        ) : null}
      </svg>
    </div>
  );
}

export function HexbinScatter(props: HexbinScatterProps) {
  const n = props.nBins ?? 60;
  const b =
    props.bins ??
    (props.points
      ? bin2d(props.points.x, props.points.y, n, n, padDomain(extent(Array.from(props.points.x)), 0.02), padDomain(extent(Array.from(props.points.y)), 0.02), props.points.selected)
      : { x_edges: [0, 1], y_edges: [0, 1], counts: [[0]] });
  const legend =
    props.legend ??
    (
      <>
        <span>colour = cells per bin (log scale)</span>
        {b.sel_counts ? (
          <span>
            <i className="box" style={{ background: "var(--s2)" }} />
            selected cells
          </span>
        ) : null}
        {props.diagonal ? <span>dashed = 1:1</span> : null}
        {props.meanLine?.length ? <span>line = binned mean</span> : null}
      </>
    );
  return (
    <ChartFrame {...props} legend={legend} table={hexbinTable(b, props)}>
      <Plot {...props} b={b} />
    </ChartFrame>
  );
}
