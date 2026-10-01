// Brushable histogram (SPEC §6.4 residual histogram, §6.5 legend histogram). Drag across the
// plot to brush a value range (snapped to bin edges); keyboard: focus a bar, Enter selects
// its bin, Shift+Enter extends the brush to it, Escape clears.
import { useRef, useState, type PointerEvent } from "react";
import { Axis, axisTitle } from "./Axis";
import { ChartFrame, markProps, useChart, useChartWidth, type ChartTable, type FrameOptions } from "./ChartFrame";
import { linearScale, niceDomain } from "./scales";
import { fmtInt, fmtNum } from "../theme/format";
import { rampColor } from "../theme/palette";

export type Bins = { edges: number[]; counts: number[] };

/** Equal-width bins over `domain` (default: finite min–max). Values outside are dropped. */
export function histogram(values: ArrayLike<number>, nBins = 30, domain?: [number, number]): Bins {
  let lo = domain?.[0] ?? Infinity;
  let hi = domain?.[1] ?? -Infinity;
  if (!domain) {
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!(lo <= hi)) return { edges: [0, 1], counts: [0] };
  if (lo === hi) {
    lo -= 0.5;
    hi += 0.5;
  }
  const n = Math.max(1, Math.floor(nBins));
  const w = (hi - lo) / n;
  const counts = new Array<number>(n).fill(0);
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v) || v < lo || v > hi) continue;
    counts[Math.min(n - 1, Math.floor((v - lo) / w))]++;
  }
  const edges = Array.from({ length: n + 1 }, (_, i) => lo + i * w);
  return { edges, counts };
}

/** Snap a value range outward to bin edges. */
export function snapToEdges(edges: number[], a: number, b: number): [number, number] {
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  let i0 = 0;
  while (i0 < edges.length - 2 && edges[i0 + 1] <= lo) i0++;
  let i1 = edges.length - 1;
  while (i1 > 1 && edges[i1 - 1] >= hi) i1--;
  return [edges[i0], edges[Math.max(i1, i0 + 1)]];
}

export type HistogramProps = FrameOptions & {
  values?: ArrayLike<number>;
  bins?: Bins;
  nBins?: number;
  domain?: [number, number];
  xLabel: string;
  unit?: string;
  countLabel?: string;
  decimals?: number;
  brush?: [number, number] | null;
  onBrush?: (range: [number, number] | null) => void;
  refLines?: { value: number; label?: string }[];
  /** Colour bars by a ramp (legend histograms). */
  ramp?: { kind: "seq" | "div"; lo: number; hi: number; center?: number };
  width?: number;
  height?: number;
};

export function histogramTable(bins: Bins, xLabel: string, unit?: string, countLabel = "Cells"): ChartTable {
  return {
    columns: [{ key: "lo", label: `${xLabel} from`, unit }, { key: "hi", label: `${xLabel} to`, unit }, { key: "n", label: countLabel }],
    rows: bins.counts.map((c, i) => [bins.edges[i], bins.edges[i + 1], c]),
  };
}

function Plot(p: HistogramProps & { bins: Bins }) {
  const ctx = useChart();
  const svg = useRef<SVGSVGElement>(null);
  const auto = useChartWidth(640);
  const W = p.width ?? auto;
  const H = p.height ?? 220;
  const M = { top: 10, right: 14, bottom: 42, left: 56 };
  const { edges, counts } = p.bins;
  const sx = linearScale([edges[0], edges[edges.length - 1]], [M.left, W - M.right]);
  const maxC = Math.max(1, ...counts);
  const sy = linearScale(niceDomain(0, maxC), [H - M.bottom, M.top]);
  const [drag, setDrag] = useState<[number, number] | null>(null);
  const anchor = useRef<number | null>(null);
  const d = p.decimals ?? 2;
  const brush = drag ?? p.brush ?? null;

  const toValue = (clientX: number): number | null => {
    const el = svg.current;
    if (!el) return null;
    const r = el.getBoundingClientRect();
    if (!r.width) return null;
    const x = ((clientX - r.left) / r.width) * W;
    return sx.invert(Math.min(W - M.right, Math.max(M.left, x)));
  };
  const onDown = (e: PointerEvent<SVGRectElement>) => {
    if (!p.onBrush) return;
    const v = toValue(e.clientX);
    if (v === null) return;
    anchor.current = v;
    setDrag([v, v]);
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
  };
  const binLabel = (i: number) =>
    `${fmtNum(edges[i], d)} to ${fmtNum(edges[i + 1], d)}${p.unit ? " " + p.unit : ""}: ${fmtInt(counts[i])} ${(p.countLabel ?? "cells").toLowerCase()}`;
  const onMove = (e: PointerEvent<SVGRectElement>) => {
    const v = toValue(e.clientX);
    if (anchor.current === null) {
      if (v === null) return;
      let i = 0;
      while (i < counts.length - 1 && edges[i + 1] <= v) i++;
      ctx.tip(binLabel(i), e.clientX, e.clientY);
      return;
    }
    if (v !== null) setDrag([Math.min(anchor.current, v), Math.max(anchor.current, v)]);
  };
  const onUp = () => {
    if (anchor.current === null || !drag) return;
    anchor.current = null;
    const [a, b] = drag;
    setDrag(null);
    if (Math.abs(sx(b) - sx(a)) < 3) p.onBrush?.(null);
    else p.onBrush?.(snapToEdges(edges, a, b));
  };
  const selectBin = (i: number, extend: boolean) => {
    if (!p.onBrush) return;
    const lo = edges[i];
    const hi = edges[i + 1];
    if (extend && p.brush) p.onBrush([Math.min(p.brush[0], lo), Math.max(p.brush[1], hi)]);
    else p.onBrush([lo, hi]);
  };

  return (
    <svg
      ref={svg}
      className="chart"
      viewBox={`0 0 ${W} ${H}`}
      role="group"
      aria-label={`${p.title}${p.onBrush ? ". Drag to select a range; on a bar, Enter selects its bin, Shift+Enter extends, Escape clears." : ""}`}
      onKeyDown={(e) => {
        if (e.key === "Escape" && p.onBrush && p.brush) {
          e.preventDefault();
          p.onBrush(null);
        }
      }}
    >
      <Axis scale={sy} orient="left" at={M.left} grid={W - M.left - M.right} title={p.countLabel ?? "Cells"} format={(v) => fmtInt(v)} />
      <Axis scale={sx} orient="bottom" at={H - M.bottom} title={axisTitle(p.xLabel, p.unit)} />
      {counts.map((c, i) => {
        const x0 = sx(edges[i]);
        const x1 = sx(edges[i + 1]);
        const inBrush = brush ? edges[i] >= brush[0] - 1e-12 && edges[i + 1] <= brush[1] + 1e-12 : true;
        const mid = (edges[i] + edges[i + 1]) / 2;
        let fill = "var(--s1)";
        if (p.ramp) {
          const { kind, lo, hi, center = 0 } = p.ramp;
          const t =
            kind === "div"
              ? 0.5 + (0.5 * (mid - center)) / (Math.max(Math.abs(lo - center), Math.abs(hi - center)) || 1e-12)
              : (mid - lo) / (hi - lo || 1e-12);
          fill = rampColor(kind, t, ctx.dark);
        }
        const label = `${binLabel(i)}${brush && inBrush ? ", selected" : ""}`;
        return (
          <rect
            key={i}
            x={x0 + 0.5}
            y={sy(c)}
            width={Math.max(0.5, x1 - x0 - 1)}
            height={Math.max(0, H - M.bottom - sy(c))}
            style={{ fill, pointerEvents: p.onBrush ? "none" : undefined }}
            opacity={brush && !inBrush ? 0.3 : 1}
            {...markProps(ctx, label)}
            onKeyDown={(e) => {
              if ((e.key === "Enter" || e.key === " ") && p.onBrush) {
                e.preventDefault();
                selectBin(i, e.shiftKey);
              }
            }}
          />
        );
      })}
      {p.onBrush ? (
        <rect
          x={M.left}
          y={M.top}
          width={W - M.left - M.right}
          height={H - M.top - M.bottom}
          fill="transparent"
          style={{ cursor: "crosshair" }}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerLeave={() => ctx.untip()}
          onPointerCancel={() => {
            anchor.current = null;
            setDrag(null);
          }}
          aria-hidden="true"
        />
      ) : null}
      {(p.refLines ?? []).map((r, i) => (
        <g key={`ref${i}`} aria-hidden="true">
          <line className="zero" x1={sx(r.value)} x2={sx(r.value)} y1={M.top} y2={H - M.bottom} />
          {r.label ? (
            <text className="t-axis" x={sx(r.value) + 4} y={M.top + 10}>
              {r.label}
            </text>
          ) : null}
        </g>
      ))}
      {brush ? <rect className="brush" x={sx(brush[0])} y={M.top} width={Math.max(1, sx(brush[1]) - sx(brush[0]))} height={H - M.top - M.bottom} pointerEvents="none" aria-hidden="true" /> : null}
    </svg>
  );
}

export function Histogram(props: HistogramProps) {
  const bins = props.bins ?? histogram(props.values ?? [], props.nBins ?? 30, props.domain);
  const n = bins.counts.reduce((a, b) => a + b, 0);
  const sel = props.brush ? bins.counts.reduce((a, c, i) => a + (bins.edges[i] >= props.brush![0] - 1e-12 && bins.edges[i + 1] <= props.brush![1] + 1e-12 ? c : 0), 0) : null;
  const legend =
    props.legend ??
    (props.brush ? (
      <span>
        Selected {fmtNum(props.brush[0], props.decimals ?? 2)} to {fmtNum(props.brush[1], props.decimals ?? 2)}
        {props.unit ? " " + props.unit : ""}: {fmtInt(sel)} of {fmtInt(n)} {(props.countLabel ?? "cells").toLowerCase()}
      </span>
    ) : undefined);
  return (
    <ChartFrame {...props} legend={legend} table={histogramTable(bins, props.xLabel, props.unit, props.countLabel)}>
      <Plot {...props} bins={bins} />
    </ChartFrame>
  );
}
