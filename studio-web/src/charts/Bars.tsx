// Bars: horizontal or vertical; single, grouped, stacked or 100%-stacked; optional
// whiskers (p10–p90) per bar. Used for range bars, exposure, stacker weights, timings.
import { Axis, BandAxis, axisTitle } from "./Axis";
import { ChartFrame, markProps, useChart, useChartWidth, type ChartTable, type FrameOptions } from "./ChartFrame";
import { seriesColors } from "./LineBand";
import { bandScale, extent, linearScale, niceDomain } from "./scales";
import { fmtNum } from "../theme/format";

export type BarSeries = {
  id: string;
  label: string;
  values: (number | null)[];
  lo?: (number | null)[];
  hi?: (number | null)[];
  color?: string;
  muted?: boolean;
};

export type BarsProps = FrameOptions & {
  categories: string[];
  series: BarSeries[];
  orientation?: "h" | "v";
  mode?: "grouped" | "stacked" | "percent";
  valueLabel: string;
  unit?: string;
  categoryLabel?: string;
  decimals?: number;
  /** Categories drawn with the accent outline (e.g. the winner, the main design). */
  highlight?: string[];
  width?: number;
  height?: number;
  domain?: [number, number];
  onBarClick?: (category: string, seriesId: string) => void;
};

export function barsTable(p: Pick<BarsProps, "categories" | "series" | "valueLabel" | "unit" | "categoryLabel">): ChartTable {
  const hasRange = p.series.some((s) => s.lo && s.hi);
  const columns: ChartTable["columns"] = [{ key: "cat", label: p.categoryLabel ?? "Category" }, { key: "series", label: "Series" }, { key: "v", label: p.valueLabel, unit: p.unit }];
  if (hasRange) columns.push({ key: "lo", label: "Low", unit: p.unit }, { key: "hi", label: "High", unit: p.unit });
  const rows: ChartTable["rows"] = [];
  p.categories.forEach((c, ci) =>
    p.series.forEach((s) => {
      const r: ChartTable["rows"][number] = [c, s.label, s.values[ci] ?? null];
      if (hasRange) r.push(s.lo?.[ci] ?? null, s.hi?.[ci] ?? null);
      rows.push(r);
    }),
  );
  return { columns, rows };
}

type Rect = { cat: string; sid: string; label: string; v: number; a: number; b: number; off: number; size: number; color: string; lo?: number | null; hi?: number | null };

function Plot(p: BarsProps) {
  const ctx = useChart();
  const horiz = (p.orientation ?? "h") === "h";
  const mode = p.mode ?? "grouped";
  const n = p.categories.length;
  const auto = useChartWidth(640);
  const W = p.width ?? auto;
  const M = horiz ? { top: 8, right: 20, bottom: 44, left: 140 } : { top: 12, right: 16, bottom: n > 8 ? 70 : 48, left: 58 };
  const H = p.height ?? (horiz ? Math.max(120, M.top + M.bottom + n * (mode === "grouped" ? 10 + 14 * p.series.length : 26)) : 300);
  const colors = seriesColors(p.series);
  const d = p.decimals ?? 2;

  // value domain
  const totals = p.categories.map((_, ci) => {
    let pos = 0;
    let neg = 0;
    for (const s of p.series) {
      const v = s.values[ci];
      if (v === null || v === undefined || !Number.isFinite(v)) continue;
      if (v >= 0) pos += v;
      else neg += v;
    }
    return [neg, pos] as const;
  });
  let dom: [number, number];
  if (p.domain) dom = p.domain;
  else if (mode === "percent") dom = [0, 100];
  else if (mode === "stacked") dom = niceDomain(Math.min(0, ...totals.map((t) => t[0])), Math.max(0, ...totals.map((t) => t[1])));
  else {
    const e = extent(p.series.flatMap((s) => [...s.values, ...(s.lo ?? []), ...(s.hi ?? [])])) ?? [0, 1];
    dom = niceDomain(Math.min(0, e[0]), Math.max(0, e[1]));
  }
  const band = bandScale(p.categories, horiz ? [M.top, H - M.bottom] : [M.left, W - M.right], 0.25, 0.1);
  const vs = horiz ? linearScale(dom, [M.left, W - M.right]) : linearScale(dom, [H - M.bottom, M.top]);

  const rects: Rect[] = [];
  p.categories.forEach((cat, ci) => {
    if (mode === "grouped") {
      const k = p.series.length;
      const size = band.bandwidth / k;
      p.series.forEach((s, si) => {
        const v = s.values[ci];
        if (v === null || v === undefined || !Number.isFinite(v)) return;
        rects.push({ cat, sid: s.id, label: s.label, v, a: Math.min(0, v), b: Math.max(0, v), off: band(cat) + si * size, size: size * 0.92, color: colors[si], lo: s.lo?.[ci], hi: s.hi?.[ci] });
      });
    } else {
      const total = p.series.reduce((acc, s) => acc + Math.abs(s.values[ci] ?? 0), 0) || 1;
      let pos = 0;
      let neg = 0;
      p.series.forEach((s, si) => {
        let v = s.values[ci];
        if (v === null || v === undefined || !Number.isFinite(v)) return;
        const raw = v;
        if (mode === "percent") v = (Math.abs(v) / total) * 100;
        const a = v >= 0 ? pos : neg + v;
        const b = v >= 0 ? pos + v : neg;
        if (v >= 0) pos += v;
        else neg += v;
        rects.push({ cat, sid: s.id, label: s.label, v: mode === "percent" ? raw : v, a, b, off: band(cat), size: band.bandwidth, color: colors[si] });
      });
    }
  });

  const unit = mode === "percent" ? "%" : p.unit;
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="group" aria-label={p.title}>
      {horiz ? (
        <>
          <Axis scale={vs} orient="bottom" at={H - M.bottom} grid={H - M.top - M.bottom} title={axisTitle(mode === "percent" ? `Share of ${p.valueLabel}` : p.valueLabel, unit)} />
          <BandAxis scale={band} orient="left" at={M.left} />
          {dom[0] < 0 && dom[1] > 0 ? <line className="zero" x1={vs(0)} x2={vs(0)} y1={M.top} y2={H - M.bottom} /> : null}
        </>
      ) : (
        <>
          <Axis scale={vs} orient="left" at={M.left} grid={W - M.left - M.right} title={axisTitle(mode === "percent" ? `Share of ${p.valueLabel}` : p.valueLabel, unit)} />
          <BandAxis scale={band} orient="bottom" at={H - M.bottom} title={p.categoryLabel} />
          {dom[0] < 0 && dom[1] > 0 ? <line className="zero" x1={M.left} x2={W - M.right} y1={vs(0)} y2={vs(0)} /> : null}
        </>
      )}
      {rects.map((r) => {
        const hl = p.highlight?.includes(r.cat);
        const label = `${r.cat}${p.series.length > 1 ? `, ${r.label}` : ""}: ${fmtNum(r.v, d)}${p.unit ? " " + p.unit : ""}${r.lo !== undefined && r.lo !== null && r.hi !== undefined && r.hi !== null ? ` (${fmtNum(r.lo, d)} to ${fmtNum(r.hi, d)})` : ""}`;
        const geom = horiz
          ? { x: vs(r.a), y: r.off, width: Math.max(0.5, vs(r.b) - vs(r.a)), height: r.size }
          : { x: r.off, y: vs(r.b), width: r.size, height: Math.max(0.5, vs(r.a) - vs(r.b)) };
        return (
          <g key={`${r.cat}-${r.sid}`}>
            <rect
              {...geom}
              rx={2}
              style={{ fill: r.color, stroke: hl ? "var(--ink)" : undefined }}
              strokeWidth={hl ? 1.5 : 0}
              {...markProps(ctx, label, p.onBarClick ? () => p.onBarClick!(r.cat, r.sid) : undefined)}
            />
            {r.lo !== undefined && r.lo !== null && r.hi !== undefined && r.hi !== null ? (
              horiz ? (
                <line x1={vs(r.lo)} x2={vs(r.hi)} y1={r.off + r.size / 2} y2={r.off + r.size / 2} style={{ stroke: "var(--ink-2)" }} strokeWidth={1.2} aria-hidden="true" />
              ) : (
                <line y1={vs(r.lo)} y2={vs(r.hi)} x1={r.off + r.size / 2} x2={r.off + r.size / 2} style={{ stroke: "var(--ink-2)" }} strokeWidth={1.2} aria-hidden="true" />
              )
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}

export function Bars(props: BarsProps) {
  const colors = seriesColors(props.series);
  const legend =
    props.legend ??
    (props.series.length > 1 ? (
      <>
        {props.series.map((s, i) => (
          <span key={s.id}>
            <i className="box" style={{ background: colors[i] }} />
            {s.label}
          </span>
        ))}
      </>
    ) : undefined);
  return (
    <ChartFrame {...props} legend={legend} table={barsTable(props)}>
      <Plot {...props} />
    </ChartFrame>
  );
}
