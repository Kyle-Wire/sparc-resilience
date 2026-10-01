// Span-tree Gantt (SPEC §5.10): run › stage › task › subtask rows, collapsible, virtualised
// (only rows in view are drawn), running bars grow to "now", warning ticks and checkpoint
// flags on the time axis, and hatched "system sleep?" bands for heartbeat gaps. The time
// axis toggles between relative and wall clock.
import { useId, useMemo, useState, type UIEvent } from "react";
import { ChartFrame, markProps, useChart, useChartWidth, type ChartTable, type FrameOptions } from "./ChartFrame";
import { durationTicks, formatDurationTick, linearScale } from "./scales";
import { fmtDuration } from "../theme/format";

export type GanttRow = {
  id: string;
  parentId: string | null;
  label: string;
  /** Unix seconds. */
  start: number;
  /** Unix seconds; null while running. */
  end: number | null;
  status: string;
};

export type GanttProps = FrameOptions & {
  rows: GanttRow[];
  /** Time origin (default: earliest start). */
  t0?: number;
  /** Current time for running bars (default: latest end or start). */
  now?: number;
  markers?: { t: number; kind: "warning" | "checkpoint"; label: string }[];
  bands?: { from: number; to: number; label?: string }[];
  timeMode?: "relative" | "wall";
  rowHeight?: number;
  /** Rows visible at once before scrolling. */
  viewportRows?: number;
  /** Rows deeper than this start collapsed (default 2: stages open, tasks folded). */
  openDepth?: number;
  width?: number;
  onRowClick?: (id: string) => void;
};

type Flat = GanttRow & { depth: number; hasChildren: boolean };

/** Depth-first order (children by start time) with depths. */
export function flattenSpans(rows: GanttRow[]): Flat[] {
  const byParent = new Map<string | null, GanttRow[]>();
  const ids = new Set(rows.map((r) => r.id));
  for (const r of rows) {
    const p = r.parentId && ids.has(r.parentId) ? r.parentId : null;
    const list = byParent.get(p) ?? [];
    list.push(r);
    byParent.set(p, list);
  }
  for (const list of byParent.values()) list.sort((a, b) => a.start - b.start || a.label.localeCompare(b.label));
  const out: Flat[] = [];
  const walk = (parent: string | null, depth: number) => {
    for (const r of byParent.get(parent) ?? []) {
      const kids = byParent.get(r.id);
      out.push({ ...r, depth, hasChildren: !!kids?.length });
      walk(r.id, depth + 1);
    }
  };
  walk(null, 0);
  return out;
}

/** Rows whose ancestors are all expanded. */
export function visibleSpans(flat: Flat[], collapsed: Set<string>): Flat[] {
  const out: Flat[] = [];
  const hiddenUnder: number[] = [];
  for (const r of flat) {
    while (hiddenUnder.length && hiddenUnder[hiddenUnder.length - 1] >= r.depth) hiddenUnder.pop();
    if (hiddenUnder.length) continue;
    out.push(r);
    if (r.hasChildren && collapsed.has(r.id)) hiddenUnder.push(r.depth);
  }
  return out;
}

const STATUS_FILL: Record<string, string> = {
  running: "var(--st-running)",
  ok: "var(--s1)",
  done: "var(--s1)",
  error: "var(--st-failed)",
  failed: "var(--st-failed)",
  cancelled: "var(--st-cancelled)",
  cached: "var(--st-cached)",
  skipped: "var(--gray-mark)",
};

export function ganttTable(rows: GanttRow[], t0: number, now: number): ChartTable {
  return {
    columns: [{ key: "span", label: "Span" }, { key: "status", label: "Status" }, { key: "start", label: "Start", unit: "s from start" }, { key: "dur", label: "Duration", unit: "s" }],
    rows: flattenSpans(rows).map((r) => ["  ".repeat(r.depth) + r.label, r.status, Number((r.start - t0).toFixed(2)), Number(((r.end ?? now) - r.start).toFixed(2))]),
  };
}

function clock(t: number): string {
  const d = new Date(t * 1000);
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}${d.getSeconds() ? ":" + String(d.getSeconds()).padStart(2, "0") : ""}`;
}

function Plot(p: GanttProps & { mode: "relative" | "wall"; t0: number; now: number }) {
  const ctx = useChart();
  const pid = useId().replace(/:/g, "");
  const rh = p.rowHeight ?? 22;
  const vrows = p.viewportRows ?? 16;
  const auto = useChartWidth(760);
  const W = p.width ?? auto;
  const labelW = Math.round(Math.min(230, W * 0.32));
  const axisH = 28;
  const openDepth = p.openDepth ?? 2;
  const flat = useMemo(() => flattenSpans(p.rows), [p.rows]);
  // A row is folded when it is at least `openDepth` deep, XOR the user toggled it; rows that
  // arrive later (live runs) follow the same default.
  const [toggled, setToggled] = useState<Set<string>>(() => new Set());
  const collapsed = useMemo(() => new Set(flat.filter((r) => r.hasChildren && (r.depth >= openDepth) !== toggled.has(r.id)).map((r) => r.id)), [flat, toggled, openDepth]);
  const [scrollTop, setScrollTop] = useState(0);
  const vis = useMemo(() => visibleSpans(flat, collapsed), [flat, collapsed]);
  const viewH = axisH + Math.min(vis.length, vrows) * rh + 6;
  const first = Math.max(0, Math.floor(scrollTop / rh) - 4);
  const last = Math.min(vis.length, Math.ceil((scrollTop + vrows * rh) / rh) + 4);
  const sx = linearScale([p.t0, Math.max(p.now, p.t0 + 1)], [labelW, W - 12]);
  const { ticks, step } = durationTicks(0, Math.max(1, p.now - p.t0), Math.max(2, Math.floor((W - 12 - labelW) / 80)));
  const toggle = (id: string) =>
    setToggled((c) => {
      const n = new Set(c);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  const yOf = (i: number) => axisH + i * rh - scrollTop;
  return (
    <div style={{ height: viewH, overflowY: vis.length > vrows ? "auto" : "hidden", position: "relative" }} onScroll={(e: UIEvent<HTMLDivElement>) => setScrollTop(e.currentTarget.scrollTop)}>
      <div style={{ height: axisH + vis.length * rh + 6, position: "relative" }}>
        <svg className="chart" viewBox={`0 0 ${W} ${viewH}`} style={{ position: "sticky", top: 0, height: viewH, width: "100%" }} role="tree" aria-label={p.title} preserveAspectRatio="xMinYMin meet">
          <defs>
            <pattern id={`hatch-${pid}`} width={6} height={6} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1={0} y1={0} x2={0} y2={6} style={{ stroke: "var(--hatch)" }} strokeWidth={1.5} opacity={0.5} />
            </pattern>
            <clipPath id={`rows-${pid}`}>
              <rect x={0} y={axisH} width={W} height={viewH - axisH} />
            </clipPath>
          </defs>
          {(p.bands ?? []).map((b, i) => (
            <g key={`band${i}`}>
              <rect x={sx(b.from)} width={Math.max(2, sx(b.to) - sx(b.from))} y={axisH} height={viewH - axisH} fill={`url(#hatch-${pid})`} {...markProps(ctx, `${b.label ?? "system sleep?"}: ${fmtDuration(b.to - b.from)} without heartbeats`)} />
              <text className="t-axis" x={sx(b.from) + 3} y={viewH - 6}>
                {b.label ?? "system sleep?"}
              </text>
            </g>
          ))}
          {ticks.map((t) => (
            <g key={t}>
              <line className="gridline" x1={sx(p.t0 + t)} x2={sx(p.t0 + t)} y1={axisH - 4} y2={viewH} />
              <text className="t-axis" x={sx(p.t0 + t)} y={12} textAnchor="middle">
                {p.mode === "wall" ? clock(p.t0 + t) : formatDurationTick(t, step)}
              </text>
            </g>
          ))}
          <line className="axisline" x1={labelW} x2={W - 12} y1={axisH - 4} y2={axisH - 4} />
          {(p.markers ?? []).map((m, i) =>
            m.kind === "warning" ? (
              <path key={`m${i}`} d={`M${sx(m.t)},${axisH - 5}l-4,-7h8z`} style={{ fill: "var(--warning)" }} {...markProps(ctx, `Warning at ${formatDurationTick(m.t - p.t0, 1)}: ${m.label}`)} />
            ) : (
              <path key={`m${i}`} d={`M${sx(m.t)},${axisH - 4}v-12h7l-2,3l2,3h-7`} style={{ fill: "var(--s3)", stroke: "var(--s3)" }} {...markProps(ctx, `Checkpoint at ${formatDurationTick(m.t - p.t0, 1)}: ${m.label}`)} />
            ),
          )}
          <g clipPath={`url(#rows-${pid})`}>
            {vis.slice(first, last).map((r, k) => {
              const i = first + k;
              const y = yOf(i);
              const end = r.end ?? p.now;
              const fill = STATUS_FILL[r.status] ?? "var(--s1)";
              const label = `${r.label}: ${r.status}, ${fmtDuration(end - r.start)}${r.end === null ? " so far" : ""}, started at ${formatDurationTick(r.start - p.t0, 1)}`;
              return (
                <g key={r.id} role="treeitem" aria-level={r.depth + 1} aria-expanded={r.hasChildren ? !collapsed.has(r.id) : undefined}>
                  {i % 2 ? <rect x={0} y={y} width={W} height={rh} style={{ fill: "var(--ink)" }} opacity={0.025} aria-hidden="true" /> : null}
                  {r.hasChildren ? (
                    <text
                      className="t-label"
                      x={6 + r.depth * 14}
                      y={y + rh / 2}
                      dy="0.32em"
                      style={{ cursor: "pointer" }}
                      {...markProps(ctx, `${collapsed.has(r.id) ? "Expand" : "Collapse"} ${r.label}`, () => toggle(r.id))}
                    >
                      {collapsed.has(r.id) ? "▸" : "▾"}
                    </text>
                  ) : null}
                  <text className={r.depth === 0 ? "t-strong" : "t-label"} x={18 + r.depth * 14} y={y + rh / 2} dy="0.32em" aria-hidden="true">
                    {(() => {
                      const max = Math.max(6, Math.floor((labelW - 22 - r.depth * 14) / 6.2));
                      return r.label.length > max ? r.label.slice(0, max - 1) + "…" : r.label;
                    })()}
                  </text>
                  <rect
                    x={sx(r.start)}
                    y={y + 4}
                    width={Math.max(2, sx(end) - sx(r.start))}
                    height={rh - 8}
                    rx={2}
                    style={{ fill, opacity: r.depth === 0 ? 0.55 : 1 }}
                    {...markProps(ctx, label, p.onRowClick ? () => p.onRowClick!(r.id) : undefined)}
                  />
                  {r.end === null ? <rect x={sx(end) - 2} y={y + 3} width={2} height={rh - 6} style={{ fill: "var(--ink)" }} aria-hidden="true" /> : null}
                </g>
              );
            })}
          </g>
        </svg>
      </div>
    </div>
  );
}

export function Gantt(props: GanttProps) {
  const [mode, setMode] = useState<"relative" | "wall">(props.timeMode ?? "relative");
  const t0 = props.t0 ?? Math.min(...props.rows.map((r) => r.start), Number.POSITIVE_INFINITY);
  const now = props.now ?? Math.max(...props.rows.map((r) => r.end ?? r.start), t0 + 1);
  const safeT0 = Number.isFinite(t0) ? t0 : 0;
  const safeNow = Number.isFinite(now) ? now : safeT0 + 1;
  const toggle = (
    <button type="button" className="btn small ghost" aria-pressed={mode === "wall"} onClick={() => setMode((m) => (m === "wall" ? "relative" : "wall"))}>
      {mode === "wall" ? "Wall clock" : "Relative time"}
    </button>
  );
  return (
    <ChartFrame {...props} actions={<>{props.actions}{toggle}</>} table={ganttTable(props.rows, safeT0, safeNow)}>
      <Plot {...props} mode={mode} t0={safeT0} now={safeNow} />
    </ChartFrame>
  );
}
