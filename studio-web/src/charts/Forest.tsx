// Forest plot: estimate ± interval per row with a zero line and an optional shaded "better"
// side (baselines ΔMSE ± 2 SE; causal θ_own/θ_nbr/θ_sum ± 1.96 SE beside model slopes).
import { Axis, axisTitle } from "./Axis";
import { ChartFrame, markProps, useChart, useChartWidth, type ChartTable, type FrameOptions } from "./ChartFrame";
import { bandScale, extent, linearScale, niceDomain } from "./scales";
import { fmtSigned } from "../theme/format";

export type ForestRow = {
  id: string;
  label: string;
  est: number | null;
  /** Interval bounds; or give `se` and `z` to derive est ± z·se. */
  lo?: number | null;
  hi?: number | null;
  se?: number | null;
  group?: string;
  /** Secondary marker drawn as a hollow square (e.g. the model's own slope). */
  compare?: number | null;
  muted?: boolean;
};

export type ForestProps = FrameOptions & {
  rows: ForestRow[];
  valueLabel: string;
  unit?: string;
  /** Multiplier for `se` (2 for ΔMSE ± 2 SE, 1.96 for 95%). */
  z?: number;
  /** Shade the side of zero that means "better", with this label. */
  better?: { side: "negative" | "positive"; label: string };
  compareLabel?: string;
  decimals?: number;
  width?: number;
  onRowClick?: (id: string) => void;
};

export function forestBounds(r: ForestRow, z: number): [number | null, number | null] {
  if (r.lo !== undefined && r.lo !== null && r.hi !== undefined && r.hi !== null) return [r.lo, r.hi];
  if (r.est !== null && r.se !== undefined && r.se !== null) return [r.est - z * r.se, r.est + z * r.se];
  return [null, null];
}

export function forestTable(p: Pick<ForestProps, "rows" | "valueLabel" | "unit" | "z" | "compareLabel">): ChartTable {
  const z = p.z ?? 1.96;
  const hasCompare = p.rows.some((r) => r.compare !== undefined && r.compare !== null);
  const columns: ChartTable["columns"] = [
    { key: "group", label: "Group" },
    { key: "label", label: "Row" },
    { key: "est", label: p.valueLabel, unit: p.unit },
    { key: "lo", label: "Lower", unit: p.unit },
    { key: "hi", label: "Upper", unit: p.unit },
    { key: "excl", label: "Excludes 0" },
  ];
  if (hasCompare) columns.push({ key: "cmp", label: p.compareLabel ?? "Comparison", unit: p.unit });
  return {
    columns,
    rows: p.rows.map((r) => {
      const [lo, hi] = forestBounds(r, z);
      const row: ChartTable["rows"][number] = [r.group ?? "", r.label, r.est, lo, hi, lo !== null && hi !== null ? (lo > 0 || hi < 0 ? "yes" : "no") : ""];
      if (hasCompare) row.push(r.compare ?? null);
      return row;
    }),
  };
}

function Plot(p: ForestProps) {
  const ctx = useChart();
  const z = p.z ?? 1.96;
  const auto = useChartWidth(640);
  const W = p.width ?? auto;
  const M = { top: 22, right: 20, bottom: 44, left: 180 };
  const H = M.top + M.bottom + p.rows.length * 24;
  const bounds = p.rows.map((r) => forestBounds(r, z));
  const e = extent([...bounds.flat(), ...p.rows.map((r) => r.est), ...p.rows.map((r) => r.compare ?? null), 0]) ?? [-1, 1];
  const dom = niceDomain(e[0], e[1]);
  const sx = linearScale(dom, [M.left, W - M.right]);
  const band = bandScale(p.rows.map((r) => r.id), [M.top, H - M.bottom], 0.35, 0.05);
  const d = p.decimals ?? 3;
  const x0 = sx(0);
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="group" aria-label={p.title}>
      {p.better ? (
        <g aria-hidden="true">
          <rect
            x={p.better.side === "negative" ? M.left : x0}
            width={p.better.side === "negative" ? x0 - M.left : W - M.right - x0}
            y={M.top}
            height={H - M.top - M.bottom}
            style={{ fill: "var(--good)" }}
            opacity={0.07}
          />
          <text className="t-axis" x={p.better.side === "negative" ? M.left + 4 : W - M.right - 4} y={M.top - 8} textAnchor={p.better.side === "negative" ? "start" : "end"}>
            {p.better.label}
          </text>
        </g>
      ) : null}
      <Axis scale={sx} orient="bottom" at={H - M.bottom} grid={H - M.top - M.bottom} title={axisTitle(p.valueLabel, p.unit)} signed />
      <line className="zero" x1={x0} x2={x0} y1={M.top} y2={H - M.bottom} />
      {p.rows.map((r, i) => {
        const cy = band(r.id) + band.bandwidth / 2;
        const [lo, hi] = bounds[i];
        const color = r.muted ? "var(--gray-mark)" : "var(--s1)";
        const label = `${r.group ? r.group + ", " : ""}${r.label}: ${fmtSigned(r.est, d)}${p.unit ? " " + p.unit : ""}${lo !== null && hi !== null ? ` (${fmtSigned(lo, d)} to ${fmtSigned(hi, d)})${lo > 0 || hi < 0 ? ", excludes zero" : ", includes zero"}` : ""}${r.compare !== undefined && r.compare !== null ? `; ${p.compareLabel ?? "comparison"} ${fmtSigned(r.compare, d)}` : ""}`;
        return (
          <g key={r.id}>
            <text className="t-label" x={M.left - 8} y={cy} dy="0.32em" textAnchor="end" aria-hidden="true">
              {(r.group && (i === 0 || p.rows[i - 1].group !== r.group) ? `${r.group} · ` : "") + (r.label.length > 24 ? r.label.slice(0, 23) + "…" : r.label)}
            </text>
            {lo !== null && hi !== null ? <line x1={sx(lo)} x2={sx(hi)} y1={cy} y2={cy} style={{ stroke: color }} strokeWidth={2} aria-hidden="true" /> : null}
            {r.compare !== undefined && r.compare !== null ? (
              <rect x={sx(r.compare) - 4} y={cy - 4} width={8} height={8} style={{ fill: "var(--surface)", stroke: "var(--s2)" }} strokeWidth={1.5} aria-hidden="true" />
            ) : null}
            {r.est !== null && Number.isFinite(r.est) ? (
              <rect
                x={sx(r.est) - 5}
                y={cy - 5}
                width={10}
                height={10}
                transform={`rotate(45 ${sx(r.est)} ${cy})`}
                style={{ fill: color }}
                {...markProps(ctx, label, p.onRowClick ? () => p.onRowClick!(r.id) : undefined)}
              />
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}

export function Forest(props: ForestProps) {
  const z = props.z ?? 1.96;
  const legend =
    props.legend ??
    (
      <>
        <span>
          <i style={{ background: "var(--s1)" }} />
          estimate ± {z === 2 ? "2" : "1.96"} SE
        </span>
        {props.rows.some((r) => r.compare !== undefined && r.compare !== null) ? (
          <span>
            <i className="box" style={{ border: "1.5px solid var(--s2)" }} />
            {props.compareLabel ?? "comparison"}
          </span>
        ) : null}
      </>
    );
  return (
    <ChartFrame {...props} legend={legend} table={forestTable(props)}>
      <Plot {...props} />
    </ChartFrame>
  );
}
