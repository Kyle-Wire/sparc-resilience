// Ring-profile spill chart (SPEC §2 J6, §7.7): mean ΔT inside the edited cells (ring 0) and in
// distance rings around them, with ±1.96 SE whiskers, so the reach of a scenario is visible.
import { Axis, BandAxis, axisTitle } from "./Axis";
import { ChartFrame, markProps, useChart, useChartWidth, type ChartTable, type FrameOptions } from "./ChartFrame";
import { bandScale, extent, linearScale, niceDomain } from "./scales";
import { fmtInt, fmtNum, fmtSigned } from "../theme/format";

export type Ring = { r0_m: number; r1_m: number; mean: number | null; se?: number | null; n?: number | null };

export type RingProfileProps = FrameOptions & {
  rings: Ring[];
  unit?: string;
  valueLabel?: string;
  decimals?: number;
  width?: number;
  height?: number;
};

export function ringLabel(r: Ring): string {
  return r.r0_m === 0 && r.r1_m === 0 ? "edited" : `${fmtNum(r.r0_m, 0)}–${fmtNum(r.r1_m, 0)} m`;
}

export function ringTable(p: Pick<RingProfileProps, "rings" | "unit" | "valueLabel">): ChartTable {
  return {
    columns: [{ key: "r0", label: "From", unit: "m" }, { key: "r1", label: "To", unit: "m" }, { key: "mean", label: p.valueLabel ?? "Mean ΔT", unit: p.unit }, { key: "se", label: "SE", unit: p.unit }, { key: "n", label: "Cells" }],
    rows: p.rings.map((r) => [r.r0_m, r.r1_m, r.mean, r.se ?? null, r.n ?? null]),
  };
}

function Plot(p: RingProfileProps) {
  const ctx = useChart();
  const auto = useChartWidth(640);
  const W = p.width ?? auto;
  const H = p.height ?? (p.rings.length > 8 ? 300 : 260);
  const keys = p.rings.map(ringLabel);
  // Many rings get rotated distance labels (BandAxis): leave room for them and the axis title.
  const many = keys.length > 8;
  const M = { top: 12, right: 16, bottom: many ? 84 : 46, left: 62 };
  const band = bandScale(keys, [M.left, W - M.right], 0.25, 0.1);
  const e = extent([0, ...p.rings.flatMap((r) => [r.mean, r.mean !== null && r.se ? r.mean - 1.96 * r.se : null, r.mean !== null && r.se ? r.mean + 1.96 * r.se : null])]) ?? [-1, 0];
  const sy = linearScale(niceDomain(e[0], e[1]), [H - M.bottom, M.top]);
  const d = p.decimals ?? 3;
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="group" aria-label={p.title}>
      <Axis scale={sy} orient="left" at={M.left} grid={W - M.left - M.right} title={axisTitle(p.valueLabel ?? "Mean ΔT", p.unit)} signed />
      <BandAxis scale={band} orient="bottom" at={H - M.bottom} title="Distance from the edited cells" titleGap={many ? 72 : undefined} />
      <line className="zero" x1={M.left} x2={W - M.right} y1={sy(0)} y2={sy(0)} />
      {p.rings.map((r, i) => {
        if (r.mean === null || !Number.isFinite(r.mean)) return null;
        const k = keys[i];
        const x = band(k);
        const edited = i === 0 && r.r0_m === 0;
        const label = `${k}: ${fmtSigned(r.mean, d)}${p.unit ? " " + p.unit : ""}${r.se ? ` ± ${fmtNum(1.96 * r.se, d)}` : ""}${r.n !== undefined && r.n !== null ? `, ${fmtInt(r.n)} cells` : ""}${r.mean < 0 ? ", cooler" : r.mean > 0 ? ", warmer" : ""}`;
        return (
          <g key={k}>
            <rect
              x={x}
              width={band.bandwidth}
              y={Math.min(sy(0), sy(r.mean))}
              height={Math.max(1, Math.abs(sy(r.mean) - sy(0)))}
              rx={2}
              style={{ fill: edited ? "var(--s1)" : "var(--s1)", fillOpacity: edited ? 1 : 0.55 }}
              {...markProps(ctx, label)}
            />
            {r.se ? (
              <line x1={x + band.bandwidth / 2} x2={x + band.bandwidth / 2} y1={sy(r.mean - 1.96 * r.se)} y2={sy(r.mean + 1.96 * r.se)} style={{ stroke: "var(--ink-2)" }} strokeWidth={1.4} aria-hidden="true" />
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}

export function RingProfile(props: RingProfileProps) {
  const legend = props.legend ?? <span>dark bar = edited cells · lighter bars = rings around them · whiskers = ±1.96 SE · negative = cooler</span>;
  return (
    <ChartFrame {...props} legend={legend} table={ringTable(props)}>
      <Plot {...props} />
    </ChartFrame>
  );
}
