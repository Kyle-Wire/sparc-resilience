// SVG axis with ticks, optional gridlines and a title that carries the units (SPEC §12.6).
import { formatLogTick, formatTick, tickStep, type BandScale, type Scale } from "./scales";

export type AxisProps = {
  scale: Scale;
  orient: "bottom" | "left";
  /** Position of the axis line: y for bottom axes, x for left axes. */
  at: number;
  /** Gridlines span from `at` across this many px (0 = none). */
  grid?: number;
  title?: string;
  ticks?: number[];
  /** Default: from the axis length (autoTickCount). */
  tickCount?: number;
  format?: (v: number) => string;
  signed?: boolean;
};

/** Ticks that fit the axis length: about one per 80 px across, one per 45 px up. */
export function autoTickCount(orient: "bottom" | "left", lengthPx: number): number {
  return Math.max(2, Math.min(8, Math.floor(Math.abs(lengthPx) / (orient === "bottom" ? 80 : 45))));
}

export function Axis({ scale, orient, at, grid = 0, title, ticks, tickCount, format, signed }: AxisProps) {
  const count = tickCount ?? autoTickCount(orient, scale.range[1] - scale.range[0]);
  const tks = ticks ?? scale.ticks(count);
  const step = scale.kind === "linear" ? tickStep(scale.domain[0], scale.domain[1], count) : undefined;
  const fmt = format ?? ((v: number) => (scale.kind === "log" ? formatLogTick(v) : formatTick(v, step, signed)));
  const [r0, r1] = scale.range;
  if (orient === "bottom") {
    return (
      <g aria-hidden="true">
        {grid
          ? tks.map((t) => <line key={`g${t}`} className="gridline" x1={scale(t)} x2={scale(t)} y1={at} y2={at - grid} />)
          : null}
        <line className="axisline" x1={Math.min(r0, r1)} x2={Math.max(r0, r1)} y1={at} y2={at} />
        {tks.map((t) => (
          <g key={t} transform={`translate(${scale(t)},${at})`}>
            <line className="axisline" y2={4} />
            <text className="t-axis" y={16} textAnchor="middle">
              {fmt(t)}
            </text>
          </g>
        ))}
        {title ? (
          <text className="t-label" x={(r0 + r1) / 2} y={at + 32} textAnchor="middle">
            {title}
          </text>
        ) : null}
      </g>
    );
  }
  return (
    <g aria-hidden="true">
      {grid ? tks.map((t) => <line key={`g${t}`} className="gridline" x1={at} x2={at + grid} y1={scale(t)} y2={scale(t)} />) : null}
      <line className="axisline" x1={at} x2={at} y1={Math.min(r0, r1)} y2={Math.max(r0, r1)} />
      {tks.map((t) => (
        <g key={t} transform={`translate(${at},${scale(t)})`}>
          <line className="axisline" x2={-4} />
          <text className="t-axis" x={-7} dy="0.32em" textAnchor="end">
            {fmt(t)}
          </text>
        </g>
      ))}
      {title ? (
        <text className="t-label" transform={`translate(${at - 46},${(r0 + r1) / 2}) rotate(-90)`} textAnchor="middle">
          {title}
        </text>
      ) : null}
    </g>
  );
}

/** Category axis for band scales (labels truncated to `maxChars`). */
export function BandAxis({ scale, orient, at, title, maxChars = 18, titleGap }: { scale: BandScale; orient: "bottom" | "left"; at: number; title?: string; maxChars?: number; titleGap?: number }) {
  const cut = (s: string) => (s.length > maxChars ? s.slice(0, maxChars - 1) + "…" : s);
  if (orient === "bottom") {
    const rotate = scale.keys.length > 8;
    // Labels every `every`-th band so neighbours never overlap: rotated labels need ~20 px between
    // bands, flat ones their own width (~6.5 px a character).
    const longest = Math.max(0, ...scale.keys.map((k) => cut(k).length));
    const every = Math.max(1, Math.ceil((rotate ? 20 : longest * 6.5 + 6) / Math.max(1e-9, Math.abs(scale.step))));
    return (
      <g aria-hidden="true">
        {scale.keys.map((k, i) => (i % every ? null :
          <text
            key={k}
            className="t-axis"
            transform={`translate(${scale(k) + scale.bandwidth / 2},${at + 14})${rotate ? " rotate(-35)" : ""}`}
            textAnchor={rotate ? "end" : "middle"}
          >
            {cut(k)}
          </text>
        ))}
        {title ? (
          <text className="t-label" x={(scale(scale.keys[0] ?? "") + scale(scale.keys[scale.keys.length - 1] ?? "") + scale.bandwidth) / 2} y={at + (titleGap ?? (rotate ? 52 : 32))} textAnchor="middle">
            {title}
          </text>
        ) : null}
      </g>
    );
  }
  return (
    <g aria-hidden="true">
      {scale.keys.map((k) => (
        <text key={k} className="t-label" x={at - 8} y={scale(k) + scale.bandwidth / 2} dy="0.32em" textAnchor="end">
          {cut(k)}
        </text>
      ))}
      {title ? (
        <text className="t-label" x={at - 8} y={-6} textAnchor="end">
          {title}
        </text>
      ) : null}
    </g>
  );
}

/** "Label (unit)" for axis titles; every axis title carries its units. */
export function axisTitle(label: string, unit?: string | null): string {
  return unit ? `${label} (${unit})` : label;
}
