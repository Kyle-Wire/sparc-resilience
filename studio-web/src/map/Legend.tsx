// Map legend (SPEC §6.3, §6.5): the colour ramp with domain ticks and units, cooler/warmer
// labels on diverging layers, category swatches for categorical layers, an embedded brushable
// histogram of the layer, and the layer summary (n, mean, p10/median/p90 or class shares).
import { useMemo } from "react";
import type { LayerMeta } from "../api/types";
import { Histogram, histogram } from "../charts/Histogram";
import { fmtInt, fmtNum, fmtPct, fmtSigned, unitLabel } from "../theme/format";
import { catColors, rampColor } from "../theme/palette";
import { categoryCounts, layerSummary, type Domain } from "./domain";

export type LegendProps = {
  meta: LayerMeta;
  domain: Domain;
  dark: boolean;
  values?: ArrayLike<number> | null;
  /** Brushed display-value range on the histogram. */
  brush?: [number, number] | null;
  onBrush?: (range: [number, number] | null) => void;
  /** Hide the histogram (compact legends). */
  compact?: boolean;
};

/** Category colours for a cat layer in class order (class 0 grey, then s1..s3). */
export function categorySwatches(meta: LayerMeta, dark: boolean): { label: string; color: string }[] {
  const labels = meta.labels ?? [];
  const { colors, gray } = catColors(dark);
  if (labels.length > 4) return labels.map((l, k) => ({ label: l, color: rampColor("seq", k / (labels.length - 1), dark) }));
  return labels.map((l, k) => ({ label: l, color: k === 0 ? gray : colors[k - 1] }));
}

const TEMP_UNITS = new Set(["°F", "°C", "K"]);

/** Words for the two ends of a diverging ramp: cooler/warmer for temperatures (SPEC §6.1). */
export function divergingEnds(meta: Pick<LayerMeta, "unit" | "sign_note">): [string, string] {
  const u = unitLabel(meta.unit);
  const note = (meta.sign_note ?? "").toLowerCase();
  if (TEMP_UNITS.has(u) || u.startsWith("°F") || u.startsWith("°C")) return /positive\s*=\s*cooler/.test(note) ? ["less cooling", "more cooling"] : ["cooler", "warmer"];
  return ["lower", "higher"];
}

export function Legend({ meta, domain, dark, values, brush, onBrush, compact }: LegendProps) {
  const u = unitLabel(meta.unit);
  const d = meta.decimals ?? 2;
  const signed = domain.kind === "div" && (domain.center ?? 0) === 0;
  const f = (v: number) => (signed ? fmtSigned(v, d) : fmtNum(v, d));

  const summary = useMemo(() => (values && domain.kind !== "cat" ? layerSummary(values, domain) : null), [values, domain]);
  const counts = useMemo(() => (values && domain.kind === "cat" ? categoryCounts(values, domain.nCat) : null), [values, domain]);
  const bins = useMemo(() => {
    if (!values || domain.kind === "cat") return null;
    const disp = new Float32Array(values.length);
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      disp[i] = domain.zeroBlank && v <= 0 ? NaN : v * domain.mult;
    }
    return histogram(disp, 32, [domain.lo, domain.hi]);
  }, [values, domain]);

  if (domain.kind === "cat") {
    const sw = categorySwatches(meta, dark);
    const total = counts ? counts.reduce((a, b) => a + b, 0) : 0;
    return (
      <div className="stack" style={{ gap: 8 }}>
        <div className="swatches" role="list" aria-label={`${meta.label} classes`}>
          {sw.map((s, k) => (
            <div key={k} role="listitem">
              <span className="sw" style={{ background: s.color }} />
              {s.label}
              {counts && total ? <span className="muted num"> · {fmtPct(counts[k] / total, 1)}</span> : null}
            </div>
          ))}
        </div>
      </div>
    );
  }

  const steps = 64;
  const ends = divergingEnds(meta);
  return (
    <div className="stack" style={{ gap: 8 }}>
      <div className="legend-ramp">
        <svg viewBox={`0 0 ${steps} 6`} preserveAspectRatio="none" role="img" aria-label={`Colour scale from ${f(domain.lo)} to ${f(domain.hi)} ${u}`} style={{ height: 14 }}>
          {Array.from({ length: steps }, (_, i) => (
            <rect key={i} x={i} width={1.05} height={6} fill={rampColor(domain.kind === "div" ? "div" : "seq", i / (steps - 1), dark)} />
          ))}
        </svg>
        <div className="legend-ticks">
          <span>≤ {f(domain.lo)}</span>
          {domain.kind === "div" && domain.center !== null ? <span>{f(domain.center)}</span> : null}
          <span>
            ≥ {f(domain.hi)} {u}
          </span>
        </div>
        {domain.kind === "div" ? (
          <div className="legend-ticks">
            <span>{ends[0]}</span>
            <span>{ends[1]}</span>
          </div>
        ) : null}
        {domain.zeroBlank ? <p className="cap">Cells at or below zero are shown as no data.</p> : null}
        {meta.mult && meta.mult !== 1 ? <p className="cap">Values are shown × {fmtNum(meta.mult, 2)}.</p> : null}
        {meta.sign_note ? <p className="cap">{meta.sign_note}</p> : null}
      </div>
      {!compact && bins ? (
        <Histogram
          bare
          title={`${meta.label} distribution`}
          bins={bins}
          xLabel={meta.label}
          unit={u}
          decimals={d}
          height={120}
          width={300}
          ramp={{ kind: domain.kind === "div" ? "div" : "seq", lo: domain.lo, hi: domain.hi, center: domain.center ?? 0 }}
          brush={brush ?? null}
          onBrush={onBrush}
        />
      ) : null}
      {summary ? (
        <dl className="kv" aria-label="Layer summary">
          <dt>Cells</dt>
          <dd>{fmtInt(summary.n)}</dd>
          <dt>Mean</dt>
          <dd>
            {f(summary.mean ?? NaN)} {u}
          </dd>
          <dt>10th percentile</dt>
          <dd>
            {f(summary.p10 ?? NaN)} {u}
          </dd>
          <dt>Median</dt>
          <dd>
            {f(summary.p50 ?? NaN)} {u}
          </dd>
          <dt>90th percentile</dt>
          <dd>
            {f(summary.p90 ?? NaN)} {u}
          </dd>
        </dl>
      ) : null}
    </div>
  );
}
