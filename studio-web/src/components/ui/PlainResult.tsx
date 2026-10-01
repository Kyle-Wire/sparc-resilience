import { useState, type ReactNode } from "react";
import type { Likely } from "../../api/types";
import { fmtInt, fmtNum, fmtPct, fmtRange, fmtValue } from "../../theme/format";
import { Badge } from "./Badge";

export type Confidence = "confident_cools" | "confident_warms" | "could_be_zero" | "unknown";

/**
 * Confidence from the 95% likely range: cools iff hi < 0, warms iff lo > 0, could be zero
 * when the range spans 0; unknown without a range (SPEC §7.7 plain-language results).
 */
export function confidenceOf(l: Pick<Likely, "lo" | "hi">): Confidence {
  if (l.lo === null || l.hi === null || !Number.isFinite(l.lo) || !Number.isFinite(l.hi)) return "unknown";
  if (l.hi < 0) return "confident_cools";
  if (l.lo > 0) return "confident_warms";
  return "could_be_zero";
}

export const CONFIDENCE_TEXT: Record<Confidence, string> = {
  confident_cools: "Confident it cools.",
  confident_warms: "Confident it warms.",
  could_be_zero: "Could be zero.",
  unknown: "No uncertainty estimate.",
};

/** "Cools the edited area by 0.41 °F (likely range 0.30–0.52 °F)." */
export function likelySentence(l: Likely, unit: string, subject: string, decimals = 2): string {
  const est = l.estimate;
  const verb = est < 0 ? "Cools" : est > 0 ? "Warms" : "Leaves";
  const mag = fmtValue(Math.abs(est), unit, decimals);
  let s = est === 0 ? `Leaves ${subject} unchanged` : `${verb} ${subject} by ${mag}`;
  if (l.lo !== null && l.hi !== null) {
    // Express the range in the same "by how much" terms as the headline.
    const lo = est < 0 ? -l.hi : l.lo;
    const hi = est < 0 ? -l.lo : l.hi;
    s += ` (likely range ${fmtRange(lo, hi, unit, decimals)})`;
  }
  return s + ".";
}

export type PlainResultProps = {
  likely: Likely;
  unit: string;
  /** What changes, e.g. "the edited area", "the city". */
  subject: string;
  /** Share of edited cells outside the observed range (0..1); > 20% adds a qualifier. */
  fracExtrapolated?: number | null;
  /** Extra qualifiers and "what it buys" lines. */
  qualifiers?: ReactNode[];
  expert?: { n?: number | null; method?: string; folds?: number | null; extra?: [string, ReactNode][] };
  decimals?: number;
  title?: ReactNode;
};

/** Plain-language likely-range card with an expert toggle (SE, n, method). */
export function PlainResult({ likely, unit, subject, fracExtrapolated, qualifiers = [], expert, decimals = 2, title }: PlainResultProps) {
  const [showExpert, setShowExpert] = useState(false);
  const conf = confidenceOf(likely);
  const lines: ReactNode[] = [...qualifiers];
  if (fracExtrapolated !== null && fracExtrapolated !== undefined && fracExtrapolated > 0.2) {
    lines.unshift(`${fmtPct(fracExtrapolated)} of the edited cells are outside the conditions the model was trained on (extrapolated), so treat this with extra caution.`);
  }
  return (
    <div className="plain-result card">
      {title ? <p className="eyebrow">{title}</p> : null}
      <p className="headline">{likelySentence(likely, unit, subject, decimals)}</p>
      <p className="row">
        <Badge tone={conf === "confident_cools" ? "accent" : conf === "confident_warms" ? "crit" : conf === "could_be_zero" ? "warn" : undefined}>
          {conf === "confident_cools" ? "cools" : conf === "confident_warms" ? "warms" : conf === "could_be_zero" ? "could be zero" : "no range"}
        </Badge>
        <span>{CONFIDENCE_TEXT[conf]}</span>
      </p>
      {lines.length ? (
        <ul>
          {lines.map((l, i) => (
            <li key={i}>{l}</li>
          ))}
        </ul>
      ) : null}
      <div>
        <button type="button" className="btn small ghost" aria-expanded={showExpert} onClick={() => setShowExpert((v) => !v)}>
          {showExpert ? "Hide details" : "Expert details"}
        </button>
      </div>
      {showExpert ? (
        <dl>
          <dt>Estimate</dt>
          <dd>{fmtValue(likely.estimate, unit, Math.max(decimals, 3))}</dd>
          <dt>Standard error</dt>
          <dd>{likely.se === null ? "not available" : fmtValue(likely.se, unit, Math.max(decimals, 3))}</dd>
          <dt>95% interval</dt>
          <dd>{fmtRange(likely.lo, likely.hi, unit, Math.max(decimals, 3))}</dd>
          {expert?.n !== undefined && expert.n !== null ? (
            <>
              <dt>Cells</dt>
              <dd>{fmtInt(expert.n)}</dd>
            </>
          ) : null}
          {expert?.folds ? (
            <>
              <dt>Folds (jackknife)</dt>
              <dd>{fmtNum(expert.folds, 0)}</dd>
            </>
          ) : null}
          {expert?.method ? (
            <>
              <dt>Method</dt>
              <dd>{expert.method}</dd>
            </>
          ) : null}
          {expert?.extra?.map(([k, v]) => (
            <span key={k} style={{ display: "contents" }}>
              <dt>{k}</dt>
              <dd>{v}</dd>
            </span>
          ))}
        </dl>
      ) : null}
    </div>
  );
}
