// The run's trust grade (Good / Fair / Limited) in plain words: how well the model predicts
// areas it was not trained on (held-out R²) and whether its likely ranges hold (interval
// coverage near its target). Pure, unit-tested (src/test/trust.test.ts). The Summary's trust
// card, How much to trust this and the top bar's trust dot (layouts/AppShell) all use it.
// The words (grade labels, sentences, the scenario verdicts) come from the shared glossary.
import { GLOSSARY, term, trustWord, type TermId } from "./glossary";

export type TrustGrade = "good" | "fair" | "limited";

/** The rule behind the grade. R² sets the grade; ranges off target by more than `coveragePts` drop it one step. */
export const TRUST_RULE = {
  /** Held-out R² at or above this is Good. */
  good: 0.6,
  /** Held-out R² at or above this (and below `good`) is Fair; below it, Limited. */
  fair: 0.4,
  /** Coverage within this many percentage points of its target counts as "ranges hold". */
  coveragePts: 5,
  /** Coverage target when the run does not say (a 90% interval). */
  defaultTarget: 0.9,
} as const;

export const TRUST_GRADE_LABEL: Record<TrustGrade, string> = { good: trustWord("good").label, fair: trustWord("fair").label, limited: trustWord("limited").label };

export type TrustInput = {
  /** Held-out (cross-validated) R² of the final model. */
  r2: number | null | undefined;
  /** Share of held-out readings inside their likely range (0..1). */
  coverage?: number | null;
  /** The share the likely range aims for (0..1); default 0.9. */
  coverageTarget?: number | null;
};

export type TrustAssessment = {
  grade: TrustGrade;
  /** "Good" | "Fair" | "Limited". */
  label: string;
  /** One plain sentence for planners (no statistics). */
  line: string;
  /** The grade from R² alone, before the ranges check. */
  r2Grade: TrustGrade;
  /** How the likely ranges behave: "ok" within the tolerance, "narrow" too few readings inside, "wide" too many; null when unknown. */
  ranges: "ok" | "narrow" | "wide" | null;
};

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

const STEP_DOWN: Record<TrustGrade, TrustGrade> = { good: "fair", fair: "limited", limited: "limited" };

const BASE_LINE: Record<TrustGrade, string> = { good: trustWord("good").sentence, fair: trustWord("fair").sentence, limited: trustWord("limited").sentence };

const RANGES_LINE: Record<"ok" | "narrow" | "wide", string> = GLOSSARY.trustRanges;

/** The grade from held-out R² alone. */
export function r2Grade(r2: number): TrustGrade {
  return r2 >= TRUST_RULE.good ? "good" : r2 >= TRUST_RULE.fair ? "fair" : "limited";
}

/**
 * The run's trust grade, or null when its held-out R² is unknown (the model has not been
 * evaluated yet). R² ≥ 0.6 is Good, ≥ 0.4 Fair, else Limited; when the interval coverage is
 * known and more than 5 percentage points from its target, the grade drops one step (Good →
 * Fair, Fair → Limited).
 */
export function trustGrade({ r2, coverage, coverageTarget }: TrustInput): TrustAssessment | null {
  if (!isNum(r2)) return null;
  const base = r2Grade(r2);
  let ranges: TrustAssessment["ranges"] = null;
  if (isNum(coverage)) {
    const target = isNum(coverageTarget) ? coverageTarget : TRUST_RULE.defaultTarget;
    // Compare in rounded percentage points so 85.0% vs a 90% target counts as within 5 pts.
    const offPts = Math.round((coverage - target) * 1000) / 10;
    ranges = Math.abs(offPts) <= TRUST_RULE.coveragePts ? "ok" : offPts < 0 ? "narrow" : "wide";
  }
  const grade = ranges === "narrow" || ranges === "wide" ? STEP_DOWN[base] : base;
  const line = ranges ? `${BASE_LINE[base]} ${RANGES_LINE[ranges]}` : BASE_LINE[base];
  return { grade, label: TRUST_GRADE_LABEL[grade], line, r2Grade: base, ranges };
}

// ---------------------------------------------------------------- scenario verdicts

/** What the uncertainty report's evidence supports for a scenario (Verdict.verdict). */
export type VerdictKey = "robust" | "direction" | "not_established" | "unknown";

/**
 * The glossary term of each verdict: planner pages show its plain label ("Solid", "Direction
 * clear", "Not proven") with the tip; Model details keeps the server's technical label.
 */
export const VERDICT_TERMS: Record<Exclude<VerdictKey, "unknown">, TermId> = {
  robust: "verdict_robust",
  direction: "verdict_direction",
  not_established: "verdict_not_established",
};

/** A verdict in plain words: its glossary label and term, else the server's label (no term). */
export function verdictWord(verdict: string, fallback?: string | null): { label: string; term: TermId | null } {
  const id = Object.prototype.hasOwnProperty.call(VERDICT_TERMS, verdict) ? VERDICT_TERMS[verdict as keyof typeof VERDICT_TERMS] : null;
  if (id) return { label: term(id).label, term: id };
  return { label: fallback?.trim() || verdict.replace(/_/g, " "), term: null };
}
