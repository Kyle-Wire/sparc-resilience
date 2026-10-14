// How much to trust this run tab (`/r/:rid/trust`, phase 2): one readable page, top to bottom.
//   1. The grade (Good / Fair / Limited, src/content/trust.ts, the same rule and inputs as the
//      Summary's trust card) with one sentence.
//   2. Four checks, each ✓ or ! with one plain result and a link to its Model details section
//      (no charts: the charts live there): accuracy in places it never saw (held-out R², RMSE),
//      whether the ranges are honest (overall coverage vs its target), whether it holds up away
//      from measurements (skill vs CV block size, and the distance bands whose ranges fall short:
//      the one home of every distance claim; when the skill-vs-distance stage was skipped it is
//      "partly checked" from the coverage by distance alone, with the re-run: the one home of
//      that notice), and cause or coincidence (one line per lever from the causal audit).
//   3. Independent checks: the validation studies' verdicts, with their cards (launch forms,
//      results) behind a disclosure that `#validation` and `#study-<kind>` open.
//   4. How wide the ranges are: the uncertainty report's summary in two lines (which scenarios
//      keep their sign, what the evidence supports), linked to Model details › Uncertainty.
//   5. Limitations from the model card, with a link to the run's caveats (their home is the
//      Summary's "Things to keep in mind").
// This page is the planner's home of the accuracy numbers (R², RMSE, coverage). The charts, the
// stacker, blend weights, physics, advection and forcing cards, the CV design, CATE maps and the
// DAG audit are Model details only; each block links to its Model details section.
// Plain words (phase 4): the checks say "accuracy score", "typical error", "test areas" and
// "likely range"; the technical meaning (held-out R², RMSE, coverage, skill vs block size, the
// causal audit) is in the ⓘ beside each check's title and in the glossary <Term>s in its line.
// The scenario verdicts use the glossary's plain labels (Solid / Direction clear / Not proven).
import { useState, type ReactNode } from "react";
import type { AccuracySections, CoverageRow, UncertaintySections, ViewKpi, ViewModelOf, ViewName } from "../../api/runs";
import { useRunDetailFull, useView } from "../../api/runs";
import { ActionButton } from "../../components/ui/EmptyState";
import { InfoTip } from "../../components/ui/InfoTip";
import { Pill } from "../../components/ui/Pill";
import { Term } from "../../components/ui/Term";
import { term, type TermId } from "../../content/glossary";
import { TRUST_RULE, r2Grade, trustGrade, verdictWord, type TrustAssessment, type TrustGrade } from "../../content/trust";
import { runIsLive } from "../../layouts/resources";
import { Link, runHref, useRoute } from "../../router";
import { fmtNum, fmtPct, fmtValue } from "../../theme/format";
import { IndependentChecks, StudyCards } from "../studies/Validation";
import { finalModel } from "./Accuracy";
import { causalAgreement, type CausalAgreement } from "./Causal";
import { missingSummary, SectionBoundary, TipChip, useRid, useSkippedStage } from "./common";
import { skillAcross, type SkillAcross } from "./Distance";
import { fmtDistance, missingText, targetUnit } from "./format";
import { KEEP_IN_MIND_ID, keepInMind, trustInputs } from "./Summary";
import { envelopeCaption } from "./Uncertainty";

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

const kpiOf = (kpis: readonly ViewKpi[] | null | undefined, id: string): ViewKpi | null => kpis?.find((k) => k.id === id) ?? null;

/** A phrase of a plain sentence and the glossary term that explains it. */
export type TermPhrase = readonly [phrase: string, id: TermId];

/**
 * A plain sentence with the first occurrence of each phrase shown as its glossary <Term> (dotted
 * underline, tooltip). The words stay exactly as written, so the text reads the same.
 */
export function TermText({ text, terms }: { text: string; terms: readonly TermPhrase[] }) {
  const parts: ReactNode[] = [];
  const pending = [...terms];
  let rest = text;
  for (;;) {
    let best = -1;
    let at = -1;
    for (let i = 0; i < pending.length; i++) {
      const j = rest.indexOf(pending[i][0]);
      if (j >= 0 && (at < 0 || j < at)) [best, at] = [i, j];
    }
    if (best < 0) break;
    const [phrase, id] = pending.splice(best, 1)[0];
    if (at > 0) parts.push(rest.slice(0, at));
    parts.push(
      <Term key={parts.length} id={id}>
        {phrase}
      </Term>,
    );
    rest = rest.slice(at + phrase.length);
  }
  if (rest) parts.push(rest);
  return <>{parts}</>;
}

/** The glossary phrases of the checks' lines. */
const ACCURACY_TERMS: readonly TermPhrase[] = [
  ["accuracy score", "r2"],
  ["typical error", "rmse"],
  ["test areas", "held_out_fold"],
];
const RANGES_TERMS: readonly TermPhrase[] = [
  ["test areas", "held_out_fold"],
  ["prediction range", "coverage"],
];

// ---------------------------------------------------------------- the checks (pure)

/** A check's mark: ✓ holds up, ! take care, ◐ partly checked (some of the evidence), – not checked in this run. */
export type CheckStatus = "ok" | "warn" | "partial" | "unknown";

export type CheckResult = { status: CheckStatus; line: string };

/** Accuracy in places it never saw: held-out R² (graded as the trust grade does) and RMSE. */
export function accuracyCheck(r2: number | null | undefined, rmse: number | null | undefined, unit: string): CheckResult {
  if (!isNum(r2)) return { status: "unknown", line: "The model has not been tested on areas it did not see yet." };
  const err = isNum(rmse) ? `, and its typical error is ${fmtValue(rmse, unit, 2)}` : "";
  const head = `Tested on areas left out of training, the model's accuracy score is ${fmtNum(r2, 2)} (1 is perfect)${err}.`;
  const g = r2Grade(r2);
  const tail =
    g === "good"
      ? ` That is good: ${fmtNum(TRUST_RULE.good, 1)} or more counts as good.`
      : g === "fair"
        ? ` That gets the broad pattern right; ${fmtNum(TRUST_RULE.good, 1)} or more would count as good.`
        : ` That is weak (below ${fmtNum(TRUST_RULE.fair, 1)}); use the maps as a rough guide only.`;
  return { status: g === "good" ? "ok" : "warn", line: head + tail };
}

/** How the likely ranges behave against their target, with the trust grade's tolerance (±5 points). */
export function rangesOf(coverage: number, target: number): "ok" | "narrow" | "wide" {
  const offPts = Math.round((coverage - target) * 1000) / 10;
  return Math.abs(offPts) <= TRUST_RULE.coveragePts ? "ok" : offPts < 0 ? "narrow" : "wide";
}

/** Are the ranges honest? The share of held-out readings inside their likely range against its target. */
export function rangesCheck(coverage: number | null | undefined, target: number | null | undefined): CheckResult {
  if (!isNum(coverage)) return { status: "unknown", line: "This run has no check of its prediction ranges yet." };
  const t = isNum(target) ? target : TRUST_RULE.defaultTarget;
  const head = `${fmtPct(coverage, 0)} of the readings in the test areas fall inside the model's prediction range, which should catch ${fmtPct(t, 0)} of them.`;
  const r = rangesOf(coverage, t);
  if (r === "ok") return { status: "ok", line: `${head} The ranges are about right.` };
  return {
    status: "warn",
    line: r === "narrow" ? `${head} The ranges are too narrow: real values fall outside them more often than stated.` : `${head} The ranges are wider than they need to be.`,
  };
}

/**
 * Does it hold up away from measurements? The stack's held-out skill with the smallest and the
 * largest CV blocks: R² still at least Fair (≥ 0.4) far out, or RMSE at most 1.5× its near value.
 * In plain words a CV block is a test area ("test areas 180 m across").
 */
export function distanceCheck(sa: SkillAcross | null, unit: string): CheckResult | null {
  if (!sa) return null;
  const r2 = sa.metric === "r2";
  const v = (x: number) => (r2 ? fmtNum(x, 2) : fmtValue(x, unit, 2));
  const head = `The ${r2 ? "accuracy score" : "typical error"} is ${v(sa.near)} with test areas ${fmtDistance(sa.nearBlockM)} across, and ${v(sa.far)} with test areas ${fmtDistance(sa.farBlockM)} across, where places lie farther from any measurement.`;
  const ok = r2 ? sa.far >= TRUST_RULE.fair : sa.near > 0 ? sa.far <= sa.near * 1.5 : true;
  return { status: ok ? "ok" : "warn", line: ok ? `${head} It holds up.` : `${head} Treat results far from measurements with care.` };
}

/** What the skill-vs-distance stage is, in the Trust page's words. */
const DISTANCE_TEST = "The distance test, which re-checks accuracy farther from the readings,";

/**
 * Does it hold up away from measurements, when skill was not tested at increasing distances
 * (the skill-vs-distance stage skipped)? Partly: the likely ranges' coverage by distance band
 * still says whether they hold farther out. Null without distance bands. `skipped`: the stage was
 * skipped, with its reason as a phrase (useSkippedStage's `why`); the reason comes first, so the line
 * never reads as if the test both ran and did not.
 */
export function distanceCoverageCheck(rows: readonly CoverageRow[], target: number, skipped?: { why?: string | null } | null): CheckResult | null {
  const banded = rows.filter((r) => r.global !== null);
  if (!banded.length) return null;
  const why = skipped?.why ?? null;
  const head = !skipped
    ? "Only the prediction ranges were checked at increasing distances from the readings, not the accuracy itself."
    : why === "turned off in the project settings"
      ? `${DISTANCE_TEST} is turned off in the project settings, so only the prediction ranges were checked by distance.`
      : `${DISTANCE_TEST} did not run${why ? ` (${why})` : ""}, so only the prediction ranges were checked by distance.`;
  const far = underTargetLine(banded, target, "they") ?? `They hold their ${fmtPct(target, 0)} target at every distance checked.`;
  return { status: "partial", line: `${head} ${far}` };
}

/** Cause or coincidence? The causal audit per lever, summed up in one line. */
export function causalCheck(levers: readonly CausalAgreement[]): CheckResult {
  const known = levers.filter((l) => l.status !== "unknown");
  if (!known.length) return { status: "unknown", line: "No cause-and-effect check in this run." };
  const n = known.length;
  const agree = known.filter((l) => l.status === "agrees").length;
  const conflict = known.filter((l) => l.status === "conflict").length;
  const lead = "The cause-and-effect check (a separate statistical analysis that does not use the model)";
  if (agree === n) return { status: "ok", line: `${lead} agrees with the model's effect for ${n === 1 ? "the lever it tested" : `all ${n} levers`}.` };
  if (conflict) return { status: "warn", line: `${lead} points the other way from the model for ${conflict === 1 ? "one lever" : `${conflict} levers`}; see below.` };
  if (agree) return { status: "warn", line: `${lead} agrees with the model for ${agree} of ${n} levers; for the rest it agrees on the direction but not the size.` };
  return { status: "warn", line: `${lead} agrees on the direction of ${n === 1 ? "the effect" : "each effect"} but not on its size.` };
}

// ---------------------------------------------------------------- building blocks

type Res<V extends ViewName> = { data: ViewModelOf<V> | undefined; error: unknown };

type ViewOf<V extends ViewName> =
  | { state: "loading"; vm: null; why?: undefined }
  | { state: "error"; vm: null; why?: undefined }
  | { state: "missing"; vm: null; why?: string }
  | { state: "ready"; vm: ViewModelOf<V>; why?: undefined };

/** A view as a check reads it: loading, unreadable, not in this run (and why), or its ViewModel. */
function viewOf<V extends ViewName>(res: Res<V>): ViewOf<V> {
  if (res.data === undefined) return res.error ? { state: "error", vm: null } : { state: "loading", vm: null };
  if (res.data.availability === "missing") {
    const one = missingSummary(res.data.missing);
    const why = one ? one.sentence : res.data.missing.length ? res.data.missing.map(missingText).join("; ") + "." : undefined;
    return { state: "missing", vm: null, why };
  }
  return { state: "ready", vm: res.data };
}

function Mark({ status }: { status: CheckStatus | "loading" }) {
  if (status === "loading")
    return (
      <span className="cap" role="status">
        <span className="spinner" aria-hidden="true" /> Loading…
      </span>
    );
  if (status === "ok")
    return (
      <Pill tone="good" icon="check">
        Holds up
      </Pill>
    );
  if (status === "warn")
    return (
      <Pill tone="warn" icon="alert">
        Take care
      </Pill>
    );
  if (status === "partial") return <Pill icon="info">Partly checked</Pill>;
  return <Pill icon="minus">Not checked</Pill>;
}

function DetailsLink({ to }: { to: string }) {
  return (
    <Link to={to} className="cap" data-details-link="true">
      Full technical details →
    </Link>
  );
}

/** A check's remedy (a re-run): sized to its label, never stretched across the card. */
function CheckAction({ action }: { action: Parameters<typeof ActionButton>[0]["action"] }) {
  return <ActionButton action={action} size="small" variant="default" style={{ alignSelf: "flex-start", justifySelf: "start" }} />;
}

/** The ⓘ beside each check's title: what the check measures, technically (glossary). */
const CHECK_TERM: Record<string, TermId> = { accuracy: "r2", ranges: "coverage", distance: "distance_skill", causal: "causal_audit" };

/** One of the four checks: its mark and title (with its ⓘ), one plain result, and the Model details link (the charts live there). */
function CheckRow({
  id,
  title,
  status,
  line,
  more,
  details,
}: {
  id: string;
  title: string;
  status: CheckStatus | "loading";
  line?: ReactNode;
  more?: ReactNode;
  details: string;
}) {
  const tip = CHECK_TERM[id];
  return (
    <article className="card" id={id} data-check={id} data-status={status} aria-labelledby={`${id}-title`}>
      <header>
        <div className="row" style={{ alignItems: "center" }}>
          <Mark status={status} />
          {/* the ⓘ sits beside the heading, not inside it, so the heading's name stays its title */}
          <span>
            <h3 id={`${id}-title`} style={{ margin: 0, display: "inline" }}>
              {title}
            </h3>
            {tip ? <InfoTip term={tip} /> : null}
          </span>
        </div>
        <DetailsLink to={details} />
      </header>
      <SectionBoundary title={title}>
        <div className="stack" style={{ gap: 8 }}>
          {line ? (
            <p className="prose" style={{ margin: 0 }} data-check-line="true">
              {line}
            </p>
          ) : null}
          {more}
        </div>
      </SectionBoundary>
    </article>
  );
}

/** A check whose view is not ready: loading, unreadable, or not in this run (with the server's remedy). */
function PendingCheck<V extends ViewName>({ id, title, res, details, absent }: { id: string; title: string; res: Res<V>; details: string; absent?: string }) {
  const v = viewOf(res);
  if (v.state === "loading") return <CheckRow id={id} title={title} status="loading" details={details} />;
  const action = res.data?.missing.find((m) => m.action)?.action ?? null;
  const line = v.state === "error" ? "This check could not be loaded." : (absent ?? `Not available for this run yet.${v.why ? ` ${v.why}` : ""}`);
  return <CheckRow id={id} title={title} status="unknown" line={line} more={action ? <CheckAction action={action} /> : null} details={details} />;
}

// ---------------------------------------------------------------- 1. the grade

const GRADE_TONE: Record<TrustGrade, "good" | "warn" | "crit"> = { good: "good", fair: "warn", limited: "crit" };

function GradeCard({ grade, loading, live }: { grade: TrustAssessment | null; loading: boolean; live: boolean }) {
  return (
    <section
      className="kpi"
      id="grade"
      aria-labelledby="trust-grade-label"
      data-grade={grade?.grade}
      data-tone={grade ? GRADE_TONE[grade.grade] : undefined}
      style={{ gap: 6, maxWidth: "70ch" }}
    >
      <span className="row" style={{ gap: 4, alignItems: "center" }}>
        <span className="label" id="trust-grade-label">
          Overall
        </span>
        <InfoTip term="trust_grade" />
      </span>
      <span className="value">
        {grade ? (
          grade.label
        ) : loading ? (
          <span className="cap" role="status">
            <span className="spinner" aria-hidden="true" /> Loading…
          </span>
        ) : (
          "Not graded yet"
        )}
      </span>
      <p style={{ margin: 0 }} data-grade-line="true">
        {grade ? grade.line : loading ? "" : live ? "Graded once the model has been tested on areas it did not see." : "This run has no accuracy check, so it cannot be graded."}
      </p>
      <p className="cap" style={{ margin: 0 }} data-grade-rule="true">
        How the grade works: an <Term id="r2">accuracy score</Term> of {fmtNum(TRUST_RULE.good, 1)} or more is Good, {fmtNum(TRUST_RULE.fair, 1)} or more Fair, lower Limited. If the{" "}
        <Term id="coverage">prediction ranges</Term> miss their target by more than {TRUST_RULE.coveragePts} percentage points, the grade drops one step.
      </p>
    </section>
  );
}

// ---------------------------------------------------------------- 2. the four checks

function AccuracyRow({ rid, res, r2, rmse }: { rid: string; res: Res<"accuracy">; r2: number | null; rmse: number | null }) {
  const details = runHref(rid, "details", "accuracy");
  const title = "Accuracy in places it never saw";
  const v = viewOf(res);
  const unit = targetUnit(res.data?.units);
  const final = v.vm ? finalModel(v.vm.sections.models) : null;
  const r2v = isNum(r2) ? r2 : (final?.r2 ?? null);
  const rmsev = isNum(rmse) ? rmse : (final?.rmse ?? null);
  if (!isNum(r2v) && v.state !== "ready") return <PendingCheck id="accuracy" title={title} res={res} details={details} />;
  const c = accuracyCheck(r2v, rmsev, unit);
  return <CheckRow id="accuracy" title={title} status={c.status} line={<TermText text={c.line} terms={ACCURACY_TERMS} />} details={details} />;
}

/** The distance bands whose ranges fall short of their target, in one line (null when none does). */
export function underTargetLine(rows: readonly CoverageRow[], target: number, subject = "the ranges"): string | null {
  const under = rows
    .filter((r) => r.global !== null && r.global < target - TRUST_RULE.coveragePts / 100)
    .map((r) => r.label.replace(/(\d)\s*-\s*(\d)/g, "$1–$2")); // "240-330 m" → "240–330 m"
  if (!under.length) return null;
  return `At ${under.join(", ")} from the nearest reading, ${subject} catch fewer than ${fmtPct(target - TRUST_RULE.coveragePts / 100, 0)} of readings (target: ${fmtPct(target, 0)}).`;
}

function RangesRow({ rid, res, coverage, target }: { rid: string; res: Res<"accuracy">; coverage: number | null; target: number | null }) {
  const details = runHref(rid, "details", "accuracy");
  const title = "Are the ranges honest?";
  const v = viewOf(res);
  const ih: AccuracySections["interval_honesty"] | null = v.vm?.sections.interval_honesty ?? null;
  const cov = isNum(coverage) ? coverage : (ih?.global ?? null);
  const tgt = isNum(target) ? target : (ih?.target ?? null);
  if (!isNum(cov) && v.state !== "ready") return <PendingCheck id="ranges" title={title} res={res} details={details} />;
  const c = rangesCheck(cov, tgt);
  // How the ranges hold farther from the measurements is the distance check's (below).
  return <CheckRow id="ranges" title={title} status={c.status} line={<TermText text={c.line} terms={RANGES_TERMS} />} details={details} />;
}

function DistanceRow({ rid, res, accuracy }: { rid: string; res: Res<"distance">; accuracy: Res<"accuracy"> }) {
  const details = runHref(rid, "details", "reach");
  const title = "Does it hold up away from measurements?";
  const skipped = useSkippedStage(rid, "cv_curve", "The distance test");
  const v = viewOf(res);
  // The ranges' coverage by distance band (the accuracy view): the other half of this check.
  const ih = accuracy.data && accuracy.data.availability !== "missing" ? accuracy.data.sections.interval_honesty : null;
  const bands = ih?.by_distance ?? [];
  const target = isNum(ih?.target) ? ih.target : TRUST_RULE.defaultTarget;
  const far = bands.length ? underTargetLine(bands, target) : null;
  const curve = v.state === "ready" ? v.vm.sections.curve : null;
  const c = v.state === "ready" ? distanceCheck(skillAcross(curve), targetUnit(v.vm.units)) : null;
  if (curve && c) {
    const more = far ? (
      <p className="cap" style={{ margin: 0 }} data-distance-ranges="true">
        {far}
      </p>
    ) : null;
    return <CheckRow id="distance" title={title} status={c.status} line={<TermText text={c.line} terms={ACCURACY_TERMS} />} more={more} details={details} />;
  }
  const accuracyLoading = accuracy.data === undefined && !accuracy.error;
  if (v.state === "loading" || v.state === "error" || accuracyLoading || (v.state === "missing" && !skipped && !bands.length))
    return <PendingCheck id="distance" title={title} res={res} details={details} />;
  const remedy = skipped?.action ?? res.data?.missing.find((m) => m.action)?.action ?? null;
  const action = remedy ? <CheckAction action={remedy} /> : null;
  // Skill was not tested at increasing distances: partly checked when the ranges were.
  const partial = distanceCoverageCheck(bands, target, skipped);
  if (partial) return <CheckRow id="distance" title={title} status="partial" line={partial.line} more={action} details={details} />;
  const line = skipped ? `Not tested in this run. ${skipped.reason}` : "This run did not test its accuracy at increasing distances from the measurements.";
  return <CheckRow id="distance" title={title} status="unknown" line={line} more={action} details={details} />;
}

const LEVER_TONE: Record<CausalAgreement["status"], { tone: "good" | "warn" | "crit" | "neutral"; icon: "check" | "alert" | "minus"; text: string; tip: string }> = {
  agrees: { tone: "good", icon: "check", text: "agrees", tip: "The cause-and-effect check finds the same direction and about the same size of effect as the model." },
  size: { tone: "warn", icon: "alert", text: "different size", tip: "Both say the change cools (or warms), but they disagree on by how much, or on how the effect grows with the amount." },
  conflict: { tone: "crit", icon: "alert", text: "direction differs", tip: "The cause-and-effect check finds the opposite effect: one says the change cools, the other that it warms." },
  unknown: { tone: "neutral", icon: "minus", text: "not checked", tip: "This lever was not part of the cause-and-effect check." },
};

function CausalRow({ rid, res }: { rid: string; res: Res<"causal"> }) {
  const details = runHref(rid, "details", "causal");
  const title = "Cause or coincidence?";
  const v = viewOf(res);
  if (v.state !== "ready") return <PendingCheck id="causal" title={title} res={res} details={details} />;
  const ts = v.vm.sections.treatments ?? {};
  const levers = Object.keys(ts).map((n) => causalAgreement(n, ts[n], v.vm.units.target, v.vm.units.levers));
  const c = causalCheck(levers);
  const more = levers.length ? (
    <ul className="stack" style={{ listStyle: "none", padding: 0, margin: 0, gap: 6 }} aria-label="Cause or coincidence, per lever">
      {levers.map((l) => {
        const t = LEVER_TONE[l.status];
        return (
          <li key={l.lever} className="row" data-lever={l.lever} data-agreement={l.status} style={{ alignItems: "baseline", flexWrap: "nowrap" }}>
            <TipChip tip={t.tip}>
              <Pill tone={t.tone} icon={t.icon}>
                {t.text}
              </Pill>
            </TipChip>
            <span>{l.line}</span>
          </li>
        );
      })}
    </ul>
  ) : null;
  return <CheckRow id="causal" title={title} status={c.status} line={c.line} more={more} details={details} />;
}

// ---------------------------------------------------------------- 3. independent checks

/** Whether the location asks for the study cards (`#validation`, `#study-<kind>`). */
export function wantsStudyCards(hash: string): boolean {
  return hash === "#validation" || hash.startsWith("#study-");
}

function IndependentChecksSection({ rid }: { rid: string }) {
  const { hash } = useRoute();
  // Open when the hash asks for a card; the viewer's own toggle wins until the hash changes.
  const [toggled, setToggled] = useState<{ hash: string; open: boolean } | null>(null);
  const open = toggled && toggled.hash === hash ? toggled.open : wantsStudyCards(hash);
  return (
    <section id="validation" className="composer-section stack" aria-labelledby="trust-validation">
      <h2 id="trust-validation" className="composer-heading">
        {term("independent_checks").label}
      </h2>
      <p className="cap prose" style={{ margin: 0 }}>
        Separate studies that test the run from other angles: simpler reference models, fake layers that should show no effect, planted effects it should recover, alternative
        modelling choices, and a repeat run that should give the same numbers.
      </p>
      <IndependentChecks rid={rid} />
      <details
        className="callout"
        data-tone="info"
        data-study-cards="true"
        open={open}
        onToggle={(e) => {
          const o = e.currentTarget.open;
          if (o !== open) setToggled({ hash, open: o });
        }}
      >
        <summary>Run a check, or see each check's full results</summary>
        {open ? (
          <div className="stack" style={{ marginTop: 10 }}>
            <StudyCards />
          </div>
        ) : null}
      </details>
    </section>
  );
}

// ---------------------------------------------------------------- 4. how wide the ranges are

type UncertaintyRow = NonNullable<UncertaintySections["rows"]>[number];

const VERDICT_ORDER = ["robust", "direction", "not_established", "unknown"] as const;

/** The verdicts of the scenarios, counted, in their plain glossary words (Solid, Direction clear, Not proven). */
function verdictCounts(rows: readonly Pick<UncertaintyRow, "verdict">[]): { label: string; term: TermId | null; n: number }[] {
  const judged = rows.filter((r) => r.verdict?.verdict);
  return VERDICT_ORDER.flatMap((v) => {
    const of = judged.filter((r) => r.verdict!.verdict === v);
    return of.length ? [{ ...verdictWord(v, of[0].verdict!.label), n: of.length }] : [];
  });
}

/** What the evidence supports across the scenarios, in one line ("…, across 3 scenarios: 2 Solid, 1 Direction clear."); null without verdicts. */
export function verdictLine(rows: readonly Pick<UncertaintyRow, "verdict">[]): string | null {
  const parts = verdictCounts(rows);
  if (!parts.length) return null;
  return `What the evidence supports, across ${rows.length} scenario${rows.length === 1 ? "" : "s"}: ${parts.map((p) => `${p.n} ${p.label}`).join(", ")}.`;
}

/** The phrases of the verdict line explained by their glossary terms. */
function verdictTerms(rows: readonly Pick<UncertaintyRow, "verdict">[]): TermPhrase[] {
  return verdictCounts(rows).flatMap((p) => (p.term ? [[p.label, p.term] as const] : []));
}

function RangesSection({ rid, res }: { rid: string; res: Res<"uncertainty"> }) {
  const v = viewOf(res);
  let body: ReactNode;
  if (v.state === "loading")
    body = (
      <p className="cap" role="status">
        <span className="spinner" aria-hidden="true" /> Loading the uncertainty report…
      </p>
    );
  else if (v.state === "error") body = <p className="cap">The uncertainty report could not be loaded.</p>;
  else if (v.state === "missing" || !v.vm.sections.rows?.length)
    body = (
      <p className="cap">
        This run has no uncertainty report yet. <Link to={`${runHref(rid, "trust")}#study-uncertainty`}>Run the uncertainty report</Link> to see how wide each scenario's range is.
      </p>
    );
  else {
    const rows = v.vm.sections.rows;
    const verdicts = verdictLine(rows);
    body = (
      <SectionBoundary title="How wide the ranges are">
        <p className="prose" style={{ margin: 0 }} data-uncertainty-summary="true">
          <TermText text={envelopeCaption(rows)} terms={[["every source of doubt", "envelope"]]} />
        </p>
        {verdicts ? (
          <p className="cap" style={{ margin: 0 }} data-verdicts="true">
            <TermText text={verdicts} terms={verdictTerms(rows)} />
          </p>
        ) : null}
      </SectionBoundary>
    );
  }
  return (
    <section id="uncertainty" className="composer-section stack" aria-labelledby="trust-uncertainty">
      <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline" }}>
        <span>
          <h2 id="trust-uncertainty" className="composer-heading" style={{ display: "inline" }}>
            How wide the ranges are
          </h2>
          <InfoTip term="envelope" />
        </span>
        <DetailsLink to={runHref(rid, "details", "uncertainty")} />
      </div>
      {body}
    </section>
  );
}

// ---------------------------------------------------------------- 5. limitations

function Limitations({ rid, res }: { rid: string; res: Res<"overview"> }) {
  const v = viewOf(res);
  const lim = v.vm?.sections.limitations ?? null;
  const caveats = keepInMind(v.vm ?? undefined).length;
  return (
    <section id="limitations" className="composer-section stack" aria-labelledby="trust-limitations">
      <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline" }}>
        <h2 id="trust-limitations" className="composer-heading">
          Limitations
        </h2>
        <Link to={runHref(rid, "details", "docs/model_card")} className="cap">
          Model card →
        </Link>
      </div>
      {v.state === "loading" ? (
        <p className="cap" role="status">
          <span className="spinner" aria-hidden="true" /> Loading…
        </p>
      ) : lim?.length ? (
        <ul className="prose" style={{ margin: 0, paddingLeft: 18 }} aria-label="Limitations">
          {lim.map((l, i) => (
            <li key={i}>{l}</li>
          ))}
        </ul>
      ) : (
        <p className="cap">The model card lists no limitations.</p>
      )}
      {caveats ? (
        <p className="cap" style={{ margin: 0 }} data-caveats-link="true">
          {caveats === 1 ? "1 more point about this run is" : `${caveats} more points about this run are`} on the Summary, under{" "}
          <Link to={`${runHref(rid)}#${KEEP_IN_MIND_ID}`}>{term("keep_in_mind").label}</Link>.
        </p>
      ) : null}
    </section>
  );
}

// ---------------------------------------------------------------- page

export default function Trust() {
  const rid = useRid();
  const run = useRunDetailFull(rid || null);
  const overview = useView(rid || null, "overview");
  const accuracy = useView(rid || null, "accuracy");
  const distance = useView(rid || null, "distance");
  const causal = useView(rid || null, "causal");
  const uncertainty = useView(rid || null, "uncertainty");
  const live = runIsLive(run.data?.run.status);

  const kpis = overview.data?.sections.kpis;
  const inputs = trustInputs(kpis, run.data?.run);
  const grade = trustGrade(inputs);
  const rmseKpi = kpiOf(kpis, "rmse");
  const rmse = isNum(rmseKpi?.value) ? rmseKpi.value : (run.data?.run.rmse ?? null);
  const gradeLoading = !grade && overview.data === undefined && !overview.error;

  return (
    <div className="stack composer" data-composer="trust">
      <section id="checks" className="composer-section stack" aria-labelledby="trust-title">
        <header className="stack" style={{ gap: 4 }}>
          <h2 id="trust-title" className="composer-heading">
            How much to trust this
          </h2>
          <p className="cap prose" style={{ margin: 0 }}>
            How well the model predicts places it has not seen, whether its ranges hold, and whether its effects stand up to an independent check.
          </p>
        </header>
        <GradeCard grade={grade} loading={gradeLoading} live={live} />
        <div className="stack" style={{ gap: 12 }}>
          <AccuracyRow rid={rid} res={accuracy} r2={isNum(inputs.r2) ? inputs.r2 : null} rmse={rmse} />
          <RangesRow rid={rid} res={accuracy} coverage={isNum(inputs.coverage) ? inputs.coverage : null} target={isNum(inputs.coverageTarget) ? inputs.coverageTarget : null} />
          <DistanceRow rid={rid} res={distance} accuracy={accuracy} />
          <CausalRow rid={rid} res={causal} />
        </div>
      </section>
      <IndependentChecksSection rid={rid} />
      <RangesSection rid={rid} res={uncertainty} />
      <Limitations rid={rid} res={overview} />
    </div>
  );
}
