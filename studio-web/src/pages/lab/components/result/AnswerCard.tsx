// The full-model answer (phase 3): a .vcard whose top rule is cool when the change cools, warm when
// it warms and neutral when it might be no real effect (or has no range); kicker "FULL MODEL RESULT";
// the plain headline; the big number with its likely-range strip on a zero line (RangeStrip); the
// confidence chip (words.CONFIDENCE_WORDS, never crimson for cooling); up to 4 facts (citywide effect ·
// residents moved out of Extreme caution · cost · cost per °F); "Keep in mind" qualifiers and "What it
// buys"; actions Pin to compare · Export decision pack · Show on map.
// Every number here is shown nowhere else in the result: the accordion under the card (ResultView)
// holds the detail behind it, never the same figure again. The cost shows once, here.
import { useId, useState, type CSSProperties, type ReactNode } from "react";
import { errorMessage } from "../../../../api/client";
import { exportPack, useLevers, type Lever, type Result } from "../../../../api/lab";
import type { Likely } from "../../../../api/types";
import { Badge } from "../../../../components/ui/Badge";
import { Button } from "../../../../components/ui/Button";
import { CONFIDENCE_TEXT, confidenceOf, type Confidence } from "../../../../components/ui/PlainResult";
import { useJobs } from "../../../../stores/jobs";
import { toast } from "../../../../stores/ui";
import { fmtDateTime, fmtInt, fmtRange, fmtSigned, fmtSignedValue, unitLabel } from "../../../../theme/format";
import { CAUSAL_QUALIFIER_LINK, answerHeadlineText, buysLines, extrapolationQualifier, isCausalQualifier, plainServerBuys, plainWording, withRange } from "../../model/plain";
import { costWords } from "../../model/quick";
import { useTray } from "../../model/tray";
import { CONFIDENCE_WORDS, LAB_WORDS, leverWords, type ConfidenceTone } from "../../model/words";
import "./result.css";

/**
 * Plain words of the result (the answer card and the accordion). They live here until phase 4 folds
 * them, with model/words.ts, into the shared glossary.
 */
export const RESULT_WORDS = {
  citywide: "Citywide",
  citywideNote: "averaged over the whole city",
  residentsOut: "Residents out of Extreme caution",
  placesOut: "Places out of Extreme caution",
  residentsIn: "Residents pushed into Extreme caution",
  placesIn: "Places pushed into Extreme caution",
  heatNote: "on a hot afternoon today (heat index of 90 °F or more)",
  heatCounting: "Counting residents…",
  heatCountingNote: "Working out who moves out of each heat-risk level (a few seconds).",
  heatFailed: "Could not count the residents",
  tryAgain: "Try again",
  likelyRange: "likely range",
  cost: "Estimated cost",
  costNote: "in your cost model's units",
  costPer: (u: string) => `Cost per ${u || "degree"}`,
  costPerNote: (u: string) => `to cool one place by 1 ${u || "degree"} (cooling added up over every place)`,
  costPerWarms: "No cooling to buy",
  costPerWarmsNote: "this change warms on balance",
  keepInMind: "Keep in mind",
  buys: "What it buys",
  stale: "The model has changed since this result was worked out. It stays viewable; run the full model again to refresh it.",
  revision: (n: number) => `revision ${n}`,
  // the accordion (ResultView)
  where: "Where it cools",
  whereTeaser: "How far the cooling reaches, by area, and the map layers",
  heat: "Heat risk & people",
  heatTeaser: "Residents by heat-risk level, who benefits, hot days",
  future: "Against future warming",
  futureTeaser: "How much of the coming warming this cancels out",
  changed: "What was actually changed",
  changedTeaser: "What you asked for and what fits the space",
  sure: "How sure are we",
  sureTeaser: "Quick estimate vs full model, ranges and cross-checks",
  tech: "Technical details",
  techTeaser: "For analysts: the statistics behind these numbers",
} as const;

/** The card's accent by confidence tone (cooling is the cool ramp end, never the crimson accent). */
const ACCENT: Record<ConfidenceTone, string> = { cool: "var(--cool-2)", warm: "var(--warm-2)", neutral: "var(--line-strong)" };

/** The main Likely of a result (where you made changes, else citywide) with its 95% range filled in. */
export function mainLikely(r: Pick<Result, "summary" | "city">): Likely {
  return withRange(r.summary.edited ?? r.city);
}

/** Confidence of the main Likely, from its range (PlainResult.confidenceOf: the same rule as the strip). */
export function answerConfidence(r: Pick<Result, "summary" | "city">): Confidence {
  return confidenceOf(mainLikely(r));
}

/** True when the card shows the "partly beyond the conditions measured" qualifier (the share then shows nowhere else). */
export function cardShowsExtrapolation(r: Pick<Result, "summary" | "extrapolated_edited">): boolean {
  return extrapolationQualifier(r.extrapolated_edited ?? r.summary.frac_extrapolated_edited) !== null;
}

/**
 * A real cost model: some changed lever has its own price (the scenario's `costs`, or a catalogue
 * `cost_per_unit` other than the default 1). Without one, `cost.total` only adds up the amounts.
 */
export function resultHasCost(r: Pick<Result, "cost" | "spec">, levers: readonly Pick<Lever, "var" | "cost_per_unit">[] | undefined): boolean {
  if (!(Number.isFinite(r.cost.total) && r.cost.total > 0)) return false;
  const costs = (r.spec as { costs?: unknown }).costs;
  const priced = costs && typeof costs === "object" ? (costs as Record<string, unknown>) : null;
  return Object.keys(r.cost.per_lever).some((v) => {
    if (priced && v in priced) return true;
    const c = levers?.find((l) => l.var === v)?.cost_per_unit;
    return typeof c === "number" && Number.isFinite(c) && c !== 1;
  });
}

/** The changed places of a result (its automatic "edited cells" area), if it lists them. */
function editedArea(r: Pick<Result, "regions">) {
  return r.regions?.find((g) => g.auto && /edited/i.test(g.name) && !/ring/i.test(g.name)) ?? null;
}

/**
 * The changes cover the whole city: the changed area holds every place of the run (`nCells`, the grid's
 * cell count), or the result has no separate figure for the changed places.
 */
export function editsCoverCity(r: Pick<Result, "summary" | "regions">, nCells?: number | null): boolean {
  if (!r.summary.edited) return true;
  const area = editedArea(r);
  return !!area && typeof nCells === "number" && nCells > 0 && area.n_cells >= nCells;
}

/**
 * The card's headline, without numbers ("Cools the places you changed"): the big number and the
 * likely-range strip under it carry the figures, so they show once.
 */
export function answerHeadline(r: Pick<Result, "summary" | "city" | "regions">, nCells?: number | null): string {
  const s = r.summary;
  const subject = !s.edited ? "the city" : editsCoverCity(r, nCells) ? "the whole city" : "the places you changed";
  return answerHeadlineText(withRange(s.edited ?? r.city), subject);
}

/**
 * The citywide effect, as the card's fact: null when it would repeat the big number (the changes
 * cover the whole city, or the two round to the same figure at the big number's 2 decimals).
 */
export function citywideFact(r: Pick<Result, "summary" | "city" | "regions">, unit: string, nCells?: number | null): string | null {
  const s = r.summary;
  if (!s.edited || editsCoverCity(r, nCells)) return null;
  const city = fmtSignedValue(r.city.estimate, unit, 2);
  return city === fmtSignedValue(s.edited.estimate, unit, 2) ? null : city;
}

/** "Keep in mind" lines: the plain qualifiers plus the server's own that say something new. */
export function answerQualifiers(r: Result, unit: string): string[] {
  const s = r.summary;
  const cc = r.causal_check;
  const causalDisagrees = cc ? !cc.model_within : false;
  // the check is of the citywide change: its direction is worded against the model's citywide estimate
  const causal = cc ? { delta: cc.delta, estimate: r.city.estimate } : null;
  const wording = plainWording({ edited: s.edited, city: r.city, unit, fracExtrapolatedEdited: r.extrapolated_edited ?? s.frac_extrapolated_edited, causalDisagrees, causal });
  const known = new Set(wording.qualifiers.map((q) => q.toLowerCase()));
  // the server's own wording of the same points (extrapolation, the causal check, preview only) is left out
  const extra = r.plain.qualifiers.filter((q) => !known.has(q.toLowerCase()) && !/outside observed conditions|beyond the conditions|causal check|cross-check|preview only|edited cells/i.test(q));
  return [...wording.qualifiers, ...extra];
}

/** The today row of the heat-risk impacts: the card's "Residents out of Extreme caution" fact (null without one). */
export function heatFactRow(r: Pick<Result, "impacts">) {
  const heat = r.impacts?.heat ?? null;
  return heat ? (heat.rows.find((x) => x.case === "today") ?? heat.rows[0] ?? null) : null;
}

/**
 * "What it buys" lines, without the cost line (the cost and the cost per degree are facts of the card).
 * When the card shows the residents-out-of-Extreme-caution fact (or is still counting it), the
 * residents-by-air-temperature line is left out: one claim about residents, one number. Before the
 * impacts are in, the server's own lines stand in, in plain words (plain.plainServerBuys).
 */
export function answerBuys(r: Result, unit: string): string[] {
  const lines = buysLines({ cost: { ...r.cost, cooling_per_cost: null }, impacts: r.impacts }, unit, { residents: !heatFactRow(r) });
  if (lines.length || r.impacts) return lines;
  return plainServerBuys(r.plain.buys);
}

/**
 * Bring the answer card into view (a result that just finished, "see the result below"). With `focus`
 * the card also takes the keyboard focus, so screen readers read it next. False when there is no card yet.
 */
export function revealAnswerCard(opts: { focus?: boolean } = {}): boolean {
  const card = typeof document !== "undefined" ? document.querySelector<HTMLElement>(".answer-card") : null;
  if (!card) return false;
  if (typeof card.scrollIntoView === "function") card.scrollIntoView({ block: "start", behavior: "smooth" });
  if (opts.focus) card.focus({ preventScroll: true });
  return true;
}

/** Show a result layer on the Design map and bring the map into view. */
export function showLayer(onShowLayer: ((key: string) => void) | undefined, key: string): void {
  if (!onShowLayer) return;
  onShowLayer(key);
  const map = typeof document !== "undefined" ? document.querySelector<HTMLElement>(".lab-map") : null;
  if (map && typeof map.scrollIntoView === "function") map.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

// ---------------------------------------------------------------- the likely-range strip

export type RangeStripProps = {
  estimate: number;
  lo: number | null;
  hi: number | null;
  unit: string;
  confidence: Confidence;
};

const finite = (v: number | null | undefined): v is number => typeof v === "number" && Number.isFinite(v);

/** Text alternative of the strip: "Likely range −0.80 to −0.44 °F; the whole range is below zero. Confident it cools." */
export function rangeStripText({ estimate, lo, hi, unit, confidence }: RangeStripProps): string {
  const dec = stripDecimals(estimate, lo, hi);
  const range = finite(lo) && finite(hi) ? `Likely range ${fmtRange(lo, hi, unit, dec)}` : `Estimate ${fmtSignedValue(estimate, unit, dec)}, no likely range`;
  const zero =
    confidence === "confident_cools" ? "the whole range is below zero" : confidence === "confident_warms" ? "the whole range is above zero" : confidence === "could_be_zero" ? "the range crosses zero" : "";
  return `${range}${zero ? `; ${zero}` : ""}. ${CONFIDENCE_TEXT[confidence]}`;
}

function stripDecimals(...vals: (number | null)[]): number {
  const m = Math.max(0, ...vals.filter(finite).map(Math.abs));
  return m > 0 && m < 0.1 ? 3 : 2;
}

/**
 * The likely range on a zero line (SVG): the bar spans lo–hi under a visible "likely range" label, the
 * dot is the estimate, the vertical rule is zero ("no change", labelled 0 where the end labels leave
 * room). Cool tint left of zero, warm right of it, so a range that might be no real effect visibly
 * straddles the line. Root data: `data-tone` (cool | warm | neutral) and `data-straddles="true"` when
 * lo < 0 < hi; the label is `.answer-strip-label`.
 */
export function RangeStrip(p: RangeStripProps) {
  const titleId = useId();
  const { estimate, lo, hi, confidence } = p;
  const W = 320;
  const PAD = 16;
  const Y = 36; // the bar's centre line (the direction words, then the "likely range" label, above it)
  const H = Y + 28;
  const LABEL_HALF = 34; // half the width of "likely range" at the strip's type size
  const hasRange = finite(lo) && finite(hi);
  const vals = [0, estimate, ...(hasRange ? [lo, hi] : [])].filter(finite);
  let min = Math.min(...vals);
  let max = Math.max(...vals);
  const span = max - min || Math.abs(estimate) || 1;
  min -= span * 0.14;
  max += span * 0.14;
  const x = (v: number) => PAD + ((v - min) / (max - min)) * (W - 2 * PAD);
  const x0 = x(0);
  const tone = CONFIDENCE_WORDS[confidence].tone;
  const straddles = hasRange && lo < 0 && hi > 0;
  const dec = stripDecimals(estimate, lo, hi);
  const clampX = (v: number) => Math.min(W - PAD - 18, Math.max(PAD + 18, v));
  const bars: { side: "cool" | "warm"; a: number; b: number }[] = [];
  if (hasRange) {
    if (lo < 0) bars.push({ side: "cool", a: lo, b: Math.min(hi, 0) });
    if (hi > 0) bars.push({ side: "warm", a: Math.max(lo, 0), b: hi });
  }
  const tight = hasRange && x(hi) - x(lo) < 64;
  // "likely range" over the middle of the bar, kept inside the strip (the zero rule stops below it)
  const labelX = hasRange ? Math.min(W - PAD - LABEL_HALF, Math.max(PAD + LABEL_HALF, (x(lo) + x(hi)) / 2)) : null;
  return (
    <svg
      className="answer-strip"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-labelledby={titleId}
      data-tone={tone}
      data-straddles={straddles || undefined}
      data-lo={hasRange ? lo : undefined}
      data-hi={hasRange ? hi : undefined}
    >
      <title id={titleId}>{rangeStripText(p)}</title>
      {/* direction words, from the zero line outwards, where there is room */}
      {x0 - PAD > 56 ? (
        <text x={x0 - 6} y={10} textAnchor="end">
          ← cooler
        </text>
      ) : null}
      {W - PAD - x0 > 56 ? (
        <text x={x0 + 6} y={10} textAnchor="start">
          warmer →
        </text>
      ) : null}
      {labelX !== null ? (
        <text className="answer-strip-label" x={labelX} y={Y - 10} textAnchor="middle">
          {RESULT_WORDS.likelyRange}
        </text>
      ) : null}
      <line className="answer-strip-axis" x1={PAD} x2={W - PAD} y1={Y} y2={Y} />
      {bars.map((b) => (
        <rect key={b.side} className="answer-strip-bar" data-side={b.side} x={x(b.a)} y={Y - 5} width={Math.max(1, x(b.b) - x(b.a))} height={10} rx={2} />
      ))}
      <line className="answer-strip-zero" x1={x0} x2={x0} y1={Y - 9} y2={Y + 12} />
      <circle className="answer-strip-dot" cx={x(estimate)} cy={Y} r={6} />
      {!hasRange || (Math.abs(x0 - x(lo)) > 34 && Math.abs(x0 - x(hi)) > 34 && !(tight && Math.abs(x0 - (x(lo) + x(hi)) / 2) < 60)) ? (
        <text x={x0} y={Y + 24} textAnchor="middle">
          0
        </text>
      ) : null}
      {hasRange ? (
        tight ? (
          <text className="answer-strip-end" x={clampX((x(lo) + x(hi)) / 2)} y={Y + 24} textAnchor="middle">
            {fmtSigned(lo, dec)} to {fmtSigned(hi, dec)}
          </text>
        ) : (
          <>
            <text className="answer-strip-end" x={clampX(x(lo))} y={Y + 24} textAnchor="middle">
              {fmtSigned(lo, dec)}
            </text>
            <text className="answer-strip-end" x={clampX(x(hi))} y={Y + 24} textAnchor="middle">
              {fmtSigned(hi, dec)}
            </text>
          </>
        )
      ) : null}
    </svg>
  );
}

// ---------------------------------------------------------------- the card

/**
 * Working out the impacts (residents by heat-risk level, futures): ResultView computes them once on its
 * own when a result arrives without them, and the card and two accordion sections share this state.
 * `compute` is the retry after an error. `noData`: the run has no resident data (the server's
 * `needs_layers`), so there is nothing to count and nothing to retry.
 */
export type ImpactsState = { busy: boolean; error: Error | null; compute: () => void; noData?: boolean };

/**
 * Props of the answer card.
 * - `result` is the loaded exact result (ResultView loads it with `useResult(resId, rid)`).
 * - `pid` null disables "Export decision pack" (the pack is a project export).
 * - `onShowLayer` shows a result layer on the Design map (`res:${resId}:delta` for "Show on map");
 *   absent hides that action.
 * - `impacts` (optional): while the result has no impacts yet, the residents fact reads "Counting
 *   residents…" (ResultView works them out on its own), offers "Try again" after an error, and is left
 *   out when the run has no resident data (`noData`); absent, the fact waits for the result's impacts.
 * - `nCells` (optional): the run's cell count, so a change of the whole city reads as one ("Cools the
 *   whole city", no separate citywide fact).
 * - `onOpenSection` (optional): opens and scrolls to an accordion section of ResultView; the "Keep in
 *   mind" cross-check line links "How sure are we" through it.
 * Facts come from `result` only: `summary.edited ?? city` (main number), `city` (citywide, unless it
 * repeats the main number, see citywideFact),
 * `impacts.heat.rows[today].ec_avoided` (residents out of Extreme caution), `cost.total` and
 * `cost.cooling_per_cost` (shown only with a real cost model, see resultHasCost), `plain.headline`,
 * `plain.qualifiers`, `plain.buys` / plain.buysLines. The headline is numberless (answerHeadline).
 * DOM contract: the body is `.plain-result` with the headline in `.headline` (e2e: `.result-inspector
 * .plain-result .headline`); the root has `data-tone` = cool | warm | neutral; the DEMO badge is a `.badge`.
 */
export type AnswerCardProps = {
  rid: string;
  resId: string;
  pid: string | null;
  result: Result;
  unit: string;
  onShowLayer?: (key: string) => void;
  impacts?: ImpactsState;
  nCells?: number | null;
  onOpenSection?: (id: "sure") => void;
};

type Fact = { id: string; label: string; value: ReactNode; note?: ReactNode };

export function AnswerCard({ rid, resId, pid, result: r, unit, onShowLayer, impacts, nCells, onOpenSection }: AnswerCardProps) {
  const pin = useTray((s) => s.pin);
  const levers = useLevers(rid).data;
  const [packBusy, setPackBusy] = useState(false);
  const s = r.summary;
  const u = unitLabel(unit);
  const main = mainLikely(r);
  const confidence = confidenceOf(main);
  const conf = CONFIDENCE_WORDS[confidence];
  const headline = answerHeadline(r, nCells);
  const headId = `lab-answer-${resId}`;
  const qualifiers = answerQualifiers(r, unit);
  const buys = answerBuys(r, unit);

  const facts: Fact[] = [];
  const city = citywideFact(r, unit, nCells);
  if (city) facts.push({ id: "city", label: RESULT_WORDS.citywide, value: city, note: RESULT_WORDS.citywideNote });
  const heat = r.impacts?.heat ?? null;
  const today = heatFactRow(r);
  if (heat && today) {
    const people = heat.measure === "people";
    const into = today.ec_avoided < 0;
    facts.push({
      id: "heat",
      label: into ? (people ? RESULT_WORDS.residentsIn : RESULT_WORDS.placesIn) : people ? RESULT_WORDS.residentsOut : RESULT_WORDS.placesOut,
      value: fmtInt(Math.abs(today.ec_avoided)),
      note: RESULT_WORDS.heatNote,
    });
  } else if (!r.impacts && impacts && !impacts.noData) {
    // worked out on its own as the result arrives: "Counting residents…" until then; a retry after an error
    facts.push(
      impacts.error
        ? {
            id: "heat",
            label: RESULT_WORDS.residentsOut,
            value: (
              <Button size="small" icon="refresh" busy={impacts.busy} onClick={impacts.compute}>
                {RESULT_WORDS.tryAgain}
              </Button>
            ),
            note: `${RESULT_WORDS.heatFailed}: ${errorMessage(impacts.error)}`,
          }
        : {
            id: "heat",
            label: RESULT_WORDS.residentsOut,
            value: (
              <span className="answer-fact-pending" role="status">
                {RESULT_WORDS.heatCounting}
              </span>
            ),
            note: RESULT_WORDS.heatCountingNote,
          },
    );
  }
  if (resultHasCost(r, levers)) {
    const perLever = Object.entries(r.cost.per_lever).filter(([, c]) => Number.isFinite(c) && c > 0);
    facts.push({
      id: "cost",
      label: RESULT_WORDS.cost,
      value: costWords(r.cost.total),
      note: perLever.length > 1 ? perLever.map(([v, c]) => `${leverWords(levers?.find((l) => l.var === v), v).name} ${costWords(c).replace(/^about /, "")}`).join(" · ") : RESULT_WORDS.costNote,
    });
    const cpc = r.cost.cooling_per_cost;
    if (cpc !== null && Number.isFinite(cpc)) {
      facts.push(
        cpc > 0
          ? { id: "per", label: RESULT_WORDS.costPer(u), value: costWords(1 / cpc), note: RESULT_WORDS.costPerNote(u) }
          : { id: "per", label: RESULT_WORDS.costPer(u), value: RESULT_WORDS.costPerWarms, note: RESULT_WORDS.costPerWarmsNote },
      );
    }
  }

  const exportDecision = async () => {
    if (!pid) return;
    setPackBusy(true);
    try {
      const { job } = await exportPack(pid, "decision_pack", { result_id: resId });
      useJobs.getState().upsert(job);
      toast("info", "Building the decision pack", { href: `/jobs/${job.id}`, linkLabel: "Track" });
    } catch (e) {
      toast("error", "Could not start the decision pack", { body: errorMessage(e) });
    } finally {
      setPackBusy(false);
    }
  };

  return (
    <article className="vcard answer-card" tabIndex={-1} data-tone={conf.tone} data-confidence={confidence} aria-labelledby={headId} style={{ "--vcard-accent": ACCENT[conf.tone] } as CSSProperties}>
      <div className="vcard-head">
        <span className="kicker">{LAB_WORDS.fullModelKicker}</span>
        <span className="vcard-meta">
          {r.scenario ? <span>{r.scenario.name}</span> : null}
          {r.scenario ? <span>{RESULT_WORDS.revision(r.scenario.revision)}</span> : null}
          <span>{fmtDateTime(s.created_utc)}</span>
          {r.demo ? <Badge tone="demo">DEMO</Badge> : null}
          {r.stale || s.stale ? <Badge tone="warn">stale</Badge> : null}
        </span>
      </div>
      <div className="vcard-body answer-body plain-result">
        <p className="answer-headline headline" id={headId}>
          {headline}
        </p>
        {r.stale || s.stale ? (
          <p className="callout" role="status" data-stale="true">
            {RESULT_WORDS.stale}
          </p>
        ) : null}
        <div className="answer-hero">
          <div className="answer-number">
            <p className="answer-big" data-estimate={main.estimate}>
              {fmtSignedValue(main.estimate, unit, 2)}
            </p>
            <span className="answer-where">{s.edited && !editsCoverCity(r, nCells) ? LAB_WORDS.whereChanged : LAB_WORDS.citywide}</span>
            <span className="pill answer-chip" data-conf={conf.tone} title={conf.tip}>
              {conf.label}
            </span>
          </div>
          <RangeStrip estimate={main.estimate} lo={main.lo} hi={main.hi} unit={unit} confidence={confidence} />
        </div>
        {facts.length ? (
          <dl className="answer-facts" aria-label="Key facts">
            {facts.slice(0, 4).map((f) => (
              <div className="answer-fact" key={f.id} data-fact={f.id}>
                <dt>{f.label}</dt>
                <dd className="answer-fact-value">{f.value}</dd>
                {f.note ? <dd className="answer-fact-note">{f.note}</dd> : null}
              </div>
            ))}
          </dl>
        ) : null}
        {qualifiers.length || buys.length ? (
          <div className="answer-notes">
            {qualifiers.length ? (
              <section aria-label={RESULT_WORDS.keepInMind}>
                <p className="kicker">{RESULT_WORDS.keepInMind}</p>
                <ul>
                  {qualifiers.map((q, i) =>
                    isCausalQualifier(q) && onOpenSection ? (
                      <li key={i} data-qualifier="causal">
                        {q.slice(0, -CAUSAL_QUALIFIER_LINK.length)}
                        <button type="button" className="answer-link" onClick={() => onOpenSection("sure")}>
                          {CAUSAL_QUALIFIER_LINK}
                        </button>
                      </li>
                    ) : (
                      <li key={i}>{q}</li>
                    ),
                  )}
                </ul>
              </section>
            ) : null}
            {buys.length ? (
              <section aria-label={RESULT_WORDS.buys}>
                <p className="kicker">{RESULT_WORDS.buys}</p>
                <ul>
                  {buys.map((b, i) => (
                    <li key={i}>{b}</li>
                  ))}
                </ul>
              </section>
            ) : null}
          </div>
        ) : null}
        <div className="row answer-actions" role="group" aria-label="Result actions">
          <Button size="small" icon="pin" onClick={() => (pin(rid, { kind: "result", id: resId }, r.scenario?.name ?? resId) ? toast("success", "Pinned to the compare tray") : toast("warning", "The compare tray holds 4 items"))}>
            {LAB_WORDS.pinToCompare}
          </Button>
          <Button size="small" icon="download" busy={packBusy} disabled={!pid} onClick={() => void exportDecision()} title="A self-contained brief with maps, tables and GIS files">
            {LAB_WORDS.exportPack}
          </Button>
          {onShowLayer ? (
            <Button size="small" variant="ghost" icon="layers" onClick={() => showLayer(onShowLayer, `res:${resId}:delta`)}>
              {LAB_WORDS.showOnMap}
            </Button>
          ) : null}
        </div>
      </div>
    </article>
  );
}
