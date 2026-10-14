// Summary run tab (`/r/:rid`, phase 2): the planner's landing page of a run (the run's name,
// mode and state are in the run header above the tabs); while the run is still going, its stage rail (the full log lives in
// Model details › Run log); four headline cards, each opening its tab: Cooling potential
// (→ Scenario Lab › Library), Residents at heat risk (→ Heat risk › Today), How much to trust
// this (→ its tab), Climate outlook (→ Heat risk › Future); the caveats behind one quiet
// "Things to keep in mind" disclosure, their one home (`#keep-in-mind` opens it; the views no
// longer repeat them, and How much to trust this links here; the model-card limitations are
// linked the other way: their home is How much to trust this); next steps; and the notes pinned
// from this run.
// Run metadata, stage timings, the outputs grid and the accuracy numbers are not here: they
// live in Model details (Provenance, Run log, Accuracy) and How much to trust this.
// Overview.tsx keeps the full technical overview.
import type { ClimateProjection, OverviewSections, RunDetail, ScenarioRow, ViewKpi, ViewModelOf } from "../../api/runs";
import { useRunDetailFull, useView } from "../../api/runs";
import type { Likely, RunSummary } from "../../api/types";
import { JobStrip } from "../../components/ui/JobStrip";
import { StageRail, type StageRailStage } from "../../components/ui/StageRail";
import { trustGrade, type TrustGrade, type TrustInput } from "../../content/trust";
import { runIsLive } from "../../layouts/resources";
import { Link, runHref, useRoute } from "../../router";
import { coolerWarmer, DASH, EMPTY, fmtDateTime, fmtDuration, fmtInt, fmtNum, fmtPct, fmtRange, fmtSigned, unitLabel } from "../../theme/format";
import { stageShort } from "../projects/model/plan";
import { railStatus } from "../tracking/mission/StageRail";
import { useRid } from "./common";
import { isTempUnit, rangeWords, targetUnit } from "./format";

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** A fetched resource as the card builders see it: data, or an error, or neither (still loading). */
export type Loaded<T> = { data: T | undefined; error: unknown };

const pending = (r: Loaded<unknown>) => r.data === undefined && !r.error;

export type HeadlineCardId = "cooling" | "heat" | "trust" | "climate";

/** One headline card. `state` "absent" means the run has no number for it; `line` says why. */
export type HeadlineCard = {
  id: HeadlineCardId;
  label: string;
  state: "ready" | "loading" | "absent";
  value?: string;
  unit?: string;
  line: string;
  tone?: "good" | "warn" | "crit";
  to: string;
  linkText: string;
  grade?: TrustGrade;
};

const kpiOf = (kpis: readonly ViewKpi[] | null | undefined, id: string): ViewKpi | null => kpis?.find((k) => k.id === id) ?? null;

/** A temperature change as value + unit: −1.31 °F → { "1.31", "°F cooler" }; other units keep their sign. */
function changeParts(v: number, unit: string, decimals: number): { value: string; unit: string } {
  if (!isTempUnit(unit)) return { value: fmtSigned(v, decimals), unit };
  const w = coolerWarmer(v, decimals);
  if (w === "no change") return { value: "No change", unit: "" };
  return { value: fmtNum(Math.abs(v), decimals), unit: `${unit} ${w}`.trim() };
}

// ---------------------------------------------------------------- cooling potential

export type CoolingPick = { name: string; likely: Pick<Likely, "estimate" | "lo" | "hi"> };

/**
 * The cooling headline: the run's cooling package (its joint scenario, the overview's `package`
 * KPI), else its headline scenario (`headline` KPI), else the configured scenario that cools
 * the city most (scenarios view). City-mean ΔT with its likely range.
 */
export function coolingPick(kpis: readonly ViewKpi[] | null | undefined, rows: readonly ScenarioRow[] | null | undefined): CoolingPick | null {
  for (const id of ["package", "headline"]) {
    const k = kpiOf(kpis, id);
    if (k && isNum(k.value)) return { name: k.label, likely: { estimate: k.value, lo: k.likely?.lo ?? null, hi: k.likely?.hi ?? null } };
  }
  let best: ScenarioRow | null = null;
  for (const r of rows ?? []) if (isNum(r.delta?.estimate) && (!best || r.delta.estimate < best.delta.estimate)) best = r;
  return best ? { name: best.name, likely: { estimate: best.delta.estimate, lo: best.delta.lo, hi: best.delta.hi } } : null;
}

export function coolingCard(
  rid: string,
  overview: Loaded<ViewModelOf<"overview">>,
  scenarios: Loaded<ViewModelOf<"scenarios">>,
  live: boolean,
): HeadlineCard {
  const base = { id: "cooling", label: "Cooling potential", to: runHref(rid, "lab", "library"), linkText: "See scenarios" } as const;
  const kpis = overview.data?.sections.kpis;
  const pick = coolingPick(kpis, scenarios.data?.availability === "missing" ? null : scenarios.data?.sections.rows);
  if (!pick) {
    if (pending(overview) || (overview.data && !kpiOf(kpis, "package") && !kpiOf(kpis, "headline") && pending(scenarios))) return { ...base, state: "loading", line: "" };
    return { ...base, state: "absent", line: live ? "Appears once the scenarios stage finishes." : "This run has no cooling scenarios." };
  }
  const unit = targetUnit(overview.data?.units ?? scenarios.data?.units);
  const { estimate, lo, hi } = pick.likely;
  const range = isNum(lo) && isNum(hi) ? ` · likely ${rangeWords(lo, hi, unit, 2)}` : "";
  return { ...base, state: "ready", ...changeParts(estimate, unit, 2), line: `${pick.name}, city-wide average${range}`, tone: isNum(hi) && hi < 0 ? "good" : undefined };
}

// ---------------------------------------------------------------- residents at heat risk

export function heatCard(rid: string, heat: Loaded<ViewModelOf<"heat">>, live: boolean): HeadlineCard {
  const base = { id: "heat", label: "Residents at heat risk", to: runHref(rid, "heat"), linkText: "See heat risk" } as const;
  if (pending(heat)) return { ...base, state: "loading", line: "" };
  const vm = heat.data;
  const today = vm && vm.availability !== "missing" ? vm.sections.today : null;
  if (!today || !isNum(today.unadapted?.ec_or_worse)) {
    const needsHumidity = !!vm?.sections.humidity?.needs_input;
    const line = needsHumidity ? "Needs a humidity setting: choose one on Heat risk." : live ? "Appears when the run finishes." : "Not available for this run.";
    return { ...base, state: "absent", line };
  }
  const people = today.measure === "people";
  const ec = today.unadapted.ec_or_worse;
  const share = isNum(today.total) && today.total > 0 ? ` · ${fmtPct(ec / today.total)} of ${fmtInt(today.total)} ${people ? "residents" : "cells"}` : "";
  return { ...base, state: "ready", value: fmtInt(ec), unit: people ? "residents" : "cells", line: `at Extreme caution or worse on the heat index today${share}` };
}

// ---------------------------------------------------------------- trust

/** The trust grade's inputs: the overview's held-out R² and interval coverage, else the run summary's. */
export function trustInputs(kpis: readonly ViewKpi[] | null | undefined, run: Pick<RunSummary, "r2" | "coverage"> | null | undefined): TrustInput {
  const r2 = kpiOf(kpis, "r2");
  const cov = kpiOf(kpis, "coverage");
  return {
    r2: isNum(r2?.value) ? r2.value : (run?.r2 ?? null),
    coverage: isNum(cov?.value) ? cov.value : (run?.coverage ?? null),
    coverageTarget: cov?.target ?? null,
  };
}

const GRADE_TONE: Record<TrustGrade, HeadlineCard["tone"]> = { good: "good", fair: "warn", limited: "crit" };

export function trustCard(rid: string, overview: Loaded<ViewModelOf<"overview">>, run: RunSummary | undefined, live: boolean): HeadlineCard {
  const base = { id: "trust", label: "How much to trust this", to: runHref(rid, "trust"), linkText: "See the checks" } as const;
  const t = trustGrade(trustInputs(overview.data?.sections.kpis, run));
  if (!t) {
    if (pending(overview)) return { ...base, state: "loading", line: "" };
    return { ...base, state: "absent", line: live ? "Graded once the model has been tested on areas it did not see." : "This run has no accuracy check." };
  }
  return { ...base, state: "ready", value: t.label, line: t.line, tone: GRADE_TONE[t.grade], grade: t.grade };
}

// ---------------------------------------------------------------- climate outlook

const MID_CENTURY = "2041-2060";

/** The mid-century warming headline: SSP2-4.5 for 2041–2060, else another scenario for that period. */
export function warmingPick(warming: readonly ClimateProjection[] | null | undefined): ClimateProjection | null {
  const mid = (warming ?? []).filter((p) => p.period === MID_CENTURY && isNum(p.median));
  return mid.find((p) => p.experiment === "ssp245") ?? mid[0] ?? null;
}

export function climateCard(rid: string, climate: Loaded<ViewModelOf<"climate">>, overview: Loaded<ViewModelOf<"overview">>, live: boolean): HeadlineCard {
  const base = { id: "climate", label: "Climate outlook", to: runHref(rid, "heat", "future"), linkText: "See future climate" } as const;
  const vm = climate.data && climate.data.availability !== "missing" ? climate.data : null;
  const p = warmingPick(vm?.sections.warming);
  if (p && isNum(p.median)) {
    const unit = targetUnit(vm?.units);
    const spread = isNum(p.p10) && isNum(p.p90) ? ` · ${fmtRange(p.p10, p.p90, unit, 1)} across ${p.n_models ? fmtInt(p.n_models) : "the"} climate models` : "";
    return { ...base, state: "ready", ...changeParts(p.median, unit, 1), line: `by ${p.period.replace("-", DASH)} (${p.label})${spread}` };
  }
  // The overview's headline number when the climate view cannot say.
  const k = kpiOf(overview.data?.sections.kpis, "warming");
  if (k && isNum(k.value)) {
    const unit = unitLabel(k.unit ?? overview.data?.units.target ?? "");
    return { ...base, state: "ready", ...changeParts(k.value, unit, k.decimals ?? 1), line: k.label };
  }
  if (pending(climate) || pending(overview)) return { ...base, state: "loading", line: "" };
  return { ...base, state: "absent", line: live ? "Appears once the climate stage finishes." : "This run has no climate projections." };
}

// ---------------------------------------------------------------- caveats

/** The run's caveats, once each. The model-card limitations live on How much to trust this (linked). */
export function keepInMind(overview: ViewModelOf<"overview"> | undefined): string[] {
  if (!overview) return [];
  const all = (overview.caveats ?? []).map((s) => s.trim()).filter(Boolean);
  return [...new Set(all)];
}

// ---------------------------------------------------------------- page

function SummaryCard({ card }: { card: HeadlineCard }) {
  const labelId = `summary-card-${card.id}`;
  return (
    <article
      className="kpi summary-card"
      aria-labelledby={labelId}
      data-card={card.id}
      data-state={card.state}
      data-grade={card.grade}
      data-tone={card.state === "ready" ? card.tone : undefined}
      style={{ gridTemplateRows: "auto auto 1fr auto", gap: 6 }}
    >
      <h2 className="label" id={labelId} style={{ margin: 0 }}>
        {card.label}
      </h2>
      <span className="value">
        {card.state === "loading" ? (
          <span className="cap" role="status">
            <span className="spinner" aria-hidden="true" /> Loading…
          </span>
        ) : card.state === "absent" ? (
          EMPTY
        ) : (
          <>
            {card.value}
            {card.unit ? <small>{card.unit}</small> : null}
          </>
        )}
      </span>
      <span className="note">{card.line}</span>
      <Link to={card.to} className="cap summary-card-link">
        {card.linkText} →
      </Link>
    </article>
  );
}

const STOPPED = new Set(["failed", "interrupted", "cancelled"]);

/** The stage rail and job progress while the run is going (or where it stopped). */
function Progress({ rid, detail, live, timings }: { rid: string; detail: RunDetail; live: boolean; timings: OverviewSections["timings"] | null | undefined }) {
  const labels = new Map((timings ?? []).map((t) => [t.stage, t.label]));
  const running = detail.stages.find((s) => railStatus(s.state) === "running");
  const stages: StageRailStage[] = detail.stages.map((s) => {
    const status = railStatus(s.state);
    return {
      id: s.id,
      label: <span title={labels.get(s.id) ?? stageShort(s.id)}>{stageShort(s.id)}</span>,
      status,
      meta: status === "done" && isNum(s.seconds) ? fmtDuration(s.seconds) : undefined,
      data: { stage: s.id },
    };
  });
  return (
    <section className="card stack" aria-label="Run progress" data-progress={live ? "running" : "stopped"} style={{ gap: 10 }}>
      <p style={{ margin: 0 }}>
        {live ? "This run is still running. The headline numbers fill in as each stage finishes." : "This run stopped before it finished."}{" "}
        <Link to={runHref(rid, "details", "runlog")}>{live ? "Follow it in the run log" : "See what happened in the run log"}</Link>
      </p>
      <JobStrip runId={rid} />
      {stages.length ? <StageRail stages={stages} orientation="horizontal" size="sm" label="Pipeline stages" current={running?.id ?? null} /> : null}
    </section>
  );
}

/** A finding's stored `url_state` when it is a same-origin app path ("/r/…?…"), else null. */
function appPath(url: string | null | undefined): string | null {
  return url && url.startsWith("/") && !url.startsWith("//") && !url.startsWith("/\\") ? url : null;
}

function Notes({ rid, findings }: { rid: string; findings: NonNullable<ViewModelOf<"overview">["sections"]["findings"]> }) {
  if (!findings.length) return null;
  return (
    <section className="stack" aria-labelledby="summary-notes" style={{ gap: 6 }}>
      <h3 id="summary-notes" className="kicker" style={{ margin: 0 }}>
        Notes from this run ({findings.length})
      </h3>
      <ul style={{ margin: 0, paddingLeft: 18 }}>
        {findings.map((f) => (
          <li key={f.id}>
            <Link to={appPath(f.url_state) ?? runHref(rid)}>{f.title}</Link> <span className="cap">{fmtDateTime(f.created_utc)}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

/** Anchor of the Summary's "Things to keep in mind" (the run's caveats; the link opens it). */
export const KEEP_IN_MIND_ID = "keep-in-mind";

function KeepInMind({ rid, items, limitations }: { rid: string; items: string[]; limitations: number }) {
  const { hash } = useRoute();
  if (!items.length && !limitations) return null;
  return (
    <details className="summary-caveats cap" id={KEEP_IN_MIND_ID} data-caveats={items.length} open={hash === `#${KEEP_IN_MIND_ID}` || undefined}>
      <summary style={{ cursor: "pointer", width: "max-content", maxWidth: "100%" }}>Things to keep in mind{items.length ? ` (${items.length})` : ""}</summary>
      {items.length ? (
        <ul className="prose" style={{ margin: "6px 0 0", paddingLeft: 18 }}>
          {items.map((c, i) => (
            <li key={i}>{c}</li>
          ))}
        </ul>
      ) : null}
      {limitations ? (
        <p style={{ margin: "6px 0 0" }}>
          The model&apos;s limitations ({limitations}) are on <Link to={`${runHref(rid, "trust")}#limitations`}>How much to trust this</Link>.
        </p>
      ) : null}
    </details>
  );
}

function NextSteps({ rid }: { rid: string }) {
  return (
    <nav className="row summary-next" aria-label="Next steps" style={{ flexWrap: "wrap", gap: 8 }}>
      <span className="kicker">Next steps</span>
      <Link className="btn" to={runHref(rid, "map")}>
        Explore the map
      </Link>
      <Link className="btn" to={runHref(rid, "lab")}>
        Try a scenario
      </Link>
      <Link className="btn" to={runHref(rid, "effects")}>
        See what cools
      </Link>
    </nav>
  );
}

export default function Summary() {
  const rid = useRid();
  const detail = useRunDetailFull(rid);
  const run = detail.data?.run;
  const live = runIsLive(run?.status);
  const overview = useView(rid, "overview");
  const heat = useView(rid, "heat");
  const climate = useView(rid, "climate");
  // The configured scenarios are fetched only when the overview names no package or headline scenario.
  const kpis = overview.data?.sections.kpis;
  const needScenarios = !!overview.data && !kpiOf(kpis, "package") && !kpiOf(kpis, "headline");
  const scenarios = useView(needScenarios ? rid : null, needScenarios ? "scenarios" : null);
  const cards = [
    coolingCard(rid, overview, scenarios, live),
    heatCard(rid, heat, live),
    trustCard(rid, overview, run, live),
    climateCard(rid, climate, overview, live),
  ];
  const showProgress = !!detail.data && (live || STOPPED.has(run?.status ?? ""));
  return (
    <section className="stack runhub-view summary" data-view="summary" aria-label="Summary">
      {showProgress ? <Progress rid={rid} detail={detail.data!} live={live} timings={overview.data?.sections.timings} /> : null}
      <section
        className="summary-cards"
        aria-label="Headline numbers"
        style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(210px, 100%), 1fr))", gap: 12 }}
      >
        {cards.map((c) => (
          <SummaryCard key={c.id} card={c} />
        ))}
      </section>
      <KeepInMind rid={rid} items={keepInMind(overview.data)} limitations={overview.data?.sections.limitations?.length ?? 0} />
      <NextSteps rid={rid} />
      {overview.data?.sections.findings ? <Notes rid={rid} findings={overview.data.sections.findings} /> : null}
    </section>
  );
}
