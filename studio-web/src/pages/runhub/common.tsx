// Shared run-hub building blocks (SPEC §6.4): the ViewModel page wrapper (live banner,
// missing-output empty state in one sentence with the server's action, the run-wide caveats only
// when asked: their home is the Summary), per-section placeholders
// (older code, or the known reason a section is absent), KPI tiles, flags, generic tables and
// the Findings pin for non-chart blocks. Plain words with tooltips (phase 4): status chips explain
// themselves on hover, focus or tap (TipChip, never a native title), KPI tiles and block titles take
// an ⓘ (InfoTip), and a Model details section can put an ⓘ beside its page's title (TitleTipProvider).
import { Component, createContext, useContext, useState, type ErrorInfo, type ReactNode } from "react";
import { api, errorMessage } from "../../api/client";
import type { GenericTable, MissingOutput, Sections, Verdict, ViewFlag, ViewKpi, ViewModelOf, ViewName, ViewParams, ViewSections, ViewUnits } from "../../api/runs";
import { useRunDetailFull, useView } from "../../api/runs";
import type { Action, Availability, Finding, FindingCreate, StageId } from "../../api/types";
import { Badge } from "../../components/ui/Badge";
import { ActionButton, EmptyState } from "../../components/ui/EmptyState";
import { Icon } from "../../components/ui/Icon";
import { InfoTip, type InfoTipProps } from "../../components/ui/InfoTip";
import { Kpi, KpiRow } from "../../components/ui/Kpi";
import { Pill } from "../../components/ui/Pill";
import { Table, type Column } from "../../components/ui/Table";
import { useTooltip } from "../../components/ui/Tooltip";
import { term } from "../../content/glossary";
import { runIsLive } from "../../layouts/resources";
import { useRoute } from "../../router";
import { toast, useUi } from "../../stores/ui";
import { planReasonText } from "../projects/model/plan";
import { cellText, kpiDisplay, missingText, producedByText, tableColumns } from "./format";

/** The `:rid` of the current run route. */
export function useRid(): string {
  return useRoute().params.rid ?? "";
}

const DEFAULT_UNITS: ViewUnits = { target: "", levers: {} };
const UnitsCtx = createContext<ViewUnits>(DEFAULT_UNITS);

/** Units of the current ViewModel (target and levers). */
export function useUnits(): ViewUnits {
  return useContext(UnitsCtx);
}

export function UnitsProvider({ units, children }: { units: ViewUnits; children: ReactNode }) {
  return <UnitsCtx.Provider value={units}>{children}</UnitsCtx.Provider>;
}

/** Whether the run is still being written (a job is running on it). */
export function useRunIsLive(rid: string): boolean {
  const d = useRunDetailFull(rid);
  return runIsLive(d.data?.run.status);
}

/** The "live" banner on mid-run tabs: the view refreshes as outputs are written. */
export function LiveBanner({ children }: { children?: ReactNode }) {
  return (
    <div className="callout live-banner" data-tone="info" role="status" data-live="true">
      <Badge tone="accent">Live</Badge> {children ?? "This run is still running. The view refreshes as each output is written."}
    </div>
  );
}

/** What "older code" means, in the ⓘ beside the placeholder. */
const OLDER_CODE_TIP = "This run was made with an earlier version of SPARC that did not write this part. Running it again with the current version adds it.";

/** Placeholder for a section an older-code run does not have. */
export function OlderCode({ title }: { title: ReactNode }) {
  return (
    <div className="callout older-code" data-tone="info" data-older-code="true">
      <strong>{title}</strong>: not in this run (older code)
      <InfoTip label="older code" text={OLDER_CODE_TIP} />
    </div>
  );
}

/**
 * A status chip (badge, pill) that explains itself on hover (after a short delay), keyboard focus
 * or tap, in place of a native `title`: the tooltip exists only while open (in a portal), so it adds
 * nothing to the page's text. Without a tip the chip renders as it is.
 */
export function TipChip({ tip, children }: { tip: ReactNode; children: ReactNode }) {
  const t = useTooltip<HTMLSpanElement>({ content: tip });
  if (tip === null || tip === undefined || tip === "") return <>{children}</>;
  return (
    // focusable so keyboard users can read the tip too (it is not a control)
    <span ref={t.anchorRef} className="tip-anchor" tabIndex={0} data-tip-chip="" {...t.triggerProps}>
      {children}
      {t.bubble}
    </span>
  );
}

/** The ⓘ a Model details section puts beside its page's title (ViewPage shows it next to a level-2 title). */
const TitleTipCtx = createContext<ReactNode>(null);

/** Gives the pages inside a Model details section the ⓘ for their title. */
export function TitleTipProvider({ tip, children }: { tip: ReactNode; children: ReactNode }) {
  return <TitleTipCtx.Provider value={tip}>{children}</TitleTipCtx.Provider>;
}

/** Why a section is absent when the reason is known (a stage the run skipped, data the run lacks): one sentence. */
/** `why`: a skipped stage's reason as a phrase ("turned off in the project settings"), for callers that word their own sentence. */
export type Absence = { reason: ReactNode; action?: Action | null; why?: string | null };

/** Placeholder for a section absent for a known reason, with its remedy when there is one. */
export function NotInRun({ title, reason, action }: { title: ReactNode } & Absence) {
  return (
    <div className="callout" data-tone="info" data-absent="true">
      <strong>{title}</strong>: not in this run. {reason}
      {action ? (
        <>
          {" "}
          <ActionButton action={action} size="small" variant="default" />
        </>
      ) : null}
    </div>
  );
}

const SKIPPED_STATES = new Set(["skipped", "disabled", "not_requested"]);

/**
 * Why `stage` did not run in this run, from the run's stage rows (`RunDetail.stages`, the
 * plan's skip reason), with the remedy the Overview's outputs grid offers for it ("Re-run
 * with …"); `what` names the stage at the start of a sentence ("The baselines stage"). Null
 * when the stage ran or is pending, or when the run cannot say: an older manifest without a
 * stage list reads "not in this run", and its sections stay "older code".
 */
export function useSkippedStage(rid: string, stage: StageId, what: string): Absence | null {
  const row = useRunDetailFull(rid).data?.stages.find((s) => s.id === stage);
  const skipped = !!row && SKIPPED_STATES.has(row.state) && row.reason !== "not in this run";
  const overview = useView(skipped ? rid : null, skipped ? "overview" : null);
  if (!skipped) return null;
  const why = row.reason && row.reason !== "not run" ? planReasonText(row.reason) : null;
  const action = overview.data?.sections.outputs_grid?.find((o) => o.produced_by === `stage:${stage}` && o.action)?.action ?? null;
  return { reason: why ? `${what} did not run: ${why}.` : `${what} did not run.`, action, why };
}

type BoundaryProps = { title: ReactNode; children: ReactNode };

/** Keeps one malformed section from taking the whole view down. */
export class SectionBoundary extends Component<BoundaryProps, { error: Error | null }> {
  override state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    console.warn("[run hub] section failed to render", error, info.componentStack);
  }

  override render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="callout" data-tone="crit" role="alert">
        <strong>{this.props.title}</strong>: this section could not be displayed ({this.state.error.message}).
      </div>
    );
  }
}

/**
 * One ViewModel section: renders `children(data)` when the section is present. A null section
 * says why when the caller knows (`absent`: a stage the config disabled, data the run has no
 * column for), else it is the older-code placeholder.
 */
export function Section<T>({
  title,
  data,
  absent,
  children,
}: {
  title: ReactNode;
  data: T | null | undefined;
  absent?: Absence | null;
  children: (d: NonNullable<T>) => ReactNode;
}) {
  if (data === null || data === undefined) return absent ? <NotInRun title={title} {...absent} /> : <OlderCode title={title} />;
  return <SectionBoundary title={title}>{children(data as NonNullable<T>)}</SectionBoundary>;
}

/** One plain sentence per producer whose outputs a view is missing (all of them share its remedy). */
const PRODUCER_SENTENCE: Record<string, string> = {
  "post:planner": "Run the planner pack to see where to act, who is exposed, and the GIS files.",
  "post:uncertainty": "Add up every source of uncertainty to see how wide each scenario's likely range is.",
  "post:baselines": "Score the simpler comparison models to see whether this model beats them.",
  "post:writeup": "Regenerate the methods and model card to see them.",
  "post:emulator": `Set up the ${term("emulator_preview").label.toLowerCase()} to see this.`,
};

/** Two remedies are the same button when they make the same request. */
const sameAction = (a: Action | null, b: Action | null) => !!a && !!b && a.kind === b.kind && (a.path ?? "") === (b.path ?? "") && (a.method ?? "") === (b.method ?? "");

/**
 * The missing outputs of a view as one plain sentence and one remedy when they all share the
 * same remedy (or none has one): "Run the planner pack to see …" with its button, never the
 * list of output keys. Null when they need different remedies (the caller lists them).
 */
export function missingSummary(missing: readonly MissingOutput[]): { sentence: string; action: Action | null } | null {
  if (!missing.length) return null;
  const first = missing[0];
  const oneAction = first.action ? missing.every((m) => sameAction(m.action, first.action)) : missing.every((m) => !m.action);
  if (!oneAction) return null;
  const producers = [...new Set(missing.map((m) => m.produced_by))];
  const known = producers.length === 1 ? PRODUCER_SENTENCE[producers[0]] : undefined;
  const sentence = known ?? `It comes from ${producers.map(producedByText).join(" and ")}, which ${producers.length === 1 ? "has" : "have"} not written it yet.`;
  return { sentence, action: first.action };
}

/**
 * The empty state of a view (or a part of one) whose outputs are all missing: one plain
 * sentence and the remedy when they share one (missingSummary), else the list of what is missing.
 */
export function MissingState({ title, missing, fallback }: { title: string; missing: readonly MissingOutput[]; fallback?: string }) {
  const one = missingSummary(missing);
  if (one || !missing.length) {
    return <EmptyState title={`${title}: not available for this run yet`} body={one?.sentence ?? fallback ?? "The outputs this view needs have not been written."} action={one?.action ?? null} />;
  }
  return <EmptyState title={`${title}: not available for this run yet`} body={missing.map(missingText).join("; ") + "."} action={missing.find((m) => m.action)?.action ?? null} />;
}

/** Missing outputs of a view: one sentence and remedy when they share one, else each with the server's one-click remedy. */
export function MissingList({ missing }: { missing: ViewModelOf<ViewName>["missing"] }) {
  if (!missing.length) return null;
  const one = missingSummary(missing);
  if (one)
    return (
      <div className="callout" role="note" data-missing={missing.length}>
        {one.sentence} {one.action ? <ActionButton action={one.action} size="small" variant="default" /> : null}
      </div>
    );
  return (
    <div className="callout" role="note">
      <strong>Not in this run yet:</strong>
      <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>
        {missing.map((m) => (
          <li key={m.output} style={{ marginBottom: 4 }}>
            {missingText(m)} {m.action ? <ActionButton action={m.action} size="small" variant="default" /> : null}
          </li>
        ))}
      </ul>
    </div>
  );
}

/** The run's caveats as one disclosure, worded as the Summary's: "Things to keep in mind (3)". */
export function Caveats({ caveats, open }: { caveats: string[]; open?: boolean }) {
  if (!caveats.length) return null;
  return (
    <details className="callout" data-tone="info" open={open}>
      <summary>
        {term("keep_in_mind").label} ({caveats.length})
      </summary>
      <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>
        {caveats.map((c, i) => (
          <li key={i}>{c}</li>
        ))}
      </ul>
    </details>
  );
}

/**
 * What Model details passes to the page of a section: the rail label as its title and the
 * section's one-line intro as its `intro` (rendered under the title; `intro: null` on a second
 * page, which drops its own), `level: 3` for a second page under that title, and
 * `detailsOnly` to leave out the charts a planner tab already shows. Without props a page renders
 * its full original content.
 */
export type DetailPageProps = { title?: string; intro?: ReactNode | null; level?: 2 | 3; detailsOnly?: boolean };

export type ViewPageProps<V extends ViewName> = {
  view: V;
  title: string;
  intro?: ReactNode;
  /** Extra condition for the live banner (e.g. Accuracy's `sections.live`). */
  live?: (vm: ViewModelOf<V>) => boolean;
  liveText?: ReactNode;
  actions?: ReactNode;
  /**
   * Show the view's caveats. Off by default: the server's caveats are run-wide (the same list on
   * every view), so their one home is the Summary's "Things to keep in mind".
   */
  showCaveats?: boolean;
  /** Show the caveats expanded (Overview). */
  caveatsOpen?: boolean;
  /** Heading level of the title: 3 for a second page under a Model details section's title. */
  level?: 2 | 3;
  /** Show the title as a kicker (the brand's small uppercase label) rather than a section title: a group inside a page that has its own title. */
  kicker?: boolean;
  /** Query parameters of the view (the Heat tab's what-if dewpoint). */
  params?: ViewParams;
  children: (sections: Sections<ViewSections[V]>, vm: ViewModelOf<V>) => ReactNode;
};

/** A view's availability in plain words, with what it means (the pill beside the title). */
export const AVAILABILITY_WORDS: Record<Exclude<Availability, "ready">, { label: string; tip: string }> = {
  partial: { label: "partly ready", tip: "Some of the outputs this page uses are not written yet. The sections below say what is missing." },
  running: { label: "updating", tip: "The run is still writing outputs. This page refreshes as they arrive." },
  stale: {
    label: "may be out of date",
    tip: "Some outputs here are older than the run's record and may be left over from an earlier attempt. Re-run the step that writes them to be sure.",
  },
  missing: { label: "not available yet", tip: "The outputs this page needs have not been written for this run." },
};

/** The DEMO badge, with what demo data means on hover, focus or tap. */
export function DemoChip() {
  return (
    <TipChip tip={term("demo").tip}>
      <Badge tone="demo">DEMO</Badge>
    </TipChip>
  );
}

/**
 * A run tab rendered from `GET /api/runs/{rid}/views/{view}`: loading and error states, the
 * missing-output empty state with its action, the live banner while the run writes, the
 * view's caveats when asked (`showCaveats`), then the sections. Inside a Model details section
 * (TitleTipProvider) a level-2 title gets the section's ⓘ.
 */
export function ViewPage<V extends ViewName>({ view, title, intro, live, liveText, actions, showCaveats = false, caveatsOpen, level = 2, kicker = false, params, children }: ViewPageProps<V>) {
  const rid = useRid();
  const res = useView(rid, view, params);
  const runLive = useRunIsLive(rid);
  const titleTip = useContext(TitleTipCtx);
  const vm = res.data;
  const isLive = !!vm && (runLive || vm.availability === "running" || (live ? live(vm) : false));
  const Heading = level === 3 ? "h3" : "h2";
  const availability = vm && vm.availability !== "ready" ? AVAILABILITY_WORDS[vm.availability] : null;
  return (
    <section className="stack runhub-view" data-view={view} aria-labelledby={`view-${view}`}>
      <header className="row" style={{ justifyContent: "space-between" }}>
        <div className="row">
          <Heading id={`view-${view}`} className={kicker ? "kicker" : undefined} style={{ margin: 0 }}>
            {title}
          </Heading>
          {level === 2 && !kicker && titleTip ? titleTip : null}
          {vm?.demo ? <DemoChip /> : null}
          {vm && availability ? (
            <TipChip tip={availability.tip}>
              <Pill tone={vm.availability === "stale" ? "warn" : "neutral"}>{availability.label}</Pill>
            </TipChip>
          ) : null}
        </div>
        {actions ? <div className="row">{actions}</div> : null}
      </header>
      {intro ? <p className="cap prose">{intro}</p> : null}
      {isLive ? <LiveBanner>{liveText}</LiveBanner> : null}
      {!vm && res.error ? <EmptyState error={res.error} /> : null}
      {!vm && !res.error ? (
        <p className="cap" role="status">
          <span className="spinner" aria-hidden="true" /> Loading {title.toLowerCase()}…
        </p>
      ) : null}
      {vm && vm.availability === "missing" ? <MissingState title={title} missing={vm.missing} /> : null}
      {vm && vm.availability !== "missing" ? (
        <UnitsProvider units={vm.units}>
          <MissingList missing={vm.missing} />
          {showCaveats ? <Caveats caveats={vm.caveats} open={caveatsOpen} /> : null}
          {children(vm.sections, vm)}
        </UnitsProvider>
      ) : null}
    </section>
  );
}

/** A KPI whose value is words, not a number (a budget's constraint, its objective). */
export const isTextKpi = (k: ViewKpi) => k.format === "text" || typeof k.value === "string";

/** The ⓘ of a KPI tile: a glossary term, or a one-off label and tip (InfoTip's props). */
export type KpiTip = InfoTipProps;

/**
 * ViewModel KPIs as tiles (units and cooler/warmer wording from kpiDisplay). Words are not
 * numbers: text KPIs ("Constraint", "Objective") read as one caption line below the tiles.
 * `tips` puts an ⓘ after the label of the tiles it names (by KPI id).
 */
export function KpiTiles({ kpis, units, label, tips }: { kpis: ViewKpi[]; units?: ViewUnits; label?: string; tips?: Partial<Record<string, KpiTip>> }) {
  const ctxUnits = useUnits();
  if (!kpis.length) return <p className="cap">No key numbers for this run.</p>;
  const tiles = kpis.filter((k) => !isTextKpi(k));
  const words = kpis.filter(isTextKpi).filter((k) => k.value !== null && k.value !== "");
  const row = tiles.length ? (
    <KpiRow label={label}>
      {tiles.map((k) => {
        const d = kpiDisplay(k, units ?? ctxUnits);
        const tip = tips?.[k.id];
        const kpiLabel = tip ? (
          <>
            {k.label}
            <InfoTip {...tip} />
          </>
        ) : (
          k.label
        );
        // Cooling added up over places (°F·places) is explained once, by the glossary's total_cooling ⓘ.
        const unit = d.unit && d.unit.endsWith("·places") ? (
          <>
            {d.unit}
            <InfoTip term="total_cooling" label={d.unit} />
          </>
        ) : (
          d.unit || undefined
        );
        return <Kpi key={k.id} label={kpiLabel} value={d.value} unit={unit} note={d.note ?? undefined} tone={k.tone ?? undefined} />;
      })}
    </KpiRow>
  ) : null;
  if (!words.length) return row;
  return (
    <div className="stack" style={{ gap: 8 }}>
      {row}
      <p className="cap" style={{ margin: 0 }} data-kpi-text="true">
        {words.map((k, i) => (
          <span key={k.id}>
            {i ? " · " : ""}
            {k.label}: <span style={{ color: "var(--ink-2)" }}>{String(k.value)}</span>
          </span>
        ))}
      </p>
    </div>
  );
}

const SEVERITY_TONE: Record<string, "crit" | "warn" | "neutral" | "accent"> = { error: "crit", critical: "crit", warn: "warn", warning: "warn", info: "neutral" };

/** QA / data-health flags (severity shown as text and icon, not colour alone). */
export function FlagList({ flags, empty = "No flags raised." }: { flags: ViewFlag[]; empty?: string }) {
  if (!flags.length) return <p className="cap">{empty}</p>;
  return (
    <ul className="stack" style={{ gap: 6, listStyle: "none", padding: 0, margin: 0 }} aria-label="Flags">
      {flags.map((f, i) => (
        <li key={`${f.code}-${i}`} className="row" style={{ alignItems: "flex-start", flexWrap: "nowrap" }}>
          <Pill tone={SEVERITY_TONE[f.severity] ?? "neutral"} icon={f.severity === "info" ? "info" : "alert"}>
            {f.severity}
          </Pill>
          <span>
            <span className="mono cap">{f.code}</span> {f.message}
          </span>
        </li>
      ))}
    </ul>
  );
}

type GenericRow = { __i: number; cells: GenericTable["rows"][number] };

/** A ViewModel table (sortable, CSV copy/download). */
export function GenericTableView({ table, caption, csvName, empty }: { table: GenericTable; caption: string; csvName?: string; empty?: string }) {
  const cols = tableColumns(table);
  const columns: Column<GenericRow>[] = cols.map((c, i) => ({
    key: c.key,
    label: c.label,
    unit: c.unit,
    align: table.rows.some((r) => typeof r[i] === "number") ? "right" : "left",
    value: (r) => {
      const v = r.cells[i];
      return typeof v === "boolean" ? (v ? "yes" : "no") : v;
    },
    render: (r) => cellText(r.cells[i]),
  }));
  const rows: GenericRow[] = table.rows.map((cells, i) => ({ __i: i, cells }));
  return <Table columns={columns} rows={rows} rowKey={(r) => r.__i} caption={caption} csvName={csvName ?? caption} empty={empty} />;
}

/**
 * "Pin to Findings" for blocks that are not kit charts (tables, tool results): posts the
 * numbers on screen as the finding's snapshot (api.md §11).
 */
export function PinButton({ title, snapshot, view }: { title: string; snapshot: Record<string, unknown>; view?: string }) {
  const context = useUi((s) => s.context);
  const route = useRoute();
  const [busy, setBusy] = useState(false);
  const pin = async () => {
    if (!context.projectId) {
      toast("warning", "Open a project to pin findings");
      return;
    }
    setBusy(true);
    try {
      const body: FindingCreate = {
        project_id: context.projectId,
        run_id: context.runId,
        view: view ?? route.route?.path ?? route.pathname,
        url_state: route.pathname + (route.query.toString() ? "?" + route.query.toString() : ""),
        title,
        note_md: "",
        snapshot: { kind: "table", title, ...snapshot },
      };
      const f = await api.post<Finding>("/api/findings", body);
      toast("success", "Pinned to Findings", { href: `/p/${encodeURIComponent(context.projectId)}/exports?tab=notes`, linkLabel: "Open Findings" });
      return f;
    } catch (e) {
      toast("error", "Could not pin to Findings", { body: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };
  return (
    <button type="button" className="btn small ghost" onClick={() => void pin()} disabled={busy} aria-label={`Pin ${title} to Findings`}>
      {busy ? <span className="spinner" aria-hidden="true" /> : <Icon name="pin" />} Pin
    </button>
  );
}

const VERDICT_TONE: Record<Verdict["verdict"], "good" | "warn" | "crit" | "neutral"> = {
  robust: "good",
  direction: "warn",
  not_established: "crit",
  unknown: "neutral",
};
const VERDICT_ICON = { robust: "check", direction: "alert", not_established: "x", unknown: "minus" } as const;

/** The plain verdict on a scenario's change (Robust / Direction only / Not established); the reasons and
 *  qualifiers are in its tooltip (hover, focus or tap) and, with `detail`, written out below the pill. */
export function VerdictPill({ verdict, detail }: { verdict: Pick<Verdict, "verdict" | "label" | "reasons" | "qualifiers"> | null | undefined; detail?: boolean }) {
  if (!verdict || !verdict.verdict) return <span className="cap">—</span>;
  const why = [...(verdict.reasons ?? []), ...(verdict.qualifiers ?? []).map((q) => `note: ${q}`)].join("; ");
  const pill = (
    // with `detail` the reasons are written out below, so the pill needs no tooltip
    <TipChip tip={detail ? null : why}>
      <Pill tone={VERDICT_TONE[verdict.verdict] ?? "neutral"} icon={VERDICT_ICON[verdict.verdict] ?? "minus"}>
        {verdict.label}
        {verdict.qualifiers?.length ? " *" : ""}
      </Pill>
    </TipChip>
  );
  if (!detail) return pill;
  return (
    <span className="stack" style={{ gap: 2 }}>
      {pill}
      {why ? <span className="cap">{why}</span> : null}
    </span>
  );
}

/** A titled card with optional actions in its header; `tip` (an InfoTip) explains the title, beside it. */
export function Block({ title, tip, actions, children, id }: { title: ReactNode; tip?: ReactNode; actions?: ReactNode; children: ReactNode; id?: string }) {
  return (
    <section className="card" id={id}>
      <header>
        <div>
          {/* the ⓘ sits beside the heading, not inside it, so the heading's name stays its title */}
          <h3 style={tip ? { display: "inline" } : undefined}>{title}</h3>
          {tip}
        </div>
        {actions ? <div className="row">{actions}</div> : null}
      </header>
      {children}
    </section>
  );
}
