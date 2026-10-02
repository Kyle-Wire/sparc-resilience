// Shared run-hub building blocks (SPEC §6.4): the ViewModel page wrapper (live banner,
// missing-output empty state with the server's action, caveats), per-section placeholders
// (older code, or the known reason a section is absent), KPI tiles, flags, generic tables and
// the Findings pin for non-chart blocks.
import { Component, createContext, useContext, useState, type ErrorInfo, type ReactNode } from "react";
import { api, errorMessage } from "../../api/client";
import type { GenericTable, Sections, ViewFlag, ViewKpi, ViewModelOf, ViewName, ViewSections, ViewUnits } from "../../api/runs";
import { useRunDetailFull, useView } from "../../api/runs";
import type { Action, Finding, FindingCreate, StageId } from "../../api/types";
import { Badge } from "../../components/ui/Badge";
import { ActionButton, EmptyState } from "../../components/ui/EmptyState";
import { Icon } from "../../components/ui/Icon";
import { Kpi, KpiRow } from "../../components/ui/Kpi";
import { Pill } from "../../components/ui/Pill";
import { Table, type Column } from "../../components/ui/Table";
import { runIsLive } from "../../layouts/resources";
import { useRoute } from "../../router";
import { toast, useUi } from "../../stores/ui";
import { planReasonText } from "../projects/model/plan";
import { cellText, kpiDisplay, missingText, tableColumns } from "./format";

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

/** Placeholder for a section an older-code run does not have. */
export function OlderCode({ title }: { title: ReactNode }) {
  return (
    <div className="callout older-code" data-tone="info" data-older-code="true">
      <strong>{title}</strong>: not in this run (older code)
    </div>
  );
}

/** Why a section is absent when the reason is known (a stage the run skipped, data the run lacks): one sentence. */
export type Absence = { reason: ReactNode; action?: Action | null };

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
  return { reason: why ? `${what} did not run: ${why}.` : `${what} did not run.`, action };
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

/** Missing outputs of a view, each with the server's one-click remedy. */
export function MissingList({ missing }: { missing: ViewModelOf<ViewName>["missing"] }) {
  if (!missing.length) return null;
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

export function Caveats({ caveats, open }: { caveats: string[]; open?: boolean }) {
  if (!caveats.length) return null;
  return (
    <details className="callout" data-tone="info" open={open}>
      <summary>
        {caveats.length} caveat{caveats.length === 1 ? "" : "s"} for this view
      </summary>
      <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>
        {caveats.map((c, i) => (
          <li key={i}>{c}</li>
        ))}
      </ul>
    </details>
  );
}

export type ViewPageProps<V extends ViewName> = {
  view: V;
  title: string;
  intro?: ReactNode;
  /** Extra condition for the live banner (e.g. Accuracy's `sections.live`). */
  live?: (vm: ViewModelOf<V>) => boolean;
  liveText?: ReactNode;
  actions?: ReactNode;
  /** Show the caveats expanded (Overview). */
  caveatsOpen?: boolean;
  children: (sections: Sections<ViewSections[V]>, vm: ViewModelOf<V>) => ReactNode;
};

/**
 * A run tab rendered from `GET /api/runs/{rid}/views/{view}`: loading and error states, the
 * missing-output empty state with its action, the live banner while the run writes, the
 * view's caveats, then the sections.
 */
export function ViewPage<V extends ViewName>({ view, title, intro, live, liveText, actions, caveatsOpen, children }: ViewPageProps<V>) {
  const rid = useRid();
  const res = useView(rid, view);
  const runLive = useRunIsLive(rid);
  const vm = res.data;
  const isLive = !!vm && (runLive || vm.availability === "running" || (live ? live(vm) : false));
  return (
    <section className="stack runhub-view" data-view={view} aria-labelledby={`view-${view}`}>
      <header className="row" style={{ justifyContent: "space-between" }}>
        <div className="row">
          <h2 id={`view-${view}`} style={{ margin: 0 }}>
            {title}
          </h2>
          {vm?.demo ? (
            <Badge tone="demo" title="Synthetic demo data at a fictional location">
              DEMO
            </Badge>
          ) : null}
          {vm && vm.availability !== "ready" ? <Pill tone={vm.availability === "stale" ? "warn" : "neutral"}>{vm.availability}</Pill> : null}
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
      {vm && vm.availability === "missing" ? (
        <EmptyState
          title={`${title}: not available for this run yet`}
          body={vm.missing.length ? vm.missing.map(missingText).join("; ") + "." : "The outputs this view needs have not been written."}
          action={vm.missing.find((m) => m.action)?.action ?? null}
        />
      ) : null}
      {vm && vm.availability !== "missing" ? (
        <UnitsProvider units={vm.units}>
          <MissingList missing={vm.missing} />
          <Caveats caveats={vm.caveats} open={caveatsOpen} />
          {children(vm.sections, vm)}
        </UnitsProvider>
      ) : null}
    </section>
  );
}

/** ViewModel KPIs as tiles (units and cooler/warmer wording from kpiDisplay). */
export function KpiTiles({ kpis, units, label }: { kpis: ViewKpi[]; units?: ViewUnits; label?: string }) {
  const ctxUnits = useUnits();
  if (!kpis.length) return <p className="cap">No key numbers for this run.</p>;
  return (
    <KpiRow label={label}>
      {kpis.map((k) => {
        const d = kpiDisplay(k, units ?? ctxUnits);
        return <Kpi key={k.id} label={k.label} value={d.value} unit={d.unit || undefined} note={d.note ?? undefined} tone={k.tone ?? undefined} />;
      })}
    </KpiRow>
  );
}

const SEVERITY_TONE: Record<string, "crit" | "warn" | "neutral" | "accent"> = { error: "crit", critical: "crit", warn: "warn", warning: "warn", info: "accent" };

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
      toast("success", "Pinned to Findings", { href: `/p/${context.projectId}/findings`, linkLabel: "Open Findings" });
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

/** A titled card with optional actions in its header. */
export function Block({ title, actions, children, id }: { title: ReactNode; actions?: ReactNode; children: ReactNode; id?: string }) {
  return (
    <section className="card" id={id}>
      <header>
        <div>
          <h3>{title}</h3>
        </div>
        {actions ? <div className="row">{actions}</div> : null}
      </header>
      {children}
    </section>
  );
}
