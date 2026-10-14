// Wraps every /r/:rid/* page: sets the shell context (run and its project), shows the run
// title row and the flat run tab bar built from the route modules' `runTab` declarations
// (planner tabs by `order`, then Model details after a divider), each with a status dot that
// aggregates its views' entries in GET /api/runs/{rid}/outputs → tabs[] (SPEC §3.2, §12.3).
import { createContext, useContext, useEffect, type ReactNode } from "react";
import type { Availability, RunOutputs, RunTabId } from "../api/types";
import { Badge, modeLabel } from "../components/ui/Badge";
import { EmptyState } from "../components/ui/EmptyState";
import { StatusChip } from "../components/ui/StatusChip";
import { Link, runTabForRoute, runTabHref, useRegistry, useRoute, type RunTabEntry } from "../router";
import { useUi } from "../stores/ui";
import { useRunDetail, useRunOutputs, type RunDetailHead } from "./resources";

export type RunContextValue = { rid: string; detail: RunDetailHead | undefined; outputs: RunOutputs | undefined };

const RunCtx = createContext<RunContextValue | null>(null);

/** The current run inside a RunLayout (null elsewhere). */
export function useRunContext(): RunContextValue | null {
  return useContext(RunCtx);
}

const AVAIL_TEXT: Record<Availability, string> = {
  ready: "ready",
  partial: "partly available",
  running: "being computed",
  missing: "not available yet",
  stale: "stale",
};

/** Severity of the unavailable states, worst last (the dot of an all-unavailable tab). */
const BAD: Availability[] = ["running", "stale", "missing"];

export type TabStatus = { availability: Availability; reason: string | null };

/**
 * Status dot of a run tab from its views' entries in `outputs.tabs[]` (views the server does
 * not list are ignored; none listed → null, no dot). All ready → ready; none ready or partial →
 * the worst of running < stale < missing; otherwise (mixed) → partial. The reason names the
 * first view that is not ready and, when known, the missing output and what produces it.
 */
export function tabStatus(outputs: RunOutputs | undefined, views: RunTabId | readonly RunTabId[]): TabStatus | null {
  const ids: readonly RunTabId[] = typeof views === "string" ? [views] : views;
  const rows = ids.flatMap((id) => outputs?.tabs.find((x) => x.id === id) ?? []);
  if (!rows.length) return null;
  const states = rows.map((r) => r.availability);
  let availability: Availability;
  if (states.every((a) => a === "ready")) availability = "ready";
  else if (states.every((a) => BAD.includes(a))) availability = states.reduce((w, a) => (BAD.indexOf(a) > BAD.indexOf(w) ? a : w));
  else availability = "partial";
  const off = rows.find((r) => r.availability !== "ready");
  const miss = off?.missing[0];
  const reason = !off ? null : miss ? `${miss.output} — produced by ${miss.produced_by}` : rows.length > 1 ? `${off.id}: ${AVAIL_TEXT[off.availability]}` : null;
  return { availability, reason };
}

function RunTabLink({ tab, rid, outputs, active }: { tab: RunTabEntry; rid: string; outputs: RunOutputs | undefined; active: boolean }) {
  const st = tabStatus(outputs, tab.views);
  return (
    <Link
      to={runTabHref(tab, rid)}
      className={tab.advanced ? "run-tab advanced" : "run-tab"}
      data-tab={tab.id}
      aria-current={active ? "page" : undefined}
      title={st ? `${tab.label}: ${AVAIL_TEXT[st.availability]}${st.reason ? ` (${st.reason})` : ""}` : tab.label}
    >
      {st ? <span className="tab-dot" data-a={st.availability} aria-hidden="true" /> : null}
      {tab.label}
      {st && st.availability !== "ready" ? <span className="sr-only"> ({AVAIL_TEXT[st.availability]})</span> : null}
    </Link>
  );
}

/**
 * The run tab bar (exported for tests and for pages that render their own header): one flat
 * row of links, the advanced tabs (Model details) right-aligned after a thin divider.
 */
export function RunTabs({ rid, tabs, outputs, activeId }: { rid: string; tabs: RunTabEntry[]; outputs: RunOutputs | undefined; activeId: string | null }) {
  const main = tabs.filter((t) => !t.advanced);
  const advanced = tabs.filter((t) => t.advanced);
  return (
    <nav className="run-tabs" aria-label="Run views">
      {main.map((t) => (
        <RunTabLink key={t.id} tab={t} rid={rid} outputs={outputs} active={t.id === activeId} />
      ))}
      {advanced.length ? <span className="run-tab-divider" aria-hidden="true" /> : null}
      {advanced.map((t) => (
        <RunTabLink key={t.id} tab={t} rid={rid} outputs={outputs} active={t.id === activeId} />
      ))}
    </nav>
  );
}

export function RunLayout({ rid, children }: { rid: string; children: ReactNode }) {
  const registry = useRegistry();
  const route = useRoute();
  const setContext = useUi((s) => s.setContext);
  const detail = useRunDetail(rid);
  const outputs = useRunOutputs(rid);
  const run = detail.data?.run;
  useEffect(() => setContext({ projectId: run?.project_id ?? useUi.getState().context.projectId, runId: rid }), [rid, run?.project_id, setContext]);
  const active = runTabForRoute(registry, route.route);
  if (detail.error && !detail.data) return <EmptyState error={detail.error} />;
  return (
    <RunCtx.Provider value={{ rid, detail: detail.data, outputs: outputs.data }}>
      <div className="run-head">
        <h1 style={{ fontSize: "1.35rem" }}>{run ? run.label || detail.data?.header.name || run.id : rid}</h1>
        {run ? <Badge tone="mode">{modeLabel(run.mode, run.coarse_m)}</Badge> : null}
        {run?.demo ? <Badge tone="demo">DEMO</Badge> : null}
        {run ? <StatusChip status={run.status} /> : null}
        <span className="cap mono">{rid}</span>
      </div>
      <RunTabs rid={rid} tabs={registry.runTabs} outputs={outputs.data} activeId={active?.id ?? null} />
      {children}
    </RunCtx.Provider>
  );
}
