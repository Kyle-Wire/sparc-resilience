// Wraps every /r/:rid/* page: sets the shell context (run and its project), shows the run
// title row and the run tab bar built from the route modules' `runTab` declarations (grouped
// Model · Effects · Decisions · Trust · Run, ordered by `order`), each with the status dot
// from GET /api/runs/{rid}/outputs → tabs[] (SPEC §3.2, §12.3).
import { createContext, useContext, useEffect, type ReactNode } from "react";
import type { Availability, RunOutputs, RunTabId } from "../api/types";
import { Badge, modeLabel } from "../components/ui/Badge";
import { EmptyState } from "../components/ui/EmptyState";
import { StatusChip } from "../components/ui/StatusChip";
import { Link, RUN_TAB_GROUPS, runTabForRoute, runTabHref, useRegistry, useRoute, type RunTabEntry } from "../router";
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

export function tabStatus(outputs: RunOutputs | undefined, id: RunTabId): { availability: Availability; reason: string | null } | null {
  const t = outputs?.tabs.find((x) => x.id === id);
  if (!t) return null;
  const miss = t.missing[0];
  return { availability: t.availability, reason: miss ? `${miss.output} — produced by ${miss.produced_by}` : null };
}

/** The run tab bar (exported for tests and for pages that render their own header). */
export function RunTabs({ rid, tabs, outputs, activeId }: { rid: string; tabs: RunTabEntry[]; outputs: RunOutputs | undefined; activeId: RunTabId | null }) {
  const groups = RUN_TAB_GROUPS.map((g) => ({ group: g, tabs: tabs.filter((t) => t.group === g) })).filter((g) => g.tabs.length);
  return (
    <nav className="run-tabs" aria-label="Run views">
      {groups.map((g) => (
        <div className="run-tab-group" key={g.group} role="group" aria-label={g.group}>
          <span className="eyebrow" aria-hidden="true">
            {g.group}
          </span>
          <div>
            {g.tabs.map((t) => {
              const st = tabStatus(outputs, t.id);
              return (
                <Link
                  key={t.id}
                  to={runTabHref(t, rid)}
                  className="run-tab"
                  data-tab={t.id}
                  aria-current={t.id === activeId ? "page" : undefined}
                  title={st ? `${t.label}: ${AVAIL_TEXT[st.availability]}${st.reason ? ` (${st.reason})` : ""}` : t.label}
                >
                  {st ? <span className="tab-dot" data-a={st.availability} aria-hidden="true" /> : null}
                  {t.label}
                  {st && st.availability !== "ready" ? <span className="sr-only"> ({AVAIL_TEXT[st.availability]})</span> : null}
                </Link>
              );
            })}
          </div>
        </div>
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
        {run ? <Badge tone="accent">{modeLabel(run.mode, run.coarse_m)}</Badge> : null}
        {run?.demo ? <Badge tone="demo">DEMO</Badge> : null}
        {run ? <StatusChip status={run.status} /> : null}
        <span className="cap mono">{rid}</span>
      </div>
      <RunTabs rid={rid} tabs={registry.runTabs} outputs={outputs.data} activeId={active?.id ?? null} />
      {children}
    </RunCtx.Provider>
  );
}
