/**
 * useProjectContext — loads data context for the AI system prompt and
 * exposes domain/EPSG metadata for the sidebar pill.
 *
 * Re-runs whenever eady, projectLoaded, or efreshKey changes.
 */
import { useState, useCallback, useEffect } from "react";
import { getConfig, dataSummary, getDag, getReportData } from "@/lib/api";
import type { ProjectConfig, DataSummary, DagDefinition, ReportPayload } from "@/lib/types";
import type { PromptDataContext } from "@/lib/prompts";

export interface ProjectContextResult {
  dataCtx: PromptDataContext | null;
  projectDomain: string | undefined;
  projectEpsg: string | undefined;
  refreshKey: number;
  refresh: () => void;
}

export function useProjectContext(
  ready: boolean,
  projectLoaded: boolean,
): ProjectContextResult {
  const [dataCtx, setDataCtx] = useState<PromptDataContext | null>(null);
  const [projectDomain, setProjectDomain] = useState<string | undefined>();
  const [projectEpsg, setProjectEpsg] = useState<string | undefined>();
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    if (!ready || !projectLoaded) return;
    Promise.all([
      getConfig().catch(() => null),
      dataSummary().catch(() => null),
      getDag().catch(() => null),
      getReportData().catch(() => null),
    ]).then(([cfg, summary, dag, report]: [
      ProjectConfig | null,
      DataSummary | null,
      DagDefinition | null,
      ReportPayload | null,
    ]) => {
      const ctx: PromptDataContext = {
        columns: summary?.columns ?? [],
        target: cfg?.data?.target_column,
        summary: summary?.numeric_summary,
      };
      if (dag?.edges && dag.edges.length > 0) ctx.dagEdges = dag.edges;
      const mono = cfg?.physics?.monotone_constraints;
      if (mono && typeof mono === "object") ctx.physicsConstraints = mono as Record<string, number>;
      const scenarios = cfg?.scenarios;
      if (Array.isArray(scenarios) && scenarios.length > 0) ctx.scenarios = scenarios;
      const causal = report?.causal_results;
      if (causal?.direct_effects && Object.keys(causal.direct_effects).length > 0) {
        ctx.causalResults = causal.direct_effects;
      }
      if (report?.scenario_summary && report.scenario_summary.length > 0) {
        ctx.scenarioSummary = report.scenario_summary;
      }
      setDataCtx(ctx);
      setProjectDomain(cfg?.project?.domain ?? undefined);
      const epsgRaw = cfg?.crs?.projected;
      setProjectEpsg(epsgRaw ? String(epsgRaw).replace("EPSG:", "") : undefined);
    });
  }, [ready, projectLoaded, refreshKey]);

  const refresh = useCallback(() => setRefreshKey((k) => k + 1), []);

  return { dataCtx, projectDomain, projectEpsg, refreshKey, refresh };
}
