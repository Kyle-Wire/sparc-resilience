/**
 * useProjectContext — loads data context for the AI system prompt and
 * exposes domain/EPSG metadata for the sidebar pill.
 *
 * Config is read from useProjectConfigStore (already fetched by useProjectConfig
 * on project load). This hook fetches the remaining three data sources
 * (dataSummary, DAG, reportData) and does NOT call getConfig() independently.
 */
import { useState, useCallback, useEffect } from "react";
import { dataSummary, getDag, getReportData } from "@/lib/api";
import { useProjectConfigStore } from "@/stores/projectConfigStore";
import type { DataSummary, DagDefinition, ReportPayload } from "@/lib/types";
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
  const config = useProjectConfigStore((s) => s.config);
  const [dataCtx, setDataCtx] = useState<PromptDataContext | null>(null);
  const [projectDomain, setProjectDomain] = useState<string | undefined>();
  const [projectEpsg, setProjectEpsg] = useState<string | undefined>();
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    if (!ready || !projectLoaded) return;
    Promise.all([
      dataSummary().catch(() => null),
      getDag().catch(() => null),
      getReportData().catch(() => null),
    ]).then(([summary, dag, report]: [
      DataSummary | null,
      DagDefinition | null,
      ReportPayload | null,
    ]) => {
      const ctx: PromptDataContext = {
        columns: summary?.columns ?? [],
        target: config?.data?.target_column,
        summary: summary?.numeric_summary,
      };
      if (dag?.edges && dag.edges.length > 0) ctx.dagEdges = dag.edges;
      const mono = config?.physics?.monotone_constraints;
      if (mono && typeof mono === "object") ctx.physicsConstraints = mono as Record<string, number>;
      const scenarios = config?.scenarios;
      if (Array.isArray(scenarios) && scenarios.length > 0) ctx.scenarios = scenarios;
      const causal = report?.causal_results;
      if (causal?.direct_effects && Object.keys(causal.direct_effects).length > 0) {
        ctx.causalResults = causal.direct_effects;
      }
      if (report?.scenario_summary && report.scenario_summary.length > 0) {
        ctx.scenarioSummary = report.scenario_summary;
      }
      setDataCtx(ctx);
      setProjectDomain(config?.project?.domain ?? undefined);
      const epsgRaw = config?.crs?.projected;
      setProjectEpsg(epsgRaw ? String(epsgRaw).replace("EPSG:", "") : undefined);
    });
  }, [ready, projectLoaded, refreshKey, config]);

  const refresh = useCallback(() => setRefreshKey((k) => k + 1), []);

  return { dataCtx, projectDomain, projectEpsg, refreshKey, refresh };
}
