import { useState, useCallback, useEffect, useMemo } from "react";
import { useServer } from "@/hooks/useServer";
import { useProject } from "@/hooks/useProject";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { NotificationContext, useNotificationState } from "@/hooks/useNotifications";
import { PipelineProvider } from "@/hooks/PipelineProvider";
import NotificationBanner from "@/components/layout/NotificationBanner";
import Splash from "@/components/layout/Splash";
import Shell from "@/components/layout/Shell";
import type { PageName } from "@/components/layout/Sidebar";
import { PAGES } from "@/components/layout/Sidebar";
import ChatPanel from "@/components/chat/ChatPanel";
import SettingsView from "@/components/pipeline/SettingsView";
import { loadTheme, applyTheme } from "@/components/pipeline/SettingsView";
import type { LogoHue } from "@/components/pipeline/SettingsView";
import { buildSystemPrompt } from "@/lib/prompts";
import { getConfig, saveConfig, dataSummary, initProject, getReportData, getDag } from "@/lib/api";
import type { ClaudeAction, DataSummary, ProjectConfig, ReportPayload, DagDefinition } from "@/lib/types";
import type { PromptDataContext } from "@/lib/prompts";

import ProjectPage from "@/components/pages/ProjectPage";
import DataPage from "@/components/pages/DataPage";
import DAGPage from "@/components/pages/DAGPage";
import RunPage from "@/components/pages/RunPage";
import ResultsPage, { type Scenario } from "@/components/pages/ResultsPage";
import DataProcessingView from "@/components/pipeline/DataProcessingView";
import VariableDashboardView from "@/components/results/VariableDashboardView";
import PhysicsView from "@/components/pipeline/PhysicsView";
import CRSView from "@/components/pipeline/CRSView";
import ScenariosView from "@/components/pipeline/ScenariosView";
import ModelsView from "@/components/pipeline/ModelsView";
import ReportView from "@/components/results/ReportView";

type AppPage = PageName | "Settings";

export default function App() {
  const { ready, status } = useServer();
  const notif = useNotificationState();
  const project = useProject(ready);
  const [page, setPage] = useState<AppPage>("Project");
  const [chatOpen, setChatOpen] = useState(false);
  const [scenario, setScenario] = useState<Scenario>("canopy+10");
  const [dataCtx, setDataCtx] = useState<PromptDataContext | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [logoHue, setLogoHue] = useState<LogoHue>(() => loadTheme().logoHue);

  // Apply persisted theme on mount and listen for changes from Settings
  useEffect(() => {
    applyTheme(loadTheme());
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      if (detail?.logoHue) setLogoHue(detail.logoHue);
    };
    window.addEventListener("sparc-theme-change", handler);
    return () => window.removeEventListener("sparc-theme-change", handler);
  }, []);

  // Gate navigation: only Project and Settings are allowed without a loaded project
  const navigate = useCallback(
    (p: AppPage) => {
      if (p !== "Project" && p !== "Settings" && !project.projectLoaded) return;
      setPage(p);
    },
    [project.projectLoaded],
  );

  // Load data context for system prompt enrichment
  useEffect(() => {
    if (!ready || !project.projectLoaded) return;
    Promise.all([
      getConfig().catch(() => null),
      dataSummary().catch(() => null),
      getDag().catch(() => null),
      getReportData().catch(() => null),
    ]).then(([cfg, summary, dag, report]: [ProjectConfig | null, DataSummary | null, DagDefinition | null, ReportPayload | null]) => {
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
    });
  }, [ready, project.projectLoaded, refreshKey]);

  // Derive prompt mode from current page
  const promptMode = (() => {
    switch (page) {
      case "Project": return "domain" as const;
      case "DAG": return "dag" as const;
      case "Physics": return "physics" as const;
      case "Results":
      case "Report": return "results" as const;
      default: return "general" as const;
    }
  })();

  const systemPrompt = buildSystemPrompt(promptMode, dataCtx ?? undefined);

  // Keyboard shortcuts
  const kbHandlers = useMemo(
    () => ({
      toggleChat: () => setChatOpen((o) => !o),
      navigateByIndex: (i: number) => {
        if (i < PAGES.length) navigate(PAGES[i]);
      },
      openSettings: () => navigate("Settings"),
      refresh: () => setRefreshKey((k) => k + 1),
    }),
    [navigate],
  );
  useKeyboardShortcuts(kbHandlers);

  // Chat → config action dispatch
  const handleAction = useCallback(async (action: ClaudeAction) => {
    try {
      switch (action.action) {
        case "suggest_template": {
          const home = prompt("Choose output directory:", `${action.template}_project`);
          if (!home) break;
          const res = await initProject(action.template, home);
          await project.openProject(res.project_yml, { template: action.template });
          navigate("Data");
          setRefreshKey((k) => k + 1);
          break;
        }
        case "propose_dag_edges": {
          localStorage.setItem("sparc-proposed-edges", JSON.stringify(action.edges));
          navigate("DAG");
          setRefreshKey((k) => k + 1);
          break;
        }
        case "suggest_physics": {
          await saveConfig({
            physics: { monotone_constraints: action.monotonic_constraints },
          });
          navigate("Physics");
          setRefreshKey((k) => k + 1);
          break;
        }
        case "suggest_predictors": {
          await saveConfig({ predictors: action.predictors });
          navigate("Variables");
          setRefreshKey((k) => k + 1);
          break;
        }
      }
    } catch (e) {
      notif.notify("error", e instanceof Error ? e.message : "Action dispatch failed");
    }
  }, [notif, project, navigate]);

  if (!ready || project.rehydrating) return <Splash logoHue={logoHue} />;

  const renderPage = () => {
    switch (page) {
      case "Project":
        return (
          <ProjectPage
            projectPath={project.projectPath}
            onProjectLoaded={async (path, meta) => {
              await project.openProject(path, meta);
              navigate("Data");
              setRefreshKey((k) => k + 1);
              notif.notify("success", "Project loaded successfully");
            }}
          />
        );
      case "Data":
        return <DataPage key={refreshKey} />;
      case "Processing":
        return <DataProcessingView />;
      case "DAG":
        return <DAGPage key={refreshKey} />;
      case "Variables":
        return <VariableDashboardView />;
      case "Physics":
        return <PhysicsView />;
      case "CRS":
        return <CRSView />;
      case "Scenarios":
        return <ScenariosView />;
      case "Models":
        return <ModelsView />;
      case "Run":
        return <RunPage />;
      case "Results":
        return <ResultsPage scenario={scenario} setScenario={setScenario} />;
      case "Report":
        return <ReportView />;
      case "Settings":
        return <SettingsView />;
    }
  };

  return (
    <NotificationContext.Provider value={notif}>
      <PipelineProvider serverReady={ready}>
        <div style={{ position: "relative", height: "100vh", width: "100vw", overflow: "hidden" }}>
          <Shell
            currentPage={page === "Settings" ? "Project" : page}
            onNavigate={navigate}
            onToggleChat={() => setChatOpen((o) => !o)}
            chatOpen={chatOpen}
            projectLoaded={project.projectLoaded}
            projectName={project.projectPath?.split(/[\\/]/).pop()?.replace(".yml", "") ?? undefined}
            logoHue={logoHue}
            status={status}
          >
            {renderPage()}
          </Shell>

          {chatOpen && (
            <ChatPanel
              onAction={handleAction}
              systemPrompt={systemPrompt}
              onClose={() => setChatOpen(false)}
            />
          )}

          <NotificationBanner />
        </div>
      </PipelineProvider>
    </NotificationContext.Provider>
  );
}
