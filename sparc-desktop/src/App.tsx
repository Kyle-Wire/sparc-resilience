import { useState, useCallback, useEffect, useMemo } from "react";
import { useServer } from "@/hooks/useServer";
import { useProject } from "@/hooks/useProject";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { NotificationContext, useNotificationState } from "@/hooks/useNotifications";
import NotificationBanner from "@/components/layout/NotificationBanner";
import Splash from "@/components/layout/Splash";
import Shell from "@/components/layout/Shell";
import type { PageName } from "@/components/layout/Sidebar";
import { PAGES } from "@/components/layout/Sidebar";
import ProjectSetup from "@/components/pipeline/ProjectSetup";
import DataView from "@/components/pipeline/DataView";
import VariablesView from "@/components/pipeline/VariablesView";
import CRSView from "@/components/pipeline/CRSView";
import DAGView from "@/components/pipeline/DAGView";
import PhysicsView from "@/components/pipeline/PhysicsView";
import ScenariosView from "@/components/pipeline/ScenariosView";
import ModelsView from "@/components/pipeline/ModelsView";
import PipelineRun from "@/components/pipeline/PipelineRun";
import ResultsView from "@/components/results/ResultsView";
import ReportView from "@/components/results/ReportView";
import SettingsView from "@/components/pipeline/SettingsView";
import ChatPanel from "@/components/chat/ChatPanel";
import { buildSystemPrompt } from "@/lib/prompts";
import { getConfig, saveConfig, dataSummary, initProject } from "@/lib/api";
import type { ClaudeAction, DataSummary, ProjectConfig } from "@/lib/types";

type AppPage = PageName | "Settings";

export default function App() {
  const { ready, status } = useServer();
  const notif = useNotificationState();
  const project = useProject(ready);
  const [page, setPage] = useState<AppPage>("Project");
  const [chatOpen, setChatOpen] = useState(false);
  const [dataCtx, setDataCtx] = useState<{ columns: string[]; target?: string; summary?: Record<string, unknown> } | null>(null);
  // Increment to force child re-render after action dispatch
  const [refreshKey, setRefreshKey] = useState(0);

  // Gate navigation: only Project and Settings are allowed without a loaded project
  const navigate = useCallback(
    (p: AppPage) => {
      if (p !== "Project" && p !== "Settings" && !project.projectLoaded) return;
      setPage(p);
    },
    [project.projectLoaded],
  );

  // Load data context when available for system prompt enrichment
  useEffect(() => {
    if (!ready) return;
    Promise.all([
      getConfig().catch(() => null),
      dataSummary().catch(() => null),
    ]).then(([cfg, summary]: [ProjectConfig | null, DataSummary | null]) => {
      if (summary) {
        setDataCtx({
          columns: summary.columns ?? [],
          target: cfg?.data?.target_column,
          summary: summary.numeric_summary,
        });
      }
    });
  }, [ready, refreshKey]);

  // Derive the prompt mode from current page
  const promptMode = (() => {
    switch (page) {
      case "Project": return "domain" as const;
      case "DAG": return "dag" as const;
      case "Physics": return "physics" as const;
      default: return "general" as const;
    }
  })();

  const systemPrompt = buildSystemPrompt(promptMode, dataCtx ?? undefined);

  // ----- Keyboard shortcuts -----
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

  // ----- Chat → config action dispatch -----
  const handleAction = useCallback(async (action: ClaudeAction) => {
    try {
      switch (action.action) {
        case "suggest_template": {
          // Init project from suggested template
          const home = prompt("Choose output directory:", `${action.template}_project`);
          if (!home) break;
          const res = await initProject(action.template, home);
          await project.openProject(res.project_yml, { template: action.template });
          navigate("Data");
          setRefreshKey((k) => k + 1);
          break;
        }
        case "propose_dag_edges": {
          // Navigate to DAG page — edges will be shown as proposals
          // Store proposed edges for DAGView to pick up
          localStorage.setItem("sparc-proposed-edges", JSON.stringify(action.edges));
          navigate("DAG");
          setRefreshKey((k) => k + 1);
          break;
        }
        case "suggest_physics": {
          // Write monotone constraints + bounds to config
          await saveConfig({
            physics: {
              monotone_constraints: action.monotonic_constraints,
            },
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

  if (!ready || project.rehydrating) return <Splash />;

  const renderPage = () => {
    switch (page) {
      case "Project":
        return (
          <ProjectSetup
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
        return <DataView key={refreshKey} onNavigateToProject={() => navigate("Project")} />;
      case "Variables":
        return <VariablesView key={refreshKey} />;
      case "CRS":
        return <CRSView />;
      case "DAG":
        return <DAGView key={refreshKey} />;
      case "Physics":
        return <PhysicsView key={refreshKey} />;
      case "Scenarios":
        return <ScenariosView />;
      case "Models":
        return <ModelsView />;
      case "Run":
        return <PipelineRun />;
      case "Results":
        return <ResultsView />;
      case "Report":
        return <ReportView />;
      case "Settings":
        return <SettingsView />;
    }
  };

  return (
    <NotificationContext.Provider value={notif}>
    <div className="flex h-screen">
      <Shell
        currentPage={page as PageName}
        onNavigate={(p) => navigate(p)}
        onSettings={() => navigate("Settings")}
        status={status}
        projectLoaded={project.projectLoaded}
      >
        <div className="flex h-full gap-0">
          {/* Chat toggle button */}
          <button
            onClick={() => setChatOpen(!chatOpen)}
            className={`fixed right-4 top-12 z-10 rounded px-3 py-1.5 text-xs font-medium transition-colors ${
              chatOpen
                ? "bg-sparc-gray-800 text-white hover:bg-black"
                : "bg-black text-white hover:bg-sparc-purple"
            }`}
          >
            {chatOpen ? "✕ Close" : "AI Assistant"}
          </button>

          {/* Main content */}
          <div className={`flex-1 ${chatOpen ? "mr-80" : ""}`}>
            {renderPage()}
          </div>

          {/* Chat panel */}
          {chatOpen && (
            <div className="fixed right-0 top-10 bottom-0 w-80 border-l border-sparc-gray-200 bg-white shadow-lg">
              <ChatPanel onAction={handleAction} systemPrompt={systemPrompt} />
            </div>
          )}
        </div>
      </Shell>
      <NotificationBanner />
    </div>
    </NotificationContext.Provider>
  );
}
