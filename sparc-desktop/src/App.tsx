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
import CommandPalette, { type PaletteItem } from "@/components/common/CommandPalette";
import OnboardingTour from "@/components/common/OnboardingTour";
import { ExplainContext, useExplainHost } from "@/hooks/ExplainContext";
import { loadTheme, applyTheme } from "@/lib/theme";
import { buildSystemPrompt } from "@/lib/prompts";
import { getConfig, saveConfig, dataSummary, initProject, getReportData, getDag } from "@/lib/api";
import type { ClaudeAction, DataSummary, ProjectConfig, ReportPayload, DagDefinition } from "@/lib/types";
import type { PromptDataContext } from "@/lib/prompts";

import ProjectPage from "@/components/pages/ProjectPage";
import DataPage from "@/components/pages/DataPage";
import ProcessingPage from "@/components/pages/ProcessingPage";
import DAGPage from "@/components/pages/DAGPage";
import VariablesPage from "@/components/pages/VariablesPage";
import PhysicsPage from "@/components/pages/PhysicsPage";
import CRSPage from "@/components/pages/CRSPage";
import ScenariosPage from "@/components/pages/ScenariosPage";
import ModelsPage from "@/components/pages/ModelsPage";
import RunPage from "@/components/pages/RunPage";
import ResultsPage from "@/components/pages/ResultsPage";
import DecisionsPage from "@/components/pages/DecisionsPage";
import ComparePage from "@/components/pages/ComparePage";
import ReportPage from "@/components/pages/ReportPage";
import BudgetOptimizerPage from "@/components/pages/BudgetOptimizerPage";
import SettingsPage from "@/components/pages/SettingsPage";

type AppPage = PageName | "Settings";

export default function App() {
  const { ready, status } = useServer();
  const notif = useNotificationState();
  const project = useProject(ready);
  const [page, setPage] = useState<AppPage>("Project");
  const [chatOpen, setChatOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const explainHost = useExplainHost();
  const explainSeed = explainHost.seed;
  const [dataCtx, setDataCtx] = useState<PromptDataContext | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  // Apply persisted theme on mount
  useEffect(() => {
    applyTheme(loadTheme());
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

  // Derive prompt mode from current page (or seed override).
  const promptMode = (() => {
    if (explainSeed?.mode === "narrator") return "narrator" as const;
    if (explainSeed?.mode === "hypothesis") return "hypothesis" as const;
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
      openPalette: () => setPaletteOpen(true),
    }),
    [navigate],
  );

  // Open the chat automatically whenever something requests an explanation.
  useEffect(() => {
    if (explainSeed) setChatOpen(true);
  }, [explainSeed]);

  // Build the command palette items from currently-known data context.
  const paletteItems = useMemo<PaletteItem[]>(() => {
    const out: PaletteItem[] = PAGES.map((p) => ({
      id: `page:${p}`, kind: "page", label: p, page: p, hint: "→ navigate",
    }));
    out.push({ id: "page:Settings", kind: "page", label: "Settings", page: "Settings", hint: "→ navigate" });
    out.push({
      id: "action:toggle-chat", kind: "action", label: "Toggle chat panel",
      hint: "Cmd+J", run: () => setChatOpen((o) => !o),
    });
    out.push({
      id: "action:hypothesis", kind: "action", label: "Generate hypotheses",
      hint: "narrator",
      run: () => explainHost.value.requestExplain(
        "Propose 3-5 ranked, testable causal hypotheses for the loaded dataset.",
        "hypothesis",
      ),
    });
    if (dataCtx?.columns) {
      for (const col of dataCtx.columns.slice(0, 80)) {
        out.push({ id: `pred:${col}`, kind: "predictor", label: col, page: "Variables" });
      }
    }
    if (dataCtx?.scenarios) {
      for (const s of dataCtx.scenarios) {
        out.push({ id: `scn:${s.name}`, kind: "scenario", label: s.name, hint: s.variable, page: "Scenarios" });
      }
    }
    if (dataCtx?.causalResults) {
      for (const treatment of Object.keys(dataCtx.causalResults)) {
        out.push({
          id: `tr:${treatment}`, kind: "treatment", label: treatment,
          hint: "explain effect",
          run: () => explainHost.value.requestExplain(
            `Explain the causal effect of ${treatment} on the outcome, citing magnitudes, mechanism, and confidence.`,
            "narrator",
          ),
        });
      }
    }
    return out;
  }, [dataCtx, explainHost.value]);
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

  if (!ready || project.rehydrating) return <Splash />;

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
        return <ProcessingPage />;
      case "DAG":
        return <DAGPage key={refreshKey} onNavigate={(p) => navigate(p as AppPage)} />;
      case "Variables":
        return <VariablesPage />;
      case "Physics":
        return <PhysicsPage />;
      case "CRS":
        return <CRSPage />;
      case "Scenarios":
        return <ScenariosPage />;
      case "Models":
        return <ModelsPage />;
      case "Run":
        return <RunPage />;
      case "Results":
        return <ResultsPage />;
      case "Budget":
        return <BudgetOptimizerPage />;
      case "Decisions":
        return <DecisionsPage />;
      case "Compare":
        return <ComparePage />;
      case "Report":
        return <ReportPage />;
      case "Settings":
        return <SettingsPage />;
    }
  };

  return (
    <NotificationContext.Provider value={notif}>
      <ExplainContext.Provider value={explainHost.value}>
      <PipelineProvider serverReady={ready}>
        <div style={{ position: "relative", height: "100vh", width: "100vw", overflow: "hidden" }}>
          <Shell
            currentPage={page as any}
            onNavigate={navigate}
            onToggleChat={() => setChatOpen((o) => !o)}
            chatOpen={chatOpen}
            projectLoaded={project.projectLoaded}
            projectName={project.projectPath?.split(/[\\/]/).pop()?.replace(".yml", "") ?? undefined}
            projectDomain={undefined}
            projectEpsg={undefined}
            status={status}
          >
            {renderPage()}
          </Shell>

          {chatOpen && (
            <ChatPanel
              onAction={handleAction}
              systemPrompt={systemPrompt}
              onClose={() => setChatOpen(false)}
              seedMessage={explainSeed?.message}
              seedNonce={explainSeed?.nonce}
              onSeedConsumed={explainHost.value.consumeSeed}
            />
          )}

          <CommandPalette
            open={paletteOpen}
            onClose={() => setPaletteOpen(false)}
            items={paletteItems}
            onNavigate={(p) => navigate(p)}
          />

          <OnboardingTour onNavigate={(p) => navigate(p as PageName)} />

          <NotificationBanner />
        </div>
      </PipelineProvider>
      </ExplainContext.Provider>
    </NotificationContext.Provider>
  );
}
