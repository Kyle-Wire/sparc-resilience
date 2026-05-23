import React, { useState, useCallback, useEffect, useMemo } from "react";
import { useServer } from "@/hooks/useServer";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { NotificationContext, useNotificationState } from "@/hooks/useNotifications";
import { PipelineProvider } from "@/hooks/PipelineProvider";
import { useProjectStore } from "@/stores/projectStore";
import { useNavigationStore } from "@/stores/navigationStore";
import NotificationBanner from "@/components/layout/NotificationBanner";
import ServerLostBanner from "@/components/layout/ServerLostBanner";
import Splash from "@/components/layout/Splash";
import LoginScreen from "@/components/layout/LoginScreen";
import AuthGate from "@/components/layout/AuthGate";
import Shell from "@/components/layout/Shell";
import type { PageName } from "@/components/layout/Sidebar";
import { PAGES } from "@/components/layout/Sidebar";
import ChatPanel from "@/components/chat/ChatPanel";
import CommandPalette, { type PaletteItem } from "@/components/common/CommandPalette";
import OnboardingTour from "@/components/common/OnboardingTour";
import ErrorBoundary from "@/components/common/ErrorBoundary";
import UpdateBanner from "@/components/updater/UpdateBanner";
import { ExplainContext, useExplainHost } from "@/hooks/ExplainContext";
import { useAuthStore } from "@/stores/authStore";
import { buildSystemPrompt } from "@/lib/prompts";
import { useStartup } from "@/hooks/useStartup";
import { useProjectContext } from "@/hooks/useProjectContext";
import { useChatActions } from "@/hooks/useChatActions";

import ProjectPage from "@/components/pages/ProjectPage";
import DataPage from "@/components/pages/DataPage";
import DataCollectionPage from "@/components/pages/DataCollectionPage";
import DAGPage from "@/components/pages/DAGPage";
import VariablesPage from "@/components/pages/VariablesPage";
import PhysicsPage from "@/components/pages/PhysicsPage";
import ScenariosPage from "@/components/pages/ScenariosPage";
import ModelsPage from "@/components/pages/ModelsPage";
import RunPage from "@/components/pages/RunPage";
import StageResultsPage from "@/components/results/StageResultsPage";
import DecisionSupportPage from "@/components/pages/DecisionSupportPage";
import ReportPage from "@/components/pages/ReportPage";
import SettingsPage from "@/components/pages/SettingsPage";
import PerformancePage from "@/components/pages/PerformancePage";


export default function App() {
  const { ready, serverLost, startupFailed, status, restart } = useServer();
  const notif = useNotificationState();
  const project = useProjectStore();
  const { currentPage: page, navigate } = useNavigationStore();
  const { user, signOut } = useAuthStore();
  const [chatOpen, setChatOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const explainHost = useExplainHost();
  const explainSeed = explainHost.seed;

  const { splashStep, splashDone, parallaxEnabled, setParallaxEnabled, handleRetry } =
    useStartup(ready, startupFailed, restart);
  const { dataCtx, projectDomain, projectEpsg, refreshKey, refresh } =
    useProjectContext(ready, project.projectLoaded);
  const { handleAction } = useChatActions(navigate, notif, project, refresh);

  // Trigger project rehydration once the sidecar is ready
  useEffect(() => {
    if (ready) project.rehydrate();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);

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
      refresh,
      openPalette: () => setPaletteOpen(true),
    }),
    [navigate, refresh],
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

  // Must be defined before any early returns to satisfy rules-of-hooks
  const navigateToSettings = useCallback(() => navigate("Settings"), [navigate]);

  // Show splash while loading (or on startup failure)
  if (!splashDone || project.rehydrating || splashStep === "failed") {
    return <Splash step={splashStep} parallaxEnabled={parallaxEnabled} onRetry={handleRetry} />;
  }

  // Show login if no authenticated user
  if (!user) {
    return <LoginScreen parallaxEnabled={parallaxEnabled} />;
  }

  const renderPage = () => {
    const gate = (el: React.ReactNode) => (
      <AuthGate onSignIn={navigateToSettings}>{el}</AuthGate>
    );
    switch (page) {
      case "Project":
        return gate(<ProjectPage />);
      case "Data":
        return gate(<DataPage key={refreshKey} />);
      case "Data Collection":
        return gate(<DataCollectionPage />);
      case "DAG":
        return gate(<DAGPage key={refreshKey} />);
      case "Variables":
        return gate(<VariablesPage />);
      case "Physics":
        return gate(<PhysicsPage />);
      case "Scenarios":
        return gate(<ScenariosPage />);
      case "Models":
        return gate(<ModelsPage />);
      case "Run":
        return gate(<RunPage />);
      case "Results":
        return gate(<StageResultsPage />);
      case "Decision Support":
        return gate(<DecisionSupportPage />);
      case "Report":
        return gate(<ReportPage />);
      case "Settings":
        return (
          <SettingsPage
            onSignOut={async () => { await signOut(); }}
            parallaxEnabled={parallaxEnabled}
            onParallaxToggle={setParallaxEnabled}
          />
        );
      case "Performance":
        return <PerformancePage />;
    }
  };

  return (
    <NotificationContext.Provider value={notif}>
      <ExplainContext.Provider value={explainHost.value}>
      <PipelineProvider serverReady={ready} serverLost={serverLost}>
        <div style={{ position: "relative", height: "100vh", width: "100vw", overflow: "hidden" }}>
          <Shell
            currentPage={page as any}
            onNavigate={navigate}
            onToggleChat={() => setChatOpen((o) => !o)}
            chatOpen={chatOpen}
            projectLoaded={project.projectLoaded}
            projectName={project.projectPath?.split(/[\\/]/).pop()?.replace(".yml", "") ?? undefined}
            projectDomain={projectDomain}
            projectEpsg={projectEpsg ? `EPSG:${projectEpsg}` : undefined}
            status={status}
          >
            <ErrorBoundary key={page} page={page}>
              {renderPage()}
            </ErrorBoundary>
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
          <ServerLostBanner serverLost={serverLost} />
          <UpdateBanner />
        </div>
      </PipelineProvider>
      </ExplainContext.Provider>
    </NotificationContext.Provider>
  );
}
