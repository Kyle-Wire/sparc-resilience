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
import CommandPalette from "@/components/common/CommandPalette";
import OnboardingTour from "@/components/common/OnboardingTour";
import ErrorBoundary from "@/components/common/ErrorBoundary";
import UpdateBanner from "@/components/updater/UpdateBanner";
import { ExplainContext, useExplainHost } from "@/hooks/ExplainContext";
import { useAuthStore } from "@/stores/authStore";
import { buildSystemPrompt } from "@/lib/prompts";
import { useStartup } from "@/hooks/useStartup";
import { useProjectContext } from "@/hooks/useProjectContext";
import { useChatActions } from "@/hooks/useChatActions";
import { usePromptMode } from "@/hooks/usePromptMode";
import { useCommandPaletteItems } from "@/hooks/useCommandPaletteItems";

const ProjectPage         = React.lazy(() => import("@/components/pages/ProjectPage"));
const DataPage            = React.lazy(() => import("@/components/pages/DataPage"));
const ConfigurePage       = React.lazy(() => import("@/components/pages/ConfigurePage"));
const RunPage             = React.lazy(() => import("@/components/pages/RunPage"));
const InsightsPage        = React.lazy(() => import("@/components/pages/InsightsPage"));
const DecisionSupportPage = React.lazy(() => import("@/components/pages/DecisionSupportPage"));
const ReportPage          = React.lazy(() => import("@/components/pages/ReportPage"));
const SettingsPage        = React.lazy(() => import("@/components/pages/SettingsPage"));
const PerformancePage     = React.lazy(() => import("@/components/pages/PerformancePage"));


export default function App() {
  const { ready, serverLost, startupFailed, status, restart } = useServer();
  const notif = useNotificationState();
  const project = useProjectStore();
  const { currentPage: page, navigate, workflowStep } = useNavigationStore();
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
  const promptMode = usePromptMode(page, explainSeed);

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
  const paletteItems = useCommandPaletteItems(dataCtx, explainHost.value, navigate, setChatOpen);
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
      case "Configure":
        return gate(<ConfigurePage />);
      case "Run":
        return gate(<RunPage />);
      case "Results":
        return gate(<InsightsPage />);
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
            workflowStep={workflowStep}
          >
            <ErrorBoundary key={page} page={page}>
              <React.Suspense fallback={null}>
                {renderPage()}
              </React.Suspense>
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
