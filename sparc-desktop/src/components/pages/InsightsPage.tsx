/**
 * Insights workspace — the new linked-dashboard page that replaces
 * ResultsPage + the visualization halves of CausalPage / ScenariosPage /
 * DecisionSupportPage.
 *
 * Phase 1: panel registry is wired but most panels are placeholders.
 * Phase 2 fills in real renderers + brushed linking across them.
 */
import { useMemo, useCallback } from "react";
import { InsightsProvider } from "@/hooks/InsightsProvider";
import InsightsShell, { type InsightsPanelDescriptor } from "@/components/insights/InsightsShell";
import { useManifest } from "@/hooks/useManifest";
import { useResult } from "@/hooks/useResult";
import { getModelPerformance, downloadResultsBundle } from "@/lib/api";
import { Btn } from "@/components/ui/DesignSystem";
import { useNotification } from "@/hooks/useNotifications";
import OverviewPanel from "@/components/insights/panels/OverviewPanel";
import ModelPerformancePanel from "@/components/insights/panels/ModelPerformancePanel";
import PredictionsMapPanel from "@/components/insights/panels/PredictionsMapPanel";
import CorrelogramPanel from "@/components/insights/panels/CorrelogramPanel";
import DoseResponsePanel from "@/components/insights/panels/DoseResponsePanel";
import CatePanel from "@/components/insights/panels/CatePanel";
import SensitivityPanel from "@/components/insights/panels/SensitivityPanel";
import NegativeControlPanel from "@/components/insights/panels/NegativeControlPanel";
import ScenarioMapPanel from "@/components/insights/panels/ScenarioMapPanel";
import ScenarioStripPanel from "@/components/insights/panels/ScenarioStripPanel";
import PdpPanel from "@/components/insights/panels/PdpPanel";
import ArtifactBrowserPanel from "@/components/insights/panels/ArtifactBrowserPanel";
import HeadlinePanel from "@/components/insights/panels/HeadlinePanel";
import EquityCostPanel from "@/components/insights/panels/EquityCostPanel";
import DivergencePanel from "@/components/insights/panels/DivergencePanel";
import KernelFieldPanel from "@/components/insights/panels/KernelFieldPanel";
import ScenarioTrajectoryPanel from "@/components/insights/panels/ScenarioTrajectoryPanel";
import ScenarioUncertaintyPanel from "@/components/insights/panels/ScenarioUncertaintyPanel";
import RoutingAuditPanel from "@/components/insights/panels/RoutingAuditPanel";
import DatasetProfilePanel from "@/components/insights/panels/DatasetProfilePanel";

function buildPanels(): InsightsPanelDescriptor[] {
  return [
    // ---- Overview (all audiences) ----
    {
      id: "overview",
      group: "Overview",
      label: "At a glance",
      audience: ["practitioner", "researcher", "public"],
      render: () => <OverviewPanel />,
    },

    // ---- Practitioner & public: decisions ----
    {
      id: "headline",
      group: "Decisions",
      label: "Best intervention",
      audience: ["practitioner", "public"],
      stage: "Stage 4",
      render: () => <HeadlinePanel />,
    },
    {
      id: "scenario-strip",
      group: "Decisions",
      label: "Scenarios",
      audience: ["practitioner", "public"],
      render: () => <ScenarioStripPanel />,
    },
    {
      id: "equity",
      group: "Decisions",
      label: "Equity & cost",
      audience: ["practitioner", "public"],
      render: () => <EquityCostPanel />,
    },

    // ---- Spatial (all audiences) ----
    {
      id: "predictions",
      group: "Spatial",
      label: "Predictions map",
      audience: ["practitioner", "researcher", "public"],
      stage: "Stage 2",
      render: () => <PredictionsMapPanel />,
    },
    {
      id: "scenario-map",
      group: "Spatial",
      label: "Scenario map",
      audience: ["practitioner", "researcher", "public"],
      render: () => <ScenarioMapPanel />,
    },
    {
      id: "cate-map",
      group: "Spatial",
      label: "Heterogeneous effects (CATE)",
      audience: "researcher",
      render: () => <CatePanel />,
    },

    // ---- Researcher: model evidence ----
    {
      id: "dataset-profile",
      group: "Evidence",
      label: "Dataset profile",
      audience: "researcher",
      stage: "Stage 0–2",
      render: () => <DatasetProfilePanel />,
    },
    {
      id: "model-perf",
      group: "Evidence",
      label: "Model performance",
      audience: "researcher",
      render: () => <ModelPerformancePanel />,
    },
    {
      id: "correlogram",
      group: "Evidence",
      label: "Correlogram",
      audience: "researcher",
      render: () => <CorrelogramPanel />,
    },
    {
      id: "pdp",
      group: "Evidence",
      label: "Response curves (PDP)",
      audience: "researcher",
      render: () => <PdpPanel />,
    },
    {
      id: "dose-response",
      group: "Evidence",
      label: "Dose-response",
      audience: ["practitioner", "researcher", "public"],
      render: () => <DoseResponsePanel />,
    },

    // ---- Researcher: causal diagnostics ----
    {
      id: "sensitivity",
      group: "Causal diagnostics",
      label: "Sensitivity",
      audience: "researcher",
      stage: "Stage 3",
      render: () => <SensitivityPanel />,
    },
    {
      id: "negative-control",
      group: "Causal diagnostics",
      label: "Negative control",
      audience: "researcher",
      render: () => <NegativeControlPanel />,
    },
    {
      id: "divergence",
      group: "Causal diagnostics",
      label: "Divergence audit",
      audience: "researcher",
      render: () => <DivergencePanel />,
    },
    {
      id: "kernel-field",
      group: "Causal diagnostics",
      label: "Kernel field",
      audience: "researcher",
      render: () => <KernelFieldPanel />,
    },

    // ---- Researcher: scenario depth ----
    {
      id: "scenario-trajectory",
      group: "Scenario depth",
      label: "Trajectory",
      audience: "researcher",
      stage: "Stage 4",
      render: () => <ScenarioTrajectoryPanel />,
    },
    {
      id: "scenario-uncertainty",
      group: "Scenario depth",
      label: "Uncertainty bands",
      audience: "researcher",
      render: () => <ScenarioUncertaintyPanel />,
    },
    {
      id: "routing-audit",
      group: "Scenario depth",
      label: "Routing audit",
      audience: "researcher",
      render: () => <RoutingAuditPanel />,
    },

    // ---- Raw artifact browser ----
    {
      id: "artifacts",
      group: "Raw",
      label: "Artifact browser",
      audience: "researcher",
      render: () => <ArtifactBrowserPanel />,
    },
  ];
}

function PipelineHealthBadge() {
  const manifest = useManifest();
  const stageCount = Object.keys(manifest.manifest?.stages ?? {}).length;
  const perfPresent = !!manifest.lookup("2", "ensemble_results");
  const { data: perfData } = useResult(
    perfPresent ? "s2:model_performance" : null,
    getModelPerformance,
  );
  const bestR2 = useMemo(() => {
    if (!perfData?.models?.length) return null;
    return perfData.models.reduce((a: any, b: any) => (b.r2 > a.r2 ? b : a)).r2 as number;
  }, [perfData]);

  if (stageCount === 0) return null;

  return (
    <div
      className="mono"
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "6px 12px",
        border: "1px solid var(--line)",
        borderRadius: 6,
        background: "#fdfbf7",
        fontSize: 10,
        color: "var(--ink-2)",
        letterSpacing: "0.06em",
        textTransform: "uppercase",
        whiteSpace: "nowrap",
      }}
    >
      <span
        style={{
          width: 6,
          height: 6,
          borderRadius: "50%",
          background: stageCount >= 5 ? "var(--green, #2f7d32)" : "var(--amber)",
          flexShrink: 0,
        }}
      />
      {stageCount}/5 stages
      {bestR2 != null && (
        <>
          <span style={{ color: "var(--line)" }}>·</span>
          R² {bestR2.toFixed(3)}
        </>
      )}
    </div>
  );
}

export default function InsightsPage() {
  const panels = useMemo(buildPanels, []);
  const { notify } = useNotification();
  const handleDataBundle = useCallback(async () => {
    notify("info", "Preparing results bundle…");
    try {
      const blob = await downloadResultsBundle();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "sparc_results_bundle.zip";
      a.click();
      URL.revokeObjectURL(url);
      notify("success", "Results bundle downloaded");
    } catch (e) {
      notify("error", e instanceof Error ? e.message : "Bundle download failed");
    }
  }, [notify]);
  return (
    <InsightsProvider>
      <InsightsShell
        panels={panels}
        headerExtras={
          <>
            <Btn small onClick={handleDataBundle}>Download ZIP</Btn>
            <PipelineHealthBadge />
          </>
        }
      />
    </InsightsProvider>
  );
}
