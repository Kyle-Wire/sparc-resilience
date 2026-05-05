/**
 * Insights workspace — the new linked-dashboard page that replaces
 * ResultsPage + the visualization halves of CausalPage / ScenariosPage /
 * DecisionSupportPage.
 *
 * Phase 1: panel registry is wired but most panels are placeholders.
 * Phase 2 fills in real renderers + brushed linking across them.
 */
import { useMemo } from "react";
import { InsightsProvider } from "@/hooks/InsightsProvider";
import InsightsShell, { type InsightsPanelDescriptor } from "@/components/insights/InsightsShell";
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
import DatasetProfilePanel from "@/components/insights/panels/DatasetProfilePanel";

function buildPanels(): InsightsPanelDescriptor[] {
  return [
    // ---- Overview (both audiences) ----
    {
      id: "overview",
      group: "Overview",
      label: "At a glance",
      audience: ["practitioner", "researcher"],
      render: () => <OverviewPanel />,
    },

    // ---- Practitioner: decisions ----
    {
      id: "headline",
      group: "Decisions",
      label: "Best intervention",
      audience: "practitioner",
      render: () => <HeadlinePanel />,
    },
    {
      id: "scenario-strip",
      group: "Decisions",
      label: "Scenarios",
      audience: "practitioner",
      render: () => <ScenarioStripPanel />,
    },
    {
      id: "equity",
      group: "Decisions",
      label: "Equity & cost",
      audience: "practitioner",
      render: () => <EquityCostPanel />,
    },

    // ---- Spatial (both audiences) ----
    {
      id: "predictions",
      group: "Spatial",
      label: "Predictions map",
      audience: ["practitioner", "researcher"],
      render: () => <PredictionsMapPanel />,
    },
    {
      id: "scenario-map",
      group: "Spatial",
      label: "Scenario map",
      audience: ["practitioner", "researcher"],
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
      audience: ["practitioner", "researcher"],
      render: () => <DoseResponsePanel />,
    },

    // ---- Researcher: causal diagnostics ----
    {
      id: "sensitivity",
      group: "Causal diagnostics",
      label: "Sensitivity",
      audience: "researcher",
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
      render: () => <ScenarioTrajectoryPanel />,
    },
    {
      id: "scenario-uncertainty",
      group: "Scenario depth",
      label: "Uncertainty bands",
      audience: "researcher",
      render: () => <ScenarioUncertaintyPanel />,
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

export default function InsightsPage() {
  const panels = useMemo(buildPanels, []);
  return (
    <InsightsProvider>
      <InsightsShell panels={panels} />
    </InsightsProvider>
  );
}
