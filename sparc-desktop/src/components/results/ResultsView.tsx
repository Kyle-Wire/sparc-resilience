import { useState } from "react";
import CorrelogramView from "@/components/results/CorrelogramView";
import GwenView from "@/components/results/GwenView";
import SpatialCvView from "@/components/results/SpatialCvView";
import CausalView from "@/components/results/CausalView";
import ScenarioResultsView from "@/components/results/ScenarioResultsView";
import VariableDashboardView from "@/components/results/VariableDashboardView";
import ArtifactPanel from "@/components/results/ArtifactPanel";

const STAGES = [
  { value: 0, label: "Stage 0 — Correlogram" },
  { value: 1, label: "Stage 1 — GWEN" },
  { value: 2, label: "Stage 2 — Spatial CV" },
  { value: 3, label: "Stage 3 — Causal" },
  { value: 4, label: "Stage 4 — Scenarios" },
  { value: 5, label: "Variable Dashboard" },
  { value: 6, label: "Artifacts" },
];

export default function ResultsView() {
  const [stage, setStage] = useState(0);

  const StageComponent = () => {
    switch (stage) {
      case 0: return <CorrelogramView />;
      case 1: return <GwenView />;
      case 2: return <SpatialCvView />;
      case 3: return <CausalView />;
      case 4: return <ScenarioResultsView />;
      case 5: return <VariableDashboardView />;
      case 6: return <ArtifactPanel />;
      default: return <p className="p-4 text-sm text-sparc-gray-600">Select a stage above.</p>;
    }
  };

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center gap-4 border-b border-sparc-gray-200 px-4 py-3">
        <h1 className="text-lg font-bold">Results</h1>
        <div className="flex gap-1">
          {STAGES.map((s) => (
            <button
              key={s.value}
              onClick={() => setStage(s.value)}
              className={`rounded px-3 py-1.5 text-xs font-medium transition-colors ${
                stage === s.value
                  ? "bg-sparc-purple text-white"
                  : "border border-sparc-gray-300 hover:bg-sparc-gray-100"
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      {/* Stage content */}
      <div className="flex-1 overflow-hidden">
        <StageComponent />
      </div>
    </div>
  );
}
