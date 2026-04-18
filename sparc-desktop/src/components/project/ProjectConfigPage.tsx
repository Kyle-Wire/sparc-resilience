import { useState } from "react";
import VariablesView from "@/components/pipeline/VariablesView";
import CRSView from "@/components/pipeline/CRSView";
import PhysicsView from "@/components/pipeline/PhysicsView";
import ScenariosView from "@/components/pipeline/ScenariosView";
import ModelsView from "@/components/pipeline/ModelsView";
import ProjectMetadataForm from "@/components/project/ProjectMetadataForm";

const TABS = [
  { id: "metadata", label: "Metadata" },
  { id: "variables", label: "Variables" },
  { id: "crs", label: "CRS" },
  { id: "physics", label: "Physics" },
  { id: "scenarios", label: "Scenarios" },
  { id: "models", label: "Models" },
] as const;

type TabId = (typeof TABS)[number]["id"];

interface Props {
  /** Start on a specific tab (e.g. when navigating from sidebar "Variables" link). */
  initialTab?: TabId;
}

export default function ProjectConfigPage({ initialTab }: Props) {
  const [tab, setTab] = useState<TabId>(initialTab ?? "metadata");

  const renderTab = () => {
    switch (tab) {
      case "metadata": return <ProjectMetadataForm />;
      case "variables": return <VariablesView />;
      case "crs": return <CRSView />;
      case "physics": return <PhysicsView />;
      case "scenarios": return <ScenariosView />;
      case "models": return <ModelsView />;
    }
  };

  return (
    <div className="flex h-full flex-col">
      {/* Tab bar */}
      <div className="flex items-center gap-1 border-b border-sparc-gray-200 px-4 py-2">
        <h1 className="mr-4 text-lg font-bold">Configuration</h1>
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`rounded px-3 py-1.5 text-xs font-medium transition-colors ${
              tab === t.id
                ? "bg-sparc-purple text-white"
                : "border border-sparc-gray-300 hover:bg-sparc-gray-100"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-auto p-6">
        {renderTab()}
      </div>
    </div>
  );
}
