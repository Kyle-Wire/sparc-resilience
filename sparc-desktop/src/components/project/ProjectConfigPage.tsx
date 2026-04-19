import { useState } from "react";

// These views are now standalone pages under /components/pages/.
// This legacy config page is kept for backward compatibility but
// uses inline placeholders instead of the removed pipeline views.
const Placeholder = ({ label }: { label: string }) => (
  <div style={{ padding: 24, color: "#6e6358", fontStyle: "italic" }}>
    {label} — see the dedicated page instead.
  </div>
);

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
      case "metadata": return <Placeholder label="Metadata" />;
      case "variables": return <Placeholder label="Variables" />;
      case "crs": return <Placeholder label="CRS" />;
      case "physics": return <Placeholder label="Physics" />;
      case "scenarios": return <Placeholder label="Scenarios" />;
      case "models": return <Placeholder label="Models" />;
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
