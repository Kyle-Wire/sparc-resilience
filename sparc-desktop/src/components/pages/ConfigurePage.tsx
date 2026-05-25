/**
 * ConfigurePage — collapsed analysis configuration (Candidate A).
 *
 * Replaces the five shallow sidebar entries (DAG, Variables, Physics,
 * Scenarios, Models) with a single deep module. Four tabs surface the
 * common paths; Physics and Models live behind an "Advanced" toggle so
 * they don't intimidate planners who never touch them.
 *
 * Each tab renders the existing page component in full — no internal
 * refactor required at this stage.
 */
import { useState } from "react";
import VariablesPage from "@/components/pages/VariablesPage";
import DAGPage from "@/components/pages/DAGPage";
import ScenariosPage from "@/components/pages/ScenariosPage";
import PhysicsPage from "@/components/pages/PhysicsPage";
import ModelsPage from "@/components/pages/ModelsPage";

type MainTab = "variables" | "dag" | "scenarios";
type AdvancedTab = "physics" | "models";

const TAB_LABELS: { id: MainTab; label: string; hint: string }[] = [
  { id: "variables", label: "Variables",        hint: "Configure predictors, target, and causal roles" },
  { id: "dag",       label: "Causal DAG",       hint: "Define and review the causal graph structure" },
  { id: "scenarios", label: "Scenarios",        hint: "Build what-if intervention scenarios" },
];

const ADVANCED_LABELS: { id: AdvancedTab; label: string }[] = [
  { id: "physics", label: "Physics constraints" },
  { id: "models",  label: "Model hyperparameters" },
];

const tabBtnStyle = (active: boolean): React.CSSProperties => ({
  padding: "7px 16px",
  border: "none",
  borderBottom: active ? "2px solid var(--crimson)" : "2px solid transparent",
  background: "transparent",
  color: active ? "var(--ink)" : "var(--muted)",
  fontWeight: active ? 700 : 500,
  fontSize: 13,
  cursor: "pointer",
  fontFamily: "inherit",
  transition: "color 0.12s, border-color 0.12s",
  whiteSpace: "nowrap" as const,
});

export default function ConfigurePage() {
  const [tab, setTab] = useState<MainTab>("variables");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [advancedTab, setAdvancedTab] = useState<AdvancedTab>("physics");

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      {/* Tab bar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          borderBottom: "1px solid var(--line)",
          background: "var(--paper)",
          padding: "0 16px",
          gap: 2,
          flexShrink: 0,
        }}
      >
        {TAB_LABELS.map(({ id, label }) => (
          <button
            key={id}
            style={tabBtnStyle(!advancedOpen && tab === id)}
            title={TAB_LABELS.find((t) => t.id === id)?.hint}
            onClick={() => { setTab(id); setAdvancedOpen(false); }}
          >
            {label}
          </button>
        ))}

        {/* Divider */}
        <div style={{ width: 1, height: 20, background: "var(--line)", margin: "0 8px" }} />

        {/* Advanced toggle */}
        <button
          style={{
            ...tabBtnStyle(advancedOpen),
            display: "flex",
            alignItems: "center",
            gap: 5,
            fontSize: 12,
            color: advancedOpen ? "var(--amber)" : "var(--muted)",
            borderBottomColor: advancedOpen ? "var(--amber)" : "transparent",
          }}
          onClick={() => setAdvancedOpen((o) => !o)}
          title="Physics constraints and model hyperparameters"
        >
          Advanced
          <span style={{ fontSize: 9, opacity: 0.7 }}>{advancedOpen ? "▲" : "▼"}</span>
        </button>

        {advancedOpen && ADVANCED_LABELS.map(({ id, label }) => (
          <button
            key={id}
            style={{
              ...tabBtnStyle(advancedTab === id),
              fontSize: 12,
              borderBottomColor: advancedTab === id ? "var(--amber)" : "transparent",
              color: advancedTab === id ? "var(--amber)" : "var(--muted)",
            }}
            onClick={() => setAdvancedTab(id)}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Tab content — full height, scrollable inside */}
      <div style={{ flex: 1, overflow: "auto", minHeight: 0 }}>
        {!advancedOpen && tab === "variables"  && <VariablesPage />}
        {!advancedOpen && tab === "dag"        && <DAGPage />}
        {!advancedOpen && tab === "scenarios"  && <ScenariosPage />}
        {advancedOpen  && advancedTab === "physics" && <PhysicsPage />}
        {advancedOpen  && advancedTab === "models"  && <ModelsPage />}
      </div>
    </div>
  );
}
