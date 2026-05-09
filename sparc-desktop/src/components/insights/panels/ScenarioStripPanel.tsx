/**
 * ScenarioStripPanel — strip of saved scenarios as clickable cards.
 * Click sets linked-selection.scenario, which the ScenarioMapPanel
 * picks up to switch its `pred_<scenario>` layer.
 */
import { Panel, PanelEmpty, Pill, Btn } from "@/components/ui/DesignSystem";
import { PanelGate } from "@/components/ui/PanelGate";
import { useManifest } from "@/hooks/useManifest";
import { useLinkedSelection } from "@/hooks/useLinkedSelection";
import { useInsightsNavigate } from "@/hooks/InsightsProvider";
import { useResult } from "@/hooks/useResult";
import { getScenarioDetail } from "@/lib/api";
import type { ScenarioDetail } from "@/lib/types";

interface SummaryRow {
  scenario_name?: string;
  scenario?: string;
  name?: string;
  delta_mean?: number;
  delta_pct?: number;
  description?: string;
  [k: string]: unknown;
}

export default function ScenarioStripPanel() {
  const manifest = useManifest();
  const linked = useLinkedSelection();
  const navigate = useInsightsNavigate();
  const present = !!manifest.stage("4");
  const { data: detail } = useResult<ScenarioDetail>(
    present ? "s4:scenario_detail" : null,
    getScenarioDetail,
  );

  if (!present) {
    return (
      <PanelGate panelId="scenario_strip" title="Saved scenarios" subtitle="stage 4">
        <Panel title="Saved scenarios" subtitle="stage 4">
          <PanelEmpty
            reason="No scenarios run"
            hint="Configure scenarios on the Scenarios page and run stage 4."
            action={navigate && <Btn small onClick={() => navigate("Run")}>Go to Run page →</Btn>}
          />
        </Panel>
      </PanelGate>
    );
  }

  const rows = (detail?.summary ?? []) as SummaryRow[];
  if (!rows.length) {
    return (
      <Panel title="Saved scenarios" subtitle="stage 4">
        <PanelEmpty reason="Loading…" />
      </Panel>
    );
  }

  const active = linked.selection.scenario;

  return (
    <Panel title="Saved scenarios" subtitle={`${rows.length} variant${rows.length === 1 ? "" : "s"}`}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
          gap: 10,
        }}
      >
        {rows.map((r, i) => {
          const name = (r.scenario_name ?? r.scenario ?? r.name ?? `Scenario ${i + 1}`) as string;
          const isActive = active === name;
          const delta = typeof r.delta_mean === "number" ? r.delta_mean : null;
          const dpct = typeof r.delta_pct === "number" ? r.delta_pct : null;
          return (
            <button
              key={name}
              onClick={() => linked.setScenario(isActive ? null : name)}
              className="panel"
              style={{
                textAlign: "left",
                padding: 12,
                cursor: "pointer",
                background: isActive ? "var(--paper-3, #ece4d4)" : undefined,
                borderColor: isActive ? "var(--purple)" : undefined,
              }}
            >
              <div className="mono" style={{ fontSize: 11, color: "var(--ink-2)", marginBottom: 4 }}>
                #{i + 1}
              </div>
              <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>{name}</div>
              {(delta != null || dpct != null) && (
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  {delta != null && (
                    <Pill color={delta >= 0 ? "var(--green, #2f7d32)" : "var(--crimson, #b91c1c)"}>
                      Δ {delta >= 0 ? "+" : ""}
                      {delta.toFixed(2)}
                    </Pill>
                  )}
                  {dpct != null && (
                    <span className="mono" style={{ fontSize: 10, color: "var(--ink-2)" }}>
                      {dpct >= 0 ? "+" : ""}
                      {dpct.toFixed(1)}%
                    </span>
                  )}
                </div>
              )}
              {typeof r.description === "string" && (
                <div style={{ fontSize: 11, color: "var(--ink-2)", marginTop: 6 }}>{r.description}</div>
              )}
            </button>
          );
        })}
      </div>
    </Panel>
  );
}
