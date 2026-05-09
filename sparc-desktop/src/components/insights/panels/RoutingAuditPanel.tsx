/**
 * RoutingAuditPanel — scenario routing audit.
 *
 * Shows how each intervention variable was routed when computing
 * scenario counterfactuals: causal CATE path vs heuristic/correlational
 * fallback. Researcher-only panel under "Scenario depth".
 */
import { Panel, PanelEmpty, Tag } from "@/components/ui/DesignSystem";
import { useManifest } from "@/hooks/useManifest";
import { useResult } from "@/hooks/useResult";
import { getScenarioRoutingAudit } from "@/lib/api";
import type { ScenarioRoutingAuditData } from "@/lib/types";

export default function RoutingAuditPanel() {
  const manifest = useManifest();
  const stage4 = manifest.stage("4");
  const present =
    !!stage4 && Object.keys(stage4.artifacts ?? {}).some((a) => a.includes("routing"));

  const { data } = useResult<ScenarioRoutingAuditData>(
    present ? "s4:routing_audit" : null,
    getScenarioRoutingAudit,
  );

  if (!present) {
    return (
      <Panel title="Routing audit" subtitle="stage 4 · scenario routing">
        <PanelEmpty
          reason="No routing audit"
          hint="Run Stage 4 (scenario simulation) to generate this artifact."
        />
      </Panel>
    );
  }

  if (!data) {
    return (
      <Panel title="Routing audit" subtitle="stage 4 · scenario routing">
        <PanelEmpty reason="Loading…" />
      </Panel>
    );
  }

  const { summary, variables, routing } = data;

  return (
    <Panel
      title="Routing audit"
      subtitle={`${summary.causal_routed}/${summary.total_variables} variables routed via causal CATE`}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: 12,
          marginBottom: 16,
        }}
      >
        {[
          { label: "Total", value: String(summary.total_variables) },
          { label: "Causal", value: String(summary.causal_routed), tint: "var(--teal, #2a9d8f)" },
          { label: "Heuristic", value: String(summary.correlational_routed), tint: "var(--amber)" },
          { label: "Anisotropic", value: String(summary.anisotropic_predictor_count), tint: "var(--crimson)" },
        ].map(({ label, value, tint }) => (
          <div key={label} style={{ textAlign: "center" }}>
            <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "var(--font-mono)", color: tint ?? "var(--ink)" }}>
              {value}
            </div>
            <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>{label}</div>
          </div>
        ))}
      </div>

      <div style={{ overflowX: "auto" }}>
        <table className="mono" style={{ width: "100%", fontSize: 11, borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ textAlign: "left", borderBottom: "1px solid var(--line)" }}>
              <th style={{ padding: "4px 8px" }}>variable</th>
              <th style={{ padding: "4px 8px" }}>route</th>
              <th style={{ padding: "4px 8px", textAlign: "right" }}>mean multiplier</th>
              <th style={{ padding: "4px 8px", textAlign: "right" }}>cells</th>
            </tr>
          </thead>
          <tbody>
            {variables.map((v) => {
              const r = routing[v];
              const isCausal = r.source === "causal_cate";
              const isMismatch = r.source === "length_mismatch";
              return (
                <tr key={v} style={{ borderBottom: "1px solid #f0f0e8" }}>
                  <td style={{ padding: "5px 8px" }}>{v}</td>
                  <td style={{ padding: "5px 8px" }}>
                    <Tag color={isCausal ? "var(--teal, #2a9d8f)" : isMismatch ? "var(--crimson)" : "var(--amber)"}>
                      {isCausal ? "causal" : isMismatch ? "mismatch" : "heuristic"}
                    </Tag>
                  </td>
                  <td style={{ padding: "5px 8px", textAlign: "right" }}>
                    {r.mean_multiplier != null ? r.mean_multiplier.toFixed(3) : "—"}
                  </td>
                  <td style={{ padding: "5px 8px", textAlign: "right" }}>
                    {r.n_cells ?? r.n_cells_actual ?? "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
