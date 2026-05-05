/**
 * SensitivityPanel — Causal sensitivity table (E-values + CIs).
 * Stage 3 diagnostics. Researcher-only.
 */
import { useEffect, useState } from "react";
import { Panel, PanelEmpty, Pill } from "@/components/ui/DesignSystem";
import { useManifest } from "@/hooks/useManifest";
import { getCausalSensitivity, type CausalSensitivity } from "@/lib/api";

export default function SensitivityPanel() {
  const manifest = useManifest();
  const stage3 = manifest.stage("3");
  const present = !!stage3 && Object.keys(stage3.artifacts ?? {}).some((a) => a.includes("sensitivity"));
  const [data, setData] = useState<CausalSensitivity | null>(null);

  useEffect(() => {
    if (!present) {
      setData(null);
      return;
    }
    getCausalSensitivity().then(setData).catch(() => setData(null));
  }, [present, manifest.lastUpdated]);

  if (!present) {
    return (
      <Panel title="Sensitivity analysis" subtitle="stage 3 · causal">
        <PanelEmpty
          reason="No sensitivity output"
          hint="Run the Causal stage; sensitivity ships alongside dose-response."
        />
      </Panel>
    );
  }
  if (!data?.results?.length) {
    return (
      <Panel title="Sensitivity analysis" subtitle="stage 3 · causal">
        <PanelEmpty reason="Loading…" />
      </Panel>
    );
  }

  return (
    <Panel title="Sensitivity analysis" subtitle={`method · ${data.method}`}>
      <div style={{ overflowX: "auto" }}>
        <table className="mono" style={{ width: "100%", fontSize: 11, borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ textAlign: "left", borderBottom: "1px solid var(--line)" }}>
              <th style={{ padding: "6px 8px" }}>Effect</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>Estimate</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>95% CI</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>RR</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>E-value</th>
              <th style={{ padding: "6px 8px" }}>Interpretation</th>
            </tr>
          </thead>
          <tbody>
            {data.results.map((r, i) => (
              <tr key={i} style={{ borderBottom: "1px solid rgba(0,0,0,0.04)" }}>
                <td style={{ padding: "6px 8px" }}>{r.effect_label}</td>
                <td style={{ padding: "6px 8px", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                  {r.point_estimate.toFixed(3)}
                </td>
                <td style={{ padding: "6px 8px", textAlign: "right", fontVariantNumeric: "tabular-nums", color: "var(--ink-2)" }}>
                  {r.ci_lower != null && r.ci_upper != null
                    ? `${r.ci_lower.toFixed(2)}, ${r.ci_upper.toFixed(2)}`
                    : "—"}
                </td>
                <td style={{ padding: "6px 8px", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                  {r.rr_equivalent.toFixed(2)}
                </td>
                <td style={{ padding: "6px 8px", textAlign: "right" }}>
                  <Pill color={r.e_value_point >= 2 ? "var(--green, #2f7d32)" : "var(--ink-2)"}>
                    {r.e_value_point.toFixed(2)}
                  </Pill>
                </td>
                <td style={{ padding: "6px 8px", color: "var(--ink-2)", fontFamily: "Inter, sans-serif" }}>
                  {r.interpretation}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
