/**
 * SensitivityPanel — Causal sensitivity table (E-values + CIs).
 * Stage 3 diagnostics. Researcher-only.
 */
import { Panel, PanelEmpty, Pill } from "@/components/ui/DesignSystem";
import { PanelGate } from "@/components/ui/PanelGate";
import { useManifest } from "@/hooks/useManifest";
import { useResult } from "@/hooks/useResult";
import { getCausalSensitivity, getCausalResults, type CausalSensitivity } from "@/lib/api";
import type { CausalResults, SensitivityBounds } from "@/lib/types";

export default function SensitivityPanel() {
  const manifest = useManifest();
  // Sensitivity is derived live from `/results/causal` (which needs
  // stage-3 `scenario_coefficients`).
  const present =
    !!manifest.lookup("3", "scenario_coefficients") ||
    !!manifest.lookup("3", "causal_diagnostics");
  const { data } = useResult<CausalSensitivity>(
    present ? "s3:causal_sensitivity" : null,
    getCausalSensitivity,
  );
  const { data: causalData } = useResult<CausalResults>(
    present ? "s3:causal_results" : null,
    getCausalResults,
  );

  // Collect Rosenbaum bounds from direct_effects
  const boundsEntries: { treatment: string; bounds: SensitivityBounds }[] = [];
  if (causalData?.direct_effects) {
    for (const [treatment, effect] of Object.entries(causalData.direct_effects)) {
      if (effect.sensitivity_bounds) {
        boundsEntries.push({ treatment, bounds: effect.sensitivity_bounds });
      }
    }
  }

  return (
    <PanelGate panelId="sensitivity" title="Sensitivity analysis" subtitle="stage 3 · causal">
      {!data?.results?.length ? (
        <Panel title="Sensitivity analysis" subtitle="stage 3 · causal">
          <PanelEmpty reason="Loading…" />
        </Panel>
      ) : (
        <>
          <SensitivityTable data={data} />
          {boundsEntries.length > 0 && <RosenbaumboundsTable entries={boundsEntries} />}
        </>
      )}
    </PanelGate>
  );
}

/** E-value robustness levels:
 * Fragile  < 1.5  — a weak confounder could explain away the effect
 * Moderate 1.5–3  — moderate confounding needed
 * Robust   > 3    — strong confounding required; result is credible
 */
function eValueLevel(v: number): { label: string; color: string; tip: string } {
  if (v >= 3) return {
    label: "Robust",
    color: "var(--green, #2f7d32)",
    tip: `E=${v.toFixed(2)}: An unmeasured confounder would need to more than triple both the exposure-outcome and confounder-outcome risk ratios to fully explain this effect. The finding is credible.`,
  };
  if (v >= 1.5) return {
    label: "Moderate",
    color: "var(--amber, #e79024)",
    tip: `E=${v.toFixed(2)}: A moderately strong unmeasured confounder could explain away this effect. Interpret with caution.`,
  };
  return {
    label: "Fragile",
    color: "var(--crimson, #c0392b)",
    tip: `E=${v.toFixed(2)}: Even a weak unmeasured confounder could nullify this finding. Results should not be acted on alone.`,
  };
}

function EValueBadge({ value }: { value: number }) {
  const { label, color, tip } = eValueLevel(value);
  return (
    <span title={tip} style={{ cursor: "help", display: "inline-flex", alignItems: "center", gap: 4 }}>
      <Pill color={color}>{label}</Pill>
      <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--ink-2)" }}>
        {value.toFixed(2)}
      </span>
    </span>
  );
}

function SensitivityTable({ data }: { data: CausalSensitivity }) {
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
                  <EValueBadge value={r.e_value_point} />
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

function RosenbaumboundsTable({
  entries,
}: {
  entries: { treatment: string; bounds: SensitivityBounds }[];
}) {
  return (
    <Panel
      title="Partial-identification bounds"
      subtitle={`Rosenbaum · γ=${entries[0]?.bounds.gamma ?? "?"}`}
    >
      <div style={{ overflowX: "auto" }}>
        <table
          className="mono"
          style={{ width: "100%", fontSize: 11, borderCollapse: "collapse" }}
        >
          <thead>
            <tr style={{ textAlign: "left", borderBottom: "1px solid var(--line)" }}>
              <th style={{ padding: "6px 8px" }}>Treatment</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>Effect</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>Lower</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>Upper</th>
              <th style={{ padding: "6px 8px" }}>Null included?</th>
              <th style={{ padding: "6px 8px" }}>Interpretation</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(({ treatment, bounds }) => (
              <tr
                key={treatment}
                style={{ borderBottom: "1px solid rgba(0,0,0,0.04)" }}
              >
                <td style={{ padding: "6px 8px" }}>{treatment}</td>
                <td style={{ padding: "6px 8px", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                  {bounds.effect.toFixed(4)}
                </td>
                <td style={{ padding: "6px 8px", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                  {bounds.lower_bound.toFixed(4)}
                </td>
                <td style={{ padding: "6px 8px", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                  {bounds.upper_bound.toFixed(4)}
                </td>
                <td style={{ padding: "6px 8px" }}>
                  <Pill color={bounds.null_included ? "var(--orange, #e65100)" : "var(--green, #2f7d32)"}>
                    {bounds.null_included ? "Yes" : "No"}
                  </Pill>
                </td>
                <td style={{ padding: "6px 8px", color: "var(--ink-2)", fontFamily: "Inter, sans-serif" }}>
                  {bounds.interpretation}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
