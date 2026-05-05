/**
 * DivergencePanel — CATE vs GWR divergence audit.
 *
 * Surfaces per-treatment flag rate, mean |Δ|, sign agreement, and
 * Pearson correlation. Practitioner mode would collapse this to a
 * single confidence chip; researcher mode shows the full table.
 */
import { useEffect, useMemo, useState } from "react";
import { Panel, PanelEmpty, Pill } from "@/components/ui/DesignSystem";
import { useManifest } from "@/hooks/useManifest";
import { getCateVsGwrDivergence } from "@/lib/api";
import type { DivergenceReport } from "@/lib/types";

export default function DivergencePanel() {
  const manifest = useManifest();
  const stage3 = manifest.stage("3");
  const present =
    !!stage3 && Object.keys(stage3.artifacts ?? {}).some((a) => a.includes("divergence"));
  const [data, setData] = useState<{ reports: Record<string, DivergenceReport> } | DivergenceReport | null>(
    null,
  );

  useEffect(() => {
    if (!present) {
      setData(null);
      return;
    }
    getCateVsGwrDivergence().then(setData).catch(() => setData(null));
  }, [present, manifest.lastUpdated]);

  const reports = useMemo<Record<string, DivergenceReport>>(() => {
    if (!data) return {};
    if ("reports" in data && data.reports) return data.reports;
    if ("treatment" in data) return { [(data as DivergenceReport).treatment]: data as DivergenceReport };
    return {};
  }, [data]);

  const names = Object.keys(reports);

  if (!present) {
    return (
      <Panel title="Divergence audit" subtitle="stage 3 · CATE vs GWR">
        <PanelEmpty
          reason="No divergence audit"
          hint="Run the Causal stage with both Bayesian CATE and GWR baseline."
        />
      </Panel>
    );
  }
  if (!names.length) {
    return (
      <Panel title="Divergence audit" subtitle="stage 3 · CATE vs GWR">
        <PanelEmpty reason="Loading…" />
      </Panel>
    );
  }

  return (
    <Panel
      title="Divergence audit"
      subtitle={`${names.length} treatment${names.length === 1 ? "" : "s"} · CATE vs GWR`}
    >
      <div style={{ overflowX: "auto" }}>
        <table className="mono" style={{ width: "100%", fontSize: 11, borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ textAlign: "left", borderBottom: "1px solid var(--line)" }}>
              <th style={{ padding: "6px 8px" }}>Treatment</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>Cells</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>Flagged</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>Mean |Δ|</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>Pearson r</th>
              <th style={{ padding: "6px 8px", textAlign: "right" }}>Sign agree</th>
            </tr>
          </thead>
          <tbody>
            {names.map((name) => {
              const r = reports[name];
              const flagFrac = r.flagged_fraction;
              const tone =
                flagFrac > 0.25
                  ? "var(--crimson, #b91c1c)"
                  : flagFrac > 0.1
                    ? "var(--amber, #b45309)"
                    : "var(--green, #2f7d32)";
              return (
                <tr key={name} style={{ borderBottom: "1px solid rgba(0,0,0,0.04)" }}>
                  <td style={{ padding: "6px 8px" }}>{name}</td>
                  <td style={cell}>{r.cell_count}</td>
                  <td style={{ padding: "6px 8px", textAlign: "right" }}>
                    <Pill color={tone}>
                      {r.flagged_count} ({(flagFrac * 100).toFixed(0)}%)
                    </Pill>
                  </td>
                  <td style={cell}>{r.mean_abs_divergence.toFixed(3)}</td>
                  <td style={cell}>{r.pearson_correlation.toFixed(3)}</td>
                  <td style={cell}>{(r.sign_agreement_rate * 100).toFixed(0)}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

const cell = {
  padding: "6px 8px",
  textAlign: "right" as const,
  fontVariantNumeric: "tabular-nums" as const,
  color: "var(--ink-2)",
};
