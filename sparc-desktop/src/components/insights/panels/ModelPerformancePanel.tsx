/**
 * ModelPerformancePanel — R² leaderboard for ensemble models.
 * Researcher mode only; the Practitioner mode collapses this into the
 * single "Confidence" badge in the header band.
 */
import { useMemo } from "react";
import { Panel, PanelEmpty, Btn } from "@/components/ui/DesignSystem";
import { useManifest } from "@/hooks/useManifest";
import { useResult } from "@/hooks/useResult";
import { useInsightsNavigate } from "@/hooks/InsightsProvider";
import { getModelPerformance } from "@/lib/api";
import { SPARC_RAMP_HEX } from "@/lib/design-tokens";

interface Row {
  name: string;
  r2: number;
  color: string;
}

export default function ModelPerformancePanel() {
  const manifest = useManifest();
  const navigate = useInsightsNavigate();
  const present = !!manifest.lookup("2", "ensemble_results");
  const { data: perfData } = useResult<{ models: { name: string; r2: number }[] }>(
    present ? "s2:model_performance" : null,
    getModelPerformance,
  );

  const rows = useMemo<Row[]>(() => {
    if (!perfData?.models?.length) return [];
    const sorted = [...perfData.models].sort((a, b) => b.r2 - a.r2);
    return sorted.map((m, i) => ({
      name: m.name,
      r2: m.r2,
      color: SPARC_RAMP_HEX[i * 2] ?? SPARC_RAMP_HEX[0],
    }));
  }, [perfData]);

  if (!present) {
    return (
      <Panel title="Model performance" subtitle="stage 2 · ensemble">
        <PanelEmpty
          reason="Awaiting stage 2"
          hint="Ensemble model results aren't in the manifest yet. Run the Models stage."
          action={navigate && <Btn small onClick={() => navigate("Run")}>Go to Run page →</Btn>}
        />
      </Panel>
    );
  }

  if (rows.length === 0) {
    return (
      <Panel title="Model performance" subtitle="stage 2 · ensemble">
        <PanelEmpty reason="Loading…" />
      </Panel>
    );
  }

  const max = Math.max(...rows.map((r) => Math.max(r.r2, 0)), 0.001);
  return (
    <Panel title="Model performance" subtitle="held-out R² · stage 2">
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {rows.map((r) => {
          const pct = Math.max(0, Math.min(100, (r.r2 / max) * 100));
          return (
            <div
              key={r.name}
              style={{ display: "grid", gridTemplateColumns: "120px 1fr 56px", gap: 10, alignItems: "center" }}
            >
              <span className="mono" style={{ fontSize: 11, color: "var(--ink-2)" }}>
                {r.name}
              </span>
              <div
                style={{
                  height: 14,
                  background: "rgba(0,0,0,0.04)",
                  borderRadius: 3,
                  position: "relative",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${pct}%`,
                    height: "100%",
                    background: r.color,
                    transition: "width 0.4s ease",
                  }}
                />
              </div>
              <span
                className="mono"
                style={{ fontSize: 11, fontVariantNumeric: "tabular-nums", textAlign: "right" }}
              >
                {r.r2.toFixed(3)}
              </span>
            </div>
          );
        })}
      </div>
    </Panel>
  );
}
