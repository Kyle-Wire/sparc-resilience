/**
 * ModelPerformancePanel — R² leaderboard for ensemble models.
 * Researcher mode only; the Practitioner mode collapses this into the
 * single "Confidence" badge in the header band.
 */
import { useEffect, useState } from "react";
import { Panel, PanelEmpty } from "@/components/ui/DesignSystem";
import { useManifest } from "@/hooks/useManifest";
import { getModelPerformance } from "@/lib/api";
import { SPARC_RAMP_HEX } from "@/lib/design-tokens";

interface Row {
  name: string;
  r2: number;
  color: string;
}

export default function ModelPerformancePanel() {
  const manifest = useManifest();
  const [rows, setRows] = useState<Row[]>([]);
  const present = !!manifest.lookup("2", "ensemble_results");

  useEffect(() => {
    if (!present) return;
    getModelPerformance()
      .then((d) => {
        if (!d.models?.length) return;
        const sorted = [...d.models].sort((a, b) => b.r2 - a.r2);
        setRows(
          sorted.map((m, i) => ({
            name: m.name,
            r2: m.r2,
            color: SPARC_RAMP_HEX[i * 2] ?? SPARC_RAMP_HEX[0],
          })),
        );
      })
      .catch(() => {});
  }, [present, manifest.lastUpdated]);

  if (!present) {
    return (
      <Panel title="Model performance" subtitle="stage 2 · ensemble">
        <PanelEmpty
          reason="Awaiting stage 2"
          hint="Ensemble model results aren't in the manifest yet. Run the Models stage."
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
