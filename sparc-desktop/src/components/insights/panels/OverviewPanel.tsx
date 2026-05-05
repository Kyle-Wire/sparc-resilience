/**
 * OverviewPanel — top-of-page summary card.
 *
 * Practitioner: stub for the "Best intervention right now" headline.
 * Researcher: high-level run health (best R², #models, manifest staleness).
 *
 * This is intentionally lightweight in Phase 1; the Practitioner headline
 * card is fully fleshed out in Phase 2 once `/insights/headline` ships.
 */
import { useEffect, useState } from "react";
import { Panel, PanelEmpty, Stat, StatGrid } from "@/components/ui/DesignSystem";
import { useAudience } from "@/hooks/InsightsProvider";
import { useManifest } from "@/hooks/useManifest";
import { getModelPerformance } from "@/lib/api";

export default function OverviewPanel() {
  const [audience] = useAudience();
  const manifest = useManifest();
  const [bestR2, setBestR2] = useState<number | null>(null);
  const [modelCount, setModelCount] = useState<number>(0);

  useEffect(() => {
    if (!manifest.lookup("2", "ensemble_results")) return;
    getModelPerformance()
      .then((d) => {
        if (!d.models?.length) return;
        const best = d.models.reduce((a, b) => (b.r2 > a.r2 ? b : a));
        setBestR2(best.r2);
        setModelCount(d.models.length);
      })
      .catch(() => {});
  }, [manifest]);

  const stages = manifest.manifest?.stages ?? {};
  const stageCount = Object.keys(stages).length;

  if (stageCount === 0) {
    return (
      <Panel title="Overview" subtitle="run summary">
        <PanelEmpty
          reason="No pipeline run yet"
          hint="Run any stage on the Run page to populate the Insights workspace."
        />
      </Panel>
    );
  }

  return (
    <Panel
      title={audience === "practitioner" ? "At a glance" : "Run health"}
      subtitle={`audience · ${audience}`}
    >
      <StatGrid>
        <Stat
          label="Best R²"
          value={bestR2 != null ? bestR2.toFixed(3) : "—"}
          tint="var(--crimson)"
        />
        <Stat label="Models" value={modelCount} />
        <Stat label="Stages run" value={stageCount} />
        <Stat
          label="Last update"
          value={manifest.lastUpdated ? manifest.lastUpdated.toLocaleTimeString() : "—"}
          sub={manifest.loading ? "refreshing…" : undefined}
        />
      </StatGrid>
    </Panel>
  );
}
