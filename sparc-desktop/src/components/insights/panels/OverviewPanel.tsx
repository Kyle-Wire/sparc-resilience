/**
 * OverviewPanel — top-of-page summary card.
 *
 * Practitioner: top recommendation hero + pipeline completion status.
 * Researcher: high-level run health (best R², #models, manifest staleness).
 */
import { useMemo } from "react";
import { Panel, PanelEmpty, Stat, StatGrid, Btn } from "@/components/ui/DesignSystem";
import { useAudience } from "@/hooks/InsightsProvider";
import { useManifest } from "@/hooks/useManifest";
import { useResult } from "@/hooks/useResult";
import { getModelPerformance, getInsightsHeadline, type InsightsHeadline } from "@/lib/api";
import { useInsightsNavigate } from "@/hooks/InsightsProvider";

export default function OverviewPanel() {
  const [audience] = useAudience();
  const manifest = useManifest();
  const navigate = useInsightsNavigate();
  const stages = manifest.manifest?.stages ?? {};
  const stageCount = Object.keys(stages).length;

  const perfPresent = !!manifest.lookup("2", "ensemble_results");
  const { data: perfData } = useResult(
    perfPresent ? "s2:model_performance" : null,
    getModelPerformance,
  );
  const { data: headline } = useResult<InsightsHeadline>(
    audience === "practitioner" ? "meta:insights_headline" : null,
    getInsightsHeadline,
  );

  const bestR2 = useMemo(() => {
    if (!perfData?.models?.length) return null;
    return perfData.models.reduce((a, b) => (b.r2 > a.r2 ? b : a)).r2;
  }, [perfData]);
  const modelCount = perfData?.models?.length ?? 0;

  if (stageCount === 0) {
    return (
      <Panel title="Overview" subtitle="run summary">
        <PanelEmpty
          reason="No pipeline run yet"
          hint="Run any stage on the Run page to populate the Insights workspace."
          action={navigate && <Btn small onClick={() => navigate("Run")}>Go to Run page →</Btn>}
        />
      </Panel>
    );
  }

  // ---- Practitioner view ----
  if (audience === "practitioner") {
    const best = headline?.best ?? null;
    const hasRec = best != null;
    return (
      <Panel title="At a glance" subtitle="decision summary">
        {hasRec ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            {/* Hero recommendation */}
            <div
              style={{
                padding: "16px 20px",
                borderRadius: 8,
                background: "linear-gradient(135deg, #f7f3ee 0%, #fdfbf7 100%)",
                border: "1px solid var(--line)",
              }}
            >
              <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--muted)", marginBottom: 6 }}>
                Top recommendation
              </div>
              <div style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em" }}>
                {best!.treatment}
              </div>
              {(best!.mean_effect != null) && (
                <div style={{ fontSize: 13, color: "var(--ink-2)", marginTop: 4 }}>
                  Mean effect: <strong>{best!.mean_effect > 0 ? "+" : ""}{best!.mean_effect.toFixed(3)}</strong>
                  {headline?.sensitivity ? <span style={{ marginLeft: 10, color: "var(--muted)" }}>sensitivity {headline.sensitivity.robust ? "robust" : "not robust"}{headline.sensitivity.e_value != null ? ` (E=${headline.sensitivity.e_value.toFixed(2)})` : ""}</span> : null}
                </div>
              )}
            </div>
            <StatGrid>
              <Stat label="Stages complete" value={`${stageCount}/5`} />
              <Stat label="Alternatives" value={headline?.alternatives ?? "—"} />
              <Stat
                label="Last update"
                value={manifest.lastUpdated ? manifest.lastUpdated.toLocaleTimeString() : "—"}
                sub={manifest.loading ? "refreshing…" : undefined}
              />
            </StatGrid>
          </div>
        ) : (
          <StatGrid>
            <Stat label="Stages complete" value={`${stageCount}/5`} />
            <Stat
              label="Last update"
              value={manifest.lastUpdated ? manifest.lastUpdated.toLocaleTimeString() : "—"}
              sub={manifest.loading ? "refreshing…" : undefined}
            />
          </StatGrid>
        )}
      </Panel>
    );
  }

  // ---- Researcher view ----
  const maupScore = (headline as any)?.maup_robustness_score as number | undefined | null;

  return (
    <Panel title="Run health" subtitle="model summary">
      <StatGrid>
        <Stat
          label="Best R²"
          value={bestR2 != null ? bestR2.toFixed(3) : "—"}
          tint="var(--crimson)"
        />
        <Stat label="Models" value={modelCount} />
        <Stat label="Stages run" value={stageCount} />
        <Stat
          label="MAUP robustness"
          value={maupScore != null ? maupScore.toFixed(2) : "—"}
          tint={maupScore == null ? undefined : maupScore >= 0.8 ? "var(--green, #2f7d32)" : "var(--amber, #e79024)"}
          sub={maupScore == null ? "not yet computed" : maupScore >= 0.8 ? "scale-stable" : "caution: scale-sensitive"}
        />
        <Stat
          label="Last update"
          value={manifest.lastUpdated ? manifest.lastUpdated.toLocaleTimeString() : "—"}
          sub={manifest.loading ? "refreshing…" : undefined}
        />
      </StatGrid>
    </Panel>
  );
}
