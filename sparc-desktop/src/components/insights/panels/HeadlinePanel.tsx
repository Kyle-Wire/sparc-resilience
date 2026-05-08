/**
 * HeadlinePanel — practitioner top-line "best intervention right now".
 *
 * Reads the Phase 4 server aggregator at `/insights/headline`, which
 * combines the best decision candidate, sensitivity strength, and DAG
 * structure counts in a single round-trip. Falls back to client-side
 * scoring of `/decision/candidates` if the aggregator is unavailable
 * (e.g. older backends).
 */
import { useMemo } from "react";
import { Panel, PanelEmpty, Pill, Stat, StatGrid } from "@/components/ui/DesignSystem";
import { useResult } from "@/hooks/useResult";
import {
  getDecisionCandidates,
  getInsightsHeadline,
  type InsightsHeadline,
  type InterventionCandidate,
} from "@/lib/api";

export default function HeadlinePanel() {
  const { data: headline, error: headlineError } = useResult<InsightsHeadline>(
    "meta:insights_headline",
    getInsightsHeadline,
  );
  // Fetch decision candidates as fallback; always cached under the same key
  // as EquityCostPanel so the second consumer is free.
  const { data: candidatesData } = useResult<{ candidates: InterventionCandidate[] }>(
    headlineError ? "meta:decision_candidates" : null,
    getDecisionCandidates,
  );
  const fallbackCandidates = candidatesData?.candidates ?? null;
  const error = headlineError && !fallbackCandidates ? headlineError : null;

  // If the aggregator returned a `best`, prefer that. Otherwise fall
  // back to the historical client-side scoring path.
  const fallbackBest = useMemo<InterventionCandidate | null>(() => {
    if (!fallbackCandidates?.length) return null;
    const scored = fallbackCandidates
      .filter((c) => Number.isFinite(c.mean_effect))
      .map((c) => {
        const denom = Math.abs(c.cost) > 1e-9 ? Math.abs(c.cost) : 1;
        return { c, score: c.mean_effect / denom };
      });
    if (!scored.length) return null;
    scored.sort((a, b) => b.score - a.score);
    return scored[0].c;
  }, [fallbackCandidates]);

  const best = (headline?.best ?? fallbackBest) as InterventionCandidate | null;
  const altCount = headline?.alternatives ?? Math.max(0, (fallbackCandidates?.length ?? 0) - 1);
  const ratio = headline?.score ?? (best && best.cost !== 0 ? best.mean_effect / best.cost : null);
  const sensitivity = headline?.sensitivity ?? null;

  if (error || (!best && !headline)) {
    return (
      <Panel title="Best intervention right now" subtitle="aggregator">
        <PanelEmpty
          reason={error ?? "No intervention candidates"}
          hint="Configure interventions on the Decision Support page; this card surfaces the top-ranked one."
        />
      </Panel>
    );
  }
  if (!best) {
    return (
      <Panel title="Best intervention right now" subtitle="aggregator">
        <PanelEmpty reason="Candidates have no usable effect estimates yet." />
      </Panel>
    );
  }

  const subtitleParts: string[] = [];
  if (best.notes) subtitleParts.push(best.notes);
  else subtitleParts.push(`treatment · ${best.treatment}`);
  if (headline?.n_treatments && headline.n_outcomes) {
    subtitleParts.push(`${headline.n_treatments} tx → ${headline.n_outcomes} outcome`);
  }

  return (
    <Panel
      title="Best intervention right now"
      subtitle={subtitleParts.join(" · ")}
      actions={<Pill color="var(--purple)">{best.name}</Pill>}
    >
      <StatGrid>
        <Stat label="treatment" value={best.treatment} />
        <Stat label="magnitude" value={best.magnitude.toFixed(2)} />
        <Stat
          label="mean effect"
          value={`${best.mean_effect >= 0 ? "+" : ""}${best.mean_effect.toFixed(3)}`}
        />
        <Stat label="effect σ" value={best.effect_std.toFixed(3)} />
        <Stat label="cost" value={best.cost.toFixed(2)} />
        <Stat label="equity" value={best.equity_weight.toFixed(2)} />
        {ratio != null && <Stat label="effect / cost" value={ratio.toFixed(3)} />}
        <Stat label="alternatives" value={String(altCount)} />
        {sensitivity?.e_value != null && (
          <Stat
            label="top E-value"
            value={`${sensitivity.e_value.toFixed(2)}${sensitivity.robust ? " ✓" : ""}`}
          />
        )}
      </StatGrid>
      <p style={{ marginTop: 12, fontSize: 12, color: "var(--ink-2)" }}>
        Picked by maximising mean_effect ÷ |cost|.
        {sensitivity?.treatment && sensitivity.e_value != null && (
          <>
            {" "}Strongest sensitivity check: <strong>{sensitivity.treatment}</strong>{" "}
            (E-value {sensitivity.e_value.toFixed(2)} —{" "}
            {sensitivity.robust ? "robust" : "an unmeasured confounder of modest strength could explain it away"}).
          </>
        )}
        {" "}Run the full optimizer on the Decision Support page to apply a budget cap or robustness penalty.
      </p>
    </Panel>
  );
}
