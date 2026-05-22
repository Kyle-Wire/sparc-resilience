import { useState, useCallback } from "react";
import { Card, Btn } from "@/components/ui/DesignSystem";
import { useNotification } from "@/hooks/useNotifications";
import { downloadAudienceReport, type ReportAudience, type ReportFormat } from "@/lib/api";

const AUDIENCE_OPTIONS: { id: ReportAudience; label: string; blurb: string; includes: string; color: string }[] = [
  {
    id: "planner",
    label: "Planner / Decision-maker",
    blurb: "Ranked interventions, cost-effectiveness, equity summary, and spatial maps — designed for city staff and policy teams.",
    includes: "Executive summary · Ranked interventions · Budget allocation · Equity profile · Key maps",
    color: "var(--crimson)",
  },
  {
    id: "public",
    label: "Public / Residents",
    blurb: "One-page plain-language summary of findings and recommended actions — for residents and elected officials.",
    includes: "Executive summary · Plain-language findings · Key recommendations",
    color: "var(--amber)",
  },
  {
    id: "technical",
    label: "Technical / Analyst",
    blurb: "Full statistics, model diagnostics, uncertainty quantification, and reproducibility details — for analysts and peer reviewers.",
    includes: "All sections · Full diagnostics · Uncertainty · DAG · Appendices",
    color: "var(--purple)",
  },
  {
    id: "council",
    label: "Council / Elected Officials",
    blurb: "Decision summary, ranked actions with cost per unit, budget allocation, equity analysis, and targeted rollout guidance.",
    includes: "Decision summary · Ranked actions · Budget allocation · Equity · Optimal rollout",
    color: "#2e7d32",
  },
  {
    id: "scientist",
    label: "Scientist / External Reviewer",
    blurb: "Method stack, posterior diagnostics (R-hat, ESS), Wager-2025 audit roll-up, refutations table, and full provenance.",
    includes: "Method stack · Posterior diagnostics · Wager-2025 audit · Refutations · Provenance",
    color: "#1565c0",
  },
  {
    id: "equity",
    label: "Equity Advocate",
    blurb: "Heterogeneous treatment effects, who benefits most, spillover decomposition, and targeted-policy fairness analysis.",
    includes: "Effect heterogeneity · Who benefits · Spillover decomposition · Fairness analysis",
    color: "#6a1b9a",
  },
  {
    id: "auditor",
    label: "Auditor / 3rd-Party",
    blurb: "Full Wager-2025 gap completion checklist, data provenance trail, and reproducibility verification.",
    includes: "Wager-2025 gap roll-up · Full provenance · Reproducibility verification",
    color: "#37474f",
  },
];

export default function ReportPage() {
  const [generating, setGenerating] = useState(false);
  const [audience, setAudience] = useState<ReportAudience>("planner");
  const { notify } = useNotification();

  const handleAudienceExport = useCallback(async (fmt: ReportFormat) => {
    setGenerating(true);
    notify("info", `Generating ${audience} report (${fmt.toUpperCase()})…`);
    try {
      const r = await downloadAudienceReport({ audience, fmt });
      if (!r.ok) throw new Error(r.error || "Failed");
      notify("success", `Saved ${r.filename}`);
    } catch (e) {
      notify("error", e instanceof Error ? e.message : "Report generation failed");
    } finally {
      setGenerating(false);
    }
  }, [audience, notify]);

  const selectedAudience = AUDIENCE_OPTIONS.find((o) => o.id === audience)!;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

      {/* ── Hero: Audience selector ── */}
      <Card
        title="Who is this for?"
        subtitle="Select an audience — the report content and language will be tailored automatically."
        actions={
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <Btn small onClick={() => handleAudienceExport("docx")} disabled={generating}>Word</Btn>
            <Btn primary onClick={() => handleAudienceExport("pdf")} disabled={generating}>
              {generating ? "Generating…" : "Export PDF"}
            </Btn>
          </div>
        }
      >
        {/* Audience cards */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 14 }}>
          {AUDIENCE_OPTIONS.map((opt) => {
            const active = audience === opt.id;
            return (
              <button
                key={opt.id}
                onClick={() => setAudience(opt.id)}
                style={{
                  textAlign: "left",
                  border: `2px solid ${active ? opt.color : "var(--line)"}`,
                  background: active ? "#fff" : "#faf8f4",
                  borderRadius: 8,
                  padding: "14px 14px",
                  cursor: "pointer",
                  fontFamily: "inherit",
                  transition: "border-color 0.15s",
                }}
              >
                <div style={{ fontSize: 13, fontWeight: 700, color: opt.color, marginBottom: 6 }}>
                  {opt.label}
                </div>
                <div style={{ fontSize: 11.5, color: "var(--muted)", lineHeight: 1.5 }}>
                  {opt.blurb}
                </div>
              </button>
            );
          })}
        </div>

        {/* What's included strip */}
        <div style={{
          padding: "8px 12px",
          borderRadius: 6,
          background: "#f5f0ea",
          fontSize: 11.5,
          color: "var(--ink-2)",
          display: "flex",
          gap: 8,
          alignItems: "baseline",
        }}>
          <span style={{ fontWeight: 600, color: selectedAudience.color, whiteSpace: "nowrap" }}>Includes:</span>
          <span>{selectedAudience.includes}</span>
        </div>

      </Card>

    </div>
  );
}
