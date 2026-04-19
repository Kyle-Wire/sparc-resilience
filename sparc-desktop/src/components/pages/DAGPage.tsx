import { useState, useEffect } from "react";
import { Btn, Card, LegendDot, SectionHeader } from "@/components/ui/DesignSystem";
import DagMini from "@/components/results/DagMini";
import { getDag, getCausalResults } from "@/lib/api";
import type { DagDefinition, CausalResults } from "@/lib/types";

const DEMO_COEFS: [string, number, string][] = [
  ["Canopy → AAT_z", -0.022, "var(--crimson)"],
  ["Impervious → AAT_z", +0.022, "var(--crimson)"],
  ["NDVI → AAT_z", -4.131, "var(--purple)"],
  ["Albedo → AAT_z", -2.759, "var(--amber)"],
  ["Canopy → NDVI", +0.003, "var(--muted)"],
  ["Canopy → Impervious", -0.630, "var(--muted)"],
];

function buildCoefs(causal: CausalResults | null): [string, number, string][] {
  if (!causal?.direct_effects) return DEMO_COEFS;
  const entries = Object.entries(causal.direct_effects);
  if (entries.length === 0) return DEMO_COEFS;
  return entries.map(([varName, effect]) => {
    const coeff = effect.structural_coeff ?? 0;
    const color = Math.abs(coeff) > 1 ? "var(--purple)" : "var(--crimson)";
    return [`${varName} → target`, coeff, color];
  });
}

export default function DAGPage() {
  const [dag, setDag] = useState<DagDefinition | null>(null);
  const [causal, setCausal] = useState<CausalResults | null>(null);

  useEffect(() => {
    getDag().then(setDag).catch(() => {});
    getCausalResults().then(setCausal).catch(() => {});
  }, []);

  const coefs = buildCoefs(causal);

  return (
    <div>
      <SectionHeader
        kicker="04 · analysis"
        label="Causal DAG"
        right={<Btn small>Add edge</Btn>}
      />
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 14 }}>
        <Card
          title="Directed acyclic graph"
          subtitle="3 treatments · 1 mediator · 2 confounders · 1 outcome"
        >
          <DagMini dag={dag} />
          <div style={{ display: "flex", gap: 14, marginTop: 10 }}>
            <LegendDot color="var(--crimson)" label="Treatment" />
            <LegendDot color="var(--purple)" label="Mediator" />
            <LegendDot color="var(--muted)" label="Confounder" />
            <LegendDot color="var(--ink)" label="Outcome" />
          </div>
        </Card>

        <Card title="Structural coefficients" subtitle="DML · 5-fold">
          {coefs.map(([l, v, c]) => (
            <div
              key={l}
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 64px",
                gap: 8,
                padding: "6px 0",
                borderTop: "1px dashed var(--line)",
              }}
            >
              <span style={{ fontSize: 12 }}>{l}</span>
              <span
                className="mono"
                style={{
                  fontSize: 11.5,
                  fontWeight: 700,
                  textAlign: "right",
                  color: v < 0 ? c : "var(--crimson)",
                }}
              >
                {v > 0 ? "+" : ""}
                {v.toFixed(3)}
              </span>
            </div>
          ))}
        </Card>
      </div>
    </div>
  );
}
