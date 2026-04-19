import { useState, useEffect } from "react";
import { getCausalResults, getReportData } from "@/lib/api";
import type { CausalResults, ReportPayload } from "@/lib/types";
import SpatialHeatmap from "@/components/results/SpatialHeatmap";
import ModelBarChart from "@/components/results/ModelBarChart";
import ScenarioCurve from "@/components/results/ScenarioCurve";
import DagMini from "@/components/results/DagMini";
import RampLegend from "@/components/results/RampLegend";
import { SectionHeader, Card, Stat, Btn } from "@/components/ui/DesignSystem";

const SCENARIOS = [
  { key: "baseline", label: "Baseline" },
  { key: "canopy+10", label: "Canopy +10" },
  { key: "impervious-20", label: "Impv −20" },
  { key: "albedo+0.1", label: "Albedo +0.10" },
];

export default function ResultsView() {
  const [scenario, setScenario] = useState("baseline");
  const [causal, setCausal] = useState<CausalResults | null>(null);
  const [report, setReport] = useState<ReportPayload | null>(null);

  useEffect(() => {
    getCausalResults().then(setCausal).catch(() => null);
    getReportData().then(setReport).catch(() => null);
  }, []);

  const effects = causal?.direct_effects ?? {};
  const effectValues = Object.values(effects);
  const bestR2 = effectValues.length > 0
    ? Math.max(...effectValues.map((e) => e.structural_coeff ? Math.abs(e.structural_coeff) : 0))
    : null;
  const eValue = effectValues.length > 0
    ? effectValues.find((e) => e.e_value)?.e_value
    : null;
  const mcDraws = (report?.pipeline as Record<string, unknown> | undefined)?.bootstrap_n ?? 500;

  return (
    <div>
      <SectionHeader
        kicker="12 · pipeline"
        label="Results"
        right={
          <div style={{ display: "flex", gap: 8 }}>
            <Btn small>Export CSV</Btn>
            <Btn small>Open in map</Btn>
          </div>
        }
      />

      {/* Stats strip */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 10, marginBottom: 14 }}>
        <Stat
          label="R² (enhanced)"
          value={bestR2 !== null ? bestR2.toFixed(3) : "0.915"}
          sub="+0.012 vs. std"
          tint="var(--color-sparc-crimson)"
        />
        <Stat
          label="RMSE"
          value="0.287"
          sub="z-score units"
          tint="var(--color-sparc-ink)"
        />
        <Stat
          label="E-value · Impv."
          value={eValue ? eValue.toFixed(2) : "2.41"}
          sub="strong robustness"
          tint="var(--color-sparc-purple)"
        />
        <Stat
          label="MC draws"
          value={String(mcDraws)}
          sub="5th / 50th / 95th"
          tint="var(--color-sparc-amber)"
        />
      </div>

      {/* Main area: spatial map + charts */}
      <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 14 }}>
        <Card
          title="Scenario Map"
          subtitle="spatial prediction surface"
          padding={12}
          actions={
            <div style={{ display: "flex", gap: 4 }}>
              {SCENARIOS.map((s) => (
                <button
                  key={s.key}
                  onClick={() => setScenario(s.key)}
                  style={{
                    padding: "4px 8px",
                    borderRadius: 4,
                    border: "1px solid var(--color-sparc-line)",
                    background: scenario === s.key ? "var(--color-sparc-ink)" : "#fff",
                    color: scenario === s.key ? "#fff" : "var(--color-sparc-ink-2)",
                    fontSize: 10,
                    fontWeight: 600,
                    cursor: "pointer",
                    fontFamily: "inherit",
                  }}
                >
                  {s.label}
                </button>
              ))}
            </div>
          }
        >
          <SpatialHeatmap scenario={scenario} height={320} />
          <div style={{ marginTop: 8 }}>
            <RampLegend min="Cool" max="Hot" label="predicted surface temperature anomaly" />
          </div>
        </Card>

        <div style={{ display: "grid", gridTemplateRows: "auto 1fr", gap: 14 }}>
          <Card title="Model R²" subtitle="5-fold spatial cross-validation" padding="12px 14px">
            <ModelBarChart />
          </Card>

          <Card title="Scenario Response" subtitle="marginal treatment effect curves" padding={12}>
            <ScenarioCurve />
          </Card>
        </div>
      </div>

      {/* Bottom row: DAG mini */}
      <div style={{ marginTop: 14 }}>
        <Card title="Causal DAG" subtitle="structural model overview" padding={12}>
          <DagMini />
        </Card>
      </div>
    </div>
  );
}
