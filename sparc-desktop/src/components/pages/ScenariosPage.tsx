import { useState, useEffect, useCallback, useRef } from "react";
import { SectionHeader, Card, Tag, Btn, Stat, StatGrid, thStyle, tdStyle } from "@/components/ui/DesignSystem";
import { getConfig, saveConfig } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import { SPARC_RAMP_HEX } from "@/lib/design-tokens";

interface Scenario {
  id: string;
  name: string;
  interventions: Record<string, number>;
  delta: number;
  status: "draft" | "computed" | "baseline";
}

interface InterventionSlider {
  variable: string;
  min: number;
  max: number;
  step: number;
  unit: string;
  value: number;
  baseline: number;
}

export default function ScenariosPage() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [activeIdx, setActiveIdx] = useState(0);
  const [sliders, setSliders] = useState<InterventionSlider[]>([]);
  const histRef = useRef<HTMLCanvasElement>(null);
  const { notify } = useNotification();

  useEffect(() => {
    getConfig()
      .then((config) => {
        const s = config.scenarios ?? [];
        if (s.length > 0) {
          setScenarios(
            s.map((sc: any, i: number) => ({
              id: `s${i}`,
              name: sc.name ?? `Scenario ${i + 1}`,
              interventions: sc.interventions ?? {},
              delta: sc.delta ?? 0,
              status: i === 0 ? "baseline" : "draft",
            })),
          );
        } else {
          setScenarios([
            { id: "baseline", name: "Baseline", interventions: {}, delta: 0, status: "baseline" },
            { id: "s1", name: "Green canopy +20%", interventions: { tree_canopy_pct: 0.2 }, delta: -1.2, status: "computed" },
            { id: "s2", name: "Cool roofs", interventions: { albedo: 0.15 }, delta: -0.8, status: "computed" },
            { id: "s3", name: "Combined strategy", interventions: { tree_canopy_pct: 0.15, albedo: 0.1, impervious_pct: -0.1 }, delta: -2.1, status: "computed" },
          ]);
        }

        setSliders([
          { variable: "tree_canopy_pct", min: -0.3, max: 0.5, step: 0.05, unit: "%", value: 0, baseline: 0.32 },
          { variable: "impervious_pct", min: -0.3, max: 0.3, step: 0.05, unit: "%", value: 0, baseline: 0.45 },
          { variable: "albedo", min: -0.1, max: 0.3, step: 0.02, unit: "", value: 0, baseline: 0.25 },
          { variable: "building_height", min: -5, max: 10, step: 1, unit: "m", value: 0, baseline: 12 },
        ]);
      })
      .catch(() => {
        setScenarios([
          { id: "baseline", name: "Baseline", interventions: {}, delta: 0, status: "baseline" },
          { id: "s1", name: "Green canopy +20%", interventions: { tree_canopy_pct: 0.2 }, delta: -1.2, status: "computed" },
          { id: "s2", name: "Cool roofs", interventions: { albedo: 0.15 }, delta: -0.8, status: "computed" },
          { id: "s3", name: "Combined strategy", interventions: { tree_canopy_pct: 0.15, albedo: 0.1, impervious_pct: -0.1 }, delta: -2.1, status: "computed" },
        ]);
        setSliders([
          { variable: "tree_canopy_pct", min: -0.3, max: 0.5, step: 0.05, unit: "%", value: 0, baseline: 0.32 },
          { variable: "impervious_pct", min: -0.3, max: 0.3, step: 0.05, unit: "%", value: 0, baseline: 0.45 },
          { variable: "albedo", min: -0.1, max: 0.3, step: 0.02, unit: "", value: 0, baseline: 0.25 },
          { variable: "building_height", min: -5, max: 10, step: 1, unit: "m", value: 0, baseline: 12 },
        ]);
      });
  }, []);

  // Posterior histogram
  useEffect(() => {
    const canvas = histRef.current;
    if (!canvas) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    const active = scenarios[activeIdx];
    if (!active) return;

    // Draw histogram of simulated posterior
    const nBins = 30;
    const bins: number[] = [];
    const mean = active.delta;
    const std = 0.5;
    for (let i = 0; i < nBins; i++) {
      const x = mean - 3 * std + (i / nBins) * 6 * std;
      bins.push(Math.exp(-0.5 * ((x - mean) / std) ** 2));
    }
    const maxBin = Math.max(...bins);

    // Axes
    ctx.strokeStyle = "var(--line)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(35, 10); ctx.lineTo(35, h - 25); ctx.lineTo(w - 10, h - 25);
    ctx.stroke();

    // Bars
    const barW = (w - 50) / nBins;
    bins.forEach((v, i) => {
      const barH = (v / maxBin) * (h - 45);
      const x = 36 + i * barW;
      const colorIdx = Math.floor((i / nBins) * (SPARC_RAMP_HEX.length - 1));
      ctx.fillStyle = SPARC_RAMP_HEX[colorIdx] + "cc";
      ctx.fillRect(x, h - 25 - barH, barW - 1, barH);
    });

    // Mean line
    const meanX = 36 + ((0 - (mean - 3 * std)) / (6 * std)) * (w - 50);
    ctx.strokeStyle = "var(--crimson)";
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(meanX, 10); ctx.lineTo(meanX, h - 25); ctx.stroke();
    ctx.setLineDash([]);

    // Labels
    ctx.fillStyle = "var(--muted)";
    ctx.font = "9px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText(`Δ target = ${mean.toFixed(1)}`, w / 2, h - 6);
  }, [scenarios, activeIdx]);

  const handleSliderChange = useCallback((variable: string, value: number) => {
    setSliders((prev) => prev.map((s) => s.variable === variable ? { ...s, value } : s));
  }, []);

  const handleAddScenario = useCallback(() => {
    const interventions: Record<string, number> = {};
    sliders.forEach((s) => {
      if (s.value !== 0) interventions[s.variable] = s.value;
    });
    const name = prompt("Scenario name:", `Scenario ${scenarios.length + 1}`);
    if (!name) return;
    setScenarios((prev) => [
      ...prev,
      { id: `s${prev.length}`, name, interventions, delta: 0, status: "draft" },
    ]);
    notify("success", `Scenario "${name}" added`);
  }, [sliders, scenarios, notify]);

  return (
    <div>
      <SectionHeader
        kicker="08 · analysis"
        label="Scenarios"
        right={
          <div style={{ display: "flex", gap: 8 }}>
            <Btn small onClick={handleAddScenario}>Add scenario</Btn>
            <Btn primary small onClick={() => notify("info", "Computing scenario deltas...")}>Compute all</Btn>
          </div>
        }
      />

      <StatGrid>
        <Stat label="Scenarios" value={String(scenarios.length)} tint="var(--ink)" />
        <Stat label="Computed" value={String(scenarios.filter((s) => s.status === "computed").length)} tint="var(--crimson)" />
        <Stat label="Best Δ" value={`${Math.min(...scenarios.map((s) => s.delta)).toFixed(1)} °C`} tint="var(--purple)" />
        <Stat label="Variables" value={String(sliders.length)} tint="var(--amber)" />
      </StatGrid>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <Card title="Scenario library" subtitle="click to select · compare to baseline">
            {scenarios.map((sc, i) => (
              <div
                key={sc.id}
                onClick={() => setActiveIdx(i)}
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr auto auto",
                  alignItems: "center",
                  gap: 10,
                  padding: "10px 8px",
                  borderTop: i > 0 ? "1px dashed var(--line)" : "none",
                  cursor: "pointer",
                  background: activeIdx === i ? "rgba(231,60,37,0.04)" : "transparent",
                  borderRadius: 4,
                }}
              >
                <div>
                  <div style={{ fontSize: 12.5, fontWeight: 600 }}>{sc.name}</div>
                  <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>
                    {Object.entries(sc.interventions)
                      .map(([k, v]) => `${k}: ${v > 0 ? "+" : ""}${v}`)
                      .join(", ") || "no interventions"}
                  </div>
                </div>
                <span
                  className="mono"
                  style={{
                    fontSize: 13,
                    fontWeight: 700,
                    color: sc.delta < 0 ? "var(--purple)" : sc.delta > 0 ? "var(--crimson)" : "var(--muted)",
                  }}
                >
                  {sc.delta === 0 ? "—" : `${sc.delta > 0 ? "+" : ""}${sc.delta.toFixed(1)} °C`}
                </span>
                <Tag
                  color={
                    sc.status === "computed"
                      ? "var(--ink)"
                      : sc.status === "baseline"
                      ? "var(--purple)"
                      : "var(--muted)"
                  }
                >
                  {sc.status}
                </Tag>
              </div>
            ))}
          </Card>

          <Card title="Intervention builder" subtitle="adjust sliders to create new scenario">
            {sliders.map((s) => (
              <div key={s.variable} style={{ padding: "10px 0", borderTop: "1px dashed var(--line)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <span style={{ fontSize: 12, fontWeight: 600 }}>{s.variable.replace(/_/g, " ")}</span>
                  <span
                    className="mono"
                    style={{
                      fontSize: 12,
                      fontWeight: 700,
                      color: s.value !== 0 ? "var(--crimson)" : "var(--muted)",
                    }}
                  >
                    {s.value > 0 ? "+" : ""}
                    {s.value.toFixed(2)} {s.unit}
                  </span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span className="mono" style={{ fontSize: 9, color: "var(--muted)", width: 36, textAlign: "right" }}>
                    {s.min}
                  </span>
                  <input
                    type="range"
                    min={s.min}
                    max={s.max}
                    step={s.step}
                    value={s.value}
                    onChange={(e) => handleSliderChange(s.variable, Number(e.target.value))}
                    style={{ flex: 1, accentColor: "var(--crimson)" }}
                  />
                  <span className="mono" style={{ fontSize: 9, color: "var(--muted)", width: 36 }}>
                    {s.max}
                  </span>
                </div>
                <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", marginTop: 2 }}>
                  baseline: {s.baseline}{s.unit}
                </div>
              </div>
            ))}
          </Card>
        </div>

        <Card title="Posterior distribution" subtitle={scenarios[activeIdx]?.name ?? "select a scenario"}>
          <canvas
            ref={histRef}
            style={{ width: "100%", height: 300, display: "block" }}
          />
          <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 8, textAlign: "center" }}>
            {scenarios[activeIdx]?.status === "computed"
              ? `Median Δ = ${scenarios[activeIdx]?.delta.toFixed(2)} °C · 95% CI: [${(scenarios[activeIdx]?.delta - 0.98).toFixed(2)}, ${(scenarios[activeIdx]?.delta + 0.98).toFixed(2)}]`
              : "Not yet computed"}
          </div>
        </Card>
      </div>
    </div>
  );
}
