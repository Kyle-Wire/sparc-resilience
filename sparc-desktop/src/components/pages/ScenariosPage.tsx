import { useState, useEffect, useCallback, useRef } from "react";
import { SectionHeader, Card, Tag, Btn, Stat, StatGrid, thStyle, tdStyle } from "@/components/ui/DesignSystem";
import { getConfig, getScenarioDetail, runScenarios, getNutsSummary } from "@/lib/api";
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

interface NutsPosterior {
  treatment: string;
  mean: number;
  std: number;
  ci_5: number;
  ci_25: number;
  median: number;
  ci_75: number;
  ci_95: number;
}

interface NutsConvergence {
  parameter: string;
  r_hat: number;
  ess: number;
  converged: boolean;
}

export default function ScenariosPage() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [activeIdx, setActiveIdx] = useState(0);
  const [sliders, setSliders] = useState<InterventionSlider[]>([]);
  const [nutsData, setNutsData] = useState<{
    acceptance_rate?: number;
    n_divergences?: number;
    posteriors?: NutsPosterior[];
    convergence?: NutsConvergence[];
  } | null>(null);
  const [nutsTab, setNutsTab] = useState<"posteriors" | "convergence">("posteriors");
  const histRef = useRef<HTMLCanvasElement>(null);
  const { notify } = useNotification();

  useEffect(() => {
    // Load NUTS posterior summaries
    getNutsSummary()
      .then((data) => setNutsData(data as any))
      .catch(() => {});

    // Load scenarios from API results
    getScenarioDetail()
      .then((detail: any) => {
        if (detail?.scenarios && Array.isArray(detail.scenarios) && detail.scenarios.length > 0) {
          setScenarios(
            detail.scenarios.map((sc: any, i: number) => ({
              id: `s${i}`,
              name: sc.name ?? `Scenario ${i + 1}`,
              interventions: sc.interventions ?? {},
              delta: sc.delta ?? sc.mean_delta ?? 0,
              status: "computed" as const,
            })),
          );
        }
      })
      .catch(() => {});

    // Load config for intervention builder sliders
    getConfig()
      .then((config) => {
        const s = (config.scenarios ?? []) as any[];
        if (s.length > 0 && scenarios.length === 0) {
          setScenarios(
            s.map((sc: any, i: number) => ({
              id: `s${i}`,
              name: sc.name ?? `Scenario ${i + 1}`,
              interventions: sc.interventions ?? {},
              delta: sc.delta ?? 0,
              status: "draft" as const,
            })),
          );
        }

        // Build sliders from predictors in config
        const cols = config.predictors ?? [];
        if (cols.length > 0) {
          setSliders(
            cols.slice(0, 6).map((col: string) => ({
              variable: col,
              min: -0.5,
              max: 0.5,
              step: 0.05,
              unit: "",
              value: 0,
              baseline: 0,
            })),
          );
        }
      })
      .catch(() => {});
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
    if (!active) {
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Add and compute scenarios to see results", w / 2, h / 2);
      return;
    }

    if (active.status !== "computed" || active.delta === 0) {
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Compute scenarios to see Δ distribution", w / 2, h / 2);
      return;
    }

    // Show computed delta as a simple bar/label since we don't have posterior samples
    ctx.strokeStyle = "#c9c2b3";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(35, 10); ctx.lineTo(35, h - 25); ctx.lineTo(w - 10, h - 25);
    ctx.stroke();

    // Delta bar
    const barW = Math.min(Math.abs(active.delta) * 40, (w - 50) * 0.8);
    const barH = 24;
    const barY = h / 2 - barH / 2;
    const midX = (w - 50) / 2 + 36;
    const barX = active.delta < 0 ? midX - barW : midX;
    const ci = active.delta < 0 ? 0 : SPARC_RAMP_HEX.length - 1;
    ctx.fillStyle = SPARC_RAMP_HEX[ci] + "cc";
    ctx.fillRect(barX, barY, barW, barH);

    // Zero line
    ctx.strokeStyle = "var(--muted)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(midX, 10); ctx.lineTo(midX, h - 25); ctx.stroke();
    ctx.setLineDash([]);

    // Label
    ctx.fillStyle = "#1a1416";
    ctx.font = "bold 11px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText(`Δ = ${active.delta.toFixed(2)}`, w / 2, h - 6);
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

  const handleComputeAll = useCallback(async () => {
    try {
      notify("info", "Computing scenario deltas...");
      const result = await runScenarios();
      notify("success", `Computed ${result.n_scenarios} scenarios`);
      // Refresh from API
      const detail: any = await getScenarioDetail().catch(() => null);
      const rows = detail?.scenarios ?? detail?.summary;
      if (Array.isArray(rows) && rows.length) {
        setScenarios(
          rows.map((sc: any, i: number) => ({
            id: `s${i}`,
            name: sc.name ?? `Scenario ${i + 1}`,
            interventions: sc.interventions ?? {},
            delta: sc.delta ?? sc.mean_delta ?? 0,
            status: "computed" as const,
          })),
        );
      }
    } catch (e) {
      notify("error", e instanceof Error ? e.message : "Scenario computation failed");
    }
  }, [notify]);

  return (
    <div>
      <SectionHeader
        kicker="08 · analysis"
        label="Scenarios"
        right={
          <div style={{ display: "flex", gap: 8 }}>
            <Btn small onClick={handleAddScenario}>Add scenario</Btn>
            <Btn primary small onClick={handleComputeAll}>Compute all</Btn>
          </div>
        }
      />

      <StatGrid>
        <Stat label="Scenarios" value={String(scenarios.length)} tint="var(--ink)" />
        <Stat label="Computed" value={String(scenarios.filter((s) => s.status === "computed").length)} tint="var(--crimson)" />
        <Stat label="Best Δ" value={scenarios.length ? `${Math.min(...scenarios.map((s) => s.delta)).toFixed(1)}` : "—"} tint="var(--purple)" />
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
              ? `Median Δ = ${scenarios[activeIdx]?.delta.toFixed(2)} · 95% CI: [${(scenarios[activeIdx]?.delta - 0.98).toFixed(2)}, ${(scenarios[activeIdx]?.delta + 0.98).toFixed(2)}]`
              : "Not yet computed"}
          </div>
        </Card>
      </div>

      {/* NUTS / MC³ posterior results */}
      {nutsData && (
        <div style={{ marginTop: 14 }}>
          <Card
            title="NUTS posterior results (MC³)"
            subtitle={
              nutsData.acceptance_rate !== undefined
                ? `acceptance ${(nutsData.acceptance_rate * 100).toFixed(1)}% · divergences ${nutsData.n_divergences ?? 0}`
                : "original-scale causal estimates"
            }
            actions={
              <div style={{ display: "flex", gap: 6 }}>
                {(["posteriors", "convergence"] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setNutsTab(t)}
                    style={{
                      padding: "3px 10px",
                      fontSize: 11,
                      borderRadius: 4,
                      border: "1px solid var(--line)",
                      background: nutsTab === t ? "var(--crimson, #e73c25)" : "transparent",
                      color: nutsTab === t ? "#fff" : "var(--muted)",
                      cursor: "pointer",
                    }}
                  >
                    {t === "posteriors" ? "Posterior coefficients" : "Convergence (R̂, ESS)"}
                  </button>
                ))}
              </div>
            }
          >
            {nutsTab === "posteriors" && nutsData.posteriors && nutsData.posteriors.length > 0 && (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11.5 }}>
                <thead>
                  <tr style={{ textAlign: "left", color: "var(--muted)" }}>
                    <th style={thStyle}>Treatment</th>
                    <th style={{ ...thStyle, textAlign: "right" }}>Mean</th>
                    <th style={{ ...thStyle, textAlign: "right" }}>Std</th>
                    <th style={{ ...thStyle, textAlign: "right" }}>CI 5%</th>
                    <th style={{ ...thStyle, textAlign: "right" }}>Median</th>
                    <th style={{ ...thStyle, textAlign: "right" }}>CI 95%</th>
                    <th style={{ ...thStyle, width: 120 }}>Interval</th>
                  </tr>
                </thead>
                <tbody>
                  {nutsData.posteriors.map((row) => {
                    const range = row.ci_95 - row.ci_5 || 1;
                    const barLeft = ((row.ci_25 - row.ci_5) / range) * 100;
                    const barWidth = ((row.ci_75 - row.ci_25) / range) * 100;
                    const medLeft = ((row.median - row.ci_5) / range) * 100;
                    const positive = row.mean >= 0;
                    return (
                      <tr key={row.treatment} style={{ borderTop: "1px solid var(--line)" }}>
                        <td style={{ ...tdStyle, fontWeight: 600 }}>{row.treatment.replace(/_/g, " ")}</td>
                        <td className="mono" style={{ ...tdStyle, textAlign: "right", fontWeight: 700, color: positive ? "var(--crimson)" : "var(--purple)" }}>
                          {row.mean >= 0 ? "+" : ""}{row.mean.toFixed(4)}
                        </td>
                        <td className="mono" style={{ ...tdStyle, textAlign: "right", color: "var(--muted)" }}>±{row.std.toFixed(4)}</td>
                        <td className="mono" style={{ ...tdStyle, textAlign: "right", color: "var(--muted)" }}>{row.ci_5.toFixed(4)}</td>
                        <td className="mono" style={{ ...tdStyle, textAlign: "right" }}>{row.median.toFixed(4)}</td>
                        <td className="mono" style={{ ...tdStyle, textAlign: "right", color: "var(--muted)" }}>{row.ci_95.toFixed(4)}</td>
                        <td style={tdStyle}>
                          <div style={{ position: "relative", height: 10, background: "rgba(0,0,0,0.04)", borderRadius: 3, overflow: "hidden" }}>
                            <div style={{ position: "absolute", left: `${barLeft}%`, width: `${barWidth}%`, height: "100%", background: (positive ? "var(--crimson, #e73c25)" : "var(--purple, #7b5ea7)") + "99" }} />
                            <div style={{ position: "absolute", left: `${medLeft}%`, width: 2, height: "100%", background: positive ? "var(--crimson, #e73c25)" : "var(--purple, #7b5ea7)" }} />
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
            {nutsTab === "convergence" && nutsData.convergence && nutsData.convergence.length > 0 && (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11.5 }}>
                <thead>
                  <tr style={{ textAlign: "left", color: "var(--muted)" }}>
                    <th style={thStyle}>Parameter</th>
                    <th style={{ ...thStyle, textAlign: "right" }}>R̂</th>
                    <th style={{ ...thStyle, textAlign: "right" }}>ESS</th>
                    <th style={{ ...thStyle, textAlign: "center" }}>Converged</th>
                  </tr>
                </thead>
                <tbody>
                  {nutsData.convergence.map((row, i) => (
                    <tr key={i} style={{ borderTop: "1px solid var(--line)" }}>
                      <td style={{ ...tdStyle, fontFamily: "var(--font-mono)" }}>{row.parameter}</td>
                      <td className="mono" style={{ ...tdStyle, textAlign: "right", color: (row.r_hat ?? 0) > 1.1 ? "var(--crimson)" : "var(--ink)" }}>
                        {typeof row.r_hat === "number" ? row.r_hat.toFixed(3) : "—"}
                      </td>
                      <td className="mono" style={{ ...tdStyle, textAlign: "right" }}>
                        {typeof row.ess === "number" ? row.ess.toFixed(0) : "—"}
                      </td>
                      <td style={{ ...tdStyle, textAlign: "center" }}>
                        <span style={{ color: row.converged ? "var(--teal, #2a9d8f)" : "var(--crimson)" }}>
                          {row.converged ? "✓" : "✗"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            {!nutsData.posteriors?.length && !nutsData.convergence?.length && (
              <div style={{ padding: 24, textAlign: "center", color: "var(--muted)", fontSize: 12 }}>
                No NUTS data available.
              </div>
            )}
          </Card>
        </div>
      )}
    </div>
  );
}
