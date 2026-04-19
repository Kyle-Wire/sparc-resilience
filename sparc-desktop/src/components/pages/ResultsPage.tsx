import { useState, useEffect, useCallback, useRef } from "react";
import { SectionHeader, Card, Btn, Stat, StatGrid } from "@/components/ui/DesignSystem";
// API integration: import { getConfig } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import { SPARC_RAMP_HEX } from "@/lib/design-tokens";

type ViewMode = "map" | "histogram";

export default function ResultsPage() {
  const [viewMode, setViewMode] = useState<ViewMode>("map");
  const [activeScenario, setActiveScenario] = useState("baseline");
  const mapCanvasRef = useRef<HTMLCanvasElement>(null);
  const histCanvasRef = useRef<HTMLCanvasElement>(null);
  const r2CanvasRef = useRef<HTMLCanvasElement>(null);
  const curveCanvasRef = useRef<HTMLCanvasElement>(null);
  const { notify } = useNotification();

  const scenarios = [
    { id: "baseline", name: "Baseline", delta: 0 },
    { id: "s1", name: "Green canopy +20%", delta: -1.2 },
    { id: "s2", name: "Cool roofs", delta: -0.8 },
    { id: "s3", name: "Combined", delta: -2.1 },
  ];

  const models = [
    { name: "OLS", r2: 0.72, color: SPARC_RAMP_HEX[0] },
    { name: "GWR", r2: 0.86, color: SPARC_RAMP_HEX[2] },
    { name: "GWRF", r2: 0.91, color: SPARC_RAMP_HEX[4] },
    { name: "GGPGAM", r2: 0.89, color: SPARC_RAMP_HEX[6] },
    { name: "Ensemble", r2: 0.93, color: SPARC_RAMP_HEX[8] },
  ];

  // Draw scenario map (canvas heatmap)
  useEffect(() => {
    const canvas = mapCanvasRef.current;
    if (!canvas || viewMode !== "map") return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    // Background
    ctx.fillStyle = "#f5f1eb";
    ctx.fillRect(0, 0, w, h);

    // Draw heatmap grid
    const gridX = 40, gridY = 30;
    const cellW = w / gridX, cellH = h / gridY;
    const ramp = SPARC_RAMP_HEX;

    for (let y = 0; y < gridY; y++) {
      for (let x = 0; x < gridX; x++) {
        // Generate smooth noise-like pattern
        const dx = (x - gridX / 2) / gridX;
        const dy = (y - gridY / 2) / gridY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        let heat = 0.5 + 0.3 * Math.sin(x * 0.5) * Math.cos(y * 0.4) - dist * 0.3;
        heat += Math.sin(x * 0.8 + y * 0.6) * 0.15;
        heat = Math.max(0, Math.min(1, heat));

        // Adjust for scenario
        const sc = scenarios.find((s) => s.id === activeScenario);
        if (sc && sc.delta !== 0) {
          heat = Math.max(0, Math.min(1, heat + sc.delta * 0.1));
        }

        const colorIdx = Math.floor(heat * (ramp.length - 1));
        ctx.fillStyle = ramp[colorIdx];
        ctx.fillRect(x * cellW, y * cellH, cellW + 0.5, cellH + 0.5);
      }
    }
  }, [viewMode, activeScenario]);

  // Draw histogram
  useEffect(() => {
    const canvas = histCanvasRef.current;
    if (!canvas || viewMode !== "histogram") return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    const nBins = 40;
    const bins: number[] = [];
    const sc = scenarios.find((s) => s.id === activeScenario);
    const mean = sc?.delta ?? 0;

    for (let i = 0; i < nBins; i++) {
      const x = mean - 4 + (i / nBins) * 8;
      bins.push(Math.exp(-0.5 * ((x - mean) / 1.2) ** 2) * 100 + Math.random() * 10);
    }
    const maxBin = Math.max(...bins);

    // Axes
    ctx.strokeStyle = "var(--line)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(40, 10); ctx.lineTo(40, h - 30); ctx.lineTo(w - 10, h - 30);
    ctx.stroke();

    // Bars
    const barW = (w - 55) / nBins;
    bins.forEach((v, i) => {
      const barH = (v / maxBin) * (h - 50);
      const x = 42 + i * barW;
      const colorIdx = Math.floor((i / nBins) * (SPARC_RAMP_HEX.length - 1));
      ctx.fillStyle = SPARC_RAMP_HEX[colorIdx] + "cc";
      ctx.fillRect(x, h - 30 - barH, barW - 1, barH);
    });

    // Mean line
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 3]);
    const meanX = 42 + ((0 - (mean - 4)) / 8) * (w - 55);
    ctx.beginPath(); ctx.moveTo(meanX, 10); ctx.lineTo(meanX, h - 30); ctx.stroke();
    ctx.setLineDash([]);

    // Labels
    ctx.fillStyle = "var(--muted)";
    ctx.font = "10px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText("Temperature anomaly (°C)", w / 2, h - 8);
  }, [viewMode, activeScenario]);

  // Model R² bars
  useEffect(() => {
    const canvas = r2CanvasRef.current;
    if (!canvas) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    const barH = 16;
    const gap = 6;
    const labelW = 60;

    models.forEach((m, i) => {
      const y = i * (barH + gap) + 4;
      // Label
      ctx.fillStyle = "var(--muted)";
      ctx.font = "10px 'JetBrains Mono'";
      ctx.textAlign = "right";
      ctx.fillText(m.name, labelW - 6, y + barH / 2 + 4);

      // Bar bg
      ctx.fillStyle = "rgba(0,0,0,0.04)";
      ctx.fillRect(labelW, y, w - labelW - 40, barH);

      // Bar
      ctx.fillStyle = m.color;
      ctx.fillRect(labelW, y, (w - labelW - 40) * m.r2, barH);

      // Value
      ctx.fillStyle = "var(--ink)";
      ctx.font = "bold 10px 'JetBrains Mono'";
      ctx.textAlign = "left";
      ctx.fillText(m.r2.toFixed(3), w - 36, y + barH / 2 + 4);
    });
  }, []);

  // Intervention response curve
  useEffect(() => {
    const canvas = curveCanvasRef.current;
    if (!canvas) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    // Axes
    ctx.strokeStyle = "var(--line)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(40, 10); ctx.lineTo(40, h - 25); ctx.lineTo(w - 10, h - 25);
    ctx.stroke();

    // Draw curves for top 3 interventions
    const interventions = [
      { name: "tree_canopy_pct", color: SPARC_RAMP_HEX[0], slope: -0.4 },
      { name: "impervious_pct", color: SPARC_RAMP_HEX[4], slope: 0.35 },
      { name: "albedo", color: SPARC_RAMP_HEX[6], slope: -0.25 },
    ];

    interventions.forEach((intv) => {
      ctx.strokeStyle = intv.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i <= 50; i++) {
        const t = i / 50;
        const x = 42 + t * (w - 54);
        const y = (h - 35) / 2 + 10 + intv.slope * t * (h - 45) * 0.5 + Math.sin(t * 4) * 8;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    });

    // Legend
    ctx.font = "9px 'JetBrains Mono'";
    interventions.forEach((intv, i) => {
      const x = 50 + i * 85;
      ctx.fillStyle = intv.color;
      ctx.fillRect(x, h - 12, 12, 3);
      ctx.fillStyle = "var(--muted)";
      ctx.textAlign = "left";
      ctx.fillText(intv.name.replace(/_/g, " ").slice(0, 10), x + 16, h - 8);
    });
  }, []);

  const handleExport = useCallback(async (format: string) => {
    notify("success", `${format} export started`);
  }, [notify]);

  return (
    <div>
      <SectionHeader
        kicker="11 · pipeline"
        label="Results"
        right={
          <div style={{ display: "flex", gap: 8 }}>
            <Btn small onClick={() => handleExport("CSV")}>Export CSV</Btn>
            <Btn small onClick={() => handleExport("GeoPackage")}>Export GPKG</Btn>
          </div>
        }
      />

      <StatGrid>
        <Stat label="Best R²" value="0.93" tint="var(--crimson)" sub="Ensemble" />
        <Stat label="RMSE" value="0.44" tint="var(--purple)" />
        <Stat label="Scenarios" value={String(scenarios.length)} tint="var(--amber)" />
        <Stat label="Best Δ" value="-2.1 °C" tint="var(--ink)" sub="Combined" />
      </StatGrid>

      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 14 }}>
        {/* Main visualization */}
        <Card
          title="Scenario map"
          subtitle={
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              {/* Scenario selector */}
              {scenarios.map((sc) => (
                <button
                  key={sc.id}
                  onClick={() => setActiveScenario(sc.id)}
                  style={{
                    padding: "3px 8px",
                    fontSize: 10,
                    borderRadius: 3,
                    border: "1px solid " + (activeScenario === sc.id ? "var(--ink)" : "var(--line)"),
                    background: activeScenario === sc.id ? "var(--ink)" : "#fff",
                    color: activeScenario === sc.id ? "#fff" : "var(--ink-2)",
                    cursor: "pointer",
                    fontFamily: "inherit",
                    fontWeight: 600,
                  }}
                >
                  {sc.name}
                </button>
              ))}
              <span style={{ width: 1, height: 16, background: "var(--line)" }} />
              {/* View mode toggle */}
              {(["map", "histogram"] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setViewMode(mode)}
                  style={{
                    padding: "3px 8px",
                    fontSize: 10,
                    borderRadius: 3,
                    border: "1px solid " + (viewMode === mode ? "var(--crimson)" : "var(--line)"),
                    background: viewMode === mode ? "var(--crimson)" : "#fff",
                    color: viewMode === mode ? "#fff" : "var(--ink-2)",
                    cursor: "pointer",
                    fontFamily: "inherit",
                    fontWeight: 600,
                    textTransform: "capitalize",
                  }}
                >
                  {mode}
                </button>
              ))}
            </div>
          }
        >
          {viewMode === "map" ? (
            <div style={{ position: "relative" }}>
              <canvas
                ref={mapCanvasRef}
                style={{ width: "100%", height: 380, display: "block", borderRadius: 4 }}
              />
              {/* Color ramp legend */}
              <div
                style={{
                  position: "absolute",
                  bottom: 12,
                  right: 12,
                  background: "rgba(255,255,255,0.92)",
                  borderRadius: 6,
                  padding: "8px 10px",
                  backdropFilter: "blur(6px)",
                }}
              >
                <div className="mono" style={{ fontSize: 8, color: "var(--muted)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.1em" }}>
                  temperature anomaly
                </div>
                <div style={{ display: "flex", height: 10, borderRadius: 2, overflow: "hidden", width: 120 }}>
                  {SPARC_RAMP_HEX.map((c, i) => (
                    <div key={i} style={{ flex: 1, background: c }} />
                  ))}
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginTop: 2 }}>
                  <span className="mono" style={{ fontSize: 8, color: "var(--muted)" }}>cool</span>
                  <span className="mono" style={{ fontSize: 8, color: "var(--muted)" }}>hot</span>
                </div>
              </div>
            </div>
          ) : (
            <canvas
              ref={histCanvasRef}
              style={{ width: "100%", height: 380, display: "block" }}
            />
          )}
        </Card>

        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {/* Model R² bars */}
          <Card title="Model R²" subtitle="cross-validated performance">
            <canvas
              ref={r2CanvasRef}
              style={{ width: "100%", height: 120, display: "block" }}
            />
          </Card>

          {/* Intervention response curves */}
          <Card title="Intervention response" subtitle="marginal effect on target">
            <canvas
              ref={curveCanvasRef}
              style={{ width: "100%", height: 160, display: "block" }}
            />
          </Card>
        </div>
      </div>
    </div>
  );
}
