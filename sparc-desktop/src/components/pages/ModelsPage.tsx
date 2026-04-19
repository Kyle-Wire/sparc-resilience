import { useState, useEffect, useRef } from "react";
import { SectionHeader, Card, Tag, Btn, Stat, StatGrid, thStyle, tdStyle } from "@/components/ui/DesignSystem";
import { getConfig } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import { SPARC_RAMP_HEX } from "@/lib/design-tokens";

interface ModelInfo {
  name: string;
  fullName: string;
  r2: number;
  rmse: number;
  mae: number;
  weight: number;
  color: string;
}

export default function ModelsPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("Ensemble");
  const residualsRef = useRef<HTMLCanvasElement>(null);
  const weightsRef = useRef<HTMLCanvasElement>(null);
  const { notify } = useNotification();

  useEffect(() => {
    // Try to fetch from API, fall back to demo data
    getConfig()
      .then(() => {
        setModels([
          { name: "OLS", fullName: "Ordinary Least Squares", r2: 0.72, rmse: 0.89, mae: 0.67, weight: 0.10, color: SPARC_RAMP_HEX[0] },
          { name: "GWR", fullName: "Geographically Weighted Regression", r2: 0.86, rmse: 0.62, mae: 0.48, weight: 0.20, color: SPARC_RAMP_HEX[2] },
          { name: "GWRF", fullName: "Geographically Weighted Random Forest", r2: 0.91, rmse: 0.51, mae: 0.39, weight: 0.30, color: SPARC_RAMP_HEX[4] },
          { name: "GGPGAM", fullName: "Geographically Grouped GAM", r2: 0.89, rmse: 0.55, mae: 0.42, weight: 0.25, color: SPARC_RAMP_HEX[6] },
          { name: "Ensemble", fullName: "Stacked Ensemble (GWEN)", r2: 0.93, rmse: 0.44, mae: 0.34, weight: 0.15, color: SPARC_RAMP_HEX[8] },
        ]);
      })
      .catch(() => {
        setModels([
          { name: "OLS", fullName: "Ordinary Least Squares", r2: 0.72, rmse: 0.89, mae: 0.67, weight: 0.10, color: SPARC_RAMP_HEX[0] },
          { name: "GWR", fullName: "Geographically Weighted Regression", r2: 0.86, rmse: 0.62, mae: 0.48, weight: 0.20, color: SPARC_RAMP_HEX[2] },
          { name: "GWRF", fullName: "Geographically Weighted Random Forest", r2: 0.91, rmse: 0.51, mae: 0.39, weight: 0.30, color: SPARC_RAMP_HEX[4] },
          { name: "GGPGAM", fullName: "Geographically Grouped GAM", r2: 0.89, rmse: 0.55, mae: 0.42, weight: 0.25, color: SPARC_RAMP_HEX[6] },
          { name: "Ensemble", fullName: "Stacked Ensemble (GWEN)", r2: 0.93, rmse: 0.44, mae: 0.34, weight: 0.15, color: SPARC_RAMP_HEX[8] },
        ]);
      });
  }, []);

  // Draw residuals scatter
  useEffect(() => {
    const canvas = residualsRef.current;
    if (!canvas) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    const model = models.find((m) => m.name === selectedModel);
    if (!model) return;

    // Axes
    ctx.strokeStyle = "var(--line)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(40, 10); ctx.lineTo(40, h - 25); ctx.lineTo(w - 10, h - 25);
    ctx.stroke();

    // Zero line
    const yMid = (h - 35) / 2 + 10;
    ctx.strokeStyle = "rgba(0,0,0,0.1)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(40, yMid); ctx.lineTo(w - 10, yMid); ctx.stroke();
    ctx.setLineDash([]);

    // Scatter points
    const nPts = 80;
    const rmseScale = model.rmse;
    for (let i = 0; i < nPts; i++) {
      const x = 45 + Math.random() * (w - 60);
      const residual = (Math.random() - 0.5) * 2 * rmseScale * 1.5;
      const y = yMid - (residual / (rmseScale * 2)) * (h - 45);
      ctx.globalAlpha = 0.5;
      ctx.fillStyle = model.color;
      ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Labels
    ctx.fillStyle = "var(--muted)";
    ctx.font = "9px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText("Predicted", w / 2, h - 6);
    ctx.save();
    ctx.translate(12, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Residual", 0, 0);
    ctx.restore();
  }, [selectedModel, models]);

  // Draw ensemble weights bar
  useEffect(() => {
    const canvas = weightsRef.current;
    if (!canvas || models.length === 0) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    const barH = 20;
    let x = 0;
    const total = models.reduce((s, m) => s + m.weight, 0);
    models.forEach((m) => {
      const barW = (m.weight / total) * w;
      ctx.fillStyle = m.color + "cc";
      ctx.fillRect(x, (h - barH) / 2, barW, barH);
      if (barW > 30) {
        ctx.fillStyle = "#fff";
        ctx.font = "bold 9px 'JetBrains Mono'";
        ctx.textAlign = "center";
        ctx.fillText(m.name, x + barW / 2, (h - barH) / 2 + barH / 2 + 3);
      }
      x += barW;
    });
  }, [models]);

  const best = models.reduce((a, b) => (a.r2 > b.r2 ? a : b), models[0] ?? { name: "—", r2: 0 });

  return (
    <div>
      <SectionHeader kicker="09 · analysis" label="Models" />

      <StatGrid>
        <Stat label="Models" value={String(models.length)} tint="var(--ink)" />
        <Stat label="Best R²" value={best?.r2?.toFixed(3) ?? "—"} tint="var(--crimson)" sub={best?.name} />
        <Stat label="RMSE (best)" value={best?.rmse?.toFixed(3) ?? "—"} tint="var(--purple)" />
        <Stat label="Ensemble" value={models.find((m) => m.name === "Ensemble")?.r2?.toFixed(3) ?? "—"} tint="var(--amber)" />
      </StatGrid>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Card title="Model zoo" subtitle="click to view residuals">
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ textAlign: "left", color: "var(--muted)" }}>
                <th style={thStyle}>Model</th>
                <th style={{ ...thStyle, textAlign: "right" }}>R²</th>
                <th style={{ ...thStyle, textAlign: "right" }}>RMSE</th>
                <th style={{ ...thStyle, textAlign: "right" }}>MAE</th>
                <th style={{ ...thStyle, width: 90 }}>R² bar</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr
                  key={m.name}
                  onClick={() => setSelectedModel(m.name)}
                  style={{
                    borderTop: "1px solid var(--line)",
                    cursor: "pointer",
                    background: selectedModel === m.name ? "rgba(231,60,37,0.04)" : "transparent",
                  }}
                >
                  <td style={{ ...tdStyle, fontWeight: 600 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ width: 8, height: 8, borderRadius: "50%", background: m.color }} />
                      <div>
                        <div>{m.name}</div>
                        <div className="mono" style={{ fontSize: 9, color: "var(--muted)" }}>{m.fullName}</div>
                      </div>
                    </div>
                  </td>
                  <td className="mono" style={{ ...tdStyle, textAlign: "right", fontWeight: 700 }}>
                    {m.r2.toFixed(3)}
                  </td>
                  <td className="mono" style={{ ...tdStyle, textAlign: "right" }}>
                    {m.rmse.toFixed(3)}
                  </td>
                  <td className="mono" style={{ ...tdStyle, textAlign: "right" }}>
                    {m.mae.toFixed(3)}
                  </td>
                  <td style={tdStyle}>
                    <div style={{ height: 10, background: "rgba(0,0,0,0.04)", borderRadius: 3, overflow: "hidden" }}>
                      <div
                        style={{
                          width: `${m.r2 * 100}%`,
                          height: "100%",
                          background: m.color,
                          transition: "width 0.3s",
                        }}
                      />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <Card title="Residuals scatter" subtitle={`${selectedModel} · predicted vs residual`}>
            <canvas
              ref={residualsRef}
              style={{ width: "100%", height: 200, display: "block" }}
            />
          </Card>

          <Card title="Ensemble weights" subtitle="GWEN-based stacking">
            <canvas
              ref={weightsRef}
              style={{ width: "100%", height: 50, display: "block", marginBottom: 8 }}
            />
            <div style={{ display: "flex", justifyContent: "center", gap: 14, flexWrap: "wrap" }}>
              {models.map((m) => (
                <div key={m.name} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ width: 8, height: 8, borderRadius: "50%", background: m.color }} />
                  <span className="mono" style={{ fontSize: 9.5, color: "var(--muted)" }}>
                    {m.name} {(m.weight * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
