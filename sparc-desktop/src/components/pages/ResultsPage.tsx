import { useState, useEffect, useCallback, useRef } from "react";
import { SectionHeader, Card, Btn, Stat, StatGrid } from "@/components/ui/DesignSystem";
import { getResults, getScenarioDetail, getPdpCurves, getGwenData, getCorrelogramData } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import { SPARC_RAMP_HEX } from "@/lib/design-tokens";
import type { ScenarioDetail, PdpCurves, CorrelogramData } from "@/lib/types";

type ViewMode = "map" | "histogram" | "correlogram";

interface ModelInfo {
  name: string;
  r2: number;
  color: string;
}

export default function ResultsPage() {
  const [viewMode, setViewMode] = useState<ViewMode>("map");
  const [activeScenarioIdx] = useState(0);
  const mapCanvasRef = useRef<HTMLCanvasElement>(null);
  const histCanvasRef = useRef<HTMLCanvasElement>(null);
  const r2CanvasRef = useRef<HTMLCanvasElement>(null);
  const curveCanvasRef = useRef<HTMLCanvasElement>(null);
  const corrCanvasRef = useRef<HTMLCanvasElement>(null);
  const { notify } = useNotification();

  const [models, setModels] = useState<ModelInfo[]>([]);
  const [scenarioDetail, setScenarioDetail] = useState<ScenarioDetail | null>(null);
  const [pdpData, setPdpData] = useState<PdpCurves | null>(null);
  const [correlogram, setCorrelogram] = useState<CorrelogramData | null>(null);

  // Load data from API
  useEffect(() => {
    // Models from stages 1-5
    Promise.all(
      [1, 2, 3, 4, 5].map((s) => getResults(s).catch(() => null))
    ).then((results) => {
      const mods: ModelInfo[] = [];
      results.forEach((r: any, i) => {
        if (!r) return;
        const name = r.model_name ?? r.name ?? `Stage ${i + 1}`;
        const r2 = r.r2 ?? r.test_r2 ?? r.cv_r2 ?? null;
        if (r2 != null) {
          mods.push({ name, r2, color: SPARC_RAMP_HEX[i * 2] ?? SPARC_RAMP_HEX[0] });
        }
      });
      setModels(mods);
    });

    // Scenario detail
    getScenarioDetail().then(setScenarioDetail).catch(() => {});

    // PDP curves for intervention response
    getPdpCurves().then(setPdpData).catch(() => {});

    // Correlogram
    getCorrelogramData().then(setCorrelogram).catch(() => {});

    // GWEN weights (update model R²)
    getGwenData().then((gwen) => {
      if (!gwen?.rows?.length) return;
      setModels((prev) => {
        if (!prev.length) return prev;
        return prev.map((m) => {
          const row = gwen.rows.find((r: any) => r.model === m.name || r.name === m.name);
          if (row && typeof (row as any).r2 === "number") return { ...m, r2: (row as any).r2 };
          return m;
        });
      });
    }).catch(() => {});
  }, []);

  const hasData = models.length > 0;
  const bestModel = hasData ? models.reduce((a, b) => (b.r2 > a.r2 ? b : a)) : null;
  const scenarioRows = scenarioDetail?.summary ?? [];

  // Draw scenario map from real GeoJSON
  useEffect(() => {
    const canvas = mapCanvasRef.current;
    if (!canvas || viewMode !== "map") return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    const features = scenarioDetail?.geojson?.features;
    if (!features?.length) {
      ctx.fillStyle = "#f5f1eb";
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Run pipeline to see scenario map", w / 2, h / 2);
      return;
    }

    ctx.fillStyle = "#f5f1eb";
    ctx.fillRect(0, 0, w, h);

    // Extract coords & target values
    const pts: { x: number; y: number; val: number }[] = [];
    for (const f of features) {
      const coords = f.geometry?.coordinates as number[] | undefined;
      if (!coords || coords.length < 2) continue;
      const val = Number(f.properties?.predicted ?? f.properties?.target ?? 0);
      pts.push({ x: coords[0], y: coords[1], val });
    }
    if (!pts.length) return;

    const minX = Math.min(...pts.map((p) => p.x));
    const maxX = Math.max(...pts.map((p) => p.x));
    const minY = Math.min(...pts.map((p) => p.y));
    const maxY = Math.max(...pts.map((p) => p.y));
    const minV = Math.min(...pts.map((p) => p.val));
    const maxV = Math.max(...pts.map((p) => p.val));
    const rangeV = maxV - minV || 1;
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const pad = 12;

    for (const p of pts) {
      const px = pad + ((p.x - minX) / rangeX) * (w - 2 * pad);
      const py = pad + ((maxY - p.y) / rangeY) * (h - 2 * pad);
      const t = (p.val - minV) / rangeV;
      const ci = Math.floor(t * (SPARC_RAMP_HEX.length - 1));
      ctx.fillStyle = SPARC_RAMP_HEX[ci];
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [viewMode, scenarioDetail, activeScenarioIdx]);

  // Draw histogram from real prediction values
  useEffect(() => {
    const canvas = histCanvasRef.current;
    if (!canvas || viewMode !== "histogram") return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    const features = scenarioDetail?.geojson?.features;
    if (!features?.length) {
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Run pipeline to see histogram", w / 2, h / 2);
      return;
    }

    const vals = features
      .map((f) => Number(f.properties?.predicted ?? f.properties?.target))
      .filter((v) => Number.isFinite(v));
    if (!vals.length) return;

    const nBins = 40;
    const minV = Math.min(...vals);
    const maxV = Math.max(...vals);
    const range = maxV - minV || 1;
    const bins = new Array(nBins).fill(0);
    for (const v of vals) {
      const idx = Math.min(Math.floor(((v - minV) / range) * nBins), nBins - 1);
      bins[idx]++;
    }
    const maxBin = Math.max(...bins);

    // Axes
    ctx.strokeStyle = "#c9c2b3";
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

    // Labels
    ctx.fillStyle = "#6e6358";
    ctx.font = "10px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText("Predicted value distribution", w / 2, h - 8);
  }, [viewMode, scenarioDetail]);

  // Model R² bars (from real data)
  useEffect(() => {
    const canvas = r2CanvasRef.current;
    if (!canvas) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    if (!models.length) {
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Run pipeline to see model R²", w / 2, h / 2);
      return;
    }

    const barH = 16;
    const gap = 6;
    const labelW = 60;

    models.forEach((m, i) => {
      const y = i * (barH + gap) + 4;
      ctx.fillStyle = "#6e6358";
      ctx.font = "10px 'JetBrains Mono'";
      ctx.textAlign = "right";
      ctx.fillText(m.name, labelW - 6, y + barH / 2 + 4);

      ctx.fillStyle = "rgba(0,0,0,0.04)";
      ctx.fillRect(labelW, y, w - labelW - 40, barH);

      ctx.fillStyle = m.color;
      ctx.fillRect(labelW, y, (w - labelW - 40) * m.r2, barH);

      ctx.fillStyle = "#1a1416";
      ctx.font = "bold 10px 'JetBrains Mono'";
      ctx.textAlign = "left";
      ctx.fillText(m.r2.toFixed(3), w - 36, y + barH / 2 + 4);
    });
  }, [models]);

  // Intervention response curve (from real PDP data)
  useEffect(() => {
    const canvas = curveCanvasRef.current;
    if (!canvas) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    const variables = pdpData ? Object.keys(pdpData) : [];
    if (!variables.length) {
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Run pipeline to see response curves", w / 2, h / 2);
      return;
    }

    // Axes
    ctx.strokeStyle = "#c9c2b3";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(40, 10); ctx.lineTo(40, h - 25); ctx.lineTo(w - 10, h - 25);
    ctx.stroke();

    const top3 = variables.slice(0, 3);
    top3.forEach((varName, vi) => {
      const curve = (pdpData as any)[varName];
      const gridVals: number[] = curve?.grid_values ?? [];
      const pdpVals: number[] = curve?.pdp_values ?? [];
      if (!gridVals.length || !pdpVals.length) return;

      const minG = Math.min(...gridVals), maxG = Math.max(...gridVals);
      const minP = Math.min(...pdpVals), maxP = Math.max(...pdpVals);
      const rG = maxG - minG || 1, rP = maxP - minP || 1;

      ctx.strokeStyle = SPARC_RAMP_HEX[vi * 3] ?? SPARC_RAMP_HEX[0];
      ctx.lineWidth = 2;
      ctx.beginPath();
      gridVals.forEach((g, i) => {
        const x = 42 + ((g - minG) / rG) * (w - 54);
        const y = (h - 35) - ((pdpVals[i] - minP) / rP) * (h - 45) + 10;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });

    // Legend
    ctx.font = "9px 'JetBrains Mono'";
    top3.forEach((varName, i) => {
      const x = 50 + i * 85;
      ctx.fillStyle = SPARC_RAMP_HEX[i * 3] ?? SPARC_RAMP_HEX[0];
      ctx.fillRect(x, h - 12, 12, 3);
      ctx.fillStyle = "#6e6358";
      ctx.textAlign = "left";
      ctx.fillText(varName.replace(/_/g, " ").slice(0, 10), x + 16, h - 8);
    });
  }, [pdpData]);

  // Correlogram canvas
  useEffect(() => {
    const canvas = corrCanvasRef.current;
    if (!canvas || viewMode !== "correlogram") return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    const results = correlogram?.individual_results;
    const varNames = results ? Object.keys(results) : [];
    if (!varNames.length) {
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Run pipeline to see correlogram", w / 2, h / 2);
      return;
    }

    const pad = { top: 20, right: 20, bottom: 35, left: 50 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    // Axes
    ctx.strokeStyle = "#c9c2b3";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, h - pad.bottom);
    ctx.lineTo(w - pad.right, h - pad.bottom);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = "#6e6358";
    ctx.font = "9px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText("Lag distance", w / 2, h - 6);
    ctx.save();
    ctx.translate(12, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Moran's I", 0, 0);
    ctx.restore();

    // Zero line
    ctx.strokeStyle = "#c9c2b3";
    ctx.setLineDash([4, 3]);
    const zeroY = pad.top + plotH;
    ctx.beginPath();
    ctx.moveTo(pad.left, zeroY);
    ctx.lineTo(w - pad.right, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw each variable's correlogram
    // Compute global ranges
    let allMinI = 0, allMaxI = 0, allMaxLag = 0;
    for (const name of varNames) {
      const cr = results![name].correlogram_results;
      const morans = cr.morans_i_values;
      const lags = cr.lag_distances;
      allMinI = Math.min(allMinI, ...morans);
      allMaxI = Math.max(allMaxI, ...morans);
      allMaxLag = Math.max(allMaxLag, ...lags);
    }
    const rangeI = allMaxI - allMinI || 1;

    const top = Math.min(varNames.length, 6);
    for (let vi = 0; vi < top; vi++) {
      const name = varNames[vi];
      const cr = results![name].correlogram_results;
      const lags = cr.lag_distances;
      const morans = cr.morans_i_values;
      const pvals = cr.p_values;

      const color = SPARC_RAMP_HEX[vi % SPARC_RAMP_HEX.length];
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < lags.length; i++) {
        const x = pad.left + (lags[i] / allMaxLag) * plotW;
        const y = pad.top + plotH - ((morans[i] - allMinI) / rangeI) * plotH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Significant points
      for (let i = 0; i < lags.length; i++) {
        if (pvals[i] < 0.05) {
          const x = pad.left + (lags[i] / allMaxLag) * plotW;
          const y = pad.top + plotH - ((morans[i] - allMinI) / rangeI) * plotH;
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // Legend
    ctx.font = "9px 'JetBrains Mono'";
    const legendY = pad.top + 4;
    for (let vi = 0; vi < top; vi++) {
      const name = varNames[vi];
      const x = pad.left + 6 + vi * 85;
      ctx.fillStyle = SPARC_RAMP_HEX[vi % SPARC_RAMP_HEX.length];
      ctx.fillRect(x, legendY, 10, 3);
      ctx.fillStyle = "#6e6358";
      ctx.textAlign = "left";
      ctx.fillText(name.replace(/_/g, " ").slice(0, 10), x + 14, legendY + 4);
    }
  }, [viewMode, correlogram]);

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
        <Stat label="Best R²" value={bestModel ? bestModel.r2.toFixed(3) : "—"} tint="var(--crimson)" sub={bestModel?.name} />
        <Stat label="Models" value={models.length ? String(models.length) : "—"} tint="var(--purple)" />
        <Stat label="Scenarios" value={scenarioRows.length ? String(scenarioRows.length) : "—"} tint="var(--amber)" />
        <Stat label="Features" value={scenarioDetail?.geojson?.features?.length ? String(scenarioDetail.geojson.features.length) : "—"} tint="var(--ink)" />
      </StatGrid>

      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 14 }}>
        {/* Main visualization */}
        <Card
          title="Predictions"
          subtitle={
            (scenarioDetail || correlogram) ? (
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                {/* View mode toggle */}
                {(["map", "histogram", "correlogram"] as const).map((mode) => (
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
            ) : undefined
          }
        >
          {viewMode === "map" ? (
            <div style={{ position: "relative" }}>
              <canvas
                ref={mapCanvasRef}
                style={{ width: "100%", height: 380, display: "block", borderRadius: 4 }}
              />
              {scenarioDetail?.geojson?.features?.length ? (
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
                    predicted value
                  </div>
                  <div style={{ display: "flex", height: 10, borderRadius: 2, overflow: "hidden", width: 120 }}>
                    {SPARC_RAMP_HEX.map((c, i) => (
                      <div key={i} style={{ flex: 1, background: c }} />
                    ))}
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginTop: 2 }}>
                    <span className="mono" style={{ fontSize: 8, color: "var(--muted)" }}>low</span>
                    <span className="mono" style={{ fontSize: 8, color: "var(--muted)" }}>high</span>
                  </div>
                </div>
              ) : null}
            </div>
          ) : viewMode === "histogram" ? (
            <canvas
              ref={histCanvasRef}
              style={{ width: "100%", height: 380, display: "block" }}
            />
          ) : (
            <canvas
              ref={corrCanvasRef}
              style={{ width: "100%", height: 380, display: "block" }}
            />
          )}
        </Card>

        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {/* Model R² bars */}
          <Card title="Model R²" subtitle={models.length ? "cross-validated performance" : undefined}>
            <canvas
              ref={r2CanvasRef}
              style={{ width: "100%", height: 120, display: "block" }}
            />
          </Card>

          {/* Intervention response curves */}
          <Card title="Intervention response" subtitle={pdpData ? "marginal effect on target" : undefined}>
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
