import { useState, useEffect, useRef, useCallback } from "react";
import { SectionHeader, Card, Btn, Stat, StatGrid, KeyVal } from "@/components/ui/DesignSystem";
import { getConfig, saveConfig, dataSummary, dataGeoJson, checkCrsDistortion } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import type { DataSummary } from "@/lib/types";

interface DistortionResult {
  k_mean: number;
  area_distortion_pct: number;
  input_crs_name: string;
  projected_crs_name: string;
  assessment: string;
}

export default function CRSPage() {
  const [inputEpsg, setInputEpsg] = useState("4326");
  const [projectedEpsg, setProjectedEpsg] = useState("3438");
  const [distortionResult, setDistortionResult] = useState<DistortionResult | null>(null);
  const [distortionLoading, setDistortionLoading] = useState(false);
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [samplePts, setSamplePts] = useState<[number, number][]>([]);
  const previewCanvasRef = useRef<HTMLCanvasElement>(null);
  const { notify } = useNotification();

  useEffect(() => {
    getConfig()
      .then((config) => {
        const crs = config.crs ?? {};
        if (crs.input) setInputEpsg(String(crs.input ?? "4326").replace("EPSG:", ""));
        if (crs.projected) setProjectedEpsg(String(crs.projected ?? "3438").replace("EPSG:", ""));
      })
      .catch(() => {});

    dataSummary()
      .then((s) => setSummary(s))
      .catch(() => {});

    dataGeoJson()
      .then((geo) => {
        if (geo?.features) {
          const pts: [number, number][] = [];
          for (const f of geo.features.slice(0, 200)) {
            const coords = f.geometry?.coordinates;
            if (Array.isArray(coords) && coords.length >= 2) {
              pts.push([coords[0] as number, coords[1] as number]);
            }
          }
          setSamplePts(pts);
        }
      })
      .catch(() => {});
  }, []);

  const handleCheckDistortion = useCallback(async () => {
    if (!projectedEpsg) { notify("error", "Projected EPSG is required"); return; }
    setDistortionLoading(true);
    try {
      const result = await checkCrsDistortion(inputEpsg, projectedEpsg);
      setDistortionResult(result);
    } catch {
      notify("error", "Could not compute distortion — check EPSG codes");
    } finally {
      setDistortionLoading(false);
    }
  }, [inputEpsg, projectedEpsg, notify]);

  // Draw reprojection preview using real data
  useEffect(() => {
    const canvas = previewCanvasRef.current;
    if (!canvas) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    // Background
    ctx.fillStyle = "#faf8f4";
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = "rgba(0,0,0,0.06)";
    ctx.lineWidth = 0.5;
    const gridSize = 20;
    for (let x = 0; x < w; x += gridSize) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += gridSize) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Use real bbox from summary or sample points
    const bbox = summary?.bbox;
    const hasPts = samplePts.length > 0;
    const hasData = bbox || hasPts;

    if (!hasData) {
      ctx.fillStyle = "var(--muted)";
      ctx.font = "12px Inter";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Load data to see spatial preview", w / 2, h / 2);
      return;
    }

    // Compute bounds from bbox object or sample points
    let xMin: number, xMax: number, yMin: number, yMax: number;
    if (bbox && typeof bbox === "object" && "minx" in bbox) {
      xMin = bbox.minx; yMin = bbox.miny; xMax = bbox.maxx; yMax = bbox.maxy;
    } else if (hasPts) {
      xMin = Math.min(...samplePts.map(p => p[0]));
      xMax = Math.max(...samplePts.map(p => p[0]));
      yMin = Math.min(...samplePts.map(p => p[1]));
      yMax = Math.max(...samplePts.map(p => p[1]));
    } else {
      return;
    }

    const pad = 20;
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    const toCanvasX = (v: number) => pad + ((v - xMin) / xRange) * (w - 2 * pad);
    const toCanvasY = (v: number) => h - pad - ((v - yMin) / yRange) * (h - 2 * pad);

    // Draw points
    if (hasPts) {
      ctx.fillStyle = "var(--crimson)";
      for (const [px, py] of samplePts) {
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.arc(toCanvasX(px), toCanvasY(py), 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    // Bounding box
    const bx1 = toCanvasX(xMin), by1 = toCanvasY(yMax), bx2 = toCanvasX(xMax), by2 = toCanvasY(yMin);
    ctx.strokeStyle = "var(--purple)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1);
    ctx.setLineDash([]);

    // Label
    ctx.fillStyle = "var(--muted)";
    ctx.font = "10px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText(`EPSG:${inputEpsg} → EPSG:${projectedEpsg}`, w / 2, h - 4);
  }, [inputEpsg, projectedEpsg, summary, samplePts]);

  const handleSaveCRS = async () => {
    try {
      await saveConfig({
        crs: { input: `EPSG:${inputEpsg}`, projected: `EPSG:${projectedEpsg}` },
      });
      notify("success", "CRS settings saved");
    } catch {
      notify("error", "Failed to save CRS");
    }
  };

  return (
    <div>
      <SectionHeader
        kicker="07 · analysis"
        label="CRS"
        right={<Btn primary small onClick={handleSaveCRS}>Save CRS</Btn>}
      />

      <StatGrid>
        <Stat label="Input EPSG" value={inputEpsg} tint="var(--ink)" />
        <Stat label="Projected EPSG" value={projectedEpsg} tint="var(--crimson)" />
        <Stat
          label="Area Distortion"
          value={distortionResult ? `${distortionResult.area_distortion_pct.toFixed(3)}%` : "—"}
          tint={distortionResult && distortionResult.area_distortion_pct > 0.5 ? "var(--crimson)" : "var(--purple)"}
        />
        <Stat label="Scale Factor" value={distortionResult ? distortionResult.k_mean.toFixed(6) : "—"} tint="var(--amber)" />
      </StatGrid>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Card title="Input CRS" subtitle={`EPSG:${inputEpsg}${distortionResult ? ` · ${distortionResult.input_crs_name}` : ""}`}>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div>
              <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>
                EPSG Code (geographic CRS)
              </div>
              <input
                type="text"
                value={inputEpsg}
                onChange={(e) => setInputEpsg(e.target.value)}
                className="mono"
                style={{
                  border: "1px solid var(--line)",
                  borderRadius: 4,
                  padding: "6px 8px",
                  fontSize: 13,
                  fontWeight: 600,
                  width: "100%",
                  fontFamily: "inherit",
                  background: "#fff",
                }}
              />
            </div>
            <div style={{ fontSize: 11.5, color: "var(--muted)", lineHeight: 1.6 }}>
              This is the native coordinate system of your input data (usually WGS 84 / EPSG:4326 for GPS-sourced data).
            </div>
          </div>
        </Card>

        <Card title="Projected CRS" subtitle={`EPSG:${projectedEpsg}${distortionResult ? ` · ${distortionResult.projected_crs_name}` : ""}`}>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div>
              <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>
                EPSG Code (projected CRS)
              </div>
              <input
                type="text"
                value={projectedEpsg}
                onChange={(e) => setProjectedEpsg(e.target.value)}
                className="mono"
                style={{
                  border: "1px solid var(--line)",
                  borderRadius: 4,
                  padding: "6px 8px",
                  fontSize: 13,
                  fontWeight: 600,
                  width: "100%",
                  fontFamily: "inherit",
                  background: "#fff",
                }}
              />
            </div>
            <div style={{ fontSize: 11.5, color: "var(--muted)", lineHeight: 1.6 }}>
              Choose a local projected CRS (State Plane, UTM, etc.) matching your study area for accurate distance/area calculations.
            </div>
          </div>
        </Card>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginTop: 14 }}>
        <Card title="Data projection preview" subtitle="spatial extent of loaded data">
          <canvas
            ref={previewCanvasRef}
            style={{ width: "100%", height: 220, display: "block" }}
          />
        </Card>

        <Card
          title="Distortion check"
          subtitle={distortionResult ? `area distortion at study center: ${distortionResult.area_distortion_pct.toFixed(3)}%` : "no pipeline required"}
          actions={
            <button
              onClick={handleCheckDistortion}
              disabled={distortionLoading}
              style={{
                padding: "4px 12px",
                background: distortionLoading ? "var(--muted)" : "var(--crimson, #e73c25)",
                color: "#fff",
                border: "none",
                borderRadius: 5,
                fontSize: 11,
                cursor: distortionLoading ? "not-allowed" : "pointer",
                fontWeight: 600,
              }}
            >
              {distortionLoading ? "Checking…" : "Check"}
            </button>
          }
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {distortionResult ? (
              <>
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ fontSize: 12, fontWeight: 600 }}>Area distortion</span>
                    <span className="mono" style={{ fontSize: 12, color: distortionResult.area_distortion_pct < 0.5 ? "var(--purple)" : "var(--crimson)", fontWeight: 700 }}>
                      {distortionResult.area_distortion_pct.toFixed(3)}%
                    </span>
                  </div>
                  <div style={{ height: 8, background: "rgba(0,0,0,0.05)", borderRadius: 4, overflow: "hidden" }}>
                    <div
                      style={{
                        width: `${Math.min(distortionResult.area_distortion_pct * 10, 100)}%`,
                        height: "100%",
                        background: distortionResult.area_distortion_pct < 0.5 ? "var(--purple)" : "var(--crimson)",
                        transition: "width 0.3s",
                      }}
                    />
                  </div>
                </div>
                <KeyVal label="Linear scale k" value={distortionResult.k_mean.toFixed(6)} />
                <KeyVal label="Input CRS" value={distortionResult.input_crs_name} />
                <KeyVal label="Projected CRS" value={distortionResult.projected_crs_name} />
                <div style={{ padding: "8px 10px", background: distortionResult.assessment === "acceptable" ? "#f0faf0" : "#fff5f0", borderRadius: 6, fontSize: 12 }}>
                  {distortionResult.assessment === "acceptable"
                    ? "✓ Distortion within acceptable range for spatial analysis"
                    : "⚠ Consider a local CRS — high distortion may affect spatial results"}
                </div>
              </>
            ) : (
              <div style={{ color: "var(--muted)", fontSize: 12, textAlign: "center", padding: 20 }}>
                Enter EPSG codes above and click <strong>Check</strong> to compute area distortion at the study center.
                <br /><em style={{ fontSize: 11 }}>No pipeline run required.</em>
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
