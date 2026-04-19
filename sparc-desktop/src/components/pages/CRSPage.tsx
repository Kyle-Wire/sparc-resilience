import { useState, useEffect, useRef } from "react";
import { SectionHeader, Card, Btn, Stat, StatGrid, KeyVal } from "@/components/ui/DesignSystem";
import { getConfig, saveConfig } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";

export default function CRSPage() {
  const [inputEpsg, setInputEpsg] = useState("4326");
  const [projectedEpsg, setProjectedEpsg] = useState("3438");
  const [inputName, setInputName] = useState("WGS 84");
  const [projectedName, setProjectedName] = useState("NAD83 / Rhode Island");
  const [distortion, _setDistortion] = useState<number>(0.12);
  const previewCanvasRef = useRef<HTMLCanvasElement>(null);
  const { notify } = useNotification();

  useEffect(() => {
    getConfig()
      .then((config) => {
        const crs = config.crs ?? {};
        if (crs.input) setInputEpsg(String(crs.input ?? "4326"));
        if (crs.projected) setProjectedEpsg(String(crs.projected ?? "3438"));
      })
      .catch(() => {});
  }, []);

  // Draw reprojection preview
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

    // Draw point cloud
    const cx = w / 2, cy = h / 2;
    const nPts = 120;
    ctx.fillStyle = "var(--crimson)";
    for (let i = 0; i < nPts; i++) {
      const angle = Math.random() * Math.PI * 2;
      const r = Math.random() * Math.min(w, h) * 0.35;
      const x = cx + Math.cos(angle) * r * (1 + Math.random() * 0.3);
      const y = cy + Math.sin(angle) * r * (0.7 + Math.random() * 0.2);
      ctx.globalAlpha = 0.4;
      ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Bounding box
    ctx.strokeStyle = "var(--purple)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(cx - w * 0.32, cy - h * 0.28, w * 0.64, h * 0.56);
    ctx.setLineDash([]);

    // Label
    ctx.fillStyle = "var(--muted)";
    ctx.font = "10px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText(`EPSG:${inputEpsg} → EPSG:${projectedEpsg}`, w / 2, h - 8);
  }, [inputEpsg, projectedEpsg]);

  const handleSaveCRS = async () => {
    try {
      await saveConfig({
        crs: { input: String(inputEpsg), projected: String(projectedEpsg) },
      });
      notify("success", "CRS settings saved");
    } catch {
      notify("error", "Failed to save CRS");
    }
  };

  const inputWkt = `GEOGCS["${inputName}",
  DATUM["World Geodetic System 1984",
    SPHEROID["WGS 84", 6378137.0, 298.257223563]],
  PRIMEM["Greenwich", 0.0],
  UNIT["degree", 0.017453292519943295]]`;

  const projectedWkt = `PROJCS["${projectedName}",
  GEOGCS["NAD83",
    DATUM["North_American_Datum_1983",
      SPHEROID["GRS 1980", 6378137.0, 298.257222101]]],
  PROJECTION["Transverse_Mercator"],
  PARAMETER["central_meridian", -71.5],
  UNIT["US survey foot", 0.30480060960122]]`;

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
        <Stat label="Distortion" value={`${distortion.toFixed(2)}%`} tint={distortion > 0.5 ? "var(--crimson)" : "var(--purple)"} />
        <Stat label="Unit" value="US ft" tint="var(--amber)" />
      </StatGrid>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Card title="Input CRS" subtitle={`EPSG:${inputEpsg} · ${inputName}`}>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <div>
                <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>
                  EPSG Code
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
              <div>
                <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>
                  Name
                </div>
                <input
                  type="text"
                  value={inputName}
                  onChange={(e) => setInputName(e.target.value)}
                  style={{
                    border: "1px solid var(--line)",
                    borderRadius: 4,
                    padding: "6px 8px",
                    fontSize: 12,
                    width: "100%",
                    fontFamily: "inherit",
                    background: "#fff",
                  }}
                />
              </div>
            </div>
            <div>
              <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>
                WKT
              </div>
              <pre className="mono" style={{ fontSize: 9.5, margin: 0, lineHeight: 1.5, color: "var(--ink-2)", background: "rgba(0,0,0,0.02)", padding: 8, borderRadius: 4, overflow: "auto", maxHeight: 140 }}>
                {inputWkt}
              </pre>
            </div>
          </div>
        </Card>

        <Card title="Projected CRS" subtitle={`EPSG:${projectedEpsg} · ${projectedName}`}>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <div>
                <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>
                  EPSG Code
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
              <div>
                <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>
                  Name
                </div>
                <input
                  type="text"
                  value={projectedName}
                  onChange={(e) => setProjectedName(e.target.value)}
                  style={{
                    border: "1px solid var(--line)",
                    borderRadius: 4,
                    padding: "6px 8px",
                    fontSize: 12,
                    width: "100%",
                    fontFamily: "inherit",
                    background: "#fff",
                  }}
                />
              </div>
            </div>
            <div>
              <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>
                WKT
              </div>
              <pre className="mono" style={{ fontSize: 9.5, margin: 0, lineHeight: 1.5, color: "var(--ink-2)", background: "rgba(0,0,0,0.02)", padding: 8, borderRadius: 4, overflow: "auto", maxHeight: 140 }}>
                {projectedWkt}
              </pre>
            </div>
          </div>
        </Card>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginTop: 14 }}>
        <Card title="Reprojection preview" subtitle="spatial extent with transformed coordinates">
          <canvas
            ref={previewCanvasRef}
            style={{ width: "100%", height: 220, display: "block" }}
          />
        </Card>

        <Card title="Distortion check" subtitle={`area distortion at study site: ${distortion.toFixed(2)}%`}>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ fontSize: 12, fontWeight: 600 }}>Area distortion</span>
                <span className="mono" style={{ fontSize: 12, color: distortion < 0.5 ? "var(--purple)" : "var(--crimson)", fontWeight: 700 }}>
                  {distortion.toFixed(2)}%
                </span>
              </div>
              <div style={{ height: 8, background: "rgba(0,0,0,0.05)", borderRadius: 4, overflow: "hidden" }}>
                <div
                  style={{
                    width: `${Math.min(distortion * 10, 100)}%`,
                    height: "100%",
                    background: distortion < 0.5 ? "var(--purple)" : "var(--crimson)",
                    transition: "width 0.3s",
                  }}
                />
              </div>
            </div>

            <KeyVal label="Scale factor" value="0.999966" />
            <KeyVal label="Central meridian" value="-71.5°" />
            <KeyVal label="False easting" value="100,000 m" />

            <div style={{ padding: "8px 10px", background: distortion < 0.5 ? "#f0faf0" : "#fff5f0", borderRadius: 6, fontSize: 12 }}>
              {distortion < 0.5
                ? "✓ Distortion within acceptable range for spatial analysis"
                : "⚠ Consider a local CRS for better accuracy"}
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
