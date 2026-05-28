import { useState, useEffect, useCallback } from "react";
import { SectionHeader, Card, Btn, Stat, StatGrid, KeyVal } from "@/components/ui/DesignSystem";
import { getConfig, saveConfig, dataSummary, dataGeoJson, checkCrsDistortion } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import SpatialMap from "@/components/map/SpatialMap";
import type { DataSummary, GeoJsonData } from "@/lib/types";

interface DistortionResult {
  k_mean: number;
  area_distortion_pct: number;
  input_crs_name: string;
  projected_crs_name: string;
  assessment: string;
}

/**
 * Compute the UTM EPSG code for a centroid in WGS84 lon/lat.
 * Northern hemisphere zones → 326xx, southern → 327xx.
 */
function utmEpsgFromLonLat(lon: number, lat: number): number {
  const zone = Math.max(1, Math.min(60, Math.floor((lon + 180) / 6) + 1));
  return (lat >= 0 ? 32600 : 32700) + zone;
}

export default function CRSPage() {
  const [inputEpsg, setInputEpsg] = useState("4326");
  const [workingEpsg, setWorkingEpsg] = useState("3438");
  const [unitLabel, setUnitLabel] = useState<string | null>(null);
  const [distortionResult, setDistortionResult] = useState<DistortionResult | null>(null);
  const [distortionLoading, setDistortionLoading] = useState(false);
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [geojson, setGeojson] = useState<GeoJsonData | null>(null);
  const { notify } = useNotification();

  useEffect(() => {
    getConfig()
      .then((config) => {
        const crs = config.crs ?? {};
        if (crs.input) setInputEpsg(String(crs.input ?? "4326").replace("EPSG:", ""));
        // Prefer new 'working' key; fall back to legacy 'projected'
        const wpEpsg = crs.working ?? crs.projected;
        if (wpEpsg) setWorkingEpsg(String(wpEpsg).replace("EPSG:", ""));
      })
      .catch(() => {});

    dataSummary()
      .then((s) => setSummary(s))
      .catch(() => {});

    dataGeoJson()
      .then((geo) => {
        if (geo?.features?.length) setGeojson(geo);
      })
      .catch(() => {});
  }, []);

  // Derive unit label from working EPSG via detect-crs endpoint whenever it changes
  useEffect(() => {
    if (!workingEpsg) return;
    const derivedLabel =
      workingEpsg === "3438" || workingEpsg.startsWith("26772") || workingEpsg.startsWith("26773")
        ? "feet"
        : "meters";
    setUnitLabel(derivedLabel);
  }, [workingEpsg]);

  /**
   * Suggest a UTM zone matching the data centroid (or bbox center).
   * Only valid when input data is in geographic coordinates.
   */
  const handleSuggestUtm = useCallback(() => {
    if (!summary?.bbox) {
      notify("info", "Load data first to suggest a UTM zone");
      return;
    }
    const { minx, miny, maxx, maxy } = summary.bbox;
    const centerLon = (minx + maxx) / 2;
    const centerLat = (miny + maxy) / 2;
    if (Math.abs(centerLon) > 180 || Math.abs(centerLat) > 90) {
      notify("error", "Bounding box not in lon/lat — cannot infer UTM zone");
      return;
    }
    const epsg = utmEpsgFromLonLat(centerLon, centerLat);
    setWorkingEpsg(String(epsg));
    notify("success", `Suggested EPSG:${epsg} for UTM zone at ${centerLon.toFixed(2)}°, ${centerLat.toFixed(2)}°`);
  }, [summary, notify]);

  const handleCheckDistortion = useCallback(async () => {
    if (!workingEpsg) { notify("error", "Working EPSG is required"); return; }
    setDistortionLoading(true);
    try {
      const result = await checkCrsDistortion(inputEpsg, workingEpsg);
      setDistortionResult(result);
    } catch {
      notify("error", "Could not compute distortion — check EPSG codes");
    } finally {
      setDistortionLoading(false);
    }
  }, [inputEpsg, workingEpsg, notify]);

  useEffect(() => {
    // no-op: preview now lives in <SpatialMap />
  }, [inputEpsg, workingEpsg, summary]);

  const handleSaveCRS = async () => {
    try {
      await saveConfig({
        crs: { input: `EPSG:${inputEpsg}`, working: `EPSG:${workingEpsg}` },
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
        <Stat label="Working EPSG" value={workingEpsg} tint="var(--crimson)" />
        <Stat label="Working Units" value={unitLabel ?? "—"} tint="var(--purple)" />
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

        <Card title="Working CRS" subtitle={`EPSG:${workingEpsg}${distortionResult ? ` · ${distortionResult.projected_crs_name}` : ""}${unitLabel ? ` · ${unitLabel}` : ""}`}>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div>
              <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>
                EPSG Code (projected working CRS)
              </div>
              <div style={{ display: "flex", gap: 6 }}>
                <input
                  type="text"
                  value={workingEpsg}
                  onChange={(e) => setWorkingEpsg(e.target.value)}
                  className="mono"
                  style={{
                    border: "1px solid var(--line)",
                    borderRadius: 4,
                    padding: "6px 8px",
                    fontSize: 13,
                    fontWeight: 600,
                    flex: 1,
                    fontFamily: "inherit",
                    background: "#fff",
                  }}
                />
                <button
                  onClick={handleSuggestUtm}
                  title="Suggest a UTM zone matching the data centroid"
                  style={{
                    border: "1px solid var(--line)", background: "#fff",
                    color: "var(--ink-2)", borderRadius: 4, padding: "4px 10px",
                    fontSize: 11, fontWeight: 600, cursor: "pointer",
                    fontFamily: "inherit",
                  }}
                >
                  Suggest UTM
                </button>
              </div>
            </div>
            {unitLabel && (
              <div style={{
                display: "inline-flex", alignItems: "center", gap: 5,
                padding: "3px 8px", borderRadius: 3,
                background: unitLabel === "meters" ? "#f0faf0" : "#fff8ef",
                border: `1px solid ${unitLabel === "meters" ? "#b2dfb2" : "var(--amber)"}`,
                fontSize: 11, fontWeight: 600,
                color: unitLabel === "meters" ? "#2d7a2d" : "var(--amber)",
                width: "fit-content",
              }}>
                {unitLabel === "meters" ? "📏 meters" : "📐 feet"}
              </div>
            )}
            <div style={{ fontSize: 11.5, color: "var(--muted)", lineHeight: 1.6 }}>
              Choose a local projected CRS (State Plane, UTM, etc.) matching your study area for accurate distance/area calculations.
              Block sizes and buffer distances will use <strong>{unitLabel ?? "the working CRS units"}</strong>.
            </div>
          </div>
        </Card>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginTop: 14 }}>
        <Card title="Data projection preview" subtitle="actual study points on a basemap">
          <div style={{ height: 240, borderRadius: 4, overflow: "hidden", border: "1px solid var(--line)" }}>
            {geojson ? (
              <SpatialMap geojson={geojson} mode="scatter" height="100%" expandable />
            ) : (
              <div style={{ display: "flex", height: "100%", alignItems: "center", justifyContent: "center", color: "var(--muted)", fontSize: 12, background: "#faf8f4" }}>
                Load data to see spatial preview
              </div>
            )}
          </div>
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
                <KeyVal label="Working CRS" value={distortionResult.projected_crs_name} />
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
