import { useState, useEffect, useCallback } from "react";
import { SectionHeader, Card, Stat, Tag, Btn, StatGrid, KeyVal, thStyle, tdStyle } from "@/components/ui/DesignSystem";
import { dataSummary, getConfig, dataGeoJson, saveConfig, checkCrsDistortion, selectDataFile } from "@/lib/api";
import { Histogram } from "@/components/data/Histogram";
import { EmptyState } from "@/components/common/EmptyState";
import SpatialMap from "@/components/map/SpatialMap";
import { useNotification } from "@/hooks/useNotifications";
import type { DataSummary, GeoJsonData } from "@/lib/types";
import { MAP_HEIGHT_DEFAULT } from "@/lib/design-tokens";
import DataCollectionPage from "@/components/pages/DataCollectionPage";
import { useNavigationStore } from "@/stores/navigationStore";

type DataTab = "my-data" | "collect-data";

interface DistortionResult {
  k_mean: number;
  area_distortion_pct: number;
  input_crs_name: string;
  projected_crs_name: string;
  assessment: string;
}

function utmEpsgFromLonLat(lon: number, lat: number): number {
  const zone = Math.max(1, Math.min(60, Math.floor((lon + 180) / 6) + 1));
  return (lat >= 0 ? 32600 : 32700) + zone;
}

function formatCrs(crs: string): string {
  if (crs.startsWith("{")) {
    try {
      const obj = JSON.parse(crs) as { name?: string };
      if (obj.name) return obj.name;
    } catch {}
  }
  return crs.length > 42 ? crs.slice(0, 42) + "…" : crs;
}

export default function DataPage() {
  const [dataTab, setDataTab] = useState<DataTab>("my-data");
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [target, setTarget] = useState<string>("AAT_z");
  const [geo, setGeo] = useState<GeoJsonData | null>(null);
  const [inputEpsg, setInputEpsg] = useState("4326");
  const [projectedEpsg, setProjectedEpsg] = useState("3438");
  const [distortionResult, setDistortionResult] = useState<DistortionResult | null>(null);
  const [distortionLoading, setDistortionLoading] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const { notify } = useNotification();

  const refresh = useCallback(async () => {
    const [s, c] = await Promise.all([
      dataSummary().catch(() => null),
      getConfig().catch(() => null),
    ]);
    if (s) setSummary(s);
    if (c?.data?.target_column) setTarget(c.data.target_column);
    if (c?.crs?.input) setInputEpsg(String(c.crs.input).replace("EPSG:", ""));
    if (c?.crs?.projected) setProjectedEpsg(String(c.crs.projected).replace("EPSG:", ""));
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // Load spatial preview whenever target changes and summary is available
  useEffect(() => {
    if (!summary) return;
    dataGeoJson(target).then(setGeo).catch(() => setGeo(null));
  }, [summary, target]);

  const handleUpload = useCallback(async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const path = await open({
        filters: [{ name: "Data", extensions: ["csv", "parquet", "gpkg"] }],
        multiple: false,
      });
      if (!path) return;
      setIsLoading(true);
      await selectDataFile(path as string);
      await refresh();
      notify("success", "Data loaded successfully");
      useNavigationStore.getState().markDataReady();
    } catch (err) {
      notify("error", err instanceof Error ? err.message : "Failed to load data");
    } finally {
      setIsLoading(false);
    }
  }, [notify, refresh]);

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
    setProjectedEpsg(String(epsg));
    notify("success", `Suggested EPSG:${epsg} for UTM zone at ${centerLon.toFixed(2)}°, ${centerLat.toFixed(2)}°`);
  }, [summary, notify]);

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

  const handleSaveCRS = useCallback(async () => {
    try {
      await saveConfig({ crs: { input: `EPSG:${inputEpsg}`, projected: `EPSG:${projectedEpsg}` } });
      notify("success", "CRS settings saved");
    } catch {
      notify("error", "Failed to save CRS");
    }
  }, [inputEpsg, projectedEpsg, notify]);

  const columns = summary?.columns ?? [];
  const numericSummary = summary?.numeric_summary ?? {};
  const nRows = summary?.row_count ?? 0;

  const getRole = (col: string) => {
    if (col === target) return { tag: "Target", color: "var(--crimson)" };
    if (col === "x" || col === "y" || col === "POINT_X" || col === "POINT_Y") return { tag: "Coord", color: "var(--purple)" };
    if (col === "id" || col === "FID") return { tag: "ID", color: "var(--muted)" };
    return { tag: "Predictor", color: "var(--ink-2)" };
  };

  const getType = (col: string) => {
    const dtype = summary?.dtypes?.[col];
    if (dtype) {
      if (dtype.includes("int")) return "int";
      if (dtype.includes("float")) return "float";
      if (dtype.includes("str") || dtype.includes("object")) return "str";
      return dtype.replace("64", "").replace("32", "");
    }
    const s = numericSummary[col];
    if (!s) return "str";
    if (Number.isInteger(s.min) && Number.isInteger(s.max)) return "int";
    return "float";
  };

  const getRangeText = (col: string) => {
    const s = numericSummary[col];
    if (!s) return "—";
    return `${s.min?.toFixed?.(2) ?? s.min} … ${s.max?.toFixed?.(2) ?? s.max}`;
  };

  const getNullCount = (col: string): number | null => {
    const s = numericSummary[col];
    if (!s || s.count == null) return null;
    return Math.max(0, nRows - s.count);
  };

  // Detect geometry type from dtypes or column names
  const geomType = (() => {
    if (!summary) return null;
    const dtypes = summary.dtypes ?? {};
    const hasGeomCol = Object.entries(dtypes).some(([k, v]) =>
      v.toLowerCase().includes("geom") || k.toLowerCase() === "geometry"
    );
    if (hasGeomCol) return "Polygon";
    const cols = summary.columns.map((c) => c.toLowerCase());
    if (cols.includes("x") || cols.includes("point_x") || cols.includes("longitude") || cols.includes("lon")) return "Point";
    if (summary.bbox) return "Spatial";
    return null;
  })();

  const tabBtnStyle = (active: boolean): React.CSSProperties => ({
    padding: "7px 16px",
    border: "none",
    borderBottom: active ? "2px solid var(--crimson)" : "2px solid transparent",
    background: "transparent",
    color: active ? "var(--ink)" : "var(--muted)",
    fontWeight: active ? 700 : 500,
    fontSize: 13,
    cursor: "pointer",
    fontFamily: "inherit",
    transition: "color 0.12s, border-color 0.12s",
  });

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      {/* Tab bar */}
      <div style={{ display: "flex", alignItems: "center", borderBottom: "1px solid var(--line)", background: "var(--paper)", padding: "0 16px", gap: 2, flexShrink: 0 }}>
        <button style={tabBtnStyle(dataTab === "my-data")} onClick={() => setDataTab("my-data")}>My Data</button>
        <button style={tabBtnStyle(dataTab === "collect-data")} onClick={() => setDataTab("collect-data")}>Collect Data</button>
      </div>

      {/* Collect Data tab */}
      {dataTab === "collect-data" && (
        <div style={{ flex: 1, overflow: "auto", minHeight: 0 }}>
          <DataCollectionPage />
        </div>
      )}

      {/* My Data tab */}
      {dataTab === "my-data" && (
      <div style={{ flex: 1, overflow: "auto", minHeight: 0 }}>
      <div>
      <SectionHeader
        kicker="02 · setup"
        label="Data"
        right={<Btn small onClick={handleUpload} disabled={isLoading}>{isLoading ? "Loading…" : "Load data…"}</Btn>}
      />

      {!summary ? (
        <Card>
          <EmptyState
            tone="blank"
            title="No data loaded yet."
            body={
              <>
                Load a <code>.csv</code>, <code>.parquet</code>, or <code>.gpkg</code> file to
                see schema, distributions, and a spatial preview. You can also build a dataset from
                rasters and a boundary on the <strong>Collect Data</strong> tab.
              </>
            }
            action={<Btn primary onClick={handleUpload}>Load data…</Btn>}
          />
        </Card>
      ) : (
        <>
          {/* Stats row */}
          <StatGrid>
            <Stat label="Rows" value={nRows.toLocaleString()} tint="var(--ink)" />
            <Stat label="Columns" value={String(columns.length)} tint="var(--ink)" />
            {geomType && <Stat label="Geometry" value={geomType} tint="var(--purple)" />}
            {summary.crs && <Stat label="Source CRS" value={formatCrs(summary.crs)} tint="var(--muted)" weight={400} />}
          </StatGrid>

          {/* Target + spatial preview side-by-side */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginTop: 14 }}>
            <Card
              title="Target variable"
              subtitle={`${target} · ${numericSummary[target] ? `mean ${numericSummary[target].mean?.toFixed(2)}` : "no stats"}`}
            >
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                <Histogram variable={target} bins={40} height={120} />
                {numericSummary[target] && (
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
                    {[
                      { label: "min", val: numericSummary[target].min },
                      { label: "mean", val: numericSummary[target].mean },
                      { label: "max", val: numericSummary[target].max },
                      { label: "std", val: numericSummary[target].std },
                    ].map(({ label, val }) => (
                      <div key={label} style={{ textAlign: "center" }}>
                        <div className="mono" style={{ fontSize: 9, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em" }}>{label}</div>
                        <div style={{ fontSize: 13, fontWeight: 700, marginTop: 2, color: "var(--ink)" }}>
                          {val != null ? val.toFixed(3) : "—"}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </Card>

            <Card title="Spatial preview" subtitle={`${nRows.toLocaleString()} points · colored by ${target}`} padding={0}>
              <div style={{ height: MAP_HEIGHT_DEFAULT, borderRadius: "0 0 8px 8px", overflow: "hidden" }}>
                <SpatialMap
                  geojson={geo}
                  colorField={target}
                  mode="scatter"
                  height="100%"
                  palette="sparc"
                />
              </div>
            </Card>
          </div>

          {/* Schema table */}
          <Card
            title="Schema"
            subtitle={`${nRows.toLocaleString()} rows · ${columns.length} columns`}
            style={{ marginTop: 14 }}
          >
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ textAlign: "left", color: "var(--muted)" }}>
                  <th style={thStyle}>#</th>
                  <th style={thStyle}>Column</th>
                  <th style={thStyle}>Type</th>
                  <th style={thStyle}>Range</th>
                  <th style={thStyle}>Missing</th>
                  <th style={thStyle}>Distribution</th>
                  <th style={thStyle}>Role</th>
                </tr>
              </thead>
              <tbody>
                {columns.map((col, i) => {
                  const role = getRole(col);
                  const nullCount = getNullCount(col);
                  const nullPct = nullCount != null && nRows > 0 ? (nullCount / nRows) * 100 : null;
                  return (
                    <tr key={col} style={{ borderTop: "1px solid var(--line)" }}>
                      <td style={{ ...tdStyle, color: "var(--muted)" }} className="mono">
                        {String(i).padStart(2, "0")}
                      </td>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{col}</td>
                      <td style={tdStyle}>
                        <span className="mono" style={{ fontSize: 10, color: "var(--purple)" }}>
                          {getType(col)}
                        </span>
                      </td>
                      <td style={{ ...tdStyle, color: "var(--muted)" }} className="mono">
                        {getRangeText(col)}
                      </td>
                      <td style={tdStyle}>
                        {nullPct != null ? (
                          <span
                            className="mono"
                            style={{
                              fontSize: 10,
                              color: nullPct > 5 ? "var(--crimson)" : nullPct > 0 ? "var(--amber)" : "var(--muted)",
                              fontWeight: nullPct > 5 ? 700 : 400,
                            }}
                          >
                            {nullPct === 0 ? "—" : `${nullPct.toFixed(1)}%`}
                          </span>
                        ) : (
                          <span className="mono" style={{ fontSize: 10, color: "var(--muted)" }}>—</span>
                        )}
                      </td>
                      <td style={{ ...tdStyle, width: 140 }}>
                        {getType(col) !== "str" ? (
                          <div style={{ width: 130 }}>
                            <Histogram variable={col} bins={28} height={22} compact />
                          </div>
                        ) : (
                          <span className="mono" style={{ fontSize: 10, color: "var(--muted)" }}>—</span>
                        )}
                      </td>
                      <td style={tdStyle}>
                        <Tag color={role.color}>{role.tag}</Tag>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </Card>

          {/* Coordinate Reference System */}
          <Card
            title="Coordinate Reference System"
            subtitle="input and projected EPSG codes required for spatial analysis"
            actions={<Btn small primary onClick={handleSaveCRS}>Save CRS</Btn>}
            style={{ marginTop: 14 }}
          >
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                <div>
                  <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>
                    Input CRS — geographic
                  </div>
                  <input
                    type="text"
                    value={inputEpsg}
                    onChange={(e) => setInputEpsg(e.target.value)}
                    placeholder="e.g. 4326"
                    className="mono"
                    style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "6px 8px", fontSize: 13, fontWeight: 600, width: "100%", fontFamily: "inherit", background: "#fff", boxSizing: "border-box" }}
                  />
                  <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>Native CRS of your input data — usually WGS 84 (4326)</div>
                </div>
                <div>
                  <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>
                    Projected CRS — local
                  </div>
                  <div style={{ display: "flex", gap: 6 }}>
                    <input
                      type="text"
                      value={projectedEpsg}
                      onChange={(e) => setProjectedEpsg(e.target.value)}
                      placeholder="e.g. 32617"
                      className="mono"
                      style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "6px 8px", fontSize: 13, fontWeight: 600, flex: 1, fontFamily: "inherit", background: "#fff" }}
                    />
                    <Btn small onClick={handleSuggestUtm}>Suggest UTM</Btn>
                  </div>
                  <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>Local projected CRS for accurate distances — State Plane, UTM, etc.</div>
                </div>
              </div>
              {distortionResult ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
                    <KeyVal label="Input CRS" value={distortionResult.input_crs_name} />
                    <KeyVal label="Projected CRS" value={distortionResult.projected_crs_name} />
                    <KeyVal label="Area distortion" value={`${distortionResult.area_distortion_pct.toFixed(3)}%`} />
                    <KeyVal label="Scale factor k" value={distortionResult.k_mean.toFixed(6)} />
                  </div>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "6px 10px", background: distortionResult.assessment === "acceptable" ? "#f0faf0" : "#fff5f0", borderRadius: 6, fontSize: 12 }}>
                    <span>
                      {distortionResult.assessment === "acceptable"
                        ? "✓ Distortion within acceptable range for spatial analysis"
                        : "⚠ Consider a local CRS — high distortion may affect spatial results"}
                    </span>
                    <Btn small onClick={handleCheckDistortion}>{distortionLoading ? "Checking…" : "Re-check"}</Btn>
                  </div>
                </div>
              ) : (
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span style={{ fontSize: 12, color: "var(--muted)" }}>Verify area distortion for the current EPSG pair</span>
                  <Btn small onClick={handleCheckDistortion} disabled={distortionLoading}>{distortionLoading ? "Checking…" : "Check distortion"}</Btn>
                </div>
              )}
            </div>
          </Card>
        </>
      )}
    </div>
    </div>
      )}
    </div>
  );
}

