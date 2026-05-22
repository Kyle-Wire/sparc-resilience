/**
 * DataCollectionPage — 5-step stepper that lets researchers:
 *   1. Define a study boundary
 *   2. Configure temporal window + temporal mode
 *   3. Fetch each variable group from free public APIs
 *   4. Inspect and confirm data quality on a 30m choropleth map
 *   5. Export GeoParquet and update project.yml
 */
import { useState, useCallback, useEffect } from "react";
import { SectionHeader, Card, Btn, StatGrid, Stat } from "@/components/ui/DesignSystem";
import { useNotification } from "@/hooks/useNotifications";
import {
  collectManifest,
  collectFetch,
  collectPreview,
  collectCellInspect,
  collectBuild,
  getConfig,
} from "@/lib/api";
import type {
  BoundaryResponse,
  CollectManifest,
  FetchGroupResponse,
} from "@/lib/api";
import type { GeoJsonData } from "@/lib/types";
import SpatialMap from "@/components/map/SpatialMap";
import BoundarySelector from "@/components/collect/BoundarySelector";
import VariableManifestTable from "@/components/collect/VariableManifestTable";
import CellInspector from "@/components/collect/CellInspector";

type Step = 1 | 2 | 3 | 4 | 5;
type TemporalMode = "single" | "composite" | "panel";
type FetchGroup = "landsat" | "nlcd" | "era5" | "capa" | "buildings" | "equity";

const FETCH_GROUPS: { id: FetchGroup; label: string; description: string }[] = [
  { id: "landsat",   label: "Landsat (LST + spectral indices)", description: "USGS STAC Landsat C2L2 — 13 indices + thermal" },
  { id: "nlcd",      label: "NLCD land cover", description: "MRLC WCS 2021 — impervious, canopy, land cover class" },
  { id: "era5",      label: "ERA5 background temperature", description: "Open-Meteo ERA5 reanalysis — 2m air temp baseline" },
  { id: "capa",      label: "CAPA air temperature", description: "NOAA CAPA via Open-Meteo — morning / midday / night AAT" },
  { id: "buildings", label: "Buildings + SVF", description: "Microsoft MLBuildings → OSHB → constant fallback" },
  { id: "equity",    label: "Equity layers", description: "HOLC redlining, CDC SVI, EPA EJScreen" },
];

const EMPTY_MANIFEST: CollectManifest = {
  variables: [],
  can_build: false,
  blocking_variables: [],
  boundary_description: null,
  temporal_description: null,
};

export default function DataCollectionPage() {
  const { notify } = useNotification();

  // Stepper state
  const [step, setStep] = useState<Step>(1);

  // Step 1 — Boundary
  const [boundary, setBoundary] = useState<BoundaryResponse | null>(null);

  // Step 2 — Temporal config
  const [dateStart, setDateStart] = useState("2022-06-01");
  const [dateEnd,   setDateEnd]   = useState("2022-08-31");
  const [cloudMax,  setCloudMax]  = useState(20);
  const [temporalMode, setTemporalMode] = useState<TemporalMode>("composite");

  // Step 3 — Fetch progress
  const [manifest, setManifest] = useState<CollectManifest>(EMPTY_MANIFEST);
  const [groupStatus, setGroupStatus] = useState<Record<FetchGroup, "idle" | "running" | "done" | "error">>({
    landsat: "idle", nlcd: "idle", era5: "idle",
    capa: "idle", buildings: "idle", equity: "idle",
  });

  // Step 4 — Map QA
  const [previewField, setPreviewField] = useState<string>("aat_residual");
  const [previewGeojson, setPreviewGeojson] = useState<GeoJsonData | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [inspectorCellId, setInspectorCellId] = useState<number | null>(null);
  const [inspectorData, setInspectorData] = useState<Record<string, unknown> | null>(null);
  const [inspectorLoading, setInspectorLoading] = useState(false);

  // Step 5 — Build
  const [building, setBuilding] = useState(false);
  const [buildResult, setBuildResult] = useState<{ geoparquet_path: string; n_cells: number } | null>(null);
  const [outputDir, setOutputDir] = useState<string>("");
  const [projectYml, setProjectYml] = useState<string>("");

  // Load project config to pre-populate output_dir and project.yml path
  useEffect(() => {
    getConfig().then((cfg: any) => {
      if (cfg?.paths?.output_dir) setOutputDir(cfg.paths.output_dir as string);
    }).catch(() => { /* not fatal */ });
  }, []);

  // -------------------------------------------------------------------------
  // Step 1 callbacks
  // -------------------------------------------------------------------------
  function handleBoundaryChange(result: BoundaryResponse) {
    setBoundary(result);
    notify("success", `Boundary resolved: ${result.place_name ?? result.source}`);
  }

  // -------------------------------------------------------------------------
  // Step 3 — individual group fetch
  // -------------------------------------------------------------------------
  async function fetchGroup(groupId: FetchGroup) {
    if (!boundary) return;
    setGroupStatus((s) => ({ ...s, [groupId]: "running" }));
    try {
      const resp: FetchGroupResponse = await collectFetch({
        group: groupId,
        config: {
          date_start: dateStart,
          date_end: dateEnd,
          cloud_cover_max: cloudMax,
          temporal_mode: temporalMode,
        },
      });
      setManifest(resp.manifest);
      setGroupStatus((s) => ({ ...s, [groupId]: "done" }));
    } catch (err) {
      setGroupStatus((s) => ({ ...s, [groupId]: "error" }));
      notify("error", `${groupId} fetch failed: ${err instanceof Error ? err.message : err}`);
      // Refresh manifest to show ERROR status
      try {
        const m = await collectManifest();
        setManifest(m);
      } catch { /* ignore */ }
    }
  }

  async function fetchAll() {
    for (const g of FETCH_GROUPS) {
      await fetchGroup(g.id);
    }
  }

  // -------------------------------------------------------------------------
  // Step 4 — map preview
  // -------------------------------------------------------------------------
  const loadPreview = useCallback(async (field: string) => {
    setPreviewLoading(true);
    try {
      const gj = await collectPreview(field);
      setPreviewGeojson(gj);
    } catch (err) {
      notify("error", `Preview failed: ${err instanceof Error ? err.message : err}`);
    } finally {
      setPreviewLoading(false);
    }
  }, [notify]);

  useEffect(() => {
    if (step === 4) {
      loadPreview(previewField);
    }
  }, [step, previewField, loadPreview]);

  async function handleCellClick(properties: Record<string, unknown>) {
    const cellId = properties["cell_id"];
    if (cellId == null) return;
    setInspectorCellId(Number(cellId));
    setInspectorData(null);
    setInspectorLoading(true);
    try {
      const data = await collectCellInspect(Number(cellId));
      setInspectorData(data);
    } catch {
      notify("error", "Failed to load cell data");
    } finally {
      setInspectorLoading(false);
    }
  }

  // -------------------------------------------------------------------------
  // Step 5 — build
  // -------------------------------------------------------------------------
  async function handleBuild() {
    if (!outputDir) {
      notify("error", "Set an output directory first");
      return;
    }
    setBuilding(true);
    try {
      const result = await collectBuild({
        output_dir: outputDir,
        project_yml: projectYml || undefined,
        temporal_mode: temporalMode,
      });
      setBuildResult({ geoparquet_path: result.geoparquet_path, n_cells: result.n_cells });
      notify("success", "GeoParquet written — dataset ready for pipeline");
    } catch (err) {
      notify("error", `Build failed: ${err instanceof Error ? err.message : err}`);
    } finally {
      setBuilding(false);
    }
  }

  // -------------------------------------------------------------------------
  // Render helpers
  // -------------------------------------------------------------------------
  const canAdvance = (s: Step) => {
    if (s === 1) return boundary !== null;
    if (s === 2) return !!dateStart && !!dateEnd;
    if (s === 3) return manifest.can_build;
    if (s === 4) return true;
    return false;
  };

  const stepLabel = (n: Step, label: string) => (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        cursor: step === n ? "default" : "pointer",
        opacity: step < n ? 0.5 : 1,
      }}
      onClick={() => n < step && setStep(n)}
    >
      <div style={{
        width: 28, height: 28, borderRadius: "50%",
        background: step > n ? "var(--sparc-purple)" : step === n ? "var(--sparc-purple)" : "var(--border)",
        color: step >= n ? "#fff" : "var(--ink)",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontWeight: 700, fontSize: 13, flexShrink: 0,
      }}>
        {step > n ? "✓" : n}
      </div>
      <span style={{ fontWeight: step === n ? 700 : 400, fontSize: 14 }}>{label}</span>
    </div>
  );

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: 24 }}>
      <SectionHeader kicker="SPARC" label="Data Collection" />

      {/* Stepper nav */}
      <div style={{
        display: "flex", gap: 32, marginBottom: 24, padding: "14px 20px",
        background: "var(--surface-raised, #f9f9f9)",
        border: "1px solid var(--border)", borderRadius: 8,
        flexWrap: "wrap",
      }}>
        {stepLabel(1, "Study Boundary")}
        {stepLabel(2, "Temporal Config")}
        {stepLabel(3, "Fetch Variables")}
        {stepLabel(4, "Confirm Quality")}
        {stepLabel(5, "Export Dataset")}
      </div>

      {/* Step panels */}
      {step === 1 && (
        <Card title="Define Study Boundary">
          <BoundarySelector onBoundaryChange={handleBoundaryChange} />
          {boundary && (
            <div style={{ marginTop: 16 }}>
              <StatGrid>
                <Stat label="Source" value={boundary.source} />
                <Stat
                  label="Bounding Box"
                  value={boundary.bbox.map((v) => v.toFixed(3)).join(", ")}
                />
              </StatGrid>
              <div style={{ marginTop: 16, height: 260 }}>
                <SpatialMap
                  geojson={boundary.geojson}
                  mode="choropleth"
                  height="260px"
                />
              </div>
            </div>
          )}
          <div style={{ marginTop: 16, display: "flex", justifyContent: "flex-end" }}>
            <Btn primary onClick={() => setStep(2)} disabled={!canAdvance(1)}>
              Next →
            </Btn>
          </div>
        </Card>
      )}

      {step === 2 && (
        <Card title="Temporal Configuration">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <FieldRow label="Start date">
              <input type="date" value={dateStart} onChange={(e) => setDateStart(e.target.value)} style={INPUT} />
            </FieldRow>
            <FieldRow label="End date">
              <input type="date" value={dateEnd} onChange={(e) => setDateEnd(e.target.value)} style={INPUT} />
            </FieldRow>
            <FieldRow label="Max cloud cover (%)">
              <input type="number" value={cloudMax} min={0} max={100}
                onChange={(e) => setCloudMax(Number(e.target.value))} style={INPUT} />
            </FieldRow>
            <FieldRow label="Temporal mode">
              <select value={temporalMode} onChange={(e) => setTemporalMode(e.target.value as TemporalMode)} style={INPUT}>
                <option value="composite">Composite (pixel-wise median)</option>
                <option value="single">Single (lowest cloud cover)</option>
                <option value="panel">Panel (multi-date GeoDataFrame)</option>
              </select>
            </FieldRow>
          </div>
          <p style={{ marginTop: 12, fontSize: 12, color: "var(--ink-muted)" }}>
            <strong>Composite</strong>: median over all valid pixels in the window —
            best for UHI seasonal analysis. <strong>Single</strong>: one cleanest scene.{" "}
            <strong>Panel</strong>: time-series rows (one per scene date).
          </p>
          <div style={{ marginTop: 20, display: "flex", justifyContent: "space-between" }}>
            <Btn onClick={() => setStep(1)}>← Back</Btn>
            <Btn primary onClick={() => setStep(3)} disabled={!canAdvance(2)}>Next →</Btn>
          </div>
        </Card>
      )}

      {step === 3 && (
        <Card title="Fetch Variables">
          <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 12 }}>
            <Btn primary onClick={fetchAll}>Fetch All Groups</Btn>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 20 }}>
            {FETCH_GROUPS.map((g) => {
              const status = groupStatus[g.id];
              return (
                <div key={g.id} style={{
                  display: "flex", alignItems: "center", gap: 12,
                  padding: "10px 14px",
                  border: "1px solid var(--border)", borderRadius: 6,
                  background: status === "done"  ? "#d4edda"
                             : status === "error" ? "#f8d7da"
                             : status === "running" ? "#fff3cd"
                             : "var(--surface)",
                }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: 600, fontSize: 13 }}>{g.label}</div>
                    <div style={{ fontSize: 11, color: "var(--ink-muted)" }}>{g.description}</div>
                  </div>
                  <span style={{ fontSize: 12, width: 80, textAlign: "center" }}>
                    {status === "running" ? "⏳ Running…"
                     : status === "done"   ? "✅ Done"
                     : status === "error"  ? "❌ Error"
                     : "—"}
                  </span>
                  <Btn
                    small
                    onClick={() => fetchGroup(g.id)}
                    disabled={status === "running" || !boundary}
                  >
                    {status === "done" || status === "error" ? "Re-fetch" : "Fetch"}
                  </Btn>
                </div>
              );
            })}
          </div>

          {manifest.variables.length > 0 && (
            <>
              <h4 style={{ marginBottom: 10 }}>Variable Manifest</h4>
              <VariableManifestTable manifest={manifest} />
            </>
          )}

          <div style={{ marginTop: 20, display: "flex", justifyContent: "space-between" }}>
            <Btn onClick={() => setStep(2)}>← Back</Btn>
            <Btn primary onClick={() => setStep(4)} disabled={!canAdvance(3)}>
              Confirm Quality →
            </Btn>
          </div>
        </Card>
      )}

      {step === 4 && (
        <Card title="Confirm Data Quality">
          {/* Variable picker */}
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
            <label style={{ fontSize: 13, fontWeight: 600 }}>Preview variable:</label>
            <select
              value={previewField}
              onChange={(e) => setPreviewField(e.target.value)}
              style={INPUT}
            >
              {manifest.variables
                .filter((v) => v.status === "COMPLETE" || v.status === "WARNING")
                .map((v) => (
                  <option key={v.name} value={v.name}>{v.name}</option>
                ))}
            </select>
            <Btn small onClick={() => loadPreview(previewField)} disabled={previewLoading}>
              {previewLoading ? "Loading…" : "Refresh"}
            </Btn>
          </div>

          {/* Map + cell inspector */}
          <div style={{ position: "relative", height: 420, borderRadius: 8, overflow: "hidden" }}>
            <SpatialMap
              geojson={previewGeojson}
              colorField={previewField}
              dimField="has_gap"
              mode="choropleth"
              height="420px"
              onFeatureClick={handleCellClick}
            />
            <CellInspector
              cellId={inspectorCellId}
              data={inspectorData}
              loading={inspectorLoading}
              onClose={() => setInspectorCellId(null)}
            />
          </div>

          {manifest.variables.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <VariableManifestTable manifest={manifest} />
            </div>
          )}

          <div style={{ marginTop: 20, display: "flex", justifyContent: "space-between" }}>
            <Btn onClick={() => setStep(3)}>← Back</Btn>
            <Btn primary onClick={() => setStep(5)}>Export Dataset →</Btn>
          </div>
        </Card>
      )}

      {step === 5 && (
        <Card title="Export Dataset">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
            <FieldRow label="Output directory">
              <input
                type="text"
                value={outputDir}
                onChange={(e) => setOutputDir(e.target.value)}
                placeholder="e.g. output/"
                style={INPUT}
              />
            </FieldRow>
            <FieldRow label="project.yml path (optional)">
              <input
                type="text"
                value={projectYml}
                onChange={(e) => setProjectYml(e.target.value)}
                placeholder="project.yml"
                style={INPUT}
              />
            </FieldRow>
          </div>

          <div style={{ marginBottom: 12, padding: "10px 14px", background: "var(--surface-raised)", borderRadius: 6, fontSize: 13 }}>
            <div><strong>Temporal mode:</strong> {temporalMode}</div>
            <div><strong>Output file:</strong> dataset_{temporalMode}.parquet</div>
            <div><strong>Target column:</strong> aat_residual</div>
            {projectYml && <div><strong>project.yml:</strong> data.file_path and data.target_column will be updated</div>}
          </div>

          {buildResult && (
            <div style={{ marginBottom: 16, padding: "12px 14px", background: "#d4edda", borderRadius: 6 }}>
              <strong>Dataset written</strong>
              <div style={{ fontSize: 12, marginTop: 4 }}>
                {buildResult.geoparquet_path} — {buildResult.n_cells.toLocaleString()} cells
              </div>
            </div>
          )}

          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <Btn onClick={() => setStep(4)}>← Back</Btn>
            <Btn primary onClick={handleBuild} disabled={building || !outputDir}>
              {building ? "Building…" : "Build GeoParquet"}
            </Btn>
          </div>
        </Card>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tiny layout helper
// ---------------------------------------------------------------------------
function FieldRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label style={{ display: "block", fontSize: 12, fontWeight: 600, marginBottom: 4, color: "var(--ink-muted)" }}>
        {label}
      </label>
      {children}
    </div>
  );
}

const INPUT: React.CSSProperties = {
  width: "100%",
  padding: "8px 10px",
  border: "1px solid var(--border)",
  borderRadius: 6,
  fontSize: 13,
  background: "var(--surface)",
  color: "var(--ink)",
};
