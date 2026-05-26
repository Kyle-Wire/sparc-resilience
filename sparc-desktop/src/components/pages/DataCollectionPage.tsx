import { useState, useCallback, useEffect, useRef } from "react";
import { SectionHeader, Card, Btn, StatGrid, Stat } from "@/components/ui/DesignSystem";
import { useNotification } from "@/hooks/useNotifications";
import {
  collectManifest,
  collectFetch,
  collectPreview,
  collectCellInspect,
  collectBuild,
  collectCapaEvents,
  collectSaveConfig,
  getConfig,
} from "@/lib/api";
import type {
  BoundaryResponse,
  CollectManifest,
  FetchGroupResponse,
  ContextLayer,
  CapaEvent,
  CapaEventsResponse,
} from "@/lib/api";
import type { GeoJsonData } from "@/lib/types";
import SpatialMap from "@/components/map/SpatialMap";
import BoundarySelector from "@/components/collect/BoundarySelector";
import VariableManifestTable from "@/components/collect/VariableManifestTable";
import CellInspector from "@/components/collect/CellInspector";


// Default basemap tile layer — CartoDB Positron (free, no API key required)
const DEFAULT_BASEMAP: { layer: ContextLayer; opacity: number } = {
  layer: {
    id: "carto-positron",
    name: "CartoDB Positron",
    kind: "basemap",
    url_template: "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
    attribution: "© OpenStreetMap contributors, © CARTO",
    description: "Light basemap",
    domains: [],
    default_for: [],
    opacity_default: 1.0,
    tile_size: 256,
    max_zoom: 19,
    available: true,
  },
  opacity: 1.0,
};

type Step = 1 | 2 | 3 | 4 | 5 | 6;
type TemporalMode = "single" | "composite" | "panel";
type FetchGroup =
  | "landsat"
  | "nlcd"
  | "era5"
  | "capa"
  | "buildings"
  | "equity"
  | "sentinel2"
  | "terrain";

// ---------------------------------------------------------------------------
// Landsat spectral indices available to users
// ---------------------------------------------------------------------------
const LANDSAT_INDICES: { id: string; label: string; defaultOn: boolean }[] = [
  { id: "ndvi",    label: "NDVI — vegetation greenness",  defaultOn: true },
  { id: "lst",     label: "LST — land surface temperature", defaultOn: true },
  { id: "ndbi",    label: "NDBI — built-up intensity",    defaultOn: true },
  { id: "albedo",  label: "Albedo — surface reflectance",  defaultOn: true },
  { id: "mndwi",   label: "MNDWI — water",               defaultOn: false },
  { id: "ndwi",    label: "NDWI — water stress",          defaultOn: false },
  { id: "savi",    label: "SAVI — soil-adjusted vegetation", defaultOn: false },
  { id: "evi",     label: "EVI — enhanced vegetation",    defaultOn: false },
  { id: "ndbai",   label: "NDBAI — bareness",             defaultOn: false },
  { id: "nbr",     label: "NBR — burn ratio",             defaultOn: false },
  { id: "ndmi",    label: "NDMI — moisture",              defaultOn: false },
  { id: "ui",      label: "UI — urban index",             defaultOn: false },
  { id: "bsi",     label: "BSI — bare soil",              defaultOn: false },
];

// Terrain variables
const TERRAIN_VARS: { id: string; label: string; defaultOn: boolean }[] = [
  { id: "elevation_m",      label: "Elevation (m NAVD88/EGM2008)", defaultOn: true },
  { id: "slope_deg",        label: "Slope (degrees)",              defaultOn: true },
  { id: "aspect_deg",       label: "Aspect (degrees from North)",  defaultOn: false },
  { id: "distance_water_m", label: "Distance to water (m)",        defaultOn: true },
];

// Per-group variable names for manifest-checking
const GROUP_VARIABLES: Record<string, string[]> = {
  landsat:   ["lst", "ndvi", "ndbi", "albedo"],
  nlcd:      ["pct_impervious", "pct_canopy", "land_cover"],
  era5:      ["era5_t2m"],
  capa:      ["aat_morning", "aat_midday", "aat_night", "diurnal_aat"],
  buildings: ["bldg_height_mean", "bldg_coverage", "svf"],
  equity:    ["holc_grade", "cdc_svi", "ejscreen_score"],
  sentinel2: ["ndvi_s2", "ndbi_s2", "mndwi_s2"],
  terrain:   ["elevation_m", "slope_deg", "aspect_deg", "distance_water_m"],
};

// Smart aggregation defaults per variable
const AGG_DEFAULTS: Record<string, string> = {
  lst: "max", ndvi: "mean", ndbi: "mean", albedo: "mean",
  mndwi: "mean", ndwi: "mean", savi: "mean", evi: "mean",
  ndbai: "mean", nbr: "mean", ndmi: "mean", ui: "mean", bsi: "mean",
  pct_impervious: "mean", pct_canopy: "mean", land_cover: "pct_cover",
  era5_t2m: "mean",
  aat_morning: "mean", aat_midday: "mean", aat_night: "mean", diurnal_aat: "mean",
  bldg_height_mean: "mean", bldg_coverage: "mean", svf: "mean",
  holc_grade: "pct_cover", cdc_svi: "mean", ejscreen_score: "mean",
  ndvi_s2: "mean", ndbi_s2: "mean", mndwi_s2: "mean",
  elevation_m: "mean", slope_deg: "mean", aspect_deg: "mean", distance_water_m: "mean",
};

interface SourceConfig {
  enabled: boolean;
  variables: string[];
  usOnly?: boolean;
}

const DEFAULT_SOURCES: Record<FetchGroup, SourceConfig> = {
  capa:      { enabled: true,  variables: ["aat_morning", "aat_midday", "aat_night"], usOnly: true },
  landsat:   { enabled: true,  variables: LANDSAT_INDICES.filter(i => i.defaultOn).map(i => i.id) },
  sentinel2: { enabled: false, variables: ["ndvi_s2", "ndbi_s2", "mndwi_s2"] },
  nlcd:      { enabled: true,  variables: ["pct_impervious", "pct_canopy", "land_cover"], usOnly: true },
  era5:      { enabled: true,  variables: ["era5_t2m"] },
  buildings: { enabled: false, variables: ["bldg_height_mean", "bldg_coverage"] },
  equity:    { enabled: false, variables: ["cdc_svi", "ejscreen_score"], usOnly: true },
  terrain:   { enabled: false, variables: TERRAIN_VARS.filter(v => v.defaultOn).map(v => v.id) },
};

const EMPTY_MANIFEST: CollectManifest = {
  variables: [],
  can_build: false,
  blocking_variables: [],
  boundary_description: null,
  temporal_description: null,
};

// ---------------------------------------------------------------------------
// WebSocket hook for collect progress events
// ---------------------------------------------------------------------------
function useCollectStream(onEvent: (evt: Record<string, unknown>) => void) {
  const wsRef = useRef<WebSocket | null>(null);
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  const connect = useCallback(() => {
    if (wsRef.current) return;
    const token = (window as any).__SPARC_TOKEN__ ?? "";
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    // In Tauri the sidecar binds to 127.0.0.1:8008
    const base = (window as any).__SPARC_BASE__ ?? "127.0.0.1:8008";
    const url = `${protocol}//${base}/collect/stream?token=${encodeURIComponent(token)}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;
    ws.onopen = () => ws.send(JSON.stringify({ subscribe: "collect" }));
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data as string) as Record<string, unknown>;
        onEventRef.current(data);
      } catch { /* ignore */ }
    };
    ws.onclose = () => { wsRef.current = null; };
    ws.onerror = () => { ws.close(); };
  }, []);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  return { connect, disconnect };
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function DataCollectionPage() {
  const { notify } = useNotification();

  // Stepper
  const [step, setStep] = useState<Step>(1);

  // Step 1 — Boundary
  const [boundary, setBoundary] = useState<BoundaryResponse | null>(null);
  const [cityName, setCityName] = useState("");

  // Step 2 — Config + CAPA event
  const [dateStart, setDateStart] = useState("2022-06-01");
  const [dateEnd,   setDateEnd]   = useState("2022-08-31");
  const [cloudMax,  setCloudMax]  = useState(20);
  const [temporalMode, setTemporalMode] = useState<TemporalMode>("composite");
  const [capaLoading, setCapaLoading] = useState(false);
  const [capaResponse, setCapaResponse] = useState<CapaEventsResponse | null>(null);
  const [selectedCapaEvent, setSelectedCapaEvent] = useState<CapaEvent | null>(null);

  // Step 3 — Variable selection
  const [sources, setSources] = useState<Record<FetchGroup, SourceConfig>>(DEFAULT_SOURCES);
  const [groupStatus, setGroupStatus] = useState<Record<string, "idle" | "running" | "done" | "warn" | "error">>({});
  const [manifest, setManifest] = useState<CollectManifest>(EMPTY_MANIFEST);
  const [progressMessages, setProgressMessages] = useState<Record<string, string>>({});

  // Step 4 — QA map
  const [previewField, setPreviewField] = useState<string>("aat_morning");
  const [previewGeojson, setPreviewGeojson] = useState<GeoJsonData | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [inspectorCellId, setInspectorCellId] = useState<number | null>(null);
  const [inspectorData, setInspectorData] = useState<Record<string, unknown> | null>(null);
  const [inspectorLoading, setInspectorLoading] = useState(false);

  // Step 5 — Spatial statistics
  const [fishnetM, setFishnetM] = useState(30);
  const [aggregation, setAggregation] = useState<Record<string, string>>({});

  // Step 6 — Export
  const [outputDir, setOutputDir] = useState<string>("");
  const [building, setBuilding] = useState(false);
  const [buildResult, setBuildResult] = useState<{ geoparquet_path: string; n_cells: number } | null>(null);

  // WebSocket collect stream
  const { connect: wsConnect, disconnect: wsDisconnect } = useCollectStream((evt) => {
    if (evt.type === "collect_progress") {
      const group = String(evt.group ?? "");
      const status = String(evt.status ?? "");
      const message = String(evt.message ?? "");
      if (group) {
        setProgressMessages(prev => ({ ...prev, [group]: message }));
        if (status === "complete") {
          setGroupStatus(prev => ({ ...prev, [group]: "done" }));
        } else if (status === "error") {
          setGroupStatus(prev => ({ ...prev, [group]: "error" }));
        } else if (status === "running") {
          setGroupStatus(prev => ({ ...prev, [group]: "running" }));
        }
      }
    }
  });

  // Pre-populate output dir from project config
  useEffect(() => {
    getConfig().then((cfg: any) => {
      if (cfg?.paths?.output_dir) setOutputDir(cfg.paths.output_dir as string);
    }).catch(() => {});
  }, []);

  // Auto-populate aggregation defaults when sources change
  useEffect(() => {
    const agg: Record<string, string> = {};
    Object.entries(sources).forEach(([, src]) => {
      if (src.enabled) {
        src.variables.forEach(v => {
          if (!agg[v]) agg[v] = AGG_DEFAULTS[v] ?? "mean";
        });
      }
    });
    setAggregation(agg);
  }, [sources]);

  // ---------------------------------------------------------------------------
  // Step 1 callbacks
  // ---------------------------------------------------------------------------
  function handleBoundaryChange(result: BoundaryResponse) {
    setBoundary(result);
    notify("success", `Boundary resolved: ${result.place_name ?? result.source}`);
  }

  // ---------------------------------------------------------------------------
  // Step 2 — CAPA event lookup
  // ---------------------------------------------------------------------------
  async function fetchCapaEvents() {
    const city = cityName.trim() || boundary?.place_name?.split(",")[0] || "";
    if (!city) {
      notify("error", "Enter a city name to look up CAPA events");
      return;
    }
    setCapaLoading(true);
    setCapaResponse(null);
    try {
      const resp = await collectCapaEvents(city);
      setCapaResponse(resp);
      if (resp.events.length > 0) {
        setSelectedCapaEvent(resp.events[0]);
      }
      if (resp.error && resp.events.length === 0) {
        notify("warn", `CAPA: ${resp.error}`);
      }
    } catch (err) {
      notify("error", `CAPA lookup failed: ${err instanceof Error ? err.message : err}`);
    } finally {
      setCapaLoading(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Step 3 — fetch a single group
  // ---------------------------------------------------------------------------
  async function fetchGroup(groupId: FetchGroup) {
    if (!boundary) return;
    setGroupStatus(prev => ({ ...prev, [groupId]: "running" }));
    try {
      const src = sources[groupId];
      const resp: FetchGroupResponse = await collectFetch({
        group: groupId === "terrain" ? "terrain" : groupId,
        config: {
          date_start: dateStart,
          date_end: dateEnd,
          cloud_cover_max: cloudMax,
          temporal_mode: temporalMode,
          enabled_indices: groupId === "landsat" ? src.variables : undefined,
          // Pass CAPA node and event date for CAPA group
          ...(groupId === "capa" && selectedCapaEvent ? {
            osf_node: selectedCapaEvent.osf_node,
            event_date: selectedCapaEvent.date,
          } : {}),
          // Pass terrain variables for terrain group
          ...(groupId === "terrain" ? { enabled: src.variables } : {}),
        },
      });
      setManifest(resp.manifest);
      const groupVars = GROUP_VARIABLES[groupId] ?? [];
      const allSkipped = groupVars.length > 0 && groupVars.every((v) => {
        const entry = resp.manifest.variables.find(m => m.name === v);
        return entry == null || entry.status === "SKIPPED";
      });
      setGroupStatus(prev => ({ ...prev, [groupId]: allSkipped ? "warn" : "done" }));
    } catch (err) {
      setGroupStatus(prev => ({ ...prev, [groupId]: "error" }));
      notify("error", `${groupId} fetch failed: ${err instanceof Error ? err.message : err}`);
      try { const m = await collectManifest(); setManifest(m); } catch { /* ignore */ }
    }
  }

  async function fetchAllEnabled() {
    wsConnect();
    const enabled = (Object.keys(sources) as FetchGroup[]).filter(g => sources[g].enabled);
    for (const g of enabled) {
      await fetchGroup(g);
    }
    wsDisconnect();
  }

  // ---------------------------------------------------------------------------
  // Step 4 — map preview
  // ---------------------------------------------------------------------------
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
    if (step === 4) loadPreview(previewField);
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
    } catch { notify("error", "Failed to load cell data"); }
    finally { setInspectorLoading(false); }
  }

  // ---------------------------------------------------------------------------
  // Step 6 — build / export
  // ---------------------------------------------------------------------------
  async function handleBuild() {
    if (!outputDir) { notify("error", "Set an output directory first"); return; }
    setBuilding(true);
    try {
      // Save wizard config to project.yml before building
      const enabledVars: Record<string, { enabled: string[] }> = {};
      (Object.keys(sources) as FetchGroup[]).forEach(g => {
        if (sources[g].enabled) enabledVars[g] = { enabled: sources[g].variables };
      });
      await collectSaveConfig({
        city_name: cityName || boundary?.place_name || "",
        capa_event_date: selectedCapaEvent?.date ?? null,
        capa_osf_node: selectedCapaEvent?.osf_node ?? capaResponse?.osf_node ?? "",
        variables: enabledVars,
        fishnet_m: fishnetM,
        aggregation,
      });

      const result = await collectBuild({
        output_dir: outputDir,
        temporal_mode: temporalMode,
      });
      setBuildResult({ geoparquet_path: result.geoparquet_path, n_cells: result.n_cells });
      notify("success", "GeoParquet written — dataset ready for pipeline");
    } catch (err) {
      notify("error", `Export failed: ${err instanceof Error ? err.message : err}`);
    } finally {
      setBuilding(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------
  const allEnabledFetched = (Object.keys(sources) as FetchGroup[])
    .filter(g => sources[g].enabled)
    .every(g => groupStatus[g] === "done" || groupStatus[g] === "warn");

  const capaFetched = groupStatus["capa"] === "done";
  const capaBlocking = sources.capa.enabled && !capaFetched;

  function toggleSource(g: FetchGroup) {
    setSources(prev => ({ ...prev, [g]: { ...prev[g], enabled: !prev[g].enabled } }));
  }

  function toggleVar(g: FetchGroup, v: string) {
    setSources(prev => {
      const current = prev[g].variables;
      const next = current.includes(v) ? current.filter(x => x !== v) : [...current, v];
      return { ...prev, [g]: { ...prev[g], variables: next } };
    });
  }

  const isUs = boundary ? true : true; // will be determined after boundary resolved

  const canAdvance = (s: Step) => {
    if (s === 1) return boundary !== null;
    if (s === 2) return !!dateStart && !!dateEnd && (selectedCapaEvent !== null || !sources.capa.enabled);
    if (s === 3) return allEnabledFetched && !capaBlocking;
    if (s === 4) return true;
    if (s === 5) return true;
    return false;
  };

  const stepLabel = (n: Step, label: string) => (
    <div
      style={{
        display: "flex", alignItems: "center", gap: 10,
        cursor: n < step ? "pointer" : "default",
        opacity: step < n ? 0.45 : 1,
      }}
      onClick={() => { if (n < step) setStep(n); }}
    >
      <div style={{
        width: 28, height: 28, borderRadius: "50%",
        background: step > n ? "var(--sparc-purple, #6c3fc5)" : step === n ? "var(--sparc-purple, #6c3fc5)" : "var(--border, #ddd)",
        color: step >= n ? "#fff" : "var(--ink)",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontWeight: 700, fontSize: 13, flexShrink: 0,
      }}>
        {step > n ? "✓" : n}
      </div>
      <span style={{ fontWeight: step === n ? 700 : 400, fontSize: 14 }}>{label}</span>
    </div>
  );

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: 24 }}>
      <SectionHeader kicker="SPARC" label="Data Collection" />

      {/* Stepper nav */}
      <div style={{
        display: "flex", gap: 24, marginBottom: 24, padding: "14px 20px",
        background: "var(--surface-raised, #f9f9f9)",
        border: "1px solid var(--border)", borderRadius: 8, flexWrap: "wrap",
      }}>
        {stepLabel(1, "Study Boundary")}
        {stepLabel(2, "Configuration")}
        {stepLabel(3, "Variables")}
        {stepLabel(4, "Quality Check")}
        {stepLabel(5, "Statistics")}
        {stepLabel(6, "Export")}
      </div>

      {/* ── STEP 1: Study Boundary ── */}
      {step === 1 && (
        <Card title="Define Study Boundary">
          <div style={{ marginBottom: 16 }}>
            <FieldRow label="City name (used to look up CAPA campaigns in Step 2)">
              <input
                type="text" value={cityName}
                onChange={e => setCityName(e.target.value)}
                placeholder="e.g. Phoenix, AZ"
                style={INPUT}
              />
            </FieldRow>
          </div>
          <BoundarySelector onBoundaryChange={handleBoundaryChange} />
          {boundary && (
            <div style={{ marginTop: 16 }}>
              <StatGrid>
                <Stat label="Source" value={boundary.source} />
                <Stat label="Bounding Box" value={boundary.bbox.map(v => v.toFixed(3)).join(", ")} />
              </StatGrid>
              <div style={{ marginTop: 16, height: 260 }}>
                <SpatialMap
                  geojson={boundary.geojson}
                  mode="choropleth"
                  height="260px"
                  contextLayers={[DEFAULT_BASEMAP]}
                />
              </div>
            </div>
          )}
          <div style={{ marginTop: 16, display: "flex", justifyContent: "flex-end" }}>
            <Btn primary onClick={() => setStep(2)} disabled={!canAdvance(1)}>Next →</Btn>
          </div>
        </Card>
      )}

      {/* ── STEP 2: Configuration + CAPA event ── */}
      {step === 2 && (
        <Card title="Configuration">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
            <FieldRow label="Date range start">
              <input type="date" value={dateStart} onChange={e => setDateStart(e.target.value)} style={INPUT} />
            </FieldRow>
            <FieldRow label="Date range end">
              <input type="date" value={dateEnd} onChange={e => setDateEnd(e.target.value)} style={INPUT} />
            </FieldRow>
            <FieldRow label="Max cloud cover (%)">
              <input type="number" value={cloudMax} min={0} max={100}
                onChange={e => setCloudMax(Number(e.target.value))} style={INPUT} />
            </FieldRow>
            <FieldRow label="Temporal mode">
              <select value={temporalMode} onChange={e => setTemporalMode(e.target.value as TemporalMode)} style={INPUT}>
                <option value="composite">Composite (pixel-wise median)</option>
                <option value="single">Single (lowest cloud cover)</option>
                <option value="panel">Panel (multi-date rows)</option>
              </select>
            </FieldRow>
          </div>

          {/* CAPA event picker */}
          <div style={{ border: "1px solid var(--border)", borderRadius: 8, padding: 16, marginBottom: 16 }}>
            <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 12 }}>
              🌡️ CAPA Heat Watch Campaign
              <span style={{ fontWeight: 400, fontSize: 12, color: "var(--ink-muted)", marginLeft: 8 }}>
                Required anchor event for heat analysis
              </span>
            </div>
            <div style={{ display: "flex", gap: 8, alignItems: "flex-end", marginBottom: 12 }}>
              <div style={{ flex: 1 }}>
                <FieldRow label="City name for CAPA lookup">
                  <input
                    type="text"
                    value={cityName || boundary?.place_name?.split(",")[0] || ""}
                    onChange={e => setCityName(e.target.value)}
                    placeholder="e.g. Phoenix"
                    style={INPUT}
                  />
                </FieldRow>
              </div>
              <Btn onClick={fetchCapaEvents} disabled={capaLoading}>
                {capaLoading ? "Searching…" : "Find Events"}
              </Btn>
            </div>

            {capaResponse && (
              <div>
                {capaResponse.found_in_catalog ? (
                  <>
                    <div style={{ fontSize: 12, color: "#2d7d32", marginBottom: 8 }}>
                      ✅ Found: <strong>{capaResponse.canonical_name}</strong>
                      {capaResponse.us_city && " (US city — all sources available)"}
                    </div>
                    {capaResponse.events.length > 0 ? (
                      <div>
                        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>Select campaign event:</div>
                        {capaResponse.events.map((evt, i) => (
                          <div key={i}
                            onClick={() => setSelectedCapaEvent(evt)}
                            style={{
                              padding: "8px 12px", borderRadius: 6, cursor: "pointer", marginBottom: 4,
                              border: `2px solid ${selectedCapaEvent === evt ? "var(--sparc-purple, #6c3fc5)" : "var(--border)"}`,
                              background: selectedCapaEvent === evt ? "var(--sparc-purple-bg, #f0ebfc)" : "var(--surface)",
                            }}
                          >
                            <div style={{ fontWeight: 600, fontSize: 13 }}>{evt.label}</div>
                            {evt.date && <div style={{ fontSize: 11, color: "var(--ink-muted)" }}>Date: {evt.date}</div>}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div style={{ color: "#c62828", fontSize: 12 }}>
                        {capaResponse.error ?? "No campaign events found for this city."}
                      </div>
                    )}
                  </>
                ) : (
                  <div>
                    {capaResponse.suggestions.length > 0 && (
                      <div style={{ fontSize: 12, marginBottom: 8 }}>
                        Did you mean:
                        {capaResponse.suggestions.map(s => (
                          <span key={s}
                            onClick={() => { setCityName(s); setCapaResponse(null); }}
                            style={{ marginLeft: 6, color: "var(--sparc-purple, #6c3fc5)", cursor: "pointer", textDecoration: "underline" }}
                          >{s}</span>
                        ))}
                      </div>
                    )}
                    <div style={{ color: "#c62828", fontSize: 12 }}>
                      {capaResponse.error ?? "City not found in catalog."}
                    </div>
                    <div style={{ fontSize: 11, color: "var(--ink-muted)", marginTop: 6 }}>
                      You can continue without CAPA by disabling it in Step 3 (Variables), but CAPA is
                      required for heat analysis.
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          <div style={{ marginTop: 16, display: "flex", justifyContent: "space-between" }}>
            <Btn onClick={() => setStep(1)}>← Back</Btn>
            <Btn primary onClick={() => setStep(3)} disabled={!canAdvance(2)}>
              Choose Variables →
            </Btn>
          </div>
        </Card>
      )}

      {/* ── STEP 3: Variable selection (accordion per source) ── */}
      {step === 3 && (
        <Card title="Variable Collection">
          <p style={{ fontSize: 12, color: "var(--ink-muted)", marginBottom: 16 }}>
            CAPA + at least one spectral source are required. Other sources are optional.
            Non-US sources are shown for global studies; US-only sources are labeled.
          </p>

          <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
            <Btn primary onClick={fetchAllEnabled} disabled={Object.values(groupStatus).some(s => s === "running")}>
              Fetch All Enabled
            </Btn>
          </div>

          {/* CAPA (required) */}
          <SourceAccordion
            id="capa"
            label="🌡️ CAPA Heat Watch"
            description="Required — NOAA NIHHIS urban heat traverse (US only)"
            required
            usOnly
            isUs={isUs}
            enabled={sources.capa.enabled}
            onToggle={() => toggleSource("capa")}
            status={groupStatus["capa"]}
            progress={progressMessages["capa"]}
            onFetch={() => fetchGroup("capa")}
          >
            <div style={{ fontSize: 12, color: "var(--ink-muted)", marginTop: 8 }}>
              {selectedCapaEvent
                ? <><strong>Event:</strong> {selectedCapaEvent.label} {selectedCapaEvent.date ? `(${selectedCapaEvent.date})` : ""}</>
                : <span style={{ color: "#c62828" }}>⚠ No event selected — go back to Step 2 to choose a CAPA event</span>
              }
            </div>
          </SourceAccordion>

          {/* Landsat */}
          <SourceAccordion
            id="landsat"
            label="🛰 Landsat C2L2 — spectral indices + LST"
            description="USGS STAC · 30m · best for US urban heat studies"
            enabled={sources.landsat.enabled}
            onToggle={() => toggleSource("landsat")}
            status={groupStatus["landsat"]}
            progress={progressMessages["landsat"]}
            onFetch={() => fetchGroup("landsat")}
          >
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4, marginTop: 8 }}>
              {LANDSAT_INDICES.map(idx => (
                <label key={idx.id} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, cursor: "pointer" }}>
                  <input type="checkbox"
                    checked={sources.landsat.variables.includes(idx.id)}
                    onChange={() => toggleVar("landsat", idx.id)}
                  />
                  {idx.label}
                </label>
              ))}
            </div>
          </SourceAccordion>

          {/* Sentinel-2 */}
          <SourceAccordion
            id="sentinel2"
            label="🌍 Sentinel-2 L2A — global spectral indices"
            description="Planetary Computer · 10m · works outside the US"
            enabled={sources.sentinel2.enabled}
            onToggle={() => toggleSource("sentinel2")}
            status={groupStatus["sentinel2"]}
            progress={progressMessages["sentinel2"]}
            onFetch={() => fetchGroup("sentinel2")}
          >
            <div style={{ fontSize: 12, color: "var(--ink-muted)", marginTop: 6 }}>Variables: NDVI, NDBI, MNDWI</div>
          </SourceAccordion>

          {/* NLCD */}
          <SourceAccordion
            id="nlcd"
            label="🗺 NLCD Land Cover"
            description="MRLC WCS 2021 · impervious, canopy, land cover class"
            usOnly enabled={sources.nlcd.enabled}
            onToggle={() => toggleSource("nlcd")}
            status={groupStatus["nlcd"]}
            progress={progressMessages["nlcd"]}
            onFetch={() => fetchGroup("nlcd")}
          >
            <div style={{ fontSize: 12, color: "var(--ink-muted)", marginTop: 6 }}>Variables: pct_impervious, pct_canopy, land_cover</div>
          </SourceAccordion>

          {/* ERA5 */}
          <SourceAccordion
            id="era5"
            label="🌬 ERA5 Background Temperature"
            description="Open-Meteo · 2m air temp reanalysis (global)"
            enabled={sources.era5.enabled}
            onToggle={() => toggleSource("era5")}
            status={groupStatus["era5"]}
            progress={progressMessages["era5"]}
            onFetch={() => fetchGroup("era5")}
          >
            <div style={{ fontSize: 12, color: "var(--ink-muted)", marginTop: 6 }}>Variable: era5_t2m</div>
          </SourceAccordion>

          {/* Terrain */}
          <SourceAccordion
            id="terrain"
            label="⛰ Terrain (DEM)"
            description="USGS 3DEP (US) · Copernicus GLO-30 (global) · elevation, slope, aspect, distance to water"
            enabled={sources.terrain.enabled}
            onToggle={() => toggleSource("terrain")}
            status={groupStatus["terrain"]}
            progress={progressMessages["terrain"]}
            onFetch={() => fetchGroup("terrain")}
          >
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4, marginTop: 8 }}>
              {TERRAIN_VARS.map(v => (
                <label key={v.id} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, cursor: "pointer" }}>
                  <input type="checkbox"
                    checked={sources.terrain.variables.includes(v.id)}
                    onChange={() => toggleVar("terrain", v.id)}
                  />
                  {v.label}
                </label>
              ))}
            </div>
          </SourceAccordion>

          {/* Buildings */}
          <SourceAccordion
            id="buildings"
            label="🏙 Buildings + SVF"
            description="Microsoft MLBuildings → height, coverage, sky-view factor (global)"
            enabled={sources.buildings.enabled}
            onToggle={() => toggleSource("buildings")}
            status={groupStatus["buildings"]}
            progress={progressMessages["buildings"]}
            onFetch={() => fetchGroup("buildings")}
          >
            <div style={{ fontSize: 12, color: "var(--ink-muted)", marginTop: 6 }}>Variables: bldg_height_mean, bldg_coverage, svf</div>
          </SourceAccordion>

          {/* Equity */}
          <SourceAccordion
            id="equity"
            label="⚖️ Equity Layers"
            description="HOLC redlining · CDC SVI · EPA EJScreen (US only)"
            usOnly
            enabled={sources.equity.enabled}
            onToggle={() => toggleSource("equity")}
            status={groupStatus["equity"]}
            progress={progressMessages["equity"]}
            onFetch={() => fetchGroup("equity")}
          >
            <div style={{ fontSize: 12, color: "var(--ink-muted)", marginTop: 6 }}>Variables: holc_grade, cdc_svi, ejscreen_score</div>
          </SourceAccordion>

          {manifest.variables.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <VariableManifestTable manifest={manifest} />
            </div>
          )}

          {capaBlocking && (
            <div style={{ marginTop: 12, padding: "10px 14px", background: "#fdecea", borderRadius: 6, fontSize: 12, color: "#c62828" }}>
              ⛔ CAPA must be fetched successfully before proceeding. If CAPA is unavailable for this
              city, disable the CAPA source above (heat analysis quality will be reduced).
            </div>
          )}

          <div style={{ marginTop: 20, display: "flex", justifyContent: "space-between" }}>
            <Btn onClick={() => setStep(2)}>← Back</Btn>
            <Btn primary onClick={() => setStep(4)} disabled={!canAdvance(3)}>
              Confirm Quality →
            </Btn>
          </div>
        </Card>
      )}

      {/* ── STEP 4: Quality Confirmation ── */}
      {step === 4 && (
        <Card title="Quality Confirmation">
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
            <label style={{ fontSize: 13, fontWeight: 600 }}>Preview variable:</label>
            <select value={previewField} onChange={e => setPreviewField(e.target.value)} style={{ ...INPUT, maxWidth: 240 }}>
              {manifest.variables
                .filter(v => v.status === "COMPLETE" || v.status === "WARNING")
                .map(v => <option key={v.name} value={v.name}>{v.name}</option>)}
            </select>
            <Btn small onClick={() => loadPreview(previewField)} disabled={previewLoading}>
              {previewLoading ? "Loading…" : "Refresh"}
            </Btn>
          </div>

          <div style={{ position: "relative", height: 420, borderRadius: 8, overflow: "hidden" }}>
            <SpatialMap
              geojson={previewGeojson}
              colorField={previewField}
              dimField="has_gap"
              mode="choropleth"
              height="420px"
              onFeatureClick={handleCellClick}
              contextLayers={[DEFAULT_BASEMAP]}
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
              {/* Per-source Re-fetch buttons */}
              <div style={{ marginTop: 12, display: "flex", flexWrap: "wrap", gap: 8 }}>
                {(Object.keys(sources) as FetchGroup[]).filter(g => sources[g].enabled).map(g => (
                  <Btn key={g} small onClick={() => { setStep(3); }} disabled={groupStatus[g] === "running"}>
                    Re-fetch {g} →
                  </Btn>
                ))}
              </div>
            </div>
          )}

          <div style={{ marginTop: 20, display: "flex", justifyContent: "space-between" }}>
            <Btn onClick={() => setStep(3)}>← Back to Variables</Btn>
            <Btn primary onClick={() => setStep(5)}>Spatial Statistics →</Btn>
          </div>
        </Card>
      )}

      {/* ── STEP 5: Spatial Statistics ── */}
      {step === 5 && (
        <Card title="Spatial Statistics">
          <div style={{ marginBottom: 20 }}>
            <FieldRow label="Fishnet cell size (metres)">
              <input
                type="number" value={fishnetM} min={10} max={1000} step={10}
                onChange={e => setFishnetM(Number(e.target.value))}
                style={{ ...INPUT, maxWidth: 180 }}
              />
            </FieldRow>
            {boundary && (
              <div style={{ marginTop: 6, fontSize: 12, color: "var(--ink-muted)" }}>
                Estimated cell count:{" "}
                <strong>
                  ~{estimateCellCount(boundary.bbox, fishnetM).toLocaleString()}
                </strong>
                {" "}cells
              </div>
            )}
          </div>

          <div style={{ marginBottom: 8, fontWeight: 600, fontSize: 14 }}>
            Aggregation method per variable
          </div>
          <p style={{ fontSize: 12, color: "var(--ink-muted)", marginBottom: 12 }}>
            Defaults are set automatically (LST→max, land_cover→pct_cover, most others→mean).
          </p>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            {Object.keys(aggregation).map(v => (
              <div key={v} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13 }}>
                <span style={{ flex: 1, fontFamily: "monospace" }}>{v}</span>
                <select
                  value={aggregation[v] ?? "mean"}
                  onChange={e => setAggregation(prev => ({ ...prev, [v]: e.target.value }))}
                  style={{ ...INPUT, width: 130 }}
                >
                  <option value="mean">mean</option>
                  <option value="max">max</option>
                  <option value="min">min</option>
                  <option value="median">median</option>
                  <option value="sum">sum</option>
                  <option value="pct_cover">pct_cover</option>
                </select>
              </div>
            ))}
          </div>

          <div style={{ marginTop: 20, display: "flex", justifyContent: "space-between" }}>
            <Btn onClick={() => setStep(4)}>← Back</Btn>
            <Btn primary onClick={() => setStep(6)}>Export →</Btn>
          </div>
        </Card>
      )}

      {/* ── STEP 6: Export ── */}
      {step === 6 && (
        <Card title="Export Dataset">
          <div style={{ marginBottom: 16 }}>
            <FieldRow label="Output directory">
              <input
                type="text" value={outputDir}
                onChange={e => setOutputDir(e.target.value)}
                placeholder="e.g. output/"
                style={INPUT}
              />
            </FieldRow>
          </div>

          <div style={{ padding: "10px 14px", background: "var(--surface-raised)", borderRadius: 6, fontSize: 13, marginBottom: 16 }}>
            <div><strong>Temporal mode:</strong> {temporalMode}</div>
            <div><strong>Cell size:</strong> {fishnetM}m</div>
            <div><strong>Enabled sources:</strong> {(Object.keys(sources) as FetchGroup[]).filter(g => sources[g].enabled).join(", ")}</div>
            <div><strong>Outputs:</strong> dataset.parquet + dataset_manifest.json (auto-loaded into project)</div>
          </div>

          {buildResult && (
            <div style={{ marginBottom: 16, padding: "12px 14px", background: "#d4edda", borderRadius: 6 }}>
              <strong>✅ Dataset written</strong>
              <div style={{ fontSize: 12, marginTop: 4 }}>
                {buildResult.geoparquet_path} — {buildResult.n_cells.toLocaleString()} cells
              </div>
              <div style={{ fontSize: 11, color: "#2d7d32", marginTop: 4 }}>
                collection: block written to project.yml · dataset available in My Data
              </div>
            </div>
          )}

          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <Btn onClick={() => setStep(5)}>← Back</Btn>
            <Btn primary onClick={handleBuild} disabled={building || !outputDir}>
              {building ? "Exporting…" : "Export GeoParquet"}
            </Btn>
          </div>
        </Card>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// SourceAccordion — collapsible source card
// ---------------------------------------------------------------------------
function SourceAccordion({
  id, label, description, required, usOnly, isUs,
  enabled, onToggle, status, progress, onFetch, children,
}: {
  id: string;
  label: string;
  description: string;
  required?: boolean;
  usOnly?: boolean;
  isUs?: boolean;
  enabled: boolean;
  onToggle: () => void;
  status?: string;
  progress?: string;
  onFetch: () => void;
  children?: React.ReactNode;
}) {
  const [open, setOpen] = useState(required || enabled);

  const statusColor =
    status === "done"  ? "#d4edda"
    : status === "warn"  ? "#fff3cd"
    : status === "error" ? "#fdecea"
    : status === "running" ? "#e8f4fe"
    : "var(--surface)";

  return (
    <div style={{
      border: "1px solid var(--border)", borderRadius: 8, marginBottom: 8,
      background: statusColor, overflow: "hidden",
    }}>
      <div style={{
        display: "flex", alignItems: "center", gap: 10, padding: "10px 14px",
        cursor: "pointer",
      }}>
        <input type="checkbox" checked={enabled} onChange={onToggle}
          onClick={e => e.stopPropagation()}
          disabled={required}
          style={{ cursor: required ? "default" : "pointer" }}
        />
        <div style={{ flex: 1 }} onClick={() => setOpen(o => !o)}>
          <div style={{ fontWeight: 600, fontSize: 13, display: "flex", alignItems: "center", gap: 6 }}>
            {label}
            {required && <span style={BADGE_STYLE("red")}>Required</span>}
            {usOnly && <span style={BADGE_STYLE("blue")}>US only</span>}
          </div>
          <div style={{ fontSize: 11, color: "var(--ink-muted)" }}>{description}</div>
          {progress && <div style={{ fontSize: 11, color: "#555", marginTop: 2 }}>{progress}</div>}
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <span style={{ fontSize: 12, minWidth: 80, textAlign: "center" }}>
            {status === "running" ? "⏳ Running…"
             : status === "done"   ? "✅ Done"
             : status === "warn"   ? "⚠️ Partial"
             : status === "error"  ? "❌ Error"
             : enabled ? "—" : "Disabled"}
          </span>
          {enabled && (
            <Btn small onClick={e => { e.stopPropagation(); onFetch(); }} disabled={status === "running"}>
              {status === "done" || status === "error" || status === "warn" ? "Re-fetch" : "Fetch"}
            </Btn>
          )}
          <span style={{ fontSize: 12, color: "var(--ink-muted)", cursor: "pointer" }}
            onClick={() => setOpen(o => !o)}>
            {open ? "▲" : "▼"}
          </span>
        </div>
      </div>
      {open && children && (
        <div style={{ padding: "4px 14px 12px" }}>
          {children}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
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

function BADGE_STYLE(color: "red" | "blue"): React.CSSProperties {
  return {
    fontSize: 9, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase",
    padding: "1px 6px", borderRadius: 10,
    background: color === "red" ? "#c62828" : "#1976d2",
    color: "#fff",
  };
}

function estimateCellCount(
  bbox: [number, number, number, number],
  cellSizeM: number,
): number {
  const [minx, miny, maxx, maxy] = bbox;
  const lat = (miny + maxy) / 2;
  const mPerDegLat = 111_320;
  const mPerDegLon = 111_320 * Math.cos((lat * Math.PI) / 180);
  const widthM = Math.abs(maxx - minx) * mPerDegLon;
  const heightM = Math.abs(maxy - miny) * mPerDegLat;
  return Math.round((widthM * heightM) / (cellSizeM * cellSizeM));
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
