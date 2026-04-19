import { useState, useEffect, useCallback, useRef } from "react";
import { SectionHeader, Card, Stat, Tag, Btn, StatGrid } from "@/components/ui/DesignSystem";
import { dataSummary, prepareData, getDataVersions, selectDataVersion, getConfig } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import type { DataSummary } from "@/lib/types";
import { SPARC_RAMP_HEX } from "@/lib/design-tokens";

type Pathway = "preprocessed" | "spatial";

interface PipelineStep {
  id: number;
  name: string;
  detail: string;
  rows: number | null;
  status: "done" | "running" | "queued";
}

const DEFAULT_STEPS: PipelineStep[] = [
  { id: 1, name: "Ingest CSV", detail: "", rows: null, status: "queued" },
  { id: 2, name: "Reproject CRS", detail: "", rows: null, status: "queued" },
  { id: 3, name: "Deduplicate coords", detail: "", rows: null, status: "queued" },
  { id: 4, name: "Impute missing", detail: "", rows: null, status: "queued" },
  { id: 5, name: "Derive features", detail: "", rows: null, status: "queued" },
  { id: 6, name: "Standardise (z-score)", detail: "", rows: null, status: "queued" },
  { id: 7, name: "Spatial block split", detail: "", rows: null, status: "queued" },
  { id: 8, name: "Write cached arrow", detail: "", rows: null, status: "queued" },
];

export default function ProcessingPage() {
  const [pathway, setPathway] = useState<Pathway>("preprocessed");
  const [steps, setSteps] = useState<PipelineStep[]>(DEFAULT_STEPS);
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const { notify } = useNotification();
  const missingCanvasRef = useRef<HTMLCanvasElement>(null);
  const foldCanvasRef = useRef<HTMLCanvasElement>(null);

  // Spatial builder state
  const [fishnetRes, setFishnetRes] = useState(30);
  const [nFolds, setNFolds] = useState(5);

  useEffect(() => {
    getConfig()
      .then((cfg) => {
        const k = cfg?.pipeline?.n_spatial_folds;
        if (typeof k === "number" && k > 0) setNFolds(k);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    dataSummary()
      .then((s) => {
        setSummary(s);
        // Auto-populate steps from data
        const nRows = s.row_count ?? 0;
        const nCols = s.columns?.length ?? 0;
        setSteps((prev) =>
          prev.map((step, i) => {
            if (i === 0) return { ...step, detail: `data · ${nCols} columns`, rows: nRows, status: "done" };
            return step;
          }),
        );
      })
      .catch(() => {});
  }, []);

  // Draw missing value matrix
  useEffect(() => {
    const canvas = missingCanvasRef.current;
    if (!canvas || !summary) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    const cols = summary.columns ?? [];
    const nCols = cols.length;
    const nBins = 14;
    const cellW = w / nCols;
    const cellH = h / nBins;

    // Draw grid — mark missing based on column-level missing count from summary
    for (let c = 0; c < nCols; c++) {
      const colName = cols[c];
      const colSummary = summary.numeric_summary?.[colName];
      const totalCount = colSummary?.count ?? summary.row_count ?? nBins;
      const missingFrac = totalCount < (summary.row_count ?? 0)
        ? 1 - totalCount / (summary.row_count ?? 1)
        : 0;
      for (let b = 0; b < nBins; b++) {
        // Show missing rows in crimson, present rows in warm neutral
        const isMissingBin = missingFrac > 0 && b < Math.ceil(missingFrac * nBins);
        ctx.fillStyle = isMissingBin ? "var(--crimson)" : `hsl(40, ${20 + b * 3}%, 88%)`;
        ctx.fillRect(c * cellW, b * cellH, cellW - 1, cellH - 1);
      }
    }

    // Column labels
    ctx.fillStyle = "var(--muted)";
    ctx.font = "9px JetBrains Mono";
    ctx.textAlign = "center";
    cols.forEach((col, i) => {
      ctx.save();
      ctx.translate(i * cellW + cellW / 2, h - 2);
      ctx.rotate(-Math.PI / 4);
      ctx.fillText(col.slice(0, 8), 0, 0);
      ctx.restore();
    });
  }, [summary]);

  // Draw fold preview
  useEffect(() => {
    const canvas = foldCanvasRef.current;
    if (!canvas) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    const nX = 10, nY = 6;
    const cellW = w / nX, cellH = h / nY;

    for (let y = 0; y < nY; y++) {
      for (let x = 0; x < nX; x++) {
        const fold = Math.floor(((x + y * 3) % nFolds));
        ctx.fillStyle = (SPARC_RAMP_HEX[fold % SPARC_RAMP_HEX.length]) + "66";
        ctx.fillRect(x * cellW + 1, y * cellH + 1, cellW - 2, cellH - 2);

        // Label fold index
        ctx.fillStyle = "var(--muted)";
        ctx.font = "bold 9px JetBrains Mono";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(fold + 1), x * cellW + cellW / 2, y * cellH + cellH / 2);
      }
    }
  }, [nFolds]);

  const handleApplyAll = useCallback(async () => {
    notify("info", "Running processing pipeline...");
    // Mark first step as running
    setSteps((prev) =>
      prev.map((s, j) => ({ ...s, status: j === 0 ? "running" : "queued" as const })),
    );
    try {
      const result = await prepareData({ raster_paths: [] });
      // Mark all steps done on success
      setSteps((prev) =>
        prev.map((s) => ({ ...s, status: "done" as const, rows: result.n_cells || s.rows })),
      );
      notify("success", `Processing complete — ${result.n_cells} cells, ${result.columns.length} columns`);
      // Refresh summary
      dataSummary().then(setSummary).catch(() => {});
    } catch (e) {
      notify("error", e instanceof Error ? e.message : "Processing failed");
      setSteps((prev) => prev.map((s) => ({ ...s, status: "queued" as const })));
    }
  }, [notify]);

  const handleRevert = useCallback(async () => {
    try {
      const { versions } = await getDataVersions();
      if (versions.length < 2) {
        notify("info", "No previous version to revert to");
        return;
      }
      // Select the earliest (original) version
      await selectDataVersion(versions[0].version ?? 0);
      setSteps(DEFAULT_STEPS);
      dataSummary().then(setSummary).catch(() => {});
      notify("success", "Reverted to original data");
    } catch {
      // If versioning not available, just reset UI
      setSteps(DEFAULT_STEPS);
      notify("info", "Processing reverted (local only)");
    }
  }, [notify]);

  const nRows = summary?.row_count ?? 0;
  const nCols = summary?.columns?.length ?? 0;
  // Compute total missing cells from numeric_summary: columns where count < row_count
  const missingCount = summary
    ? Object.values(summary.numeric_summary ?? {}).reduce(
        (sum, s) => sum + Math.max(0, nRows - (s?.count ?? nRows)),
        0,
      )
    : null;

  return (
    <div>
      <SectionHeader
        kicker="03 · setup"
        label="Processing"
        right={
          <div style={{ display: "flex", gap: 8 }}>
            {pathway === "preprocessed" && (
              <>
                <Btn small onClick={handleRevert}>Revert</Btn>
                <Btn primary small onClick={handleApplyAll}>Apply all</Btn>
              </>
            )}
          </div>
        }
      />

      {/* Pathway toggle */}
      <div style={{ display: "flex", gap: 4, marginBottom: 14 }}>
        {(["preprocessed", "spatial"] as const).map((p) => (
          <button
            key={p}
            onClick={() => setPathway(p)}
            style={{
              border: "1px solid " + (pathway === p ? "var(--ink)" : "var(--line)"),
              background: pathway === p ? "var(--ink)" : "#fff",
              color: pathway === p ? "#fff" : "var(--ink-2)",
              fontSize: 11,
              padding: "5px 12px",
              borderRadius: 4,
              fontFamily: "inherit",
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            {p === "preprocessed" ? "Preprocessed Data" : "Spatial Data Builder"}
          </button>
        ))}
      </div>

      {pathway === "preprocessed" ? (
        <>
          <StatGrid>
            <Stat label="Input Rows" value={nRows ? nRows.toLocaleString() : "—"} tint="var(--ink)" />
            <Stat label="Columns" value={nCols ? String(nCols) : "—"} tint="var(--purple)" />
            <Stat label="Missing Cells" value={missingCount != null ? String(missingCount) : "—"} tint="var(--amber)" sub={missingCount != null && nRows && nCols ? `${(missingCount / Math.max(1, nRows * nCols) * 100).toFixed(2)}% of cells` : undefined} />
            <Stat label="Spatial Folds" value={String(nFolds)} tint="var(--crimson)" />
          </StatGrid>

          <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 14 }}>
            <Card title="Transformation pipeline" subtitle="applied left-to-right · fold-aware where marked">
              {steps.map((step) => (
                <div
                  key={step.id}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "30px 1fr 68px 70px",
                    alignItems: "center",
                    gap: 10,
                    padding: "8px 0",
                    borderTop: step.id > 1 ? "1px dashed var(--line)" : "none",
                  }}
                >
                  <span
                    className="mono"
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      justifyContent: "center",
                      width: 26,
                      height: 26,
                      borderRadius: 4,
                      background:
                        step.status === "done"
                          ? "var(--ink)"
                          : step.status === "running"
                          ? "var(--crimson)"
                          : "rgba(0,0,0,0.05)",
                      color: step.status === "queued" ? "var(--muted)" : "#fff",
                      fontSize: 11,
                      fontWeight: 700,
                    }}
                  >
                    {step.id}
                  </span>
                  <div>
                    <div style={{ fontSize: 12.5, fontWeight: 600 }}>{step.name}</div>
                    {step.detail && (
                      <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>
                        {step.detail}
                      </div>
                    )}
                    {step.status === "running" && (
                      <div
                        style={{
                          height: 4,
                          background: "rgba(0,0,0,0.05)",
                          borderRadius: 2,
                          marginTop: 4,
                          overflow: "hidden",
                        }}
                      >
                        <div
                          style={{
                            width: "60%",
                            height: "100%",
                            background: "var(--crimson)",
                            animation: "loadBar 1.3s ease-out infinite",
                          }}
                        />
                      </div>
                    )}
                  </div>
                  <span className="mono" style={{ fontSize: 10, textAlign: "right", color: "var(--muted)" }}>
                    {step.rows !== null ? step.rows.toLocaleString() : "—"}
                  </span>
                  <Tag
                    color={
                      step.status === "done"
                        ? "var(--ink)"
                        : step.status === "running"
                        ? "var(--crimson)"
                        : "var(--muted)"
                    }
                  >
                    {step.status}
                  </Tag>
                </div>
              ))}
            </Card>

            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <Card title="Missing value matrix" subtitle={`${nCols} columns · 14 column bins`}>
                <canvas
                  ref={missingCanvasRef}
                  style={{ width: "100%", height: 180, display: "block" }}
                />
              </Card>

              <Card title="Fold preview" subtitle={`${nFolds} spatial blocks`}>
                <canvas
                  ref={foldCanvasRef}
                  style={{ width: "100%", height: 140, display: "block" }}
                />
              </Card>
            </div>
          </div>
        </>
      ) : (
        /* Spatial Data Builder pathway */
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          <Card title="Upload spatial data" subtitle="rasters, shapefiles, and boundary layers">
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <UploadRow label="Rasters (.tif)" accept=".tif,.tiff" />
              <UploadRow label="Shapefiles (.shp)" accept=".shp,.dbf,.prj,.shx" />
              <UploadRow label="Boundary layer" accept=".shp,.geojson,.gpkg" />

              <div style={{ borderTop: "1px dashed var(--line)", paddingTop: 12 }}>
                <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 6 }}>
                  Fishnet resolution (m)
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <input
                    type="range"
                    min={10}
                    max={500}
                    step={10}
                    value={fishnetRes}
                    onChange={(e) => setFishnetRes(Number(e.target.value))}
                    style={{ flex: 1 }}
                  />
                  <span className="mono" style={{ fontSize: 12, fontWeight: 600, width: 50, textAlign: "right" }}>
                    {fishnetRes} m
                  </span>
                </div>
              </div>

              <div style={{ display: "flex", gap: 8 }}>
                <Btn small>Preview on map</Btn>
                <Btn small>Run zonal stats</Btn>
                <Btn small primary>Save as CSV</Btn>
              </div>
            </div>
          </Card>

          <Card title="Processing steps" subtitle="spatial data builder pipeline">
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {["Upload files", "Create fishnet grid", "Run summary statistics", "Reproject to target CRS", "Clip to boundary", "Preview & validate", "Export"].map((step, i) => (
                <div key={step} style={{ display: "flex", alignItems: "center", gap: 10, padding: "6px 0", borderTop: i > 0 ? "1px dashed var(--line)" : "none" }}>
                  <span className="mono" style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: 22, height: 22, borderRadius: 4, background: "rgba(0,0,0,0.05)", color: "var(--muted)", fontSize: 10, fontWeight: 700 }}>
                    {i + 1}
                  </span>
                  <span style={{ fontSize: 12.5, fontWeight: 500 }}>{step}</span>
                  <Tag color="var(--muted)">pending</Tag>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}

function UploadRow({ label, accept }: { label: string; accept: string }) {
  const [file, setFile] = useState<string | null>(null);
  return (
    <div>
      <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <label
          style={{
            border: "1px dashed var(--line)",
            borderRadius: 5,
            padding: "6px 12px",
            fontSize: 11,
            cursor: "pointer",
            flex: 1,
            textAlign: "center",
            color: file ? "var(--ink-2)" : "var(--muted)",
            background: file ? "#fff8ef" : "#fff",
          }}
        >
          {file ?? "Drop or click to select"}
          <input
            type="file"
            accept={accept}
            style={{ display: "none" }}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) setFile(f.name);
            }}
          />
        </label>
      </div>
    </div>
  );
}
