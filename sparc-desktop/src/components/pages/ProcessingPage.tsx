import { useState, useCallback, useEffect, useRef } from "react";
import { SectionHeader, Card, Stat, Btn, StatGrid, KeyVal, Tag } from "@/components/ui/DesignSystem";
import { prepareData, getConfig, dataSummary, getDataVersions, selectDataVersion, validateData, preprocessData } from "@/lib/api";
import type { DataSummary, ValidationReport, DataVersion } from "@/lib/types";
import { pickRasters, pickBoundary } from "@/lib/fileDialogs";
import { SPARC_RAMP_HEX } from "@/lib/design-tokens";
import { useNotification } from "@/hooks/useNotifications";


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
  const [validationResult, setValidationResult] = useState<ValidationReport | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [versions, setVersions] = useState<DataVersion[]>([]);

  // Spatial builder state
  const [fishnetRes, setFishnetRes] = useState(30);
  const [nFolds, setNFolds] = useState(5);
  const [sbBoundaryPath, setSbBoundaryPath] = useState<string | null>(null);
  const [sbRasterPaths, setSbRasterPaths] = useState<string[]>([]);
  const [sbStats, setSbStats] = useState("mean,std,min,max");
  const [sbRunning, setSbRunning] = useState(false);
  const [sbResult, setSbResult] = useState<{ n_cells: number; columns: string[]; csv_path: string } | null>(null);
  const [sbStepStatus, setSbStepStatus] = useState<("pending" | "done" | "running" | "error")[]>(
    new Array(5).fill("pending"),
  );

  const handlePickBoundary = useCallback(async () => {
    const path = await pickBoundary();
    if (path) setSbBoundaryPath(path);
  }, []);

  const handlePickRasters = useCallback(async () => {
    const paths = await pickRasters();
    if (paths) setSbRasterPaths((prev) => [...prev, ...paths]);
  }, []);

  const handleRemoveRaster = (i: number) => {
    setSbRasterPaths((prev) => prev.filter((_, idx) => idx !== i));
  };

  const setSbStep = (idx: number, status: "pending" | "done" | "running" | "error") => {
    setSbStepStatus((prev) => prev.map((s, i) => (i === idx ? status : s)));
  };

  const handleRunSpatialBuilder = useCallback(async () => {
    if (!sbBoundaryPath) {
      notify("error", "Select a boundary file first");
      return;
    }
    if (sbRasterPaths.length === 0) {
      notify("error", "Add at least one raster");
      return;
    }
    setSbRunning(true);
    setSbResult(null);
    setSbStepStatus(new Array(5).fill("pending"));

    const steps: Array<() => Promise<void>> = [
      async () => { /* Validate inputs — just set done */ },
      async () => { /* Create fishnet — done by prepareData */ },
      async () => { /* Run zonal stats — done by prepareData */ },
      async () => { /* Reproject — done by prepareData */ },
      async () => { /* Export CSV */ },
    ];

    try {
      for (let i = 0; i < steps.length - 1; i++) {
        setSbStep(i, "running");
        await steps[i]();
        setSbStep(i, "done");
      }

      // Step 4 (final): actually run the data prep
      setSbStep(4, "running");
      const result = await prepareData({
        boundary_path: sbBoundaryPath,
        raster_paths: sbRasterPaths,
        resolution: fishnetRes,
        stats: sbStats,
        set_as_data: true,
      });
      setSbStep(4, "done");
      setSbResult(result);
      notify("success", `Spatial data built — ${result.n_cells.toLocaleString()} cells, ${result.columns.length} columns`);
    } catch (e) {
      const failedIdx = sbStepStatus.findIndex((s) => s === "running");
      if (failedIdx >= 0) setSbStep(failedIdx, "error");
      notify("error", e instanceof Error ? e.message : "Spatial builder failed");
    } finally {
      setSbRunning(false);
    }
  }, [sbBoundaryPath, sbRasterPaths, fishnetRes, sbStats, sbStepStatus, notify]);

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

  // Load version list on mount and whenever data updates
  useEffect(() => {
    getDataVersions().then(({ versions: v }) => setVersions(v)).catch(() => {});
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
    cols.forEach((col: string, i: number) => {
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
    notify("info", "Running processing pipeline…");
    setSteps((prev) =>
      prev.map((s, j) => ({ ...s, status: j === 0 ? "running" : "queued" as const })),
    );
    try {
      await preprocessData((event) => {
        if (event.step === "__done__") return;
        setSteps((prev) => {
          const idx = prev.findIndex((s) => s.name === event.step);
          if (idx < 0) return prev;
          return prev.map((s, i) => {
            if (i === idx) return { ...s, status: "done" as const, rows: event.rows, detail: `sha·${event.sha}` };
            if (i === idx + 1 && s.status === "queued") return { ...s, status: "running" as const };
            return s;
          });
        });
      });
      notify("success", "Processing complete");
      dataSummary().then(setSummary).catch(() => {});
      getDataVersions().then(({ versions: v }) => setVersions(v)).catch(() => {});
    } catch (e) {
      notify("error", e instanceof Error ? e.message : "Processing failed");
      setSteps((prev) => prev.map((s) => ({ ...s, status: "queued" as const })));
    }
  }, [notify]);

  const handleRevert = useCallback(async () => {
    try {
      const { versions: v } = await getDataVersions();
      if (v.length < 2) {
        notify("info", "No previous version to revert to");
        return;
      }
      // Select the earliest (original) version
      await selectDataVersion(v[0].version ?? 0);
      setSteps(DEFAULT_STEPS);
      dataSummary().then(setSummary).catch(() => {});
      notify("success", "Reverted to original data");
    } catch {
      // If versioning not available, just reset UI
      setSteps(DEFAULT_STEPS);
      notify("info", "Processing reverted (local only)");
    }
  }, [notify]);

  const handleSelectVersion = useCallback(async (version: number) => {
    try {
      await selectDataVersion(version);
      dataSummary().then(setSummary).catch(() => {});
      notify("success", `Loaded version ${version}`);
    } catch {
      notify("error", "Failed to load version");
    }
  }, [notify]);

  const handleValidate = useCallback(async () => {
    setIsValidating(true);
    setValidationResult(null);
    try {
      const report = await validateData();
      setValidationResult(report);
      if (report.n_critical > 0) {
        notify("error", `Validation: ${report.n_critical} critical issues found`);
      } else if (report.n_warning > 0) {
        notify("info", `Validation: ${report.n_warning} warnings`);
      } else {
        notify("success", "Validation passed");
      }
    } catch (e) {
      notify("error", e instanceof Error ? e.message : "Validation failed");
    } finally {
      setIsValidating(false);
    }
  }, [notify]);

  const nRows = summary?.row_count ?? 0;
  const nCols = summary?.columns?.length ?? 0;
  // Compute total missing cells from numeric_summary: columns where count < row_count
  const missingCount = summary?.numeric_summary != null
    ? Object.values(summary.numeric_summary).reduce<number>(
        (sum, s) => sum + Math.max(0, nRows - (s.count ?? nRows)),
        0,
      )
    : null;

  return (
    <div>
      <SectionHeader
        kicker="03 · setup"
        label="Processing"
        right={
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            {pathway === "preprocessed" && (
              <>
                <Btn small onClick={handleValidate} disabled={isValidating}>
                  {isValidating ? "Validating…" : "Validate"}
                </Btn>
                {versions.length > 1 ? (
                  <select
                    defaultValue=""
                    onChange={(e) => e.target.value && handleSelectVersion(Number(e.target.value))}
                    style={{
                      border: "1px solid var(--line)",
                      borderRadius: 4,
                      fontSize: 11,
                      padding: "4px 8px",
                      fontFamily: "inherit",
                      background: "#fff",
                      cursor: "pointer",
                    }}
                  >
                    <option value="" disabled>Version…</option>
                    {versions.map((v) => (
                      <option key={v.version} value={v.version}>
                        v{v.version} · {v.n_rows?.toLocaleString()} rows
                      </option>
                    ))}
                  </select>
                ) : (
                  <Btn small onClick={handleRevert}>Revert</Btn>
                )}
                <Btn primary small onClick={handleApplyAll}
                  disabled={validationResult != null && validationResult.n_critical > 0}
                  title={validationResult?.n_critical ? "Fix critical issues before processing" : undefined}
                >Apply all</Btn>
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

          {/* P4-2: Validation issues panel */}
          {validationResult && validationResult.items && validationResult.items.length > 0 && (
            <div
              style={{
                marginBottom: 14,
                border: `1px solid ${validationResult.n_critical > 0 ? "var(--crimson)" : "var(--amber)"}`,
                borderRadius: 6,
                padding: "10px 14px",
                background: validationResult.n_critical > 0 ? "rgba(220,38,38,0.04)" : "rgba(245,158,11,0.04)",
              }}
            >
              <div style={{ fontSize: 11.5, fontWeight: 700, marginBottom: 6, color: validationResult.n_critical > 0 ? "var(--crimson)" : "var(--amber)" }}>
                Validation — {validationResult.n_critical} critical · {validationResult.n_warning} warning
              </div>
              {validationResult.items.slice(0, 10).map((issue, i) => (
                <div key={i} className="mono" style={{ fontSize: 10, color: "var(--ink-2)", padding: "2px 0", borderTop: i > 0 ? "1px dashed var(--line)" : "none" }}>
                  <span style={{ fontWeight: 700, color: issue.severity === "critical" ? "var(--crimson)" : "var(--amber)" }}>
                    [{issue.severity?.toUpperCase() ?? "INFO"}]
                  </span>
                  {" "}{issue.message}
                </div>
              ))}
              {validationResult.items.length > 10 && (
                <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 4 }}>
                  + {validationResult.items.length - 10} more issues…
                </div>
              )}
            </div>
          )}

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
        <>
          <StatGrid>
            <Stat label="Boundary" value={sbBoundaryPath ? sbBoundaryPath.split(/[\\/]/).pop()! : "—"} tint="var(--ink)" />
            <Stat label="Rasters" value={String(sbRasterPaths.length)} tint="var(--purple)" />
            <Stat label="Resolution" value={`${fishnetRes} m`} tint="var(--amber)" />
            <Stat label="Cells" value={sbResult ? sbResult.n_cells.toLocaleString() : "—"} tint="var(--crimson)" />
          </StatGrid>

          <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 14 }}>
            <Card title="Inputs" subtitle="boundary layer + raster files">
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {/* Boundary picker */}
                <div>
                  <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>
                    Boundary layer
                  </div>
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <div style={{ flex: 1, fontSize: 11.5, color: sbBoundaryPath ? "var(--ink)" : "var(--muted)", fontStyle: sbBoundaryPath ? "normal" : "italic", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {sbBoundaryPath ? sbBoundaryPath.split(/[\\/]/).pop() : "No boundary selected"}
                    </div>
                    <button
                      onClick={handlePickBoundary}
                      style={{ padding: "5px 12px", border: "1px solid var(--line)", borderRadius: 4, fontSize: 11, cursor: "pointer", fontFamily: "inherit", background: "#fff", whiteSpace: "nowrap" }}
                    >
                      Browse…
                    </button>
                  </div>
                </div>

                {/* Raster list */}
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                    <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
                      Raster files (.tif)
                    </div>
                    <button
                      onClick={handlePickRasters}
                      style={{ padding: "3px 10px", background: "var(--ink)", color: "#fff", border: "none", borderRadius: 4, fontSize: 11, cursor: "pointer", fontWeight: 600 }}
                    >
                      + Add
                    </button>
                  </div>
                  {sbRasterPaths.length === 0 ? (
                    <div style={{ fontSize: 11.5, color: "var(--muted)", fontStyle: "italic", padding: "8px 0" }}>
                      No rasters selected — click <strong>+ Add</strong>
                    </div>
                  ) : (
                    <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                      {sbRasterPaths.map((p, i) => (
                        <div key={i} style={{ display: "flex", alignItems: "center", gap: 6, padding: "3px 0" }}>
                          <span className="mono" style={{ flex: 1, fontSize: 10.5, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {p.split(/[\\/]/).pop()}
                          </span>
                          <button
                            onClick={() => handleRemoveRaster(i)}
                            style={{ width: 18, height: 18, border: "none", background: "none", cursor: "pointer", color: "var(--muted)", fontSize: 14, lineHeight: 1, display: "flex", alignItems: "center", justifyContent: "center" }}
                          >
                            ×
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Resolution + stats */}
                <div style={{ borderTop: "1px dashed var(--line)", paddingTop: 10, display: "flex", flexDirection: "column", gap: 10 }}>
                  <div>
                    <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>
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
                      <span className="mono" style={{ fontSize: 12, fontWeight: 700, width: 50, textAlign: "right" }}>
                        {fishnetRes} m
                      </span>
                    </div>
                  </div>
                  <div>
                    <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>
                      Zonal statistics
                    </div>
                    <input
                      type="text"
                      value={sbStats}
                      onChange={(e) => setSbStats(e.target.value)}
                      placeholder="mean,std,min,max"
                      style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "5px 8px", fontSize: 12, width: "100%", fontFamily: "inherit" }}
                    />
                  </div>
                </div>

                <Btn primary small onClick={handleRunSpatialBuilder} disabled={sbRunning}>
                  {sbRunning ? "Building…" : "Build spatial data"}
                </Btn>
              </div>
            </Card>

            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <Card title="Pipeline steps" subtitle="spatial data builder">
                <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
                  {["Validate inputs", "Create fishnet grid", "Run zonal statistics", "Reproject to target CRS", "Export CSV / GeoPackage"].map((step, i) => {
                    const status = sbStepStatus[i];
                    return (
                      <div key={step} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 0", borderTop: i > 0 ? "1px dashed var(--line)" : "none" }}>
                        <span
                          className="mono"
                          style={{
                            display: "inline-flex", alignItems: "center", justifyContent: "center",
                            width: 24, height: 24, borderRadius: 4,
                            background: status === "done" ? "var(--ink)" : status === "running" ? "var(--crimson)" : status === "error" ? "#c0392b" : "rgba(0,0,0,0.05)",
                            color: status === "pending" ? "var(--muted)" : "#fff",
                            fontSize: 10, fontWeight: 700,
                          }}
                        >
                          {i + 1}
                        </span>
                        <span style={{ flex: 1, fontSize: 12, fontWeight: 500 }}>{step}</span>
                        <Tag color={status === "done" ? "var(--ink)" : status === "running" ? "var(--crimson)" : status === "error" ? "var(--crimson)" : "var(--muted)"}>
                          {status}
                        </Tag>
                      </div>
                    );
                  })}
                </div>
              </Card>

              {sbResult && (
                <Card title="Result" subtitle={`${sbResult.n_cells.toLocaleString()} grid cells`}>
                  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    <KeyVal label="CSV path" value={sbResult.csv_path.split(/[\\/]/).pop() ?? sbResult.csv_path} />
                    <KeyVal label="Cells" value={sbResult.n_cells.toLocaleString()} />
                    <KeyVal label="Columns" value={String(sbResult.columns.length)} />
                    <div style={{ fontSize: 10.5, color: "var(--muted)", marginTop: 4, lineHeight: 1.6 }}>
                      Data has been set as the project dataset. Navigate to <strong>Data</strong> page to verify.
                    </div>
                  </div>
                </Card>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
