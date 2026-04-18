import { useState, useEffect, useCallback, useMemo, type ErrorInfo } from "react";
import { dataPreview, dataSummary, dataGeoJson, uploadData, listDataFiles, selectDataFile } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import { pickCsv } from "@/lib/fileDialogs";
import DropZone from "@/components/common/DropZone";
import SpatialMap from "@/components/map/SpatialMap";
import DataPrepWizard from "@/components/pipeline/DataPrepWizard";
import DataValidationChecklist from "@/components/data/DataValidationChecklist";
import DataVersionSelector from "@/components/data/DataVersionSelector";
import React from "react";
import type { DataSummary, DataPreview, GeoJsonData } from "@/lib/types";

interface ProjectFile {
  name: string;
  path: string;
  relative: string;
  size: number;
}

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DataView(_props: { onNavigateToProject?: () => void }) {
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [preview, setPreview] = useState<DataPreview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [projectFiles, setProjectFiles] = useState<ProjectFile[]>([]);
  const [projectDir, setProjectDir] = useState<string>("");
  const [showFilePicker, setShowFilePicker] = useState(false);
  const [expandedGroup, setExpandedGroup] = useState<string | null>(null);
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortAsc, setSortAsc] = useState(true);
  const [mapVar, setMapVar] = useState<string>("");
  const [geojson, setGeojson] = useState<GeoJsonData | null>(null);
  const [showMap, setShowMap] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);
  const [statsVar, setStatsVar] = useState<string>("");
  const [showPrepWizard, setShowPrepWizard] = useState(false);
  const { notify } = useNotification();

  const reload = async () => {
    try {
      const [s, p] = await Promise.all([dataSummary(), dataPreview(20)]);
      setSummary(s);
      setPreview(p);
      setError(null);
      // Clear stale map data so it re-fetches on next "Show Map" click
      setGeojson(null);
      setShowMap(false);
      setMapError(null);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // "No data loaded" is expected when a project has no data file yet — not an error
      if (msg.includes("400") && msg.toLowerCase().includes("no data loaded")) {
        setSummary(null);
        setPreview(null);
      } else {
        setError(msg);
      }
    }
  };

  const loadProjectFiles = async () => {
    try {
      const res = await listDataFiles();
      setProjectFiles(res.files);
      setProjectDir(res.project_dir);
    } catch {
      // ignore — project may not be loaded yet
    }
  };

  useEffect(() => {
    reload();
    loadProjectFiles();
  }, []);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await uploadFile(file);
  };

  const uploadFile = useCallback(async (file: File) => {
    try {
      setError(null);
      await uploadData(file);
      await reload();
      await loadProjectFiles();
      notify("success", `Uploaded ${file.name}`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      notify("error", msg);
    }
  }, [notify]);

  const handleFileDrop = useCallback(async (files: File[]) => {
    const file = files[0];
    if (file) await uploadFile(file);
  }, [uploadFile]);

  const handleSelectFile = async (path: string) => {
    try {
      setError(null);
      await selectDataFile(path);
      setShowFilePicker(false);
      await reload();
      const name = path.split(/[\\/]/).pop() ?? path;
      notify("success", `Data file set to ${name}`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      notify("error", msg);
    }
  };

  const handleBrowseCsv = async () => {
    const file = await pickCsv();
    if (file) await handleSelectFile(file);
  };

  return (
    <DropZone onFileDrop={handleFileDrop} accept={[".csv"]}>
    <div>
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-bold">Data</h1>
        <div className="flex gap-2">
          <button
            onClick={handleBrowseCsv}
            className="rounded border border-sparc-gray-300 bg-white px-4 py-2 text-sm font-medium hover:bg-sparc-gray-100"
          >
            Browse Files…
          </button>
          {projectFiles.length > 0 && (
            <button
              onClick={() => setShowFilePicker(!showFilePicker)}
              className="rounded border border-sparc-gray-300 bg-white px-4 py-2 text-sm font-medium hover:bg-sparc-gray-100"
            >
              {showFilePicker ? "Hide Files" : "Select from Project"}
            </button>
          )}
          <label className="cursor-pointer rounded bg-black px-4 py-2 text-sm font-medium text-white hover:bg-sparc-gray-800">
            Upload CSV
            <input type="file" accept=".csv" onChange={handleUpload} className="hidden" />
          </label>
          <button
            onClick={() => setShowPrepWizard(!showPrepWizard)}
            className="rounded border border-sparc-purple bg-sparc-purple/10 px-4 py-2 text-sm font-medium text-sparc-purple hover:bg-sparc-purple/20"
          >
            {showPrepWizard ? "Hide Wizard" : "Raster Prep Wizard"}
          </button>
        </div>
      </div>

      {error && <p className="mb-4 rounded border border-sparc-crimson bg-red-50 p-3 text-sm text-sparc-crimson">{error}</p>}

      {showFilePicker && (
        <div className="mb-6 rounded border border-sparc-gray-200 bg-sparc-gray-50 p-4">
          <div className="mb-2 text-sm font-medium">Files in project directory</div>
          <div className="mb-1 truncate text-xs text-sparc-gray-500">{projectDir}</div>
          {projectFiles.length === 0 ? (
            <p className="text-sm text-sparc-gray-600">No CSV files found in project directory. Upload one above.</p>
          ) : (
            <ul className="max-h-48 space-y-1 overflow-y-auto">
              {projectFiles.map((f) => (
                <li key={f.path}>
                  <button
                    onClick={() => handleSelectFile(f.path)}
                    className="flex w-full items-center justify-between rounded px-3 py-2 text-left text-sm hover:bg-sparc-gray-200"
                  >
                    <span className="truncate font-mono text-xs">{f.relative}</span>
                    <span className="ml-2 shrink-0 text-xs text-sparc-gray-500">{formatSize(f.size)}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Empty state: no data file loaded yet */}
      {showPrepWizard && (
        <div className="mb-6">
          <DataPrepWizard onComplete={() => { setShowPrepWizard(false); reload(); }} />
        </div>
      )}
      {!summary && !error && !showPrepWizard && (
        <div className="flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-sparc-gray-300 py-16">
          <div className="mb-3 text-4xl">📄</div>
          <h2 className="mb-1 text-base font-semibold">No data file loaded</h2>
          <p className="mb-4 max-w-sm text-center text-sm text-sparc-gray-500">
            Upload a CSV, browse for a file, select one from your project directory, or drag and drop here.
          </p>
        </div>
      )}

      {summary && (
        <div className="mb-6 grid grid-cols-3 gap-4">
          <div className="rounded border border-sparc-gray-200 p-4">
            <div className="text-2xl font-bold">{summary.row_count.toLocaleString()}</div>
            <div className="text-xs text-sparc-gray-600">Rows</div>
          </div>
          <div className="rounded border border-sparc-gray-200 p-4">
            <div className="text-2xl font-bold">{summary.column_count}</div>
            <div className="text-xs text-sparc-gray-600">Columns</div>
          </div>
          <div className="rounded border border-sparc-gray-200 p-4">
            <div className="text-2xl font-bold truncate">{summary.crs ?? "—"}</div>
            <div className="text-xs text-sparc-gray-600">CRS</div>
          </div>
        </div>
      )}

      {/* Data validation + versioning (visible when data is loaded) */}
      {summary && (
        <div className="mb-6 grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <DataValidationChecklist />
          </div>
          <div>
            <DataVersionSelector onVersionSelected={reload} />
          </div>
        </div>
      )}

      {/* Per-variable Summary Stats — dropdown + blocky cards */}
      {summary && Object.keys(summary.numeric_summary).length > 0 && (() => {
        const numericCols = Object.keys(summary.numeric_summary);
        const activeVar = statsVar || numericCols[0] || "";
        const stats = summary.numeric_summary[activeVar];
        return (
          <div className="mb-6">
            <div className="mb-3 flex items-center gap-3">
              <h2 className="text-sm font-semibold">Variable Statistics</h2>
              <select
                value={activeVar}
                onChange={(e) => setStatsVar(e.target.value)}
                className="rounded border border-sparc-gray-300 px-2 py-1 text-xs font-mono focus:border-sparc-purple focus:outline-none"
              >
                {numericCols.map((col) => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>
            {stats && (
              <div className="grid grid-cols-4 gap-3">
                {[
                  { label: "Min", value: stats.min },
                  { label: "Max", value: stats.max },
                  { label: "Mean", value: stats.mean },
                  { label: "Median", value: stats.median ?? stats.mean },
                  { label: "Std Dev", value: stats.std },
                  { label: "Count", value: stats.count ?? summary.row_count },
                  { label: "25th Pctl", value: stats["25%"] ?? stats.q25 },
                  { label: "75th Pctl", value: stats["75%"] ?? stats.q75 },
                ].filter((s) => s.value != null).map((s) => (
                  <div key={s.label} className="rounded border border-sparc-gray-200 bg-white px-4 py-3">
                    <div className="text-lg font-bold tabular-nums">
                      {typeof s.value === "number"
                        ? Number.isInteger(s.value)
                          ? s.value.toLocaleString()
                          : s.value.toFixed(3)
                        : String(s.value)}
                    </div>
                    <div className="text-[10px] font-medium uppercase tracking-wide text-sparc-gray-500">{s.label}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })()}

      {/* Map Viewport */}
      {summary && (
        <div className="mb-6">
          <div className="mb-2 flex items-center gap-3">
            <h2 className="text-sm font-semibold">Spatial Preview</h2>
            <button
              onClick={async () => {
                if (!showMap) {
                  try {
                    setMapError(null);
                    const gj = await dataGeoJson(mapVar || undefined);
                    setGeojson(gj);
                  } catch (err) {
                    const msg = err instanceof Error ? err.message : String(err);
                    setMapError(msg);
                    notify("error", `Map load failed: ${msg}`);
                  }
                }
                setShowMap(!showMap);
              }}
              className="rounded border border-sparc-gray-300 px-3 py-1 text-xs hover:bg-sparc-gray-100"
            >
              {showMap ? "Hide Map" : "Show Map"}
            </button>
            {showMap && (
              <select
                value={mapVar}
                onChange={async (e) => {
                  setMapVar(e.target.value);
                  try {
                    const gj = await dataGeoJson(e.target.value || undefined);
                    setGeojson(gj);
                  } catch { /* ignore */ }
                }}
                className="rounded border border-sparc-gray-300 px-2 py-1 text-xs"
              >
                <option value="">All numeric</option>
                {Object.keys(summary.numeric_summary).map((col) => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            )}
          </div>
          {showMap && (
            <div className="rounded border border-sparc-gray-200 overflow-hidden" style={{ height: 520 }}>
              {mapError ? (
                <div className="flex items-center justify-center h-full bg-sparc-gray-50">
                  <p className="text-sm text-sparc-crimson px-4 text-center">Map error: {mapError}</p>
                </div>
              ) : (
                <MapErrorBoundary>
                  <SpatialMap geojson={geojson as any} colorField={mapVar || undefined} height="520px" />
                </MapErrorBoundary>
              )}
            </div>
          )}
        </div>
      )}

      {/* Collapsible Data Table */}
      <CollapsibleDataTable
        summary={summary}
        preview={preview}
        expandedGroup={expandedGroup}
        setExpandedGroup={setExpandedGroup}
        sortCol={sortCol}
        setSortCol={setSortCol}
        sortAsc={sortAsc}
        setSortAsc={setSortAsc}
      />
    </div>
    </DropZone>
  );
}

/* ----------------------------------------------------------------
   Map Error Boundary – catches deck.gl / WebGL render crashes
   ---------------------------------------------------------------- */
class MapErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; message: string }
> {
  state = { hasError: false, message: "" };
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, message: error.message };
  }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("SpatialMap render crash:", error, info);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center h-full bg-sparc-gray-50">
          <p className="text-sm text-sparc-crimson px-4 text-center">
            Map rendering failed: {this.state.message}
          </p>
        </div>
      );
    }
    return this.props.children;
  }
}

/* ----------------------------------------------------------------
   Collapsible Data Table sub-component
   ---------------------------------------------------------------- */
interface CollapsibleProps {
  summary: DataSummary | null;
  preview: DataPreview | null;
  expandedGroup: string | null;
  setExpandedGroup: (g: string | null) => void;
  sortCol: string | null;
  setSortCol: (c: string | null) => void;
  sortAsc: boolean;
  setSortAsc: (a: boolean) => void;
}

function CollapsibleDataTable({ summary, preview, expandedGroup, setExpandedGroup, sortCol, setSortCol, sortAsc, setSortAsc }: CollapsibleProps) {
  // Group columns by type
  const groups = useMemo(() => {
    if (!summary) return {};
    const g: Record<string, string[]> = { numeric: [], categorical: [], spatial: [], other: [] };
    for (const [col, dtype] of Object.entries(summary.dtypes)) {
      if (col === "geometry") { g.spatial.push(col); continue; }
      if (dtype.startsWith("float") || dtype.startsWith("int") || dtype === "number") g.numeric.push(col);
      else if (dtype === "object" || dtype === "category" || dtype === "string") g.categorical.push(col);
      else g.other.push(col);
    }
    // Remove empty groups
    return Object.fromEntries(Object.entries(g).filter(([, cols]) => cols.length > 0));
  }, [summary]);

  // Sort preview rows
  const sortedRows = useMemo(() => {
    if (!preview || !sortCol) return preview?.rows ?? [];
    return [...preview.rows].sort((a, b) => {
      const av = a[sortCol] ?? 0;
      const bv = b[sortCol] ?? 0;
      if (typeof av === "number" && typeof bv === "number") return sortAsc ? av - bv : bv - av;
      return sortAsc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
    });
  }, [preview, sortCol, sortAsc]);

  const handleSort = (col: string) => {
    if (sortCol === col) setSortAsc(!sortAsc);
    else { setSortCol(col); setSortAsc(true); }
  };

  if (!preview) return null;

  return (
    <div className="space-y-3">
      <h2 className="text-sm font-semibold">Data Preview</h2>

      {/* Column groups */}
      <div className="flex flex-wrap gap-2 mb-2">
        {Object.entries(groups).map(([group, cols]) => (
          <button
            key={group}
            onClick={() => setExpandedGroup(expandedGroup === group ? null : group)}
            className={`rounded px-3 py-1 text-xs font-medium transition-colors ${
              expandedGroup === group
                ? "bg-sparc-purple text-white"
                : "border border-sparc-gray-300 hover:bg-sparc-gray-100"
            }`}
          >
            {group} ({cols.length})
          </button>
        ))}
        {expandedGroup && (
          <button
            onClick={() => setExpandedGroup(null)}
            className="rounded px-2 py-1 text-xs text-sparc-gray-600 hover:underline"
          >
            Show all
          </button>
        )}
      </div>

      {/* Expanded group column list */}
      {expandedGroup && groups[expandedGroup] && (
        <div className="rounded border border-sparc-gray-200 bg-sparc-gray-100/50 p-3 text-xs">
          <span className="font-medium capitalize">{expandedGroup} columns:</span>{" "}
          {groups[expandedGroup].join(", ")}
        </div>
      )}

      {/* Table */}
      <div className="overflow-auto rounded border border-sparc-gray-200" style={{ maxHeight: 480 }}>
        <table className="w-full text-left text-sm">
          <thead className="sticky top-0 z-10">
            <tr className="border-b border-sparc-gray-200 bg-sparc-gray-100">
              {preview.rows[0] &&
                Object.keys(preview.rows[0])
                  .filter((col) => !expandedGroup || (groups[expandedGroup] ?? []).includes(col))
                  .map((col) => (
                    <th
                      key={col}
                      className="cursor-pointer px-3 py-2 font-medium select-none hover:bg-sparc-gray-200"
                      onClick={() => handleSort(col)}
                    >
                      {col}
                      {sortCol === col && (
                        <span className="ml-1 text-xs">{sortAsc ? "↑" : "↓"}</span>
                      )}
                    </th>
                  ))}
            </tr>
          </thead>
          <tbody>
            {sortedRows.map((row, i) => (
              <tr key={i} className="border-b border-sparc-gray-100 hover:bg-sparc-gray-100/50">
                {Object.entries(row)
                  .filter(([col]) => !expandedGroup || (groups[expandedGroup] ?? []).includes(col))
                  .map(([, val], j) => (
                    <td key={j} className="px-3 py-1.5 tabular-nums">
                      {val == null ? "—" : String(val)}
                    </td>
                  ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
