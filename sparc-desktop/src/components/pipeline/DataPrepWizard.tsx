import { useState, useCallback } from "react";
import { prepareData } from "@/lib/api";
import { pickRasters, pickBoundary } from "@/lib/fileDialogs";
import { useNotification } from "@/hooks/useNotifications";

/**
 * Wizard component for raster + boundary → fishnet → zonal‑stats → CSV.
 * Intended to be rendered inside DataView as an expandable panel.
 */
export default function DataPrepWizard({
  onComplete,
}: {
  onComplete?: (csvPath: string) => void;
}) {
  const [boundaryPath, setBoundaryPath] = useState("");
  const [rasterPaths, setRasterPaths] = useState<string[]>([]);
  const [resolution, setResolution] = useState(100);
  const [crs, setCrs] = useState("EPSG:4326");
  const [stats, setStats] = useState("mean");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<{
    n_cells: number;
    columns: string[];
    csv_path: string;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { notify } = useNotification();

  const handlePickBoundary = useCallback(async () => {
    const path = await pickBoundary();
    if (path) setBoundaryPath(path);
  }, []);

  const handlePickRasters = useCallback(async () => {
    const paths = await pickRasters();
    if (paths) setRasterPaths((prev) => [...prev, ...paths]);
  }, []);

  const removeRaster = (idx: number) =>
    setRasterPaths((prev) => prev.filter((_, i) => i !== idx));

  const handleRun = useCallback(async () => {
    if (rasterPaths.length === 0) {
      setError("Add at least one raster layer");
      return;
    }
    setRunning(true);
    setError(null);
    setResult(null);

    try {
      const res = await prepareData({
        boundary_path: boundaryPath || undefined,
        raster_paths: rasterPaths,
        resolution,
        crs,
        stats,
        set_as_data: true,
      });
      setResult(res);
      notify("success", `Prepared ${res.n_cells} cells → ${res.csv_path}`);
      onComplete?.(res.csv_path);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      notify("error", msg);
    } finally {
      setRunning(false);
    }
  }, [boundaryPath, rasterPaths, resolution, crs, stats, notify, onComplete]);

  return (
    <div className="space-y-4 rounded-lg border border-sparc-gray-200 bg-white p-4">
      <h3 className="text-sm font-semibold">
        Raster → Fishnet → Zonal Stats Wizard
      </h3>

      {/* Step 1: Boundary */}
      <div className="space-y-1">
        <label className="text-xs font-medium text-sparc-gray-600">
          Study Area Boundary (optional)
        </label>
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={boundaryPath}
            onChange={(e) => setBoundaryPath(e.target.value)}
            placeholder=".shp / .gpkg / .geojson"
            className="flex-1 rounded border border-sparc-gray-200 px-2 py-1 text-xs"
          />
          <button
            onClick={handlePickBoundary}
            className="rounded bg-sparc-gray-100 px-2 py-1 text-xs hover:bg-sparc-gray-200"
          >
            Browse…
          </button>
        </div>
      </div>

      {/* Step 2: Rasters */}
      <div className="space-y-1">
        <label className="text-xs font-medium text-sparc-gray-600">
          Raster Layers
        </label>
        <div className="flex flex-wrap gap-1.5">
          {rasterPaths.map((p, i) => (
            <span
              key={i}
              className="inline-flex items-center gap-1 rounded bg-sparc-purple/10 px-2 py-0.5 text-[10px] text-sparc-purple"
            >
              {p.split(/[\\/]/).pop()}
              <button onClick={() => removeRaster(i)} className="text-red-400 hover:text-red-600">
                ×
              </button>
            </span>
          ))}
        </div>
        <button
          onClick={handlePickRasters}
          className="rounded bg-sparc-gray-100 px-2 py-1 text-xs hover:bg-sparc-gray-200"
        >
          + Add Rasters (.tif)
        </button>
      </div>

      {/* Step 3: Parameters */}
      <div className="grid grid-cols-3 gap-3">
        <div>
          <label className="text-[10px] text-sparc-gray-500">Resolution</label>
          <input
            type="number"
            value={resolution}
            onChange={(e) => setResolution(Number(e.target.value))}
            className="w-full rounded border border-sparc-gray-200 px-2 py-1 text-xs"
          />
        </div>
        <div>
          <label className="text-[10px] text-sparc-gray-500">CRS</label>
          <input
            type="text"
            value={crs}
            onChange={(e) => setCrs(e.target.value)}
            className="w-full rounded border border-sparc-gray-200 px-2 py-1 text-xs"
          />
        </div>
        <div>
          <label className="text-[10px] text-sparc-gray-500">Stats</label>
          <input
            type="text"
            value={stats}
            onChange={(e) => setStats(e.target.value)}
            placeholder="mean min max count"
            className="w-full rounded border border-sparc-gray-200 px-2 py-1 text-xs"
          />
        </div>
      </div>

      {/* Run */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleRun}
          disabled={running || rasterPaths.length === 0}
          className="rounded bg-sparc-purple px-4 py-1.5 text-xs font-medium text-white disabled:opacity-50"
        >
          {running ? "Processing…" : "Generate Dataset"}
        </button>
        {error && <p className="text-xs text-red-600">{error}</p>}
      </div>

      {/* Result */}
      {result && (
        <div className="rounded border border-green-200 bg-green-50 p-3 text-xs space-y-1">
          <p className="font-medium text-green-700">
            Created {result.n_cells} cells
          </p>
          <p className="text-sparc-gray-600">CSV: {result.csv_path}</p>
          <p className="text-sparc-gray-500">
            Columns: {result.columns.filter((c) => c !== "geometry").join(", ")}
          </p>
        </div>
      )}
    </div>
  );
}
