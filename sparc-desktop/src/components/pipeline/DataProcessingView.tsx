import { useState } from "react";
import { useNotification } from "@/hooks/useNotifications";

const API = "http://localhost:8008";

type ProcessingStep = "fishnet" | "layers" | "clip" | "preview";

export default function DataProcessingView() {
  const [step, setStep] = useState<ProcessingStep>("fishnet");
  const [resolution, setResolution] = useState(100);
  const [crs, setCrs] = useState("EPSG:4326");
  const [bounds, setBounds] = useState<[number, number, number, number]>([-71.47, 41.77, -71.37, 41.86]);
  const [boundaryPath, setBoundaryPath] = useState("");
  const [rasterPaths, setRasterPaths] = useState<string[]>([]);
  const [newRaster, setNewRaster] = useState("");
  const [stats, setStats] = useState("mean");
  const [fishnetResult, setFishnetResult] = useState<{ n_cells: number; columns: string[] } | null>(null);
  const [zonalResult, setZonalResult] = useState<{ n_cells: number; columns: string[]; csv_path: string } | null>(null);
  const [running, setRunning] = useState(false);
  const { notify } = useNotification();

  const createFishnet = async () => {
    setRunning(true);
    try {
      const r = await fetch(`${API}/data/fishnet`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bounds,
          resolution,
          crs,
          boundary_path: boundaryPath || undefined,
        }),
      });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      setFishnetResult(data);
      notify("success", `Fishnet created: ${data.n_cells} cells`);
      setStep("layers");
    } catch (e: any) {
      notify("error", e.message);
    } finally {
      setRunning(false);
    }
  };

  const runZonal = async () => {
    if (rasterPaths.length === 0) { notify("error", "Add at least one raster"); return; }
    setRunning(true);
    try {
      const r = await fetch(`${API}/data/zonal_stats`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fishnet_path: "fishnet.gpkg",  // relative to project dir
          raster_paths: rasterPaths,
          stats,
        }),
      });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      setZonalResult(data);
      notify("success", `Zonal stats computed: ${data.columns.length} columns`);
      setStep("preview");
    } catch (e: any) {
      notify("error", e.message);
    } finally {
      setRunning(false);
    }
  };

  const addRaster = () => {
    if (newRaster.trim()) {
      setRasterPaths((prev) => [...prev, newRaster.trim()]);
      setNewRaster("");
    }
  };

  return (
    <div className="space-y-6 p-6 max-w-4xl">
      <div>
        <h2 className="text-lg font-bold">Data Processing</h2>
        <p className="text-sm text-sparc-gray-600">
          Build a spatial fishnet, assign zonal statistics from rasters or shapefiles, and clip to your study boundary.
        </p>
      </div>

      {/* Step indicator */}
      <div className="flex gap-1">
        {(["fishnet", "layers", "clip", "preview"] as const).map((s, i) => (
          <button
            key={s}
            onClick={() => setStep(s)}
            className={`flex-1 rounded-t border border-b-0 px-3 py-2 text-xs font-medium transition-colors ${
              step === s
                ? "bg-black text-white"
                : "bg-sparc-gray-100 text-sparc-gray-600 hover:bg-sparc-gray-200"
            }`}
          >
            {i + 1}. {s.charAt(0).toUpperCase() + s.slice(1)}
          </button>
        ))}
      </div>

      <div className="rounded border border-sparc-gray-200 p-6">
        {step === "fishnet" && (
          <div className="space-y-4">
            <h3 className="text-sm font-semibold">Create Fishnet Grid</h3>
            <p className="text-xs text-sparc-gray-500">
              Generate a regular grid of cells over your study area. Each cell receives a unique OBJECT_ID and centroid coordinates.
            </p>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="mb-1 block text-xs font-medium text-sparc-gray-600">
                  Cell Resolution (CRS units)
                </label>
                <input
                  type="number"
                  value={resolution}
                  onChange={(e) => setResolution(Number(e.target.value))}
                  min={1}
                  className="w-full rounded border border-sparc-gray-300 px-3 py-2 text-sm font-mono focus:border-sparc-purple focus:outline-none"
                />
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-sparc-gray-600">
                  Target CRS
                </label>
                <input
                  type="text"
                  value={crs}
                  onChange={(e) => setCrs(e.target.value)}
                  placeholder="EPSG:32618"
                  className="w-full rounded border border-sparc-gray-300 px-3 py-2 text-sm font-mono focus:border-sparc-purple focus:outline-none"
                />
              </div>
            </div>

            <div className="grid grid-cols-4 gap-2">
              {(["minX", "minY", "maxX", "maxY"] as const).map((label, i) => (
                <div key={label}>
                  <label className="mb-1 block text-[10px] uppercase text-sparc-gray-500">{label}</label>
                  <input
                    type="number"
                    step="any"
                    value={bounds[i]}
                    onChange={(e) => {
                      const nb = [...bounds] as [number, number, number, number];
                      nb[i] = parseFloat(e.target.value) || 0;
                      setBounds(nb);
                    }}
                    className="w-full rounded border border-sparc-gray-300 px-2 py-1.5 text-xs font-mono focus:border-sparc-purple focus:outline-none"
                  />
                </div>
              ))}
            </div>

            <div>
              <label className="mb-1 block text-xs font-medium text-sparc-gray-600">
                Boundary Shapefile (optional — clips fishnet)
              </label>
              <input
                type="text"
                value={boundaryPath}
                onChange={(e) => setBoundaryPath(e.target.value)}
                placeholder="path/to/boundary.shp"
                className="w-full rounded border border-sparc-gray-300 px-3 py-2 text-sm font-mono focus:border-sparc-purple focus:outline-none"
              />
            </div>

            <button
              onClick={createFishnet}
              disabled={running}
              className="rounded bg-black px-4 py-2 text-sm font-medium text-white hover:bg-sparc-gray-800 disabled:opacity-50"
            >
              {running ? "Creating…" : "Generate Fishnet"}
            </button>

            {fishnetResult && (
              <p className="text-xs text-green-600">
                Created {fishnetResult.n_cells} cells — columns: {fishnetResult.columns.join(", ")}
              </p>
            )}
          </div>
        )}

        {step === "layers" && (
          <div className="space-y-4">
            <h3 className="text-sm font-semibold">Assign Layer Statistics</h3>
            <p className="text-xs text-sparc-gray-500">
              Add raster layers (.tif) and compute zonal statistics within each fishnet cell.
            </p>

            <div className="flex items-end gap-2">
              <div className="flex-1">
                <label className="mb-1 block text-xs font-medium text-sparc-gray-600">Raster path</label>
                <input
                  type="text"
                  value={newRaster}
                  onChange={(e) => setNewRaster(e.target.value)}
                  placeholder="path/to/layer.tif"
                  className="w-full rounded border border-sparc-gray-300 px-3 py-2 text-sm font-mono focus:border-sparc-purple focus:outline-none"
                  onKeyDown={(e) => e.key === "Enter" && addRaster()}
                />
              </div>
              <button onClick={addRaster} className="rounded border border-sparc-gray-300 px-3 py-2 text-sm hover:bg-sparc-gray-100">
                Add
              </button>
            </div>

            {rasterPaths.length > 0 && (
              <ul className="space-y-1">
                {rasterPaths.map((p, i) => (
                  <li key={i} className="flex items-center gap-2 text-xs font-mono">
                    <span className="flex-1 truncate">{p}</span>
                    <button onClick={() => setRasterPaths((prev) => prev.filter((_, j) => j !== i))} className="text-red-400 hover:text-red-600">✕</button>
                  </li>
                ))}
              </ul>
            )}

            <div>
              <label className="mb-1 block text-xs font-medium text-sparc-gray-600">Statistic</label>
              <select
                value={stats}
                onChange={(e) => setStats(e.target.value)}
                className="rounded border border-sparc-gray-300 px-3 py-2 text-sm focus:border-sparc-purple focus:outline-none"
              >
                <option value="mean">Mean</option>
                <option value="max">Max</option>
                <option value="min">Min</option>
                <option value="median">Median</option>
                <option value="sum">Sum</option>
                <option value="mean min max">Mean + Min + Max</option>
              </select>
            </div>

            <button
              onClick={runZonal}
              disabled={running || rasterPaths.length === 0}
              className="rounded bg-black px-4 py-2 text-sm font-medium text-white hover:bg-sparc-gray-800 disabled:opacity-50"
            >
              {running ? "Computing…" : "Run Zonal Statistics"}
            </button>
          </div>
        )}

        {step === "clip" && (
          <div className="space-y-4">
            <h3 className="text-sm font-semibold">Clip to Boundary</h3>
            <p className="text-xs text-sparc-gray-500">
              Apply a boundary shapefile to clip the enriched fishnet to your study area.
            </p>
            <div className="rounded border-2 border-dashed border-sparc-gray-300 p-8 text-center">
              <p className="text-sm text-sparc-gray-500">
                Upload a clipping boundary shapefile.
              </p>
              <button className="mt-3 rounded bg-sparc-purple px-4 py-2 text-sm font-medium text-white hover:bg-sparc-magenta">
                Upload Clipping Boundary
              </button>
            </div>
          </div>
        )}

        {step === "preview" && (
          <div className="space-y-4">
            <h3 className="text-sm font-semibold">Preview & Export</h3>
            <p className="text-xs text-sparc-gray-500">
              Review the final fishnet data layer with all assigned statistics.
            </p>
            {zonalResult ? (
              <div className="space-y-2">
                <div className="grid grid-cols-2 gap-2">
                  <div className="rounded border border-sparc-gray-200 p-3">
                    <p className="text-[10px] uppercase text-sparc-gray-500">Cells</p>
                    <p className="text-lg font-bold">{zonalResult.n_cells}</p>
                  </div>
                  <div className="rounded border border-sparc-gray-200 p-3">
                    <p className="text-[10px] uppercase text-sparc-gray-500">Columns</p>
                    <p className="text-lg font-bold">{zonalResult.columns.length}</p>
                  </div>
                </div>
                <div className="rounded border border-sparc-gray-200 p-3">
                  <p className="text-[10px] uppercase text-sparc-gray-500 mb-1">Output CSV</p>
                  <p className="text-xs font-mono break-all">{zonalResult.csv_path}</p>
                </div>
                <div className="flex flex-wrap gap-1">
                  {zonalResult.columns.map((c) => (
                    <span key={c} className="rounded bg-sparc-gray-100 px-2 py-0.5 text-[10px] font-mono">{c}</span>
                  ))}
                </div>
              </div>
            ) : fishnetResult ? (
              <div className="rounded border border-green-200 bg-green-50 p-4 text-sm">
                <p className="font-semibold text-green-700">Fishnet ready — {fishnetResult.n_cells} cells</p>
                <p className="text-xs text-green-600 mt-1">Add raster layers in the Layers step to compute zonal statistics.</p>
              </div>
            ) : (
              <div className="flex h-64 items-center justify-center rounded border border-sparc-gray-200 bg-sparc-gray-50">
                <p className="text-sm text-sparc-gray-400">Create a fishnet first, then compute zonal stats.</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
