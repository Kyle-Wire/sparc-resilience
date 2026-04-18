import { useEffect, useState } from "react";
import { listArtifacts, downloadArtifact, downloadGeoPackage } from "@/lib/api";
import type { ArtifactInfo } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const ICON_MAP: Record<string, string> = {
  ".csv": "📊",
  ".parquet": "📦",
  ".geojson": "🗺️",
  ".gpkg": "🗺️",
  ".png": "🖼️",
  ".svg": "🖼️",
  ".html": "📄",
  ".pdf": "📑",
  ".json": "{ }",
  ".yml": "⚙️",
  ".yaml": "⚙️",
};

export default function ArtifactPanel() {
  const [artifacts, setArtifacts] = useState<ArtifactInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [collapsedStages, setCollapsedStages] = useState<Set<number>>(new Set());
  const { notify } = useNotification();

  const load = async () => {
    setLoading(true);
    try {
      const data = await listArtifacts();
      setArtifacts(data.artifacts);
    } catch {
      // May not have artifacts yet
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleDownload = (art: ArtifactInfo) => {
    downloadArtifact(art.stage, art.relative_path, art.filename);
    notify("success", `Downloading ${art.filename} (${formatSize(art.size_bytes)})`);
  };

  const handleGeoPackage = () => {
    downloadGeoPackage();
    notify("success", "Downloading merged GeoPackage…");
  };

  const toggleStage = (stage: number) => {
    setCollapsedStages((prev) => {
      const next = new Set(prev);
      if (next.has(stage)) next.delete(stage);
      else next.add(stage);
      return next;
    });
  };

  // Group by stage
  const grouped: Record<number, { label: string; files: ArtifactInfo[] }> = {};
  for (const art of artifacts) {
    if (!grouped[art.stage]) {
      grouped[art.stage] = { label: art.stage_label, files: [] };
    }
    grouped[art.stage].files.push(art);
  }

  const totalSize = artifacts.reduce((s, a) => s + a.size_bytes, 0);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold">Output Artifacts</h2>
          <p className="text-xs text-sparc-gray-500">
            {artifacts.length} files · {formatSize(totalSize)} total
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleGeoPackage}
            disabled={artifacts.length === 0}
            className="rounded border border-sparc-purple bg-sparc-purple/10 px-3 py-1.5 text-xs font-medium text-sparc-purple hover:bg-sparc-purple/20 disabled:opacity-40"
          >
            Download GeoPackage
          </button>
          <button
            onClick={load}
            disabled={loading}
            className="rounded border border-sparc-gray-300 px-3 py-1.5 text-xs font-medium hover:bg-sparc-gray-100 disabled:opacity-50"
          >
            {loading ? "Scanning…" : "Refresh"}
          </button>
        </div>
      </div>

      {artifacts.length === 0 && !loading && (
        <div className="rounded border border-sparc-gray-200 bg-sparc-gray-50 py-8 text-center text-sm text-sparc-gray-500">
          No output files found. Run the pipeline to generate results.
        </div>
      )}

      {Object.entries(grouped)
        .sort(([a], [b]) => Number(a) - Number(b))
        .map(([stageStr, { label, files }]) => {
          const stage = Number(stageStr);
          const isCollapsed = collapsedStages.has(stage);
          const stageSize = files.reduce((s, f) => s + f.size_bytes, 0);

          return (
            <div key={stage} className="rounded border border-sparc-gray-200 overflow-hidden">
              <button
                onClick={() => toggleStage(stage)}
                className="flex w-full items-center justify-between bg-sparc-gray-50 px-3 py-2 text-left text-xs font-medium hover:bg-sparc-gray-100"
              >
                <span className="flex items-center gap-2">
                  <span className={isCollapsed ? "" : "rotate-90 inline-block"}>▶</span>
                  {label}
                  <span className="rounded-full bg-sparc-gray-200 px-1.5 py-0.5 text-[10px] text-sparc-gray-600">
                    {files.length}
                  </span>
                </span>
                <span className="text-[10px] text-sparc-gray-400">{formatSize(stageSize)}</span>
              </button>

              {!isCollapsed && (
                <div className="divide-y divide-sparc-gray-100">
                  {files.map((art) => (
                    <button
                      key={art.relative_path}
                      onClick={() => handleDownload(art)}
                      className="flex w-full items-center gap-3 px-3 py-2 text-left text-xs hover:bg-sparc-gray-50"
                    >
                      <span className="text-base">{ICON_MAP[art.extension] ?? "📄"}</span>
                      <span className="flex-1 min-w-0 truncate font-mono text-sparc-gray-700">
                        {art.relative_path}
                      </span>
                      <span className="shrink-0 text-[10px] text-sparc-gray-400">
                        {formatSize(art.size_bytes)}
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          );
        })}
    </div>
  );
}
