import { useEffect, useState } from "react";
import { getDataVersions, selectDataVersion } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import type { DataVersion } from "@/lib/types";

interface Props {
  /** Called after a version is selected and loaded. */
  onVersionSelected?: () => void;
}

export default function DataVersionSelector({ onVersionSelected }: Props) {
  const [versions, setVersions] = useState<DataVersion[]>([]);
  const [loading, setLoading] = useState(false);
  const [selecting, setSelecting] = useState<number | null>(null);
  const { notify } = useNotification();

  const loadVersions = async () => {
    setLoading(true);
    try {
      const data = await getDataVersions();
      setVersions(data.versions);
    } catch {
      // Endpoint may not exist yet — ignore
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadVersions();
  }, []);

  const handleSelect = async (version: number) => {
    setSelecting(version);
    try {
      await selectDataVersion(version);
      notify("success", `Switched to data version ${version}`);
      onVersionSelected?.();
    } catch (e: any) {
      notify("error", e.message || "Failed to select version");
    } finally {
      setSelecting(null);
    }
  };

  if (versions.length === 0 && !loading) return null;

  return (
    <div className="rounded border border-sparc-gray-200 bg-white">
      <div className="flex items-center justify-between border-b border-sparc-gray-100 px-3 py-2">
        <h3 className="text-xs font-semibold text-sparc-gray-600">Data Versions</h3>
        <button
          onClick={loadVersions}
          disabled={loading}
          className="text-[10px] text-sparc-gray-400 hover:text-sparc-gray-600"
        >
          {loading ? "Loading…" : "Refresh"}
        </button>
      </div>

      <div className="max-h-48 overflow-auto">
        {versions.map((v) => (
          <button
            key={v.version}
            onClick={() => handleSelect(v.version)}
            disabled={selecting != null}
            className="flex w-full items-center gap-3 border-b border-sparc-gray-50 px-3 py-2 text-left hover:bg-sparc-gray-50 last:border-0 disabled:opacity-50"
          >
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded bg-sparc-gray-100 text-xs font-bold text-sparc-gray-600">
              {v.version}
            </span>
            <div className="flex-1 min-w-0">
              <p className="truncate text-xs font-medium text-sparc-gray-800">
                {v.description || v.filename}
              </p>
              <p className="text-[10px] text-sparc-gray-400">
                {v.n_rows.toLocaleString()} rows × {v.n_cols} cols ·{" "}
                {new Date(v.timestamp).toLocaleDateString()}
              </p>
            </div>
            {selecting === v.version && (
              <span className="text-[10px] text-sparc-purple animate-pulse">Loading…</span>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}
