import { useState, useEffect } from "react";
import { dataPreview, dataSummary, uploadData, listDataFiles, selectDataFile } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import type { DataSummary, DataPreview } from "@/lib/types";

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

export default function DataView() {
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [preview, setPreview] = useState<DataPreview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [projectFiles, setProjectFiles] = useState<ProjectFile[]>([]);
  const [projectDir, setProjectDir] = useState<string>("");
  const [showFilePicker, setShowFilePicker] = useState(false);
  const { notify } = useNotification();

  const reload = async () => {
    try {
      const [s, p] = await Promise.all([dataSummary(), dataPreview(20)]);
      setSummary(s);
      setPreview(p);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
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
  };

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

  return (
    <div>
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-bold">Data</h1>
        <div className="flex gap-2">
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

      {preview && (
        <div className="overflow-auto rounded border border-sparc-gray-200">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-sparc-gray-200 bg-sparc-gray-100">
                {preview.rows[0] &&
                  Object.keys(preview.rows[0]).map((col) => (
                    <th key={col} className="px-3 py-2 font-medium">
                      {col}
                    </th>
                  ))}
              </tr>
            </thead>
            <tbody>
              {preview.rows.map((row, i) => (
                <tr key={i} className="border-b border-sparc-gray-100">
                  {Object.values(row).map((val, j) => (
                    <td key={j} className="px-3 py-1.5 tabular-nums">
                      {val == null ? "—" : String(val)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
