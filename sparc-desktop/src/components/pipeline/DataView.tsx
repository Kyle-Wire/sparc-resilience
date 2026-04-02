import { useState, useEffect } from "react";
import { dataPreview, dataSummary, uploadData } from "@/lib/api";
import type { DataSummary, DataPreview } from "@/lib/types";

export default function DataView() {
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [preview, setPreview] = useState<DataPreview | null>(null);
  const [error, setError] = useState<string | null>(null);

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

  useEffect(() => {
    reload();
  }, []);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      await uploadData(file);
      reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div>
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-bold">Data</h1>
        <label className="cursor-pointer rounded bg-black px-4 py-2 text-sm font-medium text-white hover:bg-sparc-gray-800">
          Upload CSV
          <input type="file" accept=".csv" onChange={handleUpload} className="hidden" />
        </label>
      </div>

      {error && <p className="mb-4 rounded border border-sparc-crimson bg-red-50 p-3 text-sm text-sparc-crimson">{error}</p>}

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
