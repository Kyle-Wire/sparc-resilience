import { useEffect, useState } from "react";
import { getConfig, saveConfig, dataSummary } from "@/lib/api";
import type { ProjectConfig, DataSummary } from "@/lib/types";

export default function VariablesView() {
  const [config, setConfig] = useState<ProjectConfig | null>(null);
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    getConfig().then(setConfig).catch((e) => setError(e.message));
    dataSummary().then(setSummary).catch(() => {});
  }, []);

  const predictors = config?.predictors ?? [];
  const allColumns = summary?.columns ?? [];
  const target = config?.data?.target_column ?? "";
  const coordCols = config?.data?.coord_columns ?? [];
  const idCol = config?.data?.identifier_column ?? "";
  const reserved = new Set([target, idCol, ...coordCols]);
  const available = allColumns.filter(
    (c) => !reserved.has(c) && !predictors.includes(c)
  );

  const addPredictor = (col: string) => {
    if (!config) return;
    setConfig({ ...config, predictors: [...predictors, col] });
  };

  const removePredictor = (col: string) => {
    if (!config) return;
    setConfig({ ...config, predictors: predictors.filter((p) => p !== col) });
  };

  const save = async () => {
    if (!config) return;
    setSaving(true);
    try {
      await saveConfig({ predictors: config.predictors });
      setError(null);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  if (error && !config) {
    return (
      <div className="p-6">
        <p className="text-red-600">{error}</p>
        <p className="mt-2 text-sm text-sparc-gray-600">
          Load a project first from the Project page.
        </p>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <h2 className="text-lg font-bold">Variables</h2>
        <p className="text-sm text-sparc-gray-600">
          Select predictor columns for the spatial regression pipeline.
        </p>
      </div>

      {/* Target info */}
      <div className="rounded border border-sparc-gray-200 p-4">
        <p className="text-xs font-semibold uppercase text-sparc-gray-500">Target</p>
        <p className="mt-1 font-mono text-sm">{target || "—"}</p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Available columns */}
        <div>
          <h3 className="mb-2 text-sm font-semibold">Available Columns</h3>
          <div className="max-h-80 overflow-y-auto rounded border border-sparc-gray-200">
            {available.length === 0 ? (
              <p className="p-3 text-xs text-sparc-gray-500">
                {allColumns.length === 0 ? "Load data to see columns" : "All columns assigned"}
              </p>
            ) : (
              available.map((col) => (
                <button
                  key={col}
                  onClick={() => addPredictor(col)}
                  className="flex w-full items-center justify-between border-b border-sparc-gray-100 px-3 py-2 text-left text-sm hover:bg-sparc-gray-50"
                >
                  <span className="font-mono text-xs">{col}</span>
                  <span className="text-sparc-purple">+ Add</span>
                </button>
              ))
            )}
          </div>
        </div>

        {/* Selected predictors */}
        <div>
          <h3 className="mb-2 text-sm font-semibold">
            Selected Predictors ({predictors.length})
          </h3>
          <div className="max-h-80 overflow-y-auto rounded border border-sparc-gray-200">
            {predictors.length === 0 ? (
              <p className="p-3 text-xs text-sparc-gray-500">No predictors selected</p>
            ) : (
              predictors.map((col) => (
                <button
                  key={col}
                  onClick={() => removePredictor(col)}
                  className="flex w-full items-center justify-between border-b border-sparc-gray-100 px-3 py-2 text-left text-sm hover:bg-red-50"
                >
                  <span className="font-mono text-xs">{col}</span>
                  <span className="text-red-500">× Remove</span>
                </button>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Save */}
      <div className="flex items-center gap-3">
        <button
          onClick={save}
          disabled={saving}
          className="rounded bg-sparc-purple px-4 py-2 text-sm font-medium text-white hover:bg-sparc-magenta disabled:opacity-50"
        >
          {saving ? "Saving…" : "Save Variables"}
        </button>
        {error && <p className="text-sm text-red-600">{error}</p>}
      </div>
    </div>
  );
}
