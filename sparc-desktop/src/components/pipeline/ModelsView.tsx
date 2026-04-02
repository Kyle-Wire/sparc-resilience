import { useEffect, useState } from "react";
import { getConfig, saveConfig } from "@/lib/api";
import type { ProjectConfig } from "@/lib/types";

/** Model names with human-readable labels. */
const MODEL_SECTIONS: { key: string; label: string; description: string }[] = [
  { key: "gwr", label: "GWR", description: "Geographically Weighted Regression" },
  { key: "gwrf", label: "GWRF", description: "Geographically Weighted Random Forest" },
  { key: "ggpgam", label: "GGPGAM", description: "Geographically-Guided Penalised GAM" },
  { key: "meta_ensemble", label: "Meta-Ensemble", description: "Stacked model ensemble (LightGBM default)" },
  { key: "deep_kriging", label: "Deep Kriging", description: "Neural residual correction with spatial basis" },
  { key: "spatial_cv", label: "Spatial CV", description: "Cross-validation strategy" },
];

function ParamEditor({
  params,
  onChange,
}: {
  params: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
}) {
  return (
    <div className="space-y-2">
      {Object.entries(params).map(([key, value]) => (
        <div key={key} className="flex items-center gap-3">
          <label className="w-40 shrink-0 text-xs font-mono text-sparc-gray-600">
            {key}
          </label>
          {typeof value === "boolean" ? (
            <button
              onClick={() => onChange(key, !value)}
              className={`rounded px-3 py-1 text-xs font-medium ${
                value
                  ? "bg-sparc-purple text-white"
                  : "bg-sparc-gray-200 text-sparc-gray-600"
              }`}
            >
              {value ? "true" : "false"}
            </button>
          ) : typeof value === "number" ? (
            <input
              type="number"
              value={value}
              onChange={(e) => {
                const v = e.target.value;
                onChange(key, v.includes(".") ? parseFloat(v) : parseInt(v, 10));
              }}
              className="w-32 rounded border border-sparc-gray-300 px-2 py-1 text-xs font-mono focus:border-sparc-purple focus:outline-none"
            />
          ) : value === null ? (
            <span className="text-xs text-sparc-gray-400 italic">null (auto)</span>
          ) : Array.isArray(value) ? (
            <span className="text-xs font-mono text-sparc-gray-600">
              [{value.join(", ")}]
            </span>
          ) : (
            <input
              type="text"
              value={String(value ?? "")}
              onChange={(e) => onChange(key, e.target.value)}
              className="flex-1 rounded border border-sparc-gray-300 px-2 py-1 text-xs font-mono focus:border-sparc-purple focus:outline-none"
            />
          )}
        </div>
      ))}
      {Object.keys(params).length === 0 && (
        <p className="text-xs text-sparc-gray-500 italic">Using defaults</p>
      )}
    </div>
  );
}

export default function ModelsView() {
  const [config, setConfig] = useState<ProjectConfig | null>(null);
  const [models, setModels] = useState<Record<string, Record<string, unknown>>>({});
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    getConfig()
      .then((c) => {
        setConfig(c);
        setModels(c.models ?? {});
      })
      .catch((e) => setError(e.message));
  }, []);

  const updateParam = (section: string, key: string, value: unknown) => {
    setModels((prev) => ({
      ...prev,
      [section]: { ...(prev[section] ?? {}), [key]: value },
    }));
  };

  const save = async () => {
    setSaving(true);
    try {
      await saveConfig({ models });
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

  // Pipeline flags
  const flags = config?.flags ?? {};
  const pipeline = config?.pipeline ?? {};

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold">Models &amp; Pipeline</h2>
          <p className="text-sm text-sparc-gray-600">
            Configure per-model hyperparameters and pipeline settings.
          </p>
        </div>
        <button
          onClick={save}
          disabled={saving}
          className="rounded bg-sparc-purple px-4 py-2 text-sm font-medium text-white hover:bg-sparc-magenta disabled:opacity-50"
        >
          {saving ? "Saving…" : "Save Models"}
        </button>
      </div>

      {/* Pipeline settings summary */}
      <div className="rounded border border-sparc-gray-200 p-4">
        <h3 className="mb-2 text-sm font-semibold">Pipeline Settings</h3>
        <div className="grid grid-cols-4 gap-3 text-xs">
          <div>
            <span className="text-sparc-gray-500">Folds:</span>{" "}
            <span className="font-mono">{pipeline.n_spatial_folds ?? 5}</span>
          </div>
          <div>
            <span className="text-sparc-gray-500">Seed:</span>{" "}
            <span className="font-mono">{pipeline.random_seed ?? 42}</span>
          </div>
          <div>
            <span className="text-sparc-gray-500">Fast mode:</span>{" "}
            <span className="font-mono">{pipeline.fast_mode ? "on" : "off"}</span>
          </div>
          <div>
            <span className="text-sparc-gray-500">Overwrite:</span>{" "}
            <span className="font-mono">{pipeline.overwrite_outputs ? "on" : "off"}</span>
          </div>
        </div>
      </div>

      {/* Feature flags */}
      {Object.keys(flags).length > 0 && (
        <div className="rounded border border-sparc-gray-200 p-4">
          <h3 className="mb-2 text-sm font-semibold">Feature Flags</h3>
          <div className="flex flex-wrap gap-2">
            {Object.entries(flags).map(([key, val]) => (
              <span
                key={key}
                className={`rounded px-2 py-1 text-xs font-mono ${
                  val
                    ? "bg-green-100 text-green-700"
                    : "bg-sparc-gray-100 text-sparc-gray-500"
                }`}
              >
                {key}: {String(val)}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Model sections */}
      <div className="space-y-4">
        {MODEL_SECTIONS.map(({ key, label, description }) => (
          <details key={key} className="rounded border border-sparc-gray-200">
            <summary className="cursor-pointer px-4 py-3 hover:bg-sparc-gray-50">
              <span className="font-semibold text-sm">{label}</span>
              <span className="ml-2 text-xs text-sparc-gray-500">{description}</span>
            </summary>
            <div className="border-t border-sparc-gray-100 px-4 py-3">
              <ParamEditor
                params={models[key] ?? {}}
                onChange={(k, v) => updateParam(key, k, v)}
              />
            </div>
          </details>
        ))}
      </div>

      {error && config && <p className="text-sm text-red-600">{error}</p>}
    </div>
  );
}
