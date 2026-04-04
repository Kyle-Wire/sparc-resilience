import { useEffect, useState } from "react";
import { getConfig, saveConfig } from "@/lib/api";
import type { ProjectConfig } from "@/lib/types";

/** Model names with human-readable labels. */
const MODEL_SECTIONS: { key: string; label: string; description: string }[] = [
  { key: "ols", label: "OLS", description: "Ordinary Least Squares (baseline)" },
  { key: "gwr", label: "GWR", description: "Geographically Weighted Regression" },
  { key: "gwrf", label: "GWRF", description: "Geographically Weighted Random Forest" },
  { key: "ggpgam", label: "GGPGAM", description: "Geographically-Guided Penalised GAM" },
  { key: "meta_ensemble", label: "Meta-Ensemble", description: "Stacked model ensemble (LightGBM default)" },
  { key: "deep_kriging", label: "Deep Kriging", description: "Neural residual correction with spatial basis" },
];

const SPATIAL_CV_METHODS = [
  { value: "buffered_block", label: "Buffered Block", description: "Spatially-aware block CV with buffer zones (default)" },
  { value: "spatial_loo", label: "Spatial LOO", description: "Leave-one-out with spatial exclusion buffer" },
  { value: "random_block", label: "Random Block", description: "Randomised spatial blocks without buffering" },
  { value: "spatial_kfold", label: "Spatial K-Fold", description: "K-means clustering of coordinates into folds" },
] as const;

const GWEN_PARAMS_SCHEMA: { key: string; label: string; type: "number" | "select"; options?: string[] }[] = [
  { key: "n_neighbors", label: "K-Neighbors", type: "number" },
  { key: "selection_threshold", label: "Selection Threshold", type: "number" },
  { key: "max_features", label: "Max Features", type: "number" },
  { key: "n_alphas", label: "N Alphas", type: "number" },
  { key: "cv_folds", label: "CV Folds", type: "number" },
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
  const [enabledModels, setEnabledModels] = useState<Record<string, boolean>>({});
  const [gwenConfig, setGwenConfig] = useState<Record<string, unknown>>({});
  const [cvMethod, setCvMethod] = useState("buffered_block");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    getConfig()
      .then((c) => {
        setConfig(c);
        setModels(c.models ?? {});
        // Initialize enabled state — default all to true
        const enabled: Record<string, boolean> = {};
        for (const { key } of MODEL_SECTIONS) {
          enabled[key] = c.models?.[key]?.enabled !== false;
        }
        setEnabledModels(enabled);
        setGwenConfig(c.gwen ?? {});
        setCvMethod(c.models?.spatial_cv?.method ?? c.pipeline?.cv_method ?? "buffered_block");
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
      // Merge enabled flags into each model section
      const mergedModels = { ...models };
      for (const { key } of MODEL_SECTIONS) {
        mergedModels[key] = { ...(mergedModels[key] ?? {}), enabled: enabledModels[key] !== false };
      }
      mergedModels.spatial_cv = { ...(mergedModels.spatial_cv ?? {}), method: cvMethod };
      await saveConfig({ models: mergedModels, gwen: gwenConfig });
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

      {/* GWEN Variable Selection */}
      <div className="rounded border border-sparc-gray-200">
        <div className="px-4 py-3 bg-sparc-gray-50 border-b border-sparc-gray-100">
          <span className="font-semibold text-sm">GWEN Variable Selection</span>
          <span className="ml-2 text-xs text-sparc-gray-500">Geographically Weighted Elastic Net — feature selection</span>
        </div>
        <div className="px-4 py-3 space-y-2">
          {GWEN_PARAMS_SCHEMA.map(({ key, label, type }) => (
            <div key={key} className="flex items-center gap-3">
              <label className="w-40 shrink-0 text-xs font-mono text-sparc-gray-600">{label}</label>
              {type === "number" ? (
                <input
                  type="number"
                  value={gwenConfig[key] != null ? Number(gwenConfig[key]) : ""}
                  placeholder="auto"
                  onChange={(e) => {
                    const v = e.target.value;
                    setGwenConfig((prev) => ({
                      ...prev,
                      [key]: v === "" ? null : v.includes(".") ? parseFloat(v) : parseInt(v, 10),
                    }));
                  }}
                  className="w-32 rounded border border-sparc-gray-300 px-2 py-1 text-xs font-mono focus:border-sparc-purple focus:outline-none"
                />
              ) : null}
            </div>
          ))}
        </div>
      </div>

      {/* Spatial CV Method */}
      <div className="rounded border border-sparc-gray-200 p-4">
        <h3 className="mb-2 text-sm font-semibold">Spatial Cross-Validation</h3>
        <div className="space-y-2">
          {SPATIAL_CV_METHODS.map((m) => (
            <label key={m.value} className="flex items-center gap-3 cursor-pointer">
              <input
                type="radio"
                name="cv_method"
                value={m.value}
                checked={cvMethod === m.value}
                onChange={() => setCvMethod(m.value)}
                className="accent-sparc-purple"
              />
              <span className="text-sm font-medium">{m.label}</span>
              <span className="text-xs text-sparc-gray-500">{m.description}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Model sections with toggles */}
      <div className="space-y-4">
        {MODEL_SECTIONS.map(({ key, label, description }) => {
          const enabled = enabledModels[key] !== false;
          return (
            <details key={key} className={`rounded border ${enabled ? "border-sparc-gray-200" : "border-sparc-gray-100 opacity-60"}`}>
              <summary className="cursor-pointer px-4 py-3 hover:bg-sparc-gray-50 flex items-center justify-between">
                <div>
                  <span className="font-semibold text-sm">{label}</span>
                  <span className="ml-2 text-xs text-sparc-gray-500">{description}</span>
                </div>
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    setEnabledModels((prev) => ({ ...prev, [key]: !prev[key] }));
                  }}
                  className={`ml-4 shrink-0 rounded px-3 py-1 text-xs font-medium transition-colors ${
                    enabled
                      ? "bg-sparc-purple text-white"
                      : "bg-sparc-gray-200 text-sparc-gray-500"
                  }`}
                >
                  {enabled ? "ON" : "OFF"}
                </button>
              </summary>
              <div className="border-t border-sparc-gray-100 px-4 py-3">
                <ParamEditor
                  params={models[key] ?? {}}
                  onChange={(k, v) => updateParam(key, k, v)}
                />
              </div>
            </details>
          );
        })}
      </div>

      {error && config && <p className="text-sm text-red-600">{error}</p>}
    </div>
  );
}
