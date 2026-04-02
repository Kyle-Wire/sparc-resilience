import { useEffect, useState } from "react";
import { getConfig, runScenarios } from "@/lib/api";
import type { ProjectConfig } from "@/lib/types";

export default function ScenariosView() {
  const [config, setConfig] = useState<ProjectConfig | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<{
    status: string;
    n_scenarios: number;
    summary_rows: number;
  } | null>(null);

  useEffect(() => {
    getConfig().then(setConfig).catch((e) => setError(e.message));
  }, []);

  const actionable = config?.causal?.actionable_variables ?? [];
  const fixed = config?.causal?.fixed_variables ?? [];
  const estimator = config?.causal?.estimator ?? "hgb";
  const dagBlend = config?.causal?.dag_blend_weight ?? 0.5;

  const run = async () => {
    setRunning(true);
    setResult(null);
    try {
      const r = await runScenarios();
      setResult(r);
      setError(null);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setRunning(false);
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
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold">Scenarios</h2>
          <p className="text-sm text-sparc-gray-600">
            Run counterfactual scenario simulation using the causal DAG.
          </p>
        </div>
        <button
          onClick={run}
          disabled={running}
          className="rounded bg-sparc-purple px-4 py-2 text-sm font-medium text-white hover:bg-sparc-magenta disabled:opacity-50"
        >
          {running ? "Running…" : "Run Scenarios"}
        </button>
      </div>

      {/* Config summary */}
      <div className="grid grid-cols-2 gap-4">
        <div className="rounded border border-sparc-gray-200 p-4 space-y-2">
          <p className="text-xs font-semibold uppercase text-sparc-gray-500">
            Causal Estimator
          </p>
          <p className="font-mono text-sm">{estimator}</p>
        </div>
        <div className="rounded border border-sparc-gray-200 p-4 space-y-2">
          <p className="text-xs font-semibold uppercase text-sparc-gray-500">
            DAG Blend Weight
          </p>
          <p className="font-mono text-sm">{dagBlend}</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Actionable vars */}
        <div className="rounded border border-sparc-gray-200 p-4">
          <h3 className="mb-2 text-sm font-semibold text-sparc-purple">
            Actionable Variables ({actionable.length})
          </h3>
          {actionable.length === 0 ? (
            <p className="text-xs text-sparc-gray-500">None defined in config</p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {actionable.map((v) => (
                <span
                  key={v}
                  className="rounded bg-sparc-purple/10 px-2 py-1 text-xs font-mono text-sparc-purple"
                >
                  {v}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Fixed vars */}
        <div className="rounded border border-sparc-gray-200 p-4">
          <h3 className="mb-2 text-sm font-semibold text-sparc-gray-600">
            Fixed Variables ({fixed.length})
          </h3>
          {fixed.length === 0 ? (
            <p className="text-xs text-sparc-gray-500">None defined in config</p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {fixed.map((v) => (
                <span
                  key={v}
                  className="rounded bg-sparc-gray-100 px-2 py-1 text-xs font-mono"
                >
                  {v}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Result */}
      {result && (
        <div className="rounded border border-green-300 bg-green-50 p-4 text-sm">
          <p className="font-semibold text-green-800">Scenarios Complete</p>
          <p className="mt-1 text-green-700">
            {result.n_scenarios} scenarios → {result.summary_rows} output rows
          </p>
        </div>
      )}

      {error && config && (
        <p className="text-sm text-red-600">{error}</p>
      )}
    </div>
  );
}
