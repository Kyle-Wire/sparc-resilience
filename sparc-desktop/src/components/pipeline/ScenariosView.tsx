import { useEffect, useState, useCallback } from "react";
import { getConfig, saveConfig, runScenarios, dataSummary } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import type { ProjectConfig, DataSummary } from "@/lib/types";

interface Scenario {
  name: string;
  variable: string;
  direction?: "increase" | "decrease";
  increments: number[];
  unit?: string;
}

interface InteractionLeg {
  variable: string;
  delta: number;
}

interface InteractionScenario {
  name: string;
  legs: InteractionLeg[];
}

const ESTIMATOR_OPTIONS = [
  {
    value: "dml",
    label: "DML",
    description: "Debiased Machine Learning — K-fold cross-fitting for unbiased causal estimates",
  },
  {
    value: "hgb",
    label: "HGB",
    description: "Histogram Gradient Boosting — Non-linear treatment effect estimation",
  },
  {
    value: "ols",
    label: "OLS",
    description: "Ordinary Least Squares — Linear backdoor-adjusted causal estimation",
  },
] as const;

export default function ScenariosView() {
  const [config, setConfig] = useState<ProjectConfig | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [interactions, setInteractions] = useState<InteractionScenario[]>([]);
  const [result, setResult] = useState<{
    status: string;
    n_scenarios: number;
    summary_rows: number;
  } | null>(null);
  const { notify } = useNotification();

  useEffect(() => {
    getConfig()
      .then((c) => {
        setConfig(c);
        setScenarios((c.scenarios ?? []) as Scenario[]);
        setInteractions(((c as any).interaction_scenarios ?? []) as InteractionScenario[]);
      })
      .catch((e) => setError(e.message));
    dataSummary().then(setSummary).catch(() => {});
  }, []);

  const actionable = config?.causal?.actionable_variables ?? [];
  const fixed = config?.causal?.fixed_variables ?? [];
  const estimator = config?.causal?.estimator ?? "dml";
  const dagBlend = config?.causal?.dag_blend_weight ?? 0.5;
  const numericCols = summary ? Object.keys(summary.numeric_summary) : [];

  const handleEstimatorChange = async (value: string) => {
    try {
      await saveConfig({ causal: { ...config?.causal, estimator: value } });
      setConfig((prev) =>
        prev ? { ...prev, causal: { ...prev.causal, estimator: value } } : prev,
      );
    } catch (e: any) {
      setError(e.message);
    }
  };

  const addScenario = () => {
    setScenarios((prev) => [
      ...prev,
      {
        name: `Scenario ${prev.length + 1}`,
        variable: actionable[0] ?? numericCols[0] ?? "",
        direction: "increase",
        increments: [1, 2, 5],
      },
    ]);
  };

  const removeScenario = (idx: number) => {
    setScenarios((prev) => prev.filter((_, i) => i !== idx));
  };

  const updateScenario = (idx: number, patch: Partial<Scenario>) => {
    setScenarios((prev) =>
      prev.map((s, i) => (i === idx ? { ...s, ...patch } : s))
    );
  };

  const addInteraction = () => {
    const vars = actionable.length > 0 ? actionable : numericCols;
    setInteractions((prev) => [
      ...prev,
      {
        name: `Interaction ${prev.length + 1}`,
        legs: vars.slice(0, 2).map((v) => ({ variable: v, delta: 1 })),
      },
    ]);
  };

  const removeInteraction = (idx: number) =>
    setInteractions((prev) => prev.filter((_, i) => i !== idx));

  const updateInteraction = (idx: number, patch: Partial<InteractionScenario>) =>
    setInteractions((prev) => prev.map((s, i) => (i === idx ? { ...s, ...patch } : s)));

  const addLeg = (idx: number) => {
    const vars = actionable.length > 0 ? actionable : numericCols;
    setInteractions((prev) =>
      prev.map((s, i) =>
        i === idx ? { ...s, legs: [...s.legs, { variable: vars[0] ?? "", delta: 1 }] } : s
      )
    );
  };

  const removeLeg = (scIdx: number, legIdx: number) =>
    setInteractions((prev) =>
      prev.map((s, i) =>
        i === scIdx ? { ...s, legs: s.legs.filter((_, j) => j !== legIdx) } : s
      )
    );

  const updateLeg = (scIdx: number, legIdx: number, patch: Partial<InteractionLeg>) =>
    setInteractions((prev) =>
      prev.map((s, i) =>
        i === scIdx
          ? { ...s, legs: s.legs.map((l, j) => (j === legIdx ? { ...l, ...patch } : l)) }
          : s
      )
    );

  const saveScenarios = useCallback(async () => {
    try {
      await saveConfig({ scenarios, interaction_scenarios: interactions });
      notify("success", "Scenarios saved");
    } catch (e: any) {
      notify("error", e.message);
    }
  }, [scenarios, interactions, notify]);

  const run = async () => {
    await saveScenarios();
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
    <div className="p-6 space-y-6 max-w-4xl">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold">Scenarios</h2>
          <p className="text-sm text-sparc-gray-600">
            Build and run counterfactual scenario simulations using the causal DAG.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={saveScenarios}
            className="rounded border border-sparc-gray-300 px-4 py-2 text-sm font-medium hover:bg-sparc-gray-100"
          >
            Save
          </button>
          <button
            onClick={run}
            disabled={running || scenarios.length === 0}
            className="rounded bg-sparc-purple px-4 py-2 text-sm font-medium text-white hover:bg-sparc-magenta disabled:opacity-50"
          >
            {running ? "Running…" : "Run Scenarios"}
          </button>
        </div>
      </div>

      {/* Config summary */}
      <div className="grid grid-cols-2 gap-4">
        <div className="rounded border border-sparc-gray-200 p-4 space-y-2">
          <p className="text-xs font-semibold uppercase text-sparc-gray-500">
            Causal Estimator
          </p>
          <select
            value={estimator}
            onChange={(e) => handleEstimatorChange(e.target.value)}
            className="w-full rounded border border-sparc-gray-200 px-2 py-1.5 text-sm font-mono focus:border-sparc-purple focus:outline-none"
          >
            {ESTIMATOR_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label} — {opt.description}
              </option>
            ))}
          </select>
        </div>
        <div className="rounded border border-sparc-gray-200 p-4 space-y-2">
          <p className="text-xs font-semibold uppercase text-sparc-gray-500">
            DAG Blend Weight
          </p>
          <p className="font-mono text-sm">{dagBlend}</p>
        </div>
      </div>

      {/* Actionable / Fixed */}
      <div className="grid grid-cols-2 gap-4">
        <div className="rounded border border-sparc-gray-200 p-4">
          <h3 className="mb-2 text-sm font-semibold text-sparc-purple">
            Actionable Variables ({actionable.length})
          </h3>
          {actionable.length === 0 ? (
            <p className="text-xs text-sparc-gray-500">None defined in config</p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {actionable.map((v) => (
                <span key={v} className="rounded bg-sparc-purple/10 px-2 py-1 text-xs font-mono text-sparc-purple">{v}</span>
              ))}
            </div>
          )}
        </div>
        <div className="rounded border border-sparc-gray-200 p-4">
          <h3 className="mb-2 text-sm font-semibold text-sparc-gray-600">
            Fixed Variables ({fixed.length})
          </h3>
          {fixed.length === 0 ? (
            <p className="text-xs text-sparc-gray-500">None defined in config</p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {fixed.map((v) => (
                <span key={v} className="rounded bg-sparc-gray-100 px-2 py-1 text-xs font-mono">{v}</span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Scenario Builder */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold">Intervention Scenarios ({scenarios.length})</h3>
          <button
            onClick={addScenario}
            className="rounded border border-sparc-gray-300 px-3 py-1.5 text-xs font-medium hover:bg-sparc-gray-100"
          >
            + Add Scenario
          </button>
        </div>

        {scenarios.length === 0 && (
          <div className="rounded border-2 border-dashed border-sparc-gray-300 p-8 text-center">
            <p className="text-sm text-sparc-gray-500">
              No scenarios defined. Click "Add Scenario" to create an intervention.
            </p>
          </div>
        )}

        {scenarios.map((s, idx) => (
          <div key={idx} className="rounded border border-sparc-gray-200 p-4 space-y-3">
            <div className="flex items-center gap-3">
              <input
                value={s.name}
                onChange={(e) => updateScenario(idx, { name: e.target.value })}
                className="flex-1 rounded border border-sparc-gray-300 px-2 py-1 text-sm font-medium focus:border-sparc-purple focus:outline-none"
                placeholder="Scenario name"
              />
              <button
                onClick={() => removeScenario(idx)}
                className="text-xs text-red-500 hover:underline"
              >
                Remove
              </button>
            </div>

            <div className="grid grid-cols-3 gap-3">
              {/* Variable */}
              <div>
                <label className="mb-1 block text-xs text-sparc-gray-600">Variable</label>
                <select
                  value={s.variable}
                  onChange={(e) => updateScenario(idx, { variable: e.target.value })}
                  className="w-full rounded border border-sparc-gray-300 px-2 py-1.5 text-xs font-mono focus:border-sparc-purple focus:outline-none"
                >
                  {(actionable.length > 0 ? actionable : numericCols).map((v) => (
                    <option key={v} value={v}>{v}</option>
                  ))}
                </select>
              </div>

              {/* Direction */}
              <div>
                <label className="mb-1 block text-xs text-sparc-gray-600">Direction</label>
                <select
                  value={s.direction ?? "increase"}
                  onChange={(e) => updateScenario(idx, { direction: e.target.value as "increase" | "decrease" })}
                  className="w-full rounded border border-sparc-gray-300 px-2 py-1.5 text-xs focus:border-sparc-purple focus:outline-none"
                >
                  <option value="increase">Increase</option>
                  <option value="decrease">Decrease</option>
                </select>
              </div>

              {/* Unit */}
              <div>
                <label className="mb-1 block text-xs text-sparc-gray-600">Unit (optional)</label>
                <input
                  value={s.unit ?? ""}
                  onChange={(e) => updateScenario(idx, { unit: e.target.value })}
                  placeholder="e.g., °C, mm, %"
                  className="w-full rounded border border-sparc-gray-300 px-2 py-1.5 text-xs focus:border-sparc-purple focus:outline-none"
                />
              </div>
            </div>

            {/* Increments */}
            <div>
              <label className="mb-1 block text-xs text-sparc-gray-600">
                Increments (comma-separated)
              </label>
              <div className="flex items-center gap-2">
                <input
                  value={s.increments.join(", ")}
                  onChange={(e) => {
                    const vals = e.target.value
                      .split(",")
                      .map((v) => parseFloat(v.trim()))
                      .filter((v) => !isNaN(v));
                    updateScenario(idx, { increments: vals });
                  }}
                  placeholder="1, 2, 5, 10"
                  className="flex-1 rounded border border-sparc-gray-300 px-2 py-1.5 text-xs font-mono focus:border-sparc-purple focus:outline-none"
                />
                {summary?.numeric_summary[s.variable] && (
                  <span className="text-[10px] text-sparc-gray-500 whitespace-nowrap">
                    range: {summary.numeric_summary[s.variable].min.toFixed(1)} – {summary.numeric_summary[s.variable].max.toFixed(1)}
                  </span>
                )}
              </div>
            </div>

            {/* Quick slider for largest increment */}
            {summary?.numeric_summary[s.variable] && (
              <div>
                <label className="mb-1 block text-xs text-sparc-gray-600">
                  Max increment: {s.increments[s.increments.length - 1] ?? 0}
                </label>
                <input
                  type="range"
                  min={0}
                  max={Math.abs(summary.numeric_summary[s.variable].max - summary.numeric_summary[s.variable].min)}
                  step={Math.abs(summary.numeric_summary[s.variable].max - summary.numeric_summary[s.variable].min) / 100}
                  value={Math.abs(s.increments[s.increments.length - 1] ?? 0)}
                  onChange={(e) => {
                    const maxVal = parseFloat(e.target.value);
                    const steps = s.increments.length || 3;
                    const newInc = Array.from({ length: steps }, (_, i) =>
                      parseFloat(((maxVal * (i + 1)) / steps).toFixed(2))
                    );
                    updateScenario(idx, { increments: newInc });
                  }}
                  className="w-full accent-sparc-purple"
                />
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Interaction (Multi-Variable) Scenarios */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold">Interaction Scenarios ({interactions.length})</h3>
            <p className="text-xs text-sparc-gray-500">Simultaneous changes to multiple variables</p>
          </div>
          <button
            onClick={addInteraction}
            className="rounded border border-sparc-gray-300 px-3 py-1.5 text-xs font-medium hover:bg-sparc-gray-100"
          >
            + Add Interaction
          </button>
        </div>

        {interactions.length === 0 && (
          <div className="rounded border-2 border-dashed border-sparc-gray-300 p-6 text-center">
            <p className="text-sm text-sparc-gray-500">
              No interaction scenarios. Add one to model simultaneous multi-variable changes.
            </p>
          </div>
        )}

        {interactions.map((ix, idx) => (
          <div key={idx} className="rounded border border-sparc-gray-200 p-4 space-y-3">
            <div className="flex items-center gap-3">
              <input
                value={ix.name}
                onChange={(e) => updateInteraction(idx, { name: e.target.value })}
                className="flex-1 rounded border border-sparc-gray-300 px-2 py-1 text-sm font-medium focus:border-sparc-purple focus:outline-none"
                placeholder="Interaction name"
              />
              <button onClick={() => removeInteraction(idx)} className="text-xs text-red-500 hover:underline">
                Remove
              </button>
            </div>

            <div className="space-y-2">
              {ix.legs.map((leg, li) => (
                <div key={li} className="flex items-center gap-2 pl-3 border-l-2 border-sparc-purple/30">
                  <select
                    value={leg.variable}
                    onChange={(e) => updateLeg(idx, li, { variable: e.target.value })}
                    className="rounded border border-sparc-gray-300 px-2 py-1 text-xs font-mono focus:border-sparc-purple focus:outline-none"
                  >
                    {(actionable.length > 0 ? actionable : numericCols).map((v) => (
                      <option key={v} value={v}>{v}</option>
                    ))}
                  </select>
                  <span className="text-xs text-sparc-gray-500">Δ</span>
                  <input
                    type="number"
                    value={leg.delta}
                    onChange={(e) => updateLeg(idx, li, { delta: parseFloat(e.target.value) || 0 })}
                    className="w-24 rounded border border-sparc-gray-300 px-2 py-1 text-xs font-mono focus:border-sparc-purple focus:outline-none"
                  />
                  {ix.legs.length > 1 && (
                    <button onClick={() => removeLeg(idx, li)} className="text-xs text-red-400 hover:text-red-600">✕</button>
                  )}
                </div>
              ))}
            </div>

            <button
              onClick={() => addLeg(idx)}
              className="text-xs text-sparc-purple hover:underline"
            >
              + Add variable
            </button>
          </div>
        ))}
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
