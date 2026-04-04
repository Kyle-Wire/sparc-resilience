import { useEffect, useState } from "react";
import { getConfig, saveConfig } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import type { ProjectConfig } from "@/lib/types";

const API = "http://localhost:8008";

interface LitPrior {
  variable: string;
  coefficient: number;
  units: string;
  uncertainty: number;
  source_scale: string;
  literature_source: string;
  confidence: string;
}

export default function PhysicsView() {
  const [config, setConfig] = useState<ProjectConfig | null>(null);
  const [constraints, setConstraints] = useState<Record<string, number>>({});
  const [priorsFile, setPriorsFile] = useState("");
  const [capsFile, setCapsFile] = useState("");
  const [litWeight, setLitWeight] = useState(0.5);
  const [bounds, setBounds] = useState<Record<string, { min: number | null; max: number | null }>>({});
  const [litPriors, setLitPriors] = useState<Record<string, LitPrior>>({});
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const { notify } = useNotification();

  useEffect(() => {
    getConfig()
      .then((c) => {
        setConfig(c);
        setConstraints(c.physics?.monotone_constraints ?? {});
        setPriorsFile(c.physics?.priors_file ?? "");
        setCapsFile(c.physics?.caps_file ?? "");
        setLitWeight(c.physics?.literature_weight ?? 0.5);
        setBounds(c.physics?.variable_bounds ?? {});
      })
      .catch((e) => setError(e.message));

    fetch(`${API}/physics/defaults`)
      .then((r) => (r.ok ? r.json() : {}))
      .then(setLitPriors)
      .catch(() => {});
  }, []);

  const predictors = config?.predictors ?? [];

  const setConstraint = (variable: string, value: number) => {
    setConstraints((prev) => ({ ...prev, [variable]: value }));
  };

  const save = async () => {
    setSaving(true);
    try {
      await saveConfig({
        physics: {
          monotone_constraints: constraints,
          priors_file: priorsFile || undefined,
          caps_file: capsFile || undefined,
          literature_weight: litWeight,
          variable_bounds: bounds,
        },
      });
      setError(null);
      notify("success", "Physics constraints saved");
    } catch (e: any) {
      notify("error", e.message);
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
        <h2 className="text-lg font-bold">Physics / Domain Constraints</h2>
        <p className="text-sm text-sparc-gray-600">
          Set monotonicity constraints and point to priors/caps files for domain knowledge integration.
        </p>
      </div>

      {/* Literature Calibration Weight */}
      <div>
        <h3 className="mb-1 text-sm font-semibold">Literature Weight</h3>
        <p className="mb-2 text-xs text-sparc-gray-500">
          Blend between literature priors and data-driven coefficients. 0 = fully data-driven, 1 = fully literature.
        </p>
        <div className="flex items-center gap-3">
          <span className="text-xs text-sparc-gray-500 w-12">Data</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={litWeight}
            onChange={(e) => setLitWeight(parseFloat(e.target.value))}
            className="flex-1 accent-sparc-purple"
          />
          <span className="text-xs text-sparc-gray-500 w-12 text-right">Lit.</span>
          <span className="w-12 text-center font-mono text-sm">{litWeight.toFixed(2)}</span>
        </div>
      </div>

      {/* Literature Priors Reference */}
      {Object.keys(litPriors).length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold">Literature Priors (Reference)</h3>
          <div className="overflow-auto rounded border border-sparc-gray-200">
            <table className="w-full text-xs">
              <thead className="bg-sparc-gray-50 text-left">
                <tr>
                  <th className="px-3 py-2 font-medium">Variable</th>
                  <th className="px-3 py-2 font-medium">Coefficient</th>
                  <th className="px-3 py-2 font-medium">Units</th>
                  <th className="px-3 py-2 font-medium">±</th>
                  <th className="px-3 py-2 font-medium">Confidence</th>
                  <th className="px-3 py-2 font-medium">Source</th>
                </tr>
              </thead>
              <tbody>
                {Object.values(litPriors).map((p) => (
                  <tr key={p.variable} className="border-t border-sparc-gray-100">
                    <td className="px-3 py-2 font-mono">{p.variable}</td>
                    <td className={`px-3 py-2 font-mono ${p.coefficient < 0 ? "text-blue-600" : "text-red-600"}`}>
                      {p.coefficient > 0 ? "+" : ""}{p.coefficient.toFixed(3)}
                    </td>
                    <td className="px-3 py-2">{p.units}</td>
                    <td className="px-3 py-2">{(p.uncertainty * 100).toFixed(0)}%</td>
                    <td className="px-3 py-2">
                      <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                        p.confidence === "high" ? "bg-green-100 text-green-700" :
                        p.confidence === "medium" ? "bg-yellow-100 text-yellow-700" :
                        "bg-red-100 text-red-700"
                      }`}>{p.confidence}</span>
                    </td>
                    <td className="px-3 py-2 text-sparc-gray-500 max-w-[280px] truncate" title={p.literature_source}>
                      {p.literature_source}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Variable Bounds */}
      <div>
        <h3 className="mb-1 text-sm font-semibold">Variable Bounds</h3>
        <p className="mb-2 text-xs text-sparc-gray-500">
          Physical min/max limits for scenario clamping. Leave blank for unconstrained.
        </p>
        <div className="overflow-auto rounded border border-sparc-gray-200">
          <table className="w-full text-sm">
            <thead className="bg-sparc-gray-50 text-left">
              <tr>
                <th className="px-3 py-2 font-medium">Variable</th>
                <th className="px-3 py-2 font-medium">Min</th>
                <th className="px-3 py-2 font-medium">Max</th>
              </tr>
            </thead>
            <tbody>
              {predictors.map((v) => {
                const b = bounds[v] ?? { min: null, max: null };
                return (
                  <tr key={v} className="border-t border-sparc-gray-100">
                    <td className="px-3 py-2 font-mono text-xs">{v}</td>
                    <td className="px-3 py-2">
                      <input
                        type="number"
                        value={b.min ?? ""}
                        placeholder="—"
                        onChange={(e) => {
                          const val = e.target.value === "" ? null : parseFloat(e.target.value);
                          setBounds((prev) => ({ ...prev, [v]: { ...prev[v], min: val, max: prev[v]?.max ?? null } }));
                        }}
                        className="w-24 rounded border border-sparc-gray-300 px-2 py-1 text-xs font-mono focus:border-sparc-purple focus:outline-none"
                      />
                    </td>
                    <td className="px-3 py-2">
                      <input
                        type="number"
                        value={b.max ?? ""}
                        placeholder="—"
                        onChange={(e) => {
                          const val = e.target.value === "" ? null : parseFloat(e.target.value);
                          setBounds((prev) => ({ ...prev, [v]: { min: prev[v]?.min ?? null, max: val } }));
                        }}
                        className="w-24 rounded border border-sparc-gray-300 px-2 py-1 text-xs font-mono focus:border-sparc-purple focus:outline-none"
                      />
                    </td>
                  </tr>
                );
              })}
              {predictors.length === 0 && (
                <tr>
                  <td colSpan={3} className="px-3 py-4 text-center text-sparc-gray-500 text-xs">
                    No predictors defined. Add variables on the Variables page.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* File references */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-semibold mb-1">Priors File</label>
          <input
            type="text"
            value={priorsFile}
            onChange={(e) => setPriorsFile(e.target.value)}
            placeholder="physics/priors.yml"
            className="w-full rounded border border-sparc-gray-300 px-3 py-2 text-sm font-mono focus:border-sparc-purple focus:outline-none"
          />
        </div>
        <div>
          <label className="block text-sm font-semibold mb-1">Caps File</label>
          <input
            type="text"
            value={capsFile}
            onChange={(e) => setCapsFile(e.target.value)}
            placeholder="physics/caps.yml"
            className="w-full rounded border border-sparc-gray-300 px-3 py-2 text-sm font-mono focus:border-sparc-purple focus:outline-none"
          />
        </div>
      </div>

      {/* Monotone constraints */}
      <div>
        <h3 className="mb-2 text-sm font-semibold">Monotone Constraints</h3>
        <p className="mb-3 text-xs text-sparc-gray-500">
          −1 = decreasing effect (cooling), +1 = increasing (warming), 0 = unconstrained
        </p>
        <div className="overflow-auto rounded border border-sparc-gray-200">
          <table className="w-full text-sm">
            <thead className="bg-sparc-gray-50 text-left">
              <tr>
                <th className="px-3 py-2 font-medium">Variable</th>
                <th className="px-3 py-2 font-medium text-center">−1</th>
                <th className="px-3 py-2 font-medium text-center">0</th>
                <th className="px-3 py-2 font-medium text-center">+1</th>
              </tr>
            </thead>
            <tbody>
              {predictors.map((v) => {
                const val = constraints[v] ?? 0;
                return (
                  <tr key={v} className="border-t border-sparc-gray-100">
                    <td className="px-3 py-2 font-mono text-xs">{v}</td>
                    {([-1, 0, 1] as const).map((n) => (
                      <td key={n} className="px-3 py-2 text-center">
                        <button
                          onClick={() => setConstraint(v, n)}
                          className={`h-6 w-6 rounded-full text-xs font-bold ${
                            val === n
                              ? n === -1
                                ? "bg-blue-500 text-white"
                                : n === 1
                                ? "bg-red-500 text-white"
                                : "bg-sparc-gray-700 text-white"
                              : "bg-sparc-gray-100 text-sparc-gray-400 hover:bg-sparc-gray-200"
                          }`}
                        >
                          {n === -1 ? "−" : n === 1 ? "+" : "0"}
                        </button>
                      </td>
                    ))}
                  </tr>
                );
              })}
              {predictors.length === 0 && (
                <tr>
                  <td colSpan={4} className="px-3 py-4 text-center text-sparc-gray-500 text-xs">
                    No predictors defined. Add variables on the Variables page.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Save */}
      <div className="flex items-center gap-3">
        <button
          onClick={save}
          disabled={saving}
          className="rounded bg-sparc-purple px-4 py-2 text-sm font-medium text-white hover:bg-sparc-magenta disabled:opacity-50"
        >
          {saving ? "Saving…" : "Save Physics"}
        </button>
      </div>
    </div>
  );
}
