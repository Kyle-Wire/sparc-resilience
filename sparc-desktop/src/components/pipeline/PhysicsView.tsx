import { useEffect, useState } from "react";
import { getConfig, saveConfig } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import type { ProjectConfig } from "@/lib/types";

export default function PhysicsView() {
  const [config, setConfig] = useState<ProjectConfig | null>(null);
  const [constraints, setConstraints] = useState<Record<string, number>>({});
  const [priorsFile, setPriorsFile] = useState("");
  const [capsFile, setCapsFile] = useState("");
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
      })
      .catch((e) => setError(e.message));
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
