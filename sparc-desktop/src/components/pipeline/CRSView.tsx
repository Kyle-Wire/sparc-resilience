import { useEffect, useState } from "react";
import { getConfig, saveConfig } from "@/lib/api";
import type { ProjectConfig } from "@/lib/types";

const COMMON_CRS = [
  { code: "EPSG:4326", label: "WGS 84 (lat/lon)" },
  { code: "EPSG:3857", label: "Web Mercator" },
  { code: "EPSG:26917", label: "NAD83 / UTM zone 17N" },
  { code: "EPSG:26918", label: "NAD83 / UTM zone 18N" },
  { code: "EPSG:26919", label: "NAD83 / UTM zone 19N" },
  { code: "EPSG:32617", label: "WGS 84 / UTM zone 17N" },
  { code: "EPSG:32618", label: "WGS 84 / UTM zone 18N" },
  { code: "EPSG:32619", label: "WGS 84 / UTM zone 19N" },
  { code: "EPSG:3438", label: "Rhode Island State Plane (NAD83, ft)" },
  { code: "EPSG:2249", label: "Massachusetts State Plane (NAD83, ft)" },
];

export default function CRSView() {
  const [config, setConfig] = useState<ProjectConfig | null>(null);
  const [inputCRS, setInputCRS] = useState("");
  const [projectedCRS, setProjectedCRS] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    getConfig()
      .then((c) => {
        setConfig(c);
        setInputCRS(c.crs?.input ?? "");
        setProjectedCRS(c.crs?.projected ?? "");
      })
      .catch((e) => setError(e.message));
  }, []);

  const save = async () => {
    setSaving(true);
    try {
      await saveConfig({ crs: { input: inputCRS, projected: projectedCRS } });
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
        <h2 className="text-lg font-bold">Coordinate Reference Systems</h2>
        <p className="text-sm text-sparc-gray-600">
          Define the input CRS of your data and the projected CRS for spatial analysis.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Input CRS */}
        <div className="rounded border border-sparc-gray-200 p-4 space-y-3">
          <label className="block text-sm font-semibold">Input CRS</label>
          <p className="text-xs text-sparc-gray-500">
            The coordinate reference system of your raw input data.
          </p>
          <input
            type="text"
            value={inputCRS}
            onChange={(e) => setInputCRS(e.target.value)}
            placeholder="EPSG:4326"
            className="w-full rounded border border-sparc-gray-300 px-3 py-2 text-sm font-mono focus:border-sparc-purple focus:outline-none"
          />
          <div className="flex flex-wrap gap-1">
            {COMMON_CRS.map((c) => (
              <button
                key={c.code}
                onClick={() => setInputCRS(c.code)}
                className={`rounded px-2 py-1 text-xs ${
                  inputCRS === c.code
                    ? "bg-sparc-purple text-white"
                    : "bg-sparc-gray-100 hover:bg-sparc-gray-200"
                }`}
                title={c.label}
              >
                {c.code}
              </button>
            ))}
          </div>
        </div>

        {/* Projected CRS */}
        <div className="rounded border border-sparc-gray-200 p-4 space-y-3">
          <label className="block text-sm font-semibold">Projected CRS</label>
          <p className="text-xs text-sparc-gray-500">
            Target projected CRS for spatial operations (should be in meters).
          </p>
          <input
            type="text"
            value={projectedCRS}
            onChange={(e) => setProjectedCRS(e.target.value)}
            placeholder="EPSG:26919"
            className="w-full rounded border border-sparc-gray-300 px-3 py-2 text-sm font-mono focus:border-sparc-purple focus:outline-none"
          />
          <div className="flex flex-wrap gap-1">
            {COMMON_CRS.map((c) => (
              <button
                key={c.code}
                onClick={() => setProjectedCRS(c.code)}
                className={`rounded px-2 py-1 text-xs ${
                  projectedCRS === c.code
                    ? "bg-sparc-purple text-white"
                    : "bg-sparc-gray-100 hover:bg-sparc-gray-200"
                }`}
                title={c.label}
              >
                {c.code}
              </button>
            ))}
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
          {saving ? "Saving…" : "Save CRS"}
        </button>
        {error && <p className="text-sm text-red-600">{error}</p>}
      </div>
    </div>
  );
}
