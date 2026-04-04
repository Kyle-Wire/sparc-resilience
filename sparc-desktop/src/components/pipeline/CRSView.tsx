import { useEffect, useState, useMemo } from "react";
import { getConfig, saveConfig, dataSummary, dataGeoJson } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import SpatialMap from "@/components/map/SpatialMap";
import type { ProjectConfig, DataSummary, GeoJsonData } from "@/lib/types";

interface CRSMeta {
  code: string;
  label: string;
  type: string;
  units: string;
  centerLng: number;
  centerLat: number;
}

const COMMON_CRS: CRSMeta[] = [
  { code: "EPSG:4326", label: "WGS 84 (lat/lon)", type: "Geographic", units: "degrees", centerLng: 0, centerLat: 0 },
  { code: "EPSG:3857", label: "Web Mercator", type: "Projected (Mercator)", units: "meters", centerLng: 0, centerLat: 0 },
  { code: "EPSG:26917", label: "NAD83 / UTM zone 17N", type: "UTM Zone 17N", units: "meters", centerLng: -81, centerLat: 28 },
  { code: "EPSG:26918", label: "NAD83 / UTM zone 18N", type: "UTM Zone 18N", units: "meters", centerLng: -75, centerLat: 38 },
  { code: "EPSG:26919", label: "NAD83 / UTM zone 19N", type: "UTM Zone 19N", units: "meters", centerLng: -69, centerLat: 42 },
  { code: "EPSG:32617", label: "WGS 84 / UTM zone 17N", type: "UTM Zone 17N", units: "meters", centerLng: -81, centerLat: 28 },
  { code: "EPSG:32618", label: "WGS 84 / UTM zone 18N", type: "UTM Zone 18N", units: "meters", centerLng: -75, centerLat: 38 },
  { code: "EPSG:32619", label: "WGS 84 / UTM zone 19N", type: "UTM Zone 19N", units: "meters", centerLng: -69, centerLat: 42 },
  { code: "EPSG:3438", label: "Rhode Island State Plane (NAD83, ft)", type: "State Plane", units: "feet", centerLng: -71.5, centerLat: 41.7 },
  { code: "EPSG:2249", label: "Massachusetts State Plane (NAD83, ft)", type: "State Plane", units: "feet", centerLng: -71.8, centerLat: 42.3 },
];

export default function CRSView() {
  const [config, setConfig] = useState<ProjectConfig | null>(null);
  const [inputCRS, setInputCRS] = useState("");
  const [projectedCRS, setProjectedCRS] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [geojson, setGeojson] = useState<GeoJsonData | null>(null);
  const { notify } = useNotification();

  useEffect(() => {
    getConfig()
      .then((c) => {
        setConfig(c);
        setInputCRS(c.crs?.input ?? "");
        setProjectedCRS(c.crs?.projected ?? "");
      })
      .catch((e) => setError(e.message));
    dataSummary().then(setSummary).catch(() => {});
    dataGeoJson().then(setGeojson).catch(() => {});
  }, []);

  // CRS metadata for info panel
  const projectedMeta = useMemo(
    () => COMMON_CRS.find((c) => c.code === projectedCRS),
    [projectedCRS]
  );

  const save = async () => {
    setSaving(true);
    try {
      await saveConfig({ crs: { input: inputCRS, projected: projectedCRS } });
      setError(null);
      notify("success", "CRS saved");
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

      {/* Map + CRS Info */}
      <div className="grid grid-cols-1 gap-6">
        <div className="rounded border border-sparc-gray-200 overflow-hidden" style={{ height: 360 }}>
          <SpatialMap geojson={geojson as any} height="360px" />
        </div>

        <div className="rounded border border-sparc-gray-200 p-4 space-y-2 text-sm">
          <h3 className="font-semibold text-sm">CRS Insights</h3>
          {projectedMeta ? (
            <>
              <p><span className="font-medium">Name:</span> {projectedMeta.label}</p>
              <p><span className="font-medium">Projection Type:</span> {projectedMeta.type}</p>
              <p><span className="font-medium">Units:</span> {projectedMeta.units}</p>
              <p><span className="font-medium">Zone Center:</span> {projectedMeta.centerLng.toFixed(1)}°, {projectedMeta.centerLat.toFixed(1)}°</p>
            </>
          ) : projectedCRS ? (
            <p className="text-sparc-gray-600">
              Custom CRS: <span className="font-mono">{projectedCRS}</span>. Select a common CRS above for detailed metadata.
            </p>
          ) : (
            <p className="text-sparc-gray-600">Select a projected CRS to see details.</p>
          )}
          {summary?.bbox && (
            <div className="border-t border-sparc-gray-200 pt-2 text-xs text-sparc-gray-600">
              <p className="font-medium text-sparc-gray-800">Data Bounding Box</p>
              <p>
                ({summary.bbox.minx.toFixed(4)}, {summary.bbox.miny.toFixed(4)}) →
                ({summary.bbox.maxx.toFixed(4)}, {summary.bbox.maxy.toFixed(4)})
              </p>
            </div>
          )}
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
      </div>
    </div>
  );
}
