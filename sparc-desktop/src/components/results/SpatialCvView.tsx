import { useState, useEffect, useMemo } from "react";
import { getSpatialCvPredictions } from "@/lib/api";
import SpatialMap, { type MapMode } from "@/components/map/SpatialMap";
import type { GeoJsonData } from "@/lib/types";

const MAP_MODES: { value: MapMode; label: string }[] = [
  { value: "scatter", label: "Scatter" },
  { value: "heatmap", label: "Heatmap" },
  { value: "choropleth", label: "Choropleth" },
];

export default function SpatialCvView() {
  const [geojson, setGeojson] = useState<GeoJsonData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [colorField, setColorField] = useState("");
  const [mapMode, setMapMode] = useState<MapMode>("scatter");

  useEffect(() => {
    getSpatialCvPredictions()
      .then((g) => {
        if (g?.type === "FeatureCollection") setGeojson(g);
        else setError("Unexpected data format");
      })
      .catch((e) => setError(e.message));
  }, []);

  // Extract available numeric columns from features
  const fields = useMemo(() => {
    if (!geojson?.features?.length) return [];
    const props = geojson.features[0].properties;
    return Object.keys(props).filter(
      (k) => typeof props[k] === "number" && k !== "identifier",
    );
  }, [geojson]);

  // Auto-select first field, preferring 'observed'
  useEffect(() => {
    if (fields.length > 0 && !colorField) {
      const preferred = fields.find((f) => f === "observed") ??
        fields.find((f) => f.startsWith("pred_")) ??
        fields[0];
      setColorField(preferred);
    }
  }, [fields, colorField]);

  // Compute per-model performance metrics from the geojson features
  const modelMetrics = useMemo(() => {
    if (!geojson?.features?.length) return null;
    const features = geojson.features;
    const modelCols = fields.filter((f) => f.startsWith("pred_"));
    const hasObserved = fields.includes("observed");
    if (!hasObserved || modelCols.length === 0) return null;

    const metrics: Record<string, { r2: number; rmse: number }> = {};
    for (const col of modelCols) {
      let ss_res = 0, ss_tot = 0, mse = 0, n = 0;
      let sumY = 0;
      const vals: { obs: number; pred: number }[] = [];
      for (const f of features) {
        const obs = Number(f.properties["observed"]);
        const pred = Number(f.properties[col]);
        if (isFinite(obs) && isFinite(pred)) {
          vals.push({ obs, pred });
          sumY += obs;
        }
      }
      const meanY = sumY / vals.length;
      for (const { obs, pred } of vals) {
        ss_res += (obs - pred) ** 2;
        ss_tot += (obs - meanY) ** 2;
        mse += (obs - pred) ** 2;
        n++;
      }
      if (n > 0 && ss_tot > 0) {
        metrics[col.replace("pred_", "").toUpperCase()] = {
          r2: 1 - ss_res / ss_tot,
          rmse: Math.sqrt(mse / n),
        };
      }
    }
    return Object.keys(metrics).length > 0 ? metrics : null;
  }, [geojson, fields]);

  if (error) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-sparc-gray-600">{error}</p>
      </div>
    );
  }

  if (!geojson) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-sparc-gray-500">Loading spatial CV predictions…</p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      {/* Model performance metrics */}
      {modelMetrics && (
        <div className="flex gap-3 overflow-x-auto border-b border-sparc-gray-100 px-4 py-3">
          {Object.entries(modelMetrics).map(([name, m]) => (
            <div
              key={name}
              className="min-w-[120px] rounded-md border border-sparc-gray-200 bg-white px-3 py-2"
            >
              <p className="truncate text-[10px] font-medium text-sparc-gray-500">
                {name}
              </p>
              <p className="text-sm font-bold tabular-nums">
                R² {m.r2.toFixed(4)}
              </p>
              <p className="text-[9px] text-sparc-gray-400">
                RMSE {m.rmse.toFixed(4)}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Map controls */}
      <div className="flex items-center gap-3 border-b border-sparc-gray-100 px-4 py-2">
        <label className="text-xs text-sparc-gray-600">Mode:</label>
        <div className="flex gap-1">
          {MAP_MODES.map((m) => (
            <button
              key={m.value}
              onClick={() => setMapMode(m.value)}
              className={`rounded px-2 py-1 text-[10px] ${
                mapMode === m.value
                  ? "bg-sparc-purple text-white"
                  : "border border-sparc-gray-200 hover:bg-sparc-gray-50"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
        {fields.length > 0 && (
          <>
            <label className="ml-2 text-xs text-sparc-gray-600">
              Display Variable:
            </label>
            <select
              value={colorField}
              onChange={(e) => setColorField(e.target.value)}
              className="rounded border border-sparc-gray-200 px-2 py-1 text-xs"
            >
              {fields.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </>
        )}
      </div>

      {/* Map */}
      <div className="flex-1">
        <SpatialMap geojson={geojson as any} colorField={colorField} mode={mapMode} />
      </div>
    </div>
  );
}
