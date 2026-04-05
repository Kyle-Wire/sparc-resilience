import { useState, useEffect, useMemo, useCallback } from "react";
import {
  getLocalCoefVariables,
  getLocalCoefficients,
  getCateMapVariables,
  getCateMap,
  getPdpCurves,
  getDoseResponseCurves,
  getCausalResults,
  getScenarioDetail,
} from "@/lib/api";
import type {
  GeoJsonData,
  PdpCurves,
  DoseResponseData,
  CausalResults,
  ScenarioDetail,
} from "@/lib/types";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Area,
  AreaChart,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
} from "recharts";

// ---------------------------------------------------------------------------
// VariableDashboardView
// ---------------------------------------------------------------------------

export default function VariableDashboardView() {
  const [allVars, setAllVars] = useState<string[]>([]);
  const [selectedVar, setSelectedVar] = useState("");
  const [loading, setLoading] = useState(true);

  // Data stores
  const [pdp, setPdp] = useState<PdpCurves | null>(null);
  const [doseResponse, setDoseResponse] = useState<DoseResponseData | null>(null);
  const [causal, setCausal] = useState<CausalResults | null>(null);
  const [scenarioDetail, setScenarioDetail] = useState<ScenarioDetail | null>(null);
  const [coefGeo, setCoefGeo] = useState<GeoJsonData | null>(null);
  const [cateGeo, setCateGeo] = useState<GeoJsonData | null>(null);
  const [coefLoading, setCoefLoading] = useState(false);
  const [cateLoading, setCateLoading] = useState(false);

  // Gather the union of all variable names across endpoints
  useEffect(() => {
    Promise.allSettled([
      getLocalCoefVariables(),
      getCateMapVariables(),
      getPdpCurves(),
      getDoseResponseCurves(),
      getCausalResults(),
      getScenarioDetail(),
    ]).then(([coefRes, cateRes, pdpRes, drRes, causalRes, scenRes]) => {
      const varSet = new Set<string>();

      if (coefRes.status === "fulfilled")
        coefRes.value.variables.forEach((v) => varSet.add(v));
      if (cateRes.status === "fulfilled")
        cateRes.value.variables.forEach((v) => varSet.add(v));
      if (pdpRes.status === "fulfilled") {
        setPdp(pdpRes.value);
        Object.keys(pdpRes.value).forEach((v) => varSet.add(v));
      }
      if (drRes.status === "fulfilled") {
        setDoseResponse(drRes.value);
        Object.keys(drRes.value).forEach((v) => varSet.add(v));
      }
      if (causalRes.status === "fulfilled") {
        setCausal(causalRes.value);
        Object.keys(causalRes.value.direct_effects ?? {}).forEach((v) => varSet.add(v));
      }
      if (scenRes.status === "fulfilled") setScenarioDetail(scenRes.value);

      const vars = [...varSet].sort();
      setAllVars(vars);
      if (vars.length > 0) setSelectedVar(vars[0]);
      setLoading(false);
    });
  }, []);

  // Fetch spatial data per variable
  const fetchSpatialData = useCallback((varName: string) => {
    setCoefGeo(null);
    setCateGeo(null);
    setCoefLoading(true);
    setCateLoading(true);

    getLocalCoefficients(varName)
      .then(setCoefGeo)
      .catch(() => setCoefGeo(null))
      .finally(() => setCoefLoading(false));

    getCateMap(varName)
      .then(setCateGeo)
      .catch(() => setCateGeo(null))
      .finally(() => setCateLoading(false));
  }, []);

  useEffect(() => {
    if (selectedVar) fetchSpatialData(selectedVar);
  }, [selectedVar, fetchSpatialData]);

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-sparc-gray-500">Loading variable data…</p>
      </div>
    );
  }

  if (allVars.length === 0) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-sparc-gray-600">
          No variable data available. Run Stages 2-4 first.
        </p>
      </div>
    );
  }

  // Per-variable data slices
  const effect = causal?.direct_effects?.[selectedVar];
  const pdpEntry = pdp?.[selectedVar];
  const drEntry = doseResponse?.[selectedVar];

  return (
    <div className="flex h-full flex-col">
      {/* Toolbar */}
      <div className="flex items-center gap-4 border-b border-sparc-gray-100 px-4 py-2">
        <h3 className="text-sm font-semibold">Variable Dashboard</h3>
        <select
          value={selectedVar}
          onChange={(e) => setSelectedVar(e.target.value)}
          className="rounded border border-sparc-gray-200 px-2 py-1 text-xs"
        >
          {allVars.map((v) => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>
      </div>

      {/* Dashboard grid */}
      <div className="flex-1 overflow-auto p-4">
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">

          {/* Summary card */}
          {effect && <SummaryCard variable={selectedVar} effect={effect} />}

          {/* PDP curve */}
          {pdpEntry && <PdpCard variable={selectedVar} entry={pdpEntry} />}

          {/* Dose-response */}
          {drEntry && <DoseResponseCard variable={selectedVar} entry={drEntry} />}

          {/* Scenario increments */}
          {scenarioDetail && (
            <ScenarioCard variable={selectedVar} scenarioDetail={scenarioDetail} />
          )}

          {/* Local coefficients map */}
          {(coefGeo || coefLoading) && (
            <div className="rounded-lg border border-sparc-gray-200 p-3">
              <h4 className="mb-2 text-xs font-semibold">Local Coefficients Map</h4>
              {coefLoading ? (
                <p className="text-xs text-sparc-gray-500">Loading…</p>
              ) : coefGeo ? (
                <MiniMap geojson={coefGeo} colorField="coefficient" diverging />
              ) : null}
            </div>
          )}

          {/* CATE map */}
          {(cateGeo || cateLoading) && (
            <div className="rounded-lg border border-sparc-gray-200 p-3">
              <h4 className="mb-2 text-xs font-semibold">Spatial CATE Map</h4>
              {cateLoading ? (
                <p className="text-xs text-sparc-gray-500">Loading…</p>
              ) : cateGeo ? (
                <MiniMap
                  geojson={cateGeo}
                  colorField={
                    Object.keys(cateGeo.features[0]?.properties ?? {}).find((k) =>
                      k.startsWith("cate_"),
                    )
                  }
                  diverging
                />
              ) : null}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function SummaryCard({
  variable,
  effect,
}: {
  variable: string;
  effect: Record<string, any>;
}) {
  const sign =
    (effect.structural_coeff ?? 0) < 0
      ? "negative"
      : (effect.structural_coeff ?? 0) > 0
        ? "positive"
        : "neutral";
  const signColor =
    sign === "negative"
      ? "border-blue-300 bg-blue-50"
      : sign === "positive"
        ? "border-red-300 bg-red-50"
        : "border-sparc-gray-200 bg-white";

  return (
    <div className={`rounded-lg border-2 p-4 space-y-2 ${signColor}`}>
      <h4 className="text-sm font-bold">{variable}</h4>
      <div className="grid grid-cols-3 gap-3 text-[10px]">
        <div>
          <span className="text-sparc-gray-500">Structural Coeff</span>
          <p className="text-lg font-bold tabular-nums">
            {effect.structural_coeff?.toFixed(4) ?? "—"}
          </p>
        </div>
        {effect.elasticity != null && (
          <div>
            <span className="text-sparc-gray-500">Elasticity</span>
            <p className="font-mono font-medium">{effect.elasticity.toFixed(4)}</p>
          </div>
        )}
        {effect.relative_importance != null && (
          <div>
            <span className="text-sparc-gray-500">Importance</span>
            <p className="font-mono font-medium">
              {(effect.relative_importance * 100).toFixed(1)}%
            </p>
          </div>
        )}
        {effect.cate_mean != null && (
          <div>
            <span className="text-sparc-gray-500">CATE mean</span>
            <p className="font-mono font-medium">
              {effect.cate_mean.toFixed(4)}
              {effect.cate_std != null && ` ± ${effect.cate_std.toFixed(4)}`}
            </p>
          </div>
        )}
        {effect.e_value != null && (
          <div>
            <span className="text-sparc-gray-500">E-value</span>
            <p className="font-mono font-medium">{effect.e_value.toFixed(2)}</p>
          </div>
        )}
      </div>
      {/* Refutation badges */}
      <div className="flex flex-wrap gap-1">
        {[
          { key: "placebo_pass", label: "Placebo" },
          { key: "rcc_pass", label: "RCC" },
          { key: "data_subset_pass", label: "Subset" },
          { key: "ucc_pass", label: "UCC" },
        ].map(({ key, label }) => {
          const val = effect[key];
          if (val == null) return null;
          return (
            <span
              key={key}
              className={`rounded px-1.5 py-0.5 text-[9px] font-medium ${
                val ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"
              }`}
            >
              {val ? "✓" : "✗"} {label}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function PdpCard({
  variable,
  entry,
}: {
  variable: string;
  entry: any;
}) {
  const chartData = useMemo(() => {
    const grid = entry.pdp?.grid_values ?? entry.grid_values ?? [];
    const vals = entry.pdp?.pdp_values ?? entry.pdp_values ?? [];
    const std = entry.pdp?.pdp_std;
    return grid.map((x: number, i: number) => ({
      x,
      pdp: vals[i],
      upper: std ? vals[i] + (std[i] ?? 0) : undefined,
      lower: std ? vals[i] - (std[i] ?? 0) : undefined,
    }));
  }, [entry]);

  return (
    <div className="rounded-lg border border-sparc-gray-200 p-3">
      <h4 className="mb-2 text-xs font-semibold">PDP — {variable}</h4>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis dataKey="x" tick={{ fontSize: 9 }} />
          <YAxis tick={{ fontSize: 9 }} />
          <Tooltip contentStyle={{ fontSize: 10 }} />
          {chartData[0]?.upper !== undefined && (
            <Area type="monotone" dataKey="upper" stroke="none" fill="#a44eb4" fillOpacity={0.15} />
          )}
          {chartData[0]?.lower !== undefined && (
            <Area type="monotone" dataKey="lower" stroke="none" fill="#a44eb4" fillOpacity={0.15} />
          )}
          <Line type="monotone" dataKey="pdp" stroke="#602468" strokeWidth={2} dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function DoseResponseCard({
  variable,
  entry,
}: {
  variable: string;
  entry: any;
}) {
  const chartData = useMemo(() => {
    if (!entry) return [];
    return entry.dose_levels.map((dose: number, i: number) => ({
      dose,
      effect: entry.marginal_effects[i],
    }));
  }, [entry]);

  return (
    <div className="rounded-lg border border-sparc-gray-200 p-3">
      <h4 className="mb-2 text-xs font-semibold">
        Dose-Response — {variable}
        {entry.is_nonlinear != null && (
          <span
            className={`ml-2 rounded px-1.5 py-0.5 text-[9px] font-normal ${
              entry.is_nonlinear ? "bg-yellow-100 text-yellow-700" : "bg-green-100 text-green-700"
            }`}
          >
            {entry.is_nonlinear ? "Non-linear" : "Linear"}
          </span>
        )}
      </h4>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis dataKey="dose" tick={{ fontSize: 9 }} />
          <YAxis tick={{ fontSize: 9 }} />
          <Tooltip contentStyle={{ fontSize: 10 }} />
          <ReferenceLine y={0} stroke="#999" strokeDasharray="4 4" />
          <Line
            type="monotone"
            dataKey="effect"
            stroke="#602468"
            strokeWidth={2}
            dot={{ r: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function ScenarioCard({
  variable,
  scenarioDetail,
}: {
  variable: string;
  scenarioDetail: ScenarioDetail;
}) {
  // Filter scenario summary rows to this variable
  const rows = useMemo(() => {
    return scenarioDetail.summary.filter(
      (r) => String(r.Variable ?? "").toLowerCase() === variable.toLowerCase(),
    );
  }, [scenarioDetail, variable]);

  if (rows.length === 0) return null;

  const chartData = rows.map((r) => ({
    increment: Number(r.Increment ?? 0),
    delta: Number(r.Mean_Delta ?? r.Delta_Mean ?? r.mean_delta ?? 0),
    label: String(r.Scenario ?? ""),
  }));

  return (
    <div className="rounded-lg border border-sparc-gray-200 p-3">
      <h4 className="mb-2 text-xs font-semibold">
        Scenario Increments — {variable}
      </h4>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis dataKey="increment" tick={{ fontSize: 9 }} />
          <YAxis tick={{ fontSize: 9 }} />
          <Tooltip
            contentStyle={{ fontSize: 10 }}
            labelFormatter={(v) => `Increment: ${v}`}
          />
          <Legend wrapperStyle={{ fontSize: 10 }} />
          <ReferenceLine y={0} stroke="#999" strokeDasharray="4 4" />
          <Bar dataKey="delta" fill="#602468" name="Mean Δ Outcome" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Mini deck.gl map (lazy-loaded)
// ---------------------------------------------------------------------------

function MiniMap({
  geojson,
  colorField,
  diverging = false,
}: {
  geojson: GeoJsonData;
  colorField?: string;
  diverging?: boolean;
}) {
  const [DeckGL, setDeckGL] = useState<any>(null);
  const [MapGL, setMapGL] = useState<any>(null);
  const [layerMods, setLayerMods] = useState<any>(null);

  useEffect(() => {
    Promise.all([
      import("@deck.gl/react"),
      import("@deck.gl/layers"),
      import("react-map-gl/maplibre"),
    ]).then(([d, l, m]) => {
      setDeckGL(() => d.DeckGL);
      setMapGL(() => m.default);
      setLayerMods(l);
    });
  }, []);

  const bbox = useMemo(() => {
    let minLng = 180, maxLng = -180, minLat = 90, maxLat = -90;
    for (const f of geojson.features) {
      const c =
        f.geometry.type === "Point"
          ? [f.geometry.coordinates as number[]]
          : (f.geometry.coordinates as number[][]);
      for (const pt of c) {
        const [lng, lat] = pt as number[];
        if (lng < minLng) minLng = lng;
        if (lng > maxLng) maxLng = lng;
        if (lat < minLat) minLat = lat;
        if (lat > maxLat) maxLat = lat;
      }
    }
    const longitude = (minLng + maxLng) / 2;
    const latitude = (minLat + maxLat) / 2;
    const span = Math.max(maxLng - minLng, maxLat - minLat);
    const zoom = span > 0 ? Math.max(1, Math.min(16, Math.log2(360 / span) - 0.5)) : 10;
    return { longitude, latitude, zoom };
  }, [geojson]);

  const [viewState, setViewState] = useState({ ...bbox, pitch: 0, bearing: 0 });
  useEffect(() => setViewState((p) => ({ ...p, ...bbox })), [bbox]);

  const domain = useMemo<[number, number]>(() => {
    if (!colorField) return [0, 1];
    let min = Infinity, max = -Infinity;
    for (const f of geojson.features) {
      const v = f.properties[colorField];
      if (typeof v === "number" && isFinite(v)) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    return min < max ? [min, max] : [0, 1];
  }, [geojson, colorField]);

  if (!DeckGL || !MapGL || !layerMods) {
    return <p className="text-xs text-sparc-gray-500">Loading map…</p>;
  }

  const absMax = diverging ? Math.max(Math.abs(domain[0]), Math.abs(domain[1])) : 0;
  const symDomain: [number, number] = diverging && absMax > 0 ? [-absMax, absMax] : domain;

  const colorFn = (v: number): [number, number, number, number] => {
    if (diverging) {
      const t = Math.max(-1, Math.min(1, v / (symDomain[1] || 1)));
      if (t < 0) {
        const a = 1 + t;
        return [Math.round(80 * (1 - a)), Math.round(80 + 175 * a), Math.round(200 + 55 * (1 - a)), 200];
      }
      return [Math.round(200 + 55 * t), Math.round(80 + 175 * (1 - t)), Math.round(80 * (1 - t)), 200];
    }
    const t = (v - domain[0]) / (domain[1] - domain[0] || 1);
    const c = Math.max(0, Math.min(1, t));
    return [
      Math.round(96 + c * 155),
      Math.round(36 + c * 185),
      Math.round(104 + c * (70 - 104)),
      200,
    ];
  };

  const deckLayers = [
    new layerMods.ScatterplotLayer({
      id: "mini-scatter",
      data: geojson.features,
      getPosition: (d: any) => {
        const c = d.geometry.coordinates;
        return d.geometry.type === "Point" ? c : c[0];
      },
      getRadius: 80,
      radiusMinPixels: 3,
      radiusMaxPixels: 16,
      getFillColor: (d: any) => {
        if (!colorField) return [96, 36, 104, 200];
        const v = d.properties[colorField];
        if (typeof v !== "number") return [200, 200, 200, 100];
        return colorFn(v);
      },
      pickable: true,
    }),
  ];

  return (
    <div className="relative overflow-hidden rounded-lg border border-sparc-gray-100" style={{ height: "280px" }}>
      <DeckGL
        viewState={viewState}
        onViewStateChange={({ viewState: vs }: any) => setViewState(vs)}
        layers={deckLayers}
        controller
        getTooltip={({ object }: any) => {
          if (!object) return null;
          const props = object.properties ?? object;
          const lines = Object.entries(props)
            .filter(([k]) => !k.startsWith("_") && k !== "geometry")
            .slice(0, 4)
            .map(([k, v]) => `${k}: ${typeof v === "number" ? v.toFixed(4) : v}`);
          return { text: lines.join("\n"), style: { fontSize: "10px", fontFamily: "monospace" } };
        }}
      >
        <MapGL
          mapStyle="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
          attributionControl={false}
        />
      </DeckGL>

      {/* Legend */}
      {colorField && (
        <div className="absolute bottom-2 right-2 rounded bg-white/90 p-1.5 shadow-sm backdrop-blur-sm">
          <p className="mb-0.5 text-[9px] text-sparc-gray-600">{colorField}</p>
          <div className="flex items-center gap-1">
            <span className="text-[8px]">{symDomain[0].toFixed(2)}</span>
            <div
              className="h-2 w-16 rounded-sm"
              style={{
                background: diverging
                  ? "linear-gradient(to right, rgb(0,80,200), rgb(255,255,255), rgb(255,80,0))"
                  : "linear-gradient(to right, rgb(96,36,104), rgb(251,221,70))",
              }}
            />
            <span className="text-[8px]">{symDomain[1].toFixed(2)}</span>
          </div>
        </div>
      )}
    </div>
  );
}
