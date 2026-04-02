import { useState, useEffect, useMemo } from "react";
import Map, { NavigationControl, ScaleControl } from "react-map-gl/maplibre";
import { GeoJsonLayer, ScatterplotLayer } from "@deck.gl/layers";
import { DeckGL } from "@deck.gl/react";
import "maplibre-gl/dist/maplibre-gl.css";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface GeoJsonFeature {
  type: "Feature";
  geometry: { type: string; coordinates: number[] | number[][] };
  properties: Record<string, unknown>;
}

interface GeoJsonData {
  type: "FeatureCollection";
  features: GeoJsonFeature[];
}

export type MapMode = "scatter" | "heatmap" | "choropleth";

interface SpatialMapProps {
  geojson: GeoJsonData | null;
  colorField?: string;
  mode?: MapMode;
  height?: string;
}

// ---------------------------------------------------------------------------
// SPARC brand color ramp: purple → magenta → amber → yellow
// ---------------------------------------------------------------------------
const COLOR_RAMP = [
  [96, 36, 104],   // sparc-purple
  [140, 56, 148],
  [164, 78, 180],  // sparc-magenta
  [200, 110, 160],
  [240, 160, 176], // sparc-blush
  [251, 221, 70],  // sparc-yellow
] as const;

function interpolateColor(t: number): [number, number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  const idx = clamped * (COLOR_RAMP.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, COLOR_RAMP.length - 1);
  const frac = idx - lo;
  return [
    Math.round(COLOR_RAMP[lo][0] + frac * (COLOR_RAMP[hi][0] - COLOR_RAMP[lo][0])),
    Math.round(COLOR_RAMP[lo][1] + frac * (COLOR_RAMP[hi][1] - COLOR_RAMP[lo][1])),
    Math.round(COLOR_RAMP[lo][2] + frac * (COLOR_RAMP[hi][2] - COLOR_RAMP[lo][2])),
    200,
  ];
}

// ---------------------------------------------------------------------------
// Compute bounding box from GeoJSON
// ---------------------------------------------------------------------------
function computeBBox(geojson: GeoJsonData): {
  longitude: number;
  latitude: number;
  zoom: number;
} {
  let minLng = 180, maxLng = -180, minLat = 90, maxLat = -90;
  for (const f of geojson.features) {
    const coords = f.geometry.type === "Point"
      ? [f.geometry.coordinates as number[]]
      : (f.geometry.coordinates as number[][]);
    for (const c of coords) {
      const [lng, lat] = c;
      if (lng < minLng) minLng = lng;
      if (lng > maxLng) maxLng = lng;
      if (lat < minLat) minLat = lat;
      if (lat > maxLat) maxLat = lat;
    }
  }
  const longitude = (minLng + maxLng) / 2;
  const latitude = (minLat + maxLat) / 2;
  const spanLng = maxLng - minLng;
  const spanLat = maxLat - minLat;
  const maxSpan = Math.max(spanLng, spanLat);
  const zoom = maxSpan > 0 ? Math.max(1, Math.min(16, Math.log2(360 / maxSpan) - 0.5)) : 10;
  return { longitude, latitude, zoom };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function SpatialMap({
  geojson,
  colorField,
  mode = "scatter",
  height = "100%",
}: SpatialMapProps) {
  const [viewState, setViewState] = useState({
    longitude: -98.5,
    latitude: 39.8,
    zoom: 4,
    pitch: 0,
    bearing: 0,
  });

  // Auto-fit to data
  useEffect(() => {
    if (!geojson || geojson.features.length === 0) return;
    const bbox = computeBBox(geojson);
    setViewState((prev) => ({ ...prev, ...bbox }));
  }, [geojson]);

  // Compute value domain for color mapping
  const domain = useMemo<[number, number]>(() => {
    if (!geojson || !colorField) return [0, 1];
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

  const layers = useMemo(() => {
    if (!geojson) return [];

    if (mode === "heatmap") {
      // Dense scatter with alpha blending as heatmap proxy (no aggregation-layers)
      return [
        new ScatterplotLayer({
          id: "heatmap-scatter",
          data: geojson.features,
          getPosition: (d: any) => {
            const c = d.geometry.coordinates;
            return d.geometry.type === "Point" ? c : c[0];
          },
          getRadius: 200,
          radiusMinPixels: 6,
          radiusMaxPixels: 30,
          getFillColor: (d: any) => {
            if (!colorField) return [96, 36, 104, 120] as [number, number, number, number];
            const v = d.properties[colorField];
            if (typeof v !== "number") return [200, 200, 200, 60] as [number, number, number, number];
            const t = (v - domain[0]) / (domain[1] - domain[0] || 1);
            return interpolateColor(t);
          },
          pickable: true,
        }),
      ];
    }

    if (mode === "choropleth") {
      return [
        new GeoJsonLayer({
          id: "choropleth",
          data: geojson as any,
          filled: true,
          stroked: true,
          getFillColor: ((d: any) => {
            if (!colorField) return [96, 36, 104, 160];
            const v = d.properties[colorField];
            if (typeof v !== "number") return [200, 200, 200, 100];
            const t = (v - domain[0]) / (domain[1] - domain[0] || 1);
            return interpolateColor(t);
          }) as any,
          getLineColor: [60, 60, 60, 80] as [number, number, number, number],
          getLineWidth: 1,
          lineWidthMinPixels: 0.5,
          pickable: true,
        }),
      ];
    }

    // Default: scatter
    return [
      new ScatterplotLayer({
        id: "scatter",
        data: geojson.features,
        getPosition: (d: any) => {
          const c = d.geometry.coordinates;
          return d.geometry.type === "Point" ? c : c[0];
        },
        getRadius: 80,
        radiusMinPixels: 3,
        radiusMaxPixels: 20,
        getFillColor: (d: any) => {
          if (!colorField) return [96, 36, 104, 200] as [number, number, number, number];
          const v = d.properties[colorField];
          if (typeof v !== "number") return [200, 200, 200, 100] as [number, number, number, number];
          const t = (v - domain[0]) / (domain[1] - domain[0] || 1);
          return interpolateColor(t);
        },
        pickable: true,
      }),
    ];
  }, [geojson, colorField, mode, domain]);

  if (!geojson) {
    return (
      <div className="flex items-center justify-center" style={{ height }}>
        <p className="text-sm text-sparc-gray-500">No spatial data to display</p>
      </div>
    );
  }

  return (
    <div className="relative overflow-hidden rounded-lg border border-sparc-gray-200" style={{ height }}>
      <DeckGL
        viewState={viewState}
        onViewStateChange={({ viewState: vs }: any) => setViewState(vs)}
        layers={layers}
        controller
        getTooltip={({ object }: any) => {
          if (!object) return null;
          const props = object.properties ?? object;
          const lines = Object.entries(props)
            .filter(([k]) => !k.startsWith("_") && k !== "geometry")
            .slice(0, 8)
            .map(([k, v]) => `${k}: ${typeof v === "number" ? v.toFixed(4) : v}`);
          return { text: lines.join("\n"), style: { fontSize: "11px", fontFamily: "monospace" } };
        }}
      >
        <Map
          mapStyle="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
          attributionControl={false}
        >
          <NavigationControl position="top-right" />
          <ScaleControl position="bottom-left" />
        </Map>
      </DeckGL>

      {/* Color legend */}
      {colorField && (
        <div className="absolute bottom-4 right-4 rounded-md bg-white/90 p-2 shadow-sm backdrop-blur-sm">
          <p className="mb-1 text-[10px] font-medium text-sparc-gray-600">{colorField}</p>
          <div className="flex items-center gap-1.5">
            <span className="text-[9px] text-sparc-gray-500">{domain[0].toFixed(2)}</span>
            <div
              className="h-2.5 w-24 rounded-sm"
              style={{
                background: `linear-gradient(to right, rgb(96,36,104), rgb(164,78,180), rgb(240,160,176), rgb(251,221,70))`,
              }}
            />
            <span className="text-[9px] text-sparc-gray-500">{domain[1].toFixed(2)}</span>
          </div>
        </div>
      )}
    </div>
  );
}
