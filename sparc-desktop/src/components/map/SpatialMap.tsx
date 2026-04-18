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
export type ColorPalette = "sparc" | "viridis" | "puor";

interface SpatialMapProps {
  geojson: GeoJsonData | null;
  colorField?: string;
  mode?: MapMode;
  height?: string;
  palette?: ColorPalette;
  /** Callback when a feature is clicked. Receives properties. */
  onFeatureClick?: (properties: Record<string, unknown>) => void;
  /** Sync viewState from a parent (for side-by-side comparison) */
  syncViewState?: { longitude: number; latitude: number; zoom: number; pitch: number; bearing: number };
  onViewStateChange?: (vs: { longitude: number; latitude: number; zoom: number; pitch: number; bearing: number }) => void;
}

// ---------------------------------------------------------------------------
// Color ramps — default SPARC brand + colorblind-safe alternatives
// ---------------------------------------------------------------------------
const RAMPS: Record<ColorPalette, readonly (readonly [number, number, number])[]> = {
  sparc: [
    [96, 36, 104],   // sparc-purple
    [140, 56, 148],
    [164, 78, 180],  // sparc-magenta
    [200, 110, 160],
    [240, 160, 176], // sparc-blush
    [251, 221, 70],  // sparc-yellow
  ],
  viridis: [
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37],
  ],
  puor: [
    [127, 59, 8],    // orange
    [224, 153, 82],
    [247, 247, 247], // neutral white
    [153, 142, 195],
    [84, 39, 136],   // purple
  ],
};

const RAMP_CSS: Record<ColorPalette, string> = {
  sparc: "linear-gradient(to right, rgb(96,36,104), rgb(164,78,180), rgb(240,160,176), rgb(251,221,70))",
  viridis: "linear-gradient(to right, rgb(68,1,84), rgb(59,82,139), rgb(33,145,140), rgb(94,201,98), rgb(253,231,37))",
  puor: "linear-gradient(to right, rgb(127,59,8), rgb(224,153,82), rgb(247,247,247), rgb(153,142,195), rgb(84,39,136))",
};

function interpolateColor(t: number, palette: ColorPalette = "sparc"): [number, number, number, number] {
  const ramp = RAMPS[palette];
  const clamped = Math.max(0, Math.min(1, t));
  const idx = clamped * (ramp.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, ramp.length - 1);
  const frac = idx - lo;
  return [
    Math.round(ramp[lo][0] + frac * (ramp[hi][0] - ramp[lo][0])),
    Math.round(ramp[lo][1] + frac * (ramp[hi][1] - ramp[lo][1])),
    Math.round(ramp[lo][2] + frac * (ramp[hi][2] - ramp[lo][2])),
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
  palette = "sparc",
  onFeatureClick,
  syncViewState,
  onViewStateChange: onViewStateChangeProp,
}: SpatialMapProps) {
  const [localViewState, setLocalViewState] = useState({
    longitude: -98.5,
    latitude: 39.8,
    zoom: 4,
    pitch: 0,
    bearing: 0,
  });

  const viewState = syncViewState ?? localViewState;

  const handleViewStateChange = ({ viewState: vs }: any) => {
    if (onViewStateChangeProp) onViewStateChangeProp(vs);
    else setLocalViewState(vs);
  };

  // Auto-fit to data (only when not synced)
  useEffect(() => {
    if (syncViewState) return;
    if (!geojson || geojson.features.length === 0) return;
    const bbox = computeBBox(geojson);
    setLocalViewState((prev) => ({ ...prev, ...bbox }));
  }, [geojson, syncViewState]);

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
            return interpolateColor(t, palette);
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
            return interpolateColor(t, palette);
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
          return interpolateColor(t, palette);
        },
        pickable: true,
      }),
    ];
  }, [geojson, colorField, mode, domain, palette]);

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
        onViewStateChange={handleViewStateChange}
        layers={layers}
        controller
        onClick={({ object }: any) => {
          if (object && onFeatureClick) {
            onFeatureClick(object.properties ?? object);
          }
        }}
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
              style={{ background: RAMP_CSS[palette] }}
            />
            <span className="text-[9px] text-sparc-gray-500">{domain[1].toFixed(2)}</span>
          </div>
        </div>
      )}

      {/* North arrow */}
      <div className="absolute top-2 left-2 flex flex-col items-center">
        <svg width="20" height="28" viewBox="0 0 20 28" className="drop-shadow-sm">
          <polygon points="10,0 14,12 10,9 6,12" fill="#333" />
          <polygon points="10,9 14,12 10,28 6,12" fill="#999" />
        </svg>
        <span className="text-[8px] font-bold text-sparc-gray-600">N</span>
      </div>
    </div>
  );
}
