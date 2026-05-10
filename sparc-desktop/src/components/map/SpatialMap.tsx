import { useState, useEffect, useMemo } from "react";
import Map, { NavigationControl, ScaleControl, Source, Layer } from "react-map-gl/maplibre";
import { GeoJsonLayer, ScatterplotLayer } from "@deck.gl/layers";
import { DeckGL } from "@deck.gl/react";
import "maplibre-gl/dist/maplibre-gl.css";
import type { ContextLayer } from "@/lib/api";
import type { GeoJsonData } from "@/lib/types";
import MapLegend from "./MapLegend";

// ---------------------------------------------------------------------------
// Inline blank basemap style — avoids external tile fetches inside Tauri.
// A light neutral background with subtle grid lines is sufficient for the
// SPARC data overlays. Context tile layers (Phase 18) are added on top when
// the user explicitly enables them.
// ---------------------------------------------------------------------------
const BLANK_BASEMAP_STYLE = {
  version: 8 as const,
  name: "SPARC Blank",
  sources: {},
  layers: [
    {
      id: "background",
      type: "background" as const,
      paint: { "background-color": "#f4f4f0" },
    },
  ],
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type MapMode = "scatter" | "heatmap" | "choropleth";
export type ColorPalette = "sparc" | "viridis" | "puor" | "rdbu";

interface SpatialMapProps {
  geojson: GeoJsonData | null;
  colorField?: string;
  mode?: MapMode;
  height?: string;
  palette?: ColorPalette;
  /**
   * Override the auto-computed value domain (min/max from features). Useful for
   * diverging palettes that should be centered on a known value (e.g. ±max(|β(s)|)
   * to keep zero pinned to the neutral midpoint of "rdbu"/"puor").
   */
  domainOverride?: [number, number];
  /**
   * Property name carrying a boolean / truthy flag that marks a feature as
   * "dimmed" — e.g. cells whose 90% CI brackets zero on a CATE map. Dimmed
   * features are rendered with reduced alpha to convey "not significant"
   * without removing them from the map.
   */
  dimField?: string;
  /** Callback when a feature is clicked. Receives properties. */
  onFeatureClick?: (properties: Record<string, unknown>) => void;
  /** Sync viewState from a parent (for side-by-side comparison) */
  syncViewState?: { longitude: number; latitude: number; zoom: number; pitch: number; bearing: number };
  onViewStateChange?: (vs: { longitude: number; latitude: number; zoom: number; pitch: number; bearing: number }) => void;
  /** Context tile layers (Phase 18) — basemap is replaced if any has kind="basemap";
   * overlays are stacked above the basemap and below the deck.gl canvas. */
  contextLayers?: { layer: ContextLayer; opacity?: number }[];
  /** When true, shows a fullscreen toggle (⛶) in the top-right corner. */
  expandable?: boolean;
  /** Property name to drive 3D extrusion height (choropleth only). */
  extrudeField?: string;
  /** Pixels per unit of `extrudeField` value (default 50). */
  extrudeScale?: number;
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
  rdbu: [
    [178, 24, 43],   // red (negative effect)
    [239, 138, 98],
    [247, 247, 247], // neutral white (zero, when domain is symmetric)
    [103, 169, 207],
    [33, 102, 172],  // blue (positive effect)
  ],
};

const RAMP_CSS: Record<ColorPalette, string> = {
  sparc: "linear-gradient(to right, rgb(96,36,104), rgb(164,78,180), rgb(240,160,176), rgb(251,221,70))",
  viridis: "linear-gradient(to right, rgb(68,1,84), rgb(59,82,139), rgb(33,145,140), rgb(94,201,98), rgb(253,231,37))",
  puor: "linear-gradient(to right, rgb(127,59,8), rgb(224,153,82), rgb(247,247,247), rgb(153,142,195), rgb(84,39,136))",
  rdbu: "linear-gradient(to right, rgb(178,24,43), rgb(239,138,98), rgb(247,247,247), rgb(103,169,207), rgb(33,102,172))",
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
// Compute bounding box from GeoJSON — handles all geometry types.
// ---------------------------------------------------------------------------
function collectCoords(geometry: any, out: number[][]): void {
  if (!geometry) return;
  switch (geometry.type) {
    case "Point":
      out.push(geometry.coordinates);
      break;
    case "MultiPoint":
    case "LineString":
      for (const c of geometry.coordinates) out.push(c);
      break;
    case "MultiLineString":
    case "Polygon":
      for (const ring of geometry.coordinates)
        for (const c of ring) out.push(c);
      break;
    case "MultiPolygon":
      for (const poly of geometry.coordinates)
        for (const ring of poly)
          for (const c of ring) out.push(c);
      break;
    case "GeometryCollection":
      for (const g of geometry.geometries ?? []) collectCoords(g, out);
      break;
  }
}

function computeBBox(geojson: GeoJsonData): {
  longitude: number;
  latitude: number;
  zoom: number;
} {
  let minLng = 180, maxLng = -180, minLat = 90, maxLat = -90;
  const pts: number[][] = [];
  for (const f of geojson.features) collectCoords(f.geometry, pts);
  for (const [lng, lat] of pts) {
    if (lng < minLng) minLng = lng;
    if (lng > maxLng) maxLng = lng;
    if (lat < minLat) minLat = lat;
    if (lat > maxLat) maxLat = lat;
  }
  if (pts.length === 0) return { longitude: -98.5, latitude: 39.8, zoom: 4 };
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
  domainOverride,
  dimField,
  onFeatureClick,
  syncViewState,
  onViewStateChange: onViewStateChangeProp,
  contextLayers,
  expandable,
  extrudeField,
  extrudeScale = 50,
}: SpatialMapProps) {
  const [fullscreen, setFullscreen] = useState(false);

  // Esc closes fullscreen.
  useEffect(() => {
    if (!fullscreen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setFullscreen(false);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [fullscreen]);

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

  // Compute value domain for color mapping (callers may pin it via domainOverride)
  const domain = useMemo<[number, number]>(() => {
    if (domainOverride) return domainOverride;
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
  }, [geojson, colorField, domainOverride]);

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
            const c = interpolateColor(t, palette);
            if (dimField && d.properties[dimField]) c[3] = Math.round(c[3] * 0.25);
            return c;
          },
          pickable: true,
        }),
      ];
    }

    if (mode === "choropleth") {
      const extrudeDomain: [number, number] = (() => {
        if (!extrudeField) return [0, 1];
        let lo = Infinity, hi = -Infinity;
        for (const f of geojson.features) {
          const v = f.properties[extrudeField];
          if (typeof v === "number" && isFinite(v)) {
            if (v < lo) lo = v;
            if (v > hi) hi = v;
          }
        }
        return lo < hi ? [lo, hi] : [0, 1];
      })();
      return [
        new GeoJsonLayer({
          id: "choropleth",
          data: geojson as any,
          filled: true,
          stroked: true,
          extruded: !!extrudeField,
          getElevation: ((d: any) => {
            if (!extrudeField) return 0;
            const v = d.properties[extrudeField];
            if (typeof v !== "number") return 0;
            const t = (v - extrudeDomain[0]) / (extrudeDomain[1] - extrudeDomain[0] || 1);
            return Math.max(0, t) * extrudeScale * 1000;
          }) as any,
          getFillColor: ((d: any) => {
            if (!colorField) return [96, 36, 104, 160];
            const v = d.properties[colorField];
            if (typeof v !== "number") return [200, 200, 200, 100];
            const t = (v - domain[0]) / (domain[1] - domain[0] || 1);
            const c = interpolateColor(t, palette);
            if (dimField && d.properties[dimField]) c[3] = Math.round(c[3] * 0.25);
            return c;
          }) as any,
          getLineColor: [60, 60, 60, 80] as [number, number, number, number],
          getLineWidth: 1,
          lineWidthMinPixels: 0.5,
          pickable: true,
          updateTriggers: {
            getElevation: [extrudeField, extrudeScale],
            getFillColor: [colorField, palette, domain[0], domain[1], dimField],
          },
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
          const c = interpolateColor(t, palette);
          if (dimField && d.properties[dimField]) c[3] = Math.round(c[3] * 0.25);
          return c;
        },
        pickable: true,
      }),
    ];
  }, [geojson, colorField, mode, domain, palette, extrudeField, extrudeScale, dimField]);

  if (!geojson) {
    return (
      <div className="flex items-center justify-center" style={{ height }}>
        <p className="text-sm text-sparc-gray-500">No spatial data to display</p>
      </div>
    );
  }

  return (
    <div
      className="relative overflow-hidden rounded-lg border border-sparc-gray-200"
      style={
        fullscreen
          ? { position: "fixed", inset: 16, zIndex: 200, background: "#fff" }
          : { height }
      }
    >
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
          mapStyle={BLANK_BASEMAP_STYLE}
          attributionControl={false}
        >
          <NavigationControl position="top-right" />
          <ScaleControl position="bottom-left" />
          {(contextLayers ?? []).filter((c) => c.layer.url_template).map(({ layer, opacity }) => (
            <Source
              key={layer.id}
              id={`ctx-${layer.id}`}
              type="raster"
              tiles={[layer.url_template as string]}
              tileSize={layer.tile_size ?? 256}
              maxzoom={layer.max_zoom ?? 19}
              attribution={layer.attribution}
            >
              <Layer
                id={`ctx-${layer.id}-layer`}
                type="raster"
                paint={{ "raster-opacity": opacity ?? layer.opacity_default ?? 1.0 }}
              />
            </Source>
          ))}
        </Map>
      </DeckGL>

      {/* Color legend */}
      {colorField && (
        <div style={{ position: "absolute", bottom: 14, right: 14, zIndex: 6 }}>
          <MapLegend
            label={colorField}
            domain={domain}
            rampCss={RAMP_CSS[palette]}
          />
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

      {/* Fullscreen toggle */}
      {expandable && (
        <button
          onClick={() => setFullscreen((f) => !f)}
          title={fullscreen ? "Exit fullscreen (Esc)" : "Expand to fullscreen"}
          style={{
            position: "absolute",
            top: 8,
            right: 48,
            width: 32,
            height: 32,
            borderRadius: 6,
            border: "1px solid #ddd",
            background: "rgba(255,255,255,0.95)",
            cursor: "pointer",
            fontSize: 16,
            fontWeight: 600,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#333",
            zIndex: 5,
          }}
        >
          {fullscreen ? "✕" : "⛶"}
        </button>
      )}
    </div>
  );
}
