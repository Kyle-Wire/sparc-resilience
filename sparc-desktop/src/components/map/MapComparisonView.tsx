import { useState, useEffect } from "react";
import SpatialMap, { type ColorPalette } from "./SpatialMap";

interface GeoJsonFeature {
  type: "Feature";
  geometry: { type: string; coordinates: number[] | number[][] };
  properties: Record<string, unknown>;
}

interface GeoJsonData {
  type: "FeatureCollection";
  features: GeoJsonFeature[];
}

interface MapComparisonViewProps {
  leftGeojson: GeoJsonData | null;
  rightGeojson: GeoJsonData | null;
  leftLabel?: string;
  rightLabel?: string;
  colorField?: string;
  palette?: ColorPalette;
  height?: string;
}

/**
 * Side-by-side synced map comparison. Pan/zoom on one map mirrors to the other.
 */
export default function MapComparisonView({
  leftGeojson,
  rightGeojson,
  leftLabel = "Baseline",
  rightLabel = "Scenario",
  colorField,
  palette = "sparc",
  height = "400px",
}: MapComparisonViewProps) {
  const [sharedViewState, setSharedViewState] = useState({
    longitude: -98.5,
    latitude: 39.8,
    zoom: 4,
    pitch: 0,
    bearing: 0,
  });

  // Auto-fit to whichever dataset is available first
  useEffect(() => {
    const data = leftGeojson ?? rightGeojson;
    if (!data || data.features.length === 0) return;
    let minLng = 180, maxLng = -180, minLat = 90, maxLat = -90;
    for (const f of data.features) {
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
    const maxSpan = Math.max(maxLng - minLng, maxLat - minLat);
    const zoom = maxSpan > 0 ? Math.max(1, Math.min(16, Math.log2(360 / maxSpan) - 0.5)) : 10;
    setSharedViewState((prev) => ({ ...prev, longitude, latitude, zoom }));
  }, [leftGeojson, rightGeojson]);

  return (
    <div className="space-y-1">
      <div className="grid grid-cols-2 gap-2" style={{ height }}>
        <div className="relative">
          <div className="absolute top-2 left-8 z-10 rounded bg-white/80 px-2 py-0.5 text-xs font-medium text-sparc-gray-700 shadow-sm backdrop-blur-sm">
            {leftLabel}
          </div>
          <SpatialMap
            geojson={leftGeojson}
            colorField={colorField}
            palette={palette}
            mode="scatter"
            height="100%"
            syncViewState={sharedViewState}
            onViewStateChange={setSharedViewState}
          />
        </div>
        <div className="relative">
          <div className="absolute top-2 left-8 z-10 rounded bg-white/80 px-2 py-0.5 text-xs font-medium text-sparc-gray-700 shadow-sm backdrop-blur-sm">
            {rightLabel}
          </div>
          <SpatialMap
            geojson={rightGeojson}
            colorField={colorField}
            palette={palette}
            mode="scatter"
            height="100%"
            syncViewState={sharedViewState}
            onViewStateChange={setSharedViewState}
          />
        </div>
      </div>
    </div>
  );
}
