/**
 * MapComparisonView — draggable-divider swipe comparison.
 *
 * Two SpatialMaps are stacked in the same coordinate space; the right map is
 * clipped to the area to the right of a draggable vertical divider. Pan/zoom
 * stays synced via shared viewState. Replaces the previous side-by-side grid.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import SpatialMap, { type ColorPalette } from "./SpatialMap";
import type { GeoJsonData } from "@/lib/types";

interface MapComparisonViewProps {
  leftGeojson: GeoJsonData | null;
  rightGeojson: GeoJsonData | null;
  leftLabel?: string;
  rightLabel?: string;
  colorField?: string;
  palette?: ColorPalette;
  height?: string;
  /** Initial split position 0-1 (default 0.5). */
  initialSplit?: number;
}

function fitView(geojson: GeoJsonData) {
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
  const maxSpan = Math.max(maxLng - minLng, maxLat - minLat);
  const zoom = maxSpan > 0 ? Math.max(1, Math.min(16, Math.log2(360 / maxSpan) - 0.5)) : 10;
  return { longitude, latitude, zoom, pitch: 0, bearing: 0 };
}

export default function MapComparisonView({
  leftGeojson,
  rightGeojson,
  leftLabel = "Baseline",
  rightLabel = "Scenario",
  colorField,
  palette = "sparc",
  height = "420px",
  initialSplit = 0.5,
}: MapComparisonViewProps) {
  const [sharedViewState, setSharedViewState] = useState({
    longitude: -98.5,
    latitude: 39.8,
    zoom: 4,
    pitch: 0,
    bearing: 0,
  });
  const [split, setSplit] = useState(initialSplit);
  const dragRef = useRef(false);
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const data = leftGeojson ?? rightGeojson;
    if (!data || data.features.length === 0) return;
    setSharedViewState((prev) => ({ ...prev, ...fitView(data) }));
  }, [leftGeojson, rightGeojson]);

  const updateSplit = useCallback((clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const next = Math.max(0.05, Math.min(0.95, (clientX - rect.left) / rect.width));
    setSplit(next);
  }, []);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragRef.current) return;
      e.preventDefault();
      updateSplit(e.clientX);
    };
    const onUp = () => { dragRef.current = false; document.body.style.cursor = ""; };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [updateSplit]);

  const onHandleDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragRef.current = true;
    document.body.style.cursor = "ew-resize";
  }, []);

  const splitPct = `${(split * 100).toFixed(2)}%`;

  return (
    <div
      ref={containerRef}
      style={{
        position: "relative",
        height,
        width: "100%",
        userSelect: "none",
        overflow: "hidden",
        borderRadius: 8,
        border: "1px solid var(--line, rgba(0,0,0,0.08))",
      }}
    >
      <div style={{ position: "absolute", inset: 0 }}>
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
      <div
        style={{
          position: "absolute",
          inset: 0,
          clipPath: `inset(0 0 0 ${splitPct})`,
          WebkitClipPath: `inset(0 0 0 ${splitPct})`,
        }}
      >
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

      <div
        style={{
          position: "absolute",
          top: 10,
          left: 12,
          padding: "3px 8px",
          background: "rgba(255,255,255,0.92)",
          borderRadius: 4,
          fontSize: 11,
          fontWeight: 600,
          color: "#444",
          pointerEvents: "none",
          zIndex: 4,
        }}
      >
        {leftLabel}
      </div>
      <div
        style={{
          position: "absolute",
          top: 10,
          right: 12,
          padding: "3px 8px",
          background: "rgba(255,255,255,0.92)",
          borderRadius: 4,
          fontSize: 11,
          fontWeight: 600,
          color: "#444",
          pointerEvents: "none",
          zIndex: 4,
        }}
      >
        {rightLabel}
      </div>

      <div
        onMouseDown={onHandleDown}
        style={{
          position: "absolute",
          top: 0,
          bottom: 0,
          left: splitPct,
          width: 4,
          marginLeft: -2,
          background: "rgba(122, 58, 58, 0.85)",
          cursor: "ew-resize",
          zIndex: 5,
        }}
      >
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            width: 28,
            height: 28,
            borderRadius: "50%",
            background: "#fff",
            border: "2px solid rgba(122, 58, 58, 0.85)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "rgba(122, 58, 58, 0.85)",
            fontSize: 12,
            fontWeight: 700,
            boxShadow: "0 1px 3px rgba(0,0,0,0.18)",
          }}
        >
          ⇆
        </div>
      </div>
    </div>
  );
}
