// Swipe A|B (SPEC §12.5): two rasters stacked, the left one clipped at a divider that can be
// dragged or moved with the arrow keys (role="slider"). Both share one view transform.
import { useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";
import { GridCanvas, type GridCanvasProps } from "./GridCanvas";
import type { Domain } from "./domain";
import type { GridData } from "./grid";
import type { MapViewApi } from "./useMapView";

export type SwipeLayer = { values: ArrayLike<number> | null; domain: Domain | null; label: string };

export type SwipeCompareProps = {
  grid: GridData;
  view: MapViewApi;
  dark: boolean;
  left: SwipeLayer;
  right: SwipeLayer;
  height?: number;
  selection?: Uint8Array | null;
  /** Interaction props for the full (right) pane: hover, pick, tool… */
  interactive?: Omit<GridCanvasProps, "grid" | "values" | "domain" | "dark" | "view" | "label" | "height">;
  overlay?: ReactNode;
};

export function SwipeCompare({ grid, view, dark, left, right, height = 480, selection, interactive, overlay }: SwipeCompareProps) {
  const [at, setAt] = useState(0.5);
  const box = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const fromEvent = (e: PointerEvent) => {
    const r = box.current?.getBoundingClientRect();
    if (!r || !r.width) return;
    setAt(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)));
  };
  const onKey = (e: KeyboardEvent) => {
    const step = e.shiftKey ? 0.1 : 0.02;
    if (e.key === "ArrowLeft") setAt((a) => Math.max(0, a - step));
    else if (e.key === "ArrowRight") setAt((a) => Math.min(1, a + step));
    else if (e.key === "Home") setAt(0);
    else if (e.key === "End") setAt(1);
    else return;
    e.preventDefault();
  };
  return (
    <div ref={box} style={{ position: "relative", height }}>
      <GridCanvas {...interactive} grid={grid} view={view} dark={dark} values={right.values} domain={right.domain} selection={selection} label={`Swipe comparison: ${left.label} on the left, ${right.label} on the right`} height={height} overlay={overlay} />
      <GridCanvas grid={grid} view={view} dark={dark} values={left.values} domain={left.domain} selection={selection} label={left.label} height={height} clip={{ side: "left", at }} passive />
      <div className="swipe-divider" style={{ left: `${at * 100}%` }} onPointerDown={(e) => {
        dragging.current = true;
        (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
      }} onPointerMove={(e) => dragging.current && fromEvent(e)} onPointerUp={() => (dragging.current = false)}>
        <button
          type="button"
          role="slider"
          aria-label={`Swipe divider: ${left.label} left, ${right.label} right`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(at * 100)}
          aria-valuetext={`${Math.round(at * 100)}% ${left.label}`}
          onKeyDown={onKey}
        >
          ⇆
        </button>
      </div>
      <span className="map-readout" style={{ top: "auto", bottom: 8, left: 8 }}>
        A: {left.label}
      </span>
      <span className="map-readout" style={{ top: "auto", bottom: 8, left: "auto", right: 8 }}>
        B: {right.label}
      </span>
    </div>
  );
}
