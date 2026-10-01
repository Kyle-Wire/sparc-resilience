// Shared view transform for map panes (SPEC §12.5): screen = (tx + px·scale, ty + py·scale)
// where (px, py) are raster units (1 per cell). Side-by-side panes pass the same MapViewApi,
// so panning or zooming one moves the other.
import { useCallback, useMemo, useRef, useState } from "react";

export type ViewState = { scale: number; tx: number; ty: number };

export const MAX_SCALE = 32;

export type MapViewApi = {
  state: ViewState;
  /** True until the first fit to a viewport. */
  needsFit: boolean;
  set: (s: ViewState) => void;
  /** Fit an nx×ny raster centred in a w×h viewport. */
  fit: (nx: number, ny: number, w: number, h: number) => void;
  /** Zoom by `factor` keeping the screen point (cx, cy) fixed. */
  zoomAt: (cx: number, cy: number, factor: number) => void;
  panBy: (dx: number, dy: number) => void;
  /** Smallest allowed scale (the fit scale, or 1 px per cell if the fit is larger). */
  minScale: number;
  toScreen: (px: number, py: number) => [number, number];
  toRaster: (sx: number, sy: number) => [number, number];
};

/** Fit transform: whole raster visible, centred. */
export function fitView(nx: number, ny: number, w: number, h: number, pad = 8): ViewState {
  const scale = Math.max(1e-3, Math.min((w - 2 * pad) / Math.max(1, nx), (h - 2 * pad) / Math.max(1, ny)));
  return { scale, tx: (w - nx * scale) / 2, ty: (h - ny * scale) / 2 };
}

/** Zoom about a screen point, clamped to [minScale, MAX_SCALE] px per cell. */
export function zoomView(v: ViewState, cx: number, cy: number, factor: number, minScale: number): ViewState {
  const scale = Math.min(MAX_SCALE, Math.max(minScale, v.scale * factor));
  const k = scale / v.scale;
  return { scale, tx: cx - (cx - v.tx) * k, ty: cy - (cy - v.ty) * k };
}

export function useMapView(initial?: ViewState): MapViewApi {
  const [state, setState] = useState<ViewState>(initial ?? { scale: 1, tx: 0, ty: 0 });
  const [needsFit, setNeedsFit] = useState(!initial);
  const minRef = useRef(1);
  const stateRef = useRef(state);
  stateRef.current = state;

  const set = useCallback((s: ViewState) => setState(s), []);
  const fit = useCallback((nx: number, ny: number, w: number, h: number) => {
    const v = fitView(nx, ny, w, h);
    minRef.current = Math.min(1, v.scale);
    setState(v);
    setNeedsFit(false);
  }, []);
  const zoomAt = useCallback((cx: number, cy: number, factor: number) => setState((v) => zoomView(v, cx, cy, factor, minRef.current)), []);
  const panBy = useCallback((dx: number, dy: number) => setState((v) => ({ ...v, tx: v.tx + dx, ty: v.ty + dy })), []);

  return useMemo(
    () => ({
      state,
      needsFit,
      set,
      fit,
      zoomAt,
      panBy,
      minScale: minRef.current,
      toScreen: (px: number, py: number) => [state.tx + px * state.scale, state.ty + py * state.scale] as [number, number],
      toRaster: (sx: number, sy: number) => [(sx - state.tx) / state.scale, (sy - state.ty) / state.scale] as [number, number],
    }),
    [state, needsFit, set, fit, zoomAt, panBy],
  );
}
