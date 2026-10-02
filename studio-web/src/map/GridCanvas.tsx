// GridCanvas (SPEC §12.5): a 30 m raster on a <canvas>, one pixel per cell, north up.
// - colorize() fills an offscreen nx×ny ImageData through a Uint32 view (≈1–2 ms for
//   54,701 cells); the viewport canvas (devicePixelRatio-aware) draws it with
//   {scale, tx, ty} and imageSmoothingEnabled = false, redrawing on change via rAF.
// - Overlays are SVG in the same transform.
// - Wheel zooms about the cursor (1–32 px per cell), drag pans (space + drag while a tool is
//   active), +/−/0 zoom and fit, arrow keys move a cell cursor announced in an aria-live
//   region, Enter pins the inspector, Escape cancels the tool.
// - The wheel listener is a native non-passive one: React attaches wheel handlers as passive,
//   where preventDefault is ignored and the page would scroll under a zooming map.
import { useCallback, useEffect, useId, useImperativeHandle, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode, type Ref } from "react";
import { colorize, mapPalette, maskRaster } from "./colour";
import type { Domain } from "./domain";
import { rowAt, rowCenter, type GridData } from "./grid";
import type { MapTool, RasterPoint, ToolPreview, ToolResult } from "./tools/types";
import type { MapViewApi } from "./useMapView";
import { THEME_COLORS } from "../theme/palette";

export type GridCanvasHandle = {
  /** The offscreen nx×ny raster (for PNG export), or null without canvas support. */
  raster: () => HTMLCanvasElement | null;
  /** Milliseconds the last recolour took. */
  lastColorizeMs: () => number;
  focus: () => void;
  /** Fit the whole raster into the measured viewport. */
  fit: () => void;
};

export type GridCanvasProps = {
  grid: GridData;
  values: ArrayLike<number> | null;
  domain: Domain | null;
  dark: boolean;
  view: MapViewApi;
  selection?: Uint8Array | null;
  /** Rows drawn with hatching (extrapolation, approximate preview). */
  hatch?: Uint8Array | null;
  tool?: MapTool | null;
  opacity?: number;
  /** Clip to the left/right of a divider at this fraction of the width (swipe). */
  clip?: { side: "left" | "right"; at: number } | null;
  height?: number | string;
  label: string;
  /** Text for the aria-live readout and tooltip of a row (or −1 for an empty cell). */
  describe?: (row: number, p: RasterPoint) => string;
  onHover?: (row: number, p: RasterPoint | null) => void;
  /** Click (pan tool) or Enter on the cursor: pin the inspector. */
  onPick?: (row: number) => void;
  onToolResult?: (r: ToolResult) => void;
  /** Extra SVG content in raster units (drawn inside the view transform). */
  overlay?: ReactNode;
  /** Fixed screen-space SVG content (scale bar, north arrow). */
  chrome?: (size: { w: number; h: number }) => ReactNode;
  handleRef?: Ref<GridCanvasHandle>;
  /** Keyboard and pointer input off (the passive pane of a swipe). */
  passive?: boolean;
};

function previewShape(p: ToolPreview): ReactNode {
  switch (p.kind) {
    case "rect":
      return <rect x={Math.min(p.x0, p.x1)} y={Math.min(p.y0, p.y1)} width={Math.abs(p.x1 - p.x0)} height={Math.abs(p.y1 - p.y0)} className="brush" vectorEffect="non-scaling-stroke" />;
    case "circle":
      return <circle cx={p.cx} cy={p.cy} r={p.r} className="brush" vectorEffect="non-scaling-stroke" />;
    case "polygon":
      return <polyline points={p.points.map((q) => `${q.px},${q.py}`).join(" ")} className="brush" vectorEffect="non-scaling-stroke" />;
    case "brush":
      return <circle cx={p.cx} cy={p.cy} r={p.r} fill="none" style={{ stroke: p.erase ? "var(--critical)" : "var(--ink)", strokeDasharray: p.erase ? "4 3" : undefined }} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />;
  }
}

export function GridCanvas(props: GridCanvasProps) {
  const { grid, values, domain, dark, view, selection, hatch, tool, opacity = 1, clip, label, describe, onHover, onPick, onToolResult, passive } = props;
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const off = useRef<HTMLCanvasElement | null>(null);
  const hatchOff = useRef<HTMLCanvasElement | null>(null);
  // Scratch canvases for the hatch pass, reused across frames (pan/zoom redraws at 60 fps).
  const hatchScratch = useRef<{ tmp: HTMLCanvasElement; pattern: CanvasPattern | null; patternDark: boolean | null } | null>(null);
  const buf = useRef<{ img: ImageData | null; u32: Uint32Array }>({ img: null, u32: new Uint32Array(0) });
  const [size, setSize] = useState({ w: 640, h: 480 });
  const [cursor, setCursor] = useState<RasterPoint | null>(null);
  const [hover, setHover] = useState<RasterPoint | null>(null);
  const [live, setLive] = useState("");
  const [dragging, setDragging] = useState(false);
  const [paintSeq, bump] = useState(0);
  const lastMs = useRef(0);
  const drag = useRef<{ x: number; y: number; moved: boolean; pan: boolean } | null>(null);
  const space = useRef(false);
  const frame = useRef(0);
  const helpId = useId();

  // ---------------------------------------------------------------- size
  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) setSize((s) => (s.w === r.width && s.h === r.height ? s : { w: r.width, h: r.height }));
    };
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Fit once, and again when the box or grid changes while the view is still the automatic fit
  // (a card whose layout settles after mount); a panned or zoomed view is kept.
  const fitKey = useRef("");
  useEffect(() => {
    const key = `${grid.nx}x${grid.ny}@${size.w}x${size.h}`;
    if (view.needsFit || (view.auto && fitKey.current !== key)) {
      fitKey.current = key;
      view.fit(grid.nx, grid.ny, size.w, size.h);
    }
  }, [view, grid.nx, grid.ny, size.w, size.h]);

  // ---------------------------------------------------------------- recolour
  const recolor = useCallback(() => {
    const nx = grid.nx;
    const ny = grid.ny;
    if (buf.current.u32.length !== nx * ny) buf.current.u32 = new Uint32Array(nx * ny);
    const t0 = performance.now();
    if (values && domain) colorize(buf.current.u32, grid.rowToPix, values, domain, mapPalette(dark), { selection });
    else buf.current.u32.fill(0);
    lastMs.current = performance.now() - t0;
    if (typeof document === "undefined") return;
    if (!off.current) off.current = document.createElement("canvas");
    const oc = off.current;
    const g = oc.getContext("2d");
    if (!g) return; // no canvas (tests): the colour buffer is still computed
    if (oc.width !== nx || oc.height !== ny) {
      oc.width = nx;
      oc.height = ny;
      buf.current.img = null;
    }
    if (!buf.current.img) buf.current.img = g.createImageData(nx, ny);
    new Uint32Array(buf.current.img.data.buffer).set(buf.current.u32);
    g.putImageData(buf.current.img, 0, 0);
  }, [grid, values, domain, dark, selection]);

  const rehatch = useCallback(() => {
    if (!hatch || typeof document === "undefined") {
      hatchOff.current = null;
      return;
    }
    const hc = document.createElement("canvas");
    hc.width = grid.nx;
    hc.height = grid.ny;
    const g = hc.getContext("2d");
    if (!g) return;
    const img = g.createImageData(grid.nx, grid.ny);
    maskRaster(img.data, grid.rowToPix, hatch);
    g.putImageData(img, 0, 0);
    hatchOff.current = hc;
  }, [grid, hatch]);

  // ---------------------------------------------------------------- draw
  const draw = useCallback(() => {
    frame.current = 0;
    const c = canvas.current;
    if (!c) return;
    const g = c.getContext("2d");
    if (!g) return;
    const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
    const W = Math.round(size.w * dpr);
    const H = Math.round(size.h * dpr);
    if (c.width !== W || c.height !== H) {
      c.width = W;
      c.height = H;
    }
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, W, H);
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (clip) {
      g.save();
      g.beginPath();
      const x = clip.at * size.w;
      if (clip.side === "left") g.rect(0, 0, x, size.h);
      else g.rect(x, 0, size.w - x, size.h);
      g.clip();
    }
    g.imageSmoothingEnabled = false;
    g.globalAlpha = opacity;
    const { scale, tx, ty } = view.state;
    if (off.current) g.drawImage(off.current, tx, ty, grid.nx * scale, grid.ny * scale);
    g.globalAlpha = 1;
    if (hatchOff.current) {
      // Hatch: the mask raster scaled up, then a diagonal-line pattern kept only inside it.
      const scratch = (hatchScratch.current ??= { tmp: document.createElement("canvas"), pattern: null, patternDark: null });
      const tmp = scratch.tmp;
      const tw = Math.max(1, Math.round(size.w));
      const th = Math.max(1, Math.round(size.h));
      if (tmp.width !== tw || tmp.height !== th) {
        tmp.width = tw;
        tmp.height = th;
      }
      const t = tmp.getContext("2d");
      if (t) {
        t.globalCompositeOperation = "source-over";
        t.clearRect(0, 0, tw, th);
        t.imageSmoothingEnabled = false;
        t.drawImage(hatchOff.current, tx, ty, grid.nx * scale, grid.ny * scale);
        t.globalCompositeOperation = "source-in";
        if (scratch.patternDark !== dark) {
          const pat = document.createElement("canvas");
          pat.width = 6;
          pat.height = 6;
          const pg = pat.getContext("2d");
          if (pg) {
            pg.strokeStyle = dark ? THEME_COLORS.dark.hatch : THEME_COLORS.light.hatch;
            pg.lineWidth = 1.2;
            pg.beginPath();
            pg.moveTo(0, 6);
            pg.lineTo(6, 0);
            pg.stroke();
          }
          scratch.pattern = pg ? t.createPattern(pat, "repeat") : null;
          scratch.patternDark = dark;
        }
        if (scratch.pattern) {
          t.fillStyle = scratch.pattern;
          t.fillRect(0, 0, tw, th);
        }
        g.drawImage(tmp, 0, 0);
      }
    }
    if (clip) g.restore();
  }, [size, view.state, grid, opacity, clip, dark]);

  // The pending frame always draws the latest state.
  const drawRef = useRef(draw);
  drawRef.current = draw;
  const schedule = useCallback(() => {
    if (frame.current) return;
    if (typeof requestAnimationFrame === "function") frame.current = requestAnimationFrame(() => drawRef.current());
    else drawRef.current();
  }, []);

  useEffect(() => {
    recolor();
    bump((x) => x + 1);
  }, [recolor]);
  useEffect(() => {
    rehatch();
    schedule();
  }, [rehatch, schedule]);
  // Redraw only when something drawn changed (view, size, raster, opacity, clip, theme), not
  // on every render: hover and cursor updates re-render the overlay but not the canvas.
  useEffect(() => {
    schedule();
  }, [draw, paintSeq, schedule]);
  useEffect(
    () => () => {
      if (frame.current && typeof cancelAnimationFrame === "function") cancelAnimationFrame(frame.current);
      frame.current = 0; // a StrictMode remount must be able to schedule again
    },
    [],
  );

  useImperativeHandle(
    props.handleRef,
    () => ({ raster: () => off.current, lastColorizeMs: () => lastMs.current, focus: () => wrap.current?.focus(), fit: () => view.fit(grid.nx, grid.ny, size.w, size.h) }),
    [view, grid.nx, grid.ny, size.w, size.h],
  );

  // ---------------------------------------------------------------- input
  const toRaster = (clientX: number, clientY: number): RasterPoint => {
    const r = wrap.current?.getBoundingClientRect();
    const sx = clientX - (r?.left ?? 0);
    const sy = clientY - (r?.top ?? 0);
    const [px, py] = view.toRaster(sx, sy);
    return { px, py };
  };

  const announce = useCallback(
    (p: RasterPoint) => {
      const row = rowAt(grid, p.px, p.py);
      setLive(describe ? describe(row, p) : row < 0 ? "No observation" : `Cell row ${row}`);
    },
    [grid, describe],
  );

  // Wheel zoom about the cursor. Pixel, line and page deltas are normalised to pixels.
  const zoomAtRef = useRef(view.zoomAt);
  zoomAtRef.current = view.zoomAt;
  useEffect(() => {
    const el = wrap.current;
    if (!el || passive) return;
    const onWheel = (e: globalThis.WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaMode === 2 ? e.deltaY * (r.height || 480) : e.deltaY;
      zoomAtRef.current(e.clientX - r.left, e.clientY - r.top, Math.exp(-dy * 0.0015));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [passive]);

  const emit = (res: ToolResult | null) => {
    if (res) onToolResult?.(res);
  };

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (passive || e.button !== 0) return;
    // no scroll: a map partly below the fold must not move under the pointer between down and up
    wrap.current?.focus({ preventScroll: true });
    const pan = !tool || tool.pans || space.current;
    drag.current = { x: e.clientX, y: e.clientY, moved: false, pan };
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
    if (!pan && tool) emit(tool.down(toRaster(e.clientX, e.clientY)));
    if (pan) setDragging(true);
  };
  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    const p = toRaster(e.clientX, e.clientY);
    let d = drag.current;
    if (d && e.pointerType === "mouse" && (e.buttons & 1) === 0) {
      // The button was released where we never heard the pointerup: the drag is over.
      drag.current = d = null;
      setDragging(false);
    }
    if (d) {
      const dx = e.clientX - d.x;
      const dy = e.clientY - d.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) d.moved = true;
      if (d.pan) {
        view.panBy(dx, dy);
        d.x = e.clientX;
        d.y = e.clientY;
        return;
      }
      if (tool) emit(tool.move(p, e.buttons));
    }
    setHover(p);
    const row = rowAt(grid, p.px, p.py);
    onHover?.(row, p);
  };
  const onPointerUp = (e: PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    drag.current = null;
    setDragging(false);
    if (!d) return;
    const p = toRaster(e.clientX, e.clientY);
    if (d.pan) {
      if (!d.moved) {
        const row = rowAt(grid, p.px, p.py);
        setCursor(p);
        announce(p);
        if (row >= 0) onPick?.(row);
      }
      return;
    }
    if (tool) emit(tool.up(p));
  };

  /** The browser took the pointer away (touch scroll, focus loss): drop the drag. */
  const onPointerCancel = () => {
    const d = drag.current;
    drag.current = null;
    setDragging(false);
    if (d && !d.pan) tool?.cancel();
  };

  const moveCursor = (dx: number, dy: number) => {
    const start = cursor ?? (() => {
      const [cx, cy] = view.toRaster(size.w / 2, size.h / 2);
      return { px: Math.floor(Math.min(grid.nx - 1, Math.max(0, cx))) + 0.5, py: Math.floor(Math.min(grid.ny - 1, Math.max(0, cy))) + 0.5 };
    })();
    const next = { px: Math.min(grid.nx - 0.5, Math.max(0.5, start.px + dx)), py: Math.min(grid.ny - 0.5, Math.max(0.5, start.py + dy)) };
    setCursor(next);
    announce(next);
    onHover?.(rowAt(grid, next.px, next.py), next);
    // keep the cursor in view
    const [sx, sy] = view.toScreen(next.px, next.py);
    const m = 24;
    if (sx < m || sy < m || sx > size.w - m || sy > size.h - m) view.panBy(sx < m ? m - sx + 40 : sx > size.w - m ? size.w - m - sx - 40 : 0, sy < m ? m - sy + 40 : sy > size.h - m ? size.h - m - sy - 40 : 0);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (passive) return;
    const step = e.shiftKey ? 10 : 1;
    switch (e.key) {
      case "ArrowLeft":
        moveCursor(-step, 0);
        break;
      case "ArrowRight":
        moveCursor(step, 0);
        break;
      case "ArrowUp":
        moveCursor(0, -step);
        break;
      case "ArrowDown":
        moveCursor(0, step);
        break;
      case "Enter": {
        if (tool && tool.id === "polygon") {
          emit(tool.finish());
          break;
        }
        if (!cursor) {
          moveCursor(0, 0);
          break;
        }
        const row = rowAt(grid, cursor.px, cursor.py);
        if (row >= 0) {
          onPick?.(row);
          setLive(`Pinned. ${describe ? describe(row, cursor) : `Cell row ${row}`}`);
        }
        break;
      }
      case "+":
      case "=":
        view.zoomAt(size.w / 2, size.h / 2, 1.5);
        break;
      case "-":
      case "_":
        view.zoomAt(size.w / 2, size.h / 2, 1 / 1.5);
        break;
      case "0":
        view.fit(grid.nx, grid.ny, size.w, size.h);
        break;
      case "Escape":
        tool?.cancel();
        bump((x) => x + 1);
        break;
      case " ":
        space.current = true;
        break;
      default:
        return;
    }
    e.preventDefault();
  };

  const { scale, tx, ty } = view.state;
  const preview = tool ? tool.preview(hover) : null;
  const cursorRow = cursor ? rowAt(grid, cursor.px, cursor.py) : -1;
  const cursorBox = useMemo(() => {
    if (!cursor) return null;
    const c = cursorRow >= 0 ? rowCenter(grid, cursorRow) : cursor;
    return { x: Math.floor(c.px), y: Math.floor(c.py) };
  }, [cursor, cursorRow, grid]);

  return (
    <div
      ref={wrap}
      className="map-stage"
      style={{ height: props.height ?? 480, position: clip ? "absolute" : "relative", inset: clip ? 0 : undefined, border: clip ? 0 : undefined, background: clip ? "transparent" : undefined, pointerEvents: passive ? "none" : undefined }}
      tabIndex={passive ? -1 : 0}
      role="application"
      aria-roledescription="map"
      aria-label={label}
      aria-describedby={helpId}
      data-tool={tool?.id ?? "pan"}
      data-dragging={dragging || undefined}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerCancel}
      onPointerLeave={() => {
        setHover(null);
        onHover?.(-1, null);
      }}
      onDoubleClick={() => tool?.id === "polygon" && emit(tool.finish())}
      onKeyDown={onKeyDown}
      onKeyUp={(e) => {
        if (e.key === " ") space.current = false;
      }}
    >
      <canvas ref={canvas} style={{ width: "100%", height: "100%" }} aria-hidden="true" />
      <svg className="map-overlay" width="100%" height="100%" aria-hidden="true">
        <g transform={`translate(${tx},${ty}) scale(${scale})`}>
          {props.overlay}
          {preview ? previewShape(preview) : null}
          {cursorBox ? <rect x={cursorBox.x} y={cursorBox.y} width={1} height={1} fill="none" style={{ stroke: "var(--ink)" }} strokeWidth={2} vectorEffect="non-scaling-stroke" /> : null}
        </g>
        {props.chrome?.(size)}
      </svg>
      {passive ? null : (
        <>
          <span id={helpId} className="sr-only">
            Arrow keys move the cell cursor (Shift for 10 cells), Enter pins the inspector, plus and minus zoom, 0 fits the map, Escape cancels the current tool.
          </span>
          <div className="sr-only" aria-live="polite" role="status" data-testid="map-live">
            {live}
          </div>
        </>
      )}
    </div>
  );
}
