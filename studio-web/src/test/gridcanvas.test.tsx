import { describe, expect, it, vi } from "vitest";
import { act, createRef } from "react";
import type { LayerMeta } from "../api/types";
import { buildCellToPt, colorize, mapPalette, outlinePath } from "../map/colour";
import { computeDomain, type Domain } from "../map/domain";
import { GridCanvas, type GridCanvasHandle } from "../map/GridCanvas";
import { fmtLonLat, rasterToLonLat, rasterToXY, rowAt, xyToRaster } from "../map/grid";
import { useMapView } from "../map/useMapView";
import { getLuts, hexToRgb255, packRgba, THEME_COLORS } from "../theme/palette";
import { bigGrid, grid3 } from "./grid";
import { flush, key, render } from "./render";

const seq = (lo: number, hi: number, extra: Partial<Domain> = {}): Domain => ({ kind: "seq", lo, hi, center: null, mult: 1, zeroBlank: false, nCat: 0, ...extra });

describe("cellToPt on a 3×3 grid", () => {
  it("is north-up: raster index (ny−1−iy)·nx + ix", () => {
    const g = grid3();
    expect([...g.cellToPt]).toEqual([0, 1, -1, 2, -1, 3, 4, 5, 6]);
    expect([...buildCellToPt(g.ix, g.iy, 3, 3)]).toEqual([...g.cellToPt]);
    expect([...g.rowToPix]).toEqual([0, 1, 3, 5, 6, 7, 8]);
  });
  it("hit-tests raster points and converts coordinates", () => {
    const g = grid3();
    expect(rowAt(g, 0.5, 0.5)).toBe(0); // north-west cell
    expect(rowAt(g, 1.2, 1.7)).toBe(-1); // the empty centre
    expect(rowAt(g, 2.9, 2.9)).toBe(6); // south-east
    expect(rowAt(g, 3.1, 0)).toBe(-1);
    // cell (ix=0, iy=0) centre is (x0, y0) in metres
    expect(rasterToXY(g, 0.5, 2.5)).toEqual([1000, 2000]);
    expect(xyToRaster(g, 1060, 2060)).toEqual({ px: 2.5, py: 0.5 });
    expect(rasterToLonLat(g, 0.5, 2.5)).toEqual([-71.5, 41.7]);
    expect(fmtLonLat(rasterToLonLat(g, 2.5, 0.5))).toBe("41.9000°N, 71.3000°W");
  });
});

describe("colorize on a 3×3 grid", () => {
  const g = grid3();
  const pal = mapPalette(false);
  const l = getLuts(false);
  it("maps values through the sequential LUT; NaN is transparent; empty cells stay clear", () => {
    const out = new Uint32Array(9);
    const vals = Float32Array.from([0, 1, 2, 3, NaN, 5, 6]);
    colorize(out, g.rowToPix, vals, seq(0, 6), pal);
    expect(out[0]).toBe(l.seq32[0]);
    expect(out[1]).toBe(l.seq32[Math.round((1 / 6) * 255)]);
    expect(out[3]).toBe(l.seq32[Math.round((2 / 6) * 255)]);
    expect(out[5]).toBe(l.seq32[128]);
    expect(out[6]).toBe(0); // NaN
    expect(out[2]).toBe(0); // no cell
    expect(out[4]).toBe(0); // no cell
    expect(out[8]).toBe(l.seq32[255]);
  });
  it("clamps outside the domain and applies mult", () => {
    const out = new Uint32Array(9);
    colorize(out, g.rowToPix, Float32Array.from([-5, 100, 50, 0, 0, 0, 0]), seq(0, 1, { mult: 0.01 }), pal);
    expect(out[0]).toBe(l.seq32[0]);
    expect(out[1]).toBe(l.seq32[255]); // 100 × 0.01 = 1
    expect(out[3]).toBe(l.seq32[128]); // 50 × 0.01 = 0.5
  });
  it("diverging: the centre is index 128 and both halves stretch to the ends", () => {
    const out = new Uint32Array(9);
    const d: Domain = { kind: "div", lo: -2, hi: 2, center: 0, mult: 1, zeroBlank: false, nCat: 0 };
    colorize(out, g.rowToPix, Float32Array.from([-2, 0, 2, -1, 1, NaN, 3]), d, pal);
    expect([out[0], out[1], out[3], out[5], out[6], out[8]]).toEqual([l.div32[0], l.div32[128], l.div32[255], l.div32[64], l.div32[191], l.div32[255]]);
    expect(out[7]).toBe(0);
  });
  it("zero_blank paints values ≤ 0 as --nodata", () => {
    const out = new Uint32Array(9);
    const [r, gg, b] = hexToRgb255(THEME_COLORS.light.nodata);
    colorize(out, g.rowToPix, Float32Array.from([0, -1, 3, 6, NaN, 0, 1.5]), seq(0, 6, { zeroBlank: true }), pal);
    expect(out[0]).toBe(packRgba(r, gg, b));
    expect(out[1]).toBe(packRgba(r, gg, b));
    expect(out[3]).toBe(l.seq32[128]);
    expect(out[6]).toBe(0);
  });
  it("categorical: 0 grey, 1–3 the categorical colours, out-of-range classes no data", () => {
    const out = new Uint32Array(9);
    const meta = { scale: "cat", labels: ["insufficient", "saturating", "linear", "S-shaped"], mult: 1, zero_blank: false, stats: { n: 7, lo: 0, hi: 3, mean: null, p1: null, p2: null, p50: null, p98: null, p99: null } } as unknown as LayerMeta;
    colorize(out, g.rowToPix, Uint8Array.from([0, 1, 2, 3, 255, 1, 0]), computeDomain(meta), pal);
    expect([out[0], out[1], out[3], out[5]]).toEqual([pal.gray, pal.cat[0], pal.cat[1], pal.cat[2]]);
    expect(out[6]).toBe(pal.nodata);
  });
  it("dims cells outside the selection", () => {
    const out = new Uint32Array(9);
    colorize(out, g.rowToPix, Float32Array.from([3, 3, 3, 3, 3, 3, 3]), seq(0, 6), pal, { selection: Uint8Array.from([1, 0, 0, 0, 0, 0, 1]), dimAlpha: 70 });
    expect(out[0] >>> 24).toBe(255);
    expect(out[1] >>> 24).toBe(70);
    expect(out[8] >>> 24).toBe(255);
  });
  it("outlines classes along cell edges", () => {
    const d = outlinePath(g.cellToPt, 3, 3, (r) => (r === 0 ? 1 : null));
    expect(d.split("M").filter(Boolean).sort()).toEqual(["0,0h1", "0,0v1", "0,1h1", "1,0v1"]);
  });
});

describe("GridCanvas", () => {
  const canvasAvailable = (() => {
    try {
      return !!document.createElement("canvas").getContext("2d");
    } catch {
      return false;
    }
  })();

  it("recolours a 54,701-cell layer in under 10 ms", () => {
    const g = bigGrid();
    expect(g.n).toBe(54701);
    const vals = new Float32Array(g.n);
    for (let i = 0; i < g.n; i++) vals[i] = 80 + 15 * Math.sin(i * 0.001) + (i % 97) * 0.05;
    const meta = { scale: "div", center: 88, mult: 1, zero_blank: false, labels: null, stats: { n: g.n, lo: 70, hi: 100, mean: 88, p1: 72, p2: 73, p50: 88, p98: 99, p99: 99.5 } } as unknown as LayerMeta;
    const d = computeDomain(meta, vals);
    const out = new Uint32Array(g.nx * g.ny);
    const pal = mapPalette(false);
    for (let i = 0; i < 5; i++) colorize(out, g.rowToPix, vals, d, pal); // warm-up (JIT)
    const times: number[] = [];
    for (let i = 0; i < 15; i++) {
      const t0 = performance.now();
      colorize(out, g.rowToPix, vals, d, pal);
      times.push(performance.now() - t0);
    }
    times.sort((a, b) => a - b);
    const median = times[Math.floor(times.length / 2)];
    expect(median).toBeLessThan(10);
  });

  it("the mounted component recolours through the same path and reports its time", async () => {
    const g = bigGrid();
    const vals = Float32Array.from({ length: g.n }, (_, i) => i % 100);
    const ref = createRef<GridCanvasHandle>();
    function Host() {
      const view = useMapView();
      return <GridCanvas handleRef={ref} grid={g} view={view} dark={false} values={vals} domain={seq(0, 99)} label="bench" />;
    }
    render(<Host />);
    await flush();
    expect(ref.current).not.toBeNull();
    expect(ref.current!.lastColorizeMs()).toBeLessThan(50); // first call includes JIT warm-up; the steady state is the test above
  });

  it.skipIf(!canvasAvailable)("draws the raster into a 2D context (needs canvas support; jsdom has none, so this is skipped there)", async () => {
    const g = grid3();
    const ref = createRef<GridCanvasHandle>();
    function Host() {
      const view = useMapView();
      return <GridCanvas handleRef={ref} grid={g} view={view} dark={false} values={Float32Array.from([0, 1, 2, 3, 4, 5, 6])} domain={seq(0, 6)} label="3x3" />;
    }
    render(<Host />);
    await flush();
    expect(ref.current!.raster()?.width).toBe(3);
  });

  it("zooms on the wheel about the cursor and keeps the page from scrolling", async () => {
    const g = grid3();
    let view: ReturnType<typeof useMapView> | null = null;
    function Host() {
      const v = useMapView({ scale: 10, tx: 0, ty: 0 });
      view = v;
      return <GridCanvas grid={g} view={v} dark={false} values={Float32Array.from([0, 1, 2, 3, 4, 5, 6])} domain={seq(0, 6)} label="3x3" />;
    }
    const { container } = render(<Host />);
    await flush();
    const stage = container.querySelector('[role="application"]')!;
    const ev = new WheelEvent("wheel", { deltaY: -200, clientX: 15, clientY: 15, bubbles: true, cancelable: true });
    act(() => {
      stage.dispatchEvent(ev);
    });
    expect(ev.defaultPrevented).toBe(true); // a passive (React) listener could not do this
    const s = view!.state;
    expect(s.scale).toBeCloseTo(10 * Math.exp(0.3), 9);
    // the raster point under the cursor (1.5, 1.5) stays under it
    expect((15 - s.tx) / s.scale).toBeCloseTo(1.5, 9);
    expect((15 - s.ty) / s.scale).toBeCloseTo(1.5, 9);
  });

  it("a cancelled pointer ends the pan instead of panning on later moves", async () => {
    const g = grid3();
    let view: ReturnType<typeof useMapView> | null = null;
    function Host() {
      const v = useMapView({ scale: 10, tx: 0, ty: 0 });
      view = v;
      return <GridCanvas grid={g} view={v} dark={false} values={null} domain={null} label="3x3" />;
    }
    const { container } = render(<Host />);
    await flush();
    const stage = container.querySelector('[role="application"]')!;
    const fire = (type: string, x: number, buttons: number) =>
      act(() => {
        stage.dispatchEvent(new PointerEvent(type, { clientX: x, clientY: 5, button: 0, buttons, pointerId: 1, pointerType: "mouse", bubbles: true }));
      });
    fire("pointerdown", 5, 1);
    fire("pointermove", 25, 1);
    expect(view!.state.tx).toBe(20);
    fire("pointercancel", 25, 0);
    fire("pointermove", 60, 1);
    expect(view!.state.tx).toBe(20);
    expect(stage.getAttribute("data-dragging")).toBeNull();
  });

  it("redraws the canvas when the view changes, not on hover", async () => {
    const draws: number[] = [];
    const ctx = new Proxy({} as Record<string, unknown>, {
      get: (_t, k) =>
        k === "drawImage"
          ? () => void draws.push(1)
          : k === "createImageData"
            ? (w: number, h: number) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h })
            : () => {},
      set: () => true,
    });
    const gc = vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation((() => ctx) as unknown as typeof HTMLCanvasElement.prototype.getContext);
    try {
      const g = grid3();
      let view: ReturnType<typeof useMapView> | null = null;
      function Host() {
        const v = useMapView({ scale: 10, tx: 0, ty: 0 });
        view = v;
        return <GridCanvas grid={g} view={v} dark={false} values={Float32Array.from([0, 1, 2, 3, 4, 5, 6])} domain={seq(0, 6)} label="3x3" />;
      }
      const { container } = render(<Host />);
      await flush(3);
      await act(async () => await new Promise((r) => setTimeout(r, 40))); // let the frame run
      const first = draws.length;
      expect(first).toBeGreaterThan(0);
      const stage = container.querySelector('[role="application"]')!;
      for (const x of [3, 9, 15, 21, 27])
        act(() => {
          stage.dispatchEvent(new PointerEvent("pointermove", { clientX: x, clientY: 5, bubbles: true }));
        });
      await act(async () => await new Promise((r) => setTimeout(r, 40)));
      expect(draws.length).toBe(first); // hover only re-renders the SVG overlay
      act(() => view!.panBy(5, 0));
      await act(async () => await new Promise((r) => setTimeout(r, 40)));
      expect(draws.length).toBeGreaterThan(first);
    } finally {
      gc.mockRestore();
    }
  });

  it("moves a keyboard cursor and announces the cell in an aria-live region; Enter pins", async () => {
    const g = grid3();
    const picked: number[] = [];
    const vals = Float32Array.from([80, 81, 82, 83, 84, 85, 86]);
    function Host() {
      const view = useMapView({ scale: 10, tx: 0, ty: 0 });
      return (
        <GridCanvas
          grid={g}
          view={view}
          dark={false}
          values={vals}
          domain={seq(80, 86)}
          label="Map of observed temperature"
          describe={(row) => (row < 0 ? "No observation" : `Observed ${vals[row].toFixed(1)} °F, row ${row}`)}
          onPick={(r) => picked.push(r)}
        />
      );
    }
    const { container } = render(<Host />);
    await flush();
    const stage = container.querySelector('[role="application"]')!;
    const live = container.querySelector('[aria-live="polite"]')!;
    expect(stage.getAttribute("aria-label")).toBe("Map of observed temperature");
    expect(live.textContent).toBe("");
    key(stage, "ArrowRight"); // first key places the cursor at the view centre, then moves it
    const first = live.textContent;
    expect(first).toMatch(/Observed|No observation/);
    // Walk to the north-west corner (row 0) and pin it.
    for (let i = 0; i < 3; i++) key(stage, "ArrowLeft");
    for (let i = 0; i < 3; i++) key(stage, "ArrowUp");
    expect(live.textContent).toBe("Observed 80.0 °F, row 0");
    key(stage, "ArrowRight");
    expect(live.textContent).toBe("Observed 81.0 °F, row 1");
    key(stage, "ArrowDown");
    expect(live.textContent).toBe("No observation"); // the empty centre cell
    key(stage, "ArrowUp");
    key(stage, "Enter");
    expect(picked).toEqual([1]);
    expect(live.textContent).toBe("Pinned. Observed 81.0 °F, row 1");
  });
});
