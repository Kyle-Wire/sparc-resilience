import { describe, expect, it, vi } from "vitest";
import { lastDownload } from "../components/ui/download";
import { act } from "react";
import type { LayerGroup, LayerMeta } from "../api/types";
import { MapView, type MapSelectionEvent } from "../map/MapView";
import { Legend, divergingEnds } from "../map/Legend";
import { computeDomain } from "../map/domain";
import { runLayerLoader } from "../map/data";
import { grid3 } from "./grid";
import { byText, click, flush, key, mockFetch, render, typeInto } from "./render";

const stats = (lo: number, hi: number) => ({ n: 7, lo, hi, mean: (lo + hi) / 2, p1: lo, p2: lo, p50: (lo + hi) / 2, p98: hi, p99: hi });
const L = (p: Partial<LayerMeta> & { key: string }): LayerMeta => ({
  group: "temperature", label: p.key, unit: "degF", scale: "seq", center: null, decimals: 1, mult: 1, zero_blank: false, labels: null, desc: "", sign_note: null,
  source: null, dtype: "float32", stats: stats(80, 86), ...p,
});
const groups: LayerGroup[] = [
  { id: "temperature", label: "Temperature", layers: [L({ key: "obs", label: "Observed", scale: "div", center: 83 }), L({ key: "resid", label: "Residual", scale: "div", center: 0, stats: stats(-1, 1) })] },
  { id: "cv", label: "CV design", layers: [L({ key: "fold", label: "Fold", scale: "cat", unit: "", labels: ["train", "test", "buffer"], dtype: "uint8" })] },
];
const data: Record<string, Float32Array | Uint8Array> = {
  obs: Float32Array.from([80, 81, 82, 83, 84, 85, 86]),
  resid: Float32Array.from([-1, -0.5, 0, 0.2, 0.5, 1, NaN]),
  fold: Uint8Array.from([0, 1, 2, 0, 1, 2, 0]),
};

describe("MapView", () => {
  it("loads the chosen layer once, shows its legend and summary, and switches layers", async () => {
    const loads: string[] = [];
    const g = grid3();
    let key = "obs";
    const { container, rerender } = render(<MapView grid={g} groups={groups} layerKey={key} onLayerChange={(k) => (key = k)} loadLayer={async (m) => (loads.push(m.key), data[m.key])} />);
    await flush(4);
    expect(loads).toEqual(["obs"]);
    expect(container.querySelector(".map-side h3")!.textContent).toBe("Observed");
    expect(container.querySelector('[aria-label="Layer summary"]')!.textContent).toContain("Cells7");
    expect(container.textContent).toContain("cooler");
    expect(container.querySelector('[role="application"]')!.getAttribute("aria-label")).toBe("Map of Observed");
    typeInto(container.querySelector("select#" + CSS.escape(container.querySelector("label.cap[for]")!.getAttribute("for")!)), "resid");
    expect(key).toBe("resid");
    rerender(<MapView grid={g} groups={groups} layerKey={key} onLayerChange={(k) => (key = k)} loadLayer={async (m) => (loads.push(m.key), data[m.key])} />);
    await flush(4);
    expect(loads).toEqual(["obs", "resid"]);
    expect(container.querySelector(".map-side h3")!.textContent).toBe("Residual");
  });

  it("categorical layers show swatches with class shares", async () => {
    const { container } = render(<MapView grid={grid3()} groups={groups} layerKey="fold" loadLayer={async (m) => data[m.key]} />);
    await flush(4);
    const items = [...container.querySelectorAll('[aria-label="Fold classes"] [role="listitem"]')].map((li) => li.textContent);
    expect(items).toEqual(["train · 42.9%", "test · 28.6%", "buffer · 28.6%"]);
  });

  it("offers swipe, side-by-side and difference compare modes", async () => {
    const { container } = render(<MapView grid={grid3()} groups={groups} layerKey="obs" compareKey="resid" compareMode="swipe" loadLayer={async (m) => data[m.key]} />);
    await flush(4);
    const slider = container.querySelector('[role="slider"]')!;
    expect(slider.getAttribute("aria-valuenow")).toBe("50");
    expect(container.querySelectorAll('[role="application"]').length).toBe(2);
    const diff = render(<MapView grid={grid3()} groups={groups} layerKey="obs" compareKey="resid" compareMode="diff" allowDiff={false} loadLayer={async (m) => data[m.key]} />);
    await flush(4);
    const diffBtn = byText(diff.container, "button", "Difference")!;
    expect(diffBtn.hasAttribute("disabled")).toBe(true);
    expect(diffBtn.getAttribute("title")).toBe("Difference needs the same grid");
  });

  it("turns a legend-histogram brush into a selection mask", async () => {
    const events: MapSelectionEvent[] = [];
    const g = grid3();
    const { container } = render(<MapView grid={g} groups={groups} layerKey="obs" onSelection={(e) => events.push(e)} loadLayer={async (m) => data[m.key]} />);
    await flush(4);
    const bars = container.querySelectorAll('.map-side svg.chart rect[role="img"]');
    expect(bars.length).toBeGreaterThan(0);
    const last = bars[bars.length - 1];
    last.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    await flush(2);
    expect(events).toHaveLength(1);
    expect(events[0].source).toBe("legend");
    expect(events[0].mask).not.toBeNull();
    expect([...events[0].mask!].reduce((a, b) => a + b, 0)).toBeGreaterThan(0);
  });

  it("the end bins of the legend histogram hold the cells beyond the 2–98% clip, and brushing them selects those cells", async () => {
    // Stats clip at 81..85, but cells 80 and 86 are painted with the end colours.
    const meta = L({ key: "obs", label: "Observed", stats: { n: 7, lo: 80, hi: 86, mean: 83, p1: 80, p2: 81, p50: 83, p98: 85, p99: 86 } });
    const events: MapSelectionEvent[] = [];
    const { container } = render(<MapView grid={grid3()} groups={[{ id: "t", label: "T", layers: [meta] }]} layerKey="obs" onSelection={(e) => events.push(e)} loadLayer={async () => data.obs} />);
    await flush(4);
    const bars = container.querySelectorAll('.map-side svg.chart rect[role="img"]');
    const counts = [...bars].map((b) => Number(/: (\d+) cells/.exec(b.getAttribute("aria-label")!)![1]));
    expect(counts.reduce((a, b) => a + b, 0)).toBe(7); // every observed cell is in a bin
    expect([counts[0], counts[counts.length - 1]]).toEqual([2, 2]);
    key(bars[bars.length - 1], "Enter"); // the hottest bin
    await flush(2);
    expect([...events[0].mask!]).toEqual([0, 0, 0, 0, 0, 1, 1]); // 85 and 86 (above the clip)
    key(bars[0], "Enter"); // the coolest bin
    await flush(2);
    expect([...events[1].mask!]).toEqual([1, 1, 0, 0, 0, 0, 0]); // 80 (below the clip) and 81
  });

  it("categorical layers without labels still get one swatch per class", () => {
    const meta = L({ key: "cls", label: "Class", scale: "cat", unit: "", labels: null, dtype: "uint8", stats: stats(0, 2) });
    const vals = Uint8Array.from([0, 1, 2, 2, 1, 0, 0]);
    const { container } = render(<Legend meta={meta} domain={computeDomain(meta, vals)} dark={false} values={vals} />);
    expect([...container.querySelectorAll('[role="listitem"]')].map((li) => li.textContent)).toEqual(["Class 0 · 42.9%", "Class 1 · 28.6%", "Class 2 · 28.6%"]);
  });

  it("Legend shows ramp ends with units and the zero_blank note", () => {
    const meta = { ...groups[0].layers[0], key: "alloc_dose", label: "Planned canopy", unit: "pp", scale: "seq" as const, center: null, zero_blank: true };
    const vals = Float32Array.from([0, 0, 2, 4, 6, 8, 10]);
    const { container } = render(<Legend meta={meta} domain={computeDomain(meta, vals)} dark={false} values={vals} />);
    expect(container.querySelector(".legend-ticks")!.textContent).toMatch(/^≤ 0\.0≥ \d+\.\d pp$/);
    expect(container.textContent).toContain("at or below zero are shown as no data");
    expect(container.querySelector('[aria-label="Layer summary"]')!.textContent).toContain("Cells5");
  });

  it("drives the rect tool with the pointer and pins cells with a click (pan tool)", async () => {
    const events: MapSelectionEvent[] = [];
    const g = grid3();
    const { container } = render(<MapView grid={g} groups={groups} layerKey="obs" tools={["pan", "rect"]} onSelection={(e) => events.push(e)} loadLayer={async (m) => data[m.key]} />);
    await flush(4);
    const stage = container.querySelector('[role="application"]')!;
    // jsdom has no layout: the view fits the 3×3 raster into the default 640×480 viewport.
    const scale = Math.min((640 - 16) / 3, (480 - 16) / 3);
    const tx = (640 - 3 * scale) / 2;
    const ty = (480 - 3 * scale) / 2;
    const at = (px: number, py: number) => ({ clientX: tx + px * scale, clientY: ty + py * scale });
    const fire = (type: string, p: { clientX: number; clientY: number }, buttons = 1) =>
      act(() => {
        stage.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, button: 0, buttons, ...p }));
      });
    // pan tool: a click without movement pins the cell
    fire("pointerdown", at(0.5, 0.5));
    fire("pointerup", at(0.5, 0.5), 0);
    await flush(2);
    expect(container.querySelector('[aria-label="Pinned cell"]')!.textContent).toContain("row 0");
    // rect tool
    click(container.querySelector('button[aria-label="Select a rectangle"]'));
    expect(container.querySelector('[role="application"]')!.getAttribute("data-tool")).toBe("rect");
    fire("pointerdown", at(0, 0));
    fire("pointermove", at(2, 2));
    fire("pointerup", at(2, 2), 0);
    await flush(2);
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({ source: "rect", portable: true, spec: { kind: "rect", crs: "EPSG:4326" } });
    expect([...events[0].mask!]).toEqual([1, 1, 1, 0, 0, 0, 0]);
  });

  it("labels diverging ends by units and sign convention", () => {
    expect(divergingEnds({ unit: "degF", sign_note: "negative = cooler" })).toEqual(["cooler", "warmer"]);
    expect(divergingEnds({ unit: "degF", sign_note: "positive = cooler" })).toEqual(["less cooling", "more cooling"]);
    expect(divergingEnds({ unit: "pp", sign_note: null })).toEqual(["lower", "higher"]);
  });
});

describe("MapView readout", () => {
  it("reads zero_blank cells as 'none' and others with their unit", async () => {
    const dose = L({ key: "dose", label: "Planned dose", unit: "pp", zero_blank: true, stats: stats(0, 4) });
    const { container } = render(<MapView grid={grid3()} groups={[{ id: "budget", label: "Budget", layers: [dose] }]} layerKey="dose" loadLayer={async () => Float32Array.from([0, 2.5, 0, 1, 0, 3, 4])} />);
    await flush(4);
    const stage = container.querySelector('[role="application"]')!;
    const live = container.querySelector('[data-testid="map-live"]')!;
    key(stage, "Enter"); // places the cursor
    key(stage, "ArrowLeft");
    key(stage, "ArrowUp"); // north-west cell: row 0, untreated
    expect(live.textContent).toMatch(/^Planned dose: none · .* · row 0$/);
    key(stage, "ArrowRight"); // row 1
    expect(live.textContent).toMatch(/^Planned dose: 2\.5 pp · .* · row 1$/);
  });
});

describe("MapView PNG export", () => {
  async function exportTexts(layerKey: string): Promise<string[]> {
    const texts: string[] = [];
    const { container } = render(<MapView grid={grid3()} groups={groups} layerKey={layerKey} loadLayer={async (m) => data[m.key]} />);
    await flush(4);
    // jsdom has no canvas: record what the export draws.
    const ctx = new Proxy({} as Record<string, unknown>, {
      get: (_t, k) => (k === "fillText" ? (t: string) => void texts.push(t) : k === "measureText" ? () => ({ width: 10 }) : () => {}),
      set: () => true,
    });
    const gc = vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation((() => ctx) as unknown as typeof HTMLCanvasElement.prototype.getContext);
    const tb = vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation(function (cb: BlobCallback) {
      cb(new Blob(["png"], { type: "image/png" }));
    });
    try {
      click(container.querySelector('button[aria-label^="Export map as PNG"]'));
      await flush(3);
    } finally {
      gc.mockRestore();
      tb.mockRestore();
    }
    expect(lastDownload()!.name).toBe(`map-${layerKey}.png`);
    return texts;
  }

  it("draws the class legend for categorical layers", async () => {
    const texts = await exportTexts("fold");
    expect(texts).toEqual(expect.arrayContaining(["Fold", "train", "test", "buffer"]));
  });

  it("draws the ramp range and cooler/warmer ends for diverging temperature layers", async () => {
    const texts = await exportTexts("resid");
    expect(texts.some((t) => t.startsWith("≤ −"))).toBe(true);
    expect(texts).toEqual(expect.arrayContaining(["cooler", "warmer"]));
  });
});

describe("runLayerLoader", () => {
  it("fetches a layer once, shares concurrent loads, and refetches when the catalogue says it changed", async () => {
    let served = Float32Array.from([1, 2, 3]);
    const m = mockFetch({
      "GET /api/runs/r_live/layers/pred.bin": () => ({ raw: served.slice().buffer, headers: { "content-type": "application/octet-stream", "X-SPARC-Dtype": "float32", "X-SPARC-Length": "3" } }),
    });
    try {
      const load = runLayerLoader("r_live", "e1");
      const meta = L({ key: "pred", stats: stats(1, 3) });
      const [a, b] = await Promise.all([load(meta), load(meta)]);
      expect(a).toBe(b);
      expect([...a]).toEqual([1, 2, 3]);
      expect(await load(meta)).toBe(a); // cached
      expect(m.calls).toHaveLength(1);
      // The run wrote new predictions: the refreshed catalogue carries new stats.
      served = Float32Array.from([4, 5, 6]);
      const fresh = await load(L({ key: "pred", stats: stats(4, 6) }));
      expect([...fresh]).toEqual([4, 5, 6]);
      expect(m.calls).toHaveLength(2);
    } finally {
      m.restore();
    }
  });
});
