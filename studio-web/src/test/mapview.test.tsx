import { describe, expect, it } from "vitest";
import { act } from "react";
import type { LayerGroup, LayerMeta } from "../api/types";
import { MapView, type MapSelectionEvent } from "../map/MapView";
import { Legend, divergingEnds } from "../map/Legend";
import { computeDomain } from "../map/domain";
import { grid3 } from "./grid";
import { byText, click, flush, render, typeInto } from "./render";

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
