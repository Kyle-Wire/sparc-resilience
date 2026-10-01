// Map explorer (SPEC §6.5): layer binaries are fetched once and cached, swipe compare draws
// two layers, a legend-histogram brush becomes the run's selection (mask in the per-run
// store, blob spec in the URL), and Region stats posts the current selection.
import { afterEach, describe, expect, it } from "vitest";
import { encodeBitset } from "../../../api/binary";
import { clearResources } from "../../../api/resource";
import type { SelectionSpec } from "../../../api/types";
import { navigate } from "../../../router";
import { decodeSelection, encodeSelection, useSelectionStore } from "../../../stores/selection";
import { byText, click, flush, key, mockFetch, render, typeInto, waitFor, type MockHandler } from "../../../test/render";
import MapTab from "../MapTab";
import { mapRoutes } from "./helpers";

let seq = 0;

/** Render the Map tab of a fresh run; `{rid}` in extra route keys is replaced by its id. */
async function renderMap(query = "", extra: Record<string, MockHandler> = {}) {
  const rid = `r_map${++seq}`;
  const routes = Object.fromEntries(Object.entries(extra).map(([k, v]) => [k.replace("{rid}", rid), v]));
  const m = mockFetch({ ...mapRoutes(rid), ...routes });
  navigate(`/r/${rid}/map${query}`, { replace: true });
  const r = render(<MapTab />);
  await waitFor(() => r.container.querySelector('[role="application"]'), 5000, "map canvas");
  await flush(4);
  return { ...r, rid, m };
}

const binCalls = (calls: { method: string; url: string }[], key: string) =>
  calls.filter((c) => c.method === "GET" && c.url.endsWith(`/layers/${encodeURIComponent(key)}.bin`)).length;

function layerSelect(root: HTMLElement): HTMLSelectElement {
  const label = [...root.querySelectorAll("label.cap[for]")].find((l) => l.textContent === "Layer" || l.textContent === "Layer A")!;
  return root.querySelector(`select#${CSS.escape(label.getAttribute("for")!)}`) as HTMLSelectElement;
}

describe("Map explorer", () => {
  afterEach(() => {
    clearResources();
    useSelectionStore.setState({ byRun: {} });
  });

  it("requests each layer's .bin once and serves it from the cache afterwards", async () => {
    const { container, m } = await renderMap();
    expect(binCalls(m.calls, "obs")).toBe(1); // the first layer of the catalogue
    typeInto(layerSelect(container), "resid");
    await flush(4);
    expect(new URLSearchParams(window.location.search).get("layer")).toBe("resid");
    expect(binCalls(m.calls, "resid")).toBe(1);
    expect(container.querySelector(".map-side h3")!.textContent).toBe("Residual");
    typeInto(layerSelect(container), "obs");
    await flush(4);
    expect(container.querySelector(".map-side h3")!.textContent).toBe("Observed");
    typeInto(layerSelect(container), "resid");
    await flush(4);
    expect(binCalls(m.calls, "obs")).toBe(1);
    expect(binCalls(m.calls, "resid")).toBe(1);
    m.restore();
  });

  it("swipe compare renders two layers with a divider", async () => {
    const { container, m } = await renderMap("?layer=obs&cmp=resid&mode=swipe");
    expect(binCalls(m.calls, "obs")).toBe(1);
    expect(binCalls(m.calls, "resid")).toBe(1);
    const canvases = container.querySelectorAll('[role="application"]');
    expect(canvases.length).toBe(2);
    expect(canvases[0].getAttribute("aria-label")).toContain("Observed on the left, Residual on the right");
    expect(container.querySelector('[role="slider"]')!.getAttribute("aria-valuenow")).toBe("50");
    m.restore();
  });

  it("a legend-histogram brush updates the per-run selection store and the URL", async () => {
    const { container, rid, m } = await renderMap("?layer=obs", { "PUT /api/runs/{rid}/blobs": { status: 201, body: { blob_id: "bl_brush", bytes: 1 } } });
    const bars = container.querySelectorAll('.map-side svg.chart rect[role="img"]');
    expect(bars.length).toBeGreaterThan(0);
    key(bars[bars.length - 1], "Enter"); // the hottest bin
    await flush(4);
    await waitFor(() => useSelectionStore.getState().byRun[rid], 3000, "selection stored");
    const put = m.calls.find((c) => c.method === "PUT" && c.url.startsWith(`/api/runs/${rid}/blobs`))!;
    expect(put.url).toContain("kind=mask");
    const entry = useSelectionStore.getState().byRun[rid];
    expect([...entry.mask!]).toEqual([0, 0, 0, 0, 0, 0, 1]); // the 86 °F cell
    expect(entry.n_cells).toBe(1);
    expect(entry.source).toBe("legend");
    const spec = decodeSelection(new URLSearchParams(window.location.search).get("sel"));
    expect(spec).toEqual({ kind: "blob", blob_id: "bl_brush" });
    expect(container.querySelector('[data-testid="selection-status"]')!.textContent).toContain("1 cell");
    m.restore();
  });

  it("Region stats posts the current selection", async () => {
    const spec: SelectionSpec = { kind: "zones", values: [1] };
    const posted: unknown[] = [];
    const extra: Record<string, MockHandler> = {
      "POST /api/runs/{rid}/selection/resolve": {
        body: { n_cells: 3, area_km2: 0.0027, people: null, medians: {}, mask: encodeBitset([1, 1, 1, 0, 0, 0, 0]), portable: true, warnings: [] },
      },
      "POST /api/runs/{rid}/stats/region": (_u, init) => {
        posted.push(JSON.parse(String(init.body)));
        return {
          body: {
            n_cells: 3,
            area_km2: 0.0027,
            people: null,
            layers: { obs: { mean: 81, sd: 0.8, p10: 80.2, p50: 81, p90: 81.8, mean_outside: 84.5 } },
            scenarios: {
              "configured:canopy-increase-plus-10": {
                inside: { estimate: -0.2, se: 0.05, lo: -0.3, hi: -0.1, confidence: "confident_cools", phrase: "" },
                outside: { estimate: -0.55, se: 0.1, lo: -0.75, hi: -0.35, confidence: "confident_cools", phrase: "" },
                has_folds: true,
              },
            },
          },
        };
      },
    };
    const { container, rid, m } = await renderMap(`?layer=obs&tool=region&sel=${encodeSelection(spec)}`, extra);
    await waitFor(() => posted.length > 0 && container.querySelector('[data-tool="region"] table'), 3000, "region stats");
    expect(posted[0]).toMatchObject({ selection: spec, layers: ["obs"], scenarios: ["configured:canopy-increase-plus-10"] });
    // the URL spec was resolved into a mask for the map tint
    expect([...useSelectionStore.getState().byRun[rid].mask!]).toEqual([1, 1, 1, 0, 0, 0, 0]);
    const text = container.querySelector('[data-tool="region"]')!.textContent!;
    expect(text).toContain("Mean inside");
    expect(text).toContain("0.20 °F cooler");
    // a new selection is posted again
    click(byText(container, "button", "Clear"));
    await flush(3);
    expect(container.querySelector('[data-tool="region"]')!.textContent).toContain("Select cells first");
    m.restore();
  });

  it("pins a cell into the inspector with configured-scenario deltas and a rebuilt curve", async () => {
    const cell = {
      index: 0,
      id: 1001,
      lon: -71.5,
      lat: 41.9,
      zone: 1,
      values: { obs: 80, resid: -1 },
      curves: { canopy: { model: "saturating", A: 2, ds: 10, inflection: null, d90: 23, dmax: 40, dose: [], benefit: [] } },
      scenarios: { "configured:canopy-increase-plus-10": -0.1 },
    };
    const { container, rid, m } = await renderMap("?layer=obs", { "GET /api/runs/{rid}/cells/0": { body: cell } });
    const stage = container.querySelector('[role="application"]')!;
    key(stage, "Enter"); // cursor
    key(stage, "ArrowLeft");
    key(stage, "ArrowUp"); // north-west cell: row 0
    key(stage, "Enter"); // pin
    await waitFor(() => container.querySelector('[aria-label="Pinned cell"] h3')?.textContent === "Cell 1001", 3000, "inspector");
    const req = m.calls.find((c) => c.url.startsWith(`/api/runs/${rid}/cells/0`))!;
    expect(decodeURIComponent(req.url)).toContain("scenarios=configured:canopy-increase-plus-10");
    const insp = container.querySelector('[aria-label="Pinned cell"]')!;
    expect(insp.textContent).toContain("Canopy Increase +10");
    expect(insp.textContent).toContain("0.10 °F cooler");
    expect(insp.querySelector('svg.chart[aria-label="canopy response here"]')).not.toBeNull(); // rebuilt from A and d_s
    m.restore();
  });

  it("hex mode paints hexagon means and the drawer opens on its tabs", async () => {
    const { container, m } = await renderMap("?layer=obs&hex=250&tool=breakdown");
    expect(container.textContent).toContain("showing 250 m hexagon means");
    expect(container.querySelector('[role="tablist"][aria-label="Analysis tool"]')).not.toBeNull();
    expect(container.querySelector('[data-tool="breakdown"]')).not.toBeNull();
    m.restore();
  });
});
