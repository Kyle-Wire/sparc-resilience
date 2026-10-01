import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createRef, type ComponentProps, type ComponentType } from "react";
import { KIT, LineBand, type ChartHandle, type KitName } from "../charts";
import { lastDownload } from "../components/ui/download";
import { clearResources } from "../api/resource";
import { useUi } from "../stores/ui";
import { click, flush, mockFetch, render } from "./render";

type Props<K extends KitName> = Omit<ComponentProps<(typeof KIT)[K]>, "handleRef">;

/** One representative sample per kit chart; the type forces a sample for every chart. */
const SAMPLES: { [K in KitName]: Props<K> } = {
  LineBand: {
    title: "Skill vs block size",
    series: [
      { id: "stack", label: "Stacked model", x: [250, 500, 1000, 2000], y: [0.82, 0.78, 0.71, 0.66], lo: [0.8, 0.75, 0.68, 0.6], hi: [0.85, 0.8, 0.74, 0.7], emphasis: true },
      { id: "gam", label: "GAM", x: [250, 500, 1000, 2000], y: [0.7, 0.66, 0.6, 0.55], muted: true },
    ],
    xLabel: "Block size", xUnit: "m", yLabel: "Held-out R²", xLog: true, refLines: [{ axis: "x", value: 2000, label: "main design" }],
  },
  Bars: { title: "Exposure", categories: ["today", "SSP2-4.5 2041–60"], series: [{ id: "a", label: "No adaptation", values: [0.21, 0.48], lo: [0.2, 0.4], hi: [0.22, 0.55] }, { id: "b", label: "Package", values: [0.15, 0.39] }], valueLabel: "Share of cells ≥ 90 °F", unit: "%" , orientation: "v" },
  DotRange: { title: "Configured scenarios", rows: [{ id: "a", label: "Canopy +10", est: -0.41, lo: -0.52, hi: -0.3, lo2: -0.9, hi2: -0.05, check: { est: -0.35, lo: -0.5, hi: -0.2 } }, { id: "b", label: "Albedo +0.1", est: -0.12, lo: -0.3, hi: 0.06, hollow: true }], valueLabel: "Mean ΔT", unit: "°F", signed: true },
  Forest: { title: "Against baselines", rows: [{ id: "k", label: "Kriging", est: -0.12, se: 0.03 }, { id: "rf", label: "Random forest", est: 0.02, se: 0.02 }], valueLabel: "ΔMSE (stack − baseline)", unit: "°F²", z: 2, better: { side: "negative", label: "stack better" } },
  Heatmap: { title: "Fold × model RMSE", rows: ["fold 1", "fold 2"], cols: ["mgwr", "gam", "gwrf"], values: [[0.61, 0.66, null], [0.58, 0.7, 0.64]], unit: "°F", valueLabel: "Held-out RMSE", labels: true },
  Histogram: { title: "Residuals", values: Float32Array.from({ length: 500 }, (_, i) => Math.sin(i) * 2), xLabel: "Residual", unit: "°F", nBins: 20, onBrush: () => {} },
  HexbinScatter: { title: "Observed vs predicted", points: { x: Float32Array.from({ length: 300 }, (_, i) => 85 + (i % 30) * 0.3), y: Float32Array.from({ length: 300 }, (_, i) => 85 + (i % 30) * 0.28 + (i % 7) * 0.1) }, xLabel: "Predicted", yLabel: "Observed", xUnit: "°F", yUnit: "°F", diagonal: true, nBins: 20 },
  BoxStrip: { title: "Residuals by zone", groups: [{ label: "Zone 1", q: [-1.2, -0.4, 0, 0.5, 1.1], n: 900, mean: 0.02 }, { label: "Zone 2", q: [-0.9, -0.3, 0.1, 0.4, 1.4], n: 650 }], valueLabel: "Residual", unit: "°F", zeroLine: true },
  Gantt: {
    title: "Span tree",
    rows: [
      { id: "run", parentId: null, label: "run", start: 0, end: 120, status: "ok" },
      { id: "s1", parentId: "run", label: "S2_S3", start: 2, end: 100, status: "ok" },
      { id: "t1", parentId: "s1", label: "fold 1/5", start: 2, end: 40, status: "ok" },
      { id: "t2", parentId: "s1", label: "fold 2/5", start: 40, end: null, status: "running" },
    ],
    t0: 0, now: 120, markers: [{ t: 30, kind: "warning", label: "physics.antiphysical_a" }, { t: 100, kind: "checkpoint", label: "saved S3" }], bands: [{ from: 60, to: 90 }],
  },
  Sparkline: { title: "RSS", values: [1.2, 1.5, 2.1, 2.0, 2.6], unit: "GB", area: true },
  Pareto: { title: "Cooling vs budget", points: [{ x: 1000, y: 120 }, { x: 5000, y: 480 }, { x: 20000, y: 1704 }], realised: [{ x: 20000, y: 1037, label: "closed loop" }], selected: 2, xLabel: "Budget", xUnit: "pp·cells", yLabel: "Total cooling", yUnit: "°F·cells" },
  Rose: { title: "Anisotropy", petals: [{ angle_deg: 0, value: 1.2, reliable: true }, { angle_deg: 45, value: 0.8 }, { angle_deg: 90, value: 0.6, reliable: false }, { angle_deg: 135, value: 0.9 }], unit: "km" },
  Gauge: { title: "Warming offset", value: 0.37, label: "of median warming offset", format: (v) => `${Math.round(v * 100)}%`, ticks: [1] },
  IntervalStack: { title: "Layered uncertainty", rows: [{ id: "a", label: "Package", estimate: -0.4, layers: [{ id: "est", label: "estimation", lo: -0.5, hi: -0.3 }, { id: "spec", label: "specification", lo: -0.7, hi: -0.1 }, { id: "env", label: "envelope", lo: -0.9, hi: 0.05 }] }], valueLabel: "Mean ΔT", unit: "°F" },
  RingProfile: { title: "Spill", rings: [{ r0_m: 0, r1_m: 0, mean: -0.41, se: 0.05, n: 1284 }, { r0_m: 0, r1_m: 100, mean: -0.12, se: 0.02, n: 900 }, { r0_m: 100, r1_m: 250, mean: -0.04, se: 0.01, n: 2100 }], unit: "°F" },
  SmallMultiples: {
    title: "Dose–response by lever",
    items: [
      { id: "canopy", x: [0, 10, 20], y: [0, -0.3, -0.5] },
      { id: "albedo", x: [0, 0.05, 0.1], y: [0, -0.1, -0.18] },
    ],
    panelTitle: (it: { id: string }) => it.id,
    renderPanel: (it: { id: string; x: number[]; y: number[] }) => <LineBand bare title={it.id} series={[{ id: it.id, label: it.id, x: it.x, y: it.y }]} xLabel="Dose" yLabel="ΔT" yUnit="°F" />,
    table: { columns: [{ key: "lever", label: "Lever" }, { key: "dose", label: "Dose" }, { key: "dt", label: "ΔT", unit: "°F" }], rows: [["canopy", 10, -0.3], ["albedo", 0.1, -0.18]] },
  } as Props<"SmallMultiples">,
};

beforeEach(() => {
  useUi.setState({ context: { projectId: "p_1", runId: "r1" } });
});
afterEach(() => {
  clearResources();
  vi.restoreAllMocks();
});

function renderChart(name: KitName) {
  const C = KIT[name] as ComponentType<Record<string, unknown>>;
  const ref = createRef<ChartHandle>();
  const r = render(<C {...(SAMPLES[name] as Record<string, unknown>)} handleRef={ref} />);
  return { ...r, ref };
}

describe("chart kit: every chart renders inside ChartFrame", () => {
  for (const name of Object.keys(KIT) as KitName[]) {
    describe(name, () => {
      it("has a 'Table view' <details> with the chart's data", () => {
        const { container } = renderChart(name);
        const details = container.querySelector("details.table-view");
        expect(details, `${name} table view`).not.toBeNull();
        expect(details!.querySelector("summary")!.textContent).toBe("Table view");
        const rows = details!.querySelectorAll("tbody tr");
        expect(rows.length).toBeGreaterThan(0);
        expect(container.querySelector("figure.chart-frame h3")!.textContent).toBe(SAMPLES[name].title);
        expect(container.querySelector("svg.chart")).not.toBeNull();
      });

      it("exposes SVG, PNG and CSV export handlers", async () => {
        const { container, ref } = renderChart(name);
        const h = ref.current!;
        expect(typeof h.svg).toBe("function");
        expect(typeof h.png).toBe("function");
        expect(typeof h.csv).toBe("function");
        expect(typeof h.exportPng).toBe("function");
        // toolbar buttons are present and named
        for (const label of ["SVG", "PNG", "CSV", "Pin"]) expect([...container.querySelectorAll(".chart-tools button")].some((b) => (b.getAttribute("aria-label") ?? "").includes(label === "Pin" ? "Pin" : `as ${label}`) || (label === "CSV" && (b.getAttribute("aria-label") ?? "").includes("CSV")))).toBe(true);
        // SVG: standalone document, CSS variables resolved, interactive attributes stripped
        const svg = h.svg();
        expect(svg.startsWith("<?xml")).toBe(true);
        expect(svg).toContain('xmlns="http://www.w3.org/2000/svg"');
        expect(svg).not.toContain("var(--");
        expect(svg).not.toContain("tabindex");
        h.exportSvg();
        expect(lastDownload()!.name).toMatch(/\.svg$/);
        expect(lastDownload()!.blob.type).toBe("image/svg+xml");
        // CSV: header with units, one line per table row
        const csv = h.csv();
        const lines = csv.trim().split("\n");
        expect(lines.length).toBe(h.table.rows.length + 1);
        expect(lines[0].split(",").length).toBe(h.table.columns.length);
        h.downloadCsv();
        expect(lastDownload()!.name).toMatch(/\.csv$/);
        // PNG: rasterised through a canvas (stubbed here: jsdom has no canvas)
        const drawn: unknown[] = [];
        vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(function () {
          return { drawImage: (...a: unknown[]) => drawn.push(a), fillRect: () => {}, set fillStyle(_v: string) {} } as unknown as CanvasRenderingContext2D;
        } as unknown as typeof HTMLCanvasElement.prototype.getContext);
        vi.spyOn(HTMLCanvasElement.prototype, "toDataURL").mockReturnValue("data:image/png;base64,iVBORw0KGgo=");
        if (name === "HexbinScatter") expect(h.svg()).toContain("<image"); // the canvas raster is embedded
        vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation(function (cb: BlobCallback) {
          cb(new Blob(["png"], { type: "image/png" }));
        });
        const OrigImage = globalThis.Image;
        globalThis.Image = class {
          onload: (() => void) | null = null;
          onerror: (() => void) | null = null;
          set src(_v: string) {
            setTimeout(() => this.onload?.(), 0);
          }
        } as unknown as typeof Image;
        URL.createObjectURL = URL.createObjectURL ?? (() => "blob:x");
        URL.revokeObjectURL = URL.revokeObjectURL ?? (() => {});
        try {
          await h.exportPng();
          await flush();
        } finally {
          globalThis.Image = OrigImage;
        }
        expect(drawn.length).toBe(1);
        expect(lastDownload()!.name).toMatch(/\.png$/);
        expect(lastDownload()!.blob.type).toBe("image/png");
      });
    });
  }

  it("covers every chart the kit exports", () => {
    expect(Object.keys(SAMPLES).sort()).toEqual(Object.keys(KIT).sort());
    expect(Object.keys(KIT)).toHaveLength(16);
  });

  it("marks are keyboard-focusable and announce their value", () => {
    const { container } = renderChart("DotRange");
    const mark = container.querySelector('svg.chart [tabindex="0"][aria-label]') as SVGElement;
    expect(mark.getAttribute("aria-label")).toContain("Canopy +10");
    mark.dispatchEvent(new FocusEvent("focus"));
    mark.dispatchEvent(new FocusEvent("focusin", { bubbles: true }));
  });

  it("Pin to Findings posts the snapshot and uploads the SVG image", async () => {
    const m = mockFetch({
      "POST /api/findings": { status: 201, body: { id: "fd_1", project_id: "p_1" } },
      "PUT /api/findings/fd_1/image": { body: { image_url: "/api/findings/fd_1/image" } },
    });
    const { container } = renderChart("Forest");
    click(container.querySelector('button[aria-label^="Pin"]'));
    await flush(5);
    m.restore();
    const post = m.calls.find((c) => c.method === "POST")!;
    expect(post.body).toMatchObject({ project_id: "p_1", run_id: "r1", title: "Against baselines", snapshot: { kind: "chart", table: { columns: expect.any(Array) } } });
    expect(m.calls.some((c) => c.method === "PUT" && c.url === "/api/findings/fd_1/image")).toBe(true);
  });
});
