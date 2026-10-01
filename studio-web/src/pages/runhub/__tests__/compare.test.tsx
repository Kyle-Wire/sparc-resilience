// Compare runs (SPEC §6.8): provenance chips, config tree, metrics; the difference map and
// priority agreement are disabled with a reason when the grids differ (`same.grid` false, or
// the server's 409 grid_mismatch) and drawn when they match.
import { afterEach, describe, expect, it } from "vitest";
import { clearResources } from "../../../api/resource";
import type { CompareRuns } from "../../../api/runs";
import { navigate } from "../../../router";
import { flush, mockFetch, render, waitFor, type MockHandler } from "../../../test/render";
import Compare from "../Compare";
import { binResponse, mapRoutes, runSummary, unframedCharts } from "./helpers";

function comparison(a: string, b: string, grid: boolean): CompareRuns {
  return {
    a: runSummary(a, { label: "fast", mode: "fast" }),
    b: runSummary(b, { label: "full", mode: "full" }),
    same: { data: true, config: false, code: true, grid },
    config_diff: [
      { path: "stacker.tune_lambda", a: [0, 0.1], b: [0, 0.1, 1] },
      { path: "stacker.epochs", a: 200, b: 400 },
      { path: "cv.distance_curve.enabled", a: false, b: true },
    ],
    metrics: [
      { key: "r2", a: 0.81, b: 0.84, delta: 0.03 },
      { key: "rmse", a: 0.34, b: 0.31, delta: -0.03 },
    ],
    timings: [
      { stage: "S2_S3", a: 15.4, b: 40.2 },
      { stage: "S4", a: 5.5, b: 12.1 },
    ],
    scenarios: [
      {
        name: "Canopy Increase +10",
        a: { estimate: -1.03, se: 0.28, lo: -1.58, hi: -0.48, confidence: "confident_cools", phrase: "" },
        b: { estimate: -0.95, se: 0.2, lo: -1.34, hi: -0.56, confidence: "confident_cools", phrase: "" },
      },
    ],
    climate: { a: { warming_mid: 3.2 }, b: { warming_mid: 3.2 } },
    causal: null,
    environment: { added: ["torch==2.14.0"], removed: [], changed: [{ name: "numpy", a: "1.26.4", b: "2.1.0" }] },
    outputs: { a_only: [], b_only: ["cv_distance"] },
  };
}

let seq = 0;

async function renderCompare(grid: boolean, extra: (a: string, b: string) => Record<string, MockHandler> = () => ({})) {
  const a = `r_cmpa${++seq}`;
  const b = `r_cmpb${seq}`;
  const m = mockFetch({
    ...mapRoutes(a),
    ...mapRoutes(b),
    "GET /api/projects/p_1/runs": { body: { items: [runSummary(a), runSummary(b)], next_cursor: null } },
    "GET /api/compare/runs": { body: comparison(a, b, grid) },
    ...extra(a, b),
  });
  navigate(`/p/p_1/compare?a=${a}&b=${b}`, { replace: true });
  const r = render(<Compare />);
  await waitFor(() => r.container.querySelector('[aria-label="Provenance"]'), 5000, "comparison");
  await flush(6);
  return { ...r, a, b, m };
}

describe("Compare runs", () => {
  afterEach(() => clearResources());

  it("shows provenance chips, the config diff tree, metrics and the environment diff", async () => {
    const { container, m } = await renderCompare(true, () => ({
      "GET /api/compare/layer.bin": () => binResponse(Float32Array.from([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])),
      "POST /api/compare/priority": { body: { kendall_tau: 0.72, top_decile_jaccard: 0.61, n: 7 } },
    }));
    const chips = container.querySelector('[aria-label="Provenance"]')!.textContent!;
    expect(chips).toContain("same data");
    expect(chips).toContain("different config");
    expect(chips).toContain("same grid");
    const tree = container.querySelector('[aria-label="Config diff tree"]')!;
    expect([...tree.querySelectorAll("summary strong")].map((s) => s.textContent)).toEqual(["cv", "stacker"]);
    expect(container.textContent).toContain("numpy");
    expect(container.textContent).toContain("Only in B: cv_distance");
    m.restore();
  });

  it("disables the difference map with a reason when the grids differ", async () => {
    const { container, m } = await renderCompare(false);
    const note = container.querySelector('[data-disabled="difference-map"]');
    expect(note).not.toBeNull();
    expect(note!.textContent).toContain("different grids");
    expect(container.querySelector('[data-disabled="priority"]')).not.toBeNull();
    expect(m.calls.some((c) => c.url.startsWith("/api/compare/layer.bin"))).toBe(false);
    expect(m.calls.some((c) => c.url.startsWith("/api/compare/priority"))).toBe(false);
    m.restore();
  });

  it("disables it with the server's reason on 409 grid_mismatch", async () => {
    const mismatch = { status: 409, body: { error: { code: "grid_mismatch", message: "Run B has a 60 m grid; run A has 30 m" } } };
    const { container, m } = await renderCompare(true, () => ({ "GET /api/compare/layer.bin": mismatch, "POST /api/compare/priority": mismatch }));
    await waitFor(() => container.querySelector('[data-disabled="difference-map"]'), 3000, "mismatch note");
    const note = container.querySelector('[data-disabled="difference-map"]')!;
    expect(note.textContent).toContain("Run B has a 60 m grid");
    expect(note.textContent).toContain("grid_mismatch");
    expect(container.querySelector('[data-disabled="priority"]')!.textContent).toContain("grid_mismatch");
    expect(container.querySelectorAll('[role="application"]').length).toBe(0);
    m.restore();
  });

  it("draws the difference map and priority agreement when the grids match", async () => {
    const { container, m } = await renderCompare(true, () => ({
      "GET /api/compare/layer.bin": () => binResponse(Float32Array.from([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])),
      "POST /api/compare/priority": { body: { kendall_tau: 0.72, top_decile_jaccard: 0.61, n: 7 } },
    }));
    await waitFor(() => container.querySelector('[role="application"]'), 3000, "difference map");
    expect(container.querySelector('[role="application"]')!.getAttribute("aria-label")).toContain("B − A");
    const req = m.calls.find((c) => c.url.startsWith("/api/compare/layer.bin"))!;
    expect(req.url).toContain("key=");
    expect(container.textContent).toContain("Kendall τ");
    expect(unframedCharts(container)).toEqual([]);
    expect(container.querySelectorAll("figure.chart-frame").length).toBeGreaterThanOrEqual(2); // timings, scenario effects
    m.restore();
  });
});
