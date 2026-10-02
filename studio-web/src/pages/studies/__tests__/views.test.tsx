// Result views read from the server: the emulator's trust numbers come from the manifest section
// (the catalog output serves emulator.npz), Truth vs recovered keeps its interval ordered for a
// negative (cooling) truth, a running study's view refreshes on a timer, and the simcheck grid
// keeps its gate-failure hatching and labels the ramp's real ends.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { clearResources } from "../../../api/resource";
import { flush, mockFetch, render, waitFor } from "../../../test/render";
import { EmulatorResult, fetchEmulatorSummary, shareInterval } from "../components/RunViews";
import { SimcheckGrid } from "../components/SimcheckGrid";
import { StudyViewPanel, VIEW_REFRESH_MS } from "../components/StudyViews";
import { RID, simcheckView } from "../__fixtures__/api";

afterEach(() => clearResources());

describe("emulator summary", () => {
  const lever = { patch_pass_rate: 0.75, patch_mean_rel_err_median: 0.08, uniform_rel_err: 0.02, physics: true };

  it("reads the manifest's emulator section (not the npz-backed catalog output)", async () => {
    const m = mockFetch({ [`GET /api/runs/${RID}/manifest`]: { body: { emulator: { files: ["emulator.npz", "emulator.json"], levers: { canopy: lever } }, _sections: {} } } });
    expect(await fetchEmulatorSummary(RID)).toEqual({ files: ["emulator.npz", "emulator.json"], levers: { canopy: lever } });
    expect(m.calls.map((c) => c.url)).toEqual([`/api/runs/${RID}/manifest`]);
    m.restore();
  });

  it("falls back to emulator.json's validation blocks when the manifest has no section", async () => {
    const m = mockFetch({
      [`GET /api/runs/${RID}/manifest`]: { body: { _sections: {} } },
      [`GET /api/runs/${RID}/files/raw`]: {
        body: { levers: { canopy: { physics: false, channels: [1, 2], validation: { patch_pass_rate: 0.5, patch_mean_rel_err_median: 0.1, uniform: { rel_err: 0.04 } } } } },
      },
    });
    const s = await fetchEmulatorSummary(RID);
    expect(s.levers!.canopy).toMatchObject({ patch_pass_rate: 0.5, patch_mean_rel_err_median: 0.1, uniform_rel_err: 0.04, physics: false });
    expect(m.calls[1].url).toBe(`/api/runs/${RID}/files/raw?path=emulator.json`);
    m.restore();
  });

  it("charts the share of passing patches per lever", async () => {
    const m = mockFetch({ [`GET /api/runs/${RID}/manifest`]: { body: { emulator: { levers: { canopy: lever, albedo: { ...lever, patch_pass_rate: 1 } } } } } });
    const { container } = render(<EmulatorResult rid={RID} />);
    await waitFor(() => container.textContent!.includes("Emulator vs exact engine"), 3000, "chart");
    expect(container.textContent).toContain("canopy: median patch error 8%");
    expect(container.textContent).not.toContain("no levers");
    m.restore();
  });
});

describe("truth interval", () => {
  it("orders the ends for a negative truth", () => {
    // truth −0.5 °F, recovered −0.3 ± 0.05: share 0.6, interval (−0.3 ∓ 0.098) / −0.5
    const [lo, hi] = shareInterval({ truth: -0.5, recovered: -0.3, se: 0.05 })!;
    expect(lo).toBeCloseTo(0.404);
    expect(hi).toBeCloseTo(0.796);
    expect(lo).toBeLessThan(hi);
    const [lo2, hi2] = shareInterval({ truth: 200, recovered: 180, se: 10 })!;
    expect(lo2).toBeLessThan(hi2);
    expect(shareInterval({ truth: 0, recovered: 1, se: 1 })).toBeNull();
    expect(shareInterval({ truth: 1, recovered: null, se: 1 })).toBeNull();
  });
});

describe("live study views", () => {
  beforeEach(() => vi.useFakeTimers({ toFake: ["setInterval", "clearInterval"] }));
  afterEach(() => vi.useRealTimers());

  it("refetch while the study runs, and not once it has finished", async () => {
    const m = mockFetch({ "GET /api/studies/st_s/view": { body: simcheckView() } });
    const views = () => m.calls.filter((c) => c.url === "/api/studies/st_s/view").length;
    const r = render(<StudyViewPanel kind="simcheck" studyId="st_s" units="°F" live />);
    await waitFor(() => r.container.querySelector("table.sg-grid"), 3000, "grid");
    expect(views()).toBe(1);
    vi.advanceTimersByTime(VIEW_REFRESH_MS);
    await flush(3);
    expect(views()).toBe(2);
    r.rerender(<StudyViewPanel kind="simcheck" studyId="st_s" units="°F" live={false} />);
    vi.advanceTimersByTime(3 * VIEW_REFRESH_MS);
    await flush(3);
    expect(views()).toBe(2);
    m.restore();
  });
});

describe("simcheck grid styling", () => {
  it("sets only the background colour inline, so gate failures keep the stylesheet's hatching", () => {
    const v = simcheckView();
    v.grid = [...v.grid!, { generator: "additive", seed: 1, status: "gate_fail", share: 0.7, ci_covers: true, causal_covers: true, seconds: 30, gate_attempt: 2 }];
    const { container } = render(<SimcheckGrid view={v} />);
    const td = container.querySelector<HTMLTableCellElement>('td[data-status="gate_fail"]')!;
    expect(td.style.backgroundColor).not.toBe("");
    expect(td.style.backgroundImage).toBe("");
    expect(td.getAttribute("style")).not.toMatch(/background:/);
  });

  it("labels the ramp's low end even below zero", () => {
    const v = simcheckView();
    v.grid = [{ generator: "physics", seed: 0, status: "done", share: -0.5, ci_covers: false, causal_covers: false, seconds: 30, gate_attempt: 0 }];
    v.design = { physics: 1 };
    const { container } = render(<SimcheckGrid view={v} />);
    // span 1.5 around 1: the ramp runs from −0.5 to 2.5
    const legend = container.querySelector('[aria-label="Grid legend"]')!.textContent!;
    expect(legend).toMatch(/share [−-]0\.5/);
    expect(legend).toContain("2.5 (1 = exact)");
  });
});
