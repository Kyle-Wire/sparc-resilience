// Simcheck grid colouring (SPEC §14.3 "studies: simcheck grid colouring"): cells take the
// diverging ramp by effect share centred on 1, grey while pending, red on error, with a dot
// marker for gate redraws; run against the fixture view and rendered.
import { afterEach, describe, expect, it } from "vitest";
import { useUi } from "../../../stores/ui";
import { render } from "../../../test/render";
import { rampColor, TOKEN_VALUES } from "../../../theme/palette";
import { SimcheckGrid } from "../components/SimcheckGrid";
import { cellKey, inkOn, sharesByGenerator, shareT, simCellLook, simGrid, simProgress } from "../model/simcheck";
import { simcheckView } from "../__fixtures__/api";

const LIGHT = TOKEN_VALUES.light;

afterEach(() => useUi.setState({ dark: false }));

describe("simGrid", () => {
  it("lays out the design's generators (count > 0) by seed; missing design cells are pending", () => {
    const g = simGrid(simcheckView());
    expect(g.generators).toEqual(["physics", "additive", "null"]);
    expect(g.seeds).toEqual([0, 1, 2]);
    expect(g.cells.get(cellKey("additive", 1))!.status).toBe("pending");
    expect(g.cells.has(cellKey("additive", 2))).toBe(false); // additive planned 2 replicates
    expect(g.cells.has(cellKey("own_only", 0))).toBe(false); // zero replicates: not a row
    // the span is the largest distance from 1 (here 0.6), at least 1
    expect(g.span).toBe(1);
    expect(simGrid({ design: { physics: 1 }, grid: [{ generator: "physics", seed: 0, status: "done", share: 3.5, ci_covers: null, causal_covers: null, seconds: 1, gate_attempt: 0 }] }).span).toBeCloseTo(2.5);
  });

  it("an empty or missing view has no rows", () => {
    expect(simGrid(null).generators).toEqual([]);
    expect(simGrid({}).cells.size).toBe(0);
  });

  it("counts progress and uses the server's ETA when given", () => {
    const p = simProgress(simGrid(simcheckView()), 1, 120);
    expect(p).toEqual({ done: 4, total: 7, errors: 1, running: 1, eta_s: 120 });
    // without the server's ETA: mean seconds of timed cells × remaining ÷ workers
    const q = simProgress(simGrid(simcheckView()), 2, null);
    const secs = [40, 44, 38, 35, 3];
    expect(q.eta_s).toBeCloseTo(((secs.reduce((a, b) => a + b) / secs.length) * 2) / 2);
  });

  it("collects done shares per generator for the strip plot", () => {
    expect(sharesByGenerator(simGrid(simcheckView()))).toEqual([
      { generator: "physics", shares: [1.0, 1.6] },
      { generator: "additive", shares: [0.5] },
      { generator: "null", shares: [] },
    ]);
  });
});

describe("simCellLook", () => {
  const g = simGrid(simcheckView());
  const look = (gen: string, seed: number, dark = false) => simCellLook(g.cells.get(cellKey(gen, seed)), gen, seed, g.span, dark);

  it("places shares on the diverging ramp centred on 1", () => {
    expect(shareT(1, 1)).toBe(0.5);
    expect(shareT(0, 1)).toBe(0);
    expect(shareT(2, 1)).toBe(1);
    expect(shareT(5, 1)).toBe(1);
    expect(look("physics", 0).fill).toBe(rampColor("div", 0.5, false)); // share 1 = the neutral midpoint
    expect(look("physics", 1).fill).toBe(rampColor("div", 0.8, false)); // 1.6: warm side
    expect(look("additive", 0).fill).toBe(rampColor("div", 0.25, false)); // 0.5: cool side
    expect(look("physics", 1).fill).not.toBe(look("additive", 0).fill);
    expect(look("physics", 0, true).fill).toBe(rampColor("div", 0.5, true));
  });

  it("is grey while pending or running, red on error, neutral when finished without a share", () => {
    expect(look("additive", 1).fill).toBe(LIGHT["--gray-mark"]);
    expect(look("physics", 2).fill).toBe(LIGHT["--gray-mark"]);
    expect(look("physics", 2).marker).toBe("…");
    expect(look("null", 1).fill).toBe(LIGHT["--critical"]);
    expect(look("null", 1).marker).toBe("!");
    expect(look("null", 0).fill).toBe(LIGHT["--nodata"]);
    expect(look("additive", 1, true).fill).toBe(TOKEN_VALUES.dark["--gray-mark"]);
  });

  it("marks a gate redraw with a dot and says so in the label (never colour alone)", () => {
    const l = look("physics", 1);
    expect(l.redraw).toBe(true);
    expect(l.marker).toBe("•");
    expect(l.label).toContain("gate redraw");
    expect(l.label).toContain("share 1.60");
    expect(look("physics", 0).redraw).toBe(false);
    expect(look("physics", 0).marker).toBe("");
    expect(look("null", 1).label).toContain("error");
  });

  it("picks a readable marker colour", () => {
    expect(inkOn("#ffffff")).toBe("#111111");
    expect(inkOn("#000000")).toBe("#ffffff");
    expect(inkOn("#d03b3b")).toBe("#ffffff");
    expect(inkOn("not a colour")).toBe("#111111");
  });
});

describe("SimcheckGrid", () => {
  it("renders one coloured cell per replicate with status, fill and markers", () => {
    const { container } = render(<SimcheckGrid view={simcheckView()} />);
    const rows = [...container.querySelectorAll("tbody tr")];
    expect(rows.map((r) => r.querySelector("th")!.textContent)).toEqual(["physics", "additive", "null"]);
    const cell = (gen: number, seed: number) => rows[gen].querySelectorAll("td")[seed] as HTMLTableCellElement;
    expect(cell(0, 0).getAttribute("data-fill")).toBe(rampColor("div", 0.5, false));
    expect(cell(0, 1).getAttribute("data-redraw")).toBe("true");
    expect(cell(0, 1).textContent).toBe("•");
    expect(cell(0, 2).getAttribute("data-status")).toBe("running");
    expect(cell(1, 1).getAttribute("data-status")).toBe("pending");
    expect(cell(1, 1).getAttribute("data-fill")).toBe(LIGHT["--gray-mark"]);
    expect(cell(2, 1).getAttribute("data-fill")).toBe(LIGHT["--critical"]);
    expect(cell(2, 1).textContent).toBe("!");
    // a seed beyond the generator's design has no cell
    expect(cell(1, 2).getAttribute("data-status")).toBe("none");
    // every cell has a text label
    for (const td of container.querySelectorAll("td[data-status]:not([data-status=none])")) expect(td.getAttribute("aria-label")).toMatch(/seed \d/);
    expect(container.textContent).toContain("Replicates 4 of 7");
    expect(container.textContent).toContain("Errors 1");
  });

  it("follows the dark theme", () => {
    useUi.setState({ dark: true });
    const { container } = render(<SimcheckGrid view={simcheckView()} />);
    expect(container.querySelector("td")!.getAttribute("data-fill")).toBe(rampColor("div", 0.5, true));
  });
});
