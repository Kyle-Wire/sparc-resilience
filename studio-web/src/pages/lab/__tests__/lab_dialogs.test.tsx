// Review additions: nothing is deleted without asking (a scenario with later revisions only
// after a second confirmation), the "Around sites" template sends typed or uploaded [lon, lat]
// sites, and a selection builder tints the map only while it is being worked on.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { clearResources } from "../../../api/resource";
import type { ScenarioTemplate } from "../../../api/lab";
import type { Job, SelectionSpec } from "../../../api/types";
import { byText, click, flush, mockFetch, render, typeInto, waitFor } from "../../../test/render";
import { AcrossRunsPanel, bandNote, latestAcross } from "../components/AcrossRuns";
import { DeleteScenarioDialog } from "../components/Dialogs";
import { ResultInspector } from "../components/ResultInspector";
import { SelectionBuilder } from "../components/SelectionBuilder";
import { TemplateGallery } from "../components/TemplateGallery";
import { makeGrid } from "../../../map/grid";
import { L, LEVERS, PID, RID, gridMeta, result, scenario } from "../__fixtures__/lab";

let fetchMock: ReturnType<typeof mockFetch> | null = null;

beforeEach(() => clearResources());

afterEach(() => {
  fetchMock?.restore();
  fetchMock = null;
});

describe("Delete scenario dialog", () => {
  it("asks first, then deletes a revision with later revisions only on a second, forced confirmation", async () => {
    const calls: string[] = [];
    fetchMock = mockFetch({
      "DELETE /api/scenarios/sc_a": (u) => {
        calls.push(u.search);
        return u.searchParams.get("force") ? { body: { ok: true } } : { status: 409, body: { error: { code: "has_children", message: "has 2 later revisions", detail: { children: ["sc_b", "sc_c"] } } } };
      },
    });
    const onDeleted = vi.fn();
    const onClose = vi.fn();
    render(<DeleteScenarioDialog sid="sc_a" name="Corridor" onDeleted={onDeleted} onClose={onClose} />);
    const dialog = await waitFor(() => document.querySelector('[role="dialog"]'), 5000, "dialog");
    expect(dialog.textContent).toContain("Corridor");
    expect(calls).toEqual([]); // nothing happens until confirmed
    click(byText(dialog, "button", /^Delete$/));
    await waitFor(() => dialog.querySelector("[data-has-children]"), 5000, "children notice");
    expect(dialog.textContent).toContain("It has 2 later revisions");
    expect(onDeleted).not.toHaveBeenCalled();
    click(byText(dialog, "button", "Delete and re-link its revisions"));
    await waitFor(() => onDeleted.mock.calls.length > 0, 5000, "deleted");
    expect(calls).toEqual(["", "?force=true"]);
    expect(onClose).toHaveBeenCalled();
  });
});

const AROUND: ScenarioTemplate = {
  id: "around_sites",
  label: "Around sites",
  desc: "Edit a lever within a radius of uploaded points.",
  requires: ["crs"],
  params_schema: {
    type: "object",
    properties: {
      sites: { type: "array", title: "sites [lon, lat]" },
      lever: { type: "string", title: "lever" },
      radius_m: { type: "number", title: "radius (m)", default: 200, minimum: 0 },
      amount: { type: "number", title: "amount", default: 10 },
    },
    required: ["sites"],
  },
};

describe("Around sites template", () => {
  it("sends the typed sites as [lon, lat] pairs and the chosen lever", async () => {
    let body: Record<string, unknown> | null = null;
    fetchMock = mockFetch({
      "GET /api/scenario-templates": { body: [AROUND] },
      [`POST /api/projects/${PID}/scenarios/from-template`]: (_u, init) => {
        body = JSON.parse(String(init.body));
        return { status: 201, body: scenario("sc_t") };
      },
    });
    const onCreated = vi.fn();
    const { container } = render(<TemplateGallery pid={PID} rid={RID} levers={LEVERS} onCreated={onCreated} />);
    click(await waitFor(() => byText(container, "button", "Use template"), 5000, "card"));
    const create = byText(container, "button", "Create scenario")!;
    expect((create as HTMLButtonElement).disabled).toBe(true); // sites are required
    expect(container.textContent).toContain("Needed: sites [lon, lat]");
    typeInto(container.querySelector('textarea[aria-label="sites [lon, lat]"]'), "-71.41, 41.82\n-71.40, 41.83");
    const lever = [...container.querySelectorAll("select")].find((s) => s.closest("label")?.textContent?.startsWith("lever"))!;
    act(() => {
      lever.value = "Albedo";
      lever.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await flush(2);
    expect(container.textContent).toContain("2 sites");
    expect((create as HTMLButtonElement).disabled).toBe(false);
    click(create);
    await waitFor(() => onCreated.mock.calls.length > 0, 5000, "created");
    expect(body).toEqual({ template: "around_sites", params: { sites: [[-71.41, 41.82], [-71.4, 41.83]], lever: "Albedo", radius_m: 200, amount: 10 }, run_id: RID });
  });
});

describe("Selection builder tint", () => {
  it("tints the map with its mask only while it is focused", async () => {
    fetchMock = mockFetch({
      [`POST /api/runs/${RID}/selection/resolve`]: { body: { n_cells: 3, area_km2: 0.0027, people: 35, medians: {}, mask: btoa(String.fromCharCode(0b0000111)), portable: true, warnings: [] } },
    });
    const g = makeGrid(gridMeta(), Int32Array.from([0, 1, 0, 2, 0, 1, 2]), Int32Array.from([2, 2, 1, 1, 0, 0, 0]));
    const onMask = vi.fn();
    const { container } = render(
      <SelectionBuilder rid={RID} grid={g} value={{ kind: "zones", values: [1] }} onChange={() => {}} columns={[]} regions={[]} levers={LEVERS} onMask={onMask} idPrefix="t" />,
    );
    await waitFor(() => byText(container, '[data-testid="sel-count"]', "3 cells"), 5000, "count");
    expect(onMask).not.toHaveBeenCalled(); // another edit's builder must not repaint the tint
    act(() => {
      container.querySelector<HTMLButtonElement>(".sel-summary button")!.focus();
    });
    expect(onMask).toHaveBeenCalledTimes(1);
    expect(Array.from(onMask.mock.calls[0][0] as Uint8Array)).toEqual([1, 1, 1, 0, 0, 0, 0]);
  });
});

describe("Selection builder with a stored (server-shaped) selection", () => {
  // The server echoes unset optional fields as null: a saved "Top %" has `k: null, within: null`,
  // a top-k `frac: null`, a buffer in metres `lever_range: null`.
  const RESOLVE = { n_cells: 2, area_km2: 0.0018, people: 20, medians: {}, mask: btoa(String.fromCharCode(0b0000011)), portable: true, warnings: [] };
  const g = () => makeGrid(gridMeta(), Int32Array.from([0, 1, 0, 2, 0, 1, 2]), Int32Array.from([2, 2, 1, 1, 0, 0, 0]));
  const builder = (value: unknown) =>
    render(<SelectionBuilder rid={RID} grid={g()} value={value as SelectionSpec} onChange={() => {}} columns={[]} regions={[]} levers={LEVERS} idPrefix="t" />);

  it("counts a Top % whose k and within are null instead of reporting it invalid", async () => {
    let resolved = 0;
    fetchMock = mockFetch({ [`POST /api/runs/${RID}/selection/resolve`]: () => (resolved++, { body: RESOLVE }) });
    const { container } = builder({ kind: "top", column: "pred:target", frac: 0.1, k: null, direction: "highest", within: null });
    await waitFor(() => byText(container, '[data-testid="sel-count"]', "2 cells"), 5000, "count");
    expect(container.textContent).not.toContain("not a selection");
    expect(container.querySelector(".sel-desc")!.textContent).toBe("highest 10% by pred:target");
    click(byText(container, ".sel-summary button", "Edit"));
    expect(container.querySelector<HTMLInputElement>('input[aria-label="Share (%)"]')!.value).toBe("10");
    expect(resolved).toBeGreaterThan(0);
  });

  it("shows the count field for a top-k whose frac is null and the radius for a buffer whose lever_range is null", async () => {
    fetchMock = mockFetch({ [`POST /api/runs/${RID}/selection/resolve`]: { body: RESOLVE } });
    const top = builder({ kind: "top", column: "pred:target", frac: null, k: 50, direction: "lowest", within: null });
    await waitFor(() => byText(top.container, '[data-testid="sel-count"]', "2 cells"), 5000, "top count");
    expect(top.container.querySelector(".sel-desc")!.textContent).toBe("lowest 50 cells by pred:target");
    click(byText(top.container, ".sel-summary button", "Edit"));
    expect(top.container.querySelector<HTMLInputElement>('input[aria-label="Cells"]')!.value).toBe("50");
    expect(top.container.querySelector('input[aria-label="Share (%)"]')).toBeNull();
    const buf = builder({ kind: "buffer", of: { kind: "zones", values: [1] }, radius_m: 120, lever_range: null });
    await waitFor(() => byText(buf.container, '[data-testid="sel-count"]', "2 cells"), 5000, "buffer count");
    expect(buf.container.querySelector(".sel-desc")!.textContent).toBe("120 m around zone 1");
    click(byText(buf.container, ".sel-summary button", "Edit"));
    expect(buf.container.querySelector<HTMLInputElement>('input[aria-label="Radius"]')!.value).toBe("120");
    expect(buf.container.querySelector('select[aria-label="Lever range"]')).toBeNull();
  });
});

function acrossJob(id: string, sid: string, status: Job["status"], result: Job["result"] = null): Job {
  return {
    id, kind: "scenario.across_runs", lane: "heavy", executor: "process", label: "Check across runs", status, project_id: PID, run_id: null, study_id: null, scenario_id: sid,
    parent_job_id: null, after_job_id: null, priority: 0, params: { scenario_id: sid, run_ids: ["r_fast", "r_full"] }, created_utc: "2026-10-01T22:00:00Z", started_utc: null,
    finished_utc: status === "succeeded" ? "2026-10-01T22:10:00Z" : null, progress: null, eta_s: null, eta_lo: null, eta_hi: null, stage: null, current_path: null, exit_code: null,
    error: null, blocked: null, result, peak_rss_mb: null, threads: null,
  };
}

describe("Check across runs output", () => {
  const RESULT = {
    rows: [
      { run_id: "r_fast", city: L(-0.03, -0.04, -0.02, 0.005), ok: true, error: null },
      { run_id: "r_full", city: L(-0.021, -0.03, -0.012, 0.0046), ok: true, error: null },
      { run_id: "r_old", city: null, ok: false, error: "no checkpoint" },
    ],
    sign_stability: 1,
    spread: 0.009,
    content_hash: "h_a",
  };

  it("picks the scenario's newest finished check and a newer one still running", () => {
    const jobs = [acrossJob("j3", "sc_a", "running"), acrossJob("j2", "sc_b", "succeeded", RESULT), acrossJob("j1", "sc_a", "succeeded", RESULT), acrossJob("j0", "sc_a", "succeeded", RESULT)];
    const got = latestAcross(jobs, "sc_a");
    expect(got.done?.id).toBe("j1");
    expect(got.active?.id).toBe("j3");
    expect(latestAcross(jobs, "sc_x")).toEqual({ done: null, active: null });
    expect(latestAcross([acrossJob("j4", "sc_a", "failed")], "sc_a").done).toBeNull();
  });

  it("plots the run means with sign stability and spread, and lists refused runs", async () => {
    fetchMock = mockFetch({ "GET /api/jobs": { body: { items: [acrossJob("j1", "sc_a", "succeeded", RESULT)], next_cursor: null } } });
    const { container } = render(<AcrossRunsPanel pid={PID} sid="sc_a" unit="degF" />);
    await waitFor(() => container.querySelector('svg.chart[aria-label="The same scenario on other runs"]'), 5000, "dot plot");
    expect(fetchMock.calls[0].url).toBe(`/api/jobs?kind=scenario.across_runs&project=${PID}&limit=100`);
    expect(container.textContent).toContain("2 of 3 runs evaluated");
    expect(container.textContent).toContain("Sign stability 100%");
    expect(byText(container, ".edit-issues li", "r_old: no checkpoint")).not.toBeNull();
    // without the shown result's content the spread is not called its band
    expect(container.textContent).not.toContain("the specification band of this result");
  });

  it("calls the spread the result's specification band only for a check of the result's content", async () => {
    expect(bandNote("h_a", "h_a")).toBe("the specification band of this result");
    expect(bandNote("h_a", "h_b")).toMatch(/earlier version of the scenario, not this result's specification band/);
    expect(bandNote(undefined, "h_a")).toMatch(/not this result's specification band/);
    fetchMock = mockFetch({ "GET /api/jobs": { body: { items: [acrossJob("j1", "sc_a", "succeeded", RESULT)], next_cursor: null } } });
    const { container } = render(<AcrossRunsPanel pid={PID} sid="sc_a" unit="degF" contentHash="h_a" />);
    await waitFor(() => container.querySelector('svg.chart[aria-label="The same scenario on other runs"]'), 5000, "dot plot");
    expect(container.textContent).toContain("(the specification band of this result)");
  });
});

describe("Result inspector cost", () => {
  it("calls a negative cooling_per_cost warming in the plain card and the Cost KPI", async () => {
    // A canopy-loss scenario warms the city: cost_table's cooling_per_cost (positive = cooler) is negative.
    const warm = result("res_w", {
      city: L(0.84, 0.7, 0.98, 0.07),
      summary: { ...result("res_w").summary, city: L(0.84, 0.7, 0.98, 0.07), edited: L(1.9, 1.6, 2.2, 0.15) },
      cost: { total: 100, per_lever: { Pct_Canopy: 100 }, cooling_per_cost: -0.6117 },
      plain: { headline: "", confidence: "", qualifiers: [], buys: ["-611.73 °F·cells of cooling per 1,000 cost units"] },
    });
    fetchMock = mockFetch({ "GET /api/results/res_w": { body: warm } });
    const { container } = render(<ResultInspector rid={RID} resId="res_w" pid={PID} unit="degF" nFolds={3} />);
    await waitFor(() => container.querySelector(".plain-result"), 5000, "inspector");
    const text = container.textContent ?? "";
    expect(text).toContain("What it buys: 611.700 °F·cells of warming per 1,000 cost units.");
    expect(text).toContain("611.700 °F·cells of warming per 1k");
    expect(text).not.toContain("of cooling per");
  });
});
