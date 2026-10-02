// Review additions: nothing is deleted without asking (a scenario with later revisions only
// after a second confirmation), the "Around sites" template sends typed or uploaded [lon, lat]
// sites, and a selection builder tints the map only while it is being worked on.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { clearResources } from "../../../api/resource";
import type { ScenarioTemplate } from "../../../api/lab";
import type { Job } from "../../../api/types";
import { byText, click, flush, mockFetch, render, typeInto, waitFor } from "../../../test/render";
import { AcrossRunsPanel, latestAcross } from "../components/AcrossRuns";
import { DeleteScenarioDialog } from "../components/Dialogs";
import { SelectionBuilder } from "../components/SelectionBuilder";
import { TemplateGallery } from "../components/TemplateGallery";
import { makeGrid } from "../../../map/grid";
import { L, LEVERS, PID, RID, gridMeta, scenario } from "../__fixtures__/lab";

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
  });
});
