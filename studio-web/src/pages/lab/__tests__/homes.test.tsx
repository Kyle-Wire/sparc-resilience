// Phase 2 homes in the Scenario Lab: the Lab's own nav (Design · Library · Plans · Compare, then
// "More tools" holding Try amounts; no Climate), Library's first group "Set up with this run"
// (the run's configured scenarios, formerly the Scenarios run tab, shown once) and Plans' first
// entry "Plan from this run" (the run's S7 budget plan, formerly the Budget run tab, read-only).
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { clearResources } from "../../../api/resource";
import { navigate } from "../../../router";
import { byText, click, mockFetch, render, waitFor } from "../../../test/render";
import { viewFixture } from "../../runhub/__fixtures__/views";
import { LabFrame } from "../components/LabFrame";
import Library from "../Library";
import Plans from "../Plans";
import { PID, RID, result, runRoutes } from "../__fixtures__/lab";

let fetchMock: ReturnType<typeof mockFetch> | null = null;

beforeEach(() => clearResources());

afterEach(() => {
  fetchMock?.restore();
  fetchMock = null;
});

const navLabels = (root: ParentNode) => [...root.querySelectorAll('nav[aria-label="Scenario Lab"] > a, nav[aria-label="Scenario Lab"] > button')].map((el) => el.textContent?.trim());

describe("Lab nav", () => {
  it("lists Design · Library · Plans · Compare, then More tools (Try amounts); Climate has moved to Heat risk", () => {
    navigate(`/r/${RID}/lab/library`, { replace: true });
    const { container } = render(
      <LabFrame rid={RID} title="Scenario library" hideEngine>
        <p />
      </LabFrame>,
    );
    expect(navLabels(container)).toEqual(["Design", "Library", "Plans", "Compare", "More tools"]);
    expect(byText(container, "a", "Library")!.getAttribute("aria-current")).toBe("page");
    expect(byText(container, "a", /Climate/)).toBeNull();
    const more = byText(container, "button", "More tools")!;
    expect(more.getAttribute("aria-expanded")).toBe("false");
    expect(byText(container, "a", "Try amounts")).toBeNull();
    click(more);
    expect(more.getAttribute("aria-expanded")).toBe("true");
    const tryAmounts = byText(container, "a", "Try amounts")!;
    expect(tryAmounts.getAttribute("href")).toBe(`/r/${RID}/lab/sweeps`);
    expect(document.getElementById(more.getAttribute("aria-controls")!)!.contains(tryAmounts)).toBe(true);
  });

  it("opens More tools by itself on the sweeps page, marking Try amounts current", () => {
    navigate(`/r/${RID}/lab/sweeps/sw_1`, { replace: true });
    const { container } = render(
      <LabFrame rid={RID} title="Try amounts" hideEngine>
        <p />
      </LabFrame>,
    );
    expect(byText(container, "button", "More tools")!.getAttribute("aria-expanded")).toBe("true");
    expect(byText(container, "a", "Try amounts")!.getAttribute("aria-current")).toBe("page");
    expect(container.querySelectorAll('nav[aria-label="Scenario Lab"] a[aria-current="page"]').length).toBe(1);
  });
});

describe("Library › Set up with this run", () => {
  it("comes first with the run's configured scenarios (effects, one table; ladders and causal band left to their tabs) and the Lab's row actions", async () => {
    const vm = viewFixture("scenarios");
    const rows = vm.sections.rows!;
    vm.sections.rows = rows.map((r, i) => ({ ...r, has_folds: i !== 0 }));
    fetchMock = mockFetch({
      ...runRoutes(),
      [`GET /api/projects/${PID}/scenarios`]: { body: [{ id: "sc_a", revision: 1, parent_id: null, status: "exact", created_utc: "", updated_utc: "", name: "Downtown cool corridor", tags: [], latest: result("res_1").summary }] },
      [`GET /api/runs/${RID}/views/scenarios`]: { body: vm },
    });
    navigate(`/r/${RID}/lab/library`, { replace: true });
    const { container } = render(<Library />);
    await waitFor(() => container.querySelector('figure.chart-frame[data-chart="City-mean change per scenario"]'), 8000, "configured scenario effects");
    const groups = [...container.querySelectorAll("[data-group]")].map((g) => g.getAttribute("data-group"));
    expect(groups).toEqual(["configured", "project"]);
    const setUp = container.querySelector('[data-group="configured"]')!;
    expect(setUp.querySelector("h2")!.textContent).toBe("Set up with this run");
    // the ladders repeat What cools' dose-response curves: a one-line link instead
    expect(setUp.querySelector('figure.chart-frame[data-chart="Scenario ladders"]')).toBeNull();
    expect(setUp.querySelector("[data-ladders-link] a")!.getAttribute("href")).toBe(`/r/${RID}/effects`);
    // the causal band is How much to trust this's: no diamonds or "outside the causal band" caption, a badge per row
    const effects = setUp.querySelector('figure.chart-frame[data-chart="City-mean change per scenario"]')!;
    expect(effects.querySelector("figcaption")!.textContent).not.toMatch(/causal/i);
    expect(effects.querySelector("details.table-view")!.textContent).not.toMatch(/causal/i);
    // the configured scenarios are listed once, in this group (the old "Configured in this run" table is gone)
    const tables = [...container.querySelectorAll("table.tbl")].filter((t) => t.querySelector("caption")?.textContent === "Configured scenarios");
    expect(tables.length).toBe(1);
    expect(setUp.contains(tables[0])).toBe(true);
    expect(tables[0].querySelectorAll("tbody tr").length).toBe(rows.length);
    expect(byText(container, "h3", "Configured in this run")).toBeNull();
    // in the Lab, the rows act in place: clone, pin, map, and re-run exactly when the fold detail is missing
    const first = tables[0].querySelector(`[data-configured="${rows[0].slug}"]`)!.closest("tr")!;
    expect(byText(first, "button", "Clone to edit")).not.toBeNull();
    expect(byText(first, "button", "Pin")).not.toBeNull();
    expect(byText(first, "a", "Map")!.getAttribute("href")).toBe(`/r/${RID}/map?layer=${encodeURIComponent(`sc:${rows[0].slug}`)}`);
    expect(byText(first, "button", "Re-run exactly")).not.toBeNull();
    expect(first.textContent).toContain("causal differs");
    const second = tables[0].querySelector(`[data-configured="${rows[1].slug}"]`)!.closest("tr")!;
    expect(byText(second, "button", "Re-run exactly")).toBeNull();
    // one row has a verdict, so the Verdict column shows
    expect(byText(tables[0], "th", /^Verdict/)).not.toBeNull();
    expect(byText(setUp, "a", "Open in Lab")).toBeNull();
    // the project's own scenarios follow
    const project = container.querySelector('[data-group="project"]')!;
    expect(project.querySelector("h2")!.textContent).toBe("Project scenarios");
    expect(byText(project, "a", "Downtown cool corridor")).not.toBeNull();
  });

  it("hides the Verdict column when no configured scenario has a verdict", async () => {
    const vm = viewFixture("scenarios");
    vm.sections.rows = vm.sections.rows!.map((r) => ({ ...r, verdict: null }));
    fetchMock = mockFetch({ ...runRoutes(), [`GET /api/runs/${RID}/views/scenarios`]: { body: vm } });
    navigate(`/r/${RID}/lab/library`, { replace: true });
    const { container } = render(<Library />);
    await waitFor(() => [...container.querySelectorAll("table.tbl")].find((t) => t.querySelector("caption")?.textContent === "Configured scenarios"), 8000, "configured table");
    const table = [...container.querySelectorAll("table.tbl")].find((t) => t.querySelector("caption")?.textContent === "Configured scenarios")!;
    expect(byText(table, "th", /^Confidence/)).not.toBeNull();
    expect(byText(table, "th", /^Verdict/)).toBeNull();
  });

  it("says when ?configured= names no configured scenario of this run", async () => {
    fetchMock = mockFetch({ ...runRoutes(), [`GET /api/runs/${RID}/views/scenarios`]: { body: viewFixture("scenarios") } });
    navigate(`/r/${RID}/lab/library?configured=nope`, { replace: true });
    const { container } = render(<Library />);
    await waitFor(() => byText(container, "p.callout", /no configured scenario “nope”/), 8000, "unknown configured note");
    expect(container.querySelector("tr.hl")).toBeNull();
  });
});

describe("Plans › Plan from this run", () => {
  it("is the first entry, read-only, with the budget numbers, Pareto, dose and closed-loop maps and top cells", async () => {
    fetchMock = mockFetch({ ...runRoutes(), [`GET /api/runs/${RID}/views/budget`]: { body: viewFixture("budget") } });
    navigate(`/r/${RID}/lab/plans`, { replace: true });
    const { container } = render(<Plans />);
    await waitFor(() => container.querySelector('figure.chart-frame[data-chart^="Cooling bought per budget"]'), 8000, "run plan Pareto");
    const sections = [...container.querySelectorAll('[data-composer="plans"] > section')];
    const run = sections[0];
    expect(run.getAttribute("data-plan")).toBe("run");
    expect(run.querySelector("h2")!.textContent).toBe("Plan from this run");
    expect(run.querySelector("header")!.textContent).toContain("read-only");
    expect(run.querySelector(".kpis")!.textContent).toContain("Cells treated");
    expect(byText(run, "h3", "Dose and closed-loop maps")).not.toBeNull();
    expect([...run.querySelectorAll("table.tbl caption")].some((c) => c.textContent === "Top cells by cooling")).toBe(true);
    // read-only: no plan actions, and no link out to another plans page
    expect(byText(run, "button", /Verify|Delete|Plan → scenario/)).toBeNull();
    expect(byText(run, "a", "Re-plan in Lab")).toBeNull();
    // then a new plan, then the saved plans
    expect(sections.slice(1).map((s) => s.querySelector("h2")?.textContent)).toEqual(["New plan", "Saved plans"]);
    expect(sections[1].querySelector('[aria-label="Plan parameters"]')).not.toBeNull();
  });

  it("explains a run plan with no positive-benefit segments without linking away from the Lab", async () => {
    const vm = viewFixture("budget");
    vm.sections.status = "no positive-benefit segments";
    fetchMock = mockFetch({ ...runRoutes(), [`GET /api/runs/${RID}/views/budget`]: { body: vm } });
    navigate(`/r/${RID}/lab/plans`, { replace: true });
    const { container } = render(<Plans />);
    await waitFor(() => container.querySelector('[data-status="no positive-benefit segments"]'), 8000, "no-allocation state");
    const empty = container.querySelector('[data-plan="run"] [data-status="no positive-benefit segments"]')!;
    expect(empty.textContent).toContain("No cell has a positive cooling footprint");
    expect(empty.querySelector("a")).toBeNull();
  });
});
