// Validation tab, Studies hub and study page against fetch mocks built from api.md §9: one card
// per kind with its result view (placebo verdicts, simcheck grid, multiverse heatmaps, the
// reproduce checklist, benchmark shares, baselines forest, layered intervals) and Truth vs
// recovered on demo runs; attach/detach; the hub's matrix links; the study page's child runs,
// resume, simcheck merge and delete.
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { clearResources } from "../../../api/resource";
import { STUDY_KINDS } from "../../../api/studies";
import { ProjectLayout } from "../../../layouts/ProjectLayout";
import { RunLayout } from "../../../layouts/RunLayout";
import { navigate } from "../../../router";
import { useJobs } from "../../../stores/jobs";
import { byText, click, flush, mockFetch, render, waitFor, type MockHandler } from "../../../test/render";
import StudiesHub, { cellHref, hubColumns } from "../StudiesHub";
import StudyPage, { canResume } from "../StudyPage";
import Validation, { cardRows } from "../Validation";
import { job, PID, projectDetail, RID, run, runDetail, simcheckView, statusRow, study } from "../__fixtures__/api";
import { benchmarkView, multiverseView, placeboView, reproduceView, truthRows } from "../__fixtures__/views";

function vm(view: string, sections: Record<string, unknown>) {
  return { view, availability: "ready", missing: [], units: { target: "°F", levers: {} }, caveats: [], demo: true, sections };
}

const ROWS = [
  statusRow("baselines", { state: "done", headline: "Stack beats 4 of 5 baselines" }),
  statusRow("planner", { requirements: { ok: false, missing: ["planner.layers"] } }),
  statusRow("placebo", { state: "done", study_id: "st_p", attached: false }),
  statusRow("simcheck", { state: "running", study_id: "st_s", job_id: "j_s", attached: true }),
  statusRow("multiverse", { state: "done", study_id: "st_m", attached: true }),
  statusRow("reproduce", { state: "failed", study_id: "st_r", job_id: "j_r" }),
  statusRow("uncertainty", { state: "done" }),
  statusRow("benchmark", { state: "done", study_id: "st_b" }),
];

function validationRoutes(extra: Record<string, MockHandler> = {}): Record<string, MockHandler> {
  return {
    [`GET /api/runs/${RID}`]: { body: runDetail() },
    [`GET /api/runs/${RID}/outputs`]: { body: { outputs: [], tabs: [] } },
    [`GET /api/runs/${RID}/studies`]: { body: ROWS },
    [`GET /api/projects/${PID}/studies`]: { body: [study("st_p", "placebo"), study("st_s", "simcheck", { status: "running" }), study("st_m", "multiverse")] },
    [`GET /api/runs/${RID}/views/overview`]: { body: vm("overview", {}) },
    [`GET /api/runs/${RID}/scenarios`]: { body: { configured: [], results: [] } },
    "GET /api/settings": { body: { threads_heavy: 4 } },
    "POST /api/studies/estimate": { body: { est_s: 60, est_lo: 50, est_hi: 90, est_peak_rss_gb: 1, est_disk_gb: 0.01, n_children: 0 } },
    "GET /api/studies/st_p/view": { body: placeboView() },
    "GET /api/studies/st_s/view": { body: simcheckView() },
    "GET /api/studies/st_m/view": { body: multiverseView() },
    "GET /api/studies/st_r/view": { body: reproduceView() },
    "GET /api/studies/st_b/view": { body: benchmarkView() },
    [`GET /api/runs/${RID}/truth`]: { body: { rows: truthRows() } },
    [`GET /api/runs/${RID}/views/distance`]: {
      body: vm("distance", {
        curve: null,
        baselines: [
          { id: "hgb", label: "Gradient boosting", rmse: 0.4, r2: 0.7, delta_mse: 0.05, delta_mse_se: 0.01, stack_better: true, baseline_better: false },
          { id: "idw", label: "IDW", rmse: 0.6, r2: 0.4, delta_mse: 0.2, delta_mse_se: 0.03, stack_better: true, baseline_better: false },
        ],
        verdict: { text: "The stack beats every baseline by more than 2 SE.", best_baseline: "hgb", stack_wins: true },
        block_wins: null,
      }),
    },
    [`GET /api/runs/${RID}/views/uncertainty`]: {
      body: vm("uncertainty", {
        rows: [{ id: "c10", label: "canopy +10", estimate: -0.9, layers: [{ id: "estimation", label: "Estimation", lo: -1.0, hi: -0.8 }, { id: "envelope", label: "Envelope", lo: -1.3, hi: -0.4 }] }],
        climate: null,
        sources: [],
      }),
    },
    ...extra,
  };
}

async function mountValidation(extra: Record<string, MockHandler> = {}) {
  const m = mockFetch(validationRoutes(extra));
  navigate(`/r/${RID}/validation`, { replace: true });
  const r = render(
    <RunLayout rid={RID}>
      <Validation />
    </RunLayout>,
  );
  await waitFor(() => r.container.querySelector('[data-kind="benchmark"] [aria-label="Effect share recovered on the synthetic city"], [data-kind="benchmark"] svg'), 5000, "cards");
  await flush(4);
  return { ...r, m };
}

beforeEach(() => useJobs.getState().reset());
afterEach(() => {
  clearResources();
  navigate("/", { replace: true });
});

describe("Validation tab", () => {
  it("shows one card per kind, in order, even for kinds the server omitted", () => {
    expect(cardRows([statusRow("placebo", { state: "done" })]).map((r) => [r.kind, r.state])).toEqual(STUDY_KINDS.map((k) => [k, k === "placebo" ? "done" : "not_run"]));
  });

  it("renders every card with its status and result view", async () => {
    const { container, m } = await mountValidation();
    const cards = [...container.querySelectorAll("article.vt-card")];
    expect(cards.map((c) => c.getAttribute("data-kind"))).toEqual([...STUDY_KINDS]);
    const card = (k: string) => container.querySelector(`article[data-kind="${k}"]`)!;
    // baselines forest with the verdict
    expect(card("baselines").textContent).toContain("Stack beats 4 of 5 baselines");
    expect(card("baselines").textContent).toContain("Stack vs reference baselines");
    expect(card("baselines").textContent).toContain("2 of 2 baselines are beaten");
    // placebo verdict table and Δ per sd vs real
    const placebo = card("placebo");
    expect(placebo.textContent).toContain("1 of 2 placebo layers pass the model check");
    expect(placebo.querySelector('[aria-label="Placebo verdicts"]')!.querySelectorAll("tbody tr")).toHaveLength(3);
    expect(placebo.textContent).toContain("Δ per sd: placebo vs real");
    expect(byText(placebo, "a", "child run")!.getAttribute("href")).toBe("/r/child-grf");
    // simcheck grid, live job link, attached toggle
    const sim = card("simcheck");
    expect(sim.querySelectorAll("table.sg-grid td[data-status]").length).toBeGreaterThan(0);
    expect(sim.textContent).toContain("Bias correction");
    expect(sim.textContent).toContain("False positives (null generator)");
    expect(byText(sim, "a", "Track in Mission Control")!.getAttribute("href")).toBe("/jobs/j_s");
    expect(byText(sim, "button", "Attached")!.getAttribute("aria-pressed")).toBe("true");
    // multiverse heatmaps and R² by variant
    const mv = card("multiverse");
    expect(mv.textContent).toContain("Scenario effect by variant");
    expect(mv.textContent).toContain("Priority agreement with the baseline");
    expect(mv.textContent).toContain("Held-out R² by variant");
    expect(mv.textContent).toContain("Sign changes under some variants for albedo +0.1 (50%)");
    // reproduce checklist
    const rep = card("reproduce");
    expect(rep.querySelector('[aria-label="Reproduction checks"]')!.querySelectorAll("tbody tr")).toHaveLength(3);
    expect(rep.textContent).toContain("1 hard check differ");
    expect(byText(rep, "a", "log and error")!.getAttribute("href")).toBe("/jobs/j_r");
    // uncertainty layered intervals, benchmark shares
    expect(card("uncertainty").textContent).toContain("Layered intervals per scenario");
    expect(card("benchmark").textContent).toContain("Stacked share: Standard 0.72, Spatial+ (MGWR) 0.78");
    // planner: requirements missing → launch disabled with the server's item
    const planner = card("planner");
    expect((planner.querySelector('[data-launch="planner"]') as HTMLButtonElement).disabled).toBe(true);
    expect(planner.querySelector('[data-requirement="planner.layers"]')).not.toBeNull();
    // study pages are linked
    expect(byText(placebo, "a", "Study page")!.getAttribute("href")).toBe("/studies/st_p");
    // Truth vs recovered (demo run)
    const truth = container.querySelector('[aria-label="Truth vs recovered"]')!;
    expect(truth.querySelector('table[aria-label="Truth vs recovered"]')!.querySelectorAll("tbody tr")).toHaveLength(3);
    expect(truth.textContent).toContain("attenuated (60% of the truth)");
    expect(truth.textContent).toContain("1 of 2 planted quantities are recovered within 20%");
    expect(m.calls.some((c) => c.url === `/api/runs/${RID}/truth`)).toBe(true);
    m.restore();
  });

  it("attaches a study (re-running the uncertainty report)", async () => {
    const { container, m } = await mountValidation({
      "POST /api/studies/st_p/attach": { body: { study: study("st_p", "placebo", { attached_runs: [RID] }), job: job("j_u", "post.uncertainty") } },
    });
    click(byText(container.querySelector('article[data-kind="placebo"]')!, "button", "Attach to run"));
    await flush(4);
    const post = m.calls.find((c) => c.url === "/api/studies/st_p/attach")!;
    expect(post.body).toEqual({ run_id: RID });
    expect(useJobs.getState().jobs.j_u).toBeDefined();
    m.restore();
  });

  it("hides Truth vs recovered on real runs", async () => {
    const m = mockFetch(validationRoutes({ [`GET /api/runs/${RID}`]: { body: runDetail(RID, { demo: false }) } }));
    navigate(`/r/${RID}/validation`, { replace: true });
    const { container } = render(
      <RunLayout rid={RID}>
        <Validation />
      </RunLayout>,
    );
    await waitFor(() => container.querySelector("article.vt-card"), 5000, "cards");
    await flush(3);
    expect(container.querySelector('[aria-label="Truth vs recovered"]')).toBeNull();
    expect(m.calls.some((c) => c.url === `/api/runs/${RID}/truth`)).toBe(false);
    m.restore();
  });
});

describe("Studies hub", () => {
  const board = {
    columns: [
      { id: "S0", label: "S0", group: "stage" },
      { id: "baselines", label: "Baselines", group: "stage" },
      { id: "planner", label: "Planner", group: "post" },
      { id: "placebo", label: "Placebo", group: "study" },
      { id: "multiverse", label: "Multiverse", group: "study" },
    ],
    rows: [
      {
        run: run(),
        cells: {
          S0: { state: "done", seconds: 3, progress: null, reason: null, job_id: null, study_id: null, action: null },
          baselines: { state: "done", seconds: 30, progress: null, reason: null, job_id: null, study_id: null, action: null },
          planner: { state: "not_run", seconds: null, progress: null, reason: null, job_id: null, study_id: null, action: null },
          placebo: { state: "done", seconds: 900, progress: null, reason: null, job_id: null, study_id: "st_p", action: null },
          multiverse: { state: "running", seconds: null, progress: 0.4, reason: null, job_id: "j_m", study_id: null, action: null },
        },
      },
      { run: run("child-1", { origin: "study_child" }), cells: {} },
    ],
  };

  it("keeps the post-run and study columns and links cells to studies, jobs or Validation cards", () => {
    expect(hubColumns(board.columns).map((c) => c.id)).toEqual(["baselines", "planner", "placebo", "multiverse"]);
    expect(cellHref(RID, "placebo", board.rows[0].cells.placebo as never)).toBe("/studies/st_p");
    expect(cellHref(RID, "multiverse", board.rows[0].cells.multiverse as never)).toBe("/jobs/j_m");
    expect(cellHref(RID, "planner", board.rows[0].cells.planner as never)).toBe(`/r/${RID}/validation#study-planner`);
  });

  it("renders the matrix, the studies list and the benchmark launch", async () => {
    const m = mockFetch({
      [`GET /api/projects/${PID}`]: { body: projectDetail() },
      [`GET /api/projects/${PID}/status-board`]: { body: board },
      [`GET /api/projects/${PID}/studies`]: {
        body: [study("st_p", "placebo", { attached_runs: [RID] }), study("st_old", "simcheck", { stale_vs: [RID], created_utc: "2026-09-01T00:00:00Z" }), study("st_b", "benchmark", { target_run_id: null })],
      },
      "GET /api/settings": { body: { threads_heavy: 4 } },
      "POST /api/studies/estimate": { body: { est_s: 120, est_lo: 100, est_hi: 160, est_peak_rss_gb: 1, est_disk_gb: 0.01, n_children: 0 } },
      [`POST /api/projects/${PID}/studies/benchmark`]: { status: 202, body: { study: study("st_b2", "benchmark", { target_run_id: null }), job: job("j_b", "study.benchmark") } },
    });
    navigate(`/p/${PID}/studies`, { replace: true });
    const { container } = render(
      <ProjectLayout pid={PID}>
        <StudiesHub />
      </ProjectLayout>,
    );
    const matrix = await waitFor(() => container.querySelector('[aria-label="Studies by run"]'), 4000, "matrix");
    expect([...matrix.querySelectorAll("thead th")].map((t) => t.textContent)).toEqual(["Run", "Baselines", "Planner", "Placebo", "Multiverse"]);
    expect(matrix.querySelectorAll("tbody tr")).toHaveLength(1); // study children are not rows
    expect(matrix.querySelector('td[data-col="placebo"] a')!.getAttribute("href")).toBe("/studies/st_p");
    expect(matrix.querySelector('td[data-col="multiverse"]')!.textContent).toContain("running 40%");
    await waitFor(() => container.querySelector('[aria-label="Project studies"]'), 3000, "list");
    const list = container.querySelector('[aria-label="Project studies"]')!;
    expect([...list.querySelectorAll("tbody tr")].map((r) => r.getAttribute("data-study"))).toEqual(["st_p", "st_b", "st_old"]);
    expect(list.textContent).toContain("stale vs 1 run");
    expect(container.textContent).toContain("Last run:");
    const bench = container.querySelector('[aria-label="Effect benchmark"]')!;
    await waitFor(() => bench.querySelector('[aria-label="Estimated cost"]'), 3000, "estimate");
    click(bench.querySelector('[data-launch="benchmark"]'));
    await flush(4);
    expect(m.calls.find((c) => c.url === `/api/projects/${PID}/studies/benchmark`)!.body).toEqual({ seed: 0, ab: true, epochs: 150, n: 96 });
    m.restore();
  });
});

describe("Study page", () => {
  it("resumes only studies that ended without finishing", () => {
    expect(canResume({ status: "failed", origin: "studio" })).toBe(true);
    expect(canResume({ status: "interrupted", origin: "studio" })).toBe(true);
    expect(canResume({ status: "running", origin: "studio" })).toBe(false);
    expect(canResume({ status: "done", origin: "studio" })).toBe(false);
    expect(canResume({ status: "failed", origin: "imported" })).toBe(false);
  });

  it("shows child runs with run and tracker links, the result view, resume and the simcheck merge", async () => {
    const st = study("st_s", "simcheck", {
      status: "interrupted",
      job_id: "j_s",
      params: { workers: 2 },
      summary: { headline: "Median share 0.9 across 2 generators" },
      children: [run("child-a", { origin: "study_child", label: "physics seed 0" })],
      attached_runs: [RID],
      stale_vs: [RID],
    });
    const m = mockFetch({
      "GET /api/studies/st_s": { body: st },
      "GET /api/studies/st_s/view": { body: simcheckView() },
      [`GET /api/runs/${RID}/views/overview`]: { body: vm("overview", {}) },
      [`GET /api/projects/${PID}/studies`]: { body: [st, study("st_s2", "simcheck"), study("st_p", "placebo")] },
      "POST /api/studies/st_s/resume": { status: 202, body: job("j_res", "study.simcheck") },
      "POST /api/studies/simcheck/merge": {
        body: { summary: { generators: simcheckView().generators, bias_correction: simcheckView().bias_correction, n_rows: 9, n_errors: 1 }, markdown: "## Merged\n\n| generator | share |" },
      },
      "DELETE /api/studies/st_s": { body: { ok: true } },
    });
    navigate("/studies/st_s", { replace: true });
    const { container } = render(<StudyPage />);
    await waitFor(() => container.querySelector('[aria-label="Child runs"]'), 4000, "study page");
    await flush(3);
    const child = container.querySelector('tr[data-child="child-a"]')!;
    expect(byText(child, "a", "Run")!.getAttribute("href")).toBe("/r/child-a");
    expect(byText(child, "a", "Tracker")!.getAttribute("href")).toBe("/r/child-a/track");
    expect(container.textContent).toContain("Median share 0.9 across 2 generators");
    expect(container.textContent).toContain("changed since this study ran");
    expect(container.querySelector("table.sg-grid")).not.toBeNull();
    expect(byText(container, "a", "j_s")!.getAttribute("href")).toBe("/jobs/j_s");
    click(byText(container, "button", "Resume"));
    await flush(3);
    expect(m.calls.some((c) => c.method === "POST" && c.url === "/api/studies/st_s/resume")).toBe(true);
    expect(useJobs.getState().jobs.j_res).toBeDefined();
    // merge with the other simcheck study (placebo studies are not offered)
    const merge = container.querySelector('[aria-label="Merge simulation checks"]')!;
    await waitFor(() => byText(merge, "label", "st_s2"), 3000, "merge choices");
    expect(byText(merge, "label", "st_p")).toBeNull();
    click(byText(merge, "label", "st_s2")!.querySelector("input"));
    click(byText(merge, "button", "Merge 2 studies"));
    await flush(4);
    expect(m.calls.find((c) => c.url === "/api/studies/simcheck/merge")!.body).toEqual({ study_ids: ["st_s", "st_s2"] });
    expect(merge.textContent).toContain("9 replicates, 1 errors");
    expect(merge.querySelector('[aria-label="Simulation check by generator"]')).not.toBeNull();
    // delete with files
    click(byText(container, "button", "Delete"));
    await flush(2);
    click(byText(document.body, ".dialog label", "Also delete its folder")!.querySelector("input"));
    click(byText(document.body, ".dialog button", "Delete"));
    await flush(4);
    expect(m.calls.find((c) => c.method === "DELETE")!.url).toBe("/api/studies/st_s?files=true");
    expect(window.location.pathname).toBe(`/p/${PID}/studies`);
    m.restore();
  });

  it("says when a study does not exist", async () => {
    const m = mockFetch({ "GET /api/studies/st_x": { status: 404, body: { error: { code: "not_found", message: "No such study" } } } });
    navigate("/studies/st_x", { replace: true });
    const { container } = render(<StudyPage />);
    await waitFor(() => container.textContent!.includes("Study not found"), 3000, "not found");
    m.restore();
  });
});
