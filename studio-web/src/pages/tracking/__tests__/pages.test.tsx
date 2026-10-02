// Project overview (Status Board, readiness spine, primary CTA), Activity, run history, Settings
// and the route declarations of the tracking pages.
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { clearResources } from "../../../api/resource";
import type { StatusBoard as Board } from "../../../api/tracking";
import type { Project, ReadinessRow, RunSummary } from "../../../api/types";
import { ProjectLayout } from "../../../layouts/ProjectLayout";
import { navigate, registry } from "../../../router";
import { useJobs } from "../../../stores/jobs";
import { byText, click, flush, mockFetch, render, typeInto, waitFor } from "../../../test/render";
import boardJson from "../__fixtures__/status-board.json";
import Activity, { canRestart } from "../Activity";
import ProjectOverview, { primaryCta } from "../ProjectOverview";
import RunHistory, { compareTarget, stageHistoryChart } from "../RunHistory";
import Settings, { settingsIssues, settingsPatch } from "../Settings";
import { BOARD_REFRESH_MS, StatusBoard, StatusBoardTable } from "../StatusBoard";
import { makeJob } from "./events";

const board = boardJson as unknown as Board;
let fetchMock: ReturnType<typeof mockFetch> | null = null;

beforeEach(() => {
  clearResources();
  useJobs.getState().reset();
});
afterEach(() => {
  fetchMock?.restore();
  fetchMock = null;
  navigate("/", { replace: true });
});

const project: Project = {
  id: "p_demo", slug: "demo", name: "Demo city", dir: "/w/projects/demo", config_path: "/w/projects/demo/config.yml", template: "synthetic_demo", demo: true,
  active_run_id: null, archived: false, created_utc: "2026-10-01T00:00:00Z", updated_utc: "2026-10-01T00:00:00Z", report: { title: null, place: null, area: null },
  headline_scenario: null, cost_model: {}, n_runs: 0, last_run: null, active_jobs: 0, readiness_score: { done: 8, total: 10 },
};

const readiness: ReadinessRow[] = [
  { key: "data", label: "Temperature data", state: "ok", detail: "1,120 points", action: null },
  { key: "forcing", label: "Campaign forcing", state: "warn", detail: "station only", action: { kind: "fetch_input", label: "Fetch forcing", method: "POST", path: "/api/projects/p_demo/inputs/forcing", body: {} } },
  { key: "people_layers", label: "People and land cover", state: "missing", detail: "needed by the planner", action: { kind: "fetch_input", label: "Fetch layers", method: "POST", path: "/api/projects/p_demo/inputs/layers", body: {} } },
  { key: "emulator", label: "Emulator", state: "n/a", detail: "after a run", action: null },
];

function runSummary(patch: Partial<RunSummary>): RunSummary {
  return { ...board.rows[0].run, ...patch } as RunSummary;
}

describe("Pipeline Status Board", () => {
  it("renders one row per run and every column, chips with icon and text", () => {
    const { container } = render(<StatusBoardTable board={board} />);
    const rows = container.querySelectorAll("tbody tr");
    expect(rows).toHaveLength(board.rows.length);
    const head = [...container.querySelectorAll("thead tr:not(.groups) th")].map((th) => th.textContent);
    expect(head).toEqual(["Run", ...board.columns.map((c) => c.label)]);
    expect([...container.querySelectorAll("thead tr.groups th[scope=colgroup]")].map((th) => th.textContent)).toEqual(["Pipeline stages", "Post-run actions", "Studies"]);
    rows.forEach((tr, i) => {
      expect(tr.getAttribute("data-run")).toBe(board.rows[i].run.id);
      const cols = [...tr.querySelectorAll("td")].map((td) => td.getAttribute("data-col"));
      expect(cols).toEqual(board.columns.map((c) => c.id));
      for (const chip of tr.querySelectorAll("td .status")) {
        expect(chip.querySelector("svg")).not.toBeNull();
        expect((chip.textContent ?? "").trim()).not.toBe("");
      }
    });
    const cell = (r: number, c: string) => container.querySelectorAll("tbody tr")[r].querySelector(`td[data-col="${c}"]`)!;
    expect(cell(0, "S2_S3").textContent).toContain("done");
    expect(cell(0, "S2_S3").textContent).toContain("15 s");
    expect(cell(0, "S2_S3").querySelector("a")!.getAttribute("href")).toBe("/jobs/j_a1?stage=S2_S3");
    expect(cell(1, "S2_S3").textContent).toContain("62%");
    expect(cell(2, "S1").textContent).toContain("cached");
    expect(cell(0, "cv_curve").textContent).toContain("not requested for this run");
    expect(cell(0, "multiverse").querySelector("a")!.getAttribute("href")).toBe("/studies/st_m1");
    expect(cell(0, "uncertainty").textContent).toContain("stale");
  });

  it("offers the action of every not-run cell that has one, and runs it", async () => {
    const withAction = board.rows.flatMap((r) => Object.values(r.cells).filter((c) => c.state === "not_run" && c.action));
    const job = makeJob({ id: "j_new", kind: "post.planner", label: "Planner pack", status: "queued" });
    fetchMock = mockFetch({ "POST /api/runs/20261001-101500-fast-ab12/actions/planner": { status: 202, body: job } });
    const { container } = render(<StatusBoardTable board={board} />);
    const buttons = [...container.querySelectorAll("td button.btn")];
    expect(buttons.map((b) => b.textContent)).toEqual(withAction.map((c) => c.action!.label));
    expect(buttons).toHaveLength(4);
    click(byText(container, "td button", "Run planner"));
    await flush();
    expect(fetchMock.calls).toEqual([{ method: "POST", url: "/api/runs/20261001-101500-fast-ab12/actions/planner", body: {} }]);
    expect(useJobs.getState().jobs.j_new?.status).toBe("queued");
  });

  it("refetches while a cell is running, and stops once nothing runs", async () => {
    vi.useFakeTimers({ toFake: ["setInterval", "clearInterval"] });
    try {
      let n = 0;
      const idle = { ...board, rows: board.rows.map((r) => ({ ...r, cells: Object.fromEntries(Object.entries(r.cells).map(([k, c]) => [k, c.state === "running" ? { ...c, state: "done" as const } : c])) })) };
      fetchMock = mockFetch({ "GET /api/projects/p_demo/status-board": () => ({ body: ++n < 3 ? board : idle }) });
      const { container } = render(<StatusBoard pid="p_demo" />);
      await waitFor(() => container.querySelector("table.board"), 5000, "board");
      expect(n).toBe(1);
      act(() => void vi.advanceTimersByTime(BOARD_REFRESH_MS));
      await flush();
      expect(n).toBe(2);
      act(() => void vi.advanceTimersByTime(BOARD_REFRESH_MS));
      await flush();
      expect(n).toBe(3);
      expect(container.querySelector('td[data-state="running"]')).toBeNull();
      act(() => void vi.advanceTimersByTime(3 * BOARD_REFRESH_MS));
      await flush();
      expect(n).toBe(3);
    } finally {
      vi.useRealTimers();
    }
  });

  it("loads the board of a project from its endpoint", async () => {
    fetchMock = mockFetch({ "GET /api/projects/p_demo/status-board": { body: board } });
    const { container } = render(<StatusBoard pid="p_demo" />);
    await waitFor(() => container.querySelector("table.board"), 5000, "board");
    expect(container.querySelectorAll("tbody tr")).toHaveLength(3);
  });
});

describe("project overview", () => {
  it("chooses the primary action: set up data → launch first run → open the Lab", () => {
    const missing = [{ ...readiness[0], state: "missing" as const }];
    expect(primaryCta({ project, readiness: missing, runs: [] })).toMatchObject({ label: "Set up data", to: "/p/p_demo/setup/data" });
    expect(primaryCta({ project, readiness, runs: [] })).toMatchObject({ label: "Launch first run", to: "/p/p_demo/launch" });
    const running = runSummary({ id: "r_live", status: "running", checkpoint_bytes: null });
    expect(primaryCta({ project, readiness, runs: [running] })).toMatchObject({ label: "Launch run" });
    const done = runSummary({ id: "r_done", status: "complete", checkpoint_bytes: 4_000_000 });
    expect(primaryCta({ project, readiness, runs: [running, done] })).toMatchObject({ label: "Open Scenario Lab", to: "/r/r_done/lab" });
  });

  it("shows the readiness spine with icon + text chips and actions, and the board", async () => {
    fetchMock = mockFetch({
      "GET /api/projects/p_demo": { body: { project, readiness, runs: [], active_jobs: [], config_version: 3 } },
      "GET /api/projects/p_demo/status-board": { body: { columns: board.columns, rows: [] } },
    });
    navigate("/p/p_demo", { replace: true });
    const { container } = render(
      <ProjectLayout pid="p_demo">
        <ProjectOverview />
      </ProjectLayout>,
    );
    await waitFor(() => container.querySelector("ul.spine"), 5000, "spine");
    const items = [...container.querySelectorAll("ul.spine > li")];
    expect(items.map((li) => li.querySelector(".status")!.textContent)).toEqual(["ok", "check", "missing", "n/a"]);
    for (const li of items) expect(li.querySelector(".status svg")).not.toBeNull();
    expect(items[2].querySelector("button")!.textContent).toBe("Fetch layers");
    expect(byText(container, "a.btn.primary", "Launch first run")!.getAttribute("href")).toBe("/p/p_demo/launch");
    await waitFor(() => byText(container, ".empty-title", "No runs yet"), 5000, "empty board");
  });
});

describe("Activity", () => {
  it("shows the queue per lane, pauses it and moves a queued job up", async () => {
    const a = makeJob({ id: "j_a", label: "Full run", status: "running", lane: "heavy", priority: 0 });
    const b = makeJob({ id: "j_b", label: "Placebo", status: "queued", lane: "heavy", priority: 0, created_utc: "2026-10-01T11:00:00Z" });
    const c = makeJob({ id: "j_c", label: "Multiverse", status: "queued", lane: "heavy", priority: 0, created_utc: "2026-10-01T11:05:00Z" });
    useJobs.getState().setActive([a, b, c]);
    let paused = false;
    fetchMock = mockFetch({
      "GET /api/queue": () => ({ body: { paused, lanes: [{ lane: "heavy", slots: 1, running: ["j_a"], queued: ["j_b", "j_c"] }, { lane: "network", slots: 2, running: [], queued: [] }] } }),
      "POST /api/queue/pause": () => {
        paused = true;
        return { body: { paused, lanes: [] } };
      },
      "PATCH /api/jobs/j_c": (_u, init) => ({ body: { ...c, priority: JSON.parse(String(init.body)).priority } }),
      "GET /api/jobs": { body: { items: [makeJob({ id: "j_old", label: "Old run", status: "failed", finished_utc: "2026-10-01T09:00:00Z", error: { type: "RuntimeError", message: "boom" } })], next_cursor: null } },
      "GET /api/meta": { body: { job_kinds: [{ kind: "run.core" }], warning_codes: [], output_catalog: [] } },
      "GET /api/projects": { body: [] },
    });
    const { container } = render(<Activity />);
    await waitFor(() => byText(container, "button", "Pause queue"), 5000, "queue");
    expect(byText(container, "th", "heavy")).not.toBeNull();
    click(byText(container, "button", "Pause queue"));
    await waitFor(() => byText(container, "button", "Resume queue"), 5000, "paused");
    expect(container.textContent).toContain("The queue is paused");
    click(container.querySelector('button[aria-label="Move Multiverse up"]'));
    await flush();
    const patch = fetchMock.calls.find((x) => x.method === "PATCH")!;
    expect(patch).toEqual({ method: "PATCH", url: "/api/jobs/j_c", body: { priority: 1 } });
    await waitFor(() => byText(container, "a", "Old run"), 5000, "history");
    expect(byText(container, "td", "boom")).not.toBeNull();
  });
});

describe("Activity history actions", () => {
  it("offers Resume or Retry for finished jobs Studio can start again, not for command-line runs", async () => {
    expect(canRestart({ kind: "run.core", status: "failed" })).toBe(true);
    expect(canRestart({ kind: "post.planner", status: "interrupted" })).toBe(true);
    expect(canRestart({ kind: "run.core", status: "succeeded" })).toBe(false);
    expect(canRestart({ kind: "run.external", status: "interrupted" })).toBe(false);
    const ext = makeJob({ id: "j_ext", kind: "run.external", executor: "external", lane: "none", label: "CLI run", status: "interrupted", finished_utc: "2026-10-01T09:00:00Z" });
    const post = makeJob({ id: "j_post", kind: "post.planner", label: "Planner pack", status: "failed", finished_utc: "2026-10-01T09:30:00Z" });
    fetchMock = mockFetch({
      "GET /api/queue": { body: { paused: false, lanes: [] } },
      "GET /api/jobs": (url: URL) => ({ body: { items: url.searchParams.get("status") === "interrupted" ? [ext] : [ext, post], next_cursor: null } }),
      "GET /api/meta": { body: { job_kinds: [], warning_codes: [], output_catalog: [] } },
      "GET /api/projects": { body: [] },
    });
    const { container } = render(<Activity />);
    const rowOf = (label: string) => byText(container, "tbody tr a", label)?.closest("tr") ?? null;
    await waitFor(() => rowOf("Planner pack"), 5000, "history");
    expect(byText(rowOf("Planner pack")!, "button", "Retry")).not.toBeNull();
    expect([...rowOf("CLI run")!.querySelectorAll("button")].map((b) => b.textContent)).not.toContain("Retry");
    expect([...rowOf("CLI run")!.querySelectorAll("button")].map((b) => b.textContent)).not.toContain("Resume");
    // the interrupted card says why there is no button
    const card = await waitFor(() => [...container.querySelectorAll("ul.spine > li")].find((li) => li.textContent?.includes("CLI run")), 5000, "interrupted card");
    expect(card.querySelector("button")).toBeNull();
    expect(card.textContent).toContain("started from the command line");
  });
});

describe("run history", () => {
  const r = (id: string, project_id: string | null): RunSummary => runSummary({ id, project_id });

  it("compares two runs of the same project only", () => {
    expect(compareTarget([r("a", "p1")]).href).toBeNull();
    expect(compareTarget([r("a", "p1"), r("b", "p2")]).reason).toContain("different projects");
    expect(compareTarget([r("a", "p1"), r("b", "p1")]).href).toBe("/p/p1/compare?a=a&b=b");
  });

  it("groups the stage-duration history by commit", () => {
    const rows = [
      { run_id: "r3", label: "third", git_commit: "bbbbbbb1", stage: "S2_S3", seconds: 20, n_points: 10, mode: "fast", threads: 3 },
      { run_id: "r1", label: "first", git_commit: "aaaaaaa1", stage: "S2_S3", seconds: 10, n_points: 10, mode: "fast", threads: 3 },
      { run_id: "r2", label: "second", git_commit: "bbbbbbb1", stage: "S2_S3", seconds: 22, n_points: 10, mode: "fast", threads: 3 },
      { run_id: "r2", label: "second", git_commit: "bbbbbbb1", stage: "S4", seconds: 5, n_points: 10, mode: "fast", threads: 3 },
    ];
    const all = stageHistoryChart(rows, null);
    expect(all.categories).toEqual(["bbbbbbb · third", "bbbbbbb · second", "aaaaaaa · first"]);
    expect(all.series.map((s) => s.id)).toEqual(["S2_S3", "S4"]);
    expect(all.series[1].values).toEqual([null, 5, null]);
    expect(stageHistoryChart(rows, "S2_S3").series).toEqual([{ id: "S2_S3", label: "S2_S3", values: [20, 22, 10] }]);
  });

  it("lists runs and links two selected runs to Compare", async () => {
    const runs = [r("r_a", "p_demo"), r("r_b", "p_demo"), { ...r("r_c", "p_demo"), label: "Third" }].map((x, i) => ({ ...x, label: x.label === "Third" ? "Third" : `Run ${i + 1}` }));
    fetchMock = mockFetch({
      "GET /api/projects/p_demo/runs": { body: { items: runs, next_cursor: null } },
      "GET /api/timings": { body: { unit_rates: [], stage_history: [] } },
    });
    navigate("/p/p_demo/runs", { replace: true });
    const { container } = render(<RunHistory />);
    await waitFor(() => byText(container, "a", "Run 1"), 5000, "runs");
    expect(container.querySelector("a.btn.primary")).toBeNull();
    click(container.querySelector('input[aria-label="Select Run 1 for comparison"]'));
    click(container.querySelector('input[aria-label="Select Run 2 for comparison"]'));
    const link = await waitFor(() => byText(container, "a.btn.primary", "Compare selected"), 5000, "compare link");
    expect(link.getAttribute("href")).toBe("/p/p_demo/compare?a=r_a&b=r_b");
  });
});

describe("Settings", () => {
  const settings = {
    thread_budget: 4, threads_heavy: 3, engine_threads: 2, heavy_slots: 1, medium_slots: 1, network_slots: 2, engine_max_runs: 2, engine_mem_budget_gb: 6,
    engine_idle_min: 30, auto_uncertainty: true, watch_roots: [], upload_max_gb: 2, keep_job_logs_days: null, offline: false, notifications: false, basemap_url: null,
  };

  it("sends only changed keys and mirrors the thread-budget rule", () => {
    expect(settingsPatch(settings, { ...settings, heavy_slots: 2, watch_roots: ["/x"] })).toEqual({ heavy_slots: 2, watch_roots: ["/x"] });
    expect(settingsIssues(settings)).toEqual([]);
    expect(settingsIssues({ ...settings, threads_heavy: 4 })[0].path).toBe("threads_heavy");
  });

  it("saves through PUT /api/settings and shows the server's validation errors", async () => {
    let attempt = 0;
    fetchMock = mockFetch({
      "GET /api/settings": { body: settings },
      "PUT /api/settings": (_u, init) => {
        attempt += 1;
        const body = JSON.parse(String(init.body));
        if (attempt === 1) return { status: 422, body: { error: { code: "validation", message: "invalid settings", detail: { errors: [{ path: "watch_roots", message: "/nope is not a directory", code: "not_dir" }] } } } };
        return { body: { ...settings, ...body } };
      },
      "GET /api/system": { body: { cpu_count: 4, cpu_model: "test", mem_total_gb: 16, mem_available_gb: 8, disk_free_gb: 100, workspace: "/w", workspace_bytes: 1e9, host_id: "abc", versions: { python: "3.11", sparc: "1.0", numpy: "2", pandas: "2", torch: null, fastapi: "0.1" }, web_build: null } },
      "GET /api/storage": { body: { workspace_bytes: 1e9, free_bytes: 1e11, cache: [{ name: "ghcn_x.csv", bytes: 5e6, mtime: "2026-10-01T00:00:00Z" }], runs: [], studies: [], jobs_bytes: 1e6 } },
    });
    const { container } = render(<Settings />);
    const root = await waitFor(() => container.querySelector<HTMLInputElement>('input[aria-label="Folder to watch"]'), 5000, "form");
    typeInto(root, "/nope");
    click(byText(container, "button", "Add folder"));
    click(byText(container, "button", "Save settings"));
    await waitFor(() => container.textContent?.includes("/nope is not a directory"), 5000, "server error");
    expect(fetchMock.calls.find((c) => c.method === "PUT")!.body).toEqual({ watch_roots: ["/nope"] });
    expect(byText(container, "td", "ghcn_x.csv")).not.toBeNull();
  });
});

describe("storage manager", () => {
  it("asks again before deleting files of a run imported in place (force_files)", async () => {
    const run = { run_id: "20260901-090000-full-ee11", label: "CLI run", project_id: "p_demo", outputs_bytes: 9e8, checkpoint_bytes: 5e8 };
    fetchMock = mockFetch({
      "GET /api/settings": { body: { thread_budget: 4, threads_heavy: 3, engine_threads: 2, heavy_slots: 1, medium_slots: 1, network_slots: 2, engine_max_runs: 2, engine_mem_budget_gb: 6, engine_idle_min: 30, auto_uncertainty: true, watch_roots: [], upload_max_gb: 2, keep_job_logs_days: null, offline: false, notifications: false, basemap_url: null } },
      "GET /api/system": { body: { cpu_count: 4, cpu_model: "test", mem_total_gb: 16, mem_available_gb: 8, disk_free_gb: 100, workspace: "/w", workspace_bytes: 1e9, host_id: "abc", versions: { python: "3.11", sparc: "1.0", numpy: "2", pandas: "2", torch: null, fastapi: "0.1" }, web_build: null } },
      "GET /api/storage": { body: { workspace_bytes: 1e9, free_bytes: 1e11, cache: [], runs: [run], studies: [], jobs_bytes: 0 } },
      [`DELETE /api/runs/${run.run_id}?what=outputs`]: { status: 409, body: { error: { code: "imported_in_place", message: "this run was imported in place", detail: { run_dir: "/data/cli/run1" } } } },
      [`DELETE /api/runs/${run.run_id}?what=outputs&force_files=true`]: { body: { freed_bytes: 9e8 } },
    });
    const { container } = render(<Settings />);
    const btn = await waitFor(() => [...container.querySelectorAll("td button")].find((b) => b.textContent === "Outputs"), 5000, "storage row");
    click(btn);
    const del = await waitFor(() => [...document.querySelectorAll<HTMLButtonElement>(".dialog button")].find((b) => b.textContent === "Delete"), 5000, "dialog");
    click(del);
    await waitFor(() => document.querySelector(".dialog [role=alert]")?.textContent?.includes("/data/cli/run1"), 5000, "in-place warning");
    click([...document.querySelectorAll<HTMLButtonElement>(".dialog button")].find((b) => b.textContent === "Delete outside the workspace")!);
    await flush();
    expect(fetchMock.calls.filter((c) => c.method === "DELETE").map((c) => c.url)).toEqual([
      `/api/runs/${run.run_id}?what=outputs`,
      `/api/runs/${run.run_id}?what=outputs&force_files=true`,
    ]);
  });
});

describe("routes", () => {
  it("declares the tracking routes, the Overview and Runs nav entries and the Track tab", () => {
    const paths = registry.routes.map((r) => r.path);
    for (const p of ["/p/:pid", "/p/:pid/runs", "/runs", "/jobs", "/jobs/:jid", "/r/:rid/track", "/settings"]) expect(paths).toContain(p);
    const nav = Object.fromEntries(registry.projectNav.map((n) => [n.id, n]));
    expect(nav.overview.label).toBe("Overview");
    expect(nav.overview.to("p_1", project)).toBe("/p/p_1");
    expect(nav.runs.label).toBe("Runs");
    expect(nav.runs.to("p_1", project)).toBe("/p/p_1/runs");
    expect(nav.overview.order).toBeLessThan(nav.runs.order);
    const track = registry.runTabs.find((t) => t.id === "track")!;
    expect(track).toMatchObject({ label: "Track", group: "Run" });
    expect(registry.problems.filter((p) => p.includes("tracking"))).toEqual([]);
  });
});

