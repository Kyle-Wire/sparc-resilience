// Mission Control components: the fold × model heatmap from base_model task.end metrics, the
// stacker leaderboard from candidate_rmse metrics, Cancel / Force stop (fake timers), the stage
// rail chips (icon + text) and reduced motion.
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { clearResources } from "../../../api/resource";
import type { JobStreamHandle, StreamManager } from "../../../api/sse";
import type { getLogs, LogLine } from "../../../api/tracking";
import type { Job } from "../../../api/types";
import { navigate } from "../../../router";
import { useJobs } from "../../../stores/jobs";
import { LOG_KEEP, logLineFromEvent, replay, toSnapshot, type TrackerEntry, type TrackerState } from "../../../stores/tracker";
import { byText, click, flush, mockFetch, render, waitFor } from "../../../test/render";
import liveText from "../__fixtures__/s2s3.sample.jsonl?raw";
import sampleText from "../__fixtures__/events.sample.jsonl?raw";
import { FORCE_STOP_GRACE_S } from "../hooks";
import { railItems } from "../model";
import { JobTracker } from "../mission/JobTracker";
import { BottomTabs, loadLogLines, LOG_VIEW_MAX, resourceSeries } from "../mission/BottomTabs";
import { GanttPanel } from "../mission/GanttPanel";
import { ChildMatrix } from "../mission/ChildMatrix";
import { diagnostics } from "../mission/Header";
import { S2S3Panel, StackerLeaderboard } from "../mission/ModelPanels";
import { StageRail } from "../mission/StageRail";
import { makeJob, parseJsonl, type CursorEvent } from "./events";

const live = parseJsonl(liveText);
const sample = parseJsonl(sampleText);
const iMgwrRunning = live.findIndex((e) => e.type === "task.start" && e.name === "base_model" && e.key === "mgwr" && e.path.includes("task:fold[2/3]"));

function setReducedMotion(on: boolean) {
  window.matchMedia = ((query: string) =>
    ({
      matches: on && query.includes("prefers-reduced-motion"),
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }) as MediaQueryList) as typeof window.matchMedia;
}

const cellOf = (root: ParentNode, fold: string, model: string) =>
  [...root.querySelectorAll<HTMLTableCellElement>("table.foldgrid td")].find((td) => td.getAttribute("aria-label")?.startsWith(`${fold}, ${model}:`)) ?? null;

describe("fold × model heatmap", () => {
  beforeEach(() => setReducedMotion(false));

  it("renders seconds and held-out RMSE from base_model task.end metrics", () => {
    const state = replay(live.slice(0, iMgwrRunning + 1));
    const { container } = render(<S2S3Panel state={state} />);
    const head = [...container.querySelectorAll("table.foldgrid thead th")].map((th) => th.textContent);
    expect(head).toEqual(["Fold", "ols", "gam", "mgwr"]);
    expect(container.querySelectorAll("table.foldgrid tbody tr")).toHaveLength(3);
    const gam1 = cellOf(container, "Fold 1", "gam")!;
    expect(gam1.textContent).toContain("6 s");
    expect(gam1.textContent).toContain("0.610");
    expect(gam1.getAttribute("aria-label")).toBe("Fold 1, gam: fitted in 6 s, held-out RMSE 0.610, R² 0.74");
    expect(gam1.getAttribute("data-status")).toBe("ok");
    expect(gam1.style.background).not.toBe("");
    expect(cellOf(container, "Fold 1", "mgwr")!.textContent).toContain("0.550");
    // fold 2's mgwr fit has started and not ended: running, pulsing; fold 3 is pending
    const running = cellOf(container, "Fold 2", "mgwr")!;
    expect(running.getAttribute("data-status")).toBe("running");
    expect(running.getAttribute("data-pulse")).toBe("true");
    expect(running.textContent).toBe("running");
    expect(cellOf(container, "Fold 3", "ols")!.getAttribute("data-status")).toBe("pending");
    expect(container.querySelectorAll('td[data-pulse="true"]')).toHaveLength(1);
  });

  it("does not pulse the running cell when reduced motion is set", () => {
    setReducedMotion(true);
    const state = replay(live.slice(0, iMgwrRunning + 1));
    const { container } = render(<S2S3Panel state={state} />);
    const running = cellOf(container, "Fold 2", "mgwr")!;
    expect(running.getAttribute("data-status")).toBe("running");
    expect(running.hasAttribute("data-pulse")).toBe(false);
  });

  it("fills every cell once all folds are fitted", () => {
    const state = replay(live);
    const { container } = render(<S2S3Panel state={state} />);
    const cells = [...container.querySelectorAll("table.foldgrid td")];
    expect(cells).toHaveLength(9);
    expect(cells.every((td) => td.getAttribute("data-status") === "ok")).toBe(true);
    expect(cellOf(container, "Fold 3", "mgwr")!.getAttribute("aria-label")).toContain("held-out RMSE 0.520");
    // final skill against the 0.90 coverage target
    expect(byText(container, ".kpi", "90% interval coverage")?.textContent).toContain("87.4%");
  });
});

describe("stacker leaderboard", () => {
  it("ranks candidates by candidate_rmse and applies the 0.1% tie rule", () => {
    const state = replay(live);
    const { container } = render(<StackerLeaderboard state={state} />);
    const rows = [...container.querySelectorAll("tbody tr")].map((tr) => [...tr.querySelectorAll("td")].map((td) => td.textContent));
    expect(rows.map((r) => r[2])).toEqual(["0.5321", "0.5326", "0.5710"]);
    expect(rows[0][1]).toContain("neural residual (λ_PDE = 0.1)");
    // residual:0.1 is lowest but within 0.1% of NNLS, so the simpler NNLS blend wins
    const winner = container.querySelector('tr[data-winner="true"]')!;
    expect(winner.textContent).toContain("convex (NNLS) blend");
    expect(winner.textContent).toContain("winner");
    expect(container.textContent).toContain("within 0.1%");
  });

  it("is empty until candidates are scored", () => {
    const state = replay(live.slice(0, iMgwrRunning + 1));
    const { container } = render(<StackerLeaderboard state={state} />);
    expect(container.querySelector("table")).toBeNull();
    expect(container.textContent).toContain("scored after every fold");
  });
});

describe("stage rail chips", () => {
  it("show every stage's state with an icon and text, reasons as tooltips", () => {
    setReducedMotion(false);
    const s = replay(sample);
    const { container } = render(<StageRail items={railItems(s)} selected="S4" onSelect={() => {}} nowS={1790900200} />);
    const chips = [...container.querySelectorAll<HTMLButtonElement>("button.rail-chip")];
    expect(chips).toHaveLength(11);
    const texts = Object.fromEntries(chips.map((b) => [b.dataset.stage, b.querySelector(".status")!.textContent]));
    expect(texts).toMatchObject({ S0: "done", S1: "cached", S2_S3: "cached", baselines: "disabled", cv_curve: "done", climate: "disabled", S6: "skipped", S7: "skipped", finish: "done" });
    for (const b of chips) {
      const st = b.querySelector(".status")!;
      expect(st.querySelector("svg")).not.toBeNull();
      expect((st.textContent ?? "").trim().length).toBeGreaterThan(0);
    }
    const s2 = chips.find((b) => b.dataset.stage === "S2_S3")!;
    expect(s2.title).toContain("loaded from the checkpoint");
    expect(s2.textContent).toContain("◆");
    expect(chips.find((b) => b.dataset.stage === "S4")!.getAttribute("aria-pressed")).toBe("true");
    expect(chips.find((b) => b.dataset.stage === "baselines")!.title).toContain("disabled in the config (baselines.enabled)");
  });

  it("pulses a running chip only without reduced motion", () => {
    const s = replay(live);
    setReducedMotion(false);
    const a = render(<StageRail items={railItems(s)} />);
    const runA = a.container.querySelector('[data-stage="S4"] .status')!;
    expect(runA.textContent).toContain("running");
    expect(runA.getAttribute("data-pulse")).toBe("true");
    a.unmount();
    setReducedMotion(true);
    const b = render(<StageRail items={railItems(s)} />);
    const runB = b.container.querySelector('[data-stage="S4"] .status')!;
    expect(runB.textContent).toContain("running");
    expect(runB.hasAttribute("data-pulse")).toBe(false);
  });
});

describe("Mission Control actions", () => {
  let fetchMock: ReturnType<typeof mockFetch>;
  const opened: { jid: string; after: number | null }[] = [];
  const streams = {
    openJob: (jid: string, sub: { after?: number | null }): JobStreamHandle => {
      opened.push({ jid, after: sub.after ?? null });
      return { jid, close: () => {} };
    },
  } as unknown as StreamManager;

  function snapshotFor(job: Job, state: TrackerState) {
    return JSON.parse(JSON.stringify(toSnapshot(state, job)));
  }

  /** `GET /api/jobs/{jid}/events` over a log (after exclusive, `types` filter, `limit`). */
  const eventsRoute = (log: CursorEvent[]) => (url: URL) => {
    const a = url.searchParams.get("after");
    const after = a === null ? null : Number(a);
    const types = url.searchParams.get("types")?.split(",") ?? null;
    let rest = log.filter((e) => after === null || e.cursor > after);
    if (types) rest = rest.filter((e) => types.includes(e.type));
    const events = rest.slice(0, Number(url.searchParams.get("limit") ?? 1000));
    return { body: { events, next_cursor: events.length ? events[events.length - 1].cursor : after ?? -1, eof: events.length === rest.length } };
  };

  beforeEach(() => {
    setReducedMotion(false);
    clearResources();
    useJobs.getState().reset();
    opened.length = 0;
    navigate("/jobs/j_live", { replace: true });
  });

  afterEach(() => {
    vi.useRealTimers();
    fetchMock?.restore();
  });

  function mocks(job: Job, log: CursorEvent[]) {
    const state = replay(log);
    let current = job;
    fetchMock = mockFetch({
      "GET /api/jobs/j_live/tracker": () => ({ body: snapshotFor(current, state) }),
      "GET /api/jobs/j_live": () => ({ body: current }),
      "POST /api/jobs/j_live/cancel": () => {
        current = { ...current, status: "cancelling" };
        return { status: 202, body: current };
      },
      "POST /api/jobs/j_live/kill": () => {
        current = { ...current, status: "cancelled", finished_utc: "2026-10-01T12:30:00Z" };
        return { status: 202, body: current };
      },
      "GET /api/meta": { body: { warning_codes: [], output_catalog: [], job_kinds: [] } },
      "GET /api/runs/20261001-120000-coarse60-cd34": { body: { run: { id: "20261001-120000-coarse60-cd34", label: "City coarse", mode: "coarse", coarse_m: 60, demo: false, status: "running", project_id: "p_city" }, header: {} } },
      "GET /api/projects/p_city": { body: { project: { id: "p_city", name: "City" }, readiness: [], runs: [], active_jobs: [], config_version: 1 } },
      "GET /api/jobs/j_live/events": eventsRoute(log),
      "GET /api/jobs/j_live/logs": { body: { lines: [], next_cursor: -1 } },
    });
  }

  it("Cancel posts to /jobs/{jid}/cancel; Force stop appears only after 90 s in cancelling", async () => {
    vi.useFakeTimers({ toFake: ["Date", "setInterval", "clearInterval"], now: new Date("2026-10-01T12:20:00Z") });
    const state = replay(live);
    const job = makeJob({ id: "j_live", label: "Coarse run", project_id: "p_city", run_id: "20261001-120000-coarse60-cd34", status: "running", started_utc: "2026-10-01T12:00:00Z" });
    mocks(job, live);
    const { container } = render(<JobTracker jid="j_live" deps={{ streams }} />);
    await waitFor(() => byText(container, "button", "Cancel"), 5000, "Cancel button");
    // the job stream opens right after the snapshot's cursor
    expect(opened).toEqual([{ jid: "j_live", after: state.cursor }]);
    expect(byText(container, "button", "Force stop")).toBeNull();

    click(byText(container, "button", "Cancel"));
    await flush();
    expect(fetchMock.calls.filter((c) => c.method === "POST").map((c) => c.url)).toEqual(["/api/jobs/j_live/cancel"]);
    await waitFor(() => container.querySelector(".mc-title .status")?.textContent === "cancelling", 5000, "cancelling status");
    expect(byText(container, "button", "Cancel")).toBeNull();
    expect(byText(container, "button", "Force stop")).toBeNull();
    expect(container.textContent).toContain(`Force stop is offered in ${FORCE_STOP_GRACE_S} s`);

    act(() => void vi.advanceTimersByTime(89_000));
    expect(byText(container, "button", "Force stop")).toBeNull();
    expect(container.textContent).toContain("Force stop is offered in 1 s");

    act(() => void vi.advanceTimersByTime(1_000));
    const force = byText(container, "button", "Force stop");
    expect(force).not.toBeNull();
    click(force);
    const confirm = await waitFor(() => [...document.querySelectorAll<HTMLButtonElement>(".dialog button")].find((b) => b.textContent === "Force stop"), 5000, "confirm dialog");
    click(confirm);
    await flush();
    expect(fetchMock.calls.filter((c) => c.method === "POST").map((c) => c.url)).toEqual(["/api/jobs/j_live/cancel", "/api/jobs/j_live/kill"]);
  });

  it("offers to try again when the snapshot cannot be loaded", async () => {
    const state = replay(live);
    const job = makeJob({ id: "j_live", project_id: null, run_id: null, status: "running" });
    let fail = true;
    fetchMock = mockFetch({
      "GET /api/jobs/j_live/tracker": () => (fail ? { status: 500, body: { error: { code: "internal", message: "database is locked" } } } : { body: snapshotFor(job, state) }),
      "GET /api/meta": { body: { warning_codes: [], output_catalog: [], job_kinds: [] } },
      "GET /api/jobs/j_live/events": { body: { events: [], next_cursor: -1, eof: true } },
      "GET /api/jobs/j_live/logs": { body: { lines: [], next_cursor: -1 } },
    });
    const { container } = render(<JobTracker jid="j_live" deps={{ streams }} />);
    const again = await waitFor(() => byText(container, "button", "Try again"), 5000, "retry");
    expect(container.textContent).toContain("database is locked");
    fail = false;
    click(again);
    await waitFor(() => container.querySelector(".mc-title"), 5000, "header");
    expect(opened.map((o) => o.jid)).toEqual(["j_live"]);
  });

  it("uses the time of the cancel request from the event log for the grace period", async () => {
    vi.useFakeTimers({ toFake: ["Date", "setInterval", "clearInterval"], now: new Date((1790950000 + 400) * 1000) });
    // a cancel requested 95 s before "now": Force stop is offered at once
    const base = { v: 1, seq: 999, t_rel: 0, pid: 6000, job: "j_live", lvl: "info", span: null, parent: null, path: [], ctx: {}, cursor: 999999 };
    const log = [...live, { ...base, type: "cancel.requested", ts: 1790950000 + 305, by: "user" }] as CursorEvent[];
    mocks(makeJob({ id: "j_live", project_id: "p_city", run_id: "20261001-120000-coarse60-cd34", status: "cancelling" }), log);
    const { container } = render(<JobTracker jid="j_live" deps={{ streams }} />);
    await waitFor(() => byText(container, "button", "Force stop"), 5000, "Force stop");
    expect(byText(container, "button", "Cancel")).toBeNull();
  });
});

describe("Logs tab", () => {
  const line = (cursor: number): LogLine => ({ cursor, ts: 1790950000 + cursor, level: "info", logger: "sparc.core.pipeline", msg: `line ${cursor}`, path: [] });

  it("reads the whole log page after page and keeps the newest lines", async () => {
    const total = LOG_VIEW_MAX + 7_000;
    const seen: { after: number | null; level?: string; logger?: string }[] = [];
    const fetchLogs = (async (_jid: string, q: { after?: number | null; limit?: number; level?: string; logger?: string }) => {
      const after = q.after ?? null;
      seen.push({ after, level: q.level, logger: q.logger });
      const from = after === null ? 0 : after + 1;
      const lines = Array.from({ length: Math.max(0, Math.min(q.limit ?? 500, total - from)) }, (_, i) => line(from + i));
      return { lines, next_cursor: lines.length ? lines[lines.length - 1].cursor : total - 1 };
    }) as unknown as typeof getLogs;
    const r = await loadLogLines("j_live", { level: "info", logger: "sparc", stage: "", q: "" }, undefined, fetchLogs);
    expect(seen).toHaveLength(Math.ceil(total / 5000) + (total % 5000 === 0 ? 1 : 0));
    expect(seen[0]).toEqual({ after: null, level: "info", logger: "sparc" });
    expect(seen[1].after).toBe(4999);
    expect(r.truncated).toBe(true);
    expect(r.lines).toHaveLength(LOG_VIEW_MAX);
    expect(r.lines[r.lines.length - 1].cursor).toBe(total - 1);
    expect(r.lines[0].cursor).toBe(total - LOG_VIEW_MAX);
    const small = await loadLogLines("j_live", { level: "debug", logger: "", stage: "", q: "" }, undefined, (async () => ({ lines: [line(3), line(9)], next_cursor: 40 })) as unknown as typeof getLogs);
    expect(small).toEqual({ lines: [line(3), line(9)], next_cursor: 40, truncated: false });
  });
});

describe("study child matrix", () => {
  let fetchMockStudy: ReturnType<typeof mockFetch> | null = null;
  afterEach(() => {
    fetchMockStudy?.restore();
    fetchMockStudy = null;
  });

  it("shows multiverse variants, the scenario heatmap inputs and the priority-stability chart", async () => {
    clearResources();
    fetchMockStudy = mockFetch({
      "GET /api/studies/st_m/view": {
        body: {
          variants: [
            { name: "baseline", label: "Baseline", status: "done", r2: 0.71, rmse: 0.5, seconds: 600, run_id: "r_b" },
            { name: "coarse90", label: "Coarse 90 m", status: "done", r2: 0.66, rmse: 0.55, seconds: 300, run_id: "r_c" },
            { name: "no_physics", label: "No physics", status: "running", r2: null, rmse: null, seconds: null, run_id: null },
          ],
          effects: {},
          priority: {
            coarse90: { canopy: { kendall_tau: 0.82, top_decile_jaccard: 0.7 }, albedo: { kendall_tau: 0.64, top_decile_jaccard: 0.61 } },
            no_physics: { canopy: { kendall_tau: 0.4, top_decile_jaccard: 0.3 } },
          },
          stability: {},
        },
      },
    });
    const job = makeJob({ id: "j_mv", kind: "study.multiverse", study_id: "st_m", run_id: "r_b", params: { variants: ["baseline", "coarse90", "no_physics"] } });
    const { container } = render(<ChildMatrix job={job} state={replay([])} />);
    await waitFor(() => container.textContent?.includes("Priority stability by variant"), 5000, "stability chart");
    const rows = [...container.querySelectorAll('table[aria-label="Multiverse variants"] tbody tr')].map((tr) => tr.querySelector("td")!.textContent);
    expect(rows).toEqual(["Baseline", "Coarse 90 m", "No physics"]);
    expect(container.textContent).toContain("1 of 2 finished variants keep the baseline's priority map");
  });
});

describe("Copy diagnostics", () => {
  let fm: ReturnType<typeof mockFetch> | null = null;
  afterEach(() => {
    fm?.restore();
    fm = null;
  });

  it("holds the job, versions and the newest 200 events even when the byte window spans several pages", async () => {
    // 7,000 short lines (40 bytes each) inside the 200 × 4 KB window before the cursor
    const log = Array.from({ length: 7000 }, (_, i) => ({ v: 1, type: "tick", seq: i, ts: i, t_rel: i, pid: 1, job: "j_d", lvl: "debug", span: "1:1", parent: null, path: [], ctx: {}, k: i, n: 7000, unit: "x", frac: i / 7000, label: "", cursor: 40 * i })) as unknown as CursorEvent[];
    fm = mockFetch({
      "GET /api/jobs/j_d/events": (url: URL) => {
        const after = Number(url.searchParams.get("after"));
        const rest = log.filter((e) => e.cursor > after);
        const events = rest.slice(0, Number(url.searchParams.get("limit")));
        return { body: { events, next_cursor: events.length ? events[events.length - 1].cursor : after, eof: events.length === rest.length } };
      },
      "GET /api/system": { body: { versions: { python: "3.11", sparc: "1.0", numpy: "2", pandas: "2", torch: null, fastapi: "0.1" }, web_build: null, host_id: "h", cpu_count: 4, mem_total_gb: 16 } },
    });
    const state = { ...replay([]), cursor: log[log.length - 1].cursor };
    const text = await diagnostics(makeJob({ id: "j_d" }), state, []);
    const blob = JSON.parse(text) as { job: { id: string }; versions: { sparc: string }; events: { cursor: number }[] };
    expect(blob.job.id).toBe("j_d");
    expect(blob.versions.sparc).toBe("1.0");
    expect(blob.events).toHaveLength(200);
    expect(blob.events[0].cursor).toBe(log[6800].cursor);
    expect(blob.events[199].cursor).toBe(log[6999].cursor);
  });
});

describe("Gantt warning ticks", () => {
  let fm: ReturnType<typeof mockFetch> | null = null;
  afterEach(() => {
    fm?.restore();
    fm = null;
  });

  it("reads the warning times from the log once and adds the streamed ones", async () => {
    clearResources();
    const warnings = sample.filter((e) => e.type === "warning");
    expect(warnings).toHaveLength(4);
    // the page opened before the last warning: the log on disk holds the first three
    const iLast = sample.indexOf(warnings[3]);
    const onDisk = sample.slice(0, iLast);
    fm = mockFetch({
      "GET /api/jobs/j_g/events": (url: URL) => {
        expect(url.searchParams.get("types")).toBe("warning");
        const events = onDisk.filter((e) => e.type === "warning");
        return { body: { events, next_cursor: onDisk[onDisk.length - 1].cursor, eof: true } };
      },
    });
    const ticks = (root: ParentNode) => [...root.querySelectorAll('[aria-label^="Warning at"]')].map((el) => el.getAttribute("aria-label"));
    const before = replay(onDisk);
    const { container, rerender } = render(<GanttPanel jid="j_g" state={before} nowS={1790900100} zoom={null} logs={[]} />);
    await waitFor(() => ticks(container).length === 3 || null, 5000, "three warning ticks");
    // the last warning streams in: it is drawn without reading the log again
    const after = replay(sample.slice(0, iLast + 1));
    const streamed = [logLineFromEvent(warnings[1])!, logLineFromEvent(warnings[3])!];
    rerender(<GanttPanel jid="j_g" state={after} nowS={1790900100} zoom={null} logs={streamed} />);
    await flush();
    const all = ticks(container);
    expect(all).toHaveLength(4);
    expect(all[3]).toContain("log.sparc.core.base_models: GAM smoothing reached the upper bound on wide blocks");
    expect(fm.calls.filter((c) => c.url.startsWith("/api/jobs/j_g/events"))).toHaveLength(1);
  });
});

describe("Resources tab", () => {
  let fm: ReturnType<typeof mockFetch> | null = null;
  afterEach(() => {
    fm?.restore();
    fm = null;
    navigate("/", { replace: true });
  });

  const sample10 = (ts: number, rss: number) => ({ ts, rss_mb: rss, cpu_pct: 250, n_procs: 3, threads: 3 });
  const entryFor = (job: Job, resources: ReturnType<typeof sample10>[]): TrackerEntry => ({
    jid: job.id, phase: "ended", error: null, job, state: replay(sample), reconstructed: false, replaying: false, resources, recent: [], ended: job.status, epochs: [], logs: [], logCapped: false,
  });
  const system = { cpu_count: 4, cpu_model: "test", mem_total_gb: 16, mem_available_gb: 0.5, disk_free_gb: 100, workspace: "/w", workspace_bytes: 1e9, host_id: "abc", versions: { python: "3.11", sparc: "1.0", numpy: "2", pandas: "2", torch: null, fastapi: "0.1" }, web_build: null };

  it("puts the stored history before the tracked samples", () => {
    const hist = [sample10(100, 500), sample10(110, 600), sample10(120, 700)];
    expect(resourceSeries(hist, [sample10(115, 650), sample10(117, 660)]).map((r) => r.ts)).toEqual([100, 110, 115, 117]);
    expect(resourceSeries(undefined, [sample10(5, 1)]).map((r) => r.ts)).toEqual([5]);
    expect(resourceSeries(hist, []).map((r) => r.ts)).toEqual([100, 110, 120]);
  });

  it("shows a finished job's samples since it started, without the live memory banner", async () => {
    clearResources();
    const job = makeJob({ id: "j_r", status: "succeeded", started_utc: "2026-10-01T10:15:01Z", finished_utc: "2026-10-01T11:15:01Z", peak_rss_mb: 2048 });
    const started = Date.parse("2026-10-01T10:15:01Z") / 1000;
    fm = mockFetch({
      "GET /api/jobs/j_r/resources": (url: URL) => {
        expect(Number(url.searchParams.get("since_ts"))).toBe(started);
        return { body: [sample10(started + 10, 900), sample10(started + 20, 1800), sample10(started + 30, 4000)] };
      },
      "GET /api/system": { body: system },
    });
    navigate("/jobs/j_r?tab=resources", { replace: true });
    const { container } = render(<BottomTabs entry={entryFor(job, [])} job={job} meta={undefined} />);
    await waitFor(() => container.querySelector(".kpi"), 5000, "resource KPIs");
    expect(container.textContent).toContain("peak 3.91 GB");
    expect(container.textContent).toContain("Memory (RSS), last sample");
    // the stored samples are drawn (the snapshot held none: the job ended long ago)
    expect(container.querySelector('svg[aria-label^="Memory (RSS)"]')!.getAttribute("aria-label")).toBe("Memory (RSS): latest 4,000 MB, range 900 to 4,000");
    // 4 GB is above 80% of the 0.5 GB free now plus its own RSS, but the job has ended
    expect(container.querySelector('[role="alert"]')).toBeNull();
  });

  it("keeps the memory banner for a live job", async () => {
    clearResources();
    const job = makeJob({ id: "j_r2", status: "running", started_utc: "2026-10-01T10:15:01Z" });
    fm = mockFetch({ "GET /api/jobs/j_r2/resources": { body: [] }, "GET /api/system": { body: system } });
    navigate("/jobs/j_r2?tab=resources", { replace: true });
    const { container } = render(<BottomTabs entry={{ ...entryFor(job, [sample10(1790950000, 4000)]), phase: "live", ended: null }} job={job} meta={undefined} />);
    await waitFor(() => container.querySelector('[role="alert"]'), 5000, "memory banner");
    expect(container.querySelector('[role="alert"]')!.textContent).toContain("above 80%");
  });
});

describe("Logs tab live lines", () => {
  let fm: ReturnType<typeof mockFetch> | null = null;
  afterEach(() => {
    fm?.restore();
    fm = null;
    navigate("/", { replace: true });
  });

  it("reads the log again once more live lines arrived than the entry keeps", async () => {
    clearResources();
    const line = (cursor: number): LogLine => ({ cursor, ts: 1790950000 + cursor / 100, level: "info", logger: "sparc.core.pipeline", msg: `line ${cursor}`, path: [] });
    let reads = 0;
    fm = mockFetch({
      "GET /api/jobs/j_l/logs": () => {
        reads += 1;
        // the first read reached cursor 100; the second one reaches the end of the streamed lines
        return { body: reads === 1 ? { lines: [line(50), line(100)], next_cursor: 100 } : { lines: [line(50), line(100), line(101)], next_cursor: 200 + LOG_KEEP } };
      },
    });
    const job = makeJob({ id: "j_l", status: "running" });
    const entry = (logs: LogLine[]): TrackerEntry => ({ jid: "j_l", phase: "live", error: null, job, state: replay([]), reconstructed: false, replaying: false, resources: [], recent: [], ended: null, epochs: [], logs, logCapped: false });
    navigate("/jobs/j_l?lvl=debug", { replace: true });
    const few = [line(150), line(160)];
    const { container, rerender } = render(<BottomTabs entry={entry(few)} job={job} meta={undefined} />);
    await waitFor(() => (container.querySelector('[aria-label="4 log lines"]') ? true : null), 5000, "server + live lines");
    expect(reads).toBe(1);
    // LOG_KEEP newer lines: the entry dropped lines 101…200, so the tab reads the log again
    const many = Array.from({ length: LOG_KEEP }, (_, i) => line(201 + i));
    rerender(<BottomTabs entry={entry(many)} job={job} meta={undefined} />);
    await waitFor(() => (reads === 2 ? true : null), 5000, "second read");
    await flush();
    expect(reads).toBe(2);
  });
});
