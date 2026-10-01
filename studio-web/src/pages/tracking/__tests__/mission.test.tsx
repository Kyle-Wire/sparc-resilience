// Mission Control components: the fold × model heatmap from base_model task.end metrics, the
// stacker leaderboard from candidate_rmse metrics, Cancel / Force stop (fake timers), the stage
// rail chips (icon + text) and reduced motion.
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { clearResources } from "../../../api/resource";
import type { JobStreamHandle, StreamManager } from "../../../api/sse";
import type { Job } from "../../../api/types";
import { navigate } from "../../../router";
import { useJobs } from "../../../stores/jobs";
import { replay, toSnapshot, type TrackerState } from "../../../stores/tracker";
import { byText, click, flush, mockFetch, render, waitFor } from "../../../test/render";
import liveText from "../__fixtures__/s2s3.sample.jsonl?raw";
import sampleText from "../__fixtures__/events.sample.jsonl?raw";
import { FORCE_STOP_GRACE_S } from "../hooks";
import { railItems } from "../model";
import { JobTracker } from "../mission/JobTracker";
import { S2S3Panel, StackerLeaderboard } from "../mission/ModelPanels";
import { StageRail } from "../mission/StageRail";
import { makeJob, parseJsonl } from "./events";

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

  function mocks(job: Job, state: TrackerState) {
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
      "GET /api/jobs/j_live/events": { body: { events: [], next_cursor: -1, eof: true } },
      "GET /api/jobs/j_live/logs": { body: { lines: [], next_cursor: -1 } },
    });
  }

  it("Cancel posts to /jobs/{jid}/cancel; Force stop appears only after 90 s in cancelling", async () => {
    vi.useFakeTimers({ toFake: ["Date", "setInterval", "clearInterval"], now: new Date("2026-10-01T12:20:00Z") });
    const state = replay(live);
    const job = makeJob({ id: "j_live", label: "Coarse run", project_id: "p_city", run_id: "20261001-120000-coarse60-cd34", status: "running", started_utc: "2026-10-01T12:00:00Z" });
    mocks(job, state);
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
    const state = replay([...live, { ...base, type: "cancel.requested", ts: 1790950000 + 305, by: "user" }]);
    mocks(makeJob({ id: "j_live", project_id: "p_city", run_id: "20261001-120000-coarse60-cd34", status: "cancelling" }), state);
    const { container } = render(<JobTracker jid="j_live" deps={{ streams }} />);
    await waitFor(() => byText(container, "button", "Force stop"), 5000, "Force stop");
    expect(byText(container, "button", "Cancel")).toBeNull();
  });
});
