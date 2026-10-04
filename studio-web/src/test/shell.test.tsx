import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { App } from "../App";
import { clearResources } from "../api/resource";
import { StreamManager, setStreams, type EventSourceLike } from "../api/sse";
import type { RunSummary } from "../api/types";
import { ActiveRunChip } from "../layouts/AppShell";
import { navigate } from "../router";
import { useJobs } from "../stores/jobs";
import { useUi } from "../stores/ui";
import { byText, click, flush, key, mockFetch, render, waitFor } from "./render";

class QuietES implements EventSourceLike {
  static last: QuietES | null = null;
  onopen: ((ev: Event) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  url: string;
  listeners = new Map<string, (ev: MessageEvent) => void>();
  constructor(url: string) {
    this.url = url;
    QuietES.last = this;
  }
  addEventListener(t: string, l: (ev: MessageEvent) => void) {
    this.listeners.set(t, l);
  }
  close() {}
}

let restore: () => void = () => {};
beforeEach(() => {
  setStreams(new StreamManager({ EventSource: QuietES, frame: (cb) => cb() }));
  restore = mockFetch({
    "GET /api/jobs": { body: { items: [], next_cursor: null } },
    "GET /api/health": { body: { ok: true, version: "1.0.0", workspace: "/w", pid: 1, started_utc: "", active_jobs: 0, engine: { state: "ready" } } },
    "GET /api/projects": { body: [] },
  }).restore;
});
afterEach(() => {
  restore();
  setStreams(null);
  clearResources();
  useJobs.getState().reset();
  useUi.setState({ paletteOpen: false, toasts: [] });
  navigate("/", { replace: true });
});

describe("AppShell", () => {
  it("renders landmarks, the 404 route and live status for an unknown URL", async () => {
    navigate("/definitely/not/here");
    const { container } = render(<App />);
    // The 404 page is a lazy chunk: wait for it rather than for a fixed number of ticks.
    await waitFor(() => container.querySelector("h1"), 10000, "the 404 page");
    await flush(2);
    expect(container.querySelector("a.skip-link")!.getAttribute("href")).toBe("#main");
    expect(container.querySelector("main#main")).not.toBeNull();
    expect(container.querySelector('aside[aria-label="Sidebar"]')).not.toBeNull();
    expect(container.querySelector('nav[aria-label="Breadcrumb"]')).not.toBeNull();
    expect(container.querySelector("h1")!.textContent).toBe("Page not found");
    expect(container.textContent).toContain("/definitely/not/here");
    expect(document.title).toBe("Not found · SPARC Studio");
    expect(container.querySelector(".conn-pill")).not.toBeNull();
    expect(container.querySelector(".engine-dot")!.getAttribute("data-state")).toBe("ready");
    // every button has an accessible name
    for (const b of container.querySelectorAll("button")) expect((b.getAttribute("aria-label") ?? b.textContent ?? "").trim().length).toBeGreaterThan(0);
  });

  it("mirrors the most important running job in the tab title and the job tray", async () => {
    navigate("/nowhere");
    const { container } = render(<App />);
    await flush(4);
    const es = QuietES.last!;
    expect(es.url).toBe("/api/stream");
    es.onopen?.(new Event("open"));
    es.listeners.get("job.created")!(new MessageEvent("job.created", { data: JSON.stringify({ gseq: 1, ts: 1, job: { id: "j_1", kind: "run.core", lane: "heavy", label: "Full run", status: "running", created_utc: "2026-10-01T00:00:00Z", run_id: "r1" } }), lastEventId: "g:1" }));
    es.listeners.get("job.progress")!(new MessageEvent("job.progress", { data: JSON.stringify({ gseq: 2, ts: 2, job_id: "j_1", frac: 0.42, eta_s: 60, eta_lo: 50, eta_hi: 80, stage: "S2_S3", path_tail: [] }), lastEventId: "g:2" }));
    await flush(3);
    expect(document.title).toBe("▶ 42% S2_S3 · SPARC Studio");
    const tray = container.querySelector('.jobtray button')!;
    expect(tray.getAttribute("aria-label")).toBe("Jobs: 1 running, 0 queued");
    click(tray);
    expect(byText(container, ".jobtray-panel a", "Full run")!.getAttribute("href")).toBe("/jobs/j_1");
    expect(container.querySelector(".conn-pill")!.getAttribute("data-state")).toBe("live");
  });

  it("the run switcher tells same-config runs apart by mode, start time and R²", () => {
    const run = (id: string, extra: Partial<RunSummary>): RunSummary => ({
      id,
      project_id: "p_1",
      label: "synthetic_demo",
      origin: "studio",
      status: "complete",
      mode: "fast",
      coarse_m: null,
      created_utc: "2026-10-02T18:12:09Z",
      finished_utc: null,
      duration_s: null,
      n_points: 7,
      r2: 0.812,
      rmse: null,
      coverage: null,
      n_scenarios: null,
      checkpoint_bytes: null,
      has_emulator: false,
      studies: [],
      git_commit: null,
      git_dirty: null,
      demo: false,
      pinned: false,
      parent_run_id: null,
      study_id: null,
      last_job_id: null,
      ...extra,
    });
    const runs = [
      run("r_fast", {}),
      run("r_coarse", { mode: "coarse", coarse_m: 60, created_utc: "2026-10-02T18:40:00Z", r2: 0.774 }),
      // Same label, mode, minute, R² and status: only the id can tell it apart.
      run("r_twin", { created_utc: "2026-10-02T18:12:40Z" }),
    ];
    const { container } = render(<ActiveRunChip projectRuns={runs} run={null} activeRunId="r_fast" />);
    const texts = [...container.querySelectorAll('select[aria-label="Switch run"] option')].map((o) => o.textContent ?? "");
    expect(texts).toHaveLength(3);
    expect(new Set(texts).size).toBe(3);
    expect(texts[0]).toContain("FAST");
    expect(texts[0]).toContain("R² 0.812");
    expect(texts[1]).toContain("COARSE 60");
    expect(texts[1]).toContain("R² 0.774");
    expect(texts[0]).toMatch(/2026-10-02 \d\d:\d\d/);
    expect(texts[2]).toContain("r_twin");
    expect(texts[0]).toContain("r_fast");
    expect(texts[1]).not.toContain("r_coarse");
  });

  it("opens the command palette with Ctrl+K and toggles the theme", async () => {
    navigate("/nowhere");
    const { container } = render(<App />);
    await flush(3);
    const opener = container.querySelector<HTMLButtonElement>('button[aria-label^="Command palette"]')!;
    opener.focus();
    key(window as unknown as Element, "k", { ctrlKey: true });
    await flush(2);
    const dialog = document.querySelector('[role="dialog"][aria-label="Command palette"]');
    expect(dialog).not.toBeNull();
    expect(document.activeElement).toBe(dialog!.querySelector("input"));
    expect(byText(document.body, '[role="option"]', "Activity (jobs)")).not.toBeNull();
    key(dialog, "Escape");
    expect(document.querySelector('[aria-label="Command palette"]')).toBeNull();
    expect(document.activeElement).toBe(opener); // focus is restored
    const before = document.documentElement.getAttribute("data-theme");
    click(container.querySelector('button[aria-label^="Theme:"]'));
    const after = document.documentElement.getAttribute("data-theme");
    expect(after).not.toBe(before);
    useUi.getState().setTheme("system");
    expect(document.documentElement.hasAttribute("data-theme")).toBe(false);
  });
});
