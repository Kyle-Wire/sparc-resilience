// Lab pages against a mocked server: the Design workbench (preview, compile, the result
// inspector, and the PATCH 409 conflict_revision → fork flow continuing on the new
// revision), the across-runs dialog (estimate before Start), the plan budget slider rate
// limit, the library's configured deep link, and the route registry entries.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { clearResources } from "../../../api/resource";
import type { Job, Project } from "../../../api/types";
import { navigate, registry } from "../../../router";
import { useJobs } from "../../../stores/jobs";
import { byText, click, flush, mockFetch, render, typeInto, waitFor, type MockHandler } from "../../../test/render";
import { makeGrid } from "../../../map/grid";
import { AcrossRunsDialog } from "../components/Dialogs";
import { EngineChip } from "../components/EngineChip";
import { PLAN_PREVIEW_MS, PlanForm } from "../Plans";
import Design from "../Design";
import Library from "../Library";
import { forgetDrafts } from "../model/draft";
import { labDisabledReason, labRunOf, routes as labRoutes } from "../routes";
import { LEVERS, PID, RID, doc, gridMeta, previewBody, result, runRoutes, scenario } from "../__fixtures__/lab";

function job(id: string, kind: string): Job {
  return {
    id, kind, lane: "engine", executor: "engine", label: kind, status: "queued", project_id: PID, run_id: RID, study_id: null, scenario_id: null,
    parent_job_id: null, after_job_id: null, priority: 0, params: {}, created_utc: "2026-10-01T22:00:00Z", started_utc: null, finished_utc: null,
    progress: null, eta_s: null, eta_lo: null, eta_hi: null, stage: null, current_path: null, exit_code: null, error: null, blocked: null,
    result: null, peak_rss_mb: null, threads: null,
  };
}

let fetchMock: ReturnType<typeof mockFetch> | null = null;

beforeEach(() => {
  clearResources();
  forgetDrafts();
  useJobs.getState().reset();
});

afterEach(() => {
  fetchMock?.restore();
  fetchMock = null;
  vi.useRealTimers();
});

describe("route registry", () => {
  it("declares the Lab tab last in the Decisions group, and every Lab route", () => {
    const decisions = registry.runTabs.filter((t) => t.group === "Decisions");
    const lab = decisions.find((t) => t.id === "lab")!;
    expect(lab).toBeTruthy();
    expect(lab.order).toBeGreaterThan(40);
    expect(decisions.at(-1)!.id).toBe("lab");
    for (const p of ["/r/:rid/lab", "/r/:rid/lab/library", "/r/:rid/lab/s/:sid", "/r/:rid/lab/plans", "/r/:rid/lab/plans/:plid", "/r/:rid/lab/sweeps/:swid?", "/r/:rid/lab/climate", "/r/:rid/lab/compare"]) {
      expect(registry.routes.some((r) => r.path === p && labRoutes.includes(r)), p).toBe(true);
    }
    expect(labRoutes.every((r) => r.fullWidth)).toBe(true);
  });

  it("declares the Scenario Lab nav entry: the active run, greyed out without a run", () => {
    const nav = registry.projectNav.find((n) => n.id === "lab")!;
    expect(nav.label).toBe("Scenario Lab");
    const base = { id: "p_1", active_run_id: "r9", last_run: { id: "r9", status: "complete", created_utc: "", r2: 0.8 } } as Project;
    expect(nav.to("p_1", base)).toBe("/r/r9/lab");
    expect(labDisabledReason(base)).toBeNull();
    const none = { ...base, active_run_id: null, last_run: null } as Project;
    expect(labRunOf(none)).toBeNull();
    expect(labDisabledReason(none)).toMatch(/Launch a run first/);
    const running = { ...base, active_run_id: null, last_run: { id: "r1", status: "running", created_utc: "", r2: null } } as Project;
    expect(labDisabledReason(running)).toMatch(/still running/);
  });
});

describe("Design workbench", () => {
  it("previews, compiles, shows the exact result and forks on 409 conflict_revision", async () => {
    const sc = scenario("sc_a", {
      status: "exact",
      results: [result("res_1").summary],
    });
    let previews = 0;
    const patched: string[] = [];
    const routes: Record<string, MockHandler> = {
      ...runRoutes(),
      "GET /api/scenarios/sc_a": { body: sc },
      "GET /api/results/res_1": { body: result("res_1") },
      [`POST /api/runs/${RID}/preview`]: (_u, init) => {
        previews++;
        const seq = JSON.parse(String(init.body)).request_seq as number;
        return previewBody(seq, [-0.6, -0.5, -0.4, 0, 0, 0, -0.1], [1, 1, 1, 0, 0, 0, 0]);
      },
      [`POST /api/runs/${RID}/compile`]: {
        body: {
          content_hash: "h",
          portable: true,
          levers: { Pct_Canopy: { n_cells: 3, mean_requested: 10, total_requested: 30, predicted_mean_realised: 9.2, clipped_share: 0.08, est_cost: 27.6 } },
          union_cells: 3,
          people: 35,
          warnings: [{ code: "beyond_sd", message: "10 pp is within 1 sd", edit_index: 0, blocking: false }],
          est_exact_s: 2,
          emulator: { usable: true, hatched: false, reasons: [] },
        },
      },
      "PATCH /api/scenarios/sc_a": () => {
        patched.push("sc_a");
        return { status: 409, body: { error: { code: "conflict_revision", message: "This scenario has exact results; fork it to change its content." } } };
      },
      "POST /api/scenarios/sc_a/fork": (_u, init) => ({ status: 201, body: scenario("sc_b", { revision: 2, parent_id: "sc_a", doc: JSON.parse(String(init.body)).doc, updated_utc: "2026-10-01T12:00:00Z" }) }),
      "GET /api/scenarios/sc_b": () => ({ body: scenario("sc_b", { revision: 2, parent_id: "sc_a", doc: doc({ name: "Corridor v2" }), updated_utc: "2026-10-01T12:00:00Z" }) }),
      "PATCH /api/scenarios/sc_b": (_u, init) => {
        patched.push("sc_b");
        return { body: scenario("sc_b", { revision: 2, parent_id: "sc_a", doc: JSON.parse(String(init.body)).doc, updated_utc: "2026-10-01T12:00:00Z" }) };
      },
    };
    fetchMock = mockFetch(routes);
    navigate(`/r/${RID}/lab/s/sc_a`, { replace: true });
    const { container } = render(<Design />);
    const name = await waitFor(() => container.querySelector<HTMLInputElement>(".scenario-editor input"), 8000, "editor");
    expect(name.value).toBe("Downtown cool corridor");

    // live preview and compile
    await waitFor(() => previews > 0 && byText(container, ".map-side dd", "3"), 8000, "preview summary");
    await waitFor(() => byText(container, ".compile-panel dd", "35"), 8000, "compile people");
    expect(byText(container, ".edit-issues li", "within 1 sd")).not.toBeNull();
    // the inspector shows the plain-language card for the latest exact result
    await waitFor(() => byText(container, ".plain-result .headline", "Cools the edited area by 0.62 °F"), 8000, "inspector");
    expect(container.querySelector(".plain-result")!.textContent).toContain("Confident it cools.");
    expect(container.querySelector(".plain-result")!.textContent).toContain("partly outside observed conditions (23% of edited cells)");
    expect(byText(container, ".badge", "DEMO")).not.toBeNull();

    // edit the content: autosave on blur gets 409 conflict_revision, forks and continues
    typeInto(name, "Corridor v2");
    act(() => {
      name.dispatchEvent(new FocusEvent("focusout", { bubbles: true, relatedTarget: null }));
    });
    await waitFor(() => window.location.pathname === `/r/${RID}/lab/s/sc_b`, 8000, "navigated to the fork");
    expect(patched).toEqual(["sc_a"]);
    const forkCall = fetchMock.calls.find((c) => c.url === "/api/scenarios/sc_a/fork")!;
    expect((forkCall.body as { doc: { name: string } }).doc.name).toBe("Corridor v2");
    await flush(4);
    // editing continues on the new revision
    const name2 = container.querySelector<HTMLInputElement>(".scenario-editor input")!;
    expect(name2.value).toBe("Corridor v2");
    typeInto(name2, "Corridor v3");
    act(() => {
      name2.dispatchEvent(new FocusEvent("focusout", { bubbles: true, relatedTarget: null }));
    });
    await waitFor(() => patched.length === 2, 8000, "patch of the fork");
    expect(patched).toEqual(["sc_a", "sc_b"]);
    const last = fetchMock.calls.filter((c) => c.method === "PATCH").at(-1)!;
    expect(last.url).toBe("/api/scenarios/sc_b");
    expect((last.body as { doc: { name: string } }).doc.name).toBe("Corridor v3");
    expect(byText(container, ".save-status", "revision 2")).not.toBeNull();
  }, 30000);
});

describe("Engine chip", () => {
  it("offers the trust action naming the pickle risk when the checkpoint is untrusted, then opens the engine", async () => {
    let opens = 0;
    fetchMock = mockFetch({
      ...runRoutes(),
      [`POST /api/runs/${RID}/engine/open`]: () =>
        ++opens === 1
          ? { status: 409, body: { error: { code: "untrusted_pickle", message: "This run was imported without trusting its checkpoint." } } }
          : { status: 202, body: job("j_open", "engine.open") },
      "POST /api/runs/import": (_u, init) => ({ status: 201, body: { id: RID, ...JSON.parse(String(init.body)) } }),
    });
    const { container } = render(<EngineChip rid={RID} />);
    const open = await waitFor(() => byText(container, "button", "Open engine"), 5000, "open button");
    expect(container.textContent).toContain("engine cold");
    click(open);
    const dialog = await waitFor(() => document.querySelector('[role="dialog"]'), 5000, "trust dialog");
    expect(dialog.textContent).toContain("Checkpoint files execute code when they are loaded");
    expect(dialog.textContent).toContain(`/w/projects/demo/runs/${RID}`);
    click(byText(dialog, "button", "Trust and open the engine"));
    await waitFor(() => opens === 2, 5000, "second open");
    const imp = fetchMock.calls.find((c) => c.url === "/api/runs/import")!;
    expect(imp.body).toEqual({ dir: `/w/projects/demo/runs/${RID}`, project_id: PID, trust_pickles: true });
    expect(useJobs.getState().jobs.j_open).toBeTruthy();
    await waitFor(() => !document.querySelector('[role="dialog"]'), 5000, "dialog closed");
  });

  it("explains a memory refusal with the server's numbers and action", async () => {
    fetchMock = mockFetch({
      ...runRoutes(),
      [`POST /api/runs/${RID}/engine/open`]: {
        status: 409,
        body: { error: { code: "engine_memory", message: "Not enough memory", detail: { needed_gb: 3.2, available_gb: 1.5, holders: [{ run_id: "r_other", rss_gb: 2.1 }] }, action: { kind: "open_engine", label: "Evict r_other", method: "POST", path: "/api/engine/restart" } } },
      },
    });
    const { container } = render(<EngineChip rid={RID} />);
    click(await waitFor(() => byText(container, "button", "Open engine"), 5000, "open button"));
    const dialog = await waitFor(() => document.querySelector('[role="dialog"]'), 5000, "memory dialog");
    expect(dialog.textContent).toContain("3.2 GB");
    expect(dialog.textContent).toContain("1.5 GB");
    expect(dialog.textContent).toContain("run r_other");
    expect(byText(dialog, "button", "Evict r_other")).not.toBeNull();
  });
});

describe("Check across runs", () => {
  it("shows total_s and peak_rss_gb from the estimate endpoint before enabling Start", async () => {
    let release: (v: unknown) => void = () => {};
    const runs = [
      { id: "r_fast", label: "Fast", status: "complete", mode: "fast", coarse_m: null, checkpoint_bytes: 1e6 },
      { id: "r_full", label: "Full", status: "complete", mode: "full", coarse_m: null, checkpoint_bytes: 5e8 },
      { id: "r_old", label: "Old", status: "complete", mode: "full", coarse_m: null, checkpoint_bytes: null },
    ];
    fetchMock = mockFetch({
      [`GET /api/projects/${PID}/runs`]: { body: { items: runs, next_cursor: null } },
      "POST /api/scenarios/sc_a/across-runs/estimate": (_u, init) =>
        new Promise((res) => {
          release = () =>
            res({
              body: {
                runs: (JSON.parse(String(init.body)).run_ids as string[]).map((id) => ({ run_id: id, ok: id !== "r_old", reason: id === "r_old" ? "no checkpoint" : null, load_s: 45, exact_s: 13, rss_gb: 2.4 })),
                total_s: 116,
                peak_rss_gb: 2.4,
              },
            });
        }),
      "POST /api/scenarios/sc_a/across-runs": { status: 202, body: job("j_across", "scenario.across_runs") },
    });
    const onClose = vi.fn();
    const { container } = render(<AcrossRunsDialog sid="sc_a" pid={PID} rid={RID} onClose={onClose} />);
    const start = () => document.querySelector<HTMLButtonElement>('[data-testid="across-start"]')!;
    await waitFor(() => fetchMock!.calls.some((c) => c.url.endsWith("/estimate")), 5000, "estimate requested");
    expect(start().disabled).toBe(true); // no estimate yet
    expect(document.querySelector('[data-testid="across-total"]')).toBeNull();
    // the default choice is the runs with a checkpoint
    expect(fetchMock.calls.find((c) => c.url.endsWith("/estimate"))!.body).toEqual({ run_ids: ["r_fast", "r_full"] });
    release(null);
    await waitFor(() => document.querySelector('[data-testid="across-total"]'), 5000, "estimate shown");
    expect(document.querySelector('[data-testid="across-total"]')!.textContent).toMatch(/1 ?min 56 ?s|1:56|116/);
    expect(document.querySelector('[data-testid="across-peak"]')!.textContent).toBe("2.4 GB");
    expect(start().disabled).toBe(false);
    // choosing another run re-estimates and disables Start until the new estimate arrives
    const old = [...document.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')].find((i) => i.closest("label")?.textContent?.includes("Old"))!;
    click(old);
    await flush(2);
    expect(start().disabled).toBe(true);
    release(null);
    await waitFor(() => !start().disabled, 5000, "re-estimated");
    expect(byText(document.body, "[data-refused]", "no checkpoint")).not.toBeNull();
    click(start());
    await waitFor(() => onClose.mock.calls.length > 0, 5000, "started");
    expect(fetchMock.calls.at(-1)).toMatchObject({ method: "POST", url: "/api/scenarios/sc_a/across-runs", body: { run_ids: ["r_fast", "r_full"] } });
    expect(useJobs.getState().jobs.j_across).toBeTruthy();
    void container;
  });
});

describe("Plan budget slider", () => {
  it("sends at most one preview per 150 ms under rapid input, ending with the final budget", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] });
    const bodies: { budget: number }[] = [];
    const sentAt: number[] = [];
    fetchMock = mockFetch({
      ...runRoutes(),
      [`POST /api/runs/${RID}/plans/preview`]: (_u, init) => {
        bodies.push(JSON.parse(String(init.body)));
        sentAt.push(Date.now());
        return {
          body: { planned_total: 100, n_cells_treated: 3, mean_dose_treated: 10, total_cost: 30, gini: 0.2, min_dose_dropped_cost: 0, pareto: [{ budget: 20000, benefit: 100, n_cells: 3, n_segments: 3, gini: 0.2 }], dose: "AAAAAA==", constraint: "plantable", objective: "cooling", caption: "Plan" },
        };
      },
    });
    const g = makeGrid(gridMeta(), Int32Array.from([0, 1, 0, 2, 0, 1, 2]), Int32Array.from([2, 2, 1, 1, 0, 0, 0]));
    const { container } = render(<PlanForm rid={RID} grid={g} levers={LEVERS} onSaved={() => {}} />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(PLAN_PREVIEW_MS + 10);
    });
    const initial = bodies.length;
    expect(initial).toBe(1); // the first preview of the default params
    const slider = container.querySelector<HTMLInputElement>('.slider input[type="range"]')!;
    const DURATION = 1500;
    const STEP = 10;
    for (let t = 0, pos = 100; t < DURATION; t += STEP, pos += 5) {
      typeInto(slider, String(pos));
      await act(async () => {
        await vi.advanceTimersByTimeAsync(STEP);
      });
    }
    await act(async () => {
      await vi.advanceTimersByTimeAsync(PLAN_PREVIEW_MS * 2);
    });
    const during = bodies.length - initial;
    expect(during).toBeGreaterThanOrEqual(2); // live updates while dragging
    expect(during).toBeLessThanOrEqual(Math.ceil(DURATION / PLAN_PREVIEW_MS) + 1);
    // no two previews closer than 150 ms on the (fake) clock
    const gaps = sentAt.slice(1).map((t, i) => t - sentAt[i]);
    expect(Math.min(...gaps)).toBeGreaterThanOrEqual(PLAN_PREVIEW_MS);
    // the final request carries the final budget (the one the slider shows)
    const lastBudget = bodies.at(-1)!.budget;
    const shownBudget = container.querySelector(".slider output")!.textContent!.replace(/,/g, "");
    expect(Math.round(lastBudget)).toBe(Number(shownBudget));
  });
});

describe("Library", () => {
  it("focuses the configured scenario of ?configured=<slug>", async () => {
    const scrolled = vi.fn();
    Element.prototype.scrollIntoView = scrolled;
    fetchMock = mockFetch({
      ...runRoutes(),
      [`GET /api/projects/${PID}/scenarios`]: { body: [{ id: "sc_a", revision: 1, parent_id: null, status: "exact", created_utc: "2026-10-01T10:00:00Z", updated_utc: "2026-10-01T10:00:00Z", name: "Downtown cool corridor", tags: ["district"], latest: result("res_1").summary }] },
      [`GET /api/runs/${RID}/scenarios`]: {
        body: {
          configured: [
            { slug: "canopy-plus-10", name: "Canopy +10", city: { estimate: -0.2, se: 0.02, lo: -0.24, hi: -0.16, confidence: "confident_cools", phrase: "" }, p10: -0.3, p90: 0, frac_extrapolated: 0.1, mean_realized: {}, causal_linear: null, has_folds: true, layer_key: "sc:canopy-plus-10", doc: doc() },
            { slug: "cool-roofs", name: "Cool roofs", city: { estimate: -0.1, se: 0.05, lo: -0.2, hi: 0.0, confidence: "could_be_zero", phrase: "" }, p10: -0.2, p90: 0, frac_extrapolated: 0.4, mean_realized: {}, causal_linear: null, has_folds: false, layer_key: "sc:cool-roofs", doc: doc() },
          ],
          results: [],
        },
      },
    });
    navigate(`/r/${RID}/lab/library?configured=cool-roofs`, { replace: true });
    const { container } = render(<Library />);
    await waitFor(() => container.querySelector('[data-configured="cool-roofs"]'), 8000, "configured rows");
    const row = container.querySelector('[data-configured="cool-roofs"]')!.closest("tr")!;
    expect(row.classList.contains("hl")).toBe(true);
    expect(container.querySelector('[data-configured="canopy-plus-10"]')!.closest("tr")!.classList.contains("hl")).toBe(false);
    expect(scrolled).toHaveBeenCalled();
    // the configured scenario without fold detail offers the exact re-run; lineage lists the scenario
    expect(byText(row, "button", "Re-run exactly")).not.toBeNull();
    expect(byText(container, "a", "Downtown cool corridor")).not.toBeNull();
  });
});
