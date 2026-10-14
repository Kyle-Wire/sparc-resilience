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
import { viewFixture } from "../../runhub/__fixtures__/views";

function job(id: string, kind: string): Job {
  return {
    id, kind, lane: "engine", executor: "engine", label: kind, status: "queued", project_id: PID, run_id: RID, study_id: null, scenario_id: null,
    parent_job_id: null, after_job_id: null, priority: 0, params: {}, created_utc: "2026-10-01T22:00:00Z", started_utc: null, finished_utc: null,
    progress: null, eta_s: null, eta_lo: null, eta_hi: null, stage: null, current_path: null, exit_code: null, error: null, blocked: null,
    result: null, peak_rss_mb: null, threads: null,
  };
}

let fetchMock: ReturnType<typeof mockFetch> | null = null;

/** Open the Design page's "Advanced options" (it renders the full scenario editor only while open; it stays open across scenarios). */
async function openAdvanced(container: HTMLElement) {
  const details = await waitFor(() => container.querySelector<HTMLDetailsElement>("details.lab-advanced"), 8000, "advanced options");
  if (!details.open) click(details.querySelector("summary")!);
  return details;
}

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
  it("declares the Scenario Lab tab between Heat risk and Trust, covering the lab, scenarios and budget views, and every Lab route", () => {
    const ids = registry.runTabs.map((t) => t.id);
    const lab = registry.runTabs.find((t) => t.id === "lab")!;
    expect(lab.label).toBe("Scenario Lab");
    expect(lab.views).toEqual(["lab", "scenarios", "budget"]);
    expect(ids.indexOf("lab")).toBe(ids.indexOf("heat") + 1);
    expect(ids.indexOf("trust")).toBe(ids.indexOf("lab") + 1);
    for (const p of ["/r/:rid/lab", "/r/:rid/lab/library", "/r/:rid/lab/s/:sid", "/r/:rid/lab/plans", "/r/:rid/lab/plans/:plid", "/r/:rid/lab/sweeps/:swid?", "/r/:rid/lab/compare"]) {
      expect(registry.routes.some((r) => r.path === p && labRoutes.includes(r)), p).toBe(true);
    }
    // Climate × adaptation moved to Heat risk › Future climate (a legacy redirect now)
    expect(labRoutes.some((r) => r.path === "/r/:rid/lab/climate")).toBe(false);
    expect(labRoutes.every((r) => r.fullWidth)).toBe(true);
  });

  it("declares the Scenario Lab nav entry: the active run, greyed out only without a run", () => {
    const nav = registry.projectNav.find((n) => n.id === "lab")!;
    expect(nav.label).toBe("Scenario Lab");
    const base = { id: "p_1", active_run_id: "r9", last_run: { id: "r9", status: "complete", created_utc: "", r2: 0.8 } } as Project;
    expect(nav.to("p_1", base)).toBe("/r/r9/lab");
    expect(labDisabledReason(base)).toBeNull();
    const none = { ...base, active_run_id: null, last_run: null } as Project;
    expect(labRunOf(none)).toBeNull();
    expect(labDisabledReason(none)).toMatch(/Launch a run first/);
    // Any run opens the Lab (Library, Plans and Compare work without a checkpoint; Design's engine
    // line says when it has none): running, without a checkpoint, failed.
    const running = { ...base, active_run_id: null, last_run: { id: "r1", status: "running", created_utc: "", r2: null } } as Project;
    expect(labDisabledReason(running)).toBeNull();
    expect(nav.to("p_1", running)).toBe("/r/r1/lab");
    const noCheckpoint = { ...base, last_run: { id: "r9", status: "complete", created_utc: "", r2: 0.8, has_checkpoint: false } } as Project;
    expect(labDisabledReason(noCheckpoint)).toBeNull();
    const failed = { ...base, active_run_id: null, last_run: { id: "r2", status: "failed", created_utc: "", r2: null, has_checkpoint: false } } as Project;
    expect(labDisabledReason(failed)).toBeNull();
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
    await openAdvanced(container);
    const name = await waitFor(() => container.querySelector<HTMLInputElement>(".scenario-editor input"), 8000, "editor");
    expect(name.value).toBe("Downtown cool corridor");

    // live preview (the quick estimate pane) and compile (Advanced options)
    await waitFor(() => previews > 0 && container.querySelector('[aria-label="Preview"] [data-n-edited="3"]'), 8000, "preview summary");
    await waitFor(() => byText(container, ".compile-panel dd", "35"), 8000, "compile people");
    expect(byText(container, ".edit-issues li", "within 1 sd")).not.toBeNull();
    // the inspector shows the plain-language card for the latest exact result
    await waitFor(() => byText(container, ".plain-result .headline", "Cools the places you changed"), 8000, "inspector");
    expect(container.querySelector(".plain-result")!.textContent).toContain("Confident it cools.");
    expect(container.querySelector(".plain-result")!.textContent).toContain("partly beyond the conditions measured in this city (23% of the places you changed)");
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

  it("previews, compiles and can run a stored 'Shade the hottest X%' scenario (top selection echoed with k: null, within: null)", async () => {
    // GET /api/scenarios/{sid} after the template: pydantic echoes unset optional fields as null.
    const where = { kind: "top", column: "pred:target", frac: 0.1, k: null, direction: "highest", within: null };
    const edit = { lever: "Pct_Canopy", mode: "add", amount: 10, percentile: null, paved_share: null, per_cell_ref: null, where, label: null };
    const sc = scenario("sc_t", { doc: doc({ name: "Shade the hottest 10%", anchor_run_id: null, edits: [edit as never] }) });
    const sent: { preview: unknown[]; compile: unknown[] } = { preview: [], compile: [] };
    fetchMock = mockFetch({
      ...runRoutes(),
      "GET /api/scenarios/sc_t": { body: sc },
      [`POST /api/runs/${RID}/preview`]: (_u, init) => {
        const body = JSON.parse(String(init.body));
        sent.preview.push(body.edits);
        return previewBody(body.request_seq as number, [-0.6, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]);
      },
      [`POST /api/runs/${RID}/compile`]: (_u, init) => {
        sent.compile.push(JSON.parse(String(init.body)).scenario.edits);
        return { body: { content_hash: "h", portable: true, levers: {}, union_cells: 1, people: 12, warnings: [], est_exact_s: 2, emulator: { usable: true, hatched: false, reasons: [] } } };
      },
    });
    navigate(`/r/${RID}/lab/s/sc_t`, { replace: true });
    const { container } = render(<Design />);
    await openAdvanced(container);
    await waitFor(() => container.querySelector<HTMLInputElement>(".scenario-editor input")?.value === "Shade the hottest 10%", 8000, "editor");
    await waitFor(() => sent.preview.length > 0, 8000, "preview request");
    await waitFor(() => byText(container, ".compile-panel dd", "12"), 8000, "compile people");
    expect((sent.preview.at(-1) as { where: unknown }[])[0].where).toEqual(where);
    expect((sent.compile.at(-1) as unknown[]).length).toBe(1);
    expect(container.textContent).not.toContain("not a selection");
    expect(container.textContent).not.toContain("Add an edit to compile it");
    const run = byText(container, "button", "Run full model") as HTMLButtonElement;
    expect(run.disabled).toBe(false);
  }, 30000);
});

describe("Design step flow (layout contract)", () => {
  const compileBody = {
    content_hash: "h",
    portable: true,
    levers: { Pct_Canopy: { n_cells: 3, mean_requested: 10, total_requested: 30, predicted_mean_realised: 9.2, clipped_share: 0.08, est_cost: 27.6 } },
    union_cells: 3,
    people: 35,
    warnings: [],
    est_exact_s: 120,
    emulator: { usable: true, hatched: false, reasons: [] },
  };
  const routes = (): Record<string, MockHandler> => ({
    ...runRoutes(),
    "GET /api/scenarios/sc_a": { body: scenario("sc_a") },
    [`POST /api/runs/${RID}/preview`]: (_u, init) => previewBody(JSON.parse(String(init.body)).request_seq as number, [-0.6, -0.5, -0.4, 0, 0, 0, -0.1], [1, 1, 1, 0, 0, 0, 0]),
    [`POST /api/runs/${RID}/compile`]: { body: compileBody },
  });
  async function closeAdvanced(container: HTMLElement) {
    const details = await waitFor(() => container.querySelector<HTMLDetailsElement>("details.lab-advanced"), 8000, "advanced options");
    if (details.open) click(details.querySelector("summary")!);
    await flush(2);
  }

  it("shows the active change's three steps, one quick-estimate region and one Run full model button; ids never clash with Advanced options", async () => {
    fetchMock = mockFetch(routes());
    navigate(`/r/${RID}/lab/s/sc_a`, { replace: true });
    const { container } = render(<Design />);
    await closeAdvanced(container);
    expect(container.querySelector(".scenario-editor")).toBeNull(); // Advanced options renders only while open
    const row = await waitFor(() => container.querySelector("article.edit-row[data-active]"), 8000, "active change");
    expect(row.querySelectorAll(".stage-num").length).toBe(3); // 01 02 03
    expect(row.querySelector("#edit-0-amount")).not.toBeNull();
    expect(row.querySelector('[data-lever="Pct_Canopy"]')!.getAttribute("aria-pressed")).toBe("true");
    // the typical amounts are chips only: no slider ticks bunched at its low end
    expect(row.querySelector("datalist")).toBeNull();
    expect(row.querySelector('input[type="range"]')!.hasAttribute("list")).toBe(false);
    await waitFor(() => container.querySelector('[aria-label="Preview"] [data-n-edited="3"]'), 8000, "quick estimate");
    expect(container.querySelectorAll('[aria-label="Preview"]').length).toBe(1);
    await waitFor(() => container.querySelector("[data-compile-sentence]"), 8000, "compile sentence");
    const runs = [...container.querySelectorAll("button")].filter((b) => b.textContent?.startsWith("Run full model"));
    expect(runs.length).toBe(1);
    expect(runs[0].classList.contains("accent")).toBe(true);
    expect(runs[0].textContent).toContain("2 min");
    // opening Advanced options hands the edit-* ids to the scenario editor: still exactly one #edit-0-amount
    await openAdvanced(container);
    await waitFor(() => container.querySelector(".scenario-editor"), 8000, "editor");
    expect(container.querySelectorAll("#edit-0-amount").length).toBe(1);
    expect(container.querySelector("article.edit-row[data-active] #change-0-amount")).not.toBeNull();
    await closeAdvanced(container);
    // Done collapses the change to its sentence; "Add another change" opens a new one
    click(byText(container.querySelector("article.edit-row[data-active]")!, "button", "Done"));
    expect(container.querySelector("article.edit-row[data-active]")).toBeNull();
    expect(container.querySelector("article.edit-row .lab-edit-sentence")).not.toBeNull();
    click(byText(container, "button", "Add another change"));
    expect(container.querySelector('article.edit-row[data-pending="true"]')).not.toBeNull();
  }, 30000);

  it("starts a new scenario from a lever card", async () => {
    fetchMock = mockFetch(routes());
    navigate(`/r/${RID}/lab`, { replace: true });
    const { container } = render(<Design />);
    await closeAdvanced(container);
    const pending = await waitFor(() => container.querySelector('article.edit-row[data-pending="true"]'), 8000, "first change");
    expect(pending.querySelector("#edit-0-amount")).toBeNull(); // nothing to size before a lever is picked
    click(pending.querySelector('[data-lever="Albedo"]'));
    const row = await waitFor(() => container.querySelector("article.edit-row[data-active]:not([data-pending])"), 8000, "created change");
    expect(row.querySelector('[data-lever="Albedo"]')!.getAttribute("aria-pressed")).toBe("true");
    expect(row.querySelector("#edit-0-amount")).not.toBeNull();
  }, 30000);

  it("walks the whole flow: empty → lever → where → how much → quick estimate → Run full model → the answer card", async () => {
    let saved: ReturnType<typeof scenario> | null = null;
    let ran = 0;
    const base = result("res_9", { scenario: { id: "sc_new", revision: 1, name: "Downtown cool corridor" } });
    const exactSummary = { ...base.summary, scenario_id: "sc_new", created_utc: "2026-10-02T09:00:00Z" };
    const res = { ...base, summary: exactSummary };
    fetchMock = mockFetch({
      ...routes(),
      [`POST /api/projects/${PID}/scenarios`]: (_u, init) => {
        saved = scenario("sc_new", { doc: JSON.parse(String(init.body)).doc });
        return { status: 201, body: saved };
      },
      "GET /api/scenarios/sc_new": () => ({ body: ran ? { ...saved!, status: "exact", results: [exactSummary] } : saved! }),
      "PATCH /api/scenarios/sc_new": (_u, init) => {
        saved = { ...saved!, doc: JSON.parse(String(init.body)).doc };
        return { body: saved };
      },
      "POST /api/scenarios/sc_new/run": () => {
        ran++;
        return { body: { cached: exactSummary, job: null } };
      },
      "GET /api/results/res_9": { body: res },
    });
    // the newly finished result is scrolled into view (jsdom has no scrollIntoView: record the calls)
    const scrolled: string[] = [];
    const scrollIntoView = vi.fn(function (this: HTMLElement) {
      scrolled.push(this.className);
    });
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { value: scrollIntoView, configurable: true, writable: true });
    navigate(`/r/${RID}/lab`, { replace: true });
    const { container } = render(<Design />);
    await closeAdvanced(container);
    const mapHeading = () => container.querySelector(".lab-map .map-side .card h3")?.textContent ?? null;

    // empty: the examples and a disabled Run full model that says why; the map shows where the city is hot
    const pending = await waitFor(() => container.querySelector('article.edit-row[data-pending="true"]'), 8000, "first change");
    const run0 = byText(container, "button", "Run full model") as HTMLButtonElement;
    expect(run0.disabled).toBe(true);
    expect(container.querySelector("[data-run-why]")!.textContent).toMatch(/Make a change first/);
    await waitFor(() => mapHeading() === "Observed temperature", 8000, "the observed temperature on the map");
    expect(container.querySelector(".compare-tray")).toBeNull(); // nothing pinned: no tray bar
    expect(container.querySelector("[data-full-result]")).toBeNull();

    // 01 what: trees & canopy
    click(pending.querySelector('[data-lever="Pct_Canopy"]'));
    const row = await waitFor(() => container.querySelector("article.edit-row[data-active]:not([data-pending])"), 8000, "created change");
    // 02 where: a neighbourhood (zone 1 first)
    click(byText(row.querySelector('[data-step="where"]')!, '[role="radio"]', "A neighbourhood"));
    await waitFor(() => row.querySelector('[data-step="where"][data-preset="neighbourhood"]'), 8000, "neighbourhood preset");
    // 03 how much: a typical amount
    click(byText(row.querySelector('[data-step="howmuch"]')!, "button.chip", "20 pts"));
    expect((row.querySelector("#edit-0-amount") as HTMLInputElement).value).toBe("20");

    // the first autosave creates the scenario and opens it; its doc is what the steps built
    await waitFor(() => window.location.pathname === `/r/${RID}/lab/s/sc_new`, 8000, "first autosave");
    expect(saved!.doc.edits).toEqual([expect.objectContaining({ lever: "Pct_Canopy", mode: "add", amount: 20, where: { kind: "zones", values: [1] } })]);

    // the quick estimate and the "how much fits" sentence
    const pv = await waitFor(() => container.querySelector('[aria-label="Preview"][data-trust]'), 8000, "quick estimate");
    await waitFor(() => pv.querySelector('[data-n-edited="3"]') && pv.querySelector("[data-edited-mean]"), 8000, "preview summary");
    expect(pv.querySelector(".qe-big")!.textContent).toBe("About −0.50 °F");
    // the map switches to the quick estimate with the first change
    await waitFor(() => mapHeading() === "Quick estimate: temperature change", 8000, "the quick estimate on the map");
    await waitFor(() => container.querySelector("[data-compile-sentence]"), 8000, "compile sentence");

    // Run full model → the answer card (a cached result here)
    const run = await waitFor(() => {
      const b = byText(container, "button", "Run full model") as HTMLButtonElement | null;
      return b && !b.disabled ? b : null;
    }, 8000, "run enabled");
    click(run);
    await waitFor(() => ran === 1, 8000, "exact run requested");
    const headline = await waitFor(() => container.querySelector(".result-inspector .plain-result .headline"), 8000, "answer card");
    expect(headline.textContent).toBe("Cools the places you changed");
    // it comes into view; with nothing new to run, the run pane points to it (no accent, no second number)
    await waitFor(() => scrolled.some((c) => c.includes("answer-card")), 8000, "the answer card scrolled into view");
    const line = await waitFor(() => container.querySelector('[data-full-result="current"]'), 8000, "the result line");
    expect(line.textContent).toBe("Full model is up to date for these changes.");
    expect(line.textContent).not.toContain("0.62");
    const runPane = container.querySelector('[aria-label="Full model"]')!;
    expect(runPane.querySelector(".btn.accent")).toBeNull();
    expect([...runPane.querySelectorAll("button")].some((b) => b.textContent?.startsWith("Run full model"))).toBe(false);
    // the quick estimate steps back: muted, "before the full model"
    const pane = container.querySelector('[aria-label="Preview"]')!;
    expect(pane.getAttribute("data-superseded")).toBe("true");
    expect(pane.querySelector(".kicker")!.textContent).toBe("Quick estimate (before the full model)");
    scrolled.length = 0;
    click(byText(runPane, "button", "See full model result"));
    expect(scrolled.some((c) => c.includes("answer-card"))).toBe(true);
    expect(document.activeElement?.classList.contains("answer-card")).toBe(true);
    delete (HTMLElement.prototype as { scrollIntoView?: unknown }).scrollIntoView;
    const card = container.querySelector(".answer-card")!;
    expect(card.getAttribute("data-tone")).toBe("cool");
    expect(card.querySelector(".answer-chip")!.textContent).toBe("Confident it cools");
    expect([...container.querySelectorAll("details.rv-section")].map((d) => (d as HTMLElement).dataset.section)).toEqual(["where", "heat", "future", "changed", "sure", "tech"]);

    // Done collapses the change to its sentence
    click(byText(container.querySelector("article.edit-row[data-active]")!, "button", "Done"));
    expect(container.querySelector(".lab-edit-sentence")!.textContent).toContain("tree canopy in zone 1");
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
    const open = await waitFor(() => byText(container, "button", "Load full model"), 5000, "open button");
    expect(container.textContent).toContain("full model not loaded");
    // the chip and the button explain themselves in tooltips, not native titles
    expect(container.querySelector("[title]")).toBeNull();
    click(open);
    const dialog = await waitFor(() => document.querySelector('[role="dialog"]'), 5000, "trust dialog");
    expect(dialog.textContent).toContain("Checkpoint files execute code when they are loaded");
    expect(dialog.textContent).toContain(`/w/projects/demo/runs/${RID}`);
    expect(byText(dialog, "button", "Trust and load the full model")!.hasAttribute("title")).toBe(false);
    click(byText(dialog, "button", "Trust and load the full model"));
    await waitFor(() => opens === 2, 5000, "second open");
    const imp = fetchMock.calls.find((c) => c.url === "/api/runs/import")!;
    expect(imp.body).toEqual({ dir: `/w/projects/demo/runs/${RID}`, project_id: PID, trust_pickles: true });
    expect(useJobs.getState().jobs.j_open).toBeTruthy();
    await waitFor(() => !document.querySelector('[role="dialog"]'), 5000, "dialog closed");
  });

  it("offers Resume, not Load full model, for a run that stopped before it finished", async () => {
    // a run cancelled after S3 has a checkpoint but no manifest: every "Open engine" (now "Load full model") failed with FileNotFoundError
    fetchMock = mockFetch({
      ...runRoutes(),
      [`GET /api/runs/${RID}/engine`]: {
        body: {
          state: "error", progress: null, step: null, rss_mb: null, est_rss_mb: 300, code_match: null, loaded_utc: null, last_used_utc: null, job_id: null,
          error: { type: "RunNotFinished", message: "this run stopped before it finished: it has a checkpoint but no manifest.json" },
          action: { kind: "resume", label: "Resume the run to finish it", method: "POST", path: `/api/runs/${RID}/resume`, body: {} },
        },
      },
    });
    const { container } = render(<EngineChip rid={RID} />);
    await waitFor(() => byText(container, "button", "Resume the run to finish it"), 5000, "resume button");
    expect(container.textContent).toContain("stopped before it finished");
    expect(byText(container, "button", "Load full model")).toBeNull();
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
    click(await waitFor(() => byText(container, "button", "Load full model"), 5000, "open button"));
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

describe("Plan form fields", () => {
  it("shows the cost, plantable share and smallest amount labels, not only their aria-labels", async () => {
    fetchMock = mockFetch({
      ...runRoutes(),
      [`POST /api/runs/${RID}/plans/preview`]: {
        body: { planned_total: 100, n_cells_treated: 3, mean_dose_treated: 10, total_cost: 30, gini: 0.2, min_dose_dropped_cost: 0, pareto: [], dose: "AAAAAA==", constraint: "plantable", objective: "cooling", caption: "Plan" },
      },
    });
    const g = makeGrid(gridMeta(), Int32Array.from([0, 1, 0, 2, 0, 1, 2]), Int32Array.from([2, 2, 1, 1, 0, 0, 0]));
    const { container } = render(<PlanForm rid={RID} grid={g} levers={LEVERS} onSaved={() => {}} />);
    await flush(3);
    const field = (name: string) => [...container.querySelectorAll("label.field")].find((l) => l.querySelector(":scope > .field-label")?.textContent === name);
    for (const name of ["Cost per unit", "Plantable share of paved ground", "Smallest amount per treated place"]) {
      const input = field(name)?.querySelector<HTMLInputElement>(".numfield input");
      expect(input, name).toBeTruthy();
      expect(input!.getAttribute("aria-label")).toBe(name);
    }
    // the paved-share box is wide enough for its placeholder
    expect(field("Plantable share of paved ground")!.querySelector<HTMLInputElement>(".numfield input")!.style.width).toBe("11em");
    // the planner's words: unevenness of the amounts (not "Gini"), with the technical name in its ⓘ
    expect(container.querySelector(".kpis")!.textContent).toContain("Unevenness of the amounts");
    expect(container.querySelector(".kpis")!.textContent).not.toMatch(/Gini/);
    expect(container.querySelector('.kpis button.infotip[data-term="gini"]')).not.toBeNull();
  });
});

/** The scenarios ViewModel with two configured scenarios, the second without fold detail. */
function configuredView() {
  const vm = viewFixture("scenarios");
  const [a] = vm.sections.rows!;
  vm.sections.rows = [
    { ...a, slug: "canopy-plus-10", name: "Canopy +10", has_folds: true },
    { ...a, slug: "cool-roofs", name: "Cool roofs", has_folds: false },
  ];
  return vm;
}

describe("Library", () => {
  it("focuses the configured scenario of ?configured=<slug>", async () => {
    const scrolled = vi.fn();
    Element.prototype.scrollIntoView = scrolled;
    fetchMock = mockFetch({
      ...runRoutes(),
      [`GET /api/projects/${PID}/scenarios`]: { body: [{ id: "sc_a", revision: 1, parent_id: null, status: "exact", created_utc: "2026-10-01T10:00:00Z", updated_utc: "2026-10-01T10:00:00Z", name: "Downtown cool corridor", tags: ["district"], latest: result("res_1").summary }] },
      // "Set up with this run" reads the run's configured scenarios from the scenarios view
      [`GET /api/runs/${RID}/views/scenarios`]: { body: configuredView() },
    });
    navigate(`/r/${RID}/lab/library?configured=cool-roofs`, { replace: true });
    const { container } = render(<Library />);
    await waitFor(() => container.querySelector('[data-configured="cool-roofs"]'), 8000, "configured rows");
    const row = container.querySelector('[data-configured="cool-roofs"]')!.closest("tr")!;
    expect(row.classList.contains("hl")).toBe(true);
    expect(container.querySelector('[data-configured="canopy-plus-10"]')!.closest("tr")!.classList.contains("hl")).toBe(false);
    expect(scrolled).toHaveBeenCalled();
    // the configured scenario without a result by test area offers to run the full model again; lineage lists the scenario
    expect(byText(row, "button", "Run full model again")).not.toBeNull();
    expect(byText(container, "a", "Downtown cool corridor")).not.toBeNull();
  });
});
