// Launch forms (api.md §8–9): every kind posts exactly its documented param shape to its
// endpoint; the multiverse custom-variant editor produces {name: {"dotted.key": value}}; a card
// whose requirements are missing disables Launch and lists the server's missing items.
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { clearResources } from "../../../api/resource";
import { ACTION_KINDS, type LaunchableKind } from "../../../api/studies";
import { navigate } from "../../../router";
import { useJobs } from "../../../stores/jobs";
import { byText, click, flush, mockFetch, render, typeInto, waitFor, type MockHandler } from "../../../test/render";
import { CustomVariants } from "../components/CustomVariants";
import { LaunchPanel } from "../components/LaunchPanel";
import { StudyCard } from "../components/StudyCard";
import { customVariantsJson, draftsFromJson, parseOverrideValue, type CustomVariantDraft } from "../model/multiverse";
import { defaultForm, paramProblems, paramsFor, type Forms } from "../model/params";
import { job, PID, RID, statusRow, study } from "../__fixtures__/api";

/** The params api.md §8 allows for each kind (additionalProperties: false). */
const ALLOWED: Record<LaunchableKind, string[]> = {
  baselines: ["models"],
  planner: ["package", "thresholds", "hex_sizes", "export"],
  emulator: ["patches"],
  uncertainty: ["multiverse_study", "simcheck_studies", "placebo_study", "real_r2_gate"],
  writeup: [],
  placebo: ["kinds", "coarse_m", "seed", "grf_range_m"],
  simcheck: ["design", "coarse_m", "epochs", "workers", "threads", "continue_study_id"],
  multiverse: ["variants", "custom_variants", "coarse_m", "workers", "threads"],
  reproduce: ["stages", "tol_r2", "tol_effect"],
  benchmark: ["seed", "ab", "epochs", "n"],
};

const KINDS = Object.keys(ALLOWED) as LaunchableKind[];

beforeEach(() => useJobs.getState().reset());
afterEach(() => {
  clearResources();
  navigate("/", { replace: true });
});

describe("paramsFor: the default forms", () => {
  it("post the exact api.md §8 shapes", () => {
    expect(paramsFor("baselines", defaultForm("baselines"))).toEqual({ models: ["regression_kriging", "hgb_xy", "hgb", "idw", "hgb_focal"] });
    expect(paramsFor("planner", defaultForm("planner"))).toEqual({ hex_sizes: [250, 500], export: true });
    expect(paramsFor("emulator", defaultForm("emulator"))).toEqual({ patches: 8 });
    expect(paramsFor("uncertainty", defaultForm("uncertainty"))).toEqual({ real_r2_gate: false });
    expect(paramsFor("writeup", defaultForm("writeup"))).toEqual({});
    expect(paramsFor("placebo", defaultForm("placebo"))).toEqual({ kinds: ["grf", "shift", "rotate"], coarse_m: 60, seed: 0, grf_range_m: 600 });
    expect(paramsFor("simcheck", defaultForm("simcheck"))).toEqual({
      design: { physics: 20, additive: 20, own_only: 8, coarse_scale: 8, confounded: 8, null: 20 },
      coarse_m: 90,
      epochs: 200,
      workers: 1,
      threads: 1,
    });
    expect(paramsFor("multiverse", defaultForm("multiverse"))).toEqual({ variants: ["baseline"], coarse_m: 60, workers: 1, threads: 1 });
    expect(paramsFor("reproduce", defaultForm("reproduce"))).toEqual({ stages: ["S0", "S1", "S2", "S3"], tol_r2: 0.01, tol_effect: 0.05 });
    expect(paramsFor("benchmark", defaultForm("benchmark"))).toEqual({ seed: 0, ab: true, epochs: 150, n: 96 });
    for (const k of KINDS) {
      expect(paramProblems(k, defaultForm(k), 4), k).toEqual([]);
      for (const key of Object.keys(paramsFor(k, defaultForm(k)))) expect(ALLOWED[k], `${k}.${key}`).toContain(key);
    }
  });

  it("map edited forms: fine grid → coarse_m null, chosen sources, number lists, core order", () => {
    const placebo: Forms["placebo"] = { ...defaultForm("placebo"), kinds: ["rotate", "grf"], fine: true, seed: 7.2 };
    expect(paramsFor("placebo", placebo)).toEqual({ kinds: ["grf", "rotate"], coarse_m: null, seed: 7, grf_range_m: 600 });
    const unc: Forms["uncertainty"] = { mode: "choose", multiverse_study: "st_m", simcheck_studies: ["st_a", "st_b"], placebo_study: "", real_r2_gate: true };
    expect(paramsFor("uncertainty", unc)).toEqual({ multiverse_study: "st_m", simcheck_studies: ["st_a", "st_b"], real_r2_gate: true });
    const planner: Forms["planner"] = { package: " shade_streets ", thresholds: "90, 95 ; 100", hex_sizes: "250", export: false };
    expect(paramsFor("planner", planner)).toEqual({ package: "shade_streets", thresholds: [90, 95, 100], hex_sizes: [250], export: false });
    const sim: Forms["simcheck"] = { ...defaultForm("simcheck"), design: { physics: 2, additive: 0, own_only: 0, coarse_scale: 0, confounded: 0, null: 2 }, continue_study_id: "st_s" };
    expect(paramsFor("simcheck", sim)).toEqual({ design: { physics: 2, additive: 0, own_only: 0, coarse_scale: 0, confounded: 0, null: 2 }, coarse_m: 90, epochs: 200, workers: 1, threads: 1, continue_study_id: "st_s" });
    const rep: Forms["reproduce"] = { stages: ["S5", "S1"], tol_r2: 0.02, tol_effect: 0.1 };
    expect(paramsFor("reproduce", rep)).toEqual({ stages: ["S0", "S1", "S5"], tol_r2: 0.02, tol_effect: 0.1 });
  });

  it("refuse invalid forms with a reason", () => {
    expect(paramProblems("multiverse", { ...defaultForm("multiverse"), workers: 3, threads: 2 }, 4)[0]).toMatch(/above the heavy-job thread budget of 4/);
    expect(paramProblems("baselines", { models: [] }, null)).toHaveLength(1);
    expect(paramProblems("placebo", { ...defaultForm("placebo"), kinds: [] }, null)).toHaveLength(1);
    expect(paramProblems("simcheck", { ...defaultForm("simcheck"), design: { physics: 0, additive: 0, own_only: 0, coarse_scale: 0, confounded: 0, null: 0 } }, null)[0]).toMatch(/no replicates/);
    expect(paramProblems("planner", { ...defaultForm("planner"), thresholds: "ninety" }, null)[0]).toMatch(/Thresholds/);
  });
});

describe("custom multiverse variants", () => {
  it("produce {name: {'dotted.key': value}} with JSON-typed values", () => {
    const drafts: CustomVariantDraft[] = [
      { name: "big_blocks", rows: [{ key: "cv.block_m", value: "2000" }, { key: "models.gwrf", value: "false" }] },
      { name: "two_scales", rows: [{ key: "influence.scales", value: "[1.0, 2.0]" }, { key: "physics.forcing", value: "None" }, { key: "", value: "" }] },
      { name: "named", rows: [{ key: "data.label", value: "summer run" }, { key: "physics.lw_net", value: "−90" }] },
    ];
    const r = customVariantsJson(drafts);
    expect(r.errors).toEqual([]);
    expect(r.value).toEqual({
      big_blocks: { "cv.block_m": 2000, "models.gwrf": false },
      two_scales: { "influence.scales": [1.0, 2.0], "physics.forcing": null },
      named: { "data.label": "summer run", "physics.lw_net": -90 },
    });
    expect(draftsFromJson(r.value).map((d) => d.name)).toEqual(["big_blocks", "two_scales", "named"]);
    expect(customVariantsJson(draftsFromJson(r.value)).value).toEqual(r.value);
    expect(parseOverrideValue("True")).toBe(true);
    expect(parseOverrideValue('"2000"')).toBe("2000");
  });

  it("report clashes and bad keys", () => {
    const r = customVariantsJson([
      { name: "baseline", rows: [{ key: "cv.block_m", value: "1" }] },
      { name: "Bad Name", rows: [{ key: "cv.block_m", value: "1" }] },
      { name: "dup", rows: [{ key: "cv..x", value: "1" }, { key: "ok.key", value: "3" }] },
      { name: "dup", rows: [{ key: "a.b", value: "1" }] },
      { name: "twice", rows: [{ key: "a.b", value: "1" }, { key: "a.b", value: "2" }] },
      { name: "empty", rows: [{ key: "", value: "" }] },
    ]);
    expect(r.errors.join("\n")).toMatch(/built-in variant name/);
    expect(r.errors.join("\n")).toMatch(/lower-case letters/);
    expect(r.errors.join("\n")).toMatch(/not a dotted config key/);
    expect(r.errors.join("\n")).toMatch(/defined twice/);
    expect(r.errors.join("\n")).toMatch(/sets a.b twice/);
    expect(r.errors.join("\n")).toMatch(/changes nothing/);
    // only well-formed, non-empty, first-defined variants are kept (the errors block the launch anyway)
    expect(r.value).toEqual({ dup: { "ok.key": 3 }, twice: { "a.b": 2 } });
  });

  it("the editor builds the JSON from typed rows", () => {
    let value: CustomVariantDraft[] = [];
    const r = render(<CustomVariants value={value} onChange={(v) => (value = v)} />);
    const rerender = () => r.rerender(<CustomVariants value={value} onChange={(v) => (value = v)} />);
    click(byText(r.container, "button", "Add custom variant"));
    rerender();
    typeInto(r.container.querySelector('[aria-label="Custom variant 1 name"]'), "coarse_blocks");
    rerender();
    typeInto(r.container.querySelector('[aria-label="Variant 1 override 1 key"]'), "cv.block_m");
    rerender();
    typeInto(r.container.querySelector('[aria-label="Variant 1 override 1 value"]'), "2500");
    rerender();
    click(byText(r.container, "button", "Add override"));
    rerender();
    typeInto(r.container.querySelector('[aria-label="Variant 1 override 2 key"]'), "models.physics");
    rerender();
    typeInto(r.container.querySelector('[aria-label="Variant 1 override 2 value"]'), "false");
    rerender();
    expect(customVariantsJson(value).value).toEqual({ coarse_blocks: { "cv.block_m": 2500, "models.physics": false } });
    expect(JSON.parse(r.container.querySelector('[data-testid="custom-variants-json"]')!.textContent!)).toEqual({ coarse_blocks: { "cv.block_m": 2500, "models.physics": false } });
  });
});

function routesFor(kind: LaunchableKind, extra: Record<string, MockHandler> = {}): Record<string, MockHandler> {
  const launched = job("j_launch", (ACTION_KINDS as readonly string[]).includes(kind) ? `post.${kind}` : `study.${kind}`);
  return {
    "GET /api/settings": { body: { threads_heavy: 4 } },
    "POST /api/studies/estimate": { body: { est_s: 600, est_lo: 480, est_hi: 900, est_peak_rss_gb: 2.1, est_disk_gb: 0.3, n_children: 3 } },
    [`POST /api/runs/${RID}/actions/${kind}`]: { status: 202, body: launched },
    [`POST /api/runs/${RID}/studies/${kind}`]: { status: 202, body: { study: study("st_new", kind), job: launched } },
    [`POST /api/projects/${PID}/studies/benchmark`]: { status: 202, body: { study: study("st_b", "benchmark", { target_run_id: null }), job: launched } },
    ...extra,
  };
}

describe("LaunchPanel posts each kind to its endpoint", () => {
  for (const kind of KINDS) {
    it(kind, async () => {
      const m = mockFetch(routesFor(kind));
      const { container } = render(<LaunchPanel kind={kind} rid={kind === "benchmark" ? null : RID} pid={PID} ctx={{ studies: [], threadsHeavy: 4 }} />);
      await waitFor(() => container.querySelector('[aria-label="Estimated cost"]'), 3000, "estimate");
      expect(container.querySelector('[aria-label="Estimated cost"]')!.textContent).toContain("3 child runs");
      click(container.querySelector(`[data-launch="${kind}"]`));
      await flush(4);
      const est = m.calls.find((c) => c.url === "/api/studies/estimate")!;
      expect(est.body).toEqual({ kind, ...(kind === "benchmark" ? {} : { run_id: RID }), params: paramsFor(kind, defaultForm(kind)) });
      const post = m.calls.filter((c) => c.method === "POST" && c.url !== "/api/studies/estimate");
      expect(post).toHaveLength(1);
      const path =
        kind === "benchmark" ? `/api/projects/${PID}/studies/benchmark` : (ACTION_KINDS as readonly string[]).includes(kind) ? `/api/runs/${RID}/actions/${kind}` : `/api/runs/${RID}/studies/${kind}`;
      expect(post[0].url).toBe(path);
      expect(post[0].body).toEqual(paramsFor(kind, defaultForm(kind)));
      expect(useJobs.getState().jobs.j_launch).toBeDefined();
      m.restore();
    });
  }

  it("multiverse: ticked variants and a custom variant reach the body", async () => {
    const m = mockFetch(routesFor("multiverse"));
    const { container } = render(<LaunchPanel kind="multiverse" rid={RID} pid={PID} ctx={{ studies: [], threadsHeavy: 4 }} />);
    click(byText(container, "label", "1 km CV blocks")!.querySelector("input"));
    click(byText(container, "label", "no GW random forest")!.querySelector("input"));
    click(byText(container, "button", "Add custom variant"));
    typeInto(container.querySelector('[aria-label="Custom variant 1 name"]'), "half_km_blocks");
    typeInto(container.querySelector('[aria-label="Variant 1 override 1 key"]'), "cv.block_m");
    typeInto(container.querySelector('[aria-label="Variant 1 override 1 value"]'), "500");
    await flush(2);
    click(container.querySelector('[data-launch="multiverse"]'));
    await flush(4);
    const post = m.calls.find((c) => c.url === `/api/runs/${RID}/studies/multiverse`)!;
    expect(post.body).toEqual({ variants: ["baseline", "blocks_1km", "no_gwrf"], custom_variants: { half_km_blocks: { "cv.block_m": 500 } }, coarse_m: 60, workers: 1, threads: 1 });
    m.restore();
  });

  it("simcheck: the design editor's counts and preset reach the body", async () => {
    const m = mockFetch(routesFor("simcheck"));
    const { container } = render(<LaunchPanel kind="simcheck" rid={RID} pid={PID} ctx={{ studies: [], threadsHeavy: 4 }} />);
    click(byText(container, "button", "Quick check (2 each)"));
    expect(container.querySelector('[data-testid="design-total"]')!.textContent).toBe("12");
    const physics = byText(container, "tr", "Physics (energy balance")!.querySelector("input")!;
    typeInto(physics, "5");
    physics.dispatchEvent(new FocusEvent("blur", { bubbles: false }));
    physics.dispatchEvent(new FocusEvent("focusout", { bubbles: true }));
    await flush(2);
    expect(container.querySelector('[data-testid="design-total"]')!.textContent).toBe("15");
    click(container.querySelector('[data-launch="simcheck"]'));
    await flush(4);
    const post = m.calls.find((c) => c.url === `/api/runs/${RID}/studies/simcheck`)!;
    expect(post.body).toEqual({ design: { physics: 5, additive: 2, own_only: 2, coarse_scale: 2, confounded: 2, null: 2 }, coarse_m: 90, epochs: 200, workers: 1, threads: 1 });
    m.restore();
  });

  it("a workers × threads above the heavy budget disables Launch", async () => {
    const m = mockFetch(routesFor("simcheck"));
    const { container } = render(<LaunchPanel kind="simcheck" rid={RID} pid={PID} ctx={{ studies: [], threadsHeavy: 2 }} />);
    const workers = byText(container, ".field", "Workers")!.querySelector("input")!;
    typeInto(workers, "3");
    workers.dispatchEvent(new FocusEvent("focusout", { bubbles: true }));
    await flush(2);
    expect(container.querySelector('[aria-label="Fix before launching"]')!.textContent).toContain("above the heavy-job thread budget of 2");
    expect((container.querySelector('[data-launch="simcheck"]') as HTMLButtonElement).disabled).toBe(true);
    m.restore();
  });
});

describe("requirements", () => {
  it("a card in 'requirements missing' state disables launch and lists the server's missing items", async () => {
    const m = mockFetch(routesFor("placebo"));
    const row = statusRow("placebo", { requirements: { ok: false, missing: ["roles.canopy", "roles.impervious", "custom.thing"] } });
    const { container } = render(<StudyCard row={row} rid={RID} pid={PID} units="°F" ctx={{ studies: [], threadsHeavy: 4 }} />);
    await flush(3);
    const btn = container.querySelector('[data-launch="placebo"]') as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
    const items = [...container.querySelectorAll('[data-testid="missing-requirements"] li')];
    expect(items.map((li) => li.getAttribute("data-requirement"))).toEqual(["roles.canopy", "roles.impervious", "custom.thing"]);
    expect(items[0].textContent).toContain("canopy role");
    expect(items[2].textContent!.trim()).toBe("custom.thing");
    click(btn);
    await flush(2);
    expect(m.calls.filter((c) => c.method === "POST" && c.url.includes("/studies/placebo"))).toEqual([]);
    // no estimate is asked for a blocked launch
    expect(m.calls.some((c) => c.url === "/api/studies/estimate")).toBe(false);
    m.restore();
  });

  it("a 422 requirements refusal lists detail.missing and disables Launch", async () => {
    const m = mockFetch(
      routesFor("planner", {
        [`POST /api/runs/${RID}/actions/planner`]: { status: 422, body: { error: { code: "requirements", message: "Missing requirements", detail: { missing: ["planner.layers"] } } } },
      }),
    );
    const { container } = render(<StudyCard row={statusRow("planner")} rid={RID} pid={PID} units="°F" ctx={{ studies: [], threadsHeavy: 4 }} />);
    click(container.querySelector('[data-launch="planner"]'));
    await waitFor(() => container.querySelector('[data-requirement="planner.layers"]'), 3000, "missing list");
    expect(container.querySelector('[data-requirement="planner.layers"]')!.textContent).toContain("people and land-cover layers");
    expect((container.querySelector('[data-launch="planner"]') as HTMLButtonElement).disabled).toBe(true);
    m.restore();
  });

  it("the card shows status, estimate and headline, and literature has no launch form", async () => {
    const m = mockFetch({ "GET /api/settings": { body: { threads_heavy: 4 } }, [`GET /api/runs/${RID}/views/response`]: { body: { view: "response", availability: "missing", missing: [], units: { target: "°F", levers: {} }, caveats: [], demo: false, sections: { levers: null, literature: null } } } });
    const { container } = render(
      <>
        <StudyCard row={statusRow("multiverse", { state: "stale", headline: "Sign stable in 8 of 8 scenarios" })} rid={RID} pid={PID} units="°F" ctx={{ studies: [], threadsHeavy: 4 }} />
        <StudyCard row={statusRow("literature", { state: "done", estimate: null })} rid={RID} pid={PID} units="°F" ctx={{ studies: [], threadsHeavy: 4 }} />
      </>,
    );
    await flush(3);
    const mv = container.querySelector('[data-kind="multiverse"]')!;
    expect(mv.textContent).toContain("stale vs run");
    expect(mv.textContent).toContain("Estimated ≈");
    expect(mv.textContent).toContain("Sign stable in 8 of 8 scenarios");
    expect(mv.querySelector("details.vt-launch")!.hasAttribute("open")).toBe(false);
    const lit = container.querySelector('article[data-kind="literature"]')!;
    expect(lit.querySelector("[data-launch]")).toBeNull();
    expect(lit.textContent).toContain("no literature panel");
    m.restore();
  });
});
