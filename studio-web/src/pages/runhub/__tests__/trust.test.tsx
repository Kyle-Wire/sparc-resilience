// How much to trust this (`/r/:rid/trust`): the grade, the four checks (✓ or ! with one plain
// result and a link, no charts), the independent checks' verdicts with the study cards behind a
// disclosure, the scenarios' ranges in two lines, and the limitations with a link to the run's
// caveats. The charts and the technical cards stay under Model details.
import { afterEach, describe, expect, it, vi } from "vitest";
import { clearResources } from "../../../api/resource";
import type { CausalTreatment, Sections, ViewModelOf } from "../../../api/runs";
import { navigate } from "../../../router";
import { flush, mockFetch, render, waitFor, type MockHandler } from "../../../test/render";
import { checkVerdicts } from "../../studies/Validation";
import { statusRow } from "../../studies/__fixtures__/api";
import { viewFixture } from "../__fixtures__/views";
import { causalAgreement } from "../Causal";
import { skillAcross } from "../Distance";
import Trust, { accuracyCheck, causalCheck, distanceCheck, distanceCoverageCheck, rangesCheck, underTargetLine, verdictLine, wantsStudyCards } from "../Trust";
import { mapRoutes, runDetail } from "./helpers";

// The study cards (launch forms, result views) have their own tests; a stub keeps this one on the page.
vi.mock("../../studies/Validation", async (orig) => ({ ...(await orig<typeof import("../../studies/Validation")>()), StudyCards: () => "Study-cards stub" }));

type Treatment = Sections<CausalTreatment> & { label: string; unit: string };

const STUDY_ROWS = [
  statusRow("baselines", { state: "done", headline: "Stack beats 4 of 5 baselines" }),
  statusRow("placebo", { state: "done", study_id: "st_p", headline: "model passes 2/2, causal passes 2/2" }),
  statusRow("reproduce", { state: "stale", headline: "reproduced" }),
  statusRow("planner", { state: "done" }),
];

let seq = 0;

async function renderTrust(
  opts: { views?: Partial<Record<"overview" | "accuracy" | "distance" | "causal" | "uncertainty", MockHandler>>; hash?: string; extra?: (rid: string) => Record<string, MockHandler> } = {},
) {
  const rid = `r_trust${++seq}`;
  const view = (v: "overview" | "accuracy" | "distance" | "causal" | "uncertainty") => opts.views?.[v] ?? { body: viewFixture(v) };
  const m = mockFetch({
    ...mapRoutes(rid),
    [`GET /api/runs/${rid}/views/overview`]: view("overview"),
    [`GET /api/runs/${rid}/views/accuracy`]: view("accuracy"),
    [`GET /api/runs/${rid}/views/distance`]: view("distance"),
    [`GET /api/runs/${rid}/views/causal`]: view("causal"),
    [`GET /api/runs/${rid}/views/uncertainty`]: view("uncertainty"),
    [`GET /api/runs/${rid}/studies`]: { body: STUDY_ROWS },
    ...(opts.extra?.(rid) ?? {}),
  });
  navigate(`/r/${rid}/trust${opts.hash ?? ""}`, { replace: true });
  const r = render(<Trust />);
  await waitFor(() => r.container.querySelector('[data-check="causal"]:not([data-status="loading"])') && r.container.querySelector('[aria-label="Independent checks"]'), 5000, "trust page");
  await flush(6);
  return { ...r, rid, restore: m.restore };
}

const charts = (root: Element) => [...root.querySelectorAll("figure.chart-frame")].map((f) => f.getAttribute("data-chart") ?? "");
const check = (root: HTMLElement, id: string) => root.querySelector<HTMLElement>(`[data-check="${id}"]`)!;

describe("trust checks (pure)", () => {
  it("accuracy: Good R² holds up, Fair or Limited asks for care, unknown is not checked", () => {
    expect(accuracyCheck(0.815, 0.342, "°F")).toMatchObject({ status: "ok" });
    expect(accuracyCheck(0.815, 0.342, "°F").line).toContain("held-out R² of 0.82 and a typical error (RMSE) of 0.34 °F");
    expect(accuracyCheck(0.45, null, "°F").status).toBe("warn");
    expect(accuracyCheck(0.1, null, "°F").line).toContain("rough guide");
    expect(accuracyCheck(null, null, "°F").status).toBe("unknown");
  });

  it("ranges: within 5 points of the target hold up; too narrow or too wide ask for care", () => {
    expect(rangesCheck(0.8902, 0.9)).toMatchObject({ status: "ok" });
    expect(rangesCheck(0.8902, 0.9).line).toContain("89% of held-out readings fall inside their likely range, which aims for 90%");
    expect(rangesCheck(0.85, null).status).toBe("ok");
    expect(rangesCheck(0.8, 0.9)).toMatchObject({ status: "warn" });
    expect(rangesCheck(0.8, 0.9).line).toContain("too narrow");
    expect(rangesCheck(0.99, 0.9).line).toContain("wider than they need to be");
    expect(rangesCheck(null, 0.9).status).toBe("unknown");
  });

  it("distance: R² at least Fair far out holds up; RMSE may grow by half", () => {
    const sa = skillAcross(viewFixture("distance").sections.curve)!;
    expect(sa).toEqual({ metric: "r2", near: 0.86, far: 0.74, nearBlockM: 180, farBlockM: 780 });
    expect(distanceCheck(sa, "°F")).toMatchObject({ status: "ok" });
    expect(distanceCheck(sa, "°F")!.line).toContain("0.86 when tested on 180 m blocks and 0.74 on 780 m blocks");
    expect(distanceCheck({ ...sa, far: 0.3 }, "°F")!.status).toBe("warn");
    expect(distanceCheck({ metric: "rmse", near: 0.4, far: 0.55, nearBlockM: 100, farBlockM: 800 }, "°F")!.status).toBe("ok");
    expect(distanceCheck({ metric: "rmse", near: 0.4, far: 0.7, nearBlockM: 100, farBlockM: 800 }, "°F")!.status).toBe("warn");
    expect(distanceCheck(null, "°F")).toBeNull();
    expect(skillAcross({ ...viewFixture("distance").sections.curve!, series: [] })).toBeNull();
  });

  it("cause or coincidence: one line per lever from the causal audit", () => {
    const vm = viewFixture("causal");
    const canopy = vm.sections.treatments!.canopy as Treatment;
    const a = causalAgreement("canopy", canopy, vm.units.target, vm.units.levers);
    expect(a).toMatchObject({ lever: "canopy", status: "size" });
    expect(a.line).toBe("canopy: the independent check agrees on the direction but finds a different size (model −0.124, check −0.041 °F per pp).");
    const ok = { ...canopy, audit: canopy.audit!.map((x) => ({ ...x, flag: false, verdict: "consistent" })) };
    expect(causalAgreement("canopy", ok, "degF", {}).status).toBe("agrees");
    const conflict = { ...canopy, audit: [{ ...canopy.audit![0], verdict: "sign conflict" }] };
    expect(causalAgreement("canopy", conflict, "degF", {}).status).toBe("conflict");
    const shape = { ...canopy, audit: [{ ...canopy.audit![2], flag: true, verdict: "shape differs" }] };
    expect(causalAgreement("canopy", shape, "degF", {}).line).toContain("builds up differently");
    expect(causalAgreement("canopy", { ...canopy, audit: null }, "degF", {}).status).toBe("unknown");

    expect(causalCheck([a])).toMatchObject({ status: "warn" });
    expect(causalCheck([{ ...a, status: "agrees" }, { ...a, lever: "albedo", status: "agrees" }]).line).toContain("for all 2 levers");
    expect(causalCheck([{ ...a, status: "agrees" }, { ...a, lever: "albedo", status: "conflict" }]).line).toContain("points the other way from the model for one lever");
    expect(causalCheck([{ ...a, status: "agrees" }, { ...a, lever: "albedo" }]).line).toContain("for 1 of 2 levers");
    expect(causalCheck([]).status).toBe("unknown");
  });

  it("independent checks: the verdict kinds in card order, headline else the state in words", () => {
    const v = checkVerdicts(STUDY_ROWS);
    expect(v.map((x) => x.kind)).toEqual(["baselines", "placebo", "simcheck", "multiverse", "reproduce", "literature", "benchmark"]);
    expect(v[0].line).toBe("Stack beats 4 of 5 baselines");
    expect(v.find((x) => x.kind === "reproduce")!.line).toBe("reproduced (out of date: the run has changed since)");
    expect(v.find((x) => x.kind === "simcheck")!.line).toBe("Not run yet.");
  });

  it("the study cards open for #validation and #study-<kind>", () => {
    expect(wantsStudyCards("#validation")).toBe(true);
    expect(wantsStudyCards("#study-placebo")).toBe(true);
    expect(wantsStudyCards("")).toBe(false);
    expect(wantsStudyCards("#accuracy")).toBe(false);
  });
});

describe("trust one-liners (pure)", () => {
  it("names the distance bands whose ranges fall short, and counts the verdicts", () => {
    const band = (label: string, global: number | null) => ({ label, global }) as Parameters<typeof underTargetLine>[0][number];
    expect(underTargetLine([band("0-100 m", 0.9), band("100-300 m", 0.88)], 0.9)).toBeNull();
    expect(underTargetLine([band("0-100 m", 0.9), band("300+ m", 0.7)], 0.9)).toContain("at 300+ m");
    // without the skill-vs-distance curve, the coverage by distance is a partial answer
    expect(distanceCoverageCheck([], 0.9)).toBeNull();
    expect(distanceCoverageCheck([band("0-100 m", null)], 0.9)).toBeNull();
    expect(distanceCoverageCheck([band("0-100 m", 0.9), band("100-300 m", 0.88)], 0.9)).toEqual({
      status: "partial",
      line: "Only the likely ranges were checked at increasing distances from the measurements, not the accuracy itself. They hold their 90% target at every distance checked.",
    });
    expect(distanceCoverageCheck([band("300+ m", 0.7)], 0.9, "The stage did not run.")!.line).toMatch(/at 300\+ m\. The stage did not run\.$/);
    const v = (verdict: "robust" | "direction" | "not_established", label: string) => ({ verdict: { verdict, label, reasons: [], qualifiers: [] } });
    expect(verdictLine([{ verdict: null }])).toBeNull();
    expect(verdictLine([v("direction", "Direction only"), v("robust", "Robust"), v("robust", "Robust"), { verdict: null }])).toBe(
      "What the evidence supports: Robust 2, Direction only 1 (of 4 scenarios).",
    );
  });
});

describe("How much to trust this", () => {
  afterEach(() => clearResources());

  it("shows the grade, then the four checks with a mark and one plain result, without the Model details charts", async () => {
    const r = await renderTrust();
    try {
      const grade = r.container.querySelector<HTMLElement>("#grade")!;
      expect(grade.dataset.grade).toBe("good");
      expect(grade.querySelector(".value")!.textContent).toBe("Good");
      expect(grade.querySelector("[data-grade-line]")!.textContent).toBe("Predicts well in places it was not trained on. Its likely ranges hold.");

      const rows = [...r.container.querySelectorAll<HTMLElement>("[data-check]")];
      expect(rows.map((x) => [x.dataset.check, x.dataset.status])).toEqual([
        ["accuracy", "ok"],
        ["ranges", "ok"],
        ["distance", "ok"],
        ["causal", "warn"],
      ]);
      expect(rows.map((x) => x.querySelector("h3")!.textContent)).toEqual([
        "Accuracy in places it never saw",
        "Are the ranges honest?",
        "Does it hold up away from measurements?",
        "Cause or coincidence?",
      ]);
      expect(check(r.container, "accuracy").querySelector(".pill")!.textContent).toBe("Holds up");
      expect(check(r.container, "causal").querySelector(".pill")!.textContent).toBe("Take care");
      expect(check(r.container, "accuracy").textContent).toContain("held-out R² of 0.82 and a typical error (RMSE) of 0.34 °F");
      // observed vs predicted, coverage by distance and skill vs block size are Model details' charts
      expect(charts(r.container)).toEqual([]);
      // every distance claim is the distance check's: the ranges check says nothing about distance
      expect(check(r.container, "ranges").textContent).not.toContain("Farther");
      expect(check(r.container, "distance").textContent).toContain("0.86 when tested on 180 m blocks and 0.74 on 780 m blocks");
      expect(check(r.container, "distance").querySelector("[data-distance-ranges]")!.textContent).toBe(
        "Farther from the measurements they hold less well: under 90% by more than 5 points at 240-330 m.",
      );
      const levers = check(r.container, "causal").querySelectorAll("[data-lever]");
      expect([...levers].map((l) => l.getAttribute("data-agreement"))).toEqual(["size"]);

      const links = Object.fromEntries(rows.map((x) => [x.dataset.check, x.querySelector("[data-details-link]")!.getAttribute("href")]));
      expect(links).toEqual({
        accuracy: `/r/${r.rid}/details/accuracy`,
        ranges: `/r/${r.rid}/details/accuracy`,
        distance: `/r/${r.rid}/details/reach`,
        causal: `/r/${r.rid}/details/causal`,
      });
      expect(r.container.textContent).not.toContain("could not be displayed");
    } finally {
      r.restore();
    }
  });

  it("leaves the technical cards to Model details", async () => {
    const r = await renderTrust();
    try {
      const titles = charts(r.container);
      for (const gone of ["Stacker candidates", "Blend weights per fold", "Physics parameters per fold", "Advection ΔRMSE per fold", "Residual histogram", "Stack vs baselines", "Uncertainty by source"])
        expect(titles, gone).not.toContain(gone);
      const text = r.container.textContent ?? "";
      for (const gone of ["Blend weight", "Physics card", "Forcing", "CV design", "CATE", "DAG audit", "Evidence sources"]) expect(text, gone).not.toContain(gone);
    } finally {
      r.restore();
    }
  });

  it("lists the independent checks' verdicts and keeps the study cards behind a disclosure", async () => {
    const r = await renderTrust();
    try {
      const list = r.container.querySelector('[aria-label="Independent checks"]')!;
      const items = [...list.querySelectorAll<HTMLElement>("[data-check-kind]")];
      expect(items.map((i) => i.dataset.checkKind)).toEqual(["baselines", "placebo", "simcheck", "multiverse", "reproduce", "literature", "benchmark"]);
      expect(items[0].textContent).toContain("Stack beats 4 of 5 baselines");
      expect(items[0].querySelector("a")!.getAttribute("href")).toBe(`/r/${r.rid}/trust#study-baselines`);
      expect(items[2].querySelector("a")!.textContent).toBe("Run it");
      const cards = r.container.querySelector<HTMLDetailsElement>("details[data-study-cards]")!;
      expect(cards.open).toBe(false);
      expect(cards.textContent).not.toContain("Study-cards stub");
    } finally {
      r.restore();
    }
  });

  it("opens the study cards for a #study-<kind> link", async () => {
    const r = await renderTrust({ hash: "#study-placebo" });
    try {
      const cards = r.container.querySelector<HTMLDetailsElement>("details[data-study-cards]")!;
      expect(cards.open).toBe(true);
      expect(cards.textContent).toContain("Study-cards stub");
    } finally {
      r.restore();
    }
  });

  it("sums up the scenarios' ranges in two lines (the table is Model details'), and lists the model card's limitations", async () => {
    const r = await renderTrust();
    try {
      const sec = r.container.querySelector<HTMLElement>("#uncertainty")!;
      expect(sec.querySelector("[data-uncertainty-summary]")!.textContent).toContain("3 of 3 scenarios still cool under the widest (envelope) interval");
      expect(sec.querySelector("[data-verdicts]")!.textContent).toBe("What the evidence supports: Robust 1 (of 3 scenarios).");
      expect(sec.querySelector("[data-scenario], table")).toBeNull();
      expect(sec.querySelector('a[data-details-link]')!.getAttribute("href")).toBe(`/r/${r.rid}/details/uncertainty`);

      const lim = r.container.querySelector('#limitations [aria-label="Limitations"]')!;
      expect([...lim.querySelectorAll("li")].map((l) => l.textContent)).toEqual(viewFixture("overview").sections.limitations);
      // the run's caveats have one home, the Summary's "Things to keep in mind": linked, not repeated
      const caveats = r.container.querySelector("#limitations [data-caveats-link]")!;
      expect(caveats.textContent).toContain("caveats (1)");
      expect(caveats.querySelector("a")!.getAttribute("href")).toBe(`/r/${r.rid}#keep-in-mind`);
      expect(r.container.textContent).not.toContain("Generic physics forcing");
    } finally {
      r.restore();
    }
  });

  it("is the one home of the skipped skill-vs-distance notice, its re-run sized to its label", async () => {
    const distance = viewFixture("distance");
    distance.sections.curve = null;
    const overview = viewFixture("overview");
    const r = await renderTrust({
      views: { distance: { body: distance }, overview: { body: overview } },
      extra: (rid) => {
        const detail = runDetail(rid);
        detail.stages = [{ id: "cv_curve", state: "skipped", seconds: null, source: "manifest", reason: "disabled_by_config:cv.distance_curve.enabled" }];
        overview.sections.outputs_grid = [
          { id: "cv_distance", label: "Skill vs distance", group: "trust", state: "missing", produced_by: "stage:cv_curve", view: "distance", action: { kind: "open", label: "Re-run with the CV curve", method: "GET", path: `/p/p_1/launch?from=${rid}` } },
        ];
        return { [`GET /api/runs/${rid}`]: { body: detail } };
      },
    });
    try {
      const c = check(r.container, "distance");
      const button = await waitFor(() => [...c.querySelectorAll("button")].find((b) => b.textContent === "Re-run with the CV curve"), 3000, "the re-run");
      // skill was not tested far out, but the ranges were: partly checked, from the coverage by distance
      expect(c.dataset.status).toBe("partial");
      expect(c.querySelector(".pill")!.textContent).toBe("Partly checked");
      expect(c.textContent).toContain("Only the likely ranges were checked at increasing distances");
      expect(c.textContent).toContain("under 90% by more than 5 points at 240-330 m");
      expect(c.textContent).toContain("cv.distance_curve.enabled");
      expect(check(r.container, "ranges").textContent).not.toContain("Farther");
      expect(button.style.justifySelf).toBe("start");
      expect(button.style.alignSelf).toBe("flex-start");
    } finally {
      r.restore();
    }
  });

  it("marks a check 'Not checked' when its view is missing", async () => {
    const causal: ViewModelOf<"causal"> = { ...viewFixture("causal"), availability: "missing", missing: [{ output: "causal.json", produced_by: "stage:S6", action: null }] };
    const r = await renderTrust({ views: { causal: { body: causal } } });
    try {
      const c = check(r.container, "causal");
      expect(c.dataset.status).toBe("unknown");
      expect(c.querySelector(".pill")!.textContent).toBe("Not checked");
      expect(c.textContent).toContain("Not available for this run yet.");
    } finally {
      r.restore();
    }
  });
});
