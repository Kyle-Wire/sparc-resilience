// How much to trust this (`/r/:rid/trust`): the grade, the four checks (✓ or ! with one plain
// result and a link, no charts), the independent checks' verdicts with the study cards behind a
// disclosure, the scenarios' ranges in two lines, and the limitations with a link to the run's
// caveats. The charts and the technical cards stay under Model details. Plain words (phase 4):
// each check's ⓘ and the <Term>s in its line carry the technical meaning (glossary), the verdicts
// read Solid / Direction clear / Not proven, and there are no native titles.
import { act } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { clearResources } from "../../../api/resource";
import type { CausalTreatment, Sections, ViewModelOf } from "../../../api/runs";
import { term } from "../../../content/glossary";
import { navigate } from "../../../router";
import { flush, mockFetch, render, waitFor, type MockHandler } from "../../../test/render";
import { checkVerdicts } from "../../studies/Validation";
import { statusRow } from "../../studies/__fixtures__/api";
import { viewFixture } from "../__fixtures__/views";
import { causalAgreement } from "../Causal";
import { skillAcross } from "../Distance";
import Trust, { accuracyCheck, causalCheck, distanceCheck, distanceCoverageCheck, rangesCheck, TermText, underTargetLine, verdictLine, wantsStudyCards } from "../Trust";
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

/** Focus an element and return a copy of the tooltip it opens, then close it. */
function tipOf(el: Element): HTMLElement {
  act(() => (el as HTMLElement).focus());
  const tip = document.querySelector<HTMLElement>('[role="tooltip"]:not([hidden])');
  expect(tip, "a tooltip opens on focus").not.toBeNull();
  const copy = tip!.cloneNode(true) as HTMLElement;
  act(() => (el as HTMLElement).blur());
  return copy;
}

const charts = (root: Element) => [...root.querySelectorAll("figure.chart-frame")].map((f) => f.getAttribute("data-chart") ?? "");
const check = (root: HTMLElement, id: string) => root.querySelector<HTMLElement>(`[data-check="${id}"]`)!;

describe("trust checks (pure)", () => {
  it("accuracy: Good R² holds up, Fair or Limited asks for care, unknown is not checked", () => {
    expect(accuracyCheck(0.815, 0.342, "°F")).toMatchObject({ status: "ok" });
    expect(accuracyCheck(0.815, 0.342, "°F").line).toBe(
      "Tested on areas left out of training, the model's accuracy score is 0.82 (1 is perfect), and its typical error is 0.34 °F. That is good: 0.6 or more counts as good.",
    );
    // plain words: the technical names are in the tooltips, not the line
    expect(accuracyCheck(0.815, 0.342, "°F").line).not.toMatch(/R²|RMSE|held-out/);
    expect(accuracyCheck(0.45, null, "°F").status).toBe("warn");
    expect(accuracyCheck(0.1, null, "°F").line).toContain("rough guide");
    expect(accuracyCheck(null, null, "°F").status).toBe("unknown");
  });

  it("ranges: within 5 points of the target hold up; too narrow or too wide ask for care", () => {
    expect(rangesCheck(0.8902, 0.9)).toMatchObject({ status: "ok" });
    expect(rangesCheck(0.8902, 0.9).line).toContain("89% of the readings in the test areas fall inside the model's prediction range, which should catch 90% of them");
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
    expect(distanceCheck(sa, "°F")!.line).toContain("The accuracy score is 0.86 with test areas 180 m across, and 0.74 with test areas 780 m across");
    expect(distanceCheck({ ...sa, far: 0.3 }, "°F")!.status).toBe("warn");
    expect(distanceCheck({ metric: "rmse", near: 0.4, far: 0.55, nearBlockM: 100, farBlockM: 800 }, "°F")!.status).toBe("ok");
    expect(distanceCheck({ metric: "rmse", near: 0.4, far: 0.7, nearBlockM: 100, farBlockM: 800 }, "°F")!.status).toBe("warn");
    expect(distanceCheck({ metric: "rmse", near: 0.4, far: 0.7, nearBlockM: 100, farBlockM: 800 }, "°F")!.line).toMatch(/^The typical error is 0\.40 °F with test areas 100 m across/);
    expect(distanceCheck(null, "°F")).toBeNull();
    expect(skillAcross({ ...viewFixture("distance").sections.curve!, series: [] })).toBeNull();
  });

  it("cause or coincidence: one line per lever from the causal audit", () => {
    const vm = viewFixture("causal");
    const canopy = vm.sections.treatments!.canopy as Treatment;
    const a = causalAgreement("canopy", canopy, vm.units.target, vm.units.levers);
    expect(a).toMatchObject({ lever: "canopy", status: "size" });
    expect(a.line).toBe("Trees & canopy: the check agrees it cools but finds a smaller effect (model 0.12 °F, check 0.04 °F cooler per percentage point of tree canopy).");
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
    expect(underTargetLine([band("0-100 m", 0.9), band("300+ m", 0.7)], 0.9)).toBe("At 300+ m from the nearest reading, the ranges catch fewer than 85% of readings (target: 90%).");
    // without the skill-vs-distance curve, the coverage by distance is a partial answer
    expect(distanceCoverageCheck([], 0.9)).toBeNull();
    expect(distanceCoverageCheck([band("0-100 m", null)], 0.9)).toBeNull();
    expect(distanceCoverageCheck([band("0-100 m", 0.9), band("100-300 m", 0.88)], 0.9)).toEqual({
      status: "partial",
      line: "Only the prediction ranges were checked at increasing distances from the readings, not the accuracy itself. They hold their 90% target at every distance checked.",
    });
    // a skipped test: the reason first, then what was checked instead
    expect(distanceCoverageCheck([band("240-330 m", 0.7)], 0.9, { why: "turned off in the project settings" })!.line).toBe(
      "The distance test, which re-checks accuracy farther from the readings, is turned off in the project settings, so only the prediction ranges were checked by distance. At 240–330 m from the nearest reading, they catch fewer than 85% of readings (target: 90%).",
    );
    expect(distanceCoverageCheck([band("300+ m", 0.7)], 0.9, { why: null })!.line).toMatch(/^The distance test, which re-checks accuracy farther from the readings, did not run, so only/);
    const v = (verdict: "robust" | "direction" | "not_established", label: string) => ({ verdict: { verdict, label, reasons: [], qualifiers: [] } });
    expect(verdictLine([{ verdict: null }])).toBeNull();
    // the glossary's plain words, not the server's technical labels
    expect(verdictLine([v("direction", "Direction only"), v("robust", "Robust"), v("robust", "Robust"), { verdict: null }])).toBe(
      "What the evidence supports, across 4 scenarios: 2 Solid, 1 Direction clear.",
    );
    expect(verdictLine([v("not_established", "Not established")])).toBe("What the evidence supports, across 1 scenario: 1 Not proven.");
  });

  it("TermText keeps the words and makes the first occurrence of each phrase a glossary Term", () => {
    const { container } = render(
      <p>
        <TermText text="The accuracy score is 0.8 and the typical error 0.3 °F; the accuracy score again." terms={[["typical error", "rmse"], ["accuracy score", "r2"], ["absent", "gini"]]} />
      </p>,
    );
    expect(container.textContent).toBe("The accuracy score is 0.8 and the typical error 0.3 °F; the accuracy score again.");
    const terms = [...container.querySelectorAll<HTMLElement>("button.term")];
    expect(terms.map((t) => [t.textContent, t.dataset.term])).toEqual([
      ["accuracy score", "r2"],
      ["typical error", "rmse"],
    ]);
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
      expect(grade.querySelector("[data-grade-line]")!.textContent).toBe("Predicts well in places it was not trained on. Its prediction ranges hold.");

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
      expect(check(r.container, "accuracy").textContent).toContain("the model's accuracy score is 0.82 (1 is perfect), and its typical error is 0.34 °F");
      // observed vs predicted, coverage by distance and skill vs block size are Model details' charts
      expect(charts(r.container)).toEqual([]);
      // every distance claim is the distance check's: the ranges check says nothing about distance
      expect(check(r.container, "ranges").textContent).not.toContain("from the nearest reading");
      expect(check(r.container, "distance").textContent).toContain("0.86 with test areas 180 m across, and 0.74 with test areas 780 m across");
      expect(check(r.container, "distance").querySelector("[data-distance-ranges]")!.textContent).toBe(
        "At 240–330 m from the nearest reading, the ranges catch fewer than 85% of readings (target: 90%).",
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

  it("explains each check and the grade in a glossary tooltip, with plain words in the lines and no native titles", async () => {
    const r = await renderTrust();
    try {
      expect(r.container.querySelector("[title]")).toBeNull();
      const tips = Object.fromEntries(
        [...r.container.querySelectorAll<HTMLElement>("[data-check]")].map((c) => {
          const tip = c.querySelector<HTMLElement>("header button.infotip")!;
          // beside the heading, never inside it
          expect(c.querySelector("h3 button")).toBeNull();
          return [c.dataset.check, tip.dataset.term];
        }),
      );
      expect(tips).toEqual({ accuracy: "r2", ranges: "coverage", distance: "distance_skill", causal: "causal_audit" });
      const r2tip = tipOf(check(r.container, "accuracy").querySelector("header button.infotip")!);
      expect(r2tip.querySelector(".tip-text")!.textContent).toBe(term("r2").tip);
      expect(r2tip.querySelector(".tip-tech code")!.textContent).toBe(term("r2").technical);
      expect(tipOf(check(r.container, "distance").querySelector("header button.infotip")!).querySelector(".tip-tech code")!.textContent).toBe(term("distance_skill").technical);
      // the plain phrases of the lines are glossary terms
      const lineTerms = (id: string) => [...check(r.container, id).querySelectorAll<HTMLElement>("[data-check-line] button.term")].map((t) => [t.textContent, t.dataset.term]);
      expect(lineTerms("accuracy")).toEqual([
        ["accuracy score", "r2"],
        ["typical error", "rmse"],
      ]);
      expect(lineTerms("ranges")).toEqual([
        ["test areas", "held_out_fold"],
        ["prediction range", "coverage"],
      ]);
      expect(tipOf(check(r.container, "accuracy").querySelector('button.term[data-term="rmse"]')!).querySelector(".tip-tech code")!.textContent).toBe(term("rmse").technical);
      // the grade: what it is, and its rule in plain words
      const grade = r.container.querySelector<HTMLElement>("#grade")!;
      expect(grade.querySelector<HTMLElement>("button.infotip")!.dataset.term).toBe("trust_grade");
      expect(grade.querySelector("[data-grade-rule]")!.textContent).toBe(
        "How the grade works: an accuracy score of 0.6 or more is Good, 0.4 or more Fair, lower Limited. If the prediction ranges miss their target by more than 5 percentage points, the grade drops one step.",
      );
      // the causal check's per-lever pills explain themselves
      const pill = check(r.container, "causal").querySelector("[data-lever] [data-tip-chip]")!;
      expect(tipOf(pill).textContent).toContain("disagree on by how much");
      // the independent checks in plain words, each with its glossary ⓘ (technical name: the card's)
      const items = [...r.container.querySelectorAll<HTMLElement>('[aria-label="Independent checks"] [data-check-kind]')];
      expect(items.map((i) => i.querySelector("strong")!.textContent)).toEqual([
        term("baselines").label,
        term("placebo").label,
        term("attribution").label,
        term("multiverse").label,
        term("reproduce").label,
        term("literature_check").label,
        term("benchmark").label,
      ]);
      expect(tipOf(items[1].querySelector("button.infotip")!).querySelector(".tip-tech code")!.textContent).toBe("Placebo tests");
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
      expect(sec.querySelector("[data-uncertainty-summary]")!.textContent).toContain("3 of 3 scenarios still cool when every source of doubt is counted");
      expect(sec.querySelector('[data-uncertainty-summary] button.term[data-term="envelope"]')!.textContent).toBe("every source of doubt");
      expect(sec.querySelector("[data-verdicts]")!.textContent).toBe("What the evidence supports, across 3 scenarios: 1 Solid.");
      // the plain verdict explains itself, with its technical name
      const solid = sec.querySelector<HTMLElement>('[data-verdicts] button.term[data-term="verdict_robust"]')!;
      expect(solid.textContent).toBe("Solid");
      expect(tipOf(solid).querySelector(".tip-tech code")!.textContent).toBe("Robust");
      expect(sec.querySelector('button.infotip[data-term="envelope"]')).not.toBeNull();
      expect(sec.querySelector("[data-scenario], table")).toBeNull();
      expect(sec.querySelector('a[data-details-link]')!.getAttribute("href")).toBe(`/r/${r.rid}/details/uncertainty`);

      const lim = r.container.querySelector('#limitations [aria-label="Limitations"]')!;
      expect([...lim.querySelectorAll("li")].map((l) => l.textContent)).toEqual(viewFixture("overview").sections.limitations);
      // the run's caveats have one home, the Summary's "Things to keep in mind": linked, not repeated
      const caveats = r.container.querySelector("#limitations [data-caveats-link]")!;
      expect(caveats.textContent).toBe("1 more point about this run is on the Summary, under Things to keep in mind.");
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
          { id: "cv_distance", label: "Skill vs distance", group: "trust", state: "missing", produced_by: "stage:cv_curve", view: "distance", action: { kind: "open", label: "Re-run with the distance test", method: "GET", path: `/p/p_1/launch?from=${rid}` } },
        ];
        return { [`GET /api/runs/${rid}`]: { body: detail } };
      },
    });
    try {
      const c = check(r.container, "distance");
      const button = await waitFor(() => [...c.querySelectorAll("button")].find((b) => b.textContent === "Re-run with the distance test"), 3000, "the re-run");
      // skill was not tested far out, but the ranges were: partly checked, from the coverage by distance
      expect(c.dataset.status).toBe("partial");
      expect(c.querySelector(".pill")!.textContent).toBe("Partly checked");
      // the reason first, then what was checked instead (never "only … were checked" before "did not run")
      expect(c.textContent).toMatch(
        /The distance test, which re-checks accuracy farther from the readings, is turned off in the project settings, so only the prediction ranges were checked by distance\. At 240–330 m from the nearest reading, they catch fewer than 85% of readings \(target: 90%\)\./,
      );
      expect(c.textContent).not.toContain("did not run");
      expect(check(r.container, "ranges").textContent).not.toContain("from the nearest reading");
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
