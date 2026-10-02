// Setup wizard (SPEC §9.3) with fetch mocked against the api.md fixtures: the column
// suggestion pre-fills the mapping form, the dose-scale table marks doses beyond 1 sd amber,
// every step shows server issues at their dotted paths, missing physics roles raise the
// "placebo, simcheck, planner and emulator need these" warning, ladder names use U+2212, and
// saving sends If-Match with a 409 opening the conflict dialog.
import type { ReactElement } from "react";
import { afterEach, describe, expect, it } from "vitest";
import { useDrafts } from "../setup/store";
import { AboutStep } from "../setup/AboutStep";
import { AnalysisStep } from "../setup/AnalysisStep";
import { DataStep } from "../setup/DataStep";
import { LeversStep } from "../setup/LeversStep";
import { PhysicsStep } from "../setup/PhysicsStep";
import { ScenariosStep } from "../setup/ScenariosStep";
import Setup from "../Setup";
import { DoseScaleTable } from "../components/DoseScaleTable";
import { byText, click, flush, mockFetch, render, typeInto, waitFor } from "../../../test/render";
import { CHECK, configDoc, demoRaw, INSPECT, ISSUES, PID, SUGGESTION } from "../__fixtures__/api";
import { projectRoutes, renderAt, renderStep, resetAll, seedDraft } from "./helpers";

const MINUS = "−";

afterEach(() => resetAll());

/** The issue lines rendered inside the field bound to `path`. */
function issueText(root: ParentNode, path: string): string {
  const field = root.querySelector(`[data-cfg-path="${path}"]`);
  if (!field) throw new Error(`no field for ${path}`);
  return [...field.querySelectorAll(".issue")].map((e) => e.textContent).join(" | ");
}

function selectValue(root: ParentNode, path: string): string {
  const sel = root.querySelector<HTMLSelectElement>(`[data-cfg-path="${path}"] select`);
  if (!sel) throw new Error(`no select for ${path}`);
  return sel.value;
}

describe("data step: suggested mapping", () => {
  it("the column suggestion pre-fills the mapping form and badges the suggested fields", async () => {
    const blank = { data: { path: "data/city.csv" } };
    const m = mockFetch(
      projectRoutes({
        [`GET /api/projects/${PID}/config`]: { body: configDoc(blank, 1, "core:\n  data:\n    path: data/city.csv\n") },
        [`POST /api/projects/${PID}/columns/suggest`]: { body: SUGGESTION },
      }),
    );
    const { container } = renderAt(`/p/${PID}/setup/data`, <Setup />);
    await waitFor(() => container.querySelector('[data-cfg-path="data.target"] select'), 5000, "mapping form");
    // empty before the suggestion (x/y show their DEFAULTS)
    expect(selectValue(container, "data.target")).toBe("");
    click(byText(container, "button", "Suggest mapping"));
    await waitFor(() => selectValue(container, "data.target") === "T", 5000, "target pre-filled");
    expect(selectValue(container, "data.id")).toBe("id");
    expect(selectValue(container, "data.x")).toBe("x");
    expect(selectValue(container, "data.y")).toBe("y");
    expect(selectValue(container, "data.zone")).toBe("district");
    expect(selectValue(container, "data.coord_unit")).toBe("m");
    const badge = container.querySelector('[data-cfg-path="data.target"] .badge');
    expect(badge?.textContent).toBe("suggested");
    expect(badge?.getAttribute("title")).toContain("80%");
    expect(m.calls.some((c) => c.method === "POST" && c.url.endsWith("/columns/suggest") && (c.body as { path: string }).path === "data/city.csv")).toBe(true);
    // the draft now differs from the saved config: the save bar lists the sections
    const bar = container.querySelector('[aria-label="Unsaved changes"]')!;
    expect(bar.textContent).toContain("data");
    expect(bar.textContent).toContain("predictors");
    // the header preview lists the inspected columns
    expect(container.textContent).toContain("water_dist");
    m.restore();
  });

  it("Check data runs S0 on the draft and shows QA flags as badges and the grid stats", async () => {
    const m = mockFetch(projectRoutes({ [`POST /api/projects/${PID}/data/check`]: { body: { ...CHECK, preview_token: "" } } }));
    seedDraft();
    const { container } = renderStep(<DataStep />);
    await flush(4);
    click(byText(container, "button", "Check data"));
    await waitFor(() => container.querySelector('[aria-label="QA flags"]'), 5000, "flags");
    const flags = container.querySelector('[aria-label="QA flags"]')!.textContent!;
    expect(flags).toContain("72% of target values are whole degrees");
    expect(container.textContent).toContain("96 × 96");
    expect(container.textContent).toContain("9,216");
    const call = m.calls.find((c) => c.method === "POST" && c.url.endsWith("/data/check"))!;
    expect((call.body as { config_patch: { data: { target: string } } }).config_patch.data.target).toBe("T");
    m.restore();
  });
});

describe("levers step: dose scale", () => {
  it("DoseScaleTable marks doses beyond 1 sd amber, with words", () => {
    const { container } = render(<DoseScaleTable lever="canopy" entry={CHECK.dose_scale.canopy} unit="pp" />);
    const rows = [...container.querySelectorAll("tbody tr")];
    expect(rows).toHaveLength(6);
    const amber = rows.filter((r) => r.getAttribute("data-beyond-sd") === "true");
    expect(amber.map((r) => r.querySelector("td")!.textContent)).toEqual(["15", "20", "30", "40"]);
    for (const r of amber) {
      expect(r.className).toContain("amber");
      expect(r.textContent).toContain("beyond 1 sd");
    }
    expect(rows[1].textContent).toContain("0.85 sd");
    expect(rows[1].textContent).not.toContain("beyond");
    expect(rows[0].textContent).toContain("p63");
  });

  it("each lever shows its dose-scale table from the data check", async () => {
    const m = mockFetch(projectRoutes());
    seedDraft({ check: CHECK });
    const { container } = renderStep(<LeversStep />, `/p/${PID}/setup/levers`);
    await waitFor(() => container.querySelector('[data-lever="canopy"] .dose-table'), 5000, "dose table");
    expect(container.querySelectorAll('[data-lever="canopy"] tr[data-beyond-sd="true"]')).toHaveLength(4);
    expect(container.querySelectorAll('[data-lever="impervious"] tr[data-beyond-sd="true"]')).toHaveLength(0);
    // dose chips show the direction's sign (U+2212 for a decrease lever)
    expect(container.querySelector('[data-lever="impervious"] .chips')!.textContent).toContain(`${MINUS}10`);
    m.restore();
  });

  it("toggling a predictor into a lever writes a default actionable entry", async () => {
    const m = mockFetch(projectRoutes());
    seedDraft();
    const { container } = renderStep(<LeversStep />, `/p/${PID}/setup/levers`);
    await flush(4);
    const box = container.querySelector<HTMLInputElement>('[data-lever="elevation"] input[type="checkbox"]')!;
    expect(box.checked).toBe(false);
    click(box);
    const lever = (useDrafts.getState().drafts[PID].raw.actionable as Record<string, { min: number; max: number; doses: number[] }>).elevation;
    expect(lever.min).toBe(2);
    expect(lever.max).toBe(61);
    expect(lever.doses).toContain(0);
    m.restore();
  });
});

describe("issues at dotted paths", () => {
  it("every step shows server issues under the field of their path", async () => {
    const m = mockFetch(projectRoutes());
    seedDraft({ issues: ISSUES });
    const steps: [ReactElement, string, string][] = [
      [<DataStep key="d" />, "data.target", "data.target (the temperature column) is required"],
      [<DataStep key="d2" />, "data.crs", "objective 'people' needs data.crs"],
      [<LeversStep key="l" />, "actionable.canopy.doses", "dose 40 is beyond 3 sd"],
      [<PhysicsStep key="p" />, "physics.roles", "canopy role not mapped"],
      [<ScenariosStep key="s" />, "scenarios.1.variable", "'impervious_x' is not a lever"],
      [<AnalysisStep key="a" />, "climate.table", "climate table not found: x.csv"],
      [<AnalysisStep key="a2" />, "optimize.variable", "optimize.variable 'ndvi' is not a lever"],
      [<AboutStep key="b" />, "report.title", "No report title"],
    ];
    for (const [step, path, msg] of steps) {
      const r = renderStep(step);
      await flush(3);
      expect(issueText(r.container, path), path).toContain(msg);
      // error-level issues are prefixed so they read as errors without colour
      const issue = ISSUES.find((i) => i.path.startsWith(path));
      if (issue?.level === "error") expect(issueText(r.container, path)).toContain("Error: ");
      r.unmount();
    }
    m.restore();
  });

  it("the step summary lists the step's issues with their paths", async () => {
    const m = mockFetch(projectRoutes());
    seedDraft({ issues: ISSUES });
    const { container } = renderStep(<AnalysisStep />);
    await flush(3);
    const summary = container.querySelector(".step-issues")!;
    expect(summary.textContent).toContain("2 errors");
    expect(summary.textContent).toContain("climate.table");
    expect(summary.textContent).toContain("optimize.variable");
    m.restore();
  });

  it("missing physics roles show the warning naming who needs them", async () => {
    const m = mockFetch(projectRoutes());
    seedDraft({ issues: ISSUES });
    const { container } = renderStep(<PhysicsStep />, `/p/${PID}/setup/physics`);
    await flush(3);
    const warn = container.querySelector(".roles-warning")!;
    expect(warn.textContent).toContain("placebo, simcheck, planner and emulator need these");
    expect(warn.getAttribute("data-missing")).toBe("canopy");
    // the role-level server warning is shown at physics.roles
    expect(container.textContent).toContain("canopy role not mapped");
    // mapping the role clears the warning
    const sel = container.querySelector<HTMLSelectElement>('[data-cfg-path="physics.roles.canopy"] select')!;
    typeInto(sel, "canopy");
    expect(container.querySelector(".roles-warning")).toBeNull();
    m.restore();
  });
});

describe("scenarios step", () => {
  it("ladder names use U+2212 for decrease levers", async () => {
    const m = mockFetch(projectRoutes());
    seedDraft();
    const { container } = renderStep(<ScenariosStep />, `/p/${PID}/setup/scenarios`);
    await flush(3);
    const lists = [...container.querySelectorAll('[aria-label="Generated scenario names"]')].map((l) => [...l.querySelectorAll("li")].map((li) => li.textContent));
    expect(lists[0]).toEqual(["Canopy Increase +5", "Canopy Increase +10", "Canopy Increase +20"]);
    expect(lists[1]).toEqual([`Impervious Decrease ${MINUS}10`, `Impervious Decrease ${MINUS}20`]);
    expect(lists[1].join(" ")).not.toContain("-");
    const all = container.querySelector('[aria-label="Configured scenarios"]')!.textContent!;
    expect(all).toContain("impervious-decrease-minus-10");
    expect(all).toContain("Cooling package");
    m.restore();
  });
});

describe("saving the draft", () => {
  it("sends If-Match with the base version; a 409 opens the conflict dialog and Overwrite retries with the server version", async () => {
    const puts: { ifMatch: string | undefined; body: unknown }[] = [];
    let saved = demoRaw();
    const m = mockFetch(
      projectRoutes({
        // the server's copy after the successful overwrite is version 6
        [`GET /api/projects/${PID}/config`]: () => ({ body: configDoc(saved, puts.length >= 2 ? 6 : 3) }),
        [`PUT /api/projects/${PID}/config`]: (_u, init) => {
          const headers = init.headers as Record<string, string>;
          puts.push({ ifMatch: headers["If-Match"], body: JSON.parse(String(init.body)) });
          if (puts.length === 1) return { status: 409, body: { error: { code: "conflict", message: "stale version", detail: { current_version: 5 } } } };
          saved = (puts[1].body as { raw: Record<string, unknown> }).raw;
          return { body: { version: 6, issues: [], diff: "" } };
        },
      }),
    );
    const { container } = renderAt(`/p/${PID}/setup/about`, <Setup />);
    await waitFor(() => container.querySelector('[data-cfg-path="report.title"] input'), 5000, "about step");
    typeInto(container.querySelector('[data-cfg-path="report.title"] input'), "Demo heat study");
    const save = await waitFor(() => byText(container, "button", "Save config"), 2000, "save bar");
    click(save);
    await waitFor(() => document.querySelector(".conflict-dialog"), 5000, "conflict dialog");
    expect(puts[0].ifMatch).toBe("3");
    expect((puts[0].body as { raw: { report: { title: string } } }).raw.report.title).toBe("Demo heat study");
    expect(document.querySelector(".conflict-dialog")!.textContent).toContain("version 5");
    click(byText(document.body, ".conflict-dialog button", "Overwrite with mine"));
    await waitFor(() => puts.length === 2 && !document.querySelector(".conflict-dialog"), 5000, "overwrite");
    expect(puts[1].ifMatch).toBe("5");
    expect(useDrafts.getState().drafts[PID].baseVersion).toBe(6);
    m.restore();
  });

  it("the wizard tabs carry completion dots from readiness", async () => {
    const m = mockFetch(projectRoutes());
    const { container } = renderAt(`/p/${PID}/setup/levers`, <Setup />);
    await waitFor(() => container.querySelector('[role="tab"]'), 5000, "tabs");
    const dots = Object.fromEntries([...container.querySelectorAll<HTMLElement>(".step-tab")].map((t) => [t.textContent, t.dataset.stepDot]));
    expect(dots).toMatchObject({ Data: "ok", Levers: "ok", Physics: "warn", Inputs: "missing" });
    expect(container.querySelector('[role="tab"][aria-selected="true"]')!.textContent).toContain("Levers");
    expect(container.querySelector('[data-step="levers"]')).not.toBeNull();
    m.restore();
  });
});

// Keep the fixture honest: the inspect lists every demo predictor.
it("fixture: the header inspect covers the demo predictors", () => {
  const cols = INSPECT.columns.map((c) => c.name);
  for (const p of demoRaw().predictors as string[]) expect(cols).toContain(p);
});
