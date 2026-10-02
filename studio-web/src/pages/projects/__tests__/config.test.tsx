// Config editor (SPEC §9.4) with fetch mocked against the api.md fixtures: issues land in the
// gutter at their YAML line, the impact preview shows before the save, the save sends
// If-Match with the base version, and a 409 opens the conflict dialog. The page says that
// comments are not preserved.
import { afterEach, describe, expect, it } from "vitest";
import { byText, click, flush, mockFetch, typeInto, waitFor } from "../../../test/render";
import ConfigEditor, { lineIssues } from "../ConfigEditor";
import { configDoc, DEMO_YAML, IMPACT, PID } from "../__fixtures__/api";
import { projectRoutes, renderAt, resetAll } from "./helpers";

afterEach(() => resetAll());

const EDITED = DEMO_YAML.replace("epochs: 200", "epochs: 300");

describe("config editor", () => {
  it("places validation issues in the gutter by dotted path", () => {
    const marks = lineIssues(DEMO_YAML, [
      { level: "error", path: "data.target", code: "x", message: "bad target" },
      { level: "warn", path: "stacker.tune_lambda.1", code: "y", message: "odd lambda" },
      { level: "info", path: "nowhere.at.all", code: "z", message: "unknown key" },
    ], { line: 7, column: 3, message: "YAML: mapping values are not allowed" });
    expect(marks.map((m) => [m.line, m.level])).toEqual([
      [5, "error"],
      [25, "warn"],
      [1, "info"],
      [7, "error"],
    ]);
  });

  it("says comments are not preserved and lists the changes from DEFAULTS", async () => {
    const m = mockFetch(projectRoutes());
    const { container } = renderAt(`/p/${PID}/config`, <ConfigEditor />);
    await waitFor(() => container.querySelector("textarea"), 5000, "editor");
    expect(container.querySelector(".comments-notice")!.textContent).toContain("Comments are not preserved");
    click(byText(container, '[role="tab"]', "vs DEFAULTS"));
    const table = container.querySelector('[aria-label="Changed from DEFAULTS"]')!;
    expect(table.textContent).toContain("cv.n_folds");
    expect(table.textContent).toContain("stacker.epochs");
    await flush(2);
    m.restore();
  });

  it("shows server issues in the gutter after validating the text", async () => {
    const m = mockFetch(
      projectRoutes({
        [`POST /api/projects/${PID}/config/validate`]: { body: { ok: false, issues: [{ level: "error", path: "actionable.canopy.doses.2", code: "dose", message: "dose out of bounds" }], fast_overrides: {}, coarse_preview: null } },
      }),
    );
    const { container } = renderAt(`/p/${PID}/config`, <ConfigEditor />);
    await waitFor(() => container.querySelector('.gutter [data-level="error"]'), 5000, "gutter mark");
    const mark = container.querySelector<HTMLElement>('.gutter [data-level="error"]')!;
    expect(mark.textContent).toContain("20");
    expect(mark.title).toContain("dose out of bounds");
    m.restore();
  });

  it("save sends If-Match after the impact preview, and a 409 shows the conflict dialog", async () => {
    const puts: { ifMatch: string | undefined; body: { yaml?: string; note?: string } }[] = [];
    const m = mockFetch(
      projectRoutes({
        [`GET /api/projects/${PID}/config`]: { body: configDoc(undefined, 3) },
        [`POST /api/projects/${PID}/config/impact`]: { body: IMPACT },
        [`PUT /api/projects/${PID}/config`]: (_u, init) => {
          puts.push({ ifMatch: (init.headers as Record<string, string>)["If-Match"], body: JSON.parse(String(init.body)) });
          return { status: 409, body: { error: { code: "conflict", message: "The config was saved by someone else", detail: { current_version: 4 } } } };
        },
      }),
    );
    const { container } = renderAt(`/p/${PID}/config`, <ConfigEditor />);
    const ta = await waitFor(() => container.querySelector("textarea"), 5000, "editor");
    expect(byText(container, "button", "Review and save")!.hasAttribute("disabled")).toBe(true);
    typeInto(ta, EDITED);
    click(byText(container, "button", "Review and save"));
    // impact preview before the save
    await waitFor(() => document.querySelector('[aria-label="Impact of this change"]'), 5000, "impact");
    const impact = document.querySelector('[aria-label="Impact of this change"]')!.textContent!;
    expect(impact).toContain("2 existing runs");
    expect(impact).toContain("would need a refit");
    expect(impact).toContain("stacker");
    typeInto(document.querySelector('input[aria-label="Save note"]'), "more epochs");
    click(byText(document.body, ".dialog button", "Save version 4"));
    await waitFor(() => document.querySelector(".conflict-dialog"), 5000, "conflict dialog");
    expect(puts).toHaveLength(1);
    expect(puts[0].ifMatch).toBe("3");
    expect(puts[0].body.yaml).toContain("epochs: 300");
    expect(puts[0].body.note).toBe("more epochs");
    const dialog = document.querySelector(".conflict-dialog")!;
    expect(dialog.textContent).toContain("based on version 3");
    expect(dialog.textContent).toContain("version 4 was saved since");
    // the edited text is kept while the dialog decides
    expect((container.querySelector("textarea") as HTMLTextAreaElement).value).toContain("epochs: 300");
    m.restore();
  });

  it("Overwrite resends with the server's version and keeps the new version", async () => {
    const ifMatch: string[] = [];
    let version = 3;
    const m = mockFetch(
      projectRoutes({
        [`GET /api/projects/${PID}/config`]: () => ({ body: configDoc(undefined, version, version === 3 ? DEMO_YAML : EDITED) }),
        [`POST /api/projects/${PID}/config/impact`]: { body: { changed_sections: ["stacker"], runs: [] } },
        [`PUT /api/projects/${PID}/config`]: (_u, init) => {
          ifMatch.push((init.headers as Record<string, string>)["If-Match"]);
          if (ifMatch.length === 1) return { status: 409, body: { error: { code: "conflict", message: "stale", detail: { current_version: 4 } } } };
          version = 5;
          return { body: { version: 5, issues: [], diff: "" } };
        },
      }),
    );
    const { container } = renderAt(`/p/${PID}/config`, <ConfigEditor />);
    const ta = await waitFor(() => container.querySelector("textarea"), 5000, "editor");
    typeInto(ta, EDITED);
    click(byText(container, "button", "Review and save"));
    await waitFor(() => document.querySelector('[aria-label="Impact of this change"]'), 5000, "impact");
    expect(document.querySelector('[aria-label="Impact of this change"]')!.textContent).toContain("No existing run is affected");
    click(byText(document.body, ".dialog button", "Save version"));
    await waitFor(() => document.querySelector(".conflict-dialog"), 5000, "conflict");
    click(byText(document.body, ".conflict-dialog button", "Overwrite with mine"));
    await waitFor(() => ifMatch.length === 2 && !document.querySelector(".conflict-dialog"), 5000, "overwrite");
    expect(ifMatch).toEqual(["3", "4"]);
    await waitFor(() => container.querySelector(".badge")?.textContent === "v5", 5000, "version badge");
    expect([...container.querySelectorAll(".badge")].map((b) => b.textContent)).not.toContain("unsaved");
    m.restore();
  });
});
