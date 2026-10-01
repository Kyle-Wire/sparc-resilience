// Home and the project list (SPEC §3.2, J1, J10): project cards with readiness and last run,
// the quickstarts (synthetic city in one click, Providence example, blank project, import
// with the pickle-trust confirmation), running jobs, and archive/delete on /projects.
import { afterEach, describe, expect, it } from "vitest";
import { useJobs } from "../../../stores/jobs";
import { byText, click, flush, mockFetch, typeInto, waitFor } from "../../../test/render";
import Home, { uniqueName } from "../Home";
import Projects from "../Projects";
import { job, project, projectDetail } from "../__fixtures__/api";
import { projectRoutes, renderAt, resetAll } from "./helpers";

afterEach(() => resetAll());

const PROJECTS = [
  project(),
  project({
    id: "p_prov",
    slug: "providence",
    name: "Providence",
    template: "providence_example",
    demo: false,
    n_runs: 2,
    active_run_id: "20261001-142233-full-a1b2",
    last_run: { id: "20261001-142233-full-a1b2", status: "complete", created_utc: "2026-10-01T14:22:33Z", r2: 0.874 },
    readiness_score: { done: 11, total: 11 },
  }),
];

function homeRoutes(extra = {}) {
  return projectRoutes({
    "GET /api/projects": { body: PROJECTS },
    "GET /api/storage": { body: { workspace_bytes: 2.4e9, free_bytes: 54e9, cache: [], runs: [], jobs_bytes: 0 } },
    ...extra,
  });
}

describe("home", () => {
  it("names a new project uniquely", () => {
    expect(uniqueName("Synthetic city", PROJECTS)).toBe("Synthetic city 2");
    expect(uniqueName("Providence", [{ name: "providence" }, { name: "Providence 2" }])).toBe("Providence 3");
    expect(uniqueName("Fresh", PROJECTS)).toBe("Fresh");
  });

  it("shows project cards with readiness, last run and active run, the quickstarts and running jobs", async () => {
    useJobs.getState().setActive([job({ status: "running", progress: 0.42, label: "Fast run of Synthetic city" })]);
    const m = mockFetch(homeRoutes());
    const { container } = renderAt("/", <Home />);
    await waitFor(() => container.querySelector('[data-project="p_prov"]'), 5000, "cards");
    const prov = container.querySelector('[data-project="p_prov"]')!;
    expect(prov.textContent).toContain("Readiness 11/11");
    expect(prov.textContent).toContain("R² 0.874");
    expect(prov.querySelector('a[href="/r/20261001-142233-full-a1b2"]')).not.toBeNull();
    expect(container.querySelector('[data-project="p_demo"]')!.textContent).toContain("DEMO");
    const quick = container.querySelector('[aria-label="Quickstarts"]')!.textContent!;
    for (const t of ["Try a synthetic city", "Open Providence example", "Import a config / run folder", "New blank project"]) expect(quick).toContain(t);
    expect(container.querySelector(".jobstrip")!.textContent).toContain("Fast run of Synthetic city");
    expect(container.querySelector('[aria-label="Disk and engine"]')!.textContent).toContain("54 GB free");
    m.restore();
  });

  it("Try a synthetic city creates the demo project in one click and opens it", async () => {
    const posts: unknown[] = [];
    const m = mockFetch(
      homeRoutes({
        "POST /api/projects": (_u: URL, init: RequestInit) => {
          posts.push(JSON.parse(String(init.body)));
          return { status: 201, body: { project: project({ id: "p_new", name: "Synthetic city 2" }), imported_runs: [], warnings: [] } };
        },
      }),
    );
    const { container } = renderAt("/", <Home />);
    await waitFor(() => container.querySelector('[data-project="p_demo"]'), 5000, "cards");
    click(byText(container, '[aria-label="Quickstarts"] button', "Try a synthetic city"));
    await waitFor(() => window.location.pathname === "/p/p_new", 5000, "navigated");
    expect(posts).toEqual([{ name: "Synthetic city 2", template: "synthetic_demo" }]);
    m.restore();
  });

  it("New blank project asks for a name and opens the data step", async () => {
    const posts: unknown[] = [];
    const m = mockFetch(
      homeRoutes({
        "POST /api/projects": (_u: URL, init: RequestInit) => {
          posts.push(JSON.parse(String(init.body)));
          return { status: 201, body: { project: project({ id: "p_blank", name: "Harbour", template: "blank" }), imported_runs: [], warnings: [] } };
        },
      }),
    );
    const { container } = renderAt("/", <Home />);
    await waitFor(() => container.querySelector('[data-project="p_demo"]'), 5000, "cards");
    click(byText(container, '[aria-label="Quickstarts"] button', "New blank project"));
    const input = await waitFor(() => document.querySelector<HTMLInputElement>('.dialog input[aria-label="Project name"]'), 2000, "dialog");
    typeInto(input, "Harbour");
    click(byText(document.body, ".dialog button", "Create and set up"));
    await waitFor(() => window.location.pathname === "/p/p_blank/setup/data", 5000, "setup");
    expect(posts).toEqual([{ name: "Harbour", template: "blank" }]);
    m.restore();
  });

  it("importing with pickle trust asks for confirmation naming the risk first", async () => {
    const posts: unknown[] = [];
    const m = mockFetch(
      homeRoutes({
        "POST /api/projects/import": (_u: URL, init: RequestInit) => {
          posts.push(JSON.parse(String(init.body)));
          return { status: 201, body: { project: project({ id: "p_imp", name: "City" }), runs: [], studies: [], warnings: [] } };
        },
      }),
    );
    const { container } = renderAt("/", <Home />);
    await waitFor(() => container.querySelector('[data-project="p_demo"]'), 5000, "cards");
    click(byText(container, '[aria-label="Quickstarts"] button', "Import…"));
    typeInto(await waitFor(() => document.querySelector('.dialog input[aria-label="Config path"]'), 2000, "dialog"), "/home/me/city/core.yml");
    typeInto(document.querySelector('.dialog textarea[aria-label="Run folders"]'), "/home/me/out/run1\n/home/me/out/run2\n");
    click(document.querySelector(".trust-pickles input"));
    click(byText(document.body, ".dialog button", "Import"));
    await flush(2);
    // nothing sent yet: the confirmation names the risk
    expect(posts).toEqual([]);
    const confirm = document.querySelector(".dialog")!;
    expect(confirm.textContent).toContain("Checkpoint files execute code when loaded; import only folders you produced.");
    click(byText(document.body, ".dialog button", "I produced these folders: import"));
    await waitFor(() => window.location.pathname === "/p/p_imp", 5000, "imported");
    expect(posts).toEqual([{ config_path: "/home/me/city/core.yml", copy_data: false, run_dirs: ["/home/me/out/run1", "/home/me/out/run2"], trust_pickles: true }]);
    m.restore();
  });

  it("Open Providence example passes the import-existing-runs option", async () => {
    const posts: unknown[] = [];
    const m = mockFetch(
      homeRoutes({
        "POST /api/projects": (_u: URL, init: RequestInit) => {
          posts.push(JSON.parse(String(init.body)));
          return { status: 201, body: { project: project({ id: "p_p2", name: "Providence 2" }), imported_runs: ["20261001-142233-full-a1b2"], warnings: ["run import unavailable in this build"] } };
        },
      }),
    );
    const { container } = renderAt("/", <Home />);
    await waitFor(() => container.querySelector('[data-project="p_demo"]'), 5000, "cards");
    click(byText(container, '[aria-label="Quickstarts"] button', "Open Providence example"));
    const input = await waitFor(() => document.querySelector<HTMLInputElement>('.dialog input[aria-label="Project name"]'), 2000, "dialog");
    expect(input.value).toBe("Providence 2");
    click(byText(document.body, ".dialog button", "Create"));
    await waitFor(() => window.location.pathname === "/p/p_p2", 5000, "opened");
    expect(posts).toEqual([{ name: "Providence 2", template: "providence_example", options: { import_existing_runs: true } }]);
    m.restore();
  });
});

describe("projects list", () => {
  it("lists projects sortably and archives / deletes them", async () => {
    const calls: { method: string; url: string; body: unknown }[] = [];
    const m = mockFetch(
      projectRoutes({
        "GET /api/projects": { body: PROJECTS },
        "PATCH /api/projects/p_demo": (_u: URL, init: RequestInit) => {
          calls.push({ method: "PATCH", url: "/api/projects/p_demo", body: JSON.parse(String(init.body)) });
          return { body: { ...project(), archived: true } };
        },
        "DELETE /api/projects/p_prov": (u: URL) => {
          calls.push({ method: "DELETE", url: u.pathname + u.search, body: null });
          return { body: { ok: true } };
        },
        "GET /api/projects/p_prov": { body: projectDetail({ id: "p_prov" }) },
      }),
    );
    const { container } = renderAt("/projects", <Projects />);
    await waitFor(() => container.querySelectorAll("tbody tr").length === 2, 5000, "rows");
    expect(container.textContent).toContain("11/11");
    const demoRow = byText(container, "tbody tr", "Synthetic city")!;
    click(byText(demoRow, "button", "Archive"));
    await waitFor(() => calls.length === 1, 5000, "archive");
    expect(calls[0]).toEqual({ method: "PATCH", url: "/api/projects/p_demo", body: { archived: true } });
    const provRow = byText(container, "tbody tr", "Providence")!;
    click(byText(provRow, "button", "Delete"));
    click(await waitFor(() => document.querySelector(".dialog input[type=checkbox]"), 2000, "confirm"));
    click(byText(document.body, ".dialog button", "Delete"));
    await waitFor(() => calls.length === 2, 5000, "delete");
    expect(calls[1]).toEqual({ method: "DELETE", url: "/api/projects/p_prov?files=true", body: null });
    m.restore();
  });
});
