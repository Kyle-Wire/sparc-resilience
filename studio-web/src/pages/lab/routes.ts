// Scenario Lab routes (SPEC §3.2, §7; owner frontend-lab): Design (editor, map, live compile,
// compare tray), the scenario + result page, the Library, budget Plans, Sweeps, Climate ×
// adaptation and Compare. Declares the run tab "lab" (last in the Decisions group) and the
// emphasised project nav entry "Scenario Lab" (§3.1).
import { lazy } from "react";
import type { Project } from "../../api/types";
import type { RouteDef } from "../../router";

const Design = lazy(() => import("./Design"));
const Plans = lazy(() => import("./Plans"));

/** The run the project nav's Scenario Lab entry opens: the active run, else the latest run. */
export function labRunOf(project: Project): string | null {
  return project.active_run_id ?? project.last_run?.id ?? null;
}

/** Why the Scenario Lab entry is greyed out (null when it can open). */
export function labDisabledReason(project: Project): string | null {
  const rid = labRunOf(project);
  if (!rid) return "No run yet. Launch a run first: the Lab needs a run with a checkpoint.";
  const st = project.last_run && project.last_run.id === rid ? project.last_run.status : null;
  if (!project.active_run_id && (st === "queued" || st === "running")) return "The first run is still running. The Lab opens once it has saved a checkpoint (after S3).";
  return null;
}

export const routes: RouteDef[] = [
  {
    path: "/r/:rid/lab",
    component: Design,
    title: "Scenario Lab",
    fullWidth: true,
    runTab: { id: "lab", label: "Lab", group: "Decisions", order: 50 },
    projectNav: {
      id: "lab",
      label: "Scenario Lab",
      order: 60,
      to: (pid, project) => {
        const rid = labRunOf(project);
        // Without a run the entry is greyed out (never a link); its placeholder target is no
        // other entry's prefix, so it never claims "current" for another page.
        return rid ? `/r/${encodeURIComponent(rid)}/lab` : `/p/${encodeURIComponent(pid)}/lab`;
      },
      disabledReason: labDisabledReason,
    },
  },
  { path: "/r/:rid/lab/s/:sid", component: Design, title: "Scenario", fullWidth: true },
  { path: "/r/:rid/lab/library", component: lazy(() => import("./Library")), title: "Scenario library", fullWidth: true },
  { path: "/r/:rid/lab/plans", component: Plans, title: "Budget plans", fullWidth: true },
  { path: "/r/:rid/lab/plans/:plid", component: Plans, title: "Budget plan", fullWidth: true },
  { path: "/r/:rid/lab/sweeps/:swid?", component: lazy(() => import("./Sweeps")), title: "Sweeps", fullWidth: true },
  { path: "/r/:rid/lab/climate", component: lazy(() => import("./Climate")), title: "Climate × adaptation", fullWidth: true },
  { path: "/r/:rid/lab/compare", component: lazy(() => import("./Compare")), title: "Compare scenarios", fullWidth: true },
];

/** The run tab and project nav ids this module declares (tests check them against the registry). */
export const LAB_RUN_TABS = routes.flatMap((r) => (r.runTab ? [r.runTab.id] : []));
export const LAB_NAV_IDS = routes.flatMap((r) => (r.projectNav ? [r.projectNav.id] : []));
