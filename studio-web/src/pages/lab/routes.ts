// Scenario Lab routes (SPEC §3.2, §7; owner frontend-lab): Design (editor, map, live compile,
// compare tray), the scenario + result page, the Library (home of the run's configured
// scenarios), budget Plans (home of the run's budget plan), Compare and, under More tools, Try
// amounts (sweeps, `/r/:rid/lab/sweeps`). Declares the
// run tab "lab" (Scenario Lab, between Heat risk and How much to trust this; its dot covers the
// lab, scenarios and budget views) and the emphasised project nav entry "Scenario Lab" (§3.1).
// Climate × adaptation moved to Heat risk › Future climate (`/r/:rid/lab/climate` redirects,
// ../legacy/routes.ts).
import { lazy } from "react";
import type { Project } from "../../api/types";
import type { RouteDef } from "../../router";

const Design = lazy(() => import("./Design"));
const Plans = lazy(() => import("./Plans"));

/** The run the project nav's Scenario Lab entry opens: the active run, else the latest run. */
export function labRunOf(project: Project): string | null {
  return project.active_run_id ?? project.last_run?.id ?? null;
}

/** Why the Scenario Lab entry is greyed out (null when it can open): only when the project has
 *  no run yet. A run without a checkpoint still opens the Lab (its Library, Plans and Compare
 *  work; Design says so in its engine line). */
export function labDisabledReason(project: Project): string | null {
  return labRunOf(project) ? null : "No run yet. Launch a run first: the Lab opens on a run.";
}

export const routes: RouteDef[] = [
  {
    path: "/r/:rid/lab",
    component: Design,
    title: "Scenario Lab",
    fullWidth: true,
    hideEngine: true,
    runTab: { id: "lab", label: "Scenario Lab", order: 50, views: ["lab", "scenarios", "budget"] },
    projectNav: {
      id: "lab",
      label: "Scenario Lab",
      order: 40,
      to: (pid, project) => {
        const rid = labRunOf(project);
        // Without a run the entry is greyed out (never a link); its placeholder target is no
        // other entry's prefix, so it never claims "current" for another page.
        return rid ? `/r/${encodeURIComponent(rid)}/lab` : `/p/${encodeURIComponent(pid)}/lab`;
      },
      disabledReason: labDisabledReason,
    },
  },
  { path: "/r/:rid/lab/s/:sid", component: Design, title: "Scenario", fullWidth: true, hideEngine: true },
  { path: "/r/:rid/lab/library", component: lazy(() => import("./Library")), title: "Scenario library", fullWidth: true },
  { path: "/r/:rid/lab/plans", component: Plans, title: "Budget plans", fullWidth: true },
  { path: "/r/:rid/lab/plans/:plid", component: Plans, title: "Budget plan", fullWidth: true },
  { path: "/r/:rid/lab/sweeps/:swid?", component: lazy(() => import("./Sweeps")), title: "Try amounts", fullWidth: true },
  { path: "/r/:rid/lab/compare", component: lazy(() => import("./Compare")), title: "Compare scenarios", fullWidth: true },
];

/** The run tab and project nav ids this module declares (tests check them against the registry). */
export const LAB_RUN_TABS = routes.flatMap((r) => (r.runTab ? [r.runTab.id] : []));
export const LAB_NAV_IDS = routes.flatMap((r) => (r.projectNav ? [r.projectNav.id] : []));
