// Studies, exports and findings routes (SPEC §3.2, §6.7, §6.10, §8; owner frontend-studies-exports):
// the run's Validation tab (group Trust), the project Studies hub, the study page, Exports &
// reports, and the Findings notebook per project and across the workspace. Declares the project
// nav entries Studies, Exports and Findings (§3.1).
import { lazy } from "react";
import type { RouteDef } from "../../router";

const Findings = lazy(() => import("./Findings"));
const enc = encodeURIComponent;

export const routes: RouteDef[] = [
  {
    path: "/r/:rid/validation",
    component: lazy(() => import("./Validation")),
    title: "Validation",
    runTab: { id: "validation", label: "Validation", group: "Trust", order: 10 },
  },
  {
    path: "/p/:pid/studies",
    component: lazy(() => import("./StudiesHub")),
    title: "Studies",
    projectNav: { id: "studies", label: "Studies", order: 70, to: (pid) => `/p/${enc(pid)}/studies` },
  },
  { path: "/studies/:stid", component: lazy(() => import("./StudyPage")), title: "Study" },
  {
    path: "/p/:pid/exports",
    component: lazy(() => import("./Exports")),
    title: "Exports & reports",
    projectNav: { id: "exports", label: "Exports", order: 80, to: (pid) => `/p/${enc(pid)}/exports` },
  },
  {
    path: "/p/:pid/findings",
    component: Findings,
    title: "Findings",
    projectNav: { id: "findings", label: "Findings", order: 90, to: (pid) => `/p/${enc(pid)}/findings` },
  },
  { path: "/findings", component: Findings, title: "Findings" },
];

/** The run tab and project nav ids this module declares (tests check them against the registry). */
export const STUDIES_RUN_TABS = routes.flatMap((r) => (r.runTab ? [r.runTab.id] : []));
export const STUDIES_NAV_IDS = routes.flatMap((r) => (r.projectNav ? [r.projectNav.id] : []));
