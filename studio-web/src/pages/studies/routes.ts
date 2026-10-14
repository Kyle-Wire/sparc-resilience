// Studies, exports and findings routes (SPEC §3.2, §6.7, §6.10, §8; owner frontend-studies-exports):
// the project Studies hub, the study page, Reports (`/p/:pid/exports`: the tabs "Reports &
// exports" and "Notes" = the Findings notebook, `?tab=notes`) and the Findings notebook across
// the workspace. Declares the project nav entries Reports and, in the collapsed Advanced group,
// Studies (§3.1). The run's Validation page is part of How much to trust this (runhub Trust);
// `/r/:rid/validation` and `/p/:pid/findings` redirect (../legacy/routes.ts).
import { lazy } from "react";
import type { RouteDef } from "../../router";

const enc = encodeURIComponent;

export const routes: RouteDef[] = [
  {
    path: "/p/:pid/exports",
    component: lazy(() => import("./Reports")),
    title: "Reports",
    projectNav: { id: "reports", label: "Reports", order: 50, to: (pid) => `/p/${enc(pid)}/exports` },
  },
  {
    path: "/p/:pid/studies",
    component: lazy(() => import("./StudiesHub")),
    title: "Studies",
    projectNav: { id: "studies", label: "Studies", order: 70, group: "advanced", to: (pid) => `/p/${enc(pid)}/studies` },
  },
  { path: "/studies/:stid", component: lazy(() => import("./StudyPage")), title: "Study" },
  { path: "/findings", component: lazy(() => import("./Findings")), title: "Findings" },
];

/** The run tab and project nav ids this module declares (tests check them against the registry). */
export const STUDIES_RUN_TABS = routes.flatMap((r) => (r.runTab ? [r.runTab.id] : []));
export const STUDIES_NAV_IDS = routes.flatMap((r) => (r.projectNav ? [r.projectNav.id] : []));
