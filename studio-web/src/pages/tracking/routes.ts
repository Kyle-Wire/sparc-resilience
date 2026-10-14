// Tracking routes (SPEC §3.2, owner frontend-tracking): project overview, run history (with the
// project's Pipeline Status Board), Activity, Mission Control and Settings. Declares the project nav
// entries Overview, Runs and (in the collapsed Advanced group) Activity (§3.1). The run's Track
// tab is now Model details › Run log (runhub Details renders ./RunTrack); `/r/:rid/track`
// redirects (../legacy/routes.ts).
import { lazy } from "react";
import type { RouteDef } from "../../router";

export const routes: RouteDef[] = [
  {
    path: "/p/:pid",
    component: lazy(() => import("./ProjectOverview")),
    title: "Overview",
    projectNav: { id: "overview", label: "Overview", order: 10, to: (pid) => `/p/${encodeURIComponent(pid)}` },
  },
  {
    path: "/p/:pid/runs",
    component: lazy(() => import("./RunHistory")),
    title: "Runs",
    projectNav: { id: "runs", label: "Runs", order: 30, to: (pid) => `/p/${encodeURIComponent(pid)}/runs` },
  },
  { path: "/runs", component: lazy(() => import("./RunHistory")), title: "Run history" },
  {
    path: "/jobs",
    component: lazy(() => import("./Activity")),
    title: "Activity",
    projectNav: { id: "activity", label: "Activity", order: 90, group: "advanced", to: () => "/jobs", claims: ["/jobs/:jid"] },
  },
  { path: "/jobs/:jid", component: lazy(() => import("./MissionControl")), title: "Mission Control", fullWidth: true },
  { path: "/settings", component: lazy(() => import("./Settings")), title: "Settings" },
];
