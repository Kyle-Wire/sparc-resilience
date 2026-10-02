// Tracking routes (SPEC §3.2, owner frontend-tracking): project overview with the Pipeline
// Status Board, run history, Activity, Mission Control, the run's Track tab and Settings.
// Declares the project nav entries Overview and Runs (§3.1) and the run tab "track".
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
    projectNav: { id: "runs", label: "Runs", order: 50, to: (pid) => `/p/${encodeURIComponent(pid)}/runs` },
  },
  { path: "/runs", component: lazy(() => import("./RunHistory")), title: "Run history" },
  { path: "/jobs", component: lazy(() => import("./Activity")), title: "Activity" },
  { path: "/jobs/:jid", component: lazy(() => import("./MissionControl")), title: "Mission Control", fullWidth: true },
  {
    path: "/r/:rid/track",
    component: lazy(() => import("./RunTrack")),
    title: "Track",
    fullWidth: true,
    runTab: { id: "track", label: "Track", group: "Run", order: 10 },
  },
  { path: "/settings", component: lazy(() => import("./Settings")), title: "Settings" },
];
