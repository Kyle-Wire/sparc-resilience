// Run hub routes (SPEC §3.2, §6.4, §6.5, §6.8): the run tabs owned by frontend-run-hub, each
// declaring its `runTab` (group and order inside Model · Effects · Decisions · Trust · Run),
// plus the project-level Compare runs page. Other items own track, lab and validation.
import { lazy } from "react";
import type { RouteDef } from "../../router";

export const routes: RouteDef[] = [
  { path: "/r/:rid", component: lazy(() => import("./Overview")), title: "Run overview", runTab: { id: "overview", label: "Overview", group: "Model", order: 10 } },
  { path: "/r/:rid/data", component: lazy(() => import("./DataQa")), title: "Data & QA", runTab: { id: "data", label: "Data", group: "Model", order: 20 } },
  { path: "/r/:rid/accuracy", component: lazy(() => import("./Accuracy")), title: "Accuracy", runTab: { id: "accuracy", label: "Accuracy", group: "Model", order: 30 } },
  {
    path: "/r/:rid/distance",
    component: lazy(() => import("./Distance")),
    title: "Distance & baselines",
    runTab: { id: "distance", label: "Distance", group: "Model", order: 40 },
  },
  { path: "/r/:rid/influence", component: lazy(() => import("./Influence")), title: "Influence", runTab: { id: "influence", label: "Influence", group: "Model", order: 50 } },
  { path: "/r/:rid/response", component: lazy(() => import("./Response")), title: "Response", runTab: { id: "response", label: "Response", group: "Effects", order: 10 } },
  { path: "/r/:rid/causal", component: lazy(() => import("./Causal")), title: "Causal audit", runTab: { id: "causal", label: "Causal", group: "Effects", order: 20 } },
  { path: "/r/:rid/heat", component: lazy(() => import("./Heat")), title: "Heat brief", runTab: { id: "heat", label: "Heat", group: "Decisions", order: 5 } },
  {
    path: "/r/:rid/scenarios",
    component: lazy(() => import("./Scenarios")),
    title: "Configured scenarios",
    runTab: { id: "scenarios", label: "Scenarios", group: "Decisions", order: 10 },
  },
  { path: "/r/:rid/climate", component: lazy(() => import("./Climate")), title: "Climate", runTab: { id: "climate", label: "Climate", group: "Decisions", order: 20 } },
  { path: "/r/:rid/budget", component: lazy(() => import("./Budget")), title: "Budget", runTab: { id: "budget", label: "Budget", group: "Decisions", order: 30 } },
  { path: "/r/:rid/planner", component: lazy(() => import("./Planner")), title: "Planner pack", runTab: { id: "planner", label: "Planner", group: "Decisions", order: 40 } },
  {
    path: "/r/:rid/uncertainty",
    component: lazy(() => import("./Uncertainty")),
    title: "Uncertainty",
    runTab: { id: "uncertainty", label: "Uncertainty", group: "Trust", order: 20 },
  },
  { path: "/r/:rid/provenance", component: lazy(() => import("./Provenance")), title: "Provenance", runTab: { id: "provenance", label: "Provenance", group: "Trust", order: 30 } },
  { path: "/r/:rid/map", component: lazy(() => import("./MapTab")), title: "Map explorer", fullWidth: true, runTab: { id: "map", label: "Map", group: "Run", order: 20 } },
  { path: "/r/:rid/docs/:doc?", component: lazy(() => import("./Docs")), title: "Documents", runTab: { id: "docs", label: "Docs", group: "Run", order: 30 } },
  { path: "/r/:rid/files", component: lazy(() => import("./Files")), title: "Files", runTab: { id: "files", label: "Files", group: "Run", order: 40 } },
  { path: "/p/:pid/compare", component: lazy(() => import("./Compare")), title: "Compare runs" },
];

/** The run tab ids this module declares (tests check them against the registry). */
export const RUN_HUB_TABS = routes.flatMap((r) => (r.runTab ? [r.runTab.id] : []));
