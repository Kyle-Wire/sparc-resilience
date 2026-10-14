// Run hub routes (SPEC §3.2, §6.4, §6.5, §6.8; phase 2 information architecture): the flat run
// tab bar's planner tabs Summary · Map · What cools · Heat risk · How much to trust this (the
// Scenario Lab tab is declared by the lab module) and the quieter Model details tab, each a
// composer of the existing pages (./Summary, ./Effects, ./HeatRisk, ./Trust, ./Details), plus
// the project-level Compare runs page. Every old run tab URL redirects (../legacy/routes.ts).
// `views` are the server view ids each tab shows; its status dot aggregates theirs.
import { lazy } from "react";
import type { RouteDef } from "../../router";

const HeatRisk = lazy(() => import("./HeatRisk"));
const Details = lazy(() => import("./Details"));

export const routes: RouteDef[] = [
  {
    path: "/r/:rid",
    component: lazy(() => import("./Summary")),
    title: "Summary",
    runTab: { id: "summary", label: "Summary", order: 10, views: ["overview", "heat", "climate"] },
  },
  { path: "/r/:rid/map", component: lazy(() => import("./MapTab")), title: "Map explorer", fullWidth: true, runTab: { id: "map", label: "Map", order: 20, views: ["map"] } },
  {
    path: "/r/:rid/effects",
    component: lazy(() => import("./Effects")),
    title: "What cools",
    runTab: { id: "effects", label: "What cools", order: 30, views: ["response", "influence", "causal"] },
  },
  {
    path: "/r/:rid/heat/:section?",
    component: HeatRisk,
    title: "Heat risk",
    runTab: { id: "heat", label: "Heat risk", order: 40, views: ["heat", "climate", "planner"] },
  },
  {
    path: "/r/:rid/trust",
    component: lazy(() => import("./Trust")),
    title: "How much to trust this",
    runTab: { id: "trust", label: "How much to trust this", order: 60, views: ["accuracy", "validation", "uncertainty", "distance"] },
  },
  {
    path: "/r/:rid/details/:section?",
    component: Details,
    title: "Model details",
    runTab: {
      id: "details",
      label: "Model details",
      order: 100,
      advanced: true,
      views: ["data", "accuracy", "distance", "influence", "response", "causal", "uncertainty", "track", "provenance", "docs", "files"],
    },
  },
  { path: "/r/:rid/details/docs/:doc?", component: Details, title: "Documents" },
  { path: "/p/:pid/compare", component: lazy(() => import("./Compare")), title: "Compare runs" },
];

/** The run tab ids this module declares (tests check them against the registry). */
export const RUN_HUB_TABS = routes.flatMap((r) => (r.runTab ? [r.runTab.id] : []));
