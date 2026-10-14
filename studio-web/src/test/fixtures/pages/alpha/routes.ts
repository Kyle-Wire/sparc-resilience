// Registry fixture: run tabs declared out of order (advanced ones first), plus a project nav entry.
import { lazy } from "react";
import type { RouteDef } from "../../../../router";

const Page = lazy(() => import("../FixturePage"));

export const routes: RouteDef[] = [
  { path: "/r/:rid/files", component: Page, title: "Files", runTab: { id: "files", label: "Files", order: 90, advanced: true, views: ["files"] } },
  { path: "/r/:rid", component: Page, title: "Overview", runTab: { id: "overview", label: "Overview", order: 10, views: ["overview"] } },
  { path: "/r/:rid/accuracy", component: Page, title: "Accuracy", runTab: { id: "accuracy", label: "Accuracy", order: 30, views: ["accuracy"] } },
  { path: "/r/:rid/docs/:doc?", component: Page, title: "Docs", runTab: { id: "docs", label: "Docs", order: 80, advanced: true, views: ["docs"] } },
  {
    path: "/p/:pid/runs",
    component: Page,
    title: "Runs",
    projectNav: { id: "runs", label: "Runs", order: 50, to: (pid) => `/p/${pid}/runs` },
  },
];
