// Registry fixture: run tabs declared out of order across groups, plus a project nav entry.
import { lazy } from "react";
import type { RouteDef } from "../../../../router";

const Page = lazy(() => import("../FixturePage"));

export const routes: RouteDef[] = [
  { path: "/r/:rid/files", component: Page, title: "Files", runTab: { id: "files", label: "Files", group: "Run", order: 90 } },
  { path: "/r/:rid", component: Page, title: "Overview", runTab: { id: "overview", label: "Overview", group: "Model", order: 10 } },
  { path: "/r/:rid/accuracy", component: Page, title: "Accuracy", runTab: { id: "accuracy", label: "Accuracy", group: "Model", order: 30 } },
  { path: "/r/:rid/docs/:doc?", component: Page, title: "Docs", runTab: { id: "docs", label: "Docs", group: "Run", order: 80 } },
  {
    path: "/p/:pid/runs",
    component: Page,
    title: "Runs",
    projectNav: { id: "runs", label: "Runs", order: 50, to: (pid) => `/p/${pid}/runs` },
  },
];
