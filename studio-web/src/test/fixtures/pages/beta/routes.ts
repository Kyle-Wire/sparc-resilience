// Registry fixture: a second route module (another feature folder), with multi-view tabs, a
// redirect route, an advanced project nav entry and a duplicate tab id.
import { lazy } from "react";
import type { RouteDef } from "../../../../router";

const Page = lazy(() => import("../FixturePage"));

export const routes: RouteDef[] = [
  { path: "/r/:rid/lab", component: Page, title: "Scenario Lab", runTab: { id: "lab", label: "Lab", order: 60, views: ["lab", "scenarios"] } },
  { path: "/r/:rid/lab/library", component: Page, title: "Library" },
  { path: "/r/:rid/causal", component: Page, title: "Causal", runTab: { id: "causal", label: "Causal", order: 20, views: ["causal"] } },
  { path: "/r/:rid/trust", component: Page, title: "Trust", runTab: { id: "trust", label: "Trust", order: 40, views: ["validation", "uncertainty"] } },
  // A legacy URL: replaced by its target, query kept (the renderer keeps the hash).
  { path: "/r/:rid/old", title: "Old", redirect: (p, search) => `/r/${p.rid}/accuracy${search}` },
  {
    path: "/r/:rid/lab/x",
    component: Page,
    title: "Lab nav",
    projectNav: {
      id: "lab",
      label: "Scenario Lab",
      order: 60,
      to: (_pid, p) => (p.active_run_id ? `/r/${p.active_run_id}/lab` : `/p/${p.id}/runs`),
      disabledReason: (p) => (p.active_run_id ? null : "Needs a run with a checkpoint"),
    },
  },
  {
    path: "/p/:pid/tools",
    component: Page,
    title: "Tools",
    projectNav: { id: "tools", label: "Tools", order: 70, group: "advanced", to: (pid) => `/p/${pid}/tools`, claims: ["/p/:pid/tools/:tool"] },
  },
  { path: "/p/:pid/tools/:tool", component: Page, title: "Tool" },
  // A duplicate tab id: must be reported and ignored (each tab has one owner).
  { path: "/r/:rid/other-files", component: Page, title: "Other", runTab: { id: "files", label: "Files 2", order: 1, views: ["files"] } },
  // A tab listing a view the server does not know: reported and ignored.
  { path: "/r/:rid/bogus", component: Page, title: "Bogus", runTab: { id: "bogus", label: "Bogus", order: 5, views: ["nope" as never] } },
];
