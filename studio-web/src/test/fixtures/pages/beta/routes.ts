// Registry fixture: a second route module (another feature folder).
import { lazy } from "react";
import type { RouteDef } from "../../../../router";

const Page = lazy(() => import("../FixturePage"));

export const routes: RouteDef[] = [
  { path: "/r/:rid/lab", component: Page, title: "Scenario Lab", runTab: { id: "lab", label: "Lab", group: "Decisions", order: 60 } },
  { path: "/r/:rid/lab/library", component: Page, title: "Library" },
  { path: "/r/:rid/causal", component: Page, title: "Causal", runTab: { id: "causal", label: "Causal", group: "Effects", order: 20 } },
  { path: "/r/:rid/validation", component: Page, title: "Validation", runTab: { id: "validation", label: "Validation", group: "Trust", order: 10 } },
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
  // A duplicate tab id: must be reported and ignored (each tab has one owner).
  { path: "/r/:rid/other-files", component: Page, title: "Other", runTab: { id: "files", label: "Files 2", group: "Run", order: 1 } },
];
