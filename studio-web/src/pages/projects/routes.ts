// Projects routes (SPEC §3.2, owner frontend-projects): Home, the project list, the setup
// wizard, the config editor and Launch. Declares the project nav entries Setup, Inputs and
// Launch (§3.1; Overview and Runs belong to tracking). Inputs is the wizard's inputs step on
// its own static path so it can carry its own nav entry; the config editor is reached from
// Setup, not from the nav.
import { lazy } from "react";
import type { RouteDef } from "../../router";

const Setup = lazy(() => import("./Setup"));

export const routes: RouteDef[] = [
  { path: "/", component: lazy(() => import("./Home")), title: "Home" },
  { path: "/projects", component: lazy(() => import("./Projects")), title: "Projects" },
  {
    path: "/p/:pid/setup/:step",
    component: Setup,
    title: "Setup",
    projectNav: { id: "setup", label: "Setup", order: 20, to: (pid) => `/p/${encodeURIComponent(pid)}/setup/data` },
  },
  {
    path: "/p/:pid/setup/inputs",
    component: Setup,
    title: "Inputs",
    projectNav: { id: "inputs", label: "Inputs", order: 30, to: (pid) => `/p/${encodeURIComponent(pid)}/setup/inputs` },
  },
  { path: "/p/:pid/setup", component: Setup, title: "Setup" },
  { path: "/p/:pid/config", component: lazy(() => import("./ConfigEditor")), title: "Config" },
  {
    path: "/p/:pid/launch",
    component: lazy(() => import("./Launch")),
    title: "Launch",
    projectNav: { id: "launch", label: "Launch", order: 40, to: (pid) => `/p/${encodeURIComponent(pid)}/launch` },
  },
];

/** The project nav entries this module declares (tests check them against the registry). */
export const PROJECT_NAV_IDS = routes.flatMap((r) => (r.projectNav ? [r.projectNav.id] : []));
