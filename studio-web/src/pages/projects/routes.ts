// Projects routes (SPEC §3.2, owner frontend-projects): Home, the project list, the setup
// wizard, the config editor and Launch. Declares the project nav entries "Set up" (one entry
// for the whole flow Data · Inputs · Settings · Launch: the wizard's steps and Launch all
// highlight it) and "Config editor" in the sidebar's collapsed Advanced group (§3.1; Overview,
// Runs and Activity belong to tracking).
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
    projectNav: {
      id: "setup",
      label: "Set up",
      order: 20,
      to: (pid) => `/p/${encodeURIComponent(pid)}/setup/data`,
      claims: ["/p/:pid/setup", "/p/:pid/setup/inputs", "/p/:pid/launch"],
    },
  },
  { path: "/p/:pid/setup/inputs", component: Setup, title: "More data" },
  { path: "/p/:pid/setup", component: Setup, title: "Setup" },
  {
    path: "/p/:pid/config",
    component: lazy(() => import("./ConfigEditor")),
    title: "Config",
    projectNav: { id: "config", label: "Config editor", order: 80, group: "advanced", to: (pid) => `/p/${encodeURIComponent(pid)}/config` },
  },
  { path: "/p/:pid/launch", component: lazy(() => import("./Launch")), title: "Launch" },
];

/** The project nav entries this module declares (tests check them against the registry). */
export const PROJECT_NAV_IDS = routes.flatMap((r) => (r.projectNav ? [r.projectNav.id] : []));
