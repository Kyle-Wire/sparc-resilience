// Legacy URLs (phase 2 information architecture): every run tab and page that moved keeps its
// old URL as a redirect, so bookmarks, pinned Findings (`view`/`url_state`), exported reports
// and server links keep working. Each redirect replaces the history entry, keeps the query and
// the hash (RouteDef.redirect; App renders it). Titles stay the old page titles: Findings
// label pins by their stored route pattern.
//
// Route modules import only types from the router at top level; the helpers are used inside
// the redirect functions (called after the registry is built).
import type { RouteDef, RouteRedirect, RunSection } from "../../router";
import * as router from "../../router";

/** Redirect to a run page, keeping the query (the hash is kept by the renderer). */
const toRun =
  (section: RunSection, sub?: (params: Record<string, string>) => string | null): RouteRedirect =>
  (p, search) =>
    router.withSearch(router.runHref(p.rid, section, sub ? sub(p) : null), search);

const fixed = (sub: string) => () => sub;

export const routes: RouteDef[] = [
  // run tabs folded into the seven planner tabs and Model details
  { path: "/r/:rid/data", title: "Data & QA", redirect: toRun("details", fixed("data")) },
  { path: "/r/:rid/accuracy", title: "Accuracy", redirect: toRun("trust") },
  { path: "/r/:rid/distance", title: "Distance & baselines", redirect: toRun("details", fixed("reach")) },
  { path: "/r/:rid/influence", title: "Influence", redirect: toRun("effects") },
  { path: "/r/:rid/response", title: "Response", redirect: toRun("effects") },
  { path: "/r/:rid/causal", title: "Causal audit", redirect: toRun("details", fixed("causal")) },
  { path: "/r/:rid/scenarios", title: "Configured scenarios", redirect: toRun("lab", fixed("library")) },
  { path: "/r/:rid/climate", title: "Climate", redirect: toRun("heat", fixed("future")) },
  { path: "/r/:rid/budget", title: "Budget", redirect: toRun("lab", fixed("plans")) },
  { path: "/r/:rid/planner", title: "Planner pack", redirect: toRun("heat", fixed("people")) },
  { path: "/r/:rid/validation", title: "Validation", redirect: toRun("trust") },
  { path: "/r/:rid/uncertainty", title: "Uncertainty", redirect: toRun("trust") },
  { path: "/r/:rid/provenance", title: "Provenance", redirect: toRun("details", fixed("provenance")) },
  { path: "/r/:rid/docs/:doc?", title: "Documents", redirect: toRun("details", (p) => (p.doc ? `docs/${p.doc}` : "docs")) },
  { path: "/r/:rid/files", title: "Files", redirect: toRun("details", fixed("files")) },
  { path: "/r/:rid/track", title: "Track", redirect: toRun("details", fixed("runlog")) },
  // Lab › Climate merged into Heat risk › Future climate
  { path: "/r/:rid/lab/climate", title: "Climate × adaptation", redirect: toRun("heat", fixed("future")) },
  // Findings became the Notes tab of Reports
  { path: "/p/:pid/findings", title: "Findings", redirect: (p, search) => router.withSearch(`/p/${encodeURIComponent(p.pid)}/exports?tab=notes`, search) },
];

/** The legacy patterns this module declares (tests walk each one). */
export const LEGACY_PATHS = routes.map((r) => r.path);
