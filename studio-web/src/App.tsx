// Root component: matches the URL against the route registry and renders the page inside the
// shell and its layout (/p/:pid/* → ProjectLayout, /r/:rid/* → RunLayout). Pages are lazy
// chunks. Each page gets a fresh error boundary per path; the layouts stay mounted across
// navigation (the run tab bar keeps its focus when a tab is followed), behind a boundary of
// their own that resets on navigation.
import { Suspense, useEffect, type ReactNode } from "react";
import { AppShell } from "./layouts/AppShell";
import { ErrorBoundary } from "./layouts/ErrorBoundary";
import { ProjectLayout } from "./layouts/ProjectLayout";
import { RunLayout } from "./layouts/RunLayout";
import { useRoute, type RouteDef } from "./router";
import { useUi } from "./stores/ui";

const FULL_WIDTH = /^\/(jobs\/:jid|r\/:rid\/(track|map|lab)(\/|$))/;

export function isFullWidth(route: RouteDef | null): boolean {
  if (!route) return false;
  return !!route.fullWidth || FULL_WIDTH.test(route.path);
}

function Loading() {
  return (
    <p className="cap" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" style={{ width: 12, height: 12, marginRight: 6, verticalAlign: -1 }} />
      Loading…
    </p>
  );
}

export function App() {
  const { route, params, pathname } = useRoute();
  const setContext = useUi((s) => s.setContext);
  const Page = route?.component;

  useEffect(() => {
    if (params.pid || params.rid) return; // the layouts set the context
    // Home and the project list have no current project; other global pages keep it.
    if (pathname === "/" || pathname === "/projects") setContext({ projectId: null, runId: null });
    else setContext({ runId: null });
  }, [pathname, params.pid, params.rid, setContext]);

  const page: ReactNode = Page ? (
    <ErrorBoundary key={pathname}>
      <Suspense fallback={<Loading />}>
        <Page />
      </Suspense>
    </ErrorBoundary>
  ) : null;
  let body: ReactNode = page;
  if (route && params.rid && route.path.startsWith("/r/:rid")) body = <RunLayout rid={params.rid}>{page}</RunLayout>;
  else if (route && params.pid && route.path.startsWith("/p/:pid")) body = <ProjectLayout pid={params.pid}>{page}</ProjectLayout>;

  return (
    <AppShell pageTitle={route?.title ?? null} fullWidth={isFullWidth(route)}>
      <ErrorBoundary resetKey={pathname}>{body}</ErrorBoundary>
    </AppShell>
  );
}
