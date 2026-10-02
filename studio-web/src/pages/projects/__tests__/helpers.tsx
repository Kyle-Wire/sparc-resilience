// Test helpers for the projects pages: a registry with this module's routes, rendering at a
// URL, a seeded setup draft rendered inside the config context, and fetch-mock routes for a
// project (built on the foundation test kit).
import type { ReactNode } from "react";
import type { DataCheck } from "../../../api/projects";
import { clearResources } from "../../../api/resource";
import type { Issue } from "../../../api/types";
import { buildRegistry, navigate, RegistryProvider, type RouteModule } from "../../../router";
import { useJobs } from "../../../stores/jobs";
import { cleanup, render, type MockHandler } from "../../../test/render";
import { routes as coreRoutes } from "../../core/routes";
import { CfgProvider } from "../components/cfg";
import { routes } from "../routes";
import { resetDrafts, rawKey, useDraft, useDrafts } from "../setup/store";
import { configDoc, demoRaw, FILES, INSPECT, PID, projectDetail } from "../__fixtures__/api";

export const MODULES: Record<string, RouteModule> = { "./pages/core/routes.ts": { routes: coreRoutes }, "./pages/projects/routes.ts": { routes } };

export function projectsRegistry() {
  return buildRegistry(MODULES);
}

/** Render `ui` with this module's registry, after moving the URL to `path`. */
export function renderAt(path: string, ui: ReactNode) {
  navigate(path, { replace: true });
  return render(<RegistryProvider registry={projectsRegistry()}>{ui}</RegistryProvider>);
}

/** Reset the caches and stores the pages share (call in afterEach). */
export function resetAll(): void {
  cleanup(); // unmount first, so the store resets below re-render nothing
  clearResources();
  resetDrafts();
  useJobs.getState().reset();
  navigate("/", { replace: true });
}

/** Put a draft for PID in the store: saved = `raw`, plus issues and a data check. */
export function seedDraft(opts: { raw?: Record<string, unknown>; issues?: Issue[]; check?: DataCheck | null; version?: number } = {}) {
  const raw = opts.raw ?? demoRaw();
  const st = useDrafts.getState();
  st.adopt(PID, configDoc(raw, opts.version ?? 3));
  st.setIssues(PID, opts.issues ?? [], rawKey(raw));
  if (opts.check !== undefined) st.setCheck(PID, { check: opts.check, checkKey: opts.check ? rawKey(raw) : null });
}

function DraftHost({ children }: { children: ReactNode }) {
  const draft = useDraft(PID);
  if (!draft) return null;
  return <CfgProvider value={{ pid: PID, draft, issues: draft.issues }}>{children}</CfgProvider>;
}

/** Render a setup step on a seeded draft (issues come from the draft). */
export function renderStep(step: ReactNode, path = `/p/${PID}/setup/data`) {
  return renderAt(path, <DraftHost>{step}</DraftHost>);
}

const PAGE = { items: [], next_cursor: null };

/** Mocks every read the projects pages make for PID (override entries per test). */
export function projectRoutes(extra: Record<string, MockHandler> = {}): Record<string, MockHandler> {
  const base = `/api/projects/${PID}`;
  return {
    [`GET ${base}`]: { body: projectDetail() },
    [`GET ${base}/config`]: { body: configDoc() },
    [`GET ${base}/config/history`]: { body: [{ version: 3, saved_utc: "2026-10-01T20:05:00Z", note: "created" }] },
    [`POST ${base}/config/validate`]: { body: { ok: true, issues: [], fast_overrides: {}, coarse_preview: null } },
    [`GET ${base}/files`]: { body: FILES },
    [`GET ${base}/files/inspect`]: { body: INSPECT },
    [`GET ${base}/inputs`]: { body: { forcing: null, climate: null, layers: null, features: null, ghcn: null } },
    "GET /api/jobs": { body: PAGE },
    "GET /api/settings": { body: { thread_budget: 4, threads_heavy: 3, engine_threads: 2, offline: false, upload_max_gb: 2 } },
    "POST /api/system/netcheck": { body: { results: [] } },
    ...extra,
  };
}
