# PRD: SPARC Desktop — Architecture Hardening & Quality Pass

**Date:** 2026-05-09  
**Status:** Approved — ready for implementation  
**Scope:** Full application (React/TypeScript frontend, Tauri/Rust host, Python FastAPI sidecar)

---

## Problem Statement

The SPARC Desktop application has accumulated architectural gaps across its three layers that compromise reliability, security posture, developer experience, and user-facing performance. These gaps were identified through a systematic code review and design interview. None are blocking today's local-only usage, but several will actively prevent the planned migration to cloud-hosted compute and several cause measurable user pain right now (slow chart loading, crashes with no recovery path, broken TypeScript compilation).

---

## Goals

1. Clean TypeScript build — zero compiler errors
2. Unified, configurable server URL — foundation for future cloud mode
3. End-to-end authenticated requests — Supabase JWT flows through to the sidecar
4. Graceful mid-session server loss — reconnect, restore state, no hard crash
5. React error boundaries — crashes are caught, reported automatically, app recovers
6. Centralized state — project and navigation state in Zustand stores, no prop drilling
7. Measurably faster chart/result loading after each pipeline stage
8. Frontend unit test coverage on critical paths

---

## Non-Goals

- React Router migration (no meaningful benefit in single-window Tauri app)
- E2E / Playwright tests (out of scope for this cycle)
- Cloud server provisioning or deployment (URL foundation only; cloud mode is future work)
- Resend email setup (Supabase Edge Function is built; Resend API key configuration is an ops task)
- Python sidecar test suite changes (already well-covered; out of scope)

---

## Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Single `SERVER_ORIGIN` constant derived by all HTTP and WS clients | Three hardcoded URLs today — single source required for cloud migration |
| 2 | Supabase JWT forwarded as `Authorization: Bearer` to sidecar | User identity reaches the sidecar; enables per-user audit trail and cloud access control |
| 3 | `python-jose[cryptography]` for JWKS verification in sidecar | Offline-capable RS256 verification; no network call per request |
| 4 | WS auth via `?token=<hex>` query param | Browser WebSocket API cannot set custom headers; query param is standard practice |
| 5 | Page-level `ErrorBoundary`, not per-panel | Page granularity catches all component crashes; per-panel adds complexity without proportional benefit |
| 6 | Crash report via Supabase Edge Function → Resend | Automatic, no user action required; API key never in frontend bundle |
| 7 | `/run/session-log` new sidecar endpoint | Crash reports need persistent log; in-memory buffer may be corrupted at crash time |
| 8 | `projectStore` Zustand store | Eliminates prop drilling; makes project state available to error boundary and future components |
| 9 | `navigationStore` with per-page UI state map | Preserves tab selection / scroll position across navigation; data already cached in `resultCache` |
| 10 | `asyncio.to_thread` for artifact blob reads | Large artifact reads block the async event loop; thread pool unblocks concurrent requests |
| 11 | `/project/load` returns after YAML parse; data load backgrounded | Users see "loaded" immediately; CSV + prewarm happen asynchronously |
| 12 | `useAvailability` fast-poll 2s → 500ms during runs | Halves worst-case detection lag after stage completion |
| 13 | Eager manifest refetch on `stage_status: complete` | Bypasses poll interval entirely for stage boundaries — the highest-impact latency point |
| 14 | Vitest for frontend unit tests | Fastest setup for Vite projects; no config overhead |

---

## Technical Approach

### Layer 1 — Frontend (React/TypeScript)

#### 1.1 TypeScript Errors (9 errors, 6 files)
Fix all compiler errors before any other work:
- `ProcessingPage.tsx` — duplicate `export default function ProcessingPage` (lines 15 and 292); remove second definition
- `App.tsx` — remove unused `ComparePage` import; fix `onNavigate` type mismatch (`AppPage` vs `string`); fix `DAGPage` missing prop declaration
- `DataPage.tsx` — `bodyPadding` prop does not exist on `CardProps`; rename or add to type
- `Shell.tsx` — `status` prop declared but never read; remove from destructure
- `OverviewPanel.tsx` — `sensitivity` object assigned to `ReactNode`; wrap in string

#### 1.2 Dead Code Removal
- ~~`usePipelineStream.ts`~~ — already deleted
- `ComparePage` import in `App.tsx` — remove (covered by 1.1)
- Any other unreferenced exports discovered during TS fix pass

#### 1.3 Server URL — Single Source of Truth
Create `src/lib/server.ts`:
```ts
export const SERVER_ORIGIN = "http://127.0.0.1:8008";
export const WS_ORIGIN    = "ws://127.0.0.1:8008";
```
Update all consumers:
- `src/lib/api.ts` — replace `const BASE = "http://127.0.0.1:8008"` 
- `src/hooks/PipelineProvider.tsx` — replace inline WS URL
- `src/hooks/useArtifactStream.ts` — replace `WS_URL` constant

#### 1.4 Auth — Forward Supabase JWT
Update `src/lib/api.ts`:
- `authHeaders()` reads `supabase.auth.getSession()` (or the cached session from `authStore`) and adds `Authorization: Bearer <access_token>` alongside `X-SPARC-Token`
- Token refresh is handled by Supabase's `autoRefreshToken: true` — no manual refresh logic needed

Update both WebSocket connection points (PipelineProvider, useArtifactStream):
- Append `?token=<sidecar_hex>` to the WS URL

#### 1.5 Error Boundaries
Create `src/components/common/ErrorBoundary.tsx`:
- Class component (required by React error boundary API)
- `componentDidCatch`: call `reportCrash(error, info)` async — fire and forget
- Render fallback: card with error message, "Try reloading this page" button (calls `this.setState({ hasError: false })`), and "Report sent" toast via `useNotification`

Create `src/lib/crashReporter.ts`:
- `reportCrash(error, componentInfo)`: fetches `/run/session-log` (last 200 entries), then calls `supabase.functions.invoke('report-crash', { body: { ... } })`
- Payload: `{ error: string, stack: string, componentStack: string, page: string, appVersion: string, userId: string, sessionLog: object[] }`
- Fails silently if sidecar or Supabase unreachable (user already in an error state)

Wrap every page render in `App.tsx`:
```tsx
<ErrorBoundary page={page}>
  {renderPage()}
</ErrorBoundary>
```

#### 1.6 State — projectStore
Create `src/stores/projectStore.ts` (Zustand):
```ts
interface ProjectState {
  projectPath: string | null;
  projectLoaded: boolean;
  rehydrating: boolean;
  openProject: (path: string, meta?) => Promise<void>;
  clearProject: () => void;
}
```
Migrate logic from `useProject` hook into the store. Remove `useProject` hook. Update `App.tsx` and all pages that receive `projectPath`/`onProjectLoaded` as props to read from the store directly.

#### 1.7 State — navigationStore
Create `src/stores/navigationStore.ts` (Zustand):
```ts
interface NavigationState {
  currentPage: AppPage;
  navigate: (page: AppPage) => void;
  pageUIState: Record<string, unknown>;
  setPageUIState: (page: string, state: unknown) => void;
  getPageUIState: <T>(page: string) => T | null;
}
```
- `navigate` enforces project-load gate (same logic as current `navigate` callback)
- Pages write their UI state (selected tab, scroll position) on change and read it on mount
- Remove `page` state and `navigate` callback from `App.tsx` local state; remove prop drilling through `Shell` and `Sidebar`

#### 1.8 Server Resilience
Update `src/hooks/useServer.ts`:
- After initial `ready = true`, continue polling at 30s intervals to detect mid-session loss
- On poll failure after previously ready: set `ready = false`, emit a `serverLost` event
- On re-poll success after loss: attempt to restore pipeline state via `getRunEvents()`; set `ready = true`

Add `ServerLostBanner` component (similar to existing `NotificationBanner`):
- Shown when `serverLost` is true
- Shows reconnect attempt count and spinner
- Dismisses automatically when server returns

#### 1.9 Performance — Frontend
In `src/hooks/useAvailability.ts`:
- Change `FAST_INTERVAL_MS` from `2_000` to `500`

In `src/hooks/PipelineProvider.tsx`:
- On `stage_status: complete` event, call `refetch()` from `useManifest` directly (import the singleton `_fetchOnce` or expose a `refetch` action from the manifest singleton)

#### 1.10 Frontend Unit Tests — Vitest
Add to `sparc-desktop/package.json`:
```json
"vitest": "^3.x",
"@vitest/ui": "^3.x",
"@testing-library/react": "^16.x",
"jsdom": "^25.x"
```

Test files to create:
- `src/lib/api.test.ts` — mock `fetch`, verify `X-SPARC-Token` and `Authorization` headers present on all non-exempt requests
- `src/hooks/PipelineProvider.test.tsx` — verify `isRunning`, `stageStatuses`, `runStartedAt` transitions from a sequence of synthetic events
- `src/stores/projectStore.test.ts` — verify `openProject` sets state, `clearProject` resets, rehydration logic
- `src/stores/navigationStore.test.ts` — verify `navigate` gate (blocked without project), UI state persistence

---

### Layer 2 — Tauri/Rust Host

#### 2.1 Inject SUPABASE_URL at Sidecar Spawn
In `src-tauri/src/sidecar.rs`, read `VITE_SUPABASE_URL` from the bundled `.env.local` (or from a compiled-in constant via Tauri's `env!` macro at build time) and inject as `SUPABASE_URL` environment variable into the sidecar `Command` alongside existing `SPARC_SERVER_TOKEN` and `SPARC_SERVER_PORT`.

---

### Layer 3 — Python Sidecar

#### 3.1 Add python-jose dependency
In `pyproject.toml` under `[project] dependencies`:
```toml
"python-jose[cryptography]>=3.3",
```

#### 3.2 Supabase JWT Verification Middleware
Update `_TokenMiddleware` in `sparc/server/app.py`:
- Still validate `X-SPARC-Token` first (fast path, local guard)
- If token valid and `Authorization: Bearer <jwt>` header present: verify JWT via JWKS
- JWKS URL: `{SUPABASE_URL}/auth/v1/jwks` — fetched once at startup, cached in memory, refreshed every 24h
- Invalid JWT → 401. Missing JWT → 200 (JWT is additive for now; becomes required when cloud mode ships)
- Attach decoded `sub` (user UUID) to request state for audit logging

#### 3.3 Async Artifact Reads
All `_read_or_404` calls in result endpoints that read large blobs:
- Wrap synchronous `store.read_any(stage, artifact_id)` in `await asyncio.to_thread(store.read_any, stage, artifact_id)`
- Applies to: all `/results/*` endpoints serving parquet, npy, pkl artifacts

#### 3.4 Async Project Load
Update `/project/load` endpoint:
- Return `{"status": "loading", ...}` immediately after YAML parse + `_attach_registry` (fast — just opens SQLite)
- Move `_load_data_into_state(config)` and `_start_prewarm()` to a background `asyncio.create_task`
- Frontend detects readiness via existing `/health` poll (`project_loaded: true` is already set after YAML parse)

#### 3.5 New /run/session-log Endpoint
```python
@app.get("/run/session-log")
async def get_session_log(n: int = Query(200)):
    """Return the last n lines of the current project's session_log.jsonl."""
```
- Reads tail of `{project_dir}/session_log.jsonl`
- Returns `{"entries": [...], "total_lines": int, "path": str}`
- Returns empty entries (not 404) if no log exists yet

#### 3.6 WebSocket Token Validation
Update the `/run/stream` WebSocket handler:
- On connection upgrade, read `token` query param
- Validate against `_SERVER_TOKEN` (same constant as HTTP middleware)
- Close with code 4003 if invalid

---

### Layer 4 — Supabase Edge Function

#### 4.1 Create supabase/functions/report-crash/index.ts
```ts
// Deno Edge Function
// Receives crash report from ErrorBoundary, sends via Resend API
```
- Validates request is from authenticated Supabase user (JWT verified by Supabase platform automatically)
- Sends email to `sparcurbanlabs@gmail.com` via Resend API
- Email includes: error message, stack, component stack, page, app version, user ID, session log entries (formatted)
- Returns `{ received: true }` — frontend shows "Crash report sent" toast

---

## Acceptance Criteria

- [ ] `npx tsc --noEmit` exits 0 with no errors
- [ ] All hardcoded `127.0.0.1:8008` strings replaced by `SERVER_ORIGIN` / `WS_ORIGIN`
- [ ] Every HTTP request to sidecar includes both `X-SPARC-Token` and `Authorization: Bearer` headers when user is signed in
- [ ] Both WebSocket connections include `?token=` query param; sidecar rejects connections with invalid/missing token
- [ ] Sidecar successfully verifies a real Supabase JWT in dev (manual curl test)
- [ ] Navigating away from RunPage during an active pipeline run does not cancel the run
- [ ] Killing and restarting the sidecar mid-session shows "server lost" banner; after sidecar restarts, banner clears and pipeline state restores
- [ ] A deliberately thrown error in a page component shows the error boundary card, not a blank screen
- [ ] Crash report email arrives at `sparcurbanlabs@gmail.com` in dev test
- [ ] `/run/session-log` returns last 200 entries from current project log
- [ ] `pnpm test` runs Vitest suite with all 4 test files passing
- [ ] Subjective: result charts appear within 1–2s of stage completion (down from current 3–6s)
- [ ] `projectStore` exports `useProjectStore` — no page receives `projectPath` or `onProjectLoaded` as props
- [ ] `navigationStore` exports `useNavigationStore` — no page receives `onNavigate` as a prop

---

## Resolved Questions

- **Resend API key**: Owner will create Resend account and obtain API key. Set via `supabase secrets set RESEND_API_KEY=<key>` using Supabase CLI linked to the project. Step-by-step instructions are part of T25.

- **SUPABASE_URL in Tauri build**: Build-time injection via `cargo` `env!` macro. A `build.rs` script reads `VITE_SUPABASE_URL` from `.env.local` at compile time and exposes it as `SPARC_SUPABASE_URL` to the Rust binary. No runtime file I/O. Acceptable constraint: a new build is required if the Supabase project URL ever changes (unlikely for a desktop app).

- **JWT requirement feature flag**: Implemented as `SPARC_REQUIRE_JWT` environment variable injected by the Tauri host (consistent with `SPARC_SERVER_TOKEN` naming convention). Default `false` — additive mode. Sidecar reads: `JWT_REQUIRED = os.environ.get("SPARC_REQUIRE_JWT", "false").lower() == "true"`. Enable in a future PR when cloud mode ships by having the Tauri host inject `SPARC_REQUIRE_JWT=true`.
