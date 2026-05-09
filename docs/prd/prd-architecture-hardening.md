# PRD: Architecture Hardening (Non-Security)

**SPARC Labs LLC | May 2026**
**Status: Planning**

---

## Problem Statement

SPARC is maturing toward a professional desktop product with a published academic paper and active commercial pilots. The core pipeline and server infrastructure are well-designed, but several cross-cutting engineering practices — error handling, observability, test coverage, fetch hygiene, and server-side logging — are at prototype quality. These gaps will cause user-facing failures, debugging difficulty, and maintainability costs at scale.

This PRD covers non-security architectural improvements discovered during a full-stack code review (April–May 2026). Security items are tracked separately.

---

## Goals

1. Surface real errors to users in a consistent, actionable way.
2. Give developers structured logs they can inspect when things go wrong in production.
3. Prevent runaway resource leaks (fetch requests, WebSocket handles, pending timers).
4. Establish a minimal but working frontend test harness.
5. Eliminate patterns that will silently degrade at scale (O(n) per-event re-renders, uncapped polling, missing fetch timeouts).

## Non-Goals

- Security (auth, token rotation, RLS) — separate PRD.
- V4 transfer learning wiring — separate roadmap (SPARC_V4_Roadmap_Integrated.md).
- Adding new pipeline stages or domain templates.
- Redesigning existing UX.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Server logging | Replace `print()` in `app.py` with `logging.getLogger(__name__)` | Consistent with all other pipeline modules; allows log level control via `SPARC_LOG_LEVEL` env var |
| Fetch timeout | `AbortController` + `fetch(signal)` in `api.ts` | Native, no new dependencies; prevents hanging requests from blocking the UI indefinitely |
| Error boundary | Single `<ErrorBoundary>` wrapping each page-level component | React standard; already absent — any uncaught render error currently whitescreens the entire app |
| Frontend tests | Vitest + `@testing-library/react` | Shares Vite config; zero config overhead; standard for Vite+React |
| Notification auto-dismiss | Already exists via `useNotificationState`; ensure all error paths route through it | Consistent UX; errors are currently swallowed silently in several `api.ts` callers |
| rAF event buffering | Applied (completed) to `usePipelineStream` and `PipelineProvider` | Prevents O(n) re-render cost per WebSocket message |
| Unmount cleanup | Applied (completed) to `usePipelineStream` | WebSocket closes on unmount; pipeline keeps running server-side |

---

## Technical Approach

### Area 1 — Server Logging (`app.py`)

**Current state:** 14 `print()` calls in `app.py` for warnings and operational events. All other `sparc/run/*.py` modules use `logging.getLogger(__name__)`.

**Gap:** `print()` output goes to uvicorn's stdout with no level, no timestamp, no module name. When bundled in the PyInstaller sidecar, stdout is redirected to a log file but the messages are indistinguishable from pipeline stdout. There is no way to filter by severity.

**Fix:** Add a module-level `logger = logging.getLogger(__name__)` at the top of `app.py` and replace all `print()` calls with the appropriate `logger.warning()` / `logger.info()` call. This is a mechanical, low-risk change.

---

### Area 2 — Fetch Timeouts (`api.ts`)

**Current state:** `get()` and `post()` in `api.ts` call `fetch()` with no timeout. The `useServer` health poll fires every 500ms with no cancellation.

**Gap:** If the sidecar is slow to respond (e.g., loading a large dataset, saturated CPU during Stage 2), fetch calls queue silently. On a flaky connection or if uvicorn hangs, the UI freezes with a spinner forever. The 500ms health poll can accumulate many in-flight requests.

**Fix:**
- Wrap `get()` and `post()` with `AbortController` + a configurable timeout (default 30s for data endpoints, 5s for health).
- Add a `signal` option to each fetch wrapper so callers can cancel (e.g., `useResult` can cancel the in-flight request when the key changes).
- The health poll in `useServer` should cancel its previous request before issuing a new one.

---

### Area 3 — Frontend Error Boundaries

**Current state:** No `<ErrorBoundary>` components anywhere in the React tree. The app renders inside `App.tsx` with no catch layer.

**Gap:** Any uncaught error in a results panel (e.g., a null-dereference on malformed pipeline output) unwinds the entire React tree, whitescreen-ing the app. The user must force-quit and relaunch.

**Fix:** Add a generic `<ErrorBoundary>` component and wrap each major page at the router level. On error, render an inline fallback ("Something went wrong — try reloading this page") rather than a blank screen. The pipeline and other pages remain functional.

---

### Area 4 — Frontend Test Harness

**Current state:** `package.json` has no `test` script. Zero frontend tests exist despite 15+ page components, 12 custom hooks, a Zustand store, and a WebSocket layer.

**Gap:** The two hooks we modified (`usePipelineStream`, `PipelineProvider`) have non-trivial async/timing logic (rAF buffering, unmount cleanup, reconnection). These are exactly the bugs that are invisible in manual testing. The Python side has 50+ test files with `pytest`; the frontend has nothing.

**Fix:**
1. Add `vitest` + `@testing-library/react` + `jsdom` as dev dependencies.
2. Add a `"test": "vitest run"` script and a `"test:watch": "vitest"` script to `package.json`.
3. Add `vitest.config.ts` (can inherit from `vite.config.ts`).
4. Write smoke tests for:
   - `usePipelineStream`: verify rAF flush batches messages, verify unmount closes socket without calling `/run/cancel`
   - `PipelineProvider`: verify `startStage` resets buffer, verify `complete` event flushes synchronously
   - `useResult`: verify deduplication of in-flight fetches, verify cache hit skips fetcher

---

### Area 5 — `app.py` Global Exception Handler

**Current state:** FastAPI's default exception handler returns JSON for `HTTPException` but returns a plain HTML 500 for unhandled exceptions. Several endpoints use `except Exception: return JSONResponse({"error": str(exc)}, 500)` — inconsistent shape.

**Gap:** The frontend's `get()` / `post()` wrapper reads `res.text()` on error and throws `Error("${status}: ${body}")`. An HTML 500 body causes a confusing error message in the terminal panel.

**Fix:** Register a global `@app.exception_handler(Exception)` that returns `{"detail": str(exc), "type": "internal_error"}` as JSON. This normalises all 500 responses to the same shape the frontend already handles.

---

### Area 6 — `resultCache.ts` Staleness TTL

**Current state:** `CacheEntry` stores `fetchedAt: number` but nothing ever reads it. Cache entries are only evicted on `invalidateStage` (WebSocket artifact event) or `clear()` (project unload). There is no TTL.

**Gap:** If the WebSocket artifact stream is disconnected (e.g., the user is on a slow machine and `useArtifactStream` is in exponential backoff), results panels will serve stale data indefinitely after a stage re-run because the invalidation event was never received.

**Fix:** In `useResult`, before treating a cache hit as valid, check if `fetchedAt` is older than a configurable TTL (default 5 minutes). If stale, evict and re-fetch. This is a backstop for the primary WebSocket invalidation path; for most users with a healthy WS connection it never fires.

---

### Area 7 — `useNotificationState` `_nextId` Module Singleton

**Current state:** `let _nextId = 0` is a module-level mutable singleton in `useNotifications.ts`. In React Strict Mode (dev), components mount twice, meaning the counter can desync between the first and second mount.

**Gap:** Not a runtime bug in production, but can cause confusing duplicate notification IDs in development, masking real issues during testing.

**Fix:** Move `_nextId` to a `useRef` inside `useNotificationState` or use `crypto.randomUUID()` (available in Tauri's webview). A one-line change.

---

## Acceptance Criteria

- [ ] `app.py` has zero `print()` calls; all operational messages go through `logger`.
- [ ] `get()` and `post()` in `api.ts` throw a typed `TimeoutError` (or similar) after 30s; health poll cancels previous request before issuing a new one.
- [ ] Every top-level page is wrapped in `<ErrorBoundary>`; a render crash in one panel does not blank the entire app.
- [ ] `pnpm test` runs successfully with at least 6 passing test cases covering the hooks listed in Area 4.
- [ ] A single `@app.exception_handler(Exception)` normalises all unhandled 500 responses to `{"detail": "...", "type": "internal_error"}`.
- [ ] `useResult` evicts cache entries older than a configurable TTL (default 5 min) before serving from cache.
- [ ] Notification IDs use `crypto.randomUUID()` instead of a module-level integer counter.

## Open Questions

- Should fetch timeout be configurable via the hardware profile (longer on low-tier hardware during heavy stages)? Currently proposed as a fixed 30s.
- Should the `<ErrorBoundary>` log caught errors to the server via a `POST /client/error` endpoint so they appear in the session log? Deferred — adds complexity; could be Phase 2.

---

## Items Already Implemented (this session)

- `usePipelineStream`: rAF-based event buffering (O(1) per message vs O(n))
- `usePipelineStream`: unmount cleanup (`useEffect` teardown without `/run/cancel`)
- `PipelineProvider`: rAF-based event buffering + synchronous flush on `complete`/`error`
- `PipelineProvider`: event buffer reset on new run start
