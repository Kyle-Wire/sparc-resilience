# Tasks: SPARC Desktop — Architecture Hardening & Quality Pass

Related PRD: docs/prd/prd-sparc-desktop-hardening.md

## Phase 1 — TypeScript & Dead Code (Prerequisite for everything else)

- [x] T01 · Fix `RunPage.tsx` — restore missing `interface RunPageProps` declaration
- [ ] T02 · Fix `ProcessingPage.tsx` — remove duplicate `export default function ProcessingPage` (second definition at line 292)
- [ ] T03 · Fix `App.tsx` — remove unused `ComparePage` import
- [ ] T04 · Fix `App.tsx` — resolve `onNavigate` type mismatch (`AppPage` vs `string`) and `DAGPage` missing prop
- [ ] T05 · Fix `DataPage.tsx` — remove or type `bodyPadding` prop on `Card`
- [ ] T06 · Fix `Shell.tsx` — remove unused `status` from destructure
- [ ] T07 · Fix `OverviewPanel.tsx` — wrap `sensitivity` object correctly as `ReactNode`
- [ ] T08 · Verify `npx tsc --noEmit` exits 0

## Phase 2 — Server URL Consolidation

- [ ] T09 · Create `src/lib/server.ts` with `SERVER_ORIGIN` and `WS_ORIGIN` constants
- [ ] T10 · Update `src/lib/api.ts` to import `SERVER_ORIGIN`
- [ ] T11 · Update `PipelineProvider.tsx` WS URL to use `WS_ORIGIN`
- [ ] T12 · Update `useArtifactStream.ts` `WS_URL` constant to use `WS_ORIGIN`

## Phase 3 — Authentication

- [ ] T13 · Update `src/lib/api.ts` `authHeaders()` to include `Authorization: Bearer <supabase_jwt>` from `authStore` session
- [ ] T14 · Update `PipelineProvider.tsx` WS connection to append `?token=<sidecar_hex>`
- [ ] T15 · Update `useArtifactStream.ts` WS connection to append `?token=<sidecar_hex>`
- [ ] T16 · Add `python-jose[cryptography]>=3.3` to `pyproject.toml` dependencies
- [ ] T17 · Add JWKS fetch + cache to sidecar (`sparc/server/app.py`) — fetch once on startup, refresh every 24h
- [ ] T18 · Extend `_TokenMiddleware` to optionally verify `Authorization: Bearer` JWT via cached JWKS
- [ ] T19 · Update `/run/stream` WS handler to validate `?token=` query param; close 4003 on mismatch
- [ ] T20 · Update `src-tauri/src/sidecar.rs` to inject `SUPABASE_URL` into sidecar env at spawn

## Phase 4 — Error Boundaries & Crash Reporting

- [ ] T21 · Create `src/components/common/ErrorBoundary.tsx` — class component with reset button
- [ ] T22 · Create `src/lib/crashReporter.ts` — fetches `/run/session-log`, calls `supabase.functions.invoke('report-crash')`
- [ ] T23 · Wrap every page render in `App.tsx` with `<ErrorBoundary page={page}>`
- [ ] T24 · Add `/run/session-log` GET endpoint to sidecar — returns tail of `session_log.jsonl`
- [ ] T25 · Create `supabase/functions/report-crash/index.ts` — Deno function, sends via Resend API to `sparcurbanlabs@gmail.com`
- [ ] T26 · Add "Crash report sent" toast in `ErrorBoundary` after invoke resolves

## Phase 5 — State Management Refactor

- [ ] T27 · Create `src/stores/projectStore.ts` — Zustand store with `projectPath`, `projectLoaded`, `rehydrating`, `openProject`, `clearProject`
- [ ] T28 · Migrate logic from `useProject` hook into `projectStore`; delete `useProject` hook
- [ ] T29 · Update `App.tsx` — remove `useProject` call, remove project prop drilling to Shell/Sidebar/pages
- [ ] T30 · Update all pages that receive `projectPath` / `onProjectLoaded` as props to use `useProjectStore()` directly
- [ ] T31 · Create `src/stores/navigationStore.ts` — `currentPage`, `navigate` (with project gate), `pageUIState` map
- [ ] T32 · Migrate `page` state and `navigate` callback from `App.tsx` to `navigationStore`
- [ ] T33 · Remove `onNavigate` prop from all pages; pages call `useNavigationStore().navigate()` directly
- [ ] T34 · Wire per-page UI state: key pages (RunPage, InsightsPage, DAGPage) write/read tab and scroll state from `navigationStore`

## Phase 6 — Server Resilience

- [ ] T35 · Update `useServer.ts` to continue polling at 30s after initial `ready = true`
- [ ] T36 · Detect mid-session server loss: on poll failure after ready, emit `serverLost` state
- [ ] T37 · On server return after loss: call `getRunEvents()` and restore pipeline state in `PipelineProvider`
- [ ] T38 · Create `ServerLostBanner` component — spinner, reconnect count, auto-dismiss on recovery
- [ ] T39 · Mount `ServerLostBanner` in `App.tsx` alongside existing `NotificationBanner`

## Phase 7 — Performance

- [ ] T40 · Sidecar: wrap all `store.read_any()` calls in `/results/*` endpoints with `await asyncio.to_thread(...)`
- [ ] T41 · Sidecar: make `/project/load` return after YAML parse + `_attach_registry`; move `_load_data_into_state` + `_start_prewarm` to `asyncio.create_task`
- [ ] T42 · Frontend: change `FAST_INTERVAL_MS` in `useAvailability.ts` from `2_000` to `500`
- [ ] T43 · Frontend: on `stage_status: complete` in `PipelineProvider`, call `useManifest` singleton `_fetchOnce()` immediately

## Phase 8 — Frontend Unit Tests

- [ ] T44 · Add Vitest + `@testing-library/react` + jsdom to `sparc-desktop/package.json`; configure `vite.config.ts` test block
- [ ] T45 · Write `src/lib/api.test.ts` — verify `X-SPARC-Token` and `Authorization` headers on mocked fetch calls
- [ ] T46 · Write `src/hooks/PipelineProvider.test.tsx` — verify state transitions from synthetic pipeline event sequences
- [ ] T47 · Write `src/stores/projectStore.test.ts` — openProject, clearProject, rehydration
- [ ] T48 · Write `src/stores/navigationStore.test.ts` — navigate gate, UI state persistence
- [ ] T49 · Confirm `pnpm test` passes all 4 suites

## Completion Gate

- [ ] T50 · `npx tsc --noEmit` exits 0
- [ ] T51 · Manual smoke: start app, load project, run Stage 0, navigate away mid-run, navigate back — pipeline still running, terminal intact
- [ ] T52 · Manual smoke: kill sidecar — "server lost" banner appears; restart sidecar — banner clears
- [ ] T53 · Manual smoke: throw error in DevTools — error boundary card shown, crash email received at `sparcurbanlabs@gmail.com`
- [ ] T54 · Manual smoke: complete Stage 0 — Insights charts appear within 2s
