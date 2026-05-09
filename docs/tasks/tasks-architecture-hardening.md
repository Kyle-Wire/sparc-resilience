# Tasks: Architecture Hardening (Non-Security)

Related PRD: docs/prd/prd-architecture-hardening.md

---

## Status

- [x] usePipelineStream: rAF event buffering
- [x] usePipelineStream: unmount cleanup (no server cancel)
- [x] PipelineProvider: rAF event buffering + flush on complete/error
- [x] PipelineProvider: buffer reset on new run start

---

## Tasks

### Area 1 — Server Logging
- [ ] Add `logger = logging.getLogger(__name__)` to `sparc/server/app.py`
- [ ] Replace all 14 `print()` calls in `app.py` with `logger.warning()` / `logger.info()` as appropriate

### Area 2 — Fetch Timeouts
- [ ] Add `AbortController` + 30s timeout wrapper to `get()` in `sparc-desktop/src/lib/api.ts`
- [ ] Add `AbortController` + 30s timeout wrapper to `post()` in `api.ts`
- [ ] Add 5s timeout to health poll in `useServer.ts`; cancel previous request before issuing new one

### Area 3 — Error Boundaries
- [ ] Create `sparc-desktop/src/components/common/ErrorBoundary.tsx` (React class component)
- [ ] Wrap each page-level route in `App.tsx` with `<ErrorBoundary>`

### Area 4 — Frontend Test Harness
- [ ] Add `vitest`, `@testing-library/react`, `jsdom`, `@testing-library/jest-dom` as dev dependencies
- [ ] Add `"test": "vitest run"` and `"test:watch": "vitest"` scripts to `package.json`
- [ ] Create `sparc-desktop/vitest.config.ts`
- [ ] Write tests: `usePipelineStream` — rAF batching, unmount no-cancel
- [ ] Write tests: `PipelineProvider` — buffer reset on startStage, synchronous flush on complete
- [ ] Write tests: `useResult` — in-flight deduplication, cache hit skips fetcher

### Area 5 — Global Exception Handler
- [ ] Register `@app.exception_handler(Exception)` in `sparc/server/app.py` returning `{"detail": str(exc), "type": "internal_error"}`
- [ ] Audit existing `except Exception: return JSONResponse(...)` blocks and normalise to the same shape

### Area 6 — Result Cache TTL
- [ ] Add TTL check in `useResult.ts` — evict entries older than 5 min before serving from cache
- [ ] Make TTL configurable via a constant at the top of `useResult.ts`

### Area 7 — Notification ID Fix
- [ ] Replace `let _nextId = 0` in `useNotifications.ts` with `crypto.randomUUID()` per notification
