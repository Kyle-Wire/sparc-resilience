# PRD: Fast, Persistent Result Access

## Problem Statement

After a pipeline run completes, results feel slow to load and appear to "disappear" when the user navigates to a different page and back. The user must wait for data to reload despite having already fetched it. Additionally, the first load of each result page after a pipeline run is noticeably slow.

## Root Causes (Diagnosed)

### 1. `useManifest` is NOT a singleton — PRIMARY cause of "disappearing results"
Every component that calls `useManifest()` creates its own **new React state** starting at `manifest: null` with its own 2-second polling loop. When the Insights page unmounts (user navigates away) and remounts (user comes back), the manifest re-polls from scratch. All panels gate their data fetch on `present = !!manifest.lookup(...)`, so they render empty/loading states for 0-2 seconds before the manifest poll resolves, even though the actual result data IS cached in the Zustand store.

Contrast with `useAvailability`, which **is** a module-level singleton with a shared `cached`, `timer`, and `subscribers` — late-joining components immediately get the cached value.

### 2. New SQLite connection per read — secondary cause of slow reads
`RunRegistry.sqlite_connection()` opens and closes a brand-new `sqlite3.Connection` on every call. Each connection pays: file open, WAL discovery, PRAGMA setup (WAL, synchronous, cache_size, temp_store). The 64MB page cache (`PRAGMA cache_size=-65536`) is per-connection, so it never warms across requests.

### 3. Cold start after app restart
The server-side LRU `ResultCache` (50 entries) and the frontend Zustand cache are both in-memory. When the Tauri sidecar restarts, every result must be re-read from SQLite on first access.

### 4. Server-side LRU too small
50 entries may evict earlier stage results when many panels are requesting data simultaneously.

## Goals

1. Results panels are **instantly populated on re-navigation** — no visible blank/loading flash after first load.
2. First-load after project open is **as fast as possible** — pre-warm the server cache in the background.
3. **SQLite reads are fast** — persistent connection eliminates reconnect overhead.
4. Results survive app restarts gracefully (SQLite is on-disk; pre-warm fills cache quickly).

## Non-Goals

- Replacing SQLite with a different database engine.
- Persisting the frontend Zustand cache to IndexedDB/localStorage (future phase).
- CATE map visual bug (tracked separately).
- Budget page UX improvements (tracked separately).

## Design Decisions

### D1: Convert `useManifest` to a module-level singleton
Mirror `useAvailability` exactly:
- Module-level `_state`, `_listeners`, `_timer`, `_inFlight`.
- New callers get the cached manifest immediately on mount; no waiting for first poll.
- Shared polling loop (2s active / 30s idle). One in-flight fetch at a time.
- Export `resetManifest()` — called from `useProject.openProject()` alongside `resultCache.clear()` to prevent stale manifest from a previous project.
- Artifact stream events wired via a direct exported `_addManifestListener(fn)` on `useArtifactStream.ts` (no dynamic import). 250ms debounce before refetch.

### D2: Thread-local SQLite connection pool in RunRegistry
Replace open-close-per-call with `threading.local()`:
- Each thread holds one persistent `sqlite3.Connection`.
- Initialized once with WAL + PRAGMA settings.
- Validated via `SELECT 1` before reuse; reconnects on failure.
- `close()` method drains all thread connections.
- Implemented properly for future web-server compatibility.

### D3: Background pre-warm on `/project/load`
- Scoped to **frontend-facing artifacts only** — those with a `server:/results/...` entry in their `consumers` list in `_KNOWN_CATALOG` (~15-20 entries).
- Spawned as a daemon thread after the HTTP response is returned (load time unaffected).
- Cancelled via `threading.Event` (`_prewarm_cancel`) — module-level in `app.py`, replaced on each `/project/load`.

### D4: Increase server-side ResultCache to 256 entries
- Default bumped from 50 → 256.
- `SPARC_RESULT_CACHE_SIZE` env var preserved as override.

## Technical Approach

```
useManifest → module-level singleton (like useAvailability)
    ├─ _state: { manifest, loading, error, lastUpdated }
    ├─ _listeners: Set of setState callbacks
    ├─ shared polling timer (2s active / 30s idle)
    ├─ _addManifestListener ← registered by useArtifactStream at load time
    └─ resetManifest() ← called by useProject.openProject()

useArtifactStream → adds _addManifestListener export
    └─ client.addListener(fn) → fires notifyManifestArtifactWritten on artifact events

useProject.openProject()
    ├─ resultCache.clear()          ← existing
    └─ resetManifest()              ← new

/project/load endpoint
    └─ after response: spawn daemon thread
        ├─ cancel_event = threading.Event (stored as _prewarm_cancel)
        └─ for each frontend-facing artifact in manifest:
               if cancel_event.is_set(): break
               _read_or_404(stage, artifact_id)   → fills ResultCache

RunRegistry._sqlite() / sqlite_connection()
    └─ threading.local() → one persistent Connection per thread
        └─ SELECT 1 health-check before reuse; reconnect on failure

ResultCache: 50 → 256 (env var SPARC_RESULT_CACHE_SIZE still overrides)
```

## Acceptance Criteria

1. Navigate away from Insights page and back — results visible immediately, no loading spinner.
2. `useManifest` has no per-component state; module cache value available on mount.
3. `resetManifest()` is called on project open; stale manifest clears instantly.
4. `_addManifestListener` registered on `useArtifactStream` singleton at module load.
5. `RunRegistry.sqlite_connection()` reuses connections within a thread.
6. Pre-warm thread only reads frontend-facing artifacts; cancels cleanly on new project load.
7. `ResultCache` default max size is 256.
