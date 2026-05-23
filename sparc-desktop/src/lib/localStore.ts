/**
 * sparc-desktop/src/lib/localStore.ts
 *
 * A thin, typed adapter over `localStorage` (or any `Storage`-compatible
 * implementation) that:
 *
 * - Centralises all "sparc-*" key names so callers use semantic keys.
 * - Can be swapped for an in-memory implementation in tests.
 */

/** Semantic keys used by SPARC frontend modules. */
export type SparcLocalKey =
  | "current-project"
  | "active-page"
  | "recent-projects"
  | "workspace-dir";

/** Full localStorage key for each semantic key. */
const KEY_MAP: Record<SparcLocalKey, string> = {
  "current-project": "sparc-current-project",
  "active-page": "sparc-active-page",
  "recent-projects": "sparc-recent-projects",
  "workspace-dir": "sparc-workspace-dir",
};

/** All raw keys managed by SPARC (used by `clear()`). */
const ALL_KEYS = Object.values(KEY_MAP);

export interface LocalStoreAdapter {
  get(key: SparcLocalKey): string | null;
  set(key: SparcLocalKey, value: string): void;
  remove(key: SparcLocalKey): void;
  /** Remove every "sparc-*" key without touching any other keys. */
  clear(): void;
}

/**
 * Create a `LocalStoreAdapter` backed by `storage`.
 *
 * Pass an in-memory `Storage` stub in tests; omit (or pass `localStorage`)
 * for the production default.
 */
export function createLocalStore(storage: Storage): LocalStoreAdapter {
  return {
    get(key) {
      try {
        return storage.getItem(KEY_MAP[key]);
      } catch {
        return null;
      }
    },
    set(key, value) {
      try {
        storage.setItem(KEY_MAP[key], value);
      } catch {
        /* quota / private-mode — silently ignore */
      }
    },
    remove(key) {
      try {
        storage.removeItem(KEY_MAP[key]);
      } catch {
        /* ignore */
      }
    },
    clear() {
      for (const rawKey of ALL_KEYS) {
        try {
          storage.removeItem(rawKey);
        } catch {
          /* ignore */
        }
      }
    },
  };
}

/** Production singleton backed by `window.localStorage`. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const localStore: LocalStoreAdapter = createLocalStore(
  typeof localStorage !== "undefined"
    ? localStorage
    : (Object.create(null) as Storage), // SSR / node fallback (no-op)
);
