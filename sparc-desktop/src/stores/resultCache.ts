/**
 * Global result cache backed by Zustand.
 *
 * Panels read from this store instead of holding data in local component
 * state, so results survive React Router navigation (unmount/remount) without
 * re-fetching from the server.
 *
 * Cache keys follow the convention `s{stage}:{name}[:{param}…]`, e.g.:
 *   s0:correlogram
 *   s2:model_performance
 *   s3:cate_map:ndvi:true
 *
 * Invalidation is stage-scoped: when an artifact_written WebSocket event
 * fires for stage N, all `s{N}:*` keys are evicted so panels refetch on
 * their next render.
 */
import { create } from "zustand";

/** Entries older than this are considered stale and evicted on read. */
export const CACHE_TTL_MS = 30 * 60 * 1000; // 30 minutes

export interface CacheEntry {
  data: unknown;
  fetchedAt: number;
}

interface ResultCacheState {
  cache: Record<string, CacheEntry>;
  /** Write or overwrite a cache slot. */
  set: (key: string, data: unknown) => void;
  /**
   * Return the cache entry if it exists and has not exceeded the TTL.
   * Evicts and returns null if the entry is stale.
   */
  getIfFresh: (key: string) => CacheEntry | null;
  /** Evict all keys whose name starts with `s{stage}:`. */
  invalidateStage: (stage: string | number) => void;
  /** Evict a single key. */
  invalidateKey: (key: string) => void;
  /** Drop the entire cache (used on project unload). */
  clear: () => void;
}

export const useResultCacheStore = create<ResultCacheState>((set, get) => ({
  cache: {},

  set: (key, data) =>
    set((s) => ({
      cache: { ...s.cache, [key]: { data, fetchedAt: Date.now() } },
    })),

  getIfFresh: (key) => {
    const entry = get().cache[key];
    if (!entry) return null;
    if (Date.now() - entry.fetchedAt > CACHE_TTL_MS) {
      // Evict on read
      set((s) => {
        const next = { ...s.cache };
        delete next[key];
        return { cache: next };
      });
      return null;
    }
    return entry;
  },

  invalidateStage: (stage) =>
    set((s) => {
      const prefix = `s${stage}:`;
      const next: Record<string, CacheEntry> = {};
      for (const [k, v] of Object.entries(s.cache)) {
        if (!k.startsWith(prefix)) next[k] = v;
      }
      return { cache: next };
    }),

  invalidateKey: (key) =>
    set((s) => {
      const next = { ...s.cache };
      delete next[key];
      return { cache: next };
    }),

  clear: () => set({ cache: {} }),
}));
