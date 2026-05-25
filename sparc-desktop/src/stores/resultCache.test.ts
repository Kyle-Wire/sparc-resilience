/**
 * T48 — resultCache TTL eviction tests
 *
 * Verifies that the result cache evicts stale entries when they are read
 * past their TTL, preventing unbounded memory growth in long sessions.
 */
import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { useResultCacheStore } from "@/stores/resultCache";

const TTL_MS = 30 * 60 * 1000; // 30 minutes, must match implementation

function resetCache() {
  useResultCacheStore.setState({ cache: {} });
}

describe("resultCache — basic operations (regression)", () => {
  beforeEach(resetCache);

  it("stores and retrieves a value by key", () => {
    useResultCacheStore.getState().set("s0:correlogram", { foo: 1 });
    const entry = useResultCacheStore.getState().cache["s0:correlogram"];
    expect(entry?.data).toEqual({ foo: 1 });
  });

  it("invalidateStage removes all keys for that stage", () => {
    useResultCacheStore.getState().set("s2:model_perf", { r2: 0.9 });
    useResultCacheStore.getState().set("s2:predictions", [1, 2, 3]);
    useResultCacheStore.getState().set("s0:correlogram", { keep: true });
    useResultCacheStore.getState().invalidateStage(2);
    expect(useResultCacheStore.getState().cache["s2:model_perf"]).toBeUndefined();
    expect(useResultCacheStore.getState().cache["s2:predictions"]).toBeUndefined();
    expect(useResultCacheStore.getState().cache["s0:correlogram"]).toBeDefined();
  });

  it("clear removes everything", () => {
    useResultCacheStore.getState().set("s0:correlogram", { a: 1 });
    useResultCacheStore.getState().set("s3:cate_map", { b: 2 });
    useResultCacheStore.getState().clear();
    expect(Object.keys(useResultCacheStore.getState().cache)).toHaveLength(0);
  });
});

describe("resultCache — TTL eviction", () => {
  beforeEach(() => {
    resetCache();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("a fresh entry is returned when TTL has not elapsed", () => {
    useResultCacheStore.getState().set("s0:correlogram", { fresh: true });
    vi.advanceTimersByTime(TTL_MS - 1);
    const entry = useResultCacheStore.getState().getIfFresh("s0:correlogram");
    expect(entry).not.toBeNull();
    expect((entry as { data: unknown }).data).toEqual({ fresh: true });
  });

  it("a stale entry is evicted and returns null after TTL elapses", () => {
    useResultCacheStore.getState().set("s2:model_perf", { stale: true });
    vi.advanceTimersByTime(TTL_MS + 1);
    const entry = useResultCacheStore.getState().getIfFresh("s2:model_perf");
    expect(entry).toBeNull();
    // Also evicted from the cache map
    expect(useResultCacheStore.getState().cache["s2:model_perf"]).toBeUndefined();
  });

  it("different keys expire independently", () => {
    useResultCacheStore.getState().set("s0:old", { old: true });
    vi.advanceTimersByTime(TTL_MS - 5000);
    useResultCacheStore.getState().set("s0:new", { new: true });
    vi.advanceTimersByTime(5001); // only "old" has now exceeded TTL

    expect(useResultCacheStore.getState().getIfFresh("s0:old")).toBeNull();
    expect(useResultCacheStore.getState().getIfFresh("s0:new")).not.toBeNull();
  });

  it("invalidateStage still works alongside TTL entries", () => {
    useResultCacheStore.getState().set("s2:perf", { r2: 0.8 });
    vi.advanceTimersByTime(1000);
    useResultCacheStore.getState().invalidateStage(2);
    expect(useResultCacheStore.getState().getIfFresh("s2:perf")).toBeNull();
  });
});
