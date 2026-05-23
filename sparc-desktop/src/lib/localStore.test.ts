/**
 * Tests for the LocalStore seam (Candidate 3).
 *
 * All tests use an isolated in-memory Storage stub so they are hermetic
 * and never touch the real localStorage.
 */

import { describe, it, expect, beforeEach } from "vitest";
import { createLocalStore, type SparcLocalKey } from "./localStore";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Minimal in-memory Storage implementation for testing. */
function makeMemoryStorage(): Storage {
  const store: Record<string, string> = {};
  return {
    getItem: (k) => store[k] ?? null,
    setItem: (k, v) => {
      store[k] = v;
    },
    removeItem: (k) => {
      delete store[k];
    },
    clear: () => {
      for (const k of Object.keys(store)) delete store[k];
    },
    get length() {
      return Object.keys(store).length;
    },
    key: (i) => Object.keys(store)[i] ?? null,
  };
}

// ---------------------------------------------------------------------------
// Slice 1 — basic get / set / remove
// ---------------------------------------------------------------------------

describe("LocalStoreAdapter — get/set/remove", () => {
  it("get returns null before any set", () => {
    const ls = createLocalStore(makeMemoryStorage());
    expect(ls.get("current-project")).toBeNull();
  });

  it("get returns stored value after set", () => {
    const ls = createLocalStore(makeMemoryStorage());
    ls.set("current-project", "/home/user/project");
    expect(ls.get("current-project")).toBe("/home/user/project");
  });

  it("remove makes get return null", () => {
    const ls = createLocalStore(makeMemoryStorage());
    ls.set("active-page", "Results");
    ls.remove("active-page");
    expect(ls.get("active-page")).toBeNull();
  });

  it("overwrite via set replaces the previous value", () => {
    const ls = createLocalStore(makeMemoryStorage());
    ls.set("workspace-dir", "/old");
    ls.set("workspace-dir", "/new");
    expect(ls.get("workspace-dir")).toBe("/new");
  });
});

// ---------------------------------------------------------------------------
// Slice 2 — clear wipes sparc keys without touching other keys
// ---------------------------------------------------------------------------

describe("LocalStoreAdapter — clear", () => {
  it("clear removes all sparc keys", () => {
    const storage = makeMemoryStorage();
    const ls = createLocalStore(storage);

    const keys: SparcLocalKey[] = [
      "current-project",
      "active-page",
      "recent-projects",
      "workspace-dir",
    ];
    for (const k of keys) ls.set(k, "value");

    ls.clear();

    for (const k of keys) {
      expect(ls.get(k)).toBeNull();
    }
  });

  it("clear does not remove non-sparc keys in the same storage", () => {
    const storage = makeMemoryStorage();
    const ls = createLocalStore(storage);

    // Write a non-sparc key directly to the backing storage
    storage.setItem("my-app-theme", "dark");
    ls.set("current-project", "/p");

    ls.clear();

    expect(storage.getItem("my-app-theme")).toBe("dark");
  });
});

// ---------------------------------------------------------------------------
// Slice 3 — key isolation between separate adapter instances
// ---------------------------------------------------------------------------

describe("LocalStoreAdapter — isolation", () => {
  it("two adapters sharing the same storage see each other's writes", () => {
    const storage = makeMemoryStorage();
    const a = createLocalStore(storage);
    const b = createLocalStore(storage);

    a.set("workspace-dir", "/shared");
    expect(b.get("workspace-dir")).toBe("/shared");
  });

  it("two adapters on different storage instances are independent", () => {
    const a = createLocalStore(makeMemoryStorage());
    const b = createLocalStore(makeMemoryStorage());

    a.set("active-page", "Dashboard");
    expect(b.get("active-page")).toBeNull();
  });
});
