// Draft store (undo/redo of 50 steps, coalescing, brush steps) and the autosave loop
// (create → patch, 409 conflict_revision → fork and keep editing on the new revision).
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../../../api/client";
import { forkScenario, patchScenario, type Scenario, type ScenarioDoc } from "../../../api/lab";
import { mockFetch } from "../../../test/render";
import { newEdit } from "../model/doc";
import { COALESCE_MS, HISTORY_LIMIT, canRedo, canUndo, createAutosave, createDraftStore, docForSave, isDirty } from "../model/draft";

function scenario(id: string, doc: ScenarioDoc, extra: Partial<Scenario> = {}): Scenario {
  return {
    id,
    project_id: "p_1",
    revision: 1,
    parent_id: null,
    children: [],
    doc,
    content_hash: "h",
    status: "draft",
    created_utc: "2026-10-01T10:00:00Z",
    updated_utc: "2026-10-01T10:00:00Z",
    results: [],
    ...extra,
  };
}

const setAmount = (v: number) => (d: ScenarioDoc): ScenarioDoc => ({ ...d, edits: [{ ...(d.edits[0] ?? newEdit("Pct_Canopy")), amount: v }] });

describe("draft store: undo/redo", () => {
  it("keeps exactly 50 undo steps and redoes them all", () => {
    let t = 0;
    const store = createDraftStore({ n: 4, now: () => (t += 10_000) });
    for (let i = 1; i <= 60; i++) store.getState().change(setAmount(i));
    expect(store.getState().doc.edits[0].amount).toBe(60);
    expect(store.getState().past.length).toBe(HISTORY_LIMIT);
    for (let i = 0; i < HISTORY_LIMIT; i++) store.getState().undo();
    // 60 changes, 50 undone: the state after change #10
    expect(store.getState().doc.edits[0].amount).toBe(10);
    expect(canUndo(store.getState())).toBe(false);
    store.getState().undo(); // the 51st undo is a no-op
    expect(store.getState().doc.edits[0].amount).toBe(10);
    for (let i = 0; i < HISTORY_LIMIT; i++) store.getState().redo();
    expect(store.getState().doc.edits[0].amount).toBe(60);
    expect(canRedo(store.getState())).toBe(false);
  });

  it("a new change after undo drops the redo branch", () => {
    let t = 0;
    const store = createDraftStore({ n: 4, now: () => (t += 10_000) });
    store.getState().change(setAmount(1));
    store.getState().change(setAmount(2));
    store.getState().undo();
    expect(canRedo(store.getState())).toBe(true);
    store.getState().change(setAmount(5));
    expect(canRedo(store.getState())).toBe(false);
    store.getState().undo();
    expect(store.getState().doc.edits[0].amount).toBe(1);
  });

  it("coalesces rapid changes with the same key into one step, and ignores no-op changes", () => {
    let t = 0;
    const store = createDraftStore({ n: 4, now: () => t });
    store.getState().change((d) => ({ ...d, name: "A" }), { coalesce: "name" });
    t += COALESCE_MS / 4;
    store.getState().change((d) => ({ ...d, name: "Ab" }), { coalesce: "name" });
    t += COALESCE_MS / 4;
    store.getState().change((d) => ({ ...d, name: "Abc" }), { coalesce: "name" });
    expect(store.getState().past.length).toBe(1);
    const v = store.getState().version;
    store.getState().change((d) => ({ ...d, name: "Abc" }));
    expect(store.getState().version).toBe(v);
    t += COALESCE_MS * 2;
    store.getState().change((d) => ({ ...d, name: "Abcd" }), { coalesce: "name" });
    expect(store.getState().past.length).toBe(2);
    store.getState().undo();
    expect(store.getState().doc.name).toBe("Abc");
  });

  it("brush strokes are undo steps and restore the live arrays in place", () => {
    let t = 0;
    const store = createDraftStore({ n: 5, now: () => (t += 10_000) });
    const live = store.brushLayers.array("Pct_Canopy");
    const stroke = { mode: "add" as const, amount: 10, base: Float32Array.from([0, 10, 20, 30, 95]), bounds: [0, 100] as [number, number] };
    store.brushLayers.paint("Pct_Canopy", [0, 1], stroke);
    store.getState().commitBrush();
    store.brushLayers.paint("Pct_Canopy", [4], stroke);
    store.getState().commitBrush();
    expect(Array.from(live)).toEqual([10, 10, 0, 0, 5]);
    store.getState().undo();
    expect(store.brushLayers.array("Pct_Canopy")).toBe(live); // same array object
    expect(Array.from(live)).toEqual([10, 10, 0, 0, 0]);
    store.getState().undo();
    expect(Array.from(live)).toEqual([0, 0, 0, 0, 0]);
    store.getState().redo();
    store.getState().redo();
    expect(Array.from(live)).toEqual([10, 10, 0, 0, 5]);
  });

  it("load replaces the draft and clears the history", () => {
    const store = createDraftStore({ n: 2 });
    store.getState().change(setAmount(3));
    const doc: ScenarioDoc = { name: "Loaded", edits: [newEdit("Albedo", "set", 0.35)] };
    store.getState().load(scenario("sc_1", doc, { revision: 4 }));
    const s = store.getState();
    expect(s.sid).toBe("sc_1");
    expect(s.revision).toBe(4);
    expect(s.doc).toEqual(doc);
    expect(s.past).toEqual([]);
    expect(isDirty(s)).toBe(false);
  });
});

describe("autosave", () => {
  beforeEach(() => vi.useFakeTimers({ toFake: ["setInterval", "clearInterval", "setTimeout", "clearTimeout"] }));
  afterEach(() => vi.useRealTimers());

  it("creates a new draft once, then patches it every 2 s while dirty", async () => {
    const store = createDraftStore({ n: 2 });
    const created = scenario("sc_new", store.getState().doc);
    const create = vi.fn(async (d: ScenarioDoc) => ({ ...created, doc: d }));
    const patch = vi.fn(async (id: string, d: ScenarioDoc) => scenario(id, d, { updated_utc: "2026-10-01T10:00:05Z" }));
    const fork = vi.fn();
    const onCreated = vi.fn();
    const auto = createAutosave(store, { create, patch, fork }, { onCreated });
    auto.start();
    store.getState().change(setAmount(1));
    await vi.advanceTimersByTimeAsync(2000);
    expect(create).toHaveBeenCalledTimes(1);
    expect(onCreated).toHaveBeenCalledWith(expect.objectContaining({ id: "sc_new" }));
    expect(store.getState().sid).toBe("sc_new");
    expect(isDirty(store.getState())).toBe(false);
    await vi.advanceTimersByTimeAsync(4000);
    expect(patch).not.toHaveBeenCalled(); // nothing changed
    store.getState().change(setAmount(2));
    await vi.advanceTimersByTimeAsync(2000);
    expect(patch).toHaveBeenCalledTimes(1);
    expect(patch.mock.calls[0][0]).toBe("sc_new");
    expect(patch.mock.calls[0][1].edits[0].amount).toBe(2);
    expect(fork).not.toHaveBeenCalled();
    auto.stop();
  });

  it("flush saves immediately (blur) and saves again a change made during the save", async () => {
    vi.useRealTimers();
    const store = createDraftStore({ n: 2 });
    store.getState().load(scenario("sc_1", { name: "x", edits: [] }));
    let release: () => void = () => {};
    const patch = vi.fn((id: string, d: ScenarioDoc) => new Promise<Scenario>((res) => (release = () => res(scenario(id, d)))));
    const auto = createAutosave(store, { create: vi.fn(), patch, fork: vi.fn() });
    store.getState().change(setAmount(1));
    const done = auto.flush();
    await vi.waitFor(() => expect(patch).toHaveBeenCalledTimes(1));
    store.getState().change(setAmount(2)); // during the save
    const again = auto.flush();
    release();
    await done;
    await vi.waitFor(() => expect(patch).toHaveBeenCalledTimes(2));
    release();
    await again;
    expect(patch.mock.calls[1][1].edits[0].amount).toBe(2);
    expect(isDirty(store.getState())).toBe(false);
  });

  it("forks on 409 conflict_revision and keeps editing on the new revision", async () => {
    const store = createDraftStore({ n: 2 });
    store.getState().load(scenario("sc_old", { name: "Exact one", edits: [newEdit("Pct_Canopy", "add", 10)] }, { status: "exact", revision: 3 }));
    const patch = vi.fn(async (id: string, d: ScenarioDoc) => {
      if (id === "sc_old") throw new ApiError(409, "conflict_revision", "This scenario has exact results; fork it");
      return scenario(id, d, { revision: 4, parent_id: "sc_old" });
    });
    const fork = vi.fn(async (id: string, d: ScenarioDoc) => scenario("sc_fork", d, { revision: 4, parent_id: id }));
    const onForked = vi.fn();
    const auto = createAutosave(store, { create: vi.fn(), patch, fork }, { onForked });
    store.getState().change(setAmount(20));
    await auto.flush();
    expect(fork).toHaveBeenCalledWith("sc_old", expect.objectContaining({ edits: [expect.objectContaining({ amount: 20 })] }));
    expect(onForked).toHaveBeenCalledWith(expect.objectContaining({ id: "sc_fork" }), "sc_old");
    expect(store.getState().sid).toBe("sc_fork");
    expect(store.getState().revision).toBe(4);
    expect(store.getState().forkedFrom).toBe("sc_old");
    // the edit survived, and further edits patch the fork
    store.getState().change(setAmount(30));
    await auto.flush();
    expect(patch).toHaveBeenLastCalledWith("sc_fork", expect.objectContaining({ edits: [expect.objectContaining({ amount: 30 })] }));
    expect(fork).toHaveBeenCalledTimes(1);
  });

  it("the fork flow works over the wire: PATCH 409 → POST fork → PATCH the new id", async () => {
    vi.useRealTimers();
    const doc0: ScenarioDoc = { name: "Corridor", edits: [newEdit("Pct_Canopy", "add", 10)] };
    const m = mockFetch({
      "PATCH /api/scenarios/sc_a": { status: 409, body: { error: { code: "conflict_revision", message: "has exact results", action: { kind: "open", label: "Fork" } } } },
      "POST /api/scenarios/sc_a/fork": (_u, init) => ({ status: 201, body: scenario("sc_b", JSON.parse(String(init.body)).doc, { revision: 2, parent_id: "sc_a" }) }),
      "PATCH /api/scenarios/sc_b": (_u, init) => ({ body: scenario("sc_b", JSON.parse(String(init.body)).doc, { revision: 2, parent_id: "sc_a" }) }),
    });
    try {
      const store = createDraftStore({ n: 2 });
      store.getState().load(scenario("sc_a", doc0, { status: "exact" }));
      const auto = createAutosave(store, {
        create: () => Promise.reject(new Error("unused")),
        patch: (id, d) => patchScenario(id, { doc: d }),
        fork: (id, d) => forkScenario(id, { doc: d }),
      });
      store.getState().change(setAmount(15));
      await auto.flush();
      store.getState().change((d) => ({ ...d, name: "Corridor v2" }));
      await auto.flush();
      expect(m.calls.map((c) => `${c.method} ${c.url}`)).toEqual(["PATCH /api/scenarios/sc_a", "POST /api/scenarios/sc_a/fork", "PATCH /api/scenarios/sc_b"]);
      expect((m.calls[1].body as { doc: ScenarioDoc }).doc.edits[0].amount).toBe(15);
      expect((m.calls[2].body as { doc: ScenarioDoc }).doc.name).toBe("Corridor v2");
      expect(store.getState().sid).toBe("sc_b");
    } finally {
      m.restore();
    }
  });

  it("uploads brushed levers once per change and saves them as per-cell blob edits", async () => {
    const store = createDraftStore({ n: 3 });
    store.getState().load(scenario("sc_1", { name: "b", edits: [] }));
    store.brushLayers.paint("Pct_Canopy", [1], { mode: "add", amount: 5, base: Float32Array.from([1, 2, 3]), bounds: [0, 100] });
    store.getState().commitBrush();
    const uploadBrush = vi.fn(async () => "bl_1");
    const patch = vi.fn(async (id: string, d: ScenarioDoc) => scenario(id, d));
    const auto = createAutosave(store, { create: vi.fn(), patch, fork: vi.fn(), uploadBrush });
    await auto.flush();
    expect(uploadBrush).toHaveBeenCalledTimes(1);
    expect(patch.mock.calls[0][1].edits).toEqual([{ lever: "Pct_Canopy", mode: "per_cell", per_cell_ref: "blob:bl_1", label: "brush" }]);
    store.getState().change((d) => ({ ...d, name: "renamed" }));
    await auto.flush();
    expect(uploadBrush).toHaveBeenCalledTimes(1); // unchanged brush: the cached blob is reused
    expect(docForSave({ name: "x", edits: [] }, {})).toEqual({ name: "x", edits: [] });
  });

  it("reports a failed save and retries after the next change", async () => {
    const store = createDraftStore({ n: 2 });
    store.getState().load(scenario("sc_1", { name: "x", edits: [] }));
    const patch = vi.fn().mockRejectedValueOnce(new ApiError(0, "network", "Studio server is not reachable")).mockImplementation(async (id: string, d: ScenarioDoc) => scenario(id, d));
    const onError = vi.fn();
    const auto = createAutosave(store, { create: vi.fn(), patch, fork: vi.fn() }, { onError });
    auto.start();
    store.getState().change(setAmount(1));
    await vi.advanceTimersByTimeAsync(2000);
    expect(store.getState().save).toBe("error");
    expect(onError).toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(4000);
    expect(patch).toHaveBeenCalledTimes(1); // no hammering after an error
    store.getState().change(setAmount(2));
    await vi.advanceTimersByTimeAsync(2000);
    expect(patch).toHaveBeenCalledTimes(2);
    expect(store.getState().save).toBe("saved");
    auto.stop();
  });
});
