// Preview requests: debounced at 120 ms, request_seq increments, superseded (older) responses
// and 409 superseded errors are ignored; the packed body decodes into delta + edited mask.
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../../../api/client";
import { packBits, pack } from "../../../api/binary";
import { decodePreview, postPreview, type PreviewRequest, type PreviewResult } from "../../../api/lab";
import { mockFetch } from "../../../test/render";
import { PREVIEW_DEBOUNCE_MS, PreviewController } from "../model/preview";

function result(seq: number, v: number): PreviewResult {
  return {
    delta: Float32Array.from([v, v]),
    edited: Uint8Array.from([1, 0]),
    summary: { mean: v, edited_mean: v, n_edited: 1, outside_share: 0, trust: "good", hatched: false, reasons: [], request_seq: seq },
  };
}

type Deferred = { body: PreviewRequest; resolve: (r: PreviewResult) => void; reject: (e: unknown) => void };

function harness() {
  const pending: Deferred[] = [];
  const applied: PreviewResult[] = [];
  const errors: unknown[] = [];
  const c = new PreviewController({
    send: (body) => new Promise<PreviewResult>((resolve, reject) => pending.push({ body, resolve, reject })),
    onResult: (r) => applied.push(r),
    onError: (e) => errors.push(e),
  });
  return { c, pending, applied, errors };
}

const tick = () => new Promise((r) => setTimeout(r, 0));

describe("preview controller", () => {
  afterEach(() => vi.useRealTimers());

  it("ignores a superseded (older request_seq) response that arrives late", async () => {
    const { c, pending, applied } = harness();
    void c.send({ edits: [] });
    void c.send({ edits: [] });
    expect(pending.map((p) => p.body.request_seq)).toEqual([1, 2]);
    pending[1].resolve(result(2, -0.5)); // newer first
    await tick();
    pending[0].resolve(result(1, -0.1)); // older arrives later
    await tick();
    expect(applied.map((r) => r.summary.request_seq)).toEqual([2]);
    expect(c.lastApplied).toBe(2);
  });

  it("applies in-order responses and never goes backwards", async () => {
    const { c, pending, applied } = harness();
    void c.send({ edits: [] });
    void c.send({ edits: [] });
    void c.send({ edits: [] });
    pending[0].resolve(result(1, -0.1));
    await tick();
    pending[2].resolve(result(3, -0.3));
    await tick();
    pending[1].resolve(result(2, -0.2));
    await tick();
    expect(applied.map((r) => r.summary.request_seq)).toEqual([1, 3]);
  });

  it("swallows 409 superseded and errors of older requests, reports the latest error", async () => {
    const { c, pending, applied, errors } = harness();
    void c.send({ edits: [] });
    void c.send({ edits: [] });
    pending[0].reject(new ApiError(409, "superseded", "a newer preview was requested"));
    await tick();
    expect(errors).toEqual([]);
    void c.send({ edits: [] });
    pending[1].reject(new ApiError(422, "validation", "old"));
    await tick();
    expect(errors).toEqual([]); // request 2 is older than request 3
    pending[2].reject(new ApiError(404, "no_emulator", "No emulator"));
    await tick();
    expect(errors).toHaveLength(1);
    expect((errors[0] as ApiError).code).toBe("no_emulator");
    expect(applied).toEqual([]);
  });

  it("reset drops responses still in flight", async () => {
    const { c, pending, applied } = harness();
    void c.send({ edits: [] });
    c.reset();
    pending[0].resolve(result(1, -1));
    await tick();
    expect(applied).toEqual([]);
  });

  it("debounces input at 120 ms and sends the latest edits", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const { c, pending } = harness();
    for (let i = 0; i < 10; i++) {
      c.request({ edits: [{ lever: "Pct_Canopy", mode: "add", amount: i }] });
      vi.advanceTimersByTime(PREVIEW_DEBOUNCE_MS - 20);
    }
    expect(pending).toHaveLength(0);
    vi.advanceTimersByTime(20);
    expect(pending).toHaveLength(1);
    expect(pending[0].body.edits[0].amount).toBe(9);
    expect(pending[0].body.request_seq).toBe(1);
  });

  it("uses the server's request_seq echo from X-SPARC-Summary", () => {
    const n = 10;
    const mask = new Uint8Array(n);
    mask[2] = 1;
    mask[9] = 1;
    const { buffer, offsets } = pack([
      { name: "delta", dtype: "float32", data: Float32Array.from({ length: n }, (_, i) => -i / 10) },
      { name: "edited", dtype: "uint8", data: packBits(mask) },
    ]);
    const r = decodePreview(buffer, JSON.stringify(offsets), JSON.stringify({ mean: -0.45, edited_mean: -0.55, n_edited: 2, outside_share: 0.1, trust: "rough", hatched: true, reasons: ["big"], request_seq: 7 }), 3);
    expect(Array.from(r.edited)).toEqual(Array.from(mask));
    expect(r.delta[9]).toBeCloseTo(-0.9);
    expect(r.summary).toMatchObject({ request_seq: 7, hatched: true, trust: "rough", n_edited: 2 });
  });
});

describe("postPreview over the wire", () => {
  it("POSTs the edits with request_seq and decodes the packed body; errors become ApiErrors", async () => {
    const mask = Uint8Array.from([0, 1, 1]);
    const { buffer, offsets } = pack([
      { name: "delta", dtype: "float32", data: Float32Array.from([0, -0.2, -0.4]) },
      { name: "edited", dtype: "uint8", data: packBits(mask) },
    ]);
    const m = mockFetch({
      "POST /api/runs/r1/preview": () => ({
        raw: buffer,
        headers: {
          "content-type": "application/octet-stream",
          "X-SPARC-Offsets": JSON.stringify(offsets),
          "X-SPARC-Summary": JSON.stringify({ mean: -0.2, edited_mean: -0.3, n_edited: 2, outside_share: 0, trust: "good", hatched: false, reasons: [], request_seq: 5 }),
        },
      }),
      "POST /api/runs/r2/preview": { status: 404, body: { error: { code: "no_emulator", message: "Build the emulator first", action: { kind: "build_emulator", label: "Build emulator", method: "POST", path: "/api/runs/r2/actions/emulator" } } } },
    });
    try {
      const r = await postPreview("r1", { edits: [{ lever: "Pct_Canopy", mode: "add", amount: 10 }], request_seq: 5 });
      expect(m.calls[0].body).toEqual({ edits: [{ lever: "Pct_Canopy", mode: "add", amount: 10 }], request_seq: 5 });
      expect(Array.from(r.edited)).toEqual([0, 1, 1]);
      expect(r.summary.request_seq).toBe(5);
      await expect(postPreview("r2", { edits: [], request_seq: 1 })).rejects.toMatchObject({ status: 404, code: "no_emulator", action: { kind: "build_emulator" } });
    } finally {
      m.restore();
    }
  });
});
