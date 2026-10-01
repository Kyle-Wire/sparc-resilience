import { afterEach, describe, expect, it } from "vitest";
import { ApiError, api, apiUrl, errorFromResponse, getBin, isApiError, putRaw, request } from "../api/client";
import { pack } from "../api/binary";
import { mockFetch } from "./render";

let restore: (() => void) | null = null;
afterEach(() => {
  restore?.();
  restore = null;
});

describe("ApiError mapping from the error envelope (api.md §0.3)", () => {
  it("maps code, message, detail and action", () => {
    const e = errorFromResponse(404, {
      error: {
        code: "output_missing",
        message: "causal.json is not in this run",
        detail: { output: "causal", produced_by: "stage:S6", expected_path: "causal.json" },
        action: { kind: "resume", label: "Resume to compute S6", method: "POST", path: "/api/runs/r1/resume", body: {} },
      },
    });
    expect(e).toBeInstanceOf(ApiError);
    expect(isApiError(e)).toBe(true);
    expect(e.status).toBe(404);
    expect(e.code).toBe("output_missing");
    expect(e.message).toBe("causal.json is not in this run");
    expect(e.detail).toEqual({ output: "causal", produced_by: "stage:S6", expected_path: "causal.json" });
    expect(e.action).toEqual({ kind: "resume", label: "Resume to compute S6", method: "POST", path: "/api/runs/r1/resume", body: {} });
  });
  it("exposes validation errors", () => {
    const e = errorFromResponse(422, { error: { code: "validation", message: "Invalid body", detail: { errors: [{ path: "threads", message: "must be ≥ 1", code: "min" }] } } });
    expect(e.validationErrors).toEqual([{ path: "threads", message: "must be ≥ 1", code: "min" }]);
    expect(e.action).toBeNull();
  });
  it("falls back for bodies that are not envelopes", () => {
    expect(errorFromResponse(401, null).code).toBe("unauthorized");
    expect(errorFromResponse(500, "<html>boom</html>", "Internal Server Error")).toMatchObject({ code: "internal", message: "Internal Server Error" });
    expect(errorFromResponse(404, { detail: "Not Found" })).toMatchObject({ code: "not_found", message: "Not Found" });
    expect(errorFromResponse(418, "short text")).toMatchObject({ code: "http_418", message: "short text" });
  });
  it("throws ApiError from request() for non-2xx JSON responses", async () => {
    restore = mockFetch({
      "POST /api/jobs/j_1/cancel": { status: 409, body: { error: { code: "not_cancellable", message: "Job already finished", detail: { status: "succeeded" } } } },
    }).restore;
    const err = await api.post("/api/jobs/j_1/cancel").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ status: 409, code: "not_cancellable", message: "Job already finished", detail: { status: "succeeded" } });
  });
  it("reports network failures as code 'network'", async () => {
    const orig = globalThis.fetch;
    globalThis.fetch = (() => Promise.reject(new TypeError("Failed to fetch"))) as typeof fetch;
    restore = () => (globalThis.fetch = orig);
    await expect(request("GET", "/api/health")).rejects.toMatchObject({ status: 0, code: "network" });
  });
});

describe("requests", () => {
  it("builds query strings", () => {
    expect(apiUrl("/api/jobs", { status: "active", kind: ["run.core", "post.planner"], project: null, limit: 50 })).toBe("/api/jobs?status=active&kind=run.core%2Cpost.planner&limit=50");
    expect(apiUrl("/api/x?a=1", { b: true })).toBe("/api/x?a=1&b=true");
  });
  it("sends JSON bodies with the content type and parses JSON replies", async () => {
    const m = mockFetch({ "POST /api/findings": { status: 201, body: { id: "fd_1" } } });
    restore = m.restore;
    const r = await api.post<{ id: string }>("/api/findings", { project_id: "p_1", title: "x" });
    expect(r.id).toBe("fd_1");
    expect(m.calls[0]).toMatchObject({ method: "POST", url: "/api/findings", body: { project_id: "p_1", title: "x" } });
  });
  it("getBin picks the dtype from X-SPARC-Dtype", async () => {
    const f = Float32Array.from([1.5, NaN, -2]);
    restore = mockFetch({
      "GET /api/runs/r1/layers/obs.bin": { raw: f.buffer.slice(0), headers: { "content-type": "application/octet-stream", "X-SPARC-Dtype": "float32", "X-SPARC-Length": "3", ETag: '"abc"' } },
    }).restore;
    const r = await getBin<Float32Array>("/api/runs/r1/layers/obs.bin");
    expect(r.dtype).toBe("float32");
    expect(r.length).toBe(3);
    expect(r.etag).toBe('"abc"');
    expect(r.data[0]).toBe(1.5);
    expect(Number.isNaN(r.data[1])).toBe(true);
  });
  it("getBin exposes packed offsets", async () => {
    const { buffer, offsets } = pack([
      { name: "ix", dtype: "int32", data: Int32Array.from([0, 1]) },
      { name: "zone", dtype: "int16", data: Int16Array.from([3, 4]) },
    ]);
    restore = mockFetch({ "GET /api/runs/r1/grid.bin": { raw: buffer, headers: { "content-type": "application/octet-stream", "X-SPARC-Offsets": JSON.stringify(offsets) } } }).restore;
    const r = await getBin("/api/runs/r1/grid.bin");
    expect(r.offsets).toEqual(offsets);
    expect(r.buffer.byteLength).toBe(buffer.byteLength);
  });
  it("getBin rejects truncated or malformed binaries instead of misaligning rows", async () => {
    const f = Float32Array.from([1, 2, 3]);
    restore = mockFetch({
      "GET /api/runs/r1/layers/short.bin": { raw: f.buffer.slice(0, 8), headers: { "content-type": "application/octet-stream", "X-SPARC-Dtype": "float32", "X-SPARC-Length": "3" } },
      "GET /api/runs/r1/layers/odd.bin": { raw: f.buffer.slice(0, 7), headers: { "content-type": "application/octet-stream", "X-SPARC-Dtype": "float32" } },
      "GET /api/runs/r1/layers/kind.bin": { raw: f.buffer.slice(0), headers: { "content-type": "application/octet-stream", "X-SPARC-Dtype": "float16" } },
      "GET /api/runs/r1/grid.bin": { raw: new ArrayBuffer(8), headers: { "content-type": "application/octet-stream", "X-SPARC-Offsets": '[{"name":"ix","dtype":"int32","offset":0,"length":5}]' } },
    }).restore;
    await expect(getBin("/api/runs/r1/layers/short.bin")).rejects.toMatchObject({ code: "bad_response", message: expect.stringContaining("X-SPARC-Length is 3") });
    await expect(getBin("/api/runs/r1/layers/odd.bin")).rejects.toMatchObject({ code: "bad_response" });
    await expect(getBin("/api/runs/r1/layers/kind.bin")).rejects.toMatchObject({ code: "bad_response", message: expect.stringContaining("float16") });
    await expect(getBin("/api/runs/r1/grid.bin")).rejects.toMatchObject({ code: "bad_response", message: expect.stringContaining('"ix" overruns') });
  });
  it("getBin errors carry the envelope", async () => {
    restore = mockFetch({ "GET /api/runs/r1/layers/nope.bin": { status: 404, body: { error: { code: "unknown_layer", message: "No layer nope" } } } }).restore;
    await expect(getBin("/api/runs/r1/layers/nope.bin")).rejects.toMatchObject({ code: "unknown_layer" });
  });
  it("putRaw streams a raw body with its content type", async () => {
    const m = mockFetch({ "PUT /api/runs/r1/blobs": (_u, init) => ({ status: 201, body: { blob_id: "bl_1", bytes: (init.body as ArrayBuffer).byteLength, ct: new Headers(init.headers).get("Content-Type") } }) });
    restore = m.restore;
    const r = await putRaw<{ blob_id: string; bytes: number; ct: string }>("/api/runs/r1/blobs", Uint8Array.from([1, 2, 3]), { query: { kind: "mask" } });
    expect(r).toEqual({ blob_id: "bl_1", bytes: 3, ct: "application/octet-stream" });
    expect(m.calls[0].url).toBe("/api/runs/r1/blobs?kind=mask");
  });
});
