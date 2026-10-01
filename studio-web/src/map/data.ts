// Run map data (api.md §6.2): grid geometry, the layer catalogue and layer binaries.
// Decoded layers live in a byte-capped LRU, so switching back to a layer costs no request and
// no decode (SPEC §12.5: layer switch < 50 ms including decode from cache).
import { api, getBin, putRaw } from "../api/client";
import { packBits } from "../api/binary";
import { useResource } from "../api/resource";
import type { GridMeta, LayerGroup, LayerMeta, SelectionSpec } from "../api/types";
import { parseGridBin, type GridData } from "./grid";

export type LayerValues = Float32Array | Uint8Array;

const MAX_BYTES = 96 * 1024 * 1024;
const cache = new Map<string, LayerValues>();
let cachedBytes = 0;

function remember(key: string, v: LayerValues): void {
  if (cache.has(key)) {
    cache.delete(key);
  } else cachedBytes += v.byteLength;
  cache.set(key, v);
  for (const [k, old] of cache) {
    if (cachedBytes <= MAX_BYTES) break;
    if (k === key) continue;
    cache.delete(k);
    cachedBytes -= old.byteLength;
  }
}

/** Drop cached layers of a run (e.g. after a resume rewrote its outputs). */
export function forgetRunLayers(rid: string): void {
  for (const [k, v] of [...cache]) {
    if (k.startsWith(`${rid}|`)) {
      cache.delete(k);
      cachedBytes -= v.byteLength;
    }
  }
}

export function cachedLayerCount(): number {
  return cache.size;
}

const inflight = new Map<string, Promise<LayerValues>>();

/**
 * Content signature of a layer from its catalogue entry. While a run is still writing, a
 * layer's file can change under the same key; the catalogue is refetched on
 * `output.written`, and new stats mean new bytes, so the cached copy is not reused.
 */
export function layerSignature(meta: LayerMeta): string {
  const s = meta.stats;
  return [meta.dtype, s.n, s.lo, s.hi, s.mean, s.p2, s.p50, s.p98].map((v) => (v === null || v === undefined ? "" : String(v))).join(",");
}

/**
 * Loader for `GET /api/runs/{rid}/layers/{key}.bin`. Cached per run, layer key, grid etag
 * (a re-index invalidates it) and layer signature; concurrent requests share one fetch.
 */
export function runLayerLoader(rid: string, gridEtag = ""): (meta: LayerMeta) => Promise<LayerValues> {
  return (meta: LayerMeta) => {
    const k = `${rid}|${gridEtag}|${meta.key}|${layerSignature(meta)}`;
    const hit = cache.get(k);
    if (hit) {
      remember(k, hit);
      return Promise.resolve(hit);
    }
    let p = inflight.get(k);
    if (!p) {
      p = getBin<LayerValues>(`/api/runs/${encodeURIComponent(rid)}/layers/${encodeURIComponent(meta.key)}.bin`, meta.dtype)
        .then((r) => {
          const v = r.data as LayerValues;
          remember(k, v);
          return v;
        })
        .finally(() => inflight.delete(k));
      inflight.set(k, p);
    }
    return p;
  };
}

/** GridMeta + grid.bin → GridData. */
export async function fetchRunGrid(rid: string, signal?: AbortSignal): Promise<GridData> {
  const base = `/api/runs/${encodeURIComponent(rid)}`;
  const [meta, bin] = await Promise.all([api.get<GridMeta>(`${base}/grid`, undefined, signal), getBin(`${base}/grid.bin`, "uint8", { signal })]);
  if (!bin.offsets) throw new Error("grid.bin came without X-SPARC-Offsets");
  return parseGridBin(meta, bin.buffer, bin.offsets);
}

/** The run's grid (immutable for finished runs). */
export function useRunGrid(rid: string | null, opts: { immutable?: boolean } = {}) {
  return useResource<GridData>(rid ? `run:${rid}:grid` : null, (s) => fetchRunGrid(rid!, s), { tags: rid ? [`run:${rid}`] : [], immutable: opts.immutable });
}

/** The run's layer catalogue (refreshes when outputs are written). */
export function useRunLayers(rid: string | null) {
  return useResource<LayerGroup[]>(
    rid ? `run:${rid}:layers` : null,
    async (s) => (await api.get<{ groups: LayerGroup[] }>(`/api/runs/${encodeURIComponent(rid!)}/layers`, undefined, s)).groups,
    { tags: rid ? [`run:${rid}`, `run:${rid}:layers`] : [] },
  );
}

/**
 * Upload a brushed mask as a selection blob (`PUT /blobs?kind=mask`) and return the portable
 * `{kind: "blob"}` spec that analysis tools and scenarios accept.
 */
export async function uploadMaskSelection(rid: string, mask: ArrayLike<number>): Promise<SelectionSpec> {
  const r = await putRaw<{ blob_id: string; bytes: number }>(`/api/runs/${encodeURIComponent(rid)}/blobs`, packBits(mask), {
    query: { kind: "mask" },
    contentType: "application/octet-stream",
  });
  return { kind: "blob", blob_id: r.blob_id };
}

/** Rows whose display value lies in [lo, hi] (legend histogram brush → selection mask). */
export function maskFromRange(values: ArrayLike<number>, mult: number, lo: number, hi: number): Uint8Array {
  const out = new Uint8Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const v = values[i] * mult;
    if (v >= lo && v <= hi) out[i] = 1;
  }
  return out;
}
