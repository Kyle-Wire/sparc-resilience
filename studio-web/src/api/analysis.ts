// Analysis tools and selections (api.md §6.3, §6.4): selection resolve, saved regions, mask
// blobs, region stats, breakdown, relationships hexbin and the correlogram. The run hub's
// Map tab and its analysis-tools drawer call these; every request carries the current
// per-run SelectionSpec (SPEC §6.5, §6.6).
import { api, putRaw } from "./client";
import { countBits, decodeBitset, packBits } from "./binary";
import { useResource } from "./resource";
import type { Bitset, Likely, SelectionSpec } from "./types";

const enc = encodeURIComponent;
const base = (rid: string) => `/api/runs/${enc(rid)}`;

// ---------------------------------------------------------------- selections (§6.3)

export type ResolvedSelectionReply = {
  n_cells: number;
  area_km2: number;
  people: number | null;
  medians: Record<string, number | null>;
  mask: Bitset;
  portable: boolean;
  warnings: string[];
};

/** `POST /api/runs/{rid}/selection/resolve`. */
export function resolveSelection(rid: string, selection: SelectionSpec, signal?: AbortSignal): Promise<ResolvedSelectionReply> {
  return api.post<ResolvedSelectionReply>(`${base(rid)}/selection/resolve`, { selection }, { signal });
}

/** Resolve a spec into a 0/1 row mask of length n (plus the server's summary). */
export async function resolveMask(rid: string, selection: SelectionSpec, n: number, signal?: AbortSignal): Promise<{ mask: Uint8Array; reply: ResolvedSelectionReply }> {
  const reply = await resolveSelection(rid, selection, signal);
  return { mask: decodeBitset(reply.mask, n), reply };
}

export type Region = { id: string; name: string; spec: SelectionSpec; n_cells: number; created_utc?: string; portable?: boolean };

export function useRegions(rid: string | null) {
  return useResource<Region[]>(rid ? `run:${rid}:regions` : null, (s) => api.get<Region[]>(`${base(rid!)}/regions`, undefined, s), {
    tags: rid ? [`run:${rid}:regions`] : [],
  });
}

/** `POST /api/runs/{rid}/regions` ("Save as region"). */
export function createRegion(rid: string, name: string, spec: SelectionSpec): Promise<Region> {
  return api.post<Region>(`${base(rid)}/regions`, { name, spec });
}

export function deleteRegion(rid: string, id: string): Promise<{ ok: true }> {
  return api.del<{ ok: true }>(`${base(rid)}/regions/${enc(id)}`);
}

/**
 * Upload a row mask as a selection blob (`PUT /blobs?kind=mask`, raw LSB-first bytes) and
 * return the portable-within-run `{kind: "blob"}` spec.
 */
export async function uploadMask(rid: string, mask: ArrayLike<number>): Promise<{ spec: SelectionSpec; n_cells: number }> {
  const r = await putRaw<{ blob_id: string; bytes: number }>(`${base(rid)}/blobs`, packBits(mask), {
    query: { kind: "mask" },
    contentType: "application/octet-stream",
  });
  return { spec: { kind: "blob", blob_id: r.blob_id }, n_cells: countBits(mask) };
}

// ---------------------------------------------------------------- tools (§6.4)

export type RegionStatsRequest = {
  selection: SelectionSpec;
  layers: string[];
  weights?: "people" | null;
  /** `res_…` or `configured:<slug>`. */
  scenarios?: string[];
};

export type LayerRegionStats = {
  mean: number | null;
  sd: number | null;
  p10: number | null;
  p50: number | null;
  p90: number | null;
  mean_outside: number | null;
};

export type RegionStats = {
  n_cells: number;
  area_km2: number;
  people: number | null;
  layers: Record<string, LayerRegionStats>;
  scenarios: Record<string, { inside: Likely; outside: Likely; has_folds: boolean }>;
};

/** `POST /api/runs/{rid}/stats/region`. */
export function regionStats(rid: string, body: RegionStatsRequest, signal?: AbortSignal): Promise<RegionStats> {
  return api.post<RegionStats>(`${base(rid)}/stats/region`, body, { signal });
}

export type BreakdownBy =
  | { kind: "zone" }
  | { kind: "quantile"; layer: string; q: number }
  | { kind: "hex"; size_m: 250 | 500 }
  | { kind: "category"; layer: string }
  | { kind: "fold" };

export type BreakdownRequest = { value: string; by: BreakdownBy; weights?: "people" | null; stat: "box" | "mean" };

export type Breakdown = {
  groups: { label: string; n: number; people: number | null; mean: number | null; q: [number, number, number, number, number] | null }[];
};

/** `POST /api/runs/{rid}/stats/breakdown`. */
export function breakdown(rid: string, body: BreakdownRequest, signal?: AbortSignal): Promise<Breakdown> {
  return api.post<Breakdown>(`${base(rid)}/stats/breakdown`, body, { signal });
}

export type HexbinRequest = { x: string; y: string; bins?: number; selection?: SelectionSpec };

export type Hexbin = {
  x_edges: number[];
  y_edges: number[];
  counts: number[][];
  sel_counts: number[][] | null;
  spearman: number | null;
  binned_mean: { x: number; y: number }[];
};

/** `POST /api/runs/{rid}/stats/hexbin`. */
export function hexbin(rid: string, body: HexbinRequest, signal?: AbortSignal): Promise<Hexbin> {
  return api.post<Hexbin>(`${base(rid)}/stats/hexbin`, body, { signal });
}

export type AcfRequest = { layer: string; max_lag_m?: number; n_perm?: number };

export type Acf = { lags_m: number[]; acf: number[]; band_mean: number[]; band_sd: number[] };

/** `POST /api/runs/{rid}/stats/acf` (influence.fft_acf with a permutation band). */
export function acf(rid: string, body: AcfRequest, signal?: AbortSignal): Promise<Acf> {
  return api.post<Acf>(`${base(rid)}/stats/acf`, body, { signal });
}

/** Stable cache key for a request body (same body → same cached result). */
export function bodyKey(body: unknown): string {
  return JSON.stringify(body, (_k, v: unknown) =>
    v && typeof v === "object" && !Array.isArray(v) ? Object.fromEntries(Object.entries(v as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b))) : v,
  );
}

/**
 * A tool result as a resource: one request per distinct body, refreshed when the run writes
 * outputs (tagged with the run's layers). `body === null` skips the request.
 */
export function useAnalysis<B, T>(rid: string, tool: string, body: B | null, call: (rid: string, body: B, signal: AbortSignal) => Promise<T>) {
  return useResource<T>(body === null ? null : `run:${rid}:analysis:${tool}:${bodyKey(body)}`, (s) => call(rid, body as B, s), {
    tags: [`run:${rid}:layers`, `run:${rid}:analysis`],
    keepPrevious: true,
  });
}
