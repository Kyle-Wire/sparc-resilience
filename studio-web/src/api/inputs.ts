// Project inputs (api.md §5.4, owner backend-projects): the tracked network jobs that fetch
// campaign forcing, people & land cover, open predictors and CMIP6 change factors, their
// chart-ready views, the ISD station picker and the host pre-check (`POST
// /api/system/netcheck`, foundation). Every `start*` returns `202 Job`.
import { api } from "./client";
import { useResource } from "./resource";
import type { Job } from "./types";

const enc = encodeURIComponent;
const base = (pid: string) => `/api/projects/${enc(pid)}`;

export type InputKind = "forcing" | "layers" | "features" | "cmip6" | "ghcn" | "stations";

/** Job kind of each input endpoint (api.md §8). */
export const INPUT_JOB_KIND: Record<InputKind, string> = {
  forcing: "input.forcing",
  layers: "input.layers",
  features: "input.features",
  cmip6: "input.cmip6",
  ghcn: "input.ghcn",
  stations: "input.stations",
};

export type WindSource = "auto" | "station" | "era5";

export type ForcingBody = {
  date: string;
  hours: [number, number];
  tz: string;
  lat?: number;
  lon?: number;
  station?: string;
  wind_source: WindSource;
  link?: boolean;
};

export type LayersBody = { link?: boolean };

export type FeaturesTarget = "new_project" | "this_project";

export type FeaturesBody = { months?: string[]; max_cloud?: number; s2_tiles?: string[]; target: FeaturesTarget };

export type Cmip6Body = {
  lat?: number;
  lon?: number;
  experiments?: string[];
  periods?: Record<string, [number, number]>;
  baseline?: [number, number];
  months?: number[];
  variable?: "tasmax" | "tas";
  models?: string[];
  workers?: number;
  link?: boolean;
};

export type GhcnBody = { station?: string };

export function startForcing(pid: string, body: ForcingBody): Promise<Job> {
  return api.post<Job>(`${base(pid)}/inputs/forcing`, body);
}

export function startLayers(pid: string, body: LayersBody = {}): Promise<Job> {
  return api.post<Job>(`${base(pid)}/inputs/layers`, body);
}

export function startFeatures(pid: string, body: FeaturesBody): Promise<Job> {
  return api.post<Job>(`${base(pid)}/inputs/features`, body);
}

export function startCmip6(pid: string, body: Cmip6Body): Promise<Job> {
  return api.post<Job>(`${base(pid)}/inputs/cmip6`, body);
}

export function startGhcn(pid: string, body: GhcnBody = {}): Promise<Job> {
  return api.post<Job>(`${base(pid)}/inputs/ghcn`, body);
}

/** Download and cache the NOAA ISD station index (≈3 MB) as a tracked job. */
export function startStations(pid: string): Promise<Job> {
  return api.post<Job>(`${base(pid)}/inputs/stations`, {});
}

// ---------------------------------------------------------------- summaries and views

export type InputsSummary = {
  forcing: { path: string; date: string; physics: Record<string, unknown>; checks: string[]; linked: boolean } | null;
  climate: { path: string; n_models: number; experiments: string[]; periods: string[]; linked: boolean } | null;
  layers: { path: string; n: number; people_total: number; linked: boolean } | null;
  features: { path: string; agreement: Record<string, unknown>[]; linked: boolean } | null;
  ghcn: { station: string; years: [number, number] } | null;
};

export type ForcingView = {
  era5: Record<string, unknown>;
  station: Record<string, unknown> | null;
  compare: { name: string; era5: number | null; station: number | null }[];
  checks: string[];
};

export type ClimateView = {
  rows: { model: string; experiment: string; period: string; delta_K: number | null }[];
  summary: { experiment: string; period: string; median: number; p10: number; p90: number }[];
};

export type LayersView = {
  totals: Record<string, number | null>;
  columns: ({ name: string; [k: string]: unknown } | string)[];
};

/** One 2-D histogram of a city layer against its open counterpart (additive server field). */
export type ScatterBins = { role?: string; city_column?: string; open_column?: string; x_edges: number[]; y_edges: number[]; counts: number[][] };

export type FeaturesView = { agreement: Record<string, unknown>[]; scatter_bins: ScatterBins[] | ScatterBins | null };

export type InputViewKind = "forcing" | "climate" | "layers" | "features";
export type InputViews = { forcing: ForcingView; climate: ClimateView; layers: LayersView; features: FeaturesView };

export function getInputs(pid: string, signal?: AbortSignal): Promise<InputsSummary> {
  return api.get<InputsSummary>(`${base(pid)}/inputs`, undefined, signal);
}

export function getInputView<K extends InputViewKind>(pid: string, kind: K, signal?: AbortSignal): Promise<InputViews[K]> {
  return api.get<InputViews[K]>(`${base(pid)}/inputs/${enc(kind)}/view`, undefined, signal);
}

export function useInputs(pid: string | null) {
  return useResource<InputsSummary>(pid ? `project:${pid}:inputs` : null, (s) => getInputs(pid!, s), { tags: pid ? [`project:${pid}`] : [] });
}

export function useInputView<K extends InputViewKind>(pid: string | null, kind: K, enabled = true) {
  return useResource<InputViews[K]>(pid && enabled ? `project:${pid}:inputs:${kind}:view` : null, (s) => getInputView(pid!, kind, s), {
    tags: pid ? [`project:${pid}`] : [],
  });
}

// ---------------------------------------------------------------- stations and hosts

export type Station = { usaf_wban: string; name: string; lat: number; lon: number; dist_km: number; begin: string; end: string };

/**
 * Nearest ISD stations from the cached index. When `isd-history.csv` is not cached the
 * server answers `404 not_found` with a `fetch_input` action (it never downloads inline).
 */
export function getStations(pid: string, lat: number, lon: number, limit = 10, signal?: AbortSignal): Promise<Station[]> {
  return api.get<Station[]>(`${base(pid)}/forcing/stations`, { lat, lon, limit }, signal);
}

export type NetcheckRow = { host: string; ok: boolean; ms: number | null; error: string | null };

/** Host pre-check (cached for 10 min per host by the server). */
export function netcheck(hosts?: string[]): Promise<{ results: NetcheckRow[] }> {
  return api.post<{ results: NetcheckRow[] }>("/api/system/netcheck", hosts && hosts.length ? { hosts } : {});
}
