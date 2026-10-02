// Findings notebook endpoints (api.md §11; SPEC §6.10). A finding pins the numbers on screen
// (`snapshot`), the route and query to reopen (`view`, `url_state`), a markdown note and an
// optional chart SVG or map PNG. The Finding and FindingCreate types are shared with the
// foundation (ChartFrame's "Pin to Findings"), so they live in api/types.ts.
import { api, putRaw } from "./client";
import { useResource } from "./resource";
import type { Finding, FindingCreate } from "./types";

export type { Finding, FindingCreate } from "./types";

const enc = encodeURIComponent;

export type FindingPatch = { title?: string; note_md?: string; position?: number };

/** `GET /api/findings` (ordered by position), filtered by project and/or run. */
export function listFindings(q: { project?: string | null; run?: string | null } = {}, signal?: AbortSignal): Promise<Finding[]> {
  return api.get<Finding[]>("/api/findings", { project: q.project ?? undefined, run: q.run ?? undefined }, signal);
}

export function createFinding(body: FindingCreate): Promise<Finding> {
  return api.post<Finding>("/api/findings", body);
}

export function patchFinding(id: string, patch: FindingPatch): Promise<Finding> {
  return api.patch<Finding>(`/api/findings/${enc(id)}`, patch);
}

export function deleteFinding(id: string): Promise<{ ok: true }> {
  return api.del<{ ok: true }>(`/api/findings/${enc(id)}`);
}

/** Upload the finding's image (raw PNG or SVG, ≤ 10 MB). */
export function putFindingImage(id: string, image: Blob): Promise<{ image_url: string }> {
  const type = image.type === "image/svg+xml" ? "image/svg+xml" : "image/png";
  return putRaw<{ image_url: string }>(`/api/findings/${enc(id)}/image`, image, { contentType: type });
}

export function findingImageUrl(id: string): string {
  return `/api/findings/${enc(id)}/image`;
}

/**
 * Findings of a project (all runs; the notebook filters by run on the client so a reorder
 * always sees every position), or every finding of the workspace when `pid` is null.
 */
export function useFindings(pid: string | null) {
  const key = pid ? `project:${pid}:findings` : "findings:all";
  return useResource<Finding[]>(key, (s) => listFindings({ project: pid }, s), {
    tags: [pid ? `project:${pid}:findings` : "findings:all", "findings"],
    keepPrevious: true,
  });
}
