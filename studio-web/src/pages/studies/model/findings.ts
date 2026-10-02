// Findings notebook model (SPEC §6.10): ordering by position, the PATCH updates a move sends,
// run filtering and reading a snapshot (the numbers that were on screen) back as a table.
import type { Finding } from "../../../api/types";
import { fillPath } from "../../../router";

/** Notebook order: position, then creation time (ties from older servers). */
export function ordered(findings: readonly Finding[]): Finding[] {
  return [...findings].sort((a, b) => a.position - b.position || a.created_utc.localeCompare(b.created_utc) || a.id.localeCompare(b.id));
}

/**
 * The `PATCH {position}` updates that move the visible finding at `from` to `to`.
 *
 * `all` is every finding of the notebook (so positions hidden by a run filter are kept);
 * `visible` the ids shown, in display order. The visible findings take the same set of
 * position slots they held, in their new order. When positions tie (or are missing), the
 * whole notebook is first renumbered 0…n−1 in its current order, so the move is always
 * expressible. Only findings whose position changes are returned.
 */
export function reorderPatches(all: readonly Finding[], visible: readonly string[], from: number, to: number): { id: string; position: number }[] {
  if (from === to || from < 0 || to < 0 || from >= visible.length || to >= visible.length) return [];
  const sorted = ordered(all);
  const positions = new Map(sorted.map((f) => [f.id, f.position]));
  const distinct = new Set(sorted.map((f) => f.position)).size === sorted.length && sorted.every((f) => Number.isFinite(f.position));
  const slot = new Map<string, number>(distinct ? positions : sorted.map((f, i) => [f.id, i]));
  const ids = visible.filter((id) => slot.has(id));
  if (ids.length !== visible.length) return [];
  const slots = ids.map((id) => slot.get(id)!).sort((a, b) => a - b);
  const next = [...ids];
  const [moved] = next.splice(from, 1);
  next.splice(to, 0, moved);
  next.forEach((id, i) => slot.set(id, slots[i]));
  const out: { id: string; position: number }[] = [];
  for (const f of sorted) {
    const p = slot.get(f.id)!;
    if (p !== positions.get(f.id)) out.push({ id: f.id, position: p });
  }
  return out;
}

/** Apply position patches locally (optimistic update), returning the new order. */
export function applyPositions(all: readonly Finding[], patches: readonly { id: string; position: number }[]): Finding[] {
  const m = new Map(patches.map((p) => [p.id, p.position]));
  return ordered(all.map((f) => (m.has(f.id) ? { ...f, position: m.get(f.id)! } : f)));
}

export type SnapshotTable = { columns: string[]; rows: (string | number | boolean | null)[][] };

function cell(v: unknown): string | number | boolean | null {
  if (v === null || v === undefined) return null;
  if (typeof v === "number" || typeof v === "string" || typeof v === "boolean") return v;
  return JSON.stringify(v);
}

/**
 * The snapshot as a table: a chart pin carries `table: {columns: [{label, unit}], rows}`; a
 * generic pin carries `table: {columns: [{key, label}], rows}` or plain numbers, shown as
 * key/value rows. Returns null when there is nothing tabular.
 */
export function snapshotTable(snapshot: Record<string, unknown> | null | undefined): SnapshotTable | null {
  if (!snapshot) return null;
  const t = snapshot.table as { columns?: unknown; rows?: unknown } | undefined;
  if (t && Array.isArray(t.columns) && Array.isArray(t.rows)) {
    const columns = (t.columns as unknown[]).map((c) => {
      if (c && typeof c === "object") {
        const o = c as { label?: unknown; key?: unknown; unit?: unknown };
        const label = String(o.label ?? o.key ?? "");
        return o.unit ? `${label} (${String(o.unit)})` : label;
      }
      return String(c);
    });
    const rows = (t.rows as unknown[]).filter(Array.isArray).map((r) => (r as unknown[]).map(cell));
    return { columns, rows };
  }
  const skip = new Set(["kind", "title", "caption", "units", "table"]);
  const rows = Object.entries(snapshot)
    .filter(([k, v]) => !skip.has(k) && v !== undefined)
    .map(([k, v]) => [k, cell(v)]);
  return rows.length ? { columns: ["Field", "Value"], rows } : null;
}

/** Runs referenced by findings, for the run filter. */
export function findingRuns(findings: readonly Finding[]): string[] {
  return [...new Set(findings.map((f) => f.run_id).filter((r): r is string => !!r))];
}

/**
 * Where "Open" goes: the stored `url_state` when it is a full in-app path (pathname + query, as
 * ChartFrame pins it); otherwise the finding's route (`view`, filled with its run and project)
 * plus `url_state` read as the query.
 */
export function findingHref(f: Pick<Finding, "url_state" | "view" | "run_id" | "project_id">): string {
  const u = (f.url_state ?? "").trim();
  // Only same-origin app paths: //host and /\host are protocol-relative URLs to another site.
  if (u.startsWith("/") && !u.startsWith("//") && !u.startsWith("/\\")) return u;
  const q = u.replace(/^\?/, "");
  let path: string;
  try {
    path = f.view.startsWith("/") ? fillPath(f.view, { rid: f.run_id, pid: f.project_id }) : f.run_id ? `/r/${encodeURIComponent(f.run_id)}` : `/p/${encodeURIComponent(f.project_id)}`;
  } catch {
    path = f.run_id ? `/r/${encodeURIComponent(f.run_id)}` : `/p/${encodeURIComponent(f.project_id)}`;
  }
  return q ? `${path}?${q}` : path;
}
