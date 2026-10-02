// Per-run selection (SPEC §3.3, §6.5, §6.6). The SelectionSpec lives in the URL (`sel`):
// a saved region is `sel=rg_…`, anything else is an inline URL-safe base64 JSON spec
// prefixed with "s." (`sel=s.eyJraW5kIjoiem9uZXMi…`). The store caches what was resolved for
// that spec (mask, cell count, label) so views can tint the map without a round trip.
// Views opt in by calling useRunSelection(rid); other views ignore it.
import { useCallback, useMemo } from "react";
import { create } from "zustand";
import type { SelectionSpec } from "../api/types";
import { base64url, setQuery, useUrlState, type Codec } from "../router";

export type ResolvedSelection = {
  /** The encoded `sel` value this entry belongs to. */
  key: string;
  /** 0/1 per run row, when known (brush results or a /selection/resolve reply). */
  mask: Uint8Array | null;
  n_cells: number | null;
  label: string | null;
  source: string | null;
};

type SelectionState = {
  byRun: Record<string, ResolvedSelection>;
  put: (rid: string, entry: ResolvedSelection) => void;
  drop: (rid: string) => void;
};

export const useSelectionStore = create<SelectionState>((set, get) => ({
  byRun: {},
  put: (rid, entry) => set({ byRun: { ...get().byRun, [rid]: entry } }),
  drop: (rid) => {
    const byRun = { ...get().byRun };
    delete byRun[rid];
    set({ byRun });
  },
}));

const REGION_RE = /^rg_[A-Za-z0-9_-]+$/;

/** Encode a spec for the `sel` query parameter. */
export function encodeSelection(spec: SelectionSpec | null): string | null {
  if (!spec) return null;
  if ("kind" in spec && spec.kind === "all") return null;
  if ("kind" in spec && spec.kind === "region" && REGION_RE.test(spec.id)) return spec.id;
  return "s." + base64url.encode(JSON.stringify(spec));
}

/** Decode the `sel` query parameter; invalid values read as no selection. */
export function decodeSelection(raw: string | null): SelectionSpec | null {
  if (!raw) return null;
  if (REGION_RE.test(raw)) return { kind: "region", id: raw };
  if (!raw.startsWith("s.")) return null;
  try {
    const v = JSON.parse(base64url.decode(raw.slice(2))) as unknown;
    return isSelectionSpec(v) ? v : null;
  } catch {
    return null;
  }
}

const KINDS = new Set(["all", "zones", "polygon", "circle", "rect", "cells", "blob", "hex", "filter", "top", "buffer", "region"]);

/** Structural check of a SelectionSpec tree (kinds and combinators, not field values). */
export function isSelectionSpec(v: unknown): v is SelectionSpec {
  if (!v || typeof v !== "object") return false;
  const o = v as Record<string, unknown>;
  if (typeof o.kind === "string") {
    if (!KINDS.has(o.kind)) return false;
    if (o.kind === "top" && o.within !== undefined && o.within !== null) return isSelectionSpec(o.within);
    if (o.kind === "buffer") return isSelectionSpec(o.of);
    return true;
  }
  if (o.op === "not") return isSelectionSpec(o.arg);
  if (o.op === "and" || o.op === "or" || o.op === "minus") return Array.isArray(o.args) && o.args.length > 0 && o.args.every(isSelectionSpec);
  return false;
}

export const selectionCodec: Codec<SelectionSpec | null> = { parse: decodeSelection, format: encodeSelection };

export type RunSelection = {
  spec: SelectionSpec | null;
  /** The encoded URL value (stable identity for caching). */
  key: string | null;
  mask: Uint8Array | null;
  nCells: number | null;
  label: string | null;
  /** Replace the selection (writes the URL; `resolved` caches a mask/count for it). */
  set: (spec: SelectionSpec | null, resolved?: { mask?: Uint8Array | null; n_cells?: number | null; label?: string | null; source?: string | null }) => void;
  /** Attach a resolved mask/count to the current selection. */
  resolve: (r: { mask?: Uint8Array | null; n_cells?: number | null; label?: string | null }) => void;
  clear: () => void;
};

/** The current run's selection (URL-backed). */
export function useRunSelection(rid: string): RunSelection {
  const [spec] = useUrlState("sel", selectionCodec);
  const key = useMemo(() => encodeSelection(spec), [spec]);
  const entry = useSelectionStore((s) => s.byRun[rid]);
  const valid = entry && key !== null && entry.key === key ? entry : null;

  const set = useCallback<RunSelection["set"]>(
    (next, resolved) => {
      const k = encodeSelection(next);
      if (k === null) useSelectionStore.getState().drop(rid);
      else useSelectionStore.getState().put(rid, { key: k, mask: resolved?.mask ?? null, n_cells: resolved?.n_cells ?? null, label: resolved?.label ?? null, source: resolved?.source ?? null });
      setQuery({ sel: k }, { push: true });
    },
    [rid],
  );
  const resolve = useCallback<RunSelection["resolve"]>(
    (r) => {
      if (key === null) return;
      const cur = useSelectionStore.getState().byRun[rid];
      const base = cur && cur.key === key ? cur : { key, mask: null, n_cells: null, label: null, source: null };
      useSelectionStore.getState().put(rid, { ...base, mask: r.mask ?? base.mask, n_cells: r.n_cells ?? base.n_cells, label: r.label ?? base.label });
    },
    [rid, key],
  );
  const clear = useCallback(() => {
    useSelectionStore.getState().drop(rid);
    setQuery({ sel: null }, { push: true });
  }, [rid]);

  return { spec, key, mask: valid?.mask ?? null, nCells: valid?.n_cells ?? null, label: valid?.label ?? null, set, resolve, clear };
}
