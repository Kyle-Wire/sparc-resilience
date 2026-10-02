// Wizard actions shared by the steps: the inline data check on the draft (S0, api.md §5.2),
// the column suggestion for a data file, and the preview grid/layers of a check.
import { errorMessage } from "../../../api/client";
import { checkData, fetchPreviewColumn, fetchPreviewGrid, suggestColumns, type ConfigRaw, type DataCheck, type FileInspect } from "../../../api/projects";
import { useResource } from "../../../api/resource";
import type { LayerGroup, LayerMeta } from "../../../api/types";
import { parseGridBin, type GridData } from "../../../map/grid";
import { toast } from "../../../stores/ui";
import { getList, getRecord, getString } from "../model/dotted";
import { applySuggestion } from "../model/suggest";
import { rawKey, useDrafts } from "./store";

const isRec = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);

/** `after` plus `null` for every key `before` holds that `after` dropped, at any depth; null when nothing was dropped. */
function withRemovals(before: Record<string, unknown>, after: Record<string, unknown>): Record<string, unknown> | null {
  let out: Record<string, unknown> | null = null;
  for (const k of Object.keys(before)) {
    const b = before[k];
    const a = after[k];
    if (b === undefined || b === null) continue;
    let v: unknown;
    if (a === undefined) v = null;
    else if (isRec(a) && isRec(b)) v = withRemovals(b, a) ?? undefined;
    if (v === undefined) continue;
    out ??= { ...after };
    out[k] = v;
  }
  return out;
}

/**
 * The `config_patch` of a data check: the draft, plus `null` for every key the draft removed
 * from the saved config, at any depth. The server applies the patch to the saved config as a
 * JSON Merge Patch (api.md §5.2: `null` deletes a key), so a removed lever, `qa.clip.<col>`,
 * `physics.roles.<role>` or cleared data key (zone, crs, id, subsample, coarse_m, cell_m…)
 * does not still apply from the saved copy, and the check describes the draft.
 */
export function checkPatch(saved: ConfigRaw, raw: ConfigRaw): ConfigRaw {
  return withRemovals(saved, raw) ?? raw;
}

/** Run S0 on the draft (unsaved form values are sent as `config_patch`). */
export async function runDataCheck(pid: string): Promise<DataCheck | null> {
  const st = useDrafts.getState();
  const d = st.drafts[pid];
  if (!d) return null;
  const key = rawKey(d.raw);
  st.setCheck(pid, { checking: true, checkError: null });
  try {
    const check = await checkData(pid, checkPatch(d.saved, d.raw));
    useDrafts.getState().setCheck(pid, { check, checkKey: key, checking: false, checkError: null });
    return check;
  } catch (e) {
    useDrafts.getState().setCheck(pid, { checking: false, checkError: e });
    return null;
  }
}

/**
 * Ask the server for a mapping of `path` and pre-fill the empty fields of the draft
 * (`overwrite` replaces fields already set). Returns the paths it filled.
 */
export async function suggestMapping(pid: string, path: string, opts: { overwrite?: boolean } = {}): Promise<string[]> {
  try {
    const s = await suggestColumns(pid, path);
    const st = useDrafts.getState();
    st.setSuggestion(pid, s);
    const d = st.drafts[pid];
    if (!d) return [];
    const r = applySuggestion(d.raw, s, opts);
    if (r.filled.length) st.replace(pid, r.raw, r.filled);
    return r.filled;
  } catch (e) {
    toast("error", "Could not suggest a column mapping", { body: errorMessage(e) });
    return [];
  }
}

/** The preview grid of a data check (immutable per token). */
export function usePreviewGrid(pid: string, token: string | null) {
  return useResource<GridData>(
    token ? `preview:${pid}:${token}:grid` : null,
    async (s) => {
      const g = await fetchPreviewGrid(pid, token!, s);
      return parseGridBin(g.meta, g.buffer, g.offsets);
    },
    { immutable: true },
  );
}

/** Loader of preview columns (Float32 in row order). */
export function previewLoader(pid: string, token: string): (meta: LayerMeta) => Promise<Float32Array> {
  const cache = new Map<string, Promise<Float32Array>>();
  return (meta) => {
    let p = cache.get(meta.key);
    if (!p) {
      p = fetchPreviewColumn(pid, token, meta.key);
      p.catch(() => cache.delete(meta.key));
      cache.set(meta.key, p);
    }
    return p;
  };
}

/**
 * Layer catalogue of a data check preview: the target first, then predictors, then the
 * `planner.layers` columns the server joins in (people*, lc_*; data columns of the same name
 * win), then any other preview column. Stats come from the header inspect when present
 * (percentiles are left null so the map derives its range from the values).
 */
export function previewGroups(check: DataCheck, raw: unknown, inspect: FileInspect | null | undefined): LayerGroup[] {
  const target = getString(raw, "data.target");
  const units = getString(raw, "data.target_units") ?? "degF";
  const predictors = getList<string>(raw, "predictors").map(String);
  const levers = getRecord<{ unit?: string }>(raw, "actionable");
  const cols = check.preview_columns ?? [...(target ? [target] : []), ...predictors];
  const byName = new Map((inspect?.columns ?? []).map((c) => [c.name, c]));
  const layersPath = getString(raw, "planner.layers");
  const isLayer = (c: string) => !!layersPath && /^(people|lc_)/.test(c) && !byName.has(c) && c !== target && !predictors.includes(c);
  const meta = (name: string): LayerMeta => {
    const ic = byName.get(name);
    const isTarget = name === target;
    if (isLayer(name)) {
      const people = name.startsWith("people");
      return {
        key: name,
        group: "layers",
        label: name,
        unit: people ? "people" : "share",
        scale: "seq",
        center: null,
        decimals: people ? 1 : 2,
        mult: 1,
        zero_blank: false,
        labels: null,
        desc: people ? "Residents per cell from planner.layers (HRSL), joined by id" : "Land-cover share of the cell from planner.layers (WorldCover), joined by id",
        sign_note: null,
        source: { file: layersPath ?? "", column: name },
        dtype: "float32",
        stats: { n: check.n_points, lo: null, hi: null, mean: null, p1: null, p2: null, p50: null, p98: null, p99: null },
      };
    }
    return {
      key: name,
      group: isTarget ? "target" : "predictors",
      label: isTarget ? `${name} (temperature)` : name,
      unit: isTarget ? units : String(levers[name]?.unit ?? ""),
      scale: "seq",
      center: null,
      decimals: 2,
      mult: 1,
      zero_blank: false,
      labels: null,
      desc: isTarget ? "Target as read by S0 (after QA)" : "Predictor as read by S0 (after QA and clipping)",
      sign_note: null,
      source: { file: getString(raw, "data.path") ?? "", column: name },
      dtype: "float32",
      stats: { n: check.n_points, lo: ic?.min ?? null, hi: ic?.max ?? null, mean: null, p1: null, p2: null, p50: null, p98: null, p99: null },
    };
  };
  const tgt = cols.filter((c) => c === target);
  const preds = cols.filter((c) => c !== target && predictors.includes(c));
  const rank = (c: string) => (c.startsWith("people") ? 0 : 1);
  const lay = cols.filter(isLayer).sort((a, b) => rank(a) - rank(b)); // residents first, then land cover
  const other = cols.filter((c) => c !== target && !predictors.includes(c) && !isLayer(c));
  const groups: LayerGroup[] = [];
  if (tgt.length) groups.push({ id: "target", label: "Temperature", layers: tgt.map(meta) });
  if (preds.length) groups.push({ id: "predictors", label: "Predictors", layers: preds.map(meta) });
  if (lay.length) groups.push({ id: "layers", label: "People & land cover", layers: lay.map(meta) });
  if (other.length) groups.push({ id: "other", label: "Other columns", layers: other.map(meta) });
  return groups;
}
