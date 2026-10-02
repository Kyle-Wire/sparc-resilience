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

/** core DEFAULTS of the `data` keys whose default is not null (sparc.core.config.DEFAULTS). */
const DATA_DEFAULTS: Record<string, unknown> = { x: "x", y: "y", coord_unit: "m", target_units: "degF", background: "median", join: [] };

/**
 * The `config_patch` of a data check: the draft, plus every `data.*` key the draft removed
 * set back to its DEFAULTS value. The server deep-merges the patch onto the saved config,
 * so a key cleared in the form (zone, crs, id, subsample, coarse_m, cell_m…) would
 * otherwise still apply from the saved copy and the check would not describe the draft.
 */
export function checkPatch(saved: ConfigRaw, raw: ConfigRaw): ConfigRaw {
  const before = getRecord<unknown>(saved, "data");
  const after = getRecord<unknown>(raw, "data");
  const cleared = Object.keys(before).filter((k) => before[k] !== undefined && before[k] !== null && after[k] === undefined);
  if (!cleared.length) return raw;
  const data: Record<string, unknown> = { ...after };
  for (const k of cleared) data[k] = k in DATA_DEFAULTS ? DATA_DEFAULTS[k] : null;
  return { ...raw, data };
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
 * Layer catalogue of a data check preview: the target first, then predictors, then any other
 * preview column. Stats come from the header inspect when present (percentiles are left
 * null so the map derives its range from the values).
 */
export function previewGroups(check: DataCheck, raw: unknown, inspect: FileInspect | null | undefined): LayerGroup[] {
  const target = getString(raw, "data.target");
  const units = getString(raw, "data.target_units") ?? "degF";
  const predictors = getList<string>(raw, "predictors").map(String);
  const levers = getRecord<{ unit?: string }>(raw, "actionable");
  const cols = check.preview_columns ?? [...(target ? [target] : []), ...predictors];
  const byName = new Map((inspect?.columns ?? []).map((c) => [c.name, c]));
  const meta = (name: string): LayerMeta => {
    const ic = byName.get(name);
    const isTarget = name === target;
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
  const other = cols.filter((c) => c !== target && !predictors.includes(c));
  const groups: LayerGroup[] = [];
  if (tgt.length) groups.push({ id: "target", label: "Temperature", layers: tgt.map(meta) });
  if (preds.length) groups.push({ id: "predictors", label: "Predictors", layers: preds.map(meta) });
  if (other.length) groups.push({ id: "other", label: "Other columns", layers: other.map(meta) });
  return groups;
}
