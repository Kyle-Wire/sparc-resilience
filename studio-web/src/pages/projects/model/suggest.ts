// Column suggestions (api.md §5.1 `columns/suggest`) → config draft. Empty mapping fields are
// pre-filled from the suggestion; fields the user already set are kept unless `overwrite`.
// The result names every path it filled so the form can badge them "suggested".
import type { ColumnSuggestion } from "../../../api/projects";
import { getList, getPath, getRecord, setPath } from "./dotted";

export type Raw = Record<string, unknown>;

/** Mapping fields of the data step: config path, suggestion key and confidence key. */
export const SUGGEST_FIELDS = [
  { path: "data.target", key: "target", conf: "target", label: "Temperature (target)" },
  { path: "data.id", key: "id", conf: "id", label: "Id" },
  { path: "data.x", key: "x", conf: "x", label: "X coordinate" },
  { path: "data.y", key: "y", conf: "y", label: "Y coordinate" },
  { path: "data.zone", key: "zone", conf: "zone", label: "Zone" },
  { path: "data.coord_unit", key: "coord_unit", conf: "coord_unit", label: "Coordinate unit" },
  { path: "data.crs", key: "crs_guess", conf: "crs", label: "CRS" },
] as const;

const empty = (v: unknown) => v === undefined || v === null || v === "";

export type SuggestResult = { raw: Raw; filled: string[] };

/** The suggested value for a mapping path (null when the server had none). */
export function suggestedValue(s: ColumnSuggestion | null | undefined, path: string): string | null {
  if (!s) return null;
  const f = SUGGEST_FIELDS.find((x) => x.path === path);
  if (!f) return null;
  const v = s[f.key];
  return typeof v === "string" && v !== "" ? v : null;
}

/** Confidence (0–1) of a suggestion for a mapping path, or for `roles.<role>` / `predictors`. */
export function suggestionConfidence(s: ColumnSuggestion | null | undefined, path: string): number | null {
  if (!s) return null;
  const f = SUGGEST_FIELDS.find((x) => x.path === path);
  const key = f ? f.conf : path.startsWith("physics.roles.") ? `roles.${path.slice("physics.roles.".length)}` : path;
  const c = s.confidence?.[key];
  return typeof c === "number" ? c : null;
}

/**
 * Pre-fill the mapping (target, id, x/y, zone, coord unit, CRS), the predictors and the
 * physics roles from a suggestion.
 */
export function applySuggestion(raw: Raw, s: ColumnSuggestion, opts: { overwrite?: boolean } = {}): SuggestResult {
  let next = raw;
  const filled: string[] = [];
  for (const f of SUGGEST_FIELDS) {
    const v = suggestedValue(s, f.path);
    if (v === null) continue;
    const cur = getPath(next, f.path);
    if (!opts.overwrite && !empty(cur)) continue;
    if (cur === v) continue;
    next = setPath(next, f.path, v);
    filled.push(f.path);
  }
  if (s.predictors?.length && (opts.overwrite || getList(next, "predictors").length === 0)) {
    next = setPath(next, "predictors", [...s.predictors]);
    filled.push("predictors");
  }
  const roles = getRecord<string>(next, "physics.roles");
  for (const [role, col] of Object.entries(s.roles ?? {})) {
    if (!col) continue;
    if (!opts.overwrite && !empty(roles[role])) continue;
    if (roles[role] === col) continue;
    next = setPath(next, ["physics", "roles", role], col);
    filled.push(`physics.roles.${role}`);
  }
  return { raw: next, filled };
}
