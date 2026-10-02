// Column lists the setup steps offer, from the header inspect of the data file.
import { isNumericDtype, type FileInspect } from "../../../api/projects";
import { getString } from "./dotted";

const MAPPING_PATHS = ["data.target", "data.x", "data.y", "data.id", "data.zone"];

/** Names of the mapping columns (target, x, y, id, zone) of a config (draft, else effective). */
export function mappingColumns(raw: unknown, effective: unknown): Set<string> {
  return new Set(MAPPING_PATHS.map((p) => getString(raw, p) ?? getString(effective, p)).filter((v): v is string => !!v));
}

/** Numeric columns of the data file minus the mapping columns: the predictor candidates. */
export function predictorCandidates(inspect: FileInspect | null | undefined, raw: unknown, effective: unknown): string[] {
  const used = mappingColumns(raw, effective);
  return (inspect?.columns ?? []).filter((col) => isNumericDtype(col.dtype) && !used.has(col.name)).map((col) => col.name);
}

/** Every column of the data file. */
export function allColumns(inspect: FileInspect | null | undefined): string[] {
  return (inspect?.columns ?? []).map((c) => c.name);
}
