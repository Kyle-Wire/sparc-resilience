// Dotted-path access to a config object ("actionable.Pct_Canopy.doses.2"), immutable updates
// and section diffs. Issue paths (api.md §1 `Issue.path`) use the same notation; a segment
// that is all digits indexes an array. Pass an array path when a key itself contains a dot.

export type Path = string | readonly (string | number)[];

export function splitPath(path: Path): (string | number)[] {
  if (typeof path !== "string") return [...path];
  if (path === "") return [];
  return path.split(".").map((s) => (/^\d+$/.test(s) ? Number(s) : s));
}

export function joinPath(parts: readonly (string | number)[]): string {
  return parts.map(String).join(".");
}

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);

/** Value at a path, or undefined when any step is missing. */
export function getPath(obj: unknown, path: Path): unknown {
  let cur: unknown = obj;
  for (const seg of splitPath(path)) {
    if (cur === null || cur === undefined) return undefined;
    if (Array.isArray(cur)) cur = typeof seg === "number" ? cur[seg] : undefined;
    else if (isObj(cur)) cur = cur[String(seg)];
    else return undefined;
  }
  return cur;
}

/**
 * A copy of `obj` with `value` at `path` (structural sharing elsewhere). Missing containers
 * are created (an array when the next segment is an index). `undefined` deletes the key (or
 * the array element).
 */
export function setPath<T>(obj: T, path: Path, value: unknown): T {
  const parts = splitPath(path);
  if (!parts.length) return value as T;
  const rec = (cur: unknown, i: number): unknown => {
    const seg = parts[i];
    const last = i === parts.length - 1;
    if (Array.isArray(cur) && typeof seg === "number") {
      const out = [...cur];
      if (last) {
        if (value === undefined) out.splice(seg, 1);
        else out[seg] = value;
      } else out[seg] = rec(cur[seg], i + 1);
      return out;
    }
    const base: Record<string, unknown> = isObj(cur) ? { ...cur } : {};
    const key = String(seg);
    if (last) {
      if (value === undefined) delete base[key];
      else base[key] = value;
    } else {
      const next = base[key];
      const child = next !== undefined && next !== null ? next : typeof parts[i + 1] === "number" ? [] : {};
      base[key] = rec(child, i + 1);
    }
    return base;
  };
  return rec(obj, 0) as T;
}

/** Structural equality for JSON-like values (key order ignored). */
export function jsonEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b || a === null || b === null) return false;
  if (Array.isArray(a)) return Array.isArray(b) && a.length === b.length && a.every((v, i) => jsonEqual(v, b[i]));
  if (isObj(a) && isObj(b)) {
    const ka = Object.keys(a).filter((k) => a[k] !== undefined);
    const kb = Object.keys(b).filter((k) => b[k] !== undefined);
    return ka.length === kb.length && ka.every((k) => jsonEqual(a[k], b[k]));
  }
  return false;
}

/** Top-level keys whose value differs between two configs, in first-seen order. */
export function changedSections(saved: Record<string, unknown>, draft: Record<string, unknown>): string[] {
  const keys = [...new Set([...Object.keys(saved), ...Object.keys(draft)])];
  return keys.filter((k) => !jsonEqual(saved[k], draft[k]));
}

/** Deep copy of a JSON-like value. */
export function cloneJson<T>(v: T): T {
  return v === undefined ? v : (JSON.parse(JSON.stringify(v)) as T);
}

/** Array value at a path (non-arrays read as empty). */
export function getList<T = unknown>(obj: unknown, path: Path): T[] {
  const v = getPath(obj, path);
  return Array.isArray(v) ? (v as T[]) : [];
}

/** Object value at a path (non-objects read as empty). */
export function getRecord<T = unknown>(obj: unknown, path: Path): Record<string, T> {
  const v = getPath(obj, path);
  return isObj(v) ? (v as Record<string, T>) : {};
}

/** String value at a path (null when unset or not a string). */
export function getString(obj: unknown, path: Path): string | null {
  const v = getPath(obj, path);
  return typeof v === "string" ? v : null;
}

/** Finite number at a path (null otherwise). */
export function getNumber(obj: unknown, path: Path): number | null {
  const v = getPath(obj, path);
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}
