// Multiverse launch model: the built-in variants of sparc.core.multiverse (VARIANTS / LABELS)
// and the custom-variant editor, whose drafts become `custom_variants` =
// {name: {"dotted.key": value}} (api.md §8, passed to core as `extra_variants`).

export const BASELINE_VARIANT = "baseline";

/** sparc.core.multiverse.VARIANTS, in core order, with core's labels and what each changes. */
export const BUILTIN_VARIANTS: readonly { name: string; label: string; changes: string }[] = [
  { name: "baseline", label: "baseline", changes: "the run's own configuration (always included: the reference)" },
  { name: "blocks_1km", label: "1 km CV blocks", changes: "cv.block_m = 1000" },
  { name: "blocks_3km", label: "3 km CV blocks", changes: "cv.block_m = 3000" },
  { name: "generic_forcing", label: "generic forcing (no campaign day)", changes: "physics forcing = 800 W/m² sun, −100 W/m² longwave, no wind" },
  { name: "no_mediator", label: "no NDVI mediator", changes: "mediators = {}" },
  { name: "no_physics", label: "no physics model", changes: "models.physics = false" },
  { name: "spatial_plus", label: "Spatial+ (MGWR)", changes: "models.spatial_plus = [mgwr]" },
  { name: "focal_scale_1", label: "one neighbourhood scale", changes: "influence.scales = [1.0]" },
  { name: "lag_1km", label: "1 km correlogram horizon", changes: "influence.max_lag_m = 1000" },
  { name: "no_gwrf", label: "no GW random forest", changes: "models.gwrf = false" },
];

export const BUILTIN_NAMES: readonly string[] = BUILTIN_VARIANTS.map((v) => v.name);

export type OverrideRow = { key: string; value: string };
export type CustomVariantDraft = { name: string; rows: OverrideRow[] };

const NAME_RE = /^[a-z][a-z0-9_]*$/;
/** A dotted config path, as the server validates it: identifier segments joined by dots. */
const KEY_RE = /^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$/;
/** Segments that would reach an object's prototype instead of a config key. */
const UNSAFE_SEGMENTS = new Set(["__proto__", "prototype", "constructor"]);

/**
 * An override value as typed: JSON when it parses (numbers, true/false, null, lists, objects,
 * quoted strings), otherwise the text itself as a string. U+2212 is read as a minus.
 */
export function parseOverrideValue(text: string): unknown {
  const t = text.trim();
  if (t === "") return "";
  const j = t.replace(/^−/, "-");
  try {
    return JSON.parse(j);
  } catch {
    if (/^(True|False)$/.test(t)) return t === "True";
    if (/^(None|~)$/.test(t)) return null;
    return t;
  }
}

/** Text shown in the editor for a value (the inverse of parseOverrideValue for JSON values). */
export function formatOverrideValue(v: unknown): string {
  return typeof v === "string" ? v : JSON.stringify(v);
}

/**
 * The `custom_variants` JSON of the drafts, plus every problem that blocks the launch: names
 * must be lower-case slugs, unique and not built-in (core refuses a clash); keys must be dotted
 * config paths, unique within a variant; a variant needs at least one override. Rows with an
 * empty key and value are ignored.
 */
export function customVariantsJson(drafts: readonly CustomVariantDraft[]): { value: Record<string, Record<string, unknown>>; errors: string[] } {
  const value: Record<string, Record<string, unknown>> = {};
  const errors: string[] = [];
  const seen = new Set<string>();
  drafts.forEach((d, i) => {
    const name = d.name.trim();
    const where = name ? `Custom variant “${name}”` : `Custom variant ${i + 1}`;
    if (!name) errors.push(`${where} needs a name.`);
    else if (!NAME_RE.test(name)) errors.push(`${where}: use lower-case letters, digits and underscores, starting with a letter.`);
    else if (BUILTIN_NAMES.includes(name)) errors.push(`${where} reuses a built-in variant name.`);
    else if (seen.has(name)) errors.push(`${where} is defined twice.`);
    // A Map keeps any key as data (an object literal would treat "__proto__" specially).
    const changes = new Map<string, unknown>();
    for (const r of d.rows) {
      const key = r.key.trim();
      if (!key && !r.value.trim()) continue;
      if (!KEY_RE.test(key) || key.split(".").some((s) => UNSAFE_SEGMENTS.has(s))) {
        errors.push(`${where}: “${key || "(empty)"}” is not a dotted config key such as cv.block_m.`);
        continue;
      }
      if (changes.has(key)) errors.push(`${where} sets ${key} twice.`);
      changes.set(key, parseOverrideValue(r.value));
    }
    if (!changes.size) errors.push(`${where} changes nothing: add a dotted key and a value.`);
    if (name && NAME_RE.test(name) && !BUILTIN_NAMES.includes(name) && !seen.has(name)) {
      seen.add(name);
      if (changes.size) value[name] = Object.fromEntries(changes);
    }
  });
  return { value, errors };
}

/** Drafts back from a `custom_variants` object (re-launching a study with its params). */
export function draftsFromJson(v: Record<string, Record<string, unknown>> | null | undefined): CustomVariantDraft[] {
  return Object.entries(v ?? {}).map(([name, changes]) => ({
    name,
    rows: Object.entries(changes ?? {}).map(([key, val]) => ({ key, value: formatOverrideValue(val) })),
  }));
}
