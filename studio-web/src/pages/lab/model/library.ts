// Library view models (SPEC §7.4): the lineage tree (revisions under their parents), ladder
// groups (`ladder:<id>` tags) with the dose that varies along each ladder, and the field-kit
// downloads (CSV and GeoJSON rendered in the browser from the field-kit JSON, SPEC §7.10).
import type { FieldKit, ScenarioDoc, ScenarioSummary } from "../../../api/lab";
import { toCsv, type CsvCell } from "../../../components/ui/csv";

export type LineageNode = { s: ScenarioSummary; depth: number; children: number };

/**
 * Depth-first lineage order: roots (no parent, or a parent not in the list) sorted by most
 * recent update, each followed by its descendants (oldest revision first).
 */
export function lineage(list: ScenarioSummary[]): LineageNode[] {
  const byId = new Map(list.map((s) => [s.id, s]));
  const kids = new Map<string, ScenarioSummary[]>();
  const roots: ScenarioSummary[] = [];
  for (const s of list) {
    if (s.parent_id && byId.has(s.parent_id) && s.parent_id !== s.id) {
      const arr = kids.get(s.parent_id) ?? [];
      arr.push(s);
      kids.set(s.parent_id, arr);
    } else roots.push(s);
  }
  roots.sort((a, b) => b.updated_utc.localeCompare(a.updated_utc));
  const out: LineageNode[] = [];
  const seen = new Set<string>();
  const walk = (s: ScenarioSummary, depth: number) => {
    if (seen.has(s.id)) return; // defensive against cycles
    seen.add(s.id);
    const ch = (kids.get(s.id) ?? []).sort((a, b) => a.revision - b.revision || a.created_utc.localeCompare(b.created_utc));
    out.push({ s, depth, children: ch.length });
    for (const c of ch) walk(c, depth + 1);
  };
  for (const r of roots) walk(r, 0);
  for (const s of list) if (!seen.has(s.id)) walk(s, 0);
  return out;
}

export const LADDER_TAG = /^ladder:(.+)$/;

/** Scenarios grouped by their `ladder:<id>` tag (only ladders with at least two rungs). */
export function ladderGroups(list: ScenarioSummary[]): { id: string; members: ScenarioSummary[] }[] {
  const groups = new Map<string, ScenarioSummary[]>();
  for (const s of list)
    for (const t of s.tags) {
      const m = LADDER_TAG.exec(t);
      if (!m) continue;
      const arr = groups.get(m[1]) ?? [];
      arr.push(s);
      groups.set(m[1], arr);
    }
  return [...groups.entries()].filter(([, m]) => m.length >= 2).map(([id, members]) => ({ id, members }));
}

/**
 * The dose of each ladder rung: the amount (or percentile) of the one edit whose value
 * differs between the rungs' docs. Returns null when no single edit varies.
 */
export function ladderDoses(docs: ScenarioDoc[]): { editIndex: number; lever: string; doses: number[] } | null {
  if (docs.length < 2) return null;
  const nEdits = Math.min(...docs.map((d) => d.edits.length));
  for (let i = 0; i < nEdits; i++) {
    const vals = docs.map((d) => d.edits[i].amount ?? d.edits[i].percentile ?? NaN);
    if (vals.some((v) => !Number.isFinite(v))) continue;
    if (new Set(vals).size > 1) return { editIndex: i, lever: docs[0].edits[i].lever, doses: vals };
  }
  return null;
}

/** Parse "5, 10, 20 30" into numbers (invalid parts are reported). */
export function parseNumberList(text: string): { values: number[]; bad: string[] } {
  const parts = text.split(/[\s,;]+/).filter(Boolean);
  const values: number[] = [];
  const bad: string[] = [];
  for (const p of parts) {
    const v = Number(p.replace("−", "-"));
    if (Number.isFinite(v)) values.push(v);
    else bad.push(p);
  }
  return { values, bad };
}

// ---------------------------------------------------------------- sites (the "Around sites" template)

const LON_COL = /^(lon|lng|long|longitude|x)$/i;
const LAT_COL = /^(lat|latitude|y)$/i;

/**
 * Points as `[lon, lat]` pairs from typed text ("lon, lat" per line) or an uploaded CSV whose
 * header names the columns (lon/lng/longitude/x and lat/latitude/y, in any position). Values
 * outside ±180 / ±90 and lines that are not two numbers are reported, not guessed.
 */
export function parseSites(text: string): { sites: [number, number][]; bad: string[] } {
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const sites: [number, number][] = [];
  const bad: string[] = [];
  if (!lines.length) return { sites, bad };
  // Delimited lines keep their empty cells (columns must not shift); bare "lon lat" splits on spaces.
  const split = (l: string) => (/[,;\t]/.test(l) ? l.split(/[,;\t]/) : l.split(/\s+/)).map((c) => c.trim().replace(/^"|"$/g, ""));
  const head = split(lines[0]);
  let ix = 0;
  let iy = 1;
  let start = 0;
  const hx = head.findIndex((c) => LON_COL.test(c));
  const hy = head.findIndex((c) => LAT_COL.test(c));
  if (hx >= 0 && hy >= 0) {
    ix = hx;
    iy = hy;
    start = 1;
  }
  for (const l of lines.slice(start)) {
    const cells = split(l);
    const num = (c: string | undefined) => (c ? Number(c.replace("−", "-")) : NaN); // an empty cell is missing, not 0
    const lon = num(cells[ix]);
    const lat = num(cells[iy]);
    if (start === 0 && cells.length !== 2) bad.push(l);
    else if (!Number.isFinite(lon) || !Number.isFinite(lat) || Math.abs(lon) > 180 || Math.abs(lat) > 90) bad.push(l);
    else sites.push([lon, lat]);
  }
  return { sites, bad };
}

// ---------------------------------------------------------------- field kit downloads

export function fieldKitCsv(kit: FieldKit, which: "cells" | "sites" | "pairs"): string {
  if (which === "cells") {
    const cols = ["rank", "id", "lon", "lat", "zone", "dose", "planned_benefit", "closed_loop_delta", "people", "plantable_pp"] as const;
    return toCsv([...cols], kit.cells.map((c) => cols.map((k) => c[k] as CsvCell)));
  }
  if (which === "sites") {
    const cols = ["id", "lon", "lat", "role", "canopy", "impervious", "effect_sd"] as const;
    return toCsv([...cols], kit.sites.map((c) => cols.map((k) => c[k] as CsvCell)));
  }
  const cols = ["treated_id", "control_id", "treated_lon", "treated_lat", "control_lon", "control_lat", "covariate_distance"] as const;
  return toCsv([...cols], kit.pairs.map((c) => cols.map((k) => c[k] as CsvCell)));
}

type Feature = { type: "Feature"; geometry: { type: "Point" | "LineString"; coordinates: number[] | number[][] }; properties: Record<string, unknown> };

const finite = (...v: (number | null | undefined)[]) => v.every((x) => typeof x === "number" && Number.isFinite(x));

/**
 * GeoJSON (EPSG:4326) of the field kit: treated cells and logger sites as points, before/after
 * pairs as lines. Rows without lon/lat (runs without a CRS) are left out and counted.
 */
export function fieldKitGeoJson(kit: FieldKit): { geojson: string; skipped: number } {
  const features: Feature[] = [];
  let skipped = 0;
  for (const c of kit.cells) {
    if (!finite(c.lon, c.lat)) {
      skipped++;
      continue;
    }
    features.push({ type: "Feature", geometry: { type: "Point", coordinates: [c.lon as number, c.lat as number] }, properties: { layer: "cell", ...c } });
  }
  for (const s of kit.sites) {
    if (!finite(s.lon, s.lat)) {
      skipped++;
      continue;
    }
    features.push({ type: "Feature", geometry: { type: "Point", coordinates: [s.lon as number, s.lat as number] }, properties: { layer: "logger_site", ...s } });
  }
  for (const p of kit.pairs) {
    if (!finite(p.treated_lon, p.treated_lat, p.control_lon, p.control_lat)) {
      skipped++;
      continue;
    }
    features.push({
      type: "Feature",
      geometry: { type: "LineString", coordinates: [[p.treated_lon as number, p.treated_lat as number], [p.control_lon as number, p.control_lat as number]] },
      properties: { layer: "before_after_pair", ...p },
    });
  }
  return { geojson: JSON.stringify({ type: "FeatureCollection", features }), skipped };
}
