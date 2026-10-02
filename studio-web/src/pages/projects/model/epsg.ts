// EPSG search for the data step's CRS field (SPEC §9.3 "EPSG search box"). Offline: a
// curated list of the systems city data usually comes in (WGS 84, Web Mercator, every UTM
// zone on WGS 84 / NAD83 / ETRS89, common US State Plane feet systems and national grids).
// Any "EPSG:<code>" typed by hand is accepted too; the server validates it.

export type EpsgEntry = { code: string; name: string; unit: "degree" | "metre" | "US survey foot" };

function utm(prefix: number, datum: string, zones: [number, number], hemi: "N" | "S"): EpsgEntry[] {
  const out: EpsgEntry[] = [];
  for (let z = zones[0]; z <= zones[1]; z++) out.push({ code: `EPSG:${prefix + z}`, name: `${datum} / UTM zone ${z}${hemi}`, unit: "metre" });
  return out;
}

export const EPSG_LIST: readonly EpsgEntry[] = [
  { code: "EPSG:4326", name: "WGS 84 (longitude/latitude)", unit: "degree" },
  { code: "EPSG:3857", name: "WGS 84 / Pseudo-Mercator (web maps)", unit: "metre" },
  { code: "EPSG:3438", name: "NAD83(NSRS2007) / Rhode Island (ftUS)", unit: "US survey foot" },
  { code: "EPSG:2249", name: "NAD83 / Massachusetts Mainland (ftUS)", unit: "US survey foot" },
  { code: "EPSG:2263", name: "NAD83 / New York Long Island (ftUS)", unit: "US survey foot" },
  { code: "EPSG:2272", name: "NAD83 / Pennsylvania South (ftUS)", unit: "US survey foot" },
  { code: "EPSG:2248", name: "NAD83 / Maryland (ftUS)", unit: "US survey foot" },
  { code: "EPSG:2283", name: "NAD83 / Virginia North (ftUS)", unit: "US survey foot" },
  { code: "EPSG:2236", name: "NAD83 / Florida East (ftUS)", unit: "US survey foot" },
  { code: "EPSG:2240", name: "NAD83 / Georgia West (ftUS)", unit: "US survey foot" },
  { code: "EPSG:3435", name: "NAD83 / Illinois East (ftUS)", unit: "US survey foot" },
  { code: "EPSG:2276", name: "NAD83 / Texas North Central (ftUS)", unit: "US survey foot" },
  { code: "EPSG:2278", name: "NAD83 / Texas South Central (ftUS)", unit: "US survey foot" },
  { code: "EPSG:2229", name: "NAD83 / California zone 5 (ftUS)", unit: "US survey foot" },
  { code: "EPSG:2926", name: "NAD83(HARN) / Washington North (ftUS)", unit: "US survey foot" },
  { code: "EPSG:27700", name: "OSGB 1936 / British National Grid", unit: "metre" },
  { code: "EPSG:2154", name: "RGF93 / Lambert-93 (France)", unit: "metre" },
  { code: "EPSG:28992", name: "Amersfoort / RD New (Netherlands)", unit: "metre" },
  { code: "EPSG:3035", name: "ETRS89-extended / LAEA Europe", unit: "metre" },
  ...utm(32600, "WGS 84", [1, 60], "N"),
  ...utm(32700, "WGS 84", [1, 60], "S"),
  ...utm(26900, "NAD83", [1, 23], "N"),
  ...utm(25800, "ETRS89", [28, 38], "N"),
];

/** Normalise "3438", "epsg 3438" or "EPSG:3438" to "EPSG:3438"; null when not a code. */
export function normaliseEpsg(text: string): string | null {
  const m = /^\s*(?:epsg\s*[:\s]?\s*)?(\d{4,6})\s*$/i.exec(text);
  return m ? `EPSG:${m[1]}` : null;
}

/** Entries matching every word of the query (code or name), best matches first. */
export function searchEpsg(query: string, limit = 12): EpsgEntry[] {
  const q = query.trim().toLowerCase();
  if (!q) return EPSG_LIST.slice(0, limit);
  const code = normaliseEpsg(q);
  const words = q.replace(/^epsg\s*:?/, "").split(/\s+/).filter(Boolean);
  const scored: { e: EpsgEntry; s: number }[] = [];
  for (const e of EPSG_LIST) {
    const hay = `${e.code} ${e.name}`.toLowerCase();
    if (code && e.code === code) {
      scored.push({ e, s: 0 });
      continue;
    }
    if (!words.every((w) => hay.includes(w))) continue;
    scored.push({ e, s: e.code.toLowerCase().includes(words[0] ?? "") ? 1 : 2 });
  }
  scored.sort((a, b) => a.s - b.s);
  const out = scored.slice(0, limit).map((x) => x.e);
  if (code && !out.some((e) => e.code === code)) out.unshift({ code, name: "custom code", unit: "metre" });
  return out.slice(0, limit);
}

/** The UTM zone (WGS 84) containing a lon/lat, as an EPSG code: a CRS hint from a centroid. */
export function utmForLonLat(lon: number, lat: number): string | null {
  if (!Number.isFinite(lon) || !Number.isFinite(lat) || Math.abs(lat) > 84) return null;
  const zone = Math.min(60, Math.max(1, Math.floor((lon + 180) / 6) + 1));
  return `EPSG:${(lat >= 0 ? 32600 : 32700) + zone}`;
}
