// Colour ramps and LUTs (SPEC §6.3), built from the SPARC brand ramp. The results-page
// template (sparc/core/results_page/template.html) still carries its own copy of RAMPS.
// `lut(stops, n)`: equally spaced stops, linear interpolation in OKLab (Ottosson matrices),
// sRGB gamma, clamp, Math.round. The golden entries are asserted in src/test/palette.test.ts
// against src/test/fixtures/palette.json.
//
// Sequential: the brand ramp without pink, so OKLab lightness is monotonic (seqDark runs
// dark→light for dark backgrounds; seqLight is the reverse, starting from a pale yellow so
// low values stay visible on paper). Diverging (ΔT, negative = cooler): index 0 is the cool
// end (the brand purple, OKLab b < 0), the middle stop is a visible warm neutral (not the surface or
// page colour, so cells near the centre stay distinct from empty canvas), the last stop is the warm end.
// The dark cool end is pale so its ends stay ≥ 0.15 ΔE apart under tritan simulation.

export const RAMPS = {
  seqLight: ["#fcefb4", "#f0b632", "#e79024", "#e76c25", "#e94461", "#9e337d", "#602468"],
  seqDark: ["#602468", "#9e337d", "#e94461", "#e76c25", "#e79024", "#f0b632", "#fbdd46"],
  divLight: ["#3d1650", "#602468", "#b79ac4", "#e9e2d6", "#f3b77a", "#e76c25", "#9c2a14"],
  divDark: ["#f0dcf5", "#c58bd0", "#5a2a63", "#3a3134", "#8e3a1c", "#e76c25", "#f8ab5e"],
} as const;

export type RampName = keyof typeof RAMPS;

/**
 * Canvas-side copies of the theme tokens the renderers need (a canvas cannot read CSS
 * variables). Keep in sync with tokens.css.
 */
export const THEME_COLORS = {
  light: { s1: "#602468", s2: "#e73c25", s3: "#b9770e", s4: "#d94f9e", s5: "#f0963f", s6: "#6e6358", gray: "#b8b0a3", nodata: "#e4ded2", hatch: "#4a3f3a", ink: "#1a1416", ink2: "#4a3f3a", muted: "#6e6358", surface: "#ffffff", page: "#f7f4ee", grid: "#e6e0d4", axis: "#c9c2b3", accent: "#e73c25", select: "#1a1416" },
  dark: { s1: "#c58bd0", s2: "#f0553c", s3: "#f0b632", s4: "#e8489a", s5: "#f08a3c", s6: "#a0958a", gray: "#8a7f78", nodata: "#2e2729", hatch: "#d2c9bc", ink: "#f4efe6", ink2: "#d2c9bc", muted: "#a0958a", surface: "#1d181a", page: "#141012", grid: "#2a2326", axis: "#4a4043", accent: "#f0553c", select: "#f4efe6" },
} as const;

export type ThemeColors = { [K in keyof (typeof THEME_COLORS)["light"]]: string };

export function themeColors(dark: boolean): ThemeColors {
  return dark ? THEME_COLORS.dark : THEME_COLORS.light;
}

export function hexToRgb01(hex: string): [number, number, number] {
  let h = hex.replace("#", "");
  if (h.length === 3) h = h.split("").map((c) => c + c).join("");
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255) as [number, number, number];
}

export function hexToRgb255(hex: string): [number, number, number] {
  return hexToRgb01(hex).map((c) => Math.round(c * 255)) as [number, number, number];
}

const lin = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
const delin = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);

/** "#rrggbb" → linear-light sRGB in [0, 1]. */
export function hexToLinear(hex: string): [number, number, number] {
  return hexToRgb01(hex).map(lin) as [number, number, number];
}

export function toLab(hex: string): [number, number, number] {
  return linearToLab(hexToLinear(hex));
}

/** Linear-light sRGB → OKLab (Ottosson). */
export function linearToLab([r, g, b]: readonly [number, number, number]): [number, number, number] {
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

export function fromLab([L, a, b]: readonly [number, number, number]): [number, number, number] {
  const l = Math.pow(L + 0.3963377774 * a + 0.2158037573 * b, 3);
  const m = Math.pow(L - 0.1055613458 * a - 0.0638541728 * b, 3);
  const s = Math.pow(L - 0.0894841775 * a - 1.291485548 * b, 3);
  const rgb = [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
  return rgb.map((c) => Math.round(255 * Math.min(1, Math.max(0, delin(c))))) as [number, number, number];
}

/** n-entry RGB LUT (n·3 bytes) interpolated in OKLab between equally spaced stops. */
export function lut(stops: readonly string[], n = 256): Uint8ClampedArray {
  const labs = stops.map(toLab);
  const out = new Uint8ClampedArray(n * 3);
  for (let i = 0; i < n; i++) {
    const t = (i / (n - 1)) * (labs.length - 1);
    const k = Math.min(labs.length - 2, Math.floor(t));
    const f = t - k;
    const c = fromLab(labs[k].map((v, j) => v + (labs[k + 1][j] - v) * f) as [number, number, number]);
    out.set(c, i * 3);
  }
  return out;
}

const hex2 = (v: number) => v.toString(16).padStart(2, "0");

export function rgbToHex(r: number, g: number, b: number): string {
  return `#${hex2(r)}${hex2(g)}${hex2(b)}`;
}

/** Entry i of an RGB LUT as "#rrggbb". */
export function lutHex(table: Uint8ClampedArray, i: number): string {
  return rgbToHex(table[i * 3], table[i * 3 + 1], table[i * 3 + 2]);
}

/** Pack RGBA into the Uint32 layout of ImageData on a little-endian host (0xAABBGGRR). */
export function packRgba(r: number, g: number, b: number, a = 255): number {
  return ((a << 24) | (b << 16) | (g << 8) | r) >>> 0;
}

export function unpackRgba(v: number): [number, number, number, number] {
  return [v & 255, (v >>> 8) & 255, (v >>> 16) & 255, (v >>> 24) & 255];
}

/** An RGB LUT as packed opaque RGBA words, ready for Uint32Array pixel writes. */
export function lutToUint32(table: Uint8ClampedArray): Uint32Array {
  const n = table.length / 3;
  const out = new Uint32Array(n);
  for (let i = 0; i < n; i++) out[i] = packRgba(table[i * 3], table[i * 3 + 1], table[i * 3 + 2]);
  return out;
}

export type Luts = {
  dark: boolean;
  seq: Uint8ClampedArray;
  div: Uint8ClampedArray;
  seq32: Uint32Array;
  div32: Uint32Array;
};

const lutCache = new Map<boolean, Luts>();

/** The sequential and diverging LUTs for a theme (built once per theme). */
export function getLuts(dark: boolean): Luts {
  let l = lutCache.get(dark);
  if (!l) {
    const seq = lut(dark ? RAMPS.seqDark : RAMPS.seqLight);
    const div = lut(dark ? RAMPS.divDark : RAMPS.divLight);
    l = { dark, seq, div, seq32: lutToUint32(seq), div32: lutToUint32(div) };
    lutCache.set(dark, l);
  }
  return l;
}

/** CSS colour for a value on a ramp, t in [0, 1] (charts use the same LUTs as maps). */
export function rampColor(kind: "seq" | "div", t: number, dark: boolean): string {
  const table = getLuts(dark)[kind];
  const i = Math.round(Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0)) * 255);
  return lutHex(table, i);
}

/** NWS heat-index categories (below caution, caution, extreme caution, danger, extreme danger): warm, ordered
 *  colours for the `palette: "heat"` categorical layers and the Heat tab's charts (tokens --hc0…--hc4). */
export const HEAT_COLORS: Record<"light" | "dark", [string, string, string, string, string]> = {
  light: ["#cfc8bb", "#f0b632", "#e79024", "#e73c25", "#602468"],
  dark: ["#4a4043", "#f0b632", "#e79024", "#f0553c", "#b46cc0"],
};

export function heatColors(dark: boolean): [string, string, string, string, string] {
  return dark ? HEAT_COLORS.dark : HEAT_COLORS.light;
}

/** Categorical colours: six distinct series hues (--s1…--s6) plus grey for muted series. */
export function catColors(dark: boolean): { colors: [string, string, string, string, string, string]; gray: string } {
  const c = themeColors(dark);
  return { colors: [c.s1, c.s2, c.s3, c.s4, c.s5, c.s6], gray: c.gray };
}

/** Number of stops of the stage-tone ramp (--tone-0 … --tone-6, purple → gold). */
export const STAGE_TONE_STOPS = 7;

/** Position of stage i of n on the stage-tone ramp, in stops (0 … STAGE_TONE_STOPS − 1). */
export function stageTonePos(i: number, n: number): number {
  const t = n > 1 ? Math.min(1, Math.max(0, i / (n - 1))) : 0;
  return t * (STAGE_TONE_STOPS - 1);
}

/** The nearest stage-tone stop of stage i of n (for data-tone hooks such as dark numerals). */
export function stageToneIndex(i: number, n: number): number {
  return Math.round(stageTonePos(i, n));
}

/**
 * Colour of stage i of n: the purple→gold ramp (--tone-0…6) sampled at i/(n−1), so n stages run
 * across the ramp once without wrapping. Between stops it is an OKLab color-mix of the two
 * neighbouring tones (theme-aware; resolveCssVars flattens it to a hex for exports).
 */
export function stageTone(i: number, n: number): string {
  const p = stageTonePos(i, n);
  const k = Math.min(STAGE_TONE_STOPS - 2, Math.floor(p));
  const f = p - k;
  if (f < 0.005) return `var(--tone-${k})`;
  if (f > 0.995) return `var(--tone-${k + 1})`;
  return `color-mix(in oklab, var(--tone-${k}) ${Math.round((1 - f) * 1000) / 10}%, var(--tone-${k + 1}))`;
}

/** Mix two hex colours in OKLab, `wa` the weight of `a` in [0, 1] (CSS color-mix(in oklab, a wa, b)). */
export function mixOklab(a: string, b: string, wa: number): string {
  const la = toLab(a);
  const lb = toLab(b);
  const w = Math.min(1, Math.max(0, wa));
  return rgbToHex(...fromLab([0, 1, 2].map((j) => la[j] * w + lb[j] * (1 - w)) as [number, number, number]));
}

/** Token values for contexts without CSS (exported SVG/PNG, canvas). Mirrors tokens.css. */
export const TOKEN_VALUES: Record<"light" | "dark", Record<string, string>> = {
  light: {
    "--page": "#f7f4ee", "--surface": "#ffffff", "--surface-2": "#ede8dd", "--shell": "#ede8dd", "--ink": "#1a1416", "--ink-2": "#4a3f3a", "--muted": "#6e6358",
    "--grid": "#e6e0d4", "--axis": "#c9c2b3", "--line": "rgba(26, 20, 22, 0.12)", "--line-strong": "#c9c2b3",
    "--accent": "#e73c25", "--accent-strong": "#c4301b", "--accent-ink": "#b02a17", "--on-accent": "#ffffff", "--s1": "#602468", "--s2": "#e73c25", "--s3": "#b9770e", "--s4": "#d94f9e", "--s5": "#f0963f", "--s6": "#6e6358", "--gray-mark": "#b8b0a3",
    "--tone-0": "#602468", "--tone-1": "#9e337d", "--tone-2": "#e94d9b", "--tone-3": "#e73c25", "--tone-4": "#e76c25", "--tone-5": "#e79024", "--tone-6": "#f0b632",
    "--good": "#3a8a4a", "--warning": "#f0b632", "--serious": "#e76c25", "--critical": "#c8312b",
    "--good-ink": "#2f6b3a", "--warn-ink": "#8a5a00", "--serious-ink": "#9c4a1f", "--crit-ink": "#a8282a", "--chip": "rgba(26, 20, 22, 0.05)",
    "--cool-1": "#b79ac4", "--cool-2": "#602468", "--cool-3": "#3d1650", "--warm-1": "#f3b77a", "--warm-2": "#e76c25", "--warm-3": "#9c2a14",
    "--hc0": "#cfc8bb", "--hc1": "#f0b632", "--hc2": "#e79024", "--hc3": "#e73c25", "--hc4": "#602468",
    "--nodata": "#e4ded2", "--hatch": "#4a3f3a", "--demo": "#8a5a00",
    "--font-body": '"Inter", system-ui, sans-serif', "--font-mono": '"JetBrains Mono", ui-monospace, monospace', "--font-display": '"Inter", system-ui, sans-serif',
  },
  dark: {
    "--page": "#141012", "--surface": "#1d181a", "--surface-2": "#262022", "--shell": "#181315", "--ink": "#f4efe6", "--ink-2": "#d2c9bc", "--muted": "#a0958a",
    "--grid": "#2a2326", "--axis": "#4a4043", "--line": "rgba(247, 244, 238, 0.10)", "--line-strong": "#4a4043",
    "--accent": "#f0553c", "--accent-strong": "#e73c25", "--accent-ink": "#ff8a72", "--on-accent": "#ffffff", "--s1": "#c58bd0", "--s2": "#f0553c", "--s3": "#f0b632", "--s4": "#e8489a", "--s5": "#f08a3c", "--s6": "#a0958a", "--gray-mark": "#8a7f78",
    "--tone-0": "#b46cc0", "--tone-1": "#b8428e", "--tone-2": "#e94d9b", "--tone-3": "#f0553c", "--tone-4": "#e76c25", "--tone-5": "#e79024", "--tone-6": "#f0b632",
    "--good": "#3a8a4a", "--warning": "#f0b632", "--serious": "#e76c25", "--critical": "#c8312b",
    "--good-ink": "#7cc489", "--warn-ink": "#f0b632", "--serious-ink": "#f0935a", "--crit-ink": "#f07a6e", "--chip": "rgba(247, 244, 238, 0.06)",
    "--cool-1": "#5a2a63", "--cool-2": "#c58bd0", "--cool-3": "#f0dcf5", "--warm-1": "#8e3a1c", "--warm-2": "#e76c25", "--warm-3": "#f8ab5e",
    "--hc0": "#4a4043", "--hc1": "#f0b632", "--hc2": "#e79024", "--hc3": "#f0553c", "--hc4": "#b46cc0",
    "--nodata": "#2e2729", "--hatch": "#d2c9bc", "--demo": "#f0b632",
    "--font-body": '"Inter", system-ui, sans-serif', "--font-mono": '"JetBrains Mono", ui-monospace, monospace', "--font-display": '"Inter", system-ui, sans-serif',
  },
};

/** Resolve a CSS custom property: the live computed value, else the token table. */
export function cssVar(name: string, dark: boolean): string {
  try {
    if (typeof document !== "undefined" && typeof getComputedStyle === "function") {
      const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      if (v) return v;
    }
  } catch {
    /* no CSSOM */
  }
  return TOKEN_VALUES[dark ? "dark" : "light"][name] ?? "currentColor";
}

/**
 * Replace every var(--x[, fallback]) in a CSS/SVG string with its resolved value, then flatten
 * color-mix(in oklab, #a p%, #b) (the stage tones) to a plain hex so exports render anywhere.
 */
export function resolveCssVars(text: string, dark: boolean): string {
  return text
    .replace(/var\((--[\w-]+)(?:\s*,\s*([^)]+))?\)/g, (_, name: string, fb?: string) => {
      const v = cssVar(name, dark);
      return v === "currentColor" && fb ? fb.trim() : v;
    })
    .replace(/color-mix\(in oklab,\s*(#[0-9a-fA-F]{6})\s+([\d.]+)%,\s*(#[0-9a-fA-F]{6})\s*\)/g, (_, a: string, p: string, b: string) => mixOklab(a, b, Number(p) / 100));
}
