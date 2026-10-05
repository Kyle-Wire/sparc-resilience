// Colour ramps and LUTs (SPEC §6.3), ported verbatim from the results-page template.
// `lut(stops, n)`: equally spaced stops, linear interpolation in OKLab (Ottosson matrices),
// sRGB gamma, clamp, Math.round. The golden entries of SPEC §6.3 are asserted in
// src/test/palette.test.ts against src/test/fixtures/palette.json.

export const RAMPS = {
  seqLight: ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
  seqDark: ["#1d2f45", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
  divLight: ["#0d366b", "#256abf", "#6da7ec", "#f0efec", "#f19a8f", "#d6403f", "#8a1f22"],
  divDark: ["#9ec5f4", "#3987e5", "#1f4f8a", "#383835", "#8f3434", "#e66767", "#f6b3ab"],
} as const;

export type RampName = keyof typeof RAMPS;

/**
 * Canvas-side copies of the theme tokens the renderers need (a canvas cannot read CSS
 * variables). Keep in sync with tokens.css.
 */
export const THEME_COLORS = {
  light: { s1: "#2a78d6", s2: "#eb6834", s3: "#1baf7a", gray: "#b9b8b1", nodata: "#e4e4df", hatch: "#52514e", ink: "#0b0b0b", ink2: "#52514e", muted: "#7a7974", surface: "#fcfcfb", page: "#f5f6f4", grid: "#e1e0d9", axis: "#c3c2b7", accent: "#2a78d6", select: "#eb6834" },
  dark: { s1: "#3987e5", s2: "#d95926", s3: "#199e70", gray: "#5e5d58", nodata: "#2a2a28", hatch: "#c3c2b7", ink: "#ffffff", ink2: "#c3c2b7", muted: "#9a998f", surface: "#1a1a19", page: "#0f100f", grid: "#2c2c2a", axis: "#383835", accent: "#3987e5", select: "#d95926" },
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

export function toLab(hex: string): [number, number, number] {
  const [r, g, b] = hexToRgb01(hex).map(lin);
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
  light: ["#c9cec6", "#f2c14e", "#ec8a2c", "#d4402f", "#8c1c2e"],
  dark: ["#4a4d48", "#e9c46a", "#f08c3a", "#e8564a", "#c2364a"],
};

export function heatColors(dark: boolean): [string, string, string, string, string] {
  return dark ? HEAT_COLORS.dark : HEAT_COLORS.light;
}

/** Categorical colours: at most three plus grey (SPEC §6.3). */
export function catColors(dark: boolean): { colors: [string, string, string]; gray: string } {
  const c = themeColors(dark);
  return { colors: [c.s1, c.s2, c.s3], gray: c.gray };
}

/** Token values for contexts without CSS (exported SVG/PNG, canvas). Mirrors tokens.css. */
export const TOKEN_VALUES: Record<"light" | "dark", Record<string, string>> = {
  light: {
    "--page": "#f5f6f4", "--surface": "#fcfcfb", "--surface-2": "#f0f1ee", "--ink": "#0b0b0b", "--ink-2": "#52514e", "--muted": "#7a7974",
    "--grid": "#e1e0d9", "--axis": "#c3c2b7", "--line": "rgba(11, 11, 11, 0.10)", "--line-strong": "rgba(11, 11, 11, 0.22)",
    "--accent": "#2a78d6", "--accent-ink": "#1c5cab", "--on-accent": "#ffffff", "--s1": "#2a78d6", "--s2": "#eb6834", "--s3": "#1baf7a", "--gray-mark": "#b9b8b1",
    "--good": "#0ca30c", "--warning": "#fab219", "--serious": "#ec835a", "--critical": "#d03b3b",
    "--good-ink": "#006300", "--warn-ink": "#8a5a00", "--serious-ink": "#9c4a1f", "--crit-ink": "#a8282a", "--chip": "#eceeea",
    "--nodata": "#e4e4df", "--hatch": "#52514e", "--demo": "#8a5a00",
    "--font-body": '"Public Sans", system-ui, sans-serif', "--font-mono": '"IBM Plex Mono", ui-monospace, monospace', "--font-display": '"Archivo", "Arial Narrow", sans-serif',
  },
  dark: {
    "--page": "#0f100f", "--surface": "#1a1a19", "--surface-2": "#222220", "--ink": "#ffffff", "--ink-2": "#c3c2b7", "--muted": "#9a998f",
    "--grid": "#2c2c2a", "--axis": "#383835", "--line": "rgba(255, 255, 255, 0.10)", "--line-strong": "rgba(255, 255, 255, 0.24)",
    "--accent": "#3987e5", "--accent-ink": "#86b6ef", "--on-accent": "#ffffff", "--s1": "#3987e5", "--s2": "#d95926", "--s3": "#199e70", "--gray-mark": "#5e5d58",
    "--good": "#0ca30c", "--warning": "#fab219", "--serious": "#ec835a", "--critical": "#d03b3b",
    "--good-ink": "#0ca30c", "--warn-ink": "#fab219", "--serious-ink": "#ec835a", "--crit-ink": "#e66767", "--chip": "#242423",
    "--nodata": "#2a2a28", "--hatch": "#c3c2b7", "--demo": "#fab219",
    "--font-body": '"Public Sans", system-ui, sans-serif', "--font-mono": '"IBM Plex Mono", ui-monospace, monospace', "--font-display": '"Archivo", "Arial Narrow", sans-serif',
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

/** Replace every var(--x[, fallback]) in a CSS/SVG string with its resolved value. */
export function resolveCssVars(text: string, dark: boolean): string {
  return text.replace(/var\((--[\w-]+)(?:\s*,\s*([^)]+))?\)/g, (_, name: string, fb?: string) => {
    const v = cssVar(name, dark);
    return v === "currentColor" && fb ? fb.trim() : v;
  });
}
