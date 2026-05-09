/**
 * Theme system — 5 named presets replace the old hue+tone two-knob approach.
 * Storage key: "sparc-theme" (hyphen) — used consistently everywhere.
 */

export interface ThemePreset {
  key: string;
  label: string;
  /** preview swatch: bg + three accent chips shown in Settings */
  swatch: { bg: string; primary: string; secondary: string; warning: string };
  vars: Record<string, string>;
}

// ── Color theory notes ────────────────────────────────────────────────────────
// Each theme defines a complete set of semantic accent slots so no color is
// ever "stuck" at the warm-paper default when a dark or cool theme is active.
//
// Warm Paper  — warm-temperature analogous palette (red-orange primary,
//               plum secondary, golden amber warning)
// Dark        — same hue families, raised brightness ~+25% for dark-bg contrast
// High Contrast — maximum-contrast accessibility palette (WCAG AAA on #000):
//               cyan primary, magenta secondary, yellow warning
// SPARC Electric — neon sci-fi: cyan primary, violet secondary, orange warning
// Cool Slate  — blue-centric split-complementary: indigo primary,
//               violet secondary, warm amber warning for contrast pop
// ─────────────────────────────────────────────────────────────────────────────

export const THEME_PRESETS: ThemePreset[] = [
  {
    key: "warm-paper",
    label: "Warm Paper",
    swatch: { bg: "#f7f4ee", primary: "#e73c25", secondary: "#602468", warning: "#e79024" },
    vars: {
      // Neutrals
      "--color-sparc-paper":   "#f7f4ee",
      "--color-sparc-paper-2": "#ede8dd",
      "--color-sparc-paper-3": "#d8d3c6",
      "--color-sparc-line":    "#c9c2b3",
      "--color-sparc-ink":     "#1a1416",
      "--color-sparc-ink-2":   "#2b2327",
      "--color-sparc-muted":   "#6e6358",
      // Legacy gray aliases
      "--color-sparc-gray-100": "#f7f4ee",
      "--color-sparc-gray-200": "#ede8dd",
      "--color-sparc-gray-300": "#c9c2b3",
      "--color-sparc-gray-600": "#6e6358",
      "--color-sparc-gray-800": "#1a1416",
      // Accents — warm analogous family
      "--color-sparc-crimson":  "#e73c25",
      "--color-sparc-red":      "#e94461",
      "--color-sparc-orange":   "#e76c25",
      "--color-sparc-amber":    "#e79024",
      "--color-sparc-gold":     "#f0b632",
      "--color-sparc-yellow":   "#fbdd46",
      "--color-sparc-purple":   "#602468",
      "--color-sparc-magenta":  "#9e337d",
      "--color-sparc-pink":     "#e94d9b",
    },
  },
  {
    key: "dark",
    label: "Dark",
    swatch: { bg: "#141214", primary: "#ff5540", secondary: "#b87fd9", warning: "#fbbf24" },
    vars: {
      // Neutrals
      "--color-sparc-paper":   "#141214",
      "--color-sparc-paper-2": "#1e1a1e",
      "--color-sparc-paper-3": "#0a080c",
      "--color-sparc-line":    "#332d33",
      "--color-sparc-ink":     "#f0ebe4",
      "--color-sparc-ink-2":   "#cfc6be",
      "--color-sparc-muted":   "#7d7375",
      // Legacy gray aliases
      "--color-sparc-gray-100": "#141214",
      "--color-sparc-gray-200": "#1e1a1e",
      "--color-sparc-gray-300": "#332d33",
      "--color-sparc-gray-600": "#7d7375",
      "--color-sparc-gray-800": "#f0ebe4",
      // Accents — same hue families as warm-paper, raised ~+25% brightness for dark bg
      "--color-sparc-crimson":  "#ff5540",
      "--color-sparc-red":      "#ff6b7a",
      "--color-sparc-orange":   "#fb923c",
      "--color-sparc-amber":    "#fbbf24",
      "--color-sparc-gold":     "#fcd34d",
      "--color-sparc-yellow":   "#fde68a",
      "--color-sparc-purple":   "#b87fd9",
      "--color-sparc-magenta":  "#e879a8",
      "--color-sparc-pink":     "#f472b6",
    },
  },
  {
    key: "high-contrast",
    label: "High Contrast",
    swatch: { bg: "#000000", primary: "#00ffff", secondary: "#ff00ff", warning: "#ffff00" },
    vars: {
      // Neutrals
      "--color-sparc-paper":   "#000000",
      "--color-sparc-paper-2": "#111111",
      "--color-sparc-paper-3": "#000000",
      "--color-sparc-line":    "#555555",
      "--color-sparc-ink":     "#ffffff",
      "--color-sparc-ink-2":   "#eeeeee",
      "--color-sparc-muted":   "#bbbbbb",
      // Legacy gray aliases
      "--color-sparc-gray-100": "#000000",
      "--color-sparc-gray-200": "#111111",
      "--color-sparc-gray-300": "#555555",
      "--color-sparc-gray-600": "#bbbbbb",
      "--color-sparc-gray-800": "#ffffff",
      // Accents — WCAG AAA on black (ratio > 7:1); distinct hue per semantic role
      "--color-sparc-crimson":  "#00ffff", // cyan  → primary action
      "--color-sparc-red":      "#ff6b6b", // soft red for error states
      "--color-sparc-orange":   "#ff9f43",
      "--color-sparc-amber":    "#ffff00", // yellow → warning (highest visibility)
      "--color-sparc-gold":     "#ffd700",
      "--color-sparc-yellow":   "#ffff00",
      "--color-sparc-purple":   "#ff00ff", // magenta → secondary accent
      "--color-sparc-magenta":  "#ff00ff",
      "--color-sparc-pink":     "#ff79c6",
    },
  },
  {
    key: "sparc-electric",
    label: "SPARC Electric",
    swatch: { bg: "#080612", primary: "#00d4ff", secondary: "#a855f7", warning: "#fb923c" },
    vars: {
      // Neutrals
      "--color-sparc-paper":   "#080612",
      "--color-sparc-paper-2": "#100e1e",
      "--color-sparc-paper-3": "#04030d",
      "--color-sparc-line":    "#1e1a36",
      "--color-sparc-ink":     "#e0f4ff",
      "--color-sparc-ink-2":   "#b8d8f0",
      "--color-sparc-muted":   "#6a82a8",
      // Legacy gray aliases
      "--color-sparc-gray-100": "#080612",
      "--color-sparc-gray-200": "#100e1e",
      "--color-sparc-gray-300": "#1e1a36",
      "--color-sparc-gray-600": "#6a82a8",
      "--color-sparc-gray-800": "#e0f4ff",
      // Accents — neon sci-fi: cool primary, split-complementary warm warning
      "--color-sparc-crimson":  "#00d4ff", // cyan → primary action
      "--color-sparc-red":      "#f87171",
      "--color-sparc-orange":   "#fb923c", // neon orange (warm pop on cool dark)
      "--color-sparc-amber":    "#fb923c",
      "--color-sparc-gold":     "#fbbf24",
      "--color-sparc-yellow":   "#fde047",
      "--color-sparc-purple":   "#a855f7", // bright violet → secondary
      "--color-sparc-magenta":  "#e040fb",
      "--color-sparc-pink":     "#f472b6",
    },
  },
  {
    key: "cool-slate",
    label: "Cool Slate",
    swatch: { bg: "#f2f4f7", primary: "#5b7af0", secondary: "#8b5cf6", warning: "#d97706" },
    vars: {
      // Neutrals
      "--color-sparc-paper":   "#f2f4f7",
      "--color-sparc-paper-2": "#e4e8f0",
      "--color-sparc-paper-3": "#cdd3df",
      "--color-sparc-line":    "#c5ccd8",
      "--color-sparc-ink":     "#1e2535",
      "--color-sparc-ink-2":   "#2e3a52",
      "--color-sparc-muted":   "#6b7a96",
      // Legacy gray aliases
      "--color-sparc-gray-100": "#f2f4f7",
      "--color-sparc-gray-200": "#e4e8f0",
      "--color-sparc-gray-300": "#c5ccd8",
      "--color-sparc-gray-600": "#6b7a96",
      "--color-sparc-gray-800": "#1e2535",
      // Accents — blue-centric split-complementary
      // Primary: indigo-blue; secondary: violet (adjacent); warning: amber (complementary pop)
      "--color-sparc-crimson":  "#5b7af0", // indigo-blue → primary action
      "--color-sparc-red":      "#ef4444",
      "--color-sparc-orange":   "#ea580c",
      "--color-sparc-amber":    "#d97706", // deeper amber — more contrast on light cool bg
      "--color-sparc-gold":     "#ca8a04",
      "--color-sparc-yellow":   "#ca8a04",
      "--color-sparc-purple":   "#8b5cf6", // violet → secondary (adjacent to blue on wheel)
      "--color-sparc-magenta":  "#a21caf",
      "--color-sparc-pink":     "#ec4899",
    },
  },
];

export const DEFAULT_THEME_KEY = "warm-paper";
const STORAGE_KEY = "sparc-theme";

export function getPreset(key: string): ThemePreset {
  return THEME_PRESETS.find((p) => p.key === key) ?? THEME_PRESETS[0];
}

export function loadThemeKey(): string {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      // New format: plain string key
      if (typeof parsed === "string") return parsed;
      // Old format: { key } object
      if (parsed?.key && typeof parsed.key === "string") return parsed.key;
      // Old paperTone migration
      if (parsed?.paperTone === "dark") return "dark";
    }
  } catch {
    /* ignore */
  }
  return DEFAULT_THEME_KEY;
}

export function applyTheme(key: string) {
  const preset = getPreset(key);
  const root = document.documentElement;
  const v = preset.vars;

  // Apply all preset CSS vars
  for (const [varName, value] of Object.entries(v)) {
    root.style.setProperty(varName, value);
  }

  // Sync short-name aliases — neutrals
  root.style.setProperty("--paper",   v["--color-sparc-paper"]   ?? "");
  root.style.setProperty("--paper-2", v["--color-sparc-paper-2"] ?? "");
  root.style.setProperty("--paper-3", v["--color-sparc-paper-3"] ?? "");
  root.style.setProperty("--line",    v["--color-sparc-line"]    ?? "");
  root.style.setProperty("--ink",     v["--color-sparc-ink"]     ?? "");
  root.style.setProperty("--ink-2",   v["--color-sparc-ink-2"]   ?? "");
  root.style.setProperty("--muted",   v["--color-sparc-muted"]   ?? "");
  // Shell chrome uses paper-3
  root.style.setProperty("--bg-shell", v["--color-sparc-paper-3"] ?? "");
  root.style.setProperty("--bg",       v["--color-sparc-paper"]   ?? "");
  root.style.setProperty("--bg-elevated", v["--color-sparc-paper"] ?? "");
  root.style.setProperty("--bg-sunken",   v["--color-sparc-paper-2"] ?? "");

  // Sync short-name aliases — accents (these were previously hardcoded in :root)
  root.style.setProperty("--crimson", v["--color-sparc-crimson"] ?? "#e73c25");
  root.style.setProperty("--red",     v["--color-sparc-red"]     ?? "#e94461");
  root.style.setProperty("--orange",  v["--color-sparc-orange"]  ?? "#e76c25");
  root.style.setProperty("--amber",   v["--color-sparc-amber"]   ?? "#e79024");
  root.style.setProperty("--gold",    v["--color-sparc-gold"]    ?? "#f0b632");
  root.style.setProperty("--yellow",  v["--color-sparc-yellow"]  ?? "#fbdd46");
  root.style.setProperty("--purple",  v["--color-sparc-purple"]  ?? "#602468");
  root.style.setProperty("--magenta", v["--color-sparc-magenta"] ?? "#9e337d");
  root.style.setProperty("--pink",    v["--color-sparc-pink"]    ?? "#e94d9b");

  // Semantic aliases built from accents
  root.style.setProperty("--accent",          v["--color-sparc-crimson"] ?? "#e73c25");
  root.style.setProperty("--accent-2",        v["--color-sparc-purple"]  ?? "#602468");
  root.style.setProperty("--accent-warning",  v["--color-sparc-amber"]   ?? "#e79024");
  root.style.setProperty("--status-warn",     v["--color-sparc-amber"]   ?? "#e79024");
  root.style.setProperty("--status-ai",       v["--color-sparc-purple"]  ?? "#602468");
  root.style.setProperty("--status-live",     v["--color-sparc-crimson"] ?? "#e73c25");

  // data-theme for dark/electric CSS selectors
  const darkKeys = ["dark", "sparc-electric", "high-contrast"];
  const isDark = darkKeys.includes(key);
  if (isDark) {
    root.setAttribute("data-theme", key);
  } else {
    root.removeAttribute("data-theme");
  }

  // Grid-line overlay — light on dark, dark on light
  root.style.setProperty("--grid-line", isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.03)");

  localStorage.setItem(STORAGE_KEY, JSON.stringify(key));
}

// ── Legacy shim — keeps old call-sites in App.tsx from breaking ───────────
/** @deprecated Use THEME_PRESETS / loadThemeKey / applyTheme(key) */
export type LogoHue = "ink" | "red" | "purple" | "amber";
/** @deprecated */
export type PaperTone = "warm" | "cool" | "white" | "dark";
/** @deprecated */
export interface ThemeSettings { logoHue: LogoHue; paperTone: PaperTone; }
/** @deprecated */
export const LOGO_HUES = [
  { key: "ink" as LogoHue, label: "Ink", color: "#1a1416" },
  { key: "red" as LogoHue, label: "Red", color: "#e73c25" },
  { key: "purple" as LogoHue, label: "Purple", color: "#602468" },
  { key: "amber" as LogoHue, label: "Amber", color: "#e79024" },
];
/** @deprecated */
export const PAPER_TONES = [
  { key: "warm" as PaperTone, label: "Warm", paper: "#f7f4ee", line: "#c9c2b3" },
  { key: "cool" as PaperTone, label: "Cool", paper: "#f3f4f2", line: "#c5cac4" },
  { key: "white" as PaperTone, label: "White", paper: "#ffffff", line: "#d8d4cb" },
  { key: "dark" as PaperTone, label: "Dark", paper: "#1a1416", line: "#3a2f31" },
];
/** @deprecated */
export function loadTheme(): ThemeSettings { return { logoHue: "ink", paperTone: "warm" }; }
