/**
 * Theme system — 5 named presets replace the old hue+tone two-knob approach.
 * Storage key: "sparc-theme" (hyphen) — used consistently everywhere.
 */

export interface ThemePreset {
  key: string;
  label: string;
  /** preview swatch colors */
  swatch: { bg: string; accent: string };
  vars: Record<string, string>;
}

export const THEME_PRESETS: ThemePreset[] = [
  {
    key: "warm-paper",
    label: "Warm Paper",
    swatch: { bg: "#f7f4ee", accent: "#e73c25" },
    vars: {
      "--color-sparc-paper": "#f7f4ee",
      "--color-sparc-paper-2": "#ede8dd",
      "--color-sparc-line": "#c9c2b3",
      "--color-sparc-ink": "#1a1416",
      "--color-sparc-ink-2": "#2b2327",
      "--color-sparc-muted": "#6e6358",
      "--color-sparc-gray-100": "#f7f4ee",
      "--color-sparc-gray-200": "#ede8dd",
      "--color-sparc-gray-300": "#c9c2b3",
      "--color-sparc-gray-600": "#6e6358",
      "--color-sparc-gray-800": "#1a1416",
      "--color-sparc-crimson": "#e73c25",
    },
  },
  {
    key: "dark",
    label: "Dark",
    swatch: { bg: "#141214", accent: "#ff5540" },
    vars: {
      "--color-sparc-paper": "#141214",
      "--color-sparc-paper-2": "#1e1a1e",
      "--color-sparc-line": "#332d33",
      "--color-sparc-ink": "#f0ebe4",
      "--color-sparc-ink-2": "#cfc6be",
      "--color-sparc-muted": "#7d7375",
      "--color-sparc-gray-100": "#141214",
      "--color-sparc-gray-200": "#1e1a1e",
      "--color-sparc-gray-300": "#332d33",
      "--color-sparc-gray-600": "#7d7375",
      "--color-sparc-gray-800": "#f0ebe4",
      "--color-sparc-crimson": "#ff5540",
    },
  },
  {
    key: "high-contrast",
    label: "High Contrast",
    swatch: { bg: "#000000", accent: "#ffff00" },
    vars: {
      "--color-sparc-paper": "#000000",
      "--color-sparc-paper-2": "#111111",
      "--color-sparc-line": "#444444",
      "--color-sparc-ink": "#ffffff",
      "--color-sparc-ink-2": "#eeeeee",
      "--color-sparc-muted": "#aaaaaa",
      "--color-sparc-gray-100": "#000000",
      "--color-sparc-gray-200": "#111111",
      "--color-sparc-gray-300": "#444444",
      "--color-sparc-gray-600": "#aaaaaa",
      "--color-sparc-gray-800": "#ffffff",
      "--color-sparc-crimson": "#ffff00",
    },
  },
  {
    key: "sparc-electric",
    label: "SPARC Electric",
    swatch: { bg: "#080612", accent: "#00d4ff" },
    vars: {
      "--color-sparc-paper": "#080612",
      "--color-sparc-paper-2": "#100e1e",
      "--color-sparc-line": "#1e1a36",
      "--color-sparc-ink": "#e0f4ff",
      "--color-sparc-ink-2": "#b8d8f0",
      "--color-sparc-muted": "#6a82a8",
      "--color-sparc-gray-100": "#080612",
      "--color-sparc-gray-200": "#100e1e",
      "--color-sparc-gray-300": "#1e1a36",
      "--color-sparc-gray-600": "#6a82a8",
      "--color-sparc-gray-800": "#e0f4ff",
      "--color-sparc-crimson": "#00d4ff",
      "--color-sparc-purple": "#7c3aed",
      "--color-sparc-pink": "#e040fb",
    },
  },
  {
    key: "cool-slate",
    label: "Cool Slate",
    swatch: { bg: "#f2f4f7", accent: "#5b7af0" },
    vars: {
      "--color-sparc-paper": "#f2f4f7",
      "--color-sparc-paper-2": "#e4e8f0",
      "--color-sparc-line": "#c5ccd8",
      "--color-sparc-ink": "#1e2535",
      "--color-sparc-ink-2": "#2e3a52",
      "--color-sparc-muted": "#6b7a96",
      "--color-sparc-gray-100": "#f2f4f7",
      "--color-sparc-gray-200": "#e4e8f0",
      "--color-sparc-gray-300": "#c5ccd8",
      "--color-sparc-gray-600": "#6b7a96",
      "--color-sparc-gray-800": "#1e2535",
      "--color-sparc-crimson": "#5b7af0",
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

  // Apply preset CSS vars
  for (const [varName, value] of Object.entries(preset.vars)) {
    root.style.setProperty(varName, value);
  }

  // Sync short-name aliases used throughout JSX
  root.style.setProperty("--paper", preset.vars["--color-sparc-paper"] ?? "");
  root.style.setProperty("--paper-2", preset.vars["--color-sparc-paper-2"] ?? "");
  root.style.setProperty("--line", preset.vars["--color-sparc-line"] ?? "");
  root.style.setProperty("--ink", preset.vars["--color-sparc-ink"] ?? "");
  root.style.setProperty("--ink-2", preset.vars["--color-sparc-ink-2"] ?? "");
  root.style.setProperty("--muted", preset.vars["--color-sparc-muted"] ?? "");
  root.style.setProperty("--crimson", preset.vars["--color-sparc-crimson"] ?? "#e73c25");
  root.style.setProperty("--accent", preset.vars["--color-sparc-crimson"] ?? "#e73c25");

  // data-theme for dark/electric CSS selectors
  const darkKeys = ["dark", "sparc-electric", "high-contrast"];
  if (darkKeys.includes(key)) {
    root.setAttribute("data-theme", key);
  } else {
    root.removeAttribute("data-theme");
  }

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
