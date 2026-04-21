/**
 * Theme settings — extracted so both App.tsx and SettingsPage can share them.
 */

export const LOGO_HUES = [
  { key: "ink", label: "Ink", color: "#1a1416" },
  { key: "red", label: "Red", color: "#e73c25" },
  { key: "purple", label: "Purple", color: "#602468" },
  { key: "amber", label: "Amber", color: "#e79024" },
] as const;

export const PAPER_TONES = [
  { key: "warm", label: "Warm", paper: "#f7f4ee", line: "#c9c2b3" },
  { key: "cool", label: "Cool", paper: "#f3f4f2", line: "#c5cac4" },
  { key: "white", label: "White", paper: "#ffffff", line: "#d8d4cb" },
] as const;

export type LogoHue = (typeof LOGO_HUES)[number]["key"];
export type PaperTone = (typeof PAPER_TONES)[number]["key"];

export interface ThemeSettings {
  logoHue: LogoHue;
  paperTone: PaperTone;
}

export function loadTheme(): ThemeSettings {
  try {
    const raw = localStorage.getItem("sparc-theme");
    if (raw) return JSON.parse(raw);
  } catch {
    /* ignore */
  }
  return { logoHue: "ink", paperTone: "warm" };
}

export function applyTheme(theme: ThemeSettings) {
  const tone = PAPER_TONES.find((t) => t.key === theme.paperTone) ?? PAPER_TONES[0];
  document.documentElement.style.setProperty("--color-sparc-paper", tone.paper);
  document.documentElement.style.setProperty("--color-sparc-line", tone.line);
  document.documentElement.style.setProperty("--color-sparc-gray-100", tone.paper);
  document.documentElement.style.setProperty("--color-sparc-gray-300", tone.line);
  localStorage.setItem("sparc-theme", JSON.stringify(theme));
}
