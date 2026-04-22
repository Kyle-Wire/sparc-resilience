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
  { key: "dark", label: "Dark", paper: "#1a1416", line: "#3a2f31" },
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
  // Set/remove a `data-theme="dark"` attribute so CSS can flip ink colors
  // when the user picks the dark paper tone.
  if (theme.paperTone === "dark") {
    document.documentElement.setAttribute("data-theme", "dark");
    document.documentElement.style.setProperty("--color-sparc-ink", "#f5efea");
    document.documentElement.style.setProperty("--color-sparc-ink-2", "#cfc6bd");
    document.documentElement.style.setProperty("--color-sparc-muted", "#8a7e72");
  } else {
    document.documentElement.removeAttribute("data-theme");
    document.documentElement.style.removeProperty("--color-sparc-ink");
    document.documentElement.style.removeProperty("--color-sparc-ink-2");
    document.documentElement.style.removeProperty("--color-sparc-muted");
  }
  localStorage.setItem("sparc-theme", JSON.stringify(theme));
}
