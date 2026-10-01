// UI state (SPEC §12.4): theme, toasts, command palette, aria-live announcements, the
// current project/run context and small conveniences persisted in localStorage (theme,
// last tab, dock state). Every storage access is wrapped in try/catch.
import { create } from "zustand";
import type { Action } from "../api/types";

export type ThemePref = "system" | "light" | "dark";

const LS_PREFIX = "sparc-studio:";

export function readLocal<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(LS_PREFIX + key);
    return raw === null ? fallback : (JSON.parse(raw) as T);
  } catch {
    return fallback;
  }
}

export function writeLocal(key: string, value: unknown): void {
  try {
    if (value === undefined || value === null) window.localStorage.removeItem(LS_PREFIX + key);
    else window.localStorage.setItem(LS_PREFIX + key, JSON.stringify(value));
  } catch {
    /* storage blocked or full: conveniences only */
  }
}

function systemDark(): boolean {
  try {
    return typeof window !== "undefined" && !!window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  } catch {
    return false;
  }
}

/** Apply a theme preference to <html data-theme>; returns whether dark is in effect. */
export function applyTheme(pref: ThemePref): boolean {
  if (typeof document !== "undefined") {
    const root = document.documentElement;
    if (pref === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", pref);
  }
  return pref === "dark" || (pref === "system" && systemDark());
}

export type ToastKind = "info" | "success" | "warning" | "error";

export type Toast = {
  id: number;
  kind: ToastKind;
  title: string;
  body?: string;
  href?: string;
  linkLabel?: string;
  action?: Action;
  /** Milliseconds before auto-dismiss; 0 keeps it until closed. */
  timeout: number;
};

export type Command = {
  id: string;
  label: string;
  group: string;
  hint?: string;
  keywords?: string;
  run: () => void;
};

export type UiContext = { projectId: string | null; runId: string | null };

type UiState = {
  theme: ThemePref;
  dark: boolean;
  toasts: Toast[];
  paletteOpen: boolean;
  commandSources: Record<string, Command[]>;
  context: UiContext;
  sidebarOpen: boolean;
  notifications: boolean;
  liveMessage: string;
  dock: Record<string, unknown>;
  setTheme: (t: ThemePref) => void;
  syncSystemTheme: () => void;
  pushToast: (t: Omit<Toast, "id" | "timeout"> & { timeout?: number }) => number;
  dismissToast: (id: number) => void;
  setPaletteOpen: (open: boolean) => void;
  registerCommands: (source: string, cmds: Command[]) => void;
  unregisterCommands: (source: string) => void;
  setContext: (c: Partial<UiContext>) => void;
  setSidebarOpen: (open: boolean) => void;
  setNotifications: (on: boolean) => void;
  announce: (msg: string) => void;
  setDock: (key: string, value: unknown) => void;
};

let toastSeq = 1;
const initialTheme = readLocal<ThemePref>("theme", "system");

export const useUi = create<UiState>((set, get) => ({
  theme: initialTheme,
  dark: applyTheme(initialTheme),
  toasts: [],
  paletteOpen: false,
  commandSources: {},
  context: { projectId: null, runId: null },
  sidebarOpen: false,
  notifications: readLocal<boolean>("notifications", false),
  liveMessage: "",
  dock: readLocal<Record<string, unknown>>("dock", {}),
  setTheme: (t) => {
    writeLocal("theme", t === "system" ? null : t);
    set({ theme: t, dark: applyTheme(t) });
  },
  syncSystemTheme: () => {
    const d = applyTheme(get().theme);
    if (d !== get().dark) set({ dark: d });
  },
  pushToast: (t) => {
    const id = toastSeq++;
    const toast: Toast = { timeout: t.kind === "error" ? 0 : 6000, ...t, id };
    set({ toasts: [...get().toasts.slice(-4), toast] });
    return id;
  },
  dismissToast: (id) => set({ toasts: get().toasts.filter((t) => t.id !== id) }),
  setPaletteOpen: (open) => set({ paletteOpen: open }),
  registerCommands: (source, cmds) => set({ commandSources: { ...get().commandSources, [source]: cmds } }),
  unregisterCommands: (source) => {
    const next = { ...get().commandSources };
    delete next[source];
    set({ commandSources: next });
  },
  setContext: (c) => {
    const cur = get().context;
    const next = { ...cur, ...c };
    if (next.projectId !== cur.projectId || next.runId !== cur.runId) set({ context: next });
  },
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  setNotifications: (on) => {
    // Opt-in browser notifications on job end (SPEC §3.1): enabling asks for the permission
    // once (call this from the click that enables it, so the browser shows the prompt).
    if (on && typeof Notification !== "undefined" && Notification.permission === "default") {
      try {
        void Promise.resolve(Notification.requestPermission()).catch(() => {});
      } catch {
        /* old callback-only API or blocked */
      }
    }
    writeLocal("notifications", on);
    set({ notifications: on });
  },
  announce: (msg) => {
    // Re-announce identical messages by toggling a zero-width suffix.
    const prev = get().liveMessage;
    set({ liveMessage: prev === msg ? msg + "​" : msg });
  },
  setDock: (key, value) => {
    const dock = { ...get().dock, [key]: value };
    writeLocal("dock", dock);
    set({ dock });
  },
}));

if (typeof window !== "undefined" && window.matchMedia) {
  try {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => useUi.getState().syncSystemTheme());
  } catch {
    /* old browsers */
  }
}

/** Shorthand for toasts from non-React code. */
export function toast(kind: ToastKind, title: string, extra: Partial<Omit<Toast, "id" | "kind" | "title">> = {}): number {
  return useUi.getState().pushToast({ kind, title, ...extra });
}

/** Remember the last tab per scope (e.g. "run" → "accuracy"). */
export function rememberTab(scope: string, tab: string): void {
  writeLocal(`lasttab:${scope}`, tab);
}

export function lastTab(scope: string, fallback: string): string {
  return readLocal<string>(`lasttab:${scope}`, fallback);
}

/** Whether dark mode is in effect (re-renders on theme changes). */
export function useDark(): boolean {
  return useUi((s) => s.dark);
}
