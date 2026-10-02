// Setup wizard draft (SPEC §9.3): the project's raw config being edited, the version it is
// based on, its validation issues, the column suggestion and the last data check. Kept in a
// store keyed by project so it survives moving between steps (each step is its own URL and
// the page remounts) and the shell's navigation. Saving is one atomic `PUT /config {raw}`
// with `If-Match`; a server change while the draft is clean is adopted silently, while a
// dirty draft keeps the user's edits and flags the newer server version.
import { create } from "zustand";
import type { ColumnSuggestion, ConfigDoc, ConfigRaw, DataCheck } from "../../../api/projects";
import type { Issue } from "../../../api/types";
import { changedSections, getPath, jsonEqual, setPath, type Path } from "../model/dotted";

export type Draft = {
  pid: string;
  /** Server version the draft started from (sent as If-Match). */
  baseVersion: number;
  /** Newest version seen on the server. */
  serverVersion: number;
  saved: ConfigRaw;
  raw: ConfigRaw;
  /** DEFAULTS-merged config of the saved version: display fallback for unset keys. */
  effective: ConfigRaw;
  issues: Issue[];
  /** JSON of the raw the issues belong to (stale when it differs from the draft). */
  issuesKey: string | null;
  suggestion: ColumnSuggestion | null;
  /** Paths pre-filled from the suggestion (badged in the form until edited). */
  suggested: string[];
  check: DataCheck | null;
  checkKey: string | null;
  checkError: unknown;
  checking: boolean;
};

type DraftStore = {
  drafts: Record<string, Draft>;
  adopt: (pid: string, doc: ConfigDoc) => void;
  edit: (pid: string, path: Path, value: unknown) => void;
  replace: (pid: string, raw: ConfigRaw, suggested?: string[]) => void;
  discard: (pid: string) => void;
  forget: (pid: string) => void;
  markSaved: (pid: string, version: number) => void;
  setIssues: (pid: string, issues: Issue[], key: string) => void;
  setSuggestion: (pid: string, s: ColumnSuggestion | null) => void;
  setCheck: (pid: string, patch: Partial<Pick<Draft, "check" | "checkKey" | "checkError" | "checking">>) => void;
};

export function rawKey(raw: ConfigRaw): string {
  return JSON.stringify(raw);
}

function fresh(pid: string, doc: ConfigDoc, prev?: Draft): Draft {
  return {
    pid,
    baseVersion: doc.version,
    serverVersion: doc.version,
    saved: doc.raw ?? {},
    raw: doc.raw ?? {},
    effective: doc.effective ?? {},
    issues: prev?.issues ?? [],
    issuesKey: prev && jsonEqual(prev.raw, doc.raw) ? prev.issuesKey : null,
    suggestion: prev?.suggestion ?? null,
    suggested: [],
    check: prev?.check ?? null,
    checkKey: prev?.checkKey ?? null,
    checkError: null,
    checking: false,
  };
}

export const useDrafts = create<DraftStore>((set, get) => {
  const update = (pid: string, f: (d: Draft) => Partial<Draft>) => {
    const d = get().drafts[pid];
    if (!d) return;
    set({ drafts: { ...get().drafts, [pid]: { ...d, ...f(d) } } });
  };
  return {
    drafts: {},
    adopt: (pid, doc) => {
      const d = get().drafts[pid];
      // A clean draft follows the server (a link applied by an input job, a save elsewhere).
      if (!d || isClean(d)) set({ drafts: { ...get().drafts, [pid]: fresh(pid, doc, d) } });
      // A dirty one keeps the user's edits; the newer version is flagged and a save conflicts.
      else if (doc.version > d.serverVersion) update(pid, () => ({ serverVersion: doc.version }));
    },
    edit: (pid, path, value) =>
      update(pid, (d) => {
        const p = typeof path === "string" ? path : path.map(String).join(".");
        return { raw: setPath(d.raw, path, value), suggested: d.suggested.filter((s) => s !== p) };
      }),
    replace: (pid, raw, suggested) => update(pid, (d) => ({ raw, suggested: suggested ? [...new Set([...d.suggested, ...suggested])] : d.suggested })),
    discard: (pid) => update(pid, (d) => ({ raw: d.saved, suggested: [] })),
    forget: (pid) => {
      const drafts = { ...get().drafts };
      delete drafts[pid];
      set({ drafts });
    },
    markSaved: (pid, version) => update(pid, (d) => ({ saved: d.raw, baseVersion: version, serverVersion: Math.max(version, d.serverVersion), suggested: [] })),
    setIssues: (pid, issues, key) => update(pid, () => ({ issues, issuesKey: key })),
    setSuggestion: (pid, s) => update(pid, () => ({ suggestion: s })),
    setCheck: (pid, patch) => update(pid, () => patch),
  };
});

export function isClean(d: Draft): boolean {
  return jsonEqual(d.saved, d.raw);
}

/** Sections the draft changed (top-level config keys). */
export function dirtySections(d: Draft | undefined): string[] {
  return d ? changedSections(d.saved, d.raw) : [];
}

/** Display value: the draft's own value, else the DEFAULTS-merged one. */
export function draftValue(d: Draft | undefined, path: Path): unknown {
  if (!d) return undefined;
  const own = getPath(d.raw, path);
  return own !== undefined ? own : getPath(d.effective, path);
}

/** The draft of a project (undefined until its config loaded). */
export function useDraft(pid: string): Draft | undefined {
  return useDrafts((s) => s.drafts[pid]);
}

/** Drop every draft (tests). */
export function resetDrafts(): void {
  useDrafts.setState({ drafts: {} });
}
