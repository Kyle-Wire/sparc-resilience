// The compare tray (SPEC §7.4 "Pin to the compare tray", §7.12): up to four items per run
// (exact results, configured scenarios, plans, the baseline), kept in localStorage as a
// convenience so the tray survives reloads. The Compare page itself is driven by its URL.
import { create } from "zustand";
import type { ItemRef } from "../../../api/lab";
import { readLocal, writeLocal } from "../../../stores/ui";
import { decodeItemRef, encodeItemRef } from "./doc";

export const TRAY_MAX = 4;

type TrayItem = { ref: string; label: string };

type TrayState = {
  byRun: Record<string, TrayItem[]>;
  pin: (rid: string, ref: ItemRef, label: string) => boolean;
  unpin: (rid: string, ref: ItemRef | string) => void;
  clear: (rid: string) => void;
};

const KEY = "sparc.lab.tray";

export const useTray = create<TrayState>((set, get) => ({
  byRun: readLocal<Record<string, TrayItem[]>>(KEY, {}),
  pin: (rid, ref, label) => {
    const cur = get().byRun[rid] ?? [];
    const enc = encodeItemRef(ref);
    if (cur.some((i) => i.ref === enc)) return true;
    if (cur.length >= TRAY_MAX) return false;
    const byRun = { ...get().byRun, [rid]: [...cur, { ref: enc, label }] };
    set({ byRun });
    writeLocal(KEY, byRun);
    return true;
  },
  unpin: (rid, ref) => {
    const enc = typeof ref === "string" ? ref : encodeItemRef(ref);
    const byRun = { ...get().byRun, [rid]: (get().byRun[rid] ?? []).filter((i) => i.ref !== enc) };
    set({ byRun });
    writeLocal(KEY, byRun);
  },
  clear: (rid) => {
    const byRun = { ...get().byRun };
    delete byRun[rid];
    set({ byRun });
    writeLocal(KEY, byRun);
  },
}));

/** The tray's refs for a run (invalid stored entries dropped). */
export function trayRefs(items: TrayItem[] | undefined): { ref: ItemRef; enc: string; label: string }[] {
  return (items ?? []).flatMap((i) => {
    const ref = decodeItemRef(i.ref);
    return ref ? [{ ref, enc: i.ref, label: i.label }] : [];
  });
}

/** `/r/<rid>/lab/compare?items=a,b,…` for a set of refs. */
export function compareHref(rid: string, refs: ItemRef[]): string {
  const items = refs.map(encodeItemRef).map((s) => s.replace(/%/g, "%25").replace(/,/g, "%2C"));
  return `/r/${encodeURIComponent(rid)}/lab/compare?items=${encodeURIComponent(items.join(","))}`;
}
