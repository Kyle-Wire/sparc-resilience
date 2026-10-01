import { useEffect, useId, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useUi, type Command } from "../../stores/ui";
import { trapTab } from "./focus";

/** Every query word must appear in the label, group or keywords; label-prefix hits rank first. */
export function filterCommands(commands: Command[], query: string, limit = 60): Command[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return commands.slice(0, limit);
  const scored: { c: Command; s: number }[] = [];
  for (const c of commands) {
    const hay = `${c.label} ${c.group} ${c.keywords ?? ""} ${c.hint ?? ""}`.toLowerCase();
    if (!words.every((w) => hay.includes(w))) continue;
    const label = c.label.toLowerCase();
    const s = label.startsWith(words[0]) ? 0 : label.includes(words[0]) ? 1 : 2;
    scored.push({ c, s });
  }
  return scored.sort((a, b) => a.s - b.s).slice(0, limit).map((x) => x.c);
}

/**
 * Ctrl/Cmd-K palette: jump to runs, jobs, layers, scenarios and start actions. Pages add
 * commands with `useUi.getState().registerCommands(source, cmds)` (see useCommands).
 */
export function CommandPalette({ base = [] }: { base?: Command[] }) {
  const open = useUi((s) => s.paletteOpen);
  const setOpen = useUi((s) => s.setPaletteOpen);
  const sources = useUi((s) => s.commandSources);
  const [q, setQ] = useState("");
  const [active, setActive] = useState(0);
  const listId = useId();
  const root = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const all = useMemo(() => [...base, ...Object.values(sources).flat()], [base, sources]);
  const shown = useMemo(() => filterCommands(all, q), [all, q]);

  useEffect(() => {
    if (!open) return;
    // Focus returns to where it was (e.g. the search button) when the palette closes.
    const restore = typeof document !== "undefined" ? (document.activeElement as HTMLElement | null) : null;
    setQ("");
    setActive(0);
    const t = setTimeout(() => inputRef.current?.focus(), 0);
    return () => {
      clearTimeout(t);
      if (restore && restore.isConnected && restore !== document.body) restore.focus?.();
    };
  }, [open]);
  useEffect(() => setActive(0), [q]);
  // Keep the highlighted option visible while arrowing through a long list.
  useEffect(() => {
    if (open) document.getElementById(`${listId}-${active}`)?.scrollIntoView?.({ block: "nearest" });
  }, [open, active, listId]);

  if (!open) return null;
  const run = (c: Command | undefined) => {
    if (!c) return;
    setOpen(false);
    c.run();
  };
  return createPortal(
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && setOpen(false)}>
      <div
        ref={root}
        className="dialog palette"
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            e.stopPropagation();
            setOpen(false);
          } else if (e.key === "ArrowDown") {
            e.preventDefault();
            setActive((a) => Math.min(shown.length - 1, a + 1));
          } else if (e.key === "ArrowUp") {
            e.preventDefault();
            setActive((a) => Math.max(0, a - 1));
          } else if (e.key === "Enter") {
            e.preventDefault();
            run(shown[active]);
          }
          trapTab(e, root.current);
        }}
      >
        <input
          ref={inputRef}
          type="search"
          role="combobox"
          aria-expanded="true"
          aria-controls={listId}
          aria-activedescendant={shown[active] ? `${listId}-${active}` : undefined}
          aria-label="Search commands, runs, jobs, layers and scenarios"
          placeholder="Jump to a run, job, layer or scenario…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
        <ul id={listId} role="listbox" aria-label="Commands">
          {shown.length === 0 ? <li className="muted">No matches</li> : null}
          {shown.map((c, i) => (
            <li
              key={c.id}
              id={`${listId}-${i}`}
              role="option"
              aria-selected={i === active}
              onMouseEnter={() => setActive(i)}
              onMouseDown={(e) => {
                e.preventDefault();
                run(c);
              }}
            >
              <span>
                {c.label}
                {c.hint ? <span className="cap"> · {c.hint}</span> : null}
              </span>
              <span className="group">{c.group}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>,
    document.body,
  );
}

/** Register page-level commands while mounted (e.g. the Map registers its layers). */
export function useCommands(source: string, commands: Command[] | null): void {
  const register = useUi((s) => s.registerCommands);
  const unregister = useUi((s) => s.unregisterCommands);
  useEffect(() => {
    if (!commands) return;
    register(source, commands);
    return () => unregister(source);
  }, [source, commands, register, unregister]);
}
