import { useEffect, useMemo, useRef, useState } from "react";
import type { PageName } from "@/components/layout/Sidebar";

export interface PaletteItem {
  id: string;
  label: string;
  kind: "page" | "predictor" | "scenario" | "treatment" | "action";
  hint?: string;
  /** Optional callback. If omitted and ``page`` is set, navigates to that page. */
  run?: () => void;
  page?: PageName | "Settings";
}

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
  items: PaletteItem[];
  onNavigate: (page: PageName | "Settings") => void;
}

/**
 * Cmd+K command palette. Fuzzy-filters across pages, predictors,
 * scenarios, treatments, and quick actions provided by the host.
 */
export default function CommandPalette({ open, onClose, items, onNavigate }: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Reset on open / focus the input.
  useEffect(() => {
    if (!open) return;
    setQuery("");
    setActive(0);
    queueMicrotask(() => inputRef.current?.focus());
  }, [open]);

  const filtered = useMemo(() => {
    if (!query.trim()) return items.slice(0, 30);
    const q = query.toLowerCase();
    const scored = items
      .map((it) => {
        const hay = `${it.label} ${it.hint ?? ""} ${it.kind}`.toLowerCase();
        if (!hay.includes(q)) {
          // Cheap subsequence scoring as a fuzzy fallback.
          let i = 0;
          for (const ch of hay) {
            if (ch === q[i]) i++;
            if (i === q.length) break;
          }
          if (i < q.length) return null;
          return { item: it, score: 1000 - hay.length };
        }
        const exact = hay.indexOf(q);
        return { item: it, score: 10000 - exact };
      })
      .filter((x): x is { item: PaletteItem; score: number } => x !== null);
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, 30).map((x) => x.item);
  }, [items, query]);

  useEffect(() => {
    if (active >= filtered.length) setActive(0);
  }, [filtered.length, active]);

  if (!open) return null;

  const choose = (item: PaletteItem) => {
    onClose();
    if (item.run) {
      item.run();
    } else if (item.page) {
      onNavigate(item.page);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((i) => Math.min(filtered.length - 1, i + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => Math.max(0, i - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const it = filtered[active];
      if (it) choose(it);
    }
  };

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed", inset: 0, background: "rgba(40,28,12,0.32)",
        zIndex: 60, display: "flex", justifyContent: "center", alignItems: "flex-start",
        paddingTop: "12vh",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        onKeyDown={onKeyDown}
        style={{
          width: 560, maxWidth: "90vw", background: "#fdf6e9",
          border: "1px solid var(--line)", borderRadius: 6,
          boxShadow: "0 16px 40px rgba(0,0,0,0.18)",
          display: "flex", flexDirection: "column", maxHeight: "70vh",
        }}
      >
        <input
          ref={inputRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Jump to page, predictor, scenario, action…"
          style={{
            border: "none", outline: "none",
            background: "transparent", padding: "12px 16px",
            fontSize: 14, fontFamily: "inherit",
            borderBottom: "1px solid var(--line)", color: "var(--ink)",
          }}
        />
        <div style={{ overflowY: "auto", padding: 4 }}>
          {filtered.length === 0 ? (
            <div style={{ padding: 14, fontSize: 11, color: "var(--muted)" }}>
              no matches
            </div>
          ) : (
            filtered.map((it, idx) => (
              <div
                key={it.id}
                onMouseEnter={() => setActive(idx)}
                onClick={() => choose(it)}
                style={{
                  padding: "6px 10px",
                  borderRadius: 4,
                  background: idx === active ? "rgba(122, 78, 162, 0.12)" : "transparent",
                  cursor: "pointer",
                  display: "flex", alignItems: "center", gap: 10,
                  fontSize: 12,
                }}
              >
                <span
                  className="mono"
                  style={{
                    fontSize: 9, padding: "1px 6px",
                    border: "1px solid var(--line)", borderRadius: 3,
                    color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em",
                  }}
                >
                  {it.kind}
                </span>
                <span style={{ flex: 1 }}>{it.label}</span>
                {it.hint && (
                  <span className="mono" style={{ fontSize: 10, color: "var(--muted)" }}>
                    {it.hint}
                  </span>
                )}
              </div>
            ))
          )}
        </div>
        <div
          style={{
            borderTop: "1px solid var(--line)",
            padding: "6px 12px",
            fontSize: 9.5, color: "var(--muted)",
            display: "flex", gap: 14, fontFamily: "'JetBrains Mono', monospace",
            textTransform: "uppercase", letterSpacing: "0.08em",
          }}
        >
          <span>↑↓ navigate</span>
          <span>↵ open</span>
          <span>esc close</span>
        </div>
      </div>
    </div>
  );
}
