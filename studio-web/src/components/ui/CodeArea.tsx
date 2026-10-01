import { useId, useMemo, useRef, type KeyboardEvent } from "react";

export type LineIssue = { line: number; level: "error" | "warn" | "info"; message: string };

export type CodeAreaProps = {
  value: string;
  onChange?: (v: string) => void;
  label: string;
  issues?: LineIssue[];
  readOnly?: boolean;
  rows?: number;
  /** Line to scroll into view and select (e.g. an issue clicked in a list). */
  spellCheck?: boolean;
};

/**
 * Monospace editor (a textarea) with line numbers and an issue gutter. Tab inserts two
 * spaces; Escape then Tab leaves the field (keyboard users are never trapped).
 */
export function CodeArea({ value, onChange, label, issues = [], readOnly, rows = 24, spellCheck = false }: CodeAreaProps) {
  const id = useId();
  const gutter = useRef<HTMLDivElement>(null);
  const escaped = useRef(false);
  const lines = useMemo(() => Math.max(1, value.split("\n").length), [value]);
  const byLine = useMemo(() => {
    const m = new Map<number, LineIssue[]>();
    for (const i of issues) m.set(i.line, [...(m.get(i.line) ?? []), i]);
    return m;
  }, [issues]);
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Escape") {
      escaped.current = true;
      return;
    }
    if (e.key === "Tab" && !e.shiftKey && !escaped.current && !readOnly && onChange) {
      e.preventDefault();
      const ta = e.currentTarget;
      const { selectionStart: s, selectionEnd: en } = ta;
      const next = value.slice(0, s) + "  " + value.slice(en);
      onChange(next);
      requestAnimationFrame(() => ta.setSelectionRange(s + 2, s + 2));
      return;
    }
    escaped.current = false;
  };
  return (
    <div className="code-area">
      <div className="gutter" ref={gutter} aria-hidden="true">
        {Array.from({ length: lines }, (_, i) => {
          const iss = byLine.get(i + 1);
          const lvl = iss?.some((x) => x.level === "error") ? "error" : iss?.some((x) => x.level === "warn") ? "warn" : iss ? "info" : undefined;
          return (
            <div key={i} data-level={lvl} title={iss?.map((x) => x.message).join("\n")}>
              {lvl === "error" ? "✕" : lvl === "warn" ? "!" : ""}
              {i + 1}
            </div>
          );
        })}
      </div>
      <textarea
        id={id}
        aria-label={label}
        value={value}
        readOnly={readOnly}
        rows={rows}
        spellCheck={spellCheck}
        wrap="off"
        onChange={(e) => onChange?.(e.target.value)}
        onKeyDown={onKey}
        onScroll={(e) => {
          if (gutter.current) gutter.current.scrollTop = e.currentTarget.scrollTop;
        }}
      />
      {issues.length ? (
        <ul className="sr-only">
          {issues.map((i, k) => (
            <li key={k}>
              Line {i.line}: {i.level} {i.message}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
