/**
 * ExplainNumber (Phase 20) — small reusable tooltip widget that lets
 * any numeric readout in the UI carry a short causal/statistical
 * explanation. Hover or focus the trailing "?" to reveal the body.
 *
 * Wire-up pattern:
 *   <Stat label="Spatial R²" value={0.74} />
 *   <ExplainNumber title="Spatial R²"
 *      body="Variance explained by the model on held-out spatial folds.
 *            Higher is better; values above 0.5 are considered strong
 *            for environmental health regression." />
 *
 * This is intentionally dependency-free (no portal, no animation lib)
 * so it can be sprinkled anywhere without bundle pressure.
 */

import { useEffect, useId, useRef, useState } from "react";

interface ExplainNumberProps {
  title: string;
  body: string;
  /** Optional "Read more" link or hint shown beneath the body. */
  source?: string;
  /** Optional inline display label. Defaults to "?". */
  glyph?: string;
  /** Color of the glyph. Defaults to var(--purple). */
  color?: string;
  align?: "left" | "right";
}

export default function ExplainNumber({
  title, body, source, glyph = "?", color = "var(--purple)", align = "left",
}: ExplainNumberProps) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) { if (e.key === "Escape") setOpen(false); }
    function onClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    window.addEventListener("keydown", onKey);
    window.addEventListener("mousedown", onClick);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("mousedown", onClick);
    };
  }, [open]);

  return (
    <span ref={ref} style={{ position: "relative", display: "inline-flex", alignItems: "center", gap: 4 }}>
      <button
        type="button"
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        onMouseEnter={() => setOpen(true)}
        onFocus={() => setOpen(true)}
        style={{
          width: 14, height: 14, borderRadius: 7, padding: 0,
          border: `1px solid ${color}`, background: "transparent", color,
          fontSize: 9, lineHeight: "12px", cursor: "help",
          display: "inline-flex", alignItems: "center", justifyContent: "center",
          fontFamily: "inherit", fontWeight: 700,
        }}
      >{glyph}</button>
      {open && (
        <div
          id={id} role="tooltip"
          onMouseLeave={() => setOpen(false)}
          style={{
            position: "absolute", top: "calc(100% + 4px)", [align]: 0,
            zIndex: 20, minWidth: 220, maxWidth: 320,
            background: "#fff", border: "1px solid var(--line)",
            borderRadius: 4, padding: 10, boxShadow: "0 6px 18px rgba(0,0,0,0.08)",
            fontSize: 11.5, lineHeight: 1.5, color: "var(--ink-2)",
            textTransform: "none", letterSpacing: "normal", fontWeight: 400,
          }}
        >
          <div style={{ fontWeight: 700, color: "var(--ink)", marginBottom: 4 }}>{title}</div>
          <div>{body}</div>
          {source && (
            <div style={{ marginTop: 6, fontSize: 10, color: "var(--muted)" }}>{source}</div>
          )}
        </div>
      )}
    </span>
  );
}
