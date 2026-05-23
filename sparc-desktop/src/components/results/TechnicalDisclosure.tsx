/**
 * TechnicalDisclosure — collapsible "What this means technically" section.
 *
 * Children are always rendered in the DOM (never conditionally unmounted)
 * so that Canvas/WebGL contexts initialise on mount. Visibility is toggled
 * via the HTML `hidden` attribute so that `aria-hidden` semantics and
 * jest-dom `toBeVisible()` checks both work correctly.
 */
import { useState } from "react";

interface TechnicalDisclosureProps {
  /** Toggle button label. Defaults to "What this means technically". */
  label?: string;
  children: React.ReactNode;
}

export default function TechnicalDisclosure({
  label = "What this means technically",
  children,
}: TechnicalDisclosureProps) {
  const [open, setOpen] = useState(false);

  return (
    <section
      style={{
        borderTop: "1px solid var(--line, #c9c2b3)",
        marginTop: 32,
      }}
    >
      <button
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        style={{
          background: "none",
          border: "none",
          cursor: "pointer",
          padding: "12px 0",
          color: "var(--muted, #6e6358)",
          fontFamily: "var(--font-mono, 'JetBrains Mono', monospace)",
          fontSize: 12,
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <span
          style={{
            display: "inline-block",
            transform: open ? "rotate(90deg)" : "rotate(0deg)",
            transition: "transform 0.2s",
            fontSize: 10,
          }}
        >
          ▶
        </span>
        {label}
      </button>

      {/* Always in DOM — toggled via `hidden` so Canvas contexts initialise on mount */}
      <div hidden={!open}>{children}</div>
    </section>
  );
}
