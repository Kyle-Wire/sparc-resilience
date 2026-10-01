import { useId, useState, type ReactNode } from "react";

/**
 * Hover/focus tooltip. The trigger gets aria-describedby, so the text is read on focus too.
 * Keep the content short; use a Dialog or Drawer for anything interactive.
 */
export function Tooltip({ content, children, className }: { content: ReactNode; children: ReactNode; className?: string }) {
  const id = useId();
  const [open, setOpen] = useState(false);
  return (
    <span
      className={["tip-anchor", className ?? ""].filter(Boolean).join(" ")}
      aria-describedby={id}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
      onKeyDown={(e) => {
        if (e.key === "Escape") setOpen(false);
      }}
    >
      {children}
      <span role="tooltip" id={id} className="tip-bubble" hidden={!open}>
        {content}
      </span>
    </span>
  );
}
