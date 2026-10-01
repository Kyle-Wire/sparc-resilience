import { useEffect, useId, useRef, type ReactNode } from "react";
import { trapTab } from "./focus";
import { IconButton } from "./IconButton";

export type DrawerProps = {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  side?: "left" | "right";
  /** "overlay" floats over the page and traps focus; "inline" sits in the layout (analysis tools). */
  mode?: "overlay" | "inline";
  children?: ReactNode;
  actions?: ReactNode;
};

/** Side panel. Escape closes the overlay variant. */
export function Drawer({ open, onClose, title, side = "right", mode = "overlay", children, actions }: DrawerProps) {
  const ref = useRef<HTMLElement>(null);
  const titleId = useId();
  useEffect(() => {
    if (open && mode === "overlay") ref.current?.focus();
  }, [open, mode]);
  if (!open) return null;
  const overlay = mode === "overlay";
  return (
    <aside
      ref={ref}
      className={["drawer", overlay ? "" : "inline"].filter(Boolean).join(" ")}
      data-side={side}
      role={overlay ? "dialog" : "complementary"}
      aria-modal={overlay ? true : undefined}
      aria-labelledby={titleId}
      tabIndex={-1}
      onKeyDown={(e) => {
        if (!overlay) return;
        if (e.key === "Escape") {
          e.stopPropagation();
          onClose();
        }
        trapTab(e, ref.current);
      }}
    >
      <header>
        <h3 id={titleId}>{title}</h3>
        <div className="row">
          {actions}
          <IconButton icon="x" label="Close panel" onClick={onClose} />
        </div>
      </header>
      <div className="drawer-body">{children}</div>
    </aside>
  );
}
