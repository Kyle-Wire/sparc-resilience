import { useEffect, useId, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { focusables, trapTab } from "./focus";
import { IconButton } from "./IconButton";

export type DialogProps = {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  children?: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
  /** Closing by Escape/backdrop is blocked while true (e.g. a request in flight). */
  busy?: boolean;
  className?: string;
  describedBy?: string;
};

/**
 * Modal dialog: portal, aria-modal, labelled by its title, focus trapped and restored,
 * Escape and backdrop close.
 */
export function Dialog({ open, onClose, title, children, footer, wide, busy, className, describedBy }: DialogProps) {
  const ref = useRef<HTMLDivElement>(null);
  const titleId = useId();
  const restore = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;
    restore.current = document.activeElement as HTMLElement | null;
    const t = setTimeout(() => {
      const root = ref.current;
      if (!root) return;
      const auto = root.querySelector<HTMLElement>("[data-autofocus]") ?? focusables(root).find((el) => !el.classList.contains("dialog-close")) ?? root;
      auto.focus();
    }, 0);
    return () => {
      clearTimeout(t);
      restore.current?.focus?.();
    };
  }, [open]);

  if (!open) return null;
  return createPortal(
    <div
      className="overlay"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget && !busy) onClose();
      }}
    >
      <div
        ref={ref}
        className={["dialog", wide ? "wide" : "", className ?? ""].filter(Boolean).join(" ")}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={describedBy}
        tabIndex={-1}
        onKeyDown={(e) => {
          if (e.key === "Escape" && !busy) {
            e.stopPropagation();
            onClose();
          }
          trapTab(e, ref.current);
        }}
      >
        <header>
          <h2 id={titleId}>{title}</h2>
          <IconButton icon="x" label="Close" className="dialog-close" onClick={onClose} disabled={busy} />
        </header>
        {children}
        {footer ? <footer>{footer}</footer> : null}
      </div>
    </div>,
    document.body,
  );
}

/** Yes/no confirmation with a guarded primary action (deletes, force stop). */
export function ConfirmDialog({
  open,
  onClose,
  onConfirm,
  title,
  children,
  confirmLabel = "Confirm",
  danger,
  busy,
}: {
  open: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: ReactNode;
  children?: ReactNode;
  confirmLabel?: string;
  danger?: boolean;
  busy?: boolean;
}) {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      title={title}
      busy={busy}
      footer={
        <>
          <button type="button" className="btn" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button type="button" className={`btn ${danger ? "danger" : "primary"}`} onClick={onConfirm} disabled={busy} data-autofocus>
            {busy ? <span className="spinner" aria-hidden="true" /> : null}
            {confirmLabel}
          </button>
        </>
      }
    >
      <div className="prose">{children}</div>
    </Dialog>
  );
}
