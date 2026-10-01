import { useEffect } from "react";
import { navigate } from "../../router";
import { useUi, type Toast as ToastT } from "../../stores/ui";
import { ActionButton } from "./EmptyState";
import { Icon, type IconName } from "./Icon";

const ICON: Record<ToastT["kind"], IconName> = { info: "info", success: "check", warning: "alert", error: "x" };

function ToastItem({ t }: { t: ToastT }) {
  const dismiss = useUi((s) => s.dismissToast);
  useEffect(() => {
    if (!t.timeout) return;
    const h = setTimeout(() => dismiss(t.id), t.timeout);
    return () => clearTimeout(h);
  }, [t.id, t.timeout, dismiss]);
  return (
    <div className="toast" data-kind={t.kind} role={t.kind === "error" ? "alert" : undefined}>
      <Icon name={ICON[t.kind]} />
      <div className="stack" style={{ gap: 4 }}>
        <span className="title">{t.title}</span>
        {t.body ? <span className="ink-2">{t.body}</span> : null}
        <div className="row">
          {t.href ? (
            <a
              href={t.href}
              onClick={(e) => {
                if (e.button !== 0 || e.metaKey || e.ctrlKey) return;
                e.preventDefault();
                dismiss(t.id);
                navigate(t.href!);
              }}
            >
              {t.linkLabel ?? "Open"}
            </a>
          ) : null}
          {t.action ? <ActionButton action={t.action} size="small" onDone={() => dismiss(t.id)} /> : null}
        </div>
      </div>
      <button type="button" className="btn icon small ghost" aria-label="Dismiss notification" onClick={() => dismiss(t.id)}>
        <Icon name="x" />
      </button>
    </div>
  );
}

/** Toast stack (bottom right). Non-error toasts auto-dismiss; errors stay until closed. */
export function Toasts() {
  const toasts = useUi((s) => s.toasts);
  return (
    <div className="toasts" role="region" aria-label="Notifications" aria-live="polite">
      {toasts.map((t) => (
        <ToastItem key={t.id} t={t} />
      ))}
    </div>
  );
}
