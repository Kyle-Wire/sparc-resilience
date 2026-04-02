import { useNotification } from "@/hooks/useNotifications";

const styles: Record<string, { bg: string; border: string; text: string; icon: string }> = {
  success: { bg: "bg-green-50", border: "border-green-300", text: "text-green-800", icon: "✓" },
  error: { bg: "bg-red-50", border: "border-sparc-crimson/30", text: "text-sparc-crimson", icon: "✕" },
  warning: { bg: "bg-amber-50", border: "border-sparc-amber/40", text: "text-amber-800", icon: "!" },
  info: { bg: "bg-sparc-gray-100", border: "border-sparc-gray-300", text: "text-sparc-gray-800", icon: "i" },
};

export default function NotificationBanner() {
  const { notifications, dismiss } = useNotification();

  if (notifications.length === 0) return null;

  return (
    <div className="pointer-events-none fixed inset-x-0 top-0 z-50 flex flex-col items-center gap-2 px-4 pt-3">
      {notifications.map((n) => {
        const s = styles[n.type] ?? styles.info;
        return (
          <div
            key={n.id}
            className={`pointer-events-auto flex w-full max-w-lg items-center gap-3 rounded-lg border px-4 py-3 shadow-md backdrop-blur-sm ${s.bg} ${s.border} animate-in slide-in-from-top duration-200`}
          >
            <span className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-xs font-bold ${s.text}`}>
              {s.icon}
            </span>
            <span className={`flex-1 text-sm ${s.text}`}>{n.message}</span>
            <button
              onClick={() => dismiss(n.id)}
              className={`shrink-0 text-sm opacity-60 hover:opacity-100 ${s.text}`}
            >
              ✕
            </button>
          </div>
        );
      })}
    </div>
  );
}
