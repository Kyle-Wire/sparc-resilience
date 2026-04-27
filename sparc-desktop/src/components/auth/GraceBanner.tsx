/**
 * Non-blocking yellow banner shown when verifyLicense() returns
 * `grace_warning` (cache valid but server unreachable).
 *
 * Sits between the Topbar and main content; dismisses on the next
 * successful refresh.
 */
import { useAuth } from "@/lib/auth/AuthProvider";

export default function GraceBanner() {
  const { state, daysRemaining, refresh, busy } = useAuth();
  if (state !== "grace_warning") return null;

  const days = Math.max(0, daysRemaining);
  return (
    <div
      role="status"
      style={{
        background: "#fff7e6",
        borderBottom: "1px solid #f0c36d",
        color: "#7a5200",
        padding: "8px 16px",
        fontSize: 12.5,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 12,
      }}
    >
      <div>
        <strong>Working offline.</strong>{" "}
        License cached — {days} day{days === 1 ? "" : "s"} remaining in offline grace.
        Reconnect to revalidate.
      </div>
      <button
        onClick={() => void refresh({ force: true })}
        disabled={busy}
        style={{
          background: "transparent",
          border: "1px solid #c08a30",
          borderRadius: 4,
          color: "#7a5200",
          padding: "3px 10px",
          fontSize: 11,
          cursor: busy ? "wait" : "pointer",
        }}
      >
        {busy ? "Checking…" : "Retry now"}
      </button>
    </div>
  );
}
