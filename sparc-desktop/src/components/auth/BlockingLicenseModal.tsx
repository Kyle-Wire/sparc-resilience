/**
 * Full-screen blocker shown when verifyLicense() returns
 * `grace_expired`. The user can navigate to Settings → License &
 * Account (still reachable) or sign out.
 */
import { Btn, Card } from "@/components/ui/DesignSystem";
import { useAuth } from "@/lib/auth/AuthProvider";

interface Props {
  onGoToSettings(): void;
}

export default function BlockingLicenseModal({ onGoToSettings }: Props) {
  const { refresh, busy, signOut, daysRemaining } = useAuth();
  const overdueDays = Math.abs(Math.min(0, daysRemaining));

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(20, 16, 18, 0.55)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 9999,
      }}
    >
      <div style={{ width: 480, maxWidth: "90vw" }}>
        <Card title="License verification required" subtitle="offline grace expired">
          <p style={{ fontSize: 13, lineHeight: 1.55, margin: "0 0 12px" }}>
            Your SPARC license hasn't been revalidated in over 15 days
            {overdueDays > 0 ? ` (${overdueDays} day${overdueDays === 1 ? "" : "s"} overdue)` : ""}.
            Reconnect to the internet and refresh, or open License & Account
            settings to sign in again.
          </p>
          <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", flexWrap: "wrap" }}>
            <Btn onClick={() => void signOut()}>Sign out</Btn>
            <Btn onClick={onGoToSettings}>Open Settings</Btn>
            <Btn primary onClick={() => void refresh({ force: true })} disabled={busy}>
              {busy ? "Checking…" : "Retry now"}
            </Btn>
          </div>
        </Card>
      </div>
    </div>
  );
}
