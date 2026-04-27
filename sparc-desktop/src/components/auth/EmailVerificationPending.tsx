/**
 * Shown when verifyLicense() returns `email_unverified`. The user is
 * authenticated but Supabase says their email isn't confirmed yet.
 */
import { Btn, Card } from "@/components/ui/DesignSystem";
import { useAuth } from "@/lib/auth/AuthProvider";

export default function EmailVerificationPending() {
  const { email, refresh, signOut, busy } = useAuth();

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "var(--paper, #faf7f2)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 32,
      }}
    >
      <div style={{ width: 420, maxWidth: "100%" }}>
        <Card title="Confirm your email" subtitle="one more step">
          <p style={{ fontSize: 13, lineHeight: 1.55, margin: "0 0 14px" }}>
            We sent a confirmation link to{" "}
            <strong>{email ?? "your email"}</strong>. Click the link, then
            return here and refresh to continue.
          </p>
          <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
            <Btn onClick={() => void signOut()}>Sign out</Btn>
            <Btn primary onClick={() => void refresh({ force: true })} disabled={busy}>
              {busy ? "Checking…" : "I've confirmed"}
            </Btn>
          </div>
        </Card>
      </div>
    </div>
  );
}
