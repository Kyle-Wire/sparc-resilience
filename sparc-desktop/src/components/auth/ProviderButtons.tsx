/**
 * Provider sign-in buttons (Google, GitHub, Microsoft).
 *
 * Calls AuthProvider.signInWithProvider, surfaces errors via the
 * notification system. Disabled while the OAuth flow is in flight.
 */
import { useState } from "react";
import { Btn } from "@/components/ui/DesignSystem";
import { useAuth } from "@/lib/auth/AuthProvider";
import { useNotification } from "@/hooks/useNotifications";

export default function ProviderButtons() {
  const { signInWithProvider } = useAuth();
  const { notify } = useNotification();
  const [busyProvider, setBusyProvider] = useState<string | null>(null);

  const click = async (p: "google" | "github" | "azure") => {
    setBusyProvider(p);
    try {
      await signInWithProvider(p);
    } catch (e) {
      notify("error", e instanceof Error ? e.message : "Sign-in failed");
    } finally {
      setBusyProvider(null);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <Btn primary onClick={() => click("google")} disabled={busyProvider !== null}>
        {busyProvider === "google" ? "Opening browser…" : "Continue with Google"}
      </Btn>
      <Btn onClick={() => click("github")} disabled={busyProvider !== null}>
        {busyProvider === "github" ? "Opening browser…" : "Continue with GitHub"}
      </Btn>
      <Btn onClick={() => click("azure")} disabled={busyProvider !== null}>
        {busyProvider === "azure" ? "Opening browser…" : "Continue with Microsoft"}
      </Btn>
    </div>
  );
}
