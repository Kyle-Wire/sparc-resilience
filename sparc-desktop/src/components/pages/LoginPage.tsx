/**
 * Standalone login surface — replaces the entire app shell when no
 * valid session exists. Three tabs:
 *
 *   1. Sign in       — provider buttons + email/password
 *   2. Create account — SPARC email/password sign-up
 *   3. License key    — paste a Lemon Squeezy key
 */
import { useState } from "react";
import { Btn, Card } from "@/components/ui/DesignSystem";
import { useAuth } from "@/lib/auth/AuthProvider";
import { useNotification } from "@/hooks/useNotifications";
import ProviderButtons from "@/components/auth/ProviderButtons";

type Tab = "signin" | "signup" | "key";

export default function LoginPage() {
  const [tab, setTab] = useState<Tab>("signin");

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
        <div style={{ textAlign: "center", marginBottom: 24 }}>
          <h1
            style={{
              fontSize: 22,
              fontWeight: 700,
              margin: 0,
              letterSpacing: "-0.01em",
            }}
          >
            SPARC Labs
          </h1>
          <div
            className="mono"
            style={{ fontSize: 10, color: "var(--muted)", marginTop: 4, letterSpacing: "0.1em" }}
          >
            SIGN IN TO CONTINUE
          </div>
        </div>

        <Card>
          <div style={{ display: "flex", gap: 4, marginBottom: 14, borderBottom: "1px solid var(--line)" }}>
            <TabBtn label="Sign in" active={tab === "signin"} onClick={() => setTab("signin")} />
            <TabBtn label="Create account" active={tab === "signup"} onClick={() => setTab("signup")} />
            <TabBtn label="License key" active={tab === "key"} onClick={() => setTab("key")} />
          </div>

          {tab === "signin" && <SignInPanel />}
          {tab === "signup" && <SignUpPanel onDone={() => setTab("signin")} />}
          {tab === "key" && <LicenseKeyPanel />}
        </Card>

        <div
          className="mono"
          style={{
            textAlign: "center",
            color: "var(--muted)",
            fontSize: 10,
            marginTop: 16,
            letterSpacing: "0.05em",
          }}
        >
          By continuing you agree to the SPARC Labs terms of service.
        </div>
      </div>
    </div>
  );
}

// ─── Subcomponents ──────────────────────────────────────────────────

function TabBtn({ label, active, onClick }: { label: string; active: boolean; onClick(): void }) {
  return (
    <button
      onClick={onClick}
      style={{
        background: "transparent",
        border: "none",
        borderBottom: `2px solid ${active ? "var(--ink)" : "transparent"}`,
        color: active ? "var(--ink)" : "var(--muted)",
        padding: "8px 12px",
        fontSize: 12,
        fontWeight: active ? 600 : 500,
        cursor: "pointer",
      }}
    >
      {label}
    </button>
  );
}

function SignInPanel() {
  const { signInWithEmail, resetPassword } = useAuth();
  const { notify } = useNotification();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    if (!email || !password) return;
    setBusy(true);
    try {
      await signInWithEmail(email, password);
    } catch (e) {
      notify("error", e instanceof Error ? e.message : "Sign-in failed");
    } finally {
      setBusy(false);
    }
  };

  const reset = async () => {
    if (!email) {
      notify("info", "Enter your email first");
      return;
    }
    try {
      await resetPassword(email);
      notify("success", "Password reset email sent");
    } catch (e) {
      notify("error", e instanceof Error ? e.message : "Reset failed");
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <ProviderButtons />

      <Divider label="or" />

      <FieldLabel>Email</FieldLabel>
      <TextInput
        type="email"
        value={email}
        onChange={setEmail}
        placeholder="you@example.com"
        autoComplete="email"
      />
      <FieldLabel>Password</FieldLabel>
      <TextInput
        type="password"
        value={password}
        onChange={setPassword}
        placeholder="••••••••"
        autoComplete="current-password"
        onEnter={submit}
      />

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <button
          type="button"
          onClick={reset}
          style={{
            background: "transparent",
            border: "none",
            color: "var(--muted)",
            fontSize: 11,
            cursor: "pointer",
            padding: 0,
          }}
        >
          Forgot password?
        </button>
        <Btn primary onClick={submit} disabled={busy || !email || !password}>
          {busy ? "Signing in…" : "Sign in"}
        </Btn>
      </div>
    </div>
  );
}

function SignUpPanel({ onDone }: { onDone(): void }) {
  const { signUpWithEmail } = useAuth();
  const { notify } = useNotification();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);

  const submit = async () => {
    if (!email || !password) return;
    if (password.length < 8) {
      notify("error", "Password must be at least 8 characters");
      return;
    }
    setBusy(true);
    try {
      await signUpWithEmail(email, password);
      setDone(true);
    } catch (e) {
      notify("error", e instanceof Error ? e.message : "Sign-up failed");
    } finally {
      setBusy(false);
    }
  };

  if (done) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <p style={{ fontSize: 13, lineHeight: 1.5, margin: 0 }}>
          We just sent a confirmation email to <strong>{email}</strong>.
          Click the link, then return here and sign in.
        </p>
        <Btn onClick={onDone}>Back to sign in</Btn>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <p style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.5, margin: 0 }}>
        Create a free SPARC account. You'll get email-verified access to the
        Free tier; upgrade any time from Settings.
      </p>

      <FieldLabel>Email</FieldLabel>
      <TextInput
        type="email"
        value={email}
        onChange={setEmail}
        placeholder="you@example.com"
        autoComplete="email"
      />
      <FieldLabel>Password (min 8 chars)</FieldLabel>
      <TextInput
        type="password"
        value={password}
        onChange={setPassword}
        placeholder="••••••••"
        autoComplete="new-password"
        onEnter={submit}
      />

      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <Btn primary onClick={submit} disabled={busy || !email || !password}>
          {busy ? "Creating…" : "Create account"}
        </Btn>
      </div>
    </div>
  );
}

function LicenseKeyPanel() {
  const { redeemKey } = useAuth();
  const { notify } = useNotification();
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    if (!key.trim()) return;
    setBusy(true);
    try {
      await redeemKey(key.trim());
      notify("success", "License activated");
    } catch (e) {
      notify("error", e instanceof Error ? e.message : "License key invalid");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <p style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.5, margin: 0 }}>
        Already purchased a SPARC license? Paste your key below to activate
        this device. No account is required, but creating one lets you manage
        subscriptions across machines.
      </p>

      <FieldLabel>License key</FieldLabel>
      <TextInput
        value={key}
        onChange={setKey}
        placeholder="XXXX-XXXX-XXXX-XXXX"
        onEnter={submit}
        mono
      />

      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <Btn primary onClick={submit} disabled={busy || !key.trim()}>
          {busy ? "Activating…" : "Activate license"}
        </Btn>
      </div>
    </div>
  );
}

// ─── Tiny shared form atoms ─────────────────────────────────────────

function FieldLabel({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="mono"
      style={{
        fontSize: 9.5,
        color: "var(--muted)",
        textTransform: "uppercase",
        letterSpacing: "0.1em",
      }}
    >
      {children}
    </div>
  );
}

function TextInput({
  value,
  onChange,
  type = "text",
  placeholder,
  autoComplete,
  onEnter,
  mono,
}: {
  value: string;
  onChange(v: string): void;
  type?: string;
  placeholder?: string;
  autoComplete?: string;
  onEnter?(): void;
  mono?: boolean;
}) {
  return (
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      autoComplete={autoComplete}
      onKeyDown={(e) => {
        if (e.key === "Enter" && onEnter) onEnter();
      }}
      style={{
        border: "1px solid var(--line)",
        borderRadius: 4,
        padding: "8px 10px",
        fontSize: 13,
        fontFamily: mono ? "'JetBrains Mono', monospace" : "inherit",
        background: "#fff",
        width: "100%",
        boxSizing: "border-box",
      }}
    />
  );
}

function Divider({ label }: { label: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--muted)" }}>
      <div style={{ flex: 1, height: 1, background: "var(--line)" }} />
      <span className="mono" style={{ fontSize: 10, letterSpacing: "0.1em" }}>
        {label.toUpperCase()}
      </span>
      <div style={{ flex: 1, height: 1, background: "var(--line)" }} />
    </div>
  );
}
