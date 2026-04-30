"use client";

import * as React from "react";
import { useSearchParams } from "next/navigation";
import { createSupabaseBrowserClient } from "@/lib/supabase/client";
import { SITE_URL } from "@/lib/brand";

type Provider = "google" | "azure";

export function LoginForm({ mode }: { mode: "signin" | "signup" }) {
  const params = useSearchParams();
  const next = params.get("next") ?? "/account";
  const desktop = params.get("desktop") === "1";
  const desktopState = params.get("state") ?? "";

  const [email, setEmail] = React.useState("");
  const [state, setState] = React.useState<"idle" | "submitting" | "sent" | "error">("idle");
  const [error, setError] = React.useState("");

  const callback = `${SITE_URL}/auth/callback?next=${encodeURIComponent(next)}${desktop ? `&desktop=1&state=${encodeURIComponent(desktopState)}` : ""}`;

  async function magicLink(e: React.FormEvent) {
    e.preventDefault();
    setState("submitting");
    setError("");
    try {
      const supabase = createSupabaseBrowserClient();
      const { error } = await supabase.auth.signInWithOtp({
        email,
        options: { emailRedirectTo: callback, shouldCreateUser: mode === "signup" },
      });
      if (error) throw error;
      setState("sent");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setState("error");
    }
  }

  async function oauth(provider: Provider) {
    setState("submitting");
    setError("");
    try {
      const supabase = createSupabaseBrowserClient();
      const { error } = await supabase.auth.signInWithOAuth({
        provider,
        options: { redirectTo: callback },
      });
      if (error) throw error;
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setState("error");
    }
  }

  if (state === "sent") {
    return (
      <div className="card" style={{ padding: 28, textAlign: "center" }}>
        <div className="kicker" style={{ justifyContent: "center" }}>magic link sent</div>
        <h3 style={{ fontSize: 22, fontWeight: 700, margin: "12px 0 8px" }}>Check your inbox.</h3>
        <p style={{ color: "var(--ink-2)", fontSize: 14, lineHeight: 1.55, margin: 0 }}>
          We sent a sign-in link to <strong>{email}</strong>. It&apos;s good for 15 minutes.
        </p>
      </div>
    );
  }

  return (
    <div className="card" style={{ padding: 28 }}>
      <div className="kicker">{mode === "signup" ? "create your account" : "sign in"}</div>
      <h3 style={{ fontSize: 20, fontWeight: 700, margin: "10px 0 16px" }}>Magic link · no passwords</h3>
      <form onSubmit={magicLink}>
        <label className="label">Email</label>
        <input className="input" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@university.edu" autoComplete="email" />
        <button type="submit" className="btn btn-accent" style={{ width: "100%", justifyContent: "center", marginTop: 14 }} disabled={state === "submitting"}>
          {state === "submitting" ? "Sending…" : mode === "signup" ? "Send sign-up link" : "Send sign-in link"} <span className="arrow">→</span>
        </button>
      </form>

      <div style={{ display: "flex", alignItems: "center", gap: 10, margin: "18px 0", color: "var(--muted)", fontSize: 11, fontFamily: "var(--font-mono)", letterSpacing: "0.12em", textTransform: "uppercase" }}>
        <span style={{ flex: 1, height: 1, background: "var(--line)" }} /> or <span style={{ flex: 1, height: 1, background: "var(--line)" }} />
      </div>

      <div style={{ display: "grid", gap: 8 }}>
        <button type="button" className="btn" onClick={() => oauth("google")} style={{ width: "100%", justifyContent: "center" }} disabled={state === "submitting"}>
          Continue with Google
        </button>
        <button type="button" className="btn" onClick={() => oauth("azure")} style={{ width: "100%", justifyContent: "center" }} disabled={state === "submitting"}>
          Continue with Microsoft
        </button>
      </div>

      {error && <p style={{ color: "var(--crimson)", marginTop: 12, fontSize: 13 }}>{error}</p>}
      {desktop && (
        <p className="mono" style={{ marginTop: 14, fontSize: 11, color: "var(--muted)" }}>
          ▸ desktop sign-in · we&apos;ll redirect back to the SPARC app when you&apos;re authenticated
        </p>
      )}
    </div>
  );
}
