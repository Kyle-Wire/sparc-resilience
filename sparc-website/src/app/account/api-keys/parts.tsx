"use client";

import * as React from "react";
import type { ApiKeyRow } from "./page";

const PROVIDERS = ["anthropic", "openai", "mistral", "google", "groq", "openrouter"] as const;

export function ApiKeysClient({ initial }: { initial: ApiKeyRow[] }) {
  const [keys, setKeys] = React.useState(initial);
  const [provider, setProvider] = React.useState<typeof PROVIDERS[number]>("anthropic");
  const [label, setLabel] = React.useState("default");
  const [secret, setSecret] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [err, setErr] = React.useState("");

  async function add(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true); setErr("");
    try {
      const res = await fetch("/api/account/api-keys", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ provider, label, secret }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error ?? "could not save");
      setKeys([data.key, ...keys]);
      setSecret(""); setLabel("default");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally { setBusy(false); }
  }

  async function revoke(id: string) {
    if (!confirm("Revoke this key? Any desktop session using it will fail next time it refreshes.")) return;
    const res = await fetch(`/api/account/api-keys?id=${encodeURIComponent(id)}`, { method: "DELETE" });
    if (res.ok) setKeys(keys.filter((k) => k.id !== id));
  }

  return (
    <>
      <form className="card" style={{ padding: 22 }} onSubmit={add}>
        <div className="kicker">add a key</div>
        <div style={{ display: "grid", gridTemplateColumns: "160px 1fr 1fr auto", gap: 8, marginTop: 12, alignItems: "flex-end" }}>
          <div>
            <label className="label">Provider</label>
            <select className="input" value={provider} onChange={(e) => setProvider(e.target.value as typeof PROVIDERS[number])}>
              {PROVIDERS.map((p) => <option key={p}>{p}</option>)}
            </select>
          </div>
          <div>
            <label className="label">Label</label>
            <input className="input" value={label} onChange={(e) => setLabel(e.target.value)} required />
          </div>
          <div>
            <label className="label">API key</label>
            <input className="input mono" type="password" value={secret} onChange={(e) => setSecret(e.target.value)} required placeholder="sk-…" autoComplete="off" />
          </div>
          <button type="submit" className="btn btn-accent" disabled={busy}>{busy ? "Saving…" : "Save key"}</button>
        </div>
        {err && <p style={{ color: "var(--crimson)", marginTop: 10, fontSize: 13 }}>{err}</p>}
      </form>

      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
          <thead><tr style={{ background: "var(--paper-2)", borderBottom: "1px solid var(--line)" }}>
            {["provider", "label", "key", "added", ""].map((h) => (
              <th key={h} style={{ textAlign: "left", padding: "12px 18px", fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--muted)" }}>{h}</th>
            ))}
          </tr></thead>
          <tbody>
            {keys.length === 0 ? (
              <tr><td colSpan={5} style={{ padding: 22, textAlign: "center", color: "var(--muted)" }}>No keys yet.</td></tr>
            ) : keys.map((k) => (
              <tr key={k.id} style={{ borderBottom: "1px dotted var(--line)" }}>
                <td style={{ padding: "12px 18px" }}><span className="pill mono">{k.provider}</span></td>
                <td style={{ padding: "12px 18px" }}>{k.label}</td>
                <td style={{ padding: "12px 18px" }} className="mono">••••{k.key_last4}</td>
                <td style={{ padding: "12px 18px" }} className="mono">{new Date(k.created_at).toLocaleDateString()}</td>
                <td style={{ padding: "12px 18px", textAlign: "right" }}>
                  <button className="btn btn-sm" onClick={() => revoke(k.id)}>Revoke</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
