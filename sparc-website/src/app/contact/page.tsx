"use client";

import * as React from "react";
import { Nav, Footer } from "@/components/shell/Shell";
import { PageHero, Section } from "@/components/marketing/page-shell";
import { COMPANY, COMPANY_LEGAL, EMAIL_PRIMARY, OFFICE_FULL } from "@/lib/brand";

type Form = {
  name: string; email: string; org: string; domain: string; crs: string; reason: string;
};

const DOMAINS = ["uhi", "coastal", "drought", "geotech", "groundwater", "noise", "seismic", "stormwater", "water", "wildfire", "air", "forcesmip", "blank"];

function ContactForm() {
  const [form, setForm] = React.useState<Form>({ name: "", email: "", org: "", domain: "uhi", crs: "EPSG:32619", reason: "" });
  const [state, setState] = React.useState<"idle" | "submitting" | "ok" | "error">("idle");
  const [ticket, setTicket] = React.useState<string>("");
  const [error, setError] = React.useState<string>("");

  const update = <K extends keyof Form>(k: K, v: Form[K]) => setForm({ ...form, [k]: v });

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setState("submitting");
    setError("");
    try {
      const res = await fetch("/api/waitlist", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error ?? "Something went wrong.");
      setTicket(data.ticket ?? "");
      setState("ok");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setState("error");
    }
  }

  if (state === "ok") {
    return (
      <div className="card" style={{ padding: 32, textAlign: "center" }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "6px 12px", borderRadius: 9999, background: "var(--purple)", color: "#fff", fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 16 }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#fff" }} /> request received
        </div>
        <h3 style={{ fontSize: 28, fontWeight: 800, letterSpacing: "-0.02em", margin: "0 0 12px" }}>Welcome to the waitlist.</h3>
        <p style={{ color: "var(--ink-2)", fontSize: 14, lineHeight: 1.6, maxWidth: 480, margin: "0 auto" }}>
          We&apos;ll send a build with the <strong className="mono">{form.domain.toUpperCase()}</strong> template pre-loaded for <strong className="mono">{form.crs}</strong> within 5 business days. Watch your inbox at <strong>{form.email}</strong>.
        </p>
        {ticket && (
          <div className="mono" style={{ marginTop: 18, fontSize: 11, color: "var(--muted)", letterSpacing: "0.1em" }}>
            ▸ ticket · #{ticket} · queued
          </div>
        )}
      </div>
    );
  }

  return (
    <form className="card" style={{ padding: 28 }} onSubmit={submit}>
      <div className="kicker">request access · waitlist</div>
      <h3 style={{ fontSize: 22, fontWeight: 700, letterSpacing: "-0.015em", margin: "8px 0 18px" }}>Tell us about your work.</h3>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <div><label className="label">Name</label><input className="input" required value={form.name} onChange={(e) => update("name", e.target.value)} placeholder="Dr. Naomi Schaal" /></div>
        <div><label className="label">Email</label><input className="input" type="email" required value={form.email} onChange={(e) => update("email", e.target.value)} placeholder="naomi@berkeley.edu" /></div>
        <div><label className="label">Organization</label><input className="input" value={form.org} onChange={(e) => update("org", e.target.value)} placeholder="UC Berkeley · Geography" /></div>
        <div><label className="label">Project CRS</label><input className="input mono" value={form.crs} onChange={(e) => update("crs", e.target.value)} placeholder="EPSG:32619" /></div>
        <div style={{ gridColumn: "1 / -1" }}>
          <label className="label">Domain template</label>
          <select className="input" value={form.domain} onChange={(e) => update("domain", e.target.value)}>
            {DOMAINS.map((o) => <option key={o}>{o}</option>)}
          </select>
        </div>
        <div style={{ gridColumn: "1 / -1" }}>
          <label className="label">What are you trying to figure out?</label>
          <textarea className="input" style={{ minHeight: 100, resize: "vertical" }} value={form.reason} onChange={(e) => update("reason", e.target.value)} placeholder="Modeling canopy intervention scenarios across 287 Philadelphia census tracts. Need DAG validation + posterior σ rasters." />
        </div>
      </div>
      {state === "error" && <p style={{ color: "var(--crimson)", marginTop: 12, fontSize: 13 }}>{error}</p>}
      <div className="row" style={{ marginTop: 22, justifyContent: "space-between", alignItems: "center" }}>
        <span className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>▸ submit · build queue · ~5 business days</span>
        <button type="submit" className="btn btn-accent" disabled={state === "submitting"}>
          {state === "submitting" ? "Sending…" : "Request access"} <span className="arrow">→</span>
        </button>
      </div>
    </form>
  );
}

export default function ContactPage() {
  return (
    <>
      <Nav active="contact" />
      <PageHero
        kicker="contact · we read everything"
        title="Get in touch."
        lead="Waitlist requests, sales, support, research collaborations, press. We try to answer within two business days, and we read every line you write."
        accent="var(--amber)"
      />

      <Section>
        <div id="waitlist" style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 32, alignItems: "start" }}>
          <ContactForm />
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div className="card" style={{ borderLeft: "2px solid var(--crimson)" }}>
              <div className="kicker">all enquiries · one inbox</div>
              <p style={{ fontSize: 13, color: "var(--ink-2)", margin: "8px 0 10px", lineHeight: 1.5 }}>
                Sales, support, research collaborations, press — everything routes here while we set up role inboxes on the new domain.
              </p>
              <a className="mono" href={`mailto:${EMAIL_PRIMARY}`} style={{ fontSize: 13, color: "var(--ink)", textDecorationColor: "var(--crimson)" }}>{EMAIL_PRIMARY}</a>
            </div>

            <div className="card dashed">
              <div className="kicker">office · by appointment</div>
              <p style={{ fontSize: 13, color: "var(--ink-2)", margin: "8px 0 0", lineHeight: 1.55 }}>
                <strong>{COMPANY}</strong><br />
                {OFFICE_FULL}
              </p>
              <p className="mono" style={{ fontSize: 11, color: "var(--muted)", margin: "10px 0 0", letterSpacing: "0.1em" }}>
                {COMPANY_LEGAL}
              </p>
            </div>
          </div>
        </div>
      </Section>

      <Section sunken kicker="response times · honest" title="What to expect.">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
          {[
            { k: "waitlist", v: "≤ 5 days", c: "var(--purple)" },
            { k: "support · pursue", v: "≤ 48 hrs", c: "var(--crimson)" },
            { k: "support · converge", v: "≤ 24 hrs", c: "var(--amber)" },
            { k: "support · transcend", v: "24/7 · 99.9% SLA", c: "var(--gold)" },
          ].map((r, i) => (
            <div key={i} className="card" style={{ padding: 22, borderTop: `3px solid ${r.c}` }}>
              <div className="kicker">{r.k}</div>
              <div style={{ fontSize: 26, fontWeight: 800, letterSpacing: "-0.02em", marginTop: 6, fontFeatureSettings: "'tnum'" }}>{r.v}</div>
            </div>
          ))}
        </div>
      </Section>

      <Footer />
    </>
  );
}
