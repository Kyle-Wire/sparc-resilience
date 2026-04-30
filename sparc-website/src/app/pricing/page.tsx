"use client";

import * as React from "react";
import Link from "next/link";
import { Nav, Footer } from "@/components/shell/Shell";
import { PageHero, Section } from "@/components/marketing/page-shell";

type Tier = {
  id: "wander" | "pursue" | "converge" | "transcend";
  name: string;
  subtitle: string;
  poet: string;
  body: string;
  price: string;
  unit: string;
  cta: string;
  ctaStyle: "primary" | "accent" | "ghost";
  ctaHref: string;
  color: string;
  featured?: boolean;
  features: string[];
};

const TIERS: Tier[] = [
  {
    id: "wander", name: "Wander", subtitle: "Free",
    poet: "For the curious.",
    body: "Explore spatial relationships, run your first analyses, and get a feel for the terrain. No commitment, no ceiling on curiosity.",
    price: "$0", unit: "forever",
    cta: "Download", ctaStyle: "ghost", ctaHref: "/download",
    color: "var(--purple)",
    features: [
      "1 active project", "Local-only · no cloud sync", "All 13 domain templates",
      "Run pipeline on CPU or single GPU", "Posterior trace + 1 scenario per project",
      "Community Discord support",
    ],
  },
  {
    id: "pursue", name: "Pursue", subtitle: "Pro",
    poet: "For the serious researcher.",
    body: "Full causal inference, unlimited projects, and the complete SPARC toolkit. You've found something worth chasing — now go after it with everything.",
    price: "$49", unit: "per month · billed yearly",
    cta: "Start Pursue", ctaStyle: "accent", ctaHref: "/api/checkout?tier=pursue&period=yearly",
    color: "var(--crimson)", featured: true,
    features: [
      "Unlimited projects", "Full DAG editor + monotone constraints",
      "Unlimited scenarios + uncertainty rasters", "Multi-GPU ensemble training",
      "Claude Sonnet integration (BYOK)", "Reproducible audit trails (PDF + manifest)",
      "Priority email support",
    ],
  },
  {
    id: "converge", name: "Converge", subtitle: "Team",
    poet: "For groups closing in on something true.",
    body: "Shared projects, collaborative DAG review, and ensemble workflows built for teams that think together. The best findings rarely come from a single mind.",
    price: "$160", unit: "per seat · monthly",
    cta: "Start Converge", ctaStyle: "primary", ctaHref: "/api/checkout?tier=converge&period=monthly",
    color: "var(--amber)",
    features: [
      "Everything in Pursue", "Shared workspaces (up to 25 members)",
      "Collaborative DAG review + edge voting", "Ensemble runs across team GPU pool",
      "Project versioning + change requests", "SSO (Okta · Google · Microsoft)",
      "Dedicated Slack channel",
    ],
  },
  {
    id: "transcend", name: "Transcend", subtitle: "Agency / Institutional",
    poet: "For organizations where the work becomes the infrastructure.",
    body: "Custom onboarding, SSO, SLAs, and SPARC embedded into how your institution does science. The question gets bigger. So does the answer.",
    price: "Custom", unit: "annual contract",
    cta: "Contact sales", ctaStyle: "primary", ctaHref: "/contact?tier=transcend",
    color: "var(--gold)",
    features: [
      "Everything in Converge", "On-premises deployment · air-gapped option",
      "Custom domain templates (we co-author)", "Audit-grade compliance (FedRAMP, SOC 2)",
      "99.9% sidecar SLA + 24/7 support", "Quarterly methods review with our team",
      "Co-authored publications (where appropriate)",
    ],
  },
];

function TierCard({ t }: { t: Tier }) {
  return (
    <div style={{
      background: t.featured ? "var(--ink)" : "var(--bg-elevated)",
      color: t.featured ? "var(--paper)" : "var(--ink)",
      border: "1px solid " + (t.featured ? "var(--ink)" : "var(--line)"),
      borderRadius: 10, position: "relative", overflow: "hidden",
      display: "flex", flexDirection: "column",
      transform: t.featured ? "translateY(-12px)" : "none",
      boxShadow: t.featured ? "0 24px 60px -28px rgba(26,20,22,0.4)" : "none",
    }}>
      <div style={{ height: 4, background: t.color }} />
      {t.featured && (
        <div style={{
          position: "absolute", top: 16, right: 16,
          fontSize: 10, fontFamily: "var(--font-mono)",
          letterSpacing: "0.12em", textTransform: "uppercase",
          padding: "3px 8px", borderRadius: 3,
          background: t.color, color: "#fff", fontWeight: 700,
        }}>most popular</div>
      )}
      <div style={{ padding: "24px 22px 18px", borderBottom: "1px solid " + (t.featured ? "rgba(245,239,228,0.18)" : "var(--line)") }}>
        <div className="row" style={{ alignItems: "baseline", gap: 10, marginBottom: 6 }}>
          <h3 style={{ fontSize: 26, fontWeight: 800, letterSpacing: "-0.02em", margin: 0 }}>{t.name}</h3>
          <span className="mono" style={{ fontSize: 11, color: t.featured ? "rgba(245,239,228,0.5)" : "var(--muted)", letterSpacing: "0.12em", textTransform: "uppercase" }}>{t.subtitle}</span>
        </div>
        <p style={{ fontSize: 13, fontStyle: "italic", color: t.featured ? "rgba(245,239,228,0.7)" : "var(--ink-2)", margin: "0 0 10px" }}>{t.poet}</p>
        <p style={{ fontSize: 13, color: t.featured ? "rgba(245,239,228,0.7)" : "var(--muted)", margin: 0, lineHeight: 1.55 }}>{t.body}</p>
      </div>
      <div style={{ padding: "20px 22px", borderBottom: "1px solid " + (t.featured ? "rgba(245,239,228,0.18)" : "var(--line)") }}>
        <div style={{ fontSize: 40, fontWeight: 800, letterSpacing: "-0.03em", lineHeight: 1, fontFeatureSettings: "'tnum'" }}>{t.price}</div>
        <div className="mono" style={{ fontSize: 11, color: t.featured ? "rgba(245,239,228,0.5)" : "var(--muted)", marginTop: 4, letterSpacing: "0.08em", textTransform: "uppercase" }}>{t.unit}</div>
      </div>
      <ul style={{ listStyle: "none", padding: "20px 22px", margin: 0, flex: 1, display: "flex", flexDirection: "column", gap: 9 }}>
        {t.features.map((f, i) => (
          <li key={i} style={{ display: "flex", gap: 10, fontSize: 13, lineHeight: 1.4 }}>
            <span className="mono" style={{ color: t.color, fontWeight: 700, marginTop: 1 }}>✓</span>
            <span style={{ color: t.featured ? "rgba(245,239,228,0.92)" : "var(--ink-2)" }}>{f}</span>
          </li>
        ))}
      </ul>
      <div style={{ padding: "0 22px 24px" }}>
        <Link
          href={t.ctaHref}
          className={"btn " + (t.ctaStyle === "accent" ? "btn-accent" : t.ctaStyle === "primary" ? "btn-primary" : "")}
          style={{
            width: "100%", justifyContent: "center", padding: "11px 14px",
            ...(t.featured && t.ctaStyle === "primary" ? { background: t.color, borderColor: t.color, color: "#fff" } : {}),
          }}>
          {t.cta} <span className="arrow">→</span>
        </Link>
      </div>
    </div>
  );
}

function ComparisonTable() {
  const rows = [
    { k: "Active projects", v: ["1", "Unlimited", "Unlimited", "Unlimited"] },
    { k: "Domain templates", v: ["13 + blank", "13 + blank", "13 + blank + custom", "13 + blank + co-authored"] },
    { k: "DAG editor", v: ["View only", "Full", "Full + voting", "Full + voting + governance"] },
    { k: "Scenario runner", v: ["1 / project", "Unlimited", "Unlimited", "Unlimited"] },
    { k: "GPU compute", v: ["Local · 1", "Local · multi", "Team pool", "Dedicated cluster"] },
    { k: "Claude integration", v: ["—", "BYOK", "BYOK + shared", "Managed"] },
    { k: "Audit-grade reports", v: ["—", "PDF + manifest", "+ revisions", "+ regulatory"] },
    { k: "Collaboration", v: ["—", "—", "25 seats", "Unlimited"] },
    { k: "SSO", v: ["—", "—", "Okta · Google · MS", "+ SAML · custom"] },
    { k: "On-prem / air-gapped", v: ["—", "—", "—", "Yes"] },
    { k: "SLA", v: ["Best effort", "Email · 48h", "Slack · 24h", "99.9% · 24/7"] },
    { k: "Co-authored research", v: ["—", "—", "—", "Where appropriate"] },
  ];
  return (
    <div className="card" style={{ padding: 0, overflow: "hidden" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ borderBottom: "1px solid var(--line)", background: "var(--paper-2)" }}>
            <th style={{ textAlign: "left", padding: "14px 18px", fontFamily: "var(--font-mono)", fontSize: 10, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--muted)", fontWeight: 600, width: "28%" }}>feature</th>
            {TIERS.map((t) => (
              <th key={t.id} style={{ textAlign: "left", padding: "14px 18px", fontWeight: 700, fontSize: 13, color: "var(--ink)" }}>
                <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 2, background: t.color, marginRight: 8, verticalAlign: "middle" }} />
                {t.name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} style={{ borderBottom: "1px dotted var(--line)" }}>
              <td style={{ padding: "12px 18px", color: "var(--ink-2)", fontWeight: 500 }}>{r.k}</td>
              {r.v.map((v, j) => (
                <td key={j} style={{ padding: "12px 18px", color: v === "—" ? "var(--muted)" : "var(--ink-2)", fontFamily: v === "—" ? "var(--font-mono)" : "inherit" }}>{v}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FAQ() {
  const items = [
    { q: "Is there an academic discount?", a: "Pursue is free for verified students and 50% off for verified faculty at non-profit institutions. Use a .edu / .ac.* email at signup; we'll handle the rest." },
    { q: "What does \"BYOK\" mean for Claude?", a: "Bring your own API key. SPARC never proxies your prompts — they go directly from your desktop to Anthropic. We don't store, log, or train on your conversations." },
    { q: "Can I run SPARC offline?", a: "Yes. SPARC is a desktop app with a local Python sidecar. The only required network calls are license activation (once per 30 days) and, optionally, the Claude API if you've configured it." },
    { q: "Do you sell my data?", a: "No. We sell software. Your projects, rasters, and DAGs never leave your machine unless you explicitly export them. Transcend customers can deploy fully air-gapped." },
    { q: "Can I switch tiers?", a: "Up or down at any time. Annual plans pro-rate. Team and Institutional contracts are renewed quarterly with no auto-escalation." },
  ];
  const [open, setOpen] = React.useState<number | -1>(0);
  return (
    <div style={{ borderTop: "1px solid var(--line)" }}>
      {items.map((it, i) => (
        <div key={i} style={{ borderBottom: "1px solid var(--line)" }}>
          <button onClick={() => setOpen(open === i ? -1 : i)} style={{
            width: "100%", textAlign: "left", padding: "20px 0",
            background: "transparent", border: "none", cursor: "pointer",
            fontFamily: "inherit", fontSize: 17, fontWeight: 600, color: "var(--ink)",
            display: "flex", justifyContent: "space-between", alignItems: "center",
          }}>
            <span>{it.q}</span>
            <span className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>{open === i ? "▲" : "▼"}</span>
          </button>
          {open === i && (
            <div className="fadeup" style={{ padding: "0 0 22px", color: "var(--ink-2)", fontSize: 14, lineHeight: 1.6, maxWidth: 720 }}>{it.a}</div>
          )}
        </div>
      ))}
    </div>
  );
}

export default function PricingPage() {
  return (
    <>
      <Nav active="pricing" />
      <PageHero
        kicker="pricing · four tiers"
        title="Wander. Pursue. Converge. Transcend."
        lead="Four ways to use SPARC, named for the arc of an investigation. Start curious, get serious, work in concert, and — when the question is big enough — make SPARC the infrastructure."
      />
      <Section>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14, alignItems: "stretch" }}>
          {TIERS.map((t) => <TierCard key={t.id} t={t} />)}
        </div>
        <p className="kicker" style={{ marginTop: 28, textAlign: "center" }}>
          all tiers · 30-day refund · no auto-renewal price hikes · cancel from your account
        </p>
      </Section>

      <Section sunken kicker="compare" title="Tier-by-tier breakdown.">
        <ComparisonTable />
      </Section>

      <Section kicker="faq" title="Questions we keep getting.">
        <FAQ />
      </Section>

      <section style={{ borderTop: "1px solid var(--line)", padding: "56px 0" }} className="grid-bg">
        <div className="wrap" style={{ textAlign: "center" }}>
          <h2 style={{ fontSize: 36, fontWeight: 800, letterSpacing: "-0.025em", margin: "0 0 16px" }}>Still deciding?</h2>
          <p style={{ color: "var(--ink-2)", maxWidth: 520, margin: "0 auto 24px", fontSize: 15 }}>
            Wander is free and has no time limit. Run a project end-to-end and see if SPARC fits the way you actually work.
          </p>
          <div className="row" style={{ justifyContent: "center", gap: 10 }}>
            <Link href="/download" className="btn btn-accent">Start with Wander</Link>
            <Link href="/contact" className="btn">Talk to a human</Link>
          </div>
        </div>
      </section>

      <Footer />
    </>
  );
}
