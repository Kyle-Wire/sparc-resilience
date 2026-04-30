import { Nav, Footer } from "@/components/shell/Shell";
import { PageHero, Section } from "@/components/marketing/page-shell";
import { VERSION } from "@/lib/brand";

export const metadata = { title: "Changelog" };

const RELEASES = [
  { v: VERSION, d: "2026-01", notes: ["Lemon Squeezy billing integration", "Supabase auth + account portal", "Per-seat license keys", "BYOK API key vault"] },
  { v: "v0.4.0", d: "2025-11", notes: ["Scenario engine v4", "Phase C7 dual-write artifact store", "Memory watchdog", "Bayesian beta simulator"] },
  { v: "v0.3.0", d: "2025-08", notes: ["Causal stack", "NUTS per-edge sampling", "Stage 4 migration"] },
];

export default function ChangelogPage() {
  return (
    <>
      <Nav />
      <PageHero kicker="changelog" title="What's shipped." lead="Honest, dated, no rewrites." accent="var(--magenta)" />
      <Section>
        <div style={{ display: "grid", gap: 18 }}>
          {RELEASES.map((r) => (
            <div key={r.v} className="card" style={{ padding: 24 }}>
              <div className="row" style={{ alignItems: "baseline", justifyContent: "space-between" }}>
                <h3 style={{ fontSize: 24, fontWeight: 800, letterSpacing: "-0.02em", margin: 0 }}>{r.v}</h3>
                <span className="mono" style={{ fontSize: 12, color: "var(--muted)" }}>{r.d}</span>
              </div>
              <ul style={{ marginTop: 12, paddingLeft: 18, color: "var(--ink-2)", fontSize: 14, lineHeight: 1.65 }}>
                {r.notes.map((n, i) => <li key={i}>{n}</li>)}
              </ul>
            </div>
          ))}
        </div>
      </Section>
      <Footer />
    </>
  );
}
