import { Nav, Footer } from "@/components/shell/Shell";
import { PageHero, Section } from "@/components/marketing/page-shell";
import { COMPANY, COMPANY_LEGAL, EMAIL_PRIMARY, OFFICE_FULL, COPYRIGHT_YEAR } from "@/lib/brand";

export const metadata = { title: "Terms of Service" };

export default function TermsPage() {
  return (
    <>
      <Nav />
      <PageHero kicker="terms · honest" title="Terms of Service." accent="var(--crimson)" />
      <Section>
        <article style={{ maxWidth: 760, margin: "0 auto", display: "grid", gap: 18, color: "var(--ink-2)", fontSize: 15, lineHeight: 1.7 }}>
          <p>By using {COMPANY} you agree to these terms. The SPARC source code is licensed under <strong>AGPL-3.0</strong>; the desktop client requires an active subscription for non-Wander tiers.</p>

          <h2 style={{ fontSize: 24, fontWeight: 700, color: "var(--ink)", margin: 0 }}>Subscriptions</h2>
          <p>Billing is handled by Lemon Squeezy as merchant of record. You can cancel any time from the customer portal. Yearly plans are pro-rated on cancellation; monthly plans run to the end of the current period.</p>

          <h2 style={{ fontSize: 24, fontWeight: 700, color: "var(--ink)", margin: 0 }}>Refunds</h2>
          <p>Full refund within 30 days, no questions asked. Email <a href={`mailto:${EMAIL_PRIMARY}`}>{EMAIL_PRIMARY}</a>.</p>

          <h2 style={{ fontSize: 24, fontWeight: 700, color: "var(--ink)", margin: 0 }}>Acceptable use</h2>
          <p>Don&apos;t use SPARC to harm people. Don&apos;t use it to misrepresent uncertainty in ways a court would call fraud. Don&apos;t scrape our marketing site.</p>

          <h2 style={{ fontSize: 24, fontWeight: 700, color: "var(--ink)", margin: 0 }}>No warranty</h2>
          <p>SPARC is provided &quot;as is&quot;. The neural-causal pipeline produces statistical estimates, not engineering certifications. You are responsible for the decisions you make from its outputs.</p>

          <p style={{ fontSize: 13, color: "var(--muted)" }}>© {COPYRIGHT_YEAR} {COMPANY_LEGAL} · {OFFICE_FULL}</p>
        </article>
      </Section>
      <Footer />
    </>
  );
}
