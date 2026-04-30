import { Nav, Footer } from "@/components/shell/Shell";
import { PageHero, Section } from "@/components/marketing/page-shell";
import { COMPANY, COMPANY_LEGAL, EMAIL_PRIMARY, OFFICE_FULL, COPYRIGHT_YEAR } from "@/lib/brand";

export const metadata = { title: "Privacy" };

export default function PrivacyPage() {
  return (
    <>
      <Nav />
      <PageHero kicker="privacy · plain english" title="What we collect, what we don't." lead={`${COMPANY} is a desktop application. Your projects, rasters, and prompts stay on your machine.`} accent="var(--purple)" />
      <Section>
        <article className="prose" style={{ maxWidth: 760, margin: "0 auto", display: "grid", gap: 18, color: "var(--ink-2)", fontSize: 15, lineHeight: 1.7 }}>
          <h2 style={{ fontSize: 24, fontWeight: 700, color: "var(--ink)", margin: 0 }}>What we collect</h2>
          <p>Account information you give us (email, optional name, organization). Subscription metadata from Lemon Squeezy (status, tier, renewal dates) — never card numbers, which Lemon Squeezy holds. Anonymous metering events from your desktop client (e.g. <code>pipeline_runs += 1</code>) — never the inputs or outputs of those runs.</p>

          <h2 style={{ fontSize: 24, fontWeight: 700, color: "var(--ink)", margin: 0 }}>What we don&apos;t</h2>
          <p>Your project files, rasters, DAGs, model weights, posterior samples, scenario outputs, or LLM prompts. We do not log, proxy, or train on any of this material. BYOK API keys are encrypted with Supabase Vault and are only ever decrypted by your authenticated desktop client at runtime.</p>

          <h2 style={{ fontSize: 24, fontWeight: 700, color: "var(--ink)", margin: 0 }}>Your rights</h2>
          <p>You can export, correct, or delete your account data at any time by emailing <a href={`mailto:${EMAIL_PRIMARY}`}>{EMAIL_PRIMARY}</a>. We will respond within 30 days.</p>

          <h2 style={{ fontSize: 24, fontWeight: 700, color: "var(--ink)", margin: 0 }}>Subprocessors</h2>
          <ul><li>Supabase — auth + database (US/EU)</li><li>Lemon Squeezy — billing & merchant of record (US/EU)</li><li>Vercel — website hosting (US)</li><li>Resend — transactional email (US)</li></ul>

          <p style={{ fontSize: 13, color: "var(--muted)" }}>© {COPYRIGHT_YEAR} {COMPANY_LEGAL} · {OFFICE_FULL}</p>
        </article>
      </Section>
      <Footer />
    </>
  );
}
