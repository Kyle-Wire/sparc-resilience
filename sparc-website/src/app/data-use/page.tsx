import { Nav, Footer } from "@/components/shell/Shell";
import { PageHero, Section } from "@/components/marketing/page-shell";
import { COMPANY, EMAIL_PRIMARY } from "@/lib/brand";

export const metadata = { title: "Data Use" };

export default function DataUsePage() {
  return (
    <>
      <Nav />
      <PageHero kicker="data use · explicit" title="What happens to your data." accent="var(--amber)" />
      <Section>
        <article style={{ maxWidth: 760, margin: "0 auto", display: "grid", gap: 18, color: "var(--ink-2)", fontSize: 15, lineHeight: 1.7 }}>
          <p><strong>{COMPANY} is local-first.</strong> Your projects, rasters, vectors, model artifacts, and posterior samples are stored on your machine. They never leave it unless you explicitly export them.</p>

          <h2 style={{ fontSize: 24, fontWeight: 700, color: "var(--ink)", margin: 0 }}>What leaves your machine</h2>
          <ul>
            <li>License activation pings (every 30 days) — contains your account ID and license key, nothing else.</li>
            <li>Anonymous metering events — counts of pipeline runs and scenarios, never their contents.</li>
            <li>If you opt in, crash reports — sanitized stack traces, no project paths or data.</li>
          </ul>

          <h2 style={{ fontSize: 24, fontWeight: 700, color: "var(--ink)", margin: 0 }}>BYOK LLM keys</h2>
          <p>Your Anthropic / OpenAI / Mistral keys are encrypted at rest with Supabase Vault and only decrypted by your authenticated desktop client. Prompts go directly from your machine to the LLM provider. We never proxy them.</p>

          <h2 style={{ fontSize: 24, fontWeight: 700, color: "var(--ink)", margin: 0 }}>Air-gapped option</h2>
          <p>Transcend customers can deploy SPARC entirely offline with manual license activation. Email <a href={`mailto:${EMAIL_PRIMARY}`}>{EMAIL_PRIMARY}</a> for the air-gapped installer and signing keys.</p>
        </article>
      </Section>
      <Footer />
    </>
  );
}
