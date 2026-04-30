import Link from "next/link";
import { Nav, Footer } from "@/components/shell/Shell";
import { PageHero, Section } from "@/components/marketing/page-shell";
import { COMPANY_LONG, BRAND_LONG, FOUNDED_YEAR } from "@/lib/brand";

export const metadata = { title: "Research" };

export default function ResearchPage() {
  return (
    <>
      <Nav />
      <PageHero kicker="research · open" title={`${BRAND_LONG} as a method.`} lead={`${COMPANY_LONG} publishes the methods we use. If you build on SPARC, please cite us — and tell us what you find.`} accent="var(--crimson)" />

      <Section kicker="cite the project" title="How to cite SPARC.">
        <div className="card" style={{ padding: 24 }}>
          <pre className="mono" style={{ fontSize: 12, color: "var(--ink-2)", margin: 0, whiteSpace: "pre-wrap", lineHeight: 1.55 }}>{`@software{sparc_${FOUNDED_YEAR},
  author = {SPARC Labs},
  title  = {SPARC: ${BRAND_LONG}},
  year   = {${FOUNDED_YEAR}},
  url    = {https://sparclabs.co}
}`}</pre>
        </div>
        <p className="kicker" style={{ marginTop: 14 }}>full CITATION.cff in the repository</p>
      </Section>

      <Section sunken kicker="working papers" title="Methods, in the open.">
        <div style={{ display: "grid", gap: 10 }}>
          {[
            "A neural-causal pipeline for spatial intervention modeling (preprint, in prep.)",
            "Monotone constraints as physics priors (workshop, ML4PS 2025)",
            "Forced-response detection in CMIP6 with neural ensembles (working paper)",
          ].map((t, i) => (
            <div key={i} className="card" style={{ padding: 18 }}>
              <div style={{ fontSize: 14, color: "var(--ink-2)" }}>{t}</div>
            </div>
          ))}
        </div>
        <p style={{ marginTop: 14, fontSize: 13, color: "var(--muted)" }}>
          Want to collaborate or co-author? <Link href="/contact" style={{ color: "var(--ink)" }}>Get in touch →</Link>
        </p>
      </Section>

      <Footer />
    </>
  );
}
