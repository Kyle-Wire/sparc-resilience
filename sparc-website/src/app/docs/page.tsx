import Link from "next/link";
import { Nav, Footer } from "@/components/shell/Shell";
import { PageHero, Section } from "@/components/marketing/page-shell";

export const metadata = { title: "Documentation" };

const DOCS = [
  { t: "Manual", b: "End-to-end overview of the SPARC desktop client.", href: "https://github.com/sparclabs/sparc/blob/main/docs/MANUAL.md" },
  { t: "Pipeline guide", b: "How the neural-causal pipeline assembles a project.", href: "https://github.com/sparclabs/sparc/blob/main/docs/PIPELINE_GUIDE.md" },
  { t: "Interpretation guide", b: "Reading posteriors, intervention rasters, and audit reports.", href: "https://github.com/sparclabs/sparc/blob/main/docs/INTERPRETATION_GUIDE.md" },
  { t: "Contributing", b: "How to submit a community template or plugin.", href: "https://github.com/sparclabs/sparc/blob/main/docs/CONTRIBUTING.md" },
];

export default function DocsPage() {
  return (
    <>
      <Nav />
      <PageHero kicker="documentation" title="The manual, the methods, the pipelines." lead="All technical documentation lives in the open-source repository. Here are the most useful entry points." accent="var(--purple)" />
      <Section>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 14 }}>
          {DOCS.map((d) => (
            <Link key={d.href} href={d.href} className="card" style={{ padding: 24, textDecoration: "none", color: "inherit" }}>
              <h3 style={{ fontSize: 22, fontWeight: 700, letterSpacing: "-0.015em", margin: "0 0 8px" }}>{d.t} →</h3>
              <p style={{ color: "var(--ink-2)", fontSize: 14, lineHeight: 1.55, margin: 0 }}>{d.b}</p>
            </Link>
          ))}
        </div>
      </Section>
      <Footer />
    </>
  );
}
