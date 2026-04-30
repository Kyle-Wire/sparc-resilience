import Link from "next/link";
import { Nav, Footer } from "@/components/shell/Shell";
import { PageHero, Section, Kicker } from "@/components/marketing/page-shell";
import { COMPANY, COMPANY_LEGAL, COMPANY_LONG, EMAIL_PRIMARY, OFFICE_FULL, FOUNDED_YEAR, OFFICE_CITY, OFFICE_STATE, VERSION, BRAND_LONG } from "@/lib/brand";

export default function AboutPage() {
  return (
    <>
      <Nav active="about" />
      <PageHero
        kicker={`about · ${COMPANY.toLowerCase()}`}
        title="A research lab dressed like a research lab."
        lead={`We make tools for the people who actually have to be right — climate scientists, civil consultants, urban planners, and the grad students doing most of the work. Founded in ${FOUNDED_YEAR} in ${OFFICE_CITY}, ${OFFICE_STATE}.`}
        accent="var(--purple)"
      />

      <Section>
        <div id="thesis" style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 56, alignItems: "start" }}>
          <div>
            <Kicker>our thesis</Kicker>
            <h2 style={{ fontSize: 40, fontWeight: 800, letterSpacing: "-0.025em", lineHeight: 1.1, margin: "12px 0 18px" }}>Spatial science deserves better instruments.</h2>
            <p style={{ fontSize: 16, color: "var(--ink-2)", lineHeight: 1.65, marginBottom: 14 }}>
              Most spatial ML tools today are either too academic to ship or too commercial to trust. The first ones don&apos;t have install scripts. The second ones don&apos;t tell you what their models actually optimize.
            </p>
            <p style={{ fontSize: 16, color: "var(--ink-2)", lineHeight: 1.65, marginBottom: 14 }}>
              We started {COMPANY} ({COMPANY_LONG}) to make a third thing — a working desktop application built around a single neural pipeline that takes the rigor of academic geostatistics seriously, with the engineering hygiene of a product you can hand to a colleague.
            </p>
            <p style={{ fontSize: 16, color: "var(--ink-2)", lineHeight: 1.65, marginBottom: 0 }}>
              The model is neural. The structure is causal. The promise is that you can simulate adaptation with error bars — not just describe the world.
            </p>
          </div>
          <div className="card" style={{ padding: 28, position: "sticky", top: 80 }}>
            <div className="kicker">at a glance</div>
            <dl className="dl" style={{ marginTop: 12 }}>
              <dt>founded</dt><dd className="mono">{FOUNDED_YEAR} · {OFFICE_CITY}, {OFFICE_STATE}</dd>
              <dt>entity</dt><dd className="mono">{COMPANY_LEGAL}</dd>
              <dt>stage</dt><dd className="mono">private beta · {VERSION}</dd>
              <dt>license</dt><dd className="mono">AGPL · open core</dd>
              <dt>runtime</dt><dd className="mono">Tauri · React · Python sidecar</dd>
              <dt>compute</dt><dd className="mono">CUDA 11.8 · MPS · CPU</dd>
              <dt>contact</dt><dd className="mono">{EMAIL_PRIMARY}</dd>
            </dl>
          </div>
        </div>
      </Section>

      <Section sunken kicker="principles · 04" title="What we'll never compromise on." id="principles">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 18 }}>
          {[
            { k: "01", t: "Models that show their work.", b: "If you can't read the coefficients, set the priors, or audit the chain, it isn't science. It's marketing." },
            { k: "02", t: "Reproducibility, not replicability.", b: "A SPARC project from 2026 will run identically on a fresh install in 2031. We pin our way out of dependency hell so you don't have to." },
            { k: "03", t: "Physics is a feature.", b: "Monotone constraints, conservation laws, sign locks — these aren't restrictions, they're the reason the model is worth listening to." },
            { k: "04", t: "Your data stays yours.", b: "Local-first by default. We don't proxy, log, or train on your projects. We sell software, not your work." },
          ].map((p, i) => (
            <div key={i} className="card" style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: 16, padding: 22 }}>
              <span className="mono" style={{ fontSize: 11, fontWeight: 700, padding: "4px 8px", borderRadius: 3, background: "var(--ink)", color: "#fff", height: "fit-content", letterSpacing: "0.1em" }}>{p.k}</span>
              <div>
                <h3 style={{ fontSize: 20, fontWeight: 700, letterSpacing: "-0.015em", margin: "0 0 8px" }}>{p.t}</h3>
                <p style={{ fontSize: 14, color: "var(--ink-2)", lineHeight: 1.55, margin: 0 }}>{p.b}</p>
              </div>
            </div>
          ))}
        </div>
      </Section>

      <Section kicker="get in touch" title="Want to talk?">
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>
          <div className="card" style={{ padding: 28, borderLeft: "3px solid var(--crimson)" }}>
            <div className="kicker">researchers · waitlist</div>
            <h3 style={{ fontSize: 22, fontWeight: 700, margin: "10px 0 10px" }}>Try the private beta.</h3>
            <p style={{ fontSize: 14, color: "var(--ink-2)", lineHeight: 1.55, margin: "0 0 16px" }}>
              Tell us your domain and CRS. We&apos;ll send a build with the right template pre-loaded.
            </p>
            <Link href="/signup" className="btn btn-accent">Request access <span className="arrow">→</span></Link>
          </div>
          <div className="card" style={{ padding: 28, borderLeft: "3px solid var(--purple)" }}>
            <div className="kicker">institutions · partnerships</div>
            <h3 style={{ fontSize: 22, fontWeight: 700, margin: "10px 0 10px" }}>Bigger questions, longer arcs.</h3>
            <p style={{ fontSize: 14, color: "var(--ink-2)", lineHeight: 1.55, margin: "0 0 16px" }}>
              Embedded deployments, custom domain templates, audit-grade compliance. Annual contracts.
            </p>
            <Link href="/contact" className="btn">Talk to us <span className="arrow">→</span></Link>
          </div>
        </div>
        <p className="mono" style={{ marginTop: 24, fontSize: 11, color: "var(--muted)", textAlign: "center", letterSpacing: "0.1em" }}>
          {COMPANY_LEGAL} · {OFFICE_FULL} · {BRAND_LONG.toLowerCase()}
        </p>
      </Section>

      <Footer />
    </>
  );
}
