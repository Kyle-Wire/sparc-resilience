import Link from "next/link";
import { Nav, Footer } from "@/components/shell/Shell";
import { Kicker, Section } from "@/components/marketing/page-shell";
import { Hero, ProofStrip, PipelineFlow, PosteriorTrace, ScenarioCompare } from "@/components/marketing/landing-parts";

const PILLARS = [
  {
    kicker: "01 · neural model",
    title: "Locally-weighted neural ensembles.",
    body: "Every prediction has a local model. Coefficients vary smoothly across space; uncertainty travels with them. Spatial heterogeneity is a feature, not noise.",
    tag: "physics-constrained",
  },
  {
    kicker: "02 · causal inference",
    title: "Causal, not just correlational.",
    body: "The pipeline learns under directed causal structure. Edges have direction, signs are physically grounded, and counterfactuals respect the graph.",
    tag: "monotone constraints",
  },
  {
    kicker: "03 · adaptation engine",
    title: "Simulate \"what-if\" — with error bars.",
    body: "Increase canopy 12%. Lower impervious 8%. Get a posterior mean, σ, and a pixel-wise uncertainty raster — never a single number.",
    tag: "uncertainty quantified",
  },
];

const TEMPLATES_PEEK = [
  { name: "UHI", sub: "Urban Heat Island", n: 24, c: "var(--crimson)" },
  { name: "Coastal", sub: "Erosion + sea level", n: 18, c: "var(--purple)" },
  { name: "Drought", sub: "Soil moisture · ET", n: 21, c: "var(--amber)" },
  { name: "Geotechnical", sub: "Bearing capacity", n: 16, c: "var(--gold)" },
  { name: "Groundwater", sub: "Aquifer drawdown", n: 19, c: "var(--magenta)" },
  { name: "Wildfire", sub: "Fuel · weather · ROS", n: 27, c: "var(--orange)" },
  { name: "Air Quality", sub: "PM₂.₅ + dispersion", n: 22, c: "var(--pink)" },
  { name: "Stormwater", sub: "Runoff · flooding", n: 17, c: "var(--red)" },
];

const STEPS = [
  { no: "01", k: "Setup", title: "Bring your project", body: "Drop a project.yml, point at your raster + vector data, set CRS and time window. Or fork one of 13 templates." },
  { no: "02", k: "Causal", title: "Set the structure", body: "SPARC proposes a causal graph from the correlogram; you accept, reject, or rewrite. Physics constraints (signs, monotones) are first-class." },
  { no: "03", k: "Pipeline", title: "Run the neural ensemble", body: "Locally-weighted models train in parallel across the grid. Posterior traces stream live. χ² and σ gate every stage." },
  { no: "04", k: "Adapt", title: "Simulate and report", body: "Compare adaptation scenarios on a single map. Export pixel-wise uncertainty rasters and a fully reproducible audit trail." },
];

export default function HomePage() {
  return (
    <>
      <Nav active="home" />
      <Hero />
      <ProofStrip />

      <Section kicker="what it does" title="Three things, done with rigor." lead="An engineer's notebook for spatial science. Every step is auditable, every coefficient is interrogable, every claim has a confidence interval.">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 18 }}>
          {PILLARS.map((it, i) => (
            <div key={i} className="card" style={{ padding: 24, position: "relative" }}>
              <div className="kicker">{it.kicker}</div>
              <h3 style={{ fontSize: 22, fontWeight: 700, letterSpacing: "-0.015em", margin: "10px 0 12px", textWrap: "pretty" }}>{it.title}</h3>
              <p style={{ color: "var(--ink-2)", fontSize: 14, lineHeight: 1.55, margin: 0 }}>{it.body}</p>
              <div style={{ marginTop: 16, paddingTop: 12, borderTop: "1px dashed var(--line)" }}>
                <span className="pill" style={{ background: "transparent", border: "1px solid var(--line)" }}>{it.tag}</span>
              </div>
            </div>
          ))}
        </div>
      </Section>

      <section style={{ background: "var(--paper-2)", borderTop: "1px solid var(--line)", borderBottom: "1px solid var(--line)" }}>
        <div className="wrap">
          <div className="section-head">
            <div>
              <Kicker dot dotColor="var(--purple)">live · interactive</Kicker>
              <h2 style={{ marginTop: 8 }}>The pipeline is the product.</h2>
            </div>
            <p className="lead">A simplified urban-heat run on Philadelphia census tracts. Every stage is reproducible from <code className="mono">project.yml</code>.</p>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 18, alignItems: "stretch" }}>
            <PipelineFlow />
            <div className="col" style={{ gap: 18 }}>
              <PosteriorTrace />
              <ScenarioCompare />
            </div>
          </div>
        </div>
      </section>

      <Section
        kicker="13 templates · 1 blank slate"
        title="Domain-agnostic, not domain-naive."
        action={<Link href="/templates" className="btn">All templates <span className="arrow">→</span></Link>}
      >
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
          {TEMPLATES_PEEK.map((it) => (
            <div key={it.name} className="card" style={{ padding: 16, borderLeft: `2px solid ${it.c}` }}>
              <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline" }}>
                <h4 style={{ margin: 0, fontSize: 16, fontWeight: 700 }}>{it.name}</h4>
                <span className="mono" style={{ fontSize: 10, color: "var(--muted)", letterSpacing: "0.1em" }}>n={it.n}</span>
              </div>
              <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 4 }}>{it.sub}</div>
            </div>
          ))}
        </div>
      </Section>

      <Section kicker="how it works" title="Setup → Causal → Pipeline → Adapt." lead="Four stages, hand-numbered. The same flow whether you're modeling drought across a watershed or noise around a runway.">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 0, borderTop: "1px solid var(--line)", borderBottom: "1px solid var(--line)" }}>
          {STEPS.map((s, i) => (
            <div key={i} style={{ padding: "26px 22px", borderRight: i < 3 ? "1px solid var(--line)" : "none", position: "relative" }}>
              <div className="row" style={{ alignItems: "baseline", gap: 10, marginBottom: 12 }}>
                <span className="mono" style={{ fontSize: 11, fontWeight: 700, padding: "2px 6px", borderRadius: 3, background: "var(--ink)", color: "#fff" }}>{s.no}</span>
                <span className="kicker">{s.k}</span>
              </div>
              <h4 style={{ fontSize: 17, fontWeight: 700, letterSpacing: "-0.015em", margin: "0 0 10px" }}>{s.title}</h4>
              <p style={{ fontSize: 13, color: "var(--ink-2)", margin: 0, lineHeight: 1.55 }}>{s.body}</p>
            </div>
          ))}
        </div>
      </Section>

      <section style={{ borderTop: "1px solid var(--line)" }} className="grid-bg">
        <div className="wrap" style={{ textAlign: "center", padding: "48px 0" }}>
          <Kicker dot dotColor="var(--accent)">join the waitlist</Kicker>
          <h2 style={{ fontSize: 48, fontWeight: 800, letterSpacing: "-0.025em", lineHeight: 1.05, margin: "16px auto 18px", maxWidth: 800, textWrap: "balance" }}>
            The desktop app is in private beta.<br />Researchers go first.
          </h2>
          <p style={{ color: "var(--ink-2)", maxWidth: 540, margin: "0 auto 24px", fontSize: 16 }}>
            Tell us your domain and your CRS. We&apos;ll send a build with the right
            template pre-loaded and a sidecar binary that talks to your GPU.
          </p>
          <div className="row" style={{ justifyContent: "center", gap: 10 }}>
            <Link href="/signup" className="btn btn-accent">Request access <span className="arrow">→</span></Link>
            <Link href="/product" className="btn">Tour the pipeline</Link>
          </div>
        </div>
      </section>

      <Footer />
    </>
  );
}
