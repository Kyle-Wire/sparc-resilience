import Link from "next/link";
import * as React from "react";
import { Nav, Footer } from "@/components/shell/Shell";
import { PageHero, Section, Kicker } from "@/components/marketing/page-shell";

function FeatureBlock({
  kicker, title, body, items, side, reverse, anchor,
}: {
  kicker: string; title: string; body: string;
  items: { k: string; v: string }[];
  side: React.ReactNode; reverse?: boolean; anchor?: string;
}) {
  return (
    <section style={{ borderBottom: "1px solid var(--line)" }} id={anchor}>
      <div className="wrap" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 56, alignItems: "center" }}>
        <div style={{ order: reverse ? 2 : 1 }}>
          <Kicker dot dotColor="var(--accent)">{kicker}</Kicker>
          <h2 style={{ fontSize: 36, fontWeight: 800, letterSpacing: "-0.02em", margin: "12px 0 16px", lineHeight: 1.05, textWrap: "balance" }}>{title}</h2>
          <p style={{ fontSize: 16, color: "var(--ink-2)", lineHeight: 1.55, margin: "0 0 18px" }}>{body}</p>
          <dl className="dl">
            {items.map((it, i) => (
              <React.Fragment key={i}>
                <dt>{it.k}</dt>
                <dd>{it.v}</dd>
              </React.Fragment>
            ))}
          </dl>
        </div>
        <div style={{ order: reverse ? 1 : 2 }}>{side}</div>
      </div>
    </section>
  );
}

function PipelineDiagram() {
  const stages = [
    { n: "00", k: "ingest", note: "raster + vector + project.yml", c: "var(--purple)" },
    { n: "01", k: "correlogram", note: "Moran's I · Geary's C", c: "var(--magenta)" },
    { n: "02", k: "causal graph", note: "24 nodes · 41 edges", c: "var(--pink)" },
    { n: "03", k: "neural train", note: "GW3C · χ² 0.93", c: "var(--crimson)" },
    { n: "04", k: "posterior", note: "8 chains · σ 0.154", c: "var(--orange)" },
    { n: "05", k: "scenario", note: "adaptation · uncertainty", c: "var(--amber)" },
    { n: "06", k: "report", note: "audit trail · pdf · md", c: "var(--gold)" },
  ];
  return (
    <div className="card" style={{ padding: 0 }}>
      <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--line)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div className="kicker">pipeline · 7 stages</div>
        <span className="pill"><span className="live-dot" /> running · stage 04</span>
      </div>
      <div style={{ padding: 18, display: "flex", flexDirection: "column", gap: 6 }}>
        {stages.map((s, i) => (
          <div key={i} style={{ display: "grid", gridTemplateColumns: "40px 130px 1fr 80px", gap: 10, alignItems: "center", padding: "10px 12px", borderRadius: 5, background: i === 4 ? "var(--paper)" : "transparent", border: i === 4 ? "1px dashed var(--line)" : "1px solid transparent" }}>
            <span className="mono" style={{ fontSize: 11, fontWeight: 700, padding: "3px 6px", borderRadius: 3, background: s.c, color: "#fff", textAlign: "center" }}>{s.n}</span>
            <span style={{ fontSize: 13, fontWeight: 600 }}>{s.k}</span>
            <span className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>{s.note}</span>
            <span className="mono" style={{ fontSize: 10, color: i < 4 ? "var(--purple)" : i === 4 ? "var(--crimson)" : "var(--muted)", textAlign: "right" }}>
              {i < 4 ? "✓ done" : i === 4 ? "▸ live" : "queued"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function CausalCard() {
  const edges = [
    { from: "Pct_Canopy", to: "AAT_z", sign: "−", phys: "shading", strength: 0.78, status: "accepted" },
    { from: "Impervious", to: "AAT_z", sign: "+", phys: "albedo", strength: 0.82, status: "accepted" },
    { from: "Bldg_Height", to: "Wind_z", sign: "−", phys: "drag", strength: 0.41, status: "review" },
  ];
  return (
    <div className="card" style={{ padding: 0 }}>
      <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--line)", display: "flex", justifyContent: "space-between" }}>
        <div className="kicker">causal · edge review</div>
        <span className="mono" style={{ fontSize: 10, color: "var(--muted)" }}>3 of 41</span>
      </div>
      <div style={{ padding: 14, display: "flex", flexDirection: "column", gap: 10 }}>
        {edges.map((e, i) => (
          <div key={i} style={{ border: "1px solid var(--line)", borderRadius: 6, padding: "10px 12px" }}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <span className="mono" style={{ fontSize: 12, fontWeight: 600 }}>
                {e.from} <span style={{ color: "var(--muted)", margin: "0 6px" }}>→</span> {e.to}
              </span>
              <span className="pill" style={{ background: e.status === "accepted" ? "var(--purple)" : "var(--amber)", color: "#fff" }}>
                {e.status === "accepted" ? "✓ accepted" : "▲ review"}
              </span>
            </div>
            <div className="row" style={{ marginTop: 6, justifyContent: "space-between", fontSize: 11, color: "var(--muted)" }}>
              <span className="mono">sign · <strong style={{ color: e.sign === "−" ? "var(--purple)" : "var(--crimson)" }}>{e.sign}</strong></span>
              <span className="mono">physics · {e.phys}</span>
              <span className="mono">|β| · {e.strength}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ScenarioCard() {
  return (
    <div className="card" style={{ padding: 0 }}>
      <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--line)", display: "flex", justifyContent: "space-between" }}>
        <div className="kicker">adaptation · &quot;more shade&quot;</div>
        <span className="mono" style={{ fontSize: 10, color: "var(--muted)" }}>philadelphia · 2025</span>
      </div>
      <div style={{ padding: 16 }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(20, 1fr)", gap: 1, marginBottom: 12 }}>
          {Array.from({ length: 80 }).map((_, i) => {
            const v = Math.sin(i * 0.4) * Math.cos(i * 0.21) * 0.5 + 0.5;
            const stops = ["#602468", "#9e337d", "#e94d9b", "#e94461", "#e73c25", "#e76c25", "#e79024", "#f0b632", "#fbdd46"];
            const idx = Math.floor(v * (stops.length - 1));
            return <div key={i} style={{ aspectRatio: "1", background: stops[idx], opacity: 0.85 }} />;
          })}
        </div>
        <dl className="dl">
          <dt>Δ AAT_z mean</dt><dd className="mono">−0.258 (σ 0.154)</dd>
          <dt>peak temp</dt><dd className="mono">+9.4 °C ↓ from +11.8 °C</dd>
          <dt>tracts improved</dt><dd className="mono">218 of 287</dd>
          <dt>physics check</dt><dd className="mono" style={{ color: "var(--purple)" }}>✓ monotone · χ² 0.93</dd>
        </dl>
      </div>
    </div>
  );
}

export default function ProductPage() {
  return (
    <>
      <Nav active="product" />
      <PageHero
        kicker="product · sparc desktop"
        title="A neural pipeline that simulates adaptation."
        lead="SPARC is a Tauri/React desktop app wrapped around one thing: a 7-stage neural pipeline with built-in causal inference. Build a project, run the pipeline, simulate adaptation — every stage is reproducible, every prediction has a posterior."
      />

      <FeatureBlock
        anchor="pipeline"
        kicker="01 · pipeline · the product"
        title="Seven stages. None of them optional."
        body="The pipeline is the product. Each stage gates the next. χ², σ, and physics constraints are checked before you can advance. Stages are reproducible from project.yml — bring the same file to a different machine and get bit-identical posteriors."
        items={[
          { k: "stages", v: "ingest → correlogram → causal → neural → posterior → scenario → report" },
          { k: "checks", v: "χ² < 1.2 · σ < 0.3 · monotone signs honored" },
          { k: "runtime", v: "CUDA · 11.8 / Apple MPS / CPU fallback" },
          { k: "audit", v: "every run emits a signed manifest" },
        ]}
        side={<PipelineDiagram />}
      />

      <FeatureBlock
        anchor="causal"
        kicker="02 · neural + causal"
        title="A neural model that learns under causal structure."
        body="At the heart of SPARC: a locally-weighted neural ensemble trained under directed causal constraints. Edges are propose-then-adjudicate; signs are physically grounded; coefficients vary smoothly across space. The result is a model that doesn't just fit — it admits to a theory of how the world works."
        reverse
        items={[
          { k: "model", v: "geographically-weighted neural ensemble · 8-chain HMC" },
          { k: "structure", v: "directed causal graph · monotone signs · sign locks" },
          { k: "constraints", v: "physics-class per edge · cycle rejection · mass-balance" },
          { k: "review", v: "accept · reject · flip · constrain · audit" },
        ]}
        side={<CausalCard />}
      />

      <FeatureBlock
        anchor="scenarios"
        kicker="03 · adaptation"
        title="Simulate adaptation, with error bars."
        body="An adaptation scenario is a perturbation applied to the trained ensemble. Increase canopy. Lower impervious. Raise sea level by 0.6m. SPARC re-runs prediction across the grid and returns a posterior mean, σ, and a pixel-wise uncertainty raster — never a single number."
        items={[
          { k: "perturbations", v: "additive · multiplicative · spatial mask" },
          { k: "outputs", v: "mean raster · σ raster · diff raster" },
          { k: "compare", v: "side-by-side, paired ttest, drift map" },
          { k: "export", v: "GeoTIFF · CSV · PDF report" },
        ]}
        side={<ScenarioCard />}
      />

      <section style={{ background: "var(--ink)", color: "var(--paper)", padding: "60px 0", borderBottom: "1px solid var(--ink)" }}>
        <div className="wrap" style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 48 }}>
          {[
            { k: "philosophy · 01", t: "No black boxes.", b: "Every coefficient is interrogable. Every edge has a sign. Every prediction has a posterior." },
            { k: "philosophy · 02", t: "Physics, not vibes.", b: "Monotone constraints, mass balance, conservation laws — the model honors them or it doesn't run." },
            { k: "philosophy · 03", t: "Reproducible by default.", b: "project.yml is the source of truth. Every result is regenerable from a single command, 5 years from now." },
          ].map((p, i) => (
            <div key={i} style={{ borderTop: "1px solid rgba(245,239,228,0.3)", paddingTop: 24 }}>
              <div className="kicker" style={{ color: "rgba(245,239,228,0.5)" }}>{p.k}</div>
              <h3 style={{ fontSize: 26, fontWeight: 700, letterSpacing: "-0.02em", margin: "10px 0 12px", color: "var(--paper)" }}>{p.t}</h3>
              <p style={{ color: "rgba(245,239,228,0.75)", fontSize: 14, lineHeight: 1.55, margin: 0 }}>{p.b}</p>
            </div>
          ))}
        </div>
      </section>

      <Section kicker="under the hood" title="What's actually running.">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
          {[
            { k: "frontend", v: "Tauri 2.0 · React 18 · TypeScript", c: "var(--crimson)" },
            { k: "backend", v: "Python 3.12 · sidecar binary :8008", c: "var(--purple)" },
            { k: "model", v: "GW3C neural · 8-chain HMC", c: "var(--amber)" },
            { k: "geo stack", v: "GDAL · rasterio · GeoPandas", c: "var(--gold)" },
            { k: "compute", v: "CUDA 11.8 · MPS · CPU fallback", c: "var(--magenta)" },
            { k: "storage", v: "DuckDB · parquet · GeoTIFF", c: "var(--orange)" },
            { k: "ai assist", v: "Claude Sonnet 4.6 (BYOK)", c: "var(--pink)" },
            { k: "license", v: "AGPL · open core", c: "var(--red)" },
          ].map((r, i) => (
            <div key={i} className="card" style={{ borderLeft: `2px solid ${r.c}` }}>
              <div className="kicker">{r.k}</div>
              <div className="mono" style={{ fontSize: 13, fontWeight: 600, marginTop: 6 }}>{r.v}</div>
            </div>
          ))}
        </div>
      </Section>

      <section style={{ borderTop: "1px solid var(--line)", padding: "56px 0" }} className="grid-bg">
        <div className="wrap" style={{ textAlign: "center" }}>
          <h2 style={{ fontSize: 40, fontWeight: 800, letterSpacing: "-0.025em", margin: "0 0 16px" }}>Ready to run the pipeline?</h2>
          <div className="row" style={{ justifyContent: "center", gap: 10 }}>
            <Link href="/signup" className="btn btn-accent">Request access <span className="arrow">→</span></Link>
            <Link href="/templates" className="btn">Browse templates</Link>
          </div>
        </div>
      </section>

      <Footer />
    </>
  );
}
