"use client";

import * as React from "react";
import Link from "next/link";
import { Nav, Footer } from "@/components/shell/Shell";
import { PageHero, Section } from "@/components/marketing/page-shell";

const TEMPLATES = [
  { id: "uhi", name: "UHI", sub: "Urban Heat Island", tag: "climate · urban", vars: 24, c: "var(--crimson)", body: "Quantify and project urban heat anomalies. Inputs: LST rasters, canopy %, impervious %, building height, wind. Outputs: AAT_z scenarios, intervention rasters." },
  { id: "coastal", name: "Coastal", sub: "Erosion & sea-level", tag: "climate · coastal", vars: 18, c: "var(--purple)", body: "Shoreline retreat under SLR scenarios. Inputs: bathymetry, wave climate, sediment grain size. Outputs: retreat rate posteriors, vulnerability rasters." },
  { id: "drought", name: "Drought", sub: "Soil moisture · ET", tag: "climate · agri", vars: 21, c: "var(--amber)", body: "Soil moisture deficit and evapotranspiration with crop water stress indices. Inputs: SMAP, MODIS ET, precipitation. Outputs: SPEI, σ rasters." },
  { id: "geotech", name: "Geotechnical", sub: "Bearing capacity", tag: "civil · structural", vars: 16, c: "var(--gold)", body: "Site-specific bearing capacity from CPT/SPT logs and surface geology. Outputs: capacity rasters with depth-of-influence." },
  { id: "ground", name: "Groundwater", sub: "Aquifer drawdown", tag: "hydrology", vars: 19, c: "var(--magenta)", body: "Pump-test drawdown response across heterogeneous aquifers. Inputs: well logs, lithology, recharge. Outputs: cone-of-depression scenarios." },
  { id: "noise", name: "Noise", sub: "Lp around sources", tag: "civil · airport", vars: 14, c: "var(--pink)", body: "Sound pressure propagation around airports, highways, and industrial sites. Outputs: dBA isolines under operational scenarios." },
  { id: "seismic", name: "Seismic", sub: "PSHA + site response", tag: "civil · risk", vars: 17, c: "var(--red)", body: "Probabilistic seismic hazard with Vs30-conditioned site amplification. Outputs: PGA hazard rasters with disaggregation." },
  { id: "storm", name: "Stormwater", sub: "Runoff · flooding", tag: "hydrology · urban", vars: 17, c: "var(--orange)", body: "TR-55 + 2D routing with green-infrastructure perturbations. Outputs: depth/extent rasters, design-storm comparisons." },
  { id: "water", name: "Water Quality", sub: "DO · nutrients", tag: "hydrology · enviro", vars: 20, c: "var(--crimson)", body: "Dissolved oxygen, total N, total P, fecal coliform under land-use scenarios. Outputs: 303(d) classification rasters." },
  { id: "fire", name: "Wildfire", sub: "Fuel · weather · ROS", tag: "climate · risk", vars: 27, c: "var(--orange)", body: "Rate-of-spread and burn probability with fuel-treatment scenarios. Inputs: LANDFIRE, RAWS, topography. Outputs: BP, FLI rasters." },
  { id: "air", name: "Air Quality", sub: "PM₂.₅ + dispersion", tag: "climate · public-health", vars: 22, c: "var(--pink)", body: "PM₂.₅ and ozone with stationary + mobile sources. Inputs: CMAQ, monitoring stations, traffic. Outputs: exposure rasters with σ." },
  { id: "smip", name: "ForceSMIP", sub: "Forced response", tag: "research · ML", vars: 12, c: "var(--purple)", body: "Detection-and-attribution of forced climate response from internal variability. CMIP6 input, posterior trace output." },
  { id: "blank", name: "Blank", sub: "Start from scratch", tag: "custom", vars: 0, c: "var(--muted)", body: "Empty project skeleton. project.yml + template DAG + variable registry. Bring your own data and theory." },
];

export default function TemplatesPage() {
  const [sel, setSel] = React.useState(TEMPLATES[0]);
  return (
    <>
      <Nav active="templates" />
      <PageHero
        kicker="13 templates · 1 blank slate"
        title="Pick a domain. Bring your data. Run the pipeline."
        lead="Each template ships with a vetted DAG, variable registry, physics constraints, and an example dataset. Fork it, modify it, or start blank — the pipeline doesn't care."
      />
      <Section>
        <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 28, alignItems: "start" }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
            {TEMPLATES.map((t) => {
              const active = sel.id === t.id;
              return (
                <button key={t.id} onClick={() => setSel(t)} style={{
                  textAlign: "left", border: "1px solid var(--line)",
                  background: active ? "var(--ink)" : "var(--bg-elevated)",
                  color: active ? "var(--paper)" : "var(--ink)",
                  borderRadius: 8, padding: 16, cursor: "pointer", fontFamily: "inherit",
                  borderLeftWidth: 3, borderLeftColor: t.c, transition: "all .15s",
                }}>
                  <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline" }}>
                    <h3 style={{ fontSize: 18, fontWeight: 700, margin: 0, letterSpacing: "-0.01em" }}>{t.name}</h3>
                    <span className="mono" style={{ fontSize: 10, opacity: 0.7, letterSpacing: "0.1em" }}>{t.vars > 0 ? `n=${t.vars}` : "blank"}</span>
                  </div>
                  <div className="mono" style={{ fontSize: 11, marginTop: 4, opacity: 0.7, textTransform: "uppercase", letterSpacing: "0.1em" }}>{t.sub}</div>
                  <div className="kicker" style={{ marginTop: 14, color: active ? "rgba(245,239,228,0.55)" : "var(--muted)" }}>{t.tag}</div>
                </button>
              );
            })}
          </div>
          <div style={{ position: "sticky", top: 80 }}>
            <div className="card" style={{ padding: 0, overflow: "hidden", borderLeft: `3px solid ${sel.c}` }}>
              <div style={{ padding: 18, borderBottom: "1px solid var(--line)" }}>
                <div className="kicker">{sel.tag}</div>
                <h2 style={{ fontSize: 32, fontWeight: 800, letterSpacing: "-0.02em", margin: "8px 0 4px" }}>{sel.name}</h2>
                <div className="mono" style={{ fontSize: 12, color: "var(--muted)", letterSpacing: "0.1em", textTransform: "uppercase" }}>{sel.sub}</div>
              </div>
              <div style={{ padding: 18 }}>
                <p style={{ fontSize: 14, lineHeight: 1.55, color: "var(--ink-2)", margin: "0 0 14px" }}>{sel.body}</p>
                <dl className="dl">
                  <dt>variables</dt><dd className="mono">{sel.vars > 0 ? sel.vars : "user-defined"}</dd>
                  <dt>example</dt><dd className="mono">included · {sel.id}_demo.gpkg</dd>
                  <dt>license</dt><dd className="mono">AGPL · attribution required</dd>
                </dl>
                <div className="row" style={{ marginTop: 18, gap: 8 }}>
                  <Link className="btn btn-accent" href="/signup">Fork {sel.name} →</Link>
                  <Link className="btn" href="/product">See the pipeline</Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </Section>

      <Section sunken kicker="bring your own" title="What if your domain isn't here?">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 18 }}>
          {[
            { k: "01", t: "Start blank.", b: "The Blank template is a project skeleton — project.yml, an empty DAG, an empty variable registry. Add your variables, propose your edges, run." },
            { k: "02", t: "Fork the closest.", b: "Geotech for ground stability? Hydrology for surface water? Most domains share more structure than you'd think. Fork the nearest and rewrite." },
            { k: "03", t: "Submit one upstream.", b: "If you build a template you'd like others to use, the SPARC pipeline accepts community templates via PR. Bring data, a DAG, and a citation." },
          ].map((s, i) => (
            <div key={i} className="card dashed">
              <span className="mono" style={{ fontSize: 11, fontWeight: 700, padding: "3px 7px", borderRadius: 3, background: "var(--ink)", color: "#fff" }}>{s.k}</span>
              <h3 style={{ fontSize: 20, fontWeight: 700, margin: "12px 0 8px" }}>{s.t}</h3>
              <p style={{ fontSize: 13, color: "var(--ink-2)", lineHeight: 1.55, margin: 0 }}>{s.b}</p>
            </div>
          ))}
        </div>
      </Section>

      <Footer />
    </>
  );
}
