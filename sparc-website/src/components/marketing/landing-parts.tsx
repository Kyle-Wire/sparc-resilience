"use client";

import * as React from "react";
import Link from "next/link";
import { Kicker } from "@/components/marketing/page-shell";
import { TAGLINE_SHORT, BRAND_LONG, VERSION } from "@/lib/brand";

const BOOT_LINES = [
  { t: 80, text: "▸ sparc init --version 0.4.2", kind: "cmd" },
  { t: 220, text: "  loading sidecar :8008 ················ ok", kind: "ok" },
  { t: 360, text: "  binding CUDA · 11.8 · RTX A6000 ······· ok", kind: "ok" },
  { t: 480, text: "  warming neural · causal kernel ········ ok", kind: "ok" },
  { t: 620, text: "  validating causal graph · 24 nodes ···· ok", kind: "ok" },
  { t: 760, text: "  posterior trace · χ²=0.93 ············· ok", kind: "ok" },
  { t: 900, text: "  scenario engine · adaptation ready ···· ok", kind: "ok" },
  { t: 1020, text: "▸ ready", kind: "ready" },
];

function TerminalSplash() {
  const [shown, setShown] = React.useState(0);
  const [progress, setProgress] = React.useState(0);
  React.useEffect(() => {
    const timers = BOOT_LINES.map((l, i) => setTimeout(() => setShown(i + 1), l.t * 1.5));
    const p = setInterval(() => setProgress((v) => Math.min(100, v + 2.2)), 50);
    return () => { timers.forEach(clearTimeout); clearInterval(p); };
  }, []);
  return (
    <div style={{
      background: "#0e0a0c", color: "#f5efe4", borderRadius: 10,
      border: "1px solid var(--ink)", padding: "14px 18px 18px",
      fontFamily: "var(--font-mono)", fontSize: 12, lineHeight: 1.7,
      position: "relative", overflow: "hidden",
      boxShadow: "0 24px 60px -20px rgba(26,20,22,0.35)",
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", paddingBottom: 10, borderBottom: "1px solid #2a2227", marginBottom: 12 }}>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 10, letterSpacing: "0.12em", color: "#8a7d72", textTransform: "uppercase" }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--crimson)" }} /> sparc · boot
        </span>
        <span style={{ fontSize: 10, color: "#8a7d72", letterSpacing: "0.1em" }}>{VERSION} · {String(Math.floor(progress)).padStart(3, "0")}%</span>
      </div>
      {BOOT_LINES.slice(0, shown).map((l, i) => (
        <div key={i} className="fadeup" style={{
          color: l.kind === "cmd" ? "#fbdd46" : l.kind === "ready" ? "#e73c25" : "#d8d0c2",
          fontWeight: l.kind === "ready" ? 700 : 400, whiteSpace: "pre",
        }}>
          {l.text}
          {l.kind === "ok" && <span style={{ color: "#9e337d", marginLeft: 8 }}>✓</span>}
        </div>
      ))}
      {shown < BOOT_LINES.length && (
        <div style={{ color: "#f0b632" }}>
          ▸ <span style={{ background: "#f5efe4", color: "#0e0a0c", padding: "0 1px", animation: "blink 1s steps(2) infinite" }}>&nbsp;</span>
        </div>
      )}
      <div style={{ marginTop: 14, height: 3, background: "#2a2227", borderRadius: 2, overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${progress}%`,
          background: "linear-gradient(90deg, var(--crimson), var(--amber), var(--gold))",
          transition: "width .12s linear",
        }} />
      </div>
    </div>
  );
}

export function Hero() {
  return (
    <section style={{ padding: "60px 0 80px", borderBottom: "1px solid var(--line)", position: "relative", overflow: "hidden" }} className="grid-bg">
      <div aria-hidden style={{
        position: "absolute", right: -200, top: -200, width: 700, height: 700,
        background: "radial-gradient(circle, rgba(231,60,37,0.08), transparent 60%)",
        pointerEvents: "none",
      }} />
      <div className="wrap" style={{ display: "grid", gridTemplateColumns: "1.05fr 0.95fr", gap: 56, alignItems: "center" }}>
        <div>
          <Kicker dot dotColor="var(--accent)">{TAGLINE_SHORT.toLowerCase()}</Kicker>
          <h1 style={{ fontSize: 64, fontWeight: 800, letterSpacing: "-0.03em", lineHeight: 1.0, margin: "18px 0 22px", textWrap: "balance" }}>
            A neural pipeline that{" "}
            <span style={{
              backgroundImage: "var(--grad-ramp)",
              backgroundSize: "200% 100%",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
              animation: "rampShift 12s ease-in-out infinite",
            }}>simulates adaptation</span>.
          </h1>
          <p style={{ fontSize: 18, color: "var(--ink-2)", maxWidth: 560, lineHeight: 1.5, margin: "0 0 28px" }}>
            SPARC ({BRAND_LONG}) is a spatial neural model with built-in causal
            inference. The pipeline ingests your data, learns local
            relationships, and simulates how a place adapts under interventions
            — with posterior uncertainty on every pixel.
          </p>
          <div className="row" style={{ gap: 10, marginBottom: 28 }}>
            <Link href="/signup" className="btn btn-accent">Request access <span className="arrow">→</span></Link>
            <Link href="/product" className="btn">See the pipeline</Link>
          </div>
          <div className="row" style={{ gap: 24, flexWrap: "wrap", color: "var(--muted)", fontSize: 12 }}>
            <span className="row" style={{ gap: 6 }}><span className="dot" style={{ background: "var(--crimson)" }} /> neural · causal</span>
            <span className="row" style={{ gap: 6 }}><span className="dot" style={{ background: "var(--purple)" }} /> 13 domain templates</span>
            <span className="row" style={{ gap: 6 }}><span className="dot" style={{ background: "var(--amber)" }} /> CUDA · GPU-accelerated</span>
          </div>
        </div>
        <TerminalSplash />
      </div>
    </section>
  );
}

export function ProofStrip() {
  const items = [
    { kicker: "n=1,247 · missing 12", value: "ΔAAT_z = −0.258", sub: "Pct_Canopy (−) · σ 0.154", color: "var(--crimson)" },
    { kicker: "ensemble · posterior", value: "χ² = 0.93", sub: "8 chains · R̂ < 1.01", color: "var(--purple)" },
    { kicker: "philadelphia · scenario", value: "+12.4 °C", sub: "UHI peak · jul 2025", color: "var(--amber)" },
    { kicker: "causal · validated", value: "24 → 41", sub: "nodes → directed edges", color: "var(--gold)" },
  ];
  return (
    <section className="compact" style={{ borderBottom: "1px solid var(--line)" }}>
      <div className="wrap">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 24 }}>
          {items.map((it, i) => (
            <div key={i} style={{ borderLeft: `2px solid ${it.color}`, paddingLeft: 16 }}>
              <div className="kicker">{it.kicker}</div>
              <div style={{ fontSize: 32, fontWeight: 800, letterSpacing: "-0.02em", margin: "4px 0 2px", fontFeatureSettings: "'tnum'" }}>{it.value}</div>
              <div style={{ fontSize: 12, color: "var(--muted)" }} className="mono">{it.sub}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export function PipelineFlow() {
  const [stage, setStage] = React.useState(3);
  React.useEffect(() => {
    const iv = setInterval(() => setStage((s) => (s + 1) % 7), 2200);
    return () => clearInterval(iv);
  }, []);
  const stages = [
    { n: "00", k: "ingest", note: "raster + vector + project.yml", c: "var(--purple)" },
    { n: "01", k: "correlogram", note: "Moran's I · spatial structure", c: "var(--magenta)" },
    { n: "02", k: "causal graph", note: "24 nodes · 41 edges", c: "var(--pink)" },
    { n: "03", k: "neural train", note: "GW3C · χ² 0.93", c: "var(--crimson)" },
    { n: "04", k: "posterior", note: "8 chains · σ 0.154", c: "var(--orange)" },
    { n: "05", k: "scenario", note: "adaptation · uncertainty", c: "var(--amber)" },
    { n: "06", k: "report", note: "audit trail · pdf · md", c: "var(--gold)" },
  ];
  return (
    <div className="card" style={{ padding: 0 }}>
      <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--line)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div className="kicker">pipeline · 7 stages · the product</div>
        <span className="pill"><span className="live-dot" /> running · stage {String(stage).padStart(2, "0")}</span>
      </div>
      <div style={{ padding: 14, display: "flex", flexDirection: "column", gap: 6 }}>
        {stages.map((s, i) => {
          const status = i < stage ? "done" : i === stage ? "live" : "queued";
          return (
            <button key={i} onClick={() => setStage(i)} style={{
              display: "grid", gridTemplateColumns: "44px 130px 1fr 90px", gap: 10, alignItems: "center",
              padding: "10px 12px", borderRadius: 5, cursor: "pointer", textAlign: "left",
              fontFamily: "inherit",
              background: i === stage ? "var(--paper)" : "transparent",
              border: i === stage ? "1px dashed var(--line)" : "1px solid transparent",
              transition: "all .15s",
            }}>
              <span className="mono" style={{ fontSize: 11, fontWeight: 700, padding: "3px 6px", borderRadius: 3, background: s.c, color: "#fff", textAlign: "center" }}>{s.n}</span>
              <span style={{ fontSize: 13, fontWeight: 600 }}>{s.k}</span>
              <span className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>{s.note}</span>
              <span className="mono" style={{ fontSize: 10, color: status === "done" ? "var(--purple)" : status === "live" ? "var(--crimson)" : "var(--muted)", textAlign: "right" }}>
                {status === "done" ? "✓ done" : status === "live" ? "▸ live" : "queued"}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function PosteriorTrace() {
  const [t, setT] = React.useState(0);
  React.useEffect(() => {
    const iv = setInterval(() => setT((v) => v + 1), 80);
    return () => clearInterval(iv);
  }, []);
  const chains = [0, 1, 2, 3].map((c) => {
    const seed = c * 13 + 7;
    const pts: [number, number][] = [];
    for (let i = 0; i < 80; i++) {
      const x = i / 79;
      const noise = Math.sin((i + seed + t * 0.4) * 0.6) * 0.15
                  + Math.sin((i * 0.3 + seed) * 1.2) * 0.07;
      pts.push([x, -0.258 + noise]);
    }
    return pts;
  });
  const chainColors = ["var(--crimson)", "var(--purple)", "var(--amber)", "var(--magenta)"];
  return (
    <div className="card" style={{ padding: 0 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "12px 16px", borderBottom: "1px solid var(--line)" }}>
        <div className="kicker">posterior trace · ΔAAT_z</div>
        <span className="pill"><span className="live-dot" /> live · 8 chains</span>
      </div>
      <div style={{ padding: 16 }}>
        <svg width="100%" height="160" viewBox="0 0 400 160" preserveAspectRatio="none">
          <line x1="0" y1="100" x2="400" y2="100" stroke="var(--line)" strokeDasharray="3 3" />
          <text x="6" y="96" fontSize="9" fontFamily="var(--font-mono)" fill="var(--muted)">μ = −0.258</text>
          {chains.map((pts, c) => (
            <polyline key={c}
              points={pts.map(([x, y]) => `${x * 400},${100 + y * 40}`).join(" ")}
              fill="none" stroke={chainColors[c]} strokeWidth="1.2" opacity="0.85" />
          ))}
        </svg>
        <div className="row" style={{ gap: 16, marginTop: 8, fontSize: 10, color: "var(--muted)" }}>
          {chainColors.map((c, i) => (
            <span key={i} className="row mono" style={{ gap: 5 }}>
              <span style={{ width: 8, height: 2, background: c }} /> chain {i + 1}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

export function ScenarioCompare() {
  return (
    <div className="card" style={{ padding: 0 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "12px 16px", borderBottom: "1px solid var(--line)" }}>
        <div className="kicker">adaptation scenario</div>
        <span className="mono" style={{ fontSize: 10, color: "var(--muted)" }}>philadelphia · 1.2km grid</span>
      </div>
      <div style={{ padding: 16, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {[
          { label: "Baseline", value: "+11.8 °C", color: "var(--crimson)", sigma: "σ 0.42" },
          { label: "+12% canopy", value: "+9.4 °C", color: "var(--amber)", sigma: "σ 0.51" },
        ].map((s, i) => (
          <div key={i} style={{ border: "1px dashed var(--line)", borderRadius: 6, padding: "10px 12px" }}>
            <div className="kicker" style={{ fontSize: 9 }}>{s.label}</div>
            <div style={{ fontSize: 24, fontWeight: 800, letterSpacing: "-0.02em", color: s.color, margin: "4px 0 2px", fontFeatureSettings: "'tnum'" }}>{s.value}</div>
            <div className="mono" style={{ fontSize: 10, color: "var(--muted)" }}>{s.sigma}</div>
          </div>
        ))}
        <div style={{ gridColumn: "1 / -1", padding: "8px 0 2px", borderTop: "1px dotted var(--line)", display: "flex", justifyContent: "space-between" }}>
          <span className="kicker">Δ predicted mean</span>
          <span className="mono" style={{ fontSize: 12, fontWeight: 700, color: "var(--purple)" }}>−2.4 °C</span>
        </div>
      </div>
    </div>
  );
}
