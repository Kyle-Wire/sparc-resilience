// Page content for each sidebar destination.

const { useState: useStateP } = React;

function Card({ title, subtitle, actions, children, padding = 16, style = {} }) {
  return (
    <div style={{
      background: "#fff",
      border: "1px solid var(--line)",
      borderRadius: 8,
      display: "flex", flexDirection: "column",
      ...style,
    }}>
      {(title || actions) && (
        <div style={{
          display: "flex", alignItems: "center", gap: 10,
          padding: "10px 14px",
          borderBottom: "1px solid var(--line)",
          background: "#fdfbf7",
          borderRadius: "8px 8px 0 0",
        }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            {title && <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: "-0.01em" }}>{title}</div>}
            {subtitle && <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>{subtitle}</div>}
          </div>
          {actions}
        </div>
      )}
      <div style={{ padding, flex: 1, minHeight: 0 }}>{children}</div>
    </div>
  );
}

function Tag({ color, children }) {
  return (
    <span className="mono" style={{
      fontSize: 9.5, padding: "2px 6px", borderRadius: 3,
      background: `${color}22`, color,
      letterSpacing: "0.06em", textTransform: "uppercase", fontWeight: 600,
    }}>{children}</span>
  );
}

function SectionHeader({ label, kicker, right }) {
  return (
    <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", marginBottom: 14 }}>
      <div>
        <div className="mono" style={{ fontSize: 10, color: "var(--muted)", letterSpacing: "0.16em", textTransform: "uppercase" }}>{kicker}</div>
        <div style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em", marginTop: 3 }}>{label}</div>
      </div>
      {right}
    </div>
  );
}

// ---------- PROJECT ----------
function ProjectPage() {
  return (
    <div>
      <SectionHeader
        kicker="01 · setup"
        label="Project"
        right={<div style={{ display: "flex", gap: 8 }}>
          <Btn>Open project.yml</Btn>
          <Btn primary>New from template</Btn>
        </div>}
      />
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 14 }}>
        <Card title="Active project" subtitle="uhi · Brown University / Providence, RI">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <KeyVal label="Name" value="Brown UHI — 30 m" />
            <KeyVal label="Domain" value={<Tag color="var(--crimson)">UHI</Tag>} />
            <KeyVal label="Target" value="AAT_z (°F, z-score)" />
            <KeyVal label="Observations" value="54,701 pts" />
            <KeyVal label="CRS (in / proj)" value="EPSG:4326 → 3438" />
            <KeyVal label="Resolution" value="≈30 m" />
            <KeyVal label="Random seed" value="42" />
            <KeyVal label="Pipeline" value={<Tag color="var(--purple)">fast_mode:false</Tag>} />
          </div>
          <div style={{ marginTop: 14, borderTop: "1px dashed var(--line)", paddingTop: 12 }}>
            <div className="mono" style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 6 }}>
              project.yml · preview
            </div>
            <pre className="mono" style={{ fontSize: 10.5, margin: 0, lineHeight: 1.6, color: "var(--ink-2)" }}>{
`project:
  name: brown_uhi
  domain: uhi
  version: 2.1
data:
  path: examples/brown4.csv
  target_column: AAT_z
crs:
  input_epsg: 4326
  projected_epsg: 3438
flags:
  use_gwen: true
  use_laplacian: true`
            }</pre>
          </div>
        </Card>

        <Card title="Templates" subtitle="13 domains available">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
            {[
              ["uhi", "Urban Heat Island", "var(--crimson)"],
              ["forcesmip", "Climate Forcing", "var(--purple)"],
              ["groundwater", "Hydrogeology", "var(--magenta)"],
              ["air_quality", "Air Quality", "var(--amber)"],
              ["stormwater", "Stormwater", "var(--pink)"],
              ["coastal", "Coastal Eng.", "var(--red)"],
              ["geotechnical", "Geotechnical", "var(--orange)"],
              ["seismic", "Seismic", "var(--gold)"],
              ["noise", "Noise", "var(--muted)"],
              ["wildfire", "Wildfire", "var(--crimson)"],
              ["drought", "Drought", "var(--amber)"],
              ["water_quality", "Water Quality", "var(--purple)"],
            ].map(([k, label, col], i) => (
              <button key={k} style={{
                textAlign: "left", border: "1px solid var(--line)",
                background: k === "uhi" ? "#fff8ef" : "#fff",
                borderColor: k === "uhi" ? "var(--amber)" : "var(--line)",
                borderRadius: 5,
                padding: "7px 9px", cursor: "pointer",
                fontFamily: "inherit",
              }}>
                <div className="mono" style={{ fontSize: 9, color: col, letterSpacing: "0.08em", textTransform: "uppercase" }}>{k}</div>
                <div style={{ fontSize: 11, fontWeight: 600, color: "var(--ink-2)", marginTop: 1 }}>{label}</div>
              </button>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

function KeyVal({ label, value }) {
  return (
    <div>
      <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", letterSpacing: "0.1em", textTransform: "uppercase" }}>{label}</div>
      <div style={{ fontSize: 13, fontWeight: 600, marginTop: 3 }}>{value}</div>
    </div>
  );
}

function Btn({ children, primary, small, ...rest }) {
  return (
    <button {...rest} style={{
      border: "1px solid " + (primary ? "var(--ink)" : "var(--line)"),
      background: primary ? "var(--ink)" : "#fff",
      color: primary ? "#fff" : "var(--ink-2)",
      padding: small ? "4px 10px" : "7px 14px",
      fontSize: small ? 11 : 12, fontWeight: 600,
      borderRadius: 5, cursor: "pointer",
      fontFamily: "inherit",
    }}>{children}</button>
  );
}

// ---------- DATA ----------
function DataPage() {
  const cols = [
    ["id", "int", "54,701 unique"],
    ["x", "float", "RI SP · 237,112 … 258,441"],
    ["y", "float", "RI SP · 236,504 … 269,880"],
    ["AAT_z", "float", "target · −2.41 … +3.08"],
    ["Pct_Canopy", "float", "0 … 100"],
    ["Pct_Impervious", "float", "0 … 100"],
    ["NDVI", "float", "−0.12 … 0.91"],
    ["Albedo", "float", "0.04 … 0.48"],
    ["Elevation_m", "float", "0.4 … 88.2"],
    ["Distance_from_water_m", "float", "0 … 4,421"],
  ];
  return (
    <div>
      <SectionHeader kicker="02 · setup" label="Data"
        right={<Btn small>Upload CSV</Btn>} />

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 10, marginBottom: 14 }}>
        <Stat label="Rows" value="54,701" tint="var(--ink)" />
        <Stat label="Columns" value="10" tint="var(--ink)" />
        <Stat label="Missing" value="0.00%" tint="var(--purple)" />
        <Stat label="Spatial density" value="≈1.1 / 30m²" tint="var(--crimson)" />
      </div>

      <Card title="Schema" subtitle="examples/brown4.csv">
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ textAlign: "left", color: "var(--muted)" }}>
              <th style={th}>#</th><th style={th}>Column</th><th style={th}>Type</th><th style={th}>Summary</th><th style={th}>Role</th>
            </tr>
          </thead>
          <tbody>
            {cols.map((c, i) => (
              <tr key={c[0]} style={{ borderTop: "1px solid var(--line)" }}>
                <td style={{ ...td, color: "var(--muted)" }} className="mono">{String(i).padStart(2, "0")}</td>
                <td style={{ ...td, fontWeight: 600 }}>{c[0]}</td>
                <td style={td}><span className="mono" style={{ fontSize: 10, color: "var(--purple)" }}>{c[1]}</span></td>
                <td style={{ ...td, color: "var(--muted)" }} className="mono">{c[2]}</td>
                <td style={td}>
                  {c[0] === "AAT_z" ? <Tag color="var(--crimson)">Target</Tag>
                    : ["x","y"].includes(c[0]) ? <Tag color="var(--purple)">Coord</Tag>
                    : c[0] === "id" ? <Tag color="var(--muted)">ID</Tag>
                    : <Tag color="var(--ink-2)">Predictor</Tag>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}
const th = { padding: "6px 8px", fontWeight: 600, fontSize: 10, letterSpacing: "0.1em", textTransform: "uppercase" };
const td = { padding: "7px 8px", fontSize: 12 };

function Stat({ label, value, tint, sub }) {
  return (
    <div style={{ border: "1px solid var(--line)", borderRadius: 8, padding: "10px 14px", background: "#fff", position: "relative", overflow: "hidden" }}>
      <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", letterSpacing: "0.12em", textTransform: "uppercase" }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, letterSpacing: "-0.02em", color: tint, marginTop: 2 }}>{value}</div>
      {sub && <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

// ---------- DAG ----------
function DAGPage() {
  return (
    <div>
      <SectionHeader kicker="04 · analysis" label="Causal DAG" right={<Btn small>Add edge</Btn>} />
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 14 }}>
        <Card title="Directed acyclic graph" subtitle="3 treatments · 1 mediator · 2 confounders · 1 outcome">
          <DagMini />
          <div style={{ display: "flex", gap: 14, marginTop: 10 }}>
            <LegendDot color="var(--crimson)" label="Treatment" />
            <LegendDot color="var(--purple)" label="Mediator" />
            <LegendDot color="var(--muted)" label="Confounder" />
            <LegendDot color="var(--ink)" label="Outcome" />
          </div>
        </Card>

        <Card title="Structural coefficients" subtitle="DML · 5-fold">
          {[
            ["Canopy → AAT_z", -0.022, "var(--crimson)"],
            ["Impervious → AAT_z", +0.022, "var(--crimson)"],
            ["NDVI → AAT_z", -4.131, "var(--purple)"],
            ["Albedo → AAT_z", -2.759, "var(--amber)"],
            ["Canopy → NDVI", +0.003, "var(--muted)"],
            ["Canopy → Impervious", -0.630, "var(--muted)"],
          ].map(([l, v, c]) => (
            <div key={l} style={{ display: "grid", gridTemplateColumns: "1fr 64px", gap: 8, padding: "6px 0", borderTop: "1px dashed var(--line)" }}>
              <span style={{ fontSize: 12 }}>{l}</span>
              <span className="mono" style={{ fontSize: 11.5, fontWeight: 700, textAlign: "right", color: v < 0 ? c : "var(--crimson)" }}>
                {v > 0 ? "+" : ""}{v.toFixed(3)}
              </span>
            </div>
          ))}
        </Card>
      </div>
    </div>
  );
}

function LegendDot({ color, label }) {
  return <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 11 }}>
    <span style={{ width: 10, height: 10, border: `1.5px solid ${color}`, background: "#fff", borderRadius: 2 }}/>
    <span className="mono" style={{ color: "var(--muted)", letterSpacing: "0.08em", textTransform: "uppercase", fontSize: 10 }}>{label}</span>
  </span>;
}

// ---------- RUN ----------
function RunPage() {
  const stages = [
    { n: 0, name: "Correlogram Analysis", status: "done", time: "00:14" },
    { n: 1, name: "GWEN Variable Selection", status: "done", time: "02:37" },
    { n: 2, name: "Spatial Cross-Validation", status: "done", time: "18:42" },
    { n: 3, name: "Causal Validation", status: "running", time: "04:12", progress: 0.62 },
    { n: 4, name: "Scenario Simulation", status: "queued", time: "—" },
  ];
  return (
    <div>
      <SectionHeader kicker="11 · pipeline" label="Run"
        right={<div style={{ display: "flex", gap: 8 }}>
          <Btn small>Cancel</Btn>
          <Btn primary small>Re-run</Btn>
        </div>} />

      <div style={{ display: "grid", gridTemplateColumns: "1.1fr 1fr", gap: 14 }}>
        <Card title="Stages" subtitle="5-stage pipeline">
          {stages.map(s => (
            <div key={s.n} style={{
              display: "grid", gridTemplateColumns: "30px 1fr 68px 70px",
              alignItems: "center", gap: 10, padding: "8px 0",
              borderTop: s.n > 0 ? "1px dashed var(--line)" : "none",
            }}>
              <span className="mono" style={{
                display: "inline-flex", alignItems: "center", justifyContent: "center",
                width: 26, height: 26, borderRadius: 4,
                background: s.status === "done" ? "var(--ink)" : s.status === "running" ? "var(--crimson)" : "rgba(0,0,0,0.05)",
                color: s.status === "queued" ? "var(--muted)" : "#fff",
                fontSize: 11, fontWeight: 700,
              }}>{s.n}</span>
              <div>
                <div style={{ fontSize: 12.5, fontWeight: 600 }}>{s.name}</div>
                {s.status === "running" && (
                  <div style={{ height: 4, background: "rgba(0,0,0,0.05)", borderRadius: 2, marginTop: 4, overflow: "hidden" }}>
                    <div style={{ width: `${s.progress * 100}%`, height: "100%", background: "var(--crimson)" }}/>
                  </div>
                )}
              </div>
              <Tag color={s.status === "done" ? "var(--ink)" : s.status === "running" ? "var(--crimson)" : "var(--muted)"}>
                {s.status}
              </Tag>
              <span className="mono" style={{ fontSize: 10.5, textAlign: "right", color: "var(--muted)" }}>{s.time}</span>
            </div>
          ))}
        </Card>

        <Card title="Live terminal" subtitle="stage 3 · dml backdoor" padding={0}
          style={{ minHeight: 280 }}>
          <div style={{
            background: "#1a1416", color: "#e6ddcb", fontFamily: "JetBrains Mono, monospace",
            fontSize: 10.5, lineHeight: 1.55, padding: "12px 14px",
            height: "100%", overflow: "hidden",
            borderRadius: "0 0 8px 8px",
          }}>
            <TermLine t="var(--muted)">[00:00:02] loading project.yml …</TermLine>
            <TermLine t="var(--amber)">[00:00:03] target: AAT_z ← 6 predictors</TermLine>
            <TermLine>[stage 0] Moran's I @ lag 30m … 0.842</TermLine>
            <TermLine>[stage 0] ↳ auto bandwidth = 180 m · block = 420 m</TermLine>
            <TermLine>[stage 1] GWEN rank  1 Pct_Impervious   0.512</TermLine>
            <TermLine>[stage 1] GWEN rank  2 Pct_Canopy       0.378</TermLine>
            <TermLine>[stage 1] GWEN rank  3 NDVI             0.301</TermLine>
            <TermLine t="var(--gold)">[stage 2] OOF R² → OLS 0.294 · GWR 0.828 · GWRF 0.898</TermLine>
            <TermLine t="var(--gold)">[stage 2] meta-ensemble (enhanced) = 0.915 ✓</TermLine>
            <TermLine>[stage 3] DML fold 3/5 ATE(Canopy) = −0.015</TermLine>
            <TermLine>[stage 3] refutation: placebo   p=0.83 ✓</TermLine>
            <TermLine>[stage 3] refutation: subset    p=0.71 ✓</TermLine>
            <TermLine t="var(--crimson)">[stage 3] running causal forest … <Blink/></TermLine>
          </div>
        </Card>
      </div>
    </div>
  );
}
function TermLine({ t = "#e6ddcb", children }) {
  return <div style={{ color: t }}>{children}</div>;
}
function Blink() {
  return <span style={{ display: "inline-block", width: 7, height: 12, background: "currentColor", verticalAlign: "middle", marginLeft: 2, animation: "blink 1s steps(1) infinite" }}/>;
}

// ---------- RESULTS ----------
function ResultsPage({ scenario, setScenario }) {
  return (
    <div>
      <SectionHeader kicker="12 · pipeline" label="Results"
        right={<div style={{ display: "flex", gap: 8 }}>
          <Btn small>Export CSV</Btn>
          <Btn small>Open in map</Btn>
        </div>} />

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 10, marginBottom: 14 }}>
        <Stat label="R² (enhanced)" value="0.915" sub="+0.012 vs. std" tint="var(--crimson)" />
        <Stat label="RMSE" value="0.500" sub="z-score units" tint="var(--ink)" />
        <Stat label="E-value · Impv." value="2.47" sub="strong robustness" tint="var(--purple)" />
        <Stat label="MC draws" value="500" sub="5th / 50th / 95th" tint="var(--amber)" />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 14 }}>
        <Card
          title="Scenario map"
          subtitle={{
            "baseline": "baseline · AAT_z",
            "canopy+10": "canopy +10 pp · ΔAAT_z",
            "impervious-20": "impervious −20 pp · ΔAAT_z",
            "albedo+0.1": "albedo +0.10 · ΔAAT_z",
          }[scenario]}
          actions={
            <div style={{ display: "flex", gap: 4 }}>
              {[
                ["baseline", "Base"],
                ["canopy+10", "Canopy +10"],
                ["impervious-20", "Impv −20"],
                ["albedo+0.1", "Albedo +0.10"],
              ].map(([k, l]) => (
                <button key={k} onClick={() => setScenario(k)}
                  style={{
                    border: "1px solid " + (scenario === k ? "var(--ink)" : "var(--line)"),
                    background: scenario === k ? "var(--ink)" : "#fff",
                    color: scenario === k ? "#fff" : "var(--ink-2)",
                    fontSize: 10.5, padding: "3px 8px", borderRadius: 4,
                    fontFamily: "inherit", fontWeight: 600, cursor: "pointer",
                  }}>{l}</button>
              ))}
            </div>
          }
          padding={0}
          style={{ overflow: "hidden" }}
        >
          <div style={{ position: "relative", height: 320 }}>
            <SpatialMap scenario={scenario} />
            <div style={{ position: "absolute", left: 10, bottom: 10, right: 10, background: "rgba(255,255,255,0.92)", border: "1px solid var(--line)", borderRadius: 4, padding: "6px 10px" }}>
              <RampLegend
                label={scenario === "baseline" ? "Air Temperature (z-score)" : "ΔTemperature (z-score)"}
                min={scenario === "baseline" ? "−2.4" : "−0.8"}
                max={scenario === "baseline" ? "+3.1" : "+0.2"}
              />
            </div>
            <div className="mono" style={{ position: "absolute", top: 8, right: 10, fontSize: 9.5, color: "var(--ink-2)", background: "rgba(255,255,255,0.85)", padding: "2px 6px", borderRadius: 3 }}>
              N ↑ · 30 m · EPSG:3438
            </div>
          </div>
        </Card>

        <div style={{ display: "grid", gridTemplateRows: "auto 1fr", gap: 14 }}>
          <Card title="Model R²" subtitle="out-of-fold, spatial CV">
            <ModelBarChart />
          </Card>
          <Card title="Intervention response" subtitle="mean Δ AAT_z by lever magnitude">
            <ScenarioCurve />
          </Card>
        </div>
      </div>
    </div>
  );
}

// ---------- Placeholder for less-important pages ----------
function Placeholder({ title, kicker, description }) {
  return (
    <div>
      <SectionHeader kicker={kicker} label={title}/>
      <div style={{
        border: "1px dashed var(--line)",
        background: "repeating-linear-gradient(135deg, rgba(0,0,0,0.015) 0 8px, transparent 8px 16px)",
        borderRadius: 8, padding: 40, textAlign: "center",
      }}>
        <div className="mono" style={{ fontSize: 10, color: "var(--muted)", letterSpacing: "0.14em", textTransform: "uppercase" }}>
          view
        </div>
        <div style={{ fontSize: 16, fontWeight: 700, marginTop: 6 }}>{title}</div>
        <div style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6, maxWidth: 420, margin: "6px auto 0" }}>
          {description}
        </div>
      </div>
    </div>
  );
}

window.ProjectPage = ProjectPage;
window.DataPage = DataPage;
window.DAGPage = DAGPage;
window.RunPage = RunPage;
window.ResultsPage = ResultsPage;
window.Placeholder = Placeholder;
