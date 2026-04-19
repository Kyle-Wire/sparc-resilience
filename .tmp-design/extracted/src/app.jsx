// Main app shell — window chrome, sidebar, routing, splash, tweaks.

const { useState, useEffect } = React;

const TWEAKS = /*EDITMODE-BEGIN*/{
  "logoHue": "ink",
  "logoDensity": 1,
  "paperTone": "warm",
  "accent": "crimson"
}/*EDITMODE-END*/;

function Splash({ onReady }) {
  useEffect(() => {
    const t = setTimeout(onReady, 1400);
    return () => clearTimeout(t);
  }, []);
  return (
    <div style={{
      width: "100%", height: "100%",
      display: "flex", alignItems: "center", justifyContent: "center",
      flexDirection: "column", gap: 22,
    }}>
      <div style={{ width: 180, height: 180 }}>
        <CubeLogo size={180} density={1.4} hue="ink" />
      </div>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em" }}>SPARC LABS</div>
        <div className="mono" style={{ fontSize: 10, letterSpacing: "0.2em", color: "var(--muted)", marginTop: 4 }}>
          SPATIAL ANALYSIS & RESEARCH CORE · v0.4.2
        </div>
      </div>
      <div style={{ width: 220, height: 3, background: "rgba(0,0,0,0.08)", borderRadius: 2, overflow: "hidden" }}>
        <div style={{ width: "100%", height: "100%", background: "var(--crimson)", animation: "loadBar 1.3s ease-out" }}/>
      </div>
      <style>{`
        @keyframes loadBar { from { transform: translateX(-100%); } to { transform: translateX(0); } }
        @keyframes blink { 50% { opacity: 0; } }
      `}</style>
    </div>
  );
}

function ChatPanel({ onClose }) {
  const [msgs, setMsgs] = useState([
    { role: "assistant", text: "I can wire up your DAG from natural language. Try: 'Canopy and impervious affect air temperature, mediated by NDVI.'" },
  ]);
  const [input, setInput] = useState("");
  const send = () => {
    if (!input.trim()) return;
    const u = { role: "user", text: input };
    const replies = [
      "Proposing DAG edges: Canopy→AAT, Impervious→AAT, Canopy→NDVI→AAT. Open DAG view to accept.",
      "Added monotone constraint: Pct_Canopy (−) on AAT_z. Verified against Providence UHI priors.",
      "Scenario written: canopy +10 pp · predicted mean ΔAAT_z = −0.258 (σ = 0.154).",
    ];
    setMsgs(m => [...m, u, { role: "assistant", text: replies[m.length % replies.length] }]);
    setInput("");
  };
  return (
    <div style={{
      position: "absolute", left: 228, bottom: 0, width: 360, height: 420,
      background: "#fff", border: "1px solid var(--line)", borderRadius: "8px 8px 0 0",
      display: "flex", flexDirection: "column", zIndex: 40,
      boxShadow: "0 -8px 24px rgba(0,0,0,0.08)",
      animation: "slideUp 0.22s ease-out",
    }}>
      <style>{`@keyframes slideUp { from { transform: translateY(16px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }`}</style>
      <div style={{ padding: "10px 14px", borderBottom: "1px solid var(--line)", display: "flex", alignItems: "center" }}>
        <div style={{ width: 22, height: 22, marginRight: 8 }}>
          <CubeLogo size={22} density={0.5} />
        </div>
        <div>
          <div style={{ fontSize: 12, fontWeight: 700 }}>SPARC Assistant</div>
          <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)" }}>claude-haiku-4.5 · dag mode</div>
        </div>
        <button onClick={onClose} style={{ marginLeft: "auto", border: "none", background: "transparent", fontSize: 16, cursor: "pointer", color: "var(--muted)" }}>×</button>
      </div>
      <div className="scroll" style={{ flex: 1, overflowY: "auto", padding: 12, display: "flex", flexDirection: "column", gap: 8 }}>
        {msgs.map((m, i) => (
          <div key={i} style={{
            alignSelf: m.role === "user" ? "flex-end" : "flex-start",
            maxWidth: "85%",
            background: m.role === "user" ? "var(--ink)" : "#f7f4ee",
            color: m.role === "user" ? "#fff" : "var(--ink-2)",
            padding: "7px 10px", borderRadius: 7,
            fontSize: 12, lineHeight: 1.45,
          }}>{m.text}</div>
        ))}
      </div>
      <div style={{ padding: 10, borderTop: "1px solid var(--line)", display: "flex", gap: 6 }}>
        <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === "Enter" && send()}
          placeholder="Ask about your DAG, physics, or scenarios…"
          style={{ flex: 1, padding: "7px 10px", border: "1px solid var(--line)", borderRadius: 5, fontSize: 12, fontFamily: "inherit" }}/>
        <button onClick={send} style={{ background: "var(--ink)", color: "#fff", border: "none", padding: "0 12px", borderRadius: 5, fontSize: 12, fontWeight: 600, cursor: "pointer", fontFamily: "inherit" }}>Send</button>
      </div>
    </div>
  );
}

function TweaksPanel({ tweaks, setTweaks, onClose }) {
  const set = (k, v) => setTweaks(t => ({ ...t, [k]: v }));
  return (
    <div style={{
      position: "fixed", right: 20, bottom: 20, width: 260,
      background: "#fff", border: "1px solid var(--ink)", borderRadius: 8,
      zIndex: 60, boxShadow: "0 12px 32px rgba(0,0,0,0.18)",
      fontFamily: "inherit",
    }}>
      <div style={{ padding: "10px 12px", borderBottom: "1px solid var(--line)", display: "flex", alignItems: "center", background: "var(--ink)", color: "#fff", borderRadius: "8px 8px 0 0" }}>
        <span className="mono" style={{ fontSize: 10, letterSpacing: "0.18em", fontWeight: 700 }}>TWEAKS</span>
        <button onClick={onClose} style={{ marginLeft: "auto", color: "#fff", background: "transparent", border: "none", cursor: "pointer", fontSize: 14 }}>×</button>
      </div>
      <div style={{ padding: 12, display: "flex", flexDirection: "column", gap: 12 }}>
        <TweakField label="Logo colour">
          <div style={{ display: "flex", gap: 4 }}>
            {["ink", "red", "purple", "amber"].map(h => (
              <button key={h} onClick={() => set("logoHue", h)} style={{
                flex: 1, padding: "5px 0", fontSize: 10.5, fontFamily: "inherit",
                border: "1px solid " + (tweaks.logoHue === h ? "var(--ink)" : "var(--line)"),
                background: tweaks.logoHue === h ? "var(--ink)" : "#fff",
                color: tweaks.logoHue === h ? "#fff" : "var(--ink-2)",
                borderRadius: 4, cursor: "pointer", textTransform: "uppercase", letterSpacing: "0.05em",
              }}>{h}</button>
            ))}
          </div>
        </TweakField>
        <TweakField label={`Matter density · ${tweaks.logoDensity.toFixed(2)}`}>
          <input type="range" min="0.3" max="2" step="0.05" value={tweaks.logoDensity}
            onChange={e => set("logoDensity", parseFloat(e.target.value))} style={{ width: "100%" }}/>
        </TweakField>
        <TweakField label="Paper tone">
          <div style={{ display: "flex", gap: 4 }}>
            {["warm", "cool", "white"].map(t => (
              <button key={t} onClick={() => set("paperTone", t)} style={{
                flex: 1, padding: "5px 0", fontSize: 10.5, fontFamily: "inherit",
                border: "1px solid " + (tweaks.paperTone === t ? "var(--ink)" : "var(--line)"),
                background: tweaks.paperTone === t ? "var(--ink)" : "#fff",
                color: tweaks.paperTone === t ? "#fff" : "var(--ink-2)",
                borderRadius: 4, cursor: "pointer", textTransform: "uppercase", letterSpacing: "0.05em",
              }}>{t}</button>
            ))}
          </div>
        </TweakField>
      </div>
    </div>
  );
}
function TweakField({ label, children }) {
  return (
    <div>
      <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 5 }}>{label}</div>
      {children}
    </div>
  );
}

function App() {
  const [booting, setBooting] = useState(true);
  const [page, setPage] = useState("Results");
  const [chatOpen, setChatOpen] = useState(false);
  const [scenario, setScenario] = useState("canopy+10");
  const [tweaks, setTweaksState] = useState(TWEAKS);
  const [tweaksOpen, setTweaksOpen] = useState(false);
  const setTweaks = (fn) => {
    setTweaksState(prev => {
      const next = typeof fn === "function" ? fn(prev) : fn;
      try {
        window.parent?.postMessage({ type: "__edit_mode_set_keys", edits: next }, "*");
      } catch(e) {}
      return next;
    });
  };

  // Paper tone override
  useEffect(() => {
    const r = document.documentElement.style;
    if (tweaks.paperTone === "cool") { r.setProperty("--paper", "#f3f4f2"); r.setProperty("--line", "#c5cac4"); }
    else if (tweaks.paperTone === "white") { r.setProperty("--paper", "#ffffff"); r.setProperty("--line", "#d8d4cb"); }
    else { r.setProperty("--paper", "#f7f4ee"); r.setProperty("--line", "#c9c2b3"); }
  }, [tweaks.paperTone]);

  // Edit mode protocol
  useEffect(() => {
    const onMsg = (e) => {
      if (e.data?.type === "__activate_edit_mode") setTweaksOpen(true);
      if (e.data?.type === "__deactivate_edit_mode") setTweaksOpen(false);
    };
    window.addEventListener("message", onMsg);
    window.parent?.postMessage({ type: "__edit_mode_available" }, "*");
    return () => window.removeEventListener("message", onMsg);
  }, []);

  // Override logo params by re-rendering CubeLogo with props on each page render:
  // We do this by passing through pages via a context-free approach:
  window.__LOGO_HUE = tweaks.logoHue;
  window.__LOGO_DENSITY = tweaks.logoDensity;

  const renderPage = () => {
    switch (page) {
      case "Project": return <ProjectPage />;
      case "Data": return <DataPage />;
      case "DAG": return <DAGPage />;
      case "Run": return <RunPage />;
      case "Results": return <ResultsPage scenario={scenario} setScenario={setScenario} />;
      case "Processing": return <Placeholder kicker="03 · setup" title="Data Processing" description="Clean, derive, and standardize variables before training. Missing-value strategies, CRS reprojection, and fold-aware spatial joins." />;
      case "Variables": return <Placeholder kicker="05 · analysis" title="Variables" description="Select predictors, inspect distributions, and mark actionable vs. fixed levers for scenarios." />;
      case "Physics": return <Placeholder kicker="06 · analysis" title="Physics" description="Monotone constraints, priors, diminishing-return tapers, and caps for physical guardrails." />;
      case "CRS": return <Placeholder kicker="07 · analysis" title="CRS" description="Input EPSG, projected EPSG, and equal-area transforms for global studies." />;
      case "Scenarios": return <Placeholder kicker="08 · analysis" title="Scenarios" description="Define single- and joint-variable interventions. Defaults from template." />;
      case "Models": return <Placeholder kicker="09 · analysis" title="Models" description="OLS · GWR · GWRF · GGPGAM · Meta-ensemble. Per-model hyperparameters." />;
      case "Report": return <Placeholder kicker="13 · pipeline" title="Report" description="Generate the narrative PDF/HTML report with all stage outputs and refutation tables." />;
      default: return null;
    }
  };

  return (
    <>
      <WindowChrome>
        {booting ? <Splash onReady={() => setBooting(false)} /> : (
          <>
            <Sidebar currentPage={page} onNavigate={setPage} onToggleChat={() => setChatOpen(o => !o)} chatOpen={chatOpen} />
            <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
              <Topbar page={page}/>
              <main className="scroll" style={{
                flex: 1, overflow: "auto", padding: "20px 22px",
                background: "var(--paper)",
                backgroundImage:
                  "linear-gradient(rgba(0,0,0,0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,0.035) 1px, transparent 1px)",
                backgroundSize: "24px 24px",
                position: "relative",
              }}>
                {renderPage()}
              </main>
            </div>
            {chatOpen && <ChatPanel onClose={() => setChatOpen(false)}/>}
          </>
        )}
      </WindowChrome>
      {tweaksOpen && <TweaksPanel tweaks={tweaks} setTweaks={setTweaks} onClose={() => setTweaksOpen(false)} />}
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App/>);
