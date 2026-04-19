// Results view — charts + spatial map preview using canvas.

const { useEffect: useEffectR, useRef: useRefR, useState: useStateR } = React;

const SPARC_HEX = ["#602468", "#9e337d", "#e94d9b", "#e94461", "#e73c25", "#e76c25", "#e79024", "#f0b632", "#fbdd46"];

// Spatial heatmap — pseudo-UHI temperature field over Providence grid
function SpatialMap({ scenario }) {
  const ref = useRefR(null);
  useEffectR(() => {
    const canvas = ref.current; if (!canvas) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d"); ctx.scale(DPR, DPR);

    // Build a noise field from a few gaussian hotspots
    const hotspots = [
      { x: 0.35, y: 0.45, s: 0.18, a: 1.0 },
      { x: 0.62, y: 0.38, s: 0.14, a: 0.85 },
      { x: 0.78, y: 0.64, s: 0.16, a: 0.7 },
      { x: 0.22, y: 0.72, s: 0.20, a: 0.6 },
      { x: 0.55, y: 0.78, s: 0.12, a: 0.9 },
    ];
    const coolspots = [
      { x: 0.15, y: 0.30, s: 0.18, a: 0.8 },
      { x: 0.85, y: 0.20, s: 0.12, a: 0.7 },
    ];
    const cell = 6;
    // Scenario shift
    const shift = scenario === "canopy+10" ? -0.18
                 : scenario === "impervious-20" ? -0.25
                 : scenario === "albedo+0.1" ? -0.12
                 : 0;

    for (let y = 0; y < h; y += cell) {
      for (let x = 0; x < w; x += cell) {
        const nx = x / w, ny = y / h;
        let v = 0;
        for (const hs of hotspots) {
          const dx = nx - hs.x, dy = ny - hs.y;
          v += hs.a * Math.exp(-(dx*dx + dy*dy) / (2 * hs.s * hs.s));
        }
        for (const cs of coolspots) {
          const dx = nx - cs.x, dy = ny - cs.y;
          v -= cs.a * Math.exp(-(dx*dx + dy*dy) / (2 * cs.s * cs.s));
        }
        v += shift;
        const t = Math.max(0, Math.min(1, (v + 0.5) / 1.6));
        // sample SPARC ramp
        const idx = t * (SPARC_HEX.length - 1);
        const i0 = Math.floor(idx), i1 = Math.min(SPARC_HEX.length - 1, i0 + 1);
        const f = idx - i0;
        const c0 = SPARC_HEX[i0], c1 = SPARC_HEX[i1];
        const r = Math.round(parseInt(c0.slice(1, 3), 16) * (1-f) + parseInt(c1.slice(1, 3), 16) * f);
        const g = Math.round(parseInt(c0.slice(3, 5), 16) * (1-f) + parseInt(c1.slice(3, 5), 16) * f);
        const b = Math.round(parseInt(c0.slice(5, 7), 16) * (1-f) + parseInt(c1.slice(5, 7), 16) * f);
        ctx.fillStyle = `rgba(${r},${g},${b},0.82)`;
        ctx.fillRect(x, y, cell, cell);
      }
    }
    // Overlay street-grid-ish lines
    ctx.strokeStyle = "rgba(0,0,0,0.08)"; ctx.lineWidth = 1;
    for (let i = 0; i < 14; i++) {
      ctx.beginPath(); ctx.moveTo((i/14)*w, 0); ctx.lineTo((i/14)*w + 20, h); ctx.stroke();
    }
    for (let i = 0; i < 10; i++) {
      ctx.beginPath(); ctx.moveTo(0, (i/10)*h); ctx.lineTo(w, (i/10)*h - 12); ctx.stroke();
    }
    // Border dashes (brand)
    ctx.setLineDash([3, 3]); ctx.strokeStyle = "rgba(0,0,0,0.45)"; ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
  }, [scenario]);
  return <canvas ref={ref} style={{ width: "100%", height: "100%", display: "block" }} />;
}

// Bar chart — model R² comparison
function ModelBarChart() {
  const data = [
    { name: "OLS", r2: 0.294 },
    { name: "GWR", r2: 0.828 },
    { name: "GWRF", r2: 0.898 },
    { name: "GGPGAM", r2: 0.839 },
    { name: "Meta (std)", r2: 0.902 },
    { name: "Meta (+L)", r2: 0.915, hi: true },
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, padding: "4px 0" }}>
      {data.map(d => (
        <div key={d.name} style={{ display: "grid", gridTemplateColumns: "92px 1fr 44px", alignItems: "center", gap: 10 }}>
          <span className="mono" style={{ fontSize: 10.5, color: "var(--ink-2)" }}>{d.name}</span>
          <div style={{ height: 14, background: "rgba(0,0,0,0.05)", borderRadius: 2, position: "relative", overflow: "hidden" }}>
            <div style={{
              position: "absolute", top: 0, left: 0, bottom: 0,
              width: `${d.r2 * 100}%`,
              background: d.hi ? "var(--crimson)" : "var(--ink-2)",
            }}/>
            {d.hi && <div style={{ position: "absolute", inset: 0, backgroundImage: "repeating-linear-gradient(-45deg, transparent 0 3px, rgba(255,255,255,0.15) 3px 4px)" }}/>}
          </div>
          <span className="mono" style={{ fontSize: 10.5, textAlign: "right", fontWeight: 600, color: d.hi ? "var(--crimson)" : "var(--ink-2)" }}>
            {d.r2.toFixed(3)}
          </span>
        </div>
      ))}
    </div>
  );
}

// Scenario response curve
function ScenarioCurve() {
  const canvas = useRefR(null);
  useEffectR(() => {
    const c = canvas.current; if (!c) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = c.clientWidth, h = c.clientHeight;
    c.width = w * DPR; c.height = h * DPR;
    const ctx = c.getContext("2d"); ctx.scale(DPR, DPR);
    const PAD = { l: 36, r: 10, t: 10, b: 22 };
    const xs = [0, 5, 10, 15, 20, 30, 50];
    const canopy = [0, -0.130, -0.258, -0.437, -0.509, -0.608, -0.738];
    const imperv = [0, -0.098, -0.195, null, -0.383, -0.456, -0.550];
    const albedo = [0, -0.098, -0.196, null, -0.384, -0.450, null];
    const xMax = 50, yMin = -0.8, yMax = 0.02;
    const X = (x) => PAD.l + (x / xMax) * (w - PAD.l - PAD.r);
    const Y = (y) => PAD.t + ((yMax - y) / (yMax - yMin)) * (h - PAD.t - PAD.b);
    // axes
    ctx.strokeStyle = "rgba(0,0,0,0.15)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD.l, PAD.t); ctx.lineTo(PAD.l, h - PAD.b); ctx.lineTo(w - PAD.r, h - PAD.b); ctx.stroke();
    // grid
    for (let y = -0.8; y <= 0; y += 0.2) {
      ctx.strokeStyle = "rgba(0,0,0,0.05)";
      ctx.beginPath(); ctx.moveTo(PAD.l, Y(y)); ctx.lineTo(w - PAD.r, Y(y)); ctx.stroke();
      ctx.fillStyle = "var(--muted)"; ctx.font = "10px JetBrains Mono";
      ctx.textAlign = "right"; ctx.fillText(y.toFixed(1), PAD.l - 6, Y(y) + 3);
    }
    xs.forEach(x => {
      ctx.fillStyle = "var(--muted)"; ctx.font = "10px JetBrains Mono";
      ctx.textAlign = "center"; ctx.fillText(String(x), X(x), h - PAD.b + 13);
    });

    const series = [
      { data: canopy, color: "#602468", label: "Canopy +X pp" },
      { data: imperv, color: "#e73c25", label: "Impervious −X pp" },
      { data: albedo, color: "#e79024", label: "Albedo +X" },
    ];
    for (const s of series) {
      ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.beginPath();
      let started = false;
      xs.forEach((x, i) => {
        if (s.data[i] == null) return;
        const px = X(x), py = Y(s.data[i]);
        if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
      });
      ctx.stroke();
      xs.forEach((x, i) => {
        if (s.data[i] == null) return;
        ctx.fillStyle = s.color;
        ctx.beginPath(); ctx.arc(X(x), Y(s.data[i]), 3, 0, Math.PI*2); ctx.fill();
      });
    }
    // legend
    const lx = w - 150, ly = 18;
    series.forEach((s, i) => {
      ctx.fillStyle = s.color;
      ctx.fillRect(lx, ly + i*14 - 6, 10, 2);
      ctx.fillStyle = "var(--ink-2)"; ctx.font = "10px JetBrains Mono"; ctx.textAlign = "left";
      ctx.fillText(s.label, lx + 14, ly + i*14);
    });
  }, []);
  return <canvas ref={canvas} style={{ width: "100%", height: 220, display: "block" }} />;
}

// DAG mini
function DagMini() {
  const nodes = [
    { id: "canopy", x: 60, y: 60, label: "Canopy", type: "t" },
    { id: "impv", x: 60, y: 130, label: "Impervious", type: "t" },
    { id: "albedo", x: 60, y: 200, label: "Albedo", type: "t" },
    { id: "ndvi", x: 220, y: 95, label: "NDVI", type: "m" },
    { id: "elev", x: 220, y: 165, label: "Elevation", type: "c" },
    { id: "water", x: 220, y: 225, label: "Dist. water", type: "c" },
    { id: "aat", x: 380, y: 130, label: "AAT_z", type: "o" },
  ];
  const edges = [
    ["canopy", "ndvi"], ["canopy", "aat"], ["impv", "aat"], ["albedo", "aat"],
    ["ndvi", "aat"], ["elev", "aat"], ["water", "aat"], ["canopy", "impv"],
  ];
  const N = Object.fromEntries(nodes.map(n => [n.id, n]));
  const color = (t) => t === "t" ? "var(--crimson)" : t === "m" ? "var(--purple)" : t === "o" ? "var(--ink)" : "var(--muted)";
  return (
    <svg viewBox="0 0 460 280" style={{ width: "100%", height: 260 }}>
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="rgba(0,0,0,0.45)" />
        </marker>
      </defs>
      {edges.map(([a, b], i) => (
        <line key={i} x1={N[a].x + 34} y1={N[a].y} x2={N[b].x - 34} y2={N[b].y}
          stroke="rgba(0,0,0,0.35)" strokeWidth="1" strokeDasharray="3 3" markerEnd="url(#arr)" />
      ))}
      {nodes.map(n => (
        <g key={n.id}>
          <rect x={n.x - 42} y={n.y - 14} width="84" height="28" rx="3" fill="#fff"
            stroke={color(n.type)} strokeWidth={n.id === "aat" ? 2 : 1.2} />
          <text x={n.x} y={n.y + 4} textAnchor="middle" fontSize="11" fontFamily="JetBrains Mono" fill="var(--ink-2)">
            {n.label}
          </text>
        </g>
      ))}
    </svg>
  );
}

// Legend strip for spatial map
function RampLegend({ label = "ΔTemperature (z-score)", min = "−1.2", max = "+1.2" }) {
  return (
    <div>
      <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ height: 10, borderRadius: 2, background: `linear-gradient(90deg, ${SPARC_HEX.join(", ")})` }}/>
      <div className="mono" style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--muted)", marginTop: 2 }}>
        <span>{min}</span><span>0</span><span>{max}</span>
      </div>
    </div>
  );
}

window.SpatialMap = SpatialMap;
window.ModelBarChart = ModelBarChart;
window.ScenarioCurve = ScenarioCurve;
window.DagMini = DagMini;
window.RampLegend = RampLegend;
