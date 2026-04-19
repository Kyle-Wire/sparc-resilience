// ===== src/cube_logo.jsx =====
// Animated SPARC cube logo.
// The cube contains "matter" — flowing dashed contour rings + particles — that
// moves around inside the isometric cube bounds.
//
// Geometry matches the official logo (viewBox around 138 48 516 516).
// Cube vertices (projected 2D):
//   TOP:    (396, 72)
//   TL:     (195, 190)   — top-left face
//   TR:     (597, 190)   — top-right face
//   C:      (396, 313)   — centre (where all three top edges meet the middle diamond)
//   BL:     (195, 418)
//   BR:     (597, 418)
//   BOT:    (396, 540)

const { useEffect, useRef, useMemo } = React;

// ------- Utility: noise-like pseudo-random field (for organic motion) -------
function hash(x, y, z) {
  const s = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453;
  return s - Math.floor(s);
}

// ------- Point in cube hexagon (2D silhouette) -------
// The cube projects to a hexagon; the three visible faces are three rhombi meeting at C.
const HEX = [
  [396, 72],   // top
  [597, 190],  // upper-right
  [597, 418],  // lower-right
  [396, 540],  // bottom
  [195, 418],  // lower-left
  [195, 190],  // upper-left
];
function pointInHex(px, py) {
  let inside = false;
  for (let i = 0, j = HEX.length - 1; i < HEX.length; j = i++) {
    const [xi, yi] = HEX[i];
    const [xj, yj] = HEX[j];
    const intersect = ((yi > py) !== (yj > py)) &&
      (px < ((xj - xi) * (py - yi)) / (yj - yi + 1e-9) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

// ------- Isometric mapping: cube coords (u,v,w) in [-1,1]^3 -> 2D projection ----
// axes: u=right (→TR), v=back-left (→TL), w=up
const C = [396, 313];
const AX_U = [ 201, -123 ];   // TR - C
const AX_V = [-201, -123 ];   // TL - C
const AX_W = [   0, -241 ];   // TOP - C
function iso(u, v, w) {
  return [
    C[0] + u * AX_U[0] + v * AX_V[0] + w * AX_W[0],
    C[1] + u * AX_U[1] + v * AX_V[1] + w * AX_W[1],
  ];
}

// ------- CUBE LOGO -------
function CubeLogo({ size = 120, animate = true, hue = "ink", density = 1 }) {
  const canvasRef = useRef(null);
  const rafRef = useRef(0);
  const startRef = useRef(performance.now());

  // Resolve stroke colour
  const stroke = hue === "ink" ? "#1a1416" :
                 hue === "red" ? "#e73c25" :
                 hue === "purple" ? "#602468" :
                 hue === "amber" ? "#e79024" : "#1a1416";

  // Pre-generate "matter" seeds — blobs that float around in the cube volume
  const blobs = useMemo(() => {
    const n = Math.max(3, Math.round(5 * density));
    const out = [];
    for (let i = 0; i < n; i++) {
      out.push({
        // base position in cube-space [-0.7,0.7]
        u0: (hash(i, 1, 7) - 0.5) * 1.2,
        v0: (hash(i, 2, 7) - 0.5) * 1.2,
        w0: (hash(i, 3, 7) - 0.5) * 1.2,
        // oscillation amps / phases
        au: 0.25 + hash(i, 4, 0) * 0.25,
        av: 0.25 + hash(i, 5, 0) * 0.25,
        aw: 0.25 + hash(i, 6, 0) * 0.25,
        pu: hash(i, 7, 0) * Math.PI * 2,
        pv: hash(i, 8, 0) * Math.PI * 2,
        pw: hash(i, 9, 0) * Math.PI * 2,
        // size of blob in projected px
        r: 60 + hash(i, 10, 0) * 55,
        // number of contour rings
        rings: 5 + Math.floor(hash(i, 11, 0) * 4),
        // rotation speed
        rot: (hash(i, 12, 0) - 0.5) * 0.6,
      });
    }
    return out;
  }, [density]);

  // Particles — small dashed "sparks" floating in the cube
  const particles = useMemo(() => {
    const n = Math.round(45 * density);
    const out = [];
    for (let i = 0; i < n; i++) {
      out.push({
        u0: (hash(i, 21, 3) - 0.5) * 1.4,
        v0: (hash(i, 22, 3) - 0.5) * 1.4,
        w0: (hash(i, 23, 3) - 0.5) * 1.4,
        au: 0.35 + hash(i, 24, 0) * 0.35,
        av: 0.35 + hash(i, 25, 0) * 0.35,
        aw: 0.35 + hash(i, 26, 0) * 0.35,
        pu: hash(i, 27, 0) * Math.PI * 2,
        pv: hash(i, 28, 0) * Math.PI * 2,
        pw: hash(i, 29, 0) * Math.PI * 2,
        spd: 0.2 + hash(i, 30, 0) * 0.6,
        size: 0.8 + hash(i, 31, 0) * 1.4,
      });
    }
    return out;
  }, [density]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    // High-DPI
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const cw = 792, chh = 612;  // logo native
    canvas.width = cw * DPR;
    canvas.height = chh * DPR;
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    ctx.scale(DPR, DPR);

    // Clip path: hexagonal silhouette of the cube
    const clip = () => {
      ctx.beginPath();
      ctx.moveTo(HEX[0][0], HEX[0][1]);
      for (let i = 1; i < HEX.length; i++) ctx.lineTo(HEX[i][0], HEX[i][1]);
      ctx.closePath();
      ctx.clip();
    };

    const drawCubeFrame = () => {
      ctx.strokeStyle = stroke;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.lineWidth = 7.56;

      // Outer hexagon (cube silhouette)
      ctx.beginPath();
      ctx.moveTo(HEX[0][0], HEX[0][1]);
      for (let i = 1; i < HEX.length; i++) ctx.lineTo(HEX[i][0], HEX[i][1]);
      ctx.closePath();
      ctx.stroke();

      // Y lines: three edges from centre going to TOP, TR, TL (well, to corners via midpts)
      // The three "visible" inner edges go from C to top / bottom / but in ISO cube it's
      // from C to: TOP, BL, BR — forming a Y rotated. Actually for typical iso cube with
      // top visible, inner seams are C→TOP, C→BL, C→BR.
      ctx.beginPath();
      // C → TOP (straight up) — partial (matches the short white tick in logo)
      // but we also need the full back-edge of the top face. The logo shows:
      //   front-right face edge: C→BR edge partly via mid, and the bottom vertical C→BOT
      // Let's do the standard iso cube internal seams: centre to three alt vertices
      // which are TOP (up), BR (down-right but hidden bottom edge), BL.
      // In our hex ordering, top face edges connect: TL-TOP, TOP-TR, TR-BR-back? Hmm.
      //
      // Simpler: draw the three inner lines: C→TOP, C→BL_diag, C→BR_diag via the three
      // face diagonals. For iso cube, the three inner lines go from C to TL, TR, BOT
      // (three of the six hex corners — the alternating set).
      ctx.moveTo(C[0], C[1]); ctx.lineTo(HEX[5][0], HEX[5][1]); // C→TL
      ctx.moveTo(C[0], C[1]); ctx.lineTo(HEX[1][0], HEX[1][1]); // C→TR
      ctx.moveTo(C[0], C[1]); ctx.lineTo(HEX[3][0], HEX[3][1]); // C→BOT
      ctx.stroke();

      // Short white accent ticks at TOP and sides (from the logo)
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 7.56;
      ctx.beginPath();
      ctx.moveTo(HEX[0][0], HEX[0][1]);
      ctx.lineTo(HEX[0][0], HEX[0][1] + 74);   // down from top vertex
      ctx.moveTo(HEX[1][0], HEX[1][1] + 230);  // right face vertical-ish tick
      ctx.lineTo(HEX[1][0] - 88, HEX[1][1] + 178);
      ctx.moveTo(HEX[5][0], HEX[5][1] + 228);
      ctx.lineTo(HEX[5][0] + 84, HEX[5][1] + 179);
      ctx.stroke();
    };

    const draw = (t) => {
      ctx.clearRect(0, 0, cw, chh);

      // Soft white wash inside the cube (the matter sits on white like the logo)
      ctx.save();
      clip();
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, cw, chh);

      // ---- draw flowing "matter" as overlapping dashed rings ----
      ctx.lineWidth = 1.13;
      ctx.strokeStyle = stroke;
      ctx.lineCap = "round";

      for (const b of blobs) {
        // animate cube-space position with overlapping sines
        const u = Math.max(-0.75, Math.min(0.75, b.u0 + Math.sin(t * 0.35 + b.pu) * b.au));
        const v = Math.max(-0.75, Math.min(0.75, b.v0 + Math.sin(t * 0.42 + b.pv) * b.av));
        const w = Math.max(-0.75, Math.min(0.75, b.w0 + Math.sin(t * 0.29 + b.pw) * b.aw));
        const [cx, cy] = iso(u, v, w);

        const theta = t * b.rot;
        // rings shrink inward with a slight breathing pulse
        for (let i = 0; i < b.rings; i++) {
          const frac = (i + 1) / (b.rings + 1);
          const r = b.r * frac * (0.9 + 0.1 * Math.sin(t * 1.2 + i * 0.8));
          // isoline-ish ellipse (iso-foreshortened), slightly deformed
          const a = r;
          const bb = r * 0.56;
          const dashOn = 2.97 + (i % 3) * 0.05;
          const dashOff = dashOn;
          ctx.setLineDash([dashOn, dashOff]);
          ctx.lineDashOffset = -t * 8 - i * 4;

          ctx.beginPath();
          const steps = 56;
          for (let k = 0; k <= steps; k++) {
            const ang = (k / steps) * Math.PI * 2;
            // deform radius with low-freq noise for organic shape
            const def = 1
              + 0.12 * Math.sin(ang * 3 + theta + i * 0.7)
              + 0.08 * Math.cos(ang * 5 - theta * 0.7 + i * 1.3);
            const ca = Math.cos(ang + theta * 0.3);
            const sa = Math.sin(ang + theta * 0.3);
            const x = cx + a * def * ca;
            const y = cy + bb * def * sa;
            if (k === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.closePath();
          ctx.stroke();
        }
      }

      // ---- sparks / particles ----
      ctx.setLineDash([]);
      for (const p of particles) {
        const u = Math.max(-0.85, Math.min(0.85, p.u0 + Math.sin(t * p.spd + p.pu) * p.au));
        const v = Math.max(-0.85, Math.min(0.85, p.v0 + Math.cos(t * p.spd * 1.1 + p.pv) * p.av));
        const w = Math.max(-0.85, Math.min(0.85, p.w0 + Math.sin(t * p.spd * 0.9 + p.pw) * p.aw));
        const [x, y] = iso(u, v, w);
        const depth = 0.5 + 0.5 * (w + 1) / 2;
        ctx.fillStyle = stroke;
        ctx.globalAlpha = 0.35 + 0.5 * depth;
        ctx.beginPath();
        ctx.arc(x, y, p.size, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      // ---- solid "core" blob (the thick white-filled region from the logo) ----
      // It sits on the right-back face, a softer organic mass
      ctx.save();
      ctx.beginPath();
      const cx0 = 380 + Math.sin(t * 0.3) * 16;
      const cy0 = 335 + Math.cos(t * 0.27) * 10;
      const steps = 72;
      for (let k = 0; k <= steps; k++) {
        const ang = (k / steps) * Math.PI * 2;
        const r = 118
          + 22 * Math.sin(ang * 3 + t * 0.5)
          + 12 * Math.cos(ang * 5 - t * 0.4)
          + 8  * Math.sin(ang * 7 + t * 0.9);
        const x = cx0 + r * Math.cos(ang);
        const y = cy0 + r * 0.72 * Math.sin(ang);
        if (k === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.lineWidth = 4.54;
      ctx.strokeStyle = stroke;
      ctx.stroke();
      ctx.restore();

      ctx.restore(); // end clip

      // ---- cube frame on top ----
      drawCubeFrame();
    };

    const tick = () => {
      const now = performance.now();
      const t = animate ? (now - startRef.current) / 1000 : 0;
      draw(t);
      if (animate) rafRef.current = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(rafRef.current);
  }, [animate, blobs, particles, stroke]);

  return (
    <div style={{ width: size, height: size, position: "relative", display: "inline-block" }}>
      <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />
    </div>
  );
}

window.CubeLogo = CubeLogo;


// ===== src/sidebar.jsx =====
// Sidebar + Topbar + Window chrome for the SPARC desktop app.

const SECTIONS = [
  { label: "Setup", pages: ["Project", "Data", "Processing"] },
  { label: "Analysis", pages: ["DAG", "Variables", "Physics", "CRS", "Scenarios", "Models"] },
  { label: "Pipeline", pages: ["Run", "Results", "Report"] },
];

function Sidebar({ currentPage, onNavigate, onToggleChat, chatOpen }) {
  let idx = 0;
  return (
    <aside style={{
      width: 224, flexShrink: 0, height: "100%",
      display: "flex", flexDirection: "column",
      background: "#fdfbf7",
      borderRight: "1px solid var(--line)",
    }}>
      {/* Brand lockup */}
      <div style={{
        display: "flex", alignItems: "center", gap: 12,
        padding: "14px 16px",
        borderBottom: "1px solid var(--line)",
      }}>
        <div style={{ width: 40, height: 40, marginTop: -2 }}>
          <CubeLogo size={40} density={0.6} />
        </div>
        <div style={{ display: "flex", flexDirection: "column", lineHeight: 1, gap: 3 }}>
          <span style={{ fontSize: 13, fontWeight: 800, letterSpacing: "-0.01em" }}>SPARC</span>
          <span style={{ fontSize: 9.5, fontWeight: 600, letterSpacing: "0.18em", color: "var(--muted)" }}>LABS</span>
        </div>
      </div>

      {/* Project pill */}
      <div style={{ padding: "10px 12px 4px" }}>
        <div style={{
          border: "1px dashed var(--line)",
          background: "#fff",
          borderRadius: 8,
          padding: "8px 10px",
        }}>
          <div className="mono" style={{ fontSize: 9, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.12em" }}>
            active project
          </div>
          <div style={{ fontSize: 12, fontWeight: 700, marginTop: 3, display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--crimson)" }}></span>
            Brown UHI — 30m
          </div>
          <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>uhi · EPSG:3438</div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="scroll" style={{ flex: 1, overflowY: "auto", padding: "6px 0 12px" }}>
        {SECTIONS.map((section, si) => (
          <div key={section.label} style={{ marginTop: si === 0 ? 4 : 10 }}>
            <div style={{
              padding: "4px 18px",
              fontSize: 9.5, fontWeight: 700,
              letterSpacing: "0.16em",
              color: "var(--muted)",
              textTransform: "uppercase",
              display: "flex", justifyContent: "space-between", alignItems: "center",
            }}>
              <span>{section.label}</span>
              <span className="mono" style={{ fontSize: 9, opacity: 0.6 }}>{String(si + 1).padStart(2, "0")}</span>
            </div>
            {section.pages.map((p) => {
              const n = ++idx;
              const active = p === currentPage;
              return (
                <button
                  key={p}
                  onClick={() => onNavigate(p)}
                  style={{
                    display: "flex", alignItems: "center", gap: 10,
                    width: "100%", textAlign: "left", border: "none",
                    padding: "7px 12px 7px 16px",
                    fontFamily: "inherit",
                    fontSize: 13,
                    cursor: "pointer",
                    background: active ? "var(--ink)" : "transparent",
                    color: active ? "#fff" : "var(--ink-2)",
                    fontWeight: active ? 600 : 500,
                    transition: "background 0.15s",
                  }}
                  onMouseEnter={(e) => { if (!active) e.currentTarget.style.background = "rgba(0,0,0,0.05)"; }}
                  onMouseLeave={(e) => { if (!active) e.currentTarget.style.background = "transparent"; }}
                >
                  <span className="mono" style={{
                    display: "inline-flex", alignItems: "center", justifyContent: "center",
                    width: 20, height: 18, borderRadius: 3,
                    fontSize: 10, fontWeight: 600,
                    background: active ? "var(--crimson)" : "rgba(0,0,0,0.06)",
                    color: active ? "#fff" : "var(--muted)",
                  }}>{String(n).padStart(2, "0")}</span>
                  {p}
                </button>
              );
            })}
          </div>
        ))}
      </nav>

      {/* AI assistant + settings */}
      <div style={{ borderTop: "1px solid var(--line)", padding: 8 }}>
        <button
          onClick={onToggleChat}
          style={{
            display: "flex", alignItems: "center", gap: 10,
            width: "100%", textAlign: "left",
            padding: "8px 10px", borderRadius: 6,
            border: "1px solid " + (chatOpen ? "var(--ink)" : "transparent"),
            background: chatOpen ? "var(--ink)" : "transparent",
            color: chatOpen ? "#fff" : "var(--ink-2)",
            fontSize: 12.5, fontWeight: 600, cursor: "pointer",
            fontFamily: "inherit",
          }}
        >
          <span style={{ width: 18, height: 18, display: "inline-flex", alignItems: "center", justifyContent: "center" }}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M2 4h12v7H9l-3 3v-3H2V4z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round"/>
            </svg>
          </span>
          Assistant
          <span className="mono" style={{
            marginLeft: "auto", fontSize: 9, letterSpacing: "0.08em",
            padding: "2px 6px", borderRadius: 3,
            background: chatOpen ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.06)",
          }}>⌘K</span>
        </button>
      </div>
    </aside>
  );
}

function Topbar({ page, status }) {
  return (
    <div style={{
      height: 42, flexShrink: 0,
      display: "flex", alignItems: "center",
      borderBottom: "1px solid var(--line)",
      background: "#fdfbf7",
      paddingRight: 10,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "0 16px", flex: 1 }}>
        <span className="mono" style={{ fontSize: 10, color: "var(--muted)", letterSpacing: "0.12em", textTransform: "uppercase" }}>
          sparc · pipeline
        </span>
        <span style={{ fontSize: 11, color: "var(--line)" }}>/</span>
        <span style={{ fontSize: 12, fontWeight: 600 }}>{page}</span>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        <StatusPill color="var(--crimson)" label="Sidecar" value="ready · :17123" />
        <StatusPill color="var(--amber)" label="GPU" value="CUDA · 11.8" />
        <StatusPill color="var(--purple)" label="Claude" value="haiku-4.5" />
      </div>
    </div>
  );
}

function StatusPill({ color, label, value }) {
  return (
    <div className="mono" style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: "var(--muted)" }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: color }}></span>
      <span style={{ letterSpacing: "0.1em", textTransform: "uppercase" }}>{label}</span>
      <span style={{ color: "var(--ink-2)" }}>{value}</span>
    </div>
  );
}

// Window chrome (mac-style) — presented as a running desktop app
function WindowChrome({ children, title = "SPARC Labs — Resilience Core" }) {
  return (
    <div style={{
      width: "100%", height: "100%",
      maxWidth: 1440, maxHeight: 900,
      borderRadius: 14,
      overflow: "hidden",
      boxShadow: "0 1px 0 rgba(255,255,255,0.5) inset, 0 20px 60px rgba(0,0,0,0.22), 0 0 0 1px rgba(0,0,0,0.18)",
      display: "flex", flexDirection: "column",
      background: "var(--paper)",
    }}>
      <div style={{
        height: 36, display: "flex", alignItems: "center",
        background: "linear-gradient(180deg, #e8e2d4 0%, #dcd5c4 100%)",
        borderBottom: "1px solid rgba(0,0,0,0.18)",
        position: "relative", flexShrink: 0,
      }}>
        <div style={{ display: "flex", gap: 8, padding: "0 14px" }}>
          {["#ff5f57", "#febc2e", "#28c840"].map((c, i) => (
            <span key={i} style={{
              width: 12, height: 12, borderRadius: "50%", background: c,
              border: "0.5px solid rgba(0,0,0,0.15)",
            }}/>
          ))}
        </div>
        <div style={{
          position: "absolute", left: 0, right: 0, textAlign: "center",
          fontSize: 12, fontWeight: 600, color: "var(--ink-2)",
          letterSpacing: "0.01em", pointerEvents: "none",
        }}>{title}</div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 10, padding: "0 14px" }}>
          <span className="mono" style={{ fontSize: 10, color: "var(--muted)" }}>v0.4.2 · desktop</span>
        </div>
      </div>
      <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
        {children}
      </div>
    </div>
  );
}

window.Sidebar = Sidebar;
window.Topbar = Topbar;
window.WindowChrome = WindowChrome;


// ===== src/results.jsx =====
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


// ===== src/pages.jsx =====
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


// ===== src/app.jsx =====
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


