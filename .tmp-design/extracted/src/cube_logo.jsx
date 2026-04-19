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
