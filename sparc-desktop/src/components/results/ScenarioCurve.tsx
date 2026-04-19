import { useEffect, useRef } from "react";

export default function ScenarioCurve() {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = c.clientWidth;
    const h = c.clientHeight;
    c.width = w * DPR;
    c.height = h * DPR;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    const PAD = { l: 36, r: 10, t: 10, b: 22 };
    const xs = [0, 5, 10, 15, 20, 30, 50];
    const canopy: (number | null)[] = [0, -0.130, -0.258, -0.437, -0.509, -0.608, -0.738];
    const imperv: (number | null)[] = [0, -0.098, -0.195, null, -0.383, -0.456, -0.550];
    const albedo: (number | null)[] = [0, -0.098, -0.196, null, -0.384, -0.450, null];

    const xMax = 50,
      yMin = -0.8,
      yMax = 0.02;
    const X = (x: number) => PAD.l + (x / xMax) * (w - PAD.l - PAD.r);
    const Y = (y: number) => PAD.t + ((yMax - y) / (yMax - yMin)) * (h - PAD.t - PAD.b);

    ctx.strokeStyle = "rgba(0,0,0,0.15)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.l, PAD.t);
    ctx.lineTo(PAD.l, h - PAD.b);
    ctx.lineTo(w - PAD.r, h - PAD.b);
    ctx.stroke();

    for (let y = -0.8; y <= 0; y += 0.2) {
      ctx.strokeStyle = "rgba(0,0,0,0.05)";
      ctx.beginPath();
      ctx.moveTo(PAD.l, Y(y));
      ctx.lineTo(w - PAD.r, Y(y));
      ctx.stroke();
      ctx.fillStyle = "#6e6358";
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText(y.toFixed(1), PAD.l - 6, Y(y) + 3);
    }
    xs.forEach((x) => {
      ctx.fillStyle = "#6e6358";
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText(String(x), X(x), h - PAD.b + 13);
    });

    const series = [
      { data: canopy, color: "#602468", label: "Canopy +X pp" },
      { data: imperv, color: "#e73c25", label: "Impervious −X pp" },
      { data: albedo, color: "#e79024", label: "Albedo +X" },
    ];

    for (const s of series) {
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      let started = false;
      xs.forEach((x, i) => {
        if (s.data[i] == null) return;
        const px = X(x);
        const py = Y(s.data[i] as number);
        if (!started) {
          ctx.moveTo(px, py);
          started = true;
        } else {
          ctx.lineTo(px, py);
        }
      });
      ctx.stroke();

      xs.forEach((x, i) => {
        if (s.data[i] == null) return;
        ctx.fillStyle = s.color;
        ctx.beginPath();
        ctx.arc(X(x), Y(s.data[i] as number), 3, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    const lx = w - 150;
    const ly = 18;
    series.forEach((s, i) => {
      ctx.fillStyle = s.color;
      ctx.fillRect(lx, ly + i * 14 - 6, 10, 2);
      ctx.fillStyle = "#2b2327";
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.textAlign = "left";
      ctx.fillText(s.label, lx + 14, ly + i * 14);
    });
  }, []);

  return <canvas ref={canvas} style={{ width: "100%", height: 220, display: "block" }} />;
}
