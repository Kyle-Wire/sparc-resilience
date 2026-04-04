import { useRef, useEffect } from "react";

/**
 * Processing-language–style animated eye that tracks the mouse pointer.
 * Uses canvas + atan2 for pupil displacement, matching the classic
 * Processing `atan2()` arctangent demo.
 */
export default function ProcessingEye({ width = 280, height = 180 }: { width?: number; height?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouse = useRef({ x: width / 2, y: height / 2 });

  useEffect(() => {
    const handleMove = (e: MouseEvent) => {
      const cvs = canvasRef.current;
      if (!cvs) return;
      const r = cvs.getBoundingClientRect();
      mouse.current = { x: e.clientX - r.left, y: e.clientY - r.top };
    };
    window.addEventListener("mousemove", handleMove);
    return () => window.removeEventListener("mousemove", handleMove);
  }, []);

  useEffect(() => {
    const cvs = canvasRef.current;
    if (!cvs) return;
    const ctx = cvs.getContext("2d");
    if (!ctx) return;

    let raf: number;

    const draw = () => {
      const W = cvs.width;
      const H = cvs.height;
      ctx.clearRect(0, 0, W, H);

      // Background
      ctx.fillStyle = "#0d0d0d";
      ctx.fillRect(0, 0, W, H);

      // Scanline overlay
      ctx.fillStyle = "rgba(0,255,120,0.02)";
      for (let y = 0; y < H; y += 3) {
        ctx.fillRect(0, y, W, 1);
      }

      const cx = W / 2;
      const cy = H / 2;
      const eyeRx = W * 0.28;
      const eyeRy = H * 0.36;

      // Eye white (almond shape)
      ctx.save();
      ctx.beginPath();
      ctx.ellipse(cx, cy, eyeRx, eyeRy, 0, 0, Math.PI * 2);
      ctx.fillStyle = "#e8e0d0";
      ctx.fill();

      // Subtle sclera veins
      ctx.strokeStyle = "rgba(180,60,60,0.12)";
      ctx.lineWidth = 0.5;
      for (let i = 0; i < 6; i++) {
        const a = (Math.PI * 2 * i) / 6 + 0.3;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + Math.cos(a) * eyeRx * 0.9, cy + Math.sin(a) * eyeRy * 0.9);
        ctx.stroke();
      }
      ctx.restore();

      // Pupil tracking via atan2
      const dx = mouse.current.x - cx;
      const dy = mouse.current.y - cy;
      const angle = Math.atan2(dy, dx);
      const dist = Math.min(Math.hypot(dx, dy), eyeRx * 0.4);
      const pupilX = cx + Math.cos(angle) * dist;
      const pupilY = cy + Math.sin(angle) * dist * (eyeRy / eyeRx);

      // Iris
      const irisR = eyeRy * 0.55;
      const irisGrad = ctx.createRadialGradient(pupilX, pupilY, irisR * 0.15, pupilX, pupilY, irisR);
      irisGrad.addColorStop(0, "#3a6b35");
      irisGrad.addColorStop(0.5, "#2d5a27");
      irisGrad.addColorStop(0.8, "#1a3d17");
      irisGrad.addColorStop(1, "#102a0e");
      ctx.beginPath();
      ctx.arc(pupilX, pupilY, irisR, 0, Math.PI * 2);
      ctx.fillStyle = irisGrad;
      ctx.fill();

      // Iris texture rings
      ctx.strokeStyle = "rgba(80,160,60,0.15)";
      ctx.lineWidth = 0.5;
      for (let r = irisR * 0.3; r < irisR; r += irisR * 0.12) {
        ctx.beginPath();
        ctx.arc(pupilX, pupilY, r, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Pupil (dark center)
      const pupilR = irisR * 0.42;
      ctx.beginPath();
      ctx.arc(pupilX, pupilY, pupilR, 0, Math.PI * 2);
      ctx.fillStyle = "#080808";
      ctx.fill();

      // Specular highlight
      ctx.beginPath();
      ctx.ellipse(pupilX - pupilR * 0.35, pupilY - pupilR * 0.4, pupilR * 0.28, pupilR * 0.2, -0.3, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255,0.7)";
      ctx.fill();

      // Small secondary highlight
      ctx.beginPath();
      ctx.arc(pupilX + pupilR * 0.3, pupilY + pupilR * 0.25, pupilR * 0.1, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255,0.35)";
      ctx.fill();

      // CRT glow around eye
      ctx.save();
      ctx.shadowColor = "rgba(0,255,120,0.15)";
      ctx.shadowBlur = 20;
      ctx.beginPath();
      ctx.ellipse(cx, cy, eyeRx + 4, eyeRy + 4, 0, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(0,255,120,0.2)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.restore();

      // Retro label
      ctx.fillStyle = "rgba(0,255,120,0.5)";
      ctx.font = '10px "Courier New", monospace';
      ctx.fillText("atan2() tracking", 8, H - 8);

      raf = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="mx-auto rounded"
      style={{ imageRendering: "pixelated" }}
    />
  );
}
