import { useEffect } from "react";
import CubeLogo from "../brand/CubeLogo";

interface SplashProps {
  onReady?: () => void;
}

export default function Splash({ onReady }: SplashProps) {
  useEffect(() => {
    if (!onReady) return;
    const t = setTimeout(onReady, 1400);
    return () => clearTimeout(t);
  }, [onReady]);

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexDirection: "column",
        gap: 22,
      }}
    >
      <CubeLogo size={180} />
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em" }}>SPARC LABS</div>
        <div
          className="mono"
          style={{ fontSize: 10, letterSpacing: "0.2em", color: "var(--muted)", marginTop: 4 }}
        >
          SPATIAL ANALYSIS &amp; RESEARCH CORE · v0.4.2
        </div>
      </div>
      <div style={{ width: 220, height: 3, background: "rgba(0,0,0,0.08)", borderRadius: 2, overflow: "hidden" }}>
        <div style={{ width: "100%", height: "100%", background: "var(--crimson)", animation: "loadBar 1.3s ease-out" }} />
      </div>
    </div>
  );
}
