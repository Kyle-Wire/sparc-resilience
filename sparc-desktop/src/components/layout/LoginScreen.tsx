import { useState, useRef, useCallback, useEffect } from "react";
import CubeLogo from "@/components/brand/CubeLogo";
import { useAuthStore } from "@/stores/authStore";
import { useMouseParallax } from "@/hooks/useMouseParallax";
import { open as shellOpen } from "@tauri-apps/plugin-shell";
import { SPARC_SIGNUP_URL } from "@/lib/constants";

interface LoginScreenProps {
  parallaxEnabled?: boolean;
}

export default function LoginScreen({ parallaxEnabled = true }: LoginScreenProps) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const { signIn, loading, error, clearError } = useAuthStore();

  const cardRef = useRef<HTMLDivElement>(null);
  const logoRef = useRef<HTMLDivElement>(null);
  const glowRef = useRef<HTMLDivElement>(null);

  // [DEBUG-sparc-blank] confirm LoginScreen mounted and card ref attached
  useEffect(() => {
    console.log("[DEBUG-sparc-blank] LoginScreen mounted, cardRef attached=", !!cardRef.current);
    return () => console.log("[DEBUG-sparc-blank] LoginScreen unmounted");
  }, []);

  useMouseParallax(
    useCallback(({ nx, ny }) => {
      if (cardRef.current) {
        cardRef.current.style.transform = parallaxEnabled
          ? `perspective(900px) rotateY(${nx * 4}deg) rotateX(${-ny * 3}deg) translateZ(0)`
          : "none";
      }
      if (logoRef.current) {
        cardRef.current; // keep ref
        logoRef.current.style.transform = parallaxEnabled
          ? `translateX(${nx * 7}px) translateY(${ny * 5}px)`
          : "none";
      }
      if (glowRef.current) {
        glowRef.current.style.transform = parallaxEnabled
          ? `translate(${nx * 20}px, ${ny * 15}px)`
          : "none";
      }
    }, [parallaxEnabled]),
    { enabled: parallaxEnabled },
  );

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      clearError();
      if (mode === "signup") {
        shellOpen(SPARC_SIGNUP_URL);
        return;
      }
      await signIn(email, password);
    },
    [mode, email, password, signIn, clearError],
  );

  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "var(--paper)",
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* Ambient glow orb — moves with mouse */}
      <div
        ref={glowRef}
        style={{
          position: "absolute",
          width: 600,
          height: 600,
          borderRadius: "50%",
          background:
            "radial-gradient(circle, rgba(103,51,200,0.12) 0%, rgba(231,60,37,0.06) 50%, transparent 70%)",
          pointerEvents: "none",
          transition: "transform 0.1s linear",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
        }}
      />

      {/* Background grid */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage:
            "linear-gradient(rgba(0,0,0,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,0.03) 1px, transparent 1px)",
          backgroundSize: "28px 28px",
          pointerEvents: "none",
        }}
      />

      {/* [DEBUG-sparc-blank] visible render stamp */}
      <div
        style={{
          position: "absolute",
          top: 8,
          left: 8,
          fontSize: 9,
          fontFamily: "monospace",
          color: "var(--crimson)",
          opacity: 0.5,
          pointerEvents: "none",
          zIndex: 9999,
        }}
      >
        [debug] LoginScreen
      </div>

      {/* Login card */}
      <div
        ref={cardRef}
        style={{
          width: 360,
          background: "var(--paper)",
          border: "1px solid var(--line)",
          borderRadius: 12,
          padding: "36px 32px",
          boxShadow: "0 4px 40px rgba(0,0,0,0.10), 0 1px 4px rgba(0,0,0,0.06)",
          position: "relative",
          zIndex: 10,
          transition: "transform 0.05s linear",
          transformStyle: "preserve-3d",
        }}
      >
        {/* Card gradient border accent */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            borderRadius: 12,
            padding: 1,
            background:
              "linear-gradient(135deg, var(--crimson), var(--color-sparc-purple, #602468), transparent 60%)",
            WebkitMask: "linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)",
            WebkitMaskComposite: "xor",
            maskComposite: "exclude",
            opacity: 0.5,
            pointerEvents: "none",
          }}
        />

        {/* Logo */}
        <div
          ref={logoRef}
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            marginBottom: 28,
            transition: "transform 0.05s linear",
          }}
        >
          <CubeLogo size={56} />
          <div style={{ marginTop: 12, textAlign: "center" }}>
            <div style={{ fontSize: 18, fontWeight: 800, letterSpacing: "-0.02em" }}>SPARC LABS</div>
            <div
              className="mono"
              style={{ fontSize: 9.5, letterSpacing: "0.2em", color: "var(--muted)", marginTop: 3 }}
            >
              SPATIAL ANALYSIS &amp; RESEARCH CORE
            </div>
          </div>
        </div>

        {/* Tab switcher */}
        <div
          style={{
            display: "flex",
            gap: 0,
            marginBottom: 20,
            border: "1px solid var(--line)",
            borderRadius: 6,
            overflow: "hidden",
          }}
        >
          {(["signin", "signup"] as const).map((m) => (
            <button
              key={m}
              onClick={() => { setMode(m); clearError(); }}
              style={{
                flex: 1,
                padding: "7px 0",
                fontSize: 12,
                fontWeight: 600,
                border: "none",
                background: mode === m ? "var(--ink)" : "transparent",
                color: mode === m ? "var(--paper)" : "var(--muted)",
                cursor: "pointer",
                transition: "background 0.15s, color 0.15s",
              }}
            >
              {m === "signin" ? "Sign In" : "Create Account"}
            </button>
          ))}
        </div>

        {mode === "signup" ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <p style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.6, margin: 0 }}>
              Create your SPARC Labs account on our website to get started.
            </p>
            <button
              onClick={() => shellOpen(SPARC_SIGNUP_URL)}
              style={{
                width: "100%",
                padding: "10px 0",
                fontSize: 13,
                fontWeight: 700,
                border: "none",
                borderRadius: 6,
                background: "var(--ink)",
                color: "var(--paper)",
                cursor: "pointer",
                letterSpacing: "0.03em",
              }}
            >
              Go to sparclabs.co →
            </button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div>
              <label
                className="mono"
                style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", display: "block", marginBottom: 5 }}
              >
                Email
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                required
                autoComplete="email"
                style={{
                  width: "100%",
                  padding: "8px 10px",
                  fontSize: 13,
                  border: "1px solid var(--line)",
                  borderRadius: 5,
                  background: "var(--paper-2, var(--paper))",
                  color: "var(--ink)",
                  outline: "none",
                  boxSizing: "border-box",
                }}
              />
            </div>

            <div>
              <label
                className="mono"
                style={{ fontSize: 9.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", display: "block", marginBottom: 5 }}
              >
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                required
                autoComplete="current-password"
                style={{
                  width: "100%",
                  padding: "8px 10px",
                  fontSize: 13,
                  border: "1px solid var(--line)",
                  borderRadius: 5,
                  background: "var(--paper-2, var(--paper))",
                  color: "var(--ink)",
                  outline: "none",
                  boxSizing: "border-box",
                }}
              />
            </div>

            {error && (
              <div
                style={{
                  padding: "8px 10px",
                  background: "rgba(231,60,37,0.08)",
                  border: "1px solid rgba(231,60,37,0.25)",
                  borderRadius: 5,
                  fontSize: 12,
                  color: "var(--crimson)",
                }}
              >
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              style={{
                width: "100%",
                padding: "10px 0",
                fontSize: 13,
                fontWeight: 700,
                border: "none",
                borderRadius: 6,
                background: loading ? "var(--line)" : "var(--ink)",
                color: loading ? "var(--muted)" : "var(--paper)",
                cursor: loading ? "not-allowed" : "pointer",
                letterSpacing: "0.03em",
                transition: "background 0.15s",
              }}
            >
              {loading ? "Signing in…" : "Sign In"}
            </button>

            <div style={{ textAlign: "center" }}>
              <a
                href="https://sparclabs.co/reset"
                target="_blank"
                rel="noreferrer"
                style={{ fontSize: 11, color: "var(--muted)", textDecoration: "none" }}
              >
                Forgot password?
              </a>
            </div>
          </form>
        )}
      </div>

      {/* Version watermark */}
      <div
        className="mono"
        style={{
          position: "absolute",
          bottom: 16,
          right: 20,
          fontSize: 9.5,
          color: "var(--muted)",
          letterSpacing: "0.1em",
          opacity: 0.5,
        }}
      >
        v{__APP_VERSION__}
      </div>
    </div>
  );
}
