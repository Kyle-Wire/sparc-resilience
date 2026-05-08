import type { ReactNode } from "react";
import { useAuthStore } from "@/stores/authStore";

interface AuthGateProps {
  children: ReactNode;
  onSignIn?: () => void;
}

/**
 * Wraps a page with a frosted-glass overlay when the user is not authenticated.
 * The page content is rendered but blurred/dimmed behind the overlay,
 * giving a preview effect.
 */
export default function AuthGate({ children, onSignIn }: AuthGateProps) {
  const user = useAuthStore((s) => s.user);

  if (user) return <>{children}</>;

  return (
    <div style={{ position: "relative", height: "100%", width: "100%", overflow: "hidden" }}>
      {/* Blurred content behind gate */}
      <div
        style={{
          filter: "blur(3px) brightness(0.7)",
          pointerEvents: "none",
          userSelect: "none",
          height: "100%",
          width: "100%",
        }}
        aria-hidden="true"
      >
        {children}
      </div>

      {/* Frosted overlay */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 16,
          backdropFilter: "blur(2px)",
          background: "rgba(var(--paper-rgb, 247,244,238), 0.35)",
        }}
      >
        {/* SPARC gradient border card */}
        <div
          style={{
            padding: "28px 32px",
            borderRadius: 12,
            background: "var(--paper)",
            boxShadow: "0 8px 48px rgba(0,0,0,0.15), 0 1px 4px rgba(0,0,0,0.08)",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 12,
            position: "relative",
            border: "1px solid transparent",
            backgroundClip: "padding-box",
            maxWidth: 320,
            textAlign: "center",
          }}
        >
          {/* Gradient border accent */}
          <div
            style={{
              position: "absolute",
              inset: -1,
              borderRadius: 13,
              background:
                "linear-gradient(135deg, #602468, #e73c25, #00d4ff, #5b7af0)",
              zIndex: -1,
              opacity: 0.7,
            }}
          />

          {/* Lock icon */}
          <div style={{ fontSize: 28, lineHeight: 1 }}>🔒</div>

          <div>
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 4 }}>
              Sign in to unlock
            </div>
            <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.5 }}>
              This feature requires a SPARC Labs account.
            </div>
          </div>

          <button
            onClick={onSignIn}
            style={{
              marginTop: 4,
              padding: "9px 24px",
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
            Sign In
          </button>

          <a
            href="https://sparclabs.co"
            target="_blank"
            rel="noreferrer"
            style={{ fontSize: 11, color: "var(--muted)", textDecoration: "none" }}
          >
            Create account at sparclabs.co →
          </a>
        </div>
      </div>
    </div>
  );
}
