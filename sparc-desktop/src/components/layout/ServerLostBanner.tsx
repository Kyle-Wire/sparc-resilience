import { useEffect, useState } from "react";

interface Props {
  serverLost: boolean;
}

/**
 * Floats a dismissible banner at the top of the viewport whenever the
 * sidecar goes offline mid-session. Auto-dismisses 2s after recovery.
 */
export default function ServerLostBanner({ serverLost }: Props) {
  const [visible, setVisible] = useState(false);
  const [reconnects, setReconnects] = useState(0);

  useEffect(() => {
    if (serverLost) {
      setVisible(true);
      setReconnects((n) => n + 1);
    } else if (visible) {
      // Server recovered — auto-dismiss after a brief "reconnected" flash
      const t = setTimeout(() => setVisible(false), 2_000);
      return () => clearTimeout(t);
    }
  }, [serverLost]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!visible) return null;

  return (
    <div
      role="alert"
      aria-live="assertive"
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        zIndex: 9999,
        background: serverLost ? "var(--crimson, #e94461)" : "var(--green, #2e7d32)",
        color: "#fff",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 10,
        padding: "8px 16px",
        fontSize: 12,
        fontWeight: 600,
        fontFamily: "inherit",
        transition: "background 400ms ease",
      }}
    >
      {serverLost ? (
        <>
          <span
            style={{
              display: "inline-block",
              width: 10,
              height: 10,
              borderRadius: "50%",
              border: "2px solid #fff",
              borderTopColor: "transparent",
              animation: "spin 0.7s linear infinite",
            }}
          />
          Sidecar offline — reconnecting
          {reconnects > 1 && (
            <span style={{ opacity: 0.75, fontWeight: 400 }}>
              &nbsp;(attempt {reconnects})
            </span>
          )}
        </>
      ) : (
        <>✓ Sidecar reconnected</>
      )}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
