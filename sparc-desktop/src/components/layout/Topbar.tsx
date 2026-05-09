import { useAuthStore } from "@/stores/authStore";

interface TopbarProps {
  page: string;
  onNavigateSettings?: () => void;
}

export default function Topbar({ page, onNavigateSettings }: TopbarProps) {
  const user = useAuthStore((s) => s.user);

  return (
    <div
      style={{
        height: 42,
        flexShrink: 0,
        display: "flex",
        alignItems: "center",
        borderBottom: "1px solid var(--line)",
        background: "var(--paper)",
        paddingRight: 10,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "0 16px", flex: 1 }}>
        <span
          className="mono"
          style={{ fontSize: 10, color: "var(--muted)", letterSpacing: "0.12em", textTransform: "uppercase" }}
        >
          sparc · pipeline
        </span>
        <span style={{ fontSize: 11, color: "var(--line)" }}>/</span>
        <span style={{ fontSize: 12, fontWeight: 600 }}>{page}</span>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 16, paddingRight: 4 }}>
        {/* User avatar */}
        {user && (
          <button
            onClick={onNavigateSettings}
            title={user.email ?? "Account settings"}
            style={{
              width: 26,
              height: 26,
              borderRadius: "50%",
              border: "none",
              background: "var(--crimson)",
              color: "#fff",
              fontSize: 11,
              fontWeight: 700,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
          >
            {user.email?.[0]?.toUpperCase() ?? "?"}
          </button>
        )}
      </div>
    </div>
  );
}

