import type { ReactNode } from "react";
import type { HealthResponse } from "@/lib/types";

interface TopbarProps {
  status: HealthResponse | null;
  children?: ReactNode;
  currentPage?: string;
}

function StatusPill({ color, label, value }: { color: string; label: string; value: string }) {
  return (
    <div
      className="mono"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        fontSize: 10,
        color: 'var(--color-sparc-muted)',
      }}
    >
      <span
        style={{
          width: 6,
          height: 6,
          borderRadius: '50%',
          background: color,
        }}
      />
      <span style={{ letterSpacing: '0.1em', textTransform: 'uppercase' }}>{label}</span>
      <span style={{ color: 'var(--color-sparc-ink-2)' }}>{value}</span>
    </div>
  );
}

export default function Topbar({ status, children, currentPage }: TopbarProps) {
  return (
    <header
      style={{
        height: 42,
        flexShrink: 0,
        display: 'flex',
        alignItems: 'center',
        borderBottom: '1px solid var(--color-sparc-line)',
        background: '#fdfbf7',
        paddingRight: 10,
      }}
    >
      {/* Breadcrumb */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '0 16px', flex: 1 }}>
        <span
          className="mono"
          style={{
            fontSize: 10,
            color: 'var(--color-sparc-muted)',
            letterSpacing: '0.12em',
            textTransform: 'uppercase',
          }}
        >
          sparc · pipeline
        </span>
        {currentPage && (
          <>
            <span style={{ fontSize: 11, color: 'var(--color-sparc-line)' }}>/</span>
            <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--color-sparc-ink)' }}>{currentPage}</span>
          </>
        )}
        {status?.is_running && (
          <>
            <span style={{ fontSize: 11, color: 'var(--color-sparc-line)', margin: '0 2px' }}>·</span>
            <span
              className="mono"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
                color: 'var(--color-sparc-crimson)',
                fontSize: 11,
                fontWeight: 600,
              }}
            >
              <span
                className="animate-pulse"
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  background: 'var(--color-sparc-crimson)',
                  display: 'inline-block',
                }}
              />
              Stage {status.current_stage}
            </span>
          </>
        )}
      </div>

      {/* Status pills */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <StatusPill
          color="var(--color-sparc-crimson)"
          label="Sidecar"
          value={status?.project_loaded ? 'ready · :8008' : 'off'}
        />
        <StatusPill
          color="var(--color-sparc-amber)"
          label="GPU"
          value="cpu"
        />
        <StatusPill
          color="var(--color-sparc-purple)"
          label="Claude"
          value="haiku-4.5"
        />
        {children}
      </div>
    </header>
  );
}
