import type { ReactNode } from "react";
import type { HealthResponse } from "@/lib/types";

interface TopbarProps {
  status: HealthResponse | null;
  children?: ReactNode;
  currentPage?: string;
}

function StatusPill({ label, ok, detail }: { label: string; ok: boolean; detail?: string }) {
  return (
    <span
      className="mono inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5"
      style={{
        fontSize: 10,
        fontWeight: 600,
        letterSpacing: '0.06em',
        textTransform: 'uppercase' as const,
        borderColor: ok ? 'var(--color-sparc-line)' : 'rgba(231,60,37,0.3)',
        background: ok ? '#fff' : 'rgba(231,60,37,0.06)',
        color: ok ? 'var(--color-sparc-ink-2)' : 'var(--color-sparc-crimson)',
      }}
    >
      <span
        style={{
          width: 6,
          height: 6,
          borderRadius: '50%',
          background: ok ? '#22c55e' : 'var(--color-sparc-crimson)',
          display: 'inline-block',
        }}
      />
      {label}
      {detail && (
        <span style={{ color: 'var(--color-sparc-muted)', fontWeight: 400 }}>
          {detail}
        </span>
      )}
    </span>
  );
}

export default function Topbar({ status, children, currentPage }: TopbarProps) {
  return (
    <header
      className="flex items-center justify-between border-b px-4"
      style={{
        height: 40,
        background: '#fff',
        borderColor: 'var(--color-sparc-line)',
      }}
    >
      {/* Breadcrumb */}
      <div className="flex items-center gap-1.5" style={{ fontSize: 12 }}>
        <span className="font-semibold" style={{ color: 'var(--color-sparc-ink)' }}>
          SPARC
        </span>
        {currentPage && (
          <>
            <span style={{ color: 'var(--color-sparc-line)' }}>/</span>
            <span style={{ color: 'var(--color-sparc-muted)' }}>{currentPage}</span>
          </>
        )}
        {status?.is_running && (
          <>
            <span style={{ color: 'var(--color-sparc-line)', margin: '0 4px' }}>·</span>
            <span
              className="mono inline-flex items-center gap-1.5 font-semibold"
              style={{ color: 'var(--color-sparc-crimson)', fontSize: 11 }}
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

      {/* Status pills + children */}
      <div className="flex items-center gap-2">
        <StatusPill
          label="Sidecar"
          ok={status?.project_loaded ?? false}
          detail={status?.project_loaded ? '8008' : 'off'}
        />
        <StatusPill label="GPU" ok={false} detail="cpu" />
        {children}
      </div>
    </header>
  );
}
