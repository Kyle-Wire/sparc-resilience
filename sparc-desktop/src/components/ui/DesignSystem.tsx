import type { ReactNode, CSSProperties } from 'react';

/* ----------------------------------------------------------------
   Card — primary container with cream header accent + border
   ---------------------------------------------------------------- */
interface CardProps {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  padding?: number | string;
  style?: CSSProperties;
  className?: string;
}

export function Card({ title, subtitle, actions, children, padding, style, className }: CardProps) {
  const pad = padding !== undefined ? padding : 14;
  return (
    <div
      className={className}
      style={{
        border: '1px solid var(--color-sparc-line)',
        borderRadius: 8,
        background: '#fff',
        overflow: 'hidden',
        ...style,
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '10px 14px',
          borderBottom: '1px dashed var(--color-sparc-line)',
          background: '#fdfbf7',
        }}
      >
        <div>
          <div style={{ fontSize: 12.5, fontWeight: 700, color: 'var(--color-sparc-ink)' }}>{title}</div>
          {subtitle && (
            <div
              className="mono"
              style={{ fontSize: 10, color: 'var(--color-sparc-muted)', marginTop: 1, letterSpacing: '0.04em' }}
            >
              {subtitle}
            </div>
          )}
        </div>
        {actions}
      </div>
      <div style={{ padding: pad }}>{children}</div>
    </div>
  );
}

/* ----------------------------------------------------------------
   SectionHeader — page-level kicker + heading + optional right slot
   ---------------------------------------------------------------- */
interface SectionHeaderProps {
  kicker: string;
  label: string;
  right?: ReactNode;
}

export function SectionHeader({ kicker, label, right }: SectionHeaderProps) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'flex-end',
        justifyContent: 'space-between',
        marginBottom: 14,
        borderBottom: '1px solid var(--color-sparc-line)',
        paddingBottom: 10,
      }}
    >
      <div>
        <div
          className="mono"
          style={{
            fontSize: 10,
            fontWeight: 600,
            letterSpacing: '0.14em',
            textTransform: 'uppercase',
            color: 'var(--color-sparc-muted)',
          }}
        >
          {kicker}
        </div>
        <div
          style={{
            fontSize: 20,
            fontWeight: 800,
            letterSpacing: '-0.02em',
            color: 'var(--color-sparc-ink)',
            marginTop: 2,
          }}
        >
          {label}
        </div>
      </div>
      {right && <div>{right}</div>}
    </div>
  );
}

/* ----------------------------------------------------------------
   Tag — small pill label
   ---------------------------------------------------------------- */
interface TagProps {
  color?: string;
  children: ReactNode;
}

export function Tag({ color = 'var(--color-sparc-ink)', children }: TagProps) {
  return (
    <span
      className="mono"
      style={{
        display: 'inline-block',
        border: `1px solid ${color}`,
        color,
        padding: '1px 6px',
        borderRadius: 4,
        fontSize: 10,
        fontWeight: 600,
        letterSpacing: '0.06em',
        textTransform: 'uppercase',
        lineHeight: '16px',
      }}
    >
      {children}
    </span>
  );
}

/* ----------------------------------------------------------------
   Stat — KPI card with label + large value
   ---------------------------------------------------------------- */
interface StatProps {
  label: string;
  value: string | number;
  tint?: string;
  sub?: string;
}

export function Stat({ label, value, tint = 'var(--color-sparc-ink)', sub }: StatProps) {
  return (
    <div
      style={{
        border: '1px solid var(--color-sparc-line)',
        borderRadius: 8,
        padding: '10px 14px',
        background: '#fff',
        overflow: 'hidden',
      }}
    >
      <div
        className="mono"
        style={{
          fontSize: 9.5,
          color: 'var(--color-sparc-muted)',
          letterSpacing: '0.12em',
          textTransform: 'uppercase',
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: 22,
          fontWeight: 700,
          letterSpacing: '-0.02em',
          color: tint,
          marginTop: 2,
        }}
      >
        {value}
      </div>
      {sub && (
        <div
          className="mono"
          style={{ fontSize: 10, color: 'var(--color-sparc-muted)', marginTop: 2 }}
        >
          {sub}
        </div>
      )}
    </div>
  );
}

/* ----------------------------------------------------------------
   Btn — small action button
   ---------------------------------------------------------------- */
interface BtnProps {
  children: ReactNode;
  primary?: boolean;
  small?: boolean;
  onClick?: () => void;
  disabled?: boolean;
}

export function Btn({ children, primary, small, onClick, disabled }: BtnProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        border: `1px solid ${primary ? 'var(--color-sparc-ink)' : 'var(--color-sparc-line)'}`,
        background: primary ? 'var(--color-sparc-ink)' : '#fff',
        color: primary ? '#fff' : 'var(--color-sparc-ink-2)',
        padding: small ? '4px 10px' : '7px 14px',
        fontSize: small ? 11 : 12,
        fontWeight: 600,
        borderRadius: 5,
        cursor: disabled ? 'not-allowed' : 'pointer',
        fontFamily: 'inherit',
        opacity: disabled ? 0.5 : 1,
      }}
    >
      {children}
    </button>
  );
}

/* ----------------------------------------------------------------
   KeyVal — label: value pair
   ---------------------------------------------------------------- */
interface KeyValProps {
  label: string;
  value: string | number;
  mono?: boolean;
}

export function KeyVal({ label, value, mono: useMono }: KeyValProps) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0', borderTop: '1px dashed var(--color-sparc-line)' }}>
      <span style={{ fontSize: 12, color: 'var(--color-sparc-muted)' }}>{label}</span>
      <span className={useMono ? 'mono' : ''} style={{ fontSize: 12, fontWeight: 600, color: 'var(--color-sparc-ink)' }}>{value}</span>
    </div>
  );
}

/* ----------------------------------------------------------------
   LegendDot — small label with colored indicator
   ---------------------------------------------------------------- */
interface LegendDotProps {
  color: string;
  label: string;
}

export function LegendDot({ color, label }: LegendDotProps) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 11 }}>
      <span
        style={{
          width: 10,
          height: 10,
          border: `1.5px solid ${color}`,
          background: '#fff',
          borderRadius: 2,
        }}
      />
      <span
        className="mono"
        style={{
          color: 'var(--color-sparc-muted)',
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          fontSize: 10,
        }}
      >
        {label}
      </span>
    </span>
  );
}
