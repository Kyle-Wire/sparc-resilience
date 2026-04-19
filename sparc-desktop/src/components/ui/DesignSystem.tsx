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
  const pad = padding !== undefined ? padding : 16;
  return (
    <div
      className={className}
      style={{
        background: '#fff',
        border: '1px solid var(--color-sparc-line)',
        borderRadius: 8,
        display: 'flex',
        flexDirection: 'column',
        ...style,
      }}
    >
      {(title || actions) && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            padding: '10px 14px',
            borderBottom: '1px solid var(--color-sparc-line)',
            background: '#fdfbf7',
            borderRadius: '8px 8px 0 0',
          }}
        >
          <div style={{ flex: 1, minWidth: 0 }}>
            {title && <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: '-0.01em', color: 'var(--color-sparc-ink)' }}>{title}</div>}
            {subtitle && (
              <div
                className="mono"
                style={{ fontSize: 10, color: 'var(--color-sparc-muted)', marginTop: 2 }}
              >
                {subtitle}
              </div>
            )}
          </div>
          {actions}
        </div>
      )}
      <div style={{ padding: pad, flex: 1, minHeight: 0 }}>{children}</div>
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
      }}
    >
      <div>
        <div
          className="mono"
          style={{
            fontSize: 10,
            color: 'var(--color-sparc-muted)',
            letterSpacing: '0.16em',
            textTransform: 'uppercase',
          }}
        >
          {kicker}
        </div>
        <div
          style={{
            fontSize: 22,
            fontWeight: 800,
            letterSpacing: '-0.02em',
            color: 'var(--color-sparc-ink)',
            marginTop: 3,
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
        fontSize: 9.5,
        padding: '2px 6px',
        borderRadius: 3,
        background: `${color}22`,
        color,
        letterSpacing: '0.06em',
        textTransform: 'uppercase',
        fontWeight: 600,
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
  value: ReactNode;
}

export function KeyVal({ label, value }: KeyValProps) {
  return (
    <div>
      <div
        className="mono"
        style={{
          fontSize: 9.5,
          color: 'var(--color-sparc-muted)',
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
        }}
      >
        {label}
      </div>
      <div style={{ fontSize: 13, fontWeight: 600, marginTop: 3 }}>{value}</div>
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
