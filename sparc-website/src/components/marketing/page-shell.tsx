import * as React from "react";

export function Kicker({
  children,
  dot,
  dotColor,
}: {
  children: React.ReactNode;
  dot?: boolean;
  dotColor?: string;
}) {
  return (
    <div className="kicker" style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
      {dot && <span className="dot" style={{ background: dotColor ?? "var(--accent)" }} />}
      {children}
    </div>
  );
}

export function PageHero({
  kicker,
  title,
  lead,
  accent = "var(--crimson)",
}: {
  kicker: React.ReactNode;
  title: React.ReactNode;
  lead?: React.ReactNode;
  accent?: string;
}) {
  return (
    <section className="page-hero grid-bg">
      <div
        aria-hidden
        style={{
          position: "absolute",
          right: -100,
          top: -120,
          width: 460,
          height: 460,
          background: `radial-gradient(circle, ${accent}1f, transparent 60%)`,
          pointerEvents: "none",
        }}
      />
      <div className="wrap" style={{ position: "relative" }}>
        <Kicker dot dotColor={accent}>
          {kicker}
        </Kicker>
        <h1>{title}</h1>
        {lead && <p className="lead">{lead}</p>}
      </div>
    </section>
  );
}

export function Section({
  children,
  kicker,
  title,
  lead,
  sunken,
  action,
  id,
}: {
  children: React.ReactNode;
  kicker?: React.ReactNode;
  title?: React.ReactNode;
  lead?: React.ReactNode;
  sunken?: boolean;
  action?: React.ReactNode;
  id?: string;
}) {
  return (
    <section id={id} style={sunken ? { background: "var(--paper-2)" } : undefined}>
      <div className="wrap">
        {(kicker || title) && (
          <div className="section-head">
            <div>
              {kicker && <Kicker>{kicker}</Kicker>}
              {title && <h2 style={{ marginTop: 8 }}>{title}</h2>}
            </div>
            {action ? action : lead && <p className="lead">{lead}</p>}
          </div>
        )}
        {children}
      </div>
    </section>
  );
}
