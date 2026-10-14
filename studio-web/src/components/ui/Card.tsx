import type { CSSProperties, HTMLAttributes, ReactNode } from "react";

export type CardProps = Omit<HTMLAttributes<HTMLElement>, "title"> & {
  title?: ReactNode;
  eyebrow?: ReactNode;
  actions?: ReactNode;
  variant?: "default" | "flat" | "tight";
  as?: "section" | "div" | "article";
  children?: ReactNode;
};

export function Card({ title, eyebrow, actions, variant = "default", as = "section", className, children, ...rest }: CardProps) {
  const Tag = as;
  const cls = ["card", variant !== "default" ? variant : "", className ?? ""].filter(Boolean).join(" ");
  return (
    <Tag {...rest} className={cls}>
      {title || eyebrow || actions ? (
        <header>
          <div>
            {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
            {title ? <h3>{title}</h3> : null}
          </div>
          {actions ? <div className="row">{actions}</div> : null}
        </header>
      ) : null}
      {children}
    </Tag>
  );
}

export type VCardProps = Omit<HTMLAttributes<HTMLElement>, "title"> & {
  /** Kicker label on the left of the header row. */
  label: ReactNode;
  /** Mono meta on the right of the header row (text or small actions). */
  meta?: ReactNode;
  /** Colour of the 2px top rule (any CSS colour or var); defaults to --accent. */
  accent?: string;
  as?: "section" | "div" | "article";
  children?: ReactNode;
};

/** Card with a 2px accent top rule and a kicker / meta header row (the website's VCard). */
export function VCard({ label, meta, accent, as = "section", className, style, children, ...rest }: VCardProps) {
  const Tag = as;
  const cls = ["vcard", className ?? ""].filter(Boolean).join(" ");
  const st = accent ? ({ ...style, "--vcard-accent": accent } as CSSProperties) : style;
  return (
    <Tag {...rest} className={cls} style={st}>
      <header className="vcard-head">
        <span className="kicker">{label}</span>
        {meta ? <span className="vcard-meta">{meta}</span> : null}
      </header>
      <div className="vcard-body">{children}</div>
    </Tag>
  );
}
