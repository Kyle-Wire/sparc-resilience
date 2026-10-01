import type { HTMLAttributes, ReactNode } from "react";

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
