import type { ButtonHTMLAttributes, ReactNode } from "react";
import { Icon, type IconName } from "./Icon";

export type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "primary" | "danger" | "ghost";
  size?: "normal" | "small";
  icon?: IconName;
  /** Shows a spinner and disables the button. */
  busy?: boolean;
  children?: ReactNode;
};

export function Button({ variant = "default", size = "normal", icon, busy, disabled, className, children, type = "button", ...rest }: ButtonProps) {
  const cls = ["btn", variant !== "default" ? variant : "", size === "small" ? "small" : "", className ?? ""].filter(Boolean).join(" ");
  return (
    <button {...rest} type={type} className={cls} disabled={disabled || busy} aria-busy={busy || undefined}>
      {busy ? <span className="spinner" aria-hidden="true" /> : icon ? <Icon name={icon} /> : null}
      {children}
    </button>
  );
}
