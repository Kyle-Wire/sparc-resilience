import type { ButtonHTMLAttributes } from "react";
import { Icon, type IconName } from "./Icon";

export type IconButtonProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children"> & {
  icon: IconName;
  /** Accessible name (also the tooltip). Required: an icon alone has no name. */
  label: string;
  size?: "normal" | "small";
  pressed?: boolean;
  variant?: "default" | "ghost";
};

export function IconButton({ icon, label, size = "normal", pressed, variant = "ghost", className, type = "button", title, ...rest }: IconButtonProps) {
  const cls = ["btn", "icon", variant === "ghost" ? "ghost" : "", size === "small" ? "small" : "", className ?? ""].filter(Boolean).join(" ");
  return (
    <button {...rest} type={type} className={cls} aria-label={label} title={title ?? label} aria-pressed={pressed}>
      <Icon name={icon} />
    </button>
  );
}
