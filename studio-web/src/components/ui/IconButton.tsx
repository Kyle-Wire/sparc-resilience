import type { ButtonHTMLAttributes, ReactNode } from "react";
import { Icon, type IconName } from "./Icon";
import { useTipOn } from "./useTipOn";

export type IconButtonProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children" | "title"> & {
  icon: IconName;
  /** Accessible name, also shown in the tooltip on hover and keyboard focus. Required: an icon alone has no name. */
  label: string;
  size?: "normal" | "small";
  pressed?: boolean;
  variant?: "default" | "ghost";
  /** A longer explanation for the tooltip in place of the label (no longer a native title). */
  title?: ReactNode;
};

export function IconButton({ icon, label, size = "normal", pressed, variant = "ghost", className, type = "button", title, onMouseEnter, onMouseLeave, onFocus, onBlur, onKeyDown, onPointerDown, onClick, ...rest }: IconButtonProps) {
  const cls = ["btn", "icon", variant === "ghost" ? "ghost" : "", size === "small" ? "small" : "", className ?? ""].filter(Boolean).join(" ");
  const tip = title ?? label;
  // The label already names the button: only a different tip describes it.
  const t = useTipOn<HTMLButtonElement>(tip, { control: true, describe: tip !== label });
  return (
    <>
      <button {...rest} ref={t.ref} type={type} className={cls} aria-label={label} aria-pressed={pressed} {...t.props({ onMouseEnter, onMouseLeave, onFocus, onBlur, onKeyDown, onPointerDown, onClick })}>
        <Icon name={icon} />
      </button>
      {t.bubble}
    </>
  );
}
