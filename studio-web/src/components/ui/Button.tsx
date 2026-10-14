import { useId, type ButtonHTMLAttributes, type ReactNode } from "react";
import { Icon, type IconName } from "./Icon";
import { useTipOn } from "./useTipOn";

export type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  /** default: surface + hairline. primary: ink fill. accent: crimson fill, for the one main
   *  call to action on a view. ghost: no chrome. danger: destructive. */
  variant?: "default" | "primary" | "accent" | "danger" | "ghost";
  size?: "normal" | "small";
  icon?: IconName;
  /** Shows a spinner and disables the button. */
  busy?: boolean;
  /**
   * What the button does, or why it is disabled, in a tooltip (hover and keyboard focus), in place
   * of a native title. The button is then wrapped in a span that shows the tip (a disabled button
   * gets no pointer or focus events of its own), and the tip text describes the button for screen
   * readers. The wrapper stays while the button is disabled and enabled again, so focus is kept.
   */
  tip?: ReactNode;
  children?: ReactNode;
};

export function Button({ variant = "default", size = "normal", icon, busy, disabled, className, children, type = "button", tip, ...rest }: ButtonProps) {
  const cls = ["btn", variant !== "default" ? variant : "", size === "small" ? "small" : "", className ?? ""].filter(Boolean).join(" ");
  const hasTip = tip !== undefined && tip !== null && tip !== "" && tip !== false;
  const descId = useId();
  // hover (bubbling from the button, disabled or not) and focus (focusin bubbles) reach the wrapper
  const t = useTipOn<HTMLSpanElement>(hasTip ? tip : null, { control: true, describe: false });
  const describedBy = hasTip ? [rest["aria-describedby"], descId].filter(Boolean).join(" ") : rest["aria-describedby"];
  const button = (
    <button {...rest} type={type} className={cls} disabled={disabled || busy} aria-busy={busy || undefined} aria-describedby={describedBy}>
      {busy ? <span className="spinner" aria-hidden="true" /> : icon ? <Icon name={icon} /> : null}
      {children}
    </button>
  );
  if (!hasTip) return button;
  return (
    <>
      <span ref={t.ref} className="btn-tip" style={{ display: "inline-flex" }} {...t.props()}>
        {button}
        <span id={descId} hidden>
          {tip}
        </span>
      </span>
      {t.bubble}
    </>
  );
}
