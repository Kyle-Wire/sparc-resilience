// Client-side redirect: replaces the current history entry with `to` once mounted (legacy
// URLs, RouteDef.redirect). Renders a quiet status line for the frame before it navigates.
import { useLayoutEffect } from "react";
import { navigate } from "../../router";

export function Redirect({ to }: { to: string }) {
  useLayoutEffect(() => {
    navigate(to, { replace: true });
  }, [to]);
  return (
    <p className="cap" role="status" aria-live="polite">
      Opening {to}…
    </p>
  );
}
