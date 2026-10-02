// The Lab's own header: section links (Design · Library · Plans · Sweeps · Climate · Compare),
// the engine chip and the compare tray count. Every Lab page renders inside it.
import type { ReactNode } from "react";
import { Link, useLocation } from "../../../router";
import { useTray } from "../model/tray";
import { EngineChip } from "./EngineChip";
import "../lab.css";

export const labHref = (rid: string, sub = "") => `/r/${encodeURIComponent(rid)}/lab${sub ? "/" + sub : ""}`;

const SECTIONS: { sub: string; label: string; match: RegExp }[] = [
  { sub: "", label: "Design", match: /\/lab(\/s\/[^/]+)?\/?$/ },
  { sub: "library", label: "Library", match: /\/lab\/library/ },
  { sub: "plans", label: "Plans", match: /\/lab\/plans/ },
  { sub: "sweeps", label: "Sweeps", match: /\/lab\/sweeps/ },
  { sub: "climate", label: "Climate", match: /\/lab\/climate/ },
  { sub: "compare", label: "Compare", match: /\/lab\/compare/ },
];

export function LabFrame({ rid, title, actions, children, hideEngine }: { rid: string; title: ReactNode; actions?: ReactNode; children: ReactNode; hideEngine?: boolean }) {
  const loc = useLocation();
  const tray = useTray((s) => s.byRun[rid]?.length ?? 0);
  return (
    <div className="lab">
      <div className="lab-head">
        <nav className="lab-nav" aria-label="Scenario Lab">
          {SECTIONS.map((s) => (
            <Link key={s.label} to={labHref(rid, s.sub)} aria-current={s.match.test(loc.pathname) ? "page" : undefined}>
              {s.label}
              {s.sub === "compare" && tray ? <span className="badge">{tray}</span> : null}
            </Link>
          ))}
        </nav>
        <span className="spacer" />
        {hideEngine ? null : <EngineChip rid={rid} />}
      </div>
      <div className="lab-title row">
        <h2>{title}</h2>
        <span className="spacer" />
        {actions}
      </div>
      {children}
    </div>
  );
}
