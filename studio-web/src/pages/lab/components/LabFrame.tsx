// The Lab's own header: section links (Design · Library · Plans · Compare, then the "More tools"
// disclosure holding Try amounts, the sweeps page), the engine chip and the compare tray count.
// Every Lab page renders inside it (Climate × adaptation too, which now opens under Heat risk ›
// Future climate, so it has no Lab section link). "More tools" opens by itself while one of its
// pages is current, so the current page's link is always visible.
import { useEffect, useId, useState, type ReactNode } from "react";
import { Icon } from "../../../components/ui/Icon";
import { Link, useLocation } from "../../../router";
import { compareHref, trayRefs, useTray } from "../model/tray";
import { EngineChip } from "./EngineChip";
import "../lab.css";

export const labHref = (rid: string, sub = "") => `/r/${encodeURIComponent(rid)}/lab${sub ? "/" + sub : ""}`;

type LabSection = { sub: string; label: string; match: RegExp };

/** The Lab's main sections, in nav order. */
export const LAB_SECTIONS: LabSection[] = [
  { sub: "", label: "Design", match: /\/lab(\/s\/[^/]+)?\/?$/ },
  { sub: "library", label: "Library", match: /\/lab\/library/ },
  { sub: "plans", label: "Plans", match: /\/lab\/plans/ },
  { sub: "compare", label: "Compare", match: /\/lab\/compare/ },
];

/** The tools under "More tools". */
export const LAB_MORE_TOOLS: LabSection[] = [{ sub: "sweeps", label: "Try amounts", match: /\/lab\/sweeps/ }];

// The disclosure button reads like the nav links next to it (lab.css styles `.lab-nav a`).
const MORE_STYLE = { fontSize: "inherit", fontWeight: 500, height: "auto", padding: "5px 10px" } as const;

export function LabFrame({ rid, title, actions, children, hideEngine }: { rid: string; title: ReactNode; actions?: ReactNode; children: ReactNode; hideEngine?: boolean }) {
  const loc = useLocation();
  const trayItems = useTray((s) => s.byRun[rid]);
  const tray = trayItems?.length ?? 0;
  const moreId = useId();
  const onMore = LAB_MORE_TOOLS.some((s) => s.match.test(loc.pathname));
  const [moreOpen, setMoreOpen] = useState(onMore);
  useEffect(() => {
    if (onMore) setMoreOpen(true);
  }, [onMore]);
  // With two or more pinned items, "Compare" opens them (the tray's own Compare button does too).
  const hrefOf = (sub: string) => (sub === "compare" && tray >= 2 && !/\/lab\/compare/.test(loc.pathname) ? compareHref(rid, trayRefs(trayItems).map((i) => i.ref)) : labHref(rid, sub));
  const link = (s: LabSection) => (
    <Link key={s.label} to={hrefOf(s.sub)} aria-current={s.match.test(loc.pathname) ? "page" : undefined}>
      {s.label}
      {s.sub === "compare" && tray ? <span className="badge">{tray}</span> : null}
    </Link>
  );
  return (
    <div className="lab">
      <div className="lab-head">
        <nav className="lab-nav" aria-label="Scenario Lab">
          {LAB_SECTIONS.map(link)}
          <button
            type="button"
            className="btn small ghost"
            style={MORE_STYLE}
            aria-expanded={moreOpen}
            aria-controls={moreId}
            onClick={() => setMoreOpen(!moreOpen)}
          >
            More tools
            <Icon name={moreOpen ? "chevronDown" : "chevronRight"} size={14} />
          </button>
          {moreOpen ? (
            <span id={moreId} role="group" aria-label="More tools" style={{ display: "inline-flex", flexWrap: "wrap", gap: 2 }}>
              {LAB_MORE_TOOLS.map(link)}
            </span>
          ) : null}
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
