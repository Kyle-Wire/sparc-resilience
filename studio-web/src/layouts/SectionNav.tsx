// Section navigation inside a run tab (phase 2): the Heat risk sub-tabs (horizontal) and the
// Model details left rail (vertical). Plain links with aria-current, so every section is a
// deep-linkable, reload-safe URL (`/r/:rid/heat/:section`, `/r/:rid/details/:section`).
import { Link } from "../router";

export type SectionNavItem = { id: string; label: string; to: string };

export function SectionNav({ label, items, current, variant = "tabs" }: { label: string; items: SectionNavItem[]; current: string | null; variant?: "tabs" | "rail" }) {
  return (
    <nav className={variant === "rail" ? "section-rail" : "section-nav"} aria-label={label}>
      <ul>
        {items.map((it) => (
          <li key={it.id}>
            <Link to={it.to} data-section={it.id} aria-current={it.id === current ? "page" : undefined}>
              {it.label}
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  );
}

/** The status line a composer shows while a section's page chunk loads. */
export function SectionLoading() {
  return (
    <p className="cap" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" style={{ width: 12, height: 12, marginRight: 6, verticalAlign: -1 }} />
      Loading…
    </p>
  );
}
