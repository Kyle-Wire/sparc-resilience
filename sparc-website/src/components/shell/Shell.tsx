import Link from "next/link";
import * as React from "react";
import { BRAND_LONG, COMPANY, TAGLINE_LONG, VERSION, COPYRIGHT_YEAR, OFFICE_CITY, OFFICE_STATE } from "@/lib/brand";

export type NavKey = "home" | "product" | "templates" | "pricing" | "about" | "contact" | "download" | "docs" | "account";

const NAV_LINKS: { href: string; label: string; key: NavKey }[] = [
  { href: "/", label: "Home", key: "home" },
  { href: "/product", label: "Product", key: "product" },
  { href: "/templates", label: "Templates", key: "templates" },
  { href: "/pricing", label: "Pricing", key: "pricing" },
  { href: "/download", label: "Download", key: "download" },
  { href: "/about", label: "About", key: "about" },
];

function LogoLockup({ height = 30 }: { height?: number }) {
  // Inline SVG placeholder lockup — the prototype's PNG isn't shipped here.
  // Swap with the real /assets/logo-lockup.png once added to /public.
  return (
    <span
      aria-label={`${COMPANY} — home`}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 10,
        fontFamily: "var(--font-sans)",
        fontWeight: 800,
        fontSize: Math.round(height * 0.46),
        letterSpacing: "-0.01em",
        color: "var(--ink)",
        height,
      }}
    >
      <svg
        width={height}
        height={height}
        viewBox="0 0 64 64"
        fill="none"
        stroke="currentColor"
        strokeWidth="3"
        aria-hidden
      >
        <path d="M10 16 L32 6 L54 16 L54 48 L32 58 L10 48 Z" />
        <path d="M10 16 L32 26 L54 16" />
        <path d="M32 26 L32 58" />
      </svg>
      SPARC<span style={{ color: "var(--muted)", marginLeft: 4 }}>LABS</span>
    </span>
  );
}

export function Nav({ active }: { active?: NavKey }) {
  return (
    <header className="nav">
      <div className="nav-inner">
        <Link className="nav-logo" href="/" aria-label={`${COMPANY} — home`}>
          <LogoLockup height={30} />
        </Link>
        <nav className="nav-links">
          {NAV_LINKS.map((l) => (
            <Link key={l.key} href={l.href} className={active === l.key ? "active" : ""}>
              {l.label}
            </Link>
          ))}
        </nav>
        <div className="nav-cta">
          <Link href="/contact" className="btn btn-ghost btn-sm" style={{ fontWeight: 500, fontSize: 12 }}>
            Contact
          </Link>
          <Link href="/login" className="btn btn-ghost btn-sm" style={{ fontWeight: 500, fontSize: 12 }}>
            Sign in
          </Link>
          <Link href="/signup" className="btn btn-primary btn-sm">
            Request access <span className="arrow">→</span>
          </Link>
        </div>
      </div>
      <div className="ramp-strip" aria-hidden />
    </header>
  );
}

export function Footer() {
  return (
    <footer>
      <div className="wrap">
        <div className="foot-grid">
          <div>
            <Link className="nav-logo" href="/" style={{ marginBottom: 14 }}>
              <LogoLockup height={32} />
            </Link>
            <p className="foot-tag" style={{ marginTop: 14 }}>{TAGLINE_LONG}</p>
            <div className="row" style={{ marginTop: 18, gap: 8, flexWrap: "wrap" }}>
              <span className="pill"><span className="dot" style={{ background: "var(--crimson)" }} /> sidecar · ready</span>
              <span className="pill"><span className="dot" style={{ background: "var(--purple)" }} /> claude · sonnet-4.6</span>
            </div>
          </div>
          <div>
            <h6>Product</h6>
            <Link href="/product">Overview</Link>
            <Link href="/product#pipeline">Pipeline</Link>
            <Link href="/product#causal">Causal model</Link>
            <Link href="/product#scenarios">Scenarios</Link>
            <Link href="/templates">Templates</Link>
            <Link href="/pricing">Pricing</Link>
            <Link href="/download">Download</Link>
            <Link href="/changelog">Changelog</Link>
          </div>
          <div>
            <h6>Company</h6>
            <Link href="/about">About</Link>
            <Link href="/about#thesis">Thesis</Link>
            <Link href="/about#principles">Principles</Link>
            <Link href="/research">Research</Link>
            <Link href="/contact">Contact</Link>
          </div>
          <div>
            <h6>Legal</h6>
            <Link href="/privacy">Privacy</Link>
            <Link href="/terms">Terms</Link>
            <Link href="/data-use">Data use</Link>
            <Link href="/docs">Docs</Link>
          </div>
        </div>
        <div className="foot-bottom">
          <span>© {COPYRIGHT_YEAR} {COMPANY.toUpperCase()} · {VERSION}</span>
          <span>{OFFICE_CITY} · {OFFICE_STATE} · {BRAND_LONG.toUpperCase()}</span>
        </div>
      </div>
    </footer>
  );
}
